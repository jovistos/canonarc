from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from canonarc import (
    AliasRecord,
    CallableInflectionGenerator,
    CanonArcEngine,
    DefaultLexicalResolver,
    EntityDelta,
    EntityRecord,
    InflectionRequest,
    LLMInflectionGenerator,
    OpenAICompatibleExtractor,
    ResolvedMention,
    StepDelta,
    TokenUsage,
    TranslationPreset,
    build_batch_inflection_prompt,
    build_inflection_prompt,
)


def test_alias_record_lang_roundtrip_and_filtering() -> None:
    """AliasRecord.lang persists in CausalStore and filters cleanly by lang in DefaultLexicalResolver."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(
        EntityRecord(
            entity_id="char_elena",
            canonical_name="Elena",
            base_attributes={"target_name": "Jelena", "target_lang": "sr", "gender": "F"},
        ),
        aliases=[
            AliasRecord(alias="the Chancellor", entity_id="char_elena", kind="title", lang="en"),
            AliasRecord(alias="kancelarka", entity_id="char_elena", kind="title", lang="sr"),
            AliasRecord(alias="Lady E", entity_id="char_elena", kind="epithet"),  # language-neutral
        ],
        step=0,
    )

    stored_aliases = engine.store.get_all_active_aliases("main", 0)
    by_alias = {a.alias.lower(): a for a in stored_aliases}
    assert by_alias["the chancellor"].lang == "en"
    assert by_alias["kancelarka"].lang == "sr"
    assert by_alias["lady e"].lang is None
    # target_name auto-registered with target_lang
    assert by_alias["jelena"].lang == "sr"
    assert by_alias["jelena"].kind == "permanent"

    # Source-language retrieval ("en") matches English + universal aliases, skips Serbian-only aliases
    ctx_en = engine.retrieve_context(step=1, text="The Chancellor entered.", lang="en")
    assert [e.entity_id for e in ctx_en.entities] == ["char_elena"]

    ctx_en_skip_sr = engine.retrieve_context(
        step=1,
        text="Kancelarka je ušla.",
        lang="en",
        fallback_on_empty_mentions="none",
    )
    assert ctx_en_skip_sr.entities == []

    # Target-language retrieval ("sr") matches Serbian + target_name + universal aliases, skips English-only aliases
    ctx_sr = engine.retrieve_context(step=1, text="Kancelarka je ušla.", lang="sr")
    assert [e.entity_id for e in ctx_sr.entities] == ["char_elena"]

    ctx_sr_target = engine.retrieve_context(step=1, text="Jelena je klimnula glavom.", lang="sr")
    assert [e.entity_id for e in ctx_sr_target.entities] == ["char_elena"]

    ctx_sr_skip_en = engine.retrieve_context(
        step=1,
        text="The Chancellor entered.",
        lang="sr",
        fallback_on_empty_mentions="none",
    )
    assert ctx_sr_skip_en.entities == []

    # Language-neutral alias works in both languages
    assert [e.entity_id for e in engine.retrieve_context(step=1, text="Lady E smiled.", lang="en").entities] == [
        "char_elena"
    ]
    assert [e.entity_id for e in engine.retrieve_context(step=1, text="Lady E se osmehnu.", lang="sr").entities] == [
        "char_elena"
    ]
    engine.close()


def test_target_name_automatic_indexing_without_manual_aliases() -> None:
    """base_attributes['target_name'] resolves automatically in retrieve_context and extract_dialogue_candidates."""
    engine = CanonArcEngine(":memory:", preset=TranslationPreset())
    engine.register_entity(
        EntityRecord(
            entity_id="char_elena",
            canonical_name="Elena",
            base_attributes={"target_name": "Jelena", "gender": "F"},
        ),
        step=0,
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_marcus",
            canonical_name="Marcus",
            base_attributes={"target_name": "Markus", "gender": "M"},
        ),
        step=0,
    )

    sr_scene = (
        'Jelena pogleda Markusa preko stola. '
        '"Ovaj ugovor ne može da čeka do jutra, moramo odmah doneti konačnu odluku pre nego što savet zasedne!"'
    )
    candidates = engine.extract_dialogue_candidates(sr_scene, step=1)
    assert len(candidates) == 1
    assert candidates[0].speaker_id == "char_elena"
    assert candidates[0].addressee_id == "char_marcus"
    engine.close()


def test_short_name_inflection_fallback_and_false_positive_guards() -> None:
    """Short inflected Slavic names (3-6 chars) resolve without false positives on common words."""
    resolver = DefaultLexicalResolver(fuzzy_threshold=88.0)
    entities = [
        EntityRecord(entity_id="char_ana", canonical_name="Ana"),
        EntityRecord(entity_id="char_pera", canonical_name="Pera"),
        EntityRecord(entity_id="char_dzon", canonical_name="Džon"),
        EntityRecord(entity_id="char_lira", canonical_name="Lira"),
        EntityRecord(entity_id="char_petar", canonical_name="Petar"),
        EntityRecord(entity_id="char_art", canonical_name="Art"),
    ]

    # Positive short inflections
    for sentence, expected_id in [
        ("Rekao je Ani da dođe odmah.", "char_ana"),
        ("Anin glas je zadrhtao.", "char_ana"),
        ("Peri je predao pismo.", "char_pera"),
        ("Perom je razgovarao dugo.", "char_pera"),
        ("Videli smo Džona u luci.", "char_dzon"),
        ("Sa Džonom niko ne želi sukob.", "char_dzon"),
        ("Lirina senka se pomerila.", "char_lira"),
        ("Predao je ključ Liri.", "char_lira"),
        ("Petra nismo zatekli kod kuće.", "char_petar"),
        ("Petrov mač je ostao na stolu.", "char_petar"),
    ]:
        mentions = resolver.resolve(sentence, (), entities, [])
        assert [m.entity_id for m in mentions] == [expected_id], f"Failed on: {sentence}"
        assert mentions[0].matched_via == "fuzzy"

    # False-positive guards: common words & sentence-initial capitalized words must NOT match
    for negative_sentence in [
        "The storm started at midnight.",
        "Are you certain about that?",
        "Arm the guards immediately.",
        "Article five was repealed.",
        "Artist painted the hall.",
        "And then nothing happened.",
        "Any soldier would obey.",
        "Perhaps we should wait.",
        "Person of interest arrived.",
    ]:
        mentions = resolver.resolve(negative_sentence, (), entities, [])
        assert mentions == [], f"False positive on '{negative_sentence}': {mentions}"


def test_callable_inflection_generator_and_registration_oneshot() -> None:
    """InflectionGenerator is called once per entity at register_entity time, not per chunk."""
    calls: list[tuple[str, str, str | None]] = []

    def fake_inflect(name: str, lang: str, gender: str | None = None) -> list[str]:
        calls.append((name, lang, gender))
        table = {
            "Petar": ["Petar", "Petra", "Petru", "Petrom", "Petre", "Petrov", "Petrova", "Petrovo"],
            "Jelena": ["Jelena", "Jelene", "Jeleni", "Jelenu", "Jelenom", "Jelenin", "Jelenina"],
        }
        return table.get(name, [name])

    gen = CallableInflectionGenerator(fake_inflect)
    engine = CanonArcEngine(":memory:", preset=TranslationPreset(), inflection_generator=gen)

    engine.register_entity(
        EntityRecord(
            entity_id="char_petar",
            canonical_name="Peter",
            base_attributes={"target_name": "Petar", "gender": "M", "declension": "o-decl"},
        ),
        step=0,
        target_lang="sr",
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_elena",
            canonical_name="Elena",
            base_attributes={"target_name": "Jelena", "gender": "F", "declension": "a-decl"},
        ),
        step=0,
        target_lang="sr",
    )

    assert len(calls) == 2
    assert calls[0] == ("Petar", "sr", "M")
    assert calls[1] == ("Jelena", "sr", "F")

    # Verify aliases stored with lang='sr', base target_name='permanent', generated forms='inflection'
    stored = {a.alias: a for a in engine.store.get_all_active_aliases("main", 0)}
    assert stored["Petar"].kind == "permanent"
    assert stored["Jelena"].kind == "permanent"
    assert "Petrov" in stored and stored["Petrov"].lang == "sr" and stored["Petrov"].kind == "inflection"
    assert "Jelenin" in stored and stored["Jelenin"].lang == "sr" and stored["Jelenin"].kind == "inflection"

    # Resolve multiple chunks without any additional generator calls
    for chunk, expected_speaker in [
        (
            (
                'Petrov glas odjeknu hodnikom. '
                '"Nikome ne verujte dok pažljivo ne proverimo sve pečate i ne zatvorimo kapije tvrđave večeras!"'
            ),
            "char_petar",
        ),
        (
            (
                'Jelenin pogled pade na Petra. '
                '"Ovaj dokument je očigledan falsifikat koji su nam podmetnuli u samoj arhivi pre zore!"'
            ),
            "char_elena",
        ),
    ]:
        cands = engine.extract_dialogue_candidates(chunk, step=1, lang="sr")
        assert len(cands) == 1
        assert cands[0].speaker_id == expected_speaker

    assert len(calls) == 2  # Still only called once per registered entity!
    engine.close()


def test_llm_inflection_generator_parsing_and_disk_cache(tmp_path: Path) -> None:
    """LLMInflectionGenerator parses JSON objects/arrays and caches results to disk across instances."""
    prompt_preview = build_inflection_prompt("Petar", "Serbian", gender="M", declension="o-decl")
    assert "Petar" in prompt_preview and "Serbian" in prompt_preview

    api_calls: list[str] = []

    class StubExtractor(OpenAICompatibleExtractor):
        def __init__(self) -> None:
            super().__init__(model="stub-model", api_key="test")

        def call_json_api(self, prompt: str, **kwargs: Any) -> tuple[str, TokenUsage]:
            api_calls.append(prompt)
            return (
                '```json\n{"forms": ["Petar", "Petra", "Petru", "Petrom", "Petre", "Petrov", "Petrova", "Petrovo"]}\n```',
                TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35, api_calls=1),
            )

    cache_dir = tmp_path / "inflection_cache"
    gen1 = LLMInflectionGenerator(StubExtractor(), cache_dir=cache_dir)
    forms1 = gen1.generate_inflections("Petar", "sr", gender="M")
    assert forms1 == ["Petar", "Petra", "Petru", "Petrom", "Petre", "Petrov", "Petrova", "Petrovo"]
    assert gen1.api_calls == 1
    assert gen1.cache_hits == 0

    # Second call on same instance hits memory cache
    forms2 = gen1.generate_inflections("Petar", "sr", gender="M")
    assert forms2 == forms1
    assert gen1.api_calls == 1
    assert gen1.cache_hits == 1

    # New instance pointing to same cache_dir hits disk cache with 0 API calls
    gen2 = LLMInflectionGenerator(StubExtractor(), cache_dir=cache_dir)
    forms3 = gen2.generate_inflections("Petar", "sr", gender="M")
    assert forms3 == forms1
    assert gen2.api_calls == 0
    assert gen2.cache_hits == 1
    assert len(api_calls) == 1

    # Verify disk cache file is valid JSON
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    saved = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert saved["name"] == "Petar"
    assert saved["lang"] == "sr"


def test_zero_seed_expand_entity_inflections_and_custom_resolver_compat() -> None:
    """Option A: apply_delta stays offline, expand_entity_inflections() expands on demand, and 4-arg MentionResolvers still work."""
    calls: list[str] = []
    gen = CallableInflectionGenerator(
        lambda name, lang: (calls.append(name) or [name, f"{name}a", f"{name}u"])
    )
    engine = CanonArcEngine(":memory:", inflection_generator=gen)

    # Zero-Seed entity introduced via apply_delta does NOT call LLM during commit
    engine.apply_delta(
        step=1,
        delta=StepDelta(
            entities=[
                EntityDelta(
                    entity_id="char_marko",
                    canonical_name="Marko",
                    base_attributes={"target_name": "Marko", "target_lang": "sr", "gender": "M"},
                )
            ]
        ),
    )
    assert calls == []

    # Explicit one-shot expansion persists inflected aliases
    added = engine.expand_entity_inflections("char_marko", step=1)
    assert calls == ["Marko"]
    assert {a.alias for a in added} >= {"Marko", "Markoa", "Markou"}
    assert all(a.lang == "sr" for a in added)

    # Backwards compatibility: custom 4-arg MentionResolver without `lang` kwarg still works
    class LegacyResolver:
        def resolve(self, text, explicit_mentions, entities, aliases):
            return [ResolvedMention(surface_text="Marko", entity_id="char_marko", matched_via="canonical")]

    engine.mention_resolver = LegacyResolver()
    ctx = engine.retrieve_context(step=2, text="Marko nodded.", lang="sr")
    assert [e.entity_id for e in ctx.entities] == ["char_marko"]
    engine.close()


def test_llm_inflection_generator_batching_cache_sharing_and_fallback(tmp_path: Path) -> None:
    """generate_inflections_batch chunks by batch_size, shares cache with single-item calls, and falls back on dropped IDs."""
    preview = build_batch_inflection_prompt(
        [InflectionRequest(name="Petar", lang="sr", gender="M")]
    )
    assert '"id": 0' in preview and "Petar" in preview

    prompts_seen: list[str] = []
    drop_id_once = {"dropped": False}

    class BatchStubExtractor(OpenAICompatibleExtractor):
        def __init__(self) -> None:
            super().__init__(model="stub-batch-model", api_key="test")

        def call_json_api(self, prompt: str, **kwargs: Any) -> tuple[str, TokenUsage]:
            prompts_seen.append(prompt)
            # Batch prompt contains "Input: ["
            if "Input: [" in prompt:
                raw_json = prompt.split("Input: ", 1)[1]
                items = json.loads(raw_json)
                out_items = []
                for item in items:
                    # Simulate LLM omitting id=1 on the very first batch call to test fallback
                    if not drop_id_once["dropped"] and item["id"] == 1:
                        drop_id_once["dropped"] = True
                        continue
                    nm = item["name"]
                    out_items.append({"id": item["id"], "forms": [nm, f"{nm}a", f"{nm}u"]})
                return (
                    json.dumps({"items": out_items}, ensure_ascii=False),
                    TokenUsage(prompt_tokens=40, completion_tokens=30, total_tokens=70, api_calls=1),
                )
            # Single-item fallback prompt: extract proper name inside single quotes
            start = prompt.index("'") + 1
            end = prompt.index("'", start)
            nm = prompt[start:end]
            return (
                json.dumps({"forms": [nm, f"{nm}a", f"{nm}u"]}, ensure_ascii=False),
                TokenUsage(prompt_tokens=15, completion_tokens=10, total_tokens=25, api_calls=1),
            )

    cache_dir = tmp_path / "batch_cache"
    gen = LLMInflectionGenerator(BatchStubExtractor(), cache_dir=cache_dir)

    # 7 unique Serbian names + 1 duplicate -> with batch_size=3:
    # 3 batch calls (sizes 3, 3, 1 -> last is single-item) + 1 single-item fallback for dropped id=1 = 4 calls
    reqs = [
        InflectionRequest(name="Artur", lang="sr", gender="M"),
        InflectionRequest(name="Petar", lang="sr", gender="M"),
        InflectionRequest(name="Marko", lang="sr", gender="M"),
        InflectionRequest(name="Artur", lang="sr", gender="M"),  # duplicate within batch
        InflectionRequest(name="Jelena", lang="sr", gender="F"),
        InflectionRequest(name="Milica", lang="sr", gender="F"),
        InflectionRequest(name="Lazar", lang="sr", gender="M"),
        InflectionRequest(name="Nikola", lang="sr", gender="M"),
    ]
    results = gen.generate_inflections_batch(reqs, batch_size=3)
    assert len(results) == len(reqs)
    assert results[0] == ["Artur", "Artura", "Arturu"]
    assert results[1] == ["Petar", "Petara", "Petaru"]  # recovered via single-item fallback
    assert results[3] == results[0]  # intra-batch duplicate matched
    assert results[7] == ["Nikola", "Nikolaa", "Nikolau"]
    assert gen.api_calls == 4  # 2 multi-item batches + 1 dropped-id fallback + 1 trailing single item

    # Single-item call on a new generator pointing to same cache_dir hits disk cache with 0 API calls
    gen2 = LLMInflectionGenerator(BatchStubExtractor(), cache_dir=cache_dir)
    assert gen2.generate_inflections("Jelena", "sr", gender="F") == ["Jelena", "Jelenaa", "Jelenau"]
    assert gen2.api_calls == 0
    assert gen2.cache_hits == 1

    # seed_entities() uses batched generation (7 fresh entities with batch_size=3 -> 3 calls instead of 7)
    drop_id_once["dropped"] = True  # disable drop for clean count
    gen3 = LLMInflectionGenerator(BatchStubExtractor())
    engine = CanonArcEngine(":memory:", inflection_generator=gen3)
    entities = [
        EntityRecord(
            entity_id=f"char_{i}",
            canonical_name=f"Hero{i}",
            base_attributes={"target_name": f"Junak{i}", "target_lang": "sr", "gender": "M"},
        )
        for i in range(7)
    ]
    engine.seed_entities(entities, step=0, batch_size=3)
    assert gen3.api_calls == 3  # ceil(7 / 3) == 3

    aliases = engine.store.get_all_active_aliases("main", 0)
    by_alias = {a.alias: a for a in aliases}
    assert by_alias["Junak0"].kind == "permanent"
    assert by_alias["Junak0a"].kind == "inflection"
    assert by_alias["Junak6u"].kind == "inflection"

    # DefaultLexicalResolver still resolves kind='inflection' aliases
    ctx = engine.retrieve_context(step=1, text="Videli smo Junak0a na trgu.", lang="sr")
    assert [e.entity_id for e in ctx.entities] == ["char_0"]
    engine.close()
