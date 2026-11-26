from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from canonarc import (
    AliasRecord,
    BibleBuilder,
    BibleSection,
    CachedDeltaExtractor,
    CallableExtractor,
    CallableInflectionGenerator,
    CanonArcEngine,
    EntityRecord,
    TokenUsage,
    bootstrap_bible,
)


class _MockJsonClient:
    """Deterministic JsonCompletionClient that records prompts and returns scripted JSON payloads."""

    def __init__(self, responses: list[str], *, model: str = "mock-bible-model") -> None:
        self.model = model
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []
        self.calls: int = 0

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, TokenUsage]:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx], TokenUsage(
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            api_calls=1,
        )


def test_bible_builder_sequential_accumulation_locks_and_conflicts() -> None:
    ch1_payload = json.dumps(
        {
            "entities": [
                {
                    "canonical_name": "Marcus Vance",
                    "kind": "character",
                    "target_name": "Markus Vans",
                    "aliases": ["Vance"],
                    "notes": "Harbor physician",
                },
                {
                    "canonical_name": "Ravenport",
                    "kind": "location",
                    "target_name": "Gavranluka",
                },
            ]
        }
    )
    ch2_payload = json.dumps(
        {
            "entities": [
                {
                    "canonical_name": "Marcus Vance",
                    "kind": "character",
                    "target_name": "Marko Vanc",  # conflicts with locked 'Markus Vans'
                    "gender": "masculine",  # fills previously empty field
                    "declension": "animate_m",  # fills previously empty field
                    "aliases": ["Dr. Vance"],
                    "notes": "Served in the northern war",
                },
                {
                    "canonical_name": "Ravenport",
                    "kind": "faction",  # conflicts with locked 'location'
                    "target_name": "Gavranluka",
                },
            ],
            "alias_updates": [
                {"entity_id": "char_marcus_vance", "add_aliases": ["the Doctor"]},
                {"entity_id": "char_nonexistent", "add_aliases": ["ghost"]},
            ],
        }
    )

    client = _MockJsonClient([ch1_payload, ch2_payload])
    builder = BibleBuilder(
        client,
        source_lang="en",
        target_lang="sr",
        instructions="Use Serbian Ekavian transliteration.",
    )

    draft = builder.build(
        [
            BibleSection(section_id="ch_01", title="Arrival", text="Marcus Vance arrived at Ravenport."),
            BibleSection(section_id="ch_02", title="Clinic", text="The Doctor treated the wounded in Ravenport."),
        ]
    )

    assert draft.sections_processed == 2
    assert draft.api_calls == 2
    assert draft.usage.total_tokens == 280
    assert len(draft.entities) == 2

    # Section 2 prompt must include locked ledger from Section 1
    assert "char_marcus_vance | Marcus Vance [character] -> 'Markus Vans'" in client.prompts[1]
    assert "loc_ravenport | Ravenport [location] -> 'Gavranluka'" in client.prompts[1]
    assert "Use Serbian Ekavian transliteration." in client.prompts[0]

    by_id = {e.entity_id: e for e in draft.entities}
    marcus = by_id["char_marcus_vance"]
    assert marcus.base_attributes["target_name"] == "Markus Vans"  # locked value preserved
    assert marcus.base_attributes["target_lang"] == "sr"
    assert marcus.base_attributes["gender"] == "masculine"
    assert marcus.base_attributes["declension"] == "animate_m"
    assert "Harbor physician" in marcus.base_attributes["notes"]
    assert "Served in the northern war" in marcus.base_attributes["notes"]

    marcus_aliases = {a.alias for a in draft.aliases["char_marcus_vance"]}
    assert marcus_aliases == {"Vance", "Dr. Vance", "the Doctor"}

    ravenport = by_id["loc_ravenport"]
    assert ravenport.kind == "location"

    # Both target_name and kind contradictions are logged as conflicts
    conflict_fields = {(c.entity_id, c.field) for c in draft.conflicts}
    assert ("char_marcus_vance", "target_name") in conflict_fields
    assert ("loc_ravenport", "kind") in conflict_fields
    assert any("char_nonexistent" in w for w in draft.warnings)


def test_bible_builder_domain_agnostic_mode_and_filters() -> None:
    payload = json.dumps(
        {
            "entities": [
                {
                    "canonical_name": "Iron Citadel",
                    "kind": "location",
                    "aliases": ["the Fortress"],
                    "attributes": {"region": "Northreach", "fortified": True},
                    "notes": "Seat of the northern wardens",
                },
                {
                    "canonical_name": "TBD_PLACEHOLDER",
                    "kind": "location",
                },
                {
                    "canonical_name": "Fireball",
                    "kind": "spell",  # not in allowed_kinds
                },
            ]
        }
    )
    client = _MockJsonClient([payload])
    builder = BibleBuilder(
        client,
        target_lang=None,
        allowed_kinds=("character", "location"),
        ignore_patterns=(r"TBD_.*",),
    )

    draft = builder.build(["The Iron Citadel stood over Northreach."])
    assert len(draft.entities) == 1
    citadel = draft.entities[0]
    assert citadel.entity_id == "loc_iron_citadel"
    assert "target_name" not in citadel.base_attributes
    assert citadel.base_attributes["region"] == "Northreach"
    assert citadel.base_attributes["fortified"] is True
    assert any("spell" in w for w in draft.warnings)


def test_bible_builder_disk_cache_and_downstream_invalidation(tmp_path: Path) -> None:
    ch1 = json.dumps({"entities": [{"canonical_name": "Lyra", "kind": "character", "target_name": "Lira"}]})
    ch2 = json.dumps({"entities": [{"canonical_name": "Kael", "kind": "character", "target_name": "Kael"}]})

    cache_dir = tmp_path / "bible_cache"
    client1 = _MockJsonClient([ch1, ch2])
    builder1 = BibleBuilder(client1, target_lang="sr", cache_dir=cache_dir)
    sections = [
        BibleSection(section_id="ch_01", text="Lyra opened the gate."),
        BibleSection(section_id="ch_02", text="Kael followed Lyra inside."),
    ]
    draft1 = builder1.build(sections)
    assert draft1.api_calls == 2
    assert draft1.cache_hits == 0

    # Identical re-run hits cache completely
    client2 = _MockJsonClient([ch1, ch2])
    builder2 = BibleBuilder(client2, target_lang="sr", cache_dir=cache_dir)
    draft2 = builder2.build(sections)
    assert draft2.api_calls == 0
    assert draft2.cache_hits == 2
    assert client2.calls == 0

    # Changing section 1 output invalidates section 2 because locked ledger changes
    ch1_edited = json.dumps(
        {"entities": [{"canonical_name": "Lyra", "kind": "character", "target_name": "Lajra"}]}
    )
    client3 = _MockJsonClient([ch1_edited, ch2])
    builder3 = BibleBuilder(client3, target_lang="sr", cache_dir=cache_dir)
    draft3 = builder3.build(
        [
            BibleSection(section_id="ch_01", text="Lyra opened the iron gate."),
            BibleSection(section_id="ch_02", text="Kael followed Lyra inside."),
        ]
    )
    assert draft3.api_calls == 2
    assert draft3.cache_hits == 0
    assert "Lajra" in client3.prompts[1]


def test_bible_draft_export_round_trip_with_sync_bible(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "entities": [
                {
                    "canonical_name": "Arthur",
                    "kind": "character",
                    "target_name": "Artur",
                    "gender": "masculine",
                    "aliases": ["the young king"],
                    "notes": "Heir to the throne",
                }
            ]
        }
    )
    client = _MockJsonClient([payload])
    draft = BibleBuilder(client, source_lang="en", target_lang="sr").build(["Arthur drew the sword."])

    json_path = draft.export(tmp_path / "bible.json")
    csv_path = draft.export(tmp_path / "glossary.csv")

    eng_json = CanonArcEngine(":memory:")
    rep_json = eng_json.sync_bible(json_path, step=0)
    assert rep_json.created_entities == ["char_arthur"]

    eng_csv = CanonArcEngine(":memory:")
    rep_csv = eng_csv.sync_bible(csv_path, step=0)
    assert rep_csv.created_entities == ["char_arthur"]
    ents = eng_csv.store.get_all_active_entities("main", 0)
    assert ents[0].base_attributes["target_name"] == "Artur"
    assert ents[0].base_attributes["notes"] == "Heir to the throne"


def test_bootstrap_bible_commits_inflects_detects_duplicates_and_inherits(tmp_path: Path) -> None:
    inflector = CallableInflectionGenerator(
        lambda name, lang, **kw: [name, f"{name}a", f"{name}u"]
    )
    engine = CanonArcEngine(":memory:", inflection_generator=inflector)

    # Pre-seed an entity (simulating Book 1 cast)
    engine.register_entity(
        EntityRecord(
            entity_id="char_lyra",
            kind="character",
            canonical_name="Lyra",
            base_attributes={"target_name": "Lira", "target_lang": "sr", "gender": "feminine"},
        ),
        aliases=[AliasRecord(alias="the Ash Queen", entity_id="char_lyra", kind="epithet")],
        step=0,
        target_lang="sr",
        auto_inflect=False,
    )

    ch1 = json.dumps(
        {
            "entities": [
                {
                    "canonical_name": "Lyra",
                    "kind": "character",
                    "target_name": "Lajra",  # should keep inherited 'Lira' and record conflict
                    "aliases": ["Lady Lyra"],
                },
                {
                    "canonical_name": "Alexander",
                    "kind": "character",
                    "target_name": "Aleksandar",
                    "gender": "masculine",
                },
                {
                    "canonical_name": "Aleksander",  # near-duplicate of Alexander
                    "kind": "character",
                    "target_name": "Aleksander",
                    "gender": "masculine",
                },
            ]
        }
    )

    client = _MockJsonClient([ch1])
    export_file = tmp_path / "bootstrapped.json"
    report = engine.bootstrap_bible(
        [BibleSection(section_id="ch_01", text="Lyra met Alexander and Aleksander.")],
        client=client,
        source_lang="en",
        target_lang="sr",
        export_path=export_file,
        step=0,
        fuzzy_threshold=85.0,
    )

    assert export_file.is_file()
    assert report.sync_report is not None
    assert "char_alexander" in report.sync_report.created_entities
    assert "char_aleksander" in report.sync_report.created_entities
    assert report.sync_report.inflections_generated > 0

    # Inherited Book 1 lock preserved
    ents = {e.entity_id: e for e in engine.store.get_all_active_entities("main", 0)}
    assert ents["char_lyra"].base_attributes["target_name"] == "Lira"
    assert any(c.entity_id == "char_lyra" and c.field == "target_name" for c in report.draft.conflicts)

    # Inflections generated on newly committed entities
    alex_aliases = {
        a.alias for a in engine.store.get_all_active_aliases("main", 0) if a.entity_id == "char_alexander"
    }
    assert {"Aleksandar", "Aleksandara", "Aleksandaru"}.issubset(alex_aliases)

    # Fuzzy near-duplicate surfaced without blind auto-merge
    assert len(report.duplicate_candidates) == 1
    pair = {report.duplicate_candidates[0].duplicate_id, report.duplicate_candidates[0].canonical_id}
    assert pair == {"char_alexander", "char_aleksander"}


def test_bootstrap_bible_accepts_cached_extractor_and_commit_false(tmp_path: Path) -> None:
    def _fake_extract(prompt: str, schema: dict[str, Any]) -> str:
        return json.dumps({"entities": [{"canonical_name": "Vesper", "kind": "character"}]})

    # CallableExtractor implements JsonCompletionClient; wrapping in CachedDeltaExtractor must still work
    wrapped = CachedDeltaExtractor(CallableExtractor(_fake_extract), cache_dir=tmp_path / "delta_cache")
    engine = CanonArcEngine(":memory:", extractor=wrapped)

    report = bootstrap_bible(
        engine,
        ["Vesper walked through the fog."],
        commit=False,
        export_path=tmp_path / "review_me.json",
    )

    assert report.sync_report is None
    assert report.duplicate_candidates == []
    assert len(report.draft.entities) == 1
    assert engine.store.get_all_active_entities("main", 0) == []
    assert (tmp_path / "review_me.json").is_file()


def test_bible_builder_retries_on_malformed_json() -> None:
    client = _MockJsonClient(
        [
            "not valid json at all",
            '```json\n{"entities": [{"canonical_name": "Sera", "kind": "character"}]}\n```',
        ]
    )
    builder = BibleBuilder(client, max_retries=1)
    draft = builder.build(["Sera smiled."])
    assert len(draft.entities) == 1
    assert draft.entities[0].entity_id == "char_sera"
    assert draft.usage.retry_count == 1
    assert draft.usage.validation_errors == 1
