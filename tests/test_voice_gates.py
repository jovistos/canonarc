from canonarc.models import VoiceExemplar
from canonarc.voice import IngestionGate, OfflineNgramEmbedder, VoiceBank


def test_anti_drift_gate():
    gate = IngestionGate(min_words=12, max_words=60, min_distinctiveness=0.40)

    # 1. Short utterance -> rejected by word count
    short_text = 'He shouted, "Yes, let us go!"'
    candidates = gate.extract_candidates(short_text)
    assert len(candidates) == 0

    # 2. Distinctive line (15 words) -> accepted
    rich_text = (
        'Merlin looked sternly and said, "The ancient parchment reveals an ominous portent '
        'that threatens the sovereign destiny of Camelot."'
    )
    candidates = gate.extract_candidates(rich_text)
    assert len(candidates) == 1
    assert candidates[0].word_count >= 12
    assert candidates[0].distinctiveness_score >= 0.40

    # 3. Quality gate check
    bad_ex = VoiceExemplar(speaker_id="merlin", step=1, text="...", quality_score=6.5)
    good_ex = VoiceExemplar(speaker_id="merlin", step=1, text="...", quality_score=8.5)
    assert not gate.passes_quality_gate(bad_ex)
    assert gate.passes_quality_gate(good_ex)


def test_voice_bank_hard_register_cascade():
    embedder = OfflineNgramEmbedder()
    bank = VoiceBank(embedder=embedder)

    ex_formal = VoiceExemplar(
        exemplar_id="ex1",
        step=5,
        speaker_id="char_arthur",
        text="I beseech your wisdom, venerable master of the council.",
        metadata={"register": "formal", "tone": "respectful"},
        quality_score=9.0,
    )
    ex_informal_angry = VoiceExemplar(
        exemplar_id="ex2",
        step=10,
        speaker_id="char_arthur",
        text="Listen to me now, you traitorous fool, or taste my blade!",
        metadata={"register": "informal", "tone": "furious"},
        quality_score=9.5,
    )

    candidates = [
        (ex_formal, embedder.embed_text(ex_formal.text).tobytes()),
        (ex_informal_angry, embedder.embed_text(ex_informal_angry.text).tobytes()),
    ]

    # Query with target_register="informal" and tone="furious"
    # MUST exclude ex_formal completely (hard filter)
    ranked = bank.rank_exemplars(
        query_text="Traitorous fool!",
        candidates=candidates,
        target_register="informal",
        target_tone="furious",
        top_k=2,
    )
    assert len(ranked) == 1
    assert ranked[0].exemplar_id == "ex2"

    # Query with target_register="formal" -> MUST exclude ex_informal_angry
    ranked_formal = bank.rank_exemplars(
        query_text="wisdom council",
        candidates=candidates,
        target_register="formal",
        target_tone="respectful",
        top_k=2,
    )
    assert len(ranked_formal) == 1
    assert ranked_formal[0].exemplar_id == "ex1"


def test_offline_embedder_deterministic_across_pythonhashseed():
    """OfflineNgramEmbedder produces identical bytes across processes with different PYTHONHASHSEED."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    code = (
        "from canonarc.voice import OfflineNgramEmbedder; "
        "e = OfflineNgramEmbedder(); "
        "print(e.embed_text('The ancient parchment reveals an ominous portent.').tobytes().hex())"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONHASHSEED": "1", "PYTHONPATH": src_dir},
        text=True,
    ).strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONHASHSEED": "99999", "PYTHONPATH": src_dir},
        text=True,
    ).strip()
    assert out1 == out2
    assert len(out1) > 0


def test_cjk_ingestion_gate_and_boilerplate_rejection():
    """IngestionGate supports Chinese/Japanese/Korean dialogue and rejects repetitive CJK filler."""
    gate = IngestionGate(min_words=12, max_words=60, min_distinctiveness=0.40)

    # Rich Chinese dialogue in CJK corner brackets
    zh_text = '爱丽丝皱起眉头说道：「这本古老的羊皮卷揭示了一个不祥的预兆，正威胁着卡美洛的命运。」'
    zh_cands = gate.extract_candidates(zh_text)
    assert len(zh_cands) == 1
    assert zh_cands[0].word_count >= 12
    assert zh_cands[0].distinctiveness_score >= 0.40

    # Rich Japanese dialogue
    ja_text = 'マーリンは静かに言った。「古い羊皮紙はキャメロットの運命を脅かす不吉な前兆を明らかにした。」'
    ja_cands = gate.extract_candidates(ja_text)
    assert len(ja_cands) == 1

    # Rich Korean dialogue
    ko_text = '멀린이 엄숙하게 말했다. "오래된 양피지는 카멜롯의 운명을 위협하는 불길한 징조를 드러냈다."'
    ko_cands = gate.extract_candidates(ko_text)
    assert len(ko_cands) == 1

    # Repetitive CJK filler must be rejected by distinctiveness
    filler_text = '他重复道：「好好好好好好好好好好好好好好」'
    assert gate.extract_candidates(filler_text) == []


def test_dialogue_candidate_speaker_attribution_and_indexing():
    """extract_dialogue_candidates attributes speaker_id/addressee_id and bridges to index_voice_exemplars."""
    from canonarc import CanonArcEngine, EntityRecord

    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)

    scene = (
        'Merlin turned to Arthur and said, "The ancient parchment reveals an ominous portent '
        'that threatens the sovereign destiny of Camelot."\n'
        '"I will not yield to fear while the knights of the realm still stand beside me," '
        'Arthur replied.'
    )
    candidates = engine.extract_dialogue_candidates(scene, step=1)
    assert len(candidates) == 2

    assert candidates[0].speaker_id == "char_merlin"
    assert candidates[0].addressee_id == "char_arthur"
    assert candidates[1].speaker_id == "char_arthur"

    exemplars = [
        c.to_voice_exemplar(step=1, metadata={"register": "formal"}, quality_score=9.5)
        for c in candidates
    ]
    indexed = engine.index_voice_exemplars(step=1, exemplars=exemplars)
    assert len(indexed) == 2
    assert indexed[0].speaker_id == "char_merlin"
    assert indexed[1].speaker_id == "char_arthur"
    engine.close()


def test_voice_bank_register_fallback_note_and_disable():
    """Switching to a new register falls back with a warning note by default, or returns [] if disabled."""
    from canonarc.models import StepContext
    from canonarc.presets import (
        InteractiveFictionPreset,
        SchemaPolicy,
        SimulationPreset,
        TranslationPreset,
    )

    embedder = OfflineNgramEmbedder()
    bank = VoiceBank(embedder=embedder)

    ex_formal = VoiceExemplar(
        exemplar_id="ex_formal_only",
        step=1,
        speaker_id="char_arthur",
        text="I beseech your wisdom, venerable master of the council.",
        metadata={"register": "formal", "tone": "respectful"},
        quality_score=9.0,
    )
    candidates = [(ex_formal, embedder.embed_text(ex_formal.text).tobytes())]

    # Default policy: falls back to formal exemplar and tags it with [Note: different register]
    ranked = bank.rank_exemplars(
        query_text="Hey buddy!",
        candidates=candidates,
        target_register="informal",
        top_k=2,
    )
    assert len(ranked) == 1
    assert ranked[0].exemplar_id == "ex_formal_only"
    assert ranked[0].metadata.get("fallback_note") == "[Note: different register]"
    # Original exemplar metadata was not mutated
    assert "fallback_note" not in ex_formal.metadata

    ctx = StepContext(timeline_id="main", step=2, voice_exemplars=ranked)
    for preset in (InteractiveFictionPreset(), TranslationPreset(), SimulationPreset()):
        rendered = preset.format_prompt_context(ctx)
        assert "[Note: different register]" in rendered

    # Strict policy with fallback disabled returns []
    strict_policy = SchemaPolicy(fallback_on_empty_hard_filter=False)
    strict_ranked = bank.rank_exemplars(
        query_text="Hey buddy!",
        candidates=candidates,
        target_register="informal",
        policy=strict_policy,
        top_k=2,
    )
    assert strict_ranked == []


def test_retrieve_context_voice_constraints_channel_filtering():
    """Scene-level voice_constraints filter Tier 5 exemplars via SchemaPolicy hard_filter_keys."""
    from canonarc import (
        CanonArcEngine,
        EdgeDelta,
        EntityRecord,
        SchemaPolicy,
        SimulationPreset,
        StepDelta,
    )

    policy = SchemaPolicy(
        hard_filter_keys=("channel",),
        soft_boost_weights={"stance": 0.20, "urgency": 0.15},
        fallback_on_empty_hard_filter=True,
    )
    engine = CanonArcEngine(":memory:", preset=SimulationPreset(policy=policy))
    engine.register_entity(EntityRecord(entity_id="agent_alice", canonical_name="Alice"), step=0)
    engine.register_entity(EntityRecord(entity_id="agent_bob", canonical_name="Bob"), step=0)
    engine.commit_step(
        step=1,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="agent_alice",
                    target_id="agent_bob",
                    state={"affinity": 0.7, "stance": "cooperative"},
                )
            ]
        ),
        exemplars=[
            VoiceExemplar(
                step=1,
                speaker_id="agent_alice",
                text="Med-1 to all stations: storm barometer is dropping fast, secure your thermal seals now.",
                metadata={"channel": "radio", "stance": "cooperative", "urgency": "high"},
                quality_score=9.2,
            ),
            VoiceExemplar(
                step=1,
                speaker_id="agent_alice",
                text="Sit down and let me check those frostbitten fingers before you head back.",
                metadata={"channel": "face_to_face", "stance": "cooperative", "urgency": "low"},
                quality_score=9.0,
            ),
        ],
    )

    radio_ctx = engine.retrieve_context(
        step=2,
        mentions=["Alice", "Bob"],
        voice_constraints={"channel": "radio", "urgency": "high"},
    )
    assert len(radio_ctx.voice_exemplars) == 1
    assert radio_ctx.voice_exemplars[0].metadata["channel"] == "radio"

    f2f_ctx = engine.session("main").retrieve_context(
        step=2,
        mentions=["Alice", "Bob"],
        voice_constraints={"channel": "face_to_face", "urgency": "low"},
    )
    assert len(f2f_ctx.voice_exemplars) == 1
    assert f2f_ctx.voice_exemplars[0].metadata["channel"] == "face_to_face"

    # Also works when speaker has no outgoing edges
    engine.register_entity(EntityRecord(entity_id="agent_solo", canonical_name="Solo"), step=1)
    engine.index_voice_exemplars(
        step=2,
        exemplars=[
            VoiceExemplar(
                step=2,
                speaker_id="agent_solo",
                text="Solo beacon online on emergency radio frequency alpha.",
                metadata={"channel": "radio"},
                quality_score=9.0,
            ),
            VoiceExemplar(
                step=2,
                speaker_id="agent_solo",
                text="Keep your voice down inside the corridor.",
                metadata={"channel": "face_to_face"},
                quality_score=9.0,
            ),
        ],
    )
    solo_ctx = engine.retrieve_context(
        step=3,
        mentions=["Solo"],
        voice_constraints={"channel": "radio"},
    )
    assert len(solo_ctx.voice_exemplars) == 1
    assert solo_ctx.voice_exemplars[0].metadata["channel"] == "radio"
    engine.close()


def test_european_balkan_dash_dialogue_and_interjection_stripping():
    """En-dash/em-dash/quotation-dash dialogue strips narrator beats and routes them to attribution."""
    from canonarc import CanonArcEngine, EntityRecord

    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_artur", canonical_name="Artur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_elena", canonical_name="Elena"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_marko", canonical_name="Marko"), step=0)

    scene = (
        "– Ne znam – reče Artur tiho – šta da radim ako nas čuvari pronađu pre svitanja u ovoj kuli.\n"
        "— Moraš da odlučiš odmah i povedeš ljude kroz severnu kapiju dok još imamo vremena — odgovori Elena.\n"
        "Marko se okrenu prema njima:\n"
        "― Između 10–15 vojnika već patrolira duž zidina, pa moramo krenuti zaista oprezno i bez buke."
    )
    candidates = engine.extract_dialogue_candidates(scene, step=1)
    assert len(candidates) == 3

    # 1. En-dash 3-part line: middle narrator beat stripped from text and used for attribution
    assert "reče Artur" not in candidates[0].text
    assert candidates[0].text == "Ne znam šta da radim ako nas čuvari pronađu pre svitanja u ovoj kuli."
    assert candidates[0].speaker_id == "char_artur"

    # 2. Em-dash 2-part line: trailing beat stripped, previous turn's Artur did NOT bleed into before_win
    assert "odgovori Elena" not in candidates[1].text
    assert candidates[1].text == "Moraš da odlučiš odmah i povedeš ljude kroz severnu kapiju dok još imamo vremena"
    assert candidates[1].speaker_id == "char_elena"

    # 3. Quotation-dash line with narrative lead-in preserved and unspaced range '10–15' intact
    assert "10–15" in candidates[2].text
    assert candidates[2].speaker_id == "char_marko"
    engine.close()


def test_dash_dialogue_preserves_rhetorical_aside_ending_in_question():
    """In-speech parenthetical asides ending with '?' or '!' are not stripped as narrator beats."""
    gate = IngestionGate(min_words=10, max_words=60, min_distinctiveness=0.40)
    text = "– Bio je to – zar ne misliš tako? – zaista neobičan i opasan susret u zamku."
    cands = gate.extract_candidates(text)
    assert len(cands) == 1
    assert "zar ne misliš tako?" in cands[0].text


def test_serbian_german_quotes_and_attribution_window_no_bleed():
    """Supports both „…“ and „…” quotes and prevents next „ quote from bleeding into after_win."""
    from canonarc import CanonArcEngine, EntityRecord

    gate = IngestionGate(min_words=10, max_words=60, min_distinctiveness=0.40)

    # Serbian/Croatian/Polish „…” closing variant (U+201D)
    sr_right_quote = "Artur šapnu: „Ne mešajte se u opasne stvari koje uopšte ne razumete, inače ćete zažaliti.”"
    cands_right = gate.extract_candidates(sr_right_quote)
    assert len(cands_right) == 1
    assert cands_right[0].text.startswith("Ne mešajte se")

    # Attribution window must stop at opening „ of the next quote
    scene = (
        "„Ne verujem da ćemo ikada pronaći izlaz iz ove mračne i zaboravljene tvrđave!“ "
        "„Elena, moramo odmah požuriti pre nego što stražari zatvore sve gradske kapije večeras!“ viknu Marko."
    )
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_elena", canonical_name="Elena"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_marko", canonical_name="Marko"), step=0)

    cands = engine.extract_dialogue_candidates(scene, step=1)
    assert len(cands) == 2
    # First quote has no speaker after it; Elena inside the second quote MUST NOT bleed into its attribution
    assert cands[0].speaker_id is None
    assert cands[0].addressee_id is None
    # Second quote is attributed to Marko
    assert cands[1].speaker_id == "char_marko"
    engine.close()


def test_inward_and_outward_guillemets_and_mixed_quotes():
    """Handles French/Russian «…», Croatian/German »…«, and mixed „…“ + straight quotes without collisions."""
    gate = IngestionGate(min_words=10, max_words=60, min_distinctiveness=0.40)

    # French outward guillemets: narrator text between quotes must NOT be matched
    fr_text = (
        "«Le vieux parchemin révèle un présage sombre qui menace toute la cité royale.» "
        "murmura Merlin lentement. "
        "«Nous devons agir ensemble avant la tombée de la nuit glaciale.»"
    )
    fr_cands = gate.extract_candidates(fr_text)
    assert len(fr_cands) == 2
    assert all("murmura Merlin" not in c.text for c in fr_cands)

    # Croatian/German inward guillemets »…«
    hr_text = (
        "»Stari pergament otkriva mračno proročanstvo koje ugrožava čitavo naše kraljevstvo.« "
        "reče Artur tiho. "
        "»Moramo odmah upozoriti stražare na severnim kulama pre nego što padne noć.«"
    )
    hr_cands = gate.extract_candidates(hr_text)
    assert len(hr_cands) == 2
    assert all("reče Artur" not in c.text for c in hr_cands)

    # Mixed „…“ and straight "..." quotes in the same paragraph must not cross-match
    mixed = (
        '„Drevni zapis jasno upozorava na veliku opasnost koja preti našoj tvrđavi.“ '
        'reče Artur, '
        '"The ancient parchment reveals an ominous portent that threatens sovereign Camelot."'
    )
    mixed_cands = gate.extract_candidates(mixed)
    assert len(mixed_cands) == 2
    assert all("reče Artur" not in c.text for c in mixed_cands)
