from narrative_memory.voice import IngestionGate, OfflineNgramEmbedder, VoiceBank
from narrative_memory.models import VoiceExemplar


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
        "from narrative_memory.voice import OfflineNgramEmbedder; "
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
    from narrative_memory import EntityRecord, NarrativeMemoryEngine

    engine = NarrativeMemoryEngine(":memory:")
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
