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
