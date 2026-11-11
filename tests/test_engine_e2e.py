from narrative_memory.engine import NarrativeMemoryEngine
from narrative_memory.models import (
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    StepDelta,
    VoiceExemplar,
)
from narrative_memory.presets import TranslationPreset


def test_unified_chunk_85_scenario():
    """Validates the exact Chunk 85 Arthur vs. Merlin scenario from the user request."""
    engine = NarrativeMemoryEngine(":memory:", preset=TranslationPreset())

    # Tier 1: Static Attributes
    engine.register_entity(
        EntityRecord(
            entity_id="char_arthur",
            canonical_name="Arthur",
            base_attributes={"target_name": "Artur", "gender": "M", "declension": "type_1"},
        ),
        step=0,
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_merlin",
            canonical_name="Merlin",
            base_attributes={"target_name": "Merlin", "gender": "M", "declension": "type_1"},
        ),
        aliases=["the old wizard"],
        step=0,
    )

    # Step 10: Arthur -> Merlin initial edge: formal (Vi)
    engine.commit_step(
        step=10,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    state={"register": "Vi", "tone": "calm"},
                )
            ]
        ),
    )

    # Seed past dialogue lines for Arthur in both Vi and ti registers
    ex_vi = VoiceExemplar(
        step=15,
        speaker_id="char_arthur",
        text="Molim vas da me saslušate pažljivo, jer kraljevstvo visi o koncu sudbine.",
        metadata={"register": "Vi", "tone": "calm"},
        quality_score=9.2,
    )
    ex_ti_angry = VoiceExemplar(
        step=50,
        speaker_id="char_arthur",
        text="Prestani sa lažima, starče, pre nego što izgubim i poslednje zrno strpljenja!",
        metadata={"register": "ti", "tone": "angry/confrontational"},
        quality_score=9.5,
    )
    engine.index_voice_exemplars(15, [ex_vi])
    engine.index_voice_exemplars(50, [ex_ti_angry])

    # Step 82: Argument occurs -> Arthur's mood becomes furious and register shifts to ti
    engine.commit_step(
        step=82,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"mood": "furious"})],
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    state={"register": "ti", "tone": "angry/confrontational"},
                )
            ],
        ),
    )

    # Now execute Chunk 85
    # Text mentions Arthur and "the old wizard"
    chunk_85_text = (
        'Arthur glared at the old wizard with clenched fists, his patience shattered. '
        '"You deceived me from the very start!"'
    )

    ctx = engine.retrieve_context(
        step=85,
        text=chunk_85_text,
        mentions=["Arthur", "the old wizard"],
        top_k_voice=2,
    )

    # 1. 1-Hop Resolution check: "the old wizard" -> char_merlin
    matched_ids = {m.entity_id for m in ctx.resolved_mentions}
    assert "char_arthur" in matched_ids
    assert "char_merlin" in matched_ids

    # 2. Tier 1 base attributes
    merlin_node = next(e for e in ctx.entities if e.entity_id == "char_merlin")
    assert merlin_node.base_attributes["declension"] == "type_1"

    # 3. Temporal edge lookup at chunk 85: register == ti, tone == angry/confrontational
    art_to_merlin = next(
        edge for edge in ctx.active_edges
        if edge.source_id == "char_arthur" and edge.target_id == "char_merlin"
    )
    assert art_to_merlin.state["register"] == "ti"
    assert art_to_merlin.state["tone"] == "angry/confrontational"

    # 4. Voice Bank Retrieval: Arthur has register='ti' -> MUST retrieve ex_ti_angry, NOT ex_vi!
    assert len(ctx.voice_exemplars) == 1
    assert ctx.voice_exemplars[0].metadata["register"] == "ti"
    assert "Prestani sa lažima" in ctx.voice_exemplars[0].text

    # 5. Formatted context check
    prompt_str = engine.format_context(ctx)
    assert "## Scene character state & register — binding" in prompt_str
    assert "register=ti, tone=angry/confrontational" in prompt_str
    assert "Prestani sa lažima" in prompt_str


def test_step_transaction():
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="alice", canonical_name="Alice", state={"mood": "happy"}),
        step=0,
    )

    with engine.step_transaction(step=10, text="Alice entered the room.") as tx:
        assert tx.context.entities[0].state["mood"] == "happy"
        tx.commit(
            delta=StepDelta(
                entities=[EntityDelta(entity_id="alice", state={"mood": "surprised"})]
            )
        )

    # After transaction, step 11 sees the updated state
    ctx11 = engine.retrieve_context(step=11)
    assert ctx11.entities[0].state["mood"] == "surprised"
