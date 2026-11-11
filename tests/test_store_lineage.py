import tempfile
from pathlib import Path
import pytest
from narrative_memory.store import CausalStore
from narrative_memory.models import EntityRecord, StepDelta, EdgeDelta, EntityDelta


def test_lineage_resolution_and_branching():
    store = CausalStore(":memory:")

    # Step 0: Seed Arthur and Merlin on 'main'
    store.register_entity(
        EntityRecord(
            entity_id="char_arthur",
            canonical_name="Arthur",
            base_attributes={"gender": "M"},
            state={"mood": "calm"},
        ),
        step=0,
        timeline_id="main",
    )
    store.register_entity(
        EntityRecord(
            entity_id="char_merlin",
            canonical_name="Merlin",
            base_attributes={"gender": "M"},
        ),
        aliases=["the old wizard"],
        step=0,
        timeline_id="main",
    )

    # Step 10: Arthur -> Merlin (formal, respectful)
    store.apply_delta(
        step=10,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    state={"register": "formal", "tone": "respectful"},
                )
            ]
        ),
        timeline_id="main",
    )

    # Fork branch 'route_b' at step 50
    store.create_timeline("route_b", parent_id="main", fork_step=50)

    # Step 82 on 'main': Arthur becomes furious and switches to informal
    store.apply_delta(
        step=82,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"mood": "furious"})],
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    state={"register": "informal", "tone": "furious"},
                )
            ],
        ),
        timeline_id="main",
    )

    # Step 55 on 'route_b': Arthur becomes warm and friendly
    store.apply_delta(
        step=55,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"mood": "warm"})],
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    state={"register": "formal", "tone": "warm"},
                )
            ],
        ),
        timeline_id="route_b",
    )

    # Query main at step 85: sees furious state
    main_art = store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=85)
    assert main_art is not None
    assert main_art.state["mood"] == "furious"
    main_edge = store.get_active_edge("char_arthur", "char_merlin", timeline_id="main", cutoff_step=85)
    assert main_edge is not None
    assert main_edge.state["register"] == "informal"
    assert main_edge.state["tone"] == "furious"

    # Query route_b at step 85: MUST ignore main's step 82 mutation!
    b_art = store.get_active_entity("char_arthur", timeline_id="route_b", cutoff_step=85)
    assert b_art is not None
    assert b_art.state["mood"] == "warm"
    b_edge = store.get_active_edge("char_arthur", "char_merlin", timeline_id="route_b", cutoff_step=85)
    assert b_edge is not None
    assert b_edge.state["register"] == "formal"
    assert b_edge.state["tone"] == "warm"

    # Verify 1-hop alias inherited across branches
    aliases_b = store.get_all_active_aliases("route_b", cutoff_step=85)
    assert any(a.alias == "the old wizard" and a.entity_id == "char_merlin" for a in aliases_b)


def test_rollback_to():
    store = CausalStore(":memory:")
    store.register_entity(
        EntityRecord(entity_id="char_arthur", canonical_name="Arthur", state={"mood": "calm"}),
        step=0,
    )
    store.apply_delta(
        step=10,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_arthur", state={"mood": "tense"})]),
    )
    store.apply_delta(
        step=20,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_arthur", state={"mood": "furious"})]),
    )

    assert store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=25).state["mood"] == "furious"

    # Rollback to step 15 (deletes step 20)
    store.rollback_to(step=15, timeline_id="main")
    art = store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=25)
    assert art.state["mood"] == "tense"

    # Rollback to step 5 (deletes step 10)
    store.rollback_to(step=5, timeline_id="main")
    art0 = store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=25)
    assert art0.state["mood"] == "calm"


def test_export_snapshot():
    store = CausalStore(":memory:")
    store.register_entity(
        EntityRecord(entity_id="hero", canonical_name="Hero", state={"level": 5}),
        step=0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "slot_1.sqlite"
        store.export_snapshot(snap_path, at_step=10)
        assert snap_path.exists()

        snap_store = CausalStore(snap_path)
        hero = snap_store.get_active_entity("hero", timeline_id="main", cutoff_step=1)
        assert hero is not None
        assert hero.state["level"] == 5
        snap_store.close()
