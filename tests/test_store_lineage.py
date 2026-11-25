import tempfile
from pathlib import Path

from canonarc.models import EdgeDelta, EntityDelta, EntityRecord, StepDelta
from canonarc.store import CausalStore


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


def test_multigraph_independent_edge_lifecycles():
    store = CausalStore(":memory:")
    store.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    store.register_entity(EntityRecord(entity_id="char_kay", canonical_name="Sir Kay"), step=0)

    # Step 1: Kinship edge
    store.apply_delta(
        step=1,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_kay",
                    relation_type="kinship",
                    state={"relation": "foster_brother"},
                )
            ]
        ),
    )

    # Step 10: Financial debt edge
    store.apply_delta(
        step=10,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_kay",
                    relation_type="debt",
                    state={"gold": 100},
                )
            ]
        ),
    )

    # Step 20: Conversational stance edge
    store.apply_delta(
        step=20,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_kay",
                    relation_type="stance",
                    state={"register": "informal", "tone": "warm"},
                )
            ]
        ),
    )

    # Step 25 assertions: 3 distinct typed edges exist concurrently
    kinship = store.get_active_edge("char_arthur", "char_kay", relation_type="kinship", timeline_id="main", cutoff_step=25)
    assert kinship is not None
    assert kinship.relation_type == "kinship"
    assert kinship.state["relation"] == "foster_brother"
    assert kinship.valid_from_step == 1

    debt = store.get_active_edge("char_arthur", "char_kay", relation_type="debt", timeline_id="main", cutoff_step=25)
    assert debt is not None
    assert debt.relation_type == "debt"
    assert debt.state["gold"] == 100
    assert debt.valid_from_step == 10

    stance = store.get_active_edge("char_arthur", "char_kay", relation_type="stance", timeline_id="main", cutoff_step=25)
    assert stance is not None
    assert stance.relation_type == "stance"
    assert stance.state["tone"] == "warm"
    assert stance.valid_from_step == 20

    all_kay_edges = store.get_active_edges_between("char_arthur", "char_kay", timeline_id="main", cutoff_step=25)
    assert len(all_kay_edges) == 3

    # Step 30: Mutate conversational stance only
    store.apply_delta(
        step=30,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_kay",
                    relation_type="stance",
                    state={"tone": "annoyed"},
                )
            ]
        ),
    )

    # Step 35 assertions: stance valid_from_step updated, but kinship valid_from_step is NOT clobbered
    new_stance = store.get_active_edge("char_arthur", "char_kay", relation_type="stance", timeline_id="main", cutoff_step=35)
    assert new_stance.state["tone"] == "annoyed"
    assert new_stance.state["register"] == "informal"  # merged
    assert new_stance.valid_from_step == 30

    kinship_after = store.get_active_edge("char_arthur", "char_kay", relation_type="kinship", timeline_id="main", cutoff_step=35)
    assert kinship_after.valid_from_step == 1

    # Step 40: Pay off debt (tombstone ONLY debt relation)
    store.apply_delta(
        step=40,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_kay",
                    relation_type="debt",
                    remove=True,
                )
            ]
        ),
    )

    # Step 45 assertions: debt is gone, kinship and stance remain active!
    assert store.get_active_edge("char_arthur", "char_kay", relation_type="debt", timeline_id="main", cutoff_step=45) is None
    assert store.get_active_edge("char_arthur", "char_kay", relation_type="kinship", timeline_id="main", cutoff_step=45) is not None
    assert store.get_active_edge("char_arthur", "char_kay", relation_type="stance", timeline_id="main", cutoff_step=45) is not None

    # Filtered active edges query
    kinship_only = store.get_all_active_edges("main", cutoff_step=45, relation_type="kinship")
    assert len(kinship_only) == 1
    assert kinship_only[0].relation_type == "kinship"

    # Export snapshot and verify typed edges are preserved
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "multigraph_snap.sqlite"
        store.export_snapshot(snap_path, at_step=45)
        snap_store = CausalStore(snap_path)
        snap_kinship = snap_store.get_active_edge("char_arthur", "char_kay", relation_type="kinship", timeline_id="main", cutoff_step=1)
        assert snap_kinship is not None
        assert snap_kinship.relation_type == "kinship"
        assert snap_kinship.state["relation"] == "foster_brother"
        snap_store.close()


def test_bounded_episode_loading_and_sql_prefilter():
    """get_lineage_episodes(limit=...) bounds rows loaded while preserving pinned, cue, FTS, and POV matches."""
    from canonarc.engine import CanonArcEngine
    from canonarc.models import EpisodeRecord, InformDelta

    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)

    # Insert an old episode at step 1 that should be found via FTS + cue even after 300 filler episodes
    engine.store.insert_episodes(
        [
            (
                EpisodeRecord(
                    episode_id="ep_old_sword",
                    timeline_id="main",
                    step=1,
                    summary="Arthur forged the enchanted broadsword in the hidden smithy.",
                    location_id="loc_smithy",
                    salience=5.0,
                    participants={"char_arthur": "smith"},
                    witnesses=[],
                    known_by={"char_arthur": "direct"},
                ),
                b"",
            ),
            (
                EpisodeRecord(
                    episode_id="ep_secret_merlin",
                    timeline_id="main",
                    step=2,
                    summary="Merlin hid the ancient crown.",
                    salience=9.0,
                    participants={"char_merlin": "actor"},
                    known_by={"char_merlin": "direct"},
                ),
                b"",
            ),
        ],
        timeline_id="main",
    )

    # Insert 300 newer filler episodes
    filler = [
        (
            EpisodeRecord(
                episode_id=f"ep_filler_{i}",
                timeline_id="main",
                step=10 + i,
                summary=f"Routine patrol number {i} along the eastern wall.",
                salience=3.0,
                participants={"char_merlin": "observer"},
                known_by={"char_arthur": "direct", "char_merlin": "direct"},
            ),
            b"",
        )
        for i in range(300)
    ]
    engine.store.insert_episodes(filler, timeline_id="main")

    # Bounded query with limit=10: must still surface ep_old_sword via FTS + cue, and exclude ep_secret_merlin for Arthur POV
    bounded = engine.store.get_lineage_episodes(
        "main",
        400,
        limit=10,
        pov_entity_ids=["char_arthur"],
        pov_mode="any",
        cue_entity_ids=["char_arthur"],
        query_text="enchanted broadsword smithy",
    )
    assert len(bounded) == 10
    bounded_ids = {ep.episode_id for ep, _ in bounded}
    assert "ep_old_sword" in bounded_ids
    assert "ep_secret_merlin" not in bounded_ids

    # Forgotten override in branch should be respected by get_known_episode_ids & POV-filtered get_lineage_episodes
    engine.branch(350, "alt", from_timeline="main")
    engine.store.record_epistemic(
        InformDelta(episode_id="ep_old_sword", entity_id="char_arthur", epistemic="forgotten"),
        step=360,
        timeline_id="alt",
    )
    assert "ep_old_sword" not in engine.store.get_known_episode_ids(
        "alt", 400, ["char_arthur"], episode_ids=["ep_old_sword", "ep_filler_0"]
    )
    assert "ep_filler_0" in engine.store.get_known_episode_ids(
        "alt", 400, ["char_arthur"], episode_ids=["ep_old_sword", "ep_filler_0"]
    )

    # End-to-end retrieve_context() on main with > 200 episodes still recalls ep_old_sword
    ctx = engine.retrieve_context(
        400,
        text="Arthur asked about the enchanted broadsword.",
        mentions=["char_arthur"],
        pov_entity_ids=["char_arthur"],
        top_k_episodes=3,
    )
    assert any(ep.episode_id == "ep_old_sword" for ep in ctx.episodes)
    engine.close()
