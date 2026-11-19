import tempfile
from pathlib import Path

from canonarc import (
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    CanonArcEngine,
    StepDelta,
)


def test_pitfall_1_and_5_episodic_survival_and_trajectory_after_mood_reset():
    """Tests that when Arthur's mood returns to 'calm' at Step 90, the Step 82 argument
    is still preserved in StepContext.episodes AND trajectory metadata (prior_state/revision_count)
    records the prior furious state without stuffing state dicts."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="char_arthur", canonical_name="Arthur", state={"mood": "calm"}),
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_merlin", canonical_name="Merlin"),
        step=0,
    )

    # Step 82: Argument occurs -> mood becomes furious + episode recorded
    engine.commit_step(
        step=82,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"mood": "furious"})],
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    state={"register": "informal", "tone": "hostile"},
                )
            ],
            episodes=[
                EpisodeRecord(
                    episode_id="ep_arg_82",
                    predicate="argument",
                    summary="Arthur and Merlin had a bitter argument over the stolen crown.",
                    participants={"char_arthur": "actor", "char_merlin": "target"},
                    salience=8.5,
                )
            ],
        ),
    )

    # Step 90: Arthur cools down -> mood returns to "calm", stance returns to "formal/neutral"
    engine.commit_step(
        step=90,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"mood": "calm"})],
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    state={"register": "formal", "tone": "neutral"},
                )
            ],
        ),
    )

    # Step 95: Query when Arthur and Merlin meet again
    ctx = engine.retrieve_context(
        step=95,
        text="Arthur looked across the council table at Merlin.",
        mentions=["Arthur", "Merlin"],
    )

    # 1. State dict is clean (only {'mood': 'calm'}, no stuffed keys)
    arthur = next(e for e in ctx.entities if e.entity_id == "char_arthur")
    assert arthur.state == {"mood": "calm"}

    # 2. Trajectory metadata (Pitfall 5) tracks the prior state and revision count
    assert arthur.prior_state == {"mood": "furious"}
    assert arthur.prior_step == 82
    assert arthur.revision_count == 3  # step 0, 82, 90

    edge = ctx.active_edges[0]
    assert edge.state == {"register": "formal", "tone": "neutral"}
    assert edge.prior_state == {"register": "informal", "tone": "hostile"}
    assert edge.prior_step == 82
    assert edge.revision_count == 2  # step 82, 90

    # 3. Episodic memory (Pitfall 1) still remembers the Step 82 argument!
    assert len(ctx.episodes) == 1
    assert ctx.episodes[0].episode_id == "ep_arg_82"
    assert ctx.episodes[0].step == 82
    assert "bitter argument" in ctx.episodes[0].summary

    # 4. Prompt formatting includes both trajectory and episodic memory
    prompt = engine.format_context(ctx)
    assert "(was: furious at step 82)" in prompt
    assert "[Step 82 | argument | known to: char_arthur, char_merlin]" in prompt


def test_pitfall_2_causal_provenance_pinning():
    """Tests that caused_by_episode_id and reason link active states to their triggering episode
    and automatically pin that episode in StepContext even among many competing episodes."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_lancelot", canonical_name="Lancelot"), step=0)

    # Step 42: Betrayal causes broken trust
    engine.commit_step(
        step=42,
        delta=StepDelta(
            episodes=[
                EpisodeRecord(
                    episode_id="ep_betrayal_42",
                    predicate="betrayal",
                    summary="Arthur sabotaged Lancelot's lance at the royal tournament.",
                    participants={"char_arthur": "actor", "char_lancelot": "target"},
                    salience=9.5,
                )
            ],
            edges=[
                EdgeDelta(
                    source_id="char_lancelot",
                    target_id="char_arthur",
                    relation_type="stance",
                    state={"trust": "broken", "tone": "cold"},
                    caused_by_episode_id="ep_betrayal_42",
                    reason="Arthur sabotaged his lance at the tournament",
                )
            ],
        ),
    )

    # Add 5 unrelated recent minor episodes involving other characters so top_k=1 must rely on relevance/pinning
    for s in range(50, 55):
        engine.index_episodes(
            step=s,
            episodes=[
                EpisodeRecord(
                    predicate="feast",
                    summary=f"The court held a routine banquet on day {s}.",
                    participants={"char_gawain": "actor"},
                    salience=7.0,
                )
            ],
        )

    ctx = engine.retrieve_context(
        step=120,
        text="Lancelot stood silently before Arthur.",
        mentions=["Lancelot", "Arthur"],
        top_k_episodes=1,
    )

    edge = ctx.active_edges[0]
    assert edge.caused_by_episode_id == "ep_betrayal_42"
    assert edge.reason == "Arthur sabotaged his lance at the tournament"

    # The pinned causal episode is guaranteed to be top-1
    assert len(ctx.episodes) == 1
    assert ctx.episodes[0].episode_id == "ep_betrayal_42"


def test_pitfall_3_n_ary_event_hyperedge_and_structured_queries():
    """Tests N-ary episodes involving >2 entities, artifacts, locations, and witnesses."""
    engine = CanonArcEngine(":memory:")
    for eid, name, kind in [
        ("char_arthur", "Arthur", "character"),
        ("char_lancelot", "Lancelot", "character"),
        ("char_merlin", "Merlin", "character"),
        ("item_excalibur", "Excalibur", "artifact"),
        ("loc_camelot", "Camelot", "location"),
    ]:
        engine.register_entity(EntityRecord(entity_id=eid, canonical_name=name, kind=kind), step=0)

    # Step 42: 5-way N-ary transfer episode
    engine.index_episodes(
        step=42,
        episodes=[
            EpisodeRecord(
                episode_id="ep_sword_transfer",
                predicate="item_transfer",
                summary="Arthur entrusted Excalibur to Lancelot at Camelot before Merlin.",
                location_id="loc_camelot",
                participants={
                    "char_arthur": "giver",
                    "item_excalibur": "instrument",
                    "char_lancelot": "recipient",
                },
                witnesses=["char_merlin"],
                salience=9.0,
            )
        ],
    )

    # Query by co-participation (Arthur + Excalibur + Lancelot)
    matches = engine.query_episodes(
        participants=["char_arthur", "item_excalibur", "char_lancelot"],
        match_all_participants=True,
    )
    assert len(matches) == 1
    assert matches[0].episode_id == "ep_sword_transfer"
    assert matches[0].participants["item_excalibur"] == "instrument"
    assert matches[0].location_id == "loc_camelot"


def test_pitfall_4_asymmetric_epistemic_memory_and_secondhand_inform():
    """Tests that characters only recall episodes they witnessed or were told about (no metagaming)."""
    engine = CanonArcEngine(":memory:")
    for eid, name in [
        ("char_arthur", "Arthur"),
        ("char_lancelot", "Lancelot"),
        ("char_guinevere", "Guinevere"),
        ("char_percival", "Percival"),
    ]:
        engine.register_entity(EntityRecord(entity_id=eid, canonical_name=name), step=0)

    # Step 42: Secret dungeon betrayal known ONLY to Arthur and Lancelot
    engine.index_episodes(
        step=42,
        episodes=[
            EpisodeRecord(
                episode_id="ep_secret_dungeon",
                predicate="betrayal",
                summary="Arthur locked Lancelot in the secret dungeon.",
                participants={"char_arthur": "actor", "char_lancelot": "target"},
                salience=9.5,
            )
        ],
    )

    # Step 50: Query from Percival's POV -> Percival was NOT in the room, MUST NOT see the secret episode!
    ctx_percival = engine.retrieve_context(
        step=50,
        text="Percival greeted Arthur in the courtyard.",
        mentions=["Percival", "Arthur"],
        pov_entity_ids=["char_percival"],
    )
    assert len(ctx_percival.episodes) == 0

    # Step 50: Query from Guinevere's POV before she is told -> empty
    ctx_guin_before = engine.retrieve_context(
        step=50,
        text="Guinevere spoke with Arthur.",
        mentions=["Guinevere", "Arthur"],
        pov_entity_ids=["char_guinevere"],
    )
    assert len(ctx_guin_before.episodes) == 0

    # Step 60: Lancelot tells Guinevere about the dungeon betrayal
    engine.commit_step(
        step=60,
        delta=StepDelta(
            inform=[
                InformDelta(
                    episode_id="ep_secret_dungeon",
                    entity_id="char_guinevere",
                    epistemic="told",
                )
            ]
        ),
    )

    # Step 65: Query from Guinevere's POV -> Now she knows ('told')!
    ctx_guin_after = engine.retrieve_context(
        step=65,
        text="Guinevere confronted Arthur.",
        mentions=["Guinevere", "Arthur"],
        pov_entity_ids=["char_guinevere"],
    )
    assert len(ctx_guin_after.episodes) == 1
    assert ctx_guin_after.episodes[0].episode_id == "ep_secret_dungeon"
    assert ctx_guin_after.episodes[0].known_by["char_guinevere"] == "told"
    assert ctx_guin_after.episodes[0].known_by["char_arthur"] == "direct"

    # Causal check: Querying Guinevere's POV back at Step 55 still sees nothing (no spoiler leakage!)
    ctx_guin_past = engine.retrieve_context(
        step=55,
        text="Guinevere walked through the garden.",
        mentions=["Guinevere"],
        pov_entity_ids=["char_guinevere"],
    )
    assert len(ctx_guin_past.episodes) == 0


def test_pitfall_6_cue_based_location_and_semantic_plot_recall():
    """Tests that returning to a location or mentioning semantic plot cues at Step 300
    surfaces a high-salience episode from Step 42 over low-salience recent episodes."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(
        EntityRecord(entity_id="loc_tournament", canonical_name="tournament grounds", kind="location"),
        step=0,
    )

    # Step 42: High-salience tournament trauma
    engine.index_episodes(
        step=42,
        episodes=[
            EpisodeRecord(
                episode_id="ep_tournament_42",
                predicate="betrayal",
                summary="Arthur shattered the silver lance during the joust at the tournament grounds.",
                location_id="loc_tournament",
                participants={"char_arthur": "actor"},
                salience=9.8,
            )
        ],
    )

    # Steps 250..255: Low-salience mundane episodes
    for s in range(250, 255):
        engine.index_episodes(
            step=s,
            episodes=[
                EpisodeRecord(
                    episode_id=f"ep_mundane_{s}",
                    predicate="routine",
                    summary=f"Arthur inspected the granary ledger on morning {s}.",
                    participants={"char_arthur": "actor"},
                    salience=1.5,
                )
            ],
        )

    # Step 300: Arthur walks back into the tournament grounds and sees a shattered silver lance
    ctx = engine.retrieve_context(
        step=300,
        text="Arthur walked onto the tournament grounds and noticed a shattered silver lance in the mud.",
        top_k_episodes=1,
    )
    assert len(ctx.episodes) == 1
    assert ctx.episodes[0].episode_id == "ep_tournament_42"


def test_episodic_time_travel_branching_rollback_and_snapshot():
    """Verifies that Tier 6 episodes and epistemics obey timeline branching, rollback_to, and export_snapshot."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)

    # Step 20: Shared episode on 'main'
    engine.index_episodes(
        step=20,
        episodes=[
            EpisodeRecord(
                episode_id="ep_shared_20",
                predicate="coronation",
                summary="Arthur drew the sword from the stone.",
                participants={"char_arthur": "actor"},
                salience=10.0,
            )
        ],
    )

    # Fork 'peace_route' at step 50
    engine.branch(at_step=50, name="peace_route")

    # Step 82 on 'main': War declared
    engine.index_episodes(
        step=82,
        episodes=[
            EpisodeRecord(
                episode_id="ep_war_82",
                predicate="war",
                summary="Arthur declared war on the northern barons.",
                participants={"char_arthur": "actor"},
                salience=9.0,
            )
        ],
        timeline_id="main",
    )

    # 'peace_route' at Step 90 sees ep_shared_20, but NOT ep_war_82!
    peace_eps = engine.query_episodes(cutoff_step=90, timeline_id="peace_route")
    assert [e.episode_id for e in peace_eps] == ["ep_shared_20"]

    # 'main' at Step 90 sees both
    main_eps = engine.query_episodes(cutoff_step=90, timeline_id="main")
    assert [e.episode_id for e in main_eps] == ["ep_shared_20", "ep_war_82"]

    # Export snapshot of 'main' at Step 90 and verify episodes are persisted
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "save_90.sqlite"
        engine.export_snapshot(snap_path, at_step=90, timeline_id="main")
        loaded = CanonArcEngine(snap_path)
        snap_eps = loaded.query_episodes(cutoff_step=100)
        assert {e.episode_id for e in snap_eps} == {"ep_shared_20", "ep_war_82"}
        loaded.close()

    # Rollback 'main' to Step 50 -> ep_war_82 is atomically removed
    engine.rollback_to(step=50, timeline_id="main")
    main_eps_after = engine.query_episodes(cutoff_step=90, timeline_id="main")
    assert [e.episode_id for e in main_eps_after] == ["ep_shared_20"]
