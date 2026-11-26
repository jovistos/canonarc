from __future__ import annotations

import json
from pathlib import Path

import pytest

from canonarc import (
    AliasRecord,
    CanonArcEngine,
    DuplicateCandidate,
    EdgeDelta,
    EntityDelta,
    EntityMergeReport,
    EntityRecord,
    EpisodeRecord,
    StepDelta,
    VoiceExemplar,
)
from canonarc.cli import main as cli_main


def test_merge_promotes_names_aliases_and_tombstones_duplicate() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_hercule_poirot",
                kind="character",
                canonical_name="Hercule Poirot",
                base_attributes={"gender": "male", "target_name": "Herkul Poaro", "target_lang": "sr"},
                state={"mood": "analytical"},
            ),
            aliases=[AliasRecord(alias="the Belgian detective", entity_id="char_hercule_poirot", kind="epithet")],
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_poirot",
                kind="character",
                canonical_name="Mr. Poirot",
                base_attributes={"archetype": "detective", "target_name": "Gospodin Poaro", "target_lang": "sr"},
                state={"location_id": "loc_train", "mood": "suspicious"},
            ),
            aliases=[
                AliasRecord(alias="the little man", entity_id="char_poirot", kind="epithet"),
                AliasRecord(alias="Poaroa", entity_id="char_poirot", kind="inflection", lang="sr"),
            ],
            step=1,
        )

        report = engine.merge_entities("char_poirot", "char_hercule_poirot", step=2)
        assert isinstance(report, EntityMergeReport)
        assert report.duplicate_id == "char_poirot"
        assert report.canonical_id == "char_hercule_poirot"
        assert report.step == 2
        assert "Mr. Poirot" in report.aliases_added
        assert "Gospodin Poaro" in report.aliases_added
        assert "the little man" in report.aliases_added
        assert "Poaroa" in report.aliases_added
        assert set(report.aliases_tombstoned) == {"the little man", "Poaroa", "Gospodin Poaro"}
        assert "archetype" in report.base_attributes_merged
        assert "location_id" in report.state_keys_merged

        # Duplicate is tombstoned, canonical survives with backfilled attributes/state (canonical wins on mood)
        active_entities = engine.store.get_all_active_entities("main", 2)
        active_ids = {e.entity_id for e in active_entities}
        assert "char_hercule_poirot" in active_ids
        assert "char_poirot" not in active_ids

        canon = next(e for e in active_entities if e.entity_id == "char_hercule_poirot")
        assert canon.base_attributes["archetype"] == "detective"
        assert canon.base_attributes["gender"] == "male"
        assert canon.base_attributes["merged_from"] == ["char_poirot"]
        assert canon.state["mood"] == "analytical"  # canonical_wins
        assert canon.state["location_id"] == "loc_train"

        # Old aliases on duplicate_id are tombstoned; no active alias points to char_poirot
        active_aliases = engine.store.get_all_active_aliases("main", 2)
        assert all(a.entity_id != "char_poirot" for a in active_aliases)
        canon_alias_map = {a.alias: a for a in active_aliases if a.entity_id == "char_hercule_poirot"}
        assert canon_alias_map["Mr. Poirot"].kind == "permanent"
        assert canon_alias_map["Gospodin Poaro"].lang == "sr"
        assert canon_alias_map["Poaroa"].kind == "inflection"
        assert canon_alias_map["Poaroa"].lang == "sr"

        # Lexical resolution resolves all former duplicate names/aliases to canonical_id
        ctx = engine.retrieve_context(3, text="Mr. Poirot spoke to the little man about Poaroa.")
        resolved_ids = {m.entity_id for m in ctx.resolved_mentions}
        assert resolved_ids == {"char_hercule_poirot"}

        # Tombstone history records merged_into provenance
        hist = engine.get_history("char_poirot", record_type="entity")
        assert hist[-1]["is_tombstone"] is True
        assert hist[-1]["payload"]["merged_into"] == "char_hercule_poirot"


def test_merge_normalize_delta_redirects_retired_id() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0
        )
        engine.register_entity(
            EntityRecord(entity_id="char_stranger", canonical_name="the stranger"), step=1
        )
        engine.merge_entities("char_stranger", "char_arthur", step=2)

        raw = StepDelta(
            entities=[EntityDelta(entity_id="char_stranger", state={"mood": "determined"})],
            edges=[EdgeDelta(source_id="char_stranger", target_id="char_merlin", state={"register": "ti"})],
        )
        norm = engine.normalize_delta(raw, step=3)
        assert norm.entities[0].entity_id == "char_arthur"
        assert norm.edges[0].source_id == "char_arthur"


def test_merge_edge_migration_collision_self_loop_and_audit_compat() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(entity_id="char_darcy", canonical_name="Fitzwilliam Darcy"),
            aliases=[AliasRecord(alias="Mr. Darcy", entity_id="char_darcy", kind="epithet")],
            step=0,
        )
        for eid, name in [
            ("char_mr_darcy", "Mr. Darcy"),
            ("char_elizabeth", "Elizabeth Bennet"),
            ("char_bingley", "Charles Bingley"),
        ]:
            engine.register_entity(EntityRecord(entity_id=eid, canonical_name=name), step=0)

        engine.apply_delta(
            1,
            StepDelta(
                edges=[
                    # Self-loop candidate (between duplicate and canonical)
                    EdgeDelta(source_id="char_mr_darcy", target_id="char_darcy", state={"tone": "neutral"}),
                    # Unique edge only on duplicate -> should migrate cleanly
                    EdgeDelta(source_id="char_mr_darcy", target_id="char_bingley", state={"register": "ti", "tone": "warm"}),
                    EdgeDelta(source_id="char_bingley", target_id="char_mr_darcy", state={"register": "ti"}),
                    # Colliding edge with Elizabeth (different register)
                    EdgeDelta(source_id="char_darcy", target_id="char_elizabeth", state={"register": "Vi", "trust": "low"}),
                    EdgeDelta(source_id="char_mr_darcy", target_id="char_elizabeth", state={"register": "ti", "admiration": "high"}),
                ]
            ),
        )

        # Audit detects near-duplicate before merge
        pre_audit = engine.audit_step(0, include_rules=["near_duplicate_entity"])
        assert any(i.code == "near_duplicate_entity" for i in pre_audit.issues)

        report = engine.merge_entities("char_mr_darcy", "char_darcy", step=2)
        assert len(report.self_loops_dropped) == 1
        assert "char_darcy:stance->char_bingley" in report.edges_migrated
        assert "char_bingley:stance->char_darcy" in report.edges_migrated
        assert "char_darcy:stance->char_elizabeth" in report.edges_merged_on_collision
        assert any("register" in w for w in report.warnings)

        active_edges = engine.store.get_all_active_edges("main", 2)
        assert all(e.source_id != "char_mr_darcy" and e.target_id != "char_mr_darcy" for e in active_edges)
        assert all(e.source_id != e.target_id for e in active_edges)

        by_pair = {(e.source_id, e.target_id, e.relation_type): e for e in active_edges}
        assert by_pair[("char_darcy", "char_bingley", "stance")].state == {"register": "ti", "tone": "warm"}
        assert by_pair[("char_bingley", "char_darcy", "stance")].state == {"register": "ti"}
        # Colliding edge kept canonical register='Vi' and backfilled admiration='high'
        elizabeth_edge = by_pair[("char_darcy", "char_elizabeth", "stance")]
        assert elizabeth_edge.state == {"register": "Vi", "trust": "low", "admiration": "high"}
        assert elizabeth_edge.reason is not None

        # Post-merge audit: no near-duplicate and no unexplained register flip
        post_audit = engine.audit_step(
            2, include_rules=["near_duplicate_entity", "unexplained_register_flip"]
        )
        assert post_audit.issues == []


def test_merge_duplicate_wins_strategy_and_location_refs() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(entity_id="loc_mansion", kind="location", canonical_name="Pemberley", state={"weather": "sunny"}),
            step=0,
        )
        engine.register_entity(
            EntityRecord(entity_id="loc_estate", kind="location", canonical_name="the Darcy estate", state={"weather": "rainy"}),
            step=0,
        )
        engine.register_entity(
            EntityRecord(entity_id="char_darcy", kind="character", canonical_name="Darcy", state={"location_id": "loc_estate"}),
            step=0,
        )

        report = engine.merge_entities(
            "loc_estate",
            "loc_mansion",
            step=1,
            conflict_strategy="duplicate_wins",
        )
        assert report.location_refs_updated == ["char_darcy"]

        entities = {e.entity_id: e for e in engine.store.get_all_active_entities("main", 1)}
        assert entities["loc_mansion"].state["weather"] == "rainy"
        assert entities["char_darcy"].state["location_id"] == "loc_mansion"


def test_merge_migrates_voice_exemplars_and_episodes_with_branch_isolation() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
        engine.register_entity(EntityRecord(entity_id="char_stranger", canonical_name="the stranger"), step=0)
        engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)

        engine.commit_step(
            1,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_tavern",
                        step=1,
                        predicate="meeting",
                        summary="The stranger met Merlin at the tavern.",
                        participants={"char_stranger": "actor", "char_merlin": "target"},
                        witnesses=["char_stranger", "char_merlin"],
                    )
                ]
            ),
            exemplars=[
                VoiceExemplar(
                    exemplar_id="vx_1",
                    step=1,
                    speaker_id="char_stranger",
                    addressee_id="char_merlin",
                    text="Keep your voice down, old man, the walls have ears.",
                    quality_score=9.0,
                )
            ],
            bypass_gate=True,
        )

        # Fork a child branch and merge ONLY on the child branch
        branch_sess = engine.session("main").branch(at_step=1, name="merged_branch")
        report = branch_sess.merge_entities("char_stranger", "char_arthur", step=2)
        assert report.voice_exemplars_migrated == 1
        assert report.episodes_migrated == 1
        assert report.epistemics_migrated == 1

        # On child branch: Arthur now owns the voice exemplar and episode awareness
        branch_voice = engine.store.get_lineage_voice_exemplars("merged_branch", 2, "char_arthur")
        assert [ex.exemplar_id for ex, _ in branch_voice] == ["vx_1"]

        branch_ep = engine.store.get_episode("ep_tavern", timeline_id="merged_branch", cutoff_step=2)
        assert branch_ep is not None
        assert "char_arthur" in branch_ep.participants
        assert "char_stranger" not in branch_ep.participants
        assert branch_ep.known_by.get("char_arthur") == "direct"
        assert "char_stranger" not in branch_ep.known_by

        # On parent 'main': untouched!
        main_entities = {e.entity_id for e in engine.store.get_all_active_entities("main", 2)}
        assert {"char_arthur", "char_stranger", "char_merlin"} == main_entities
        main_voice = engine.store.get_lineage_voice_exemplars("main", 2, "char_stranger")
        assert [ex.exemplar_id for ex, _ in main_voice] == ["vx_1"]
        main_ep = engine.store.get_episode("ep_tavern", timeline_id="main", cutoff_step=2)
        assert main_ep is not None
        assert "char_stranger" in main_ep.participants
        assert "char_arthur" not in main_ep.known_by


def test_merge_rollback_undoes_graph_merge() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(EntityRecord(entity_id="char_a", canonical_name="Alice"), step=0)
        engine.register_entity(EntityRecord(entity_id="char_a2", canonical_name="Ms. Alice"), step=1)
        engine.merge_entities("char_a2", "char_a", step=2)

        assert {e.entity_id for e in engine.store.get_all_active_entities("main", 2)} == {"char_a"}

        engine.rollback_to(1)
        assert {e.entity_id for e in engine.store.get_all_active_entities("main", 2)} == {"char_a", "char_a2"}


def test_merge_validation_and_dry_run() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(EntityRecord(entity_id="char_a", kind="character", canonical_name="Alice"), step=0)
        engine.register_entity(EntityRecord(entity_id="loc_a", kind="location", canonical_name="Alice Hall"), step=0)

        with pytest.raises(ValueError, match="into itself"):
            engine.merge_entities("char_a", "char_a")
        with pytest.raises(ValueError, match="not active"):
            engine.merge_entities("char_missing", "char_a")
        with pytest.raises(ValueError, match="Kind mismatch"):
            engine.merge_entities("loc_a", "char_a")

        # Dry run with allow_kind_mismatch=True reports warning and mutates nothing
        dry = engine.merge_entities("loc_a", "char_a", allow_kind_mismatch=True, dry_run=True)
        assert dry.dry_run is True
        assert dry.aliases_added == ["Alice Hall"]
        assert len(dry.warnings) == 1
        assert {e.entity_id for e in engine.store.get_all_active_entities("main", 0)} == {"char_a", "loc_a"}


def test_cli_merge_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "book.db"
    with CanonArcEngine(db_path) as engine:
        engine.register_entity(EntityRecord(entity_id="char_hercule_poirot", canonical_name="Hercule Poirot"), step=0)
        engine.register_entity(EntityRecord(entity_id="char_poirot", canonical_name="Mr. Poirot"), step=1)

    # Dry-run JSON preview
    rc_dry = cli_main(["merge", str(db_path), "char_poirot", "char_hercule_poirot", "--dry-run", "--json"])
    assert rc_dry == 0
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["dry_run"] is True
    assert dry_payload["aliases_added"] == ["Mr. Poirot"]

    # Real merge
    rc = cli_main(["merge", str(db_path), "char_poirot", "char_hercule_poirot"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Merged 'char_poirot' -> 'char_hercule_poirot'" in out

    # Merging again fails cleanly with exit code 1 because duplicate is now tombstoned
    rc_err = cli_main(["merge", str(db_path), "char_poirot", "char_hercule_poirot"])
    assert rc_err == 1
    assert "not active" in capsys.readouterr().err


def test_find_duplicate_candidates_and_two_line_merge_workflow() -> None:
    with CanonArcEngine(":memory:") as engine:
        # Post-glossary step=0: richer entity 'char_darcy' sorts AFTER sparse stub 'char_a_mr_darcy'
        # alphabetically, but richness tie-breaking should still elect 'char_darcy' as canonical_id.
        engine.register_entity(
            EntityRecord(
                entity_id="char_darcy",
                kind="character",
                canonical_name="Fitzwilliam Darcy",
                base_attributes={"target_name": "Ficvilijam Darsi", "gender": "male"},
            ),
            aliases=[AliasRecord(alias="Mr. Darcy", entity_id="char_darcy", kind="epithet")],
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_a_mr_darcy",
                kind="character",
                canonical_name="Mr. Darcy",
            ),
            step=0,
        )
        # Later step adds a fuzzy duplicate of a third entity, then advances head_step to 3
        engine.register_entity(
            EntityRecord(entity_id="char_elizabeth", canonical_name="Elizabeth Bennet"),
            step=1,
        )
        engine.register_entity(
            EntityRecord(entity_id="char_elizabth", canonical_name="Elizabth Bennet"),
            step=2,
        )
        engine.apply_delta(
            3,
            StepDelta(
                entities=[EntityDelta(entity_id="char_elizabeth", state={"mood": "composed"})]
            ),
        )

        # Default audit_step at head_step=3 misses older duplicates because check_all_entities=False
        default_audit = engine.audit_step(3, include_rules=["near_duplicate_entity"])
        assert default_audit.issues == []

        # find_duplicate_candidates() defaults to head_step and scans the full active roster
        candidates = engine.find_duplicate_candidates()
        assert len(candidates) == 2
        assert all(isinstance(c, DuplicateCandidate) for c in candidates)

        # Sorted by descending score: exact alias collision (100.0) before fuzzy name match
        exact, fuzzy = candidates
        assert exact.duplicate_id == "char_a_mr_darcy"
        assert exact.canonical_id == "char_darcy"
        assert exact.score == 100.0
        assert exact.duplicate_kind == "character"
        assert exact.canonical_kind == "character"
        assert "alias:Mr. Darcy" in exact.reason
        assert exact.step == 3

        assert fuzzy.duplicate_id == "char_elizabth"
        assert fuzzy.canonical_id == "char_elizabeth"
        assert 90.0 <= fuzzy.score < 100.0

        # Explicit at_step=0 finds only the post-glossary duplicate
        step0_cands = engine.find_duplicate_candidates(at_step=0)
        assert [(c.duplicate_id, c.canonical_id) for c in step0_cands] == [
            ("char_a_mr_darcy", "char_darcy")
        ]

        # 2-line explicit merge workflow
        for cand in candidates:
            engine.merge_entities(cand.duplicate_id, cand.canonical_id)

        assert engine.find_duplicate_candidates() == []

        # Validation errors
        with pytest.raises(ValueError, match="does not exist"):
            engine.find_duplicate_candidates(timeline_id="missing_tl")
        with pytest.raises(ValueError, match="non-negative"):
            engine.find_duplicate_candidates(at_step=-1)


def test_audit_report_duplicate_candidates_and_session_isolation() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(entity_id="char_arthur", canonical_name="Arthur Pendragon"),
            step=0,
        )
        engine.register_entity(
            EntityRecord(entity_id="char_arthur_typo", canonical_name="Arther Pendragon"),
            step=1,
        )
        engine.register_entity(
            EntityRecord(entity_id="char_arthur_dup", canonical_name="Arthur Pendragon"),
            step=2,
        )

        report = engine.audit_timeline(include_rules=["near_duplicate_entity"])
        all_cands = report.duplicate_candidates()
        assert len(all_cands) >= 2
        assert all_cands[0].score == 100.0
        assert all_cands[0].duplicate_id == "char_arthur_dup"
        assert all_cands[0].canonical_id == "char_arthur"

        # min_score filters out fuzzy matches
        exact_only = report.duplicate_candidates(min_score=100.0)
        assert [(c.duplicate_id, c.canonical_id) for c in exact_only] == [
            ("char_arthur_dup", "char_arthur")
        ]

        # Branch isolation via TimelineSession.find_duplicate_candidates()
        branch = engine.session("main").branch(at_step=2, name="cleaned")
        branch.merge_entities("char_arthur_dup", "char_arthur", step=3)
        branch.merge_entities("char_arthur_typo", "char_arthur", step=3)

        assert branch.find_duplicate_candidates() == []
        assert len(engine.session("main").find_duplicate_candidates()) >= 2


def test_cli_duplicates_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "dups.db"
    with CanonArcEngine(db_path) as engine:
        engine.register_entity(
            EntityRecord(entity_id="char_poirot", kind="character", canonical_name="Hercule Poirot"),
            step=0,
        )
        engine.register_entity(
            EntityRecord(entity_id="char_poirot_2", kind="character", canonical_name="Hercules Poirot"),
            step=1,
        )
        engine.register_entity(
            EntityRecord(entity_id="loc_poirot", kind="location", canonical_name="Hercule Poirot"),
            step=1,
        )

    # Default human output (same_kind_only=True)
    rc = cli_main(["duplicates", str(db_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "char_poirot_2" in out
    assert "char_poirot" in out
    assert "canonarc merge" in out
    assert "loc_poirot" not in out

    # JSON output with --allow-cross-kind catches loc_poirot too
    rc_json = cli_main(["duplicates", str(db_path), "--allow-cross-kind", "--json"])
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    dup_ids = {item["duplicate_id"] for item in payload}
    assert "char_poirot_2" in dup_ids
    assert "loc_poirot" in dup_ids

    # After merging both duplicates, CLI reports no duplicates
    assert cli_main(["merge", str(db_path), "char_poirot_2", "char_poirot"]) == 0
    assert (
        cli_main(["merge", str(db_path), "loc_poirot", "char_poirot", "--allow-kind-mismatch"])
        == 0
    )
    capsys.readouterr()
    assert cli_main(["duplicates", str(db_path)]) == 0
    assert "No near-duplicate entities found" in capsys.readouterr().out
