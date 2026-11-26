from __future__ import annotations

import json
from pathlib import Path

from canonarc import (
    AliasRecord,
    AuditContext,
    AuditIssue,
    CallableAuditRule,
    CanonArcEngine,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    ExonymLeakRule,
    NearDuplicateEntityRule,
    RegisterVocabularyDriftRule,
    SchemaPolicy,
    StepDelta,
    TranslationPreset,
    UnexplainedRegisterFlipRule,
)
from canonarc.cli import main as cli_main


def _seed_translation_engine(db_path: str | Path = ":memory:") -> CanonArcEngine:
    engine = CanonArcEngine(db_path, preset=TranslationPreset())
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
            entity_id="char_julian",
            canonical_name="Julian",
            base_attributes={"target_name": "Julijan", "gender": "M"},
        ),
        step=0,
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_lyra",
            canonical_name="Lyra",
            base_attributes={"target_name": "Lira", "gender": "F"},
        ),
        aliases=[
            AliasRecord(
                alias="hooded courier",
                entity_id="char_lyra",
                kind="disguise",
                persona_name="the hooded courier",
                known_to=["char_lyra"],
            )
        ],
        step=0,
    )
    engine.commit_step(
        step=0,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_elena",
                    target_id="char_julian",
                    state={"register": "Vi", "tone": "composed"},
                )
            ]
        ),
    )
    return engine


def test_exonym_leak_rule_catches_leak_and_passes_clean_target() -> None:
    with _seed_translation_engine() as engine:
        bad = engine.audit_step(
            step=1,
            source_text="Elena turned to Julian.",
            target_text="Elena se okrenula ka Julijanu.",
        )
        assert not bad.passed
        leaks = bad.by_code("exonym_leak")
        assert len(leaks) == 1
        assert leaks[0].severity == "error"
        assert leaks[0].entity_ids == ["char_elena"]
        assert leaks[0].evidence["expected_target_name"] == "Jelena"
        assert "Jelena" in (leaks[0].suggestion or "")

        clean = engine.audit_step(
            step=1,
            source_text="Elena turned to Julian.",
            target_text="Jelena se okrenula ka Julijanu.",
        )
        assert clean.passed
        assert clean.by_code("exonym_leak") == []


def test_exonym_leak_false_positive_guards_and_cjk() -> None:
    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_mark",
                canonical_name="Mark",
                base_attributes={"target_name": "Marko"},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="term_guild",
                canonical_name="Guild",
                base_attributes={"target_name": "Esnaf", "allow_source_name": True},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_sakura",
                canonical_name="桜",
                base_attributes={"target_name": "Sakura"},
            ),
            step=0,
        )

        # Lowercase 'mark' does not trigger case-sensitive ExonymLeakRule; allow_source_name skipped
        report = engine.audit_step(
            step=1,
            source_text="Mark visited the Guild.",
            target_text="Marko je ostavio mark u Guild dvorani.",
        )
        assert report.by_code("exonym_leak") == []

        # Source scoping: char_sakura not mentioned in source_text -> skipped
        scoped = engine.audit_step(
            step=1,
            source_text="Mark walked away.",
            target_text="桜の花が咲いた。",
        )
        assert scoped.by_code("exonym_leak") == []

        # CJK leak caught when mentioned in source_text (or when source_text omitted)
        cjk_leak = engine.audit_step(
            step=1,
            source_text="桜は静かに微笑んだ。",
            target_text="桜 smiled quietly.",
        )
        assert len(cjk_leak.by_code("exonym_leak")) == 1


def test_near_duplicate_entity_catches_alias_hijack_and_fuzzy_name() -> None:
    with _seed_translation_engine() as engine:
        # Step 1: LLM mistakenly creates char_courier despite 'hooded courier' alias on char_lyra
        engine.commit_step(
            step=1,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_courier",
                        canonical_name="Hooded Courier",
                        kind="character",
                    )
                ]
            ),
        )
        r1 = engine.audit_step(step=1)
        dups1 = r1.by_code("near_duplicate_entity")
        assert len(dups1) == 1
        assert set(dups1[0].entity_ids) == {"char_courier", "char_lyra"}
        assert "alias:hooded courier" in dups1[0].evidence["matched_against"]

        # Step 2: Near-duplicate typo of Elena -> 'Elenna'
        engine.commit_step(
            step=2,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_elenna",
                        canonical_name="Elenna",
                        kind="character",
                    )
                ]
            ),
        )
        r2 = engine.audit_step(step=2)
        dups2 = r2.by_code("near_duplicate_entity")
        assert len(dups2) == 1
        assert set(dups2[0].entity_ids) == {"char_elenna", "char_elena"}

        # Step 3: Intentional twin with ignore_duplicate_of is suppressed
        engine.commit_step(
            step=3,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_elena_twin",
                        canonical_name="Elenaa",
                        kind="character",
                        base_attributes={"ignore_duplicate_of": ["char_elena", "char_elenna"]},
                    )
                ]
            ),
        )
        r3 = engine.audit_step(step=3)
        assert r3.by_code("near_duplicate_entity") == []


def test_unexplained_register_flip_and_dangling_causal_episode() -> None:
    with _seed_translation_engine() as engine:
        # Step 1: Flip Vi -> ti WITHOUT reason or caused_by_episode_id -> warning
        engine.commit_step(
            step=1,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        state={"register": "ti", "tone": "furious"},
                    )
                ]
            ),
        )
        r1 = engine.audit_step(step=1)
        flips = r1.by_code("unexplained_register_flip")
        assert len(flips) == 1
        assert flips[0].severity == "warning"
        assert flips[0].evidence["prior_value"] == "Vi"
        assert flips[0].evidence["new_value"] == "ti"
        assert r1.passed  # warning does not fail report.passed

        # Step 2: Flip ti -> Vi WITH valid caused_by_episode_id -> clean
        engine.commit_step(
            step=2,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_truce",
                        step=2,
                        predicate="reconciliation",
                        summary="Elena and Julian restored formal court protocol.",
                        witnesses=["char_elena", "char_julian"],
                    )
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        state={"register": "Vi", "tone": "formal"},
                        caused_by_episode_id="ep_truce",
                    )
                ],
            ),
        )
        r2 = engine.audit_step(step=2)
        assert r2.by_code("unexplained_register_flip") == []
        assert r2.by_code("dangling_causal_episode") == []

        # Step 3: Flip Vi -> ti referencing non-existent episode -> error
        engine.commit_step(
            step=3,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        state={"register": "ti"},
                        caused_by_episode_id="ep_does_not_exist",
                    )
                ]
            ),
        )
        r3 = engine.audit_step(step=3)
        assert not r3.passed
        assert len(r3.by_code("dangling_causal_episode")) == 1


def test_register_vocabulary_drift_detects_mixed_vocabularies() -> None:
    with _seed_translation_engine() as engine:
        engine.commit_step(
            step=1,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_julian",
                        target_id="char_elena",
                        state={"register": "informal", "tone": "panicked"},
                        reason="Panicked plea",
                    )
                ]
            ),
        )
        report = engine.audit_step(step=1)
        drift = report.by_code("register_vocabulary_drift")
        assert len(drift) == 1
        assert "Vi" in drift[0].evidence["tv_values"]
        assert "informal" in drift[0].evidence["generic_values"]

        # Explicit allowed_values mode
        strict = engine.audit_step(
            step=1,
            rules=[RegisterVocabularyDriftRule(allowed_values=("Vi", "ti"))],
        )
        assert len(strict.by_code("register_vocabulary_drift")) == 1
        assert strict.issues[0].subject_key == "char_julian:stance->char_elena"


def test_pluggable_callable_rule_and_lexicon_only_tier_gating() -> None:
    def no_slang(ctx: AuditContext) -> list[AuditIssue]:
        if "okay" in ctx.target_text.lower():
            return [
                AuditIssue(
                    code="modern_slang",
                    rule_name="no_slang",
                    severity="warning",
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    message="Modern slang 'okay' found in historical prose.",
                )
            ]
        return []

    custom_rule = CallableAuditRule("no_slang", no_slang)
    preset = TranslationPreset(policy=SchemaPolicy.lexicon_only())
    with CanonArcEngine(":memory:", preset=preset, audit_rules=[
        ExonymLeakRule(),
        NearDuplicateEntityRule(),
        UnexplainedRegisterFlipRule(),
        RegisterVocabularyDriftRule(),
        custom_rule,
    ]) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_elena",
                canonical_name="Elena",
                base_attributes={"target_name": "Jelena"},
            ),
            step=0,
        )
        report = engine.audit_step(
            step=1,
            source_text="Elena nodded.",
            target_text="Jelena je rekla okay.",
        )
        # Tier 2/3 rules automatically skipped under LEXICON_ONLY
        assert "unexplained_register_flip" not in report.rules_executed
        assert "register_vocabulary_drift" not in report.rules_executed
        assert "no_slang" in report.rules_executed
        assert len(report.by_code("modern_slang")) == 1


def test_audit_timeline_and_session_and_summary() -> None:
    with _seed_translation_engine() as engine:
        engine.commit_step(
            step=1,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        state={"register": "ti"},
                    )
                ]
            ),
        )
        sess = engine.session("main")
        report = sess.audit_timeline(
            texts={
                1: ("Elena spoke to Julian.", "Elena je govorila Julijanu."),
            }
        )
        assert not report.passed
        assert len(report.by_code("exonym_leak")) == 1
        assert len(report.by_code("unexplained_register_flip")) == 1
        summary = report.summary()
        assert "Audit [FAIL]" in summary
        assert "exonym_leak" in summary


def test_cli_audit_subcommand_db_only_single_step_and_pairs_jsonl(
    tmp_path: Path, capsys
) -> None:
    db_path = tmp_path / "story.db"
    with _seed_translation_engine(db_path) as engine:
        engine.commit_step(
            step=1,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        state={"register": "ti"},
                    )
                ]
            ),
        )

    # 1. DB-only audit: warning only -> exit 0 with default --fail-on error, exit 1 with --fail-on warning
    rc_ok = cli_main(["audit", str(db_path), "--json"])
    assert rc_ok == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(i["code"] == "unexplained_register_flip" for i in payload["issues"])

    rc_warn = cli_main(["audit", str(db_path), "--fail-on", "warning"])
    assert rc_warn == 1
    assert "unexplained_register_flip" in capsys.readouterr().out

    # 2. Single-step audit with --source and --target triggering exonym_leak error
    src_file = tmp_path / "s1_en.txt"
    tgt_file = tmp_path / "s1_sr.txt"
    src_file.write_text("Elena looked at Julian.", encoding="utf-8")
    tgt_file.write_text("Elena je pogledala Julijana.", encoding="utf-8")

    rc_err = cli_main(
        [
            "audit",
            str(db_path),
            "--step",
            "1",
            "--source",
            str(src_file),
            "--target",
            str(tgt_file),
            "--rules",
            "exonym_leak",
            "--json",
        ]
    )
    assert rc_err == 1
    step_payload = json.loads(capsys.readouterr().out)
    assert step_payload["rules_executed"] == ["exonym_leak"]
    assert len(step_payload["issues"]) == 1

    # 3. Multi-step --pairs-jsonl audit with --fail-on none
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text(
        json.dumps(
            {
                "step": 1,
                "source": "Elena looked at Julian.",
                "target": "Elena je pogledala Julijana.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rc_none = cli_main(
        [
            "audit",
            str(db_path),
            "--pairs-jsonl",
            str(pairs_file),
            "--fail-on",
            "none",
        ]
    )
    assert rc_none == 0
    assert "Audit [FAIL]" in capsys.readouterr().out
