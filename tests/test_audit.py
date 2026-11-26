from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    ScriptConsistencyRule,
    StepDelta,
    TargetTermComplianceRule,
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


def test_target_term_compliance_catches_wrong_spellings_and_allows_inflections() -> None:
    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_phoebe",
                canonical_name="Phoebe",
                base_attributes={"target_name": "Fibi", "target_lang": "sr"},
            ),
            aliases=[
                AliasRecord(alias="Fibin", entity_id="char_phoebe", kind="inflection", lang="sr"),
                AliasRecord(alias="Fibina", entity_id="char_phoebe", kind="inflection", lang="sr"),
            ],
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_poirot",
                canonical_name="Hercule Poirot",
                base_attributes={"target_name": "Erkil Poaro", "target_lang": "sr"},
            ),
            aliases=[
                AliasRecord(alias="Poirot", entity_id="char_poirot", kind="permanent", lang="en"),
                AliasRecord(alias="Erkila Poaroa", entity_id="char_poirot", kind="inflection", lang="sr"),
                AliasRecord(alias="Poaroa", entity_id="char_poirot", kind="inflection", lang="sr"),
            ],
            step=0,
        )

        # 1. Wrong phonetic spellings ("Febi" instead of "Fibi", "Herkul Poaro" instead of "Erkil Poaro") -> error
        bad = engine.audit_step(
            step=1,
            source_text="Phoebe handed the letter to Hercule Poirot.",
            target_text="Febi je predala pismo inspektoru Herkul Poaro.",
            lang="sr",
        )
        assert not bad.passed
        assert bad.by_code("exonym_leak") == []
        mismatches = bad.by_code("target_term_mismatch")
        assert len(mismatches) == 2
        by_ent = {m.subject_key: m for m in mismatches}
        assert by_ent["char_phoebe"].severity == "error"
        assert by_ent["char_phoebe"].evidence["suspected_variant"] == "Febi"
        assert by_ent["char_poirot"].severity == "error"
        assert by_ent["char_poirot"].evidence["suspected_variant"] == "Herkul Poaro"

        # 2. Valid inflected forms pass cleanly
        clean_inflected = engine.audit_step(
            step=1,
            source_text="Phoebe handed the letter to Hercule Poirot.",
            target_text="Fibin osmeh je umirio Erkila Poaroa.",
            lang="sr",
        )
        assert clean_inflected.passed
        assert clean_inflected.by_code("target_term_mismatch") == []
        assert clean_inflected.by_code("target_term_missing") == []

        # 3. Single-word source alias ("Poirot") allows partial target match ("Poaroa")
        partial_ok = engine.audit_step(
            step=1,
            source_text="Poirot raised an eyebrow.",
            target_text="Nije iznenadilo Poaroa.",
            lang="sr",
        )
        assert partial_ok.passed
        assert partial_ok.by_code("target_term_missing") == []
        assert partial_ok.by_code("target_term_mismatch") == []

        # 4. Pure omission / pronominalization emits warning (not error) by default
        pronoun = engine.audit_step(
            step=1,
            source_text="Phoebe smiled softly.",
            target_text="Ona se tiho nasmejala.",
            lang="sr",
        )
        assert pronoun.passed
        missing = pronoun.by_code("target_term_missing")
        assert len(missing) == 1
        assert missing[0].severity == "warning"
        assert missing[0].subject_key == "char_phoebe"
        assert pronoun.by_code("target_term_mismatch") == []

        # 5. Verbatim exonym leak is owned by ExonymLeakRule without double-firing TargetTermComplianceRule
        verbatim = engine.audit_step(
            step=1,
            source_text="Phoebe smiled softly.",
            target_text="Phoebe se tiho nasmejala.",
            lang="sr",
        )
        assert not verbatim.passed
        assert len(verbatim.by_code("exonym_leak")) == 1
        assert verbatim.by_code("target_term_mismatch") == []
        assert verbatim.by_code("target_term_missing") == []


def test_target_term_compliance_opt_out_and_kinds_filter() -> None:
    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_phoebe",
                canonical_name="Phoebe",
                kind="character",
                base_attributes={"target_name": "Fibi", "allow_omission": True},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="loc_abbey",
                canonical_name="Blackwood Abbey",
                kind="location",
                base_attributes={"target_name": "Opatija Blekvud"},
            ),
            step=0,
        )

        # allow_omission skips char_phoebe; kinds=("character",) skips loc_abbey
        report = engine.audit_step(
            step=1,
            source_text="Phoebe left Blackwood Abbey.",
            target_text="Ona je otišla iz manastira.",
            rules=[TargetTermComplianceRule(kinds=("character",))],
        )
        assert report.issues == []


def test_tx_audit_draft_pre_commit_retry_loop_and_format_repair_instructions() -> None:
    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_phoebe",
                canonical_name="Phoebe",
                base_attributes={"target_name": "Fibi", "target_lang": "sr"},
            ),
            aliases=[
                AliasRecord(alias="Fibin", entity_id="char_phoebe", kind="inflection", lang="sr"),
            ],
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_poirot",
                canonical_name="Hercule Poirot",
                base_attributes={"target_name": "Erkil Poaro", "target_lang": "sr"},
            ),
            step=0,
        )
        engine.commit_step(step=0)

        # Simulate a stale prior run at step 1 introducing a bogus entity; rerun_clean=True
        # must ensure tx.audit_draft() still audits against pre-commit state (step 0)
        engine.commit_step(
            step=1,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_stale",
                        canonical_name="Stale",
                        base_attributes={"target_name": "Zastareli"},
                    )
                ]
            ),
        )

        source_chunk = "Phoebe asked Hercule Poirot about the key."
        with engine.step_transaction(1, text=source_chunk, lang="sr", rerun_clean=True) as tx:
            # Draft 1: one verbatim leak ("Phoebe") + one wrong transliteration ("Herkul Poaro")
            draft1 = "Phoebe je pitala inspektora Herkul Poaro za ključ."
            r1 = tx.audit_draft(draft1)
            assert not r1.passed
            assert len(r1.by_code("exonym_leak")) == 1
            assert len(r1.by_code("target_term_mismatch")) == 1
            assert "target_term_compliance" in r1.rules_executed

            repair1 = r1.format_repair_instructions()
            assert "Fibi" in repair1
            assert "Erkil Poaro" in repair1
            assert "Herkul Poaro" in repair1
            assert "[char_phoebe]" in repair1
            assert "[char_poirot]" in repair1

            # max_issues and include_snippets=False
            compact = r1.format_repair_instructions(max_issues=1, include_snippets=False)
            assert compact.count("\n- ") == 1
            assert "(at:" not in compact

            # Draft 2: fixed translation passes cleanly before commit
            draft2 = "Fibi je pitala Erkila Poaroa za ključ."
            r2 = tx.audit_draft(draft2)
            assert r2.passed
            assert r2.format_repair_instructions() == ""

            # Draft 3: warning-only omission -> format_repair_instructions() is "" at default min_severity="error",
            # but populated when min_severity="warning"
            draft3 = "Ona je pitala Erkila Poaroa za ključ."
            r3 = tx.audit_draft(draft3)
            assert r3.passed
            assert r3.warning_count == 1
            assert r3.format_repair_instructions() == ""
            assert "Fibi" in r3.format_repair_instructions(min_severity="warning")

            tx.commit()


def test_tx_audit_draft_works_under_lexicon_only_profile() -> None:
    preset = TranslationPreset(policy=SchemaPolicy.lexicon_only())
    with CanonArcEngine(":memory:", preset=preset) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_phoebe",
                canonical_name="Phoebe",
                base_attributes={"target_name": "Fibi", "target_lang": "sr"},
            ),
            step=0,
        )
        with engine.step_transaction(1, text="Phoebe opened the gate.", lang="sr") as tx:
            report = tx.audit_draft("Febi je otvorila kapiju.")
            assert not report.passed
            assert len(report.by_code("target_term_mismatch")) == 1
            assert "unexplained_register_flip" not in report.rules_executed
            assert "Fibi" in report.format_repair_instructions()
            tx.commit()


def test_script_consistency_catches_drift_and_mixed_script_homoglyphs() -> None:
    rule = ScriptConsistencyRule(expected_script="latin")
    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        # 1. Clean Serbian Latin passes
        clean = engine.audit_step(
            step=1,
            target_text="Jelena je otvorila teška drvena vrata i nasmešila se.",
            rules=[rule],
        )
        assert clean.passed
        assert clean.issues == []

        # 2. Whole-word Cyrillic drift is aggregated into a single script_drift issue with snippet
        drifted = engine.audit_step(
            step=1,
            target_text="Jelena je rekla: Нисам сигурна да је то истина.",
            rules=[rule],
        )
        assert not drifted.passed
        drift_issues = drifted.by_code("script_drift")
        assert len(drift_issues) == 1
        assert drift_issues[0].subject_key == "script:cyrillic"
        assert drift_issues[0].evidence["found_script"] == "cyrillic"
        assert drift_issues[0].evidence["offending_token_count"] == 6
        assert "Нисам" in drift_issues[0].evidence["offending_tokens"]
        assert "Нисам" in drift_issues[0].evidence["snippet"]
        repair = drifted.format_repair_instructions()
        assert "cyrillic" in repair.lower()
        assert "latin" in repair.lower()
        assert "(at:" in repair

        # 3. Mixed-script homoglyph (Cyrillic 'о' \u043e inside Latin 'policija')
        homoglyph_word = "p\u043elicija"
        mixed = engine.audit_step(
            step=1,
            target_text=f"Stigla je {homoglyph_word} ispred kuće.",
            rules=[ScriptConsistencyRule(expected_script="latin", severity="warning", mixed_script_severity="error")],
        )
        assert not mixed.passed
        mixed_issues = mixed.by_code("mixed_script_token")
        assert len(mixed_issues) == 1
        assert mixed_issues[0].severity == "error"
        assert mixed_issues[0].subject_key == "script:mixed"
        assert homoglyph_word in mixed_issues[0].evidence["mixed_tokens"]
        assert mixed_issues[0].evidence["mixed_scripts_by_token"][homoglyph_word] == ["cyrillic", "latin"]
        assert mixed.by_code("script_drift") == []


def test_script_consistency_false_positive_guards_and_exonym_dedup() -> None:
    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="org_fbi",
                canonical_name="FBI",
                kind="organization",
                base_attributes={"target_name": "FBI", "allow_source_name": True},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="loc_london",
                canonical_name="London",
                kind="location",
                base_attributes={"target_name": "Лондон"},
            ),
            step=0,
        )

        rule = ScriptConsistencyRule(
            expected_script="cyrillic",
            allowed_tokens=("Wi-Fi", "mon Dieu"),
            allowed_patterns=(r"\[[A-Z_]+\]",),
        )

        # Roman numerals (IV, V), URLs, markup, placeholders, allowlisted tokens/phrases,
        # and canon-approved entity names (FBI) all pass cleanly in Russian Cyrillic
        clean_ru = engine.audit_step(
            step=1,
            source_text="The FBI checked Chapter IV.",
            target_text=(
                "Глава IV: агент FBI проверил <color=red>{player_name}</color> "
                "[PAUSE] через Wi-Fi (%s) на https://example.com — mon Dieu, Карл V!"
            ),
            rules=[rule],
        )
        assert clean_ru.passed
        assert clean_ru.issues == []

        # Verbatim exonym leak ("London" -> "Лондон") is owned by ExonymLeakRule
        # and not double-reported by ScriptConsistencyRule when skip_exonym_leaks=True
        both = engine.audit_step(
            step=1,
            source_text="He arrived in London.",
            target_text="Он прибыл в London вечером.",
            rules=[ExonymLeakRule(), rule],
        )
        assert not both.passed
        assert len(both.by_code("exonym_leak")) == 1
        assert both.by_code("script_drift") == []

        # Unrelated non-entity Latin drift is still caught
        stray = engine.audit_step(
            step=1,
            source_text="He arrived in London.",
            target_text="Он прибыл в Лондон actually вечером.",
            rules=[ExonymLeakRule(), rule],
        )
        assert not stray.passed
        assert stray.by_code("exonym_leak") == []
        assert len(stray.by_code("script_drift")) == 1
        assert stray.by_code("script_drift")[0].evidence["offending_tokens"] == ["actually"]


def test_script_consistency_multiscript_and_lang_inference() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported script"):
        ScriptConsistencyRule(expected_script="cyrilic")

    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        # Japanese legitimately combines han + hiragana + katakana inside a single token (食べる)
        ja_rule = ScriptConsistencyRule(expected_script=("han", "hiragana", "katakana"))
        ja_clean = engine.audit_step(
            step=1,
            target_text="彼女はリンゴを食べる。",
            rules=[ja_rule],
        )
        assert ja_clean.passed
        assert ja_clean.issues == []

        ja_bad = engine.audit_step(
            step=1,
            target_text="彼女はリンゴを食べる привет.",
            rules=[ja_rule],
        )
        assert not ja_bad.passed
        assert len(ja_bad.by_code("script_drift")) == 1
        assert ja_bad.by_code("script_drift")[0].evidence["found_script"] == "cyrillic"

        # Auto-inference from unambiguous BCP-47 ctx.lang when expected_script=None
        auto_rule = ScriptConsistencyRule()
        sr_latn = engine.audit_step(
            step=1,
            target_text="Dobro jutro пријатељу.",
            lang="sr-Latn",
            rules=[auto_rule],
        )
        assert not sr_latn.passed
        assert sr_latn.by_code("script_drift")[0].evidence["found_script"] == "cyrillic"

        sr_cyrl = engine.audit_step(
            step=1,
            target_text="Добро јутро prijatelju.",
            lang="sr-Cyrl",
            rules=[auto_rule],
        )
        assert not sr_cyrl.passed
        assert sr_cyrl.by_code("script_drift")[0].evidence["found_script"] == "latin"

        # Ambiguous 'sr' or None -> safe no-op
        ambiguous = engine.audit_step(
            step=1,
            target_text="Dobro jutro пријатељу.",
            lang="sr",
            rules=[auto_rule],
        )
        assert ambiguous.passed
        assert ambiguous.issues == []


def test_audit_timeline_text_provider_and_mutual_exclusion() -> None:
    with _seed_translation_engine() as engine:
        engine.commit_step(
            step=1,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        state={"register": "ti"},
                        reason="Shared ordeal.",
                    )
                ]
            ),
        )
        engine.commit_step(step=2, delta=StepDelta())

        calls: list[int] = []

        def provider(step: int) -> str | tuple[str, str] | None:
            calls.append(step)
            if step == 1:
                return ("Elena spoke to Julian.", "Elena je govorila Julijanu.")
            if step == 2:
                return "Jelena i Julijan su stigli."
            return None

        sess = engine.session("main")
        report = sess.audit_timeline(
            from_step=0,
            to_step=2,
            text_provider=provider,
            include_rules=["exonym_leak"],
        )
        assert calls == [0, 1, 2]
        assert report.steps_audited == [0, 1, 2]
        assert not report.passed
        assert len(report.by_code("exonym_leak")) == 1
        assert report.by_code("exonym_leak")[0].step == 1

        with pytest.raises(ValueError, match="not both"):
            engine.audit_timeline(
                texts={1: ("Elena", "Elena")},
                text_provider=provider,
            )


def test_audit_tier_gated_episode_loading_and_dangling_ref_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _seed_translation_engine() as engine:
        ep_calls = 0
        orig_get_episodes = engine.store.get_lineage_episodes

        def tracked_get_episodes(*args, **kwargs):
            nonlocal ep_calls
            ep_calls += 1
            return orig_get_episodes(*args, **kwargs)

        monkeypatch.setattr(engine.store, "get_lineage_episodes", tracked_get_episodes)

        # 1. Text-only Tier-1 rule under FULL policy (Tier 6 active) skips episode queries
        engine.commit_step(step=1, delta=StepDelta())
        engine.commit_step(step=2, delta=StepDelta())
        ep_calls = 0
        report = engine.audit_timeline(
            from_step=0,
            to_step=2,
            text_provider=lambda s: ("Elena smiled.", "Jelena se osmehnula.") if s == 1 else None,
            include_rules=["exonym_leak", "unexplained_register_flip"],
        )
        assert report.passed
        assert ep_calls == 0

        # 2. Edge mutation with caused_by_episode_id still loads episodes and catches dangling ref
        engine.commit_step(
            step=3,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        state={"register": "ti"},
                        caused_by_episode_id="ep_missing_999",
                    )
                ]
            ),
        )
        ep_calls = 0
        dangling_report = engine.audit_step(
            step=3,
            include_rules=["unexplained_register_flip"],
        )
        assert ep_calls == 1
        assert len(dangling_report.by_code("dangling_causal_episode")) == 1

        # 3. Custom Tier-6 rule always triggers episode loading
        seen_ep_counts: list[int] = []
        ep_rule = CallableAuditRule(
            "tier6_inspector",
            lambda ctx: (seen_ep_counts.append(len(ctx.episodes)), [])[1],
            required_tiers=frozenset({6}),
        )
        ep_calls = 0
        engine.audit_step(step=1, rules=[ep_rule])
        assert ep_calls == 1
        assert seen_ep_counts == [0]


def test_audit_report_export_formats_and_cli_output(
    tmp_path: Path, capsys
) -> None:
    import csv
    import io

    db_path = tmp_path / "export_audit.db"
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
        report = engine.audit_timeline(
            texts={1: ("Elena looked at Julian.", "Elena je pogledala Julijana.")},
            include_rules=["exonym_leak", "unexplained_register_flip"],
        )

    # Programmatic render/export: JSON, JSONL, CSV (+ excel_bom)
    json_str = report.to_json()
    assert json.loads(json_str)["timeline_id"] == "main"
    assert len(json.loads(json_str)["issues"]) == 2

    jsonl_lines = [json.loads(line) for line in report.to_jsonl().splitlines() if line.strip()]
    assert len(jsonl_lines) == 2
    assert {row["code"] for row in jsonl_lines} == {"exonym_leak", "unexplained_register_flip"}

    csv_rows = list(csv.DictReader(io.StringIO(report.to_csv())))
    assert len(csv_rows) == 2
    assert csv_rows[0]["timeline_id"] == "main"
    assert "entity_ids" in csv_rows[0]

    csv_file = tmp_path / "out" / "qa.csv"
    report.export(csv_file, excel_bom=True)
    raw_bytes = csv_file.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")

    jsonl_file = tmp_path / "out" / "qa.jsonl"
    report.export(jsonl_file)
    assert len(jsonl_file.read_text(encoding="utf-8").splitlines()) == 2

    with pytest.raises(ValueError, match="Cannot infer audit export format"):
        report.export(tmp_path / "qa.unknown")

    # CLI: --output with extension auto-detection + --format jsonl to stdout + conflicting flags
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

    cli_csv = tmp_path / "cli_report.csv"
    rc_csv = cli_main(
        [
            "audit",
            str(db_path),
            "--pairs-jsonl",
            str(pairs_file),
            "--output",
            str(cli_csv),
            "--excel-bom",
            "--fail-on",
            "error",
        ]
    )
    assert rc_csv == 1
    assert cli_csv.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "exonym_leak" in cli_csv.read_text(encoding="utf-8-sig")

    rc_jsonl = cli_main(
        [
            "audit",
            str(db_path),
            "--pairs-jsonl",
            str(pairs_file),
            "--format",
            "jsonl",
            "--fail-on",
            "none",
        ]
    )
    assert rc_jsonl == 0
    stdout_lines = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    assert any(row["code"] == "exonym_leak" for row in stdout_lines)

    rc_conflict = cli_main(
        ["audit", str(db_path), "--json", "--format", "csv"]
    )
    assert rc_conflict == 2
    assert "cannot be combined" in capsys.readouterr().err
