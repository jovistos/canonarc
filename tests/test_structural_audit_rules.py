from __future__ import annotations

from pathlib import Path

from canonarc import (
    COMMON_MARKUP_PATTERNS,
    COMMON_PLACEHOLDER_PATTERNS,
    AuditContext,
    CanonArcEngine,
    EntityDelta,
    EntityRecord,
    OmissionGuardRule,
    PlaceholderPreservationRule,
    StepDelta,
    TranslationPreset,
    UntranslatedSourceRule,
    default_audit_rules,
)


def _make_ctx(
    source_text: str,
    target_text: str,
    *,
    ignore_patterns=(),
    entities: tuple[EntityRecord, ...] = (),
    lang: str | None = "sr-Latn",
) -> AuditContext:
    return AuditContext(
        timeline_id="main",
        step=1,
        source_text=source_text,
        target_text=target_text,
        ignore_patterns=ignore_patterns,
        entities=entities,
        lang=lang,
    )


def test_default_audit_rules_keeps_structural_rules_opt_in() -> None:
    names = {r.name for r in default_audit_rules()}
    assert "placeholder_preservation" not in names
    assert "untranslated_source" not in names
    assert "omission_guard" not in names


# ---------------------------------------------------------------------------
# PlaceholderPreservationRule
# ---------------------------------------------------------------------------


def test_placeholder_preservation_passes_exact_and_reordered() -> None:
    rule = PlaceholderPreservationRule()
    src = "[[IMG_1]] {player_name} gave %s to {npc_name} (%.1f gold)."
    tgt = "{npc_name} je dobio %s od {player_name} (%.1f zlata). [[IMG_1]]"
    issues = rule.check(_make_ctx(src, tgt))
    assert issues == []


def test_placeholder_preservation_catches_mutated_tokens() -> None:
    rule = PlaceholderPreservationRule()
    src = "See [[IMG_1]] and talk to {player}."
    tgt = "Pogledaj [[SLIKA_1]] i pričaj sa {igrač}."
    issues = rule.check(_make_ctx(src, tgt))

    assert len(issues) == 2
    by_subj = {i.subject_key: i for i in issues}
    assert by_subj["placeholder:[[IMG_1]]"].code == "placeholder_mutated"
    assert by_subj["placeholder:[[IMG_1]]"].severity == "error"
    assert by_subj["placeholder:[[IMG_1]]"].evidence["suspected_replacement"] == "[[SLIKA_1]]"
    assert "[[IMG_1]]" in (by_subj["placeholder:[[IMG_1]]"].suggestion or "")
    assert "[[SLIKA_1]]" in (by_subj["placeholder:[[IMG_1]]"].suggestion or "")

    assert by_subj["placeholder:{player}"].code == "placeholder_mutated"
    assert by_subj["placeholder:{player}"].evidence["suspected_replacement"] == "{igrač}"


def test_placeholder_preservation_catches_missing_duplicated_and_unexpected() -> None:
    rule = PlaceholderPreservationRule()
    src = "Hello {a}, welcome to {b}."
    tgt = "Zdravo {a} i opet {a}, dobrodošli [[EXTRA_1]]."
    issues = rule.check(_make_ctx(src, tgt))

    codes = {i.code for i in issues}
    assert codes == {"placeholder_missing", "placeholder_duplicated", "placeholder_unexpected"}

    by_code = {i.code: i for i in issues}
    assert by_code["placeholder_missing"].evidence["token"] == "{b}"
    assert by_code["placeholder_duplicated"].evidence["token"] == "{a}"
    assert by_code["placeholder_duplicated"].evidence["actual_count"] == 2
    assert by_code["placeholder_unexpected"].evidence["token"] == "[[EXTRA_1]]"


def test_placeholder_preservation_check_order_for_positional_printf() -> None:
    rule = PlaceholderPreservationRule(check_order=True)
    src = "Found %d items in %s."
    tgt = "Pronađeno u %s ukupno %d predmeta."
    issues = rule.check(_make_ctx(src, tgt))
    assert len(issues) == 1
    assert issues[0].code == "placeholder_order_mismatch"
    assert issues[0].severity == "warning"
    assert issues[0].evidence["expected_order"] == ["%d", "%s"]
    assert issues[0].evidence["actual_order"] == ["%s", "%d"]


def test_placeholder_preservation_default_ignores_translatable_markdown_alt_text() -> None:
    rule = PlaceholderPreservationRule()
    src = "![A red dragon](img/dragon.png) approaches {hero}!"
    tgt = "![Crveni zmaj](img/dragon.png) prilazi {hero}!"
    # Even when engine ignore_patterns has COMMON_MARKUP_PATTERNS, default rule uses COMMON_PLACEHOLDER_PATTERNS
    issues = rule.check(_make_ctx(src, tgt, ignore_patterns=COMMON_MARKUP_PATTERNS))
    assert issues == []


def test_placeholder_preservation_no_double_counting_nested_patterns() -> None:
    # With COMMON_MARKUP_PATTERNS via use_context_ignore_patterns=True, {x} inside an HTML comment
    # is consumed by the comment pattern and not double-counted as a brace placeholder.
    rule = PlaceholderPreservationRule(use_context_ignore_patterns=True)
    src = "<!-- note {x} --> Hello {hero}"
    tgt = "<!-- note {x} --> Zdravo {hero}"
    issues = rule.check(_make_ctx(src, tgt, ignore_patterns=COMMON_MARKUP_PATTERNS))
    assert issues == []


def test_placeholder_preservation_custom_patterns() -> None:
    rule = PlaceholderPreservationRule(patterns=[r"<<[A-Z0-9_]+>>"])
    src = "Press <<BTN_A>> to jump."
    tgt = "Pritisni <<DUGME_A>> za skok."
    issues = rule.check(_make_ctx(src, tgt))
    assert len(issues) == 1
    assert issues[0].code == "placeholder_mutated"
    assert issues[0].evidence["token"] == "<<BTN_A>>"
    assert issues[0].evidence["suspected_replacement"] == "<<DUGME_A>>"


# ---------------------------------------------------------------------------
# UntranslatedSourceRule
# ---------------------------------------------------------------------------


def test_untranslated_source_passes_normal_translation() -> None:
    rule = UntranslatedSourceRule()
    src = "The old captain looked across the dark harbor and lowered his lantern."
    tgt = "Stari kapetan pogleda preko tamne luke i spusti svoj fenjer."
    assert rule.check(_make_ctx(src, tgt)) == []


def test_untranslated_source_catches_whole_step_echo() -> None:
    rule = UntranslatedSourceRule()
    src = "The old captain looked across the dark harbor and lowered his lantern."
    tgt = "The old captain looked across the dark harbor and lowered his lantern!"
    issues = rule.check(_make_ctx(src, tgt))
    assert len(issues) == 1
    assert issues[0].code == "untranslated_source_echo"
    assert issues[0].severity == "error"


def test_untranslated_source_catches_single_untranslated_paragraph() -> None:
    rule = UntranslatedSourceRule()
    src = (
        "The storm broke over the cliffs before midnight.\n\n"
        "Nobody dared to open the iron gate while the bells were ringing."
    )
    tgt = (
        "Oluja se sručila na litice pre ponoći.\n\n"
        "Nobody dared to open the iron gate while the bells were ringing."
    )
    issues = rule.check(_make_ctx(src, tgt))
    assert len(issues) == 1
    assert issues[0].code == "untranslated_paragraph_echo"
    assert issues[0].evidence["source_paragraph_index"] == 1


def test_untranslated_source_exempts_short_lines_and_allowed_entity_names() -> None:
    rule = UntranslatedSourceRule()
    # Short universal utterance
    assert rule.check(_make_ctx("OK, Arthur!", "OK, Arthur!")) == []

    # Line consisting mainly of exempt entity names
    ent = EntityRecord(
        entity_id="loc_new_york",
        timeline_id="main",
        kind="location",
        canonical_name="New York Central Station",
        base_attributes={"allow_source_name": True},
        valid_from_step=0,
    )
    assert (
        rule.check(
            _make_ctx(
                "New York Central Station — Platform 9",
                "New York Central Station — Platform 9",
                entities=(ent,),
            )
        )
        == []
    )


# ---------------------------------------------------------------------------
# OmissionGuardRule
# ---------------------------------------------------------------------------


def test_omission_guard_passes_balanced_translation() -> None:
    rule = OmissionGuardRule()
    src = (
        "First paragraph introduces the quiet village by the river.\n\n"
        "Second paragraph describes the market square at dawn.\n\n"
        "Third paragraph follows Elena as she walks toward the bridge."
    )
    tgt = (
        "Prvi pasus predstavlja mirno selo pored reke.\n\n"
        "Drugi pasus opisuje pijačni trg u zoru.\n\n"
        "Treći pasus prati Elenu dok hoda ka mostu."
    )
    assert rule.check(_make_ctx(src, tgt)) == []


def test_omission_guard_catches_empty_and_truncated_and_structure_collapse() -> None:
    rule = OmissionGuardRule()
    src = (
        "One sentence here. Two sentences here.\n\n"
        "Three sentences here. Four sentences here.\n\n"
        "Five sentences here. Six sentences here."
    )

    empty_issues = rule.check(_make_ctx(src, "   "))
    assert len(empty_issues) == 1
    assert empty_issues[0].code == "empty_translation"
    assert empty_issues[0].severity == "error"

    trunc_issues = rule.check(_make_ctx(src, "Samo jedna kratka rečenica."))
    codes = {i.code: i.severity for i in trunc_issues}
    assert codes["translation_truncated"] == "error"
    assert codes["paragraph_collapse"] == "warning"
    assert codes["sentence_collapse"] == "warning"


def test_omission_guard_catches_bloat_as_warning_and_ignores_short_sources() -> None:
    rule = OmissionGuardRule()
    # Short source below min_source_chars=80 should not trigger length ratio
    assert rule.check(_make_ctx("Thank you very much.", "Hvala.")) == []

    # Severe bloat triggers warning
    src = "A" * 100
    tgt = "B" * 300
    issues = rule.check(_make_ctx(src, tgt))
    assert len(issues) == 1
    assert issues[0].code == "translation_bloated"
    assert issues[0].severity == "warning"


def test_omission_guard_strips_ignored_markup_before_ratio() -> None:
    rule = OmissionGuardRule()
    huge_comment = "<!-- " + ("x" * 400) + " -->"
    src = f"{huge_comment} Short real sentence."
    tgt = "Kratka prava rečenica."
    # Without stripping ignore_patterns, length ratio would be ~0.05 and falsely trigger truncation
    issues = rule.check(_make_ctx(src, tgt, ignore_patterns=COMMON_MARKUP_PATTERNS))
    assert issues == []


# ---------------------------------------------------------------------------
# End-to-end unified pre-commit gate via StepTransaction.audit_draft()
# ---------------------------------------------------------------------------


def test_tx_audit_draft_unified_terminology_and_structural_repair_loop(tmp_path: Path) -> None:
    engine = CanonArcEngine(
        db_path=str(tmp_path / "unified_qa.db"),
        preset=TranslationPreset(),
        ignore_patterns=COMMON_MARKUP_PATTERNS,
        audit_rules=[
            *default_audit_rules(),
            PlaceholderPreservationRule(patterns=COMMON_PLACEHOLDER_PATTERNS),
            UntranslatedSourceRule(),
            OmissionGuardRule(),
        ],
    )

    with engine.step_transaction(0, text="Seed", lang="sr-Latn") as tx:
        tx.commit(
            StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_phoebe",
                        kind="character",
                        canonical_name="Phoebe",
                        base_attributes={"target_name": "Fibi"},
                    )
                ]
            )
        )

    source = (
        "[[IMG_1]] Phoebe opened the heavy oak door and looked down the corridor "
        "where {guard_name} was waiting with a brass key."
    )

    with engine.step_transaction(1, text=source, lang="sr-Latn") as tx:
        # Bad draft: exonym leak ("Phoebe"), mutated placeholder ("[[SLIKA_1]]"), missing "{guard_name}"
        bad_draft = (
            "[[SLIKA_1]] Phoebe otvori teška hrastova vrata i pogleda niz hodnik "
            "gde je stražar čekao sa mesinganim ključem."
        )
        r1 = tx.audit_draft(bad_draft, label="v1")
        assert not r1.passed
        codes = {i.code for i in r1.issues}
        assert "exonym_leak" in codes
        assert "placeholder_mutated" in codes
        assert "placeholder_missing" in codes

        instructions = r1.format_repair_instructions(min_severity="error")
        assert "Fibi" in instructions
        assert "[[IMG_1]]" in instructions
        assert "{guard_name}" in instructions

        # Repaired draft passes all rules in one unified gate
        good_draft = (
            "[[IMG_1]] Fibi otvori teška hrastova vrata i pogleda niz hodnik "
            "gde je {guard_name} čekao sa mesinganim ključem."
        )
        r2 = tx.audit_draft(good_draft, label="v2")
        assert r2.passed
        assert r2.format_repair_instructions() == ""

        tx.commit(StepDelta())

    engine.close()
