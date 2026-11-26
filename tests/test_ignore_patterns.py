from __future__ import annotations

import re

import pytest

from canonarc import (
    COMMON_MARKUP_PATTERNS,
    MARKUP_BRACKET_PLACEHOLDERS,
    MARKUP_HTML_TAGS,
    MARKUP_MARKDOWN_IMAGES,
    AuditContext,
    CallableAuditRule,
    CanonArcEngine,
    DefaultLexicalResolver,
    EntityRecord,
    ExonymLeakRule,
    HeuristicVoiceQualityScorer,
    IngestionGate,
    ScriptConsistencyRule,
    TargetTermComplianceRule,
    TranslationPreset,
    compile_ignore_patterns,
    mask_ignored_spans,
    strip_ignored_spans,
)


def test_text_masking_utilities_preserve_offsets_and_newlines() -> None:
    patterns = compile_ignore_patterns(COMMON_MARKUP_PATTERNS)
    raw = (
        "<!-- Line 1\nLine 2 -->\n"
        "Visit [London](https://example.com/paris) with <b>Arthur</b>!\n"
        "![Portrait of London](assets/london.png) [[IMG_1]] {player_name} %s"
    )
    masked = mask_ignored_spans(raw, patterns)

    # Exact length and newline positions preserved
    assert len(masked) == len(raw)
    assert [i for i, ch in enumerate(masked) if ch == "\n"] == [
        i for i, ch in enumerate(raw) if ch == "\n"
    ]

    # Opaque constructs masked out
    assert "Line 1" not in masked
    assert "Line 2" not in masked
    assert "paris" not in masked
    assert "Portrait" not in masked
    assert "IMG_1" not in masked
    assert "player_name" not in masked
    assert "%s" not in masked
    assert "<b>" not in masked

    # Visible prose (including Markdown link anchor and bolded name) preserved at original offsets
    assert "London" in masked
    assert "Arthur" in masked
    arthur_idx = raw.index("Arthur")
    assert masked[arthur_idx : arthur_idx + len("Arthur")] == "Arthur"

    stripped = strip_ignored_spans(raw, patterns)
    assert "  " not in stripped
    assert "Arthur" in stripped
    assert "London" in stripped
    assert "IMG_1" not in stripped


def test_compile_ignore_patterns_validation() -> None:
    assert compile_ignore_patterns(None) == ()
    assert compile_ignore_patterns(()) == ()
    compiled = compile_ignore_patterns([r"\[\[IMG_\d+\]\]", re.compile(r"\{[^}]+\}")])
    assert len(compiled) == 2
    with pytest.raises(TypeError):
        compile_ignore_patterns([123])  # type: ignore[list-item]


def test_resolver_ignores_markup_and_fixes_multiword_fuzzy_windows() -> None:
    entities = [
        EntityRecord(entity_id="loc_london", canonical_name="London", kind="location"),
        EntityRecord(entity_id="loc_paris", canonical_name="Paris", kind="location"),
        EntityRecord(entity_id="char_poirot", canonical_name="Hercule Poirot", kind="character"),
    ]
    resolver = DefaultLexicalResolver(ignore_patterns=COMMON_MARKUP_PATTERNS)

    # Entities inside comments, image links, URLs, HTML attributes, and placeholders are ignored
    text = (
        "<!-- London --> "
        "![Map of London](assets/london.png) "
        '<span data-city="Paris">{London_id}</span> '
        "[[PARIS_1]] "
        "<i>Hercule</i> <b>Poirot</b> walked slowly."
    )
    resolved = resolver.resolve(text, (), entities, ())
    ids = {r.entity_id for r in resolved}
    assert ids == {"char_poirot"}

    # Markdown link anchor text is preserved while URL is masked
    link_text = "They boarded the train to [London](https://example.com/paris) at dusk."
    link_resolved = resolver.resolve(link_text, (), entities, ())
    assert {r.entity_id for r in link_resolved} == {"loc_london"}

    # Multi-word fuzzy window works across inline HTML tags (inflected/typo form)
    fuzzy_text = "Everyone spoke quietly of <i>Hercule</i> <b>Poirots</b> brilliant deduction."
    fuzzy_resolved = resolver.resolve(fuzzy_text, (), entities, ())
    assert {r.entity_id for r in fuzzy_resolved} == {"char_poirot"}


def test_engine_retrieve_context_and_halo_respect_ignore_patterns() -> None:
    engine = CanonArcEngine(
        ":memory:",
        preset=TranslationPreset(),
        ignore_patterns=COMMON_MARKUP_PATTERNS,
    )
    engine.seed_entities(
        [
            EntityRecord(
                entity_id="char_arthur",
                canonical_name="Arthur",
                kind="character",
                base_attributes={"target_name": "Artur"},
            ),
            EntityRecord(
                entity_id="loc_london",
                canonical_name="London",
                kind="location",
                base_attributes={"target_name": "London"},
            ),
        ]
    )

    ctx = engine.retrieve_context(
        1,
        text="![View of London](img/london.png) [[IMG_1]] He stepped onto the pier.",
        prev_text="<!-- London --> <b>Arthur</b> checked his pocket watch.",
    )
    entity_ids = {e.entity_id for e in ctx.entities}
    assert "char_arthur" in entity_ids
    assert "loc_london" not in entity_ids


def test_voice_gate1_and_gate2_strip_markup_and_preserve_attribution() -> None:
    gate = IngestionGate(
        min_words=10,
        max_words=60,
        min_distinctiveness=0.35,
        ignore_patterns=COMMON_MARKUP_PATTERNS,
    )

    # 1. Short line padded with placeholders must NOT artificially pass min_words=10
    padded_short = (
        '"[[IMG_1]] {player_name} <b>Come</b> <i>here</i> %s right now!"'
    )
    assert gate.extract_candidates(padded_short) == []

    # 2. HTML attribute quotes must NOT collide with real dialogue quotes
    html_text = (
        '<div class="scene-intro" data-speaker="ghost">'
        '![A very long illustration caption that would normally push the speaker name past 140 characters in raw text](assets/chapters/chapter_03_ballroom_illustration_highres.png)'
        '</div>'
        'Arthur whispered grimly, "We cannot trust <i>anyone</i> inside this marble cathedral until the lantern burns out completely. [[IMG_2]]"'
    )
    html_cands = gate.extract_candidates(html_text)
    assert len(html_cands) == 1
    assert "<i>" not in html_cands[0].text
    assert "[[IMG_2]]" not in html_cands[0].text
    assert "marble cathedral" in html_cands[0].text

    # Wire through CanonArcEngine.harvest_voice() end-to-end
    engine = CanonArcEngine(
        ":memory:",
        preset=TranslationPreset(),
        gate=IngestionGate(min_words=10, max_words=60, min_distinctiveness=0.35, min_quality=7.0),
        voice_scorer=HeuristicVoiceQualityScorer(),
        ignore_patterns=COMMON_MARKUP_PATTERNS,
    )
    engine.seed_entities(
        [
            EntityRecord(entity_id="char_arthur", canonical_name="Arthur", kind="character"),
        ]
    )

    # Also test attribution surviving a long Markdown image link between speaker and quote
    narrator_with_img = (
        "Arthur leaned forward and said, "
        "![A very long illustration caption exceeding 140 characters so raw slicing would lose Arthur completely in the attribution window](assets/chapter_03_ballroom_highres.png) "
        '"We cannot trust <i>anyone</i> inside this marble cathedral until the lantern burns out completely. [[IMG_2]]"'
    )

    with engine.step_transaction(1, text=narrator_with_img) as tx:
        exemplars = tx.harvest_voice(narrator_with_img)
        tx.commit()

    assert len(exemplars) == 1
    ex = exemplars[0]
    assert ex.speaker_id == "char_arthur"
    assert "<i>" not in ex.text
    assert "[[IMG_2]]" not in ex.text
    assert "marble cathedral" in ex.text
    assert ex.quality_score >= 7.0

    # 3. HTML attribute quotes alone produce no dialogue candidates
    assert gate.extract_candidates('<span class="a very long attribute value with many distinct words here">Hello.</span>') == []

    # 4. Unexpected artifacts (not in ignore_patterns) STILL incur Gate-2 penalty
    unexpected_engine = CanonArcEngine(
        ":memory:",
        preset=TranslationPreset(),
        gate=IngestionGate(min_words=10, max_words=60, min_distinctiveness=0.35, min_quality=7.0),
        voice_scorer=HeuristicVoiceQualityScorer(),
        ignore_patterns=[MARKUP_HTML_TAGS, MARKUP_BRACKET_PLACEHOLDERS, MARKUP_MARKDOWN_IMAGES],
    )
    unexpected_engine.seed_entities(
        [EntityRecord(entity_id="char_arthur", canonical_name="Arthur", kind="character")]
    )
    dirty_line = (
        'Arthur muttered, "We cannot trust anyone inside this marble cathedral until TODO fix this line."'
    )
    with unexpected_engine.step_transaction(1, text=dirty_line) as tx:
        rejected = tx.harvest_voice(dirty_line)
        tx.commit()
    assert rejected == []


def test_audit_rules_respect_engine_ignore_patterns_and_preserve_snippets() -> None:
    engine = CanonArcEngine(
        ":memory:",
        preset=TranslationPreset(),
        ignore_patterns=COMMON_MARKUP_PATTERNS,
        audit_rules=[
            ExonymLeakRule(),
            TargetTermComplianceRule(),
            ScriptConsistencyRule(),
        ],
    )
    engine.seed_entities(
        [
            EntityRecord(
                entity_id="loc_vienna",
                canonical_name="Vienna",
                kind="location",
                base_attributes={"target_name": "Беч", "target_lang": "sr-Cyrl"},
            ),
            EntityRecord(
                entity_id="char_poirot",
                canonical_name="Hercule Poirot",
                kind="character",
                base_attributes={"target_name": "Еркил Поаро", "target_lang": "sr-Cyrl"},
            ),
        ]
    )

    # 1. Clean Cyrillic translation with Latin markup/placeholders & canonical name only in image/comment
    source_clean = "<!-- Vienna --> ![Map of Vienna](vienna.png) <b>Hercule Poirot</b> arrived. [[IMG_1]]"
    target_clean = "![Map of Vienna](vienna.png) <b>Еркил</b> <i>Поаро</i> је стигао. [[IMG_1]] {player_name}"

    with engine.step_transaction(1, text=source_clean, lang="sr-Cyrl") as tx:
        report = tx.audit_draft(target_clean, lang="sr-Cyrl")
        tx.commit()

    assert report.passed, report.summary()
    assert report.issues == []

    # 2. Real exonym leak AFTER markup still gets caught with exact original-text offsets/snippet
    source_leak = "Hercule Poirot left Vienna yesterday."
    target_leak = "[[IMG_1]] <b>Еркил Поаро</b> је напустио Vienna јуче."

    with engine.step_transaction(2, text=source_leak, lang="sr-Cyrl") as tx:
        leak_report = tx.audit_draft(target_leak, lang="sr-Cyrl")
        tx.commit()

    exonym_issues = leak_report.by_code("exonym_leak")
    assert len(exonym_issues) == 1
    issue = exonym_issues[0]
    spans = issue.evidence["matched_spans"]
    assert len(spans) == 1
    start, end = spans[0]
    # Exact slice in raw target_leak!
    assert target_leak[start:end] == "Vienna"
    assert "Vienna" in issue.evidence["snippet"]


def test_custom_text_normalizer_and_audit_context_helpers() -> None:
    def strip_custom_shortcodes(s: str) -> str:
        return re.sub(r"::[A-Z_]+::", "", s)

    seen_masked: list[tuple[str, str]] = []

    def inspect_rule(ctx: AuditContext) -> list:
        seen_masked.append((ctx.masked_source_text, ctx.masked_target_text))
        return []

    engine = CanonArcEngine(
        ":memory:",
        preset=TranslationPreset(),
        ignore_patterns=[r"\[\[IMG_\d+\]\]"],
        text_normalizer=strip_custom_shortcodes,
        audit_rules=[
            ExonymLeakRule(),
            CallableAuditRule("inspect", inspect_rule),
        ],
    )
    engine.seed_entities(
        [
            EntityRecord(
                entity_id="loc_vienna",
                canonical_name="Vienna",
                kind="location",
                base_attributes={"target_name": "Beč"},
            )
        ]
    )

    with engine.step_transaction(1, text="::SCENE_START:: [[IMG_1]] Vienna") as tx:
        report = tx.audit_draft("::SCENE_START:: [[IMG_1]] Vienna")
        tx.commit()

    assert not report.passed
    assert len(seen_masked) == 1
    assert "::SCENE_START::" not in seen_masked[0][0]
    assert "IMG_1" not in seen_masked[0][1]
    assert "Vienna" in report.by_code("exonym_leak")[0].evidence["snippet"]
