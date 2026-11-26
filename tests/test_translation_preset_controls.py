from __future__ import annotations

import logging

import pytest

from canonarc import (
    CanonArcEngine,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    SchemaPolicy,
    StepContext,
    StepDelta,
    TranslationPreset,
    VoiceExemplar,
)


def _sample_context() -> StepContext:
    arthur = EntityRecord(
        entity_id="char_arthur",
        kind="character",
        canonical_name="Arthur",
        base_attributes={
            "target_name": "Artur",
            "gender": "masculine",
            "declension": "animate",
            "notes": "Uses clipped royal cadence.",
            "honorific": "Your Grace",
        },
        state={"mood": "furious"},
    )
    merlin = EntityRecord(
        entity_id="char_merlin",
        kind="character",
        canonical_name="Merlin",
        base_attributes={"target_name": "Merlin", "gender": "masculine"},
        state={"mood": "calm"},
    )
    citadel = EntityRecord(
        entity_id="loc_citadel",
        kind="location",
        canonical_name="Iron Citadel",
        base_attributes={"target_name": "Gvozdena citadela", "gender": "feminine"},
    )
    percival = EntityRecord(
        entity_id="char_percival",
        kind="character",
        canonical_name="Percival",
        base_attributes={"target_name": "Persival", "gender": "masculine"},
    )
    return StepContext(
        timeline_id="main",
        step=5,
        entities=[arthur, merlin, citadel, percival],
        active_edges=[
            DirectedEdgeRecord(
                source_id="char_arthur",
                target_id="char_merlin",
                relation_type="stance",
                state={"register": "ti", "tone": "angry"},
            )
        ],
        background_edges=[
            DirectedEdgeRecord(
                source_id="char_arthur",
                target_id="char_morgana",
                relation_type="stance",
                state={"register": "Vi", "tone": "cold"},
            )
        ],
        episodes=[
            EpisodeRecord(
                episode_id="ep_1",
                step=4,
                predicate="confrontation",
                summary="Arthur accused Merlin of hiding the seal.",
                known_by={"char_arthur": "direct", "char_merlin": "direct", "char_king": "told"},
            )
        ],
        voice_exemplars=[
            VoiceExemplar(
                exemplar_id="v1",
                step=2,
                speaker_id="char_arthur",
                text="You dare lie to me in my own hall?",
                metadata={"register": "ti"},
            ),
            VoiceExemplar(
                exemplar_id="v2",
                step=3,
                speaker_id="char_arthur",
                text="Bring me the seal before sundown.",
                metadata={"register": "ti"},
            ),
            VoiceExemplar(
                exemplar_id="v3",
                step=3,
                speaker_id="char_merlin",
                text="Patience, my lord, the truth is rarely simple.",
                metadata={"register": "Vi"},
            ),
        ],
        entity_names={
            "char_arthur": "Arthur",
            "char_merlin": "Merlin",
            "loc_citadel": "Iron Citadel",
            "char_percival": "Percival",
            "char_morgana": "Morgana",
            "char_king": "King Aldric",
        },
    )


def test_translation_preset_hides_ids_and_includes_kind_and_notes_by_default() -> None:
    ctx = _sample_context()
    preset = TranslationPreset(include_background_edges=True)
    rendered = preset.format_prompt_context(ctx)

    # No internal IDs leak anywhere in the translator prompt
    for internal_id in (
        "char_arthur",
        "char_merlin",
        "loc_citadel",
        "char_percival",
        "char_morgana",
        "char_king",
    ):
        assert internal_id not in rendered

    # Human names, kinds, and notes render cleanly
    assert "- Arthur [character]: target='Artur', masculine, animate [mood: furious]; notes: Uses clipped royal cadence." in rendered
    assert "- Iron Citadel [location]: target='Gvozdena citadela', feminine" in rendered
    assert "- Arthur → Merlin: register=ti, tone=angry" in rendered
    assert "- Arthur → Morgana: register=Vi, tone=cold" in rendered
    assert "known to: Arthur, King Aldric, Merlin | ignorant: Percival" in rendered
    assert '- Arthur (ti): "You dare lie to me in my own hall?"' in rendered


def test_translation_preset_show_entity_ids_true_preserves_legacy_ids() -> None:
    ctx = _sample_context()
    preset = TranslationPreset(show_entity_ids=True, include_kind=False, extra_attributes=())
    rendered = preset.format_prompt_context(ctx)

    assert "- Arthur (char_arthur): target='Artur', masculine, animate [mood: furious]" in rendered
    assert "notes:" not in rendered
    assert "[location]" not in rendered
    assert "- char_arthur → char_merlin: register=ti, tone=angry" in rendered
    assert "known to: char_arthur, char_king, char_merlin | ignorant: char_percival" in rendered
    assert '- char_arthur (ti): "You dare lie to me in my own hall?"' in rendered


def test_include_kind_non_character_and_custom_extra_attributes() -> None:
    ctx = _sample_context()
    preset = TranslationPreset(
        include_kind="non_character",
        extra_attributes=("notes", "honorific"),
    )
    rendered = preset.format_prompt_context(ctx)

    assert "- Arthur: target='Artur', masculine, animate [mood: furious]; notes: Uses clipped royal cadence.; honorific: Your Grace" in rendered
    assert "- Iron Citadel [location]: target='Gvozdena citadela', feminine" in rendered
    assert "[character]" not in rendered


def test_extra_attributes_truncate_verbose_notes() -> None:
    long_note = " ".join(["Keep archaic phrasing for this noble house."] * 10)
    ctx = StepContext(
        timeline_id="main",
        step=1,
        entities=[
            EntityRecord(
                entity_id="char_arthur",
                kind="character",
                canonical_name="Arthur",
                base_attributes={"target_name": "Artur", "notes": long_note},
            )
        ],
    )
    rendered = TranslationPreset(max_attr_chars=40).format_prompt_context(ctx)
    assert "notes:" in rendered
    assert rendered.endswith("…")
    note_part = rendered.split("notes: ", 1)[1]
    assert len(note_part) <= 40


def test_empty_context_returns_empty_text_and_omits_bare_headers() -> None:
    empty_ctx = StepContext(timeline_id="main", step=1)
    assert TranslationPreset().format_prompt_context(empty_ctx) == ""
    assert (
        TranslationPreset(policy=SchemaPolicy.lexicon_only()).format_prompt_context(empty_ctx)
        == ""
    )
    assert (
        TranslationPreset(empty_text="(none)").format_prompt_context(empty_ctx)
        == "(none)"
    )

    # Voice-only context omits bare state/glossary header
    voice_only_ctx = StepContext(
        timeline_id="main",
        step=2,
        voice_exemplars=[
            VoiceExemplar(
                exemplar_id="v1",
                step=1,
                speaker_id="char_arthur",
                text="Hold the gate.",
                metadata={"register": "ti"},
            )
        ],
    )
    rendered = TranslationPreset().format_prompt_context(voice_only_ctx)
    assert "## Scene character state & register — binding" not in rendered
    assert "## Character voice bank — stylistic anchor" in rendered
    assert '- char_arthur (ti): "Hold the gate."' in rendered


def test_auto_disambiguates_colliding_canonical_names() -> None:
    ctx = StepContext(
        timeline_id="main",
        step=1,
        entities=[
            EntityRecord(entity_id="char_raven", kind="character", canonical_name="Raven"),
            EntityRecord(entity_id="loc_raven", kind="location", canonical_name="Raven"),
            EntityRecord(entity_id="char_john_sr", kind="character", canonical_name="John"),
            EntityRecord(entity_id="char_john_jr", kind="character", canonical_name="John"),
        ],
        active_edges=[
            DirectedEdgeRecord(
                source_id="char_john_sr",
                target_id="char_john_jr",
                relation_type="stance",
                state={"register": "ti", "tone": "warm"},
            ),
            DirectedEdgeRecord(
                source_id="char_raven",
                target_id="loc_raven",
                relation_type="located_in",
                state={"status": "inside"},
            ),
        ],
    )
    rendered = TranslationPreset(include_kind=False).format_prompt_context(ctx)

    # Different kinds colliding -> disambiguated by [kind]
    assert "- Raven [character]: target='Raven'" in rendered
    assert "- Raven [location]: target='Raven'" in rendered
    assert "- Raven [character] → Raven [location] [located_in]: status='inside'" in rendered

    # Same name AND same kind colliding -> disambiguated by (entity_id)
    assert "- John (char_john_sr): target='John'" in rendered
    assert "- John (char_john_jr): target='John'" in rendered
    assert "- John (char_john_sr) → John (char_john_jr): register=ti, tone=warm" in rendered


def test_max_voice_per_speaker_caps_preserving_rank_order() -> None:
    ctx = _sample_context()
    rendered = TranslationPreset(max_voice_per_speaker=1).format_prompt_context(ctx)

    assert "You dare lie to me in my own hall?" in rendered
    assert "Bring me the seal before sundown." not in rendered
    assert "Patience, my lord, the truth is rarely simple." in rendered


def test_max_context_chars_trims_in_priority_order_and_warns_on_binding_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = _sample_context()
    full_preset = TranslationPreset(include_background_edges=True)
    full_text = full_preset.format_prompt_context(ctx)

    # Budget that forces dropping background edges and episodes, while keeping binding state
    mid_budget = len(full_text) - 180
    mid_text = TranslationPreset(
        include_background_edges=True,
        max_context_chars=mid_budget,
    ).format_prompt_context(ctx)
    assert len(mid_text) <= mid_budget
    assert "## Off-screen character ties — reference only" not in mid_text
    assert "## Scene character state & register — binding" in mid_text
    assert "- Arthur → Merlin: register=ti, tone=angry" in mid_text

    # Tight budget that forces dropping binding lines logs a warning and stays within limit
    tight_budget = 140
    with caplog.at_level(logging.WARNING, logger="canonarc.presets"):
        tight_text = TranslationPreset(
            include_background_edges=True,
            max_context_chars=tight_budget,
        ).format_prompt_context(ctx)
    assert len(tight_text) <= tight_budget
    assert any("truncated" in rec.message and "binding" in rec.message for rec in caplog.records)


def test_engine_populates_entity_names_for_offscreen_episode_witnesses_and_extractor_keeps_ids() -> None:
    with CanonArcEngine(":memory:", preset=TranslationPreset()) as engine:
        engine.seed_entities(
            [
                EntityRecord(entity_id="char_arthur", canonical_name="Arthur", kind="character"),
                EntityRecord(entity_id="char_merlin", canonical_name="Merlin", kind="character"),
                EntityRecord(entity_id="char_king", canonical_name="King Aldric", kind="character"),
            ]
        )
        engine.commit_step(
            1,
            StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_oath",
                        predicate="oath",
                        summary="Arthur swore an oath before King Aldric.",
                        participants={"char_arthur": "actor", "char_king": "witness"},
                        witnesses=["char_arthur", "char_king"],
                    )
                ]
            ),
        )
        # Step 2 only mentions Arthur and Merlin (King Aldric is off-screen)
        ctx = engine.retrieve_context(2, text="Arthur looked at Merlin.")
        assert ctx.entity_names.get("char_king") == "King Aldric"

        translator_ctx = engine.format_context(ctx)
        assert "King Aldric" in translator_ctx
        assert "char_king" not in translator_ctx
        assert "char_arthur" not in translator_ctx

        # Extractor prompt still receives full canonical entity_ids
        _, extractor_user = engine.build_extraction_messages(ctx, "Arthur looked at Merlin.")
        assert "char_arthur" in extractor_user
        assert "char_merlin" in extractor_user


def test_translation_preset_validates_arguments() -> None:
    with pytest.raises(ValueError, match="include_kind"):
        TranslationPreset(include_kind="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_attr_chars"):
        TranslationPreset(max_attr_chars=0)
    with pytest.raises(ValueError, match="max_voice_per_speaker"):
        TranslationPreset(max_voice_per_speaker=-1)
    with pytest.raises(ValueError, match="max_context_chars"):
        TranslationPreset(max_context_chars=-10)
    with pytest.raises(ValueError, match="entity_grouping"):
        TranslationPreset(entity_grouping="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown section_headers"):
        TranslationPreset(section_headers={"unknown": "## Bad"})
    with pytest.raises(ValueError, match="section_headers"):
        TranslationPreset(section_headers={"glossary": "   "})


def _hybrid_context() -> StepContext:
    return StepContext(
        timeline_id="main",
        step=7,
        entities=[
            EntityRecord(
                entity_id="char_arthur",
                kind="character",
                canonical_name="Arthur",
                base_attributes={"target_name": "Artur", "gender": "M"},
                state={"mood": "tense"},
            ),
            EntityRecord(
                entity_id="term_drift",
                kind="terminology",
                canonical_name="Resonance Drift",
                base_attributes={"target_name": "Rezonantni pomak", "gender": "M"},
            ),
            EntityRecord(
                entity_id="loc_citadel",
                kind="location",
                canonical_name="Iron Citadel",
                base_attributes={"target_name": "Gvozdena citadela", "gender": "F"},
            ),
            EntityRecord(
                entity_id="org_council",
                kind="organization",
                canonical_name="Council of Nine",
                base_attributes={"target_name": "Veće devetorice", "gender": "N"},
            ),
            EntityRecord(
                entity_id="item_loom",
                kind="artifact",
                canonical_name="Aether Loom",
                base_attributes={"target_name": "Eterski razboj", "gender": "M"},
            ),
            EntityRecord(
                entity_id="custom_rune",
                kind="spell_school",
                canonical_name="Pyromancy",
                base_attributes={"target_name": "Piromantija", "gender": "F"},
            ),
        ],
        active_edges=[
            DirectedEdgeRecord(
                source_id="char_arthur",
                target_id="org_council",
                relation_type="stance",
                state={"register": "Vi", "tone": "formal"},
            )
        ],
    )


def test_entity_grouping_cast_and_glossary_splits_binding_blocks_and_omits_empty() -> None:
    ctx = _hybrid_context()
    rendered = TranslationPreset(entity_grouping="cast_and_glossary").format_prompt_context(ctx)

    state_idx = rendered.index("## Scene character state & register — binding")
    glossary_idx = rendered.index("## Terminology & glossary — binding")
    assert state_idx < glossary_idx

    state_block = rendered[state_idx:glossary_idx]
    glossary_block = rendered[glossary_idx:]

    assert "- Arthur [character]: target='Artur', M [mood: tense]" in state_block
    assert "- Arthur → Council of Nine: register=Vi, tone=formal" in state_block
    assert "Iron Citadel" not in state_block

    assert "- Iron Citadel [location]: target='Gvozdena citadela', F" in glossary_block
    assert "- Council of Nine [organization]: target='Veće devetorice', N" in glossary_block
    assert "- Aether Loom [artifact]: target='Eterski razboj', M" in glossary_block
    assert "- Resonance Drift [terminology]: target='Rezonantni pomak', M" in glossary_block

    # Terminology-only chunk omits character state header completely
    term_only_ctx = StepContext(
        timeline_id="main",
        step=8,
        entities=[ctx.entities[1], ctx.entities[2]],
    )
    term_only_rendered = TranslationPreset(
        entity_grouping="cast_and_glossary"
    ).format_prompt_context(term_only_ctx)
    assert "## Scene character state & register — binding" not in term_only_rendered
    assert "## Terminology & glossary — binding" in term_only_rendered

    # Character-only chunk omits glossary header completely
    char_only_ctx = StepContext(
        timeline_id="main",
        step=9,
        entities=[ctx.entities[0]],
    )
    char_only_rendered = TranslationPreset(
        entity_grouping="cast_and_glossary"
    ).format_prompt_context(char_only_ctx)
    assert "## Scene character state & register — binding" in char_only_rendered
    assert "## Terminology & glossary — binding" not in char_only_rendered

    # LEXICON_ONLY keeps a single glossary block without character state header
    lex_rendered = TranslationPreset(
        policy=SchemaPolicy.lexicon_only(),
        entity_grouping="cast_and_glossary",
    ).format_prompt_context(ctx)
    assert "## Scene character state & register — binding" not in lex_rendered
    assert lex_rendered.count("## Terminology & glossary — binding") == 1


def test_entity_grouping_by_kind_orders_subsections_and_suppresses_redundant_kind_tags() -> None:
    ctx = _hybrid_context()

    # FULL: characters + edges in state block; non-characters grouped under glossary subsections
    full_rendered = TranslationPreset(entity_grouping="by_kind").format_prompt_context(ctx)
    assert "## Scene character state & register — binding" in full_rendered
    assert "- Arthur: target='Artur', M [mood: tense]" in full_rendered
    assert "[character]" not in full_rendered
    assert "## Terminology & glossary — binding" in full_rendered

    headings = [
        "### Locations",
        "### Organizations",
        "### Artifacts",
        "### Terminology",
        "### Spell School",
    ]
    positions = [full_rendered.index(h) for h in headings]
    assert positions == sorted(positions)
    assert "[location]" not in full_rendered
    assert "[terminology]" not in full_rendered
    assert "- Iron Citadel: target='Gvozdena citadela', F" in full_rendered
    assert "- Pyromancy: target='Piromantija', F" in full_rendered

    # LEXICON_ONLY: all kinds (including Characters) grouped under glossary subsections
    lex_rendered = TranslationPreset(
        policy=SchemaPolicy.lexicon_only(),
        entity_grouping="by_kind",
    ).format_prompt_context(ctx)
    assert "## Scene character state & register — binding" not in lex_rendered
    assert "## Terminology & glossary — binding" in lex_rendered
    assert lex_rendered.index("### Characters") < lex_rendered.index("### Locations")
    assert "[mood:" not in lex_rendered

    # Colliding names across kinds still preserve disambiguation in by_kind
    collision_ctx = StepContext(
        timeline_id="main",
        step=1,
        entities=[
            EntityRecord(entity_id="char_raven", kind="character", canonical_name="Raven"),
            EntityRecord(entity_id="loc_raven", kind="location", canonical_name="Raven"),
        ],
    )
    coll_rendered = TranslationPreset(
        policy=SchemaPolicy.lexicon_only(),
        entity_grouping="by_kind",
    ).format_prompt_context(collision_ctx)
    assert "- Raven [character]: target='Raven'" in coll_rendered
    assert "- Raven [location]: target='Raven'" in coll_rendered


def test_custom_section_headers_override_defaults() -> None:
    ctx = _sample_context()
    preset = TranslationPreset(
        include_background_edges=True,
        entity_grouping="cast_and_glossary",
        section_headers={
            "state": "## Cast & Register (Strict)",
            "glossary": "## World Terms (Strict)",
            "episodes": "## Prior Plot Beats",
        },
    )
    rendered = preset.format_prompt_context(ctx)
    assert "## Cast & Register (Strict)" in rendered
    assert "## World Terms (Strict)" in rendered
    assert "## Prior Plot Beats" in rendered
    # Unoverridden headers retain defaults
    assert "## Off-screen character ties — reference only" in rendered
    assert "## Character voice bank — stylistic anchor" in rendered


def test_max_context_chars_prunes_empty_grouped_subheadings_without_orphans(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = _hybrid_context()
    full_text = TranslationPreset(entity_grouping="by_kind").format_prompt_context(ctx)

    # Trim enough to drop the trailing Spell School and Terminology subsections
    tight_budget = len(full_text) - 120
    with caplog.at_level(logging.WARNING, logger="canonarc.presets"):
        trimmed = TranslationPreset(
            entity_grouping="by_kind",
            max_context_chars=tight_budget,
        ).format_prompt_context(ctx)

    assert len(trimmed) <= tight_budget
    assert "### Spell School" not in trimmed
    assert "Pyromancy" not in trimmed
    # No orphaned ### heading at the end of the prompt
    assert not trimmed.rstrip().endswith("### Terminology")
    assert not trimmed.rstrip().endswith("### Artifacts")
    assert any("truncated" in rec.message and "binding" in rec.message for rec in caplog.records)
