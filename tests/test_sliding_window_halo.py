from __future__ import annotations

from canonarc import (
    AliasRecord,
    CallableExtractor,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    CanonArcEngine,
    StepDelta,
)


def _seed_council_cast(engine: CanonArcEngine) -> None:
    engine.register_entity(
        EntityRecord(entity_id="char_elena", canonical_name="Elena", state={"mood": "calm"}),
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_julian", canonical_name="Julian", state={"mood": "nervous"}),
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_marcus", canonical_name="Marcus", state={"mood": "watchful"}),
        step=0,
    )
    engine.commit_step(
        step=0,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_elena",
                    target_id="char_julian",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "suspicious", "trust": -0.3},
                )
            ]
        ),
    )


def test_retrieve_context_prev_text_fallback_resolves_pronouns_and_edges() -> None:
    """prev_text resolves active characters & edges when target chunk has < 2 character mentions."""
    with CanonArcEngine(":memory:") as engine:
        _seed_council_cast(engine)

        # Without prev_text: pronoun-only chunk resolves nothing
        ctx_no_halo = engine.retrieve_context(
            step=5,
            text="Still furious, she slammed her palm against the oak table and demanded that he empty his pockets.",
            fallback_on_empty_mentions="none",
        )
        assert ctx_no_halo.resolved_mentions == []
        assert ctx_no_halo.entities == []
        assert ctx_no_halo.active_edges == []

        # With prev_text: fallback resolves Elena & Julian and keeps their active edge
        ctx_with_halo = engine.retrieve_context(
            step=5,
            text="Still furious, she slammed her palm against the oak table and demanded that he empty his pockets.",
            prev_text="Elena confronted Julian over the missing treasury ledger.",
            fallback_on_empty_mentions="none",
        )
        ent_ids = {e.entity_id for e in ctx_with_halo.entities}
        assert ent_ids == {"char_elena", "char_julian"}
        assert len(ctx_with_halo.active_edges) == 1
        assert ctx_with_halo.active_edges[0].source_id == "char_elena"
        assert ctx_with_halo.active_edges[0].target_id == "char_julian"

        # When target chunk ALREADY has >= 2 characters, prev_text does not pull in extra characters
        ctx_two_present = engine.retrieve_context(
            step=5,
            text="Elena faced Julian across the oak table.",
            prev_text="Marcus paced near the terrace doors.",
            fallback_on_empty_mentions="none",
        )
        assert {e.entity_id for e in ctx_two_present.entities} == {"char_elena", "char_julian"}


def test_retrieve_context_prev_text_respects_epistemic_disguise() -> None:
    """Fallback mentions from prev_text still obey POV disguise masking."""
    with CanonArcEngine(":memory:") as engine:
        _seed_council_cast(engine)
        engine.register_entity(
            EntityRecord(entity_id="char_lyra", canonical_name="Lyra", state={"mood": "watchful"}),
            aliases=[
                AliasRecord(
                    alias="the hooded courier",
                    entity_id="char_lyra",
                    kind="disguise",
                    persona_name="Courier",
                    known_to=["char_elena"],
                )
            ],
            step=0,
        )

        ctx_marcus = engine.retrieve_context(
            step=2,
            text="She lowered her voice and waited for an answer.",
            prev_text="The hooded courier bowed low before Marcus.",
            pov_entity_ids=["char_marcus"],
        )
        seen_ids = {e.entity_id for e in ctx_marcus.entities}
        assert "masked:char_lyra" in seen_ids
        assert "char_lyra" not in seen_ids


def test_extract_dialogue_candidates_split_attribution_and_no_halo_leak() -> None:
    """prev_text / next_text attribute boundary-split quotes without extracting quotes from the halo."""
    with CanonArcEngine(":memory:") as engine:
        _seed_council_cast(engine)

        # 1. Trailing quote in chunk k, speaker tag at start of chunk k+1 (next_text)
        chunk_k = '"Drop the obsidian signet right there on the oak table before I call the royal guard!"'
        chunk_k_plus_1 = (
            'Marcus barked from the doorway, drawing his sword. '
            'Julian replied, "Please, General, this is all a terrible misunderstanding among cousins!"'
        )
        cands_no_next = engine.extract_dialogue_candidates(chunk_k, step=1)
        assert len(cands_no_next) == 1
        assert cands_no_next[0].speaker_id is None

        cands_with_next = engine.extract_dialogue_candidates(
            chunk_k,
            step=1,
            next_text=chunk_k_plus_1,
        )
        # Must extract ONLY the quote inside chunk_k (NOT Julian's quote inside next_text!)
        assert len(cands_with_next) == 1
        assert cands_with_next[0].text.startswith("Drop the obsidian signet")
        assert cands_with_next[0].speaker_id == "char_marcus"
        assert cands_with_next[0].speaker_surface == "marcus"

        # 2. Leading quote in chunk k, speaker tag at end of chunk k-1 (prev_text)
        prev_chunk = (
            'Marcus whispered, "Keep your voice down near the stairs." '
            "Elena leaned across the table toward Julian."
        )
        curr_chunk = '"You will hand over the forged treasury ledger before the bells ring at midnight."'
        cands_with_prev = engine.extract_dialogue_candidates(
            curr_chunk,
            step=2,
            prev_text=prev_chunk,
        )
        # Must extract ONLY the quote inside curr_chunk (NOT Marcus's quote inside prev_text!)
        assert len(cands_with_prev) == 1
        assert cands_with_prev[0].text.startswith("You will hand over")
        assert cands_with_prev[0].speaker_id == "char_elena"
        assert cands_with_prev[0].addressee_id == "char_julian"


def test_build_extraction_prompt_and_step_transaction_fence_read_only_halo() -> None:
    """build_extraction_prompt and StepTransaction fence prev_text/next_text as read-only."""
    with CanonArcEngine(":memory:") as engine:
        _seed_council_cast(engine)

        captured_prompts: list[str] = []

        def _fake_llm(prompt: str, schema: dict) -> StepDelta:  # noqa: ARG001
            captured_prompts.append(prompt)
            return StepDelta(
                entities=[EntityDelta(entity_id="char_elena", state={"mood": "furious"})]
            )

        extractor = CallableExtractor(_fake_llm)
        prev_text = "Elena confronted Julian over the missing ledger."
        scene_text = '"Drop the obsidian signet right now!"'
        next_text = "Marcus barked from the doorway, drawing his sword."

        with engine.step_transaction(
            step=1,
            text=scene_text,
            prev_text=prev_text,
            next_text=next_text,
        ) as tx:
            assert {e.entity_id for e in tx.context.entities} == {"char_elena", "char_julian"}
            tx.extract_and_commit(scene_text, extractor=extractor)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "### Preceding Context (Already Committed — Use ONLY to resolve pronouns like he/she/they; DO NOT re-extract events):" in prompt
        assert prev_text in prompt
        assert "### Target Scene Chunk (EXTRACT ONLY state changes and episodes occurring in THIS text):" in prompt
        assert scene_text in prompt
        assert "### Following Context (Read-Only Lookahead — Use ONLY to identify speakers/addressees of trailing dialogue; DO NOT extract future events):" in prompt
        assert next_text in prompt


def test_store_apply_delta_idempotent_state_guard_preserves_trajectory() -> None:
    """store.apply_delta skips no-op entity/edge revisions while still applying alias updates."""
    with CanonArcEngine(":memory:") as engine:
        _seed_council_cast(engine)

        # Step 4: Elena calm -> furious; stance Vi/suspicious -> ti/furious
        engine.apply_delta(
            step=4,
            delta=StepDelta(
                entities=[EntityDelta(entity_id="char_elena", state={"mood": "furious"})],
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        relation_type="stance",
                        state={"register": "ti", "tone": "furious", "trust": -0.95},
                    )
                ],
            ),
        )

        # Step 5: Re-emitting identical mood="furious" (plus adding an alias) & identical edge state
        engine.apply_delta(
            step=5,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_elena",
                        state={"mood": "furious"},
                        add_aliases=["the angry chancellor"],
                    )
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_julian",
                        relation_type="stance",
                        state={"register": "ti", "tone": "furious", "trust": -0.95},
                    )
                ],
            ),
        )

        elena = engine.store.get_active_entity("char_elena", timeline_id="main", cutoff_step=5)
        assert elena is not None
        assert elena.state == {"mood": "furious"}
        assert elena.prior_state == {"mood": "calm"}
        assert elena.prior_step == 0
        assert elena.revision_count == 2  # step 0 and step 4 only

        aliases = engine.store.get_all_active_aliases("main", 5)
        assert any(a.alias == "the angry chancellor" and a.entity_id == "char_elena" for a in aliases)

        edge = engine.store.get_active_edge(
            "char_elena", "char_julian", relation_type="stance", timeline_id="main", cutoff_step=5
        )
        assert edge is not None
        assert edge.state == {"register": "ti", "tone": "furious", "trust": -0.95}
        assert edge.prior_state == {"register": "Vi", "tone": "suspicious", "trust": -0.3}
        assert edge.prior_step == 0
        assert edge.revision_count == 2  # step 0 and step 4 only
