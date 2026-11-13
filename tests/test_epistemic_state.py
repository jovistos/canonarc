from narrative_memory import (
    AliasRecord,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    NarrativeMemoryEngine,
    StepDelta,
    TranslationPreset,
    VoiceExemplar,
)


def test_disguise_alias_scoped_resolution():
    """Tests that a scoped disguise alias masks the entity for fooled observers
    while resolving to the true identity for observers who see through it."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(
        EntityRecord(
            entity_id="char_odysseus",
            canonical_name="Odysseus",
            base_attributes={"role": "King of Ithaca"},
            private_state={"intent": "reclaim_palace"},
        ),
        step=0,
    )
    engine.register_entity(EntityRecord(entity_id="char_suitor", canonical_name="Antinous"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_eurycleia", canonical_name="Eurycleia"), step=0)

    # Step 10: Odysseus adopts beggar disguise; only Odysseus & Eurycleia know
    engine.commit_step(
        step=10,
        delta=StepDelta(
            entities=[
                EntityDelta(
                    entity_id="char_odysseus",
                    add_aliases=[
                        AliasRecord(
                            alias="the old beggar",
                            entity_id="char_odysseus",
                            kind="disguise",
                            known_to=["char_odysseus", "char_eurycleia"],
                            persona_name="The Old Beggar",
                        )
                    ],
                )
            ],
            edges=[
                EdgeDelta(
                    source_id="char_suitor",
                    target_id="char_odysseus",
                    relation_type="stance",
                    state={"tone": "mortal_enemies", "trust": -1.0},
                )
            ],
        ),
    )

    # Antinous (fooled) sees only the masked persona, NOT Odysseus's identity or edges
    ctx_suitor = engine.retrieve_context(
        step=20,
        text="Antinous mocked the old beggar by the hearth.",
        mentions=["Antinous", "the old beggar"],
        pov_entity_ids=["char_suitor"],
    )
    suitor_entity_ids = {e.entity_id for e in ctx_suitor.entities}
    assert "char_odysseus" not in suitor_entity_ids
    assert "masked:char_odysseus" in suitor_entity_ids
    masked = next(e for e in ctx_suitor.entities if e.entity_id == "masked:char_odysseus")
    assert masked.canonical_name == "The Old Beggar"
    assert masked.private_state == {}
    assert len(ctx_suitor.active_edges) == 0

    # Eurycleia (sees through disguise) resolves to Odysseus and sees pierced annotation
    ctx_eurycleia = engine.retrieve_context(
        step=20,
        text="Eurycleia washed the feet of the old beggar.",
        mentions=["Eurycleia", "the old beggar"],
        pov_entity_ids=["char_eurycleia"],
    )
    eury_ids = {e.entity_id for e in ctx_eurycleia.entities}
    assert "char_odysseus" in eury_ids
    odysseus_view = next(e for e in ctx_eurycleia.entities if e.entity_id == "char_odysseus")
    assert odysseus_view.disguised_as == "The Old Beggar"
    # Eurycleia still cannot read Odysseus's private_state
    assert odysseus_view.private_state == {}


def test_private_state_and_provenance_redaction():
    """Tests that private_state and unwitnessed causal provenance (reason / caused_by_episode_id)
    are redacted when querying from an ignorant character's POV."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_lancelot", canonical_name="Lancelot"), step=0)

    engine.commit_step(
        step=42,
        delta=StepDelta(
            episodes=[
                EpisodeRecord(
                    episode_id="ep_poison_42",
                    predicate="sabotage",
                    summary="Arthur slipped nightshade into Lancelot's chalice.",
                    participants={"char_arthur": "actor"},
                    salience=9.5,
                )
            ],
            entities=[
                EntityDelta(
                    entity_id="char_arthur",
                    state={"mood": "tense"},
                    private_state={"secret_action": "poisoned_chalice", "guilt": "high"},
                    caused_by_episode_id="ep_poison_42",
                    reason="Secretly poisoned Lancelot's wine",
                )
            ],
        ),
    )

    # Lancelot's POV: sees observable mood='tense', but private_state and secret reason are redacted
    ctx_lancelot = engine.retrieve_context(
        step=50,
        text="Lancelot observed Arthur across the table.",
        mentions=["Lancelot", "Arthur"],
        pov_entity_ids=["char_lancelot"],
    )
    arthur_for_lancelot = next(e for e in ctx_lancelot.entities if e.entity_id == "char_arthur")
    assert arthur_for_lancelot.state == {"mood": "tense"}
    assert arthur_for_lancelot.private_state == {}
    assert arthur_for_lancelot.reason is None
    assert arthur_for_lancelot.caused_by_episode_id is None
    assert len(ctx_lancelot.episodes) == 0

    # Arthur's POV: retains his own private_state, reason, and causal episode
    ctx_arthur = engine.retrieve_context(
        step=50,
        text="Arthur watched Lancelot lift the chalice.",
        mentions=["Arthur", "Lancelot"],
        pov_entity_ids=["char_arthur"],
    )
    arthur_for_arthur = next(e for e in ctx_arthur.entities if e.entity_id == "char_arthur")
    assert arthur_for_arthur.private_state == {"secret_action": "poisoned_chalice", "guilt": "high"}
    assert arthur_for_arthur.reason == "Secretly poisoned Lancelot's wine"
    assert arthur_for_arthur.caused_by_episode_id == "ep_poison_42"
    assert len(ctx_arthur.episodes) == 1


def test_deception_and_believed_summary():
    """Tests that InformDelta(epistemic='deceived', believed_summary=...) gives the deceived
    observer a false summary while preserving ground truth for others."""
    engine = NarrativeMemoryEngine(":memory:")
    for eid, name in [
        ("char_mordred", "Mordred"),
        ("char_gawain", "Gawain"),
        ("char_guinevere", "Guinevere"),
    ]:
        engine.register_entity(EntityRecord(entity_id=eid, canonical_name=name), step=0)

    engine.commit_step(
        step=30,
        delta=StepDelta(
            episodes=[
                EpisodeRecord(
                    episode_id="ep_crown_theft",
                    predicate="theft",
                    summary="Mordred stole the royal crown from the vault.",
                    participants={"char_mordred": "thief"},
                    salience=9.0,
                )
            ]
        ),
    )

    # Step 35: Mordred deceives Guinevere into believing Gawain stole it
    engine.inform_entity(
        step=35,
        episode_id="ep_crown_theft",
        entity_id="char_guinevere",
        epistemic="deceived",
        believed_summary="Gawain stole the royal crown from the vault.",
    )

    # Guinevere's POV recalls the false believed_summary
    ctx_guin = engine.retrieve_context(
        step=40,
        text="Guinevere confronted the knights about the crown.",
        mentions=["Guinevere"],
        pov_entity_ids=["char_guinevere"],
    )
    assert len(ctx_guin.episodes) == 1
    assert ctx_guin.episodes[0].summary == "Gawain stole the royal crown from the vault."
    assert ctx_guin.episodes[0].known_by["char_guinevere"] == "deceived"

    # Mordred's POV recalls the true summary
    ctx_mordred = engine.retrieve_context(
        step=40,
        text="Mordred smiled in the shadows.",
        mentions=["Mordred"],
        pov_entity_ids=["char_mordred"],
    )
    assert len(ctx_mordred.episodes) == 1
    assert ctx_mordred.episodes[0].summary == "Mordred stole the royal crown from the vault."


def test_stale_belief_state_override():
    """Tests that a belief_state edge overrides ground-truth entity state for a character
    who has not yet learned of a distant change (e.g. a battlefield death)."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="char_arthur", canonical_name="Arthur", state={"status": "alive"}),
        step=0,
    )
    engine.register_entity(EntityRecord(entity_id="char_guinevere", canonical_name="Guinevere"), step=0)

    # Step 90: Guinevere believes Arthur is alive on campaign
    # Step 100: Arthur dies in battle (ground truth)
    engine.commit_step(
        step=90,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_guinevere",
                    target_id="char_arthur",
                    relation_type="belief_state",
                    state={"status": "alive"},
                )
            ]
        ),
    )
    engine.commit_step(
        step=100,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"status": "dead"})]
        ),
    )

    # Step 105: Guinevere's POV still projects Arthur as status='alive'
    ctx_guin = engine.retrieve_context(
        step=105,
        text="Guinevere waited for Arthur's return.",
        mentions=["Guinevere", "Arthur"],
        pov_entity_ids=["char_guinevere"],
    )
    arthur_for_guin = next(e for e in ctx_guin.entities if e.entity_id == "char_arthur")
    assert arthur_for_guin.state["status"] == "alive"

    # Omniscient POV sees ground truth status='dead'
    ctx_omni = engine.retrieve_context(
        step=105,
        text="Arthur lay on the battlefield.",
        mentions=["Arthur"],
    )
    arthur_omni = next(e for e in ctx_omni.entities if e.entity_id == "char_arthur")
    assert arthur_omni.state["status"] == "dead"


def test_facade_vs_private_stance_voice_selection_and_irony_prompt():
    """Tests that facade edges drive outward voice exemplar selection and hide private
    stance edges from non-POV observers, while omniscient presets render dramatic irony tags."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_mordred", canonical_name="Mordred"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_percival", canonical_name="Percival"), step=0)

    engine.commit_step(
        step=10,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_mordred",
                    target_id="char_arthur",
                    relation_type="stance",
                    state={"register": "formal", "tone": "hostile", "trust": -0.9},
                ),
                EdgeDelta(
                    source_id="char_mordred",
                    target_id="char_arthur",
                    relation_type="facade",
                    state={"register": "formal", "tone": "devoted", "trust": 0.8},
                ),
            ],
            episodes=[
                EpisodeRecord(
                    episode_id="ep_secret_plot",
                    predicate="conspiracy",
                    summary="Mordred forged a secret alliance against Arthur.",
                    participants={"char_mordred": "actor"},
                    salience=9.0,
                )
            ],
        ),
        exemplars=[
            VoiceExemplar(
                step=10,
                speaker_id="char_mordred",
                addressee_id="char_arthur",
                text="Your Majesty's wisdom guides us all in these dark hours.",
                metadata={"register": "formal", "tone": "devoted"},
                quality_score=9.5,
            ),
            VoiceExemplar(
                step=10,
                speaker_id="char_mordred",
                addressee_id="char_arthur",
                text="You are a weakling king who deserves to fall by my blade.",
                metadata={"register": "formal", "tone": "hostile"},
                quality_score=9.5,
            ),
        ],
    )

    # Voice selection for Mordred speaking to Arthur prioritizes facade ('devoted') over private stance ('hostile')
    ctx_arthur = engine.retrieve_context(
        step=20,
        text="Mordred bowed before Arthur.",
        mentions=["Mordred", "Arthur"],
        pov_entity_ids=["char_arthur"],
        top_k_voice=1,
    )
    assert len(ctx_arthur.voice_exemplars) == 1
    assert "Your Majesty's wisdom" in ctx_arthur.voice_exemplars[0].text
    # Arthur does NOT see Mordred's private hostile stance edge when facade exists
    assert all(e.relation_type == "facade" for e in ctx_arthur.active_edges)

    # Omniscient scene with Mordred & Percival shows dramatic irony ignorant tag only for co-present Percival
    ctx_omni = engine.retrieve_context(
        step=20,
        text="Mordred greeted Percival in the hall.",
        mentions=["Mordred", "Percival"],
    )
    assert "char_arthur" not in {e.entity_id for e in ctx_omni.entities}
    assert "char_arthur" in {e.entity_id for e in ctx_omni.related_entities}
    prompt = engine.format_context(ctx_omni)
    assert "ignorant: char_percival" in prompt

    # TranslationPreset also preserves epistemic boundaries
    trans_prompt = TranslationPreset().format_prompt_context(ctx_omni)
    assert "ignorant: char_percival" in trans_prompt
