from canonarc import (
    CanonArcEngine,
    EdgeDelta,
    EntityRecord,
    StepDelta,
    VoiceExemplar,
)


def test_multiparty_voice_resolution_preserves_all_copresent_registers():
    """Arthur in a room with both Merlin (ti/furious) and the Queen (Vi/reverent)
    must receive exemplars for both registers via Round-Robin rather than locking
    onto an arbitrary first edge."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_queen", canonical_name="Queen"), step=0)

    engine.commit_step(
        step=5,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    relation_type="stance",
                    state={"register": "ti", "tone": "furious"},
                ),
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_queen",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "reverent"},
                ),
            ]
        ),
        exemplars=[
            VoiceExemplar(
                exemplar_id="ex_merlin_ti",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_merlin",
                text="Prestani sa tim trikovima odmah, Merline!",
                metadata={"register": "ti", "tone": "furious"},
                quality_score=9.5,
            ),
            VoiceExemplar(
                exemplar_id="ex_queen_vi",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_queen",
                text="Vaše Veličanstvo, molim Vas za savet u ovom času.",
                metadata={"register": "Vi", "tone": "reverent"},
                quality_score=9.5,
            ),
        ],
    )

    # Mention Queen first, then Merlin -> Queen's Vi exemplar leads, Merlin's ti exemplar follows
    ctx_queen_first = engine.retrieve_context(
        step=10,
        text="Arthur bowed to the Queen, then glared at Merlin.",
        top_k_voice=2,
    )
    ids_q_first = [ex.exemplar_id for ex in ctx_queen_first.voice_exemplars]
    assert ids_q_first == ["ex_queen_vi", "ex_merlin_ti"]
    assert ctx_queen_first.voice_exemplars[0].metadata.get("for_addressee") == "char_queen"
    assert ctx_queen_first.voice_exemplars[1].metadata.get("for_addressee") == "char_merlin"

    # Mention Merlin first, then Queen -> Merlin's ti exemplar leads, Queen's Vi exemplar follows
    ctx_merlin_first = engine.retrieve_context(
        step=10,
        text="Arthur snapped at Merlin before turning to the Queen.",
        top_k_voice=2,
    )
    ids_m_first = [ex.exemplar_id for ex in ctx_merlin_first.voice_exemplars]
    assert ids_m_first == ["ex_merlin_ti", "ex_queen_vi"]


def test_per_pair_facade_does_not_contaminate_other_addressees():
    """A facade edge toward a Spy must not override Arthur's genuine stance toward Lancelot."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_spy", canonical_name="Spy"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_lancelot", canonical_name="Lancelot"), step=0)

    engine.commit_step(
        step=5,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_spy",
                    relation_type="facade",
                    state={"register": "Vi", "tone": "cordial"},
                ),
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_lancelot",
                    relation_type="stance",
                    state={"register": "ti", "tone": "urgent"},
                ),
            ]
        ),
        exemplars=[
            VoiceExemplar(
                exemplar_id="ex_spy_facade",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_spy",
                text="Dobrodošli na naš dvor, cenjeni goste.",
                metadata={"register": "Vi", "tone": "cordial"},
                quality_score=9.5,
            ),
            VoiceExemplar(
                exemplar_id="ex_lancelot_stance",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_lancelot",
                text="Lanselote, uzmi mač i kreni odmah!",
                metadata={"register": "ti", "tone": "urgent"},
                quality_score=9.5,
            ),
        ],
    )

    ctx = engine.retrieve_context(
        step=10,
        text="Arthur whispered to Lancelot while the Spy watched.",
        top_k_voice=2,
    )
    ids = [ex.exemplar_id for ex in ctx.voice_exemplars]
    assert ids == ["ex_lancelot_stance", "ex_spy_facade"]


def test_offscreen_facade_does_not_override_inscene_stance():
    """An active facade toward an absent character must not override an in-scene stance."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_absent_king", canonical_name="Leodegrance"), step=0)

    engine.commit_step(
        step=5,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_absent_king",
                    relation_type="facade",
                    state={"register": "Vi", "tone": "cordial"},
                ),
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    relation_type="stance",
                    state={"register": "ti", "tone": "furious"},
                ),
            ]
        ),
        exemplars=[
            VoiceExemplar(
                exemplar_id="ex_king_vi",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_absent_king",
                text="Vaše Veličanstvo, čast mi je da Vas pozdravim.",
                metadata={"register": "Vi", "tone": "cordial"},
                quality_score=9.5,
            ),
            VoiceExemplar(
                exemplar_id="ex_merlin_ti",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_merlin",
                text="Ne igraj se sa mnom, Merline!",
                metadata={"register": "ti", "tone": "furious"},
                quality_score=9.5,
            ),
        ],
    )

    # Leodegrance is NOT in the scene
    ctx = engine.retrieve_context(
        step=10,
        text="Arthur confronted Merlin in the tower.",
        mentions=["Arthur", "Merlin"],
        top_k_voice=2,
    )
    ids = [ex.exemplar_id for ex in ctx.voice_exemplars]
    assert ids == ["ex_merlin_ti"]


def test_per_pair_edge_layer_merging_preserves_split_register_and_tone():
    """A base/kinship edge carrying register and a stance edge carrying tone
    for the same target must merge rather than dropping one attribute."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_morgana", canonical_name="Morgana"), step=0)

    engine.commit_step(
        step=5,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_morgana",
                    relation_type="kinship",
                    state={"register": "ti"},
                ),
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_morgana",
                    relation_type="stance",
                    state={"tone": "suspicious"},
                ),
            ]
        ),
        exemplars=[
            VoiceExemplar(
                exemplar_id="ex_formal_suspicious",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_morgana",
                text="Recite mi iskreno šta ste smerali.",
                metadata={"register": "Vi", "tone": "suspicious"},
                quality_score=9.5,
            ),
            VoiceExemplar(
                exemplar_id="ex_ti_suspicious",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_morgana",
                text="Šta to kriješ iza leđa, Morgana?",
                metadata={"register": "ti", "tone": "suspicious"},
                quality_score=9.5,
            ),
            VoiceExemplar(
                exemplar_id="ex_ti_warm",
                step=5,
                speaker_id="char_arthur",
                addressee_id="char_morgana",
                text="Draga sestro, uvek si dobrodošla u Kamelot.",
                metadata={"register": "ti", "tone": "warm"},
                quality_score=9.5,
            ),
        ],
    )

    ctx = engine.retrieve_context(
        step=10,
        text="Arthur watched Morgana closely.",
        mentions=["Arthur", "Morgana"],
        top_k_voice=1,
    )
    assert len(ctx.voice_exemplars) == 1
    assert ctx.voice_exemplars[0].exemplar_id == "ex_ti_suspicious"
