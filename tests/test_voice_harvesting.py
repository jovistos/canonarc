from __future__ import annotations

import json
from typing import Any

import pytest

from canonarc import (
    AliasRecord,
    BaseEngineHooks,
    CallableVoiceQualityScorer,
    CanonArcEngine,
    DialogueCandidate,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    HeuristicVoiceQualityScorer,
    InteractiveFictionPreset,
    LLMVoiceQualityScorer,
    SchemaPolicy,
    StepDelta,
    VoiceScoreResult,
)


class _RecordingHooks(BaseEngineHooks):
    def __init__(self) -> None:
        self.gate_rejections: list[tuple[str, str, str, dict[str, Any]]] = []
        self.after_commits: list[tuple[int, int]] = []

    def on_gate_rejected(
        self,
        stage: str,
        text: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> None:
        self.gate_rejections.append((stage, text, reason, dict(metadata)))

    def on_after_commit(
        self,
        step: int,
        timeline_id: str,
        committed_episodes: Any,
        indexed_exemplars: Any,
    ) -> None:
        del timeline_id, committed_episodes
        self.after_commits.append((step, len(indexed_exemplars)))


class _FakeExtractor:
    """Minimal stand-in for OpenAICompatibleExtractor.call_json_api."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> tuple[str, None]:
        del max_tokens, temperature
        self.calls.append((prompt, system_prompt))
        return self.responses.pop(0), None


def test_heuristic_voice_quality_scorer_calibration_and_penalties():
    scorer = HeuristicVoiceQualityScorer()

    strong = DialogueCandidate(
        candidate_id="c1",
        text="Ne mešajte se u stvari koje ne razumete, Marko, inače ćete gorko zažaliti!",
        word_count=14,
        distinctiveness_score=0.86,
        lexical_diversity=0.93,
        speaker_id="char_elena",
        addressee_id="char_marko",
    )
    strong_res = scorer.score_candidate(strong, metadata={"register": "Vi", "tone": "cold"})
    assert strong_res.score >= 8.0

    weak = DialogueCandidate(
        candidate_id="c2",
        text="We should go there now and we should go right away now",
        word_count=12,
        distinctiveness_score=0.45,
        lexical_diversity=0.58,
        speaker_id="char_elena",
        addressee_id=None,
    )
    weak_res = scorer.score_candidate(weak, metadata={})
    assert weak_res.score < 8.0

    artifact = DialogueCandidate(
        candidate_id="c3",
        text="Listen [INSERT_NAME], the ancient parchment reveals an ominous portent across {loc_tower}!",
        word_count=14,
        distinctiveness_score=0.88,
        lexical_diversity=0.93,
        speaker_id="char_elena",
        addressee_id="char_marko",
    )
    artifact_res = scorer.score_candidate(artifact, metadata={"register": "Vi"})
    assert artifact_res.score < 8.0
    assert "template_or_markup_artifact" in artifact_res.reason


def test_callable_voice_quality_scorer_signatures():
    cand = DialogueCandidate(
        candidate_id="c1",
        text="The ancient parchment reveals an ominous portent that threatens Camelot.",
        word_count=12,
        distinctiveness_score=0.85,
        lexical_diversity=0.92,
        speaker_id="char_merlin",
    )
    s1 = CallableVoiceQualityScorer(lambda c: 9.25)
    assert s1.score_candidate(cand, metadata={}).score == 9.25

    s2 = CallableVoiceQualityScorer(
        lambda c, meta: VoiceScoreResult(
            score=12.0,  # clamped to 10.0
            reason="great",
            metadata_updates={"tone": meta.get("tone", "stern")},
        )
    )
    res2 = s2.score_candidate(cand, metadata={"tone": "solemn"})
    assert res2.score == 10.0
    assert res2.reason == "great"
    assert res2.metadata_updates == {"tone": "solemn"}

    s3 = CallableVoiceQualityScorer(lambda c, meta, ctx: 7.5 if ctx is None else 8.5)
    assert s3.score_candidate(cand, metadata={}, context=None).score == 7.5


def test_llm_voice_quality_scorer_batching_and_fallback():
    c1 = DialogueCandidate(
        candidate_id="cand_a",
        text="Ne mešajte se u stvari koje ne razumete, Marko, inače ćete zažaliti!",
        source_text="Do not meddle in matters you do not understand, Marko, or you will regret it!",
        word_count=13,
        distinctiveness_score=0.86,
        lexical_diversity=0.92,
        speaker_id="char_elena",
        addressee_id="char_marko",
    )
    c2 = DialogueCandidate(
        candidate_id="cand_b",
        text="Nemoj da se mešaš u stvari koje uopšte ne razumeš, prijatelju moj.",
        word_count=13,
        distinctiveness_score=0.82,
        lexical_diversity=0.90,
        speaker_id="char_elena",
        addressee_id="char_marko",
    )
    payload = json.dumps(
        {
            "items": [
                {
                    "candidate_id": "cand_a",
                    "score": 9.1,
                    "reason": "Correct Vi register and sharp voice",
                    "metadata_updates": {"tone": "stern"},
                },
                {
                    "candidate_id": "cand_b",
                    "score": 4.2,
                    "reason": "Register mismatch: uses ti instead of expected Vi",
                },
            ]
        }
    )
    fake = _FakeExtractor([f"```json\n{payload}\n```"])
    scorer = LLMVoiceQualityScorer(fake)  # type: ignore[arg-type]

    results = scorer.score_candidates_batch(
        [c1, c2],
        [{"register": "Vi"}, {"register": "Vi"}],
    )
    assert scorer.api_calls == 1
    assert len(results) == 2
    assert results[0].score == 9.1
    assert results[0].metadata_updates == {"tone": "stern"}
    assert results[1].score == 4.2
    assert "Register mismatch" in results[1].reason

    # Missing item falls back cleanly to fallback_scorer
    partial_fake = _FakeExtractor([json.dumps({"items": [{"candidate_id": "cand_a", "score": 8.8}]})])
    scorer_with_fb = LLMVoiceQualityScorer(
        partial_fake,  # type: ignore[arg-type]
        fallback_scorer=CallableVoiceQualityScorer(lambda c: 6.0),
    )
    fb_results = scorer_with_fb.score_candidates_batch([c1, c2], [{}, {}])
    assert fb_results[0].score == 8.8
    assert fb_results[1].score == 6.0


def test_step_transaction_harvest_voice_serbian_dyadic_edges_and_retrieval():
    hooks = _RecordingHooks()
    engine = CanonArcEngine(":memory:", hooks=[hooks])

    engine.register_entity(
        EntityRecord(entity_id="char_elena", canonical_name="Elena", kind="character"),
        aliases=[AliasRecord(alias="Eleni", entity_id="char_elena")],
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_marko", canonical_name="Marko", kind="character"),
        aliases=[AliasRecord(alias="Marka", entity_id="char_marko"), AliasRecord(alias="Marku", entity_id="char_marko")],
        step=0,
    )
    engine.apply_delta(
        0,
        StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_elena",
                    target_id="char_marko",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "cold", "trust": -0.4},
                )
            ]
        ),
    )

    sr_text = (
        "Elena oštro pogleda Marku u oči. "
        "„Ne mešajte se u stvari koje ne razumete, gospodine, inače ćete gorko zažaliti zbog svoje radoznalosti!“\n"
        "Marko slegnu ramenima. „Dobro.“"
    )

    with engine.step_transaction(1, text=sr_text, lang="sr") as tx:
        harvested = tx.harvest_voice(
            sr_text,
            source_text='Elena looked sharply at Marko. "Do not meddle..."',
            lang="sr",
        )
        assert len(harvested) == 1
        assert harvested[0].speaker_id == "char_elena"
        assert harvested[0].addressee_id == "char_marko"
        assert harvested[0].metadata["register"] == "Vi"
        assert harvested[0].metadata["tone"] == "cold"
        assert harvested[0].quality_score >= 8.0

    # Committed atomically on __exit__ and retrievable at step 2 with register hard filter
    assert hooks.after_commits[-1] == (1, 1)
    ctx2 = engine.retrieve_context(2, text="Elena i Marko razgovaraju u biblioteci.", lang="sr")
    assert len(ctx2.voice_exemplars) == 1
    assert ctx2.voice_exemplars[0].speaker_id == "char_elena"
    assert ctx2.voice_exemplars[0].metadata["register"] == "Vi"


def test_harvest_voice_staged_delta_overlay_facade_and_persona():
    """Intra-step register shifts, facade overrides, and disguise persona are reflected immediately."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="char_elena", canonical_name="Elena", kind="character"),
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_marko", canonical_name="Marko", kind="character"),
        step=0,
    )
    engine.apply_delta(
        0,
        StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_elena",
                    target_id="char_marko",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "formal"},
                )
            ]
        ),
    )

    scene = (
        'Elena stepped closer to Marko and snapped, '
        '"Stop pretending you never saw the forged seal inside the archive vault tonight!"'
    )

    with engine.step_transaction(1, text=scene) as tx:
        # Stage a register shift + facade + disguise in the same transaction before harvesting
        tx.commit(
            StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_elena",
                        state={"disguise": "archivist_cloak", "mood": "furious"},
                    )
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_marko",
                        relation_type="stance",
                        state={"register": "ti", "tone": "furious"},
                    ),
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="char_marko",
                        relation_type="facade",
                        state={"tone": "icy"},  # facade overrides stance tone, inherits register
                    ),
                ],
            )
        )
        harvested = tx.harvest_voice(scene)
        assert len(harvested) == 1
        ex = harvested[0]
        assert ex.metadata["register"] == "ti"
        assert ex.metadata["tone"] == "icy"
        assert ex.metadata["persona"] == "archivist_cloak"


def test_harvest_voice_infers_addressee_in_two_character_scene():
    engine = CanonArcEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="char_merlin", canonical_name="Merlin", kind="character"),
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_arthur", canonical_name="Arthur", kind="character"),
        step=0,
    )
    engine.apply_delta(
        0,
        StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_merlin",
                    target_id="char_arthur",
                    relation_type="stance",
                    state={"register": "formal", "tone": "solemn"},
                )
            ]
        ),
    )

    # Both characters appear in the scene, but Arthur is not inside Merlin's immediate attribution window
    scene = (
        "Arthur stood silently beside the hearth watching the embers fade into ash. "
        "The heavy oak door groaned against the winter wind outside the citadel walls.\n\n"
        'Merlin whispered, "The ancient parchment reveals an ominous portent that threatens the sovereign destiny of Camelot!"'
    )
    with engine.step_transaction(1, text=scene) as tx:
        harvested = tx.harvest_voice(scene)
        assert len(harvested) == 1
        assert harvested[0].speaker_id == "char_merlin"
        assert harvested[0].addressee_id == "char_arthur"
        assert harvested[0].metadata["register"] == "formal"
        assert harvested[0].metadata["tone"] == "solemn"


def test_harvest_voice_gate2_rejection_hooks_dry_run_and_rollback():
    hooks = _RecordingHooks()
    engine = CanonArcEngine(
        ":memory:",
        voice_scorer=CallableVoiceQualityScorer(
            lambda c: VoiceScoreResult(score=5.5, reason="too generic")
        ),
        hooks=[hooks],
    )
    engine.register_entity(
        EntityRecord(entity_id="char_merlin", canonical_name="Merlin", kind="character"),
        step=0,
    )
    scene = (
        'Merlin sighed, "The ancient parchment reveals an ominous portent that threatens '
        'the sovereign destiny of Camelot!"'
    )

    # 1. Rejected by Gate 2 -> fires hook, commits 0 exemplars
    with engine.step_transaction(1, text=scene) as tx:
        harvested = tx.harvest_voice(scene)
        assert harvested == []

    gate2_rejections = [r for r in hooks.gate_rejections if r[0] == "gate2"]
    assert len(gate2_rejections) == 1
    assert "too generic" in gate2_rejections[0][2]
    assert engine.store.get_lineage_voice_exemplars("main", 1, "char_merlin") == []

    # 2. stage=False returns passing exemplars without committing them
    with engine.step_transaction(2, text=scene) as tx:
        preview = tx.harvest_voice(
            scene,
            scorer=CallableVoiceQualityScorer(lambda c: 9.2),
            stage=False,
        )
        assert len(preview) == 1
        assert preview[0].quality_score == 9.2
    assert engine.store.get_lineage_voice_exemplars("main", 2, "char_merlin") == []

    # 3. Exception inside transaction rolls back harvested exemplars atomically
    with pytest.raises(RuntimeError, match="boom"):
        with engine.step_transaction(3, text=scene) as tx:
            staged = tx.harvest_voice(
                scene,
                scorer=CallableVoiceQualityScorer(lambda c: 9.2),
            )
            assert len(staged) == 1
            raise RuntimeError("boom")
    assert engine.store.get_lineage_voice_exemplars("main", 3, "char_merlin") == []


def test_harvest_voice_noop_when_tier5_disabled():
    engine = CanonArcEngine(
        ":memory:",
        preset=InteractiveFictionPreset(policy=SchemaPolicy.state_only()),
    )
    engine.register_entity(
        EntityRecord(entity_id="char_merlin", canonical_name="Merlin", kind="character"),
        step=0,
    )
    scene = (
        'Merlin whispered, "The ancient parchment reveals an ominous portent that threatens '
        'the sovereign destiny of Camelot!"'
    )
    with engine.step_transaction(1, text=scene) as tx:
        assert tx.harvest_voice(scene) == []
