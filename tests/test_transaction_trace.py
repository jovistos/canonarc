from __future__ import annotations

import json
from pathlib import Path

import pytest

from canonarc import (
    AliasRecord,
    CachedDeltaExtractor,
    CallableExtractor,
    CallableVoiceQualityScorer,
    CanonArcEngine,
    DraftAuditTrace,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    LLMVoiceQualityScorer,
    SchemaPolicy,
    StageUsageSummary,
    StepContextSummary,
    StepDelta,
    StepTransactionTrace,
    TokenUsage,
    TransactionUsageSummary,
    TranslationPreset,
    VoiceExemplar,
    VoiceHarvestTrace,
    VoiceScoreResult,
)


class _StubVoiceJsonClient:
    """Minimal JsonCompletionClient returning valid batch voice scores + TokenUsage."""

    def __init__(self, *, model: str, score: float, reason: str, usage: TokenUsage) -> None:
        self.model = model
        self._score = score
        self._reason = reason
        self._usage = usage

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, TokenUsage]:
        import re

        cand_ids = re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
        payload = {
            "scores": [
                {"candidate_id": cid, "score": self._score, "reason": self._reason}
                for cid in cand_ids
            ]
        }
        return json.dumps(payload), TokenUsage.from_dict(self._usage.to_dict())


def test_step_context_to_summary_and_alias_provenance() -> None:
    """StepContext.to_summary() captures compact IDs, counts, and alias_kind/lang provenance."""
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_arthur",
                canonical_name="Arthur",
                base_attributes={"target_name": "Artur"},
            ),
            aliases=[
                AliasRecord(alias="Arture", entity_id="char_arthur", kind="inflection", lang="sr"),
                AliasRecord(alias="the young king", entity_id="char_arthur", kind="epithet", lang="en"),
            ],
            step=0,
        )
        engine.register_entity(
            EntityRecord(entity_id="char_merlin", canonical_name="Merlin"),
            step=0,
        )
        engine.commit_step(
            step=1,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_arthur",
                        target_id="char_merlin",
                        relation_type="stance",
                        state={"register": "vi"},
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_oath",
                        predicate="oath",
                        summary="Arthur swore an oath to Merlin.",
                        participants={"char_arthur": "actor", "char_merlin": "target"},
                    )
                ],
            ),
            exemplars=[
                VoiceExemplar(
                    exemplar_id="vx_1",
                    step=1,
                    speaker_id="char_arthur",
                    addressee_id="char_merlin",
                    text="We shall ride at dawn, old friend.",
                    quality_score=9.5,
                )
            ],
        )

        ctx = engine.retrieve_context(2, text="Arture spoke with Merlin.", lang="sr")
        summary = ctx.to_summary()

        assert isinstance(summary, StepContextSummary)
        assert summary.step == 2
        assert summary.timeline_id == "main"
        assert "char_arthur" in summary.entity_ids
        assert "char_merlin" in summary.entity_ids
        assert "char_arthur->char_merlin:stance" in summary.active_edge_keys
        assert summary.episode_ids == ["ep_oath"]
        assert summary.voice_exemplar_ids == ["vx_1"]
        assert summary.counts == {
            "resolved_mentions": len(ctx.resolved_mentions),
            "entities": len(ctx.entities),
            "related_entities": len(ctx.related_entities),
            "active_edges": len(ctx.active_edges),
            "background_edges": len(ctx.background_edges),
            "episodes": len(ctx.episodes),
            "voice_exemplars": len(ctx.voice_exemplars),
        }

        arthur_mentions = [m for m in summary.resolved_mentions if m.entity_id == "char_arthur"]
        assert len(arthur_mentions) == 1
        assert arthur_mentions[0].matched_via == "alias"
        assert arthur_mentions[0].alias_kind == "inflection"
        assert arthur_mentions[0].lang == "sr"


def test_transaction_trace_full_lifecycle_audit_extraction_voice_and_json() -> None:
    """Covers open -> committed status, multi-attempt audit_draft, extraction, voice harvesting, and JSON round-trip."""
    raw_delta = StepDelta(
        entities=[EntityDelta(entity_id="Arthur", state={"mood": "determined"})],
        episodes=[
            EpisodeRecord(
                predicate="departure",
                summary="Arthur left Camelot with Merlin.",
                participants={"Arthur": "actor", "Merlin": "companion"},
            )
        ],
    )
    extractor = CallableExtractor(
        lambda prompt, text: (
            raw_delta,
            TokenUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200, api_calls=1, latency_ms=50.0),
        ),
        model="deepseek-chat",
    )
    scorer = CallableVoiceQualityScorer(
        lambda cand, meta, ctx: VoiceScoreResult(
            score=9.2 if "dawn" in cand.text.lower() else 4.0,
            reason="distinctive" if "dawn" in cand.text.lower() else "generic",
        )
    )

    with CanonArcEngine(
        ":memory:",
        preset=TranslationPreset(),
        extractor=extractor,
        voice_scorer=scorer,
    ) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_arthur",
                canonical_name="Arthur",
                base_attributes={"target_name": "Artur"},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_merlin",
                canonical_name="Merlin",
                base_attributes={"target_name": "Merlin"},
            ),
            step=0,
        )

        source_text = 'Arthur looked at Merlin. "We ride at dawn!" he said. "Okay." said someone.'
        with engine.step_transaction(
            1,
            text=source_text,
            lang="sr",
            content_hash="abc123",
            checkpoint_metadata={"chunk_id": "ch01_001"},
        ) as tx:
            # Draft 1 leaks English exonym "Arthur" instead of Serbian "Artur"
            r1 = tx.audit_draft("Arthur pogleda Merlina.", label="initial")
            assert not r1.passed

            # Draft 2 fixes exonym
            r2 = tx.audit_draft("Artur pogleda Merlina.", label="repair_1")
            assert r2.passed

            tx.extract_and_commit(source_text)
            tx.harvest_voice(
                'Arthur said, "We ride at dawn, Merlin, before the river mist lifts over the valley!" '
                'Arthur muttered, "Very well, we shall leave the courtyard whenever the horses are ready." '
                'A stranger whispered, "Watch the eastern gate carefully until the bells ring twice at midnight!"',
                source_text=source_text,
            )

            mid_trace = tx.trace()
            assert mid_trace.status == "open"
            assert mid_trace.duration_ms is not None and mid_trace.duration_ms >= 0.0
            assert mid_trace.raw_delta is None  # omitted by default

        # Post-exit trace captures committed state & final IDs
        trace = tx.trace()
        assert isinstance(trace, StepTransactionTrace)
        assert trace.status == "committed"
        assert trace.step == 1
        assert trace.timeline_id == "main"
        assert trace.lang == "sr"
        assert trace.content_hash == "abc123"
        assert trace.checkpoint_metadata == {"chunk_id": "ch01_001"}
        assert trace.error is None
        assert trace.duration_ms is not None and trace.duration_ms >= 0.0

        # Audits recorded in attempt order
        assert len(trace.audit_reports) == 2
        assert isinstance(trace.audit_reports[0], DraftAuditTrace)
        assert trace.audit_reports[0].attempt == 1
        assert trace.audit_reports[0].label == "initial"
        assert trace.audit_reports[0].passed is False
        assert trace.audit_reports[0].error_count >= 1
        assert trace.audit_reports[1].attempt == 2
        assert trace.audit_reports[1].label == "repair_1"
        assert trace.audit_reports[1].passed is True

        # Delta normalized and committed
        assert trace.raw_delta is None
        assert trace.staged_delta is not None
        assert trace.staged_delta.entities[0].entity_id == "char_arthur"
        assert trace.delta_counts == {"entities": 1, "edges": 0, "episodes": 1, "inform": 0}
        assert len(trace.committed_episode_ids) == 1

        # Opt-in raw_delta preserves pre-normalization entity_id
        raw_trace = tx.trace(include_raw_delta=True)
        assert raw_trace.raw_delta is not None
        assert raw_trace.raw_delta.entities[0].entity_id == "Arthur"

        # Voice harvesting counts & indexed IDs
        assert isinstance(trace.voice_harvest, VoiceHarvestTrace)
        assert trace.voice_harvest.candidates_extracted >= 2
        assert trace.voice_harvest.gate1_rejected >= 1
        assert trace.voice_harvest.gate2_rejected >= 1
        assert trace.voice_harvest.staged_count == 1
        assert trace.harvested_voice_count == 1
        assert len(trace.indexed_exemplar_ids) == 1
        assert trace.voice_harvest.staged_exemplar_ids == trace.indexed_exemplar_ids

        # Usage captured for extraction
        assert "extraction" in trace.usage.stages
        ext_usage = trace.usage.stages["extraction"]
        assert ext_usage.model == "deepseek-chat"
        assert ext_usage.prompt_tokens == 1000
        assert ext_usage.completion_tokens == 200
        assert ext_usage.api_calls == 1
        assert trace.usage.total_tokens == 1200
        assert trace.usage.estimated_cost_usd == ext_usage.estimated_cost_usd

        # Plain JSON serialization round-trip
        trace_dict = tx.to_trace_dict()
        dumped = json.dumps(trace_dict, ensure_ascii=False)
        reloaded = StepTransactionTrace.model_validate(json.loads(dumped))
        assert reloaded.status == "committed"
        assert reloaded.step == 1
        assert len(reloaded.audit_reports) == 2
        assert reloaded.harvested_voice_count == 1


def test_transaction_trace_skipped_rolled_back_and_unentered() -> None:
    """Verifies skipped transactions, rolled-back transactions on exception, and unentered guard."""
    with CanonArcEngine(":memory:") as engine:
        unentered = engine.step_transaction(1, text="Hello")
        with pytest.raises(RuntimeError, match="has not been entered"):
            unentered.trace()

        # Skipped: no commit() or staged mutations
        with engine.step_transaction(1, text="Pure read-only lookup") as tx_skip:
            pass
        skip_trace = tx_skip.trace()
        assert skip_trace.status == "skipped"
        assert skip_trace.committed_episode_ids == []
        assert skip_trace.indexed_exemplar_ids == []
        assert skip_trace.error is None

        # Rolled back: exception raised inside block
        with pytest.raises(ValueError, match="boom"), engine.step_transaction(2, text="Failing step") as tx_err:
            tx_err.commit(StepDelta())
            raise ValueError("boom")
        err_trace = tx_err.trace()
        assert err_trace.status == "rolled_back"
        assert err_trace.error == "ValueError: boom"
        assert err_trace.duration_ms is not None
        assert json.dumps(tx_err.to_trace_dict())


def test_transaction_trace_lexicon_only_policy_and_manual_commit_exemplars() -> None:
    """TierProfile.LEXICON_ONLY marks extraction_skipped_by_policy; manual exemplars don't pollute voice_harvest."""
    with CanonArcEngine(":memory:", preset=TranslationPreset(policy=SchemaPolicy.lexicon_only())) as engine:
        with engine.step_transaction(1, text="Technical manual section.") as tx:
            tx.extract_and_commit("Technical manual section.")
        tr = tx.trace()
        assert tr.status == "committed"
        assert tr.extraction_skipped_by_policy is True
        assert tr.usage.api_calls == 0

    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(EntityRecord(entity_id="char_a", canonical_name="Alice"), step=0)
        manual_ex = VoiceExemplar(
            exemplar_id="vx_manual",
            step=1,
            speaker_id="char_a",
            text="A manually staged exemplar line with distinctive cadence.",
            quality_score=9.5,
        )
        with engine.step_transaction(1, text="Alice spoke.") as tx2:
            tx2.commit(StepDelta(), exemplars=[manual_ex])
        tr2 = tx2.trace()
        assert tr2.status == "committed"
        assert tr2.staged_extra_exemplars == 1
        assert tr2.indexed_exemplar_ids == ["vx_manual"]
        assert tr2.harvested_voice_count == 0
        assert tr2.voice_harvest.staged_count == 0
        assert tr2.voice_harvest.staged_exemplar_ids == []


def test_transaction_trace_heterogeneous_model_usage_and_cache_hits(tmp_path: Path) -> None:
    """Sums per-stage model-specific costs accurately and reflects CachedDeltaExtractor cache hits."""
    # Extraction on gemini-2.5-pro ($1.25 / $10.00 per 1M)
    ext_client = CallableExtractor(
        lambda prompt, text: (
            StepDelta(),
            TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000, api_calls=1, latency_ms=100.0),
        ),
        model="gemini-2.5-pro",
    )
    cached_ext = CachedDeltaExtractor(ext_client, cache_dir=tmp_path / "ext_cache", model_id="gemini-2.5-pro")

    # Voice scoring on deepseek-chat ($0.14 / $0.28 per 1M)
    voice_client = _StubVoiceJsonClient(
        model="deepseek-chat",
        score=9.0,
        reason="strong voice",
        usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000, api_calls=1, latency_ms=40.0),
    )
    voice_scorer = LLMVoiceQualityScorer(voice_client)

    dialogue_line = (
        'Arthur said, "Hold the northern bridge until sunrise, and let no rider cross without my royal seal!"'
    )
    with CanonArcEngine(
        ":memory:",
        extractor=cached_ext,
        voice_scorer=voice_scorer,
    ) as engine:
        engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)

        # First run: cache miss on extraction ($1.25) + voice scoring ($0.14) = $1.39 exact
        with engine.step_transaction(1, text=dialogue_line) as tx1:
            tx1.extract_and_commit(dialogue_line)
            tx1.harvest_voice(dialogue_line)

        t1 = tx1.trace()
        assert isinstance(t1.usage, TransactionUsageSummary)
        assert isinstance(t1.usage.stages["extraction"], StageUsageSummary)
        assert t1.usage.stages["extraction"].model == "gemini-2.5-pro"
        assert t1.usage.stages["extraction"].cache_misses == 1
        assert t1.usage.stages["extraction"].estimated_cost_usd == pytest.approx(1.25, abs=1e-6)

        assert t1.usage.stages["voice_scoring"].model == "deepseek-chat"
        assert t1.usage.stages["voice_scoring"].estimated_cost_usd == pytest.approx(0.14, abs=1e-6)

        # Total cost is the sum of model-specific stage costs, NOT a single-model re-pricing
        assert t1.usage.total_tokens == 2_000_000
        assert t1.usage.estimated_cost_usd == pytest.approx(1.39, abs=1e-6)

        # Second run (same step/text): CachedDeltaExtractor serves cache hit ($0.00 billable)
        with engine.step_transaction(1, text=dialogue_line) as tx2:
            tx2.extract_and_commit(dialogue_line)

        t2 = tx2.trace()
        assert "extraction" in t2.usage.stages
        assert t2.usage.stages["extraction"].cache_hits == 1
        assert t2.usage.stages["extraction"].api_calls == 0
        assert t2.usage.stages["extraction"].total_tokens == 0
        assert t2.usage.estimated_cost_usd == 0.0
