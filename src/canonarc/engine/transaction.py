from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self

from ..audit import AuditReport, AuditRule, AuditSeverity, run_audit_draft as _run_audit_draft
from ..extractors import DeltaExtractor, TokenUsage
from ..models import (
    DialogueCandidate,
    EpisodeRecord,
    StageUsageSummary,
    StepContext,
    StepDelta,
    VoiceExemplar,
    VoiceHarvestTrace,
)
from ..store import TimelineBranchConflictError
from ..voice import VoiceQualityScorer, score_voice_candidates_batch
from .traces import (
    _ACTIVE_COMMIT_EPISODES,
    DraftAuditTrace,
    KnownEntityScope,
    StepTransactionTrace,
    _build_transaction_usage_summary,
    _diff_component_usage,
    _merge_stage_usage,
    _snapshot_component_usage,
)

if TYPE_CHECKING:
    from .core import CanonArcEngine


class StepTransaction:
    """Context manager for reading state at `step` and atomically committing mutations on exit."""

    def __init__(
        self,
        engine: CanonArcEngine,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int | None = None,
        top_k_episodes: int | None = None,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        location_id: str | None = None,
        timeline_id: str | None = None,
        rerun_clean: bool = True,
        prev_text: str = "",
        next_text: str = "",
        lang: str | None = None,
        content_hash: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.step = step
        self.text = text
        self.mentions = mentions
        self.top_k_voice = top_k_voice
        self.top_k_episodes = top_k_episodes
        self.pov_entity_ids = pov_entity_ids
        self.pov_mode = pov_mode
        self.edge_types = edge_types
        self.location_id = location_id
        self.timeline_id = timeline_id or engine.active_timeline
        self.rerun_clean = rerun_clean
        self.prev_text = prev_text
        self.next_text = next_text
        self.lang = lang
        self._context: StepContext | None = None
        self._staged_delta: StepDelta | None = None
        self.last_raw_delta: StepDelta | None = None
        self._staged_exemplars: list[VoiceExemplar] = []
        self._staged_episodes: list[EpisodeRecord] = []
        self._bypass_gate = False
        self._committed = False
        self._content_hash: str | None = content_hash
        self._checkpoint_metadata: dict[str, Any] | None = (
            dict(checkpoint_metadata) if checkpoint_metadata is not None else None
        )
        self._audit_traces: list[DraftAuditTrace] = []
        self._extraction_skipped_by_policy: bool = False
        self._voice_candidates_extracted: int = 0
        self._voice_gate1_rejected: int = 0
        self._voice_gate2_rejected: int = 0
        self._voice_harvested_count: int = 0
        self._voice_harvested_ids: list[str] = []
        self._staged_extra_exemplars: int = 0
        self._staged_extra_episodes: int = 0
        self._stage_usages: dict[str, StageUsageSummary] = {}
        self._status: Literal["open", "committed", "skipped", "rolled_back"] = "open"
        self._committed_episode_ids: list[str] = []
        self._indexed_exemplar_ids: list[str] = []
        self._error: str | None = None
        self._started_perf: float | None = None
        self._duration_ms: float | None = None

    def _record_stage_usage(self, stage: str, component: Any, before: TokenUsage | None) -> None:
        diff = _diff_component_usage(stage, component, before)
        if diff is not None:
            self._stage_usages[stage] = _merge_stage_usage(self._stage_usages.get(stage), diff)

    def __enter__(self) -> Self:
        self._started_perf = time.perf_counter()
        if self.rerun_clean:
            for tl in self.engine.list_timelines():
                if (
                    tl.parent_id == self.timeline_id
                    and tl.fork_step is not None
                    and tl.fork_step > self.step - 1
                ):
                    raise TimelineBranchConflictError(
                        f"Cannot roll back '{self.timeline_id}' to step {self.step - 1}: child branch "
                        f"'{tl.timeline_id}' forked at step {tl.fork_step}. "
                        "Pass force=True to store.rollback_to to override."
                    )

        self._context = self.engine.retrieve_context(
            self.step,
            text=self.text,
            prev_text=self.prev_text,
            mentions=self.mentions,
            top_k_voice=self.top_k_voice,
            top_k_episodes=self.top_k_episodes,
            pov_entity_ids=self.pov_entity_ids,
            pov_mode=self.pov_mode,
            edge_types=self.edge_types,
            location_id=self.location_id,
            timeline_id=self.timeline_id,
            lang=self.lang,
        )
        return self

    @property
    def context(self) -> StepContext:
        if self._context is None:
            raise RuntimeError("StepTransaction is not active.")
        return self._context

    def audit_draft(
        self,
        target_text: str,
        *,
        source_text: str | None = None,
        rules: Sequence[AuditRule] | None = None,
        include_rules: Sequence[str] | None = None,
        exclude_rules: Sequence[str] | None = None,
        min_severity: AuditSeverity = "info",
        lang: str | None = None,
        include_all_timeline_entities: bool = False,
        label: str | None = None,
    ) -> AuditReport:
        """Audits a pre-commit draft translation against tx.context before commit()/extract_and_commit()."""
        report = _run_audit_draft(
            self.engine,
            self.step,
            source_text=self.text if source_text is None else source_text,
            target_text=target_text,
            context=self.context,
            timeline_id=self.timeline_id,
            rules=rules,
            include_rules=include_rules,
            exclude_rules=exclude_rules,
            min_severity=min_severity,
            lang=self.lang if lang is None else lang,
            include_all_timeline_entities=include_all_timeline_entities,
        )
        self._audit_traces.append(
            DraftAuditTrace(
                attempt=len(self._audit_traces) + 1,
                label=label,
                passed=report.passed,
                error_count=report.error_count,
                warning_count=report.warning_count,
                info_count=report.info_count,
                report=report,
            )
        )
        return report

    def commit(
        self,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        *,
        bypass_gate: bool = False,
        content_hash: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stage mutations to be committed when the transaction context exits successfully."""
        self._committed = True
        self._staged_delta = delta
        self._staged_exemplars.extend(exemplars)
        self._staged_episodes.extend(episodes)
        self._staged_extra_exemplars += len(exemplars)
        self._staged_extra_episodes += len(episodes)
        self._bypass_gate = bypass_gate
        if content_hash is not None:
            self._content_hash = content_hash
        if checkpoint_metadata is not None:
            self._checkpoint_metadata = dict(checkpoint_metadata)

    def extract_and_commit(
        self,
        scene_text: str,
        *,
        extractor: DeltaExtractor | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        bypass_gate: bool = False,
        prev_text: str | None = None,
        next_text: str | None = None,
        compact_schema: bool = False,
        known_entity_scope: KnownEntityScope = "scene",
        max_roster_entities: int | None = None,
        content_hash: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> StepDelta:
        """Runs the configured DeltaExtractor on scene_text, normalizes IDs, and stages the StepDelta."""
        if not self.engine._policy.has_tier(3):
            self._extraction_skipped_by_policy = True
            empty = StepDelta()
            self.last_raw_delta = empty
            self.commit(
                None,
                exemplars=exemplars,
                episodes=episodes,
                bypass_gate=bypass_gate,
                content_hash=content_hash,
                checkpoint_metadata=checkpoint_metadata,
            )
            return empty
        active_extractor = extractor or self.engine.extractor
        if active_extractor is None:
            raise ValueError("No DeltaExtractor provided to StepTransaction or CanonArcEngine.")
        effective_prev = self.prev_text if prev_text is None else prev_text
        effective_next = self.next_text if next_text is None else next_text
        system_prompt, user_prompt = self.engine.build_extraction_messages(
            self.context,
            scene_text,
            prev_text=effective_prev,
            next_text=effective_next,
            compact_schema=compact_schema,
            known_entity_scope=known_entity_scope,
            max_roster_entities=max_roster_entities,
        )
        extract_messages = getattr(active_extractor, "extract_delta_messages", None)
        before_usage = _snapshot_component_usage(active_extractor)
        try:
            if callable(extract_messages):
                raw_delta = extract_messages(system_prompt, user_prompt, scene_text)
            else:
                raw_delta = active_extractor.extract_delta(
                    f"{system_prompt}\n\n{user_prompt}",
                    scene_text,
                )
        finally:
            self._record_stage_usage("extraction", active_extractor, before_usage)
        self.last_raw_delta = raw_delta
        normalized = self.engine.normalize_delta(raw_delta, step=self.step, timeline_id=self.timeline_id)
        self.commit(
            normalized,
            exemplars=exemplars,
            episodes=episodes,
            bypass_gate=bypass_gate,
            content_hash=content_hash,
            checkpoint_metadata=checkpoint_metadata,
        )
        return normalized

    def harvest_voice(
        self,
        text: str,
        *,
        source_text: str = "",
        lang: str | None = None,
        prev_text: str | None = None,
        next_text: str | None = None,
        scorer: VoiceQualityScorer | None = None,
        min_quality: float | None = None,
        require_speaker: bool = True,
        infer_addressee: bool = True,
        extra_metadata: Mapping[str, str] | None = None,
        stage: bool = True,
    ) -> list[VoiceExemplar]:
        """Runs Gate-1 extraction, attaches active dyadic edge/persona metadata, scores via Gate 2, and stages passing VoiceExemplars."""
        if not self.engine._policy.has_tier(5) or not text.strip():
            return []

        effective_lang = self.lang if lang is None else lang
        effective_prev = self.prev_text if prev_text is None else prev_text
        effective_next = self.next_text if next_text is None else next_text
        cutoff_step = max(0, self.step - 1)

        candidates = self.engine._extract_dialogue_candidates_with_delta(
            text,
            source_text=source_text,
            step=cutoff_step,
            timeline_id=self.timeline_id,
            prev_text=effective_prev,
            next_text=effective_next,
            lang=effective_lang,
            staged_delta=self._staged_delta,
        )
        self._voice_candidates_extracted += len(candidates)
        if not candidates:
            return []

        eligible: list[DialogueCandidate] = []
        for cand in candidates:
            if require_speaker and not cand.speaker_id:
                self._voice_gate1_rejected += 1
                for hook in self.engine._hooks:
                    hook.on_gate_rejected(
                        "gate1",
                        cand.text,
                        "unresolved_speaker",
                        {"candidate_id": cand.candidate_id},
                    )
                continue
            eligible.append(cand)
        if not eligible:
            return []

        scored_candidates: list[DialogueCandidate] = []
        base_metadatas: list[dict[str, str]] = []
        for cand in eligible:
            eff_addressee, meta = self.engine._resolve_harvest_metadata(
                speaker_id=cand.speaker_id,
                addressee_id=cand.addressee_id,
                context=self.context,
                staged_delta=self._staged_delta,
                cutoff_step=cutoff_step,
                timeline_id=self.timeline_id,
                infer_addressee=infer_addressee,
            )
            if extra_metadata:
                for k, v in extra_metadata.items():
                    ks, vs = str(k).strip(), str(v).strip()
                    if ks and vs:
                        meta[ks] = vs
            scored_cand = (
                cand.model_copy(update={"addressee_id": eff_addressee})
                if eff_addressee != cand.addressee_id
                else cand
            )
            scored_candidates.append(scored_cand)
            base_metadatas.append(meta)

        active_scorer = scorer or self.engine.voice_scorer
        before_usage = _snapshot_component_usage(active_scorer)
        try:
            score_results = score_voice_candidates_batch(
                active_scorer,
                scored_candidates,
                base_metadatas,
                context=self.context,
            )
        finally:
            self._record_stage_usage("voice_scoring", active_scorer, before_usage)

        gate_min = self.engine.voice_bank.gate.min_quality
        effective_min_quality = gate_min if min_quality is None else float(min_quality)

        passing: list[VoiceExemplar] = []
        for cand, base_meta, score_res in zip(
            scored_candidates, base_metadatas, score_results
        ):
            merged_meta = dict(base_meta)
            merged_meta.update(score_res.metadata_updates)
            if extra_metadata:
                for k, v in extra_metadata.items():
                    ks, vs = str(k).strip(), str(v).strip()
                    if ks and vs:
                        merged_meta[ks] = vs

            if not cand.speaker_id:
                continue
            exemplar = cand.to_voice_exemplar(
                step=self.step,
                speaker_id=cand.speaker_id,
                addressee_id=cand.addressee_id,
                metadata=merged_meta,
                quality_score=score_res.score,
                timeline_id=self.timeline_id,
            )
            if exemplar.quality_score < effective_min_quality:
                self._voice_gate2_rejected += 1
                reason_suffix = f" ({score_res.reason})" if score_res.reason else ""
                for hook in self.engine._hooks:
                    hook.on_gate_rejected(
                        "gate2",
                        exemplar.text,
                        f"quality_score {exemplar.quality_score} < {effective_min_quality}{reason_suffix}",
                        {
                            "speaker_id": exemplar.speaker_id,
                            "addressee_id": exemplar.addressee_id,
                            "quality_score": exemplar.quality_score,
                            "threshold": effective_min_quality,
                            "reason": score_res.reason,
                        },
                    )
                continue
            passing.append(exemplar)

        if stage and passing:
            self._staged_exemplars.extend(passing)
            self._voice_harvested_count += len(passing)
            self._voice_harvested_ids.extend(
                ex.exemplar_id for ex in passing if ex.exemplar_id
            )
            if any(ex.quality_score < gate_min for ex in passing):
                self._bypass_gate = True
        return passing

    def trace(self, *, include_raw_delta: bool = False) -> StepTransactionTrace:
        """Builds a structured StepTransactionTrace snapshot of this transaction."""
        if self._context is None:
            raise RuntimeError("StepTransaction has not been entered.")
        duration_ms = self._duration_ms
        if duration_ms is None and self._started_perf is not None:
            duration_ms = round((time.perf_counter() - self._started_perf) * 1000.0, 3)

        delta_counts: dict[str, int] = {}
        if self._staged_delta is not None:
            delta_counts = {
                "entities": len(self._staged_delta.entities),
                "edges": len(self._staged_delta.edges),
                "episodes": len(self._staged_delta.episodes),
                "inform": len(self._staged_delta.inform),
            }

        if self._voice_harvested_count > 0:
            harvest_ids = (
                list(self._voice_harvested_ids)
                if self._voice_harvested_ids
                else list(self._indexed_exemplar_ids)
            )
        else:
            harvest_ids = []
        return StepTransactionTrace(
            timeline_id=self.timeline_id,
            step=self.step,
            lang=self.lang,
            rerun_clean=self.rerun_clean,
            status=self._status,
            content_hash=self._content_hash,
            checkpoint_metadata=dict(self._checkpoint_metadata) if self._checkpoint_metadata else {},
            duration_ms=duration_ms,
            context=self._context.to_summary(),
            audit_reports=[t.model_copy(deep=True) for t in self._audit_traces],
            extraction_skipped_by_policy=self._extraction_skipped_by_policy,
            raw_delta=self.last_raw_delta.model_copy(deep=True) if (include_raw_delta and self.last_raw_delta is not None) else None,
            staged_delta=self._staged_delta.model_copy(deep=True) if self._staged_delta is not None else None,
            delta_counts=delta_counts,
            staged_extra_episodes=self._staged_extra_episodes,
            staged_extra_exemplars=self._staged_extra_exemplars,
            voice_harvest=VoiceHarvestTrace(
                candidates_extracted=self._voice_candidates_extracted,
                gate1_rejected=self._voice_gate1_rejected,
                gate2_rejected=self._voice_gate2_rejected,
                staged_count=self._voice_harvested_count,
                staged_exemplar_ids=harvest_ids,
            ),
            harvested_voice_count=self._voice_harvested_count,
            committed_episode_ids=list(self._committed_episode_ids),
            indexed_exemplar_ids=list(self._indexed_exemplar_ids),
            error=self._error,
            usage=_build_transaction_usage_summary(self._stage_usages),
        )

    def to_trace_dict(
        self,
        *,
        include_raw_delta: bool = False,
        exclude_none: bool = True,
    ) -> dict[str, Any]:
        """Serializes tx.trace() to a plain JSON-compatible dict for host-app debug logs."""
        return self.trace(include_raw_delta=include_raw_delta).model_dump(
            mode="json",
            exclude_none=exclude_none,
        )

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            if exc_type is not None:
                self._status = "rolled_back"
                exc_name = getattr(exc_type, "__name__", str(exc_type))
                self._error = f"{exc_name}: {exc_val}"
                return
            try:
                with self.engine._store_transaction():
                    if self.rerun_clean:
                        self.engine.rollback_to(self.step - 1, timeline_id=self.timeline_id)
                    if (
                        self._committed
                        or self._staged_delta
                        or self._staged_exemplars
                        or self._staged_episodes
                    ):
                        inflection_gen = self.engine.inflection_generator
                        before_inflect = _snapshot_component_usage(inflection_gen)
                        ep_sink: list[EpisodeRecord] = []
                        ep_token = _ACTIVE_COMMIT_EPISODES.set(ep_sink)
                        try:
                            indexed_ex = self.engine.commit_step(
                                self.step,
                                delta=self._staged_delta,
                                exemplars=self._staged_exemplars,
                                episodes=self._staged_episodes,
                                bypass_gate=self._bypass_gate,
                                timeline_id=self.timeline_id,
                                content_hash=self._content_hash,
                                checkpoint_metadata=self._checkpoint_metadata,
                            )
                        finally:
                            _ACTIVE_COMMIT_EPISODES.reset(ep_token)
                            self._record_stage_usage("inflection", inflection_gen, before_inflect)
                        self._committed_episode_ids = [ep.episode_id for ep in ep_sink]
                        self._indexed_exemplar_ids = [ex.exemplar_id for ex in indexed_ex]
                        self._status = "committed"
                    else:
                        self._status = "skipped"
            except Exception as exc:
                self._status = "rolled_back"
                self._error = f"{type(exc).__name__}: {exc}"
                raise
        finally:
            if self._started_perf is not None:
                self._duration_ms = round((time.perf_counter() - self._started_perf) * 1000.0, 3)
