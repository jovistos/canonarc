from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import re
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal, Self

from .audit import (
    AuditReport,
    AuditRule,
    AuditSeverity,
    run_audit_draft as _run_audit_draft,
    run_audit_step as _run_audit_step,
    run_audit_timeline as _run_audit_timeline,
)
from .bible import (
    BibleExportFormat,
    BibleExportResult,
    BibleSyncMode,
    BibleSyncReport,
    export_bible as _export_bible,
    render_bible as _render_bible,
    sync_bible as _sync_bible,
)
from .embedders import (
    Embedder,
    EmbedderMismatchError,
    embed_texts_batch,
    get_embedder_info,
)
from .episodic import DEFAULT_EPISODIC_POLICY, EpisodeRanker
from .extractors import DeltaExtractor
from .hooks import EngineHooks
from .inflections import InflectionGenerator, InflectionRequest, generate_inflections_batch
from .models import (
    AliasRecord,
    DialogueCandidate,
    DirectedEdgeRecord,
    EdgeDelta,
    EntityDelta,
    EntityMergeReport,
    EntityRecord,
    EpisodeRecord,
    EpistemicMode,
    InformDelta,
    ResolvedMention,
    ResumePlan,
    StepCheckpoint,
    StepContext,
    StepDelta,
    TimelineInfo,
    VoiceExemplar,
)
from .presets import (
    DEFAULT_SCHEMA_POLICY,
    DomainPreset,
    InteractiveFictionPreset,
    SchemaPolicy,
)
from .resolver import DefaultLexicalResolver, MentionResolver
from .store import CausalStore, CheckpointStore, MemoryStore, TimelineBranchConflictError
from .voice import (
    HeuristicVoiceQualityScorer,
    IngestionGate,
    VoiceBank,
    VoiceQualityScorer,
    score_voice_candidates_batch,
)


KnownEntityScope = Literal["scene", "characters_compact", "all_compact"]


def compute_step_hash(
    text: str,
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Computes a deterministic SHA-256 hex digest for step resumability checks."""
    payload: dict[str, Any] = {"text": text}
    if model is not None:
        payload["model"] = model
    if prompt_version is not None:
        payload["prompt_version"] = prompt_version
    if extra:
        payload["extra"] = extra
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    def __enter__(self) -> Self:
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
    ) -> AuditReport:
        """Audits a pre-commit draft translation against tx.context before commit()/extract_and_commit()."""
        return _run_audit_draft(
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
        """Runs the configured DeltaExtractor on scene_text, normalizes IDs, and stages the StepDelta.

        When Tier 3 (rolling state deltas) is disabled by the active ``SchemaPolicy`` (e.g.,
        ``TierProfile.LEXICON_ONLY``), this is a zero-LLM-call no-op that still stages the
        step checkpoint so callers can share one unified loop across fiction and non-fiction.
        """
        if not self.engine._policy.has_tier(3):
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
        if callable(extract_messages):
            raw_delta = extract_messages(system_prompt, user_prompt, scene_text)
        else:
            raw_delta = active_extractor.extract_delta(
                f"{system_prompt}\n\n{user_prompt}",
                scene_text,
            )
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
        """Runs Gate-1 extraction, attaches active dyadic edge/persona metadata, scores via Gate 2, and stages passing VoiceExemplars.

        When Tier 5 (Voice Bank) is disabled by the active ``SchemaPolicy`` (e.g., ``LEXICON_ONLY``
        or ``STATE_ONLY``), this is a zero-work no-op returning ``[]``.
        """
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
        if not candidates:
            return []

        eligible: list[DialogueCandidate] = []
        for cand in candidates:
            if require_speaker and not cand.speaker_id:
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
        score_results = score_voice_candidates_batch(
            active_scorer,
            scored_candidates,
            base_metadatas,
            context=self.context,
        )

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
            if any(ex.quality_score < gate_min for ex in passing):
                self._bypass_gate = True
        return passing

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_type is None:
            with self.engine._store_transaction():
                if self.rerun_clean:
                    self.engine.rollback_to(self.step - 1, timeline_id=self.timeline_id)
                if self._committed or self._staged_delta or self._staged_exemplars or self._staged_episodes:
                    self.engine.commit_step(
                        self.step,
                        delta=self._staged_delta,
                        exemplars=self._staged_exemplars,
                        episodes=self._staged_episodes,
                        bypass_gate=self._bypass_gate,
                        timeline_id=self.timeline_id,
                        content_hash=self._content_hash,
                        checkpoint_metadata=self._checkpoint_metadata,
                    )


class TimelineSession:
    """Concurrency-safe, immutable-timeline session view bound to a single timeline_id."""

    def __init__(self, engine: CanonArcEngine, timeline_id: str) -> None:
        self.engine = engine
        self.timeline_id = timeline_id

    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str | AliasRecord] = (),
        step: int = 0,
        target_lang: str | None = None,
        auto_inflect: bool = True,
    ) -> None:
        self.engine.register_entity(
            entity,
            aliases=aliases,
            step=step,
            timeline_id=self.timeline_id,
            target_lang=target_lang,
            auto_inflect=auto_inflect,
        )

    def seed_entities(
        self,
        entities: Sequence[EntityRecord],
        *,
        step: int = 0,
        target_lang: str | None = None,
        auto_inflect: bool = True,
        batch_size: int = 25,
    ) -> None:
        self.engine.seed_entities(
            entities,
            step=step,
            timeline_id=self.timeline_id,
            target_lang=target_lang,
            auto_inflect=auto_inflect,
            batch_size=batch_size,
        )

    def expand_entity_inflections(
        self,
        entity_id: str,
        *,
        target_lang: str | None = None,
        step: int | None = None,
    ) -> list[AliasRecord]:
        return self.engine.expand_entity_inflections(
            entity_id,
            target_lang=target_lang,
            step=step,
            timeline_id=self.timeline_id,
        )

    def merge_entities(
        self,
        duplicate_id: str,
        canonical_id: str,
        *,
        step: int | None = None,
        merge_attributes: bool = True,
        migrate_edges: bool = True,
        migrate_voice: bool = True,
        migrate_episodes: bool = True,
        conflict_strategy: Literal["canonical_wins", "duplicate_wins"] = "canonical_wins",
        allow_kind_mismatch: bool = False,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> EntityMergeReport:
        return self.engine.merge_entities(
            duplicate_id,
            canonical_id,
            step=step,
            timeline_id=self.timeline_id,
            merge_attributes=merge_attributes,
            migrate_edges=migrate_edges,
            migrate_voice=migrate_voice,
            migrate_episodes=migrate_episodes,
            conflict_strategy=conflict_strategy,
            allow_kind_mismatch=allow_kind_mismatch,
            reason=reason,
            dry_run=dry_run,
        )

    def retrieve_context(
        self,
        step: int,
        *,
        text: str = "",
        prev_text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int | None = None,
        top_k_episodes: int | None = None,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        location_id: str | None = None,
        inclusive: bool = False,
        fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only",
        story_time: float | str | None = None,
        chapter_id: str | None = None,
        scene_id: str | None = None,
        voice_constraints: dict[str, str] | None = None,
        lang: str | None = None,
    ) -> StepContext:
        return self.engine.retrieve_context(
            step,
            text=text,
            prev_text=prev_text,
            mentions=mentions,
            top_k_voice=top_k_voice,
            top_k_episodes=top_k_episodes,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            edge_types=edge_types,
            location_id=location_id,
            inclusive=inclusive,
            timeline_id=self.timeline_id,
            fallback_on_empty_mentions=fallback_on_empty_mentions,
            story_time=story_time,
            chapter_id=chapter_id,
            scene_id=scene_id,
            voice_constraints=voice_constraints,
            lang=lang,
        )

    def step_transaction(
        self,
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
        rerun_clean: bool = True,
        prev_text: str = "",
        next_text: str = "",
        lang: str | None = None,
        content_hash: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> StepTransaction:
        return self.engine.step_transaction(
            step,
            text=text,
            mentions=mentions,
            top_k_voice=top_k_voice,
            top_k_episodes=top_k_episodes,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            edge_types=edge_types,
            location_id=location_id,
            timeline_id=self.timeline_id,
            rerun_clean=rerun_clean,
            prev_text=prev_text,
            next_text=next_text,
            lang=lang,
            content_hash=content_hash,
            checkpoint_metadata=checkpoint_metadata,
        )

    def apply_delta(self, step: int, delta: StepDelta) -> list[EpisodeRecord]:
        return self.engine.apply_delta(step, delta, timeline_id=self.timeline_id)

    def index_episodes(self, step: int, episodes: Sequence[EpisodeRecord]) -> list[EpisodeRecord]:
        return self.engine.index_episodes(step, episodes, timeline_id=self.timeline_id)

    def inform_entity(
        self,
        step: int,
        *,
        episode_id: str,
        entity_id: str,
        epistemic: EpistemicMode = "told",
        believed_summary: str | None = None,
    ) -> None:
        self.engine.inform_entity(
            step,
            episode_id=episode_id,
            entity_id=entity_id,
            epistemic=epistemic,
            believed_summary=believed_summary,
            timeline_id=self.timeline_id,
        )

    def query_episodes(self, **kwargs: Any) -> list[EpisodeRecord]:
        return self.engine.query_episodes(timeline_id=self.timeline_id, **kwargs)

    def extract_dialogue_candidates(
        self,
        text: str,
        *,
        source_text: str = "",
        step: int | None = None,
        prev_text: str = "",
        next_text: str = "",
        lang: str | None = None,
    ) -> list[DialogueCandidate]:
        return self.engine.extract_dialogue_candidates(
            text,
            source_text=source_text,
            step=step,
            timeline_id=self.timeline_id,
            prev_text=prev_text,
            next_text=next_text,
            lang=lang,
        )

    def index_voice_exemplars(
        self,
        step: int,
        exemplars: Sequence[VoiceExemplar],
        *,
        bypass_gate: bool = False,
    ) -> list[VoiceExemplar]:
        return self.engine.index_voice_exemplars(
            step, exemplars, bypass_gate=bypass_gate, timeline_id=self.timeline_id
        )

    def commit_step(
        self,
        step: int,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        *,
        bypass_gate: bool = False,
        content_hash: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> list[VoiceExemplar]:
        return self.engine.commit_step(
            step,
            delta=delta,
            exemplars=exemplars,
            episodes=episodes,
            bypass_gate=bypass_gate,
            timeline_id=self.timeline_id,
            content_hash=content_hash,
            checkpoint_metadata=checkpoint_metadata,
        )

    def get_step_checkpoint(self, step: int) -> StepCheckpoint | None:
        return self.engine.get_step_checkpoint(step, timeline_id=self.timeline_id)

    def is_step_committed(
        self,
        step: int,
        *,
        content_hash: str | None = None,
        metadata_match: dict[str, Any] | None = None,
    ) -> bool:
        return self.engine.is_step_committed(
            step,
            content_hash=content_hash,
            metadata_match=metadata_match,
            timeline_id=self.timeline_id,
        )

    def last_committed_step(self) -> int | None:
        return self.engine.last_committed_step(timeline_id=self.timeline_id)

    def list_step_checkpoints(self, *, cutoff_step: int | None = None) -> list[StepCheckpoint]:
        return self.engine.list_step_checkpoints(
            timeline_id=self.timeline_id,
            cutoff_step=cutoff_step,
        )

    def consolidate_episodes(self, step: int, **kwargs: Any) -> EpisodeRecord | None:
        return self.engine.consolidate_episodes(step, timeline_id=self.timeline_id, **kwargs)

    def rollback_to(self, step: int, *, force: bool = False) -> int:
        return self.engine.rollback_to(step, timeline_id=self.timeline_id, force=force)

    def plan_resume(
        self,
        step_hashes: Sequence[tuple[int, str]],
        *,
        causal: bool | None = None,
        metadata_match: Mapping[str, Any] | None = None,
        apply_rollback: bool = False,
        force: bool = False,
        prune_tail: bool = True,
    ) -> ResumePlan:
        return self.engine.plan_resume(
            step_hashes,
            causal=causal,
            metadata_match=metadata_match,
            timeline_id=self.timeline_id,
            apply_rollback=apply_rollback,
            force=force,
            prune_tail=prune_tail,
        )

    def branch(
        self,
        at_step: int,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineSession:
        self.engine.branch(at_step, name, from_timeline=self.timeline_id, metadata=metadata)
        return TimelineSession(self.engine, name)

    def get_history(self, subject_key: str, *, record_type: str | None = None) -> list[dict[str, Any]]:
        return self.engine.get_history(
            subject_key, timeline_id=self.timeline_id, record_type=record_type
        )

    def export_snapshot(self, target_path: str | Path, *, at_step: int) -> Path:
        return self.engine.export_snapshot(target_path, at_step=at_step, timeline_id=self.timeline_id)

    def render_bible(
        self,
        *,
        format: BibleExportFormat = "md",
        at_step: int | None = None,
        kinds: Sequence[str] | None = None,
        include_state: bool = False,
        include_edges: bool | None = None,
        include_aliases: bool = True,
        include_inflections: bool | None = None,
        lang: str | None = None,
    ) -> tuple[str, BibleExportResult]:
        return self.engine.render_bible(
            format=format,
            at_step=at_step,
            timeline_id=self.timeline_id,
            kinds=kinds,
            include_state=include_state,
            include_edges=include_edges,
            include_aliases=include_aliases,
            include_inflections=include_inflections,
            lang=lang,
        )

    def export_bible(
        self,
        path: str | Path,
        *,
        format: Literal["auto", "csv", "json", "md"] = "auto",
        at_step: int | None = None,
        kinds: Sequence[str] | None = None,
        include_state: bool = False,
        include_edges: bool | None = None,
        include_aliases: bool = True,
        include_inflections: bool | None = None,
        lang: str | None = None,
        excel_bom: bool = False,
    ) -> BibleExportResult:
        return self.engine.export_bible(
            path,
            format=format,
            at_step=at_step,
            timeline_id=self.timeline_id,
            kinds=kinds,
            include_state=include_state,
            include_edges=include_edges,
            include_aliases=include_aliases,
            include_inflections=include_inflections,
            lang=lang,
            excel_bom=excel_bom,
        )

    def sync_bible(
        self,
        path: str | Path,
        *,
        format: Literal["auto", "csv", "json"] = "auto",
        step: int = 0,
        mode: BibleSyncMode = "upsert",
        target_lang: str | None = None,
        auto_inflect: bool = True,
        batch_size: int = 25,
        dry_run: bool = False,
        column_map: dict[str, str] | None = None,
    ) -> BibleSyncReport:
        return self.engine.sync_bible(
            path,
            format=format,
            step=step,
            timeline_id=self.timeline_id,
            mode=mode,
            target_lang=target_lang,
            auto_inflect=auto_inflect,
            batch_size=batch_size,
            dry_run=dry_run,
            column_map=column_map,
        )

    def audit_step(
        self,
        step: int,
        source_text: str = "",
        target_text: str = "",
        *,
        rules: Sequence[AuditRule] | None = None,
        include_rules: Sequence[str] | None = None,
        exclude_rules: Sequence[str] | None = None,
        min_severity: AuditSeverity = "info",
        lang: str | None = None,
    ) -> AuditReport:
        return self.engine.audit_step(
            step,
            source_text=source_text,
            target_text=target_text,
            timeline_id=self.timeline_id,
            rules=rules,
            include_rules=include_rules,
            exclude_rules=exclude_rules,
            min_severity=min_severity,
            lang=lang,
        )

    def audit_timeline(
        self,
        *,
        from_step: int = 0,
        to_step: int | None = None,
        texts: Mapping[int, str | tuple[str, str]] | None = None,
        rules: Sequence[AuditRule] | None = None,
        include_rules: Sequence[str] | None = None,
        exclude_rules: Sequence[str] | None = None,
        min_severity: AuditSeverity = "info",
        lang: str | None = None,
    ) -> AuditReport:
        return self.engine.audit_timeline(
            timeline_id=self.timeline_id,
            from_step=from_step,
            to_step=to_step,
            texts=texts,
            rules=rules,
            include_rules=include_rules,
            exclude_rules=exclude_rules,
            min_severity=min_severity,
            lang=lang,
        )

    def normalize_delta(self, delta: StepDelta, *, step: int | None = None) -> StepDelta:
        return self.engine.normalize_delta(delta, step=step, timeline_id=self.timeline_id)

    def format_context(self, ctx: StepContext) -> str:
        return self.engine.format_context(ctx)

    def build_extraction_messages(
        self,
        ctx: StepContext,
        scene_text: str,
        *,
        prev_text: str = "",
        next_text: str = "",
        compact_schema: bool = False,
        known_entity_scope: KnownEntityScope = "scene",
        max_roster_entities: int | None = None,
    ) -> tuple[str, str]:
        return self.engine.build_extraction_messages(
            ctx,
            scene_text,
            prev_text=prev_text,
            next_text=next_text,
            compact_schema=compact_schema,
            known_entity_scope=known_entity_scope,
            max_roster_entities=max_roster_entities,
        )

    def build_extraction_prompt(
        self,
        ctx: StepContext,
        scene_text: str,
        *,
        prev_text: str = "",
        next_text: str = "",
        compact_schema: bool = False,
        known_entity_scope: KnownEntityScope = "scene",
        max_roster_entities: int | None = None,
    ) -> str:
        return self.engine.build_extraction_prompt(
            ctx,
            scene_text,
            prev_text=prev_text,
            next_text=next_text,
            compact_schema=compact_schema,
            known_entity_scope=known_entity_scope,
            max_roster_entities=max_roster_entities,
        )


class CanonArcEngine:
    """High-level facade for Tier 1-6 narrative memory, episodic recall, time travel, and voice retrieval."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        store: MemoryStore | None = None,
        timeline_id: str = "main",
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
        preset: DomainPreset | None = None,
        mention_resolver: MentionResolver | None = None,
        extractor: DeltaExtractor | None = None,
        episode_ranker: EpisodeRanker | None = None,
        inflection_generator: InflectionGenerator | None = None,
        voice_scorer: VoiceQualityScorer | None = None,
        hooks: Sequence[EngineHooks] = (),
        audit_rules: Sequence[AuditRule] | None = None,
        on_embedder_mismatch: Literal["raise", "warn", "reindex", "ignore"] = "warn",
    ) -> None:
        self.store: MemoryStore = store if store is not None else CausalStore(db_path)
        self._default_timeline = timeline_id
        self._timeline_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"canonarc_timeline_{id(self)}", default=None
        )
        self.voice_bank = VoiceBank(
            embedder=embedder, gate=gate, episode_ranker=episode_ranker
        )
        self.preset: DomainPreset = preset or InteractiveFictionPreset()
        self.mention_resolver: MentionResolver = mention_resolver or DefaultLexicalResolver()
        self.extractor: DeltaExtractor | None = extractor
        self.inflection_generator: InflectionGenerator | None = inflection_generator
        self.voice_scorer: VoiceQualityScorer = voice_scorer or HeuristicVoiceQualityScorer()
        self._hooks: list[EngineHooks] = list(hooks)
        self.audit_rules: list[AuditRule] | None = (
            list(audit_rules) if audit_rules is not None else None
        )
        self._on_embedder_mismatch = on_embedder_mismatch
        self._embedder_verified = False
        if self._policy.has_tier(5) or self._policy.has_tier(6):
            self._check_embedder_compatibility()

    @property
    def active_timeline(self) -> str:
        scoped = self._timeline_var.get()
        return scoped if scoped is not None else self._default_timeline

    @active_timeline.setter
    def active_timeline(self, value: str) -> None:
        self._default_timeline = value
        if self._timeline_var.get() is not None:
            self._timeline_var.set(value)

    @contextlib.contextmanager
    def use_timeline(self, timeline_id: str) -> Iterator[TimelineSession]:
        """Task/thread-local context manager scoping active_timeline without mutating other requests."""
        if not self.store.get_timeline(timeline_id):
            raise ValueError(f"Timeline '{timeline_id}' does not exist.")
        token = self._timeline_var.set(timeline_id)
        try:
            yield TimelineSession(self, timeline_id)
        finally:
            self._timeline_var.reset(token)

    def session(self, timeline_id: str | None = None) -> TimelineSession:
        """Returns a concurrency-safe TimelineSession bound to timeline_id."""
        tl = timeline_id or self.active_timeline
        if not self.store.get_timeline(tl):
            raise ValueError(f"Timeline '{tl}' does not exist.")
        return TimelineSession(self, tl)

    def add_hook(self, hook: EngineHooks) -> None:
        self._hooks.append(hook)

    def remove_hook(self, hook: EngineHooks) -> None:
        self._hooks = [h for h in self._hooks if h is not hook]

    @property
    def _policy(self) -> SchemaPolicy:
        return getattr(self.preset, "policy", DEFAULT_SCHEMA_POLICY)

    def close(self) -> None:
        self.store.close()

    def _check_embedder_compatibility(self) -> None:
        """Verifies that the active embedder matches any embedder/dimension recorded in the store."""
        get_meta = getattr(self.store, "get_embedding_metadata", None)
        if not callable(get_meta):
            return
        stored_id, stored_dim = get_meta()
        if stored_id is None and stored_dim is None:
            return
        current_id, current_dim = get_embedder_info(self.voice_bank.embedder)
        id_mismatch = stored_id is not None and stored_id != current_id
        dim_mismatch = stored_dim is not None and current_dim is not None and stored_dim != current_dim
        if not id_mismatch and not dim_mismatch:
            self._embedder_verified = True
            return

        msg = (
            f"Store embeddings were created with embedder={stored_id!r}, dim={stored_dim}, "
            f"but engine was initialized with embedder={current_id!r}, dim={current_dim}."
        )
        policy = self._on_embedder_mismatch
        if policy == "raise":
            raise EmbedderMismatchError(msg)
        if policy == "reindex":
            self.reindex_embeddings()
            return
        if policy == "warn":
            warnings.warn(
                f"{msg} Call engine.reindex_embeddings() to migrate stored vectors.",
                UserWarning,
                stacklevel=3,
            )

    def _record_embedder_metadata(self, vectors: Sequence[Any]) -> None:
        """Records embedder_id and vector dimension in store_metadata on first write."""
        if self._embedder_verified or not vectors:
            return
        set_meta = getattr(self.store, "set_embedding_metadata", None)
        get_meta = getattr(self.store, "get_embedding_metadata", None)
        if not callable(set_meta) or not callable(get_meta):
            self._embedder_verified = True
            return
        dim = len(vectors[0])
        if dim <= 0:
            return
        stored_id, stored_dim = get_meta()
        current_id, _ = get_embedder_info(self.voice_bank.embedder)
        if stored_id is None and stored_dim is None:
            set_meta(current_id, dim)
            self._embedder_verified = True
            return
        if (stored_id is not None and stored_id != current_id) or (
            stored_dim is not None and stored_dim != dim
        ):
            msg = (
                f"Store embeddings were created with embedder={stored_id!r}, dim={stored_dim}, "
                f"but active embedder produced embedder={current_id!r}, dim={dim}."
            )
            if self._on_embedder_mismatch == "raise":
                raise EmbedderMismatchError(msg)
            if self._on_embedder_mismatch == "warn":
                warnings.warn(
                    f"{msg} Call engine.reindex_embeddings() to migrate stored vectors.",
                    UserWarning,
                    stacklevel=3,
                )
        self._embedder_verified = True

    def reindex_embeddings(self, *, batch_size: int = 128) -> dict[str, int]:
        """Re-embeds all stored episodes and voice exemplars in batches using the active embedder."""
        list_ep = getattr(self.store, "list_all_episodes_for_reembed", None)
        list_vo = getattr(self.store, "list_all_voice_exemplars_for_reembed", None)
        upd_ep = getattr(self.store, "update_episode_embeddings", None)
        upd_vo = getattr(self.store, "update_voice_embeddings", None)
        set_meta = getattr(self.store, "set_embedding_metadata", None)
        if not (
            callable(list_ep)
            and callable(list_vo)
            and callable(upd_ep)
            and callable(upd_vo)
            and callable(set_meta)
        ):
            raise NotImplementedError("Active store backend does not support reindex_embeddings().")

        ep_rows: list[tuple[str, str, str]] = list_ep()
        vo_rows: list[tuple[str, str, str]] = list_vo()
        step_size = max(1, batch_size)
        detected_dim: int | None = None

        ep_updates: list[tuple[str, str, bytes]] = []
        for i in range(0, len(ep_rows), step_size):
            chunk = ep_rows[i : i + step_size]
            vecs = embed_texts_batch(self.voice_bank.embedder, [r[2] for r in chunk])
            if vecs and detected_dim is None:
                detected_dim = len(vecs[0])
            for (tl, eid, _), vec in zip(chunk, vecs):
                ep_updates.append((tl, eid, vec.tobytes()))

        vo_updates: list[tuple[str, str, bytes]] = []
        for i in range(0, len(vo_rows), step_size):
            chunk = vo_rows[i : i + step_size]
            vecs = embed_texts_batch(self.voice_bank.embedder, [r[2] for r in chunk])
            if vecs and detected_dim is None:
                detected_dim = len(vecs[0])
            for (tl, xid, _), vec in zip(chunk, vecs):
                vo_updates.append((tl, xid, vec.tobytes()))

        current_id, known_dim = get_embedder_info(self.voice_bank.embedder)
        final_dim = detected_dim or known_dim
        if final_dim is None:
            probe = self.voice_bank.embedder.embed_text("probe")
            final_dim = len(probe)

        with self._store_transaction():
            n_ep = upd_ep(ep_updates)
            n_vo = upd_vo(vo_updates)
            set_meta(current_id, final_dim)
        self._embedder_verified = True
        return {"episodes": n_ep, "voice_exemplars": n_vo}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # --- Tier 1 Seeding ---

    def _resolve_mentions(
        self,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
        *,
        lang: str | None = None,
    ) -> list[ResolvedMention]:
        if lang is not None:
            try:
                return self.mention_resolver.resolve(  # type: ignore[call-arg]
                    text=text,
                    explicit_mentions=explicit_mentions,
                    entities=entities,
                    aliases=aliases,
                    lang=lang,
                )
            except TypeError:
                pass
        return self.mention_resolver.resolve(
            text=text,
            explicit_mentions=explicit_mentions,
            entities=entities,
            aliases=aliases,
        )

    def _make_inflection_request(
        self,
        entity: EntityRecord,
        *,
        target_lang: str | None = None,
    ) -> InflectionRequest | None:
        raw_target = entity.base_attributes.get("target_name")
        if not isinstance(raw_target, str) or not raw_target.strip():
            return None
        raw_lang = (
            target_lang
            or entity.base_attributes.get("target_lang")
            or entity.base_attributes.get("lang")
        )
        eff_lang = raw_lang.strip().lower() if isinstance(raw_lang, str) and raw_lang.strip() else None
        if not eff_lang:
            return None
        gender = entity.base_attributes.get("gender")
        declension = entity.base_attributes.get("declension")
        return InflectionRequest(
            name=raw_target.strip(),
            lang=eff_lang,
            gender=str(gender) if gender is not None else None,
            kind=entity.kind,
            declension=str(declension) if declension is not None else None,
        )

    def _build_target_aliases(
        self,
        entity: EntityRecord,
        existing_aliases: Sequence[str | AliasRecord],
        *,
        target_lang: str | None = None,
        auto_inflect: bool = True,
        prefetched_forms: Sequence[str] | None = None,
    ) -> list[AliasRecord]:
        """Auto-registers base_attributes['target_name'] and optional one-shot LLM inflections."""
        raw_target = entity.base_attributes.get("target_name")
        if not isinstance(raw_target, str) or not raw_target.strip():
            return []
        target_name = raw_target.strip()
        raw_lang = (
            target_lang
            or entity.base_attributes.get("target_lang")
            or entity.base_attributes.get("lang")
        )
        eff_lang = raw_lang.strip().lower() if isinstance(raw_lang, str) and raw_lang.strip() else None

        existing_keys: set[str] = set()
        for a in existing_aliases:
            s = a.alias.strip() if isinstance(a, AliasRecord) else a.strip()
            if s:
                existing_keys.add(s.lower())
        canonical_lower = entity.canonical_name.strip().lower()

        generated: list[AliasRecord] = []

        def _maybe_add(
            form: str,
            *,
            kind: Literal["permanent", "inflection"] = "permanent",
            allow_canonical: bool = False,
        ) -> None:
            clean = form.strip()
            if not clean:
                return
            key = clean.lower()
            if key in existing_keys:
                return
            if not allow_canonical and key == canonical_lower:
                return
            existing_keys.add(key)
            generated.append(
                AliasRecord(
                    alias=clean,
                    entity_id=entity.entity_id,
                    kind=kind,
                    lang=eff_lang,
                )
            )

        _maybe_add(target_name, kind="permanent", allow_canonical=eff_lang is not None)

        forms: Sequence[str] = ()
        if prefetched_forms is not None:
            forms = prefetched_forms
        elif auto_inflect and eff_lang and self.inflection_generator is not None:
            gender = entity.base_attributes.get("gender")
            declension = entity.base_attributes.get("declension")
            forms = self.inflection_generator.generate_inflections(
                target_name,
                eff_lang,
                gender=str(gender) if gender is not None else None,
                kind=entity.kind,
                declension=str(declension) if declension is not None else None,
            )
        for form in forms:
            _maybe_add(form, kind="inflection")

        return generated

    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str | AliasRecord] = (),
        step: int = 0,
        timeline_id: str | None = None,
        target_lang: str | None = None,
        auto_inflect: bool = True,
    ) -> None:
        tl = timeline_id or self.active_timeline
        extra_aliases = self._build_target_aliases(
            entity,
            aliases,
            target_lang=target_lang,
            auto_inflect=auto_inflect,
        )
        all_aliases: list[str | AliasRecord] = [*aliases, *extra_aliases]
        self.store.register_entity(entity, aliases=all_aliases, step=step, timeline_id=tl)

    def seed_entities(
        self,
        entities: Sequence[EntityRecord],
        *,
        step: int = 0,
        timeline_id: str | None = None,
        target_lang: str | None = None,
        auto_inflect: bool = True,
        batch_size: int = 25,
    ) -> None:
        tl = timeline_id or self.active_timeline
        prefetched_by_idx: dict[int, list[str]] = {}
        if auto_inflect and self.inflection_generator is not None:
            pending_indices: list[int] = []
            pending_requests: list[InflectionRequest] = []
            for idx, ent in enumerate(entities):
                req = self._make_inflection_request(ent, target_lang=target_lang)
                if req is not None:
                    pending_indices.append(idx)
                    pending_requests.append(req)
            if pending_requests:
                batched_forms = generate_inflections_batch(
                    self.inflection_generator,
                    pending_requests,
                    batch_size=batch_size,
                )
                for idx, forms in zip(pending_indices, batched_forms):
                    prefetched_by_idx[idx] = forms

        with self._store_transaction():
            for idx, ent in enumerate(entities):
                extra_aliases = self._build_target_aliases(
                    ent,
                    (),
                    target_lang=target_lang,
                    auto_inflect=False,
                    prefetched_forms=prefetched_by_idx.get(idx),
                )
                self.store.register_entity(
                    ent,
                    aliases=extra_aliases,
                    step=step,
                    timeline_id=tl,
                )

    def expand_entity_inflections(
        self,
        entity_id: str,
        *,
        target_lang: str | None = None,
        step: int | None = None,
        timeline_id: str | None = None,
    ) -> list[AliasRecord]:
        """One-shot inflection expansion for an already-registered or Zero-Seed discovered entity."""
        tl = timeline_id or self.active_timeline
        if step is None:
            tl_info = self.store.get_timeline(tl)
            eff_step = tl_info.head_step if tl_info is not None else 0
        else:
            eff_step = step

        entities = self.store.get_all_active_entities(tl, eff_step)
        entity = next((e for e in entities if e.entity_id == entity_id), None)
        if entity is None:
            raise ValueError(f"Entity '{entity_id}' not found on timeline '{tl}' at step {eff_step}.")

        active_aliases = [
            a for a in self.store.get_all_active_aliases(tl, eff_step) if a.entity_id == entity_id
        ]
        new_aliases = self._build_target_aliases(
            entity,
            active_aliases,
            target_lang=target_lang,
            auto_inflect=True,
        )
        if new_aliases:
            self.apply_delta(
                eff_step,
                StepDelta(
                    entities=[
                        EntityDelta(
                            entity_id=entity_id,
                            add_aliases=list(new_aliases),
                        )
                    ]
                ),
                timeline_id=tl,
            )
        return new_aliases

    def merge_entities(
        self,
        duplicate_id: str,
        canonical_id: str,
        *,
        step: int | None = None,
        timeline_id: str | None = None,
        merge_attributes: bool = True,
        migrate_edges: bool = True,
        migrate_voice: bool = True,
        migrate_episodes: bool = True,
        conflict_strategy: Literal["canonical_wins", "duplicate_wins"] = "canonical_wins",
        allow_kind_mismatch: bool = False,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> EntityMergeReport:
        """Atomically merges duplicate_id into canonical_id at (timeline_id, step).

        Promotes duplicate_id's canonical_name, base_attributes['target_name'], and active
        AliasRecords onto canonical_id, tombstones duplicate_id and its old aliases, and
        optionally merges attributes/state and migrates active edges, voice exemplars, and
        episodes/epistemics.
        """
        dup_id = duplicate_id.strip()
        canon_id = canonical_id.strip()
        if not dup_id or not canon_id:
            raise ValueError("Both duplicate_id and canonical_id must be non-empty.")
        if dup_id == canon_id:
            raise ValueError(f"Cannot merge entity '{dup_id}' into itself.")
        if dup_id.startswith("masked:") or canon_id.startswith("masked:"):
            raise ValueError("Cannot merge synthetic 'masked:' disguise entity IDs.")
        if conflict_strategy not in ("canonical_wins", "duplicate_wins"):
            raise ValueError(
                f"Unsupported conflict_strategy: {conflict_strategy!r}. "
                "Expected 'canonical_wins' or 'duplicate_wins'."
            )

        tl = timeline_id or self.active_timeline
        tl_info = self.store.get_timeline(tl)
        if tl_info is None:
            raise ValueError(f"Timeline '{tl}' does not exist.")
        eff_step = tl_info.head_step if step is None else step
        if eff_step < 0:
            raise ValueError(f"step must be non-negative, got {eff_step}.")

        get_graph_fn = getattr(self.store, "get_active_graph_state", None)
        if callable(get_graph_fn):
            active_entities, active_aliases, active_edges = get_graph_fn(tl, eff_step)
        else:
            active_entities = self.store.get_all_active_entities(tl, eff_step)
            active_aliases = self.store.get_all_active_aliases(tl, eff_step)
            active_edges = self.store.get_all_active_edges(tl, eff_step)

        entity_by_id = {e.entity_id: e for e in active_entities}
        dup_ent = entity_by_id.get(dup_id)
        if dup_ent is None:
            raise ValueError(
                f"Duplicate entity '{dup_id}' is not active on timeline '{tl}' at step {eff_step}."
            )
        canon_ent = entity_by_id.get(canon_id)
        if canon_ent is None:
            raise ValueError(
                f"Canonical entity '{canon_id}' is not active on timeline '{tl}' at step {eff_step}."
            )

        warnings_list: list[str] = []
        if dup_ent.kind != canon_ent.kind:
            msg = (
                f"Kind mismatch: duplicate '{dup_id}' has kind='{dup_ent.kind}', "
                f"canonical '{canon_id}' has kind='{canon_ent.kind}'."
            )
            if not allow_kind_mismatch:
                raise ValueError(f"{msg} Pass allow_kind_mismatch=True to override.")
            warnings_list.append(msg)

        eff_reason = reason if reason is not None else f"Merged entity '{dup_id}' into '{canon_id}'"

        def _remap_known_to(known_to: list[str] | None) -> list[str] | None:
            if known_to is None:
                return None
            out: list[str] = []
            seen: set[str] = set()
            for eid in known_to:
                mapped = canon_id if eid == dup_id else eid
                if mapped not in seen:
                    seen.add(mapped)
                    out.append(mapped)
            return out

        # 1. Collect aliases to promote onto canonical_id and old aliases to tombstone
        canon_aliases = [a for a in active_aliases if a.entity_id == canon_id]
        dup_aliases = [a for a in active_aliases if a.entity_id == dup_id]
        existing_canon_alias_keys = {
            a.alias.strip().lower() for a in canon_aliases if a.alias.strip()
        }
        canon_name_lower = canon_ent.canonical_name.strip().lower()

        aliases_to_add: list[AliasRecord] = []

        def _queue_alias(rec: AliasRecord, *, allow_canonical: bool = False) -> None:
            clean = rec.alias.strip()
            if not clean:
                return
            key = clean.lower()
            if key in existing_canon_alias_keys:
                return
            if not allow_canonical and key == canon_name_lower:
                return
            existing_canon_alias_keys.add(key)
            aliases_to_add.append(
                rec.model_copy(
                    update={
                        "alias": clean,
                        "entity_id": canon_id,
                        "known_to": _remap_known_to(rec.known_to),
                    }
                )
            )

        # Promote duplicate's canonical_name
        _queue_alias(
            AliasRecord(alias=dup_ent.canonical_name, entity_id=canon_id, kind="permanent")
        )

        # Promote duplicate's target_name (if present and distinct from canonical's target_name)
        dup_target = dup_ent.base_attributes.get("target_name")
        if isinstance(dup_target, str) and dup_target.strip():
            raw_lang = (
                dup_ent.base_attributes.get("target_lang")
                or dup_ent.base_attributes.get("lang")
            )
            eff_lang = (
                raw_lang.strip().lower()
                if isinstance(raw_lang, str) and raw_lang.strip()
                else None
            )
            canon_target = canon_ent.base_attributes.get("target_name")
            canon_target_lower = (
                canon_target.strip().lower()
                if isinstance(canon_target, str) and canon_target.strip()
                else ""
            )
            if dup_target.strip().lower() != canon_target_lower:
                _queue_alias(
                    AliasRecord(
                        alias=dup_target.strip(),
                        entity_id=canon_id,
                        kind="permanent",
                        lang=eff_lang,
                    ),
                    allow_canonical=eff_lang is not None,
                )

        # Transfer all active AliasRecords from duplicate_id (preserving kind, lang, known_to, persona_name)
        for dup_alias in dup_aliases:
            _queue_alias(dup_alias, allow_canonical=dup_alias.lang is not None)

        dup_aliases_to_tombstone = [a.alias for a in dup_aliases]

        # Remap known_to references on other entities' aliases
        other_alias_updates: dict[str, list[AliasRecord]] = {}
        for a in active_aliases:
            if a.entity_id in (dup_id, canon_id):
                continue
            if a.known_to and dup_id in a.known_to:
                other_alias_updates.setdefault(a.entity_id, []).append(
                    a.model_copy(update={"known_to": _remap_known_to(a.known_to)})
                )

        # 2. Merge base_attributes, state, and private_state
        merged_base_attrs: dict[str, Any] = dict(canon_ent.base_attributes)
        base_attrs_merged: list[str] = []
        merged_state: dict[str, Any] = dict(canon_ent.state)
        state_keys_merged: list[str] = []
        merged_private: dict[str, Any] = dict(canon_ent.private_state)

        def _is_empty_val(val: Any) -> bool:
            return val is None or (isinstance(val, str) and not val.strip())

        if merge_attributes:
            for k, v in dup_ent.base_attributes.items():
                if k == "merged_from" or _is_empty_val(v):
                    continue
                if conflict_strategy == "duplicate_wins" or k not in merged_base_attrs or _is_empty_val(merged_base_attrs[k]):
                    if merged_base_attrs.get(k) != v:
                        merged_base_attrs[k] = v
                        base_attrs_merged.append(k)

            if self._policy.has_tier(3):
                for k, v in dup_ent.state.items():
                    if _is_empty_val(v):
                        continue
                    if conflict_strategy == "duplicate_wins" or k not in merged_state or _is_empty_val(merged_state[k]):
                        if merged_state.get(k) != v:
                            merged_state[k] = v
                            state_keys_merged.append(k)
                for k, v in dup_ent.private_state.items():
                    if _is_empty_val(v):
                        continue
                    if conflict_strategy == "duplicate_wins" or k not in merged_private or _is_empty_val(merged_private[k]):
                        if merged_private.get(k) != v:
                            merged_private[k] = v

        # Track merged_from provenance for normalize_delta redirects
        merged_from_list: list[str] = []
        for src_list in (
            canon_ent.base_attributes.get("merged_from"),
            dup_ent.base_attributes.get("merged_from"),
            [dup_id],
        ):
            if isinstance(src_list, list):
                for item in src_list:
                    if isinstance(item, str) and item.strip() and item != canon_id and item not in merged_from_list:
                        merged_from_list.append(item)
        merged_base_attrs["merged_from"] = merged_from_list

        # 3. Update location_id references if merging locations
        location_ref_deltas: list[EntityDelta] = []
        location_refs_updated: list[str] = []
        if dup_ent.kind == "location" and self._policy.has_tier(3):
            for ent in active_entities:
                if ent.entity_id == dup_id:
                    continue
                if ent.state.get("location_id") == dup_id:
                    location_refs_updated.append(ent.entity_id)
                    if ent.entity_id == canon_id:
                        merged_state["location_id"] = canon_id
                    else:
                        location_ref_deltas.append(
                            EntityDelta(
                                entity_id=ent.entity_id,
                                state={"location_id": canon_id},
                                merge_state=True,
                                reason=eff_reason,
                            )
                        )

        # 4. Migrate active edges
        edges_to_write: list[EdgeDelta] = []
        edges_to_tombstone: list[DirectedEdgeRecord] = []
        edges_migrated: list[str] = []
        edges_merged_on_collision: list[str] = []
        edges_tombstoned_keys: list[str] = []
        self_loops_dropped: list[str] = []

        if migrate_edges and self._policy.has_tier(2):
            edge_index: dict[tuple[str, str, str], DirectedEdgeRecord] = {
                (e.source_id, e.target_id, e.relation_type): e for e in active_edges
            }
            hard_keys = set(self._policy.hard_filter_keys)

            for edge in active_edges:
                if edge.source_id != dup_id and edge.target_id != dup_id:
                    continue
                old_key = CausalStore.edge_subject_key(
                    edge.source_id, edge.target_id, edge.relation_type
                )
                edges_to_tombstone.append(edge)
                edges_tombstoned_keys.append(old_key)

                new_src = canon_id if edge.source_id == dup_id else edge.source_id
                new_tgt = canon_id if edge.target_id == dup_id else edge.target_id
                if new_src == new_tgt:
                    self_loops_dropped.append(old_key)
                    continue

                new_key = CausalStore.edge_subject_key(new_src, new_tgt, edge.relation_type)
                existing_edge = edge_index.get((new_src, new_tgt, edge.relation_type))
                if existing_edge is None:
                    edges_to_write.append(
                        EdgeDelta(
                            source_id=new_src,
                            target_id=new_tgt,
                            relation_type=edge.relation_type,
                            state=dict(edge.state),
                            merge=False,
                            caused_by_episode_id=edge.caused_by_episode_id,
                            reason=eff_reason,
                        )
                    )
                    edges_migrated.append(new_key)
                else:
                    for hk in sorted(hard_keys):
                        dup_val = edge.state.get(hk)
                        canon_val = existing_edge.state.get(hk)
                        if (
                            dup_val is not None
                            and canon_val is not None
                            and str(dup_val).strip().lower() != str(canon_val).strip().lower()
                        ):
                            winner = canon_val if conflict_strategy == "canonical_wins" else dup_val
                            warnings_list.append(
                                f"Edge collision on {new_key}: '{hk}' conflicted "
                                f"(canonical={canon_val!r} vs duplicate={dup_val!r}); kept {winner!r}."
                            )
                    merged_edge_state = (
                        {**edge.state, **existing_edge.state}
                        if conflict_strategy == "canonical_wins"
                        else {**existing_edge.state, **edge.state}
                    )
                    if merged_edge_state != existing_edge.state:
                        edges_to_write.append(
                            EdgeDelta(
                                source_id=new_src,
                                target_id=new_tgt,
                                relation_type=edge.relation_type,
                                state=merged_edge_state,
                                merge=False,
                                caused_by_episode_id=existing_edge.caused_by_episode_id
                                or edge.caused_by_episode_id,
                                reason=eff_reason,
                            )
                        )
                    edges_merged_on_collision.append(new_key)

        # 5. Execute (or preview for dry_run)
        migrate_voice_fn = getattr(self.store, "migrate_entity_voice_exemplars", None)
        migrate_episodes_fn = getattr(self.store, "migrate_entity_episodes", None)

        voice_migrated = 0
        episodes_migrated = 0
        epistemics_migrated = 0

        if dry_run:
            if migrate_voice and self._policy.has_tier(5) and callable(migrate_voice_fn):
                voice_migrated = migrate_voice_fn(
                    dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=True
                )
            if migrate_episodes and self._policy.has_tier(6) and callable(migrate_episodes_fn):
                episodes_migrated, epistemics_migrated = migrate_episodes_fn(
                    dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=True
                )
        else:
            entity_deltas: list[EntityDelta] = [
                EntityDelta(
                    entity_id=canon_id,
                    base_attributes=merged_base_attrs,
                    state=merged_state if self._policy.has_tier(3) else {},
                    private_state=merged_private if self._policy.has_tier(3) else {},
                    merge_state=False if self._policy.has_tier(3) else True,
                    add_aliases=list(aliases_to_add),
                    reason=eff_reason,
                )
            ]
            if dup_aliases_to_tombstone:
                entity_deltas.append(
                    EntityDelta(
                        entity_id=dup_id,
                        remove_aliases=dup_aliases_to_tombstone,
                    )
                )
            for other_eid, updated_aliases in other_alias_updates.items():
                entity_deltas.append(
                    EntityDelta(
                        entity_id=other_eid,
                        add_aliases=list(updated_aliases),
                    )
                )
            entity_deltas.extend(location_ref_deltas)

            tombstone_entity_fn = getattr(self.store, "tombstone_entity", None)
            tombstone_edge_fn = getattr(self.store, "tombstone_edge", None)

            with self._store_transaction():
                # Apply canonical updates, alias transfers, duplicate alias tombstones, and migrated edges
                self.store.apply_delta(
                    eff_step,
                    StepDelta(entities=entity_deltas, edges=edges_to_write),
                    timeline_id=tl,
                )
                # Tombstone old edges referencing duplicate_id
                if edges_to_tombstone:
                    if callable(tombstone_edge_fn):
                        for old_edge in edges_to_tombstone:
                            tombstone_edge_fn(
                                old_edge.source_id,
                                old_edge.target_id,
                                old_edge.relation_type,
                                step=eff_step,
                                timeline_id=tl,
                            )
                    else:
                        self.store.apply_delta(
                            eff_step,
                            StepDelta(
                                edges=[
                                    EdgeDelta(
                                        source_id=old_edge.source_id,
                                        target_id=old_edge.target_id,
                                        relation_type=old_edge.relation_type,
                                        remove=True,
                                    )
                                    for old_edge in edges_to_tombstone
                                ]
                            ),
                            timeline_id=tl,
                        )
                # Tombstone duplicate_id itself
                if callable(tombstone_entity_fn):
                    tombstone_entity_fn(
                        dup_id,
                        step=eff_step,
                        timeline_id=tl,
                        payload={"merged_into": canon_id, "reason": eff_reason},
                    )
                else:
                    self.store.apply_delta(
                        eff_step,
                        StepDelta(entities=[EntityDelta(entity_id=dup_id, remove=True)]),
                        timeline_id=tl,
                    )
                # Migrate Tier 5 voice exemplars
                if migrate_voice and self._policy.has_tier(5) and callable(migrate_voice_fn):
                    voice_migrated = migrate_voice_fn(
                        dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=False
                    )
                # Migrate Tier 6 episodes & epistemics
                if migrate_episodes and self._policy.has_tier(6) and callable(migrate_episodes_fn):
                    episodes_migrated, epistemics_migrated = migrate_episodes_fn(
                        dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=False
                    )

        return EntityMergeReport(
            timeline_id=tl,
            step=eff_step,
            duplicate_id=dup_id,
            canonical_id=canon_id,
            dry_run=dry_run,
            aliases_added=[a.alias for a in aliases_to_add],
            aliases_tombstoned=dup_aliases_to_tombstone,
            base_attributes_merged=base_attrs_merged,
            state_keys_merged=state_keys_merged,
            edges_migrated=edges_migrated,
            edges_merged_on_collision=edges_merged_on_collision,
            edges_tombstoned=edges_tombstoned_keys,
            self_loops_dropped=self_loops_dropped,
            location_refs_updated=location_refs_updated,
            voice_exemplars_migrated=voice_migrated,
            episodes_migrated=episodes_migrated,
            epistemics_migrated=epistemics_migrated,
            warnings=warnings_list,
        )

    # --- Unified Tier 1-6 Retrieval ---

    def retrieve_context(
        self,
        step: int,
        *,
        text: str = "",
        prev_text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int | None = None,
        top_k_episodes: int | None = None,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        location_id: str | None = None,
        inclusive: bool = False,
        timeline_id: str | None = None,
        fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only",
        story_time: float | str | None = None,
        chapter_id: str | None = None,
        scene_id: str | None = None,
        voice_constraints: dict[str, str] | None = None,
        lang: str | None = None,
    ) -> StepContext:
        tl = timeline_id or self.active_timeline
        policy = self._policy
        cutoff_step = step if inclusive else (step - 1)

        has_edges = policy.has_tier(2)
        has_state = policy.has_tier(3)
        has_voice = policy.has_tier(5)
        has_episodes = policy.has_tier(6)
        eff_top_k_voice = 0 if not has_voice else (3 if top_k_voice is None else top_k_voice)
        eff_top_k_episodes = 0 if not has_episodes else (5 if top_k_episodes is None else top_k_episodes)
        eff_edge_types: Sequence[str] | None = () if not has_edges else edge_types

        # 1. Single-pass fetch of active entities, aliases, and edges (with custom MemoryStore fallback)
        get_graph_fn = getattr(self.store, "get_active_graph_state", None)
        if callable(get_graph_fn):
            active_entities, active_aliases, active_edges = get_graph_fn(
                tl, cutoff_step, edge_types=eff_edge_types
            )
        else:
            active_entities = self.store.get_all_active_entities(tl, cutoff_step)
            active_aliases = self.store.get_all_active_aliases(tl, cutoff_step)
            if has_edges:
                active_edges = self.store.get_all_active_edges(tl, cutoff_step)
                if eff_edge_types is not None:
                    active_edges = [e for e in active_edges if e.relation_type in eff_edge_types]
            else:
                active_edges = []

        # 2. Entity resolution via pluggable MentionResolver + POV epistemic disguise projection
        pov_set = set(pov_entity_ids) if pov_entity_ids else set()
        raw_mentions = self._resolve_mentions(
            text=text,
            explicit_mentions=mentions,
            entities=active_entities,
            aliases=active_aliases,
            lang=lang,
        )
        resolved_mentions, disguise_masks, pierced_disguises = self._apply_epistemic_disguises(
            raw_mentions=raw_mentions,
            aliases=active_aliases,
            pov_set=pov_set,
            pov_mode=pov_mode,
        )

        # Read-Only Coreference Halo: if target chunk has < 2 characters (e.g. pronoun-only),
        # resolve fallback mentions from prev_text so active characters/edges stay in context.
        # Skipped in LEXICON_ONLY (no Tier 2/3) where entities are glossary terms rather than scene actors.
        entity_by_id = {e.entity_id: e for e in active_entities}
        primary_char_ids = {
            m.entity_id
            for m in resolved_mentions
            if m.entity_id.startswith("masked:")
            or (
                (ent := entity_by_id.get(m.entity_id)) is None
                or ent.kind == "character"
            )
        }
        if (has_edges or has_state) and prev_text.strip() and len(primary_char_ids) < 2:
            halo_raw = self._resolve_mentions(
                text=prev_text,
                explicit_mentions=(),
                entities=active_entities,
                aliases=active_aliases,
                lang=lang,
            )
            halo_mentions, halo_masks, halo_pierced = self._apply_epistemic_disguises(
                raw_mentions=halo_raw,
                aliases=active_aliases,
                pov_set=pov_set,
                pov_mode=pov_mode,
            )
            seen_resolved_ids = {m.entity_id for m in resolved_mentions}
            for m in halo_mentions:
                real_id = m.entity_id.removeprefix("masked:")
                if (
                    m.entity_id in seen_resolved_ids
                    or real_id in seen_resolved_ids
                    or f"masked:{real_id}" in seen_resolved_ids
                ):
                    continue
                seen_resolved_ids.add(m.entity_id)
                resolved_mentions.append(m)
                if real_id in halo_masks:
                    disguise_masks.setdefault(real_id, halo_masks[real_id])
            for k, v in halo_pierced.items():
                if k not in disguise_masks:
                    pierced_disguises.setdefault(k, v)

        resolved_entity_ids = {m.entity_id for m in resolved_mentions}
        masked_real_ids = set(disguise_masks.keys())

        # 3. Separate co-present scene entities from 1-hop social graph neighbors
        present_entity_ids = {
            eid for eid in resolved_entity_ids if not eid.startswith("masked:")
        }
        active_location_ids: set[str] = set()
        if location_id is not None:
            active_location_ids.add(location_id)
        for ent in active_entities:
            if ent.entity_id in present_entity_ids and ent.kind == "location":
                active_location_ids.add(ent.entity_id)
        if active_location_ids:
            for ent in active_entities:
                if ent.entity_id in masked_real_ids:
                    continue
                ent_loc = ent.state.get("location_id") or ent.base_attributes.get("location_id")
                if ent_loc in active_location_ids:
                    present_entity_ids.add(ent.entity_id)

        if not present_entity_ids and not disguise_masks:
            if fallback_on_empty_mentions == "all":
                present_entity_ids = {
                    e.entity_id for e in active_entities if e.entity_id not in masked_real_ids
                }
            elif fallback_on_empty_mentions == "pov_only" and pov_set:
                present_entity_ids = {
                    eid for eid in pov_set if eid not in masked_real_ids
                }

        in_scene_edges: list[DirectedEdgeRecord] = []
        background_edges: list[DirectedEdgeRecord] = []
        neighbor_entity_ids: set[str] = set()
        for edge in active_edges:
            if edge.relation_type == policy.belief_relation:
                continue
            if edge.source_id in masked_real_ids or edge.target_id in masked_real_ids:
                continue
            src_present = edge.source_id in present_entity_ids
            tgt_present = edge.target_id in present_entity_ids
            if src_present and tgt_present:
                in_scene_edges.append(edge)
            elif src_present or tgt_present:
                background_edges.append(edge)
                if not src_present:
                    neighbor_entity_ids.add(edge.source_id)
                if not tgt_present:
                    neighbor_entity_ids.add(edge.target_id)

        scene_entities = [e for e in active_entities if e.entity_id in present_entity_ids]
        neighbor_entities = [e for e in active_entities if e.entity_id in neighbor_entity_ids]

        pov_known_episode_ids: set[str] = set()
        fallback_lineage_episodes: list[tuple[EpisodeRecord, bytes]] | None = None
        if pov_set and has_episodes:
            provenance_ep_ids = {
                eid
                for eid in (
                    *(e.caused_by_episode_id for e in scene_entities),
                    *(e.caused_by_episode_id for e in neighbor_entities),
                    *(e.caused_by_episode_id for e in in_scene_edges),
                    *(e.caused_by_episode_id for e in background_edges),
                )
                if eid
            }
            if provenance_ep_ids:
                get_known_fn = getattr(self.store, "get_known_episode_ids", None)
                if callable(get_known_fn):
                    pov_known_episode_ids = get_known_fn(
                        tl,
                        cutoff_step,
                        pov_entity_ids or (),
                        pov_mode=pov_mode,
                        episode_ids=sorted(provenance_ep_ids),
                    )
                else:
                    fallback_lineage_episodes = self.store.get_lineage_episodes(tl, cutoff_step)
                    if pov_mode == "all":
                        pov_known_episode_ids = {
                            ep.episode_id
                            for ep, _ in fallback_lineage_episodes
                            if all(ep.known_by.get(pid) in ("direct", "told") for pid in pov_set)
                        }
                    else:
                        pov_known_episode_ids = {
                            ep.episode_id
                            for ep, _ in fallback_lineage_episodes
                            if any(ep.known_by.get(pid) in ("direct", "told") for pid in pov_set)
                        }

        projected_entities, projected_edges = self._project_pov_state(
            scene_entities=scene_entities,
            scene_edges=in_scene_edges,
            all_active_edges=active_edges,
            pov_set=pov_set,
            pov_mode=pov_mode,
            pov_known_episode_ids=pov_known_episode_ids,
            disguise_masks=disguise_masks,
            pierced_disguises=pierced_disguises,
            policy=policy,
        )
        projected_related_entities, projected_background_edges = self._project_pov_state(
            scene_entities=neighbor_entities,
            scene_edges=background_edges,
            all_active_edges=active_edges,
            pov_set=pov_set,
            pov_mode=pov_mode,
            pov_known_episode_ids=pov_known_episode_ids,
            disguise_masks={},
            pierced_disguises=pierced_disguises,
            policy=policy,
        )

        # 4. Tier 5 Voice Exemplars retrieval
        voice_exemplars: list[VoiceExemplar] = []
        speaker_ids = resolved_entity_ids | {f"masked:{rid}" for rid in masked_real_ids}
        speakers = (
            [e for e in projected_entities if e.entity_id in speaker_ids]
            if speaker_ids
            else projected_entities
        )
        if eff_top_k_voice > 0 and speakers:
            in_scene_ids = resolved_entity_ids or {e.entity_id for e in projected_entities}
            for speaker in speakers:
                is_masked_speaker = speaker.entity_id.startswith("masked:")
                real_speaker_id = (
                    speaker.entity_id.removeprefix("masked:")
                    if is_masked_speaker
                    else speaker.entity_id
                )
                edge_pool = active_edges if is_masked_speaker else (in_scene_edges + background_edges)
                voice_targets = self._resolve_speaker_voice_targets(
                    real_speaker_id=real_speaker_id,
                    is_masked_speaker=is_masked_speaker,
                    edges=edge_pool,
                    in_scene_ids=in_scene_ids,
                    resolved_mentions=resolved_mentions,
                    text=text,
                    policy=policy,
                    voice_constraints=voice_constraints,
                )

                candidates = self.store.get_lineage_voice_exemplars(tl, cutoff_step, real_speaker_id)
                disguise_persona = (
                    speaker.state.get(policy.entity_disguise_key) if is_masked_speaker else None
                )
                if disguise_persona:
                    persona_candidates = [
                        (ex, emb)
                        for ex, emb in candidates
                        if ex.metadata.get(policy.exemplar_persona_key, "").lower()
                        == str(disguise_persona).lower()
                    ]
                    if persona_candidates:
                        candidates = persona_candidates

                ranked = self.voice_bank.rank_exemplars_for_targets(
                    query_text=text,
                    candidates=candidates,
                    targets=voice_targets,
                    policy=policy,
                    top_k=eff_top_k_voice,
                )
                if is_masked_speaker:
                    ranked = [
                        ex.model_copy(update={"speaker_id": speaker.entity_id})
                        for ex in ranked
                    ]
                voice_exemplars.extend(ranked)

        # 5. Tier 6 Episodic Memory retrieval
        episodes: list[EpisodeRecord] = []
        if eff_top_k_episodes > 0:
            pinned_episode_ids: set[str] = set()
            for ent in projected_entities:
                if ent.caused_by_episode_id:
                    pinned_episode_ids.add(ent.caused_by_episode_id)
            for edge in projected_edges:
                if edge.caused_by_episode_id:
                    pinned_episode_ids.add(edge.caused_by_episode_id)

            cue_entity_ids = (
                resolved_entity_ids
                if resolved_entity_ids
                else {e.entity_id for e in projected_entities}
            )
            if fallback_lineage_episodes is not None:
                candidate_episodes = fallback_lineage_episodes
            else:
                candidate_limit = max(eff_top_k_episodes * 20, 200)
                sql_cue_ids = sorted(cue_entity_ids | active_location_ids)
                try:
                    candidate_episodes = self.store.get_lineage_episodes(
                        tl,
                        cutoff_step,
                        limit=candidate_limit,
                        pov_entity_ids=pov_entity_ids,
                        pov_mode=pov_mode,
                        cue_entity_ids=sql_cue_ids,
                        pinned_episode_ids=sorted(pinned_episode_ids),
                        query_text=text,
                    )
                except TypeError:
                    candidate_episodes = self.store.get_lineage_episodes(tl, cutoff_step)

            if candidate_episodes:
                episodic_policy = getattr(policy, "episodic_policy", DEFAULT_EPISODIC_POLICY)
                episodes = self.voice_bank.rank_episodes(
                    current_step=step,
                    query_text=text,
                    candidates=candidate_episodes,
                    scene_entity_ids=cue_entity_ids,
                    pinned_episode_ids=pinned_episode_ids,
                    pov_entity_ids=pov_entity_ids,
                    pov_mode=pov_mode,
                    policy=episodic_policy,
                    top_k=eff_top_k_episodes,
                )

        entity_names: dict[str, str] = {
            e.entity_id: e.canonical_name
            for e in (*projected_entities, *projected_related_entities)
        }
        referenced_ids: set[str] = set()
        for edge in (*projected_edges, *projected_background_edges):
            referenced_ids.add(edge.source_id)
            referenced_ids.add(edge.target_id)
        for ex in voice_exemplars:
            if ex.speaker_id:
                referenced_ids.add(ex.speaker_id)
            if ex.addressee_id:
                referenced_ids.add(ex.addressee_id)
        for ep in episodes:
            referenced_ids.update(ep.known_by.keys())
        for missing_id in sorted(referenced_ids - entity_names.keys()):
            if missing_id.startswith("masked:"):
                continue
            rec = self.store.get_active_entity(
                missing_id, timeline_id=tl, cutoff_step=cutoff_step
            )
            if rec is not None:
                entity_names[missing_id] = rec.canonical_name

        return StepContext(
            timeline_id=tl,
            step=step,
            story_time=story_time,
            chapter_id=chapter_id,
            scene_id=scene_id,
            resolved_mentions=resolved_mentions,
            entities=projected_entities,
            active_edges=projected_edges,
            related_entities=projected_related_entities,
            background_edges=projected_background_edges,
            voice_exemplars=voice_exemplars,
            episodes=episodes,
            entity_names=entity_names,
        )

    def _resolve_speaker_voice_targets(
        self,
        *,
        real_speaker_id: str,
        is_masked_speaker: bool,
        edges: Sequence[DirectedEdgeRecord],
        in_scene_ids: set[str],
        resolved_mentions: Sequence[ResolvedMention],
        text: str,
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
        voice_constraints: dict[str, str] | None = None,
    ) -> list[tuple[str | None, dict[str, str]]]:
        """Resolves ordered (addressee_id, constraints) dyadic voice targets for a speaker."""
        extra_constraints = {
            k: str(v) for k, v in (voice_constraints or {}).items() if str(v).strip()
        }
        outgoing = [
            e
            for e in edges
            if e.source_id == real_speaker_id
            and (not is_masked_speaker or e.relation_type == policy.facade_relation)
        ]
        if not outgoing:
            return [(None, dict(extra_constraints))]

        in_scene_edges = [
            e
            for e in outgoing
            if e.target_id in in_scene_ids and e.target_id != real_speaker_id
        ]
        active_pool = in_scene_edges if in_scene_edges else outgoing

        layer_ranks = {rel: idx + 1 for idx, rel in enumerate(policy.edge_layer_order)}

        def _layer_rank(rel_type: str) -> int:
            return layer_ranks.get(rel_type, 0)

        by_target: dict[str, list[DirectedEdgeRecord]] = {}
        for edge in active_pool:
            by_target.setdefault(edge.target_id, []).append(edge)

        text_lower = text.lower()
        mention_pos: dict[str, int] = {}
        for idx, m in enumerate(resolved_mentions):
            if m.entity_id in mention_pos:
                continue
            pos = text_lower.find(m.surface_text.lower()) if text_lower else -1
            mention_pos[m.entity_id] = pos if pos >= 0 else (1_000_000 + idx)

        constraint_keys = set(policy.hard_filter_keys) | set(policy.soft_boost_weights.keys())
        resolved_targets: list[tuple[tuple[int, int, str], str | None, dict[str, str]]] = []
        for target_id, target_edges in by_target.items():
            target_edges_sorted = sorted(
                target_edges,
                key=lambda e: (_layer_rank(e.relation_type), e.valid_from_step),
            )
            merged_constraints: dict[str, str] = {}
            max_step = 0
            for e in target_edges_sorted:
                for ck in constraint_keys:
                    if e.state.get(ck):
                        merged_constraints[ck] = str(e.state[ck])
                max_step = max(max_step, e.valid_from_step)
            if extra_constraints:
                merged_constraints.update(extra_constraints)

            sort_key = (mention_pos.get(target_id, 2_000_000), -max_step, target_id)
            resolved_targets.append((sort_key, target_id, merged_constraints))

        resolved_targets.sort(key=lambda item: item[0])
        return [(addr, constraints) for _, addr, constraints in resolved_targets]

    def _project_pov_state(
        self,
        *,
        scene_entities: Sequence[EntityRecord],
        scene_edges: Sequence[DirectedEdgeRecord],
        all_active_edges: Sequence[DirectedEdgeRecord],
        pov_set: set[str],
        pov_mode: Literal["any", "all"] = "any",
        pov_known_episode_ids: set[str],
        disguise_masks: dict[str, AliasRecord],
        pierced_disguises: dict[str, AliasRecord],
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
    ) -> tuple[list[EntityRecord], list[DirectedEdgeRecord]]:
        """Projects entities and edges through the active POV's epistemic lens."""
        belief_overrides: dict[str, dict[str, Any]] = {}
        if pov_set:
            pov_edges = [
                edge
                for edge in all_active_edges
                if edge.relation_type == policy.belief_relation and edge.source_id in pov_set
            ]
            pov_edges.sort(key=lambda e: (e.source_id, e.valid_from_step))
            if len(pov_set) > 1 and pov_mode == "all":
                per_pov_beliefs: dict[str, dict[str, dict[str, Any]]] = {pid: {} for pid in pov_set}
                for edge in pov_edges:
                    per_pov_beliefs[edge.source_id].setdefault(edge.target_id, {}).update(edge.state)
                common_targets = set.intersection(
                    *(set(per_pov_beliefs[pid].keys()) for pid in pov_set)
                )
                for target_id in common_targets:
                    first_pid = next(iter(pov_set))
                    shared_state: dict[str, Any] = {}
                    for k, v in per_pov_beliefs[first_pid][target_id].items():
                        if all(
                            k in per_pov_beliefs[pid][target_id]
                            and per_pov_beliefs[pid][target_id][k] == v
                            for pid in pov_set
                        ):
                            shared_state[k] = v
                    if shared_state:
                        belief_overrides[target_id] = shared_state
            else:
                for edge in pov_edges:
                    belief_overrides.setdefault(edge.target_id, {}).update(edge.state)

        facade_pairs: set[tuple[str, str]] = {
            (edge.source_id, edge.target_id)
            for edge in scene_edges
            if edge.relation_type == policy.facade_relation
        }

        is_multipov_all = len(pov_set) > 1 and pov_mode == "all"
        projected_entities: list[EntityRecord] = []
        for ent in scene_entities:
            ent_copy = ent.model_copy(deep=True)
            if ent_copy.entity_id in pierced_disguises:
                alias_rec = pierced_disguises[ent_copy.entity_id]
                ent_copy.disguised_as = alias_rec.persona_name or alias_rec.alias

            if pov_set:
                if ent_copy.entity_id not in pov_set or is_multipov_all:
                    ent_copy.private_state = {}
                    if (
                        ent_copy.caused_by_episode_id
                        and ent_copy.caused_by_episode_id not in pov_known_episode_ids
                    ):
                        ent_copy.caused_by_episode_id = None
                        ent_copy.reason = None
                if ent_copy.entity_id in belief_overrides:
                    ent_copy.state = {**ent_copy.state, **belief_overrides[ent_copy.entity_id]}

            projected_entities.append(ent_copy)

        for real_id, alias_rec in disguise_masks.items():
            persona_label = alias_rec.persona_name or alias_rec.alias
            projected_entities.append(
                EntityRecord(
                    entity_id=f"masked:{real_id}",
                    kind=policy.masked_entity_kind,
                    canonical_name=persona_label,
                    state={policy.entity_disguise_key: persona_label},
                )
            )

        projected_edges: list[DirectedEdgeRecord] = []
        for edge in scene_edges:
            is_self_edge = edge.source_id in pov_set and not is_multipov_all
            if (
                pov_set
                and edge.relation_type in policy.edge_layer_order
                and edge.relation_type != policy.facade_relation
                and not is_self_edge
                and (edge.source_id, edge.target_id) in facade_pairs
            ):
                continue
            edge_copy = edge.model_copy(deep=True)
            if (
                pov_set
                and not is_self_edge
                and edge_copy.caused_by_episode_id
                and edge_copy.caused_by_episode_id not in pov_known_episode_ids
            ):
                edge_copy.caused_by_episode_id = None
                edge_copy.reason = None
            projected_edges.append(edge_copy)

        return projected_entities, projected_edges

    def _apply_epistemic_disguises(
        self,
        *,
        raw_mentions: Sequence[ResolvedMention],
        aliases: Sequence[AliasRecord],
        pov_set: set[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
    ) -> tuple[list[ResolvedMention], dict[str, AliasRecord], dict[str, AliasRecord]]:
        """Applies POV-aware disguise masking & piercing on top of pluggable mention matches."""
        resolved: list[ResolvedMention] = []
        seen_entities: set[str] = set()
        disguise_masks: dict[str, AliasRecord] = {}
        pierced_disguises: dict[str, AliasRecord] = {}

        alias_by_key: dict[tuple[str, str], AliasRecord] = {
            (a.entity_id, a.alias.lower()): a for a in aliases if a.alias
        }
        alias_by_surface: dict[str, AliasRecord] = {
            a.alias.lower(): a for a in aliases if a.alias
        }

        # Process alias matches (including fuzzy-matched aliases) first so canonical mentions cannot shadow them
        ordered_mentions = sorted(
            raw_mentions,
            key=lambda m: (0 if (m.matched_via == "alias" or m.matched_source == "alias") else 1),
        )

        for m in ordered_mentions:
            is_alias_match = m.matched_via == "alias" or m.matched_source == "alias"
            lookup_key = (m.matched_phrase or m.surface_text).strip().lower()
            alias_rec = (
                alias_by_key.get((m.entity_id, lookup_key)) or alias_by_surface.get(lookup_key)
                if is_alias_match
                else None
            )
            if alias_rec is not None and alias_rec.kind == "disguise":
                ent_id = alias_rec.entity_id
                if alias_rec.known_to is not None and pov_set:
                    known_set = set(alias_rec.known_to or ())
                    pierced = (
                        pov_set.issubset(known_set)
                        if pov_mode == "all"
                        else bool(pov_set & known_set)
                    )
                    if not pierced:
                        masked_id = f"masked:{ent_id}"
                        if masked_id not in seen_entities:
                            seen_entities.add(masked_id)
                            disguise_masks[ent_id] = alias_rec
                            resolved.append(
                                ResolvedMention(
                                    surface_text=m.surface_text,
                                    entity_id=masked_id,
                                    matched_via="alias",
                                    confidence=m.confidence,
                                    matched_source="alias",
                                    matched_phrase=alias_rec.alias.lower(),
                                )
                            )
                        continue
                pierced_disguises[ent_id] = alias_rec

            if m.entity_id in disguise_masks:
                continue

            if m.entity_id not in seen_entities:
                seen_entities.add(m.entity_id)
                resolved.append(m)

        return resolved, disguise_masks, pierced_disguises

    # --- Transactional Context Manager ---

    def step_transaction(
        self,
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
    ) -> StepTransaction:
        return StepTransaction(
            self,
            step,
            text=text,
            mentions=mentions,
            top_k_voice=top_k_voice,
            top_k_episodes=top_k_episodes,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            edge_types=edge_types,
            location_id=location_id,
            timeline_id=timeline_id,
            rerun_clean=rerun_clean,
            prev_text=prev_text,
            next_text=next_text,
            lang=lang,
            content_hash=content_hash,
            checkpoint_metadata=checkpoint_metadata,
        )

    # --- State & Episode Mutation ---

    def apply_delta(self, step: int, delta: StepDelta, *, timeline_id: str | None = None) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        policy = self._policy
        has_edges = policy.has_tier(2)
        has_state = policy.has_tier(3)
        has_episodes = policy.has_tier(6)

        prepared_episodes: list[EpisodeRecord] = []
        if has_episodes:
            for ep in delta.episodes:
                ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
                updates: dict[str, Any] = {"episode_id": ep_id, "step": step, "timeline_id": tl}
                if ep.story_time is None and delta.story_time is not None:
                    updates["story_time"] = delta.story_time
                if ep.chapter_id is None and delta.chapter_id is not None:
                    updates["chapter_id"] = delta.chapter_id
                if ep.scene_id is None and delta.scene_id is not None:
                    updates["scene_id"] = delta.scene_id
                prepared_episodes.append(ep.model_copy(update=updates))

        ep_embeddings: dict[str, bytes] = {}
        if prepared_episodes:
            vectors = embed_texts_batch(
                self.voice_bank.embedder, [ep.summary for ep in prepared_episodes]
            )
            self._record_embedder_metadata(vectors)
            for ep, vec in zip(prepared_episodes, vectors):
                ep_embeddings[ep.episode_id] = vec.tobytes()

        delta_updates: dict[str, Any] = {}
        if has_episodes:
            if prepared_episodes:
                delta_updates["episodes"] = prepared_episodes
        else:
            if delta.episodes:
                delta_updates["episodes"] = []
            if delta.inform:
                delta_updates["inform"] = []

        if not has_edges and delta.edges:
            delta_updates["edges"] = []

        if not has_state and delta.entities:
            stripped_entities: list[EntityDelta] = []
            for ed in delta.entities:
                cleaned = ed.model_copy(
                    update={
                        "state": {},
                        "private_state": {},
                        "caused_by_episode_id": None,
                        "reason": None,
                    }
                )
                if (
                    cleaned.canonical_name is not None
                    or cleaned.kind is not None
                    or cleaned.base_attributes
                    or cleaned.add_aliases
                    or cleaned.remove_aliases
                    or cleaned.remove
                ):
                    stripped_entities.append(cleaned)
            delta_updates["entities"] = stripped_entities

        if delta.is_flashback:
            # Flashbacks must not overwrite already-established present-day entity/edge states.
            source_entities = delta_updates.get("entities", delta.entities)
            existing_ids = {
                e.entity_id for e in self.store.get_all_active_entities(tl, step)
            }
            delta_updates["entities"] = [
                ed for ed in source_entities if not ed.remove and ed.entity_id not in existing_ids
            ]
            delta_updates["edges"] = []

        delta_to_apply = delta.model_copy(update=delta_updates) if delta_updates else delta
        return self.store.apply_delta(
            step, delta_to_apply, timeline_id=tl, episode_embeddings=ep_embeddings
        )

    def index_episodes(
        self,
        step: int,
        episodes: Sequence[EpisodeRecord],
        *,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        if not episodes or not self._policy.has_tier(6):
            return []
        indexed: list[EpisodeRecord] = []
        for ep in episodes:
            ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
            indexed.append(ep.model_copy(update={"episode_id": ep_id, "step": step, "timeline_id": tl}))

        vectors = embed_texts_batch(self.voice_bank.embedder, [ep.summary for ep in indexed])
        self._record_embedder_metadata(vectors)
        to_store = [(item, vec.tobytes()) for item, vec in zip(indexed, vectors)]
        self.store.insert_episodes(to_store, timeline_id=tl)
        return indexed

    def consolidate_episodes(
        self,
        step: int,
        *,
        before_step: int,
        summary: str,
        max_salience: float = 4.0,
        predicate: str = "consolidated_summary",
        episode_id: str | None = None,
        chapter_id: str | None = None,
        salience: float = 7.5,
        timeline_id: str | None = None,
    ) -> EpisodeRecord | None:
        """Compacts old low-salience episodes (< before_step) into a single summary episode."""
        tl = timeline_id or self.active_timeline
        candidates = self.store.query_episodes(
            timeline_id=tl,
            cutoff_step=before_step - 1,
        )
        targets = [
            ep
            for ep in candidates
            if ep.salience <= max_salience
            and ep.predicate != predicate
            and (chapter_id is None or ep.chapter_id == chapter_id)
        ]
        if not targets:
            return None

        participants: dict[str, str] = {}
        witnesses: list[str] = []
        seen_witnesses: set[str] = set()
        known_by: dict[str, Literal["direct", "told", "deceived"]] = {}
        for ep in targets:
            for eid, role in ep.participants.items():
                participants.setdefault(eid, role)
            for wid in ep.witnesses:
                if wid not in seen_witnesses:
                    seen_witnesses.add(wid)
                    witnesses.append(wid)
            for eid, mode in ep.known_by.items():
                if mode == "deceived":
                    continue
                if known_by.get(eid) != "direct":
                    known_by[eid] = mode

        summary_ep = EpisodeRecord(
            episode_id=episode_id or f"ep_consolidated_{step}_{uuid.uuid4().hex[:8]}",
            timeline_id=tl,
            step=step,
            predicate=predicate,
            summary=summary,
            participants=participants,
            witnesses=witnesses,
            known_by=known_by,
            chapter_id=chapter_id,
            consolidated_from=[ep.episode_id for ep in targets if ep.episode_id],
            salience=salience,
        )
        with self._store_transaction():
            committed = self.index_episodes(step, [summary_ep], timeline_id=tl)[0]
            delete_fn = getattr(self.store, "delete_episodes", None)
            if callable(delete_fn):
                delete_fn([ep.episode_id for ep in targets], timeline_id=tl)
        return committed

    def inform_entity(
        self,
        step: int,
        *,
        episode_id: str,
        entity_id: str,
        epistemic: EpistemicMode = "told",
        believed_summary: str | None = None,
        timeline_id: str | None = None,
    ) -> None:
        tl = timeline_id or self.active_timeline
        self.store.record_epistemic(
            InformDelta(
                episode_id=episode_id,
                entity_id=entity_id,
                epistemic=epistemic,
                believed_summary=believed_summary,
            ),
            step=step,
            timeline_id=tl,
        )

    def query_episodes(
        self,
        *,
        cutoff_step: int | None = None,
        participants: Sequence[str] = (),
        match_all_participants: bool = False,
        predicate: str | None = None,
        location_id: str | None = None,
        known_to: Sequence[str] = (),
        min_salience: float = 0.0,
        step_range: tuple[int, int] | None = None,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        return self.store.query_episodes(
            timeline_id=tl,
            cutoff_step=cutoff_step,
            participants=participants,
            match_all_participants=match_all_participants,
            predicate=predicate,
            location_id=location_id,
            known_to=known_to,
            min_salience=min_salience,
            step_range=step_range,
        )

    def _extract_dialogue_candidates_with_delta(
        self,
        text: str,
        *,
        source_text: str = "",
        step: int | None = None,
        timeline_id: str | None = None,
        prev_text: str = "",
        next_text: str = "",
        lang: str | None = None,
        staged_delta: StepDelta | None = None,
    ) -> list[DialogueCandidate]:
        tl = timeline_id or self.active_timeline
        if step is None:
            tl_info = self.store.get_timeline(tl)
            cutoff_step = tl_info.head_step if tl_info is not None else 0
        else:
            cutoff_step = step

        active_entities = list(self.store.get_all_active_entities(tl, cutoff_step))
        active_aliases = list(self.store.get_all_active_aliases(tl, cutoff_step))

        if staged_delta is not None and staged_delta.entities:
            ent_by_id = {e.entity_id: e for e in active_entities}
            for ed in staged_delta.entities:
                if ed.remove:
                    ent_by_id.pop(ed.entity_id, None)
                    continue
                existing = ent_by_id.get(ed.entity_id)
                if existing is None:
                    ent_by_id[ed.entity_id] = EntityRecord(
                        entity_id=ed.entity_id,
                        canonical_name=ed.canonical_name or ed.entity_id,
                        kind=ed.kind or "character",
                        base_attributes=dict(ed.base_attributes),
                        state=dict(ed.state),
                    )
                else:
                    merged_base = {**existing.base_attributes, **ed.base_attributes}
                    merged_state = {**existing.state, **ed.state} if ed.merge_state else dict(ed.state)
                    ent_by_id[ed.entity_id] = existing.model_copy(
                        update={
                            "canonical_name": ed.canonical_name or existing.canonical_name,
                            "kind": ed.kind or existing.kind,
                            "base_attributes": merged_base,
                            "state": merged_state,
                        }
                    )
                for alias_item in ed.add_aliases:
                    if isinstance(alias_item, AliasRecord):
                        active_aliases.append(alias_item)
                    elif isinstance(alias_item, str) and alias_item.strip():
                        active_aliases.append(
                            AliasRecord(alias=alias_item.strip(), entity_id=ed.entity_id)
                        )
            active_entities = list(ent_by_id.values())

        speaker_entities = [
            e for e in active_entities if e.kind not in ("location", "artifact", "item", "term")
        ]
        speaker_entity_ids = {e.entity_id for e in speaker_entities}
        speaker_aliases = [
            a for a in active_aliases if not speaker_entity_ids or a.entity_id in speaker_entity_ids
        ]

        def _resolve_in_segment(segment: str) -> tuple[str | None, str | None, str | None]:
            mentions = self._resolve_mentions(
                text=segment,
                explicit_mentions=(),
                entities=speaker_entities,
                aliases=speaker_aliases,
                lang=lang,
            )
            if not mentions:
                return None, None, None
            seg_lower = segment.lower()
            ordered = sorted(
                mentions,
                key=lambda m: (
                    idx if (idx := seg_lower.find(m.surface_text.lower())) != -1 else len(seg_lower)
                ),
            )
            speaker_id = ordered[0].entity_id
            speaker_surface = ordered[0].surface_text
            addressee_id = next(
                (m.entity_id for m in ordered[1:] if m.entity_id != speaker_id),
                None,
            )
            return speaker_id, addressee_id, speaker_surface

        def _resolve_speaker(before_win: str, after_win: str) -> tuple[str | None, str | None, str | None]:
            if not speaker_entities and not speaker_aliases:
                return None, None, None
            if before_win:
                # Prefer the immediate sentence/clause preceding the quote before falling back to full window
                immediate_clause = re.split(r"[.!?]\s+", before_win.strip())[-1]
                if immediate_clause and immediate_clause != before_win.strip():
                    spk, addr, surf = _resolve_in_segment(immediate_clause)
                    if spk is not None:
                        return spk, addr, surf
                spk, addr, surf = _resolve_in_segment(before_win)
                if spk is not None:
                    return spk, addr, surf
            if after_win:
                spk, addr, surf = _resolve_in_segment(after_win)
                if spk is not None:
                    return spk, addr, surf
            return None, None, None

        def _on_gate1_rejected(span: str, reason: str, meta: dict[str, Any]) -> None:
            for hook in self._hooks:
                hook.on_gate_rejected("gate1", span, reason, meta)

        return self.voice_bank.gate.extract_candidates(
            text,
            source_text=source_text,
            prev_text=prev_text,
            next_text=next_text,
            speaker_resolver=_resolve_speaker,
            on_rejected=_on_gate1_rejected if self._hooks else None,
        )

    def _resolve_harvest_metadata(
        self,
        *,
        speaker_id: str | None,
        addressee_id: str | None,
        context: StepContext,
        staged_delta: StepDelta | None,
        cutoff_step: int,
        timeline_id: str,
        infer_addressee: bool = True,
    ) -> tuple[str | None, dict[str, str]]:
        """Resolves dyadic edge metadata (register/tone) and speaker persona with staged-delta overlay."""
        policy = self._policy
        meta: dict[str, str] = {}
        if not speaker_id:
            return addressee_id, meta

        # Build effective entity map (context + store fallback + staged_delta overlay)
        ent_by_id: dict[str, EntityRecord] = {
            e.entity_id: e
            for e in (*context.entities, *context.related_entities)
            if not e.entity_id.startswith("masked:")
        }
        if speaker_id not in ent_by_id:
            for e in self.store.get_all_active_entities(timeline_id, cutoff_step):
                ent_by_id.setdefault(e.entity_id, e)

        if staged_delta is not None:
            for ed in staged_delta.entities:
                if ed.remove:
                    ent_by_id.pop(ed.entity_id, None)
                    continue
                existing = ent_by_id.get(ed.entity_id)
                if existing is None:
                    ent_by_id[ed.entity_id] = EntityRecord(
                        entity_id=ed.entity_id,
                        canonical_name=ed.canonical_name or ed.entity_id,
                        kind=ed.kind or "character",
                        base_attributes=dict(ed.base_attributes),
                        state=dict(ed.state),
                    )
                else:
                    merged_state = {**existing.state, **ed.state} if ed.merge_state else dict(ed.state)
                    ent_by_id[ed.entity_id] = existing.model_copy(
                        update={
                            "canonical_name": ed.canonical_name or existing.canonical_name,
                            "kind": ed.kind or existing.kind,
                            "state": merged_state,
                        }
                    )

        # Two-person scene addressee inference when Gate 1 found speaker but no explicit addressee
        eff_addressee = addressee_id
        if eff_addressee is None and infer_addressee:
            scene_chars = [
                e.entity_id
                for e in context.entities
                if not e.entity_id.startswith("masked:")
                and e.kind == "character"
                and e.entity_id != speaker_id
            ]
            if len(scene_chars) == 1:
                eff_addressee = scene_chars[0]

        target_keys = tuple(
            dict.fromkeys((*policy.hard_filter_keys, *policy.soft_boost_weights.keys()))
        )

        # Resolve effective dyadic edges (store at cutoff_step + staged_delta overlay)
        if policy.has_tier(2) and eff_addressee:
            edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
            for edge in self.store.get_all_active_edges(timeline_id, cutoff_step):
                if edge.source_id == speaker_id and edge.target_id == eff_addressee:
                    edge_map[(edge.source_id, edge.target_id, edge.relation_type)] = dict(edge.state)
            if staged_delta is not None:
                for rd in staged_delta.edges:
                    if rd.source_id != speaker_id or rd.target_id != eff_addressee:
                        continue
                    key = (rd.source_id, rd.target_id, rd.relation_type)
                    if rd.remove:
                        edge_map.pop(key, None)
                    elif rd.merge and key in edge_map:
                        edge_map[key] = {**edge_map[key], **rd.state}
                    else:
                        edge_map[key] = dict(rd.state)

            # Apply layers in SchemaPolicy.edge_layer_order (later layers e.g. facade override stance)
            for rel_type in policy.edge_layer_order:
                state = edge_map.get((speaker_id, eff_addressee, rel_type))
                if not state:
                    continue
                for k in target_keys:
                    val = state.get(k)
                    if val is not None and str(val).strip():
                        meta[k] = str(val).strip()

            # Fallback to any other non-belief edge if none of edge_layer_order populated a key
            for (_, _, rel_type), state in edge_map.items():
                if rel_type in policy.edge_layer_order or rel_type == policy.belief_relation:
                    continue
                for k in target_keys:
                    if k not in meta:
                        val = state.get(k)
                        if val is not None and str(val).strip():
                            meta[k] = str(val).strip()

        # Speaker disguise / persona tagging + optional mood -> tone fallback
        speaker_ent = ent_by_id.get(speaker_id)
        if speaker_ent is not None:
            disguise = speaker_ent.state.get(policy.entity_disguise_key)
            if disguise is not None and str(disguise).strip():
                meta[policy.exemplar_persona_key] = str(disguise).strip()
            if "tone" in target_keys and "tone" not in meta:
                mood = speaker_ent.state.get("mood")
                if mood is not None and str(mood).strip():
                    meta["tone"] = str(mood).strip()

        return eff_addressee, meta

    def extract_dialogue_candidates(
        self,
        text: str,
        *,
        source_text: str = "",
        step: int | None = None,
        timeline_id: str | None = None,
        prev_text: str = "",
        next_text: str = "",
        lang: str | None = None,
    ) -> list[DialogueCandidate]:
        return self._extract_dialogue_candidates_with_delta(
            text,
            source_text=source_text,
            step=step,
            timeline_id=timeline_id,
            prev_text=prev_text,
            next_text=next_text,
            lang=lang,
            staged_delta=None,
        )

    def index_voice_exemplars(
        self,
        step: int,
        exemplars: Sequence[VoiceExemplar],
        *,
        bypass_gate: bool = False,
        timeline_id: str | None = None,
    ) -> list[VoiceExemplar]:
        tl = timeline_id or self.active_timeline
        if not exemplars or not self._policy.has_tier(5):
            return []
        indexed: list[VoiceExemplar] = []
        for ex in exemplars:
            if not bypass_gate and not self.voice_bank.gate.passes_quality_gate(ex):
                for hook in self._hooks:
                    hook.on_gate_rejected(
                        "gate2",
                        ex.text,
                        f"quality_score {ex.quality_score} < {self.voice_bank.gate.min_quality}",
                        {"speaker_id": ex.speaker_id, "quality_score": ex.quality_score},
                    )
                continue
            ex_id = ex.exemplar_id or str(uuid.uuid4())
            indexed.append(ex.model_copy(update={"exemplar_id": ex_id, "step": step, "timeline_id": tl}))

        if indexed:
            vectors = embed_texts_batch(self.voice_bank.embedder, [ex.text for ex in indexed])
            self._record_embedder_metadata(vectors)
            to_store = [(item, vec.tobytes()) for item, vec in zip(indexed, vectors)]
            self.store.insert_voice_exemplars(to_store, timeline_id=tl)
        return indexed

    def _store_transaction(self) -> AbstractContextManager[None]:
        tx_fn = getattr(self.store, "transaction", None)
        return tx_fn() if callable(tx_fn) else contextlib.nullcontext()

    def commit_step(
        self,
        step: int,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        *,
        bypass_gate: bool = False,
        timeline_id: str | None = None,
        content_hash: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> list[VoiceExemplar]:
        tl = timeline_id or self.active_timeline
        for hook in self._hooks:
            hook.on_before_commit(step, tl, delta, exemplars, episodes)
        committed_episodes: list[EpisodeRecord] = []
        indexed: list[VoiceExemplar] = []
        record_cp = getattr(self.store, "record_step_checkpoint", None)
        if not callable(record_cp) and (content_hash is not None or checkpoint_metadata is not None):
            raise NotImplementedError("Active store backend does not support step checkpoints.")
        with self._store_transaction():
            if delta:
                committed_episodes.extend(self.apply_delta(step, delta, timeline_id=tl))
            if episodes and self._policy.has_tier(6):
                committed_episodes.extend(self.index_episodes(step, episodes, timeline_id=tl))
            if exemplars and self._policy.has_tier(5):
                indexed = self.index_voice_exemplars(
                    step, exemplars, bypass_gate=bypass_gate, timeline_id=tl
                )
            if callable(record_cp):
                record_cp(
                    step,
                    timeline_id=tl,
                    content_hash=content_hash,
                    metadata=checkpoint_metadata,
                )
        for hook in self._hooks:
            hook.on_after_commit(step, tl, committed_episodes, indexed)
        return indexed

    # --- Step Checkpoints (Resumability) ---

    def _require_checkpoint_store(self) -> CheckpointStore:
        required = (
            "record_step_checkpoint",
            "get_step_checkpoint",
            "get_last_committed_step",
            "list_step_checkpoints",
        )
        if not all(callable(getattr(self.store, name, None)) for name in required):
            raise NotImplementedError("Active store backend does not support step checkpoints.")
        return self.store  # type: ignore[return-value]

    def get_step_checkpoint(
        self,
        step: int,
        *,
        timeline_id: str | None = None,
    ) -> StepCheckpoint | None:
        """Returns the lineage-resolved StepCheckpoint for `step`, or None if not committed."""
        tl = timeline_id or self.active_timeline
        return self._require_checkpoint_store().get_step_checkpoint(step, timeline_id=tl)

    def is_step_committed(
        self,
        step: int,
        *,
        content_hash: str | None = None,
        metadata_match: dict[str, Any] | None = None,
        timeline_id: str | None = None,
    ) -> bool:
        """Returns True if `step` is checkpointed on the timeline lineage (and matches hash/metadata if given)."""
        cp = self.get_step_checkpoint(step, timeline_id=timeline_id)
        if cp is None:
            return False
        if content_hash is not None and cp.content_hash != content_hash:
            return False
        if metadata_match is not None:
            for key, expected in metadata_match.items():
                if key not in cp.metadata or cp.metadata[key] != expected:
                    return False
        return True

    def last_committed_step(self, *, timeline_id: str | None = None) -> int | None:
        """Returns the highest checkpointed step along the timeline lineage, or None if none committed."""
        tl = timeline_id or self.active_timeline
        return self._require_checkpoint_store().get_last_committed_step(timeline_id=tl)

    def list_step_checkpoints(
        self,
        *,
        timeline_id: str | None = None,
        cutoff_step: int | None = None,
    ) -> list[StepCheckpoint]:
        """Returns lineage-resolved StepCheckpoints in ascending step order."""
        tl = timeline_id or self.active_timeline
        return self._require_checkpoint_store().list_step_checkpoints(
            timeline_id=tl,
            cutoff_step=cutoff_step,
        )

    @staticmethod
    def _checkpoint_matches(
        cp: StepCheckpoint | None,
        *,
        content_hash: str | None,
        metadata_match: Mapping[str, Any] | None,
    ) -> bool:
        if cp is None:
            return False
        if content_hash is not None and cp.content_hash != content_hash:
            return False
        if metadata_match is not None:
            for key, expected in metadata_match.items():
                if key not in cp.metadata or cp.metadata[key] != expected:
                    return False
        return True

    def plan_resume(
        self,
        step_hashes: Sequence[tuple[int, str]],
        *,
        causal: bool | None = None,
        metadata_match: Mapping[str, Any] | None = None,
        timeline_id: str | None = None,
        apply_rollback: bool = False,
        force: bool = False,
        prune_tail: bool = True,
    ) -> ResumePlan:
        """Plans which ordered (step, content_hash) pairs can be reused vs. re-run.

        In causal mode (default when any stateful tier 2, 3, 5, or 6 is active), the first
        missing or mismatched step invalidates all subsequent steps and computes the exact
        `rollback_to_step` watermark. In non-causal mode (e.g. LEXICON_ONLY), each step is
        evaluated independently and `apply_rollback=True` purges only the stale/orphaned steps.
        """
        tl = timeline_id or self.active_timeline
        is_causal = (
            bool(causal)
            if causal is not None
            else any(self._policy.has_tier(t) for t in (2, 3, 5, 6))
        )

        normalized: list[tuple[int, str]] = []
        prev_step: int | None = None
        for raw_step, content_hash in step_hashes:
            step = int(raw_step)
            if prev_step is not None:
                if step == prev_step:
                    raise ValueError(f"Duplicate step {step} in step_hashes.")
                if step < prev_step:
                    raise ValueError(
                        f"step_hashes must be ordered by strictly ascending step numbers "
                        f"(got {step} after {prev_step})."
                    )
            normalized.append((step, content_hash))
            prev_step = step

        checkpoints = {cp.step: cp for cp in self.list_step_checkpoints(timeline_id=tl)}
        tl_info = self.store.get_timeline(tl)
        fork_step = tl_info.fork_step if tl_info is not None else None
        head_step = tl_info.head_step if tl_info is not None else None

        if prune_tail:
            if normalized:
                max_planned = normalized[-1][0]
                orphaned_steps = sorted(s for s in checkpoints if s > max_planned)
            else:
                orphaned_steps = sorted(checkpoints.keys())
        else:
            orphaned_steps = []

        reusable_steps: list[int] = []
        pending_steps: list[int] = []
        missing_steps: list[int] = []
        stale_steps: list[int] = []
        invalidated_steps: list[int] = []
        first_stale_step: int | None = None
        rollback_to_step: int | None = None
        fork_conflict_step: int | None = None
        rollback_applied = False
        deleted_rows = 0

        if is_causal:
            diverged = False
            for step, content_hash in normalized:
                cp = checkpoints.get(step)
                matched = self._checkpoint_matches(
                    cp, content_hash=content_hash, metadata_match=metadata_match
                )
                if not diverged:
                    if matched:
                        reusable_steps.append(step)
                    else:
                        diverged = True
                        pending_steps.append(step)
                        if cp is not None:
                            stale_steps.append(step)
                            first_stale_step = step
                        else:
                            missing_steps.append(step)
                else:
                    pending_steps.append(step)
                    if cp is not None:
                        if matched:
                            invalidated_steps.append(step)
                        else:
                            stale_steps.append(step)
                            if first_stale_step is None:
                                first_stale_step = step
                    else:
                        missing_steps.append(step)

            if reusable_steps:
                watermark: int | None = reusable_steps[-1]
            elif normalized:
                watermark = normalized[0][0] - 1
            elif orphaned_steps:
                watermark = orphaned_steps[0] - 1
            else:
                watermark = None

            if watermark is not None:
                has_stale_tail = bool(stale_steps or invalidated_steps or orphaned_steps)
                has_uncheckpointed_writes = (
                    head_step is not None
                    and head_step > watermark
                    and (watermark >= 0 or head_step > 0)
                    and (fork_step is None or head_step > fork_step)
                    and (prune_tail or (bool(normalized) and head_step <= normalized[-1][0]))
                )
                if has_stale_tail or has_uncheckpointed_writes:
                    rollback_to_step = watermark

            if (
                rollback_to_step is not None
                and fork_step is not None
                and rollback_to_step < fork_step
            ):
                fork_conflict_step = fork_step
                if apply_rollback:
                    parent_hint = (
                        f" Roll back parent '{tl_info.parent_id}' or fork a new branch at step {rollback_to_step}."
                        if tl_info is not None and tl_info.parent_id
                        else ""
                    )
                    raise TimelineBranchConflictError(
                        f"Cannot roll back branch '{tl}' to step {rollback_to_step}: "
                        f"branch forked at step {fork_step}.{parent_hint}"
                    )

            if apply_rollback and rollback_to_step is not None:
                deleted_rows = self.rollback_to(rollback_to_step, timeline_id=tl, force=force)
                rollback_applied = True
        else:
            for step, content_hash in normalized:
                cp = checkpoints.get(step)
                if self._checkpoint_matches(
                    cp, content_hash=content_hash, metadata_match=metadata_match
                ):
                    reusable_steps.append(step)
                else:
                    pending_steps.append(step)
                    if cp is not None:
                        stale_steps.append(step)
                        if first_stale_step is None:
                            first_stale_step = step
                    else:
                        missing_steps.append(step)

            if apply_rollback:
                purge_steps = sorted(set(stale_steps) | set(orphaned_steps))
                if purge_steps:
                    delete_steps_fn = getattr(self.store, "delete_steps", None)
                    if callable(delete_steps_fn):
                        deleted_rows = int(delete_steps_fn(purge_steps, timeline_id=tl, force=force))
                        rollback_applied = True

        return ResumePlan(
            timeline_id=tl,
            causal=is_causal,
            reusable_steps=reusable_steps,
            pending_steps=pending_steps,
            missing_steps=missing_steps,
            stale_steps=stale_steps,
            invalidated_steps=invalidated_steps,
            orphaned_steps=orphaned_steps,
            first_stale_step=first_stale_step,
            rollback_to_step=rollback_to_step,
            fork_conflict_step=fork_conflict_step,
            rollback_applied=rollback_applied,
            deleted_rows=deleted_rows,
        )

    # --- Time-Travel Operations ---

    def rollback_to(
        self,
        step: int,
        *,
        timeline_id: str | None = None,
        force: bool = False,
    ) -> int:
        tl = timeline_id or self.active_timeline
        deleted = self.store.rollback_to(step, timeline_id=tl, force=force)
        for hook in self._hooks:
            hook.on_rollback(step, tl, deleted)
        return deleted

    def branch(
        self,
        at_step: int,
        name: str,
        *,
        from_timeline: str | None = None,
        switch: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineInfo:
        parent = from_timeline or self.active_timeline
        info = self.store.create_timeline(name, parent_id=parent, fork_step=at_step, metadata=metadata)
        if switch:
            self.active_timeline = name
        for hook in self._hooks:
            hook.on_branch(info)
        return info

    def delete_timeline(self, timeline_id: str, *, cascade: bool = False) -> int:
        deleted = self.store.delete_timeline(timeline_id, cascade=cascade)
        if self._default_timeline == timeline_id:
            self._default_timeline = "main"
        if self._timeline_var.get() == timeline_id:
            self._timeline_var.set(None)
        return deleted

    def checkout(self, timeline_id: str) -> None:
        if not self.store.get_timeline(timeline_id):
            raise ValueError(f"Timeline '{timeline_id}' does not exist.")
        self.active_timeline = timeline_id

    def list_timelines(self) -> list[TimelineInfo]:
        return self.store.list_timelines()

    def get_history(
        self,
        subject_key: str,
        *,
        timeline_id: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        tl = timeline_id or self.active_timeline
        return self.store.get_history(subject_key, timeline_id=tl, record_type=record_type)

    def diff_timelines(self, timeline_a: str, timeline_b: str, *, at_step: int) -> dict[str, Any]:
        return self.store.diff_timelines(timeline_a, timeline_b, at_step=at_step)

    def export_snapshot(
        self,
        target_path: str | Path,
        *,
        at_step: int,
        timeline_id: str | None = None,
    ) -> Path:
        tl = timeline_id or self.active_timeline
        return self.store.export_snapshot(target_path, at_step=at_step, timeline_id=tl)

    def render_bible(
        self,
        *,
        format: BibleExportFormat = "md",
        at_step: int | None = None,
        timeline_id: str | None = None,
        kinds: Sequence[str] | None = None,
        include_state: bool = False,
        include_edges: bool | None = None,
        include_aliases: bool = True,
        include_inflections: bool | None = None,
        lang: str | None = None,
    ) -> tuple[str, BibleExportResult]:
        """Renders the active Series Bible at `(timeline_id, at_step)` to a CSV, JSON, or Markdown string."""
        return _render_bible(
            self,
            format=format,
            at_step=at_step,
            timeline_id=timeline_id,
            kinds=kinds,
            include_state=include_state,
            include_edges=include_edges,
            include_aliases=include_aliases,
            include_inflections=include_inflections,
            lang=lang,
        )

    def export_bible(
        self,
        path: str | Path,
        *,
        format: Literal["auto", "csv", "json", "md"] = "auto",
        at_step: int | None = None,
        timeline_id: str | None = None,
        kinds: Sequence[str] | None = None,
        include_state: bool = False,
        include_edges: bool | None = None,
        include_aliases: bool = True,
        include_inflections: bool | None = None,
        lang: str | None = None,
        excel_bom: bool = False,
    ) -> BibleExportResult:
        """Exports the active Series Bible at `(timeline_id, at_step)` to CSV, JSON, or Markdown."""
        return _export_bible(
            self,
            path,
            format=format,
            at_step=at_step,
            timeline_id=timeline_id,
            kinds=kinds,
            include_state=include_state,
            include_edges=include_edges,
            include_aliases=include_aliases,
            include_inflections=include_inflections,
            lang=lang,
            excel_bom=excel_bom,
        )

    def sync_bible(
        self,
        path: str | Path,
        *,
        format: Literal["auto", "csv", "json"] = "auto",
        step: int = 0,
        timeline_id: str | None = None,
        mode: BibleSyncMode = "upsert",
        target_lang: str | None = None,
        auto_inflect: bool = True,
        batch_size: int = 25,
        dry_run: bool = False,
        column_map: dict[str, str] | None = None,
    ) -> BibleSyncReport:
        """Syncs a human-editable CSV or JSON Series Bible into the causal store with diff/upsert semantics."""
        return _sync_bible(
            self,
            path,
            format=format,
            step=step,
            timeline_id=timeline_id,
            mode=mode,
            target_lang=target_lang,
            auto_inflect=auto_inflect,
            batch_size=batch_size,
            dry_run=dry_run,
            column_map=column_map,
        )

    def format_context(self, ctx: StepContext) -> str:
        """Formats StepContext into prompt text using the active DomainPreset."""
        return self.preset.format_prompt_context(ctx)

    def _format_compact_entity_roster(
        self,
        ctx: StepContext,
        *,
        seen_entity_ids: set[str],
        scope: KnownEntityScope,
        max_roster_entities: int | None = None,
    ) -> tuple[str, str] | None:
        """Builds a terse cast/entity roster for out-of-scene active entities on ctx.timeline_id."""
        if scope == "scene":
            return None
        if max_roster_entities is not None and max_roster_entities < 0:
            raise ValueError("max_roster_entities must be non-negative.")

        cutoff_step = max(0, ctx.step - 1)
        active_entities = self.store.get_all_active_entities(ctx.timeline_id, cutoff_step)
        candidates: list[EntityRecord] = []
        for e in active_entities:
            if e.entity_id.startswith("masked:") or e.entity_id in seen_entity_ids:
                continue
            if scope == "characters_compact" and e.kind != "character":
                continue
            candidates.append(e)

        if not candidates or max_roster_entities == 0:
            return None

        if max_roster_entities is not None and len(candidates) > max_roster_entities:
            candidates.sort(key=lambda e: (-e.revision_count, -e.valid_from_step, e.entity_id))
            candidates = candidates[:max_roster_entities]

        # Append-stable ordering (valid_from_step, entity_id) preserves prefix cache across steps.
        candidates.sort(key=lambda e: (e.valid_from_step, e.entity_id))

        def _entry_str(ent: EntityRecord) -> str:
            names: list[str] = []
            canon = ent.canonical_name.strip()
            if canon:
                names.append(canon)
            target_name = ent.base_attributes.get("target_name")
            if isinstance(target_name, str) and target_name.strip():
                tname = target_name.strip()
                if not any(n.lower() == tname.lower() for n in names):
                    names.append(tname)
            label = "/".join(names) if names else ent.entity_id
            return f"{ent.entity_id}={label}"

        if scope == "characters_compact":
            header = (
                "### Other Known Characters (reference only — NOT necessarily in this scene; "
                "reuse these exact entity_ids if any reappear under a new name or epithet):"
            )
            body = ", ".join(_entry_str(e) for e in candidates)
            return header, body

        # scope == "all_compact": group by kind with characters first
        grouped: dict[str, list[EntityRecord]] = {}
        for e in candidates:
            grouped.setdefault(e.kind or "entity", []).append(e)
        kind_order = sorted(grouped.keys(), key=lambda k: (0 if k == "character" else 1, k))
        lines = [
            f"{kind}: {', '.join(_entry_str(e) for e in grouped[kind])}"
            for kind in kind_order
        ]
        header = (
            "### Other Known Entities (reference only — NOT necessarily in this scene; "
            "reuse these exact entity_ids if any reappear under a new name or epithet):"
        )
        return header, "\n".join(lines)

    def build_extraction_messages(
        self,
        ctx: StepContext,
        scene_text: str,
        *,
        prev_text: str = "",
        next_text: str = "",
        compact_schema: bool = False,
        known_entity_scope: KnownEntityScope = "scene",
        max_roster_entities: int | None = None,
    ) -> tuple[str, str]:
        """Builds prefix-cache-friendly (system_prompt, user_prompt) extraction messages.

        Static content (preset extraction instructions and the StepDelta JSON schema) is placed
        in ``system_prompt`` so provider prompt caches and local KV caches reuse it across steps,
        while dynamic per-step context (known entity IDs, optional compact cast roster, recent
        episodes, halo, and scene text) is placed in ``user_prompt``.

        Args:
            known_entity_scope:
                - ``"scene"`` (default): Only entities in ``ctx.entities`` and ``ctx.related_entities``.
                - ``"characters_compact"``: Scene/related entities plus a terse 1-line roster of all
                  other active ``kind == "character"`` entities on ``ctx.timeline_id``.
                - ``"all_compact"``: Scene/related entities plus a terse kind-grouped roster of all
                  other active entities on ``ctx.timeline_id``.
            max_roster_entities:
                Optional cap on the number of out-of-scene entities included in the compact roster.
        """
        if known_entity_scope not in ("scene", "characters_compact", "all_compact"):
            raise ValueError(
                f"Unsupported known_entity_scope: {known_entity_scope!r}. "
                "Expected 'scene', 'characters_compact', or 'all_compact'."
            )

        policy = self._policy
        raw_schema = StepDelta.model_json_schema()
        props = raw_schema.get("properties", {})
        defs = raw_schema.get("$defs", {})
        if not policy.has_tier(6):
            for k in ("episodes", "inform", "is_flashback", "story_time", "chapter_id", "scene_id"):
                props.pop(k, None)
            for d in ("EpisodeRecord", "InformDelta"):
                defs.pop(d, None)
        if not policy.has_tier(2):
            props.pop("edges", None)
            defs.pop("EdgeDelta", None)

        schema_json = (
            json.dumps(raw_schema, separators=(",", ":"))
            if compact_schema
            else json.dumps(raw_schema, indent=2)
        )
        system_prompt = "\n\n".join(
            [
                self.preset.extraction_instructions(),
                "### Target StepDelta JSON Schema:",
                schema_json,
            ]
        )

        seen_entity_ids: set[str] = set()
        entity_lines: list[str] = []
        for e in (*ctx.entities, *ctx.related_entities):
            if e.entity_id.startswith("masked:") or e.entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(e.entity_id)
            entity_lines.append(f"- {e.entity_id} (name: {e.canonical_name}, kind: {e.kind})")
        episode_lines = (
            [
                f"- {ep.episode_id} (step {ep.step}): {ep.summary}"
                for ep in ctx.episodes
                if ep.episode_id
            ]
            if policy.has_tier(6)
            else []
        )

        user_parts = [
            "### Known Entity IDs (use exact entity_id values, not display names):",
            "\n".join(entity_lines) if entity_lines else "(none)",
        ]
        compact_roster = self._format_compact_entity_roster(
            ctx,
            seen_entity_ids=seen_entity_ids,
            scope=known_entity_scope,
            max_roster_entities=max_roster_entities,
        )
        if compact_roster is not None:
            roster_header, roster_body = compact_roster
            user_parts.append(roster_header)
            user_parts.append(roster_body)
        if episode_lines:
            user_parts.append("### Recent Episode IDs (for caused_by_episode_id or InformDelta):")
            user_parts.append("\n".join(episode_lines))
        has_halo = bool(prev_text.strip() or next_text.strip())
        if prev_text.strip():
            user_parts.append(
                "### Preceding Context (Already Committed — Use ONLY to resolve pronouns like he/she/they; DO NOT re-extract events):"
            )
            user_parts.append(prev_text.strip())
        if has_halo:
            user_parts.append(
                "### Target Scene Chunk (EXTRACT ONLY state changes and episodes occurring in THIS text):"
            )
        else:
            user_parts.append("### Scene Text to Extract From:")
        user_parts.append(scene_text)
        if next_text.strip():
            user_parts.append(
                "### Following Context (Read-Only Lookahead — Use ONLY to identify speakers/addressees of trailing dialogue; DO NOT extract future events):"
            )
            user_parts.append(next_text.strip())
        return system_prompt, "\n\n".join(user_parts)

    def build_extraction_prompt(
        self,
        ctx: StepContext,
        scene_text: str,
        *,
        prev_text: str = "",
        next_text: str = "",
        compact_schema: bool = False,
        known_entity_scope: KnownEntityScope = "scene",
        max_roster_entities: int | None = None,
    ) -> str:
        """Builds a single-string extraction prompt with static rules and schema first for prefix-cache friendliness."""
        system_prompt, user_prompt = self.build_extraction_messages(
            ctx,
            scene_text,
            prev_text=prev_text,
            next_text=next_text,
            compact_schema=compact_schema,
            known_entity_scope=known_entity_scope,
            max_roster_entities=max_roster_entities,
        )
        return f"{system_prompt}\n\n{user_prompt}"

    def normalize_delta(
        self,
        delta: StepDelta,
        *,
        step: int | None = None,
        timeline_id: str | None = None,
    ) -> StepDelta:
        """Normalizes canonical names/aliases into entity_ids and resolves intra-step episode IDs."""
        tl = timeline_id or self.active_timeline
        entities = self.store.get_all_active_entities(tl, step)
        aliases = self.store.get_all_active_aliases(tl, step)

        lookup: dict[str, str] = {}
        for e in entities:
            lookup[e.entity_id.lower()] = e.entity_id
            if e.canonical_name:
                lookup[e.canonical_name.lower()] = e.entity_id
            target_name = e.base_attributes.get("target_name")
            if isinstance(target_name, str) and target_name.strip():
                lookup.setdefault(target_name.strip().lower(), e.entity_id)
            merged_from = e.base_attributes.get("merged_from")
            if isinstance(merged_from, list):
                for old_id in merged_from:
                    if isinstance(old_id, str) and old_id.strip():
                        lookup.setdefault(old_id.strip().lower(), e.entity_id)
        for a in aliases:
            if a.alias and a.kind != "disguise":
                lookup.setdefault(a.alias.lower(), a.entity_id)

        # Include entities and non-disguise aliases introduced within this same StepDelta
        for ed in delta.entities:
            if ed.remove:
                continue
            lookup.setdefault(ed.entity_id.lower(), ed.entity_id)
            if ed.canonical_name:
                lookup.setdefault(ed.canonical_name.lower(), ed.entity_id)
            ed_target = ed.base_attributes.get("target_name")
            if isinstance(ed_target, str) and ed_target.strip():
                lookup.setdefault(ed_target.strip().lower(), ed.entity_id)
            for alias_item in ed.add_aliases:
                if isinstance(alias_item, AliasRecord):
                    if alias_item.alias and alias_item.kind != "disguise":
                        lookup.setdefault(alias_item.alias.lower(), ed.entity_id)
                elif isinstance(alias_item, str) and alias_item.strip():
                    lookup.setdefault(alias_item.strip().lower(), ed.entity_id)

        def _norm_id(raw: str) -> str:
            return lookup.get(raw.strip().lower(), raw)

        ep_id_map: dict[str, str] = {}
        norm_episodes: list[EpisodeRecord] = []
        for idx, ep in enumerate(delta.episodes):
            old_id = ep.episode_id
            if old_id:
                new_id = old_id
            else:
                digest = hashlib.sha256(
                    f"{tl}:{step if step is not None else 0}:{idx}:{ep.predicate}:{ep.summary}".encode()
                ).hexdigest()[:8]
                new_id = f"ep_{step if step is not None else 0}_{digest}"
            if old_id:
                ep_id_map[old_id] = new_id
            norm_participants = {_norm_id(k): v for k, v in ep.participants.items()}
            norm_witnesses = [_norm_id(w) for w in ep.witnesses]
            norm_known_by = {_norm_id(k): v for k, v in ep.known_by.items()}
            norm_believed = {_norm_id(k): v for k, v in ep.believed_summaries.items()}
            norm_loc = _norm_id(ep.location_id) if ep.location_id else None
            ep_updates: dict[str, Any] = {
                "episode_id": new_id,
                "participants": norm_participants,
                "witnesses": norm_witnesses,
                "known_by": norm_known_by,
                "believed_summaries": norm_believed,
                "location_id": norm_loc,
            }
            if step is not None:
                ep_updates["step"] = step
            norm_episodes.append(ep.model_copy(update=ep_updates))

        def _norm_entity_state(state_dict: dict[str, Any]) -> dict[str, Any]:
            if not state_dict:
                return state_dict
            out = dict(state_dict)
            if "location_id" in out and isinstance(out["location_id"], str):
                out["location_id"] = _norm_id(out["location_id"])
            else:
                for alt_loc_key in ("location", "current_location"):
                    if alt_loc_key in out and isinstance(out[alt_loc_key], str):
                        resolved_loc = _norm_id(out[alt_loc_key])
                        if resolved_loc.startswith("loc_") or resolved_loc in {
                            e.entity_id for e in entities if e.kind == "location"
                        }:
                            out.pop(alt_loc_key, None)
                            out["location_id"] = resolved_loc
                            break
            if "mood" not in out:
                for alt_mood_key in ("emotional_state", "emotion"):
                    if alt_mood_key in out and isinstance(out[alt_mood_key], str) and out[alt_mood_key].strip():
                        out["mood"] = out[alt_mood_key].strip()
                        break
            return out

        raw_norm_entities = [
            ed.model_copy(
                update={
                    "entity_id": _norm_id(ed.entity_id),
                    "state": _norm_entity_state(ed.state),
                    "caused_by_episode_id": ep_id_map.get(ed.caused_by_episode_id, ed.caused_by_episode_id)
                    if ed.caused_by_episode_id
                    else None,
                }
            )
            for ed in delta.entities
        ]
        # Merge multiple intra-step EntityDeltas for the same entity_id in order
        merged_entities_by_id: dict[str, Any] = {}
        for ed in raw_norm_entities:
            existing = merged_entities_by_id.get(ed.entity_id)
            if existing is None or ed.remove or existing.remove:
                merged_entities_by_id[ed.entity_id] = ed
            else:
                merged_entities_by_id[ed.entity_id] = existing.model_copy(
                    update={
                        "kind": ed.kind or existing.kind,
                        "canonical_name": ed.canonical_name or existing.canonical_name,
                        "base_attributes": {**existing.base_attributes, **ed.base_attributes},
                        "state": {**existing.state, **ed.state},
                        "private_state": {**existing.private_state, **ed.private_state},
                        "add_aliases": [*existing.add_aliases, *ed.add_aliases],
                        "remove_aliases": [*existing.remove_aliases, *ed.remove_aliases],
                        "caused_by_episode_id": ed.caused_by_episode_id or existing.caused_by_episode_id,
                        "reason": ed.reason or existing.reason,
                    }
                )
        norm_entities = list(merged_entities_by_id.values())
        norm_edges = [
            rd.model_copy(
                update={
                    "source_id": _norm_id(rd.source_id),
                    "target_id": _norm_id(rd.target_id),
                    "caused_by_episode_id": ep_id_map.get(rd.caused_by_episode_id, rd.caused_by_episode_id)
                    if rd.caused_by_episode_id
                    else None,
                }
            )
            for rd in delta.edges
        ]
        norm_inform = [
            inf.model_copy(
                update={
                    "entity_id": _norm_id(inf.entity_id),
                    "episode_id": ep_id_map.get(inf.episode_id, inf.episode_id),
                }
            )
            for inf in delta.inform
        ]

        return StepDelta(
            entities=norm_entities,
            edges=norm_edges,
            episodes=norm_episodes,
            inform=norm_inform,
            is_flashback=delta.is_flashback,
            story_time=delta.story_time,
            chapter_id=delta.chapter_id,
            scene_id=delta.scene_id,
        )

    # --- Pluggable QA & Continuity Audit ---

    def audit_step(
        self,
        step: int,
        source_text: str = "",
        target_text: str = "",
        *,
        timeline_id: str | None = None,
        rules: Sequence[AuditRule] | None = None,
        include_rules: Sequence[str] | None = None,
        exclude_rules: Sequence[str] | None = None,
        min_severity: AuditSeverity = "info",
        lang: str | None = None,
    ) -> AuditReport:
        """Runs pluggable QA & continuity audit rules against a single committed step."""
        return _run_audit_step(
            self,
            step,
            source_text=source_text,
            target_text=target_text,
            timeline_id=timeline_id,
            rules=rules,
            include_rules=include_rules,
            exclude_rules=exclude_rules,
            min_severity=min_severity,
            lang=lang,
        )

    def audit_timeline(
        self,
        *,
        timeline_id: str | None = None,
        from_step: int = 0,
        to_step: int | None = None,
        texts: Mapping[int, str | tuple[str, str]] | None = None,
        rules: Sequence[AuditRule] | None = None,
        include_rules: Sequence[str] | None = None,
        exclude_rules: Sequence[str] | None = None,
        min_severity: AuditSeverity = "info",
        lang: str | None = None,
    ) -> AuditReport:
        """Audits a range of steps (or the entire timeline) and aggregates all findings."""
        return _run_audit_timeline(
            self,
            timeline_id=timeline_id,
            from_step=from_step,
            to_step=to_step,
            texts=texts,
            rules=rules,
            include_rules=include_rules,
            exclude_rules=exclude_rules,
            min_severity=min_severity,
            lang=lang,
        )
