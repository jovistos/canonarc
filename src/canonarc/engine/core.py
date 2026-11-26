from __future__ import annotations

import contextlib
import contextvars
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal, Self

from ..audit import (
    AuditReport,
    AuditRule,
    AuditSeverity,
    DuplicateCandidate,
    StepTextEntry,
    StepTextProvider,
    find_duplicate_candidates as _find_duplicate_candidates,
    run_audit_step as _run_audit_step,
    run_audit_timeline as _run_audit_timeline,
)
from ..bible import (
    DEFAULT_ALLOWED_KINDS,
    BibleBootstrapReport,
    BibleBuilder,
    BibleExportFormat,
    BibleExportResult,
    BibleSection,
    BibleSyncMode,
    BibleSyncReport,
    bootstrap_bible as _bootstrap_bible,
    export_bible as _export_bible,
    render_bible as _render_bible,
    sync_bible as _sync_bible,
    sync_bible_if_changed as _sync_bible_if_changed,
)
from ..embedders import Embedder
from ..episodic import EpisodeConsolidator, EpisodeRanker
from ..extractors import DeltaExtractor, JsonCompletionClient
from ..hooks import EngineHooks
from ..inflections import InflectionGenerator, InflectionRequest
from ..models import (
    AliasRecord,
    DialogueCandidate,
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
from ..presets import (
    DEFAULT_SCHEMA_POLICY,
    DomainPreset,
    InteractiveFictionPreset,
    SchemaPolicy,
)
from ..resolver import DefaultLexicalResolver, MentionResolver
from ..store import CausalStore, CheckpointStore, MemoryStore
from ..text_masking import (
    TextNormalizer,
    apply_text_masking,
    compile_ignore_patterns,
    strip_ignored_spans,
)
from ..voice import (
    HeuristicVoiceQualityScorer,
    IngestionGate,
    VoiceBank,
    VoiceQualityScorer,
)
from .consolidation import (
    consolidate_chapter_impl,
    consolidate_episodes_auto_impl,
    consolidate_episodes_impl,
    count_consolidatable_episodes_impl,
)
from .embeddings import (
    check_embedder_compatibility_impl,
    record_embedder_metadata_impl,
    reindex_embeddings_impl,
)
from .harvesting import (
    extract_dialogue_candidates_with_delta,
    index_voice_exemplars_impl,
    resolve_harvest_metadata,
)
from .merging import merge_entities_impl
from .mutations import apply_delta_impl, commit_step_impl, index_episodes_impl
from .normalization import (
    build_extraction_messages_impl,
    normalize_delta_impl,
)
from .resume import checkpoint_matches, plan_resume_impl
from .retrieval import retrieve_context_impl
from .seeding import (
    build_target_aliases_impl,
    expand_entity_inflections_impl,
    make_inflection_request_impl,
    register_entity_impl,
    seed_entities_impl,
)
from .session import TimelineSession
from .traces import KnownEntityScope
from .transaction import StepTransaction


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
        episode_consolidator: EpisodeConsolidator | None = None,
        inflection_generator: InflectionGenerator | None = None,
        voice_scorer: VoiceQualityScorer | None = None,
        hooks: Sequence[EngineHooks] = (),
        audit_rules: Sequence[AuditRule] | None = None,
        on_embedder_mismatch: Literal["raise", "warn", "reindex", "ignore"] = "warn",
        ignore_patterns: Sequence[str | re.Pattern[str]] = (),
        text_normalizer: TextNormalizer | None = None,
    ) -> None:
        self.store: MemoryStore = store if store is not None else CausalStore(db_path)
        self._default_timeline = timeline_id
        self._timeline_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"canonarc_timeline_{id(self)}", default=None
        )
        self._ignore_patterns: tuple[re.Pattern[str], ...] = compile_ignore_patterns(
            ignore_patterns
        )
        self.text_normalizer: TextNormalizer | None = text_normalizer
        self.voice_bank = VoiceBank(
            embedder=embedder, gate=gate, episode_ranker=episode_ranker
        )
        self.preset: DomainPreset = preset or InteractiveFictionPreset()
        self.mention_resolver: MentionResolver = mention_resolver or DefaultLexicalResolver()
        self.extractor: DeltaExtractor | None = extractor
        self.episode_consolidator: EpisodeConsolidator | None = episode_consolidator
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
    def ignore_patterns(self) -> tuple[re.Pattern[str], ...]:
        """Compiled regex patterns masked/stripped during lexical resolution, voice scoring, and audits."""
        return self._ignore_patterns

    def _mask_text(self, text: str) -> str:
        if not text or (not self._ignore_patterns and self.text_normalizer is None):
            return text
        return apply_text_masking(
            text, self._ignore_patterns, normalizer=self.text_normalizer
        )

    def _strip_text(self, text: str) -> str:
        if not text or (not self._ignore_patterns and self.text_normalizer is None):
            return text
        return strip_ignored_spans(
            text, self._ignore_patterns, normalizer=self.text_normalizer
        )

    @property
    def _policy(self) -> SchemaPolicy:
        return getattr(self.preset, "policy", DEFAULT_SCHEMA_POLICY)

    def close(self) -> None:
        self.store.close()

    def _check_embedder_compatibility(self) -> None:
        """Verifies that the active embedder matches any embedder/dimension recorded in the store."""
        check_embedder_compatibility_impl(self)

    def _record_embedder_metadata(self, vectors: Sequence[Any]) -> None:
        """Records embedder_id and vector dimension in store_metadata on first write."""
        record_embedder_metadata_impl(self, vectors)

    def reindex_embeddings(self, *, batch_size: int = 128) -> dict[str, int]:
        """Re-embeds all stored episodes and voice exemplars in batches using the active embedder."""
        return reindex_embeddings_impl(self, batch_size=batch_size)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # --- Tier 1 Seeding & Inflections ---

    def _resolve_mentions(
        self,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
        *,
        lang: str | None = None,
    ) -> list[ResolvedMention]:
        masked_text = self._mask_text(text)
        if lang is not None:
            try:
                return self.mention_resolver.resolve(  # type: ignore[call-arg]
                    text=masked_text,
                    explicit_mentions=explicit_mentions,
                    entities=entities,
                    aliases=aliases,
                    lang=lang,
                )
            except TypeError:
                pass
        return self.mention_resolver.resolve(
            text=masked_text,
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
        return make_inflection_request_impl(entity, target_lang=target_lang)

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
        return build_target_aliases_impl(
            self,
            entity,
            existing_aliases,
            target_lang=target_lang,
            auto_inflect=auto_inflect,
            prefetched_forms=prefetched_forms,
        )

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
        register_entity_impl(
            self,
            entity,
            aliases=aliases,
            step=step,
            timeline_id=timeline_id,
            target_lang=target_lang,
            auto_inflect=auto_inflect,
        )

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
        seed_entities_impl(
            self,
            entities,
            step=step,
            timeline_id=timeline_id,
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
        timeline_id: str | None = None,
    ) -> list[AliasRecord]:
        """One-shot inflection expansion for an already-registered or Zero-Seed discovered entity."""
        return expand_entity_inflections_impl(
            self,
            entity_id,
            target_lang=target_lang,
            step=step,
            timeline_id=timeline_id,
        )

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
        """Atomically merges duplicate_id into canonical_id at (timeline_id, step)."""
        return merge_entities_impl(
            self,
            duplicate_id,
            canonical_id,
            step=step,
            timeline_id=timeline_id,
            merge_attributes=merge_attributes,
            migrate_edges=migrate_edges,
            migrate_voice=migrate_voice,
            migrate_episodes=migrate_episodes,
            conflict_strategy=conflict_strategy,
            allow_kind_mismatch=allow_kind_mismatch,
            reason=reason,
            dry_run=dry_run,
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
        return retrieve_context_impl(
            self,
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
            timeline_id=timeline_id,
            fallback_on_empty_mentions=fallback_on_empty_mentions,
            story_time=story_time,
            chapter_id=chapter_id,
            scene_id=scene_id,
            voice_constraints=voice_constraints,
            lang=lang,
        )

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
        return apply_delta_impl(self, step, delta, timeline_id=timeline_id)

    def index_episodes(
        self,
        step: int,
        episodes: Sequence[EpisodeRecord],
        *,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        return index_episodes_impl(self, step, episodes, timeline_id=timeline_id)

    def count_consolidatable_episodes(
        self,
        step: int,
        *,
        before_step: int,
        max_salience: float = 4.0,
        predicate: str = "consolidated_summary",
        chapter_id: str | None = None,
        preserve_provenance: bool = True,
        preserve_deceived: bool = True,
        timeline_id: str | None = None,
    ) -> int:
        """Zero-LLM readiness check returning how many episodes match consolidation criteria."""
        return count_consolidatable_episodes_impl(
            self,
            step,
            before_step=before_step,
            max_salience=max_salience,
            predicate=predicate,
            chapter_id=chapter_id,
            preserve_provenance=preserve_provenance,
            preserve_deceived=preserve_deceived,
            timeline_id=timeline_id,
        )

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
        location_id: str | None = None,
        key_participant_ids: Sequence[str] | None = None,
        max_participants: int | None = None,
        preserve_provenance: bool = True,
        preserve_deceived: bool = True,
        epistemic_merge: Literal["union", "intersection"] = "union",
        anchor_step: Literal["current", "last_target"] = "current",
        force: bool = False,
        timeline_id: str | None = None,
    ) -> EpisodeRecord | None:
        """Compacts old low-salience episodes (< before_step) into a single summary episode."""
        return consolidate_episodes_impl(
            self,
            step,
            before_step=before_step,
            summary=summary,
            max_salience=max_salience,
            predicate=predicate,
            episode_id=episode_id,
            chapter_id=chapter_id,
            salience=salience,
            location_id=location_id,
            key_participant_ids=key_participant_ids,
            max_participants=max_participants,
            preserve_provenance=preserve_provenance,
            preserve_deceived=preserve_deceived,
            epistemic_merge=epistemic_merge,
            anchor_step=anchor_step,
            force=force,
            timeline_id=timeline_id,
        )

    def consolidate_episodes_auto(
        self,
        step: int,
        *,
        before_step: int,
        consolidator: EpisodeConsolidator | None = None,
        client: JsonCompletionClient | None = None,
        max_salience: float = 4.0,
        min_episodes: int = 3,
        max_episodes_per_batch: int = 30,
        predicate: str = "consolidated_summary",
        chapter_id: str | None = None,
        salience: float | None = None,
        default_salience: float = 6.0,
        max_participants: int | None = 5,
        preserve_provenance: bool = True,
        preserve_deceived: bool = True,
        epistemic_merge: Literal["union", "intersection"] = "union",
        anchor_step: Literal["last_target", "current"] = "last_target",
        force: bool = False,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        """Queries, summarizes via an EpisodeConsolidator / LLM client, and commits compacted Tier 6 episodes."""
        return consolidate_episodes_auto_impl(
            self,
            step,
            before_step=before_step,
            consolidator=consolidator,
            client=client,
            max_salience=max_salience,
            min_episodes=min_episodes,
            max_episodes_per_batch=max_episodes_per_batch,
            predicate=predicate,
            chapter_id=chapter_id,
            salience=salience,
            default_salience=default_salience,
            max_participants=max_participants,
            preserve_provenance=preserve_provenance,
            preserve_deceived=preserve_deceived,
            epistemic_merge=epistemic_merge,
            anchor_step=anchor_step,
            force=force,
            timeline_id=timeline_id,
        )

    def consolidate_chapter(
        self,
        step: int,
        chapter_id: str,
        *,
        before_step: int | None = None,
        consolidator: EpisodeConsolidator | None = None,
        client: JsonCompletionClient | None = None,
        max_salience: float = 4.0,
        min_episodes: int = 3,
        max_episodes_per_batch: int = 30,
        predicate: str = "consolidated_summary",
        salience: float | None = None,
        default_salience: float = 6.0,
        max_participants: int | None = 5,
        preserve_provenance: bool = True,
        preserve_deceived: bool = True,
        epistemic_merge: Literal["union", "intersection"] = "union",
        anchor_step: Literal["last_target", "current"] = "last_target",
        force: bool = False,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        """Convenience helper to auto-consolidate low-salience episodes belonging to a single chapter."""
        return consolidate_chapter_impl(
            self,
            step,
            chapter_id,
            before_step=before_step,
            consolidator=consolidator,
            client=client,
            max_salience=max_salience,
            min_episodes=min_episodes,
            max_episodes_per_batch=max_episodes_per_batch,
            predicate=predicate,
            salience=salience,
            default_salience=default_salience,
            max_participants=max_participants,
            preserve_provenance=preserve_provenance,
            preserve_deceived=preserve_deceived,
            epistemic_merge=epistemic_merge,
            anchor_step=anchor_step,
            force=force,
            timeline_id=timeline_id,
        )

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

    # --- Voice Harvesting & Indexing ---

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
        return extract_dialogue_candidates_with_delta(
            self,
            text,
            source_text=source_text,
            step=step,
            timeline_id=timeline_id,
            prev_text=prev_text,
            next_text=next_text,
            lang=lang,
            staged_delta=staged_delta,
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
        return resolve_harvest_metadata(
            self,
            speaker_id=speaker_id,
            addressee_id=addressee_id,
            context=context,
            staged_delta=staged_delta,
            cutoff_step=cutoff_step,
            timeline_id=timeline_id,
            infer_addressee=infer_addressee,
        )

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
        return extract_dialogue_candidates_with_delta(
            self,
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
        return index_voice_exemplars_impl(
            self,
            step,
            exemplars,
            bypass_gate=bypass_gate,
            timeline_id=timeline_id,
        )

    # --- Step Commit & Checkpoints ---

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
        return commit_step_impl(
            self,
            step,
            delta,
            exemplars,
            episodes,
            bypass_gate=bypass_gate,
            timeline_id=timeline_id,
            content_hash=content_hash,
            checkpoint_metadata=checkpoint_metadata,
        )

    def _require_checkpoint_store(self) -> CheckpointStore:
        if not isinstance(self.store, CheckpointStore):
            raise NotImplementedError("Active store backend does not support step checkpoints.")
        return self.store

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
        return checkpoint_matches(cp, content_hash=content_hash, metadata_match=metadata_match)

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
        """Plans which ordered (step, content_hash) pairs can be reused vs. re-run."""
        return plan_resume_impl(
            self,
            step_hashes,
            causal=causal,
            metadata_match=metadata_match,
            timeline_id=timeline_id,
            apply_rollback=apply_rollback,
            force=force,
            prune_tail=prune_tail,
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

    # --- Series Bible Export / Sync ---

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
        include_provenance: bool = False,
        lang: str | None = None,
    ) -> tuple[str, BibleExportResult]:
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
            include_provenance=include_provenance,
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
        include_provenance: bool = False,
        lang: str | None = None,
        excel_bom: bool = False,
    ) -> BibleExportResult:
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
            include_provenance=include_provenance,
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
        ignore_columns: Sequence[str] | None = None,
    ) -> BibleSyncReport:
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
            ignore_columns=ignore_columns,
        )

    def sync_bible_if_changed(
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
        ignore_columns: Sequence[str] | None = None,
        force: bool = False,
    ) -> BibleSyncReport:
        return _sync_bible_if_changed(
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
            ignore_columns=ignore_columns,
            force=force,
        )

    def bootstrap_bible(
        self,
        sections: Sequence[BibleSection | Mapping[str, Any] | str],
        *,
        client: JsonCompletionClient | None = None,
        builder: BibleBuilder | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
        allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
        instructions: str = "",
        system_prompt: str | None = None,
        ignore_patterns: Sequence[str | re.Pattern[str]] = (),
        max_notes_chars: int = 300,
        cache_dir: str | Path | None = None,
        bypass_cache: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 1,
        step: int = 0,
        timeline_id: str | None = None,
        mode: BibleSyncMode = "upsert",
        inherit_existing: bool = True,
        auto_inflect: bool = True,
        batch_size: int = 25,
        commit: bool = True,
        dry_run: bool = False,
        export_path: str | Path | None = None,
        export_format: Literal["auto", "csv", "json"] = "auto",
        check_duplicates: bool = True,
        fuzzy_threshold: float = 88.0,
        same_kind_only: bool = True,
    ) -> BibleBootstrapReport:
        return _bootstrap_bible(
            self,
            sections,
            client=client,
            builder=builder,
            source_lang=source_lang,
            target_lang=target_lang,
            allowed_kinds=allowed_kinds,
            instructions=instructions,
            system_prompt=system_prompt,
            ignore_patterns=ignore_patterns,
            max_notes_chars=max_notes_chars,
            cache_dir=cache_dir,
            bypass_cache=bypass_cache,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            step=step,
            timeline_id=timeline_id,
            mode=mode,
            inherit_existing=inherit_existing,
            auto_inflect=auto_inflect,
            batch_size=batch_size,
            commit=commit,
            dry_run=dry_run,
            export_path=export_path,
            export_format=export_format,
            check_duplicates=check_duplicates,
            fuzzy_threshold=fuzzy_threshold,
            same_kind_only=same_kind_only,
        )

    # --- Formatting, Extraction Prompts & Normalization ---

    def format_context(self, ctx: StepContext) -> str:
        """Formats StepContext into prompt text using the active DomainPreset."""
        return self.preset.format_prompt_context(ctx)

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
        return build_extraction_messages_impl(
            self,
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
        return normalize_delta_impl(self, delta, step=step, timeline_id=timeline_id)

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
        texts: Mapping[int, StepTextEntry] | None = None,
        text_provider: StepTextProvider | None = None,
        rules: Sequence[AuditRule] | None = None,
        include_rules: Sequence[str] | None = None,
        exclude_rules: Sequence[str] | None = None,
        min_severity: AuditSeverity = "info",
        lang: str | None = None,
    ) -> AuditReport:
        return _run_audit_timeline(
            self,
            timeline_id=timeline_id,
            from_step=from_step,
            to_step=to_step,
            texts=texts,
            text_provider=text_provider,
            rules=rules,
            include_rules=include_rules,
            exclude_rules=exclude_rules,
            min_severity=min_severity,
            lang=lang,
        )

    def find_duplicate_candidates(
        self,
        *,
        at_step: int | None = None,
        timeline_id: str | None = None,
        fuzzy_threshold: float = 88.0,
        same_kind_only: bool = True,
    ) -> list[DuplicateCandidate]:
        return _find_duplicate_candidates(
            self,
            at_step=at_step,
            timeline_id=timeline_id,
            fuzzy_threshold=fuzzy_threshold,
            same_kind_only=same_kind_only,
        )
