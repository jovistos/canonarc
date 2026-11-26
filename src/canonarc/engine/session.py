from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..audit import AuditReport, AuditRule, AuditSeverity, DuplicateCandidate
from ..bible import (
    DEFAULT_ALLOWED_KINDS,
    BibleBootstrapReport,
    BibleBuilder,
    BibleExportFormat,
    BibleExportResult,
    BibleSection,
    BibleSyncMode,
    BibleSyncReport,
)
from ..extractors import JsonCompletionClient
from ..models import (
    AliasRecord,
    DialogueCandidate,
    EntityMergeReport,
    EntityRecord,
    EpisodeRecord,
    EpistemicMode,
    ResumePlan,
    StepCheckpoint,
    StepContext,
    StepDelta,
    VoiceExemplar,
)
from .traces import KnownEntityScope
from .transaction import StepTransaction

if TYPE_CHECKING:
    from .core import CanonArcEngine


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

    def count_consolidatable_episodes(self, step: int, **kwargs: Any) -> int:
        return self.engine.count_consolidatable_episodes(
            step, timeline_id=self.timeline_id, **kwargs
        )

    def consolidate_episodes(self, step: int, **kwargs: Any) -> EpisodeRecord | None:
        return self.engine.consolidate_episodes(step, timeline_id=self.timeline_id, **kwargs)

    def consolidate_episodes_auto(self, step: int, **kwargs: Any) -> list[EpisodeRecord]:
        return self.engine.consolidate_episodes_auto(
            step, timeline_id=self.timeline_id, **kwargs
        )

    def consolidate_chapter(
        self, step: int, chapter_id: str, **kwargs: Any
    ) -> list[EpisodeRecord]:
        return self.engine.consolidate_chapter(
            step, chapter_id, timeline_id=self.timeline_id, **kwargs
        )

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
        cache_dir: str | Path | None = None,
        bypass_cache: bool = False,
        step: int = 0,
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
        return self.engine.bootstrap_bible(
            sections,
            client=client,
            builder=builder,
            source_lang=source_lang,
            target_lang=target_lang,
            allowed_kinds=allowed_kinds,
            instructions=instructions,
            system_prompt=system_prompt,
            cache_dir=cache_dir,
            bypass_cache=bypass_cache,
            step=step,
            timeline_id=self.timeline_id,
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

    def find_duplicate_candidates(
        self,
        *,
        at_step: int | None = None,
        fuzzy_threshold: float = 88.0,
        same_kind_only: bool = True,
    ) -> list[DuplicateCandidate]:
        return self.engine.find_duplicate_candidates(
            at_step=at_step,
            timeline_id=self.timeline_id,
            fuzzy_threshold=fuzzy_threshold,
            same_kind_only=same_kind_only,
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
