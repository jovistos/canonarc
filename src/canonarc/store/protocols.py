from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from ..models import (
    AliasRecord,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    StepCheckpoint,
    StepDelta,
    TimelineInfo,
    VoiceExemplar,
)


class TimelineBranchConflictError(RuntimeError):
    """Raised when rolling back a parent timeline would invalidate child branches."""


class MemoryStore(Protocol):
    """Backend storage protocol for CanonArcEngine."""

    def close(self) -> None: ...
    def create_timeline(
        self,
        timeline_id: str,
        parent_id: str | None = None,
        fork_step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineInfo: ...
    def get_timeline(self, timeline_id: str) -> TimelineInfo | None: ...
    def list_timelines(self) -> list[TimelineInfo]: ...
    def delete_timeline(self, timeline_id: str, *, cascade: bool = False) -> int: ...
    def rollback_to(self, step: int, *, timeline_id: str, force: bool = False) -> int: ...
    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str | AliasRecord] = (),
        step: int = 0,
        timeline_id: str = "main",
    ) -> None: ...
    def get_active_entity(
        self,
        entity_id: str,
        *,
        timeline_id: str,
        cutoff_step: int,
    ) -> EntityRecord | None: ...
    def get_all_active_entities(self, timeline_id: str, cutoff_step: int | None) -> list[EntityRecord]: ...
    def get_all_active_edges(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        *,
        relation_type: str | None = None,
    ) -> list[DirectedEdgeRecord]: ...
    def get_all_active_aliases(self, timeline_id: str, cutoff_step: int | None) -> list[AliasRecord]: ...
    def apply_delta(
        self,
        step: int,
        delta: StepDelta,
        *,
        timeline_id: str = "main",
        episode_embeddings: dict[str, bytes] | None = None,
    ) -> list[EpisodeRecord]: ...
    def insert_episodes(
        self,
        episodes: Sequence[tuple[EpisodeRecord, bytes]],
        *,
        timeline_id: str = "main",
    ) -> None: ...
    def record_epistemic(
        self,
        inform: InformDelta,
        *,
        step: int,
        timeline_id: str = "main",
    ) -> None: ...
    def get_lineage_episodes(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        *,
        episode_id: str | None = None,
        episode_ids: Sequence[str] | None = None,
        limit: int | None = None,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        cue_entity_ids: Sequence[str] | None = None,
        pinned_episode_ids: Sequence[str] | None = None,
        query_text: str = "",
    ) -> list[tuple[EpisodeRecord, bytes]]: ...
    def get_known_episode_ids(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        pov_entity_ids: Sequence[str],
        *,
        pov_mode: Literal["any", "all"] = "any",
        episode_ids: Sequence[str] | None = None,
    ) -> set[str]: ...
    def query_episodes(
        self,
        *,
        timeline_id: str = "main",
        cutoff_step: int | None = None,
        participants: Sequence[str] = (),
        match_all_participants: bool = False,
        predicate: str | None = None,
        location_id: str | None = None,
        known_to: Sequence[str] = (),
        min_salience: float = 0.0,
        step_range: tuple[int, int] | None = None,
    ) -> list[EpisodeRecord]: ...
    def insert_voice_exemplars(
        self,
        exemplars: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        timeline_id: str = "main",
    ) -> None: ...
    def get_lineage_voice_exemplars(
        self,
        timeline_id: str,
        cutoff_step: int,
        speaker_id: str,
    ) -> list[tuple[VoiceExemplar, bytes]]: ...
    def get_history(
        self,
        subject_key: str,
        *,
        timeline_id: str,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]: ...
    def diff_timelines(self, timeline_a: str, timeline_b: str, *, at_step: int) -> dict[str, Any]: ...
    def export_snapshot(
        self,
        target_path: str | Path,
        *,
        at_step: int,
        timeline_id: str = "main",
    ) -> Path: ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Optional extension protocol for backends supporting step-level resumability checkpoints."""

    def record_step_checkpoint(
        self,
        step: int,
        *,
        timeline_id: str = "main",
        content_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepCheckpoint: ...
    def get_step_checkpoint(
        self,
        step: int,
        *,
        timeline_id: str = "main",
    ) -> StepCheckpoint | None: ...
    def get_last_committed_step(self, *, timeline_id: str = "main") -> int | None: ...
    def list_step_checkpoints(
        self,
        *,
        timeline_id: str = "main",
        cutoff_step: int | None = None,
    ) -> list[StepCheckpoint]: ...


@runtime_checkable
class MetadataStore(Protocol):
    """Optional extension protocol for backends supporting key-value store_metadata persistence."""

    def get_metadata(self, key: str) -> str | None: ...
    def set_metadata(self, key: str, value: str) -> None: ...
    def delete_metadata(self, key: str) -> bool: ...
    def delete_metadata_prefix(self, prefix: str) -> int: ...


def fetch_active_graph_state(
    store: MemoryStore,
    timeline_id: str,
    cutoff_step: int,
    *,
    edge_types: Sequence[str] | None = None,
    include_edges: bool = True,
) -> tuple[list[EntityRecord], list[AliasRecord], list[DirectedEdgeRecord]]:
    """Fetches active (entities, aliases, edges) using single-pass CTE when available."""
    eff_edge_types: Sequence[str] | None = () if not include_edges else edge_types
    get_graph_fn = getattr(store, "get_active_graph_state", None)
    if callable(get_graph_fn):
        if eff_edge_types is None:
            return get_graph_fn(timeline_id, cutoff_step)
        return get_graph_fn(timeline_id, cutoff_step, edge_types=eff_edge_types)

    entities = store.get_all_active_entities(timeline_id, cutoff_step)
    aliases = store.get_all_active_aliases(timeline_id, cutoff_step)
    if not include_edges:
        edges: list[DirectedEdgeRecord] = []
    else:
        edges = store.get_all_active_edges(timeline_id, cutoff_step)
        if eff_edge_types is not None:
            allowed = set(eff_edge_types)
            edges = [e for e in edges if e.relation_type in allowed]
    return entities, aliases, edges
