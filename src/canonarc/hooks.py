from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from .models import EpisodeRecord, StepDelta, TimelineInfo, VoiceExemplar


class EngineHooks(Protocol):
    """Observability and middleware lifecycle hook interface for CanonArcEngine."""

    def on_before_commit(
        self,
        step: int,
        timeline_id: str,
        delta: StepDelta | None,
        exemplars: Sequence[VoiceExemplar],
        episodes: Sequence[EpisodeRecord],
    ) -> None: ...

    def on_after_commit(
        self,
        step: int,
        timeline_id: str,
        committed_episodes: Sequence[EpisodeRecord],
        indexed_exemplars: Sequence[VoiceExemplar],
    ) -> None: ...

    def on_rollback(self, step: int, timeline_id: str, deleted_count: int) -> None: ...

    def on_branch(self, timeline_info: TimelineInfo) -> None: ...

    def on_gate_rejected(
        self,
        stage: Literal["gate1", "gate2"],
        text: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> None: ...


class BaseEngineHooks:
    """No-op base class for subclassing only the lifecycle hooks you need."""

    def on_before_commit(
        self,
        step: int,
        timeline_id: str,
        delta: StepDelta | None,
        exemplars: Sequence[VoiceExemplar],
        episodes: Sequence[EpisodeRecord],
    ) -> None:
        return None

    def on_after_commit(
        self,
        step: int,
        timeline_id: str,
        committed_episodes: Sequence[EpisodeRecord],
        indexed_exemplars: Sequence[VoiceExemplar],
    ) -> None:
        return None

    def on_rollback(self, step: int, timeline_id: str, deleted_count: int) -> None:
        return None

    def on_branch(self, timeline_info: TimelineInfo) -> None:
        return None

    def on_gate_rejected(
        self,
        stage: Literal["gate1", "gate2"],
        text: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> None:
        return None
