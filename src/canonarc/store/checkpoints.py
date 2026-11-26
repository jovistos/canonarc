from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from ..models import StepCheckpoint
from .schema import LINEAGE_CTE_SQL


class _CheckpointStoreMixin:
    """Lineage-aware step checkpoint persistence for resumable pipelines."""

    if TYPE_CHECKING:
        conn: sqlite3.Connection
        _lock: threading.RLock

        def transaction(self) -> AbstractContextManager[None]: ...

    def record_step_checkpoint(
        self,
        step: int,
        *,
        timeline_id: str = "main",
        content_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepCheckpoint:
        """Records or updates a checkpoint proving `step` was committed on `timeline_id`."""
        meta = dict(metadata) if metadata else {}
        with self.transaction():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO step_checkpoints (timeline_id, step, content_hash, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (timeline_id, step, content_hash, json.dumps(meta)),
            )
        return StepCheckpoint(
            timeline_id=timeline_id,
            step=step,
            content_hash=content_hash,
            metadata=meta,
        )

    def get_step_checkpoint(
        self,
        step: int,
        *,
        timeline_id: str = "main",
    ) -> StepCheckpoint | None:
        """Returns the lineage-resolved checkpoint for `step` on `timeline_id`, if any."""
        query = (
            LINEAGE_CTE_SQL
            + """
        SELECT c.timeline_id, c.step, c.content_hash, c.metadata_json
        FROM step_checkpoints c
        JOIN lineage l
          ON c.timeline_id = l.timeline_id
         AND (l.max_step IS NULL OR c.step <= l.max_step)
        WHERE c.step = :step
        ORDER BY l.depth ASC
        LIMIT 1;
        """
        )
        with self._lock:
            row = self.conn.execute(
                query,
                {"timeline_id": timeline_id, "cutoff_step": step, "step": step},
            ).fetchone()
        if not row:
            return None
        return StepCheckpoint(
            timeline_id=row["timeline_id"],
            step=int(row["step"]),
            content_hash=row["content_hash"],
            metadata=json.loads(row["metadata_json"]),
        )

    def get_last_committed_step(self, *, timeline_id: str = "main") -> int | None:
        """Returns the highest checkpointed step visible along `timeline_id`'s lineage, or None."""
        query = (
            LINEAGE_CTE_SQL
            + """
        SELECT MAX(c.step) AS last_step
        FROM step_checkpoints c
        JOIN lineage l
          ON c.timeline_id = l.timeline_id
         AND (l.max_step IS NULL OR c.step <= l.max_step);
        """
        )
        with self._lock:
            row = self.conn.execute(
                query,
                {"timeline_id": timeline_id, "cutoff_step": None},
            ).fetchone()
        if not row or row["last_step"] is None:
            return None
        return int(row["last_step"])

    def list_step_checkpoints(
        self,
        *,
        timeline_id: str = "main",
        cutoff_step: int | None = None,
    ) -> list[StepCheckpoint]:
        """Returns lineage-resolved checkpoints in ascending step order up to `cutoff_step`."""
        query = (
            LINEAGE_CTE_SQL
            + """,
        ranked_checkpoints AS (
            SELECT
                c.timeline_id,
                c.step,
                c.content_hash,
                c.metadata_json,
                ROW_NUMBER() OVER (
                    PARTITION BY c.step
                    ORDER BY l.depth ASC
                ) AS rn
            FROM step_checkpoints c
            JOIN lineage l
              ON c.timeline_id = l.timeline_id
             AND (l.max_step IS NULL OR c.step <= l.max_step)
        )
        SELECT timeline_id, step, content_hash, metadata_json
        FROM ranked_checkpoints
        WHERE rn = 1
        ORDER BY step ASC;
        """
        )
        with self._lock:
            rows = self.conn.execute(
                query,
                {"timeline_id": timeline_id, "cutoff_step": cutoff_step},
            ).fetchall()
        return [
            StepCheckpoint(
                timeline_id=row["timeline_id"],
                step=int(row["step"]),
                content_hash=row["content_hash"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]
