from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from .models import (
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

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS timelines (
    timeline_id TEXT PRIMARY KEY,
    parent_id   TEXT REFERENCES timelines(timeline_id),
    fork_step   INTEGER,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS graph_revisions (
    revision_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    timeline_id  TEXT NOT NULL REFERENCES timelines(timeline_id),
    step         INTEGER NOT NULL,
    record_type  TEXT NOT NULL, -- 'entity' | 'edge' | 'alias'
    subject_key  TEXT NOT NULL, -- 'char_arthur' | 'char_arthur:stance->char_merlin' | 'char_merlin:the old wizard'
    relation_type TEXT,         -- predicate for 'edge' (e.g. 'stance', 'kinship'); NULL for 'entity'/'alias'
    source_id    TEXT,          -- entity_id for 'entity'/'alias'; source_id for 'edge'
    target_id    TEXT,          -- target_id for 'edge'
    is_tombstone INTEGER NOT NULL DEFAULT 0, -- 1 if deleted/expired at this step
    payload      TEXT NOT NULL  -- Complete materialized JSON state at this step
);

CREATE INDEX IF NOT EXISTS idx_revisions_lookup
    ON graph_revisions(timeline_id, record_type, subject_key, step DESC, revision_id DESC);

CREATE INDEX IF NOT EXISTS idx_revisions_timeline_step
    ON graph_revisions(timeline_id, step DESC);

CREATE INDEX IF NOT EXISTS idx_revisions_edge_type
    ON graph_revisions(timeline_id, record_type, relation_type, source_id, target_id);

CREATE TABLE IF NOT EXISTS voice_exemplars (
    exemplar_id           TEXT NOT NULL,
    timeline_id           TEXT NOT NULL REFERENCES timelines(timeline_id),
    step                  INTEGER NOT NULL,
    speaker_id            TEXT NOT NULL,
    addressee_id          TEXT,
    text                  TEXT NOT NULL,
    source_text           TEXT NOT NULL DEFAULT '',
    metadata              TEXT NOT NULL DEFAULT '{}',
    quality_score         REAL NOT NULL,
    distinctiveness_score REAL NOT NULL,
    embedding             BLOB NOT NULL,
    PRIMARY KEY (timeline_id, exemplar_id)
);

CREATE INDEX IF NOT EXISTS idx_voice_lookup
    ON voice_exemplars(timeline_id, speaker_id, step DESC);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT NOT NULL,
    timeline_id  TEXT NOT NULL REFERENCES timelines(timeline_id),
    step         INTEGER NOT NULL,
    predicate    TEXT NOT NULL DEFAULT 'event',
    summary      TEXT NOT NULL,
    location_id  TEXT,
    salience     REAL NOT NULL DEFAULT 5.0,
    participants TEXT NOT NULL DEFAULT '{}',
    witnesses    TEXT NOT NULL DEFAULT '[]',
    metadata     TEXT NOT NULL DEFAULT '{}',
    embedding    BLOB NOT NULL DEFAULT X'',
    PRIMARY KEY (timeline_id, episode_id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_lookup
    ON episodes(timeline_id, step DESC);

CREATE TABLE IF NOT EXISTS episode_epistemics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timeline_id      TEXT NOT NULL REFERENCES timelines(timeline_id),
    step             INTEGER NOT NULL,
    episode_id       TEXT NOT NULL,
    entity_id        TEXT NOT NULL,
    epistemic        TEXT NOT NULL DEFAULT 'direct', -- 'direct' | 'told' | 'deceived'
    believed_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_episode_epistemics_lookup
    ON episode_epistemics(timeline_id, episode_id, step DESC);

CREATE INDEX IF NOT EXISTS idx_episode_epistemics_timeline_step
    ON episode_epistemics(timeline_id, step DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    timeline_id UNINDEXED,
    episode_id UNINDEXED,
    summary
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    DELETE FROM episodes_fts WHERE rowid = NEW.rowid;
    INSERT INTO episodes_fts(rowid, timeline_id, episode_id, summary)
    VALUES (NEW.rowid, NEW.timeline_id, NEW.episode_id, NEW.summary);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    DELETE FROM episodes_fts WHERE rowid = OLD.rowid;
END;

CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE OF summary ON episodes BEGIN
    DELETE FROM episodes_fts WHERE rowid = OLD.rowid;
    INSERT INTO episodes_fts(rowid, timeline_id, episode_id, summary)
    VALUES (NEW.rowid, NEW.timeline_id, NEW.episode_id, NEW.summary);
END;

CREATE TABLE IF NOT EXISTS store_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS step_checkpoints (
    timeline_id   TEXT NOT NULL REFERENCES timelines(timeline_id),
    step          INTEGER NOT NULL,
    content_hash  TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (timeline_id, step)
);
"""

LINEAGE_CTE_SQL = """
WITH RECURSIVE lineage(timeline_id, parent_id, max_step, depth) AS (
    SELECT timeline_id, parent_id, :cutoff_step, 0
    FROM timelines
    WHERE timeline_id = :timeline_id
    UNION ALL
    SELECT
        t.timeline_id,
        t.parent_id,
        CASE
            WHEN l.max_step IS NULL THEN t_child.fork_step
            WHEN t_child.fork_step IS NULL THEN l.max_step
            ELSE MIN(l.max_step, t_child.fork_step)
        END,
        l.depth + 1
    FROM lineage l
    JOIN timelines t_child ON t_child.timeline_id = l.timeline_id
    JOIN timelines t       ON t.timeline_id = l.parent_id
    WHERE l.parent_id IS NOT NULL
)
"""

LINEAGE_ACTIVE_REVISIONS_CTE = LINEAGE_CTE_SQL + """,
step_deduped AS (
    SELECT
        r.revision_id,
        r.timeline_id,
        r.step,
        r.record_type,
        r.subject_key,
        r.relation_type,
        r.source_id,
        r.target_id,
        r.is_tombstone,
        r.payload,
        l.depth,
        ROW_NUMBER() OVER (
            PARTITION BY r.record_type, r.subject_key, r.step
            ORDER BY l.depth ASC, r.revision_id DESC
        ) AS step_rn
    FROM graph_revisions r
    JOIN lineage l
      ON r.timeline_id = l.timeline_id
     AND (l.max_step IS NULL OR r.step <= l.max_step)
    /*EXTRA_FILTERS*/
),
ranked_revisions AS (
    SELECT
        revision_id,
        timeline_id,
        step AS valid_from_step,
        record_type,
        subject_key,
        relation_type,
        source_id,
        target_id,
        is_tombstone,
        payload,
        ROW_NUMBER() OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY record_type, subject_key
        ) AS revision_count,
        LEAD(payload) OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS prior_payload,
        LEAD(step) OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS prior_step,
        LEAD(is_tombstone) OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS prior_is_tombstone
    FROM step_deduped
    WHERE step_rn = 1
)
SELECT
    record_type,
    subject_key,
    relation_type,
    source_id,
    target_id,
    valid_from_step,
    payload,
    revision_count,
    prior_payload,
    prior_step,
    prior_is_tombstone
FROM ranked_revisions
WHERE rn = 1 AND is_tombstone = 0;
"""


def _unpack_edge_payload(raw_payload: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    """Separates clean edge state dict from optional _provenance metadata."""
    state = dict(raw_payload)
    prov = state.pop("_provenance", None)
    if isinstance(prov, dict):
        return state, prov.get("caused_by_episode_id"), prov.get("reason")
    return state, None, None


def _pack_episode_metadata(ep: EpisodeRecord) -> dict[str, Any]:
    """Serializes optional narrative-time and consolidation attributes into episode metadata."""
    meta = dict(ep.metadata)
    if ep.story_time is not None:
        meta["_story_time"] = ep.story_time
    if ep.chapter_id is not None:
        meta["_chapter_id"] = ep.chapter_id
    if ep.scene_id is not None:
        meta["_scene_id"] = ep.scene_id
    if ep.consolidated_from:
        meta["_consolidated_from"] = list(ep.consolidated_from)
    return meta


def _unpack_episode_metadata(
    raw_meta: dict[str, Any],
) -> tuple[dict[str, Any], int | float | str | None, str | None, str | None, list[str]]:
    """Extracts narrative-time and consolidation fields from stored episode metadata."""
    meta = dict(raw_meta)
    story_time = meta.pop("_story_time", None)
    chapter_id = meta.pop("_chapter_id", None)
    scene_id = meta.pop("_scene_id", None)
    consolidated_from = meta.pop("_consolidated_from", None) or []
    return meta, story_time, chapter_id, scene_id, list(consolidated_from)


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


class CausalStore:
    """Manages SQLite storage, recursive lineage resolution, and time-travel operations."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Reentrant atomic SQLite transaction guarded by an RLock."""
        with self._lock:
            is_outermost = self._tx_depth == 0
            if is_outermost and not self.conn.in_transaction:
                self.conn.execute("BEGIN")
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                self._tx_depth -= 1
                if is_outermost and self.conn.in_transaction:
                    self.conn.rollback()
                raise
            else:
                self._tx_depth -= 1
                if is_outermost and self.conn.in_transaction:
                    self.conn.commit()

    def _init_db(self) -> None:
        self.conn.execute("PRAGMA foreign_keys = ON;")
        if self.db_path != ":memory:" and not self.db_path.startswith("file::memory:"):
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA busy_timeout = 5000;")
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)
            # Safe migration for existing DBs lacking believed_summary column
            cols = {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(episode_epistemics)").fetchall()
            }
            if "believed_summary" not in cols:
                self.conn.execute("ALTER TABLE episode_epistemics ADD COLUMN believed_summary TEXT")
            # Backfill FTS5 index for legacy databases created before episodes_fts existed
            if not self.conn.execute("SELECT 1 FROM episodes_fts LIMIT 1").fetchone():
                self.conn.execute(
                    "INSERT INTO episodes_fts(rowid, timeline_id, episode_id, summary) "
                    "SELECT rowid, timeline_id, episode_id, summary FROM episodes"
                )
            # Ensure default 'main' timeline exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM timelines WHERE timeline_id = 'main'")
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO timelines (timeline_id, parent_id, fork_step, metadata) VALUES (?, NULL, NULL, ?)",
                    ("main", json.dumps({"description": "Primary timeline"})),
                )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # --- Embedding Provenance & Re-indexing ---

    def get_embedding_metadata(self) -> tuple[str | None, int | None]:
        """Returns (embedder_id, embedding_dim) recorded in store_metadata, or infers dim from existing BLOBs."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT key, value FROM store_metadata WHERE key IN ('embedder_id', 'embedding_dim')"
            ).fetchall()
            meta = {row["key"]: row["value"] for row in rows}
            embedder_id = meta.get("embedder_id")
            raw_dim = meta.get("embedding_dim")
            if raw_dim is not None:
                return embedder_id, int(raw_dim)

            # Fallback for legacy DBs created before store_metadata existed
            for table in ("episodes", "voice_exemplars"):
                row = self.conn.execute(
                    f"SELECT length(embedding) AS byte_len FROM {table} WHERE length(embedding) > 0 LIMIT 1"
                ).fetchone()
                if row and row["byte_len"]:
                    return embedder_id, int(row["byte_len"]) // 4
            return embedder_id, None

    def set_embedding_metadata(self, embedder_id: str, embedding_dim: int) -> None:
        """Persists the active embedder_id and embedding_dim in store_metadata."""
        with self.transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO store_metadata (key, value) VALUES ('embedder_id', ?)",
                (embedder_id,),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO store_metadata (key, value) VALUES ('embedding_dim', ?)",
                (str(int(embedding_dim)),),
            )

    def list_all_episodes_for_reembed(self) -> list[tuple[str, str, str]]:
        """Returns (timeline_id, episode_id, summary) for all stored episodes."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT timeline_id, episode_id, summary FROM episodes"
            ).fetchall()
            return [(row["timeline_id"], row["episode_id"], row["summary"]) for row in rows]

    def list_all_voice_exemplars_for_reembed(self) -> list[tuple[str, str, str]]:
        """Returns (timeline_id, exemplar_id, text) for all stored voice exemplars."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT timeline_id, exemplar_id, text FROM voice_exemplars"
            ).fetchall()
            return [(row["timeline_id"], row["exemplar_id"], row["text"]) for row in rows]

    def update_episode_embeddings(self, updates: Sequence[tuple[str, str, bytes]]) -> int:
        """Updates embedding BLOBs for (timeline_id, episode_id, emb_bytes) pairs."""
        if not updates:
            return 0
        with self.transaction():
            self.conn.executemany(
                "UPDATE episodes SET embedding = ? WHERE timeline_id = ? AND episode_id = ?",
                [(emb, tl, eid) for tl, eid, emb in updates],
            )
            return len(updates)

    def update_voice_embeddings(self, updates: Sequence[tuple[str, str, bytes]]) -> int:
        """Updates embedding BLOBs for (timeline_id, exemplar_id, emb_bytes) pairs."""
        if not updates:
            return 0
        with self.transaction():
            self.conn.executemany(
                "UPDATE voice_exemplars SET embedding = ? WHERE timeline_id = ? AND exemplar_id = ?",
                [(emb, tl, xid) for tl, xid, emb in updates],
            )
            return len(updates)

    # --- Timelines & Branching ---

    def _compute_head_step(self, timeline_id: str, fork_step: int | None) -> int:
        row = self.conn.execute(
            """
            SELECT MAX(s) AS max_step FROM (
                SELECT MAX(step) AS s FROM graph_revisions WHERE timeline_id = ?
                UNION ALL
                SELECT MAX(step) AS s FROM episodes WHERE timeline_id = ?
                UNION ALL
                SELECT MAX(step) AS s FROM voice_exemplars WHERE timeline_id = ?
                UNION ALL
                SELECT MAX(step) AS s FROM episode_epistemics WHERE timeline_id = ?
                UNION ALL
                SELECT MAX(step) AS s FROM step_checkpoints WHERE timeline_id = ?
            )
            """,
            (timeline_id, timeline_id, timeline_id, timeline_id, timeline_id),
        ).fetchone()
        max_step = row["max_step"] if row else None
        if max_step is not None:
            return int(max_step)
        return int(fork_step) if fork_step is not None else 0

    def create_timeline(
        self,
        timeline_id: str,
        parent_id: str | None = None,
        fork_step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineInfo:
        if parent_id is not None and fork_step is None:
            raise ValueError("fork_step is required when parent_id is specified.")
        with self.transaction():
            self.conn.execute(
                "INSERT INTO timelines (timeline_id, parent_id, fork_step, metadata) VALUES (?, ?, ?, ?)",
                (timeline_id, parent_id, fork_step, json.dumps(metadata or {})),
            )
        return TimelineInfo(
            timeline_id=timeline_id,
            parent_id=parent_id,
            fork_step=fork_step,
            head_step=fork_step if fork_step is not None else 0,
            metadata=metadata or {},
        )

    def get_timeline(self, timeline_id: str) -> TimelineInfo | None:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timeline_id, parent_id, fork_step, metadata FROM timelines WHERE timeline_id = ?",
                (timeline_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return TimelineInfo(
                timeline_id=row["timeline_id"],
                parent_id=row["parent_id"],
                fork_step=row["fork_step"],
                head_step=self._compute_head_step(row["timeline_id"], row["fork_step"]),
                metadata=json.loads(row["metadata"]),
            )

    def list_timelines(self) -> list[TimelineInfo]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                WITH all_steps AS (
                    SELECT timeline_id, MAX(step) AS max_step FROM graph_revisions GROUP BY timeline_id
                    UNION ALL
                    SELECT timeline_id, MAX(step) AS max_step FROM episodes GROUP BY timeline_id
                    UNION ALL
                    SELECT timeline_id, MAX(step) AS max_step FROM voice_exemplars GROUP BY timeline_id
                    UNION ALL
                    SELECT timeline_id, MAX(step) AS max_step FROM episode_epistemics GROUP BY timeline_id
                    UNION ALL
                    SELECT timeline_id, MAX(step) AS max_step FROM step_checkpoints GROUP BY timeline_id
                ),
                head_steps AS (
                    SELECT timeline_id, MAX(max_step) AS head_step
                    FROM all_steps
                    GROUP BY timeline_id
                )
                SELECT t.timeline_id, t.parent_id, t.fork_step, t.metadata,
                       COALESCE(h.head_step, t.fork_step, 0) AS head_step
                FROM timelines t
                LEFT JOIN head_steps h ON h.timeline_id = t.timeline_id
                """
            )
            return [
                TimelineInfo(
                    timeline_id=row["timeline_id"],
                    parent_id=row["parent_id"],
                    fork_step=row["fork_step"],
                    head_step=int(row["head_step"]),
                    metadata=json.loads(row["metadata"]),
                )
                for row in cursor.fetchall()
            ]

    def rollback_to(self, step: int, *, timeline_id: str, force: bool = False) -> int:
        """Atomic rollback: deletes all revisions, exemplars, and episodes at > step for this timeline."""
        with self.transaction():
            if not force:
                conflict = self.conn.execute(
                    "SELECT timeline_id, fork_step FROM timelines WHERE parent_id = ? AND fork_step > ?",
                    (timeline_id, step),
                ).fetchone()
                if conflict:
                    raise TimelineBranchConflictError(
                        f"Cannot roll back '{timeline_id}' to step {step}: child branch "
                        f"'{conflict['timeline_id']}' forked at step {conflict['fork_step']}. "
                        "Pass force=True to override."
                    )
            c1 = self.conn.execute(
                "DELETE FROM graph_revisions WHERE timeline_id = ? AND step > ?",
                (timeline_id, step),
            )
            c2 = self.conn.execute(
                "DELETE FROM voice_exemplars WHERE timeline_id = ? AND step > ?",
                (timeline_id, step),
            )
            c3 = self.conn.execute(
                "DELETE FROM episodes WHERE timeline_id = ? AND step > ?",
                (timeline_id, step),
            )
            c4 = self.conn.execute(
                "DELETE FROM episode_epistemics WHERE timeline_id = ? AND step > ?",
                (timeline_id, step),
            )
            c5 = self.conn.execute(
                "DELETE FROM step_checkpoints WHERE timeline_id = ? AND step > ?",
                (timeline_id, step),
            )
            return c1.rowcount + c2.rowcount + c3.rowcount + c4.rowcount + c5.rowcount

    def delete_timeline(self, timeline_id: str, *, cascade: bool = False) -> int:
        """Deletes a timeline branch and all of its owned records."""
        if timeline_id == "main":
            raise ValueError("Cannot delete the primary 'main' timeline.")
        with self.transaction():
            exists = self.conn.execute(
                "SELECT 1 FROM timelines WHERE timeline_id = ?",
                (timeline_id,),
            ).fetchone()
            if not exists:
                raise ValueError(f"Timeline '{timeline_id}' does not exist.")

            direct_children = [
                row["timeline_id"]
                for row in self.conn.execute(
                    "SELECT timeline_id FROM timelines WHERE parent_id = ?",
                    (timeline_id,),
                ).fetchall()
            ]
            if direct_children and not cascade:
                raise TimelineBranchConflictError(
                    f"Cannot delete timeline '{timeline_id}': child branches exist "
                    f"({', '.join(direct_children)}). Pass cascade=True to delete descendants."
                )

            # Collect subtree in BFS order, then delete bottom-up to satisfy parent_id FK
            to_delete: list[str] = [timeline_id]
            queue: list[str] = list(direct_children)
            while queue:
                cid = queue.pop(0)
                to_delete.append(cid)
                sub_children = [
                    row["timeline_id"]
                    for row in self.conn.execute(
                        "SELECT timeline_id FROM timelines WHERE parent_id = ?",
                        (cid,),
                    ).fetchall()
                ]
                queue.extend(sub_children)

            deleted_rows = 0
            for tid in reversed(to_delete):
                for table in (
                    "graph_revisions",
                    "voice_exemplars",
                    "episodes",
                    "episode_epistemics",
                    "step_checkpoints",
                    "timelines",
                ):
                    cur = self.conn.execute(
                        f"DELETE FROM {table} WHERE timeline_id = ?",
                        (tid,),
                    )
                    deleted_rows += cur.rowcount
            return deleted_rows

    # --- Step Checkpoints (Resumability) ---

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

    # --- Lineage Queries ---

    def get_active_revisions(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        record_type: str | None = None,
        *,
        subject_key: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        relation_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns the winning (latest) non-tombstone revision for every subject_key along the lineage."""
        filters: list[str] = []
        params: dict[str, Any] = {"timeline_id": timeline_id, "cutoff_step": cutoff_step}
        if record_type:
            filters.append("r.record_type = :record_type")
            params["record_type"] = record_type
        if subject_key is not None:
            filters.append("r.subject_key = :subject_key")
            params["subject_key"] = subject_key
        if source_id is not None:
            filters.append("r.source_id = :source_id")
            params["source_id"] = source_id
        if target_id is not None:
            filters.append("r.target_id = :target_id")
            params["target_id"] = target_id
        if relation_type is not None:
            if relation_type == "stance":
                filters.append("(r.relation_type = :relation_type OR r.relation_type IS NULL)")
            else:
                filters.append("r.relation_type = :relation_type")
            params["relation_type"] = relation_type

        extra_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = LINEAGE_ACTIVE_REVISIONS_CTE.replace("/*EXTRA_FILTERS*/", extra_sql)

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        results = []
        for row in rows:
            payload_dict = json.loads(row["payload"])
            rev_offset = 0
            if isinstance(payload_dict, dict):
                raw_offset = payload_dict.pop("_revision_count_offset", 0)
                if isinstance(raw_offset, (int, float)):
                    rev_offset = int(raw_offset)
            prior_payload = None
            prior_step = None
            if row["prior_payload"] is not None and not row["prior_is_tombstone"]:
                prior_payload = json.loads(row["prior_payload"])
                if isinstance(prior_payload, dict):
                    prior_payload.pop("_revision_count_offset", None)
                prior_step = row["prior_step"]
            results.append(
                {
                    "record_type": row["record_type"],
                    "subject_key": row["subject_key"],
                    "relation_type": row["relation_type"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "valid_from_step": row["valid_from_step"],
                    "payload": payload_dict,
                    "revision_count": row["revision_count"] + rev_offset,
                    "prior_payload": prior_payload,
                    "prior_step": prior_step,
                }
            )
        return results

    @staticmethod
    def edge_subject_key(source_id: str, target_id: str, relation_type: str = "stance") -> str:
        return f"{source_id}:{relation_type}->{target_id}"

    def _row_to_entity(self, rev: dict[str, Any]) -> EntityRecord:
        p = rev["payload"]
        prior_p = rev.get("prior_payload")
        prior_state = prior_p.get("state") if isinstance(prior_p, dict) else None
        return EntityRecord(
            entity_id=rev["subject_key"],
            kind=p.get("kind", "character"),
            canonical_name=p.get("canonical_name", rev["subject_key"]),
            base_attributes=p.get("base_attributes", {}),
            state=p.get("state", {}),
            private_state=p.get("private_state", {}),
            valid_from_step=rev["valid_from_step"],
            caused_by_episode_id=p.get("caused_by_episode_id"),
            reason=p.get("reason"),
            prior_state=prior_state,
            prior_step=rev.get("prior_step") if prior_state is not None else None,
            revision_count=rev.get("revision_count", 1),
        )

    def _row_to_edge(self, rev: dict[str, Any], default_relation: str = "stance") -> DirectedEdgeRecord:
        state, caused_by, reason = _unpack_edge_payload(rev["payload"])
        prior_p = rev.get("prior_payload")
        prior_state = None
        if isinstance(prior_p, dict):
            prior_state, _, _ = _unpack_edge_payload(prior_p)
        return DirectedEdgeRecord(
            source_id=rev["source_id"],
            target_id=rev["target_id"],
            relation_type=rev.get("relation_type") or default_relation,
            state=state,
            valid_from_step=rev["valid_from_step"],
            caused_by_episode_id=caused_by,
            reason=reason,
            prior_state=prior_state,
            prior_step=rev.get("prior_step") if prior_state is not None else None,
            revision_count=rev.get("revision_count", 1),
        )

    def get_active_entity(
        self,
        entity_id: str,
        *,
        timeline_id: str,
        cutoff_step: int,
    ) -> EntityRecord | None:
        revisions = self.get_active_revisions(
            timeline_id, cutoff_step, record_type="entity", subject_key=entity_id
        )
        if revisions:
            return self._row_to_entity(revisions[0])
        return None

    def get_active_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        relation_type: str = "stance",
        timeline_id: str,
        cutoff_step: int,
    ) -> DirectedEdgeRecord | None:
        subject_key = self.edge_subject_key(source_id, target_id, relation_type)
        revisions = self.get_active_revisions(
            timeline_id, cutoff_step, record_type="edge", subject_key=subject_key
        )
        if revisions:
            return self._row_to_edge(revisions[0], default_relation=relation_type)
        if relation_type == "stance":
            legacy_key = f"{source_id}->{target_id}"
            legacy_revs = self.get_active_revisions(
                timeline_id, cutoff_step, record_type="edge", subject_key=legacy_key
            )
            if legacy_revs:
                return self._row_to_edge(legacy_revs[0], default_relation=relation_type)
        return None

    def get_active_edges_between(
        self,
        source_id: str,
        target_id: str,
        *,
        timeline_id: str,
        cutoff_step: int,
    ) -> list[DirectedEdgeRecord]:
        revisions = self.get_active_revisions(
            timeline_id,
            cutoff_step,
            record_type="edge",
            source_id=source_id,
            target_id=target_id,
        )
        return [self._row_to_edge(rev) for rev in revisions]

    def get_all_active_entities(
        self,
        timeline_id: str,
        cutoff_step: int | None,
    ) -> list[EntityRecord]:
        revs = self.get_active_revisions(timeline_id, cutoff_step, record_type="entity")
        return [self._row_to_entity(r) for r in revs]

    def get_all_active_edges(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        *,
        relation_type: str | None = None,
    ) -> list[DirectedEdgeRecord]:
        revs = self.get_active_revisions(
            timeline_id,
            cutoff_step,
            record_type="edge",
            relation_type=relation_type,
        )
        return [
            self._row_to_edge(r, default_relation=r.get("relation_type") or "stance")
            for r in revs
        ]

    def _row_to_alias(self, rev: dict[str, Any]) -> AliasRecord:
        p = rev["payload"]
        return AliasRecord(
            alias=p.get("alias") or rev["subject_key"],
            entity_id=rev["source_id"],
            kind=p.get("kind", "epithet"),
            lang=p.get("lang"),
            known_to=p.get("known_to"),
            persona_name=p.get("persona_name"),
            valid_from_step=rev["valid_from_step"],
        )

    def get_all_active_aliases(
        self,
        timeline_id: str,
        cutoff_step: int | None,
    ) -> list[AliasRecord]:
        revs = self.get_active_revisions(timeline_id, cutoff_step, record_type="alias")
        return [self._row_to_alias(r) for r in revs]

    def get_active_graph_state(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        *,
        edge_types: Sequence[str] | None = None,
    ) -> tuple[list[EntityRecord], list[AliasRecord], list[DirectedEdgeRecord]]:
        """Single-pass CTE retrieval of all active entities, aliases, and edges."""
        revs = self.get_active_revisions(timeline_id, cutoff_step)
        entities: list[EntityRecord] = []
        aliases: list[AliasRecord] = []
        edges: list[DirectedEdgeRecord] = []
        allowed_edge_types = set(edge_types) if edge_types is not None else None
        for rev in revs:
            rtype = rev["record_type"]
            if rtype == "entity":
                entities.append(self._row_to_entity(rev))
            elif rtype == "alias":
                aliases.append(self._row_to_alias(rev))
            elif rtype == "edge":
                rel = rev.get("relation_type") or "stance"
                if allowed_edge_types is None or rel in allowed_edge_types:
                    edges.append(self._row_to_edge(rev, default_relation=rel))
        return entities, aliases, edges

    # --- Write-Time State Materialization (Resolve-on-Write) ---

    @staticmethod
    def alias_subject_key(entity_id: str, alias: str) -> str:
        return f"{entity_id}:{alias.strip().lower()}"

    def _insert_alias_revision(
        self,
        *,
        timeline_id: str,
        step: int,
        entity_id: str,
        alias: str | AliasRecord,
        default_kind: str = "epithet",
    ) -> None:
        """Inserts an alias revision preserving kind, lang, known_to, and persona_name."""
        if isinstance(alias, AliasRecord):
            alias_str = alias.alias.strip()
            target_entity_id = alias.entity_id or entity_id
            alias_payload: dict[str, Any] = {"alias": alias_str, "kind": alias.kind}
            if alias.lang is not None and alias.lang.strip():
                alias_payload["lang"] = alias.lang.strip().lower()
            if alias.known_to is not None:
                alias_payload["known_to"] = alias.known_to
            if alias.persona_name is not None:
                alias_payload["persona_name"] = alias.persona_name
        else:
            alias_str = alias.strip()
            target_entity_id = entity_id
            alias_payload = {"alias": alias_str, "kind": default_kind}

        subject_key = self.alias_subject_key(target_entity_id, alias_str)
        existing_revs = self.get_active_revisions(
            timeline_id, step, record_type="alias", subject_key=subject_key
        )
        if existing_revs:
            existing_payload = existing_revs[0]["payload"]
            # Protect active disguise aliases from accidental downgrade to 'epithet' when an LLM re-emits the alias
            if (
                existing_payload.get("kind") == "disguise"
                and alias_payload.get("kind") == "epithet"
                and alias_payload.get("known_to") is None
            ):
                return
            if existing_payload == alias_payload:
                return

        self.conn.execute(
            """
            INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
            VALUES (?, ?, 'alias', ?, ?, NULL, 0, ?)
            """,
            (timeline_id, step, subject_key, target_entity_id, json.dumps(alias_payload)),
        )

    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str | AliasRecord] = (),
        step: int = 0,
        timeline_id: str = "main",
    ) -> None:
        payload: dict[str, Any] = {
            "kind": entity.kind,
            "canonical_name": entity.canonical_name,
            "base_attributes": entity.base_attributes,
            "state": entity.state,
        }
        if entity.private_state:
            payload["private_state"] = entity.private_state
        if entity.caused_by_episode_id is not None:
            payload["caused_by_episode_id"] = entity.caused_by_episode_id
        if entity.reason is not None:
            payload["reason"] = entity.reason

        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                VALUES (?, ?, 'entity', ?, ?, NULL, 0, ?)
                """,
                (timeline_id, step, entity.entity_id, entity.entity_id, json.dumps(payload)),
            )
            for alias in aliases:
                self._insert_alias_revision(
                    timeline_id=timeline_id,
                    step=step,
                    entity_id=entity.entity_id,
                    alias=alias,
                    default_kind="permanent",
                )

    def apply_delta(
        self,
        step: int,
        delta: StepDelta,
        *,
        timeline_id: str = "main",
        episode_embeddings: dict[str, bytes] | None = None,
    ) -> list[EpisodeRecord]:
        """Applies StepDelta by resolving current active state and writing complete materialized revisions."""
        committed_episodes: list[EpisodeRecord] = []

        with self.transaction():
            # 0. Process episodes first so their IDs exist if referenced by deltas in the same step
            if delta.episodes:
                to_insert: list[tuple[EpisodeRecord, bytes]] = []
                for ep in delta.episodes:
                    ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
                    item = ep.model_copy(update={"episode_id": ep_id, "step": step, "timeline_id": timeline_id})
                    emb = (episode_embeddings or {}).get(ep_id) or (episode_embeddings or {}).get(ep.episode_id, b"")
                    to_insert.append((item, emb))
                    committed_episodes.append(item)
                self._insert_episodes_in_tx(to_insert, timeline_id=timeline_id)

            # 0b. Process epistemic inform deltas
            for inf in delta.inform:
                self.conn.execute(
                    """
                    INSERT INTO episode_epistemics (timeline_id, step, episode_id, entity_id, epistemic, believed_summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timeline_id, step, inf.episode_id, inf.entity_id, inf.epistemic, inf.believed_summary),
                )

            # 1. Process entity deltas
            for ed in delta.entities:
                if ed.remove:
                    self.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                        VALUES (?, ?, 'entity', ?, ?, NULL, 1, '{}')
                        """,
                        (timeline_id, step, ed.entity_id, ed.entity_id),
                    )
                    continue

                active = self.get_active_entity(ed.entity_id, timeline_id=timeline_id, cutoff_step=step)
                base_attrs = dict(active.base_attributes) if active else {}
                base_attrs.update(ed.base_attributes)

                current_state = dict(active.state) if (active and ed.merge_state) else {}
                current_state.update(ed.state)
                current_state = {k: v for k, v in current_state.items() if v is not None}

                current_private = dict(active.private_state) if (active and ed.merge_state) else {}
                current_private.update(ed.private_state)
                current_private = {k: v for k, v in current_private.items() if v is not None}

                canonical_name = ed.canonical_name or (active.canonical_name if active else ed.entity_id)
                kind = ed.kind or (active.kind if active else "character")

                has_state_mutation = bool(ed.state or ed.private_state or not ed.merge_state)
                caused_by = ed.caused_by_episode_id
                if caused_by is None and active and ed.merge_state and not has_state_mutation:
                    caused_by = active.caused_by_episode_id
                reason = ed.reason
                if reason is None and active and ed.merge_state and not has_state_mutation:
                    reason = active.reason

                is_noop_entity_state = (
                    active is not None
                    and kind == active.kind
                    and canonical_name == active.canonical_name
                    and base_attrs == active.base_attributes
                    and current_state == active.state
                    and current_private == active.private_state
                )
                if not is_noop_entity_state:
                    payload: dict[str, Any] = {
                        "kind": kind,
                        "canonical_name": canonical_name,
                        "base_attributes": base_attrs,
                        "state": current_state,
                    }
                    if current_private:
                        payload["private_state"] = current_private
                    if caused_by is not None:
                        payload["caused_by_episode_id"] = caused_by
                    if reason is not None:
                        payload["reason"] = reason

                    self.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                        VALUES (?, ?, 'entity', ?, ?, NULL, 0, ?)
                        """,
                        (timeline_id, step, ed.entity_id, ed.entity_id, json.dumps(payload)),
                    )

                # Add new aliases
                for alias in ed.add_aliases:
                    self._insert_alias_revision(
                        timeline_id=timeline_id,
                        step=step,
                        entity_id=ed.entity_id,
                        alias=alias,
                        default_kind="epithet",
                    )

                # Remove aliases (tombstone scoped to entity_id)
                for alias in ed.remove_aliases:
                    alias_key = self.alias_subject_key(ed.entity_id, alias)
                    self.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                        VALUES (?, ?, 'alias', ?, ?, NULL, 1, '{}')
                        """,
                        (timeline_id, step, alias_key, ed.entity_id),
                    )

            # 2. Process edge deltas
            for rd in delta.edges:
                subject_key = self.edge_subject_key(rd.source_id, rd.target_id, rd.relation_type)
                if rd.remove:
                    self.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, relation_type, source_id, target_id, is_tombstone, payload)
                        VALUES (?, ?, 'edge', ?, ?, ?, ?, 1, '{}')
                        """,
                        (timeline_id, step, subject_key, rd.relation_type, rd.source_id, rd.target_id),
                    )
                    continue

                active_edge = self.get_active_edge(
                    rd.source_id,
                    rd.target_id,
                    relation_type=rd.relation_type,
                    timeline_id=timeline_id,
                    cutoff_step=step,
                )
                edge_state = dict(active_edge.state) if (active_edge and rd.merge) else {}
                edge_state.update(rd.state)
                edge_state = {k: v for k, v in edge_state.items() if v is not None}

                has_edge_mutation = bool(rd.state or not rd.merge)
                caused_by = rd.caused_by_episode_id
                if caused_by is None and active_edge and rd.merge and not has_edge_mutation:
                    caused_by = active_edge.caused_by_episode_id
                reason = rd.reason
                if reason is None and active_edge and rd.merge and not has_edge_mutation:
                    reason = active_edge.reason

                if active_edge is not None and edge_state == active_edge.state:
                    continue

                edge_payload = dict(edge_state)
                if caused_by is not None or reason is not None:
                    edge_payload["_provenance"] = {
                        "caused_by_episode_id": caused_by,
                        "reason": reason,
                    }

                self.conn.execute(
                    """
                    INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, relation_type, source_id, target_id, is_tombstone, payload)
                    VALUES (?, ?, 'edge', ?, ?, ?, ?, 0, ?)
                    """,
                    (timeline_id, step, subject_key, rd.relation_type, rd.source_id, rd.target_id, json.dumps(edge_payload)),
                )

        return committed_episodes

    # --- Tier 6: Episodic Event Ledger Storage & Retrieval ---

    def _insert_episodes_in_tx(
        self,
        episodes: Sequence[tuple[EpisodeRecord, bytes]],
        *,
        timeline_id: str = "main",
    ) -> None:
        for ep, emb_bytes in episodes:
            ep_id = ep.episode_id or f"ep_{ep.step}_{uuid.uuid4().hex[:8]}"
            self.conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                (episode_id, timeline_id, step, predicate, summary, location_id, salience, participants, witnesses, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ep_id,
                    timeline_id,
                    ep.step,
                    ep.predicate,
                    ep.summary,
                    ep.location_id,
                    ep.salience,
                    json.dumps(ep.participants),
                    json.dumps(ep.witnesses),
                    json.dumps(_pack_episode_metadata(ep)),
                    emb_bytes,
                ),
            )
            # Determine initial epistemic awareness at ep.step
            initial_known: dict[str, str] = {}
            for ent_id in ep.participants:
                initial_known[ent_id] = "direct"
            for wit_id in ep.witnesses:
                initial_known[wit_id] = "direct"
            for ent_id, mode in ep.known_by.items():
                initial_known[ent_id] = mode

            for ent_id, raw_mode in initial_known.items():
                self.conn.execute(
                    """
                    INSERT INTO episode_epistemics (timeline_id, step, episode_id, entity_id, epistemic, believed_summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timeline_id, ep.step, ep_id, ent_id, raw_mode, ep.believed_summaries.get(ent_id)),
                )

    def insert_episodes(
        self,
        episodes: Sequence[tuple[EpisodeRecord, bytes]],
        *,
        timeline_id: str = "main",
    ) -> None:
        with self.transaction():
            self._insert_episodes_in_tx(episodes, timeline_id=timeline_id)

    def record_epistemic(
        self,
        inform: InformDelta,
        *,
        step: int,
        timeline_id: str = "main",
    ) -> None:
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO episode_epistemics (timeline_id, step, episode_id, entity_id, epistemic, believed_summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timeline_id, step, inform.episode_id, inform.entity_id, inform.epistemic, inform.believed_summary),
            )

    def delete_episodes(
        self,
        episode_ids: Sequence[str],
        *,
        timeline_id: str = "main",
    ) -> int:
        """Deletes specific episodes and their epistemic records on a timeline (used for consolidation)."""
        ids = [eid for eid in episode_ids if eid]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self.transaction():
            c1 = self.conn.execute(
                f"DELETE FROM episodes WHERE timeline_id = ? AND episode_id IN ({placeholders})",
                (timeline_id, *ids),
            )
            c2 = self.conn.execute(
                f"DELETE FROM episode_epistemics WHERE timeline_id = ? AND episode_id IN ({placeholders})",
                (timeline_id, *ids),
            )
            return c1.rowcount + c2.rowcount

    @staticmethod
    def _bind_in_clause(
        prefix: str, values: Sequence[str], params: dict[str, Any]
    ) -> str:
        """Binds a non-empty sequence of strings into named SQL placeholders."""
        placeholders: list[str] = []
        for idx, val in enumerate(values):
            key = f"{prefix}_{idx}"
            params[key] = val
            placeholders.append(f":{key}")
        return ", ".join(placeholders)

    @staticmethod
    def _build_fts_query(query_text: str, *, max_terms: int = 12) -> str:
        """Extracts safe FTS5 OR-joined terms from arbitrary narrative prose."""
        if not query_text:
            return ""
        tokens = re.findall(r"[A-Za-z0-9_]{2,}", query_text)
        seen: set[str] = set()
        terms: list[str] = []
        for tok in tokens:
            low = tok.lower()
            if low in seen:
                continue
            seen.add(low)
            terms.append(f'"{tok}"')
            if len(terms) >= max_terms:
                break
        return " OR ".join(terms)

    def get_known_episode_ids(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        pov_entity_ids: Sequence[str],
        *,
        pov_mode: Literal["any", "all"] = "any",
        episode_ids: Sequence[str] | None = None,
    ) -> set[str]:
        """Returns the subset of episode_ids known to pov_entity_ids under pov_mode."""
        pov_list = sorted({pid for pid in pov_entity_ids if pid})
        if not pov_list:
            return set()
        if episode_ids is not None and len(episode_ids) == 0:
            return set()

        params: dict[str, Any] = {"timeline_id": timeline_id, "cutoff_step": cutoff_step}
        pov_in = self._bind_in_clause("pov", pov_list, params)
        ep_filter = ""
        if episode_ids is not None:
            ep_list = sorted({eid for eid in episode_ids if eid})
            if not ep_list:
                return set()
            ep_in = self._bind_in_clause("ep", ep_list, params)
            ep_filter = f"AND ep.episode_id IN ({ep_in})"

        if pov_mode == "all":
            params["pov_count"] = len(pov_list)
            outer_select = (
                "SELECT episode_id FROM latest_epistemics "
                "WHERE ep_rn = 1 AND epistemic NOT IN ('forgotten', 'suppressed', 'deceived') "
                "GROUP BY episode_id HAVING COUNT(DISTINCT entity_id) = :pov_count"
            )
        else:
            outer_select = (
                "SELECT DISTINCT episode_id FROM latest_epistemics "
                "WHERE ep_rn = 1 AND epistemic NOT IN ('forgotten', 'suppressed', 'deceived')"
            )

        query = (
            LINEAGE_CTE_SQL
            + f""",
        latest_epistemics AS (
            SELECT ep.episode_id, ep.entity_id, ep.epistemic,
                   ROW_NUMBER() OVER (
                       PARTITION BY ep.episode_id, ep.entity_id
                       ORDER BY ep.step DESC, l.depth ASC, ep.id DESC
                   ) AS ep_rn
            FROM episode_epistemics ep
            JOIN lineage l
              ON ep.timeline_id = l.timeline_id
             AND (l.max_step IS NULL OR ep.step <= l.max_step)
            WHERE ep.entity_id IN ({pov_in})
            {ep_filter}
        )
        {outer_select};
        """
        )
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return {row["episode_id"] for row in rows}

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
    ) -> list[tuple[EpisodeRecord, bytes]]:
        """Retrieves episodes visible along the lineage with step <= cutoff_step, with resolved epistemic map."""
        if episode_ids is not None and len(episode_ids) == 0:
            return []

        params: dict[str, Any] = {"timeline_id": timeline_id, "cutoff_step": cutoff_step}
        ep_where_clauses: list[str] = []
        if episode_id is not None:
            params["episode_id"] = episode_id
            ep_where_clauses.append("e.episode_id = :episode_id")
        if episode_ids is not None:
            ep_list = sorted({eid for eid in episode_ids if eid})
            if not ep_list:
                return []
            ep_in = self._bind_in_clause("req_ep", ep_list, params)
            ep_where_clauses.append(f"e.episode_id IN ({ep_in})")

        ep_where_sql = ("WHERE " + " AND ".join(ep_where_clauses)) if ep_where_clauses else ""

        ctes: list[str] = []
        pov_list = sorted({pid for pid in (pov_entity_ids or ()) if pid})
        pov_filter_sql = ""
        if pov_list:
            pov_in = self._bind_in_clause("pov", pov_list, params)
            ctes.append(
                f"""latest_epistemics AS (
            SELECT ep.episode_id, ep.entity_id, ep.epistemic,
                   ROW_NUMBER() OVER (
                       PARTITION BY ep.episode_id, ep.entity_id
                       ORDER BY ep.step DESC, l.depth ASC, ep.id DESC
                   ) AS ep_rn
            FROM episode_epistemics ep
            JOIN lineage l
              ON ep.timeline_id = l.timeline_id
             AND (l.max_step IS NULL OR ep.step <= l.max_step)
            WHERE ep.entity_id IN ({pov_in})
        )"""
            )
            if pov_mode == "all":
                params["pov_count"] = len(pov_list)
                ctes.append(
                    """pov_visible AS (
            SELECT episode_id
            FROM latest_epistemics
            WHERE ep_rn = 1 AND epistemic NOT IN ('forgotten', 'suppressed')
            GROUP BY episode_id
            HAVING COUNT(DISTINCT entity_id) = :pov_count
        )"""
                )
            else:
                ctes.append(
                    """pov_visible AS (
            SELECT DISTINCT episode_id
            FROM latest_epistemics
            WHERE ep_rn = 1 AND epistemic NOT IN ('forgotten', 'suppressed')
        )"""
                )
            pov_filter_sql = "AND episode_id IN (SELECT episode_id FROM pov_visible)"

        fts_query = self._build_fts_query(query_text) if limit is not None else ""
        if fts_query:
            params["fts_query"] = fts_query
            ctes.append(
                """fts_matches AS (
            SELECT DISTINCT episode_id
            FROM episodes_fts
            WHERE episodes_fts MATCH :fts_query
        )"""
            )

        ctes.append(
            f"""deduped_episodes AS (
            SELECT e.episode_id, e.timeline_id, e.step, e.predicate, e.summary,
                   e.location_id, e.salience, e.participants, e.witnesses, e.metadata, e.embedding,
                   l.depth,
                   ROW_NUMBER() OVER (
                       PARTITION BY e.episode_id
                       ORDER BY l.depth ASC, e.step DESC
                   ) AS rn
            FROM episodes e
            JOIN lineage l
              ON e.timeline_id = l.timeline_id
             AND (l.max_step IS NULL OR e.step <= l.max_step)
            {ep_where_sql}
        )"""
        )

        if limit is not None:
            params["ep_limit"] = max(1, int(limit))
            order_terms: list[str] = []
            pinned_list = sorted({pid for pid in (pinned_episode_ids or ()) if pid})
            if pinned_list:
                pin_in = self._bind_in_clause("pin", pinned_list, params)
                order_terms.append(f"(CASE WHEN episode_id IN ({pin_in}) THEN 1 ELSE 0 END) DESC")

            cue_list = sorted({cid for cid in (cue_entity_ids or ()) if cid})
            match_terms: list[str] = []
            if cue_list:
                cue_in = self._bind_in_clause("cue", cue_list, params)
                match_terms.append(
                    f"(CASE WHEN location_id IN ({cue_in}) "
                    f"OR EXISTS (SELECT 1 FROM json_each(participants) WHERE key IN ({cue_in})) "
                    f"OR EXISTS (SELECT 1 FROM json_each(witnesses) WHERE value IN ({cue_in})) "
                    f"THEN 1 ELSE 0 END)"
                )
            if fts_query:
                match_terms.append(
                    "(CASE WHEN episode_id IN (SELECT episode_id FROM fts_matches) THEN 1 ELSE 0 END)"
                )
            if match_terms:
                order_terms.append(f"({' + '.join(match_terms)}) DESC")

            order_terms.extend(["salience DESC", "step DESC"])
            order_limit_sql = f"ORDER BY {', '.join(order_terms)} LIMIT :ep_limit"
        else:
            order_limit_sql = "ORDER BY depth ASC, step DESC"

        ep_query = (
            LINEAGE_CTE_SQL
            + ",\n        "
            + ",\n        ".join(ctes)
            + f"""
        SELECT episode_id, timeline_id, step, predicate, summary,
               location_id, salience, participants, witnesses, metadata, embedding
        FROM deduped_episodes
        WHERE rn = 1 {pov_filter_sql}
        {order_limit_sql};
        """
        )

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(ep_query, params)
            ep_rows = cursor.fetchall()
            if not ep_rows:
                return []

            epistemic_params: dict[str, Any] = {
                "timeline_id": timeline_id,
                "cutoff_step": cutoff_step,
            }
            if episode_id is not None:
                epistemic_params["episode_id"] = episode_id
                epistemic_filter = "WHERE ep.episode_id = :episode_id"
            elif limit is not None or episode_ids is not None:
                selected_ids = [row["episode_id"] for row in ep_rows]
                sel_in = self._bind_in_clause("sel_ep", selected_ids, epistemic_params)
                epistemic_filter = f"WHERE ep.episode_id IN ({sel_in})"
            else:
                epistemic_filter = ""

            epistemic_query = (
                LINEAGE_CTE_SQL
                + f"""
            SELECT ep.episode_id, ep.entity_id, ep.epistemic, ep.believed_summary, ep.step, l.depth
            FROM episode_epistemics ep
            JOIN lineage l
              ON ep.timeline_id = l.timeline_id
             AND (l.max_step IS NULL OR ep.step <= l.max_step)
            {epistemic_filter}
            ORDER BY ep.step ASC, l.depth DESC, ep.id ASC;
            """
            )
            cursor.execute(epistemic_query, epistemic_params)
            epistemic_rows = cursor.fetchall()

        known_map: dict[str, dict[str, Literal["direct", "told", "deceived"]]] = {}
        believed_map: dict[str, dict[str, str]] = {}
        for row in epistemic_rows:
            ep_id = row["episode_id"]
            ent_id = row["entity_id"]
            raw_mode = row["epistemic"]
            if raw_mode in ("forgotten", "suppressed"):
                if ep_id in known_map:
                    known_map[ep_id].pop(ent_id, None)
                if ep_id in believed_map:
                    believed_map[ep_id].pop(ent_id, None)
                continue
            if raw_mode == "direct":
                mode: Literal["direct", "told", "deceived"] = "direct"
            elif raw_mode == "deceived":
                mode = "deceived"
            else:
                mode = "told"
            known_map.setdefault(ep_id, {})[ent_id] = mode
            if mode == "deceived":
                if row["believed_summary"]:
                    believed_map.setdefault(ep_id, {})[ent_id] = row["believed_summary"]
            elif ep_id in believed_map and ent_id in believed_map[ep_id]:
                believed_map[ep_id].pop(ent_id, None)

        items: list[tuple[EpisodeRecord, bytes]] = []
        seen_ids: set[str] = set()
        for row in ep_rows:
            ep_id = row["episode_id"]
            if ep_id in seen_ids:
                continue
            seen_ids.add(ep_id)
            meta, story_time, chapter_id, scene_id, consolidated_from = _unpack_episode_metadata(
                json.loads(row["metadata"])
            )
            rec = EpisodeRecord(
                episode_id=ep_id,
                timeline_id=row["timeline_id"],
                step=row["step"],
                predicate=row["predicate"],
                summary=row["summary"],
                location_id=row["location_id"],
                story_time=story_time,
                chapter_id=chapter_id,
                scene_id=scene_id,
                consolidated_from=consolidated_from,
                salience=row["salience"],
                participants=json.loads(row["participants"]),
                witnesses=json.loads(row["witnesses"]),
                known_by=known_map.get(ep_id, {}),
                believed_summaries=believed_map.get(ep_id, {}),
                metadata=meta,
            )
            items.append((rec, bytes(row["embedding"])))
        items.sort(key=lambda item: item[0].step, reverse=True)
        return items

    def get_episode(
        self,
        episode_id: str,
        *,
        timeline_id: str = "main",
        cutoff_step: int | None = None,
    ) -> EpisodeRecord | None:
        items = self.get_lineage_episodes(timeline_id, cutoff_step, episode_id=episode_id)
        return items[0][0] if items else None

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
    ) -> list[EpisodeRecord]:
        """Structured episodic query supporting participant co-occurrence, epistemic filters, and step intervals."""
        all_eps = [ep for ep, _ in self.get_lineage_episodes(timeline_id, cutoff_step)]
        results: list[EpisodeRecord] = []
        part_set = set(participants)
        known_set = set(known_to)

        for ep in all_eps:
            if ep.salience < min_salience:
                continue
            if predicate is not None and ep.predicate != predicate:
                continue
            if location_id is not None and ep.location_id != location_id:
                continue
            if step_range is not None and not (step_range[0] <= ep.step <= step_range[1]):
                continue
            if part_set:
                ep_entities = set(ep.participants.keys()) | set(ep.witnesses)
                if match_all_participants:
                    if not part_set.issubset(ep_entities):
                        continue
                else:
                    if not (part_set & ep_entities):
                        continue
            if known_set and not (known_set & set(ep.known_by.keys())):
                continue
            results.append(ep)

        results.sort(key=lambda e: e.step)
        return results

    # --- Voice Exemplars Storage ---

    def insert_voice_exemplars(
        self,
        exemplars: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        timeline_id: str = "main",
    ) -> None:
        with self.transaction():
            for ex, emb_bytes in exemplars:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO voice_exemplars
                    (exemplar_id, timeline_id, step, speaker_id, addressee_id, text, source_text, metadata, quality_score, distinctiveness_score, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ex.exemplar_id,
                        timeline_id,
                        ex.step,
                        ex.speaker_id,
                        ex.addressee_id,
                        ex.text,
                        ex.source_text,
                        json.dumps(ex.metadata),
                        ex.quality_score,
                        ex.distinctiveness_score,
                        emb_bytes,
                    ),
                )

    def get_lineage_voice_exemplars(
        self,
        timeline_id: str,
        cutoff_step: int | None,
        speaker_id: str,
    ) -> list[tuple[VoiceExemplar, bytes]]:
        """Retrieves voice exemplars for speaker_id visible along the lineage with step <= cutoff_step."""
        query = (
            LINEAGE_CTE_SQL
            + """
        SELECT v.exemplar_id, v.timeline_id, v.step, v.speaker_id, v.addressee_id,
               v.text, v.source_text, v.metadata, v.quality_score, v.distinctiveness_score, v.embedding
        FROM voice_exemplars v
        JOIN lineage l
          ON v.timeline_id = l.timeline_id
         AND (l.max_step IS NULL OR v.step <= l.max_step)
        WHERE v.speaker_id = :speaker_id
        ORDER BY l.depth ASC, v.step DESC;
        """
        )
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, {"timeline_id": timeline_id, "cutoff_step": cutoff_step, "speaker_id": speaker_id})
            rows = cursor.fetchall()
        items: list[tuple[VoiceExemplar, bytes]] = []
        seen_ids: set[str] = set()
        for row in rows:
            ex_id = row["exemplar_id"]
            if ex_id in seen_ids:
                continue
            seen_ids.add(ex_id)
            ex = VoiceExemplar(
                exemplar_id=ex_id,
                timeline_id=row["timeline_id"],
                step=row["step"],
                speaker_id=row["speaker_id"],
                addressee_id=row["addressee_id"],
                text=row["text"],
                source_text=row["source_text"],
                metadata=json.loads(row["metadata"]),
                quality_score=row["quality_score"],
                distinctiveness_score=row["distinctiveness_score"],
            )
            items.append((ex, bytes(row["embedding"])))
        items.sort(key=lambda item: item[0].step, reverse=True)
        return items

    # --- History & Branch Diffing ---

    def get_history(
        self,
        subject_key: str,
        *,
        timeline_id: str,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns the full chronological interval history of a subject_key along the lineage."""
        type_filter = "AND r.record_type = :record_type" if record_type is not None else ""
        query = (
            LINEAGE_CTE_SQL
            + f""",
        visible_revs AS (
            SELECT
                r.*,
                ROW_NUMBER() OVER (
                    PARTITION BY r.record_type, r.step
                    ORDER BY l.depth ASC, r.revision_id DESC
                ) AS step_rn
            FROM graph_revisions r
            JOIN lineage l
              ON r.timeline_id = l.timeline_id
             AND (l.max_step IS NULL OR r.step <= l.max_step)
            WHERE r.subject_key = :subject_key
            {type_filter}
        ),
        projected AS (
            SELECT
                revision_id,
                timeline_id,
                record_type,
                step AS valid_from_step,
                LEAD(step - 1) OVER (
                    PARTITION BY record_type
                    ORDER BY step ASC, revision_id ASC
                ) AS valid_to_step,
                is_tombstone,
                payload
            FROM visible_revs
            WHERE step_rn = 1
        )
        SELECT * FROM projected ORDER BY valid_from_step ASC, revision_id ASC;
        """
        )
        params: dict[str, Any] = {
            "timeline_id": timeline_id,
            "cutoff_step": None,
            "subject_key": subject_key,
        }
        if record_type is not None:
            params["record_type"] = record_type
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            is_tombstone = bool(r["is_tombstone"])
            valid_to = r["valid_to_step"]
            if is_tombstone:
                valid_to = r["valid_from_step"]
            results.append(
                {
                    "revision_id": r["revision_id"],
                    "timeline_id": r["timeline_id"],
                    "record_type": r["record_type"],
                    "valid_from_step": r["valid_from_step"],
                    "valid_to_step": valid_to,
                    "is_tombstone": is_tombstone,
                    "payload": json.loads(r["payload"]),
                }
            )
        return results

    def diff_timelines(
        self,
        timeline_a: str,
        timeline_b: str,
        *,
        at_step: int,
    ) -> dict[str, Any]:
        """Compares active world states between two timelines at at_step."""
        revs_a = {r["subject_key"]: r for r in self.get_active_revisions(timeline_a, at_step)}
        revs_b = {r["subject_key"]: r for r in self.get_active_revisions(timeline_b, at_step)}

        all_keys = set(revs_a.keys()) | set(revs_b.keys())
        diff: dict[str, Any] = {
            "only_in_a": [],
            "only_in_b": [],
            "changed": [],
            "identical": [],
        }

        for k in sorted(all_keys):
            in_a = revs_a.get(k)
            in_b = revs_b.get(k)
            if in_a and not in_b:
                diff["only_in_a"].append(k)
            elif in_b and not in_a:
                diff["only_in_b"].append(k)
            elif in_a and in_b:
                if in_a["payload"] == in_b["payload"]:
                    diff["identical"].append(k)
                else:
                    diff["changed"].append(
                        {
                            "subject_key": k,
                            "timeline_a": in_a["payload"],
                            "timeline_b": in_b["payload"],
                        }
                    )
        return diff

    def export_snapshot(
        self,
        target_path: str | Path,
        *,
        at_step: int,
        timeline_id: str = "main",
    ) -> Path:
        """Exports a flattened, standalone SQLite save-slot representing active state and episodes at at_step."""
        path = Path(target_path)
        if path.exists():
            path.unlink()

        export_store = CausalStore(path)
        with self._lock:
            active_revs = self.get_active_revisions(timeline_id, at_step)
            lineage_episodes = self.get_lineage_episodes(timeline_id, at_step)
            lineage_checkpoints = self.list_step_checkpoints(
                timeline_id=timeline_id, cutoff_step=at_step
            )
            epistemic_query = (
                LINEAGE_CTE_SQL
                + """,
        ranked_epistemics AS (
            SELECT
                ep.step,
                ep.episode_id,
                ep.entity_id,
                ep.epistemic,
                ep.believed_summary,
                ep.id,
                ROW_NUMBER() OVER (
                    PARTITION BY ep.episode_id, ep.entity_id, ep.step
                    ORDER BY l.depth ASC, ep.id DESC
                ) AS rn
            FROM episode_epistemics ep
            JOIN lineage l
              ON ep.timeline_id = l.timeline_id
             AND (l.max_step IS NULL OR ep.step <= l.max_step)
        )
        SELECT step, episode_id, entity_id, epistemic, believed_summary
        FROM ranked_epistemics
        WHERE rn = 1
        ORDER BY step ASC, id ASC;
        """
            )
            epistemic_rows = self.conn.execute(
                epistemic_query, {"timeline_id": timeline_id, "cutoff_step": at_step}
            ).fetchall()

            voice_query = (
                LINEAGE_CTE_SQL
                + """
        SELECT v.exemplar_id, v.step, v.speaker_id, v.addressee_id,
               v.text, v.source_text, v.metadata, v.quality_score, v.distinctiveness_score, v.embedding
        FROM voice_exemplars v
        JOIN lineage l
          ON v.timeline_id = l.timeline_id
         AND (l.max_step IS NULL OR v.step <= l.max_step)
        ORDER BY l.depth ASC, v.step ASC;
        """
            )
            voice_rows = self.conn.execute(
                voice_query, {"timeline_id": timeline_id, "cutoff_step": at_step}
            ).fetchall()
        lineage_exemplars: list[tuple[VoiceExemplar, bytes]] = []
        seen_exemplar_ids: set[str] = set()
        for row in voice_rows:
            ex_id = row["exemplar_id"]
            if ex_id in seen_exemplar_ids:
                continue
            seen_exemplar_ids.add(ex_id)
            lineage_exemplars.append(
                (
                    VoiceExemplar(
                        exemplar_id=ex_id,
                        timeline_id="main",
                        step=row["step"],
                        speaker_id=row["speaker_id"],
                        addressee_id=row["addressee_id"],
                        text=row["text"],
                        source_text=row["source_text"],
                        metadata=json.loads(row["metadata"]),
                        quality_score=row["quality_score"],
                        distinctiveness_score=row["distinctiveness_score"],
                    ),
                    bytes(row["embedding"]),
                )
            )

        with export_store.conn:
            for r in active_revs:
                has_prior = r.get("prior_payload") is not None and r.get("prior_step") is not None
                exported_rows = 2 if has_prior else 1
                rev_offset = max(0, int(r.get("revision_count", 1)) - exported_rows)
                payload_to_export = dict(r["payload"])
                if rev_offset > 0:
                    payload_to_export["_revision_count_offset"] = rev_offset

                if has_prior:
                    export_store.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, relation_type, source_id, target_id, is_tombstone, payload)
                        VALUES ('main', ?, ?, ?, ?, ?, ?, 0, ?)
                        """,
                        (
                            r["prior_step"],
                            r["record_type"],
                            r["subject_key"],
                            r.get("relation_type"),
                            r["source_id"],
                            r["target_id"],
                            json.dumps(r["prior_payload"]),
                        ),
                    )
                export_store.conn.execute(
                    """
                    INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, relation_type, source_id, target_id, is_tombstone, payload)
                    VALUES ('main', ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        r["valid_from_step"],
                        r["record_type"],
                        r["subject_key"],
                        r.get("relation_type"),
                        r["source_id"],
                        r["target_id"],
                        json.dumps(payload_to_export),
                    ),
                )
            for ep, emb_bytes in lineage_episodes:
                export_store.conn.execute(
                    """
                    INSERT OR REPLACE INTO episodes
                    (episode_id, timeline_id, step, predicate, summary, location_id, salience, participants, witnesses, metadata, embedding)
                    VALUES (?, 'main', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ep.episode_id,
                        ep.step,
                        ep.predicate,
                        ep.summary,
                        ep.location_id,
                        ep.salience,
                        json.dumps(ep.participants),
                        json.dumps(ep.witnesses),
                        json.dumps(_pack_episode_metadata(ep)),
                        emb_bytes,
                    ),
                )
            for row in epistemic_rows:
                export_store.conn.execute(
                    """
                    INSERT INTO episode_epistemics (timeline_id, step, episode_id, entity_id, epistemic, believed_summary)
                    VALUES ('main', ?, ?, ?, ?, ?)
                    """,
                    (
                        row["step"],
                        row["episode_id"],
                        row["entity_id"],
                        row["epistemic"],
                        row["believed_summary"],
                    ),
                )
            if lineage_exemplars:
                export_store.insert_voice_exemplars(lineage_exemplars, timeline_id="main")
            for cp in lineage_checkpoints:
                export_store.conn.execute(
                    """
                    INSERT OR REPLACE INTO step_checkpoints (timeline_id, step, content_hash, metadata_json)
                    VALUES ('main', ?, ?, ?)
                    """,
                    (cp.step, cp.content_hash, json.dumps(cp.metadata)),
                )
        export_store.close()
        return path
