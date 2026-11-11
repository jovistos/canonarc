from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from .models import (
    AliasRecord,
    DirectedEdgeRecord,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
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
    subject_key  TEXT NOT NULL, -- 'char_arthur' | 'char_arthur->char_merlin' | 'the old wizard'
    source_id    TEXT,          -- entity_id for 'entity'/'alias'; source_id for 'edge'
    target_id    TEXT,          -- target_id for 'edge'
    is_tombstone INTEGER NOT NULL DEFAULT 0, -- 1 if deleted/expired at this step
    payload      TEXT NOT NULL  -- Complete materialized JSON state at this step
);

CREATE INDEX IF NOT EXISTS idx_revisions_lookup
    ON graph_revisions(timeline_id, record_type, subject_key, step DESC, revision_id DESC);

CREATE TABLE IF NOT EXISTS voice_exemplars (
    exemplar_id           TEXT PRIMARY KEY,
    timeline_id           TEXT NOT NULL REFERENCES timelines(timeline_id),
    step                  INTEGER NOT NULL,
    speaker_id            TEXT NOT NULL,
    addressee_id          TEXT,
    text                  TEXT NOT NULL,
    source_text           TEXT NOT NULL DEFAULT '',
    metadata              TEXT NOT NULL DEFAULT '{}',
    quality_score         REAL NOT NULL,
    distinctiveness_score REAL NOT NULL,
    embedding             BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voice_lookup
    ON voice_exemplars(timeline_id, speaker_id, step DESC);
"""

LINEAGE_ACTIVE_REVISIONS_CTE = """
WITH RECURSIVE lineage(timeline_id, parent_id, max_step, depth) AS (
    SELECT timeline_id, parent_id, :cutoff_step, 0
    FROM timelines
    WHERE timeline_id = :timeline_id
    UNION ALL
    SELECT
        t.timeline_id,
        t.parent_id,
        MIN(l.max_step, t_child.fork_step),
        l.depth + 1
    FROM lineage l
    JOIN timelines t_child ON t_child.timeline_id = l.timeline_id
    JOIN timelines t       ON t.timeline_id = l.parent_id
    WHERE l.parent_id IS NOT NULL
),
ranked_revisions AS (
    SELECT
        r.revision_id,
        r.timeline_id,
        r.step AS valid_from_step,
        r.record_type,
        r.subject_key,
        r.source_id,
        r.target_id,
        r.is_tombstone,
        r.payload,
        ROW_NUMBER() OVER (
            PARTITION BY r.record_type, r.subject_key
            ORDER BY l.depth ASC, r.step DESC, r.revision_id DESC
        ) AS rn
    FROM graph_revisions r
    JOIN lineage l
      ON r.timeline_id = l.timeline_id
     AND r.step <= l.max_step
)
SELECT record_type, subject_key, source_id, target_id, valid_from_step, payload
FROM ranked_revisions
WHERE rn = 1 AND is_tombstone = 0;
"""


class CausalStore:
    """Manages SQLite storage, recursive lineage resolution, and time-travel operations."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)
            # Ensure default 'main' timeline exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM timelines WHERE timeline_id = 'main'")
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO timelines (timeline_id, parent_id, fork_step, metadata) VALUES (?, NULL, NULL, ?)",
                    ("main", json.dumps({"description": "Primary timeline"})),
                )

    def close(self) -> None:
        self.conn.close()

    # --- Timelines & Branching ---

    def create_timeline(
        self,
        timeline_id: str,
        parent_id: str | None = None,
        fork_step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineInfo:
        with self.conn:
            self.conn.execute(
                "INSERT INTO timelines (timeline_id, parent_id, fork_step, metadata) VALUES (?, ?, ?, ?)",
                (timeline_id, parent_id, fork_step, json.dumps(metadata or {})),
            )
        return TimelineInfo(
            timeline_id=timeline_id,
            parent_id=parent_id,
            fork_step=fork_step,
            metadata=metadata or {},
        )

    def get_timeline(self, timeline_id: str) -> TimelineInfo | None:
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
            metadata=json.loads(row["metadata"]),
        )

    def list_timelines(self) -> list[TimelineInfo]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT timeline_id, parent_id, fork_step, metadata FROM timelines")
        return [
            TimelineInfo(
                timeline_id=row["timeline_id"],
                parent_id=row["parent_id"],
                fork_step=row["fork_step"],
                metadata=json.loads(row["metadata"]),
            )
            for row in cursor.fetchall()
        ]

    def rollback_to(self, step: int, *, timeline_id: str) -> int:
        """Atomic rollback: deletes all revisions and exemplars at > step for this timeline."""
        with self.conn:
            c1 = self.conn.execute(
                "DELETE FROM graph_revisions WHERE timeline_id = ? AND step > ?",
                (timeline_id, step),
            )
            c2 = self.conn.execute(
                "DELETE FROM voice_exemplars WHERE timeline_id = ? AND step > ?",
                (timeline_id, step),
            )
            return c1.rowcount + c2.rowcount

    # --- Lineage Queries ---

    def get_active_revisions(
        self,
        timeline_id: str,
        cutoff_step: int,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns the winning (latest) non-tombstone revision for every subject_key along the lineage."""
        query = LINEAGE_ACTIVE_REVISIONS_CTE
        params: dict[str, Any] = {"timeline_id": timeline_id, "cutoff_step": cutoff_step}
        if record_type:
            query = query.rstrip("; \n") + " AND record_type = :record_type;"
            params["record_type"] = record_type

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "record_type": row["record_type"],
                    "subject_key": row["subject_key"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "valid_from_step": row["valid_from_step"],
                    "payload": json.loads(row["payload"]),
                }
            )
        return results

    def get_active_entity(
        self,
        entity_id: str,
        *,
        timeline_id: str,
        cutoff_step: int,
    ) -> EntityRecord | None:
        revisions = self.get_active_revisions(timeline_id, cutoff_step, record_type="entity")
        for rev in revisions:
            if rev["subject_key"] == entity_id:
                p = rev["payload"]
                return EntityRecord(
                    entity_id=entity_id,
                    kind=p.get("kind", "character"),
                    canonical_name=p.get("canonical_name", entity_id),
                    base_attributes=p.get("base_attributes", {}),
                    state=p.get("state", {}),
                    valid_from_step=rev["valid_from_step"],
                )
        return None

    def get_active_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        timeline_id: str,
        cutoff_step: int,
    ) -> DirectedEdgeRecord | None:
        subject_key = f"{source_id}->{target_id}"
        revisions = self.get_active_revisions(timeline_id, cutoff_step, record_type="edge")
        for rev in revisions:
            if rev["subject_key"] == subject_key:
                return DirectedEdgeRecord(
                    source_id=source_id,
                    target_id=target_id,
                    state=rev["payload"],
                    valid_from_step=rev["valid_from_step"],
                )
        return None

    def get_all_active_entities(
        self,
        timeline_id: str,
        cutoff_step: int,
    ) -> list[EntityRecord]:
        revs = self.get_active_revisions(timeline_id, cutoff_step, record_type="entity")
        entities: list[EntityRecord] = []
        for r in revs:
            p = r["payload"]
            entities.append(
                EntityRecord(
                    entity_id=r["subject_key"],
                    kind=p.get("kind", "character"),
                    canonical_name=p.get("canonical_name", r["subject_key"]),
                    base_attributes=p.get("base_attributes", {}),
                    state=p.get("state", {}),
                    valid_from_step=r["valid_from_step"],
                )
            )
        return entities

    def get_all_active_edges(
        self,
        timeline_id: str,
        cutoff_step: int,
    ) -> list[DirectedEdgeRecord]:
        revs = self.get_active_revisions(timeline_id, cutoff_step, record_type="edge")
        edges: list[DirectedEdgeRecord] = []
        for r in revs:
            edges.append(
                DirectedEdgeRecord(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    state=r["payload"],
                    valid_from_step=r["valid_from_step"],
                )
            )
        return edges

    def get_all_active_aliases(
        self,
        timeline_id: str,
        cutoff_step: int,
    ) -> list[AliasRecord]:
        revs = self.get_active_revisions(timeline_id, cutoff_step, record_type="alias")
        aliases: list[AliasRecord] = []
        for r in revs:
            p = r["payload"]
            aliases.append(
                AliasRecord(
                    alias=r["subject_key"],
                    entity_id=r["source_id"],
                    kind=p.get("kind", "epithet"),
                    valid_from_step=r["valid_from_step"],
                )
            )
        return aliases

    # --- Write-Time State Materialization (Resolve-on-Write) ---

    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str] = (),
        step: int = 0,
        timeline_id: str = "main",
    ) -> None:
        payload = {
            "kind": entity.kind,
            "canonical_name": entity.canonical_name,
            "base_attributes": entity.base_attributes,
            "state": entity.state,
        }
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                VALUES (?, ?, 'entity', ?, ?, NULL, 0, ?)
                """,
                (timeline_id, step, entity.entity_id, entity.entity_id, json.dumps(payload)),
            )
            for alias in aliases:
                self.conn.execute(
                    """
                    INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                    VALUES (?, ?, 'alias', ?, ?, NULL, 0, ?)
                    """,
                    (timeline_id, step, alias, entity.entity_id, json.dumps({"kind": "permanent"})),
                )

    def apply_delta(
        self,
        step: int,
        delta: StepDelta,
        *,
        timeline_id: str = "main",
    ) -> None:
        """Applies StepDelta by resolving current active state and writing complete materialized revisions."""
        with self.conn:
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

                canonical_name = ed.canonical_name or (active.canonical_name if active else ed.entity_id)
                kind = ed.kind or (active.kind if active else "character")

                payload = {
                    "kind": kind,
                    "canonical_name": canonical_name,
                    "base_attributes": base_attrs,
                    "state": current_state,
                }
                self.conn.execute(
                    """
                    INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                    VALUES (?, ?, 'entity', ?, ?, NULL, 0, ?)
                    """,
                    (timeline_id, step, ed.entity_id, ed.entity_id, json.dumps(payload)),
                )

                # Add new aliases
                for alias in ed.add_aliases:
                    self.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                        VALUES (?, ?, 'alias', ?, ?, NULL, 0, ?)
                        """,
                        (timeline_id, step, alias, ed.entity_id, json.dumps({"kind": "epithet"})),
                    )

                # Remove aliases (tombstone)
                for alias in ed.remove_aliases:
                    self.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                        VALUES (?, ?, 'alias', ?, ?, NULL, 1, '{}')
                        """,
                        (timeline_id, step, alias, ed.entity_id),
                    )

            # 2. Process edge deltas
            for rd in delta.edges:
                subject_key = f"{rd.source_id}->{rd.target_id}"
                if rd.remove:
                    self.conn.execute(
                        """
                        INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                        VALUES (?, ?, 'edge', ?, ?, ?, 1, '{}')
                        """,
                        (timeline_id, step, subject_key, rd.source_id, rd.target_id),
                    )
                    continue

                active_edge = self.get_active_edge(rd.source_id, rd.target_id, timeline_id=timeline_id, cutoff_step=step)
                edge_state = dict(active_edge.state) if (active_edge and rd.merge) else {}
                edge_state.update(rd.state)

                self.conn.execute(
                    """
                    INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                    VALUES (?, ?, 'edge', ?, ?, ?, 0, ?)
                    """,
                    (timeline_id, step, subject_key, rd.source_id, rd.target_id, json.dumps(edge_state)),
                )

    # --- Voice Exemplars Storage ---

    def insert_voice_exemplars(
        self,
        exemplars: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        timeline_id: str = "main",
    ) -> None:
        with self.conn:
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
        cutoff_step: int,
        speaker_id: str,
    ) -> list[tuple[VoiceExemplar, bytes]]:
        """Retrieves voice exemplars for speaker_id visible along the lineage with step < cutoff_step."""
        query = """
        WITH RECURSIVE lineage(timeline_id, parent_id, max_step, depth) AS (
            SELECT timeline_id, parent_id, :cutoff_step, 0
            FROM timelines
            WHERE timeline_id = :timeline_id
            UNION ALL
            SELECT
                t.timeline_id,
                t.parent_id,
                MIN(l.max_step, t_child.fork_step),
                l.depth + 1
            FROM lineage l
            JOIN timelines t_child ON t_child.timeline_id = l.timeline_id
            JOIN timelines t       ON t.timeline_id = l.parent_id
            WHERE l.parent_id IS NOT NULL
        )
        SELECT v.exemplar_id, v.timeline_id, v.step, v.speaker_id, v.addressee_id,
               v.text, v.source_text, v.metadata, v.quality_score, v.distinctiveness_score, v.embedding
        FROM voice_exemplars v
        JOIN lineage l
          ON v.timeline_id = l.timeline_id
         AND v.step < l.max_step
        WHERE v.speaker_id = :speaker_id
        ORDER BY v.step DESC;
        """
        cursor = self.conn.cursor()
        cursor.execute(query, {"timeline_id": timeline_id, "cutoff_step": cutoff_step, "speaker_id": speaker_id})
        items = []
        for row in cursor.fetchall():
            ex = VoiceExemplar(
                exemplar_id=row["exemplar_id"],
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
        return items

    # --- History & Branch Diffing ---

    def get_history(self, subject_key: str, *, timeline_id: str) -> list[dict[str, Any]]:
        """Returns the full chronological interval history of a subject_key along the lineage."""
        query = """
        WITH RECURSIVE lineage(timeline_id, parent_id, max_step, depth) AS (
            SELECT timeline_id, parent_id, 999999999, 0
            FROM timelines
            WHERE timeline_id = :timeline_id
            UNION ALL
            SELECT
                t.timeline_id,
                t.parent_id,
                MIN(l.max_step, t_child.fork_step),
                l.depth + 1
            FROM lineage l
            JOIN timelines t_child ON t_child.timeline_id = l.timeline_id
            JOIN timelines t       ON t.timeline_id = l.parent_id
            WHERE l.parent_id IS NOT NULL
        ),
        visible_revs AS (
            SELECT r.*
            FROM graph_revisions r
            JOIN lineage l
              ON r.timeline_id = l.timeline_id
             AND r.step <= l.max_step
            WHERE r.subject_key = :subject_key
        ),
        projected AS (
            SELECT
                revision_id,
                timeline_id,
                step AS valid_from_step,
                LEAD(step - 1) OVER (ORDER BY step ASC, revision_id ASC) AS valid_to_step,
                is_tombstone,
                payload
            FROM visible_revs
        )
        SELECT * FROM projected ORDER BY valid_from_step ASC;
        """
        cursor = self.conn.cursor()
        cursor.execute(query, {"timeline_id": timeline_id, "subject_key": subject_key})
        rows = cursor.fetchall()
        return [
            {
                "revision_id": r["revision_id"],
                "timeline_id": r["timeline_id"],
                "valid_from_step": r["valid_from_step"],
                "valid_to_step": r["valid_to_step"],
                "is_tombstone": bool(r["is_tombstone"]),
                "payload": json.loads(r["payload"]),
            }
            for r in rows
        ]

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
        """Exports a flattened, standalone SQLite save-slot representing active state at at_step."""
        path = Path(target_path)
        if path.exists():
            path.unlink()

        export_store = CausalStore(path)
        active_revs = self.get_active_revisions(timeline_id, at_step)
        with export_store.conn:
            for r in active_revs:
                export_store.conn.execute(
                    """
                    INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                    VALUES ('main', 0, ?, ?, ?, ?, 0, ?)
                    """,
                    (r["record_type"], r["subject_key"], r["source_id"], r["target_id"], json.dumps(r["payload"])),
                )
        export_store.close()
        return path
