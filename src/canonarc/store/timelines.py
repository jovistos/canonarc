from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..models import EpisodeRecord, StepCheckpoint, TimelineInfo, VoiceExemplar
from .protocols import TimelineBranchConflictError
from .schema import LINEAGE_CTE_SQL, TIMELINE_OWNED_TABLES, _pack_episode_metadata

if TYPE_CHECKING:
    from .causal_store import CausalStore


class _TimelineStoreMixin:
    """Timeline management, checkpoints, embedding provenance, history, diffing, and snapshot export."""

    if TYPE_CHECKING:
        conn: sqlite3.Connection
        _lock: threading.RLock

        def transaction(self) -> AbstractContextManager[None]: ...
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
        ) -> list[dict[str, Any]]: ...
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
        unions = "\n                UNION ALL\n                ".join(
            f"SELECT MAX(step) AS s FROM {table} WHERE timeline_id = ?"
            for table in TIMELINE_OWNED_TABLES
        )
        row = self.conn.execute(
            f"SELECT MAX(s) AS max_step FROM ({unions})",
            tuple(timeline_id for _ in TIMELINE_OWNED_TABLES),
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
        unions = "\n                    UNION ALL\n                    ".join(
            f"SELECT timeline_id, MAX(step) AS max_step FROM {table} GROUP BY timeline_id"
            for table in TIMELINE_OWNED_TABLES
        )
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                f"""
                WITH all_steps AS (
                    {unions}
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
            deleted = 0
            for table in TIMELINE_OWNED_TABLES:
                cur = self.conn.execute(
                    f"DELETE FROM {table} WHERE timeline_id = ? AND step > ?",
                    (timeline_id, step),
                )
                deleted += cur.rowcount
            return deleted

    def delete_steps(
        self,
        steps: Sequence[int],
        *,
        timeline_id: str,
        force: bool = False,
    ) -> int:
        """Atomically deletes all records owned by `timeline_id` at the specified exact `steps`."""
        unique_steps = sorted({int(s) for s in steps})
        if not unique_steps:
            return 0
        placeholders = ", ".join("?" for _ in unique_steps)
        with self.transaction():
            if not force:
                min_step = unique_steps[0]
                conflict = self.conn.execute(
                    "SELECT timeline_id, fork_step FROM timelines WHERE parent_id = ? AND fork_step >= ?",
                    (timeline_id, min_step),
                ).fetchone()
                if conflict:
                    raise TimelineBranchConflictError(
                        f"Cannot delete step(s) {unique_steps} on '{timeline_id}': child branch "
                        f"'{conflict['timeline_id']}' forked at step {conflict['fork_step']}. "
                        "Pass force=True to override."
                    )
            params: tuple[Any, ...] = (timeline_id, *unique_steps)
            deleted = 0
            for table in TIMELINE_OWNED_TABLES:
                cur = self.conn.execute(
                    f"DELETE FROM {table} WHERE timeline_id = ? AND step IN ({placeholders})",
                    params,
                )
                deleted += cur.rowcount
            return deleted

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
                for table in (*TIMELINE_OWNED_TABLES, "timelines"):
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

    # --- History, Branch Diffing & Snapshot Export ---

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
        from .causal_store import CausalStore

        path = Path(target_path)
        if path.exists():
            path.unlink()

        export_store: CausalStore = CausalStore(path)
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
