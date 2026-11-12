from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from .models import (
    AliasRecord,
    DirectedEdgeRecord,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
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
    subject_key  TEXT NOT NULL, -- 'char_arthur' | 'char_arthur:stance->char_merlin' | 'the old wizard'
    relation_type TEXT,         -- predicate for 'edge' (e.g. 'stance', 'kinship'); NULL for 'entity'/'alias'
    source_id    TEXT,          -- entity_id for 'entity'/'alias'; source_id for 'edge'
    target_id    TEXT,          -- target_id for 'edge'
    is_tombstone INTEGER NOT NULL DEFAULT 0, -- 1 if deleted/expired at this step
    payload      TEXT NOT NULL  -- Complete materialized JSON state at this step
);

CREATE INDEX IF NOT EXISTS idx_revisions_lookup
    ON graph_revisions(timeline_id, record_type, subject_key, step DESC, revision_id DESC);

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
        r.relation_type,
        r.source_id,
        r.target_id,
        r.is_tombstone,
        r.payload,
        ROW_NUMBER() OVER (
            PARTITION BY r.record_type, r.subject_key
            ORDER BY l.depth ASC, r.step DESC, r.revision_id DESC
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY r.record_type, r.subject_key
        ) AS revision_count,
        LEAD(r.payload) OVER (
            PARTITION BY r.record_type, r.subject_key
            ORDER BY l.depth ASC, r.step DESC, r.revision_id DESC
        ) AS prior_payload,
        LEAD(r.step) OVER (
            PARTITION BY r.record_type, r.subject_key
            ORDER BY l.depth ASC, r.step DESC, r.revision_id DESC
        ) AS prior_step,
        LEAD(r.is_tombstone) OVER (
            PARTITION BY r.record_type, r.subject_key
            ORDER BY l.depth ASC, r.step DESC, r.revision_id DESC
        ) AS prior_is_tombstone
    FROM graph_revisions r
    JOIN lineage l
      ON r.timeline_id = l.timeline_id
     AND r.step <= l.max_step
    /*EXTRA_FILTERS*/
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


class TimelineBranchConflictError(RuntimeError):
    """Raised when rolling back a parent timeline would invalidate child branches."""


class MemoryStore(Protocol):
    """Backend storage protocol for NarrativeMemoryEngine."""

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
    def rollback_to(self, step: int, *, timeline_id: str, force: bool = False) -> int: ...
    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str | AliasRecord] = (),
        step: int = 0,
        timeline_id: str = "main",
    ) -> None: ...
    def get_all_active_entities(self, timeline_id: str, cutoff_step: int) -> list[EntityRecord]: ...
    def get_all_active_edges(
        self,
        timeline_id: str,
        cutoff_step: int,
        *,
        relation_type: str | None = None,
    ) -> list[DirectedEdgeRecord]: ...
    def get_all_active_aliases(self, timeline_id: str, cutoff_step: int) -> list[AliasRecord]: ...
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
        cutoff_step: int,
    ) -> list[tuple[EpisodeRecord, bytes]]: ...
    def query_episodes(
        self,
        *,
        timeline_id: str = "main",
        cutoff_step: int = 999999999,
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
    def get_history(self, subject_key: str, *, timeline_id: str) -> list[dict[str, Any]]: ...
    def diff_timelines(self, timeline_a: str, timeline_b: str, *, at_step: int) -> dict[str, Any]: ...
    def export_snapshot(
        self,
        target_path: str | Path,
        *,
        at_step: int,
        timeline_id: str = "main",
    ) -> Path: ...


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
            # Safe migration for existing DBs lacking believed_summary column
            cols = {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(episode_epistemics)").fetchall()
            }
            if "believed_summary" not in cols:
                self.conn.execute("ALTER TABLE episode_epistemics ADD COLUMN believed_summary TEXT")
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

    def rollback_to(self, step: int, *, timeline_id: str, force: bool = False) -> int:
        """Atomic rollback: deletes all revisions, exemplars, and episodes at > step for this timeline."""
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
        with self.conn:
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
            return c1.rowcount + c2.rowcount + c3.rowcount + c4.rowcount

    # --- Lineage Queries ---

    def get_active_revisions(
        self,
        timeline_id: str,
        cutoff_step: int,
        record_type: str | None = None,
        *,
        subject_key: str | None = None,
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

        extra_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = LINEAGE_ACTIVE_REVISIONS_CTE.replace("/*EXTRA_FILTERS*/", extra_sql)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = []
        for row in cursor.fetchall():
            prior_payload = None
            prior_step = None
            if row["prior_payload"] is not None and not row["prior_is_tombstone"]:
                prior_payload = json.loads(row["prior_payload"])
                prior_step = row["prior_step"]
            results.append(
                {
                    "record_type": row["record_type"],
                    "subject_key": row["subject_key"],
                    "relation_type": row["relation_type"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "valid_from_step": row["valid_from_step"],
                    "payload": json.loads(row["payload"]),
                    "revision_count": row["revision_count"],
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
        revisions = self.get_active_revisions(timeline_id, cutoff_step, record_type="edge")
        edges: list[DirectedEdgeRecord] = []
        for rev in revisions:
            if rev["source_id"] == source_id and rev["target_id"] == target_id:
                edges.append(self._row_to_edge(rev))
        return edges

    def get_all_active_entities(
        self,
        timeline_id: str,
        cutoff_step: int,
    ) -> list[EntityRecord]:
        revs = self.get_active_revisions(timeline_id, cutoff_step, record_type="entity")
        return [self._row_to_entity(r) for r in revs]

    def get_all_active_edges(
        self,
        timeline_id: str,
        cutoff_step: int,
        *,
        relation_type: str | None = None,
    ) -> list[DirectedEdgeRecord]:
        revs = self.get_active_revisions(timeline_id, cutoff_step, record_type="edge")
        edges: list[DirectedEdgeRecord] = []
        for r in revs:
            r_type = r.get("relation_type") or "stance"
            if relation_type is not None and r_type != relation_type:
                continue
            edges.append(self._row_to_edge(r, default_relation=r_type))
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
                    known_to=p.get("known_to"),
                    persona_name=p.get("persona_name"),
                    valid_from_step=r["valid_from_step"],
                )
            )
        return aliases

    # --- Write-Time State Materialization (Resolve-on-Write) ---

    def _insert_alias_revision(
        self,
        *,
        timeline_id: str,
        step: int,
        entity_id: str,
        alias: str | AliasRecord,
        default_kind: str = "epithet",
    ) -> None:
        """Inserts an alias revision preserving kind, known_to, and persona_name."""
        if isinstance(alias, AliasRecord):
            alias_str = alias.alias
            target_entity_id = alias.entity_id or entity_id
            alias_payload: dict[str, Any] = {"kind": alias.kind}
            if alias.known_to is not None:
                alias_payload["known_to"] = alias.known_to
            if alias.persona_name is not None:
                alias_payload["persona_name"] = alias.persona_name
        else:
            alias_str = alias
            target_entity_id = entity_id
            alias_payload = {"kind": default_kind}

        self.conn.execute(
            """
            INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
            VALUES (?, ?, 'alias', ?, ?, NULL, 0, ?)
            """,
            (timeline_id, step, alias_str, target_entity_id, json.dumps(alias_payload)),
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

        with self.conn:
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

        with self.conn:
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

                caused_by = ed.caused_by_episode_id
                if caused_by is None and active and ed.merge_state:
                    caused_by = active.caused_by_episode_id
                reason = ed.reason
                if reason is None and active and ed.merge_state:
                    reason = active.reason

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

                caused_by = rd.caused_by_episode_id
                if caused_by is None and active_edge and rd.merge:
                    caused_by = active_edge.caused_by_episode_id
                reason = rd.reason
                if reason is None and active_edge and rd.merge:
                    reason = active_edge.reason

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
                    json.dumps(ep.metadata),
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

            for ent_id, mode in initial_known.items():
                self.conn.execute(
                    """
                    INSERT INTO episode_epistemics (timeline_id, step, episode_id, entity_id, epistemic, believed_summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timeline_id, ep.step, ep_id, ent_id, mode, ep.believed_summaries.get(ent_id)),
                )

    def insert_episodes(
        self,
        episodes: Sequence[tuple[EpisodeRecord, bytes]],
        *,
        timeline_id: str = "main",
    ) -> None:
        with self.conn:
            self._insert_episodes_in_tx(episodes, timeline_id=timeline_id)

    def record_epistemic(
        self,
        inform: InformDelta,
        *,
        step: int,
        timeline_id: str = "main",
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO episode_epistemics (timeline_id, step, episode_id, entity_id, epistemic, believed_summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timeline_id, step, inform.episode_id, inform.entity_id, inform.epistemic, inform.believed_summary),
            )

    def get_lineage_episodes(
        self,
        timeline_id: str,
        cutoff_step: int,
    ) -> list[tuple[EpisodeRecord, bytes]]:
        """Retrieves all episodes visible along the lineage with step <= cutoff_step, with resolved epistemic map."""
        ep_query = """
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
        SELECT e.episode_id, e.timeline_id, e.step, e.predicate, e.summary,
               e.location_id, e.salience, e.participants, e.witnesses, e.metadata, e.embedding
        FROM episodes e
        JOIN lineage l
          ON e.timeline_id = l.timeline_id
         AND e.step <= l.max_step
        ORDER BY e.step DESC;
        """

        epistemic_query = """
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
        SELECT ep.episode_id, ep.entity_id, ep.epistemic, ep.believed_summary, ep.step, l.depth
        FROM episode_epistemics ep
        JOIN lineage l
          ON ep.timeline_id = l.timeline_id
         AND ep.step <= l.max_step
        ORDER BY l.depth DESC, ep.step ASC, ep.id ASC;
        """
        cursor = self.conn.cursor()
        cursor.execute(epistemic_query, {"timeline_id": timeline_id, "cutoff_step": cutoff_step})
        known_map: dict[str, dict[str, Literal["direct", "told", "deceived"]]] = {}
        believed_map: dict[str, dict[str, str]] = {}
        for row in cursor.fetchall():
            ep_id = row["episode_id"]
            ent_id = row["entity_id"]
            raw_mode = row["epistemic"]
            if raw_mode == "direct":
                mode: Literal["direct", "told", "deceived"] = "direct"
            elif raw_mode == "deceived":
                mode = "deceived"
            else:
                mode = "told"
            if ep_id not in known_map:
                known_map[ep_id] = {}
            # 'direct' awareness is never downgraded to 'told' or 'deceived'
            if known_map[ep_id].get(ent_id) != "direct":
                known_map[ep_id][ent_id] = mode
                if mode == "deceived" and row["believed_summary"]:
                    believed_map.setdefault(ep_id, {})[ent_id] = row["believed_summary"]
                elif ep_id in believed_map and ent_id in believed_map[ep_id]:
                    believed_map[ep_id].pop(ent_id, None)

        cursor.execute(ep_query, {"timeline_id": timeline_id, "cutoff_step": cutoff_step})
        items: list[tuple[EpisodeRecord, bytes]] = []
        seen_ids: set[str] = set()
        for row in cursor.fetchall():
            ep_id = row["episode_id"]
            if ep_id in seen_ids:
                continue
            seen_ids.add(ep_id)
            rec = EpisodeRecord(
                episode_id=ep_id,
                timeline_id=row["timeline_id"],
                step=row["step"],
                predicate=row["predicate"],
                summary=row["summary"],
                location_id=row["location_id"],
                salience=row["salience"],
                participants=json.loads(row["participants"]),
                witnesses=json.loads(row["witnesses"]),
                known_by=known_map.get(ep_id, {}),
                believed_summaries=believed_map.get(ep_id, {}),
                metadata=json.loads(row["metadata"]),
            )
            items.append((rec, bytes(row["embedding"])))
        return items

    def get_episode(
        self,
        episode_id: str,
        *,
        timeline_id: str = "main",
        cutoff_step: int = 999999999,
    ) -> EpisodeRecord | None:
        for ep, _ in self.get_lineage_episodes(timeline_id, cutoff_step):
            if ep.episode_id == episode_id:
                return ep
        return None

    def query_episodes(
        self,
        *,
        timeline_id: str = "main",
        cutoff_step: int = 999999999,
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
            if known_set:
                if not (known_set & set(ep.known_by.keys())):
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
         AND v.step <= l.max_step
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
            SELECT
                r.*,
                ROW_NUMBER() OVER (
                    PARTITION BY r.step
                    ORDER BY l.depth ASC, r.revision_id DESC
                ) AS step_rn
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
            WHERE step_rn = 1
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
        """Exports a flattened, standalone SQLite save-slot representing active state and episodes at at_step."""
        path = Path(target_path)
        if path.exists():
            path.unlink()

        export_store = CausalStore(path)
        active_revs = self.get_active_revisions(timeline_id, at_step)
        lineage_episodes = self.get_lineage_episodes(timeline_id, at_step)
        # Export visible voice exemplars along the lineage up to at_step
        voice_query = """
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
         AND v.step <= l.max_step
        ORDER BY v.step ASC;
        """
        voice_rows = self.conn.execute(
            voice_query, {"timeline_id": timeline_id, "cutoff_step": at_step}
        ).fetchall()
        lineage_exemplars = [
            (
                VoiceExemplar(
                    exemplar_id=row["exemplar_id"],
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
            for row in voice_rows
        ]
        with export_store.conn:
            for r in active_revs:
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
                        json.dumps(r["payload"]),
                    ),
                )
            export_store._insert_episodes_in_tx(lineage_episodes, timeline_id="main")
            if lineage_exemplars:
                export_store.insert_voice_exemplars(lineage_exemplars, timeline_id="main")
        export_store.close()
        return path
