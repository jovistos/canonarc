from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..models import EpisodeRecord

TIMELINE_OWNED_TABLES: tuple[str, ...] = (
    "graph_revisions",
    "voice_exemplars",
    "episodes",
    "episode_epistemics",
    "step_checkpoints",
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
        MIN(CASE WHEN is_tombstone = 0 THEN step END) OVER (
            PARTITION BY record_type, subject_key
        ) AS first_seen_step,
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
    first_seen_step,
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


def _bind_in_clause(prefix: str, values: Sequence[str], params: dict[str, Any]) -> str:
    """Binds a non-empty sequence of strings into named SQL placeholders."""
    placeholders: list[str] = []
    for idx, val in enumerate(values):
        key = f"{prefix}_{idx}"
        params[key] = val
        placeholders.append(f":{key}")
    return ", ".join(placeholders)


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
