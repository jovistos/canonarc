from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal

from ..models import EpisodeRecord, InformDelta, VoiceExemplar
from .schema import (
    LINEAGE_CTE_SQL,
    _bind_in_clause,
    _build_fts_query,
    _pack_episode_metadata,
    _unpack_episode_metadata,
)


class _EpisodeStoreMixin:
    """Tier 5 voice exemplar and Tier 6 episodic ledger storage and lineage retrieval."""

    if TYPE_CHECKING:
        conn: sqlite3.Connection
        _lock: threading.RLock

        def transaction(self) -> AbstractContextManager[None]: ...

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
        return _bind_in_clause(prefix, values, params)

    @staticmethod
    def _build_fts_query(query_text: str, *, max_terms: int = 12) -> str:
        return _build_fts_query(query_text, max_terms=max_terms)

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
        pov_in = _bind_in_clause("pov", pov_list, params)
        ep_filter = ""
        if episode_ids is not None:
            ep_list = sorted({eid for eid in episode_ids if eid})
            if not ep_list:
                return set()
            ep_in = _bind_in_clause("ep", ep_list, params)
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
            ep_in = _bind_in_clause("req_ep", ep_list, params)
            ep_where_clauses.append(f"e.episode_id IN ({ep_in})")

        ep_where_sql = ("WHERE " + " AND ".join(ep_where_clauses)) if ep_where_clauses else ""

        ctes: list[str] = []
        pov_list = sorted({pid for pid in (pov_entity_ids or ()) if pid})
        pov_filter_sql = ""
        if pov_list:
            pov_in = _bind_in_clause("pov", pov_list, params)
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

        fts_query = _build_fts_query(query_text) if limit is not None else ""
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
                pin_in = _bind_in_clause("pin", pinned_list, params)
                order_terms.append(f"(CASE WHEN episode_id IN ({pin_in}) THEN 1 ELSE 0 END) DESC")

            cue_list = sorted({cid for cid in (cue_entity_ids or ()) if cid})
            match_terms: list[str] = []
            if cue_list:
                cue_in = _bind_in_clause("cue", cue_list, params)
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
                sel_in = _bind_in_clause("sel_ep", selected_ids, epistemic_params)
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
