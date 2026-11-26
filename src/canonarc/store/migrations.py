from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Literal

from ..models import EpisodeRecord
from .schema import LINEAGE_CTE_SQL, _pack_episode_metadata


class _MigrationStoreMixin:
    """Lineage-safe entity merge migration helpers for voice exemplars and episodes."""

    if TYPE_CHECKING:
        conn: sqlite3.Connection
        _lock: threading.RLock

        def transaction(self) -> AbstractContextManager[None]: ...
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

    def migrate_entity_voice_exemplars(
        self,
        duplicate_id: str,
        canonical_id: str,
        *,
        step: int,
        timeline_id: str = "main",
        dry_run: bool = False,
    ) -> int:
        """Lineage-safely rebinds voice exemplars referencing duplicate_id to canonical_id."""
        query = (
            LINEAGE_CTE_SQL
            + """
        SELECT v.exemplar_id, v.timeline_id, v.step, v.speaker_id, v.addressee_id,
               v.text, v.source_text, v.metadata, v.quality_score, v.distinctiveness_score, v.embedding,
               l.depth
        FROM voice_exemplars v
        JOIN lineage l
          ON v.timeline_id = l.timeline_id
         AND (l.max_step IS NULL OR v.step <= l.max_step)
        WHERE v.speaker_id = :dup_id OR v.addressee_id = :dup_id
        ORDER BY l.depth ASC, v.step DESC;
        """
        )
        with self._lock:
            rows = self.conn.execute(
                query,
                {"timeline_id": timeline_id, "cutoff_step": step, "dup_id": duplicate_id},
            ).fetchall()

        seen_ids: set[str] = set()
        targets: list[sqlite3.Row] = []
        for row in rows:
            ex_id = row["exemplar_id"]
            if ex_id in seen_ids:
                continue
            seen_ids.add(ex_id)
            targets.append(row)

        if dry_run or not targets:
            return len(targets)

        with self.transaction():
            for row in targets:
                new_speaker = canonical_id if row["speaker_id"] == duplicate_id else row["speaker_id"]
                new_addressee = (
                    canonical_id if row["addressee_id"] == duplicate_id else row["addressee_id"]
                )
                if new_addressee == new_speaker:
                    new_addressee = None
                if row["timeline_id"] == timeline_id:
                    self.conn.execute(
                        """
                        UPDATE voice_exemplars
                        SET speaker_id = ?, addressee_id = ?
                        WHERE timeline_id = ? AND exemplar_id = ?
                        """,
                        (new_speaker, new_addressee, timeline_id, row["exemplar_id"]),
                    )
                else:
                    # Shadow inherited parent exemplar on child branch without mutating parent timeline
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO voice_exemplars
                        (exemplar_id, timeline_id, step, speaker_id, addressee_id, text, source_text, metadata, quality_score, distinctiveness_score, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["exemplar_id"],
                            timeline_id,
                            row["step"],
                            new_speaker,
                            new_addressee,
                            row["text"],
                            row["source_text"],
                            row["metadata"],
                            row["quality_score"],
                            row["distinctiveness_score"],
                            row["embedding"],
                        ),
                    )
        return len(targets)

    def migrate_entity_episodes(
        self,
        duplicate_id: str,
        canonical_id: str,
        *,
        step: int,
        timeline_id: str = "main",
        dry_run: bool = False,
    ) -> tuple[int, int]:
        """Lineage-safely migrates episode participants/witnesses/locations and epistemic awareness."""
        ep_items = self.get_lineage_episodes(timeline_id, step)
        if not ep_items:
            return 0, 0

        awareness_rank = {"direct": 3, "told": 2, "deceived": 1}
        ep_updates: list[tuple[EpisodeRecord, bytes]] = []
        epistemic_inserts: list[tuple[str, int, str, str, str, str | None]] = []

        for ep, emb_bytes in ep_items:
            ep_changed = False
            new_participants = dict(ep.participants)
            if duplicate_id in new_participants:
                dup_role = new_participants.pop(duplicate_id)
                new_participants.setdefault(canonical_id, dup_role)
                ep_changed = True

            new_witnesses: list[str] = []
            seen_w: set[str] = set()
            witnesses_changed = False
            for w in ep.witnesses:
                mapped = canonical_id if w == duplicate_id else w
                if mapped != w:
                    witnesses_changed = True
                if mapped not in seen_w:
                    seen_w.add(mapped)
                    new_witnesses.append(mapped)
            if witnesses_changed:
                ep_changed = True

            new_location = canonical_id if ep.location_id == duplicate_id else ep.location_id
            if new_location != ep.location_id:
                ep_changed = True

            if ep_changed:
                updated_ep = ep.model_copy(
                    update={
                        "participants": new_participants,
                        "witnesses": new_witnesses,
                        "location_id": new_location,
                    }
                )
                ep_updates.append((updated_ep, emb_bytes))

            if duplicate_id in ep.known_by:
                dup_mode = ep.known_by[duplicate_id]
                canon_mode = ep.known_by.get(canonical_id)
                if awareness_rank.get(dup_mode, 0) > awareness_rank.get(canon_mode or "", 0):
                    believed = ep.believed_summaries.get(duplicate_id) if dup_mode == "deceived" else None
                    epistemic_inserts.append(
                        (timeline_id, step, ep.episode_id, canonical_id, dup_mode, believed)
                    )
                epistemic_inserts.append(
                    (timeline_id, step, ep.episode_id, duplicate_id, "suppressed", None)
                )

        migrated_epistemics = sum(1 for row in epistemic_inserts if row[3] == canonical_id)
        if dry_run:
            return len(ep_updates), migrated_epistemics

        if ep_updates or epistemic_inserts:
            with self.transaction():
                for updated_ep, emb_bytes in ep_updates:
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO episodes
                        (episode_id, timeline_id, step, predicate, summary, location_id, salience, participants, witnesses, metadata, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            updated_ep.episode_id,
                            timeline_id,
                            updated_ep.step,
                            updated_ep.predicate,
                            updated_ep.summary,
                            updated_ep.location_id,
                            updated_ep.salience,
                            json.dumps(updated_ep.participants),
                            json.dumps(updated_ep.witnesses),
                            json.dumps(_pack_episode_metadata(updated_ep)),
                            emb_bytes,
                        ),
                    )
                for row in epistemic_inserts:
                    self.conn.execute(
                        """
                        INSERT INTO episode_epistemics (timeline_id, step, episode_id, entity_id, epistemic, believed_summary)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        row,
                    )
        return len(ep_updates), migrated_epistemics
