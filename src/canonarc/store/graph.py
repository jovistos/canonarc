from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from ..models import (
    AliasRecord,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    StepDelta,
)
from .schema import (
    LINEAGE_ACTIVE_REVISIONS_CTE,
    _unpack_edge_payload,
)


class _GraphStoreMixin:
    """Lineage-resolved graph revision queries and write-time materialization."""

    if TYPE_CHECKING:
        conn: sqlite3.Connection
        _lock: threading.RLock

        def transaction(self) -> AbstractContextManager[None]: ...
        def _insert_episodes_in_tx(
            self,
            episodes: Sequence[tuple[EpisodeRecord, bytes]],
            *,
            timeline_id: str = "main",
        ) -> None: ...

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
            stored_first_seen: int | None = None
            if isinstance(payload_dict, dict):
                raw_offset = payload_dict.pop("_revision_count_offset", 0)
                if isinstance(raw_offset, (int, float)):
                    rev_offset = int(raw_offset)
                raw_first_seen = payload_dict.pop("_first_seen_step", None)
                if isinstance(raw_first_seen, (int, float)):
                    stored_first_seen = int(raw_first_seen)
            prior_payload = None
            prior_step = None
            if row["prior_payload"] is not None and not row["prior_is_tombstone"]:
                prior_payload = json.loads(row["prior_payload"])
                if isinstance(prior_payload, dict):
                    prior_payload.pop("_revision_count_offset", None)
                    prior_payload.pop("_first_seen_step", None)
                prior_step = row["prior_step"]
            computed_first_seen = row["first_seen_step"]
            if stored_first_seen is not None and computed_first_seen is not None:
                eff_first_seen: int | None = min(int(computed_first_seen), stored_first_seen)
            elif stored_first_seen is not None:
                eff_first_seen = stored_first_seen
            elif computed_first_seen is not None:
                eff_first_seen = int(computed_first_seen)
            else:
                eff_first_seen = row["valid_from_step"]
            results.append(
                {
                    "record_type": row["record_type"],
                    "subject_key": row["subject_key"],
                    "relation_type": row["relation_type"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "valid_from_step": row["valid_from_step"],
                    "first_seen_step": eff_first_seen,
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

    @staticmethod
    def alias_subject_key(entity_id: str, alias: str) -> str:
        return f"{entity_id}:{alias.strip().lower()}"

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
            first_seen_step=rev.get("first_seen_step", rev["valid_from_step"]),
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
            first_seen_step=rev.get("first_seen_step", rev["valid_from_step"]),
            caused_by_episode_id=caused_by,
            reason=reason,
            prior_state=prior_state,
            prior_step=rev.get("prior_step") if prior_state is not None else None,
            revision_count=rev.get("revision_count", 1),
        )

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

                for alias in ed.add_aliases:
                    self._insert_alias_revision(
                        timeline_id=timeline_id,
                        step=step,
                        entity_id=ed.entity_id,
                        alias=alias,
                        default_kind="epithet",
                    )

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

    def tombstone_entity(
        self,
        entity_id: str,
        *,
        step: int,
        timeline_id: str = "main",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Writes an entity tombstone revision at (timeline_id, step) with optional metadata payload."""
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, source_id, target_id, is_tombstone, payload)
                VALUES (?, ?, 'entity', ?, ?, NULL, 1, ?)
                """,
                (timeline_id, step, entity_id, entity_id, json.dumps(payload or {})),
            )

    def tombstone_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "stance",
        *,
        step: int,
        timeline_id: str = "main",
    ) -> None:
        """Writes tombstone revision(s) closing an active edge (including legacy stance keys if present)."""
        keys_to_close = [self.edge_subject_key(source_id, target_id, relation_type)]
        if relation_type == "stance":
            legacy_key = f"{source_id}->{target_id}"
            if self.get_active_revisions(
                timeline_id, step, record_type="edge", subject_key=legacy_key
            ):
                keys_to_close.append(legacy_key)
        with self.transaction():
            for key in keys_to_close:
                self.conn.execute(
                    """
                    INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, relation_type, source_id, target_id, is_tombstone, payload)
                    VALUES (?, ?, 'edge', ?, ?, ?, ?, 1, '{}')
                    """,
                    (timeline_id, step, key, relation_type, source_id, target_id),
                )
