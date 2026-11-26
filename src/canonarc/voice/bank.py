from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..embedders import Embedder, OfflineNgramEmbedder
from ..episodic import (
    DEFAULT_EPISODIC_POLICY,
    DefaultEpisodeRanker,
    EpisodeRanker,
    EpisodicScoringPolicy,
    cosine_similarity,
)
from ..models import EpisodeRecord, VoiceExemplar
from ..presets import DEFAULT_SCHEMA_POLICY, SchemaPolicy
from .gate import IngestionGate


class VoiceBank:
    """Tier 5 Gated Voice Bank (with optional backwards-compatible delegation to EpisodeRanker)."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
        episode_ranker: EpisodeRanker | None = None,
    ) -> None:
        self.embedder: Embedder = embedder or OfflineNgramEmbedder()
        self.gate: IngestionGate = gate or IngestionGate()
        self.episode_ranker: EpisodeRanker = episode_ranker or DefaultEpisodeRanker(
            embedder=self.embedder
        )

    def _cosine_sim(
        self,
        q_vec: np.ndarray | None,
        q_norm: float,
        emb_bytes: bytes,
        fallback_text: str,
    ) -> float:
        return cosine_similarity(self.embedder, q_vec, q_norm, emb_bytes, fallback_text)

    def rank_exemplars(
        self,
        query_text: str,
        candidates: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        target_register: str = "",
        target_tone: str = "",
        target_addressee: str | None = None,
        target_constraints: dict[str, str] | None = None,
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
        top_k: int = 3,
    ) -> list[VoiceExemplar]:
        """Applies policy-driven hard filters + soft attribute boosts + vector cosine similarity."""
        if not candidates or top_k <= 0:
            return []

        q_vec = self.embedder.embed_text(query_text) if query_text else None
        q_norm = float(np.linalg.norm(q_vec)) if q_vec is not None else 0.0

        constraints: dict[str, str] = {}
        if target_register:
            constraints["register"] = target_register
        if target_tone:
            constraints["tone"] = target_tone
        if target_constraints:
            constraints.update(target_constraints)
        norm_constraints = {
            k.lower(): str(v).strip().lower()
            for k, v in constraints.items()
            if str(v).strip()
        }

        def _score_pass(allow_hard_mismatch: bool) -> list[tuple[float, VoiceExemplar]]:
            scored_items: list[tuple[float, VoiceExemplar]] = []
            for ex, emb_bytes in candidates:
                meta = {k.lower(): str(v).strip().lower() for k, v in ex.metadata.items()}
                penalty = 0.0
                rejected = False
                mismatched_keys: list[str] = []
                for hk in policy.hard_filter_keys:
                    hk_lower = hk.lower()
                    req_val = norm_constraints.get(hk_lower, "")
                    ex_val = meta.get(hk_lower, "")
                    if req_val and ex_val and ex_val != req_val:
                        if not allow_hard_mismatch:
                            rejected = True
                            break
                        penalty += policy.hard_filter_mismatch_penalty
                        mismatched_keys.append(hk)
                if rejected:
                    continue

                sim = self._cosine_sim(q_vec, q_norm, emb_bytes, ex.text) if q_vec is not None else 0.5

                soft_bonus = 0.0
                for sk, weight in policy.soft_boost_weights.items():
                    sk_lower = sk.lower()
                    req_val = norm_constraints.get(sk_lower, "")
                    if req_val and meta.get(sk_lower, "") == req_val:
                        soft_bonus += weight

                addr_bonus = (
                    policy.addressee_boost
                    if (target_addressee and ex.addressee_id == target_addressee)
                    else 0.0
                )
                quality_factor = policy.quality_weight * (ex.quality_score / 10.0)

                total_score = sim + soft_bonus + addr_bonus + quality_factor - penalty
                ex_copy = ex.model_copy()
                ex_copy.match_score = round(total_score, 4)
                if mismatched_keys and policy.hard_filter_fallback_note:
                    note = policy.hard_filter_fallback_note.format(keys=", ".join(mismatched_keys))
                    ex_copy.metadata = {**ex_copy.metadata, "fallback_note": note}
                scored_items.append((total_score, ex_copy))
            return scored_items

        scored_items = _score_pass(allow_hard_mismatch=False)
        if not scored_items and policy.fallback_on_empty_hard_filter:
            scored_items = _score_pass(allow_hard_mismatch=True)

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_items[:top_k]]

    def rank_exemplars_for_targets(
        self,
        query_text: str,
        candidates: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        targets: Sequence[tuple[str | None, dict[str, str]] | tuple[str | None, str, str]],
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
        top_k: int = 3,
    ) -> list[VoiceExemplar]:
        """Ranks exemplars across multiple co-present dyadic targets via Round-Robin allocation."""
        if not candidates or top_k <= 0:
            return []
        if not targets:
            return self.rank_exemplars(query_text, candidates, policy=policy, top_k=top_k)

        normalized_targets: list[tuple[str | None, dict[str, str]]] = []
        for t in targets:
            if len(t) == 2:
                normalized_targets.append((t[0], t[1]))
            else:
                addr, reg, tone = t
                c: dict[str, str] = {}
                if reg:
                    c["register"] = reg
                if tone:
                    c["tone"] = tone
                normalized_targets.append((addr, c))

        if len(normalized_targets) == 1:
            addr, c = normalized_targets[0]
            return self.rank_exemplars(
                query_text=query_text,
                candidates=candidates,
                target_constraints=c,
                target_addressee=addr,
                policy=policy,
                top_k=top_k,
            )

        per_target_queues: list[list[VoiceExemplar]] = []
        for addr, c in normalized_targets:
            ranked = self.rank_exemplars(
                query_text=query_text,
                candidates=candidates,
                target_constraints=c,
                target_addressee=addr,
                policy=policy,
                top_k=top_k,
            )
            for ex in ranked:
                if addr and "for_addressee" not in ex.metadata:
                    ex.metadata = {**ex.metadata, "for_addressee": addr}
            if ranked:
                per_target_queues.append(ranked)

        merged: list[VoiceExemplar] = []
        seen_ids: set[str] = set()
        max_len = max((len(q) for q in per_target_queues), default=0)
        for depth in range(max_len):
            for queue in per_target_queues:
                if depth < len(queue):
                    ex = queue[depth]
                    dedup_key = ex.exemplar_id or ex.text
                    if dedup_key not in seen_ids:
                        seen_ids.add(dedup_key)
                        merged.append(ex)
                        if len(merged) >= top_k:
                            return merged

        return merged

    def rank_episodes(
        self,
        *,
        current_step: int,
        query_text: str,
        candidates: Sequence[tuple[EpisodeRecord, bytes]],
        scene_entity_ids: set[str],
        pinned_episode_ids: set[str] | None = None,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        policy: EpisodicScoringPolicy = DEFAULT_EPISODIC_POLICY,
        top_k: int = 5,
    ) -> list[EpisodeRecord]:
        """Delegates to the configured EpisodeRanker (kept here for backwards compatibility)."""
        return self.episode_ranker.rank_episodes(
            current_step=current_step,
            query_text=query_text,
            candidates=candidates,
            scene_entity_ids=scene_entity_ids,
            pinned_episode_ids=pinned_episode_ids,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            policy=policy,
            top_k=top_k,
        )
