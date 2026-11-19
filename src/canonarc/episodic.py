from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np

from .embedders import Embedder, OfflineNgramEmbedder
from .models import EpisodeRecord


@dataclass(frozen=True)
class EpisodicScoringPolicy:
    """Tunable weights for the 4-signal Tier 6 episodic memory ranker."""

    provenance_bonus: float = 1.0
    multi_entity_overlap_score: float = 0.60
    single_entity_overlap_score: float = 0.30
    location_bonus: float = 0.25
    semantic_weight: float = 0.40
    salience_weight: float = 0.25
    recency_weight: float = 0.20
    base_decay_rate: float = 0.04
    salience_decay_resistance: float = 0.90


DEFAULT_EPISODIC_POLICY = EpisodicScoringPolicy()


def cosine_similarity(
    embedder: Embedder,
    q_vec: np.ndarray | None,
    q_norm: float,
    emb_bytes: bytes,
    fallback_text: str,
) -> float:
    """Computes cosine similarity against stored float32 bytes, re-embedding on dimension mismatch."""
    if q_vec is None or q_norm <= 1e-6:
        return 0.5
    if len(emb_bytes) > 0:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        if emb.shape != q_vec.shape:
            emb = embedder.embed_text(fallback_text)
    else:
        emb = embedder.embed_text(fallback_text)
    e_norm = float(np.linalg.norm(emb))
    return float(np.dot(q_vec, emb)) / (q_norm * e_norm) if e_norm > 1e-6 else 0.0


class EpisodeRanker(Protocol):
    """Protocol for pluggable Tier 6 episodic memory ranking."""

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
    ) -> list[EpisodeRecord]: ...


class DefaultEpisodeRanker:
    """Hybrid 4-signal Tier 6 episodic memory scorer:
    1. Epistemic POV filtering (direct / told / deceived awareness)
    2. Causal provenance pinning (why an active entity/edge is in its current state)
    3. Entity / Location / Artifact cue overlap (with dyadic co-occurrence boost)
    4. Semantic cue similarity (deceived-POV aware) + Salience-modulated recency decay
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder or OfflineNgramEmbedder()

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
        if not candidates or top_k <= 0:
            return []

        q_vec = self.embedder.embed_text(query_text) if query_text else None
        q_norm = float(np.linalg.norm(q_vec)) if q_vec is not None else 0.0
        pinned = pinned_episode_ids or set()
        pov_set = set(pov_entity_ids) if pov_entity_ids else set()

        scored: list[tuple[float, int, EpisodeRecord]] = []

        for ep, emb_bytes in candidates:
            # 1. Hard Epistemic POV filter (prevent metagaming)
            if pov_set:
                known_entities = set(ep.known_by.keys())
                if pov_mode == "all":
                    if not pov_set.issubset(known_entities):
                        continue
                else:
                    if not (pov_set & known_entities):
                        continue

            # Resolve POV-projected summary before semantic scoring so deceived observers
            # match against their believed_summary rather than the hidden ground truth.
            effective_summary = ep.summary
            if pov_set:
                matched_povs = sorted(pid for pid in pov_set if pid in ep.known_by)
                if matched_povs and all(
                    ep.known_by[pid] == "deceived" and pid in ep.believed_summaries
                    for pid in matched_povs
                ):
                    effective_summary = ep.believed_summaries[matched_povs[0]]

            # 2. Causal provenance pin
            provenance_bonus = (
                policy.provenance_bonus if (ep.episode_id and ep.episode_id in pinned) else 0.0
            )

            # 3. Entity / Location / Artifact overlap
            ep_entities = set(ep.participants.keys()) | set(ep.witnesses)
            overlap = scene_entity_ids & ep_entities
            if len(overlap) >= 2:
                overlap_score = policy.multi_entity_overlap_score
            elif len(overlap) == 1:
                overlap_score = policy.single_entity_overlap_score
            else:
                overlap_score = 0.0

            location_bonus = (
                policy.location_bonus
                if (ep.location_id and ep.location_id in scene_entity_ids)
                else 0.0
            )

            # 4. Semantic cue similarity
            if q_vec is not None and q_norm > 1e-6:
                emb_for_sim = b"" if effective_summary != ep.summary else emb_bytes
                sim = cosine_similarity(
                    self.embedder, q_vec, q_norm, emb_for_sim, effective_summary
                )
                semantic_score = policy.semantic_weight * max(0.0, sim)
            else:
                semantic_score = 0.0

            # 5. Salience & Salience-Modulated Recency Decay
            salience_norm = max(0.0, min(1.0, ep.salience / 10.0))
            salience_score = policy.salience_weight * salience_norm

            delta_steps = max(0, current_step - ep.step)
            decay_rate = policy.base_decay_rate * (
                1.0 - policy.salience_decay_resistance * salience_norm
            )
            recency_score = policy.recency_weight * math.exp(-decay_rate * delta_steps)

            total_score = (
                provenance_bonus
                + overlap_score
                + location_bonus
                + semantic_score
                + salience_score
                + recency_score
            )
            ep_copy = ep.model_copy()
            ep_copy.summary = effective_summary
            ep_copy.match_score = round(total_score, 4)
            scored.append((total_score, ep.step, ep_copy))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:top_k]]
