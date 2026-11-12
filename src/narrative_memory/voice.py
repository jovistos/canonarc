from __future__ import annotations

import math
import re
import uuid
import zlib
from typing import Any, Literal, Protocol, Sequence

import numpy as np

from .models import DialogueCandidate, EpisodeRecord, VoiceExemplar
from .presets import DEFAULT_SCHEMA_POLICY, SchemaPolicy

# Quoted dialogue patterns across English, European, and Balkan typographic conventions
DIALOGUE_PATTERNS = [
    re.compile(r'["“]([^"”]+)["”]'),           # Standard English / curved quotes
    re.compile(r'„([^“]+)“'),                   # Serbian / Croatian / German
    re.compile(r'«([^»]+)»'),                   # French / Russian guillemets
    re.compile(r'(?:^|\n)\s*—\s*([^\n]+)'),     # Em-dash dialogue lines
]


class Embedder(Protocol):
    """Protocol for dense or sparse text embedding."""
    def embed_text(self, text: str) -> np.ndarray: ...


class OfflineNgramEmbedder:
    """Zero-dependency offline character + word n-gram feature hashing embedder.

    Produces normalized float32 vectors of fixed dimension with cosine similarity
    reflective of lexical overlap, morphology, and style keywords.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        clean = " ".join(text.lower().split())
        if not clean:
            return np.zeros(self.dim, dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        # Word tokens (deterministic crc32 so persisted vectors remain valid across processes)
        words = re.findall(r"\w+", clean)
        for w in words:
            h = zlib.crc32(w.encode("utf-8")) % self.dim
            vec[h] += 1.0

        # Character 3-grams
        for i in range(len(clean) - 2):
            tri = clean[i : i + 3]
            h = zlib.crc32(tri.encode("utf-8")) % self.dim
            vec[h] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec.astype(np.float32)


class IngestionGate:
    """Two-stage anti-drift gate to prevent early generation drift and error compounding."""

    def __init__(
        self,
        min_words: int = 12,
        max_words: int = 60,
        min_distinctiveness: float = 0.40,
        min_quality: float = 8.0,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.min_distinctiveness = min_distinctiveness
        self.min_quality = min_quality

    def extract_candidates(self, text: str, *, source_text: str = "") -> list[DialogueCandidate]:
        """Extracts quoted dialogue spans and applies Gate 1 (length + distinctiveness pre-filter)."""
        candidates: list[DialogueCandidate] = []
        seen_spans: set[str] = set()

        for pattern in DIALOGUE_PATTERNS:
            for match in pattern.finditer(text):
                span = match.group(1).strip()
                if not span or span in seen_spans:
                    continue
                seen_spans.add(span)

                words = re.findall(r"\w+", span)
                word_count = len(words)
                if not (self.min_words <= word_count <= self.max_words):
                    continue

                # Compute distinctiveness & lexical diversity
                lower_words = [w.lower() for w in words]
                unique_words = set(lower_words)
                # Root Type-Token Ratio: |V| / sqrt(N)
                root_ttr = len(unique_words) / math.sqrt(word_count)
                # Long words density (>= 6 chars)
                long_words = [w for w in lower_words if len(w) >= 6]
                long_ratio = len(long_words) / word_count

                # Normalized distinctiveness score in [0.0, 1.0]
                distinctiveness = min(1.0, (root_ttr / 5.0) * 0.6 + long_ratio * 0.4)
                lexical_diversity = len(unique_words) / word_count

                if distinctiveness >= self.min_distinctiveness:
                    candidates.append(
                        DialogueCandidate(
                            candidate_id=str(uuid.uuid4())[:8],
                            text=span,
                            source_text=source_text,
                            word_count=word_count,
                            distinctiveness_score=round(distinctiveness, 3),
                            lexical_diversity=round(lexical_diversity, 3),
                        )
                    )

        return candidates

    def passes_quality_gate(self, exemplar: VoiceExemplar) -> bool:
        """Gate 2 check: post-polish / quality score >= threshold."""
        return exemplar.quality_score >= self.min_quality


class VoiceBank:
    """Tier 5 Gated Voice Bank & Tier 6 Hybrid Episodic Retriever."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
    ) -> None:
        self.embedder = embedder or OfflineNgramEmbedder()
        self.gate = gate or IngestionGate()

    def _cosine_sim(
        self,
        q_vec: np.ndarray | None,
        q_norm: float,
        emb_bytes: bytes,
        fallback_text: str,
    ) -> float:
        if q_vec is None or q_norm <= 1e-6:
            return 0.5
        if len(emb_bytes) > 0:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.shape != q_vec.shape:
                emb = self.embedder.embed_text(fallback_text)
        else:
            emb = self.embedder.embed_text(fallback_text)
        e_norm = float(np.linalg.norm(emb))
        return float(np.dot(q_vec, emb)) / (q_norm * e_norm) if e_norm > 1e-6 else 0.0

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
        norm_constraints = {k.lower(): str(v).strip().lower() for k, v in constraints.items() if str(v).strip()}

        def _score_pass(allow_hard_mismatch: bool) -> list[tuple[float, VoiceExemplar]]:
            scored_items: list[tuple[float, VoiceExemplar]] = []
            for ex, emb_bytes in candidates:
                meta = {k.lower(): str(v).strip().lower() for k, v in ex.metadata.items()}
                penalty = 0.0
                rejected = False
                for hk in policy.hard_filter_keys:
                    hk_lower = hk.lower()
                    req_val = norm_constraints.get(hk_lower, "")
                    ex_val = meta.get(hk_lower, "")
                    if req_val and ex_val and ex_val != req_val:
                        if not allow_hard_mismatch:
                            rejected = True
                            break
                        penalty += policy.hard_filter_mismatch_penalty
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
        top_k: int = 5,
    ) -> list[EpisodeRecord]:
        """Hybrid 4-signal episodic memory scorer:
        1. Epistemic POV filtering (direct / told awareness)
        2. Causal provenance pinning (why an active entity/edge is in its current state)
        3. Entity / Location / Artifact cue overlap (with dyadic co-occurrence boost)
        4. Semantic cue similarity + Salience-modulated recency decay
        """
        if not candidates or top_k <= 0:
            return []

        q_vec = self.embedder.embed_text(query_text) if query_text else None
        q_norm = float(np.linalg.norm(q_vec)) if q_vec is not None else 0.0
        pinned = pinned_episode_ids or set()
        pov_set = set(pov_entity_ids) if pov_entity_ids else set()

        scored: list[tuple[float, int, EpisodeRecord]] = []

        for ep, emb_bytes in candidates:
            # 1. Hard Epistemic POV filter (Pitfall 4: prevent metagaming)
            if pov_set:
                known_entities = set(ep.known_by.keys())
                if pov_mode == "all":
                    if not pov_set.issubset(known_entities):
                        continue
                else:
                    if not (pov_set & known_entities):
                        continue

            # 2. Causal provenance pin (Pitfall 2)
            provenance_bonus = 1.0 if (ep.episode_id and ep.episode_id in pinned) else 0.0

            # 3. Entity / Location / Artifact overlap (Pitfalls 3 & 6)
            ep_entities = set(ep.participants.keys()) | set(ep.witnesses)
            overlap = scene_entity_ids & ep_entities
            if len(overlap) >= 2:
                overlap_score = 0.60
            elif len(overlap) == 1:
                overlap_score = 0.30
            else:
                overlap_score = 0.0

            location_bonus = 0.25 if (ep.location_id and ep.location_id in scene_entity_ids) else 0.0

            # 4. Semantic cue similarity (Pitfall 6)
            if q_vec is not None and q_norm > 1e-6:
                sim = self._cosine_sim(q_vec, q_norm, emb_bytes, ep.summary)
                semantic_score = 0.40 * max(0.0, sim)
            else:
                semantic_score = 0.0

            # 5. Salience & Salience-Modulated Recency Decay (Pitfall 1 & 6)
            salience_norm = max(0.0, min(1.0, ep.salience / 10.0))
            salience_score = 0.25 * salience_norm

            delta_steps = max(0, current_step - ep.step)
            # High-salience episodes decay ~10x slower than low-salience episodes
            decay_rate = 0.04 * (1.0 - 0.9 * salience_norm)
            recency_score = 0.20 * math.exp(-decay_rate * delta_steps)

            total_score = (
                provenance_bonus
                + overlap_score
                + location_bonus
                + semantic_score
                + salience_score
                + recency_score
            )
            ep_copy = ep.model_copy()
            if pov_set:
                matched_povs = sorted(pid for pid in pov_set if pid in ep.known_by)
                if matched_povs and all(
                    ep.known_by[pid] == "deceived" and pid in ep.believed_summaries
                    for pid in matched_povs
                ):
                    ep_copy.summary = ep.believed_summaries[matched_povs[0]]
            ep_copy.match_score = round(total_score, 4)
            scored.append((total_score, ep.step, ep_copy))

        # Sort by descending match score, breaking ties by newer step
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:top_k]]
