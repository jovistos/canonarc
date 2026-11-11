from __future__ import annotations

import math
import re
import uuid
from typing import Any, Protocol, Sequence

import numpy as np

from .models import DialogueCandidate, VoiceExemplar

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
        # Word tokens
        words = re.findall(r"\w+", clean)
        for w in words:
            h = hash(w) % self.dim
            vec[h] += 1.0

        # Character 3-grams
        for i in range(len(clean) - 2):
            tri = clean[i : i + 3]
            h = hash(tri) % self.dim
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
    """Tier 5 Gated Voice Bank & Metadata-Filtered Vector Retriever."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
    ) -> None:
        self.embedder = embedder or OfflineNgramEmbedder()
        self.gate = gate or IngestionGate()

    def rank_exemplars(
        self,
        query_text: str,
        candidates: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        target_register: str = "",
        target_tone: str = "",
        target_addressee: str | None = None,
        top_k: int = 3,
    ) -> list[VoiceExemplar]:
        """Applies Hard-Register filtering + Soft-Tone boost + Vector cosine similarity ranking."""
        if not candidates:
            return []

        q_vec = self.embedder.embed_text(query_text) if query_text else None
        scored_items: list[tuple[float, VoiceExemplar]] = []

        norm_register = target_register.strip().lower()
        norm_tone = target_tone.strip().lower()

        for ex, emb_bytes in candidates:
            meta = {k.lower(): str(v).lower() for k, v in ex.metadata.items()}
            ex_register = meta.get("register", "")
            ex_tone = meta.get("tone", "")

            # Hard Register Filter: If active scene requires a specific grammatical register,
            # reject exemplars with conflicting register.
            if norm_register and ex_register and ex_register != norm_register:
                continue

            # Cosine similarity
            if q_vec is not None and len(emb_bytes) > 0:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                dot = float(np.dot(q_vec, emb))
                q_norm = float(np.linalg.norm(q_vec))
                e_norm = float(np.linalg.norm(emb))
                sim = dot / (q_norm * e_norm) if (q_norm > 1e-6 and e_norm > 1e-6) else 0.0
            else:
                sim = 0.5  # Neutral baseline when no query text is provided

            # Soft tone bonus (+0.15) & addressee bonus (+0.10)
            tone_bonus = 0.15 if (norm_tone and ex_tone == norm_tone) else 0.0
            addr_bonus = (
                0.10
                if (target_addressee and ex.addressee_id == target_addressee)
                else 0.0
            )
            quality_factor = 0.05 * (ex.quality_score / 10.0)

            total_score = sim + tone_bonus + addr_bonus + quality_factor
            ex_copy = ex.model_copy()
            ex_copy.match_score = round(total_score, 4)
            scored_items.append((total_score, ex_copy))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_items[:top_k]]
