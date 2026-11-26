from __future__ import annotations

import inspect
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

from .embedders import Embedder, OfflineNgramEmbedder
from .extractors import (
    JsonCompletionClient,
    TokenUsage,
    _strip_json_fences,
    validate_json_completion_client,
)
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
            is_deceived_view = False
            if pov_set:
                matched_povs = sorted(pid for pid in pov_set if pid in ep.known_by)
                if matched_povs and all(
                    ep.known_by[pid] == "deceived" and pid in ep.believed_summaries
                    for pid in matched_povs
                ):
                    believed_vals = {ep.believed_summaries[pid] for pid in matched_povs}
                    if pov_mode == "all" and len(believed_vals) > 1:
                        continue
                    effective_summary = ep.believed_summaries[matched_povs[0]]
                    is_deceived_view = True
                elif pov_mode == "all" and any(
                    ep.known_by[pid] == "deceived" for pid in matched_povs
                ):
                    # Mixed deceived + direct/told POVs do not share ground-truth common ground
                    continue

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
            if is_deceived_view:
                ep_copy.location_id = None
            ep_copy.match_score = round(total_score, 4)
            scored.append((total_score, ep.step, ep_copy))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:top_k]]


# --- Tier 6 Episode Consolidation ---

_CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a precise narrative memory consolidator. "
    "Compact a chronological list of low-salience narrative episodes into one concise, "
    "information-dense summary episode.\n"
    "Rules:\n"
    "1. Preserve concrete plot facts, named entities, objects, locations, and decisions.\n"
    "2. Do NOT invent events, motives, or causal links not stated in the input episodes.\n"
    "3. Select only the primary participants (up to 5) from the provided entity IDs.\n"
    "4. Return ONLY a valid JSON object with schema:\n"
    '{"summary": "<1-3 concise sentences>", "salience": <float 4.5-7.5>, '
    '"key_participant_ids": ["<entity_id>", ...], "location_id": "<location_id or null>"}'
)


@dataclass(frozen=True)
class ConsolidationSummary:
    """Structured output produced by an EpisodeConsolidator."""

    summary: str
    salience: float | None = None
    key_participant_ids: tuple[str, ...] = ()
    location_id: str | None = None


_EMPTY_ENTITY_NAMES: Mapping[str, str] = MappingProxyType({})


class EpisodeConsolidator(Protocol):
    """Protocol for summarizing a batch of Tier 6 episodes into a single compacted episode."""

    def summarize_episodes(
        self,
        episodes: Sequence[EpisodeRecord],
        *,
        entity_names: Mapping[str, str] = ...,
        chapter_id: str | None = None,
    ) -> ConsolidationSummary: ...


def build_consolidation_prompt(
    episodes: Sequence[EpisodeRecord],
    *,
    entity_names: Mapping[str, str] | None = None,
    chapter_id: str | None = None,
) -> tuple[str, str]:
    """Builds a prefix-cache-friendly (system_prompt, user_prompt) pair for episode consolidation."""
    names = entity_names or {}
    lines: list[str] = []
    if chapter_id:
        lines.append(f"Chapter: {chapter_id}")
    lines.append(f"Episodes to consolidate ({len(episodes)}):")
    for ep in episodes:
        meta_bits: list[str] = [f"step={ep.step}"]
        if ep.chapter_id and ep.chapter_id != chapter_id:
            meta_bits.append(f"chapter={ep.chapter_id}")
        if ep.scene_id:
            meta_bits.append(f"scene={ep.scene_id}")
        if ep.predicate and ep.predicate != "event":
            meta_bits.append(f"type={ep.predicate}")
        meta_bits.append(f"salience={ep.salience:.1f}")

        entry = f"- [{', '.join(meta_bits)}] {ep.summary.strip()}"
        if ep.participants:
            part_strs = [
                f"{names.get(eid, eid)} ({eid}: {role})" if eid in names else f"{eid} ({role})"
                for eid, role in ep.participants.items()
            ]
            entry += f" | participants: {', '.join(part_strs)}"
        if ep.location_id:
            loc_name = names.get(ep.location_id)
            loc_str = f"{loc_name} ({ep.location_id})" if loc_name else ep.location_id
            entry += f" | location: {loc_str}"
        lines.append(entry)
    lines.append("Return ONLY valid JSON.")
    return _CONSOLIDATION_SYSTEM_PROMPT, "\n".join(lines)


def _parse_consolidation_summary(
    raw: ConsolidationSummary | Mapping[str, Any] | str,
    episodes: Sequence[EpisodeRecord],
    *,
    min_salience: float = 4.5,
    max_salience: float = 7.5,
    strict_json: bool = False,
) -> ConsolidationSummary:
    """Validates and normalizes a consolidator response against the source episodes."""
    if isinstance(raw, ConsolidationSummary):
        summary_text = raw.summary.strip()
        if not summary_text:
            raise ValueError("ConsolidationSummary.summary must be non-empty.")
        clamped_salience = (
            max(min_salience, min(max_salience, float(raw.salience)))
            if raw.salience is not None
            else None
        )
        return ConsolidationSummary(
            summary=summary_text,
            salience=clamped_salience,
            key_participant_ids=tuple(raw.key_participant_ids),
            location_id=raw.location_id,
        )

    parsed: Any = raw
    if isinstance(raw, str):
        cleaned = _strip_json_fences(raw).strip()
        if not cleaned:
            raise ValueError("Consolidation response was empty.")
        if cleaned.startswith("{") or strict_json:
            parsed = json.loads(cleaned)
        else:
            # Plain-text summary fallback when a callable returns raw prose
            return ConsolidationSummary(summary=cleaned)

    if not isinstance(parsed, Mapping):
        raise ValueError(f"Unsupported consolidation response type: {type(parsed).__name__}")

    summary_val = parsed.get("summary")
    if not isinstance(summary_val, str) or not summary_val.strip():
        raise ValueError("Consolidation JSON must contain a non-empty 'summary' string.")
    summary_text = summary_val.strip()

    salience_val: float | None = None
    raw_salience = parsed.get("salience")
    if raw_salience is not None:
        try:
            salience_val = max(min_salience, min(max_salience, float(raw_salience)))
        except (TypeError, ValueError):
            salience_val = None

    valid_entities: set[str] = set()
    valid_locations: set[str] = set()
    for ep in episodes:
        valid_entities.update(ep.participants.keys())
        valid_entities.update(ep.witnesses)
        if ep.location_id:
            valid_locations.add(ep.location_id)

    key_participants: list[str] = []
    seen_parts: set[str] = set()
    raw_parts = parsed.get("key_participant_ids") or parsed.get("participants") or []
    if isinstance(raw_parts, (list, tuple)):
        for item in raw_parts:
            if isinstance(item, str):
                eid = item.strip()
                if eid and eid in valid_entities and eid not in seen_parts:
                    seen_parts.add(eid)
                    key_participants.append(eid)

    location_id: str | None = None
    raw_loc = parsed.get("location_id")
    if isinstance(raw_loc, str):
        loc_clean = raw_loc.strip()
        if loc_clean and (not valid_locations or loc_clean in valid_locations):
            location_id = loc_clean

    return ConsolidationSummary(
        summary=summary_text,
        salience=salience_val,
        key_participant_ids=tuple(key_participants),
        location_id=location_id,
    )


class CallableEpisodeConsolidator:
    """Adapter turning any Python callable into an EpisodeConsolidator."""

    def __init__(
        self,
        fn: Callable[..., ConsolidationSummary | Mapping[str, Any] | str],
        *,
        min_salience: float = 4.5,
        max_salience: float = 7.5,
    ) -> None:
        self.fn = fn
        self.min_salience = float(min_salience)
        self.max_salience = float(max_salience)
        try:
            self._params = tuple(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            self._params = ()

    def summarize_episodes(
        self,
        episodes: Sequence[EpisodeRecord],
        *,
        entity_names: Mapping[str, str] = _EMPTY_ENTITY_NAMES,
        chapter_id: str | None = None,
    ) -> ConsolidationSummary:
        if not episodes:
            raise ValueError("Cannot consolidate an empty sequence of episodes.")
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in self._params)
        kw_names = {
            p.name
            for p in self._params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        kwargs: dict[str, Any] = {}
        if accepts_var_kw or "entity_names" in kw_names:
            kwargs["entity_names"] = dict(entity_names) if isinstance(entity_names, Mapping) else {}
        if accepts_var_kw or "chapter_id" in kw_names:
            kwargs["chapter_id"] = chapter_id
        raw = self.fn(episodes, **kwargs)
        return _parse_consolidation_summary(
            raw,
            episodes,
            min_salience=self.min_salience,
            max_salience=self.max_salience,
        )


class HeuristicEpisodeConsolidator:
    """Zero-LLM deterministic episode consolidator for offline runs and tests."""

    def __init__(
        self,
        *,
        max_chars: int = 600,
        default_salience: float = 6.0,
        max_participants: int = 5,
    ) -> None:
        self.max_chars = max(80, int(max_chars))
        self.default_salience = float(default_salience)
        self.max_participants = max(1, int(max_participants))

    def summarize_episodes(
        self,
        episodes: Sequence[EpisodeRecord],
        *,
        entity_names: Mapping[str, str] = _EMPTY_ENTITY_NAMES,
        chapter_id: str | None = None,
    ) -> ConsolidationSummary:
        if not episodes:
            raise ValueError("Cannot consolidate an empty sequence of episodes.")

        seen_summaries: set[str] = set()
        snippets: list[str] = []
        participant_counts: Counter[str] = Counter()
        location_counts: Counter[str] = Counter()

        for ep in episodes:
            clean = " ".join(ep.summary.strip().split())
            if clean and clean.lower() not in seen_summaries:
                seen_summaries.add(clean.lower())
                snippets.append(clean.rstrip(".") + ".")
            for eid in ep.participants:
                participant_counts[eid] += 1
            if ep.location_id:
                location_counts[ep.location_id] += 1

        prefix = f"[{chapter_id}] " if chapter_id else ""
        joined = prefix + " ".join(snippets)
        if len(joined) > self.max_chars:
            joined = joined[: self.max_chars - 3].rstrip() + "..."

        top_participants = tuple(
            eid for eid, _ in participant_counts.most_common(self.max_participants)
        )
        top_location = location_counts.most_common(1)[0][0] if location_counts else None
        return ConsolidationSummary(
            summary=joined,
            salience=self.default_salience,
            key_participant_ids=top_participants,
            location_id=top_location,
        )


class LLMEpisodeConsolidator:
    """LLM-backed Tier 6 episode consolidator reusing any JsonCompletionClient."""

    def __init__(
        self,
        client: JsonCompletionClient,
        *,
        max_retries: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        min_salience: float = 4.5,
        max_salience: float = 7.5,
    ) -> None:
        self.client: JsonCompletionClient = validate_json_completion_client(client)
        self.max_retries = max(0, int(max_retries))
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_salience = float(min_salience)
        self.max_salience = float(max_salience)
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()

    def summarize_episodes(
        self,
        episodes: Sequence[EpisodeRecord],
        *,
        entity_names: Mapping[str, str] = _EMPTY_ENTITY_NAMES,
        chapter_id: str | None = None,
    ) -> ConsolidationSummary:
        if not episodes:
            raise ValueError("Cannot consolidate an empty sequence of episodes.")

        names_map = dict(entity_names) if isinstance(entity_names, Mapping) else {}
        system_prompt, base_user_prompt = build_consolidation_prompt(
            episodes,
            entity_names=names_map,
            chapter_id=chapter_id,
        )
        current_prompt = base_user_prompt
        call_usage = TokenUsage()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            raw_content, usage = self.client.call_json_api(
                current_prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if attempt > 0:
                usage.retry_count += 1
            call_usage.add(usage)
            try:
                result = _parse_consolidation_summary(
                    raw_content,
                    episodes,
                    min_salience=self.min_salience,
                    max_salience=self.max_salience,
                    strict_json=True,
                )
                self.last_usage = call_usage
                self.total_usage.add(call_usage)
                return result
            except Exception as exc:  # noqa: BLE001
                call_usage.validation_errors += 1
                last_exc = exc
                current_prompt = (
                    f"{base_user_prompt}\n\n"
                    f"Previous output failed JSON validation with error: {exc}\n"
                    'Return ONLY valid JSON: {"summary": "...", "salience": 6.0, '
                    '"key_participant_ids": [...], "location_id": null}'
                )

        self.last_usage = call_usage
        self.total_usage.add(call_usage)
        assert last_exc is not None
        raise last_exc
