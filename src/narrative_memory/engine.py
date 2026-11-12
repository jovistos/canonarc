from __future__ import annotations

import contextlib
import json
import re
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from rapidfuzz import fuzz

from .models import (
    AliasRecord,
    DialogueCandidate,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    ResolvedMention,
    StepContext,
    StepDelta,
    TimelineInfo,
    VoiceExemplar,
)
from .presets import DEFAULT_SCHEMA_POLICY, DomainPreset, InteractiveFictionPreset, SchemaPolicy
from .store import CausalStore, MemoryStore, TimelineBranchConflictError
from .voice import Embedder, IngestionGate, VoiceBank


class MentionResolver(Protocol):
    """Protocol for pluggable entity mention & coreference resolution."""

    def resolve(
        self,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
    ) -> list[ResolvedMention]: ...


class DeltaExtractor(Protocol):
    """Protocol for LLM-backed structured StepDelta extraction."""

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta: ...


_CJK_CHAR_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


class DefaultLexicalResolver:
    """Word-boundary lexical + alias resolver with optional rapidfuzz inflection fallback."""

    def __init__(self, fuzzy_threshold: float | None = 88.0) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def _phrase_pattern(phrase_lower: str) -> str:
        prefix = "" if _CJK_CHAR_RE.match(phrase_lower[0]) else r"(?<!\w)"
        suffix = "" if _CJK_CHAR_RE.match(phrase_lower[-1]) else r"(?!\w)"
        return rf"{prefix}{re.escape(phrase_lower)}{suffix}"

    @classmethod
    def _find_whole_phrase(cls, text_lower: str, phrase_lower: str) -> int:
        if not phrase_lower:
            return -1
        match = re.search(cls._phrase_pattern(phrase_lower), text_lower)
        return match.start() if match else -1

    def resolve(
        self,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
    ) -> list[ResolvedMention]:
        resolved: list[ResolvedMention] = []
        seen_pairs: set[tuple[str, str]] = set()  # (entity_id, surface_lower)

        name_to_id = {e.canonical_name.lower(): e.entity_id for e in entities if e.canonical_name}
        alias_map: dict[str, list[AliasRecord]] = {}
        for a in aliases:
            if a.alias:
                alias_map.setdefault(a.alias.lower(), []).append(a)

        def _add(surface: str, entity_id: str, via: Literal["canonical", "alias", "fuzzy"], conf: float = 1.0) -> None:
            key = (entity_id, surface.strip().lower())
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            resolved.append(
                ResolvedMention(
                    surface_text=surface,
                    entity_id=entity_id,
                    matched_via=via,
                    confidence=conf,
                )
            )

        # 1. Explicit mentions
        for m in explicit_mentions:
            m_clean = m.strip().lower()
            if not m_clean:
                continue
            if m_clean in name_to_id:
                _add(m, name_to_id[m_clean], "canonical")
            elif m_clean in alias_map:
                for alias_rec in alias_map[m_clean]:
                    _add(m, alias_rec.entity_id, "alias")

        # 2. Word-boundary text scan (longest phrases first to avoid partial shadowing)
        if text:
            text_lower = text.lower()
            candidates: list[tuple[str, str, Literal["canonical", "alias"]]] = []
            for name, ent_id in name_to_id.items():
                candidates.append((name, ent_id, "canonical"))
            for alias_str, alias_recs in alias_map.items():
                for alias_rec in alias_recs:
                    candidates.append((alias_str, alias_rec.entity_id, "alias"))
            candidates.sort(key=lambda item: len(item[0]), reverse=True)

            matched_spans: list[tuple[int, int]] = []
            for phrase, ent_id, via in candidates:
                pattern = self._phrase_pattern(phrase)
                for match in re.finditer(pattern, text_lower):
                    start, end = match.start(), match.end()
                    # Avoid matching a shorter sub-phrase inside an already-matched longer phrase
                    if any(s <= start and end <= e for s, e in matched_spans):
                        continue
                    matched_spans.append((start, end))
                    _add(phrase, ent_id, via)
                    break

            # 3. Optional fuzzy fallback for inflected languages / typos
            if self.fuzzy_threshold is not None:
                matched_entity_ids = {r.entity_id for r in resolved}
                tokens = re.findall(r"\w+", text_lower)
                for phrase, ent_id, _ in candidates:
                    if ent_id in matched_entity_ids or len(phrase) < 4:
                        continue
                    n_words = len(phrase.split())
                    if n_words <= 0 or len(tokens) < n_words:
                        continue
                    for idx in range(len(tokens) - n_words + 1):
                        window = " ".join(tokens[idx : idx + n_words])
                        score = fuzz.WRatio(phrase, window)
                        if score >= self.fuzzy_threshold:
                            _add(window, ent_id, "fuzzy", conf=round(score / 100.0, 3))
                            matched_entity_ids.add(ent_id)
                            break

        return resolved


class StepTransaction:
    """Context manager for reading state at `step` and atomically committing mutations on exit."""

    def __init__(
        self,
        engine: NarrativeMemoryEngine,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        top_k_episodes: int = 5,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        timeline_id: str | None = None,
        rerun_clean: bool = True,
    ) -> None:
        self.engine = engine
        self.step = step
        self.text = text
        self.mentions = mentions
        self.top_k_voice = top_k_voice
        self.top_k_episodes = top_k_episodes
        self.pov_entity_ids = pov_entity_ids
        self.pov_mode = pov_mode
        self.edge_types = edge_types
        self.timeline_id = timeline_id or engine.active_timeline
        self.rerun_clean = rerun_clean
        self._context: StepContext | None = None
        self._staged_delta: StepDelta | None = None
        self._staged_exemplars: list[VoiceExemplar] = []
        self._staged_episodes: list[EpisodeRecord] = []
        self._bypass_gate = False

    def __enter__(self) -> StepTransaction:
        if self.rerun_clean:
            # Fail fast if a child branch depends on steps >= self.step, without deleting yet
            for tl in self.engine.list_timelines():
                if (
                    tl.parent_id == self.timeline_id
                    and tl.fork_step is not None
                    and tl.fork_step > self.step - 1
                ):
                    raise TimelineBranchConflictError(
                        f"Cannot roll back '{self.timeline_id}' to step {self.step - 1}: child branch "
                        f"'{tl.timeline_id}' forked at step {tl.fork_step}. "
                        "Pass force=True to store.rollback_to to override."
                    )

        self._context = self.engine.retrieve_context(
            self.step,
            text=self.text,
            mentions=self.mentions,
            top_k_voice=self.top_k_voice,
            top_k_episodes=self.top_k_episodes,
            pov_entity_ids=self.pov_entity_ids,
            pov_mode=self.pov_mode,
            edge_types=self.edge_types,
            timeline_id=self.timeline_id,
        )
        return self

    @property
    def context(self) -> StepContext:
        if self._context is None:
            raise RuntimeError("StepTransaction is not active.")
        return self._context

    def commit(
        self,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        *,
        bypass_gate: bool = False,
    ) -> None:
        """Stage mutations to be committed when the transaction context exits successfully."""
        self._staged_delta = delta
        self._staged_exemplars.extend(exemplars)
        self._staged_episodes.extend(episodes)
        self._bypass_gate = bypass_gate

    def extract_and_commit(
        self,
        scene_text: str,
        *,
        extractor: DeltaExtractor | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        bypass_gate: bool = False,
    ) -> StepDelta:
        """Runs the configured DeltaExtractor on scene_text, normalizes IDs, and stages the StepDelta."""
        active_extractor = extractor or self.engine.extractor
        if active_extractor is None:
            raise ValueError("No DeltaExtractor provided to StepTransaction or NarrativeMemoryEngine.")
        prompt = self.engine.build_extraction_prompt(self.context, scene_text)
        raw_delta = active_extractor.extract_delta(prompt, scene_text)
        normalized = self.engine.normalize_delta(raw_delta, step=self.step, timeline_id=self.timeline_id)
        self.commit(normalized, exemplars=exemplars, episodes=episodes, bypass_gate=bypass_gate)
        return normalized

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            with self.engine._store_transaction():
                if self.rerun_clean:
                    self.engine.rollback_to(self.step - 1, timeline_id=self.timeline_id)
                if self._staged_delta or self._staged_exemplars or self._staged_episodes:
                    self.engine.commit_step(
                        self.step,
                        delta=self._staged_delta,
                        exemplars=self._staged_exemplars,
                        episodes=self._staged_episodes,
                        bypass_gate=self._bypass_gate,
                        timeline_id=self.timeline_id,
                    )


class NarrativeMemoryEngine:
    """High-level facade for Tier 1-6 narrative memory, episodic recall, time travel, and voice retrieval."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        store: MemoryStore | None = None,
        timeline_id: str = "main",
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
        preset: DomainPreset | None = None,
        mention_resolver: MentionResolver | None = None,
        extractor: DeltaExtractor | None = None,
    ) -> None:
        self.store: MemoryStore = store if store is not None else CausalStore(db_path)
        self.active_timeline = timeline_id
        self.voice_bank = VoiceBank(embedder=embedder, gate=gate)
        self.preset: DomainPreset = preset or InteractiveFictionPreset()
        self.mention_resolver: MentionResolver = mention_resolver or DefaultLexicalResolver()
        self.extractor: DeltaExtractor | None = extractor

    @property
    def _policy(self) -> SchemaPolicy:
        return getattr(self.preset, "policy", DEFAULT_SCHEMA_POLICY)

    def close(self) -> None:
        self.store.close()

    # --- Tier 1 Seeding ---

    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str | AliasRecord] = (),
        step: int = 0,
        timeline_id: str | None = None,
    ) -> None:
        tl = timeline_id or self.active_timeline
        self.store.register_entity(entity, aliases=aliases, step=step, timeline_id=tl)

    def seed_entities(
        self,
        entities: Sequence[EntityRecord],
        *,
        step: int = 0,
        timeline_id: str | None = None,
    ) -> None:
        for ent in entities:
            self.register_entity(ent, step=step, timeline_id=timeline_id)

    # --- Unified Tier 1-6 Retrieval ---

    def retrieve_context(
        self,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        top_k_episodes: int = 5,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        inclusive: bool = False,
        timeline_id: str | None = None,
        fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only",
    ) -> StepContext:
        tl = timeline_id or self.active_timeline
        policy = self._policy
        # Causal cutoff: step < current_step unless inclusive is True
        cutoff_step = step if inclusive else (step - 1)

        # 1. Fetch active entities, aliases, and edges from store
        active_entities = self.store.get_all_active_entities(tl, cutoff_step)
        active_aliases = self.store.get_all_active_aliases(tl, cutoff_step)
        active_edges = self.store.get_all_active_edges(tl, cutoff_step)
        if edge_types is not None:
            active_edges = [e for e in active_edges if e.relation_type in edge_types]

        # 2. Entity resolution via pluggable MentionResolver + POV epistemic disguise projection
        pov_set = set(pov_entity_ids) if pov_entity_ids else set()
        raw_mentions = self.mention_resolver.resolve(
            text=text,
            explicit_mentions=mentions,
            entities=active_entities,
            aliases=active_aliases,
        )
        resolved_mentions, disguise_masks, pierced_disguises = self._apply_epistemic_disguises(
            raw_mentions=raw_mentions,
            aliases=active_aliases,
            pov_set=pov_set,
            pov_mode=pov_mode,
        )

        resolved_entity_ids = {m.entity_id for m in resolved_mentions}
        masked_real_ids = set(disguise_masks.keys())

        # 3. Filter relevant entities: present entities + 1-hop edge neighbors
        # Exclude edges belonging to a masked entity's true identity
        relevant_entity_ids = set(resolved_entity_ids)
        scene_edges: list[DirectedEdgeRecord] = []
        for edge in active_edges:
            if edge.relation_type == policy.belief_relation:
                continue
            if edge.source_id in masked_real_ids or edge.target_id in masked_real_ids:
                continue
            if edge.source_id in resolved_entity_ids or edge.target_id in resolved_entity_ids:
                scene_edges.append(edge)
                relevant_entity_ids.add(edge.source_id)
                relevant_entity_ids.add(edge.target_id)

        scene_entities = [e for e in active_entities if e.entity_id in relevant_entity_ids]

        # Fallback when no explicit/lexical mentions were found
        if not scene_entities and not disguise_masks:
            if fallback_on_empty_mentions == "all":
                scene_entities = active_entities
                scene_edges = [e for e in active_edges if e.relation_type != policy.belief_relation]
            elif fallback_on_empty_mentions == "pov_only" and pov_set:
                scene_entities = [e for e in active_entities if e.entity_id in pov_set]
                scene_edges = [
                    e
                    for e in active_edges
                    if e.relation_type != policy.belief_relation
                    and (e.source_id in pov_set or e.target_id in pov_set)
                ]

        # Fetch candidate episodes early so we know which causal episodes are known to POV
        candidate_episodes = (
            self.store.get_lineage_episodes(tl, cutoff_step)
            if (top_k_episodes > 0 or pov_set)
            else []
        )
        pov_known_episode_ids: set[str] = set()
        if pov_set and candidate_episodes:
            if pov_mode == "all":
                pov_known_episode_ids = {
                    ep.episode_id
                    for ep, _ in candidate_episodes
                    if pov_set.issubset(ep.known_by.keys())
                }
            else:
                pov_known_episode_ids = {
                    ep.episode_id
                    for ep, _ in candidate_episodes
                    if pov_set & set(ep.known_by.keys())
                }

        # Project entities and edges through POV epistemic lens
        projected_entities, projected_edges = self._project_pov_state(
            scene_entities=scene_entities,
            scene_edges=scene_edges,
            all_active_edges=active_edges,
            pov_set=pov_set,
            pov_mode=pov_mode,
            pov_known_episode_ids=pov_known_episode_ids,
            disguise_masks=disguise_masks,
            pierced_disguises=pierced_disguises,
            policy=policy,
        )

        # 4. Tier 5 Voice Exemplars retrieval (facade-first & disguise-aware)
        voice_exemplars: list[VoiceExemplar] = []
        speaker_ids = resolved_entity_ids | {f"masked:{rid}" for rid in masked_real_ids}
        speakers = (
            [e for e in projected_entities if e.entity_id in speaker_ids]
            if speaker_ids
            else projected_entities
        )
        if top_k_voice > 0 and speakers:
            in_scene_ids = resolved_entity_ids or {e.entity_id for e in projected_entities}
            for speaker in speakers:
                is_masked_speaker = speaker.entity_id.startswith("masked:")
                real_speaker_id = (
                    speaker.entity_id.removeprefix("masked:")
                    if is_masked_speaker
                    else speaker.entity_id
                )
                edge_pool = active_edges if is_masked_speaker else scene_edges
                voice_targets = self._resolve_speaker_voice_targets(
                    real_speaker_id=real_speaker_id,
                    is_masked_speaker=is_masked_speaker,
                    edges=edge_pool,
                    in_scene_ids=in_scene_ids,
                    resolved_mentions=resolved_mentions,
                    text=text,
                    policy=policy,
                )

                candidates = self.store.get_lineage_voice_exemplars(tl, cutoff_step, real_speaker_id)
                disguise_persona = speaker.state.get(policy.entity_disguise_key) if is_masked_speaker else None
                if disguise_persona:
                    persona_candidates = [
                        (ex, emb)
                        for ex, emb in candidates
                        if ex.metadata.get(policy.exemplar_persona_key, "").lower()
                        == str(disguise_persona).lower()
                    ]
                    if persona_candidates:
                        candidates = persona_candidates

                ranked = self.voice_bank.rank_exemplars_for_targets(
                    query_text=text,
                    candidates=candidates,
                    targets=voice_targets,
                    policy=policy,
                    top_k=top_k_voice,
                )
                voice_exemplars.extend(ranked)

        # 5. Tier 6 Episodic Memory retrieval
        episodes: list[EpisodeRecord] = []
        if top_k_episodes > 0 and candidate_episodes:
            pinned_episode_ids: set[str] = set()
            for ent in projected_entities:
                if ent.caused_by_episode_id:
                    pinned_episode_ids.add(ent.caused_by_episode_id)
            for edge in projected_edges:
                if edge.caused_by_episode_id:
                    pinned_episode_ids.add(edge.caused_by_episode_id)

            cue_entity_ids = resolved_entity_ids if resolved_entity_ids else {e.entity_id for e in projected_entities}
            episodes = self.voice_bank.rank_episodes(
                current_step=step,
                query_text=text,
                candidates=candidate_episodes,
                scene_entity_ids=cue_entity_ids,
                pinned_episode_ids=pinned_episode_ids,
                pov_entity_ids=pov_entity_ids,
                pov_mode=pov_mode,
                top_k=top_k_episodes,
            )

        return StepContext(
            timeline_id=tl,
            step=step,
            resolved_mentions=resolved_mentions,
            entities=projected_entities,
            active_edges=projected_edges,
            voice_exemplars=voice_exemplars,
            episodes=episodes,
        )

    def _resolve_speaker_voice_targets(
        self,
        *,
        real_speaker_id: str,
        is_masked_speaker: bool,
        edges: Sequence[DirectedEdgeRecord],
        in_scene_ids: set[str],
        resolved_mentions: Sequence[ResolvedMention],
        text: str,
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
    ) -> list[tuple[str | None, dict[str, str]]]:
        """Resolves ordered (addressee_id, constraints) dyadic voice targets for a speaker.

        Layers edge state per (speaker, target) pair using policy.edge_layer_order and
        prioritizes in-scene addressees deterministically by text mention order,
        edge recency, and canonical target ID.
        """
        outgoing = [
            e
            for e in edges
            if e.source_id == real_speaker_id
            and (not is_masked_speaker or e.relation_type == policy.facade_relation)
        ]
        if not outgoing:
            return [(None, {})]

        in_scene_edges = [
            e
            for e in outgoing
            if e.target_id in in_scene_ids and e.target_id != real_speaker_id
        ]
        active_pool = in_scene_edges if in_scene_edges else outgoing

        layer_ranks = {rel: idx + 1 for idx, rel in enumerate(policy.edge_layer_order)}

        def _layer_rank(rel_type: str) -> int:
            return layer_ranks.get(rel_type, 0)

        by_target: dict[str, list[DirectedEdgeRecord]] = {}
        for edge in active_pool:
            by_target.setdefault(edge.target_id, []).append(edge)

        text_lower = text.lower()
        mention_pos: dict[str, int] = {}
        for idx, m in enumerate(resolved_mentions):
            if m.entity_id in mention_pos:
                continue
            pos = text_lower.find(m.surface_text.lower()) if text_lower else -1
            mention_pos[m.entity_id] = pos if pos >= 0 else (1_000_000 + idx)

        constraint_keys = set(policy.hard_filter_keys) | set(policy.soft_boost_weights.keys())
        resolved_targets: list[tuple[tuple[int, int, str], str | None, dict[str, str]]] = []
        for target_id, target_edges in by_target.items():
            target_edges_sorted = sorted(
                target_edges,
                key=lambda e: (_layer_rank(e.relation_type), e.valid_from_step),
            )
            merged_constraints: dict[str, str] = {}
            max_step = 0
            for e in target_edges_sorted:
                for ck in constraint_keys:
                    if e.state.get(ck):
                        merged_constraints[ck] = str(e.state[ck])
                max_step = max(max_step, e.valid_from_step)

            sort_key = (mention_pos.get(target_id, 2_000_000), -max_step, target_id)
            resolved_targets.append((sort_key, target_id, merged_constraints))

        resolved_targets.sort(key=lambda item: item[0])
        return [(addr, constraints) for _, addr, constraints in resolved_targets]

    def _project_pov_state(
        self,
        *,
        scene_entities: Sequence[EntityRecord],
        scene_edges: Sequence[DirectedEdgeRecord],
        all_active_edges: Sequence[DirectedEdgeRecord],
        pov_set: set[str],
        pov_mode: Literal["any", "all"] = "any",
        pov_known_episode_ids: set[str],
        disguise_masks: dict[str, AliasRecord],
        pierced_disguises: dict[str, AliasRecord],
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
    ) -> tuple[list[EntityRecord], list[DirectedEdgeRecord]]:
        """Projects entities and edges through the active POV's epistemic lens."""
        # Map belief_state overrides for active POV(s): target_id -> merged belief state
        belief_overrides: dict[str, dict[str, Any]] = {}
        if pov_set:
            pov_edges = [
                edge
                for edge in all_active_edges
                if edge.relation_type == policy.belief_relation and edge.source_id in pov_set
            ]
            # Sort deterministically by source_id then valid_from_step
            pov_edges.sort(key=lambda e: (e.source_id, e.valid_from_step))
            for edge in pov_edges:
                belief_overrides.setdefault(edge.target_id, {}).update(edge.state)

        # Identify (source_id, target_id) pairs that have an active facade edge
        facade_pairs: set[tuple[str, str]] = {
            (edge.source_id, edge.target_id)
            for edge in scene_edges
            if edge.relation_type == policy.facade_relation
        }

        projected_entities: list[EntityRecord] = []
        for ent in scene_entities:
            ent_copy = ent.model_copy(deep=True)
            if ent_copy.entity_id in pierced_disguises:
                alias_rec = pierced_disguises[ent_copy.entity_id]
                ent_copy.disguised_as = alias_rec.persona_name or alias_rec.alias

            if pov_set:
                if ent_copy.entity_id not in pov_set or (len(pov_set) > 1 and pov_mode == "all"):
                    ent_copy.private_state = {}
                    if (
                        ent_copy.caused_by_episode_id
                        and ent_copy.caused_by_episode_id not in pov_known_episode_ids
                    ):
                        ent_copy.caused_by_episode_id = None
                        ent_copy.reason = None
                if ent_copy.entity_id in belief_overrides:
                    ent_copy.state = {**ent_copy.state, **belief_overrides[ent_copy.entity_id]}

            projected_entities.append(ent_copy)

        # Append masked persona records for unpierced disguises
        for real_id, alias_rec in disguise_masks.items():
            persona_label = alias_rec.persona_name or alias_rec.alias
            projected_entities.append(
                EntityRecord(
                    entity_id=f"masked:{real_id}",
                    kind=policy.masked_entity_kind,
                    canonical_name=persona_label,
                    state={policy.entity_disguise_key: persona_label},
                )
            )

        projected_edges: list[DirectedEdgeRecord] = []
        for edge in scene_edges:
            if (
                pov_set
                and edge.relation_type in policy.edge_layer_order
                and edge.relation_type != policy.facade_relation
                and edge.source_id not in pov_set
            ):
                if (edge.source_id, edge.target_id) in facade_pairs:
                    continue
            edge_copy = edge.model_copy(deep=True)
            if pov_set and edge_copy.source_id not in pov_set:
                if (
                    edge_copy.caused_by_episode_id
                    and edge_copy.caused_by_episode_id not in pov_known_episode_ids
                ):
                    edge_copy.caused_by_episode_id = None
                    edge_copy.reason = None
            projected_edges.append(edge_copy)

        return projected_entities, projected_edges

    def _apply_epistemic_disguises(
        self,
        *,
        raw_mentions: Sequence[ResolvedMention],
        aliases: Sequence[AliasRecord],
        pov_set: set[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
    ) -> tuple[list[ResolvedMention], dict[str, AliasRecord], dict[str, AliasRecord]]:
        """Applies POV-aware disguise masking & piercing on top of pluggable mention matches."""
        resolved: list[ResolvedMention] = []
        seen_entities: set[str] = set()
        disguise_masks: dict[str, AliasRecord] = {}
        pierced_disguises: dict[str, AliasRecord] = {}

        alias_by_key: dict[tuple[str, str], AliasRecord] = {
            (a.entity_id, a.alias.lower()): a for a in aliases if a.alias
        }
        alias_by_surface: dict[str, AliasRecord] = {
            a.alias.lower(): a for a in aliases if a.alias
        }

        # Process alias matches first (especially disguises) so canonical matches don't shadow them
        ordered_mentions = sorted(
            raw_mentions,
            key=lambda m: (0 if m.matched_via == "alias" else 1),
        )

        for m in ordered_mentions:
            surface_lower = m.surface_text.strip().lower()
            alias_rec = (
                alias_by_key.get((m.entity_id, surface_lower)) or alias_by_surface.get(surface_lower)
                if m.matched_via == "alias"
                else None
            )
            if alias_rec is not None and alias_rec.kind == "disguise":
                ent_id = alias_rec.entity_id
                is_scoped_disguise = alias_rec.known_to is not None and bool(pov_set)
                if is_scoped_disguise:
                    known_set = set(alias_rec.known_to or ())
                    pierced = (
                        pov_set.issubset(known_set)
                        if pov_mode == "all"
                        else bool(pov_set & known_set)
                    )
                    if not pierced:
                        masked_id = f"masked:{ent_id}"
                        if masked_id not in seen_entities:
                            seen_entities.add(masked_id)
                            disguise_masks[ent_id] = alias_rec
                            resolved.append(
                                ResolvedMention(
                                    surface_text=m.surface_text,
                                    entity_id=masked_id,
                                    matched_via="alias",
                                    confidence=m.confidence,
                                )
                            )
                        continue
                pierced_disguises[ent_id] = alias_rec

            if m.entity_id not in seen_entities:
                seen_entities.add(m.entity_id)
                resolved.append(m)

        return resolved, disguise_masks, pierced_disguises

    # --- Transactional Context Manager ---

    def step_transaction(
        self,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        top_k_episodes: int = 5,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        timeline_id: str | None = None,
        rerun_clean: bool = True,
    ) -> StepTransaction:
        return StepTransaction(
            self,
            step,
            text=text,
            mentions=mentions,
            top_k_voice=top_k_voice,
            top_k_episodes=top_k_episodes,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            edge_types=edge_types,
            timeline_id=timeline_id,
            rerun_clean=rerun_clean,
        )

    # --- State & Episode Mutation ---

    def apply_delta(self, step: int, delta: StepDelta, *, timeline_id: str | None = None) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        ep_embeddings: dict[str, bytes] = {}
        prepared_episodes: list[EpisodeRecord] = []
        for ep in delta.episodes:
            ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
            item = ep.model_copy(update={"episode_id": ep_id, "step": step, "timeline_id": tl})
            emb = self.voice_bank.embedder.embed_text(item.summary)
            ep_embeddings[ep_id] = emb.tobytes()
            prepared_episodes.append(item)

        delta_to_apply = delta.model_copy(update={"episodes": prepared_episodes}) if prepared_episodes else delta
        return self.store.apply_delta(step, delta_to_apply, timeline_id=tl, episode_embeddings=ep_embeddings)

    def index_episodes(
        self,
        step: int,
        episodes: Sequence[EpisodeRecord],
        *,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        to_store: list[tuple[EpisodeRecord, bytes]] = []
        indexed: list[EpisodeRecord] = []
        for ep in episodes:
            ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
            item = ep.model_copy(update={"episode_id": ep_id, "step": step, "timeline_id": tl})
            emb = self.voice_bank.embedder.embed_text(item.summary)
            to_store.append((item, emb.tobytes()))
            indexed.append(item)
        if to_store:
            self.store.insert_episodes(to_store, timeline_id=tl)
        return indexed

    def inform_entity(
        self,
        step: int,
        *,
        episode_id: str,
        entity_id: str,
        epistemic: Literal["direct", "told", "deceived"] = "told",
        believed_summary: str | None = None,
        timeline_id: str | None = None,
    ) -> None:
        tl = timeline_id or self.active_timeline
        self.store.record_epistemic(
            InformDelta(
                episode_id=episode_id,
                entity_id=entity_id,
                epistemic=epistemic,
                believed_summary=believed_summary,
            ),
            step=step,
            timeline_id=tl,
        )

    def query_episodes(
        self,
        *,
        cutoff_step: int | None = None,
        participants: Sequence[str] = (),
        match_all_participants: bool = False,
        predicate: str | None = None,
        location_id: str | None = None,
        known_to: Sequence[str] = (),
        min_salience: float = 0.0,
        step_range: tuple[int, int] | None = None,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        return self.store.query_episodes(
            timeline_id=tl,
            cutoff_step=cutoff_step,
            participants=participants,
            match_all_participants=match_all_participants,
            predicate=predicate,
            location_id=location_id,
            known_to=known_to,
            min_salience=min_salience,
            step_range=step_range,
        )

    def extract_dialogue_candidates(
        self,
        text: str,
        *,
        source_text: str = "",
        step: int | None = None,
        timeline_id: str | None = None,
    ) -> list[DialogueCandidate]:
        tl = timeline_id or self.active_timeline
        if step is None:
            tl_info = self.store.get_timeline(tl)
            cutoff_step = tl_info.head_step if tl_info is not None else 0
        else:
            cutoff_step = step

        active_entities = self.store.get_all_active_entities(tl, cutoff_step)
        active_aliases = self.store.get_all_active_aliases(tl, cutoff_step)

        def _resolve_speaker(before_win: str, after_win: str) -> tuple[str | None, str | None, str | None]:
            if not active_entities and not active_aliases:
                return None, None, None
            for window in (before_win, after_win):
                if not window:
                    continue
                mentions = self.mention_resolver.resolve(
                    text=window,
                    explicit_mentions=(),
                    entities=active_entities,
                    aliases=active_aliases,
                )
                if mentions:
                    # Order by earliest appearance in the attribution clause
                    win_lower = window.lower()
                    ordered = sorted(
                        mentions,
                        key=lambda m: (
                            idx if (idx := win_lower.find(m.surface_text.lower())) != -1 else len(win_lower)
                        ),
                    )
                    speaker_id = ordered[0].entity_id
                    speaker_surface = ordered[0].surface_text
                    addressee_id = next(
                        (m.entity_id for m in ordered[1:] if m.entity_id != speaker_id),
                        None,
                    )
                    return speaker_id, addressee_id, speaker_surface
            return None, None, None

        return self.voice_bank.gate.extract_candidates(
            text,
            source_text=source_text,
            speaker_resolver=_resolve_speaker,
        )

    def index_voice_exemplars(
        self,
        step: int,
        exemplars: Sequence[VoiceExemplar],
        *,
        bypass_gate: bool = False,
        timeline_id: str | None = None,
    ) -> list[VoiceExemplar]:
        tl = timeline_id or self.active_timeline
        indexed: list[VoiceExemplar] = []
        to_store: list[tuple[VoiceExemplar, bytes]] = []

        for ex in exemplars:
            if not bypass_gate and not self.voice_bank.gate.passes_quality_gate(ex):
                continue
            ex_id = ex.exemplar_id or str(uuid.uuid4())
            item = ex.model_copy(update={"exemplar_id": ex_id, "step": step, "timeline_id": tl})
            emb = self.voice_bank.embedder.embed_text(item.text)
            to_store.append((item, emb.tobytes()))
            indexed.append(item)

        if to_store:
            self.store.insert_voice_exemplars(to_store, timeline_id=tl)
        return indexed

    def _store_transaction(self) -> AbstractContextManager[None]:
        tx_fn = getattr(self.store, "transaction", None)
        return tx_fn() if callable(tx_fn) else contextlib.nullcontext()

    def commit_step(
        self,
        step: int,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        *,
        bypass_gate: bool = False,
        timeline_id: str | None = None,
    ) -> list[VoiceExemplar]:
        tl = timeline_id or self.active_timeline
        with self._store_transaction():
            if delta:
                self.apply_delta(step, delta, timeline_id=tl)
            if episodes:
                self.index_episodes(step, episodes, timeline_id=tl)
            indexed: list[VoiceExemplar] = []
            if exemplars:
                indexed = self.index_voice_exemplars(step, exemplars, bypass_gate=bypass_gate, timeline_id=tl)
            return indexed

    # --- Time-Travel Operations ---

    def rollback_to(self, step: int, *, timeline_id: str | None = None) -> int:
        tl = timeline_id or self.active_timeline
        return self.store.rollback_to(step, timeline_id=tl)

    def branch(
        self,
        at_step: int,
        name: str,
        *,
        from_timeline: str | None = None,
        switch: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineInfo:
        parent = from_timeline or self.active_timeline
        info = self.store.create_timeline(name, parent_id=parent, fork_step=at_step, metadata=metadata)
        if switch:
            self.active_timeline = name
        return info

    def checkout(self, timeline_id: str) -> None:
        if not self.store.get_timeline(timeline_id):
            raise ValueError(f"Timeline '{timeline_id}' does not exist.")
        self.active_timeline = timeline_id

    def list_timelines(self) -> list[TimelineInfo]:
        return self.store.list_timelines()

    def get_history(
        self,
        subject_key: str,
        *,
        timeline_id: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        tl = timeline_id or self.active_timeline
        return self.store.get_history(subject_key, timeline_id=tl, record_type=record_type)

    def diff_timelines(self, timeline_a: str, timeline_b: str, *, at_step: int) -> dict[str, Any]:
        return self.store.diff_timelines(timeline_a, timeline_b, at_step=at_step)

    def export_snapshot(
        self,
        target_path: str | Path,
        *,
        at_step: int,
        timeline_id: str | None = None,
    ) -> Path:
        tl = timeline_id or self.active_timeline
        return self.store.export_snapshot(target_path, at_step=at_step, timeline_id=tl)

    def format_context(self, ctx: StepContext) -> str:
        """Formats StepContext into prompt text using the active DomainPreset."""
        return self.preset.format_prompt_context(ctx)

    def build_extraction_prompt(self, ctx: StepContext, scene_text: str) -> str:
        """Builds a structured extraction prompt combining preset instructions, active IDs, and StepDelta schema."""
        entity_lines = [
            f"- {e.entity_id} (name: {e.canonical_name}, kind: {e.kind})"
            for e in ctx.entities
            if not e.entity_id.startswith("masked:")
        ]
        episode_lines = [
            f"- {ep.episode_id} (step {ep.step}): {ep.summary}"
            for ep in ctx.episodes
            if ep.episode_id
        ]
        schema_json = json.dumps(StepDelta.model_json_schema(), indent=2)

        parts = [
            self.preset.extraction_instructions(),
            "### Known Entity IDs (use exact entity_id values, not display names):",
            "\n".join(entity_lines) if entity_lines else "(none)",
        ]
        if episode_lines:
            parts.append("### Recent Episode IDs (for caused_by_episode_id or InformDelta):")
            parts.append("\n".join(episode_lines))
        parts.append("### Target StepDelta JSON Schema:")
        parts.append(schema_json)
        parts.append("### Scene Text to Extract From:")
        parts.append(scene_text)
        return "\n\n".join(parts)

    def normalize_delta(
        self,
        delta: StepDelta,
        *,
        step: int | None = None,
        timeline_id: str | None = None,
    ) -> StepDelta:
        """Normalizes canonical names/aliases into entity_ids and resolves intra-step episode IDs."""
        tl = timeline_id or self.active_timeline
        entities = self.store.get_all_active_entities(tl, step)
        aliases = self.store.get_all_active_aliases(tl, step)

        lookup: dict[str, str] = {}
        for e in entities:
            lookup[e.entity_id.lower()] = e.entity_id
            if e.canonical_name:
                lookup[e.canonical_name.lower()] = e.entity_id
        for a in aliases:
            if a.alias and a.kind != "disguise":
                lookup.setdefault(a.alias.lower(), a.entity_id)

        def _norm_id(raw: str) -> str:
            return lookup.get(raw.strip().lower(), raw)

        # Resolve intra-step episode IDs (assign concrete IDs to blank ones and map symbolic refs)
        ep_id_map: dict[str, str] = {}
        norm_episodes: list[EpisodeRecord] = []
        for ep in delta.episodes:
            old_id = ep.episode_id
            new_id = old_id or f"ep_{step if step is not None else 0}_{uuid.uuid4().hex[:8]}"
            if old_id:
                ep_id_map[old_id] = new_id
            norm_participants = {_norm_id(k): v for k, v in ep.participants.items()}
            norm_witnesses = [_norm_id(w) for w in ep.witnesses]
            norm_known_by = {_norm_id(k): v for k, v in ep.known_by.items()}
            norm_believed = {_norm_id(k): v for k, v in ep.believed_summaries.items()}
            norm_loc = _norm_id(ep.location_id) if ep.location_id else None
            norm_episodes.append(
                ep.model_copy(
                    update={
                        "episode_id": new_id,
                        "participants": norm_participants,
                        "witnesses": norm_witnesses,
                        "known_by": norm_known_by,
                        "believed_summaries": norm_believed,
                        "location_id": norm_loc,
                    }
                )
            )

        norm_entities = [
            ed.model_copy(
                update={
                    "entity_id": _norm_id(ed.entity_id),
                    "caused_by_episode_id": ep_id_map.get(ed.caused_by_episode_id, ed.caused_by_episode_id)
                    if ed.caused_by_episode_id
                    else None,
                }
            )
            for ed in delta.entities
        ]
        norm_edges = [
            rd.model_copy(
                update={
                    "source_id": _norm_id(rd.source_id),
                    "target_id": _norm_id(rd.target_id),
                    "caused_by_episode_id": ep_id_map.get(rd.caused_by_episode_id, rd.caused_by_episode_id)
                    if rd.caused_by_episode_id
                    else None,
                }
            )
            for rd in delta.edges
        ]
        norm_inform = [
            inf.model_copy(
                update={
                    "entity_id": _norm_id(inf.entity_id),
                    "episode_id": ep_id_map.get(inf.episode_id, inf.episode_id),
                }
            )
            for inf in delta.inform
        ]

        return StepDelta(
            entities=norm_entities,
            edges=norm_edges,
            episodes=norm_episodes,
            inform=norm_inform,
        )
