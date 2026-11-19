from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import re
import uuid
import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal, Sequence

from .embedders import (
    Embedder,
    EmbedderMismatchError,
    embed_texts_batch,
    get_embedder_info,
)
from .episodic import DEFAULT_EPISODIC_POLICY, EpisodeRanker
from .extractors import DeltaExtractor
from .hooks import EngineHooks
from .models import (
    AliasRecord,
    DialogueCandidate,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    EpistemicMode,
    InformDelta,
    ResolvedMention,
    StepContext,
    StepDelta,
    TimelineInfo,
    VoiceExemplar,
)
from .presets import DEFAULT_SCHEMA_POLICY, DomainPreset, InteractiveFictionPreset, SchemaPolicy
from .resolver import DefaultLexicalResolver, MentionResolver
from .store import CausalStore, MemoryStore, TimelineBranchConflictError
from .voice import IngestionGate, VoiceBank


class StepTransaction:
    """Context manager for reading state at `step` and atomically committing mutations on exit."""

    def __init__(
        self,
        engine: CanonArcEngine,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        top_k_episodes: int = 5,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        location_id: str | None = None,
        timeline_id: str | None = None,
        rerun_clean: bool = True,
        prev_text: str = "",
        next_text: str = "",
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
        self.location_id = location_id
        self.timeline_id = timeline_id or engine.active_timeline
        self.rerun_clean = rerun_clean
        self.prev_text = prev_text
        self.next_text = next_text
        self._context: StepContext | None = None
        self._staged_delta: StepDelta | None = None
        self.last_raw_delta: StepDelta | None = None
        self._staged_exemplars: list[VoiceExemplar] = []
        self._staged_episodes: list[EpisodeRecord] = []
        self._bypass_gate = False

    def __enter__(self) -> StepTransaction:
        if self.rerun_clean:
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
            prev_text=self.prev_text,
            mentions=self.mentions,
            top_k_voice=self.top_k_voice,
            top_k_episodes=self.top_k_episodes,
            pov_entity_ids=self.pov_entity_ids,
            pov_mode=self.pov_mode,
            edge_types=self.edge_types,
            location_id=self.location_id,
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
        prev_text: str | None = None,
        next_text: str | None = None,
    ) -> StepDelta:
        """Runs the configured DeltaExtractor on scene_text, normalizes IDs, and stages the StepDelta."""
        active_extractor = extractor or self.engine.extractor
        if active_extractor is None:
            raise ValueError("No DeltaExtractor provided to StepTransaction or CanonArcEngine.")
        effective_prev = self.prev_text if prev_text is None else prev_text
        effective_next = self.next_text if next_text is None else next_text
        prompt = self.engine.build_extraction_prompt(
            self.context,
            scene_text,
            prev_text=effective_prev,
            next_text=effective_next,
        )
        raw_delta = active_extractor.extract_delta(prompt, scene_text)
        self.last_raw_delta = raw_delta
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


class TimelineSession:
    """Concurrency-safe, immutable-timeline session view bound to a single timeline_id."""

    def __init__(self, engine: CanonArcEngine, timeline_id: str) -> None:
        self.engine = engine
        self.timeline_id = timeline_id

    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str | AliasRecord] = (),
        step: int = 0,
    ) -> None:
        self.engine.register_entity(entity, aliases=aliases, step=step, timeline_id=self.timeline_id)

    def seed_entities(self, entities: Sequence[EntityRecord], *, step: int = 0) -> None:
        self.engine.seed_entities(entities, step=step, timeline_id=self.timeline_id)

    def retrieve_context(
        self,
        step: int,
        *,
        text: str = "",
        prev_text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        top_k_episodes: int = 5,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        location_id: str | None = None,
        inclusive: bool = False,
        fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only",
        story_time: int | float | str | None = None,
        chapter_id: str | None = None,
        scene_id: str | None = None,
    ) -> StepContext:
        return self.engine.retrieve_context(
            step,
            text=text,
            prev_text=prev_text,
            mentions=mentions,
            top_k_voice=top_k_voice,
            top_k_episodes=top_k_episodes,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            edge_types=edge_types,
            location_id=location_id,
            inclusive=inclusive,
            timeline_id=self.timeline_id,
            fallback_on_empty_mentions=fallback_on_empty_mentions,
            story_time=story_time,
            chapter_id=chapter_id,
            scene_id=scene_id,
        )

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
        location_id: str | None = None,
        rerun_clean: bool = True,
        prev_text: str = "",
        next_text: str = "",
    ) -> StepTransaction:
        return self.engine.step_transaction(
            step,
            text=text,
            mentions=mentions,
            top_k_voice=top_k_voice,
            top_k_episodes=top_k_episodes,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            edge_types=edge_types,
            location_id=location_id,
            timeline_id=self.timeline_id,
            rerun_clean=rerun_clean,
            prev_text=prev_text,
            next_text=next_text,
        )

    def apply_delta(self, step: int, delta: StepDelta) -> list[EpisodeRecord]:
        return self.engine.apply_delta(step, delta, timeline_id=self.timeline_id)

    def index_episodes(self, step: int, episodes: Sequence[EpisodeRecord]) -> list[EpisodeRecord]:
        return self.engine.index_episodes(step, episodes, timeline_id=self.timeline_id)

    def inform_entity(
        self,
        step: int,
        *,
        episode_id: str,
        entity_id: str,
        epistemic: EpistemicMode = "told",
        believed_summary: str | None = None,
    ) -> None:
        self.engine.inform_entity(
            step,
            episode_id=episode_id,
            entity_id=entity_id,
            epistemic=epistemic,
            believed_summary=believed_summary,
            timeline_id=self.timeline_id,
        )

    def query_episodes(self, **kwargs: Any) -> list[EpisodeRecord]:
        return self.engine.query_episodes(timeline_id=self.timeline_id, **kwargs)

    def extract_dialogue_candidates(
        self,
        text: str,
        *,
        source_text: str = "",
        step: int | None = None,
        prev_text: str = "",
        next_text: str = "",
    ) -> list[DialogueCandidate]:
        return self.engine.extract_dialogue_candidates(
            text,
            source_text=source_text,
            step=step,
            timeline_id=self.timeline_id,
            prev_text=prev_text,
            next_text=next_text,
        )

    def index_voice_exemplars(
        self,
        step: int,
        exemplars: Sequence[VoiceExemplar],
        *,
        bypass_gate: bool = False,
    ) -> list[VoiceExemplar]:
        return self.engine.index_voice_exemplars(
            step, exemplars, bypass_gate=bypass_gate, timeline_id=self.timeline_id
        )

    def commit_step(
        self,
        step: int,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        episodes: Sequence[EpisodeRecord] = (),
        *,
        bypass_gate: bool = False,
    ) -> list[VoiceExemplar]:
        return self.engine.commit_step(
            step,
            delta=delta,
            exemplars=exemplars,
            episodes=episodes,
            bypass_gate=bypass_gate,
            timeline_id=self.timeline_id,
        )

    def consolidate_episodes(self, step: int, **kwargs: Any) -> EpisodeRecord | None:
        return self.engine.consolidate_episodes(step, timeline_id=self.timeline_id, **kwargs)

    def rollback_to(self, step: int) -> int:
        return self.engine.rollback_to(step, timeline_id=self.timeline_id)

    def branch(
        self,
        at_step: int,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineSession:
        self.engine.branch(at_step, name, from_timeline=self.timeline_id, metadata=metadata)
        return TimelineSession(self.engine, name)

    def get_history(self, subject_key: str, *, record_type: str | None = None) -> list[dict[str, Any]]:
        return self.engine.get_history(
            subject_key, timeline_id=self.timeline_id, record_type=record_type
        )

    def export_snapshot(self, target_path: str | Path, *, at_step: int) -> Path:
        return self.engine.export_snapshot(target_path, at_step=at_step, timeline_id=self.timeline_id)

    def normalize_delta(self, delta: StepDelta, *, step: int | None = None) -> StepDelta:
        return self.engine.normalize_delta(delta, step=step, timeline_id=self.timeline_id)

    def format_context(self, ctx: StepContext) -> str:
        return self.engine.format_context(ctx)

    def build_extraction_prompt(
        self,
        ctx: StepContext,
        scene_text: str,
        *,
        prev_text: str = "",
        next_text: str = "",
    ) -> str:
        return self.engine.build_extraction_prompt(
            ctx, scene_text, prev_text=prev_text, next_text=next_text
        )


class CanonArcEngine:
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
        episode_ranker: EpisodeRanker | None = None,
        hooks: Sequence[EngineHooks] = (),
        on_embedder_mismatch: Literal["raise", "warn", "reindex", "ignore"] = "warn",
    ) -> None:
        self.store: MemoryStore = store if store is not None else CausalStore(db_path)
        self._default_timeline = timeline_id
        self._timeline_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"canonarc_timeline_{id(self)}", default=None
        )
        self.voice_bank = VoiceBank(
            embedder=embedder, gate=gate, episode_ranker=episode_ranker
        )
        self.preset: DomainPreset = preset or InteractiveFictionPreset()
        self.mention_resolver: MentionResolver = mention_resolver or DefaultLexicalResolver()
        self.extractor: DeltaExtractor | None = extractor
        self._hooks: list[EngineHooks] = list(hooks)
        self._on_embedder_mismatch = on_embedder_mismatch
        self._embedder_verified = False
        self._check_embedder_compatibility()

    @property
    def active_timeline(self) -> str:
        scoped = self._timeline_var.get()
        return scoped if scoped is not None else self._default_timeline

    @active_timeline.setter
    def active_timeline(self, value: str) -> None:
        self._default_timeline = value
        if self._timeline_var.get() is not None:
            self._timeline_var.set(value)

    @contextlib.contextmanager
    def use_timeline(self, timeline_id: str) -> Iterator[TimelineSession]:
        """Task/thread-local context manager scoping active_timeline without mutating other requests."""
        if not self.store.get_timeline(timeline_id):
            raise ValueError(f"Timeline '{timeline_id}' does not exist.")
        token = self._timeline_var.set(timeline_id)
        try:
            yield TimelineSession(self, timeline_id)
        finally:
            self._timeline_var.reset(token)

    def session(self, timeline_id: str | None = None) -> TimelineSession:
        """Returns a concurrency-safe TimelineSession bound to timeline_id."""
        tl = timeline_id or self.active_timeline
        if not self.store.get_timeline(tl):
            raise ValueError(f"Timeline '{tl}' does not exist.")
        return TimelineSession(self, tl)

    def add_hook(self, hook: EngineHooks) -> None:
        self._hooks.append(hook)

    def remove_hook(self, hook: EngineHooks) -> None:
        self._hooks = [h for h in self._hooks if h is not hook]

    @property
    def _policy(self) -> SchemaPolicy:
        return getattr(self.preset, "policy", DEFAULT_SCHEMA_POLICY)

    def close(self) -> None:
        self.store.close()

    def _check_embedder_compatibility(self) -> None:
        """Verifies that the active embedder matches any embedder/dimension recorded in the store."""
        get_meta = getattr(self.store, "get_embedding_metadata", None)
        if not callable(get_meta):
            return
        stored_id, stored_dim = get_meta()
        if stored_id is None and stored_dim is None:
            return
        current_id, current_dim = get_embedder_info(self.voice_bank.embedder)
        id_mismatch = stored_id is not None and stored_id != current_id
        dim_mismatch = stored_dim is not None and current_dim is not None and stored_dim != current_dim
        if not id_mismatch and not dim_mismatch:
            self._embedder_verified = True
            return

        msg = (
            f"Store embeddings were created with embedder={stored_id!r}, dim={stored_dim}, "
            f"but engine was initialized with embedder={current_id!r}, dim={current_dim}."
        )
        policy = self._on_embedder_mismatch
        if policy == "raise":
            raise EmbedderMismatchError(msg)
        if policy == "reindex":
            self.reindex_embeddings()
            return
        if policy == "warn":
            warnings.warn(
                f"{msg} Call engine.reindex_embeddings() to migrate stored vectors.",
                UserWarning,
                stacklevel=3,
            )

    def _record_embedder_metadata(self, vectors: Sequence[Any]) -> None:
        """Records embedder_id and vector dimension in store_metadata on first write."""
        if self._embedder_verified or not vectors:
            return
        set_meta = getattr(self.store, "set_embedding_metadata", None)
        get_meta = getattr(self.store, "get_embedding_metadata", None)
        if not callable(set_meta) or not callable(get_meta):
            self._embedder_verified = True
            return
        dim = int(len(vectors[0]))
        if dim <= 0:
            return
        stored_id, stored_dim = get_meta()
        current_id, _ = get_embedder_info(self.voice_bank.embedder)
        if stored_id is None and stored_dim is None:
            set_meta(current_id, dim)
            self._embedder_verified = True
            return
        if (stored_id is not None and stored_id != current_id) or (
            stored_dim is not None and stored_dim != dim
        ):
            msg = (
                f"Store embeddings were created with embedder={stored_id!r}, dim={stored_dim}, "
                f"but active embedder produced embedder={current_id!r}, dim={dim}."
            )
            if self._on_embedder_mismatch == "raise":
                raise EmbedderMismatchError(msg)
            if self._on_embedder_mismatch == "warn":
                warnings.warn(
                    f"{msg} Call engine.reindex_embeddings() to migrate stored vectors.",
                    UserWarning,
                    stacklevel=3,
                )
        self._embedder_verified = True

    def reindex_embeddings(self, *, batch_size: int = 128) -> dict[str, int]:
        """Re-embeds all stored episodes and voice exemplars in batches using the active embedder."""
        list_ep = getattr(self.store, "list_all_episodes_for_reembed", None)
        list_vo = getattr(self.store, "list_all_voice_exemplars_for_reembed", None)
        upd_ep = getattr(self.store, "update_episode_embeddings", None)
        upd_vo = getattr(self.store, "update_voice_embeddings", None)
        set_meta = getattr(self.store, "set_embedding_metadata", None)
        if not all(callable(fn) for fn in (list_ep, list_vo, upd_ep, upd_vo, set_meta)):
            raise NotImplementedError("Active store backend does not support reindex_embeddings().")

        ep_rows: list[tuple[str, str, str]] = list_ep()
        vo_rows: list[tuple[str, str, str]] = list_vo()
        step_size = max(1, batch_size)
        detected_dim: int | None = None

        ep_updates: list[tuple[str, str, bytes]] = []
        for i in range(0, len(ep_rows), step_size):
            chunk = ep_rows[i : i + step_size]
            vecs = embed_texts_batch(self.voice_bank.embedder, [r[2] for r in chunk])
            if vecs and detected_dim is None:
                detected_dim = int(len(vecs[0]))
            for (tl, eid, _), vec in zip(chunk, vecs):
                ep_updates.append((tl, eid, vec.tobytes()))

        vo_updates: list[tuple[str, str, bytes]] = []
        for i in range(0, len(vo_rows), step_size):
            chunk = vo_rows[i : i + step_size]
            vecs = embed_texts_batch(self.voice_bank.embedder, [r[2] for r in chunk])
            if vecs and detected_dim is None:
                detected_dim = int(len(vecs[0]))
            for (tl, xid, _), vec in zip(chunk, vecs):
                vo_updates.append((tl, xid, vec.tobytes()))

        current_id, known_dim = get_embedder_info(self.voice_bank.embedder)
        final_dim = detected_dim or known_dim
        if final_dim is None:
            probe = self.voice_bank.embedder.embed_text("probe")
            final_dim = int(len(probe))

        with self._store_transaction():
            n_ep = upd_ep(ep_updates)
            n_vo = upd_vo(vo_updates)
            set_meta(current_id, final_dim)
        self._embedder_verified = True
        return {"episodes": n_ep, "voice_exemplars": n_vo}

    def __enter__(self) -> CanonArcEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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
        prev_text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        top_k_episodes: int = 5,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        edge_types: Sequence[str] | None = None,
        location_id: str | None = None,
        inclusive: bool = False,
        timeline_id: str | None = None,
        fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only",
        story_time: int | float | str | None = None,
        chapter_id: str | None = None,
        scene_id: str | None = None,
    ) -> StepContext:
        tl = timeline_id or self.active_timeline
        policy = self._policy
        cutoff_step = step if inclusive else (step - 1)

        # 1. Single-pass fetch of active entities, aliases, and edges (with custom MemoryStore fallback)
        get_graph_fn = getattr(self.store, "get_active_graph_state", None)
        if callable(get_graph_fn):
            active_entities, active_aliases, active_edges = get_graph_fn(
                tl, cutoff_step, edge_types=edge_types
            )
        else:
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

        # Read-Only Coreference Halo: if target chunk has < 2 characters (e.g. pronoun-only),
        # resolve fallback mentions from prev_text so active characters/edges stay in context.
        entity_by_id = {e.entity_id: e for e in active_entities}
        primary_char_ids = {
            m.entity_id
            for m in resolved_mentions
            if m.entity_id.startswith("masked:")
            or (
                (ent := entity_by_id.get(m.entity_id)) is None
                or ent.kind == "character"
            )
        }
        if prev_text.strip() and len(primary_char_ids) < 2:
            halo_raw = self.mention_resolver.resolve(
                text=prev_text,
                explicit_mentions=(),
                entities=active_entities,
                aliases=active_aliases,
            )
            halo_mentions, halo_masks, halo_pierced = self._apply_epistemic_disguises(
                raw_mentions=halo_raw,
                aliases=active_aliases,
                pov_set=pov_set,
                pov_mode=pov_mode,
            )
            seen_resolved_ids = {m.entity_id for m in resolved_mentions}
            for m in halo_mentions:
                real_id = m.entity_id.removeprefix("masked:")
                if (
                    m.entity_id in seen_resolved_ids
                    or real_id in seen_resolved_ids
                    or f"masked:{real_id}" in seen_resolved_ids
                ):
                    continue
                seen_resolved_ids.add(m.entity_id)
                resolved_mentions.append(m)
                if real_id in halo_masks:
                    disguise_masks.setdefault(real_id, halo_masks[real_id])
            for k, v in halo_pierced.items():
                if k not in disguise_masks:
                    pierced_disguises.setdefault(k, v)

        resolved_entity_ids = {m.entity_id for m in resolved_mentions}
        masked_real_ids = set(disguise_masks.keys())

        # 3. Separate co-present scene entities from 1-hop social graph neighbors
        present_entity_ids = {
            eid for eid in resolved_entity_ids if not eid.startswith("masked:")
        }
        active_location_ids: set[str] = set()
        if location_id is not None:
            active_location_ids.add(location_id)
        for ent in active_entities:
            if ent.entity_id in present_entity_ids and ent.kind == "location":
                active_location_ids.add(ent.entity_id)
        if active_location_ids:
            for ent in active_entities:
                if ent.entity_id in masked_real_ids:
                    continue
                ent_loc = ent.state.get("location_id") or ent.base_attributes.get("location_id")
                if ent_loc in active_location_ids:
                    present_entity_ids.add(ent.entity_id)

        if not present_entity_ids and not disguise_masks:
            if fallback_on_empty_mentions == "all":
                present_entity_ids = {
                    e.entity_id for e in active_entities if e.entity_id not in masked_real_ids
                }
            elif fallback_on_empty_mentions == "pov_only" and pov_set:
                present_entity_ids = {
                    eid for eid in pov_set if eid not in masked_real_ids
                }

        in_scene_edges: list[DirectedEdgeRecord] = []
        background_edges: list[DirectedEdgeRecord] = []
        neighbor_entity_ids: set[str] = set()
        for edge in active_edges:
            if edge.relation_type == policy.belief_relation:
                continue
            if edge.source_id in masked_real_ids or edge.target_id in masked_real_ids:
                continue
            src_present = edge.source_id in present_entity_ids
            tgt_present = edge.target_id in present_entity_ids
            if src_present and tgt_present:
                in_scene_edges.append(edge)
            elif src_present or tgt_present:
                background_edges.append(edge)
                if not src_present:
                    neighbor_entity_ids.add(edge.source_id)
                if not tgt_present:
                    neighbor_entity_ids.add(edge.target_id)

        scene_entities = [e for e in active_entities if e.entity_id in present_entity_ids]
        neighbor_entities = [e for e in active_entities if e.entity_id in neighbor_entity_ids]

        pov_known_episode_ids: set[str] = set()
        fallback_lineage_episodes: list[tuple[EpisodeRecord, bytes]] | None = None
        if pov_set:
            provenance_ep_ids = {
                eid
                for eid in (
                    *(e.caused_by_episode_id for e in scene_entities),
                    *(e.caused_by_episode_id for e in neighbor_entities),
                    *(e.caused_by_episode_id for e in in_scene_edges),
                    *(e.caused_by_episode_id for e in background_edges),
                )
                if eid
            }
            if provenance_ep_ids:
                get_known_fn = getattr(self.store, "get_known_episode_ids", None)
                if callable(get_known_fn):
                    pov_known_episode_ids = get_known_fn(
                        tl,
                        cutoff_step,
                        pov_entity_ids or (),
                        pov_mode=pov_mode,
                        episode_ids=sorted(provenance_ep_ids),
                    )
                else:
                    fallback_lineage_episodes = self.store.get_lineage_episodes(tl, cutoff_step)
                    if pov_mode == "all":
                        pov_known_episode_ids = {
                            ep.episode_id
                            for ep, _ in fallback_lineage_episodes
                            if pov_set.issubset(ep.known_by.keys())
                        }
                    else:
                        pov_known_episode_ids = {
                            ep.episode_id
                            for ep, _ in fallback_lineage_episodes
                            if pov_set & set(ep.known_by.keys())
                        }

        projected_entities, projected_edges = self._project_pov_state(
            scene_entities=scene_entities,
            scene_edges=in_scene_edges,
            all_active_edges=active_edges,
            pov_set=pov_set,
            pov_mode=pov_mode,
            pov_known_episode_ids=pov_known_episode_ids,
            disguise_masks=disguise_masks,
            pierced_disguises=pierced_disguises,
            policy=policy,
        )
        projected_related_entities, projected_background_edges = self._project_pov_state(
            scene_entities=neighbor_entities,
            scene_edges=background_edges,
            all_active_edges=active_edges,
            pov_set=pov_set,
            pov_mode=pov_mode,
            pov_known_episode_ids=pov_known_episode_ids,
            disguise_masks={},
            pierced_disguises=pierced_disguises,
            policy=policy,
        )

        # 4. Tier 5 Voice Exemplars retrieval
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
                edge_pool = active_edges if is_masked_speaker else (in_scene_edges + background_edges)
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
                disguise_persona = (
                    speaker.state.get(policy.entity_disguise_key) if is_masked_speaker else None
                )
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
        if top_k_episodes > 0:
            pinned_episode_ids: set[str] = set()
            for ent in projected_entities:
                if ent.caused_by_episode_id:
                    pinned_episode_ids.add(ent.caused_by_episode_id)
            for edge in projected_edges:
                if edge.caused_by_episode_id:
                    pinned_episode_ids.add(edge.caused_by_episode_id)

            cue_entity_ids = (
                resolved_entity_ids
                if resolved_entity_ids
                else {e.entity_id for e in projected_entities}
            )
            if fallback_lineage_episodes is not None:
                candidate_episodes = fallback_lineage_episodes
            else:
                candidate_limit = max(top_k_episodes * 20, 200)
                sql_cue_ids = sorted(cue_entity_ids | active_location_ids)
                try:
                    candidate_episodes = self.store.get_lineage_episodes(
                        tl,
                        cutoff_step,
                        limit=candidate_limit,
                        pov_entity_ids=pov_entity_ids,
                        pov_mode=pov_mode,
                        cue_entity_ids=sql_cue_ids,
                        pinned_episode_ids=sorted(pinned_episode_ids),
                        query_text=text,
                    )
                except TypeError:
                    candidate_episodes = self.store.get_lineage_episodes(tl, cutoff_step)

            if candidate_episodes:
                episodic_policy = getattr(policy, "episodic_policy", DEFAULT_EPISODIC_POLICY)
                episodes = self.voice_bank.rank_episodes(
                    current_step=step,
                    query_text=text,
                    candidates=candidate_episodes,
                    scene_entity_ids=cue_entity_ids,
                    pinned_episode_ids=pinned_episode_ids,
                    pov_entity_ids=pov_entity_ids,
                    pov_mode=pov_mode,
                    policy=episodic_policy,
                    top_k=top_k_episodes,
                )

        return StepContext(
            timeline_id=tl,
            step=step,
            story_time=story_time,
            chapter_id=chapter_id,
            scene_id=scene_id,
            resolved_mentions=resolved_mentions,
            entities=projected_entities,
            active_edges=projected_edges,
            related_entities=projected_related_entities,
            background_edges=projected_background_edges,
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
        """Resolves ordered (addressee_id, constraints) dyadic voice targets for a speaker."""
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
        belief_overrides: dict[str, dict[str, Any]] = {}
        if pov_set:
            pov_edges = [
                edge
                for edge in all_active_edges
                if edge.relation_type == policy.belief_relation and edge.source_id in pov_set
            ]
            pov_edges.sort(key=lambda e: (e.source_id, e.valid_from_step))
            if len(pov_set) > 1 and pov_mode == "all":
                per_pov_beliefs: dict[str, dict[str, dict[str, Any]]] = {pid: {} for pid in pov_set}
                for edge in pov_edges:
                    per_pov_beliefs[edge.source_id].setdefault(edge.target_id, {}).update(edge.state)
                common_targets = set.intersection(
                    *(set(per_pov_beliefs[pid].keys()) for pid in pov_set)
                )
                for target_id in common_targets:
                    first_pid = next(iter(pov_set))
                    shared_state: dict[str, Any] = {}
                    for k, v in per_pov_beliefs[first_pid][target_id].items():
                        if all(
                            k in per_pov_beliefs[pid][target_id]
                            and per_pov_beliefs[pid][target_id][k] == v
                            for pid in pov_set
                        ):
                            shared_state[k] = v
                    if shared_state:
                        belief_overrides[target_id] = shared_state
            else:
                for edge in pov_edges:
                    belief_overrides.setdefault(edge.target_id, {}).update(edge.state)

        facade_pairs: set[tuple[str, str]] = {
            (edge.source_id, edge.target_id)
            for edge in scene_edges
            if edge.relation_type == policy.facade_relation
        }

        is_multipov_all = len(pov_set) > 1 and pov_mode == "all"
        projected_entities: list[EntityRecord] = []
        for ent in scene_entities:
            ent_copy = ent.model_copy(deep=True)
            if ent_copy.entity_id in pierced_disguises:
                alias_rec = pierced_disguises[ent_copy.entity_id]
                ent_copy.disguised_as = alias_rec.persona_name or alias_rec.alias

            if pov_set:
                if ent_copy.entity_id not in pov_set or is_multipov_all:
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
            is_self_edge = edge.source_id in pov_set and not is_multipov_all
            if (
                pov_set
                and edge.relation_type in policy.edge_layer_order
                and edge.relation_type != policy.facade_relation
                and not is_self_edge
            ):
                if (edge.source_id, edge.target_id) in facade_pairs:
                    continue
            edge_copy = edge.model_copy(deep=True)
            if pov_set and not is_self_edge:
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

        # Process alias matches (including fuzzy-matched aliases) first so canonical mentions cannot shadow them
        ordered_mentions = sorted(
            raw_mentions,
            key=lambda m: (0 if (m.matched_via == "alias" or m.matched_source == "alias") else 1),
        )

        for m in ordered_mentions:
            is_alias_match = m.matched_via == "alias" or m.matched_source == "alias"
            lookup_key = (m.matched_phrase or m.surface_text).strip().lower()
            alias_rec = (
                alias_by_key.get((m.entity_id, lookup_key)) or alias_by_surface.get(lookup_key)
                if is_alias_match
                else None
            )
            if alias_rec is not None and alias_rec.kind == "disguise":
                ent_id = alias_rec.entity_id
                if alias_rec.known_to is not None and pov_set:
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
                                    matched_source="alias",
                                    matched_phrase=alias_rec.alias.lower(),
                                )
                            )
                        continue
                pierced_disguises[ent_id] = alias_rec

            if m.entity_id in disguise_masks:
                continue

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
        location_id: str | None = None,
        timeline_id: str | None = None,
        rerun_clean: bool = True,
        prev_text: str = "",
        next_text: str = "",
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
            location_id=location_id,
            timeline_id=timeline_id,
            rerun_clean=rerun_clean,
            prev_text=prev_text,
            next_text=next_text,
        )

    # --- State & Episode Mutation ---

    def apply_delta(self, step: int, delta: StepDelta, *, timeline_id: str | None = None) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        prepared_episodes: list[EpisodeRecord] = []
        for ep in delta.episodes:
            ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
            updates: dict[str, Any] = {"episode_id": ep_id, "step": step, "timeline_id": tl}
            if ep.story_time is None and delta.story_time is not None:
                updates["story_time"] = delta.story_time
            if ep.chapter_id is None and delta.chapter_id is not None:
                updates["chapter_id"] = delta.chapter_id
            if ep.scene_id is None and delta.scene_id is not None:
                updates["scene_id"] = delta.scene_id
            prepared_episodes.append(ep.model_copy(update=updates))

        ep_embeddings: dict[str, bytes] = {}
        if prepared_episodes:
            vectors = embed_texts_batch(
                self.voice_bank.embedder, [ep.summary for ep in prepared_episodes]
            )
            self._record_embedder_metadata(vectors)
            for ep, vec in zip(prepared_episodes, vectors):
                ep_embeddings[ep.episode_id] = vec.tobytes()

        delta_updates: dict[str, Any] = {}
        if prepared_episodes:
            delta_updates["episodes"] = prepared_episodes
        if delta.is_flashback:
            # Flashbacks must not overwrite already-established present-day entity/edge states.
            existing_ids = {
                e.entity_id for e in self.store.get_all_active_entities(tl, step)
            }
            delta_updates["entities"] = [
                ed for ed in delta.entities if not ed.remove and ed.entity_id not in existing_ids
            ]
            delta_updates["edges"] = []

        delta_to_apply = delta.model_copy(update=delta_updates) if delta_updates else delta
        return self.store.apply_delta(
            step, delta_to_apply, timeline_id=tl, episode_embeddings=ep_embeddings
        )

    def index_episodes(
        self,
        step: int,
        episodes: Sequence[EpisodeRecord],
        *,
        timeline_id: str | None = None,
    ) -> list[EpisodeRecord]:
        tl = timeline_id or self.active_timeline
        if not episodes:
            return []
        indexed: list[EpisodeRecord] = []
        for ep in episodes:
            ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
            indexed.append(ep.model_copy(update={"episode_id": ep_id, "step": step, "timeline_id": tl}))

        vectors = embed_texts_batch(self.voice_bank.embedder, [ep.summary for ep in indexed])
        self._record_embedder_metadata(vectors)
        to_store = [(item, vec.tobytes()) for item, vec in zip(indexed, vectors)]
        self.store.insert_episodes(to_store, timeline_id=tl)
        return indexed

    def consolidate_episodes(
        self,
        step: int,
        *,
        before_step: int,
        summary: str,
        max_salience: float = 4.0,
        predicate: str = "consolidated_summary",
        episode_id: str | None = None,
        chapter_id: str | None = None,
        salience: float = 7.5,
        timeline_id: str | None = None,
    ) -> EpisodeRecord | None:
        """Compacts old low-salience episodes (< before_step) into a single summary episode."""
        tl = timeline_id or self.active_timeline
        candidates = self.store.query_episodes(
            timeline_id=tl,
            cutoff_step=before_step - 1,
        )
        targets = [
            ep
            for ep in candidates
            if ep.salience <= max_salience
            and ep.predicate != predicate
            and (chapter_id is None or ep.chapter_id == chapter_id)
        ]
        if not targets:
            return None

        participants: dict[str, str] = {}
        witnesses: list[str] = []
        seen_witnesses: set[str] = set()
        known_by: dict[str, Literal["direct", "told", "deceived"]] = {}
        for ep in targets:
            for eid, role in ep.participants.items():
                participants.setdefault(eid, role)
            for wid in ep.witnesses:
                if wid not in seen_witnesses:
                    seen_witnesses.add(wid)
                    witnesses.append(wid)
            for eid, mode in ep.known_by.items():
                if mode == "deceived":
                    continue
                if known_by.get(eid) != "direct":
                    known_by[eid] = mode

        summary_ep = EpisodeRecord(
            episode_id=episode_id or f"ep_consolidated_{step}_{uuid.uuid4().hex[:8]}",
            timeline_id=tl,
            step=step,
            predicate=predicate,
            summary=summary,
            participants=participants,
            witnesses=witnesses,
            known_by=known_by,
            chapter_id=chapter_id,
            consolidated_from=[ep.episode_id for ep in targets if ep.episode_id],
            salience=salience,
        )
        with self._store_transaction():
            committed = self.index_episodes(step, [summary_ep], timeline_id=tl)[0]
            delete_fn = getattr(self.store, "delete_episodes", None)
            if callable(delete_fn):
                delete_fn([ep.episode_id for ep in targets], timeline_id=tl)
        return committed

    def inform_entity(
        self,
        step: int,
        *,
        episode_id: str,
        entity_id: str,
        epistemic: EpistemicMode = "told",
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
        prev_text: str = "",
        next_text: str = "",
    ) -> list[DialogueCandidate]:
        tl = timeline_id or self.active_timeline
        if step is None:
            tl_info = self.store.get_timeline(tl)
            cutoff_step = tl_info.head_step if tl_info is not None else 0
        else:
            cutoff_step = step

        active_entities = self.store.get_all_active_entities(tl, cutoff_step)
        active_aliases = self.store.get_all_active_aliases(tl, cutoff_step)
        speaker_entities = [
            e for e in active_entities if e.kind not in ("location", "artifact", "term")
        ]
        speaker_entity_ids = {e.entity_id for e in speaker_entities}
        speaker_aliases = [
            a for a in active_aliases if not speaker_entity_ids or a.entity_id in speaker_entity_ids
        ]

        def _resolve_in_segment(segment: str) -> tuple[str | None, str | None, str | None]:
            mentions = self.mention_resolver.resolve(
                text=segment,
                explicit_mentions=(),
                entities=speaker_entities,
                aliases=speaker_aliases,
            )
            if not mentions:
                return None, None, None
            seg_lower = segment.lower()
            ordered = sorted(
                mentions,
                key=lambda m: (
                    idx if (idx := seg_lower.find(m.surface_text.lower())) != -1 else len(seg_lower)
                ),
            )
            speaker_id = ordered[0].entity_id
            speaker_surface = ordered[0].surface_text
            addressee_id = next(
                (m.entity_id for m in ordered[1:] if m.entity_id != speaker_id),
                None,
            )
            return speaker_id, addressee_id, speaker_surface

        def _resolve_speaker(before_win: str, after_win: str) -> tuple[str | None, str | None, str | None]:
            if not speaker_entities and not speaker_aliases:
                return None, None, None
            if before_win:
                # Prefer the immediate sentence/clause preceding the quote before falling back to full window
                immediate_clause = re.split(r"[.!?]\s+", before_win.strip())[-1]
                if immediate_clause and immediate_clause != before_win.strip():
                    spk, addr, surf = _resolve_in_segment(immediate_clause)
                    if spk is not None:
                        return spk, addr, surf
                spk, addr, surf = _resolve_in_segment(before_win)
                if spk is not None:
                    return spk, addr, surf
            if after_win:
                spk, addr, surf = _resolve_in_segment(after_win)
                if spk is not None:
                    return spk, addr, surf
            return None, None, None

        def _on_gate1_rejected(span: str, reason: str, meta: dict[str, Any]) -> None:
            for hook in self._hooks:
                hook.on_gate_rejected("gate1", span, reason, meta)

        return self.voice_bank.gate.extract_candidates(
            text,
            source_text=source_text,
            prev_text=prev_text,
            next_text=next_text,
            speaker_resolver=_resolve_speaker,
            on_rejected=_on_gate1_rejected if self._hooks else None,
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
        for ex in exemplars:
            if not bypass_gate and not self.voice_bank.gate.passes_quality_gate(ex):
                for hook in self._hooks:
                    hook.on_gate_rejected(
                        "gate2",
                        ex.text,
                        f"quality_score {ex.quality_score} < {self.voice_bank.gate.min_quality}",
                        {"speaker_id": ex.speaker_id, "quality_score": ex.quality_score},
                    )
                continue
            ex_id = ex.exemplar_id or str(uuid.uuid4())
            indexed.append(ex.model_copy(update={"exemplar_id": ex_id, "step": step, "timeline_id": tl}))

        if indexed:
            vectors = embed_texts_batch(self.voice_bank.embedder, [ex.text for ex in indexed])
            self._record_embedder_metadata(vectors)
            to_store = [(item, vec.tobytes()) for item, vec in zip(indexed, vectors)]
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
        for hook in self._hooks:
            hook.on_before_commit(step, tl, delta, exemplars, episodes)
        committed_episodes: list[EpisodeRecord] = []
        indexed: list[VoiceExemplar] = []
        with self._store_transaction():
            if delta:
                committed_episodes.extend(self.apply_delta(step, delta, timeline_id=tl))
            if episodes:
                committed_episodes.extend(self.index_episodes(step, episodes, timeline_id=tl))
            if exemplars:
                indexed = self.index_voice_exemplars(
                    step, exemplars, bypass_gate=bypass_gate, timeline_id=tl
                )
        for hook in self._hooks:
            hook.on_after_commit(step, tl, committed_episodes, indexed)
        return indexed

    # --- Time-Travel Operations ---

    def rollback_to(self, step: int, *, timeline_id: str | None = None) -> int:
        tl = timeline_id or self.active_timeline
        deleted = self.store.rollback_to(step, timeline_id=tl)
        for hook in self._hooks:
            hook.on_rollback(step, tl, deleted)
        return deleted

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
        for hook in self._hooks:
            hook.on_branch(info)
        return info

    def delete_timeline(self, timeline_id: str, *, cascade: bool = False) -> int:
        deleted = self.store.delete_timeline(timeline_id, cascade=cascade)
        if self._default_timeline == timeline_id:
            self._default_timeline = "main"
        if self._timeline_var.get() == timeline_id:
            self._timeline_var.set(None)
        return deleted

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

    def build_extraction_prompt(
        self,
        ctx: StepContext,
        scene_text: str,
        *,
        prev_text: str = "",
        next_text: str = "",
    ) -> str:
        """Builds a structured extraction prompt combining preset instructions, active IDs, and StepDelta schema."""
        seen_entity_ids: set[str] = set()
        entity_lines: list[str] = []
        for e in (*ctx.entities, *ctx.related_entities):
            if e.entity_id.startswith("masked:") or e.entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(e.entity_id)
            entity_lines.append(f"- {e.entity_id} (name: {e.canonical_name}, kind: {e.kind})")
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
        has_halo = bool(prev_text.strip() or next_text.strip())
        if prev_text.strip():
            parts.append(
                "### Preceding Context (Already Committed — Use ONLY to resolve pronouns like he/she/they; DO NOT re-extract events):"
            )
            parts.append(prev_text.strip())
        if has_halo:
            parts.append(
                "### Target Scene Chunk (EXTRACT ONLY state changes and episodes occurring in THIS text):"
            )
        else:
            parts.append("### Scene Text to Extract From:")
        parts.append(scene_text)
        if next_text.strip():
            parts.append(
                "### Following Context (Read-Only Lookahead — Use ONLY to identify speakers/addressees of trailing dialogue; DO NOT extract future events):"
            )
            parts.append(next_text.strip())
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

        # Include entities and non-disguise aliases introduced within this same StepDelta
        for ed in delta.entities:
            if ed.remove:
                continue
            lookup.setdefault(ed.entity_id.lower(), ed.entity_id)
            if ed.canonical_name:
                lookup.setdefault(ed.canonical_name.lower(), ed.entity_id)
            for alias_item in ed.add_aliases:
                if isinstance(alias_item, AliasRecord):
                    if alias_item.alias and alias_item.kind != "disguise":
                        lookup.setdefault(alias_item.alias.lower(), ed.entity_id)
                elif isinstance(alias_item, str) and alias_item.strip():
                    lookup.setdefault(alias_item.strip().lower(), ed.entity_id)

        def _norm_id(raw: str) -> str:
            return lookup.get(raw.strip().lower(), raw)

        ep_id_map: dict[str, str] = {}
        norm_episodes: list[EpisodeRecord] = []
        for idx, ep in enumerate(delta.episodes):
            old_id = ep.episode_id
            if old_id:
                new_id = old_id
            else:
                digest = hashlib.sha256(
                    f"{tl}:{step if step is not None else 0}:{idx}:{ep.predicate}:{ep.summary}".encode("utf-8")
                ).hexdigest()[:8]
                new_id = f"ep_{step if step is not None else 0}_{digest}"
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

        def _norm_entity_state(state_dict: dict[str, Any]) -> dict[str, Any]:
            if not state_dict:
                return state_dict
            out = dict(state_dict)
            if "location_id" in out and isinstance(out["location_id"], str):
                out["location_id"] = _norm_id(out["location_id"])
            else:
                for alt_loc_key in ("location", "current_location"):
                    if alt_loc_key in out and isinstance(out[alt_loc_key], str):
                        resolved_loc = _norm_id(out[alt_loc_key])
                        if resolved_loc.startswith("loc_") or resolved_loc in {
                            e.entity_id for e in entities if e.kind == "location"
                        }:
                            out.pop(alt_loc_key, None)
                            out["location_id"] = resolved_loc
                            break
            if "mood" not in out:
                for alt_mood_key in ("emotional_state", "emotion"):
                    if alt_mood_key in out and isinstance(out[alt_mood_key], str) and out[alt_mood_key].strip():
                        out["mood"] = out[alt_mood_key].strip()
                        break
            return out

        raw_norm_entities = [
            ed.model_copy(
                update={
                    "entity_id": _norm_id(ed.entity_id),
                    "state": _norm_entity_state(ed.state),
                    "caused_by_episode_id": ep_id_map.get(ed.caused_by_episode_id, ed.caused_by_episode_id)
                    if ed.caused_by_episode_id
                    else None,
                }
            )
            for ed in delta.entities
        ]
        # Merge multiple intra-step EntityDeltas for the same entity_id in order
        merged_entities_by_id: dict[str, Any] = {}
        for ed in raw_norm_entities:
            existing = merged_entities_by_id.get(ed.entity_id)
            if existing is None or ed.remove or existing.remove:
                merged_entities_by_id[ed.entity_id] = ed
            else:
                merged_entities_by_id[ed.entity_id] = existing.model_copy(
                    update={
                        "kind": ed.kind or existing.kind,
                        "canonical_name": ed.canonical_name or existing.canonical_name,
                        "base_attributes": {**existing.base_attributes, **ed.base_attributes},
                        "state": {**existing.state, **ed.state},
                        "private_state": {**existing.private_state, **ed.private_state},
                        "add_aliases": [*existing.add_aliases, *ed.add_aliases],
                        "remove_aliases": [*existing.remove_aliases, *ed.remove_aliases],
                        "caused_by_episode_id": ed.caused_by_episode_id or existing.caused_by_episode_id,
                        "reason": ed.reason or existing.reason,
                    }
                )
        norm_entities = list(merged_entities_by_id.values())
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
            is_flashback=delta.is_flashback,
            story_time=delta.story_time,
            chapter_id=delta.chapter_id,
            scene_id=delta.scene_id,
        )
