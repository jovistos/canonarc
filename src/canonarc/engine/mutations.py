from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..embedders import embed_texts_batch
from ..models import EntityDelta, EpisodeRecord, StepDelta, VoiceExemplar
from .traces import _ACTIVE_COMMIT_EPISODES

if TYPE_CHECKING:
    from .core import CanonArcEngine


def _embed_prepared_episodes(
    engine: CanonArcEngine,
    episodes: Sequence[EpisodeRecord],
) -> list[tuple[EpisodeRecord, bytes]]:
    if not episodes:
        return []
    vectors = embed_texts_batch(engine.voice_bank.embedder, [ep.summary for ep in episodes])
    engine._record_embedder_metadata(vectors)
    return [(ep, vec.tobytes()) for ep, vec in zip(episodes, vectors)]


def apply_delta_impl(
    engine: CanonArcEngine,
    step: int,
    delta: StepDelta,
    *,
    timeline_id: str | None = None,
) -> list[EpisodeRecord]:
    tl = timeline_id or engine.active_timeline
    policy = engine._policy
    has_edges = policy.has_tier(2)
    has_state = policy.has_tier(3)
    has_episodes = policy.has_tier(6)

    prepared_episodes: list[EpisodeRecord] = []
    if has_episodes:
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

    ep_embeddings: dict[str, bytes] = {
        ep.episode_id: emb_bytes
        for ep, emb_bytes in _embed_prepared_episodes(engine, prepared_episodes)
    }

    delta_updates: dict[str, Any] = {}
    if has_episodes:
        if prepared_episodes:
            delta_updates["episodes"] = prepared_episodes
    else:
        if delta.episodes:
            delta_updates["episodes"] = []
        if delta.inform:
            delta_updates["inform"] = []

    if not has_edges and delta.edges:
        delta_updates["edges"] = []

    if not has_state and delta.entities:
        stripped_entities: list[EntityDelta] = []
        for ed in delta.entities:
            cleaned = ed.model_copy(
                update={
                    "state": {},
                    "private_state": {},
                    "caused_by_episode_id": None,
                    "reason": None,
                }
            )
            if (
                cleaned.canonical_name is not None
                or cleaned.kind is not None
                or cleaned.base_attributes
                or cleaned.add_aliases
                or cleaned.remove_aliases
                or cleaned.remove
            ):
                stripped_entities.append(cleaned)
        delta_updates["entities"] = stripped_entities

    if delta.is_flashback:
        source_entities = delta_updates.get("entities", delta.entities)
        existing_ids = {
            e.entity_id for e in engine.store.get_all_active_entities(tl, step)
        }
        delta_updates["entities"] = [
            ed for ed in source_entities if not ed.remove and ed.entity_id not in existing_ids
        ]
        delta_updates["edges"] = []

    delta_to_apply = delta.model_copy(update=delta_updates) if delta_updates else delta
    return engine.store.apply_delta(
        step, delta_to_apply, timeline_id=tl, episode_embeddings=ep_embeddings
    )


def index_episodes_impl(
    engine: CanonArcEngine,
    step: int,
    episodes: Sequence[EpisodeRecord],
    *,
    timeline_id: str | None = None,
) -> list[EpisodeRecord]:
    tl = timeline_id or engine.active_timeline
    if not episodes or not engine._policy.has_tier(6):
        return []
    indexed: list[EpisodeRecord] = []
    for ep in episodes:
        ep_id = ep.episode_id or f"ep_{step}_{uuid.uuid4().hex[:8]}"
        indexed.append(ep.model_copy(update={"episode_id": ep_id, "step": step, "timeline_id": tl}))

    to_store = _embed_prepared_episodes(engine, indexed)
    engine.store.insert_episodes(to_store, timeline_id=tl)
    return indexed


def commit_step_impl(
    engine: CanonArcEngine,
    step: int,
    delta: StepDelta | None = None,
    exemplars: Sequence[VoiceExemplar] = (),
    episodes: Sequence[EpisodeRecord] = (),
    *,
    bypass_gate: bool = False,
    timeline_id: str | None = None,
    content_hash: str | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> list[VoiceExemplar]:
    tl = timeline_id or engine.active_timeline
    for hook in engine._hooks:
        hook.on_before_commit(step, tl, delta, exemplars, episodes)
    committed_episodes: list[EpisodeRecord] = []
    indexed: list[VoiceExemplar] = []
    record_cp = getattr(engine.store, "record_step_checkpoint", None)
    if not callable(record_cp) and (content_hash is not None or checkpoint_metadata is not None):
        raise NotImplementedError("Active store backend does not support step checkpoints.")
    with engine._store_transaction():
        if delta:
            committed_episodes.extend(engine.apply_delta(step, delta, timeline_id=tl))
        if episodes and engine._policy.has_tier(6):
            committed_episodes.extend(engine.index_episodes(step, episodes, timeline_id=tl))
        if exemplars and engine._policy.has_tier(5):
            indexed = engine.index_voice_exemplars(
                step, exemplars, bypass_gate=bypass_gate, timeline_id=tl
            )
        if callable(record_cp):
            record_cp(
                step,
                timeline_id=tl,
                content_hash=content_hash,
                metadata=checkpoint_metadata,
            )
    for hook in engine._hooks:
        hook.on_after_commit(step, tl, committed_episodes, indexed)
    ep_sink = _ACTIVE_COMMIT_EPISODES.get()
    if ep_sink is not None:
        ep_sink.extend(committed_episodes)
    return indexed
