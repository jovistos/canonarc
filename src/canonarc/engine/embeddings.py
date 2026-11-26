from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..embedders import (
    EmbedderMismatchError,
    embed_texts_batch,
    get_embedder_info,
)

if TYPE_CHECKING:
    from .core import CanonArcEngine


def _handle_embedder_mismatch(
    engine: CanonArcEngine,
    msg: str,
    *,
    allow_reindex: bool = False,
    stacklevel: int = 4,
) -> None:
    policy = engine._on_embedder_mismatch
    if policy == "raise":
        raise EmbedderMismatchError(msg)
    if allow_reindex and policy == "reindex":
        reindex_embeddings_impl(engine)
        return
    if policy == "warn":
        warnings.warn(
            f"{msg} Call engine.reindex_embeddings() to migrate stored vectors.",
            UserWarning,
            stacklevel=stacklevel,
        )


def check_embedder_compatibility_impl(engine: CanonArcEngine) -> None:
    """Verifies that the active embedder matches any embedder/dimension recorded in the store."""
    get_meta = getattr(engine.store, "get_embedding_metadata", None)
    if not callable(get_meta):
        return
    stored_id, stored_dim = get_meta()
    if stored_id is None and stored_dim is None:
        return
    current_id, current_dim = get_embedder_info(engine.voice_bank.embedder)
    id_mismatch = stored_id is not None and stored_id != current_id
    dim_mismatch = stored_dim is not None and current_dim is not None and stored_dim != current_dim
    if not id_mismatch and not dim_mismatch:
        engine._embedder_verified = True
        return

    msg = (
        f"Store embeddings were created with embedder={stored_id!r}, dim={stored_dim}, "
        f"but engine was initialized with embedder={current_id!r}, dim={current_dim}."
    )
    _handle_embedder_mismatch(engine, msg, allow_reindex=True, stacklevel=4)


def record_embedder_metadata_impl(engine: CanonArcEngine, vectors: Sequence[Any]) -> None:
    """Records embedder_id and vector dimension in store_metadata on first write."""
    if engine._embedder_verified or not vectors:
        return
    set_meta = getattr(engine.store, "set_embedding_metadata", None)
    get_meta = getattr(engine.store, "get_embedding_metadata", None)
    if not callable(set_meta) or not callable(get_meta):
        engine._embedder_verified = True
        return
    dim = len(vectors[0])
    if dim <= 0:
        return
    stored_id, stored_dim = get_meta()
    current_id, _ = get_embedder_info(engine.voice_bank.embedder)
    if stored_id is None and stored_dim is None:
        set_meta(current_id, dim)
        engine._embedder_verified = True
        return
    if (stored_id is not None and stored_id != current_id) or (
        stored_dim is not None and stored_dim != dim
    ):
        msg = (
            f"Store embeddings were created with embedder={stored_id!r}, dim={stored_dim}, "
            f"but active embedder produced embedder={current_id!r}, dim={dim}."
        )
        _handle_embedder_mismatch(engine, msg, allow_reindex=False, stacklevel=4)
    engine._embedder_verified = True


def _reembed_rows_batched(
    engine: CanonArcEngine,
    rows: Sequence[tuple[str, str, str]],
    *,
    step_size: int,
    detected_dim: int | None,
) -> tuple[list[tuple[str, str, bytes]], int | None]:
    updates: list[tuple[str, str, bytes]] = []
    for i in range(0, len(rows), step_size):
        chunk = rows[i : i + step_size]
        vecs = embed_texts_batch(engine.voice_bank.embedder, [r[2] for r in chunk])
        if vecs and detected_dim is None:
            detected_dim = len(vecs[0])
        for (tl, item_id, _), vec in zip(chunk, vecs):
            updates.append((tl, item_id, vec.tobytes()))
    return updates, detected_dim


def reindex_embeddings_impl(engine: CanonArcEngine, *, batch_size: int = 128) -> dict[str, int]:
    """Re-embeds all stored episodes and voice exemplars in batches using the active embedder."""
    list_ep = getattr(engine.store, "list_all_episodes_for_reembed", None)
    list_vo = getattr(engine.store, "list_all_voice_exemplars_for_reembed", None)
    upd_ep = getattr(engine.store, "update_episode_embeddings", None)
    upd_vo = getattr(engine.store, "update_voice_embeddings", None)
    set_meta = getattr(engine.store, "set_embedding_metadata", None)
    if not (
        callable(list_ep)
        and callable(list_vo)
        and callable(upd_ep)
        and callable(upd_vo)
        and callable(set_meta)
    ):
        raise NotImplementedError("Active store backend does not support reindex_embeddings().")

    ep_rows: list[tuple[str, str, str]] = list_ep()
    vo_rows: list[tuple[str, str, str]] = list_vo()
    step_size = max(1, batch_size)

    ep_updates, detected_dim = _reembed_rows_batched(
        engine, ep_rows, step_size=step_size, detected_dim=None
    )
    vo_updates, detected_dim = _reembed_rows_batched(
        engine, vo_rows, step_size=step_size, detected_dim=detected_dim
    )

    current_id, known_dim = get_embedder_info(engine.voice_bank.embedder)
    final_dim = detected_dim or known_dim
    if final_dim is None:
        probe = engine.voice_bank.embedder.embed_text("probe")
        final_dim = len(probe)

    with engine._store_transaction():
        n_ep = upd_ep(ep_updates)
        n_vo = upd_vo(vo_updates)
        set_meta(current_id, final_dim)
    engine._embedder_verified = True
    return {"episodes": n_ep, "voice_exemplars": n_vo}
