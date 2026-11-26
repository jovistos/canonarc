from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..embedders import embed_texts_batch
from ..models import (
    AliasRecord,
    DialogueCandidate,
    EntityRecord,
    StepContext,
    StepDelta,
    VoiceExemplar,
)

if TYPE_CHECKING:
    from .core import CanonArcEngine


def extract_dialogue_candidates_with_delta(
    engine: CanonArcEngine,
    text: str,
    *,
    source_text: str = "",
    step: int | None = None,
    timeline_id: str | None = None,
    prev_text: str = "",
    next_text: str = "",
    lang: str | None = None,
    staged_delta: StepDelta | None = None,
) -> list[DialogueCandidate]:
    tl = timeline_id or engine.active_timeline
    if step is None:
        tl_info = engine.store.get_timeline(tl)
        cutoff_step = tl_info.head_step if tl_info is not None else 0
    else:
        cutoff_step = step

    active_entities = list(engine.store.get_all_active_entities(tl, cutoff_step))
    active_aliases = list(engine.store.get_all_active_aliases(tl, cutoff_step))

    if staged_delta is not None and staged_delta.entities:
        ent_by_id = {e.entity_id: e for e in active_entities}
        for ed in staged_delta.entities:
            if ed.remove:
                ent_by_id.pop(ed.entity_id, None)
                continue
            existing = ent_by_id.get(ed.entity_id)
            if existing is None:
                ent_by_id[ed.entity_id] = EntityRecord(
                    entity_id=ed.entity_id,
                    canonical_name=ed.canonical_name or ed.entity_id,
                    kind=ed.kind or "character",
                    base_attributes=dict(ed.base_attributes),
                    state=dict(ed.state),
                )
            else:
                merged_base = {**existing.base_attributes, **ed.base_attributes}
                merged_state = {**existing.state, **ed.state} if ed.merge_state else dict(ed.state)
                ent_by_id[ed.entity_id] = existing.model_copy(
                    update={
                        "canonical_name": ed.canonical_name or existing.canonical_name,
                        "kind": ed.kind or existing.kind,
                        "base_attributes": merged_base,
                        "state": merged_state,
                    }
                )
            for alias_item in ed.add_aliases:
                if isinstance(alias_item, AliasRecord):
                    active_aliases.append(alias_item)
                elif isinstance(alias_item, str) and alias_item.strip():
                    active_aliases.append(
                        AliasRecord(alias=alias_item.strip(), entity_id=ed.entity_id)
                    )
        active_entities = list(ent_by_id.values())

    speaker_entities = [
        e for e in active_entities if e.kind not in ("location", "artifact", "item", "term")
    ]
    speaker_entity_ids = {e.entity_id for e in speaker_entities}
    speaker_aliases = [
        a for a in active_aliases if not speaker_entity_ids or a.entity_id in speaker_entity_ids
    ]

    def _resolve_in_segment(segment: str) -> tuple[str | None, str | None, str | None]:
        mentions = engine._resolve_mentions(
            text=segment,
            explicit_mentions=(),
            entities=speaker_entities,
            aliases=speaker_aliases,
            lang=lang,
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
        for hook in engine._hooks:
            hook.on_gate_rejected("gate1", span, reason, meta)

    return engine.voice_bank.gate.extract_candidates(
        text,
        source_text=source_text,
        prev_text=prev_text,
        next_text=next_text,
        speaker_resolver=_resolve_speaker,
        on_rejected=_on_gate1_rejected if engine._hooks else None,
        ignore_patterns=engine._ignore_patterns,
        text_normalizer=engine.text_normalizer,
    )


def resolve_harvest_metadata(
    engine: CanonArcEngine,
    *,
    speaker_id: str | None,
    addressee_id: str | None,
    context: StepContext,
    staged_delta: StepDelta | None,
    cutoff_step: int,
    timeline_id: str,
    infer_addressee: bool = True,
) -> tuple[str | None, dict[str, str]]:
    """Resolves dyadic edge metadata (register/tone) and speaker persona with staged-delta overlay."""
    policy = engine._policy
    meta: dict[str, str] = {}
    if not speaker_id:
        return addressee_id, meta

    ent_by_id: dict[str, EntityRecord] = {
        e.entity_id: e
        for e in (*context.entities, *context.related_entities)
        if not e.entity_id.startswith("masked:")
    }
    if speaker_id not in ent_by_id:
        for e in engine.store.get_all_active_entities(timeline_id, cutoff_step):
            ent_by_id.setdefault(e.entity_id, e)

    if staged_delta is not None:
        for ed in staged_delta.entities:
            if ed.remove:
                ent_by_id.pop(ed.entity_id, None)
                continue
            existing = ent_by_id.get(ed.entity_id)
            if existing is None:
                ent_by_id[ed.entity_id] = EntityRecord(
                    entity_id=ed.entity_id,
                    canonical_name=ed.canonical_name or ed.entity_id,
                    kind=ed.kind or "character",
                    base_attributes=dict(ed.base_attributes),
                    state=dict(ed.state),
                )
            else:
                merged_state = {**existing.state, **ed.state} if ed.merge_state else dict(ed.state)
                ent_by_id[ed.entity_id] = existing.model_copy(
                    update={
                        "canonical_name": ed.canonical_name or existing.canonical_name,
                        "kind": ed.kind or existing.kind,
                        "state": merged_state,
                    }
                )

    eff_addressee = addressee_id
    if eff_addressee is None and infer_addressee:
        scene_chars = [
            e.entity_id
            for e in context.entities
            if not e.entity_id.startswith("masked:")
            and e.kind == "character"
            and e.entity_id != speaker_id
        ]
        if len(scene_chars) == 1:
            eff_addressee = scene_chars[0]

    target_keys = tuple(
        dict.fromkeys((*policy.hard_filter_keys, *policy.soft_boost_weights.keys()))
    )

    if policy.has_tier(2) and eff_addressee:
        edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
        for edge in engine.store.get_all_active_edges(timeline_id, cutoff_step):
            if edge.source_id == speaker_id and edge.target_id == eff_addressee:
                edge_map[(edge.source_id, edge.target_id, edge.relation_type)] = dict(edge.state)
        if staged_delta is not None:
            for rd in staged_delta.edges:
                if rd.source_id != speaker_id or rd.target_id != eff_addressee:
                    continue
                key = (rd.source_id, rd.target_id, rd.relation_type)
                if rd.remove:
                    edge_map.pop(key, None)
                elif rd.merge and key in edge_map:
                    edge_map[key] = {**edge_map[key], **rd.state}
                else:
                    edge_map[key] = dict(rd.state)

        for rel_type in policy.edge_layer_order:
            state = edge_map.get((speaker_id, eff_addressee, rel_type))
            if not state:
                continue
            for k in target_keys:
                val = state.get(k)
                if val is not None and str(val).strip():
                    meta[k] = str(val).strip()

        for (_, _, rel_type), state in edge_map.items():
            if rel_type in policy.edge_layer_order or rel_type == policy.belief_relation:
                continue
            for k in target_keys:
                if k not in meta:
                    val = state.get(k)
                    if val is not None and str(val).strip():
                        meta[k] = str(val).strip()

    speaker_ent = ent_by_id.get(speaker_id)
    if speaker_ent is not None:
        disguise = speaker_ent.state.get(policy.entity_disguise_key)
        if disguise is not None and str(disguise).strip():
            meta[policy.exemplar_persona_key] = str(disguise).strip()
        if "tone" in target_keys and "tone" not in meta:
            mood = speaker_ent.state.get("mood")
            if mood is not None and str(mood).strip():
                meta["tone"] = str(mood).strip()

    return eff_addressee, meta


def index_voice_exemplars_impl(
    engine: CanonArcEngine,
    step: int,
    exemplars: Sequence[VoiceExemplar],
    *,
    bypass_gate: bool = False,
    timeline_id: str | None = None,
) -> list[VoiceExemplar]:
    tl = timeline_id or engine.active_timeline
    if not exemplars or not engine._policy.has_tier(5):
        return []
    indexed: list[VoiceExemplar] = []
    for ex in exemplars:
        if not bypass_gate and not engine.voice_bank.gate.passes_quality_gate(ex):
            for hook in engine._hooks:
                hook.on_gate_rejected(
                    "gate2",
                    ex.text,
                    f"quality_score {ex.quality_score} < {engine.voice_bank.gate.min_quality}",
                    {"speaker_id": ex.speaker_id, "quality_score": ex.quality_score},
                )
            continue
        ex_id = ex.exemplar_id or str(uuid.uuid4())
        indexed.append(ex.model_copy(update={"exemplar_id": ex_id, "step": step, "timeline_id": tl}))

    if indexed:
        vectors = embed_texts_batch(engine.voice_bank.embedder, [ex.text for ex in indexed])
        engine._record_embedder_metadata(vectors)
        to_store = [(item, vec.tobytes()) for item, vec in zip(indexed, vectors)]
        engine.store.insert_voice_exemplars(to_store, timeline_id=tl)
    return indexed
