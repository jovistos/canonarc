from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from ..inflections import InflectionRequest, generate_inflections_batch
from ..models import AliasRecord, EntityDelta, EntityRecord, StepDelta

if TYPE_CHECKING:
    from .core import CanonArcEngine


def make_inflection_request_impl(
    entity: EntityRecord,
    *,
    target_lang: str | None = None,
) -> InflectionRequest | None:
    raw_target = entity.base_attributes.get("target_name")
    if not isinstance(raw_target, str) or not raw_target.strip():
        return None
    raw_lang = (
        target_lang
        or entity.base_attributes.get("target_lang")
        or entity.base_attributes.get("lang")
    )
    eff_lang = raw_lang.strip().lower() if isinstance(raw_lang, str) and raw_lang.strip() else None
    if not eff_lang:
        return None
    gender = entity.base_attributes.get("gender")
    declension = entity.base_attributes.get("declension")
    return InflectionRequest(
        name=raw_target.strip(),
        lang=eff_lang,
        gender=str(gender) if gender is not None else None,
        kind=entity.kind,
        declension=str(declension) if declension is not None else None,
    )


def build_target_aliases_impl(
    engine: CanonArcEngine,
    entity: EntityRecord,
    existing_aliases: Sequence[str | AliasRecord],
    *,
    target_lang: str | None = None,
    auto_inflect: bool = True,
    prefetched_forms: Sequence[str] | None = None,
) -> list[AliasRecord]:
    """Auto-registers base_attributes['target_name'] and optional one-shot LLM inflections."""
    raw_target = entity.base_attributes.get("target_name")
    if not isinstance(raw_target, str) or not raw_target.strip():
        return []
    target_name = raw_target.strip()
    raw_lang = (
        target_lang
        or entity.base_attributes.get("target_lang")
        or entity.base_attributes.get("lang")
    )
    eff_lang = raw_lang.strip().lower() if isinstance(raw_lang, str) and raw_lang.strip() else None

    existing_keys: set[str] = set()
    for a in existing_aliases:
        s = a.alias.strip() if isinstance(a, AliasRecord) else a.strip()
        if s:
            existing_keys.add(s.lower())
    canonical_lower = entity.canonical_name.strip().lower()

    generated: list[AliasRecord] = []

    def _maybe_add(
        form: str,
        *,
        kind: Literal["permanent", "inflection"] = "permanent",
        allow_canonical: bool = False,
    ) -> None:
        clean = form.strip()
        if not clean:
            return
        key = clean.lower()
        if key in existing_keys:
            return
        if not allow_canonical and key == canonical_lower:
            return
        existing_keys.add(key)
        generated.append(
            AliasRecord(
                alias=clean,
                entity_id=entity.entity_id,
                kind=kind,
                lang=eff_lang,
            )
        )

    _maybe_add(target_name, kind="permanent", allow_canonical=eff_lang is not None)

    forms: Sequence[str] = ()
    if prefetched_forms is not None:
        forms = prefetched_forms
    elif auto_inflect and eff_lang and engine.inflection_generator is not None:
        gender = entity.base_attributes.get("gender")
        declension = entity.base_attributes.get("declension")
        forms = engine.inflection_generator.generate_inflections(
            target_name,
            eff_lang,
            gender=str(gender) if gender is not None else None,
            kind=entity.kind,
            declension=str(declension) if declension is not None else None,
        )
    for form in forms:
        _maybe_add(form, kind="inflection")

    return generated


def register_entity_impl(
    engine: CanonArcEngine,
    entity: EntityRecord,
    *,
    aliases: Sequence[str | AliasRecord] = (),
    step: int = 0,
    timeline_id: str | None = None,
    target_lang: str | None = None,
    auto_inflect: bool = True,
) -> None:
    tl = timeline_id or engine.active_timeline
    extra_aliases = engine._build_target_aliases(
        entity,
        aliases,
        target_lang=target_lang,
        auto_inflect=auto_inflect,
    )
    all_aliases: list[str | AliasRecord] = [*aliases, *extra_aliases]
    engine.store.register_entity(entity, aliases=all_aliases, step=step, timeline_id=tl)


def seed_entities_impl(
    engine: CanonArcEngine,
    entities: Sequence[EntityRecord],
    *,
    step: int = 0,
    timeline_id: str | None = None,
    target_lang: str | None = None,
    auto_inflect: bool = True,
    batch_size: int = 25,
) -> None:
    tl = timeline_id or engine.active_timeline
    prefetched_by_idx: dict[int, list[str]] = {}
    if auto_inflect and engine.inflection_generator is not None:
        pending_indices: list[int] = []
        pending_requests: list[InflectionRequest] = []
        for idx, ent in enumerate(entities):
            req = engine._make_inflection_request(ent, target_lang=target_lang)
            if req is not None:
                pending_indices.append(idx)
                pending_requests.append(req)
        if pending_requests:
            batched_forms = generate_inflections_batch(
                engine.inflection_generator,
                pending_requests,
                batch_size=batch_size,
            )
            for idx, forms in zip(pending_indices, batched_forms):
                prefetched_by_idx[idx] = forms

    with engine._store_transaction():
        for idx, ent in enumerate(entities):
            extra_aliases = engine._build_target_aliases(
                ent,
                (),
                target_lang=target_lang,
                auto_inflect=False,
                prefetched_forms=prefetched_by_idx.get(idx),
            )
            engine.store.register_entity(
                ent,
                aliases=extra_aliases,
                step=step,
                timeline_id=tl,
            )


def expand_entity_inflections_impl(
    engine: CanonArcEngine,
    entity_id: str,
    *,
    target_lang: str | None = None,
    step: int | None = None,
    timeline_id: str | None = None,
) -> list[AliasRecord]:
    """One-shot inflection expansion for an already-registered or Zero-Seed discovered entity."""
    tl = timeline_id or engine.active_timeline
    if step is None:
        tl_info = engine.store.get_timeline(tl)
        eff_step = tl_info.head_step if tl_info is not None else 0
    else:
        eff_step = step

    entities = engine.store.get_all_active_entities(tl, eff_step)
    entity = next((e for e in entities if e.entity_id == entity_id), None)
    if entity is None:
        raise ValueError(f"Entity '{entity_id}' not found on timeline '{tl}' at step {eff_step}.")

    active_aliases = [
        a for a in engine.store.get_all_active_aliases(tl, eff_step) if a.entity_id == entity_id
    ]
    new_aliases = engine._build_target_aliases(
        entity,
        active_aliases,
        target_lang=target_lang,
        auto_inflect=True,
    )
    if new_aliases:
        engine.apply_delta(
            eff_step,
            StepDelta(
                entities=[
                    EntityDelta(
                        entity_id=entity_id,
                        add_aliases=list(new_aliases),
                    )
                ]
            ),
            timeline_id=tl,
        )
    return new_aliases
