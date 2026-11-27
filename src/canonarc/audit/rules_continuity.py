from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz import fuzz

from ..models import AliasRecord, EntityRecord
from .models import AuditContext, AuditIssue, AuditSeverity
from .text_utils import _entity_id_slug, _normalize_label

_TV_REGISTER_TOKENS: frozenset[str] = frozenset(
    {"vi", "ti", "vous", "tu", "sie", "du", "usted", "lei", "vy", "ty"}
)
_GENERIC_REGISTER_TOKENS: frozenset[str] = frozenset(
    {"formal", "informal", "casual", "polite", "intimate", "familiar"}
)


class NearDuplicateEntityRule:
    """Detects near-duplicate EntityRecord creation via fuzzy name or alias/exonym collision."""

    name: str = "near_duplicate_entity"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        fuzzy_threshold: float = 88.0,
        severity: AuditSeverity = "warning",
        same_kind_only: bool = True,
        check_all_entities: bool = False,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.severity: AuditSeverity = severity
        self.same_kind_only = same_kind_only
        self.check_all_entities = check_all_entities

    @staticmethod
    def _ignored_targets(entity: EntityRecord) -> set[str]:
        raw = entity.base_attributes.get("ignore_duplicate_of")
        if isinstance(raw, str) and raw.strip():
            return {raw.strip()}
        if isinstance(raw, (list, tuple, set)):
            return {str(x).strip() for x in raw if str(x).strip()}
        return set()

    @staticmethod
    def _candidate_labels(entity: EntityRecord) -> list[tuple[str, str]]:
        labels: list[tuple[str, str]] = []
        if entity.canonical_name.strip():
            labels.append(("canonical_name", _normalize_label(entity.canonical_name)))
        tname = entity.base_attributes.get("target_name")
        if isinstance(tname, str) and tname.strip():
            norm_t = _normalize_label(tname)
            if norm_t and norm_t not in {lbl for _, lbl in labels}:
                labels.append(("target_name", norm_t))
        slug = _entity_id_slug(entity.entity_id)
        if slug and len(slug) >= 4 and slug not in {lbl for _, lbl in labels}:
            labels.append(("entity_id", slug))
        return [(src, lbl) for src, lbl in labels if lbl]

    def _compare_pair(
        self,
        candidate: EntityRecord,
        existing: EntityRecord,
        existing_aliases: Sequence[AliasRecord],
    ) -> tuple[float, str, str, str] | None:
        """Returns (similarity, candidate_field, existing_field, matched_value) if colliding."""
        if existing.entity_id in self._ignored_targets(candidate):
            return None
        if candidate.entity_id in self._ignored_targets(existing):
            return None

        cand_labels = self._candidate_labels(candidate)

        exist_labels: list[tuple[str, str]] = []
        if existing.canonical_name.strip():
            exist_labels.append(("canonical_name", _normalize_label(existing.canonical_name)))
        etname = existing.base_attributes.get("target_name")
        if isinstance(etname, str) and etname.strip():
            exist_labels.append(("target_name", _normalize_label(etname)))
        for al in existing_aliases:
            if al.alias.strip():
                exist_labels.append((f"alias:{al.alias}", _normalize_label(al.alias)))
            if al.persona_name and al.persona_name.strip():
                exist_labels.append((f"persona:{al.persona_name}", _normalize_label(al.persona_name)))

        best: tuple[float, str, str, str] | None = None
        for c_src, c_lbl in cand_labels:
            for e_src, e_lbl in exist_labels:
                if not e_lbl:
                    continue
                is_alias = e_src.startswith(("alias:", "persona:"))
                if self.same_kind_only and not is_alias and candidate.kind != existing.kind:
                    continue
                if c_lbl == e_lbl:
                    score = 100.0
                elif len(c_lbl) >= 4 and len(e_lbl) >= 4:
                    score = float(fuzz.ratio(c_lbl, e_lbl))
                else:
                    continue

                if score >= self.fuzzy_threshold and (best is None or score > best[0]):
                    best = (score, c_src, e_src, e_lbl)
        return best

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if len(ctx.entities) < 2:
            return []

        aliases_by_entity: dict[str, list[AliasRecord]] = {}
        for al in ctx.aliases:
            aliases_by_entity.setdefault(al.entity_id, []).append(al)

        issues: list[AuditIssue] = []
        seen_pairs: set[tuple[str, str]] = set()

        if self.check_all_entities:
            def _canonical_sort_key(e: EntityRecord) -> tuple[int, int, int, int, int, int, str]:
                has_target = 1 if isinstance(e.base_attributes.get("target_name"), str) and e.base_attributes["target_name"].strip() else 0
                alias_count = len(aliases_by_entity.get(e.entity_id, ()))
                attr_count = len(e.base_attributes)
                name_len = len(e.canonical_name.strip())
                first_known_step = (
                    e.prior_step
                    if (e.revision_count > 1 and e.prior_step is not None)
                    else e.valid_from_step
                )
                return (
                    -e.revision_count,
                    first_known_step,
                    -has_target,
                    -alias_count,
                    -attr_count,
                    -name_len,
                    e.entity_id,
                )

            sorted_entities = sorted(ctx.entities, key=_canonical_sort_key)
            pairs_to_check: list[tuple[EntityRecord, EntityRecord]] = []
            for idx, cand in enumerate(sorted_entities):
                for prev in sorted_entities[:idx]:
                    pairs_to_check.append((cand, prev))
        else:
            if not ctx.new_entities:
                return []
            new_ids = {e.entity_id for e in ctx.new_entities}
            prior_entities = [e for e in ctx.entities if e.entity_id not in new_ids]
            pairs_to_check = []
            sorted_new = sorted(ctx.new_entities, key=lambda e: e.entity_id)
            for idx, cand in enumerate(sorted_new):
                for existing in prior_entities:
                    pairs_to_check.append((cand, existing))
                for earlier_new in sorted_new[:idx]:
                    pairs_to_check.append((cand, earlier_new))

        for cand, existing in pairs_to_check:
            pair_key = (
                min(cand.entity_id, existing.entity_id),
                max(cand.entity_id, existing.entity_id),
            )
            if pair_key in seen_pairs:
                continue
            hit = self._compare_pair(
                cand, existing, aliases_by_entity.get(existing.entity_id, ())
            )
            if hit is None:
                continue
            score, c_src, e_src, matched_val = hit
            seen_pairs.add(pair_key)
            issues.append(
                AuditIssue(
                    code="near_duplicate_entity",
                    rule_name=self.name,
                    severity=self.severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key=cand.entity_id,
                    entity_ids=[cand.entity_id, existing.entity_id],
                    message=(
                        f"Entity '{cand.entity_id}' ({cand.canonical_name}) collides with existing "
                        f"entity '{existing.entity_id}' ({existing.canonical_name}) via {e_src} "
                        f"(similarity={score:.1f})."
                    ),
                    evidence={
                        "new_entity_id": cand.entity_id,
                        "new_canonical_name": cand.canonical_name,
                        "new_kind": cand.kind,
                        "existing_entity_id": existing.entity_id,
                        "existing_canonical_name": existing.canonical_name,
                        "existing_kind": existing.kind,
                        "candidate_field": c_src,
                        "matched_against": e_src,
                        "matched_value": matched_val,
                        "similarity": round(score, 2),
                    },
                    suggestion=(
                        f"Merge '{cand.entity_id}' into '{existing.entity_id}' via "
                        f"engine.merge_entities('{cand.entity_id}', '{existing.entity_id}') "
                        f"(or register '{cand.canonical_name}' as an AliasRecord on '{existing.entity_id}')."
                    ),
                )
            )

        return issues


class UnexplainedRegisterFlipRule:
    """Flags mutations to hard-filter edge keys (e.g. 'register') lacking causal provenance."""

    name: str = "unexplained_register_flip"
    required_tiers: frozenset[int] = frozenset({2, 3})

    def __init__(
        self,
        *,
        severity: AuditSeverity = "warning",
        keys: Sequence[str] | None = None,
        skip_step_zero: bool = True,
        check_dangling_episode: bool = True,
    ) -> None:
        self.severity: AuditSeverity = severity
        self.keys: tuple[str, ...] | None = tuple(keys) if keys is not None else None
        self.skip_step_zero = skip_step_zero
        self.check_dangling_episode = check_dangling_episode

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if self.skip_step_zero and ctx.step == 0:
            return []

        tracked_keys = self.keys if self.keys is not None else ctx.policy.hard_filter_keys
        if not tracked_keys:
            return []

        issues: list[AuditIssue] = []
        for edge in ctx.mutated_edges:
            if edge.prior_state is None or edge.revision_count <= 1:
                continue
            subject_key = f"{edge.source_id}:{edge.relation_type}->{edge.target_id}"

            for key in tracked_keys:
                prior_val = edge.prior_state.get(key)
                new_val = edge.state.get(key)
                if prior_val is None or new_val is None:
                    continue
                if str(prior_val).strip().lower() == str(new_val).strip().lower():
                    continue

                has_ep = bool(edge.caused_by_episode_id and edge.caused_by_episode_id.strip())
                has_reason = bool(edge.reason and edge.reason.strip())

                if not has_ep and not has_reason:
                    issues.append(
                        AuditIssue(
                            code="unexplained_register_flip",
                            rule_name=self.name,
                            severity=self.severity,
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key=subject_key,
                            entity_ids=[edge.source_id, edge.target_id],
                            message=(
                                f"Edge {edge.source_id} → {edge.target_id} ({edge.relation_type}) "
                                f"flipped '{key}' from '{prior_val}' to '{new_val}' without "
                                "caused_by_episode_id or reason."
                            ),
                            evidence={
                                "key": key,
                                "prior_value": prior_val,
                                "new_value": new_val,
                                "prior_step": edge.prior_step,
                                "caused_by_episode_id": edge.caused_by_episode_id,
                                "reason": edge.reason,
                            },
                            suggestion=(
                                "Attach caused_by_episode_id or a concise reason on the EdgeDelta "
                                "to document why this register shift occurred."
                            ),
                        )
                    )
                elif (
                    self.check_dangling_episode
                    and has_ep
                    and ctx.policy.has_tier(6)
                    and edge.caused_by_episode_id not in ctx.all_episode_ids
                ):
                    issues.append(
                        AuditIssue(
                            code="dangling_causal_episode",
                            rule_name=self.name,
                            severity="error",
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key=subject_key,
                            entity_ids=[edge.source_id, edge.target_id],
                            message=(
                                f"Edge {edge.source_id} → {edge.target_id} ({edge.relation_type}) "
                                f"flipped '{key}' referencing unknown caused_by_episode_id "
                                f"'{edge.caused_by_episode_id}'."
                            ),
                            evidence={
                                "key": key,
                                "prior_value": prior_val,
                                "new_value": new_val,
                                "caused_by_episode_id": edge.caused_by_episode_id,
                            },
                            suggestion="Ensure caused_by_episode_id matches a committed EpisodeRecord.episode_id.",
                        )
                    )

        return issues


class RegisterVocabularyDriftRule:
    """Detects mixed register vocabularies (e.g. Vi/ti mixed with formal/informal) or invalid values."""

    name: str = "register_vocabulary_drift"
    required_tiers: frozenset[int] = frozenset({2})

    def __init__(
        self,
        *,
        severity: AuditSeverity = "warning",
        key: str = "register",
        allowed_values: Sequence[str] | None = None,
    ) -> None:
        self.severity: AuditSeverity = severity
        self.key = key
        self.allowed_values: frozenset[str] | None = (
            frozenset(v.strip().lower() for v in allowed_values if v.strip())
            if allowed_values is not None
            else None
        )

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        issues: list[AuditIssue] = []

        if self.allowed_values is not None:
            for edge in ctx.edges:
                val = edge.state.get(self.key)
                if val is None or not str(val).strip():
                    continue
                if str(val).strip().lower() not in self.allowed_values:
                    subject_key = f"{edge.source_id}:{edge.relation_type}->{edge.target_id}"
                    issues.append(
                        AuditIssue(
                            code="register_vocabulary_drift",
                            rule_name=self.name,
                            severity=self.severity,
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key=subject_key,
                            entity_ids=[edge.source_id, edge.target_id],
                            message=(
                                f"Edge {edge.source_id} → {edge.target_id} has {self.key}='{val}', "
                                f"which is outside allowed values {sorted(self.allowed_values)}."
                            ),
                            evidence={
                                "key": self.key,
                                "value": val,
                                "allowed_values": sorted(self.allowed_values),
                            },
                            suggestion=f"Normalize '{val}' to one of {sorted(self.allowed_values)}.",
                        )
                    )
            return issues

        tv_seen: set[str] = set()
        generic_seen: set[str] = set()
        for edge in ctx.edges:
            val = edge.state.get(self.key)
            if val is None:
                continue
            low = str(val).strip().lower()
            if low in _TV_REGISTER_TOKENS:
                tv_seen.add(str(val).strip())
            elif low in _GENERIC_REGISTER_TOKENS:
                generic_seen.add(str(val).strip())

        if tv_seen and generic_seen:
            issues.append(
                AuditIssue(
                    code="register_vocabulary_drift",
                    rule_name=self.name,
                    severity=self.severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    message=(
                        f"Mixed '{self.key}' vocabularies active on timeline '{ctx.timeline_id}': "
                        f"T-V tokens {sorted(tv_seen)} mixed with generic labels {sorted(generic_seen)}."
                    ),
                    evidence={
                        "key": self.key,
                        "tv_values": sorted(tv_seen),
                        "generic_values": sorted(generic_seen),
                    },
                    suggestion=(
                        "Standardize on one register vocabulary (e.g. Vi/ti or formal/informal) "
                        "so VoiceBank hard-filtering stays consistent."
                    ),
                )
            )
        return issues
