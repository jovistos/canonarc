from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from .embedders import _CJK_CHAR_RE
from .models import AliasRecord, DirectedEdgeRecord, EntityRecord, EpisodeRecord
from .presets import DEFAULT_SCHEMA_POLICY, DomainPreset, SchemaPolicy

if TYPE_CHECKING:
    from .engine import CanonArcEngine

AuditSeverity = Literal["info", "warning", "error"]

_SEVERITY_RANK: dict[AuditSeverity, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
}

_ENTITY_ID_PREFIXES: tuple[str, ...] = (
    "char_",
    "loc_",
    "item_",
    "fac_",
    "term_",
    "org_",
    "ent_",
)

_LEADING_ARTICLES_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

_TV_REGISTER_TOKENS: frozenset[str] = frozenset(
    {"vi", "ti", "vous", "tu", "sie", "du", "usted", "lei", "vy", "ty"}
)
_GENERIC_REGISTER_TOKENS: frozenset[str] = frozenset(
    {"formal", "informal", "casual", "polite", "intimate", "familiar"}
)


class AuditIssue(BaseModel):
    """Single continuity or localization QA finding emitted by an AuditRule."""

    code: str = Field(description="Machine-readable issue code, e.g. 'exonym_leak'")
    rule_name: str = Field(description="Name of the AuditRule that emitted this issue")
    severity: AuditSeverity = "warning"
    step: int
    timeline_id: str = "main"
    message: str
    subject_key: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    suggestion: str | None = None


class AuditReport(BaseModel):
    """Structured result returned by engine.audit_step() and engine.audit_timeline()."""

    timeline_id: str
    steps_audited: list[int] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    rules_executed: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no error-severity issues were detected."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")

    def has_issues_at_or_above(self, min_severity: AuditSeverity) -> bool:
        threshold = _SEVERITY_RANK[min_severity]
        return any(_SEVERITY_RANK[i.severity] >= threshold for i in self.issues)

    def by_severity(self, severity: AuditSeverity) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == severity]

    def by_code(self, code: str) -> list[AuditIssue]:
        return [i for i in self.issues if i.code == code]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        steps_desc = (
            f"step {self.steps_audited[0]}"
            if len(self.steps_audited) == 1
            else f"{len(self.steps_audited)} step(s)"
        )
        header = (
            f"Audit [{status}] timeline={self.timeline_id} ({steps_desc}): "
            f"{self.error_count} error(s), {self.warning_count} warning(s), {self.info_count} info"
        )
        if not self.issues:
            return header
        lines = [header]
        for issue in self.issues:
            subj = f" [{issue.subject_key}]" if issue.subject_key else ""
            lines.append(
                f"  - [{issue.severity.upper()}] step {issue.step} ({issue.code}){subj}: {issue.message}"
            )
            if issue.suggestion:
                lines.append(f"      suggestion: {issue.suggestion}")
        return "\n".join(lines)


@dataclass(frozen=True)
class AuditContext:
    """Read-only point-in-time snapshot passed to every AuditRule."""

    timeline_id: str
    step: int
    source_text: str = ""
    target_text: str = ""
    entities: tuple[EntityRecord, ...] = ()
    new_entities: tuple[EntityRecord, ...] = ()
    aliases: tuple[AliasRecord, ...] = ()
    edges: tuple[DirectedEdgeRecord, ...] = ()
    mutated_edges: tuple[DirectedEdgeRecord, ...] = ()
    episodes: tuple[EpisodeRecord, ...] = ()
    all_episode_ids: frozenset[str] = frozenset()
    policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY
    lang: str | None = None


class AuditRule(Protocol):
    """Protocol for pluggable QA and continuity audit rules."""

    name: str
    required_tiers: frozenset[int]

    def check(self, ctx: AuditContext) -> Sequence[AuditIssue]: ...


class CallableAuditRule:
    """Adapter turning a plain Python function into an AuditRule."""

    def __init__(
        self,
        name: str,
        fn: Callable[[AuditContext], Sequence[AuditIssue]],
        *,
        required_tiers: frozenset[int] = frozenset({1}),
    ) -> None:
        self.name = name
        self._fn = fn
        self.required_tiers = frozenset(required_tiers)

    def check(self, ctx: AuditContext) -> Sequence[AuditIssue]:
        return list(self._fn(ctx))


def _phrase_pattern(phrase: str) -> str:
    prefix = "" if _CJK_CHAR_RE.match(phrase[0]) else r"(?<!\w)"
    suffix = "" if _CJK_CHAR_RE.match(phrase[-1]) else r"(?!\w)"
    return rf"{prefix}{re.escape(phrase)}{suffix}"


def _extract_snippet(text: str, start: int, end: int, radius: int = 30) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}"


def _normalize_label(text: str) -> str:
    cleaned = re.sub(r"[\s_\-]+", " ", text.strip().lower()).strip()
    return _LEADING_ARTICLES_RE.sub("", cleaned).strip()


def _entity_id_slug(entity_id: str) -> str:
    low = entity_id.strip().lower()
    for prefix in _ENTITY_ID_PREFIXES:
        if low.startswith(prefix):
            low = low[len(prefix) :]
            break
    return _normalize_label(low)


class ExonymLeakRule:
    """Detects untranslated canonical_name leakage in target_text when target_name != canonical_name."""

    name: str = "exonym_leak"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        severity: AuditSeverity = "error",
        case_sensitive: bool = True,
        only_source_mentioned: bool = True,
        snippet_radius: int = 30,
    ) -> None:
        self.severity: AuditSeverity = severity
        self.case_sensitive = case_sensitive
        self.only_source_mentioned = only_source_mentioned
        self.snippet_radius = snippet_radius

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if not ctx.target_text.strip():
            return []

        issues: list[AuditIssue] = []
        aliases_by_entity: dict[str, list[str]] = {}
        for a in ctx.aliases:
            if a.alias.strip():
                aliases_by_entity.setdefault(a.entity_id, []).append(a.alias.strip())

        flags = 0 if self.case_sensitive else re.IGNORECASE
        source_text = ctx.source_text
        has_source = bool(source_text.strip())

        for entity in ctx.entities:
            if entity.base_attributes.get("allow_source_name"):
                continue
            canonical = entity.canonical_name.strip()
            raw_target = entity.base_attributes.get("target_name")
            if not canonical or not isinstance(raw_target, str) or not raw_target.strip():
                continue
            target_name = raw_target.strip()
            if canonical.lower() == target_name.lower():
                continue

            # Guard against canonical_name appearing as a valid word inside target_name
            if re.search(_phrase_pattern(canonical), target_name, re.IGNORECASE):
                continue

            if self.only_source_mentioned and has_source:
                cues = [canonical, *aliases_by_entity.get(entity.entity_id, ())]
                if not any(
                    re.search(_phrase_pattern(cue), source_text, re.IGNORECASE)
                    for cue in cues
                    if cue
                ):
                    continue

            matches = list(re.finditer(_phrase_pattern(canonical), ctx.target_text, flags))
            if not matches:
                continue

            spans = [(m.start(), m.end()) for m in matches]
            first = matches[0]
            issues.append(
                AuditIssue(
                    code="exonym_leak",
                    rule_name=self.name,
                    severity=self.severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key=entity.entity_id,
                    entity_ids=[entity.entity_id],
                    message=(
                        f"Untranslated canonical_name '{canonical}' leaked into target_text "
                        f"({len(matches)} occurrence(s)); expected target_name '{target_name}'."
                    ),
                    evidence={
                        "canonical_name": canonical,
                        "expected_target_name": target_name,
                        "occurrence_count": len(matches),
                        "matched_spans": spans,
                        "snippet": _extract_snippet(
                            ctx.target_text, first.start(), first.end(), self.snippet_radius
                        ),
                    },
                    suggestion=f"Replace '{canonical}' with '{target_name}' (or its appropriate inflected form).",
                )
            )

        return issues


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

        # Build existing labels: canonical_name, target_name, and aliases/personas
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
                is_alias = e_src.startswith("alias:") or e_src.startswith("persona:")
                # Respect same_kind_only for name-to-name comparisons; aliases still catch cross-kind splits
                if self.same_kind_only and not is_alias and candidate.kind != existing.kind:
                    continue
                if c_lbl == e_lbl:
                    score = 100.0
                elif len(c_lbl) >= 4 and len(e_lbl) >= 4:
                    # Use ratio (strict) rather than WRatio to avoid partial-substring false positives
                    score = float(fuzz.ratio(c_lbl, e_lbl))
                else:
                    continue

                if score >= self.fuzzy_threshold:
                    if best is None or score > best[0]:
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
            sorted_entities = sorted(
                ctx.entities, key=lambda e: (e.valid_from_step, e.entity_id)
            )
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
                        "existing_entity_id": existing.entity_id,
                        "existing_canonical_name": existing.canonical_name,
                        "candidate_field": c_src,
                        "matched_against": e_src,
                        "matched_value": matched_val,
                        "similarity": round(score, 2),
                    },
                    suggestion=(
                        f"Merge '{cand.entity_id}' into '{existing.entity_id}' (or register "
                        f"'{cand.canonical_name}' as an AliasRecord on '{existing.entity_id}')."
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

        # Auto-detect mixed T-V tokens (Vi/ti, vous/tu, Sie/du) alongside generic English tokens (formal/informal)
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


def default_audit_rules(preset: DomainPreset | None = None) -> list[AuditRule]:
    """Returns the default built-in AuditRule suite."""
    _ = preset  # Reserved for future preset-specific defaults; ExonymLeakRule is a no-op without target_text.
    return [
        ExonymLeakRule(),
        NearDuplicateEntityRule(),
        UnexplainedRegisterFlipRule(),
        RegisterVocabularyDriftRule(),
    ]


def _load_graph_state(
    engine: CanonArcEngine,
    timeline_id: str,
    step: int,
) -> tuple[list[EntityRecord], list[AliasRecord], list[DirectedEdgeRecord]]:
    get_state = getattr(engine.store, "get_active_graph_state", None)
    if callable(get_state):
        return get_state(timeline_id, step)
    entities = engine.store.get_all_active_entities(timeline_id, step)
    aliases = engine.store.get_all_active_aliases(timeline_id, step)
    edges = engine.store.get_all_active_edges(timeline_id, step)
    return entities, aliases, edges


def build_audit_context(
    engine: CanonArcEngine,
    step: int,
    *,
    source_text: str = "",
    target_text: str = "",
    timeline_id: str | None = None,
    lang: str | None = None,
) -> AuditContext:
    """Constructs a read-only AuditContext snapshot for a single step."""
    tl = timeline_id or engine.active_timeline
    entities, aliases, edges = _load_graph_state(engine, tl, step)
    new_entities = tuple(
        e for e in entities if e.valid_from_step == step and e.revision_count <= 1
    )
    mutated_edges = tuple(e for e in edges if e.valid_from_step == step)

    episodes_at_step: tuple[EpisodeRecord, ...] = ()
    all_episode_ids: frozenset[str] = frozenset()
    if engine._policy.has_tier(6):
        ep_rows = engine.store.get_lineage_episodes(tl, step)
        all_eps = [ep for ep, _ in ep_rows]
        all_episode_ids = frozenset(ep.episode_id for ep in all_eps if ep.episode_id)
        episodes_at_step = tuple(ep for ep in all_eps if ep.step == step)

    return AuditContext(
        timeline_id=tl,
        step=step,
        source_text=source_text,
        target_text=target_text,
        entities=tuple(entities),
        new_entities=new_entities,
        aliases=tuple(aliases),
        edges=tuple(edges),
        mutated_edges=mutated_edges,
        episodes=episodes_at_step,
        all_episode_ids=all_episode_ids,
        policy=engine._policy,
        lang=lang,
    )


def _select_rules(
    engine: CanonArcEngine,
    rules: Sequence[AuditRule] | None,
    *,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
) -> list[AuditRule]:
    if rules is not None:
        selected = list(rules)
    elif engine.audit_rules is not None:
        selected = list(engine.audit_rules)
    else:
        selected = default_audit_rules(engine.preset)

    if include_rules is not None:
        allowed = {r.strip() for r in include_rules if r.strip()}
        selected = [r for r in selected if r.name in allowed]
    if exclude_rules is not None:
        blocked = {r.strip() for r in exclude_rules if r.strip()}
        selected = [r for r in selected if r.name not in blocked]

    active_tiers = engine._policy.active_tiers
    return [r for r in selected if r.required_tiers.issubset(active_tiers)]


def run_audit_step(
    engine: CanonArcEngine,
    step: int,
    *,
    source_text: str = "",
    target_text: str = "",
    timeline_id: str | None = None,
    rules: Sequence[AuditRule] | None = None,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
    min_severity: AuditSeverity = "info",
    lang: str | None = None,
) -> AuditReport:
    """Runs all applicable AuditRules against a single timeline step."""
    tl = timeline_id or engine.active_timeline
    ctx = build_audit_context(
        engine,
        step,
        source_text=source_text,
        target_text=target_text,
        timeline_id=tl,
        lang=lang,
    )
    active_rules = _select_rules(
        engine, rules, include_rules=include_rules, exclude_rules=exclude_rules
    )
    min_rank = _SEVERITY_RANK[min_severity]

    issues: list[AuditIssue] = []
    for rule in active_rules:
        for issue in rule.check(ctx):
            if _SEVERITY_RANK[issue.severity] >= min_rank:
                issues.append(issue)

    return AuditReport(
        timeline_id=tl,
        steps_audited=[step],
        issues=issues,
        rules_executed=[r.name for r in active_rules],
    )


def run_audit_timeline(
    engine: CanonArcEngine,
    *,
    timeline_id: str | None = None,
    from_step: int = 0,
    to_step: int | None = None,
    texts: Mapping[int, str | tuple[str, str]] | None = None,
    rules: Sequence[AuditRule] | None = None,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
    min_severity: AuditSeverity = "info",
    lang: str | None = None,
) -> AuditReport:
    """Audits every step in [from_step..to_step] on a timeline and aggregates issues."""
    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")

    end_step = tl_info.head_step if to_step is None else to_step
    if texts:
        max_text_step = max(texts.keys())
        if to_step is None and max_text_step > end_step:
            end_step = max_text_step

    steps = list(range(max(0, from_step), end_step + 1))
    active_rules = _select_rules(
        engine, rules, include_rules=include_rules, exclude_rules=exclude_rules
    )
    min_rank = _SEVERITY_RANK[min_severity]

    all_issues: list[AuditIssue] = []
    seen_global_keys: set[tuple[str, str, str | None]] = set()

    for s in steps:
        src = ""
        tgt = ""
        if texts and s in texts:
            entry = texts[s]
            if isinstance(entry, tuple):
                src, tgt = entry[0], entry[1]
            else:
                tgt = entry

        ctx = build_audit_context(
            engine,
            s,
            source_text=src,
            target_text=tgt,
            timeline_id=tl,
            lang=lang,
        )
        for rule in active_rules:
            for issue in rule.check(ctx):
                if _SEVERITY_RANK[issue.severity] < min_rank:
                    continue
                # Deduplicate timeline-wide drift warnings across consecutive steps
                if issue.code == "register_vocabulary_drift":
                    dedup_key = (issue.code, issue.message, issue.subject_key)
                    if dedup_key in seen_global_keys:
                        continue
                    seen_global_keys.add(dedup_key)
                all_issues.append(issue)

    return AuditReport(
        timeline_id=tl,
        steps_audited=steps,
        issues=all_issues,
        rules_executed=[r.name for r in active_rules],
    )
