from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..models import (
    AliasRecord,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    StepContext,
)
from ..presets import DomainPreset
from .models import (
    _SEVERITY_RANK,
    AuditContext,
    AuditIssue,
    AuditReport,
    AuditRule,
    AuditSeverity,
    DuplicateCandidate,
    StepTextEntry,
    StepTextProvider,
    _duplicate_candidate_from_issue,
)
from .rules_continuity import (
    NearDuplicateEntityRule,
    RegisterVocabularyDriftRule,
    UnexplainedRegisterFlipRule,
)
from .rules_translation import ExonymLeakRule, TargetTermComplianceRule

if TYPE_CHECKING:
    from ..engine import CanonArcEngine


def default_audit_rules(preset: DomainPreset | None = None) -> list[AuditRule]:
    """Returns the default built-in AuditRule suite."""
    _ = preset
    return [
        ExonymLeakRule(),
        TargetTermComplianceRule(),
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
    load_episodes: bool | None = None,
    load_episodes_if_edge_refs: bool = False,
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
        if load_episodes is None:
            should_load_episodes = True
        elif load_episodes:
            should_load_episodes = True
        else:
            should_load_episodes = bool(
                load_episodes_if_edge_refs
                and any(
                    e.caused_by_episode_id and e.caused_by_episode_id.strip()
                    for e in mutated_edges
                )
            )
        if should_load_episodes:
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
        ignore_patterns=engine.ignore_patterns,
        text_normalizer=engine.text_normalizer,
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


def _unpack_step_text(entry: StepTextEntry | None) -> tuple[str, str]:
    if entry is None:
        return "", ""
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return "", entry


def _episode_loading_flags(active_rules: Sequence[AuditRule]) -> tuple[bool, bool]:
    needs_episodes = any(6 in r.required_tiers for r in active_rules)
    needs_edge_ep_refs = any(
        bool(getattr(r, "check_dangling_episode", False)) for r in active_rules
    )
    return needs_episodes, needs_edge_ep_refs


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
    active_rules = _select_rules(
        engine, rules, include_rules=include_rules, exclude_rules=exclude_rules
    )
    needs_episodes, needs_edge_ep_refs = _episode_loading_flags(active_rules)
    ctx = build_audit_context(
        engine,
        step,
        source_text=source_text,
        target_text=target_text,
        timeline_id=tl,
        lang=lang,
        load_episodes=needs_episodes,
        load_episodes_if_edge_refs=needs_edge_ep_refs,
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


def run_audit_draft(
    engine: CanonArcEngine,
    step: int,
    *,
    source_text: str = "",
    target_text: str = "",
    context: StepContext | None = None,
    timeline_id: str | None = None,
    rules: Sequence[AuditRule] | None = None,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
    min_severity: AuditSeverity = "info",
    lang: str | None = None,
    include_all_timeline_entities: bool = False,
) -> AuditReport:
    """Audits a pre-commit draft translation against the pre-commit state (step - 1 / StepContext)."""
    tl = timeline_id or (context.timeline_id if context is not None else engine.active_timeline)
    cutoff_step = max(0, step - 1)
    all_entities, all_aliases, all_edges = _load_graph_state(engine, tl, cutoff_step)

    if context is not None and not include_all_timeline_entities:
        ctx_entity_map: dict[str, EntityRecord] = {e.entity_id: e for e in context.entities}
        for rel in context.related_entities:
            ctx_entity_map.setdefault(rel.entity_id, rel)
        entities = tuple(ctx_entity_map.values())
        active_ids = set(ctx_entity_map.keys())
        aliases = tuple(a for a in all_aliases if a.entity_id in active_ids)
        edges = tuple(context.active_edges)
        resolved_mentions = tuple(context.resolved_mentions)
    else:
        entities = tuple(all_entities)
        aliases = tuple(all_aliases)
        edges = tuple(all_edges)
        resolved_mentions = tuple(context.resolved_mentions) if context is not None else ()

    ctx = AuditContext(
        timeline_id=tl,
        step=step,
        source_text=source_text,
        target_text=target_text,
        entities=entities,
        new_entities=(),
        aliases=aliases,
        edges=edges,
        mutated_edges=(),
        episodes=tuple(context.episodes) if context is not None else (),
        all_episode_ids=frozenset(
            ep.episode_id for ep in (context.episodes if context is not None else ()) if ep.episode_id
        ),
        resolved_mentions=resolved_mentions,
        policy=engine._policy,
        lang=lang,
        ignore_patterns=engine.ignore_patterns,
        text_normalizer=engine.text_normalizer,
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
    texts: Mapping[int, StepTextEntry] | None = None,
    text_provider: StepTextProvider | None = None,
    rules: Sequence[AuditRule] | None = None,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
    min_severity: AuditSeverity = "info",
    lang: str | None = None,
) -> AuditReport:
    """Audits every step in [from_step..to_step] on a timeline and aggregates issues.

    Per-step prose can be supplied either as an in-memory ``texts`` mapping or lazily via
    ``text_provider(step) -> target_text | (source_text, target_text) | None``.
    """
    if texts is not None and text_provider is not None:
        raise ValueError("Pass either 'texts' or 'text_provider', not both.")

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
    needs_episodes, needs_edge_ep_refs = _episode_loading_flags(active_rules)
    min_rank = _SEVERITY_RANK[min_severity]

    all_issues: list[AuditIssue] = []
    seen_global_keys: set[tuple[str, str, str | None]] = set()

    for s in steps:
        if text_provider is not None:
            src, tgt = _unpack_step_text(text_provider(s))
        elif texts is not None and s in texts:
            src, tgt = _unpack_step_text(texts[s])
        else:
            src, tgt = "", ""

        ctx = build_audit_context(
            engine,
            s,
            source_text=src,
            target_text=tgt,
            timeline_id=tl,
            lang=lang,
            load_episodes=needs_episodes,
            load_episodes_if_edge_refs=needs_edge_ep_refs,
        )
        for rule in active_rules:
            for issue in rule.check(ctx):
                if _SEVERITY_RANK[issue.severity] < min_rank:
                    continue
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


def find_duplicate_candidates(
    engine: CanonArcEngine,
    *,
    at_step: int | None = None,
    timeline_id: str | None = None,
    fuzzy_threshold: float = 88.0,
    same_kind_only: bool = True,
) -> list[DuplicateCandidate]:
    """Finds all near-duplicate entity pairs active at (timeline_id, at_step)."""
    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")
    eff_step = tl_info.head_step if at_step is None else at_step
    if eff_step < 0:
        raise ValueError(f"at_step must be non-negative, got {eff_step}.")

    ctx = build_audit_context(engine, eff_step, timeline_id=tl)
    rule = NearDuplicateEntityRule(
        fuzzy_threshold=fuzzy_threshold,
        same_kind_only=same_kind_only,
        check_all_entities=True,
    )
    issues = rule.check(ctx)
    candidates = [
        cand for issue in issues if (cand := _duplicate_candidate_from_issue(issue)) is not None
    ]
    candidates.sort(key=lambda c: (-c.score, c.duplicate_id, c.canonical_id))
    return candidates
