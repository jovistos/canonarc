from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ..models import (
    AliasRecord,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    ResolvedMention,
)
from ..presets import DEFAULT_SCHEMA_POLICY, SchemaPolicy
from ..text_masking import TextNormalizer, apply_text_masking

AuditSeverity = Literal["info", "warning", "error"]

_SEVERITY_RANK: dict[AuditSeverity, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
}


class DuplicateCandidate(BaseModel):
    """Structured near-duplicate entity pair ready for inspection or engine.merge_entities()."""

    duplicate_id: str
    canonical_id: str
    score: float = Field(description="Similarity score in [0.0, 100.0]")
    reason: str = Field(description="Human-readable explanation of why the pair collided")
    duplicate_name: str = ""
    canonical_name: str = ""
    duplicate_kind: str = ""
    canonical_kind: str = ""
    candidate_field: str = ""
    matched_against: str = ""
    matched_value: str = ""
    step: int = 0
    timeline_id: str = "main"


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


def _duplicate_candidate_from_issue(issue: AuditIssue) -> DuplicateCandidate | None:
    ev = issue.evidence or {}
    dup_id = str(ev.get("new_entity_id") or (issue.entity_ids[0] if len(issue.entity_ids) >= 1 else "")).strip()
    canon_id = str(ev.get("existing_entity_id") or (issue.entity_ids[1] if len(issue.entity_ids) >= 2 else "")).strip()
    if not dup_id or not canon_id or dup_id == canon_id:
        return None
    raw_score = ev.get("similarity", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    cand_field = str(ev.get("candidate_field") or "")
    matched_against = str(ev.get("matched_against") or "")
    matched_value = str(ev.get("matched_value") or "")
    if cand_field and matched_against:
        reason = f"{cand_field} matched {matched_against} ({score:.1f}%)"
    elif matched_against:
        reason = f"matched {matched_against} ({score:.1f}%)"
    else:
        reason = issue.message
    return DuplicateCandidate(
        duplicate_id=dup_id,
        canonical_id=canon_id,
        score=round(score, 2),
        reason=reason,
        duplicate_name=str(ev.get("new_canonical_name") or ""),
        canonical_name=str(ev.get("existing_canonical_name") or ""),
        duplicate_kind=str(ev.get("new_kind") or ""),
        canonical_kind=str(ev.get("existing_kind") or ""),
        candidate_field=cand_field,
        matched_against=matched_against,
        matched_value=matched_value,
        step=issue.step,
        timeline_id=issue.timeline_id,
    )


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

    def duplicate_candidates(self, *, min_score: float = 0.0) -> list[DuplicateCandidate]:
        """Extracts typed DuplicateCandidate records from near_duplicate_entity issues.

        Deduplicates repeated (duplicate_id, canonical_id) pairs across multi-step
        reports by keeping the highest-scoring occurrence, sorted by descending score.
        """
        best_by_pair: dict[tuple[str, str], DuplicateCandidate] = {}
        for issue in self.issues:
            if issue.code != "near_duplicate_entity":
                continue
            cand = _duplicate_candidate_from_issue(issue)
            if cand is None or cand.score < min_score:
                continue
            key = (cand.duplicate_id, cand.canonical_id)
            prev = best_by_pair.get(key)
            if prev is None or cand.score > prev.score:
                best_by_pair[key] = cand
        return sorted(
            best_by_pair.values(),
            key=lambda c: (-c.score, c.duplicate_id, c.canonical_id),
        )

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

    def format_repair_instructions(
        self,
        *,
        min_severity: AuditSeverity = "error",
        max_issues: int | None = None,
        header: str | None = (
            "Fix the following terminology/continuity errors in your translation "
            "while preserving everything else:"
        ),
        include_snippets: bool = True,
    ) -> str:
        """Formats failed issues at or above min_severity as a compact LLM-ready repair bullet list."""
        threshold = _SEVERITY_RANK[min_severity]
        filtered = [i for i in self.issues if _SEVERITY_RANK[i.severity] >= threshold]
        if not filtered:
            return ""

        filtered.sort(key=lambda i: _SEVERITY_RANK[i.severity], reverse=True)

        deduped: list[AuditIssue] = []
        seen_term_subjects: set[str] = set()
        term_codes = {"exonym_leak", "target_term_mismatch", "target_term_missing"}
        for issue in filtered:
            if issue.subject_key and issue.code in term_codes:
                if issue.subject_key in seen_term_subjects:
                    continue
                seen_term_subjects.add(issue.subject_key)
            deduped.append(issue)

        if max_issues is not None and max_issues >= 0:
            deduped = deduped[:max_issues]
        if not deduped:
            return ""

        lines: list[str] = []
        if header:
            lines.append(header)
        for issue in deduped:
            subj = f"[{issue.subject_key}] " if issue.subject_key else ""
            body = (
                issue.suggestion.strip()
                if issue.suggestion and issue.suggestion.strip()
                else issue.message.strip()
            )
            snippet = issue.evidence.get("snippet") if include_snippets else None
            if isinstance(snippet, str) and snippet.strip():
                lines.append(f"- {subj}{body} (at: {snippet!r})")
            else:
                lines.append(f"- {subj}{body}")
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
    resolved_mentions: tuple[ResolvedMention, ...] = ()
    policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY
    lang: str | None = None
    ignore_patterns: tuple[re.Pattern[str], ...] = ()
    text_normalizer: TextNormalizer | None = None

    @property
    def masked_source_text(self) -> str:
        """Returns source_text with engine ignore_patterns (and optional text_normalizer) applied."""
        return apply_text_masking(
            self.source_text, self.ignore_patterns, normalizer=self.text_normalizer
        )

    @property
    def masked_target_text(self) -> str:
        """Returns target_text with engine ignore_patterns (and optional text_normalizer) applied."""
        return apply_text_masking(
            self.target_text, self.ignore_patterns, normalizer=self.text_normalizer
        )

    @property
    def target_snippet_text(self) -> str:
        """Base text whose character offsets match masked_target_text for snippet extraction."""
        return self.masked_target_text if self.text_normalizer is not None else self.target_text


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
