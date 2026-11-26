from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EpistemicMode = Literal["direct", "told", "deceived", "forgotten", "suppressed"]


class TimelineInfo(BaseModel):
    """Metadata for a timeline branch."""
    timeline_id: str
    parent_id: str | None = None
    fork_step: int | None = None
    head_step: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepCheckpoint(BaseModel):
    """Recorded proof that a discrete step was committed on a timeline."""
    timeline_id: str = "main"
    step: int
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResumePlan(BaseModel):
    """Structured resume and invalidation plan produced by CanonArcEngine.plan_resume."""
    timeline_id: str = "main"
    causal: bool = True
    reusable_steps: list[int] = Field(default_factory=list)
    pending_steps: list[int] = Field(default_factory=list)
    missing_steps: list[int] = Field(default_factory=list)
    stale_steps: list[int] = Field(default_factory=list)
    invalidated_steps: list[int] = Field(default_factory=list)
    orphaned_steps: list[int] = Field(default_factory=list)
    first_stale_step: int | None = None
    rollback_to_step: int | None = None
    fork_conflict_step: int | None = None
    rollback_applied: bool = False
    deleted_rows: int = 0

    @property
    def steps_to_purge(self) -> list[int]:
        """Sorted union of stale, causally invalidated, and orphaned steps whose external outputs should be purged."""
        return sorted(set(self.stale_steps) | set(self.invalidated_steps) | set(self.orphaned_steps))

    @property
    def is_up_to_date(self) -> bool:
        """True when all planned steps are reusable and no rollback or purge is pending."""
        return not self.pending_steps and self.rollback_to_step is None and not self.orphaned_steps

    @property
    def needs_rollback(self) -> bool:
        """True when a database rollback or targeted step purge is required and has not yet been applied."""
        if self.rollback_applied:
            return False
        if self.causal:
            return self.rollback_to_step is not None
        return bool(self.stale_steps or self.orphaned_steps)


class EntityRecord(BaseModel):
    """Unified Tier 1 (base_attributes) + Tier 3 (rolling state) entity record."""
    entity_id: str = Field(description="Canonical ID, e.g. 'char_arthur'")
    kind: str = Field(default="character", description="'character', 'faction', 'location', 'artifact', 'term'")
    canonical_name: str = Field(description="Primary display name, e.g. 'Arthur'")
    base_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Tier 1 attributes (gender, declension, archetype, target_rendering)",
    )
    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Tier 3 rolling state (e.g. {'mood': 'furious', 'disguise': 'beggar'})",
    )
    private_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal state hidden from other observers (true intent, secret injury, guilt)",
    )
    disguised_as: str | None = Field(
        default=None,
        description="Epistemic annotation when a disguise is active and pierced by the POV observer",
    )
    valid_from_step: int = 0
    valid_to_step: int | None = None
    # Causal provenance (Pitfall 2) & state trajectory (Pitfall 5)
    caused_by_episode_id: str | None = None
    reason: str | None = None
    prior_state: dict[str, Any] | None = None
    prior_step: int | None = None
    revision_count: int = 1


class AliasRecord(BaseModel):
    """Tier 4: 1-hop alias / coreference / disguise mapping."""
    alias: str = Field(description="Surface string or epithet, e.g. 'the old wizard'")
    entity_id: str = Field(description="Target canonical entity_id, e.g. 'char_merlin'")
    kind: Literal["permanent", "epithet", "disguise", "title", "inflection"] = "epithet"
    lang: str | None = Field(
        default=None,
        description="Optional ISO/BCP-47 language tag (e.g. 'en', 'sr', 'ru'); None means language-neutral",
    )
    known_to: list[str] | None = Field(
        default=None,
        description="If set, only these entity_ids resolve this alias to entity_id; None means universally known",
    )
    persona_name: str | None = Field(
        default=None,
        description="Surface identity shown to fooled observers when kind='disguise'",
    )
    valid_from_step: int = 0
    valid_to_step: int | None = None


class DirectedEdgeRecord(BaseModel):
    """Tier 2, 3 & 4: Directed typed relationship (source_id -[relation_type]-> target_id)."""
    source_id: str
    target_id: str
    relation_type: str = Field(
        default="stance",
        description="Predicate/category of the edge, e.g. 'stance', 'kinship', 'debt', 'allegiance'",
    )
    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Materialized state for this relation_type, e.g. {'register': 'ti', 'tone': 'angry'}",
    )
    valid_from_step: int = 0
    valid_to_step: int | None = None
    # Causal provenance (Pitfall 2) & state trajectory (Pitfall 5)
    caused_by_episode_id: str | None = None
    reason: str | None = None
    prior_state: dict[str, Any] | None = None
    prior_step: int | None = None
    revision_count: int = 1


# --- Tier 6: Episodic Event Ledger ---

class EpisodeRecord(BaseModel):
    """Tier 6: Discrete N-ary narrative event / episode anchored at a causal step."""
    episode_id: str = ""
    timeline_id: str = "main"
    step: int = 0
    predicate: str = Field(
        default="event",
        description="Category of episode, e.g. 'argument', 'betrayal', 'item_transfer', 'revelation'",
    )
    summary: str = Field(
        description="Concise narrative summary, e.g. 'Arthur betrayed Lancelot at the tournament'",
    )
    participants: dict[str, str] = Field(
        default_factory=dict,
        description="N-ary role map: entity_id -> semantic role (e.g. {'char_arthur': 'actor', 'char_lancelot': 'target', 'item_excalibur': 'instrument'})",
    )
    witnesses: list[str] = Field(
        default_factory=list,
        description="Entity IDs who directly witnessed the episode",
    )
    known_by: dict[str, Literal["direct", "told", "deceived"]] = Field(
        default_factory=dict,
        description="Resolved epistemic awareness map at query time (entity_id -> 'direct' | 'told' | 'deceived')",
    )
    believed_summaries: dict[str, str] = Field(
        default_factory=dict,
        description="Per-entity distorted/false summary when epistemic mode is 'deceived'",
    )
    location_id: str | None = Field(
        default=None,
        description="Optional canonical location entity_id where the episode occurred",
    )
    story_time: int | float | str | None = Field(
        default=None,
        description="Optional diegetic / story-world time independent of discourse step",
    )
    chapter_id: str | None = Field(
        default=None,
        description="Optional hierarchical chapter or act identifier",
    )
    scene_id: str | None = Field(
        default=None,
        description="Optional hierarchical scene identifier",
    )
    consolidated_from: list[str] = Field(
        default_factory=list,
        description="Optional list of source episode_ids compacted into this summary episode",
    )
    salience: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Plot importance score in [0.0, 10.0]; high-salience events resist temporal decay",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    match_score: float = 0.0


class InformDelta(BaseModel):
    """Records that an entity learned about a past episode (secondhand, belatedly, or via deception) at the current step."""
    episode_id: str
    entity_id: str
    epistemic: EpistemicMode = "told"
    believed_summary: str | None = Field(
        default=None,
        description="Optional false/distorted summary believed by entity_id when epistemic='deceived'",
    )


# --- Tier 3 Deltas (Resolved-on-Write) ---

class EntityDelta(BaseModel):
    """Mutation to an entity at a discrete step."""
    entity_id: str
    canonical_name: str | None = None
    kind: str | None = None
    base_attributes: dict[str, Any] = Field(default_factory=dict, description="Merged into base_attributes")
    state: dict[str, Any] = Field(default_factory=dict, description="Merged into rolling state")
    private_state: dict[str, Any] = Field(default_factory=dict, description="Merged into private_state")
    merge_state: bool = True
    add_aliases: list[str | AliasRecord] = Field(default_factory=list)
    remove_aliases: list[str] = Field(default_factory=list)
    remove: bool = False
    caused_by_episode_id: str | None = None
    reason: str | None = None


class EdgeDelta(BaseModel):
    """Mutation to a directed typed edge at a discrete step."""
    source_id: str
    target_id: str
    relation_type: str = Field(
        default="stance",
        description="Predicate/category of the edge to mutate",
    )
    state: dict[str, Any] = Field(default_factory=dict)
    merge: bool = Field(default=True, description="Resolved-on-write: merges with prior active state for this predicate")
    remove: bool = Field(default=False, description="Writes a tombstone closing this predicate edge at this step")
    caused_by_episode_id: str | None = None
    reason: str | None = None


class StepDelta(BaseModel):
    """Structured JSON diff applied at a discrete step."""
    entities: list[EntityDelta] = Field(default_factory=list)
    edges: list[EdgeDelta] = Field(default_factory=list)
    episodes: list[EpisodeRecord] = Field(default_factory=list)
    inform: list[InformDelta] = Field(default_factory=list)
    is_flashback: bool = False
    story_time: int | float | str | None = None
    chapter_id: str | None = None
    scene_id: str | None = None


# --- Tier 5 Voice Bank & Anti-Drift Gates ---

class DialogueCandidate(BaseModel):
    """Candidate dialogue span extracted deterministically."""
    candidate_id: str
    text: str
    source_text: str = ""
    word_count: int
    distinctiveness_score: float
    lexical_diversity: float
    speaker_id: str | None = None
    addressee_id: str | None = None
    speaker_surface: str | None = None

    def to_voice_exemplar(
        self,
        *,
        step: int,
        speaker_id: str | None = None,
        addressee_id: str | None = None,
        metadata: dict[str, str] | None = None,
        quality_score: float = 10.0,
        timeline_id: str = "main",
    ) -> VoiceExemplar:
        """Converts this candidate into a VoiceExemplar ready for index_voice_exemplars()."""
        resolved_speaker = speaker_id or self.speaker_id
        if not resolved_speaker:
            raise ValueError(
                "Cannot convert DialogueCandidate to VoiceExemplar without speaker_id."
            )
        resolved_addressee = addressee_id if addressee_id is not None else self.addressee_id
        return VoiceExemplar(
            timeline_id=timeline_id,
            step=step,
            speaker_id=resolved_speaker,
            addressee_id=resolved_addressee,
            text=self.text,
            source_text=self.source_text,
            metadata=dict(metadata) if metadata else {},
            quality_score=quality_score,
            distinctiveness_score=self.distinctiveness_score,
        )


class VoiceExemplar(BaseModel):
    """Empirical dialogue exemplar stored in the Tier 5 Voice Bank."""
    exemplar_id: str = ""
    timeline_id: str = "main"
    step: int
    speaker_id: str
    addressee_id: str | None = None
    text: str
    source_text: str = ""
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'register': 'ti', 'tone': 'angry'}",
    )
    quality_score: float = Field(default=10.0, ge=0.0, le=10.0)
    distinctiveness_score: float = 0.0
    match_score: float = 0.0


class ResolvedMention(BaseModel):
    """Entity mention matched in text."""
    surface_text: str
    entity_id: str
    matched_via: Literal["canonical", "alias", "fuzzy"]
    confidence: float = 1.0
    matched_source: Literal["canonical", "alias"] | None = None
    matched_phrase: str | None = None


class StepContext(BaseModel):
    """Unified Tier 1-6 state retrieved for a specific step."""
    timeline_id: str
    step: int
    story_time: int | float | str | None = None
    chapter_id: str | None = None
    scene_id: str | None = None
    resolved_mentions: list[ResolvedMention] = Field(default_factory=list)
    entities: list[EntityRecord] = Field(default_factory=list)
    active_edges: list[DirectedEdgeRecord] = Field(default_factory=list)
    related_entities: list[EntityRecord] = Field(default_factory=list)
    background_edges: list[DirectedEdgeRecord] = Field(default_factory=list)
    voice_exemplars: list[VoiceExemplar] = Field(default_factory=list)
    episodes: list[EpisodeRecord] = Field(default_factory=list)
    entity_names: dict[str, str] = Field(
        default_factory=dict,
        description="Optional entity_id -> canonical_name lookup for entities referenced in edges, episodes, or voice exemplars",
    )


class EntityMergeReport(BaseModel):
    """Structured summary of an atomic entity merge operation."""
    timeline_id: str
    step: int
    duplicate_id: str
    canonical_id: str
    dry_run: bool = False
    aliases_added: list[str] = Field(default_factory=list)
    aliases_tombstoned: list[str] = Field(default_factory=list)
    base_attributes_merged: list[str] = Field(default_factory=list)
    state_keys_merged: list[str] = Field(default_factory=list)
    edges_migrated: list[str] = Field(default_factory=list)
    edges_merged_on_collision: list[str] = Field(default_factory=list)
    edges_tombstoned: list[str] = Field(default_factory=list)
    self_loops_dropped: list[str] = Field(default_factory=list)
    location_refs_updated: list[str] = Field(default_factory=list)
    voice_exemplars_migrated: int = 0
    episodes_migrated: int = 0
    epistemics_migrated: int = 0
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        prefix = "[DRY RUN] " if self.dry_run else ""
        lines = [
            f"{prefix}Merged '{self.duplicate_id}' -> '{self.canonical_id}' "
            f"(timeline={self.timeline_id}, step={self.step})",
            f"  Aliases promoted: {len(self.aliases_added)}, tombstoned: {len(self.aliases_tombstoned)}",
            f"  Attributes merged: {len(self.base_attributes_merged)}, state keys merged: {len(self.state_keys_merged)}",
            f"  Edges migrated: {len(self.edges_migrated)}, merged on collision: {len(self.edges_merged_on_collision)}, "
            f"tombstoned: {len(self.edges_tombstoned)}, self-loops dropped: {len(self.self_loops_dropped)}",
        ]
        if self.location_refs_updated:
            lines.append(f"  Location refs updated: {len(self.location_refs_updated)}")
        if self.voice_exemplars_migrated or self.episodes_migrated or self.epistemics_migrated:
            lines.append(
                f"  Voice exemplars migrated: {self.voice_exemplars_migrated}, "
                f"episodes migrated: {self.episodes_migrated}, "
                f"epistemics migrated: {self.epistemics_migrated}"
            )
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)
