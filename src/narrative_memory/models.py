from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class TimelineInfo(BaseModel):
    """Metadata for a timeline branch."""
    timeline_id: str
    parent_id: str | None = None
    fork_step: int | None = None
    head_step: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    kind: Literal["permanent", "epithet", "disguise", "title"] = "epithet"
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
    epistemic: Literal["direct", "told", "deceived"] = "told"
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


class StepContext(BaseModel):
    """Unified Tier 1-6 state retrieved for a specific step."""
    timeline_id: str
    step: int
    resolved_mentions: list[ResolvedMention] = Field(default_factory=list)
    entities: list[EntityRecord] = Field(default_factory=list)
    active_edges: list[DirectedEdgeRecord] = Field(default_factory=list)
    voice_exemplars: list[VoiceExemplar] = Field(default_factory=list)
    episodes: list[EpisodeRecord] = Field(default_factory=list)
