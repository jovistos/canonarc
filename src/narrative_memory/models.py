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
    valid_from_step: int = 0
    valid_to_step: int | None = None


class AliasRecord(BaseModel):
    """Tier 4: 1-hop alias / coreference / disguise mapping."""
    alias: str = Field(description="Surface string or epithet, e.g. 'the old wizard'")
    entity_id: str = Field(description="Target canonical entity_id, e.g. 'char_merlin'")
    kind: Literal["permanent", "epithet", "disguise", "title"] = "epithet"
    valid_from_step: int = 0
    valid_to_step: int | None = None


class DirectedEdgeRecord(BaseModel):
    """Tier 2, 3 & 4: Directed pairwise relationship (source_id -> target_id)."""
    source_id: str
    target_id: str
    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Materialized state, e.g. {'register': 'ti', 'tone': 'angry', 'trust': -0.6}",
    )
    valid_from_step: int = 0
    valid_to_step: int | None = None


# --- Tier 3 Deltas (Resolved-on-Write) ---

class EntityDelta(BaseModel):
    """Mutation to an entity at a discrete step."""
    entity_id: str
    canonical_name: str | None = None
    kind: str | None = None
    base_attributes: dict[str, Any] = Field(default_factory=dict, description="Merged into base_attributes")
    state: dict[str, Any] = Field(default_factory=dict, description="Merged into rolling state")
    merge_state: bool = True
    add_aliases: list[str] = Field(default_factory=list)
    remove_aliases: list[str] = Field(default_factory=list)
    remove: bool = False


class EdgeDelta(BaseModel):
    """Mutation to a directed pairwise edge at a discrete step."""
    source_id: str
    target_id: str
    state: dict[str, Any] = Field(default_factory=dict)
    merge: bool = Field(default=True, description="Resolved-on-write: merges with prior active A->B state")
    remove: bool = Field(default=False, description="Writes a tombstone closing A->B at this step")


class StepDelta(BaseModel):
    """Structured JSON diff applied at a discrete step."""
    entities: list[EntityDelta] = Field(default_factory=list)
    edges: list[EdgeDelta] = Field(default_factory=list)


# --- Tier 5 Voice Bank & Anti-Drift Gates ---

class DialogueCandidate(BaseModel):
    """Candidate dialogue span extracted deterministically."""
    candidate_id: str
    text: str
    source_text: str = ""
    word_count: int
    distinctiveness_score: float
    lexical_diversity: float


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
    """Unified Tier 1-5 state retrieved for a specific step."""
    timeline_id: str
    step: int
    resolved_mentions: list[ResolvedMention] = Field(default_factory=list)
    entities: list[EntityRecord] = Field(default_factory=list)
    active_edges: list[DirectedEdgeRecord] = Field(default_factory=list)
    voice_exemplars: list[VoiceExemplar] = Field(default_factory=list)
