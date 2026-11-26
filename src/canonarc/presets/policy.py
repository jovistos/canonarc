from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..episodic import EpisodicScoringPolicy
from ..models import StepContext


class TierProfile(str, Enum):
    """Built-in declarative tier profiles for CanonArcEngine."""

    LEXICON_ONLY = "lexicon_only"  # Tiers 1 + 4 (base attributes + temporal/alias coreference)
    STATE_ONLY = "state_only"      # Tiers 1-4 (adds directed edges + rolling state deltas)
    FULL = "full"                  # Tiers 1-6 (adds Voice Bank + Episodic Memory/POV)


ALL_TIERS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6})

PROFILE_TIERS: dict[TierProfile, frozenset[int]] = {
    TierProfile.LEXICON_ONLY: frozenset({1, 4}),
    TierProfile.STATE_ONLY: frozenset({1, 2, 3, 4}),
    TierProfile.FULL: ALL_TIERS,
}


@dataclass(frozen=True)
class SchemaPolicy:
    """Declarative domain policy controlling active tiers, edge layering, disguise keys, and voice ranking.

    Allows any domain (interactive fiction, localization, non-fiction glossary translation,
    financial/social simulations, healthcare, etc.) to enable/disable architectural tiers
    and customize which state/metadata keys act as hard filters vs. soft boosts without
    modifying the core engine or VoiceBank.
    """

    profile: TierProfile = TierProfile.FULL
    enabled_tiers: frozenset[int] | None = None
    edge_layer_order: tuple[str, ...] = ("stance", "facade")
    facade_relation: str = "facade"
    belief_relation: str = "belief_state"
    entity_disguise_key: str = "disguise"
    exemplar_persona_key: str = "persona"
    masked_entity_kind: str = "character"
    hard_filter_keys: tuple[str, ...] = ("register",)
    soft_boost_weights: dict[str, float] = field(default_factory=lambda: {"tone": 0.15})
    addressee_boost: float = 0.10
    quality_weight: float = 0.05
    fallback_on_empty_hard_filter: bool = True
    hard_filter_mismatch_penalty: float = 0.25
    hard_filter_fallback_note: str = "[Note: different {keys}]"
    episodic_policy: EpisodicScoringPolicy = field(default_factory=EpisodicScoringPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, TierProfile):
            object.__setattr__(self, "profile", TierProfile(self.profile))
        if self.enabled_tiers is not None:
            tiers = frozenset(int(t) for t in self.enabled_tiers)
            invalid = tiers - ALL_TIERS
            if invalid:
                raise ValueError(
                    f"Invalid tier(s) in enabled_tiers: {sorted(invalid)}. Expected values in 1..6."
                )
            if 1 not in tiers:
                raise ValueError("Tier 1 (base entities) must remain enabled in SchemaPolicy.")
            object.__setattr__(self, "enabled_tiers", tiers)

    @property
    def active_tiers(self) -> frozenset[int]:
        if self.enabled_tiers is not None:
            return self.enabled_tiers
        return PROFILE_TIERS[self.profile]

    def has_tier(self, tier: int) -> bool:
        return tier in self.active_tiers

    @classmethod
    def lexicon_only(cls, **kwargs: Any) -> SchemaPolicy:
        """Creates a LEXICON_ONLY policy (Tiers 1 + 4: glossary, aliases, inflections, checkpoints)."""
        return cls(profile=TierProfile.LEXICON_ONLY, **kwargs)

    @classmethod
    def state_only(cls, **kwargs: Any) -> SchemaPolicy:
        """Creates a STATE_ONLY policy (Tiers 1-4: entities, aliases, edges, rolling state)."""
        return cls(profile=TierProfile.STATE_ONLY, **kwargs)

    @classmethod
    def full(cls, **kwargs: Any) -> SchemaPolicy:
        """Creates a FULL policy (Tiers 1-6: complete narrative memory with voice & episodes)."""
        return cls(profile=TierProfile.FULL, **kwargs)

    @classmethod
    def with_tiers(cls, tiers: Iterable[int], **kwargs: Any) -> SchemaPolicy:
        """Creates a policy with an explicit custom subset of active tiers."""
        return cls(enabled_tiers=frozenset(int(t) for t in tiers), **kwargs)


DEFAULT_SCHEMA_POLICY = SchemaPolicy()


class DomainPreset(Protocol):
    """Interface for rendering StepContext into prompt blocks and formatting extraction instructions."""

    policy: SchemaPolicy

    def format_prompt_context(self, ctx: StepContext) -> str: ...
    def extraction_instructions(self) -> str: ...
