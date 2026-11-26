from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol

from .episodic import EpisodicScoringPolicy
from .models import DirectedEdgeRecord, EntityRecord, StepContext, VoiceExemplar

logger = logging.getLogger(__name__)


def _cap_voice_exemplars_per_speaker(
    exemplars: Sequence[VoiceExemplar],
    max_per_speaker: int | None,
) -> list[VoiceExemplar]:
    """Caps voice exemplars per speaker while preserving existing rank order."""
    if max_per_speaker is None:
        return list(exemplars)
    if max_per_speaker <= 0:
        return []
    counts: dict[str, int] = {}
    capped: list[VoiceExemplar] = []
    for ex in exemplars:
        n = counts.get(ex.speaker_id, 0)
        if n < max_per_speaker:
            capped.append(ex)
            counts[ex.speaker_id] = n + 1
    return capped


def _truncate_attr_value(value: Any, max_chars: int | None) -> str:
    """Normalizes whitespace and optionally truncates an attribute value at a word boundary."""
    text = " ".join(str(value).split())
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…"[:max_chars]
    cut = text[: max_chars - 1]
    space_idx = cut.rfind(" ")
    if space_idx > max_chars // 2:
        cut = cut[:space_idx]
    return cut.rstrip(" ,;:-") + "…"


def _render_budgeted_blocks(
    *,
    state_header: str,
    entity_lines: list[str],
    active_edge_lines: list[str],
    bg_header: str,
    bg_lines: list[str],
    ep_header: str,
    ep_lines: list[str],
    voice_header: str,
    voice_lines: list[str],
    empty_text: str,
    max_context_chars: int | None,
) -> tuple[str, int]:
    """Renders prompt sections and trims whole lines in reverse priority order to fit max_context_chars.

    Priority kept longest:
      1. Binding entities
      2. Binding active edges
      3. Voice bank exemplars
      4. Narrative episodes
      5. Off-screen background edges (dropped first)

    Returns ``(rendered_text, dropped_binding_count)``.
    """

    def _assemble() -> str:
        blocks: list[str] = []
        if entity_lines or active_edge_lines:
            blocks.append("\n".join([state_header, *entity_lines, *active_edge_lines]))
        if bg_lines:
            blocks.append("\n".join([bg_header, *bg_lines]))
        if ep_lines:
            blocks.append("\n".join([ep_header, *ep_lines]))
        if voice_lines:
            blocks.append("\n".join([voice_header, *voice_lines]))
        if not blocks:
            return empty_text
        return "\n\n".join(blocks)

    rendered = _assemble()
    if max_context_chars is None or len(rendered) <= max_context_chars:
        return rendered, 0

    dropped_binding = 0
    while len(rendered) > max_context_chars:
        if bg_lines:
            bg_lines.pop()
        elif ep_lines:
            ep_lines.pop()
        elif voice_lines:
            voice_lines.pop()
        elif active_edge_lines:
            active_edge_lines.pop()
            dropped_binding += 1
        elif entity_lines:
            entity_lines.pop()
            dropped_binding += 1
        else:
            rendered = empty_text[:max_context_chars]
            break
        rendered = _assemble()
    return rendered, dropped_binding


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


class InteractiveFictionPreset:
    """Preset for Ren'Py, RPG dialogue engines, and text adventures."""

    def __init__(
        self,
        policy: SchemaPolicy | None = None,
        *,
        include_background_edges: bool = False,
    ) -> None:
        self.policy: SchemaPolicy = policy or DEFAULT_SCHEMA_POLICY
        self.include_background_edges: bool = include_background_edges

    @staticmethod
    def _format_edge_line(edge: DirectedEdgeRecord) -> str:
        if edge.relation_type in ("stance", "facade") or (
            "register" in edge.state or "tone" in edge.state or "trust" in edge.state
        ):
            reg = edge.state.get("register", "standard")
            tone = edge.state.get("tone", "neutral")
            trust = edge.state.get("trust", 0.0)
            tag = " [facade]" if edge.relation_type == "facade" else ""
            line = f"- {edge.source_id} → {edge.target_id}{tag}: register={reg}, tone={tone}, trust={trust}"
        else:
            pairs = ", ".join(f"{k}={v}" for k, v in edge.state.items())
            detail = f": {pairs}" if pairs else ""
            line = f"- {edge.source_id} → {edge.target_id} [{edge.relation_type}]{detail}"
        if edge.reason:
            line += f" (because: {edge.reason})"
        elif edge.caused_by_episode_id:
            line += f" (caused by: {edge.caused_by_episode_id})"
        return line

    def format_prompt_context(self, ctx: StepContext) -> str:
        lines: list[str] = [f"### Active Scene Narrative Context (Step: {ctx.step})"]

        if ctx.entities:
            lines.append("\n#### Characters Present & State:")
            for e in ctx.entities:
                if e.kind == "character" or "mood" in e.state:
                    mood = e.state.get("mood", "neutral")
                    disguise = e.disguised_as or e.state.get("disguise")
                    state_str = f"mood: {mood}" + (f", disguised as: {disguise}" if disguise else "")
                    if e.disguised_as:
                        state_str += " (pierced)"
                    if e.state.get("status"):
                        state_str += f", status: {e.state['status']}"
                else:
                    non_loc = [(k, v) for k, v in e.state.items() if k != "location_id"]
                    state_str = (
                        ", ".join(f"{k}: {v}" for k, v in non_loc)
                        if non_loc
                        else e.kind
                    )
                if e.private_state:
                    priv_pairs = ", ".join(f"{k}={v}" for k, v in e.private_state.items())
                    state_str += f" [private: {priv_pairs}]"
                if (
                    (e.kind == "character" or "mood" in e.state)
                    and e.prior_state
                    and e.prior_state.get("mood")
                    and e.prior_state.get("mood") != e.state.get("mood", "neutral")
                ):
                    state_str += f" (was: {e.prior_state['mood']} at step {e.prior_step})"
                if e.reason:
                    state_str += f" [reason: {e.reason}]"
                lines.append(f"- **{e.canonical_name}** ({e.entity_id}): {state_str}")

        if ctx.active_edges:
            lines.append("\n#### Interpersonal Stances & Register:")
            for edge in ctx.active_edges:
                lines.append(self._format_edge_line(edge))

        if self.include_background_edges and ctx.background_edges:
            lines.append("\n#### Off-Screen Ties (Not Present in Scene):")
            for edge in ctx.background_edges:
                lines.append(self._format_edge_line(edge))

        if ctx.episodes:
            lines.append("\n#### Episodic Memory (Relevant Past Events):")
            scene_char_ids = {
                e.entity_id
                for e in ctx.entities
                if e.kind == "character" and not e.entity_id.startswith("masked:")
            }
            for ep in ctx.episodes:
                known_str = f" | known to: {', '.join(sorted(ep.known_by.keys()))}" if ep.known_by else ""
                ignorant = sorted(scene_char_ids - set(ep.known_by.keys())) if ep.known_by else []
                ignorant_str = f" | ignorant: {', '.join(ignorant)}" if ignorant else ""
                loc_str = f" @ {ep.location_id}" if ep.location_id else ""
                lines.append(f"- [Step {ep.step} | {ep.predicate}{loc_str}{known_str}{ignorant_str}] {ep.summary}")

        if ctx.voice_exemplars:
            lines.append("\n#### Voice Bank (Historical Dialogue Anchors):")
            for ex in ctx.voice_exemplars:
                reg = ex.metadata.get("register", "")
                tone = ex.metadata.get("tone", "")
                tag = f"[{reg}, {tone}]" if (reg or tone) else ""
                fallback_note = ex.metadata.get("fallback_note", "")
                note_str = f" {fallback_note}" if fallback_note else ""
                lines.append(f"- {ex.speaker_id} {tag}{note_str}: \"{ex.text}\"")

        return "\n".join(lines)

    def extraction_instructions(self) -> str:
        return (
            "Analyze the scene chunk and emit a StepDelta JSON diff following these 6-Tier rules:\n"
            "1. **Entities (`entities`) — State Keys & Delta Discipline**:\n"
            "   - Emit an `EntityDelta` ONLY for characters whose `location_id`, `mood`, or `add_aliases` changed in this chunk.\n"
            "   - **Room Moves (`state['location_id']`)**: When a character exits, enters, or flees into a different room/location, set `state['location_id']` to the exact canonical `loc_*` ID where they are at the END of the chunk. If multiple characters move or chase each other into a room, update `location_id` for ALL of them.\n"
            "   - **Mood Shifts (`state['mood']`)**: When a character's emotion changes, set `state['mood']` to a single lowercase canonical word (prefer: `'furious'`, `'panicked'`, `'restless'`, `'determined'`, `'shocked'`, `'indignant'`, `'calm'`, `'confident'`, `'watchful'`, `'nervous'`). Set `caused_by_episode_id` to the triggering episode's ID.\n"
            "   - **Disguise Unmasking (`add_aliases`)**: Do NOT re-add existing aliases unless a disguise is unmasked. When a disguised character reveals their true identity to someone, emit `add_aliases=[AliasRecord(alias='<disguise_alias>', entity_id='<char_id>', kind='disguise', persona_name='<persona>', known_to=[...])]` containing BOTH previously aware character IDs and the newly informed character ID.\n"
            "2. **Edges (`edges`) — Stance & Artifact Possession**:\n"
            "   - **Item Handoff (`relation_type='possesses'`)**: When an artifact/item is handed to another character, emit `EdgeDelta(source_id='<recipient_char_id>', target_id='<artifact_id>', relation_type='possesses', state={'status': 'concealed'}, caused_by_episode_id='<ep_id>')`.\n"
            "   - **Stance & Register Shift (`relation_type='stance'`)**: When a character's interpersonal attitude or grammatical register shifts (e.g., dropping formal `'Vi'` to informal `'ti'` during an accusation, unmasking, or reprimand), emit `EdgeDelta(source_id='<speaker_id>', target_id='<target_id>', relation_type='stance', state={'register': 'ti', 'tone': '<furious|commanding|stern>', 'trust': <negative_float>}, caused_by_episode_id='<ep_id>')`.\n"
            "3. **Episodes (`episodes`) — Single Primary Event & Strict Witness Scoping**:\n"
            "   - Emit at most **1 primary `EpisodeRecord`** for the main plot event in the chunk (using `predicate` such as `'briefing'`, `'confrontation'`, `'item_transfer'`, `'flashback'`, or `'revelation'`). If a chunk only contains a truth reveal / memory wipe referencing earlier events, emit 0 new episodes and use `inform` (`InformDelta`) instead.\n"
            "   - **CRITICAL Witness Rule**: Every character in `participants` or `witnesses` is granted direct epistemic knowledge of the event. NEVER include a character in `participants` or `witnesses` if they already stepped out of the room (e.g., onto a balcony) before the event started, or if the event occurs in another room (e.g., inside a vault). Conversely, ALWAYS include all characters who are physically inside that room during the event in `witnesses`—even if they remain silent in the shadows or in disguise.\n"
            "4. **Flashbacks (`is_flashback=True`)**:\n"
            "   - If the chunk depicts a memory/flashback to the past (e.g., ten years ago), set `is_flashback=True` and `story_time='10_years_ago'` on the `StepDelta` and record a single `'flashback'` episode (`witnesses` = only the characters involved in that past memory).\n"
            "5. **Epistemic Transitions on Prior Episodes (`inform` / `InformDelta`)**:\n"
            "   - Do NOT emit `InformDelta` with `epistemic='direct'` (direct witnesses belong in `episode.witnesses`).\n"
            "   - **`epistemic='deceived'`**: When a character lies to someone (`entity_id`) with a cover story to explain away an **earlier confrontation/theft episode** from `### Recent Episode IDs`, do NOT create a new episode for the lie; instead emit `InformDelta(episode_id='<earlier_confrontation_episode_id>', entity_id='<listener_id>', epistemic='deceived', believed_summary='<the false cover story>')`.\n"
            "   - **`epistemic='told'`**: When someone informs a character (`entity_id`) of the truth about an **earlier confrontation/theft episode** from `### Recent Episode IDs`, emit `InformDelta(episode_id='<earlier_confrontation_episode_id>', entity_id='<listener_id>', epistemic='told')`.\n"
            "   - **`epistemic='forgotten'`**: When a character (`entity_id`) drinks a draught or triggers a device that wipes their memory of an **earlier flashback episode** from `### Recent Episode IDs`, emit `InformDelta(episode_id='<earlier_flashback_episode_id>', entity_id='<target_id>', epistemic='forgotten')`."
        )


class TranslationPreset:
    """Preset for literary and game localization (Tier 1 base declension/gender + Tier 2 register)."""

    def __init__(
        self,
        policy: SchemaPolicy | None = None,
        *,
        include_background_edges: bool = False,
        show_entity_ids: bool = False,
        include_kind: bool | Literal["non_character"] = True,
        extra_attributes: Sequence[str] = ("notes",),
        max_attr_chars: int | None = 160,
        empty_text: str = "",
        max_voice_per_speaker: int | None = None,
        max_context_chars: int | None = None,
    ) -> None:
        if include_kind not in (True, False, "non_character"):
            raise ValueError(
                f"Invalid include_kind: {include_kind!r}. Expected True, False, or 'non_character'."
            )
        if max_attr_chars is not None and max_attr_chars <= 0:
            raise ValueError("max_attr_chars must be a positive integer or None.")
        if max_voice_per_speaker is not None and max_voice_per_speaker < 0:
            raise ValueError("max_voice_per_speaker must be non-negative or None.")
        if max_context_chars is not None and max_context_chars < 0:
            raise ValueError("max_context_chars must be non-negative or None.")

        self.policy: SchemaPolicy = policy or DEFAULT_SCHEMA_POLICY
        self.include_background_edges: bool = include_background_edges
        self.show_entity_ids: bool = show_entity_ids
        self.include_kind: bool | Literal["non_character"] = include_kind
        self.extra_attributes: tuple[str, ...] = tuple(extra_attributes)
        self.max_attr_chars: int | None = max_attr_chars
        self.empty_text: str = empty_text
        self.max_voice_per_speaker: int | None = max_voice_per_speaker
        self.max_context_chars: int | None = max_context_chars

    def _should_show_kind(self, kind: str) -> bool:
        if not kind:
            return False
        if self.include_kind is True:
            return True
        if self.include_kind == "non_character":
            return kind != "character"
        return False

    @staticmethod
    def _build_display_labels(ctx: StepContext) -> dict[str, str]:
        """Builds human-readable labels for entity_ids, auto-disambiguating colliding names."""
        name_by_id: dict[str, str] = dict(ctx.entity_names)
        kind_by_id: dict[str, str] = {}
        for e in (*ctx.entities, *ctx.related_entities):
            name_by_id[e.entity_id] = e.canonical_name
            if e.kind:
                kind_by_id[e.entity_id] = e.kind

        name_counts = Counter(name.casefold() for name in name_by_id.values())
        name_kind_counts = Counter(
            (name.casefold(), kind_by_id.get(eid, ""))
            for eid, name in name_by_id.items()
        )

        labels: dict[str, str] = {}
        for eid, name in name_by_id.items():
            key = name.casefold()
            if name_counts[key] <= 1:
                labels[eid] = name
            else:
                kind = kind_by_id.get(eid, "")
                if kind and name_kind_counts[(key, kind)] == 1:
                    labels[eid] = f"{name} [{kind}]"
                else:
                    labels[eid] = f"{name} ({eid})"
        return labels

    @staticmethod
    def _format_edge_line(
        edge: DirectedEdgeRecord,
        *,
        id_labels: dict[str, str] | None = None,
    ) -> str:
        src = id_labels.get(edge.source_id, edge.source_id) if id_labels is not None else edge.source_id
        dst = id_labels.get(edge.target_id, edge.target_id) if id_labels is not None else edge.target_id
        if edge.relation_type in ("stance", "facade") or ("register" in edge.state or "tone" in edge.state):
            parts: list[str] = []
            if edge.state.get("register"):
                parts.append(f"register={edge.state['register']}")
            if edge.state.get("tone"):
                parts.append(f"tone={edge.state['tone']}")
            for k, v in edge.state.items():
                if k not in ("register", "tone") and v not in (None, ""):
                    parts.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")
            tag = " [facade]" if edge.relation_type == "facade" else ""
            detail = ", ".join(parts) if parts else "active"
            return f"- {src} → {dst}{tag}: {detail}"
        pairs = ", ".join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in edge.state.items())
        return f"- {src} → {dst} [{edge.relation_type}]: {pairs}"

    def _format_entity_line(
        self,
        e: EntityRecord,
        *,
        is_lexicon_only: bool,
        id_labels: dict[str, str] | None,
    ) -> str:
        if self.show_entity_ids:
            kind_tag = f" [{e.kind}]" if self._should_show_kind(e.kind) else ""
            subject = f"{e.canonical_name} ({e.entity_id}){kind_tag}"
        else:
            base_label = id_labels.get(e.entity_id, e.canonical_name) if id_labels is not None else e.canonical_name
            if self._should_show_kind(e.kind) and f"[{e.kind}]" not in base_label:
                subject = f"{base_label} [{e.kind}]"
            else:
                subject = base_label

        gender = e.base_attributes.get("gender", "")
        decl = e.base_attributes.get("declension", "")
        target_name = e.base_attributes.get("target_name", e.canonical_name)
        attr_str = (
            f"target='{target_name}'"
            + (f", {gender}" if gender else "")
            + (f", {decl}" if decl else "")
        )
        mood = "" if is_lexicon_only else e.state.get("mood", "")
        mood_str = f" [mood: {mood}]" if mood else ""

        extra_parts: list[str] = []
        for attr_key in self.extra_attributes:
            if attr_key in ("target_name", "gender", "declension"):
                continue
            raw_val = e.base_attributes.get(attr_key)
            if raw_val is None or raw_val == "" or raw_val == [] or raw_val == {}:
                continue
            rendered_val = _truncate_attr_value(raw_val, self.max_attr_chars)
            if rendered_val:
                extra_parts.append(f"{attr_key}: {rendered_val}")
        extra_str = f"; {'; '.join(extra_parts)}" if extra_parts else ""
        return f"- {subject}: {attr_str}{mood_str}{extra_str}"

    def format_prompt_context(self, ctx: StepContext) -> str:
        id_labels: dict[str, str] | None = (
            None if self.show_entity_ids else self._build_display_labels(ctx)
        )

        # Tier 1 & 2 binding state block (or Tier 1 lexicon block in LEXICON_ONLY)
        is_lexicon_only = not self.policy.has_tier(2) and not self.policy.has_tier(3)
        state_header = (
            "## Terminology & glossary — binding"
            if is_lexicon_only
            else "## Scene character state & register — binding"
        )
        entity_lines = [
            self._format_entity_line(e, is_lexicon_only=is_lexicon_only, id_labels=id_labels)
            for e in ctx.entities
        ]
        active_edge_lines = (
            []
            if is_lexicon_only
            else [self._format_edge_line(edge, id_labels=id_labels) for edge in ctx.active_edges]
        )

        bg_header = "## Off-screen character ties — reference only"
        bg_lines = (
            [self._format_edge_line(edge, id_labels=id_labels) for edge in ctx.background_edges]
            if self.include_background_edges and ctx.background_edges
            else []
        )

        # Tier 6 Episodic context block (when present)
        ep_header = "## Relevant narrative episodes — context"
        ep_lines: list[str] = []
        if ctx.episodes:
            scene_char_ids = {
                e.entity_id
                for e in ctx.entities
                if e.kind == "character" and not e.entity_id.startswith("masked:")
            }
            for ep in ctx.episodes:
                if ep.known_by:
                    known_ids = sorted(ep.known_by.keys())
                    known_labels = (
                        known_ids
                        if id_labels is None
                        else [id_labels.get(eid, eid) for eid in known_ids]
                    )
                    known_str = f" | known to: {', '.join(known_labels)}"
                    ignorant_ids = sorted(scene_char_ids - set(ep.known_by.keys()))
                    ignorant_labels = (
                        ignorant_ids
                        if id_labels is None
                        else [id_labels.get(eid, eid) for eid in ignorant_ids]
                    )
                    ignorant_str = f" | ignorant: {', '.join(ignorant_labels)}" if ignorant_labels else ""
                else:
                    known_str = ""
                    ignorant_str = ""
                ep_lines.append(f"- [Step {ep.step}{known_str}{ignorant_str}] {ep.summary}")

        # Tier 5 Voice Bank anchor block
        voice_header = "## Character voice bank — stylistic anchor"
        voice_lines: list[str] = []
        capped_exemplars = _cap_voice_exemplars_per_speaker(
            ctx.voice_exemplars, self.max_voice_per_speaker
        )
        for ex in capped_exemplars:
            speaker = ex.speaker_id if id_labels is None else id_labels.get(ex.speaker_id, ex.speaker_id)
            reg = ex.metadata.get("register", "")
            reg_str = f" ({reg})" if reg else ""
            fallback_note = ex.metadata.get("fallback_note", "")
            note_str = f" {fallback_note}" if fallback_note else ""
            voice_lines.append(f'- {speaker}{reg_str}{note_str}: "{ex.text}"')

        rendered, dropped_binding = _render_budgeted_blocks(
            state_header=state_header,
            entity_lines=entity_lines,
            active_edge_lines=active_edge_lines,
            bg_header=bg_header,
            bg_lines=bg_lines,
            ep_header=ep_header,
            ep_lines=ep_lines,
            voice_header=voice_header,
            voice_lines=voice_lines,
            empty_text=self.empty_text,
            max_context_chars=self.max_context_chars,
        )
        if dropped_binding > 0:
            logger.warning(
                "TranslationPreset truncated %d binding entity/edge line(s) to fit max_context_chars=%d",
                dropped_binding,
                self.max_context_chars,
            )
        return rendered

    def extraction_instructions(self) -> str:
        return (
            "Analyze the literary prose chunk and emit a strict StepDelta JSON diff for localization consistency:\n"
            "1. **Entities (`entities`) — Reuse Existing IDs & Delta Discipline**:\n"
            "   - ALWAYS reuse existing canonical `entity_id`s from the known entities/aliases list. "
            "NEVER invent a new entity for an epithet, title, or disguise that already aliases an existing character "
            "(for example, if 'hooded courier' aliases `char_lyra`, emit updates on `char_lyra`, NOT a new `char_courier`).\n"
            "   - Emit an `EntityDelta` ONLY when `mood`, `location_id`, or `add_aliases` changed in this chunk.\n"
            "   - Set `state['mood']` to a concise lowercase word or short phrase (e.g., `'calm'`, `'furious'`, `'panicked'`, `'suspicious'`, `'grief'`).\n"
            "   - When a disguised character unmasks to someone, emit `add_aliases` with `kind='disguise'` and updated `known_to`.\n"
            "2. **Edges (`edges`) — Binding T–V Register & Tone on `relation_type='stance'`**:\n"
            "   - Whenever a speaker's grammatical address register (e.g., formal `'Vi'` vs. informal `'ti'`) or interpersonal tone shifts "
            "toward an addressee (such as an angry accusation, unmasking, or furious reprimand), emit "
            "`EdgeDelta(source_id='<speaker_id>', target_id='<addressee_id>', relation_type='stance', "
            "state={'register': '<Vi|ti|formal|informal>', 'tone': '<concise_tone>'}, caused_by_episode_id='<ep_id>')`.\n"
            "   - ALWAYS use `relation_type='stance'` and ALWAYS include `'register'` and `'tone'` keys for interpersonal register shifts "
            "(do not hide register shifts under ad-hoc relation types like `'accusation'` or omit `'register'`).\n"
            "3. **Episodes (`episodes`) — Concise Plot Beats & Strict Witness Scoping**:\n"
            "   - Emit 1–2 primary `EpisodeRecord`s for the major plot beats in the chunk.\n"
            "   - Only include characters physically present in the room during the beat in `participants`/`witnesses` "
            "(exclude anyone who already stepped outside, e.g., onto a balcony; include silent/disguised observers inside the room using their canonical `entity_id`).\n"
            "4. **Flashbacks (`is_flashback=True`)**:\n"
            "   - If the chunk depicts a memory or flashback to an earlier time (e.g., ten years ago), set `is_flashback=True` "
            "and `story_time` on the `StepDelta` so present-day character moods and T–V registers are preserved.\n"
            "5. **Epistemic Transitions (`inform` / `InformDelta`)**:\n"
            "   - Do NOT emit `InformDelta` with `epistemic='direct'` (direct witnesses belong in `episode.witnesses`).\n"
            "   - Use `epistemic='deceived'` (with `believed_summary`), `'told'`, or `'forgotten'` when characters lie about, reveal, or lose memory of an earlier episode."
        )


class SimulationPreset:
    """Preset for multi-agent social, financial, or domain-agnostic simulations."""

    def __init__(
        self,
        policy: SchemaPolicy | None = None,
        *,
        include_background_edges: bool = False,
    ) -> None:
        self.policy: SchemaPolicy = policy or SchemaPolicy(
            hard_filter_keys=(),
            soft_boost_weights={"stance": 0.15, "tone": 0.10},
        )
        self.include_background_edges: bool = include_background_edges

    def format_prompt_context(self, ctx: StepContext) -> str:
        lines: list[str] = [f"=== Agent Simulation State (Tick {ctx.step}) ==="]
        for e in ctx.entities:
            priv = f", private_state={e.private_state}" if e.private_state else ""
            lines.append(f"Agent {e.entity_id} ({e.canonical_name}): state={e.state}{priv}")
        for edge in ctx.active_edges:
            lines.append(f"Relation [{edge.relation_type}] {edge.source_id} -> {edge.target_id}: {edge.state}")
        if self.include_background_edges and ctx.background_edges:
            lines.append("Off-Screen Relations:")
            for edge in ctx.background_edges:
                lines.append(
                    f"  Relation [{edge.relation_type}] {edge.source_id} -> {edge.target_id}: {edge.state}"
                )
        if ctx.episodes:
            lines.append("Episodic Memory:")
            for ep in ctx.episodes:
                lines.append(f"  [Tick {ep.step} | {ep.predicate}] {ep.summary} (known_by={list(ep.known_by.keys())})")
        if ctx.voice_exemplars:
            lines.append("Past Utterance Anchors:")
            for ex in ctx.voice_exemplars:
                fallback_note = ex.metadata.get("fallback_note", "")
                note_str = f" {fallback_note}" if fallback_note else ""
                lines.append(f"  {ex.speaker_id}{note_str}: {ex.text}")
        return "\n".join(lines)

    def extraction_instructions(self) -> str:
        return (
            "Analyze the simulation tick log and emit a strict StepDelta JSON diff following these 6-Tier rules:\n"
            "1. **Entities (`entities`) — Observable Agent State Only**:\n"
            "   - ALWAYS reuse existing canonical `entity_id`s from the known entities/aliases list (including callsigns/handles).\n"
            "   - Emit an `EntityDelta` ONLY for agents whose observable internal state changed in this tick "
            "(e.g., `energy`, `morale`, `location_id`, resources).\n"
            "   - NEVER store interpersonal stances (such as `stance_toward_*` or `affinity_to_*`) inside an entity's `state` dict; "
            "model all interpersonal attitudes strictly on directed `edges`.\n"
            "2. **Edges (`edges`) — Canonical Directed `stance` & `facade` Relations**:\n"
            "   - ALWAYS emit interpersonal affinity or stance changes using `relation_type='stance'` "
            "(NEVER invent ad-hoc relation types like `'affinity'` or `'trust'`): "
            "`EdgeDelta(source_id='<agent_a>', target_id='<agent_b>', relation_type='stance', "
            "state={'affinity': <float -1.0..1.0>, 'stance': '<cooperative|neutral|cautious|guarded|suspicious|hostile|allied>'}, "
            "caused_by_episode_id='<ep_id>')`.\n"
            "   - Use `relation_type='facade'` ONLY when an agent deliberately displays a deceptive public posture that differs from their true `stance`.\n"
            "3. **Episodes (`episodes`) — Single Primary Event & Strict Witness Scoping**:\n"
            "   - Emit at most **1 primary `EpisodeRecord`** for the main event in the tick.\n"
            "   - **CRITICAL Witness Rule**: Every agent in `participants`, `witnesses`, or `known_by` is granted knowledge of the event. "
            "NEVER include sleeping, absent, or unaware agents in `participants`, `witnesses`, or `known_by`, "
            "and NEVER mark an absent/sleeping agent as `epistemic='deceived'` with a `believed_summaries` entry. "
            "Include ONLY agents who directly acted in or physically observed the event.\n"
            "4. **Epistemic Transitions (`inform` / `InformDelta`)**:\n"
            "   - Do NOT emit `InformDelta` with `epistemic='direct'` (direct observers belong in `episode.witnesses`).\n"
            "   - Emit `InformDelta(episode_id='<earlier_ep_id>', entity_id='<listener_id>', epistemic='told')` when an agent is told secondhand about an earlier event.\n"
            "   - Emit `InformDelta(..., epistemic='deceived', believed_summary='<cover story>')` ONLY when someone actively feeds a false cover story to an agent about an earlier event.\n"
            "   - Emit `InformDelta(..., epistemic='forgotten')` or `'suppressed'` when an agent loses or suppresses memory of an earlier event."
        )
