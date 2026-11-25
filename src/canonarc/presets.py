from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .episodic import EpisodicScoringPolicy
from .models import DirectedEdgeRecord, StepContext


@dataclass(frozen=True)
class SchemaPolicy:
    """Declarative domain policy controlling edge layering, disguise keys, and voice ranking.

    Allows any domain (interactive fiction, localization, financial/social simulations,
    healthcare, etc.) to customize which state/metadata keys act as hard filters vs.
    soft boosts without modifying the core engine or VoiceBank.
    """

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
    ) -> None:
        self.policy: SchemaPolicy = policy or DEFAULT_SCHEMA_POLICY
        self.include_background_edges: bool = include_background_edges

    @staticmethod
    def _format_edge_line(edge: DirectedEdgeRecord) -> str:
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
            return f"- {edge.source_id} → {edge.target_id}{tag}: {detail}"
        pairs = ", ".join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in edge.state.items())
        return f"- {edge.source_id} → {edge.target_id} [{edge.relation_type}]: {pairs}"

    def format_prompt_context(self, ctx: StepContext) -> str:
        blocks: list[str] = []

        # Tier 1 & 2 binding state block
        state_lines: list[str] = ["## Scene character state & register — binding"]
        for e in ctx.entities:
            gender = e.base_attributes.get("gender", "")
            decl = e.base_attributes.get("declension", "")
            target_name = e.base_attributes.get("target_name", e.canonical_name)
            mood = e.state.get("mood", "")
            attr_str = f"target='{target_name}'" + (f", {gender}" if gender else "") + (f", {decl}" if decl else "")
            mood_str = f" [mood: {mood}]" if mood else ""
            state_lines.append(f"- {e.canonical_name} ({e.entity_id}): {attr_str}{mood_str}")

        for edge in ctx.active_edges:
            state_lines.append(self._format_edge_line(edge))

        blocks.append("\n".join(state_lines))

        if self.include_background_edges and ctx.background_edges:
            bg_lines: list[str] = ["## Off-screen character ties — reference only"]
            for edge in ctx.background_edges:
                bg_lines.append(self._format_edge_line(edge))
            blocks.append("\n".join(bg_lines))

        # Tier 6 Episodic context block (when present)
        if ctx.episodes:
            ep_lines: list[str] = ["## Relevant narrative episodes — context"]
            scene_char_ids = {
                e.entity_id
                for e in ctx.entities
                if e.kind == "character" and not e.entity_id.startswith("masked:")
            }
            for ep in ctx.episodes:
                known_str = f" | known to: {', '.join(sorted(ep.known_by.keys()))}" if ep.known_by else ""
                ignorant = sorted(scene_char_ids - set(ep.known_by.keys())) if ep.known_by else []
                ignorant_str = f" | ignorant: {', '.join(ignorant)}" if ignorant else ""
                ep_lines.append(f"- [Step {ep.step}{known_str}{ignorant_str}] {ep.summary}")
            blocks.append("\n".join(ep_lines))

        # Tier 5 Voice Bank anchor block
        if ctx.voice_exemplars:
            voice_lines: list[str] = ["## Character voice bank — stylistic anchor"]
            for ex in ctx.voice_exemplars:
                reg = ex.metadata.get("register", "")
                fallback_note = ex.metadata.get("fallback_note", "")
                note_str = f" {fallback_note}" if fallback_note else ""
                voice_lines.append(f'- {ex.speaker_id} ({reg}){note_str}: "{ex.text}"')
            blocks.append("\n".join(voice_lines))

        return "\n\n".join(blocks)

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
