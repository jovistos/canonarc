from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import StepContext, StepDelta


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


DEFAULT_SCHEMA_POLICY = SchemaPolicy()


class DomainPreset(Protocol):
    """Interface for rendering StepContext into prompt blocks and formatting extraction instructions."""

    policy: SchemaPolicy

    def format_prompt_context(self, ctx: StepContext) -> str: ...
    def extraction_instructions(self) -> str: ...


class InteractiveFictionPreset:
    """Preset for Ren'Py, RPG dialogue engines, and text adventures."""

    def __init__(self, policy: SchemaPolicy | None = None) -> None:
        self.policy: SchemaPolicy = policy or DEFAULT_SCHEMA_POLICY

    def format_prompt_context(self, ctx: StepContext) -> str:
        lines: list[str] = ["### Active Scene Narrative Context (Step: {})".format(ctx.step)]

        if ctx.entities:
            lines.append("\n#### Characters Present & State:")
            for e in ctx.entities:
                mood = e.state.get("mood", "neutral")
                disguise = e.disguised_as or e.state.get("disguise")
                state_str = f"mood: {mood}" + (f", disguised as: {disguise}" if disguise else "")
                if e.disguised_as:
                    state_str += " (pierced)"
                if e.private_state:
                    priv_pairs = ", ".join(f"{k}={v}" for k, v in e.private_state.items())
                    state_str += f" [private: {priv_pairs}]"
                if e.prior_state and e.prior_state.get("mood") and e.prior_state.get("mood") != mood:
                    state_str += f" (was: {e.prior_state['mood']} at step {e.prior_step})"
                if e.reason:
                    state_str += f" [reason: {e.reason}]"
                lines.append(f"- **{e.canonical_name}** ({e.entity_id}): {state_str}")

        if ctx.active_edges:
            lines.append("\n#### Interpersonal Stances & Register:")
            for edge in ctx.active_edges:
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
                lines.append(line)

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
                lines.append(f"- {ex.speaker_id} {tag}: \"{ex.text}\"")

        return "\n".join(lines)

    def extraction_instructions(self) -> str:
        return (
            "Analyze the scene outcome and emit a StepDelta JSON diff recording any changes in "
            "character mood, disguises, relationship shifts, new aliases, or discrete plot episodes."
        )


class TranslationPreset:
    """Preset for literary and game localization (Tier 1 base declension/gender + Tier 2 register)."""

    def __init__(self, policy: SchemaPolicy | None = None) -> None:
        self.policy: SchemaPolicy = policy or DEFAULT_SCHEMA_POLICY

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
            if edge.relation_type in ("stance", "facade") or ("register" in edge.state or "tone" in edge.state):
                reg = edge.state.get("register", "")
                tone = edge.state.get("tone", "")
                tag = " [facade]" if edge.relation_type == "facade" else ""
                state_lines.append(f"- {edge.source_id} → {edge.target_id}{tag}: register={reg}, tone={tone}")
            else:
                pairs = ", ".join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in edge.state.items())
                state_lines.append(f"- {edge.source_id} → {edge.target_id} [{edge.relation_type}]: {pairs}")

        blocks.append("\n".join(state_lines))

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
                voice_lines.append(f'- {ex.speaker_id} ({reg}): "{ex.text}"')
            blocks.append("\n".join(voice_lines))

        return "\n\n".join(blocks)

    def extraction_instructions(self) -> str:
        return (
            "Extract any shifts in character mood, grammatical register (formal/informal), "
            "newly revealed character aliases, or key narrative episodes produced in this translated chunk."
        )


class SimulationPreset:
    """Preset for multi-agent social, financial, or domain-agnostic simulations."""

    def __init__(self, policy: SchemaPolicy | None = None) -> None:
        self.policy: SchemaPolicy = policy or SchemaPolicy(
            hard_filter_keys=(),
            soft_boost_weights={"stance": 0.15, "tone": 0.10},
        )

    def format_prompt_context(self, ctx: StepContext) -> str:
        lines: list[str] = [f"=== Agent Simulation State (Tick {ctx.step}) ==="]
        for e in ctx.entities:
            priv = f", private_state={e.private_state}" if e.private_state else ""
            lines.append(f"Agent {e.entity_id} ({e.canonical_name}): state={e.state}{priv}")
        for edge in ctx.active_edges:
            lines.append(f"Relation [{edge.relation_type}] {edge.source_id} -> {edge.target_id}: {edge.state}")
        if ctx.episodes:
            lines.append("Episodic Memory:")
            for ep in ctx.episodes:
                lines.append(f"  [Tick {ep.step} | {ep.predicate}] {ep.summary} (known_by={list(ep.known_by.keys())})")
        if ctx.voice_exemplars:
            lines.append("Past Utterance Anchors:")
            for ex in ctx.voice_exemplars:
                lines.append(f"  {ex.speaker_id}: {ex.text}")
        return "\n".join(lines)

    def extraction_instructions(self) -> str:
        return "Return state deltas for agent perceptions, emotions, interpersonal affinity changes, and notable episodes."
