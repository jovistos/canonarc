from __future__ import annotations

from typing import Protocol

from .models import StepContext, StepDelta


class DomainPreset(Protocol):
    """Interface for rendering StepContext into prompt blocks and formatting extraction instructions."""

    def format_prompt_context(self, ctx: StepContext) -> str: ...
    def extraction_instructions(self) -> str: ...


class InteractiveFictionPreset:
    """Preset for Ren'Py, RPG dialogue engines, and text adventures."""

    def format_prompt_context(self, ctx: StepContext) -> str:
        lines: list[str] = ["### Active Scene Narrative Context (Step: {})".format(ctx.step)]

        if ctx.entities:
            lines.append("\n#### Characters Present & State:")
            for e in ctx.entities:
                mood = e.state.get("mood", "neutral")
                disguise = e.state.get("disguise")
                state_str = f"mood: {mood}" + (f", disguised as: {disguise}" if disguise else "")
                lines.append(f"- **{e.canonical_name}** ({e.entity_id}): {state_str}")

        if ctx.active_edges:
            lines.append("\n#### Interpersonal Stances & Register:")
            for edge in ctx.active_edges:
                reg = edge.state.get("register", "standard")
                tone = edge.state.get("tone", "neutral")
                trust = edge.state.get("trust", 0.0)
                lines.append(f"- {edge.source_id} → {edge.target_id}: register={reg}, tone={tone}, trust={trust}")

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
            "character mood, disguises, relationship shifts, or new aliases."
        )


class TranslationPreset:
    """Preset for literary and game localization (Tier 1 base declension/gender + Tier 2 register)."""

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
            reg = edge.state.get("register", "")
            tone = edge.state.get("tone", "")
            state_lines.append(f"- {edge.source_id} → {edge.target_id}: register={reg}, tone={tone}")

        blocks.append("\n".join(state_lines))

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
            "or newly revealed character aliases produced in this translated chunk."
        )


class SimulationPreset:
    """Preset for multi-agent social simulations."""

    def format_prompt_context(self, ctx: StepContext) -> str:
        lines: list[str] = [f"=== Agent Simulation State (Tick {ctx.step}) ==="]
        for e in ctx.entities:
            lines.append(f"Agent {e.entity_id} ({e.canonical_name}): state={e.state}")
        for edge in ctx.active_edges:
            lines.append(f"Relation {edge.source_id} -> {edge.target_id}: {edge.state}")
        if ctx.voice_exemplars:
            lines.append("Past Utterance Anchors:")
            for ex in ctx.voice_exemplars:
                lines.append(f"  {ex.speaker_id}: {ex.text}")
        return "\n".join(lines)

    def extraction_instructions(self) -> str:
        return "Return state deltas for agent perceptions, emotions, and interpersonal affinity changes."
