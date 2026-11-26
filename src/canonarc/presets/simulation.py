from __future__ import annotations

from ..models import StepContext
from .policy import SchemaPolicy


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
