from __future__ import annotations

from ..models import DirectedEdgeRecord, StepContext
from .policy import DEFAULT_SCHEMA_POLICY, SchemaPolicy


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
