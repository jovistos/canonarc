from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..models import VoiceExemplar


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
    glossary_header: str = "",
    glossary_sections: list[tuple[str | None, list[str]]] | None = None,
) -> tuple[str, int]:
    """Renders prompt sections and trims whole lines in reverse priority order to fit max_context_chars.

    Priority kept longest:
      1. Binding character/state entities
      2. Binding glossary/grouped entities (empty subheadings pruned automatically)
      3. Binding active edges
      4. Voice bank exemplars
      5. Narrative episodes
      6. Off-screen background edges (dropped first)

    Returns ``(rendered_text, dropped_binding_count)``.
    """
    sections = glossary_sections or []

    def _assemble() -> str:
        blocks: list[str] = []
        if entity_lines or active_edge_lines:
            blocks.append("\n".join([state_header, *entity_lines, *active_edge_lines]))
        non_empty_glossary = [(sub, lines) for sub, lines in sections if lines]
        if non_empty_glossary:
            if all(sub is None for sub, _ in non_empty_glossary):
                flat_lines = [line for _, lines in non_empty_glossary for line in lines]
                blocks.append("\n".join([glossary_header, *flat_lines]))
            else:
                parts: list[str] = [glossary_header] if glossary_header else []
                for sub, lines in non_empty_glossary:
                    if sub:
                        parts.append("\n".join([sub, *lines]))
                    else:
                        parts.append("\n".join(lines))
                blocks.append("\n\n".join(parts))
        if bg_lines:
            blocks.append("\n".join([bg_header, *bg_lines]))
        if ep_lines:
            blocks.append("\n".join([ep_header, *ep_lines]))
        if voice_lines:
            blocks.append("\n".join([voice_header, *voice_lines]))
        if not blocks:
            return empty_text
        return "\n\n".join(blocks)

    def _pop_glossary_line() -> bool:
        for _, lines in reversed(sections):
            if lines:
                lines.pop()
                return True
        return False

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
        elif _pop_glossary_line():
            dropped_binding += 1
        elif entity_lines:
            entity_lines.pop()
            dropped_binding += 1
        else:
            rendered = empty_text[:max_context_chars]
            break
        rendered = _assemble()
    return rendered, dropped_binding
