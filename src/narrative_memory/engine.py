from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

from rapidfuzz import fuzz, process

from .models import (
    DialogueCandidate,
    DirectedEdgeRecord,
    EntityRecord,
    ResolvedMention,
    StepContext,
    StepDelta,
    TimelineInfo,
    VoiceExemplar,
)
from .presets import DomainPreset, InteractiveFictionPreset
from .store import CausalStore
from .voice import Embedder, IngestionGate, VoiceBank


class StepTransaction:
    """Context manager for reading state at `step` and atomically committing mutations on exit."""

    def __init__(
        self,
        engine: NarrativeMemoryEngine,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        timeline_id: str | None = None,
        rerun_clean: bool = True,
    ) -> None:
        self.engine = engine
        self.step = step
        self.text = text
        self.mentions = mentions
        self.top_k_voice = top_k_voice
        self.timeline_id = timeline_id or engine.active_timeline
        self.rerun_clean = rerun_clean
        self._context: StepContext | None = None
        self._staged_delta: StepDelta | None = None
        self._staged_exemplars: list[VoiceExemplar] = []
        self._bypass_gate = False

    def __enter__(self) -> StepTransaction:
        if self.rerun_clean:
            # Clean rerun: roll back any prior data at >= step on this timeline
            self.engine.rollback_to(self.step - 1, timeline_id=self.timeline_id)

        self._context = self.engine.retrieve_context(
            self.step,
            text=self.text,
            mentions=self.mentions,
            top_k_voice=self.top_k_voice,
            timeline_id=self.timeline_id,
        )
        return self

    @property
    def context(self) -> StepContext:
        if self._context is None:
            raise RuntimeError("StepTransaction is not active.")
        return self._context

    def commit(
        self,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        *,
        bypass_gate: bool = False,
    ) -> None:
        """Stage mutations to be committed when the transaction context exits successfully."""
        self._staged_delta = delta
        self._staged_exemplars.extend(exemplars)
        self._bypass_gate = bypass_gate

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            if self._staged_delta or self._staged_exemplars:
                self.engine.commit_step(
                    self.step,
                    delta=self._staged_delta,
                    exemplars=self._staged_exemplars,
                    bypass_gate=self._bypass_gate,
                    timeline_id=self.timeline_id,
                )


class NarrativeMemoryEngine:
    """High-level facade for Tier 1-5 narrative memory, time travel, and voice retrieval."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        timeline_id: str = "main",
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
        preset: DomainPreset | None = None,
    ) -> None:
        self.store = CausalStore(db_path)
        self.active_timeline = timeline_id
        self.voice_bank = VoiceBank(embedder=embedder, gate=gate)
        self.preset = preset or InteractiveFictionPreset()

    def close(self) -> None:
        self.store.close()

    # --- Tier 1 Seeding ---

    def register_entity(
        self,
        entity: EntityRecord,
        *,
        aliases: Sequence[str] = (),
        step: int = 0,
        timeline_id: str | None = None,
    ) -> None:
        tl = timeline_id or self.active_timeline
        self.store.register_entity(entity, aliases=aliases, step=step, timeline_id=tl)

    def seed_entities(
        self,
        entities: Sequence[EntityRecord],
        *,
        step: int = 0,
        timeline_id: str | None = None,
    ) -> None:
        for ent in entities:
            self.register_entity(ent, step=step, timeline_id=timeline_id)

    # --- Unified Tier 1-5 Retrieval ---

    def retrieve_context(
        self,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        inclusive: bool = False,
        timeline_id: str | None = None,
    ) -> StepContext:
        tl = timeline_id or self.active_timeline
        # Causal cutoff: step < current_step unless inclusive is True
        cutoff_step = step if inclusive else (step - 1)

        # 1. Fetch active entities, aliases, and edges from store
        active_entities = self.store.get_all_active_entities(tl, cutoff_step)
        active_aliases = self.store.get_all_active_aliases(tl, cutoff_step)
        active_edges = self.store.get_all_active_edges(tl, cutoff_step)

        # 2. Entity resolution (canonical names + 1-hop aliases)
        resolved_mentions = self._resolve_mentions(
            text=text,
            explicit_mentions=mentions,
            entities=active_entities,
            aliases=active_aliases,
        )

        resolved_entity_ids = {m.entity_id for m in resolved_mentions}

        # 3. Filter relevant entities: present entities + 1-hop edge neighbors
        relevant_entity_ids = set(resolved_entity_ids)
        scene_edges: list[DirectedEdgeRecord] = []
        for edge in active_edges:
            if edge.source_id in resolved_entity_ids or edge.target_id in resolved_entity_ids:
                scene_edges.append(edge)
                relevant_entity_ids.add(edge.source_id)
                relevant_entity_ids.add(edge.target_id)

        scene_entities = [e for e in active_entities if e.entity_id in relevant_entity_ids]

        # If no explicit mentions were found, return all active entities/edges in scene
        if not scene_entities:
            scene_entities = active_entities
            scene_edges = active_edges

        # 4. Tier 5 Voice Exemplars retrieval
        voice_exemplars: list[VoiceExemplar] = []
        if top_k_voice > 0 and scene_entities:
            # Query voice exemplars for resolved speakers
            for speaker in scene_entities:
                # Find matching edge for register/tone if any
                matching_edges = [edge for edge in scene_edges if edge.source_id == speaker.entity_id]
                reg = matching_edges[0].state.get("register", "") if matching_edges else ""
                tone = matching_edges[0].state.get("tone", "") if matching_edges else ""
                addr = matching_edges[0].target_id if matching_edges else None

                candidates = self.store.get_lineage_voice_exemplars(tl, cutoff_step + 1, speaker.entity_id)
                ranked = self.voice_bank.rank_exemplars(
                    query_text=text,
                    candidates=candidates,
                    target_register=reg,
                    target_tone=tone,
                    target_addressee=addr,
                    top_k=top_k_voice,
                )
                voice_exemplars.extend(ranked)

        return StepContext(
            timeline_id=tl,
            step=step,
            resolved_mentions=resolved_mentions,
            entities=scene_entities,
            active_edges=scene_edges,
            voice_exemplars=voice_exemplars,
        )

    def _resolve_mentions(
        self,
        *,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[Any],
    ) -> list[ResolvedMention]:
        resolved: list[ResolvedMention] = []
        seen_entities: set[str] = set()

        name_to_id = {e.canonical_name.lower(): e.entity_id for e in entities}
        alias_to_id = {a.alias.lower(): a.entity_id for a in aliases}

        # Check explicit mentions
        for m in explicit_mentions:
            m_clean = m.strip().lower()
            if m_clean in name_to_id:
                ent_id = name_to_id[m_clean]
                if ent_id not in seen_entities:
                    seen_entities.add(ent_id)
                    resolved.append(ResolvedMention(surface_text=m, entity_id=ent_id, matched_via="canonical"))
            elif m_clean in alias_to_id:
                ent_id = alias_to_id[m_clean]
                if ent_id not in seen_entities:
                    seen_entities.add(ent_id)
                    resolved.append(ResolvedMention(surface_text=m, entity_id=ent_id, matched_via="alias"))

        # Scan text for names and aliases
        text_lower = text.lower()
        for name, ent_id in name_to_id.items():
            if ent_id not in seen_entities and name in text_lower:
                seen_entities.add(ent_id)
                resolved.append(ResolvedMention(surface_text=name, entity_id=ent_id, matched_via="canonical"))

        for alias, ent_id in alias_to_id.items():
            if ent_id not in seen_entities and alias in text_lower:
                seen_entities.add(ent_id)
                resolved.append(ResolvedMention(surface_text=alias, entity_id=ent_id, matched_via="alias"))

        return resolved

    # --- Transactional Context Manager ---

    def step_transaction(
        self,
        step: int,
        *,
        text: str = "",
        mentions: Sequence[str] = (),
        top_k_voice: int = 3,
        timeline_id: str | None = None,
        rerun_clean: bool = True,
    ) -> StepTransaction:
        return StepTransaction(
            self,
            step,
            text=text,
            mentions=mentions,
            top_k_voice=top_k_voice,
            timeline_id=timeline_id,
            rerun_clean=rerun_clean,
        )

    # --- State Mutation ---

    def apply_delta(self, step: int, delta: StepDelta, *, timeline_id: str | None = None) -> None:
        tl = timeline_id or self.active_timeline
        self.store.apply_delta(step, delta, timeline_id=tl)

    def extract_dialogue_candidates(self, text: str, *, source_text: str = "") -> list[DialogueCandidate]:
        return self.voice_bank.gate.extract_candidates(text, source_text=source_text)

    def index_voice_exemplars(
        self,
        step: int,
        exemplars: Sequence[VoiceExemplar],
        *,
        bypass_gate: bool = False,
        timeline_id: str | None = None,
    ) -> list[VoiceExemplar]:
        tl = timeline_id or self.active_timeline
        indexed: list[VoiceExemplar] = []
        to_store: list[tuple[VoiceExemplar, bytes]] = []

        for ex in exemplars:
            if not bypass_gate and not self.voice_bank.gate.passes_quality_gate(ex):
                continue
            ex_id = ex.exemplar_id or str(uuid.uuid4())
            item = ex.model_copy(update={"exemplar_id": ex_id, "step": step, "timeline_id": tl})
            emb = self.voice_bank.embedder.embed_text(item.text)
            to_store.append((item, emb.tobytes()))
            indexed.append(item)

        if to_store:
            self.store.insert_voice_exemplars(to_store, timeline_id=tl)
        return indexed

    def commit_step(
        self,
        step: int,
        delta: StepDelta | None = None,
        exemplars: Sequence[VoiceExemplar] = (),
        *,
        bypass_gate: bool = False,
        timeline_id: str | None = None,
    ) -> list[VoiceExemplar]:
        tl = timeline_id or self.active_timeline
        if delta:
            self.apply_delta(step, delta, timeline_id=tl)
        indexed: list[VoiceExemplar] = []
        if exemplars:
            indexed = self.index_voice_exemplars(step, exemplars, bypass_gate=bypass_gate, timeline_id=tl)
        return indexed

    # --- Time-Travel Operations ---

    def rollback_to(self, step: int, *, timeline_id: str | None = None) -> int:
        tl = timeline_id or self.active_timeline
        return self.store.rollback_to(step, timeline_id=tl)

    def branch(
        self,
        at_step: int,
        name: str,
        *,
        from_timeline: str | None = None,
        switch: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineInfo:
        parent = from_timeline or self.active_timeline
        info = self.store.create_timeline(name, parent_id=parent, fork_step=at_step, metadata=metadata)
        if switch:
            self.active_timeline = name
        return info

    def checkout(self, timeline_id: str) -> None:
        if not self.store.get_timeline(timeline_id):
            raise ValueError(f"Timeline '{timeline_id}' does not exist.")
        self.active_timeline = timeline_id

    def list_timelines(self) -> list[TimelineInfo]:
        return self.store.list_timelines()

    def get_history(self, subject_key: str, *, timeline_id: str | None = None) -> list[dict[str, Any]]:
        tl = timeline_id or self.active_timeline
        return self.store.get_history(subject_key, timeline_id=tl)

    def diff_timelines(self, timeline_a: str, timeline_b: str, *, at_step: int) -> dict[str, Any]:
        return self.store.diff_timelines(timeline_a, timeline_b, at_step=at_step)

    def export_snapshot(
        self,
        target_path: str | Path,
        *,
        at_step: int,
        timeline_id: str | None = None,
    ) -> Path:
        tl = timeline_id or self.active_timeline
        return self.store.export_snapshot(target_path, at_step=at_step, timeline_id=tl)

    def format_context(self, ctx: StepContext) -> str:
        """Formats StepContext into prompt text using the active DomainPreset."""
        return self.preset.format_prompt_context(ctx)
