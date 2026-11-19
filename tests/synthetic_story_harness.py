from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from canonarc import (
    AliasRecord,
    DialogueCandidate,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    InteractiveFictionPreset,
    CanonArcEngine,
    StepDelta,
    VoiceExemplar,
)
from canonarc.extractors import (
    CallableExtractor,
    DeltaExtractor,
    OpenAICompatibleExtractor,
    _strip_json_fences,
)


# ============================================================================
# 1. Starting Synthetic Parameters (Ground-Truth Seed Specification)
# ============================================================================


@dataclass
class CharacterSeed:
    entity_id: str
    canonical_name: str
    base_attributes: dict[str, Any]
    initial_mood: str
    initial_location: str
    private_state: dict[str, Any] = field(default_factory=dict)
    aliases: list[str | AliasRecord] = field(default_factory=list)


@dataclass
class LocationSeed:
    entity_id: str
    canonical_name: str
    description: str


@dataclass
class ArtifactSeed:
    entity_id: str
    canonical_name: str
    initial_holder_id: str


@dataclass
class InitialEdgeSeed:
    source_id: str
    target_id: str
    relation_type: str
    state: dict[str, Any]


@dataclass
class SyntheticChunkSpec:
    """A single small-chunk step compiled from the starting synthetic parameters."""

    step: int
    scene_id: str
    chapter_id: str
    location_id: str
    story_time: str
    text: str
    # Ground-truth expectations after this chunk is committed:
    expected_room_occupants: dict[str, set[str]]
    expected_moods: dict[str, str]
    expected_prior_moods: dict[str, tuple[str, int]] = field(default_factory=dict)
    expected_edges: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    # Oracle StepDelta & Voice metadata for Mode 1 and ground-truth scoring:
    oracle_delta: StepDelta = field(default_factory=StepDelta)
    expected_gate1_passed_speakers: list[str] = field(default_factory=list)
    expected_gate1_rejected_count: int = 0
    voice_metadata_by_speaker: dict[str, dict[str, str]] = field(default_factory=dict)
    voice_quality_by_speaker: dict[str, float] = field(default_factory=dict)
    # Diagnostic tags for uncovered/boundary cases tested at this chunk:
    boundary_tags: list[str] = field(default_factory=list)
    acceptable_moods: dict[str, set[str]] = field(default_factory=dict)
    optional_inform: list[InformDelta] = field(default_factory=list)


@dataclass
class SynthStoryParams:
    """Starting synthetic parameters driving both Mode 1 (Deterministic) and Mode 2 (Live/Simulated LLM)."""

    title: str = "The Obsidian Signet Conspiracy"
    chunk_target_words: tuple[int, int] = (18, 55)
    characters: list[CharacterSeed] = field(default_factory=list)
    locations: list[LocationSeed] = field(default_factory=list)
    artifacts: list[ArtifactSeed] = field(default_factory=list)
    initial_edges: list[InitialEdgeSeed] = field(default_factory=list)

    @classmethod
    def default_conspiracy_seed(cls) -> SynthStoryParams:
        """Creates a rich 4-character, 3-room, 1-artifact synthetic story seed."""
        return cls(
            title="The Obsidian Signet Conspiracy",
            chunk_target_words=(18, 60),
            locations=[
                LocationSeed("loc_council", "Council Chamber", "Candlelit chamber with a long oak table"),
                LocationSeed("loc_vault", "Archive Vault", "Subterranean stone vault behind an iron grate"),
                LocationSeed("loc_balcony", "Windy Balcony", "High stone balcony overlooking the courtyard"),
            ],
            characters=[
                CharacterSeed(
                    entity_id="char_elena",
                    canonical_name="Elena",
                    base_attributes={"title": "Chancellor", "gender": "F", "declension": "type_a"},
                    initial_mood="calm",
                    initial_location="loc_council",
                    private_state={"secret_agenda": "expose_council_traitor"},
                    aliases=["the Iron Chancellor"],
                ),
                CharacterSeed(
                    entity_id="char_marcus",
                    canonical_name="Marcus",
                    base_attributes={"title": "General", "gender": "M", "declension": "type_1"},
                    initial_mood="confident",
                    initial_location="loc_council",
                    private_state={"secret_agenda": "protect_garrison_funds"},
                    aliases=["the old general"],
                ),
                CharacterSeed(
                    entity_id="char_lyra",
                    canonical_name="Lyra",
                    base_attributes={"title": "Spymaster", "gender": "F", "declension": "type_a"},
                    initial_mood="watchful",
                    initial_location="loc_council",
                    private_state={"true_loyalty": "crown_intelligence"},
                    aliases=[
                        AliasRecord(
                            alias="the hooded courier",
                            entity_id="char_lyra",
                            kind="disguise",
                            persona_name="Courier",
                            known_to=["char_elena"],  # Elena knows Lyra's disguise; Marcus & Julian do NOT
                        )
                    ],
                ),
                CharacterSeed(
                    entity_id="char_julian",
                    canonical_name="Julian",
                    base_attributes={"title": "Treasurer", "gender": "M", "declension": "type_1"},
                    initial_mood="nervous",
                    initial_location="loc_council",
                    private_state={"guilt": "forged_royal_seal"},
                    aliases=["the young treasurer"],
                ),
            ],
            artifacts=[
                ArtifactSeed(
                    entity_id="item_signet",
                    canonical_name="Obsidian Signet",
                    initial_holder_id="char_julian",
                )
            ],
            initial_edges=[
                # Elena <-> Marcus: formal respect initially
                InitialEdgeSeed(
                    source_id="char_elena",
                    target_id="char_marcus",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "respectful", "trust": 0.7},
                ),
                InitialEdgeSeed(
                    source_id="char_marcus",
                    target_id="char_elena",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "guarded", "trust": 0.4},
                ),
                # Elena <-> Julian: kinship (cousins) + suspicious stance + polite facade
                InitialEdgeSeed(
                    source_id="char_elena",
                    target_id="char_julian",
                    relation_type="kinship",
                    state={"relation": "cousin"},
                ),
                InitialEdgeSeed(
                    source_id="char_elena",
                    target_id="char_julian",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "suspicious", "trust": -0.3},
                ),
                InitialEdgeSeed(
                    source_id="char_julian",
                    target_id="char_elena",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "fearful", "trust": -0.5},
                ),
                InitialEdgeSeed(
                    source_id="char_julian",
                    target_id="char_elena",
                    relation_type="facade",
                    state={"register": "Vi", "tone": "polite", "trust": 0.8},
                ),
                # Marcus falsely believes Julian is innocent & calm
                InitialEdgeSeed(
                    source_id="char_marcus",
                    target_id="char_julian",
                    relation_type="belief_state",
                    state={"mood": "composed", "loyalty": "devoted"},
                ),
                # Lyra (disguised as Courier) facade toward Marcus
                InitialEdgeSeed(
                    source_id="char_lyra",
                    target_id="char_marcus",
                    relation_type="facade",
                    state={"register": "Vi", "tone": "deferential", "trust": 0.5},
                ),
                # Artifact possession edge
                InitialEdgeSeed(
                    source_id="char_julian",
                    target_id="item_signet",
                    relation_type="possesses",
                    state={"status": "hidden_in_cloak"},
                ),
            ],
        )


# ============================================================================
# 2. Mode 1: Deterministic Story & Ground-Truth Compiler (Small Chunks)
# ============================================================================


class DeterministicStoryCompiler:
    """Compiles SynthStoryParams into 18 small prose chunks (1-3 sentences each)
    covering all Tiers 1-6 plus 5 uncovered boundary stress-test scenarios.
    """

    def __init__(self, params: SynthStoryParams) -> None:
        self.params = params

    def compile_chunks(self) -> list[SyntheticChunkSpec]:
        # Track rolling ground truth as we compile steps 1..18
        rooms: dict[str, set[str]] = {
            "loc_council": {"char_elena", "char_marcus", "char_lyra", "char_julian"},
            "loc_vault": set(),
            "loc_balcony": set(),
        }
        moods: dict[str, str] = {c.entity_id: c.initial_mood for c in self.params.characters}
        prior_moods: dict[str, tuple[str, int]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {
            (e.source_id, e.target_id, e.relation_type): dict(e.state)
            for e in self.params.initial_edges
        }

        def _snap_rooms() -> dict[str, set[str]]:
            return {k: set(v) for k, v in rooms.items()}

        def _snap_moods() -> dict[str, str]:
            return dict(moods)

        def _snap_prior() -> dict[str, tuple[str, int]]:
            return dict(prior_moods)

        def _snap_edges() -> dict[tuple[str, str, str], dict[str, Any]]:
            return {k: dict(v) for k, v in edges.items()}

        def _set_mood(cid: str, new_mood: str, prev_step: int) -> None:
            old = moods[cid]
            if old != new_mood:
                prior_moods[cid] = (old, prev_step)
            moods[cid] = new_mood

        def _move_char(cid: str, target_room: str) -> None:
            for rset in rooms.values():
                rset.discard(cid)
            rooms[target_room].add(cid)

        chunks: list[SyntheticChunkSpec] = []

        # --- CHUNK 1: Opening in Council Chamber (4 characters in room; formal Vi voice + filler quote) ---
        chunks.append(
            SyntheticChunkSpec(
                step=1,
                scene_id="scene_1_council",
                chapter_id="ch_1",
                location_id="loc_council",
                story_time="day_1_dusk",
                text=(
                    'Inside the Council Chamber, Elena addressed the old general with measured solemnity. '
                    '"Molim vas da pažljivo pregledate ove poverljive izveštaje pre nego što kraljevska straža zatvori kapije." '
                    'Julian murmured, "Yes, cousin."'
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_1_council",
                    chapter_id="ch_1",
                    story_time="day_1_dusk",
                    episodes=[
                        EpisodeRecord(
                            episode_id="ep_step1_opening_brief",
                            predicate="briefing",
                            summary="Elena presented the confidential garrison reports to Marcus in the Council Chamber.",
                            participants={"char_elena": "actor", "char_marcus": "target"},
                            witnesses=["char_elena", "char_marcus", "char_lyra", "char_julian"],
                            location_id="loc_council",
                            salience=3.0,
                        )
                    ],
                ),
                expected_gate1_passed_speakers=["char_elena"],
                expected_gate1_rejected_count=1,  # "Yes, cousin." rejected by Gate 1 (too short)
                voice_metadata_by_speaker={"char_elena": {"register": "Vi", "tone": "respectful"}},
                voice_quality_by_speaker={"char_elena": 9.3},
            )
        )

        # --- CHUNK 2: Disguised character ("the hooded courier") speaks to Marcus ---
        chunks.append(
            SyntheticChunkSpec(
                step=2,
                scene_id="scene_1_council",
                chapter_id="ch_1",
                location_id="loc_council",
                story_time="day_1_dusk",
                text=(
                    'The hooded courier bowed low before Marcus near the council brazier. '
                    '"Ponizno vam donosim zapečaćene depeše sa severne granice koje zahtevaju vašu hitnu vojnu procenu."'
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_1_council",
                    chapter_id="ch_1",
                    story_time="day_1_dusk",
                    episodes=[
                        EpisodeRecord(
                            episode_id="ep_step2_courier_dispatch",
                            predicate="delivery",
                            summary="The hooded courier delivered northern border dispatches to Marcus.",
                            participants={"char_lyra": "actor", "char_marcus": "target"},
                            witnesses=["char_elena", "char_marcus", "char_lyra", "char_julian"],
                            location_id="loc_council",
                            salience=3.5,
                        )
                    ],
                ),
                expected_gate1_passed_speakers=["char_lyra"],
                voice_metadata_by_speaker={
                    "char_lyra": {"register": "Vi", "tone": "deferential", "persona": "Courier"}
                },
                voice_quality_by_speaker={"char_lyra": 9.1},
            )
        )

        # --- CHUNK 3: Room headcount changes (Marcus leaves Council Chamber for Windy Balcony: 4 -> 3) ---
        _move_char("char_marcus", "loc_balcony")
        _set_mood("char_marcus", "restless", 0)
        chunks.append(
            SyntheticChunkSpec(
                step=3,
                scene_id="scene_1_council",
                chapter_id="ch_1",
                location_id="loc_council",
                story_time="day_1_dusk",
                text=(
                    "Growing restless from the smoky air, Marcus excused himself from the Council Chamber "
                    "and stepped out onto the Windy Balcony, pulling the heavy terrace doors shut."
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_1_council",
                    chapter_id="ch_1",
                    story_time="day_1_dusk",
                    entities=[
                        EntityDelta(
                            entity_id="char_marcus",
                            state={"mood": "restless", "location_id": "loc_balcony"},
                            reason="Stepped out to the balcony for air",
                        )
                    ],
                    episodes=[
                        EpisodeRecord(
                            episode_id="ep_step3_marcus_exits",
                            predicate="room_exit",
                            summary="Marcus left the Council Chamber and went to the Windy Balcony.",
                            participants={"char_marcus": "actor"},
                            witnesses=["char_elena", "char_marcus", "char_lyra", "char_julian"],
                            location_id="loc_council",
                            salience=2.5,
                        )
                    ],
                ),
            )
        )

        # --- CHUNK 4: Secret event in Council Chamber while Marcus is on the Balcony! ---
        # Also tests UNCOVERED BOUNDARY 1 (implicit room occupant witnessing: Lyra is in loc_council
        # but her name/alias is NOT mentioned in Chunk 4's prose!).
        _set_mood("char_elena", "furious", 0)
        _set_mood("char_julian", "panicked", 0)
        edges[("char_elena", "char_julian", "stance")] = {
            "register": "ti",
            "tone": "furious",
            "trust": -0.95,
        }
        chunks.append(
            SyntheticChunkSpec(
                step=4,
                scene_id="scene_1_council",
                chapter_id="ch_1",
                location_id="loc_council",
                story_time="day_1_dusk",
                text=(
                    "As soon as the balcony doors latched, Elena confronted Julian over the missing treasury ledger. "
                    '"Prestani sa tim bednim lažima, izdajniče, jer znam da si sinoć ukrao crni pečat iz komore!" '
                    "Julian trembled in panic as the accusation echoed through the silent room."
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_1_council",
                    chapter_id="ch_1",
                    story_time="day_1_dusk",
                    entities=[
                        EntityDelta(
                            entity_id="char_elena",
                            state={"mood": "furious"},
                            caused_by_episode_id="ep_step4_seal_accusation",
                            reason="Confronted Julian over the stolen Obsidian Signet",
                        ),
                        EntityDelta(
                            entity_id="char_julian",
                            state={"mood": "panicked"},
                            caused_by_episode_id="ep_step4_seal_accusation",
                            reason="Exposed by Elena in the Council Chamber",
                        ),
                    ],
                    edges=[
                        EdgeDelta(
                            source_id="char_elena",
                            target_id="char_julian",
                            relation_type="stance",
                            state={"register": "ti", "tone": "furious", "trust": -0.95},
                            caused_by_episode_id="ep_step4_seal_accusation",
                            reason="Dropped formal register after catching Julian's theft",
                        )
                    ],
                    episodes=[
                        EpisodeRecord(
                            episode_id="ep_step4_seal_accusation",
                            predicate="confrontation",
                            summary="Elena accused Julian of stealing the Obsidian Signet while Marcus was outside on the balcony.",
                            participants={
                                "char_elena": "actor",
                                "char_julian": "target",
                                "item_signet": "object",
                            },
                            # Ground-truth witnesses: Elena, Julian, and Lyra (who is still in loc_council), NOT Marcus!
                            witnesses=["char_elena", "char_julian", "char_lyra"],
                            location_id="loc_council",
                            salience=9.2,
                        )
                    ],
                ),
                expected_gate1_passed_speakers=["char_elena"],
                voice_metadata_by_speaker={"char_elena": {"register": "ti", "tone": "furious"}},
                voice_quality_by_speaker={"char_elena": 9.6},
                boundary_tags=["implicit_room_witness_unmentioned_in_text"],
            )
        )

        # --- CHUNK 5: UNCOVERED BOUNDARY 2 (Cross-Chunk Pronoun Coreference) + UNCOVERED BOUNDARY 3 (Same-Mood Repeat) ---
        # Notice: Text only uses "She" and "him" with NO character names or aliases!
        chunks.append(
            SyntheticChunkSpec(
                step=5,
                scene_id="scene_1_council",
                chapter_id="ch_1",
                location_id="loc_council",
                story_time="day_1_dusk",
                text=(
                    "Still furious, she slammed her palm against the oak table and demanded that he empty his pockets immediately. "
                    "He backed against the tapestry, clutching his velvet cloak."
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),  # prior_mood for Elena MUST remain ("calm", 0), NOT ("furious", 4)!
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_1_council",
                    chapter_id="ch_1",
                    story_time="day_1_dusk",
                    # Intentionally re-emitting mood="furious" to test whether repeated mood in small chunks clobbers prior_state!
                    entities=[
                        EntityDelta(
                            entity_id="char_elena",
                            state={"mood": "furious"},
                            reason="Continued pressing Julian in the Council Chamber",
                        )
                    ],
                ),
                boundary_tags=["cross_chunk_pronoun_coreference", "intra_scene_same_mood_repeat"],
            )
        )

        # --- CHUNK 6: UNCOVERED BOUNDARY 4 (Artifact Possession Transfer) + Low-Quality Gate 2 Rejection ---
        # Julian hands the Obsidian Signet to the hooded courier (Lyra) and flees to the Archive Vault (3 -> 2 in Council).
        _move_char("char_julian", "loc_vault")
        _move_char("char_lyra", "loc_vault")
        edges[("char_lyra", "item_signet", "possesses")] = {"status": "concealed_in_satchel"}
        chunks.append(
            SyntheticChunkSpec(
                step=6,
                scene_id="scene_1_council",
                chapter_id="ch_1",
                location_id="loc_council",
                story_time="day_1_dusk",
                text=(
                    "In a desperate move, Julian pressed the Obsidian Signet into the hooded courier's gloved hand "
                    "before bolting down the spiral stairs into the Archive Vault, with the hooded courier chasing after him. "
                    'Julian stammered, "Take this cursed ring and hide it deep inside the stone vault before the guards arrive!"'
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_1_council",
                    chapter_id="ch_1",
                    story_time="day_1_dusk",
                    entities=[
                        EntityDelta(
                            entity_id="char_julian",
                            state={"location_id": "loc_vault"},
                        ),
                        EntityDelta(
                            entity_id="char_lyra",
                            state={"location_id": "loc_vault"},
                        ),
                    ],
                    edges=[
                        # Note: Without explicit remove=True on (char_julian, item_signet, possesses),
                        # both Julian and Lyra would have active 'possesses' edges!
                        EdgeDelta(
                            source_id="char_lyra",
                            target_id="item_signet",
                            relation_type="possesses",
                            state={"status": "concealed_in_satchel"},
                            caused_by_episode_id="ep_step6_signet_handoff",
                        )
                    ],
                    episodes=[
                        EpisodeRecord(
                            episode_id="ep_step6_signet_handoff",
                            predicate="item_transfer",
                            summary="Julian handed the Obsidian Signet to the hooded courier and fled into the Archive Vault.",
                            participants={
                                "char_julian": "actor",
                                "char_lyra": "recipient",
                                "item_signet": "instrument",
                            },
                            witnesses=["char_elena", "char_julian", "char_lyra"],
                            location_id="loc_council",
                            salience=8.8,
                        )
                    ],
                ),
                expected_gate1_passed_speakers=["char_julian"],
                voice_metadata_by_speaker={"char_julian": {"register": "ti", "tone": "panicked"}},
                voice_quality_by_speaker={"char_julian": 5.5},  # Must be REJECTED by Gate 2 (< 8.0)!
                boundary_tags=["exclusive_inventory_transfer"],
            )
        )

        # --- CHUNK 7: Flashback Chunk (10 Years Ago) ---
        # Must record the flashback episode WITHOUT overwriting Elena's or Marcus's present-day mood/edges!
        chunks.append(
            SyntheticChunkSpec(
                step=7,
                scene_id="scene_2_flashback",
                chapter_id="ch_1",
                location_id="loc_council",
                story_time="10_years_ago",
                text=(
                    "Elena stared at the empty doorway, remembering the winter siege ten years ago when she was a naive apprentice "
                    "and trusted Julian blindly during the treaty signing."
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_2_flashback",
                    chapter_id="ch_1",
                    story_time="10_years_ago",
                    is_flashback=True,
                    # Even if an extractor emits past mood/edges here, is_flashback=True must block overwriting present state!
                    entities=[
                        EntityDelta(
                            entity_id="char_elena",
                            state={"mood": "naive_trusting"},
                        )
                    ],
                    edges=[
                        EdgeDelta(
                            source_id="char_elena",
                            target_id="char_julian",
                            relation_type="stance",
                            state={"register": "ti", "tone": "affectionate", "trust": 0.9},
                        )
                    ],
                    episodes=[
                        EpisodeRecord(
                            episode_id="ep_step7_siege_memory",
                            predicate="flashback",
                            summary="Ten years ago during the winter siege, Elena trusted Julian at the treaty signing.",
                            participants={"char_elena": "experiencer", "char_julian": "companion"},
                            witnesses=["char_elena", "char_julian"],
                            location_id="loc_council",
                            story_time="10_years_ago",
                            salience=6.5,
                        )
                    ],
                ),
            )
        )

        # --- CHUNK 8: Archive Vault Scene (2 characters in Vault: Lyra pierces disguise to Julian!) ---
        _set_mood("char_lyra", "determined", 0)
        _set_mood("char_julian", "shocked", 4)
        edges[("char_lyra", "char_julian", "stance")] = {
            "register": "ti",
            "tone": "commanding",
            "trust": -0.6,
        }
        chunks.append(
            SyntheticChunkSpec(
                step=8,
                scene_id="scene_3_vault",
                chapter_id="ch_2",
                location_id="loc_vault",
                story_time="day_1_night",
                text=(
                    "Down in the Archive Vault, the hooded courier threw back her cowl, revealing to the shocked young treasurer "
                    "that she was Lyra, the Crown Spymaster. "
                    '"Predaj mi sve lažne povelje odmah, Julijane, ili ćeš do zore biti u kraljevskoj tamnici!"'
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_3_vault",
                    chapter_id="ch_2",
                    story_time="day_1_night",
                    entities=[
                        EntityDelta(
                            entity_id="char_lyra",
                            state={"mood": "determined"},
                            add_aliases=[
                                AliasRecord(
                                    alias="the hooded courier",
                                    entity_id="char_lyra",
                                    kind="disguise",
                                    persona_name="Courier",
                                    # Now BOTH Elena and Julian know the disguise; Marcus still does NOT!
                                    known_to=["char_elena", "char_julian"],
                                )
                            ],
                        ),
                        EntityDelta(
                            entity_id="char_julian",
                            state={"mood": "shocked"},
                            caused_by_episode_id="ep_step8_lyra_unmasked",
                            reason="Discovered the hooded courier is Spymaster Lyra",
                        ),
                    ],
                    edges=[
                        EdgeDelta(
                            source_id="char_lyra",
                            target_id="char_julian",
                            relation_type="stance",
                            state={"register": "ti", "tone": "commanding", "trust": -0.6},
                            caused_by_episode_id="ep_step8_lyra_unmasked",
                        )
                    ],
                    episodes=[
                        EpisodeRecord(
                            episode_id="ep_step8_lyra_unmasked",
                            predicate="revelation",
                            summary="Lyra unmasked her courier disguise in the Archive Vault and seized the Obsidian Signet from Julian.",
                            participants={"char_lyra": "actor", "char_julian": "target"},
                            witnesses=["char_lyra", "char_julian"],
                            location_id="loc_vault",
                            salience=9.0,
                        )
                    ],
                ),
                expected_gate1_passed_speakers=["char_lyra"],
                voice_metadata_by_speaker={"char_lyra": {"register": "ti", "tone": "commanding"}},
                voice_quality_by_speaker={"char_lyra": 9.4},
            )
        )

        # --- CHUNK 9: Epistemic Deception + Sarcastic Surface Dialogue (UNCOVERED BOUNDARY 5) ---
        # Marcus returns from the balcony to the Council Chamber (now Elena + Marcus = 2 in Council).
        # Julian also returns and LIES to Marcus about ep_step4_seal_accusation!
        _move_char("char_marcus", "loc_council")
        _move_char("char_julian", "loc_council")
        chunks.append(
            SyntheticChunkSpec(
                step=9,
                scene_id="scene_4_return",
                chapter_id="ch_2",
                location_id="loc_council",
                story_time="day_1_night",
                text=(
                    "Marcus stepped back into the Council Chamber just as Julian returned from the stairs wearing a polite smile. "
                    "Julian told the old general that northern raiders had broken into the annex and stolen the Obsidian Signet. "
                    'Though still furious inside, Elena said with icy sarcasm, "Oh, what a wonderfully convenient tale of heroic vigilance, dear cousin."'
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_4_return",
                    chapter_id="ch_2",
                    story_time="day_1_night",
                    entities=[
                        EntityDelta(
                            entity_id="char_marcus",
                            state={"location_id": "loc_council"},
                        ),
                        EntityDelta(
                            entity_id="char_julian",
                            state={"location_id": "loc_council"},
                        ),
                    ],
                    inform=[
                        InformDelta(
                            episode_id="ep_step4_seal_accusation",
                            entity_id="char_marcus",
                            epistemic="deceived",
                            believed_summary="Northern raiders broke into the annex and stole the Obsidian Signet while Julian defended the ledger.",
                        )
                    ],
                ),
                boundary_tags=["sarcastic_surface_dialogue_vs_true_mood"],
            )
        )

        # --- CHUNK 10: Secondhand Truth ("told") + Memory Suppression ("forgotten") ---
        # Lyra returns to the Council Chamber (all 4 back in loc_council!), informs Marcus of the truth,
        # and Julian drinks a draught of Lethe to wipe his memory of the siege flashback!
        _move_char("char_lyra", "loc_council")
        _set_mood("char_marcus", "indignant", 3)
        edges[("char_marcus", "char_julian", "stance")] = {
            "register": "ti",
            "tone": "stern",
            "trust": -0.8,
        }
        chunks.append(
            SyntheticChunkSpec(
                step=10,
                scene_id="scene_4_return",
                chapter_id="ch_2",
                location_id="loc_council",
                story_time="day_1_night",
                text=(
                    "Lyra entered the Council Chamber holding the recovered Obsidian Signet and informed Marcus of Julian's theft. "
                    "Cornered at the table, Julian swallowed a bitter alchemical draught that blanked his memory of the old winter siege. "
                    'Marcus barked at Julian, "Kako se usuđuješ da obmanjuješ vojni savet dok držimo dokaze u rukama!"'
                ),
                expected_room_occupants=_snap_rooms(),
                expected_moods=_snap_moods(),
                expected_prior_moods=_snap_prior(),
                expected_edges=_snap_edges(),
                oracle_delta=StepDelta(
                    scene_id="scene_4_return",
                    chapter_id="ch_2",
                    story_time="day_1_night",
                    entities=[
                        EntityDelta(
                            entity_id="char_lyra",
                            state={"location_id": "loc_council"},
                        ),
                        EntityDelta(
                            entity_id="char_marcus",
                            state={"mood": "indignant"},
                            caused_by_episode_id="ep_step4_seal_accusation",
                            reason="Learned from Lyra that Julian stole the signet and lied",
                        ),
                    ],
                    edges=[
                        EdgeDelta(
                            source_id="char_marcus",
                            target_id="char_julian",
                            relation_type="stance",
                            state={"register": "ti", "tone": "stern", "trust": -0.8},
                            caused_by_episode_id="ep_step4_seal_accusation",
                        )
                    ],
                    inform=[
                        # Marcus transitions from 'deceived' -> 'told' on ep_step4_seal_accusation
                        InformDelta(
                            episode_id="ep_step4_seal_accusation",
                            entity_id="char_marcus",
                            epistemic="told",
                        ),
                        # Julian forgets ep_step7_siege_memory
                        InformDelta(
                            episode_id="ep_step7_siege_memory",
                            entity_id="char_julian",
                            epistemic="forgotten",
                        ),
                    ],
                ),
                expected_gate1_passed_speakers=["char_marcus"],
                voice_metadata_by_speaker={"char_marcus": {"register": "ti", "tone": "stern"}},
                voice_quality_by_speaker={"char_marcus": 9.2},
            )
        )

        return chunks


# ============================================================================
# 3. Swappable Extractors for Mode 1 (Oracle & Heuristic) and Mode 2 (LLM)
# ============================================================================


class BlueprintOracleExtractor:
    """Mode 1A Extractor: Returns the ground-truth StepDelta for each chunk via StepTransaction.extract_and_commit()."""

    def __init__(self, chunks: Sequence[SyntheticChunkSpec]) -> None:
        self._by_text: dict[str, StepDelta] = {c.text.strip(): c.oracle_delta for c in chunks}
        self.call_count: int = 0
        self.last_prompt: str = ""

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        self.call_count += 1
        self.last_prompt = prompt
        delta = self._by_text.get(scene_text.strip())
        if delta is None:
            return StepDelta()
        return delta.model_copy(deep=True)


class HeuristicNarrativeExtractor:
    """Mode 1B / Offline NLP Extractor: Parses small prose chunks and extraction prompts into StepDelta JSON
    without looking at oracle_delta—testing what a purely text-driven extractor can recover from small chunks.
    """

    def __init__(self) -> None:
        self.call_count: int = 0

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        self.call_count += 1
        text_lower = scene_text.lower()
        entities: list[EntityDelta] = []
        edges: list[EdgeDelta] = []
        episodes: list[EpisodeRecord] = []
        inform: list[InformDelta] = []
        is_flashback = "ten years ago" in text_lower or "remembering" in text_lower

        # Room transitions
        if "marcus" in text_lower and "stepped out onto the windy balcony" in text_lower:
            entities.append(
                EntityDelta(
                    entity_id="Marcus",
                    state={"mood": "restless", "location_id": "loc_balcony"},
                    reason="Stepped out onto the Windy Balcony",
                )
            )
            episodes.append(
                EpisodeRecord(
                    episode_id="ep_step3_marcus_exits",
                    predicate="room_exit",
                    summary="Marcus left the Council Chamber for the Windy Balcony.",
                    participants={"Marcus": "actor"},
                    witnesses=["Elena", "Marcus", "Lyra", "Julian"],
                    location_id="loc_council",
                    salience=2.5,
                )
            )

        # Confrontation in Chunk 4
        if "elena confronted julian" in text_lower:
            entities.extend(
                [
                    EntityDelta(
                        entity_id="Elena",
                        state={"mood": "furious"},
                        caused_by_episode_id="ep_step4_seal_accusation",
                        reason="Confronted Julian over the missing ledger and stolen signet",
                    ),
                    EntityDelta(
                        entity_id="Julian",
                        state={"mood": "panicked"},
                        caused_by_episode_id="ep_step4_seal_accusation",
                    ),
                ]
            )
            edges.append(
                EdgeDelta(
                    source_id="Elena",
                    target_id="Julian",
                    relation_type="stance",
                    state={"register": "ti", "tone": "furious", "trust": -0.95},
                    caused_by_episode_id="ep_step4_seal_accusation",
                )
            )
            # Notice: A text-only extractor on Chunk 4 only sees "Elena" and "Julian" in the text,
            # NOT "Lyra" (who is sitting quietly in loc_council)!
            episodes.append(
                EpisodeRecord(
                    episode_id="ep_step4_seal_accusation",
                    predicate="confrontation",
                    summary="Elena accused Julian of stealing the Obsidian Signet in the Council Chamber.",
                    participants={"Elena": "actor", "Julian": "target"},
                    witnesses=["Elena", "Julian"],
                    location_id="loc_council",
                    salience=9.2,
                )
            )

        # Pronoun-only Chunk 5 ("Still furious, she slammed her palm...")
        if text_lower.startswith("still furious, she slammed"):
            # Text has no names! Naive text extractor cannot resolve "she" without discourse carry-over.
            pass

        # Handoff & Vault flight in Chunk 6
        if "bolting down the spiral stairs into the archive vault" in text_lower:
            entities.extend(
                [
                    EntityDelta(entity_id="Julian", state={"location_id": "loc_vault"}),
                    EntityDelta(entity_id="char_lyra", state={"location_id": "loc_vault"}),
                ]
            )
            edges.append(
                EdgeDelta(
                    source_id="char_lyra",
                    target_id="Obsidian Signet",
                    relation_type="possesses",
                    state={"status": "concealed_in_satchel"},
                )
            )
            episodes.append(
                EpisodeRecord(
                    episode_id="ep_step6_signet_handoff",
                    predicate="item_transfer",
                    summary="Julian handed the Obsidian Signet to the hooded courier and fled into the Archive Vault.",
                    participants={"Julian": "actor", "char_lyra": "recipient"},
                    witnesses=["Elena", "Julian", "char_lyra"],
                    location_id="loc_council",
                    salience=8.8,
                )
            )

        # Flashback in Chunk 7
        if is_flashback:
            episodes.append(
                EpisodeRecord(
                    episode_id="ep_step7_siege_memory",
                    predicate="flashback",
                    summary="Ten years ago during the winter siege, Elena trusted Julian.",
                    participants={"Elena": "experiencer", "Julian": "companion"},
                    witnesses=["Elena", "Julian"],
                    location_id="loc_council",
                    story_time="10_years_ago",
                    salience=6.5,
                )
            )

        # Unmasking in Chunk 8
        if "threw back her cowl" in text_lower:
            entities.extend(
                [
                    EntityDelta(
                        entity_id="Lyra",
                        state={"mood": "determined"},
                        add_aliases=[
                            AliasRecord(
                                alias="the hooded courier",
                                entity_id="char_lyra",
                                kind="disguise",
                                persona_name="Courier",
                                known_to=["char_elena", "char_julian"],
                            )
                        ],
                    ),
                    EntityDelta(entity_id="Julian", state={"mood": "shocked"}),
                ]
            )
            edges.append(
                EdgeDelta(
                    source_id="Lyra",
                    target_id="Julian",
                    relation_type="stance",
                    state={"register": "ti", "tone": "commanding", "trust": -0.6},
                )
            )
            episodes.append(
                EpisodeRecord(
                    episode_id="ep_step8_lyra_unmasked",
                    predicate="revelation",
                    summary="Lyra unmasked her courier disguise to Julian in the Archive Vault.",
                    participants={"Lyra": "actor", "Julian": "target"},
                    witnesses=["Lyra", "Julian"],
                    location_id="loc_vault",
                    salience=9.0,
                )
            )

        # Deception + Sarcasm in Chunk 9
        if "northern raiders had broken into the annex" in text_lower:
            entities.extend(
                [
                    EntityDelta(entity_id="Marcus", state={"location_id": "loc_council"}),
                    EntityDelta(entity_id="Julian", state={"location_id": "loc_council"}),
                ]
            )
            inform.append(
                InformDelta(
                    episode_id="ep_step4_seal_accusation",
                    entity_id="Marcus",
                    epistemic="deceived",
                    believed_summary="Northern raiders broke into the annex and stole the Obsidian Signet.",
                )
            )

        # Truth told + memory forgotten in Chunk 10
        if "informed marcus of julian's theft" in text_lower:
            entities.extend(
                [
                    EntityDelta(entity_id="Lyra", state={"location_id": "loc_council"}),
                    EntityDelta(entity_id="Marcus", state={"mood": "indignant"}),
                ]
            )
            edges.append(
                EdgeDelta(
                    source_id="Marcus",
                    target_id="Julian",
                    relation_type="stance",
                    state={"register": "ti", "tone": "stern", "trust": -0.8},
                )
            )
            inform.extend(
                [
                    InformDelta(
                        episode_id="ep_step4_seal_accusation",
                        entity_id="Marcus",
                        epistemic="told",
                    ),
                    InformDelta(
                        episode_id="ep_step7_siege_memory",
                        entity_id="Julian",
                        epistemic="forgotten",
                    ),
                ]
            )

        return StepDelta(
            entities=entities,
            edges=edges,
            episodes=episodes,
            inform=inform,
            is_flashback=is_flashback,
        )


# ============================================================================
# 4. Mode 2: Live / Simulated LLM Closed-Loop Story Generator & Extractor
# ============================================================================


class LLMStoryClosedLoopHarness:
    """Mode 2 Harness:
    1. Builds a structured LLM prompt from SynthStoryParams to generate a multi-chunk short story.
    2. Slices or parses the LLM story into small chunks (1-3 sentences each).
    3. Runs StepTransaction.extract_and_commit() using CallableExtractor / OpenAICompatibleExtractor
       (including Markdown fence stripping, schema validation retry, and entity name normalization).
    """

    def __init__(
        self,
        params: SynthStoryParams,
        *,
        story_writer_fn: Callable[[str], str] | None = None,
        delta_llm_fn: Callable[[str, dict[str, Any]], str | dict[str, Any]] | None = None,
        max_retries: int = 2,
    ) -> None:
        self.params = params
        self.compiler = DeterministicStoryCompiler(params)
        self.story_writer_fn = story_writer_fn or self._default_simulated_llm_writer
        self.delta_llm_fn = delta_llm_fn or self._default_simulated_llm_extractor
        self.extractor = CallableExtractor(self.delta_llm_fn, max_retries=max_retries)
        self.raw_llm_calls: list[dict[str, Any]] = []

    @classmethod
    def from_openai_compatible(
        cls,
        params: SynthStoryParams,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        max_retries: int = 2,
    ) -> LLMStoryClosedLoopHarness:
        """Creates a Live LLM Mode 2 harness backed by any OpenAI-compatible or Ollama endpoint."""
        live_extractor = OpenAICompatibleExtractor(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
        )

        def _writer_fn(prompt: str) -> str:
            return live_extractor._call_api(prompt, {})

        def _delta_fn(prompt: str, schema: dict[str, Any]) -> str:
            return live_extractor._call_api(prompt, schema)

        return cls(
            params,
            story_writer_fn=_writer_fn,
            delta_llm_fn=_delta_fn,
            max_retries=max_retries,
        )

    def build_story_generation_prompt(self) -> str:
        """Builds the prompt instructing an LLM to write a synthetic story from SynthStoryParams."""
        char_lines = [
            f"- {c.canonical_name} ({c.entity_id}): initial_mood={c.initial_mood}, "
            f"room={c.initial_location}, private={c.private_state}"
            for c in self.params.characters
        ]
        loc_lines = [f"- {loc.canonical_name} ({loc.entity_id}): {loc.description}" for loc in self.params.locations]
        return (
            f"Write a short story titled '{self.params.title}' split into 10 small chunks "
            f"({self.params.chunk_target_words[0]}-{self.params.chunk_target_words[1]} words each).\n\n"
            "### Characters:\n" + "\n".join(char_lines) + "\n\n"
            "### Locations:\n" + "\n".join(loc_lines) + "\n\n"
            "Return JSON with key 'chunks', where each item has 'step', 'location_id', and 'text'."
        )

    def _default_simulated_llm_writer(self, prompt: str) -> str:  # noqa: ARG002
        """Simulates an LLM story writer returning Markdown-fenced JSON chunks."""
        compiled = self.compiler.compile_chunks()
        payload = {
            "title": self.params.title,
            "chunks": [
                {
                    "step": c.step,
                    "scene_id": c.scene_id,
                    "chapter_id": c.chapter_id,
                    "location_id": c.location_id,
                    "story_time": c.story_time,
                    "text": c.text,
                }
                for c in compiled
            ],
        }
        return f"```json\n{json.dumps(payload, indent=2)}\n```"

    def _default_simulated_llm_extractor(self, prompt: str, schema: dict[str, Any]) -> str:  # noqa: ARG002
        """Simulates a realistic LLM JSON extractor:
        - Uses canonical names ("Elena", "Marcus", "Julian") instead of internal IDs to test normalize_delta()
        - Wraps output in ```json ... ``` code fences to test fence stripping
        - Intentionally emits a malformed JSON payload on the first attempt of Step 4 to test retry recovery!
        """
        target_marker = "### Target Scene Chunk (EXTRACT ONLY state changes and episodes occurring in THIS text):\n\n"
        scene_marker = "### Scene Text to Extract From:\n\n"
        if target_marker in prompt:
            raw_tail = prompt.split(target_marker)[-1]
        elif scene_marker in prompt:
            raw_tail = prompt.split(scene_marker)[-1]
        else:
            raw_tail = prompt
        following_marker = "\n\n### Following Context ("
        if following_marker in raw_tail:
            raw_tail = raw_tail.split(following_marker)[0]
        retry_marker = "\n\nPrevious output failed StepDelta JSON validation"
        scene_text = raw_tail.split(retry_marker)[0].strip()
        attempt_num = sum(1 for call in self.raw_llm_calls if call["scene_text"] == scene_text) + 1
        self.raw_llm_calls.append({"scene_text": scene_text, "attempt": attempt_num, "prompt": prompt})

        # Simulate transient malformed JSON on attempt 1 of Step 4 to verify CallableExtractor retry!
        if "Elena confronted Julian" in scene_text and attempt_num == 1:
            return "```json\n{\"entities\": [{\"entity_id\": \"Elena\", INVALID_JSON\n```"

        compiled_by_text = {c.text.strip(): c for c in self.compiler.compile_chunks()}
        spec = compiled_by_text.get(scene_text)
        if spec is None:
            return json.dumps(StepDelta().model_dump())

        # Convert canonical IDs to surface names in the raw LLM JSON to test normalize_delta()
        id_to_surface = {
            "char_elena": "Elena",
            "char_marcus": "Marcus",
            "char_julian": "the young treasurer",
            "item_signet": "Obsidian Signet",
        }
        raw_dict = spec.oracle_delta.model_dump()
        for ent in raw_dict.get("entities", []):
            ent["entity_id"] = id_to_surface.get(ent["entity_id"], ent["entity_id"])
        for edge in raw_dict.get("edges", []):
            edge["source_id"] = id_to_surface.get(edge["source_id"], edge["source_id"])
            edge["target_id"] = id_to_surface.get(edge["target_id"], edge["target_id"])
        for ep in raw_dict.get("episodes", []):
            ep["participants"] = {
                id_to_surface.get(k, k): v for k, v in ep.get("participants", {}).items()
            }
            ep["witnesses"] = [id_to_surface.get(w, w) for w in ep.get("witnesses", [])]
        for inf in raw_dict.get("inform", []):
            inf["entity_id"] = id_to_surface.get(inf["entity_id"], inf["entity_id"])

        return f"```json\n{json.dumps(raw_dict)}\n```"

    def generate_story_chunks(self) -> list[SyntheticChunkSpec]:
        """Generates the story via the LLM writer and pairs each chunk with expected ground truth."""
        prompt = self.build_story_generation_prompt()
        raw_response = self.story_writer_fn(prompt)
        cleaned = _strip_json_fences(raw_response)
        parsed = json.loads(cleaned)
        llm_chunks = parsed.get("chunks", [])

        baseline_specs = {c.step: c for c in self.compiler.compile_chunks()}
        result: list[SyntheticChunkSpec] = []
        for item in llm_chunks:
            step = int(item["step"])
            base = baseline_specs[step]
            result.append(
                SyntheticChunkSpec(
                    step=step,
                    scene_id=item.get("scene_id", base.scene_id),
                    chapter_id=item.get("chapter_id", base.chapter_id),
                    location_id=item.get("location_id", base.location_id),
                    story_time=item.get("story_time", base.story_time),
                    text=str(item["text"]),
                    expected_room_occupants=base.expected_room_occupants,
                    expected_moods=base.expected_moods,
                    expected_prior_moods=base.expected_prior_moods,
                    expected_edges=base.expected_edges,
                    oracle_delta=base.oracle_delta,
                    expected_gate1_passed_speakers=base.expected_gate1_passed_speakers,
                    expected_gate1_rejected_count=base.expected_gate1_rejected_count,
                    voice_metadata_by_speaker=base.voice_metadata_by_speaker,
                    voice_quality_by_speaker=base.voice_quality_by_speaker,
                    boundary_tags=base.boundary_tags,
                )
            )
        return result


# ============================================================================
# 5. Unified E2E Runner & Scorecard Evaluator (Covered vs. Uncovered Report)
# ============================================================================


@dataclass
class BoundaryDiagnostic:
    tag: str
    step: int
    description: str
    passed_strict_ideal: bool
    observed_behavior: str


@dataclass
class E2ERunReport:
    mode: Literal["deterministic_oracle", "deterministic_heuristic", "live_llm_closed_loop"]
    total_chunks: int
    chunk_word_counts: list[int]
    gate1_accepted_total: int = 0
    gate1_rejected_total: int = 0
    gate2_accepted_total: int = 0
    gate2_rejected_total: int = 0
    covered_checks_passed: int = 0
    covered_checks_total: int = 0
    covered_failures: list[str] = field(default_factory=list)
    boundary_diagnostics: list[BoundaryDiagnostic] = field(default_factory=list)

    @property
    def covered_pass_rate(self) -> float:
        if self.covered_checks_total == 0:
            return 1.0
        return self.covered_checks_passed / self.covered_checks_total


class SyntheticStoryE2ERunner:
    """Seeds an engine from SynthStoryParams, processes small chunks end-to-end,
    and evaluates both Covered Mechanics and Uncovered/Boundary Diagnostics.
    """

    def __init__(
        self,
        params: SynthStoryParams,
        *,
        engine: CanonArcEngine | None = None,
    ) -> None:
        self.params = params
        self.engine = engine or CanonArcEngine(
            ":memory:",
            preset=InteractiveFictionPreset(include_background_edges=True),
        )
        self.gate_rejections: list[tuple[str, str, str]] = []

        class _GateAuditHook:
            def __init__(self, outer: SyntheticStoryE2ERunner) -> None:
                self.outer = outer

            def on_before_commit(self, *args: Any) -> None:
                pass

            def on_after_commit(self, *args: Any) -> None:
                pass

            def on_rollback(self, *args: Any) -> None:
                pass

            def on_branch(self, *args: Any) -> None:
                pass

            def on_gate_rejected(
                self, stage: Literal["gate1", "gate2"], text: str, reason: str, metadata: dict[str, Any]
            ) -> None:
                self.outer.gate_rejections.append((stage, text, reason))

        self.engine.add_hook(_GateAuditHook(self))

    def seed_world(self) -> None:
        """Seeds Step 0 entities, locations, artifacts, aliases, and initial edges from SynthStoryParams."""
        for loc in self.params.locations:
            self.engine.register_entity(
                EntityRecord(
                    entity_id=loc.entity_id,
                    kind="location",
                    canonical_name=loc.canonical_name,
                    base_attributes={"description": loc.description},
                ),
                step=0,
            )

        for art in self.params.artifacts:
            self.engine.register_entity(
                EntityRecord(
                    entity_id=art.entity_id,
                    kind="artifact",
                    canonical_name=art.canonical_name,
                    state={"holder_id": art.initial_holder_id},
                ),
                step=0,
            )

        for char in self.params.characters:
            self.engine.register_entity(
                EntityRecord(
                    entity_id=char.entity_id,
                    kind="character",
                    canonical_name=char.canonical_name,
                    base_attributes=dict(char.base_attributes),
                    state={"mood": char.initial_mood, "location_id": char.initial_location},
                    private_state=dict(char.private_state),
                ),
                aliases=char.aliases,
                step=0,
            )

        self.engine.commit_step(
            step=0,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id=e.source_id,
                        target_id=e.target_id,
                        relation_type=e.relation_type,
                        state=dict(e.state),
                    )
                    for e in self.params.initial_edges
                ]
            ),
        )

    def run_pipeline(
        self,
        chunks: Sequence[SyntheticChunkSpec],
        *,
        extractor: DeltaExtractor,
        mode: Literal["deterministic_oracle", "deterministic_heuristic", "live_llm_closed_loop"],
        use_sliding_window: bool = False,
    ) -> E2ERunReport:
        self.seed_world()
        self.engine.extractor = extractor

        word_counts = [len(re.findall(r"\w+", c.text)) for c in chunks]
        report = E2ERunReport(
            mode=mode,
            total_chunks=len(chunks),
            chunk_word_counts=word_counts,
        )

        def _check(condition: bool, description: str) -> None:
            report.covered_checks_total += 1
            if condition:
                report.covered_checks_passed += 1
            else:
                report.covered_failures.append(description)

        for idx, chunk in enumerate(chunks):
            before_rejections = len(self.gate_rejections)
            prev_text = chunks[idx - 1].text if (use_sliding_window and idx > 0) else ""
            next_text = (
                chunks[idx + 1].text if (use_sliding_window and idx + 1 < len(chunks)) else ""
            )

            # 1. Extract dialogue candidates deterministically (Tier 5 Gate 1)
            candidates: list[DialogueCandidate] = self.engine.extract_dialogue_candidates(
                chunk.text,
                step=chunk.step - 1,
                prev_text=prev_text,
                next_text=next_text,
            )
            gate1_rejected_now = sum(
                1 for stage, _, _ in self.gate_rejections[before_rejections:] if stage == "gate1"
            )
            report.gate1_accepted_total += len(candidates)
            report.gate1_rejected_total += gate1_rejected_now

            exemplars_to_stage: list[VoiceExemplar] = []
            for cand in candidates:
                if cand.speaker_id:
                    meta = chunk.voice_metadata_by_speaker.get(cand.speaker_id, {})
                    q_score = chunk.voice_quality_by_speaker.get(cand.speaker_id, 9.0)
                    exemplars_to_stage.append(
                        cand.to_voice_exemplar(
                            step=chunk.step,
                            metadata=meta,
                            quality_score=q_score,
                        )
                    )

            # 2. Execute transactional step (retrieve point-in-time context -> extract & commit)
            with self.engine.step_transaction(
                step=chunk.step,
                text=chunk.text,
                prev_text=prev_text,
                next_text=next_text,
                location_id=chunk.location_id,
                rerun_clean=True,
            ) as tx:
                tx.extract_and_commit(
                    chunk.text,
                    extractor=extractor,
                    exemplars=exemplars_to_stage,
                )

            gate2_rejected_now = sum(
                1 for stage, _, _ in self.gate_rejections[before_rejections:] if stage == "gate2"
            )
            report.gate2_rejected_total += gate2_rejected_now
            report.gate2_accepted_total += max(0, len(exemplars_to_stage) - gate2_rejected_now)

            # 3. Verify Covered Mechanics at key steps (when using oracle or LLM closed-loop)
            if mode in ("deterministic_oracle", "live_llm_closed_loop"):
                self._verify_step_against_synth_params(
                    chunk, candidates, _check, report, prev_text=prev_text
                )
            else:
                self._record_boundary_diagnostics(chunk, report, prev_text=prev_text)

        # 4. Post-Story Covered Mechanics: Consolidation, Branching & Rollback
        if mode in ("deterministic_oracle", "live_llm_closed_loop"):
            self._verify_consolidation_and_time_travel(_check)

        return report

    def _verify_step_against_synth_params(
        self,
        chunk: SyntheticChunkSpec,
        candidates: Sequence[DialogueCandidate],
        _check: Callable[[bool, str], None],
        report: E2ERunReport,
        *,
        prev_text: str = "",
    ) -> None:
        step = chunk.step

        # A. Verify Gate 1 speaker attribution on small chunks
        if chunk.expected_gate1_passed_speakers:
            actual_speakers = [c.speaker_id for c in candidates if c.speaker_id]
            _check(
                actual_speakers == chunk.expected_gate1_passed_speakers,
                f"Step {step}: Expected Gate 1 speakers {chunk.expected_gate1_passed_speakers}, got {actual_speakers}",
            )

        # B. Verify Room Occupancy via location_id co-presence
        for room_id, expected_occupants in chunk.expected_room_occupants.items():
            room_ctx = self.engine.retrieve_context(
                step=step,
                location_id=room_id,
                inclusive=True,
                fallback_on_empty_mentions="none",
            )
            actual_chars_in_room = {
                e.entity_id for e in room_ctx.entities if e.kind == "character"
            }
            _check(
                actual_chars_in_room == expected_occupants,
                f"Step {step} room '{room_id}': expected {expected_occupants}, got {actual_chars_in_room}",
            )

        # C. Verify Character Moods against SynthParams
        all_entities = {
            e.entity_id: e
            for e in self.engine.store.get_all_active_entities(self.engine.active_timeline, step)
        }
        for cid, exp_mood in chunk.expected_moods.items():
            actual_mood = all_entities[cid].state.get("mood")
            _check(
                actual_mood == exp_mood,
                f"Step {step} entity '{cid}' mood: expected '{exp_mood}', got '{actual_mood}'",
            )

        # D. Step-specific deep Tier 1-6 checks
        if step == 2:
            # Disguise check: Marcus is fooled by "the hooded courier", Elena pierces it!
            ctx_marcus = self.engine.retrieve_context(
                step=2,
                text="The hooded courier waited near the table.",
                pov_entity_ids=["char_marcus"],
                inclusive=True,
            )
            ctx_elena = self.engine.retrieve_context(
                step=2,
                text="The hooded courier waited near the table.",
                pov_entity_ids=["char_elena"],
                inclusive=True,
            )
            marcus_seen_ids = {e.entity_id for e in ctx_marcus.entities}
            elena_seen_ids = {e.entity_id for e in ctx_elena.entities}
            _check(
                "masked:char_lyra" in marcus_seen_ids and "char_lyra" not in marcus_seen_ids,
                f"Step 2: Marcus should see masked:char_lyra, got {marcus_seen_ids}",
            )
            _check(
                "char_lyra" in elena_seen_ids,
                f"Step 2: Elena should pierce disguise to char_lyra, got {elena_seen_ids}",
            )

        elif step == 4:
            # Marcus was on the balcony during Step 4 -> Marcus MUST NOT see ep_step4_seal_accusation!
            ctx_marcus_ep = self.engine.retrieve_context(
                step=4,
                text="Elena and Julian argued about the Obsidian Signet.",
                pov_entity_ids=["char_marcus"],
                inclusive=True,
            )
            marcus_ep_ids = {ep.episode_id for ep in ctx_marcus_ep.episodes}
            _check(
                "ep_step4_seal_accusation" not in marcus_ep_ids,
                "Step 4: Absent character Marcus leaked secret episode ep_step4_seal_accusation!",
            )
            # Voice Bank register shift check: Elena -> Julian is now 'ti' / 'furious'
            ctx_elena_voice = self.engine.retrieve_context(
                step=4,
                text="Elena addressed Julian in the chamber.",
                mentions=["Elena", "Julian"],
                inclusive=True,
            )
            elena_ex = [ex for ex in ctx_elena_voice.voice_exemplars if ex.speaker_id == "char_elena"]
            _check(
                bool(elena_ex) and elena_ex[0].metadata.get("register") == "ti",
                f"Step 4: Expected Elena's 'ti' voice exemplar toward Julian, got {[e.metadata for e in elena_ex]}",
            )

        elif step == 7:
            # Flashback protection check: Elena's mood must still be 'furious', NOT 'naive_trusting'
            _check(
                all_entities["char_elena"].state.get("mood") == "furious",
                f"Step 7: Flashback overwrote Elena's present mood to {all_entities['char_elena'].state.get('mood')}",
            )

        elif step == 8:
            # Disguise unmasking check: Julian now pierces "the hooded courier", Marcus still does not!
            ctx_julian = self.engine.retrieve_context(
                step=8,
                text="The hooded courier spoke in the Archive Vault.",
                pov_entity_ids=["char_julian"],
                inclusive=True,
            )
            _check(
                any(e.entity_id == "char_lyra" and e.disguised_as == "Courier" for e in ctx_julian.entities),
                "Step 8: Julian should now pierce Lyra's disguise after unmasking in the vault.",
            )

        elif step == 9:
            # Deceived POV check: Marcus believes northern raiders stole the signet!
            ctx_marcus_deceived = self.engine.retrieve_context(
                step=9,
                text="Northern raiders broke into the annex and stole the Obsidian Signet.",
                pov_entity_ids=["char_marcus"],
                inclusive=True,
            )
            ep4_for_marcus = next(
                (ep for ep in ctx_marcus_deceived.episodes if ep.episode_id == "ep_step4_seal_accusation"),
                None,
            )
            _check(
                ep4_for_marcus is not None and "Northern raiders" in ep4_for_marcus.summary,
                f"Step 9: Marcus POV should see deceived summary for ep_step4_seal_accusation, got {ep4_for_marcus}",
            )

        elif step == 10:
            # Truth told check: Marcus now sees the true summary of ep_step4_seal_accusation
            ctx_marcus_truth = self.engine.retrieve_context(
                step=10,
                text="Elena accused Julian of stealing the Obsidian Signet.",
                pov_entity_ids=["char_marcus"],
                inclusive=True,
            )
            ep4_truth = next(
                (ep for ep in ctx_marcus_truth.episodes if ep.episode_id == "ep_step4_seal_accusation"),
                None,
            )
            _check(
                ep4_truth is not None and "Elena accused Julian" in ep4_truth.summary,
                "Step 10: Marcus should see ground-truth episode summary after being informed ('told').",
            )
            # Forgotten check: Julian no longer recalls ep_step7_siege_memory
            ctx_julian_mem = self.engine.retrieve_context(
                step=10,
                text="Ten years ago during the winter siege treaty signing.",
                pov_entity_ids=["char_julian"],
                inclusive=True,
            )
            _check(
                all(ep.episode_id != "ep_step7_siege_memory" for ep in ctx_julian_mem.episodes),
                "Step 10: Julian recalled ep_step7_siege_memory despite epistemic='forgotten'!",
            )

        # Record uncovered/boundary diagnostics for this step
        self._record_boundary_diagnostics(chunk, report, prev_text=prev_text)

    def _record_boundary_diagnostics(
        self,
        chunk: SyntheticChunkSpec,
        report: E2ERunReport,
        *,
        prev_text: str = "",
    ) -> None:
        step = chunk.step
        tl = self.engine.active_timeline

        for tag in chunk.boundary_tags:
            if tag == "implicit_room_witness_unmentioned_in_text":
                # In Chunk 4, Lyra is in loc_council, but her name/alias is not in Chunk 4's text.
                # If we query retrieve_context WITH location_id="loc_council", is she present?
                # And if we query WITHOUT location_id (text-only), is she present?
                ctx_with_loc = self.engine.retrieve_context(
                    step=step, text=chunk.text, location_id="loc_council", inclusive=True
                )
                ctx_text_only = self.engine.retrieve_context(
                    step=step, text=chunk.text, inclusive=True
                )
                in_with_loc = any(e.entity_id == "char_lyra" for e in ctx_with_loc.entities)
                in_text_only = any(e.entity_id == "char_lyra" for e in ctx_text_only.entities)
                ep4_list = self.engine.query_episodes(cutoff_step=step)
                ep4 = next((e for e in ep4_list if e.episode_id == "ep_step4_seal_accusation"), None)
                lyra_witnessed = bool(ep4 and "char_lyra" in ep4.known_by)

                report.boundary_diagnostics.append(
                    BoundaryDiagnostic(
                        tag=tag,
                        step=step,
                        description=(
                            "Co-present character (Lyra) is in the room (loc_council) but not named in the chunk prose."
                        ),
                        passed_strict_ideal=(in_text_only and lyra_witnessed),
                        observed_behavior=(
                            f"present_with_location_id={in_with_loc}, "
                            f"present_from_text_alone={in_text_only}, "
                            f"episode_witnessed={lyra_witnessed}"
                        ),
                    )
                )

            elif tag == "cross_chunk_pronoun_coreference":
                # Chunk 5 uses only "she", "her", "he", "his" with no character names.
                ctx_pronoun = self.engine.retrieve_context(
                    step=step,
                    text=chunk.text,
                    prev_text=prev_text,
                    inclusive=True,
                    fallback_on_empty_mentions="none",
                )
                resolved_ids = {m.entity_id for m in ctx_pronoun.resolved_mentions}
                report.boundary_diagnostics.append(
                    BoundaryDiagnostic(
                        tag=tag,
                        step=step,
                        description=(
                            "Small chunk contains only pronouns ('she', 'he') referring to entities from Chunk 4."
                        ),
                        passed_strict_ideal=("char_elena" in resolved_ids and "char_julian" in resolved_ids),
                        observed_behavior=(
                            f"resolved_mentions_from_pronouns={sorted(resolved_ids)} "
                            "(DefaultLexicalResolver matches names/aliases only, not cross-chunk pronouns)"
                        ),
                    )
                )

            elif tag == "intra_scene_same_mood_repeat":
                # Chunk 4 set Elena mood='furious' (prior='calm' at step 0).
                # Chunk 5 re-emits mood='furious' because she is still angry in the next small chunk.
                elena_ent = next(
                    e
                    for e in self.engine.store.get_all_active_entities(tl, step)
                    if e.entity_id == "char_elena"
                )
                prior_mood = (elena_ent.prior_state or {}).get("mood")
                report.boundary_diagnostics.append(
                    BoundaryDiagnostic(
                        tag=tag,
                        step=step,
                        description=(
                            "Extractor re-emits mood='furious' in consecutive small chunks (Step 4 & Step 5)."
                        ),
                        passed_strict_ideal=(prior_mood == "calm"),
                        observed_behavior=(
                            f"current_mood={elena_ent.state.get('mood')!r}, "
                            f"prior_mood={prior_mood!r} at prior_step={elena_ent.prior_step}, "
                            f"revision_count={elena_ent.revision_count}"
                        ),
                    )
                )

            elif tag == "exclusive_inventory_transfer":
                # In Chunk 6, Julian hands item_signet to Lyra without explicit remove=True on Julian's edge.
                active_edges = self.engine.store.get_all_active_edges(tl, step)
                holders = [
                    e.source_id
                    for e in active_edges
                    if e.relation_type == "possesses" and e.target_id == "item_signet"
                ]
                report.boundary_diagnostics.append(
                    BoundaryDiagnostic(
                        tag=tag,
                        step=step,
                        description=(
                            "Artifact (item_signet) transferred from Julian to Lyra without explicit tombstone remove=True."
                        ),
                        passed_strict_ideal=(holders == ["char_lyra"]),
                        observed_behavior=(
                            f"active_possesses_holders={holders} "
                            "(Requires explicit EdgeDelta(remove=True) to enforce exclusive ownership)"
                        ),
                    )
                )

            elif tag == "sarcastic_surface_dialogue_vs_true_mood":
                # In Chunk 9, Elena says a sarcastic polite line while her true mood is 'furious'.
                elena_ent = next(
                    e
                    for e in self.engine.store.get_all_active_entities(tl, step)
                    if e.entity_id == "char_elena"
                )
                report.boundary_diagnostics.append(
                    BoundaryDiagnostic(
                        tag=tag,
                        step=step,
                        description=(
                            "Character speaks sarcastic praise while true mood is 'furious'."
                        ),
                        passed_strict_ideal=(elena_ent.state.get("mood") == "furious"),
                        observed_behavior=f"elena_mood_after_sarcasm={elena_ent.state.get('mood')!r}",
                    )
                )

    def _verify_consolidation_and_time_travel(
        self,
        _check: Callable[[bool, str], None],
    ) -> None:
        # 1. Consolidate low-salience Act 1 episodes (< step 4, salience <= 4.0)
        consolidated = self.engine.consolidate_episodes(
            step=11,
            before_step=4,
            max_salience=4.0,
            summary="Act 1 Opening: Elena briefed Marcus, the courier delivered dispatches, and Marcus stepped out.",
            chapter_id="ch_1",
        )
        _check(
            consolidated is not None and len(consolidated.consolidated_from) == 3,
            f"Step 11 consolidation: expected 3 low-salience episodes compacted, got {consolidated}",
        )

        # 2. Branch at Step 8 into an alternate timeline ("julian_confesses_route")
        branch_session = self.engine.session("main").branch(
            at_step=8,
            name="julian_confesses_route",
        )
        branch_session.commit_step(
            step=9,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_julian",
                        state={"mood": "repentant"},
                        reason="Confessed freely in the Archive Vault",
                    )
                ]
            ),
        )
        diff = self.engine.diff_timelines("main", "julian_confesses_route", at_step=10)
        changed_keys = {item["subject_key"] for item in diff.get("changed", [])}
        _check(
            "char_julian" in changed_keys,
            f"Timeline diff at step 10 should show char_julian diverging, got {changed_keys}.",
        )

        # 3. Rollback the branch to Step 8 and verify clean state restoration
        branch_session.rollback_to(8)
        ctx_branch_rb = branch_session.retrieve_context(step=8, mentions=["Julian"], inclusive=True)
        _check(
            ctx_branch_rb.entities[0].state.get("mood") == "shocked",
            "Rollback of 'julian_confesses_route' to step 8 should restore Julian's mood to 'shocked'.",
        )
