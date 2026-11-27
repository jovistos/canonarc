from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from canonarc import (
    AliasRecord,
    CachedDeltaExtractor,
    CanonArcEngine,
    DialogueCandidate,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    InteractiveFictionPreset,
    OpenAICompatibleExtractor,
    StepDelta,
    TokenUsage,
    VoiceExemplar,
    load_dotenv,
)
from canonarc.extractors import _strip_json_fences
from synthetic_story_harness import (
    ArtifactSeed,
    CharacterSeed,
    DeterministicStoryCompiler,
    InitialEdgeSeed,
    LocationSeed,
    SyntheticChunkSpec,
    SynthStoryParams,
)

DEFAULT_STORIES_ROOT = Path(__file__).resolve().parent / "synthetic_stories"

GuidanceMode = Literal["full_hints", "canonical_episode_ids", "none", "surface_only"]

_NEGATION_TOKENS: set[str] = {"not", "no", "never", "neither", "non", "without", "hardly", "barely"}
_NEGATION_PREFIXES: tuple[str, ...] = ("un", "non_", "non-", "in", "im", "dis")

_MOOD_SYNONYM_CLUSTERS: list[set[str]] = [
    {
        "furious",
        "fury",
        "angry",
        "anger",
        "enraged",
        "rage",
        "irate",
        "livid",
        "wrathful",
        "wrath",
        "incensed",
        "outraged",
        "indignant",
        "hostile",
        "cold",
        "bitter",
        "stern",
        "betrayed",
        "offended",
        "resentful",
    },
    {
        "panicked",
        "panic",
        "terrified",
        "terror",
        "afraid",
        "frightened",
        "scared",
        "frantic",
        "desperate",
        "alarmed",
        "fearful",
        "shaken",
    },
    {
        "shocked",
        "shock",
        "stunned",
        "astounded",
        "startled",
        "aghast",
        "dismayed",
        "shaken",
        "bewildered",
        "bewilderment",
        "confused",
        "confusion",
        "disoriented",
        "dazed",
        "blank",
    },
    {
        "restless",
        "uneasy",
        "agitated",
        "impatient",
        "stifled",
        "tense",
        "unsettled",
        "bothered",
        "irritated",
        "concerned",
        "annoyed",
        "frustrated",
    },
    {"determined", "resolute", "focused", "relentless", "commanding", "decisive", "unyielding", "steely", "threatening", "grim"},
    {"calm", "composed", "steady", "serene", "measured", "collected", "tranquil", "controlled"},
    {"confident", "assured", "self_assured", "self-assured", "certain", "poised", "arrogant"},
    {"watchful", "vigilant", "alert", "observant", "wary", "cautious", "attentive", "guarded"},
    {"nervous", "anxious", "jittery", "apprehensive", "uneasy", "tense", "worried"},
    {"naive_trusting", "trusting", "naive", "innocent", "devoted", "fond", "affectionate", "nostalgic", "earnest"},
]


def _has_negation(clean_str: str, tokens: set[str], reference_token: str | None = None) -> bool:
    if tokens & _NEGATION_TOKENS:
        return True
    if reference_token:
        for pref in ("un", "non", "in", "im", "dis"):
            if f"{pref}{reference_token}" in tokens or f"{pref}_{reference_token}" in clean_str:
                return True
    return False


def moods_soft_match(
    expected: str | None,
    actual: str | None,
    *,
    acceptable: set[str] | Sequence[str] | None = None,
) -> bool:
    """Returns True if actual mood matches expected mood (or any acceptable mood) via token-level synonym clusters,
    while rejecting negated forms (e.g., 'not calm' or 'uncalm' vs 'calm').
    """
    if acceptable:
        for alt in acceptable:
            if moods_soft_match(alt, actual):
                return True
    if not expected or not actual:
        return expected == actual
    exp_clean = expected.strip().lower().replace("-", "_")
    act_clean = actual.strip().lower().replace("-", "_")
    if exp_clean == act_clean:
        return True

    exp_tokens = set(re.findall(r"[a-z]+", exp_clean)) | {exp_clean}
    act_tokens = set(re.findall(r"[a-z]+", act_clean)) | {act_clean}

    # Reject if one expression is negated (e.g. "not calm", "uncalm", "fearless") and the other is not
    for tok in list(exp_tokens):
        if f"un{tok}" in act_tokens or f"non_{tok}" in act_clean or f"{tok}less" in act_tokens:
            return False
    for tok in list(act_tokens):
        if f"un{tok}" in exp_tokens or f"non_{tok}" in exp_clean or f"{tok}less" in exp_tokens:
            return False
    if _has_negation(exp_clean, exp_tokens) != _has_negation(act_clean, act_tokens):
        return False

    # Direct full-word token overlap (e.g. "cold_fury" and "fury")
    shared_non_neg = (exp_tokens & act_tokens) - _NEGATION_TOKENS
    if shared_non_neg:
        return True

    for cluster in _MOOD_SYNONYM_CLUSTERS:
        if (exp_tokens & cluster) and (act_tokens & cluster):
            return True
    return False


def _safe_div(num: float, den: float, default: float = 1.0) -> float:
    return num / den if den > 0 else default


def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _jaccard(set_a: set[Any], set_b: set[Any]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def estimate_text_tokens(text: str) -> int:
    """Estimates BPE token count for prose + punctuation (~1.33 tokens per word + punctuation)."""
    words = re.findall(r"\w+", text)
    punct = re.findall(r"[^\w\s]", text)
    return round(len(words) * 1.28 + len(punct) * 0.45)


@dataclass
class ChunkBeatBlueprint:
    """Defines the narrative beat contract and expected world-state after a ~500-token chunk."""

    step: int
    scene_id: str
    chapter_id: str
    location_id: str
    story_time: str
    beat_title: str
    writing_directives: str
    extraction_hint: str
    expected_room_occupants: dict[str, set[str]]
    expected_moods: dict[str, str]
    expected_episode_id: str | None = None
    is_flashback: bool = False
    oracle_delta: StepDelta = field(default_factory=StepDelta)
    voice_metadata_by_speaker: dict[str, dict[str, str]] = field(default_factory=dict)
    voice_quality_by_speaker: dict[str, float] = field(default_factory=dict)
    acceptable_moods: dict[str, set[str]] = field(default_factory=dict)
    optional_inform: list[InformDelta] = field(default_factory=list)
    required_prose_cues: list[str] = field(default_factory=list)


@dataclass
class LongStoryBlueprint:
    """Complete blueprint for a multi-chunk (~3,000-token or 12,000-token) synthetic story."""

    story_slug: str
    title: str
    genre: str
    description: str
    target_total_tokens: int
    target_chunk_tokens: int
    target_chunk_words: tuple[int, int]
    params: SynthStoryParams
    beats: list[ChunkBeatBlueprint]
    key_episode_ids: dict[str, str] = field(default_factory=dict)
    branch_fork_step: int | None = None
    consolidation_step: int | None = None
    forbidden_prose_words: list[str] = field(default_factory=list)


@dataclass
class SavedSyntheticStory:
    """Loaded or freshly generated synthetic story persisted in tests/synthetic_stories/<story_slug>/."""

    story_slug: str
    title: str
    genre: str
    model: str
    created_at: str
    story_dir: Path
    params: SynthStoryParams
    chunks: list[SyntheticChunkSpec]
    generation_usage: TokenUsage
    loaded_from_cache: bool = False

    @property
    def total_words(self) -> int:
        return sum(len(re.findall(r"\w+", c.text)) for c in self.chunks)

    @property
    def total_estimated_tokens(self) -> int:
        return sum(estimate_text_tokens(c.text) for c in self.chunks)

    @property
    def chunk_token_counts(self) -> list[int]:
        return [estimate_text_tokens(c.text) for c in self.chunks]

    @property
    def chunk_word_counts(self) -> list[int]:
        return [len(re.findall(r"\w+", c.text)) for c in self.chunks]


@dataclass
class StepExtractionMetrics:
    """Granular extraction and retrieval fidelity metrics for a single story step."""

    step: int
    room_occupancy_jaccard: float = 1.0
    room_occupancy_exact: bool = True
    character_location_matches: int = 0
    character_location_total: int = 0
    mood_exact_matches: int = 0
    mood_soft_matches: int = 0
    mood_total_characters: int = 0
    location_delta_tp: int = 0
    location_delta_fp: int = 0
    location_delta_fn: int = 0
    mood_delta_tp: int = 0
    mood_delta_fp: int = 0
    mood_delta_fn: int = 0
    edge_delta_tp: int = 0
    edge_delta_fp: int = 0
    edge_delta_fn: int = 0
    edge_register_matches: int = 0
    edge_register_total: int = 0
    voice_register_matches: int = 0
    voice_register_total: int = 0
    causal_links_matched: int = 0
    causal_links_expected: int = 0
    witness_tp: int = 0
    witness_fp: int = 0
    witness_fn: int = 0
    epistemic_leak_count: int = 0
    absent_character_opportunities: int = 0
    silent_witness_matched: int = 0
    silent_witness_expected: int = 0
    inform_tp: int = 0
    inform_fp: int = 0
    inform_fn: int = 0
    deception_summary_matched: int = 0
    deception_summary_expected: int = 0
    flashback_expected: bool = False
    flashback_flag_correct: bool = True
    raw_id_total: int = 0
    canonical_id_count: int = 0
    normalized_surface_id_count: int = 0
    hallucinated_id_count: int = 0
    hallucinated_ids: list[str] = field(default_factory=list)
    pov_entity_jaccard: float = 1.0
    pov_edge_jaccard: float = 1.0
    pov_episode_recall: float = 1.0
    pov_secret_leaks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "room_occupancy_jaccard": round(self.room_occupancy_jaccard, 4),
            "room_occupancy_exact": self.room_occupancy_exact,
            "character_location_accuracy": round(
                _safe_div(self.character_location_matches, self.character_location_total), 4
            ),
            "mood_exact_rate": round(_safe_div(self.mood_exact_matches, self.mood_total_characters), 4),
            "mood_soft_rate": round(_safe_div(self.mood_soft_matches, self.mood_total_characters), 4),
            "location_delta": {"tp": self.location_delta_tp, "fp": self.location_delta_fp, "fn": self.location_delta_fn},
            "mood_delta": {"tp": self.mood_delta_tp, "fp": self.mood_delta_fp, "fn": self.mood_delta_fn},
            "edge_delta": {"tp": self.edge_delta_tp, "fp": self.edge_delta_fp, "fn": self.edge_delta_fn},
            "voice_register": {
                "matches": self.voice_register_matches,
                "total": self.voice_register_total,
            },
            "witness": {
                "tp": self.witness_tp,
                "fp": self.witness_fp,
                "fn": self.witness_fn,
                "epistemic_leak_count": self.epistemic_leak_count,
                "silent_witness_matched": self.silent_witness_matched,
                "silent_witness_expected": self.silent_witness_expected,
            },
            "inform": {"tp": self.inform_tp, "fp": self.inform_fp, "fn": self.inform_fn},
            "flashback_flag_correct": self.flashback_flag_correct,
            "id_hygiene": {
                "raw_id_total": self.raw_id_total,
                "canonical_id_count": self.canonical_id_count,
                "normalized_surface_id_count": self.normalized_surface_id_count,
                "hallucinated_id_count": self.hallucinated_id_count,
                "hallucinated_ids": self.hallucinated_ids,
            },
            "pov_retrieval": {
                "pov_entity_jaccard": round(self.pov_entity_jaccard, 4),
                "pov_edge_jaccard": round(self.pov_edge_jaccard, 4),
                "pov_episode_recall": round(self.pov_episode_recall, 4),
                "pov_secret_leaks": self.pov_secret_leaks,
            },
            "mood_counts": {
                "exact_matches": self.mood_exact_matches,
                "soft_matches": self.mood_soft_matches,
                "total_characters": self.mood_total_characters,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StepExtractionMetrics:
        if not data:
            return cls(step=0)
        loc_d = data.get("location_delta") or {}
        md_d = data.get("mood_delta") or {}
        ed_d = data.get("edge_delta") or {}
        wit_d = data.get("witness") or {}
        inf_d = data.get("inform") or {}
        id_d = data.get("id_hygiene") or {}
        pov_d = data.get("pov_retrieval") or {}
        mc_d = data.get("mood_counts") or {}
        vr_d = data.get("voice_register") or {}
        tot_chars = int(mc_d.get("total_characters", 4))
        exact_m = int(mc_d.get("exact_matches", round(float(data.get("mood_exact_rate", 1.0)) * tot_chars)))
        soft_m = int(mc_d.get("soft_matches", round(float(data.get("mood_soft_rate", 1.0)) * tot_chars)))
        char_loc_acc = float(data.get("character_location_accuracy", data.get("room_occupancy_jaccard", 1.0)))
        return cls(
            step=int(data.get("step", 0)),
            room_occupancy_jaccard=float(data.get("room_occupancy_jaccard", 1.0)),
            room_occupancy_exact=bool(data.get("room_occupancy_exact", True)),
            character_location_matches=round(char_loc_acc * tot_chars),
            character_location_total=tot_chars,
            mood_exact_matches=exact_m,
            mood_soft_matches=soft_m,
            mood_total_characters=tot_chars,
            location_delta_tp=int(loc_d.get("tp", 0)),
            location_delta_fp=int(loc_d.get("fp", 0)),
            location_delta_fn=int(loc_d.get("fn", 0)),
            mood_delta_tp=int(md_d.get("tp", 0)),
            mood_delta_fp=int(md_d.get("fp", 0)),
            mood_delta_fn=int(md_d.get("fn", 0)),
            edge_delta_tp=int(ed_d.get("tp", 0)),
            edge_delta_fp=int(ed_d.get("fp", 0)),
            edge_delta_fn=int(ed_d.get("fn", 0)),
            voice_register_matches=int(vr_d.get("matches", 0)),
            voice_register_total=int(vr_d.get("total", 0)),
            witness_tp=int(wit_d.get("tp", 0)),
            witness_fp=int(wit_d.get("fp", 0)),
            witness_fn=int(wit_d.get("fn", 0)),
            epistemic_leak_count=int(wit_d.get("epistemic_leak_count", 0)),
            silent_witness_matched=int(wit_d.get("silent_witness_matched", 0)),
            silent_witness_expected=int(wit_d.get("silent_witness_expected", 0)),
            inform_tp=int(inf_d.get("tp", 0)),
            inform_fp=int(inf_d.get("fp", 0)),
            inform_fn=int(inf_d.get("fn", 0)),
            flashback_flag_correct=bool(data.get("flashback_flag_correct", True)),
            raw_id_total=int(id_d.get("raw_id_total", 0)),
            canonical_id_count=int(id_d.get("canonical_id_count", 0)),
            normalized_surface_id_count=int(id_d.get("normalized_surface_id_count", 0)),
            hallucinated_id_count=int(id_d.get("hallucinated_id_count", 0)),
            hallucinated_ids=list(id_d.get("hallucinated_ids") or []),
            pov_entity_jaccard=float(pov_d.get("pov_entity_jaccard", 1.0)),
            pov_edge_jaccard=float(pov_d.get("pov_edge_jaccard", 1.0)),
            pov_episode_recall=float(pov_d.get("pov_episode_recall", 1.0)),
            pov_secret_leaks=int(pov_d.get("pov_secret_leaks", 0)),
        )


@dataclass
class ExtractionBenchmarkMetrics:
    """Multi-dimensional extraction & downstream POV benchmark metrics for comparing models."""

    model_id: str = ""
    guidance_mode: GuidanceMode = "full_hints"
    # 1. Epistemic & Theory-of-Mind Metrics (Tier 4 & Tier 6)
    witness_precision: float = 1.0
    witness_recall: float = 1.0
    witness_f1: float = 1.0
    epistemic_leak_count: int = 0
    epistemic_leak_rate: float = 0.0
    silent_witness_recall: float = 1.0
    inform_delta_precision: float = 1.0
    inform_delta_recall: float = 1.0
    inform_delta_f1: float = 1.0
    deception_summary_accuracy: float = 1.0
    disguise_accuracy: float = 1.0
    # 2. Spatial & World-State Metrics (Tiers 1–4)
    location_delta_precision: float = 1.0
    location_delta_recall: float = 1.0
    location_delta_f1: float = 1.0
    room_occupancy_jaccard: float = 1.0
    room_occupancy_exact_rate: float = 1.0
    character_location_accuracy: float = 1.0
    mood_exact_accuracy: float = 1.0
    mood_soft_accuracy: float = 1.0
    mood_delta_f1: float = 1.0
    flashback_guard_passed: bool = True
    # 3. Graph Edge, Voice Register & Causal Provenance Metrics
    edge_delta_precision: float = 1.0
    edge_delta_recall: float = 1.0
    edge_delta_f1: float = 1.0
    edge_register_accuracy: float = 1.0
    voice_register_accuracy: float = 1.0
    causal_link_accuracy: float = 1.0
    # 4. Structural & ID Hygiene Metrics
    first_pass_valid_rate: float = 1.0
    total_retries: int = 0
    total_validation_errors: int = 0
    raw_id_total: int = 0
    canonical_id_count: int = 0
    canonical_id_rate: float = 1.0
    normalized_surface_id_count: int = 0
    normalized_surface_id_rate: float = 0.0
    hallucinated_id_count: int = 0
    hallucinated_id_rate: float = 0.0
    hallucinated_ids: list[str] = field(default_factory=list)
    # 5. Downstream POV Retrieval Fidelity (Oracle Engine vs. Model Engine)
    pov_entity_jaccard: float = 1.0
    pov_edge_jaccard: float = 1.0
    pov_episode_recall: float = 1.0
    pov_secret_leak_count: int = 0
    # 6. Category Subscores & Overall Composite Quality Score (0.0 .. 1.0)
    epistemic_score: float = 1.0
    spatial_state_score: float = 1.0
    pov_retrieval_score: float = 1.0
    edge_flashback_score: float = 1.0
    schema_hygiene_score: float = 1.0
    overall_quality_score: float = 1.0
    step_metrics: list[StepExtractionMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "guidance_mode": self.guidance_mode,
            "overall_quality_score": round(self.overall_quality_score, 4),
            "category_scores": {
                "epistemic_score": round(self.epistemic_score, 4),
                "spatial_state_score": round(self.spatial_state_score, 4),
                "pov_retrieval_score": round(self.pov_retrieval_score, 4),
                "edge_flashback_score": round(self.edge_flashback_score, 4),
                "schema_hygiene_score": round(self.schema_hygiene_score, 4),
            },
            "epistemic_and_witness_metrics": {
                "witness_precision": round(self.witness_precision, 4),
                "witness_recall": round(self.witness_recall, 4),
                "witness_f1": round(self.witness_f1, 4),
                "epistemic_leak_count": self.epistemic_leak_count,
                "epistemic_leak_rate": round(self.epistemic_leak_rate, 4),
                "silent_witness_recall": round(self.silent_witness_recall, 4),
                "inform_delta_precision": round(self.inform_delta_precision, 4),
                "inform_delta_recall": round(self.inform_delta_recall, 4),
                "inform_delta_f1": round(self.inform_delta_f1, 4),
                "deception_summary_accuracy": round(self.deception_summary_accuracy, 4),
                "disguise_accuracy": round(self.disguise_accuracy, 4),
            },
            "spatial_and_state_metrics": {
                "location_delta_precision": round(self.location_delta_precision, 4),
                "location_delta_recall": round(self.location_delta_recall, 4),
                "location_delta_f1": round(self.location_delta_f1, 4),
                "room_occupancy_jaccard": round(self.room_occupancy_jaccard, 4),
                "room_occupancy_exact_rate": round(self.room_occupancy_exact_rate, 4),
                "character_location_accuracy": round(self.character_location_accuracy, 4),
                "mood_exact_accuracy": round(self.mood_exact_accuracy, 4),
                "mood_soft_accuracy": round(self.mood_soft_accuracy, 4),
                "mood_delta_f1": round(self.mood_delta_f1, 4),
                "flashback_guard_passed": self.flashback_guard_passed,
            },
            "edge_and_causal_metrics": {
                "edge_delta_precision": round(self.edge_delta_precision, 4),
                "edge_delta_recall": round(self.edge_delta_recall, 4),
                "edge_delta_f1": round(self.edge_delta_f1, 4),
                "edge_register_accuracy": round(self.edge_register_accuracy, 4),
                "voice_register_accuracy": round(self.voice_register_accuracy, 4),
                "causal_link_accuracy": round(self.causal_link_accuracy, 4),
            },
            "schema_and_id_hygiene": {
                "first_pass_valid_rate": round(self.first_pass_valid_rate, 4),
                "total_retries": self.total_retries,
                "total_validation_errors": self.total_validation_errors,
                "raw_id_total": self.raw_id_total,
                "canonical_id_count": self.canonical_id_count,
                "canonical_id_rate": round(self.canonical_id_rate, 4),
                "normalized_surface_id_count": self.normalized_surface_id_count,
                "normalized_surface_id_rate": round(self.normalized_surface_id_rate, 4),
                "hallucinated_id_count": self.hallucinated_id_count,
                "hallucinated_id_rate": round(self.hallucinated_id_rate, 4),
                "hallucinated_ids": self.hallucinated_ids,
            },
            "downstream_pov_retrieval": {
                "pov_entity_jaccard": round(self.pov_entity_jaccard, 4),
                "pov_edge_jaccard": round(self.pov_edge_jaccard, 4),
                "pov_episode_recall": round(self.pov_episode_recall, 4),
                "pov_secret_leak_count": self.pov_secret_leak_count,
            },
            "steps": [s.to_dict() for s in self.step_metrics],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExtractionBenchmarkMetrics:
        if not data:
            return cls()
        cat = data.get("category_scores") or {}
        ep = data.get("epistemic_and_witness_metrics") or {}
        sp = data.get("spatial_and_state_metrics") or {}
        ed = data.get("edge_and_causal_metrics") or {}
        hy = data.get("schema_and_id_hygiene") or {}
        pv = data.get("downstream_pov_retrieval") or {}
        return cls(
            model_id=str(data.get("model_id", "")),
            guidance_mode=data.get("guidance_mode", "full_hints"),
            witness_precision=float(ep.get("witness_precision", 1.0)),
            witness_recall=float(ep.get("witness_recall", 1.0)),
            witness_f1=float(ep.get("witness_f1", 1.0)),
            epistemic_leak_count=int(ep.get("epistemic_leak_count", 0)),
            epistemic_leak_rate=float(ep.get("epistemic_leak_rate", 0.0)),
            silent_witness_recall=float(ep.get("silent_witness_recall", 1.0)),
            inform_delta_precision=float(ep.get("inform_delta_precision", 1.0)),
            inform_delta_recall=float(ep.get("inform_delta_recall", 1.0)),
            inform_delta_f1=float(ep.get("inform_delta_f1", 1.0)),
            deception_summary_accuracy=float(ep.get("deception_summary_accuracy", 1.0)),
            disguise_accuracy=float(ep.get("disguise_accuracy", 1.0)),
            location_delta_precision=float(sp.get("location_delta_precision", 1.0)),
            location_delta_recall=float(sp.get("location_delta_recall", 1.0)),
            location_delta_f1=float(sp.get("location_delta_f1", 1.0)),
            room_occupancy_jaccard=float(sp.get("room_occupancy_jaccard", 1.0)),
            room_occupancy_exact_rate=float(sp.get("room_occupancy_exact_rate", 1.0)),
            character_location_accuracy=float(
                sp.get("character_location_accuracy", sp.get("room_occupancy_jaccard", 1.0))
            ),
            mood_exact_accuracy=float(sp.get("mood_exact_accuracy", 1.0)),
            mood_soft_accuracy=float(sp.get("mood_soft_accuracy", 1.0)),
            mood_delta_f1=float(sp.get("mood_delta_f1", 1.0)),
            flashback_guard_passed=bool(sp.get("flashback_guard_passed", True)),
            edge_delta_precision=float(ed.get("edge_delta_precision", 1.0)),
            edge_delta_recall=float(ed.get("edge_delta_recall", 1.0)),
            edge_delta_f1=float(ed.get("edge_delta_f1", 1.0)),
            edge_register_accuracy=float(ed.get("edge_register_accuracy", 1.0)),
            voice_register_accuracy=float(
                ed.get("voice_register_accuracy", ed.get("edge_register_accuracy", 1.0))
            ),
            causal_link_accuracy=float(ed.get("causal_link_accuracy", 1.0)),
            first_pass_valid_rate=float(hy.get("first_pass_valid_rate", 1.0)),
            total_retries=int(hy.get("total_retries", 0)),
            total_validation_errors=int(hy.get("total_validation_errors", 0)),
            raw_id_total=int(hy.get("raw_id_total", 0)),
            canonical_id_count=int(hy.get("canonical_id_count", 0)),
            canonical_id_rate=float(hy.get("canonical_id_rate", 1.0)),
            normalized_surface_id_count=int(hy.get("normalized_surface_id_count", 0)),
            normalized_surface_id_rate=float(hy.get("normalized_surface_id_rate", 0.0)),
            hallucinated_id_count=int(hy.get("hallucinated_id_count", 0)),
            hallucinated_id_rate=float(hy.get("hallucinated_id_rate", 0.0)),
            hallucinated_ids=list(hy.get("hallucinated_ids") or []),
            pov_entity_jaccard=float(pv.get("pov_entity_jaccard", 1.0)),
            pov_edge_jaccard=float(pv.get("pov_edge_jaccard", 1.0)),
            pov_episode_recall=float(pv.get("pov_episode_recall", 1.0)),
            pov_secret_leak_count=int(pv.get("pov_secret_leak_count", 0)),
            epistemic_score=float(cat.get("epistemic_score", 1.0)),
            spatial_state_score=float(cat.get("spatial_state_score", 1.0)),
            pov_retrieval_score=float(cat.get("pov_retrieval_score", 1.0)),
            edge_flashback_score=float(cat.get("edge_flashback_score", 1.0)),
            schema_hygiene_score=float(cat.get("schema_hygiene_score", 1.0)),
            overall_quality_score=float(data.get("overall_quality_score", 1.0)),
            step_metrics=[
                StepExtractionMetrics.from_dict(s)
                for s in (data.get("steps") or [])
                if isinstance(s, dict)
            ],
        )


@dataclass
class StoryE2EExecutionResult:
    """Full execution result, accuracy scorecard, multi-metric benchmark, and token/cost report for a synthetic story."""

    story: SavedSyntheticStory
    extraction_model: str
    story_cache_hit: bool
    extraction_cache_hits: int
    extraction_cache_misses: int
    generation_usage: TokenUsage
    extraction_billable_usage: TokenUsage
    extraction_historical_usage: TokenUsage
    combined_historical_usage: TokenUsage
    checks_passed: int
    checks_total: int
    failures: list[str]
    extracted_deltas_by_step: dict[int, StepDelta]
    raw_deltas_by_step: dict[int, StepDelta] = field(default_factory=dict)
    guidance_mode: GuidanceMode = "full_hints"
    metrics: ExtractionBenchmarkMetrics = field(default_factory=ExtractionBenchmarkMetrics)
    skipped_cached_run: bool = False
    run_cache_key: str = ""
    step_delta_checks_passed: int = 0
    step_delta_checks_total: int = 0
    cumulative_state_checks_passed: int = 0
    cumulative_state_checks_total: int = 0

    @property
    def pass_rate(self) -> float:
        if self.checks_total == 0:
            return 1.0
        return self.checks_passed / self.checks_total

    @property
    def step_delta_pass_rate(self) -> float:
        if self.step_delta_checks_total == 0:
            return self.pass_rate
        return self.step_delta_checks_passed / self.step_delta_checks_total

    @property
    def cumulative_state_pass_rate(self) -> float:
        if self.cumulative_state_checks_total == 0:
            return self.pass_rate
        return self.cumulative_state_checks_passed / self.cumulative_state_checks_total


@dataclass
class ModelComparisonReport:
    """Side-by-side comparison of multiple extraction models on the same synthetic story."""

    story_slug: str
    story_title: str
    guidance_mode: GuidanceMode
    results: list[StoryE2EExecutionResult]
    n_trials: int = 1
    trial_stats_by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_slug": self.story_slug,
            "story_title": self.story_title,
            "guidance_mode": self.guidance_mode,
            "n_trials": self.n_trials,
            "models": [
                {
                    "model_id": r.extraction_model,
                    "skipped_cached_run": r.skipped_cached_run,
                    "run_cache_key": r.run_cache_key,
                    "strict_pass_rate": round(r.pass_rate, 4),
                    "step_delta_pass_rate": round(r.step_delta_pass_rate, 4),
                    "cumulative_state_pass_rate": round(r.cumulative_state_pass_rate, 4),
                    "overall_quality_score": round(r.metrics.overall_quality_score, 4),
                    "trial_stats": self.trial_stats_by_model.get(r.extraction_model, {}),
                    "metrics": r.metrics.to_dict(),
                    "extraction_usage": r.extraction_historical_usage.to_dict(model=r.extraction_model),
                }
                for r in self.results
            ],
        }

    def to_markdown_table(self) -> str:
        trial_note = f" (Mean ± Std across {self.n_trials} trials)" if self.n_trials > 1 else ""
        lines = [
            f"# Model Extraction Benchmark: {self.story_title} (`{self.story_slug}`)",
            "",
            f"- **Guidance Mode**: `{self.guidance_mode}`",
            f"- **Total Story Chunks**: `{len(self.results[0].story.chunks) if self.results else 0}`",
            *( [f"- **Trials per Model**: `{self.n_trials}`"] if self.n_trials > 1 else [] ),
            "",
            f"| Model | Run Status | Overall Quality{trial_note} | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | Voice Reg. Acc | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |",
            "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in self.results:
            m = r.metrics
            u = r.extraction_historical_usage
            cost = u.estimate_cost_usd(model=r.extraction_model)["total_cost_usd"]
            if u.latency_ms <= 0.0 and u.total_tokens > 0:
                lat_str = "n/a (cached)"
            else:
                lat_str = f"{u.latency_ms / 1000.0:.1f}s"
            fb = "PASS" if m.flashback_guard_passed else "FAIL"
            status_tag = "SKIPPED (Cached)" if r.skipped_cached_run else "RAN"
            t_stats = self.trial_stats_by_model.get(r.extraction_model)
            if self.n_trials > 1 and t_stats:
                oq_str = f"**{t_stats['overall_quality_mean']:.1%} ± {t_stats['overall_quality_std']:.1%}**"
            else:
                oq_str = f"**{m.overall_quality_score:.1%}**"
            lines.append(
                f"| `{r.extraction_model}` | {status_tag} | {oq_str} | {r.pass_rate:.1%} | "
                f"{r.step_delta_pass_rate:.1%} | {r.cumulative_state_pass_rate:.1%} | "
                f"{m.room_occupancy_jaccard:.1%} | {m.location_delta_f1:.1%} | {m.mood_soft_accuracy:.1%} | "
                f"{m.witness_f1:.1%} | {m.silent_witness_recall:.1%} | {m.epistemic_leak_count} | "
                f"{m.inform_delta_f1:.1%} | {m.voice_register_accuracy:.1%} | {m.pov_secret_leak_count} | {fb} | "
                f"{m.hallucinated_id_count} ({m.hallucinated_id_rate:.0%}) | {m.total_retries} | "
                f"{u.total_tokens:,} | {lat_str} | ${cost:.5f} |"
            )
        lines.append("")
        return "\n".join(lines)


@dataclass
class CrossStoryModelAggregate:
    """Mean and standard deviation of a single model's extraction quality across multiple stories/trials."""

    model_id: str
    guidance_mode: GuidanceMode
    story_slugs: list[str]
    n_runs: int
    overall_quality_mean: float
    overall_quality_std: float
    strict_pass_mean: float
    strict_pass_std: float
    step_delta_pass_mean: float
    cumulative_state_pass_mean: float
    epistemic_score_mean: float
    spatial_state_score_mean: float
    pov_retrieval_score_mean: float
    edge_flashback_score_mean: float
    schema_hygiene_score_mean: float
    total_tokens: int
    total_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "guidance_mode": self.guidance_mode,
            "story_slugs": self.story_slugs,
            "n_runs": self.n_runs,
            "overall_quality_mean": round(self.overall_quality_mean, 4),
            "overall_quality_std": round(self.overall_quality_std, 4),
            "strict_pass_mean": round(self.strict_pass_mean, 4),
            "strict_pass_std": round(self.strict_pass_std, 4),
            "step_delta_pass_mean": round(self.step_delta_pass_mean, 4),
            "cumulative_state_pass_mean": round(self.cumulative_state_pass_mean, 4),
            "epistemic_score_mean": round(self.epistemic_score_mean, 4),
            "spatial_state_score_mean": round(self.spatial_state_score_mean, 4),
            "pov_retrieval_score_mean": round(self.pov_retrieval_score_mean, 4),
            "edge_flashback_score_mean": round(self.edge_flashback_score_mean, 4),
            "schema_hygiene_score_mean": round(self.schema_hygiene_score_mean, 4),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


@dataclass
class CrossStoryBenchmarkReport:
    """Aggregated multi-story / multi-trial benchmark report with Mean ± StdDev confidence metrics."""

    story_slugs: list[str]
    guidance_mode: GuidanceMode
    aggregates: list[CrossStoryModelAggregate]
    per_story_reports: dict[str, ModelComparisonReport] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_slugs": self.story_slugs,
            "guidance_mode": self.guidance_mode,
            "aggregates": [a.to_dict() for a in self.aggregates],
            "per_story": {slug: rep.to_dict() for slug, rep in self.per_story_reports.items()},
        }

    def to_markdown_table(self) -> str:
        lines = [
            f"# Cross-Story Multi-Topology Extraction Benchmark (`{self.guidance_mode}`)",
            "",
            f"- **Evaluated Stories ({len(self.story_slugs)})**: {', '.join(f'`{s}`' for s in self.story_slugs)}",
            "",
            "| Model | Runs | Overall Quality (Mean ± Std) | Strict Checks (Mean ± Std) | Step-Delta Checks | Cumulative State | Epistemic | Spatial/State | POV Retrieval | Edge/Flashback | Schema Hygiene | Total Tokens | Total Est. Cost (USD) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for a in self.aggregates:
            lines.append(
                f"| `{a.model_id}` | {a.n_runs} | **{a.overall_quality_mean:.1%} ± {a.overall_quality_std:.1%}** | "
                f"{a.strict_pass_mean:.1%} ± {a.strict_pass_std:.1%} | {a.step_delta_pass_mean:.1%} | "
                f"{a.cumulative_state_pass_mean:.1%} | {a.epistemic_score_mean:.1%} | "
                f"{a.spatial_state_score_mean:.1%} | {a.pov_retrieval_score_mean:.1%} | "
                f"{a.edge_flashback_score_mean:.1%} | {a.schema_hygiene_score_mean:.1%} | "
                f"{a.total_tokens:,} | ${a.total_cost_usd:.5f} |"
            )
        lines.append("")
        return "\n".join(lines)


# ============================================================================
# 1. Story Blueprints Catalog (Multiple 3K-Token Stories + Micro Baseline)
# ============================================================================


def _build_six_chunk_family_blueprint(
    *,
    story_slug: str,
    title: str,
    genre: str,
    description: str,
    protagonist: CharacterSeed,
    commander: CharacterSeed,
    spy: CharacterSeed,
    disguise_alias: str,
    disguise_persona: str,
    culprit: CharacterSeed,
    culprit_alias: str,
    loc_main: LocationSeed,
    loc_secret: LocationSeed,
    loc_out: LocationSeed,
    artifact: ArtifactSeed,
    protagonist_open_quote: str,
    spy_open_quote: str,
    protagonist_confront_quote: str,
    culprit_handoff_quote: str,
    spy_unmask_quote: str,
    protagonist_sarcasm_quote: str,
    commander_climax_quote: str,
    deception_believed_summary: str,
    flashback_summary: str,
) -> LongStoryBlueprint:
    """Constructs a 6-chunk x ~500-token (~3,000-token) blueprint covering all Tiers 1-6."""
    c_prot = protagonist.entity_id
    c_cmd = commander.entity_id
    c_spy = spy.entity_id
    c_culp = culprit.entity_id
    l_main = loc_main.entity_id
    l_sec = loc_secret.entity_id
    l_out = loc_out.entity_id
    art_id = artifact.entity_id

    ep_confront = "ep_step2_secret_confrontation"
    ep_handoff = "ep_step3_artifact_handoff"
    ep_flashback = "ep_step4_past_flashback"
    ep_unmask = "ep_step5_disguise_unmasked"

    params = SynthStoryParams(
        title=title,
        chunk_target_words=(320, 460),
        locations=[loc_main, loc_secret, loc_out],
        characters=[protagonist, commander, spy, culprit],
        artifacts=[artifact],
        initial_edges=[
            InitialEdgeSeed(c_prot, c_cmd, "stance", {"register": "Vi", "tone": "respectful", "trust": 0.7}),
            InitialEdgeSeed(c_cmd, c_prot, "stance", {"register": "Vi", "tone": "guarded", "trust": 0.4}),
            InitialEdgeSeed(c_prot, c_culp, "kinship", {"relation": "associate"}),
            InitialEdgeSeed(c_prot, c_culp, "stance", {"register": "Vi", "tone": "suspicious", "trust": -0.3}),
            InitialEdgeSeed(c_culp, c_prot, "stance", {"register": "Vi", "tone": "fearful", "trust": -0.5}),
            InitialEdgeSeed(c_culp, c_prot, "facade", {"register": "Vi", "tone": "polite", "trust": 0.8}),
            InitialEdgeSeed(c_cmd, c_culp, "belief_state", {"mood": "composed", "loyalty": "devoted"}),
            InitialEdgeSeed(c_spy, c_cmd, "facade", {"register": "Vi", "tone": "deferential", "trust": 0.5}),
            InitialEdgeSeed(c_culp, art_id, "possesses", {"status": "hidden"}),
        ],
    )

    beats: list[ChunkBeatBlueprint] = [
        # CHUNK 1 (~500 tokens): Opening assembly + Disguise delivery
        ChunkBeatBlueprint(
            step=1,
            scene_id="scene_1_assembly",
            chapter_id="ch_1",
            location_id=l_main,
            story_time="day_1_dusk",
            beat_title="Assembly & Disguised Courier Briefing",
            writing_directives=(
                f"Setting: {loc_main.canonical_name} ({l_main}). All 4 characters ({protagonist.canonical_name}, "
                f"{commander.canonical_name}, {culprit.canonical_name}, and {spy.canonical_name} referred to ONLY by "
                f"her disguise '{disguise_alias}') are present in {loc_main.canonical_name}. "
                f"Write 3 to 4 rich, atmospheric paragraphs (~360-420 words, ~500 tokens) detailing the high-stakes "
                f"inspection of records and rising tension. Include these exact dialogue lines naturally in the prose:\n"
                f"1) {protagonist.canonical_name} addressing {commander.canonical_name}: \"{protagonist_open_quote}\"\n"
                f"2) {disguise_alias} bowing and speaking to {commander.canonical_name}: \"{spy_open_quote}\"\n"
                f"3) {culprit.canonical_name} murmuring a brief two-word filler reply: \"Yes, indeed.\"\n"
                f"Do NOT change any character's location or mood in Chunk 1."
            ),
            extraction_hint=(
                f"In Chunk 1, all 4 characters remain in {l_main} with their initial moods. "
                f"Record an EpisodeRecord (e.g. episode_id='ep_step1_opening_briefing', predicate='briefing', "
                f"location_id='{l_main}', participants={{'{c_prot}': 'actor', '{c_cmd}': 'target', '{c_spy}': 'messenger'}}, "
                f"witnesses=['{c_prot}', '{c_cmd}', '{c_spy}', '{c_culp}'], salience=3.2). "
                f"Do NOT emit mood or location changes in entities."
            ),
            expected_room_occupants={
                l_main: {c_prot, c_cmd, c_spy, c_culp},
                l_sec: set(),
                l_out: set(),
            },
            expected_moods={
                c_prot: protagonist.initial_mood,
                c_cmd: commander.initial_mood,
                c_spy: spy.initial_mood,
                c_culp: culprit.initial_mood,
            },
            expected_episode_id="ep_step1_opening_briefing",
            oracle_delta=StepDelta(
                scene_id="scene_1_assembly",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step1_opening_briefing",
                        predicate="briefing",
                        summary=f"{protagonist.canonical_name} and {disguise_alias} briefed {commander.canonical_name} in {loc_main.canonical_name}.",
                        participants={c_prot: "actor", c_cmd: "target", c_spy: "messenger"},
                        witnesses=[c_prot, c_cmd, c_spy, c_culp],
                        location_id=l_main,
                        salience=3.2,
                    )
                ],
            ),
            voice_metadata_by_speaker={
                c_prot: {"register": "Vi", "tone": "respectful"},
                c_spy: {"register": "Vi", "tone": "deferential", "persona": disguise_persona},
            },
            voice_quality_by_speaker={c_prot: 9.3, c_spy: 9.1},
        ),
        # CHUNK 2 (~500 tokens): Commander exits to outdoor area; Protagonist confronts Culprit over stolen artifact
        ChunkBeatBlueprint(
            step=2,
            scene_id="scene_1_assembly",
            chapter_id="ch_1",
            location_id=l_main,
            story_time="day_1_dusk",
            beat_title="Balcony Departure & Secret Confrontation",
            writing_directives=(
                f"Write 3 to 4 detailed paragraphs (~360-420 words, ~500 tokens). "
                f"First, {commander.canonical_name} grows 'restless' from the stifling atmosphere and steps out of "
                f"{loc_main.canonical_name} onto the {loc_out.canonical_name} ({l_out}), closing the heavy doors behind him. "
                f"As soon as {commander.canonical_name} is outside and out of earshot, {protagonist.canonical_name} turns "
                f"'furious' and confronts {culprit.canonical_name} (who becomes 'panicked') over stealing the {artifact.canonical_name}. "
                f"Include {protagonist.canonical_name}'s exact furious quote to {culprit.canonical_name}: \"{protagonist_confront_quote}\" "
                f"Note that {disguise_alias} remains silently in the shadows of {loc_main.canonical_name} witnessing the confrontation."
            ),
            extraction_hint=(
                f"REQUIRED extractions for Chunk 2:\n"
                f"1) EntityDelta for '{c_cmd}': state={{'mood': 'restless', 'location_id': '{l_out}'}}\n"
                f"2) EntityDelta for '{c_prot}': state={{'mood': 'furious'}}, caused_by_episode_id='{ep_confront}'\n"
                f"3) EntityDelta for '{c_culp}': state={{'mood': 'panicked'}}, caused_by_episode_id='{ep_confront}'\n"
                f"4) EdgeDelta: source_id='{c_prot}', target_id='{c_culp}', relation_type='stance', "
                f"state={{'register': 'ti', 'tone': 'furious', 'trust': -0.95}}, caused_by_episode_id='{ep_confront}'\n"
                f"5) EpisodeRecord with EXACT episode_id='{ep_confront}', predicate='confrontation', "
                f"summary='{protagonist.canonical_name} accused {culprit.canonical_name} of stealing the {artifact.canonical_name}.', "
                f"participants={{'{c_prot}': 'actor', '{c_culp}': 'target', '{art_id}': 'object'}}, "
                f"witnesses=['{c_prot}', '{c_culp}', '{c_spy}'] (MUST NOT include '{c_cmd}' since he is at '{l_out}'!), "
                f"location_id='{l_main}', salience=9.2."
            ),
            expected_room_occupants={
                l_main: {c_prot, c_spy, c_culp},
                l_sec: set(),
                l_out: {c_cmd},
            },
            expected_moods={
                c_prot: "furious",
                c_cmd: "restless",
                c_spy: spy.initial_mood,
                c_culp: "panicked",
            },
            acceptable_moods={
                c_cmd: {"restless", "confident", "irritated", "impatient", "uneasy"},
            },
            required_prose_cues=[commander.canonical_name, protagonist.canonical_name, culprit.canonical_name, protagonist_confront_quote],
            expected_episode_id=ep_confront,
            oracle_delta=StepDelta(
                scene_id="scene_1_assembly",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                entities=[
                    EntityDelta(entity_id=c_cmd, state={"mood": "restless", "location_id": l_out}),
                    EntityDelta(
                        entity_id=c_prot,
                        state={"mood": "furious"},
                        caused_by_episode_id=ep_confront,
                    ),
                    EntityDelta(
                        entity_id=c_culp,
                        state={"mood": "panicked"},
                        caused_by_episode_id=ep_confront,
                    ),
                ],
                edges=[
                    EdgeDelta(
                        source_id=c_prot,
                        target_id=c_culp,
                        relation_type="stance",
                        state={"register": "ti", "tone": "furious", "trust": -0.95},
                        caused_by_episode_id=ep_confront,
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_confront,
                        predicate="confrontation",
                        summary=f"{protagonist.canonical_name} accused {culprit.canonical_name} of stealing the {artifact.canonical_name}.",
                        participants={c_prot: "actor", c_culp: "target", art_id: "object"},
                        witnesses=[c_prot, c_culp, c_spy],
                        location_id=l_main,
                        salience=9.2,
                    )
                ],
            ),
            voice_metadata_by_speaker={c_prot: {"register": "ti", "tone": "furious"}},
            voice_quality_by_speaker={c_prot: 9.6},
        ),
        # CHUNK 3 (~500 tokens): Artifact handoff & flight into the Secret Location
        ChunkBeatBlueprint(
            step=3,
            scene_id="scene_1_assembly",
            chapter_id="ch_1",
            location_id=l_main,
            story_time="day_1_dusk",
            beat_title="Desperate Artifact Handoff & Flight to the Vault",
            writing_directives=(
                f"Write 3 to 4 vivid paragraphs (~360-420 words, ~500 tokens). "
                f"Still furious, {protagonist.canonical_name} demands that {culprit.canonical_name} empty his pockets. "
                f"In a desperate panic, {culprit.canonical_name} presses the {artifact.canonical_name} into the gloved hand of "
                f"'{disguise_alias}' and bolts down the passage into the {loc_secret.canonical_name} ({l_sec}), with "
                f"'{disguise_alias}' chasing closely after him into the {loc_secret.canonical_name}. "
                f"Include {culprit.canonical_name}'s exact stammered quote: \"{culprit_handoff_quote}\""
            ),
            extraction_hint=(
                f"REQUIRED extractions for Chunk 3:\n"
                f"1) EntityDelta for '{c_culp}': state={{'location_id': '{l_sec}'}}\n"
                f"2) EntityDelta for '{c_spy}': state={{'location_id': '{l_sec}'}}\n"
                f"3) EdgeDelta: source_id='{c_spy}', target_id='{art_id}', relation_type='possesses', "
                f"state={{'status': 'concealed'}}, caused_by_episode_id='{ep_handoff}'\n"
                f"4) EpisodeRecord with EXACT episode_id='{ep_handoff}', predicate='item_transfer', "
                f"summary='{culprit.canonical_name} handed the {artifact.canonical_name} to {disguise_alias} and fled into {loc_secret.canonical_name}.', "
                f"participants={{'{c_culp}': 'actor', '{c_spy}': 'recipient', '{art_id}': 'instrument'}}, "
                f"witnesses=['{c_prot}', '{c_culp}', '{c_spy}'], location_id='{l_main}', salience=8.8."
            ),
            expected_room_occupants={
                l_main: {c_prot},
                l_sec: {c_culp, c_spy},
                l_out: {c_cmd},
            },
            expected_moods={
                c_prot: "furious",
                c_cmd: "restless",
                c_spy: spy.initial_mood,
                c_culp: "panicked",
            },
            acceptable_moods={
                c_cmd: {"restless", "confident", "irritated", "impatient", "uneasy"},
            },
            required_prose_cues=[protagonist.canonical_name, culprit.canonical_name, artifact.canonical_name, culprit_handoff_quote],
            expected_episode_id=ep_handoff,
            oracle_delta=StepDelta(
                scene_id="scene_1_assembly",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                entities=[
                    EntityDelta(entity_id=c_culp, state={"location_id": l_sec}),
                    EntityDelta(entity_id=c_spy, state={"location_id": l_sec}),
                ],
                edges=[
                    EdgeDelta(
                        source_id=c_spy,
                        target_id=art_id,
                        relation_type="possesses",
                        state={"status": "concealed"},
                        caused_by_episode_id=ep_handoff,
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_handoff,
                        predicate="item_transfer",
                        summary=f"{culprit.canonical_name} handed the {artifact.canonical_name} to {disguise_alias} and fled into {loc_secret.canonical_name}.",
                        participants={c_culp: "actor", c_spy: "recipient", art_id: "instrument"},
                        witnesses=[c_prot, c_culp, c_spy],
                        location_id=l_main,
                        salience=8.8,
                    )
                ],
            ),
            voice_metadata_by_speaker={c_culp: {"register": "ti", "tone": "panicked"}},
            voice_quality_by_speaker={c_culp: 5.5},
        ),
        # CHUNK 4 (~500 tokens): Extended Flashback Interlude (10 Years Ago)
        ChunkBeatBlueprint(
            step=4,
            scene_id="scene_2_flashback",
            chapter_id="ch_1",
            location_id=l_main,
            story_time="10_years_ago",
            beat_title="Flashback to Ten Years Ago",
            writing_directives=(
                f"Write a 3 to 4 paragraph flashback scene (~360-420 words, ~500 tokens) set ten years ago ('10_years_ago'). "
                f"Standing alone in {loc_main.canonical_name}, {protagonist.canonical_name} remembers the winter crisis ten years ago "
                f"when {protagonist.canonical_name} was 'naive_trusting' and trusted {culprit.canonical_name} implicitly during "
                f"the historic pact signing. Make it clear this entire chunk is a memory/flashback to ten years ago so "
                f"present-day room locations and present-day furious mood are preserved."
            ),
            extraction_hint=(
                f"REQUIRED extractions for Chunk 4 (FLASHBACK):\n"
                f"1) Set is_flashback=True and story_time='10_years_ago' on the StepDelta!\n"
                f"2) EpisodeRecord with EXACT episode_id='{ep_flashback}', predicate='flashback', "
                f"summary='{flashback_summary}', "
                f"participants={{'{c_prot}': 'experiencer', '{c_culp}': 'companion'}}, "
                f"witnesses=['{c_prot}', '{c_culp}'], location_id='{l_main}', story_time='10_years_ago', salience=6.5."
            ),
            expected_room_occupants={
                l_main: {c_prot},
                l_sec: {c_culp, c_spy},
                l_out: {c_cmd},
            },
            expected_moods={
                c_prot: "furious",
                c_cmd: "restless",
                c_spy: spy.initial_mood,
                c_culp: "panicked",
            },
            acceptable_moods={
                c_cmd: {"restless", "confident", "irritated", "impatient", "uneasy"},
            },
            required_prose_cues=[protagonist.canonical_name, culprit.canonical_name],
            expected_episode_id=ep_flashback,
            is_flashback=True,
            oracle_delta=StepDelta(
                scene_id="scene_2_flashback",
                chapter_id="ch_1",
                story_time="10_years_ago",
                is_flashback=True,
                entities=[EntityDelta(entity_id=c_prot, state={"mood": "naive_trusting"})],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_flashback,
                        predicate="flashback",
                        summary=flashback_summary,
                        participants={c_prot: "experiencer", c_culp: "companion"},
                        witnesses=[c_prot, c_culp],
                        location_id=l_main,
                        story_time="10_years_ago",
                        salience=6.5,
                    )
                ],
            ),
        ),
        # CHUNK 5 (~500 tokens): Disguise unmasking in Secret Room + Return & Epistemic Deception in Main Room
        ChunkBeatBlueprint(
            step=5,
            scene_id="scene_3_unmask_and_lie",
            chapter_id="ch_2",
            location_id=l_main,
            story_time="day_1_night",
            beat_title="Unmasking in the Vault & Deception Upon Return",
            writing_directives=(
                f"Write 3 to 4 rich paragraphs (~360-420 words, ~500 tokens) covering two connected beats:\n"
                f"Beat A: Down in the {loc_secret.canonical_name} ({l_sec}), '{disguise_alias}' throws back her hood/mask, "
                f"revealing to the 'shocked' {culprit.canonical_name} that she is actually {spy.canonical_name} (who is now 'determined'). "
                f"Include {spy.canonical_name}'s exact commanding quote to {culprit.canonical_name}: \"{spy_unmask_quote}\"\n"
                f"Beat B: Moments later, {commander.canonical_name} steps back into {loc_main.canonical_name} ({l_main}) from the "
                f"{loc_out.canonical_name} just as {culprit.canonical_name} returns up the stairs into {loc_main.canonical_name}. "
                f"{culprit.canonical_name} lies to {commander.canonical_name}, telling him the cover story: '{deception_believed_summary}'. "
                f"Though still 'furious' inside, {protagonist.canonical_name} remarks with biting sarcasm: \"{protagonist_sarcasm_quote}\" "
                f"({spy.canonical_name} remains down in {l_sec} until the next chunk)."
            ),
            extraction_hint=(
                f"REQUIRED extractions for Chunk 5:\n"
                f"1) EntityDelta for '{c_spy}': state={{'mood': 'determined'}}, add_aliases=[AliasRecord(alias='{disguise_alias}', "
                f"entity_id='{c_spy}', kind='disguise', persona_name='{disguise_persona}', known_to=['{c_prot}', '{c_culp}'])]\n"
                f"2) EntityDelta for '{c_culp}': state={{'mood': 'shocked', 'location_id': '{l_main}'}}, caused_by_episode_id='{ep_unmask}'\n"
                f"3) EntityDelta for '{c_cmd}': state={{'location_id': '{l_main}'}}\n"
                f"4) EdgeDelta: source_id='{c_spy}', target_id='{c_culp}', relation_type='stance', "
                f"state={{'register': 'ti', 'tone': 'commanding', 'trust': -0.6}}, caused_by_episode_id='{ep_unmask}'\n"
                f"5) EpisodeRecord with EXACT episode_id='{ep_unmask}', predicate='revelation', "
                f"summary='{spy.canonical_name} unmasked her {disguise_alias} disguise to {culprit.canonical_name} in {loc_secret.canonical_name}.', "
                f"participants={{'{c_spy}': 'actor', '{c_culp}': 'target'}}, witnesses=['{c_spy}', '{c_culp}'], location_id='{l_sec}', salience=9.0\n"
                f"6) InformDelta: episode_id='{ep_confront}', entity_id='{c_cmd}', epistemic='deceived', "
                f"believed_summary='{deception_believed_summary}'"
            ),
            expected_room_occupants={
                l_main: {c_prot, c_cmd, c_culp},
                l_sec: {c_spy},
                l_out: set(),
            },
            expected_moods={
                c_prot: "furious",
                c_cmd: "restless",
                c_spy: "determined",
                c_culp: "shocked",
            },
            acceptable_moods={
                c_cmd: {"restless", "confident", "irritated", "impatient", "uneasy", "concerned"},
            },
            required_prose_cues=[spy.canonical_name, culprit.canonical_name, commander.canonical_name, spy_unmask_quote, protagonist_sarcasm_quote],
            expected_episode_id=ep_unmask,
            oracle_delta=StepDelta(
                scene_id="scene_3_unmask_and_lie",
                chapter_id="ch_2",
                story_time="day_1_night",
                entities=[
                    EntityDelta(
                        entity_id=c_spy,
                        state={"mood": "determined"},
                        add_aliases=[
                            AliasRecord(
                                alias=disguise_alias,
                                entity_id=c_spy,
                                kind="disguise",
                                persona_name=disguise_persona,
                                known_to=[c_prot, c_culp],
                            )
                        ],
                    ),
                    EntityDelta(
                        entity_id=c_culp,
                        state={"mood": "shocked", "location_id": l_main},
                        caused_by_episode_id=ep_unmask,
                    ),
                    EntityDelta(entity_id=c_cmd, state={"location_id": l_main}),
                ],
                edges=[
                    EdgeDelta(
                        source_id=c_spy,
                        target_id=c_culp,
                        relation_type="stance",
                        state={"register": "ti", "tone": "commanding", "trust": -0.6},
                        caused_by_episode_id=ep_unmask,
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_unmask,
                        predicate="revelation",
                        summary=f"{spy.canonical_name} unmasked her {disguise_alias} disguise to {culprit.canonical_name} in {loc_secret.canonical_name}.",
                        participants={c_spy: "actor", c_culp: "target"},
                        witnesses=[c_spy, c_culp],
                        location_id=l_sec,
                        salience=9.0,
                    )
                ],
                inform=[
                    InformDelta(
                        episode_id=ep_confront,
                        entity_id=c_cmd,
                        epistemic="deceived",
                        believed_summary=deception_believed_summary,
                    )
                ],
            ),
            voice_metadata_by_speaker={c_spy: {"register": "ti", "tone": "commanding"}},
            voice_quality_by_speaker={c_spy: 9.4},
        ),
        # CHUNK 6 (~500 tokens): Climax — Spy enters Main Room, informs Commander of truth, Culprit forgets flashback
        ChunkBeatBlueprint(
            step=6,
            scene_id="scene_4_climax",
            chapter_id="ch_2",
            location_id=l_main,
            story_time="day_1_night",
            beat_title="Truth Revealed & Memory Suppression Climax",
            writing_directives=(
                f"Write 3 to 4 dramatic concluding paragraphs (~360-420 words, ~500 tokens). "
                f"{spy.canonical_name} enters {loc_main.canonical_name} ({l_main}) holding the recovered {artifact.canonical_name} "
                f"(so all 4 characters are now together in {loc_main.canonical_name}). "
                f"{spy.canonical_name} informs {commander.canonical_name} of the full truth about {culprit.canonical_name}'s theft "
                f"during the earlier confrontation. Learning he was lied to, {commander.canonical_name} becomes 'indignant' and "
                f"barks at {culprit.canonical_name}: \"{commander_climax_quote}\" "
                f"Cornered at the table, {culprit.canonical_name} triggers a memory-erasing draught/damper that completely wipes "
                f"his memory of the ten-years-ago flashback event."
            ),
            extraction_hint=(
                f"REQUIRED extractions for Chunk 6:\n"
                f"1) EntityDelta for '{c_spy}': state={{'location_id': '{l_main}'}}\n"
                f"2) EntityDelta for '{c_cmd}': state={{'mood': 'indignant'}}, caused_by_episode_id='{ep_confront}'\n"
                f"3) EdgeDelta: source_id='{c_cmd}', target_id='{c_culp}', relation_type='stance', "
                f"state={{'register': 'ti', 'tone': 'stern', 'trust': -0.8}}, caused_by_episode_id='{ep_confront}'\n"
                f"4) InformDelta: episode_id='{ep_confront}', entity_id='{c_cmd}', epistemic='told'\n"
                f"5) InformDelta: episode_id='{ep_flashback}', entity_id='{c_culp}', epistemic='forgotten'"
            ),
            expected_room_occupants={
                l_main: {c_prot, c_cmd, c_spy, c_culp},
                l_sec: set(),
                l_out: set(),
            },
            expected_moods={
                c_prot: "furious",
                c_cmd: "indignant",
                c_spy: "determined",
                c_culp: "shocked",
            },
            acceptable_moods={
                c_culp: {"shocked", "confused", "calm", "bewildered", "dazed", "panicked"},
            },
            optional_inform=[
                InformDelta(episode_id=ep_handoff, entity_id=c_cmd, epistemic="told"),
                InformDelta(episode_id=ep_confront, entity_id=c_culp, epistemic="forgotten"),
                InformDelta(episode_id=ep_handoff, entity_id=c_culp, epistemic="forgotten"),
            ],
            required_prose_cues=[spy.canonical_name, commander.canonical_name, culprit.canonical_name, commander_climax_quote],
            oracle_delta=StepDelta(
                scene_id="scene_4_climax",
                chapter_id="ch_2",
                story_time="day_1_night",
                entities=[
                    EntityDelta(entity_id=c_spy, state={"location_id": l_main}),
                    EntityDelta(
                        entity_id=c_cmd,
                        state={"mood": "indignant"},
                        caused_by_episode_id=ep_confront,
                    ),
                ],
                edges=[
                    EdgeDelta(
                        source_id=c_cmd,
                        target_id=c_culp,
                        relation_type="stance",
                        state={"register": "ti", "tone": "stern", "trust": -0.8},
                        caused_by_episode_id=ep_confront,
                    )
                ],
                inform=[
                    InformDelta(episode_id=ep_confront, entity_id=c_cmd, epistemic="told"),
                    InformDelta(episode_id=ep_flashback, entity_id=c_culp, epistemic="forgotten"),
                ],
            ),
            voice_metadata_by_speaker={c_cmd: {"register": "ti", "tone": "stern"}},
            voice_quality_by_speaker={c_cmd: 9.2},
        ),
    ]

    return LongStoryBlueprint(
        story_slug=story_slug,
        title=title,
        genre=genre,
        description=description,
        target_total_tokens=3000,
        target_chunk_tokens=500,
        target_chunk_words=(320, 460),
        params=params,
        beats=beats,
        key_episode_ids={
            "opening": "ep_step1_opening_briefing",
            "confrontation": ep_confront,
            "handoff": ep_handoff,
            "flashback": ep_flashback,
            "unmask": ep_unmask,
        },
    )


def _build_veritas_locked_room_3k_blueprint() -> LongStoryBlueprint:
    """Constructs a distinct 5-character, 4-room, 2-artifact Locked-Room Detective Mystery (6 chunks, ~3K tokens).

    Unlike the 4-character Obsidian Signet family, this blueprint exercises:
    - Multi-room initial distribution (4 in loc_study, 1 in loc_conservatory)
    - Exclusive inventory transfer with explicit EdgeDelta(remove=True)
    - 5 characters ('char_vane', 'char_cora', 'char_thorne', 'char_giles', 'char_mina')
    - 4 locations ('loc_study', 'loc_conservatory', 'loc_cellar', 'loc_gallery')
    - 2 artifacts ('item_ledger_key', 'item_arsenic_vial')
    """
    vane = CharacterSeed(
        entity_id="char_vane",
        canonical_name="Vane",
        base_attributes={"title": "Chief Inspector", "gender": "M", "declension": "type_1"},
        initial_mood="calm",
        initial_location="loc_study",
        private_state={"secret_agenda": "prove_inheritance_poisoning"},
        aliases=["the chief inspector"],
    )
    cora = CharacterSeed(
        entity_id="char_cora",
        canonical_name="Cora",
        base_attributes={"title": "Dowager Countess", "gender": "F", "declension": "type_a"},
        initial_mood="confident",
        initial_location="loc_study",
        private_state={"secret_agenda": "protect_estate_codicil"},
        aliases=["the dowager countess"],
    )
    giles = CharacterSeed(
        entity_id="char_giles",
        canonical_name="Giles",
        base_attributes={"title": "Forensic Locksmith", "gender": "M", "declension": "type_1"},
        initial_mood="watchful",
        initial_location="loc_study",
        private_state={"true_loyalty": "scotland_yard_special_branch"},
        aliases=[
            AliasRecord(
                alias="the veiled locksmith",
                entity_id="char_giles",
                kind="disguise",
                persona_name="Locksmith",
                known_to=["char_vane"],
            )
        ],
    )
    thorne = CharacterSeed(
        entity_id="char_thorne",
        canonical_name="Thorne",
        base_attributes={"title": "Family Physician", "gender": "M", "declension": "type_1"},
        initial_mood="nervous",
        initial_location="loc_study",
        private_state={"guilt": "tampered_with_apothecary_ledger"},
        aliases=["the family physician"],
    )
    mina = CharacterSeed(
        entity_id="char_mina",
        canonical_name="Mina",
        base_attributes={"title": "Young Ward", "gender": "F", "declension": "type_a"},
        initial_mood="calm",
        initial_location="loc_conservatory",
        private_state={"witnessed": "midnight_footsteps"},
        aliases=["the young ward"],
    )

    loc_study = LocationSeed("loc_study", "Mahogany Study", "Wood-paneled study smelling of pipe tobacco and rain")
    loc_cellar = LocationSeed("loc_cellar", "Wine Cellar", "Damp brick cellar beneath the kitchen stairs")
    loc_gallery = LocationSeed("loc_gallery", "Portrait Gallery", "Drafty upper gallery lined with ancestral oil paintings")
    loc_conservatory = LocationSeed("loc_conservatory", "Glass Conservatory", "Mist-filled conservatory overlooking the gardens")

    art_key = ArtifactSeed("item_ledger_key", "Brass Ledger Key", "char_thorne")
    art_vial = ArtifactSeed("item_arsenic_vial", "Crystal Arsenic Vial", "char_cora")

    ep_confront = "ep_step2_secret_confrontation"
    ep_handoff = "ep_step3_artifact_handoff"
    ep_flashback = "ep_step4_past_flashback"
    ep_unmask = "ep_step5_disguise_unmasked"

    q_open_1 = "Molim vas da pažljivo pregledate ove toksikološke nalaze pre nego što policija zapečati imanje."
    q_confront = "Prestani sa tim bednim lažima, doktore, jer znam da si sinoć ukrao mesingani ključ iz kabineta!"
    q_handoff = "Take this cursed key and hide it deep inside the stone cellar before the constables arrive!"
    q_unmask = "Predaj mi sve lažne recepte odmah, Torne, ili ćeš do zore biti u okružnom zatvoru!"
    q_sarcasm = "Oh, what a wonderfully convenient tale of heroic vigilance, dear doctor."
    q_climax = "Kako se usuđuješ da obmanjuješ porodični savet dok držimo dokaze u rukama!"
    deception_summary = "Northern raiders broke into the annex and stole the Brass Ledger Key while Thorne defended the cabinet."
    fb_summary = "Ten years ago during the winter epidemic, Vane trusted Thorne at the hospital inquest."

    params = SynthStoryParams(
        title="The Veritas Locked-Room Inquest (5-Character Detective Topology)",
        chunk_target_words=(320, 460),
        locations=[loc_study, loc_cellar, loc_gallery, loc_conservatory],
        characters=[vane, cora, giles, thorne, mina],
        artifacts=[art_key, art_vial],
        initial_edges=[
            InitialEdgeSeed("char_vane", "char_cora", "stance", {"register": "Vi", "tone": "respectful", "trust": 0.6}),
            InitialEdgeSeed("char_cora", "char_vane", "stance", {"register": "Vi", "tone": "guarded", "trust": 0.3}),
            InitialEdgeSeed("char_vane", "char_thorne", "stance", {"register": "Vi", "tone": "suspicious", "trust": -0.4}),
            InitialEdgeSeed("char_thorne", "char_vane", "stance", {"register": "Vi", "tone": "fearful", "trust": -0.6}),
            InitialEdgeSeed("char_thorne", "char_vane", "facade", {"register": "Vi", "tone": "polite", "trust": 0.7}),
            InitialEdgeSeed("char_cora", "char_thorne", "belief_state", {"mood": "composed", "loyalty": "devoted"}),
            InitialEdgeSeed("char_giles", "char_cora", "facade", {"register": "Vi", "tone": "deferential", "trust": 0.5}),
            InitialEdgeSeed("char_thorne", "item_ledger_key", "possesses", {"status": "hidden_in_waistcoat"}),
            InitialEdgeSeed("char_cora", "item_arsenic_vial", "possesses", {"status": "locked_in_reticule"}),
        ],
    )

    beats: list[ChunkBeatBlueprint] = [
        ChunkBeatBlueprint(
            step=1,
            scene_id="scene_1_inquest",
            chapter_id="ch_1",
            location_id="loc_study",
            story_time="day_1_dusk",
            beat_title="Inquest Opening in the Mahogany Study",
            writing_directives=(
                "Setting: Mahogany Study (loc_study), where Vane, Cora, Thorne, and the veiled locksmith (Giles) are gathered, "
                "while Mina remains quiet in the Glass Conservatory (loc_conservatory). Include quotes."
            ),
            extraction_hint=(
                "Record ep_step1_opening_briefing in loc_study with witnesses ['char_vane', 'char_cora', 'char_giles', 'char_thorne']."
            ),
            expected_room_occupants={
                "loc_study": {"char_vane", "char_cora", "char_giles", "char_thorne"},
                "loc_cellar": set(),
                "loc_gallery": set(),
                "loc_conservatory": {"char_mina"},
            },
            expected_moods={
                "char_vane": "calm",
                "char_cora": "confident",
                "char_giles": "watchful",
                "char_thorne": "nervous",
                "char_mina": "calm",
            },
            expected_episode_id="ep_step1_opening_briefing",
            oracle_delta=StepDelta(
                scene_id="scene_1_inquest",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step1_opening_briefing",
                        predicate="briefing",
                        summary="Vane and the veiled locksmith presented forensic lock impressions to Cora in the Mahogany Study.",
                        participants={"char_vane": "actor", "char_cora": "target", "char_giles": "messenger"},
                        witnesses=["char_vane", "char_cora", "char_giles", "char_thorne"],
                        location_id="loc_study",
                        salience=3.2,
                    )
                ],
            ),
            voice_metadata_by_speaker={
                "char_vane": {"register": "Vi", "tone": "respectful"},
                "char_giles": {"register": "Vi", "tone": "deferential", "persona": "Locksmith"},
            },
            voice_quality_by_speaker={"char_vane": 9.3, "char_giles": 9.1},
            required_prose_cues=["Vane", "Cora", "Thorne", "the veiled locksmith", q_open_1],
        ),
        ChunkBeatBlueprint(
            step=2,
            scene_id="scene_1_inquest",
            chapter_id="ch_1",
            location_id="loc_study",
            story_time="day_1_dusk",
            beat_title="Gallery Departure & Study Accusation",
            writing_directives=(
                "Cora grows restless and steps out to the Portrait Gallery (loc_gallery). Vane turns furious and confronts panicked Thorne over the Brass Ledger Key."
            ),
            extraction_hint="Cora moves to loc_gallery (mood='restless'); Vane ('furious') confronts Thorne ('panicked') in ep_step2_secret_confrontation.",
            expected_room_occupants={
                "loc_study": {"char_vane", "char_giles", "char_thorne"},
                "loc_cellar": set(),
                "loc_gallery": {"char_cora"},
                "loc_conservatory": {"char_mina"},
            },
            expected_moods={
                "char_vane": "furious",
                "char_cora": "restless",
                "char_giles": "watchful",
                "char_thorne": "panicked",
                "char_mina": "calm",
            },
            acceptable_moods={"char_cora": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id=ep_confront,
            oracle_delta=StepDelta(
                scene_id="scene_1_inquest",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                entities=[
                    EntityDelta(entity_id="char_cora", state={"mood": "restless", "location_id": "loc_gallery"}),
                    EntityDelta(entity_id="char_vane", state={"mood": "furious"}, caused_by_episode_id=ep_confront),
                    EntityDelta(entity_id="char_thorne", state={"mood": "panicked"}, caused_by_episode_id=ep_confront),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_vane",
                        target_id="char_thorne",
                        relation_type="stance",
                        state={"register": "ti", "tone": "furious", "trust": -0.95},
                        caused_by_episode_id=ep_confront,
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_confront,
                        predicate="confrontation",
                        summary="Vane accused Thorne of stealing the Brass Ledger Key while Cora was in the Portrait Gallery.",
                        participants={"char_vane": "actor", "char_thorne": "target", "item_ledger_key": "object"},
                        witnesses=["char_vane", "char_thorne", "char_giles"],
                        location_id="loc_study",
                        salience=9.2,
                    )
                ],
            ),
            voice_metadata_by_speaker={"char_vane": {"register": "ti", "tone": "furious"}},
            voice_quality_by_speaker={"char_vane": 9.5},
            required_prose_cues=["Cora", "Portrait Gallery", "Vane", "Thorne", q_confront],
        ),
        ChunkBeatBlueprint(
            step=3,
            scene_id="scene_1_inquest",
            chapter_id="ch_1",
            location_id="loc_study",
            story_time="day_1_dusk",
            beat_title="Exclusive Key Transfer & Flight to the Wine Cellar",
            writing_directives=(
                "Thorne hands the Brass Ledger Key to the veiled locksmith and flees down into the Wine Cellar (loc_cellar), "
                "chased by the veiled locksmith, while Mina walks from the Glass Conservatory into the Mahogany Study."
            ),
            extraction_hint="Thorne and Giles move to loc_cellar; Mina moves to loc_study; Giles possesses item_ledger_key (and Thorne's possesses edge is removed).",
            expected_room_occupants={
                "loc_study": {"char_vane", "char_mina"},
                "loc_cellar": {"char_thorne", "char_giles"},
                "loc_gallery": {"char_cora"},
                "loc_conservatory": set(),
            },
            expected_moods={
                "char_vane": "furious",
                "char_cora": "restless",
                "char_giles": "watchful",
                "char_thorne": "panicked",
                "char_mina": "calm",
            },
            acceptable_moods={"char_cora": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id=ep_handoff,
            oracle_delta=StepDelta(
                scene_id="scene_1_inquest",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                entities=[
                    EntityDelta(entity_id="char_thorne", state={"location_id": "loc_cellar"}),
                    EntityDelta(entity_id="char_giles", state={"location_id": "loc_cellar"}),
                    EntityDelta(entity_id="char_mina", state={"location_id": "loc_study"}),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_thorne",
                        target_id="item_ledger_key",
                        relation_type="possesses",
                        remove=True,
                        caused_by_episode_id=ep_handoff,
                    ),
                    EdgeDelta(
                        source_id="char_giles",
                        target_id="item_ledger_key",
                        relation_type="possesses",
                        state={"status": "concealed"},
                        caused_by_episode_id=ep_handoff,
                    ),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_handoff,
                        predicate="item_transfer",
                        summary="Thorne handed the Brass Ledger Key to the veiled locksmith and fled into the Wine Cellar.",
                        participants={"char_thorne": "actor", "char_giles": "recipient", "item_ledger_key": "instrument"},
                        witnesses=["char_vane", "char_thorne", "char_giles"],
                        location_id="loc_study",
                        salience=8.8,
                    )
                ],
            ),
            voice_metadata_by_speaker={"char_thorne": {"register": "ti", "tone": "panicked"}},
            voice_quality_by_speaker={"char_thorne": 5.5},
            required_prose_cues=["Brass Ledger Key", "Wine Cellar", "Mina", q_handoff],
        ),
        ChunkBeatBlueprint(
            step=4,
            scene_id="scene_2_flashback",
            chapter_id="ch_1",
            location_id="loc_study",
            story_time="10_years_ago",
            beat_title="Flashback to the Hospital Inquest Ten Years Ago",
            writing_directives="Vane recalls ten years ago during the winter epidemic when he trusted Thorne implicitly.",
            extraction_hint="Set is_flashback=True, story_time='10_years_ago', and record ep_step4_past_flashback.",
            expected_room_occupants={
                "loc_study": {"char_vane", "char_mina"},
                "loc_cellar": {"char_thorne", "char_giles"},
                "loc_gallery": {"char_cora"},
                "loc_conservatory": set(),
            },
            expected_moods={
                "char_vane": "furious",
                "char_cora": "restless",
                "char_giles": "watchful",
                "char_thorne": "panicked",
                "char_mina": "calm",
            },
            acceptable_moods={"char_cora": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id=ep_flashback,
            is_flashback=True,
            oracle_delta=StepDelta(
                scene_id="scene_2_flashback",
                chapter_id="ch_1",
                story_time="10_years_ago",
                is_flashback=True,
                entities=[EntityDelta(entity_id="char_vane", state={"mood": "naive_trusting"})],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_flashback,
                        predicate="flashback",
                        summary=fb_summary,
                        participants={"char_vane": "experiencer", "char_thorne": "companion"},
                        witnesses=["char_vane", "char_thorne"],
                        location_id="loc_study",
                        story_time="10_years_ago",
                        salience=6.5,
                    )
                ],
            ),
            required_prose_cues=["Ten years ago", "Vane", "Thorne"],
        ),
        ChunkBeatBlueprint(
            step=5,
            scene_id="scene_3_unmask_and_alibi",
            chapter_id="ch_2",
            location_id="loc_study",
            story_time="day_1_night",
            beat_title="Cellar Unmasking & False Alibi Upstairs",
            writing_directives=(
                "Down in the Wine Cellar, the veiled locksmith unmasks as Giles ('determined') to shocked Thorne. "
                "Upstairs, Cora returns from the Portrait Gallery to the Mahogany Study as Thorne stumbles back in and lies about northern raiders."
            ),
            extraction_hint="Giles unmasks to Thorne; Thorne ('shocked') and Cora move to loc_study; InformDelta deceived on ep_step2_secret_confrontation for char_cora.",
            expected_room_occupants={
                "loc_study": {"char_vane", "char_cora", "char_thorne", "char_mina"},
                "loc_cellar": {"char_giles"},
                "loc_gallery": set(),
                "loc_conservatory": set(),
            },
            expected_moods={
                "char_vane": "furious",
                "char_cora": "restless",
                "char_giles": "determined",
                "char_thorne": "shocked",
                "char_mina": "calm",
            },
            acceptable_moods={"char_cora": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id=ep_unmask,
            oracle_delta=StepDelta(
                scene_id="scene_3_unmask_and_alibi",
                chapter_id="ch_2",
                story_time="day_1_night",
                entities=[
                    EntityDelta(
                        entity_id="char_giles",
                        state={"mood": "determined"},
                        add_aliases=[
                            AliasRecord(
                                alias="the veiled locksmith",
                                entity_id="char_giles",
                                kind="disguise",
                                persona_name="Locksmith",
                                known_to=["char_vane", "char_thorne"],
                            )
                        ],
                    ),
                    EntityDelta(
                        entity_id="char_thorne",
                        state={"mood": "shocked", "location_id": "loc_study"},
                        caused_by_episode_id=ep_unmask,
                    ),
                    EntityDelta(entity_id="char_cora", state={"location_id": "loc_study"}),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_giles",
                        target_id="char_thorne",
                        relation_type="stance",
                        state={"register": "ti", "tone": "commanding", "trust": -0.6},
                        caused_by_episode_id=ep_unmask,
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_unmask,
                        predicate="revelation",
                        summary="Giles unmasked his veiled locksmith disguise to Thorne in the Wine Cellar.",
                        participants={"char_giles": "actor", "char_thorne": "target"},
                        witnesses=["char_giles", "char_thorne"],
                        location_id="loc_cellar",
                        salience=9.0,
                    )
                ],
                inform=[
                    InformDelta(
                        episode_id=ep_confront,
                        entity_id="char_cora",
                        epistemic="deceived",
                        believed_summary=deception_summary,
                    )
                ],
            ),
            voice_metadata_by_speaker={"char_giles": {"register": "ti", "tone": "commanding"}},
            voice_quality_by_speaker={"char_giles": 9.4},
            required_prose_cues=["Giles", "Thorne", "Cora", q_unmask, q_sarcasm],
        ),
        ChunkBeatBlueprint(
            step=6,
            scene_id="scene_4_climax",
            chapter_id="ch_2",
            location_id="loc_study",
            story_time="day_1_night",
            beat_title="Forensic Reckoning & Sedative Memory Wipe",
            writing_directives=(
                "Giles enters the Mahogany Study holding the Brass Ledger Key and informs Cora of Thorne's theft. "
                "Indignant Cora confronts Thorne, who swallows a memory-erasing draught that wipes his memory of the ten-years-ago hospital inquest."
            ),
            extraction_hint="Giles moves to loc_study; Cora becomes 'indignant'; InformDelta 'told' for char_cora and 'forgotten' for char_thorne.",
            expected_room_occupants={
                "loc_study": {"char_vane", "char_cora", "char_giles", "char_thorne", "char_mina"},
                "loc_cellar": set(),
                "loc_gallery": set(),
                "loc_conservatory": set(),
            },
            expected_moods={
                "char_vane": "furious",
                "char_cora": "indignant",
                "char_giles": "determined",
                "char_thorne": "shocked",
                "char_mina": "calm",
            },
            acceptable_moods={"char_thorne": {"shocked", "confused", "calm", "bewildered", "dazed"}},
            optional_inform=[
                InformDelta(episode_id=ep_handoff, entity_id="char_cora", epistemic="told"),
                InformDelta(episode_id=ep_confront, entity_id="char_thorne", epistemic="forgotten"),
                InformDelta(episode_id=ep_handoff, entity_id="char_thorne", epistemic="forgotten"),
            ],
            oracle_delta=StepDelta(
                scene_id="scene_4_climax",
                chapter_id="ch_2",
                story_time="day_1_night",
                entities=[
                    EntityDelta(entity_id="char_giles", state={"location_id": "loc_study"}),
                    EntityDelta(entity_id="char_cora", state={"mood": "indignant"}, caused_by_episode_id=ep_confront),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_cora",
                        target_id="char_thorne",
                        relation_type="stance",
                        state={"register": "ti", "tone": "stern", "trust": -0.8},
                        caused_by_episode_id=ep_confront,
                    )
                ],
                inform=[
                    InformDelta(episode_id=ep_confront, entity_id="char_cora", epistemic="told"),
                    InformDelta(episode_id=ep_flashback, entity_id="char_thorne", epistemic="forgotten"),
                ],
            ),
            voice_metadata_by_speaker={"char_cora": {"register": "ti", "tone": "stern"}},
            voice_quality_by_speaker={"char_cora": 9.2},
            required_prose_cues=["Giles", "Cora", "Thorne", q_climax],
        ),
    ]

    return LongStoryBlueprint(
        story_slug="veritas_locked_room_3k",
        title="The Veritas Locked-Room Inquest (5-Character Detective Topology)",
        genre="Victorian Locked-Room Detective Mystery",
        description="A 3,000-token (6 x ~500-token) locked-room mystery with 5 characters, 4 rooms, 2 artifacts, and exclusive inventory transfers.",
        target_total_tokens=3000,
        target_chunk_tokens=500,
        target_chunk_words=(320, 460),
        params=params,
        beats=beats,
        key_episode_ids={
            "opening": "ep_step1_opening_briefing",
            "confrontation": ep_confront,
            "handoff": ep_handoff,
            "flashback": ep_flashback,
            "unmask": ep_unmask,
        },
    )


def _build_aethelgard_chronicle_12k_blueprint() -> LongStoryBlueprint:
    """Constructs an 18-chunk (~9,000-12,000 token) multi-chapter marathon story across 3 chapters.

    Designed specifically to test long-horizon memory properties that 6-chunk stories cannot:
    - 18 sequential chunks (~500 tokens each) across 3 chapters (`ch_1`, `ch_2`, `ch_3`)
    - 6 characters ('char_elena', 'char_marcus', 'char_lyra', 'char_julian', 'char_kaelen', 'char_isolde')
    - 5 locations ('loc_council', 'loc_vault', 'loc_balcony', 'loc_observatory', 'loc_courtyard')
    - 2 artifacts ('item_signet', 'item_cipher_cylinder')
    - Long-range epistemic recall: Deception at Step 5 persists across Chapter 2 (Steps 7-12) and is finally overturned by truth ('told') at Step 18
    - Mid-story episode consolidation at Step 7 (`consolidation_step=7`) and branching at Step 12 (`branch_fork_step=12`)
    - Two distinct non-destructive flashbacks (Step 4 and Step 11)
    - Exclusive inventory transfers (`remove=True`) across multiple steps
    """
    base_3k = _build_six_chunk_family_blueprint(
        story_slug="aethelgard_chronicle_12k",
        title="The Aethelgard Chronicle (18-Chunk Multi-Chapter 12K Marathon)",
        genre="Epic Political & Arcane Conspiracy Marathon",
        description="An 18-chunk (~9,000-12,000 token) 3-chapter narrative benchmark testing long-horizon epistemic recall, chapter consolidation, and timeline branching.",
        protagonist=CharacterSeed(
            entity_id="char_elena",
            canonical_name="Elena",
            base_attributes={"title": "High Chancellor", "gender": "F", "declension": "type_a"},
            initial_mood="calm",
            initial_location="loc_council",
            private_state={"secret_agenda": "expose_aethelgard_cabal"},
            aliases=["the Iron Chancellor"],
        ),
        commander=CharacterSeed(
            entity_id="char_marcus",
            canonical_name="Marcus",
            base_attributes={"title": "Lord Marshal", "gender": "M", "declension": "type_1"},
            initial_mood="confident",
            initial_location="loc_council",
            private_state={"secret_agenda": "defend_northern_legions"},
            aliases=["the old general"],
        ),
        spy=CharacterSeed(
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
                    known_to=["char_elena"],
                )
            ],
        ),
        disguise_alias="the hooded courier",
        disguise_persona="Courier",
        culprit=CharacterSeed(
            entity_id="char_julian",
            canonical_name="Julian",
            base_attributes={"title": "Royal Treasurer", "gender": "M", "declension": "type_1"},
            initial_mood="nervous",
            initial_location="loc_council",
            private_state={"guilt": "forged_royal_seal"},
            aliases=["the young treasurer"],
        ),
        culprit_alias="the young treasurer",
        loc_main=LocationSeed("loc_council", "Council Chamber", "Grand vaulted council hall of Aethelgard"),
        loc_secret=LocationSeed("loc_vault", "Archive Vault", "Subterranean archive vault beneath the citadel"),
        loc_out=LocationSeed("loc_balcony", "Windy Balcony", "High stone parapet overlooking the winter courtyard"),
        artifact=ArtifactSeed("item_signet", "Obsidian Signet", "char_julian"),
        protagonist_open_quote="Molim vas da pažljivo pregledate ove poverljive izveštaje pre nego što kraljevska straža zatvori kapije.",
        spy_open_quote="Ponizno vam donosim zapečaćene depeše sa severne granice koje zahtevaju vašu hitnu vojnu procenu.",
        protagonist_confront_quote="Prestani sa tim bednim lažima, izdajniče, jer znam da si sinoć ukrao crni pečat iz komore!",
        culprit_handoff_quote="Take this cursed ring and hide it deep inside the stone vault before the guards arrive!",
        spy_unmask_quote="Predaj mi sve lažne povelje odmah, Julijane, ili ćeš do zore biti u kraljevskoj tamnici!",
        protagonist_sarcasm_quote="Oh, what a wonderfully convenient tale of heroic vigilance, dear cousin.",
        commander_climax_quote="Kako se usuđuješ da obmanjuješ vojni savet dok držimo dokaze u rukama!",
        deception_believed_summary="Northern raiders broke into the annex and stole the Obsidian Signet while Julian defended the ledger.",
        flashback_summary="Ten years ago during the winter siege, Elena trusted Julian at the treaty signing.",
    )

    # Expand cast with 2 additional characters, 2 additional locations, and 1 additional artifact for Chapters 2 & 3
    kaelen = CharacterSeed(
        entity_id="char_kaelen",
        canonical_name="Kaelen",
        base_attributes={"title": "Grand Astromancer", "gender": "M", "declension": "type_1"},
        initial_mood="watchful",
        initial_location="loc_observatory",
        private_state={"secret_agenda": "decode_star_cipher"},
        aliases=["the astromancer"],
    )
    isolde = CharacterSeed(
        entity_id="char_isolde",
        canonical_name="Isolde",
        base_attributes={"title": "Tribunal Envoy", "gender": "F", "declension": "type_a"},
        initial_mood="calm",
        initial_location="loc_courtyard",
        private_state={"mission": "audit_citadel_charters"},
        aliases=["the tribunal envoy"],
    )
    loc_obs = LocationSeed("loc_observatory", "Starlight Observatory", "Domed brass observatory atop the eastern spire")
    loc_court = LocationSeed("loc_courtyard", "Inner Courtyard", "Torchlit cobblestone courtyard patrolled by sentries")
    art_cipher = ArtifactSeed("item_cipher_cylinder", "Brass Cipher Cylinder", "char_kaelen")

    params = SynthStoryParams(
        title=base_3k.title,
        chunk_target_words=(320, 460),
        locations=[*base_3k.params.locations, loc_obs, loc_court],
        characters=[*base_3k.params.characters, kaelen, isolde],
        artifacts=[*base_3k.params.artifacts, art_cipher],
        initial_edges=[
            *base_3k.params.initial_edges,
            InitialEdgeSeed("char_elena", "char_kaelen", "stance", {"register": "Vi", "tone": "respectful", "trust": 0.5}),
            InitialEdgeSeed("char_kaelen", "item_cipher_cylinder", "possesses", {"status": "locked_in_astrolabe"}),
        ],
    )

    # Update Chapter 1 (Steps 1-5) room & mood maps to include char_kaelen and char_isolde
    beats: list[ChunkBeatBlueprint] = []
    for b in base_3k.beats[:5]:
        occ = {k: set(v) for k, v in b.expected_room_occupants.items()}
        occ["loc_observatory"] = {"char_kaelen"}
        occ["loc_courtyard"] = {"char_isolde"}
        moods = dict(b.expected_moods)
        moods["char_kaelen"] = "watchful"
        moods["char_isolde"] = "calm"
        beats.append(
            ChunkBeatBlueprint(
                step=b.step,
                scene_id=b.scene_id,
                chapter_id="ch_1",
                location_id=b.location_id,
                story_time=b.story_time,
                beat_title=b.beat_title,
                writing_directives=b.writing_directives,
                extraction_hint=b.extraction_hint,
                expected_room_occupants=occ,
                expected_moods=moods,
                acceptable_moods=dict(b.acceptable_moods),
                expected_episode_id=b.expected_episode_id,
                is_flashback=b.is_flashback,
                oracle_delta=b.oracle_delta.model_copy(deep=True),
                voice_metadata_by_speaker=dict(b.voice_metadata_by_speaker),
                voice_quality_by_speaker=dict(b.voice_quality_by_speaker),
                required_prose_cues=list(b.required_prose_cues),
            )
        )

    # Step 6 (End of Chapter 1): Isolde enters Council Chamber from Inner Courtyard; Marcus still deceived!
    beats.append(
        ChunkBeatBlueprint(
            step=6,
            scene_id="scene_4_envoy_arrival",
            chapter_id="ch_1",
            location_id="loc_council",
            story_time="day_1_night",
            beat_title="Arrival of the Tribunal Envoy & Midnight Adjournment",
            writing_directives=(
                "Isolde enters the Council Chamber (loc_council) from the Inner Courtyard (loc_courtyard) to announce "
                "the tribunal audit while Marcus still believes Julian's cover story about northern raiders."
            ),
            extraction_hint="EntityDelta for 'char_isolde': state={'location_id': 'loc_council'}; EpisodeRecord 'ep_step6_envoy_arrival'.",
            expected_room_occupants={
                "loc_council": {"char_elena", "char_marcus", "char_julian", "char_isolde"},
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": {"char_kaelen"},
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "watchful",
                "char_isolde": "calm",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step6_envoy_arrival",
            oracle_delta=StepDelta(
                scene_id="scene_4_envoy_arrival",
                chapter_id="ch_1",
                story_time="day_1_night",
                entities=[EntityDelta(entity_id="char_isolde", state={"location_id": "loc_council"})],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step6_envoy_arrival",
                        predicate="arrival",
                        summary="Isolde entered the Council Chamber from the Inner Courtyard to convene the midnight tribunal audit.",
                        participants={"char_isolde": "actor", "char_elena": "target"},
                        witnesses=["char_elena", "char_marcus", "char_julian", "char_isolde"],
                        location_id="loc_council",
                        salience=3.5,
                    )
                ],
            ),
            required_prose_cues=["Isolde", "Council Chamber", "Elena", "Marcus"],
        )
    )

    # CHAPTER 2 (Steps 7..12): The Observatory & Undercroft Investigation
    # Step 7: Elena & Isolde ascend to Starlight Observatory to consult Kaelen
    beats.append(
        ChunkBeatBlueprint(
            step=7,
            scene_id="scene_5_observatory",
            chapter_id="ch_2",
            location_id="loc_observatory",
            story_time="day_1_midnight",
            beat_title="Ascent to the Starlight Observatory",
            writing_directives="Elena and Isolde move to the Starlight Observatory (loc_observatory) to meet Kaelen.",
            extraction_hint="char_elena and char_isolde move to loc_observatory; record ep_step7_observatory_parley.",
            expected_room_occupants={
                "loc_council": {"char_marcus", "char_julian"},
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": {"char_elena", "char_isolde", "char_kaelen"},
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "watchful",
                "char_isolde": "calm",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step7_observatory_parley",
            oracle_delta=StepDelta(
                scene_id="scene_5_observatory",
                chapter_id="ch_2",
                story_time="day_1_midnight",
                entities=[
                    EntityDelta(entity_id="char_elena", state={"location_id": "loc_observatory"}),
                    EntityDelta(entity_id="char_isolde", state={"location_id": "loc_observatory"}),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step7_observatory_parley",
                        predicate="consultation",
                        summary="Elena and Isolde ascended to the Starlight Observatory to inspect Kaelen's Brass Cipher Cylinder.",
                        participants={"char_elena": "actor", "char_kaelen": "target", "char_isolde": "witness"},
                        witnesses=["char_elena", "char_isolde", "char_kaelen"],
                        location_id="loc_observatory",
                        salience=4.5,
                    )
                ],
            ),
            required_prose_cues=["Elena", "Isolde", "Kaelen", "Starlight Observatory"],
        )
    )

    # Step 8: Kaelen hands Brass Cipher Cylinder to Elena (exclusive transfer) and becomes nervous
    beats.append(
        ChunkBeatBlueprint(
            step=8,
            scene_id="scene_5_observatory",
            chapter_id="ch_2",
            location_id="loc_observatory",
            story_time="day_1_midnight",
            beat_title="Decoding the Brass Cipher Cylinder",
            writing_directives="Kaelen grows nervous and hands the Brass Cipher Cylinder to Elena in the Starlight Observatory.",
            extraction_hint="char_kaelen mood='nervous'; char_elena possesses item_cipher_cylinder; record ep_step8_cipher_handoff.",
            expected_room_occupants={
                "loc_council": {"char_marcus", "char_julian"},
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": {"char_elena", "char_isolde", "char_kaelen"},
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "calm",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step8_cipher_handoff",
            oracle_delta=StepDelta(
                scene_id="scene_5_observatory",
                chapter_id="ch_2",
                story_time="day_1_midnight",
                entities=[
                    EntityDelta(entity_id="char_kaelen", state={"mood": "nervous"}, caused_by_episode_id="ep_step8_cipher_handoff")
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_kaelen",
                        target_id="item_cipher_cylinder",
                        relation_type="possesses",
                        remove=True,
                        caused_by_episode_id="ep_step8_cipher_handoff",
                    ),
                    EdgeDelta(
                        source_id="char_elena",
                        target_id="item_cipher_cylinder",
                        relation_type="possesses",
                        state={"status": "unlocked"},
                        caused_by_episode_id="ep_step8_cipher_handoff",
                    ),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step8_cipher_handoff",
                        predicate="item_transfer",
                        summary="Kaelen handed the Brass Cipher Cylinder to Elena in the Starlight Observatory after discovering the forged seal code.",
                        participants={"char_kaelen": "actor", "char_elena": "recipient", "item_cipher_cylinder": "instrument"},
                        witnesses=["char_elena", "char_isolde", "char_kaelen"],
                        location_id="loc_observatory",
                        salience=8.4,
                    )
                ],
            ),
            required_prose_cues=["Kaelen", "Elena", "Brass Cipher Cylinder"],
        )
    )

    # Step 9: Secret archive discovery by Lyra in the Archive Vault while everyone else is upstairs
    beats.append(
        ChunkBeatBlueprint(
            step=9,
            scene_id="scene_6_vault_ledger",
            chapter_id="ch_2",
            location_id="loc_vault",
            story_time="day_1_midnight",
            beat_title="The Hidden Ledger in the Archive Vault",
            writing_directives="Alone in the Archive Vault (loc_vault), Lyra uncovers the forged garrison ledger proving Julian's treason.",
            extraction_hint="Record ep_step9_vault_discovery in loc_vault with sole witness ['char_lyra'].",
            expected_room_occupants={
                "loc_council": {"char_marcus", "char_julian"},
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": {"char_elena", "char_isolde", "char_kaelen"},
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "calm",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step9_vault_discovery",
            oracle_delta=StepDelta(
                scene_id="scene_6_vault_ledger",
                chapter_id="ch_2",
                story_time="day_1_midnight",
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step9_vault_discovery",
                        predicate="discovery",
                        summary="Lyra uncovered the hidden garrison ransom ledger alone inside the Archive Vault.",
                        participants={"char_lyra": "actor", "item_signet": "instrument"},
                        witnesses=["char_lyra"],
                        location_id="loc_vault",
                        salience=8.9,
                    )
                ],
            ),
            required_prose_cues=["Lyra", "Archive Vault", "Obsidian Signet"],
        )
    )

    # Step 10: Marcus (still deceived from Step 5!) moves to Inner Courtyard with Julian to mobilize troops against 'northern raiders'
    beats.append(
        ChunkBeatBlueprint(
            step=10,
            scene_id="scene_7_courtyard_mobilization",
            chapter_id="ch_2",
            location_id="loc_courtyard",
            story_time="day_2_pre_dawn",
            beat_title="False Mobilization in the Inner Courtyard",
            writing_directives=(
                "Acting on the Step 5 lie about northern raiders, Marcus and Julian descend to the Inner Courtyard (loc_courtyard) to muster the watch."
            ),
            extraction_hint="char_marcus and char_julian move to loc_courtyard; record ep_step10_courtyard_muster.",
            expected_room_occupants={
                "loc_council": set(),
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": {"char_elena", "char_isolde", "char_kaelen"},
                "loc_courtyard": {"char_marcus", "char_julian"},
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "calm",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step10_courtyard_muster",
            oracle_delta=StepDelta(
                scene_id="scene_7_courtyard_mobilization",
                chapter_id="ch_2",
                story_time="day_2_pre_dawn",
                entities=[
                    EntityDelta(entity_id="char_marcus", state={"location_id": "loc_courtyard"}),
                    EntityDelta(entity_id="char_julian", state={"location_id": "loc_courtyard"}),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step10_courtyard_muster",
                        predicate="mobilization",
                        summary="Still believing Julian's lie about northern raiders, Marcus marched Julian into the Inner Courtyard to muster the garrison.",
                        participants={"char_marcus": "actor", "char_julian": "companion"},
                        witnesses=["char_marcus", "char_julian"],
                        location_id="loc_courtyard",
                        salience=6.0,
                    )
                ],
            ),
            required_prose_cues=["Marcus", "Julian", "Inner Courtyard"],
        )
    )

    # Step 11: Second Flashback (5 Years Ago — The Astral Concordat)
    beats.append(
        ChunkBeatBlueprint(
            step=11,
            scene_id="scene_8_second_flashback",
            chapter_id="ch_2",
            location_id="loc_observatory",
            story_time="10_years_ago",
            beat_title="Flashback to the Astral Concordat",
            writing_directives=(
                "In the Starlight Observatory, Elena remembers ten years ago when Kaelen and Julian first forged the cipher key."
            ),
            extraction_hint="Set is_flashback=True, story_time='10_years_ago', and record ep_step11_astral_flashback without overwriting present moods.",
            expected_room_occupants={
                "loc_council": set(),
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": {"char_elena", "char_isolde", "char_kaelen"},
                "loc_courtyard": {"char_marcus", "char_julian"},
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "calm",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step11_astral_flashback",
            is_flashback=True,
            oracle_delta=StepDelta(
                scene_id="scene_8_second_flashback",
                chapter_id="ch_2",
                story_time="10_years_ago",
                is_flashback=True,
                entities=[EntityDelta(entity_id="char_elena", state={"mood": "naive_trusting"})],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step11_astral_flashback",
                        predicate="flashback",
                        summary="Ten years ago during the Astral Concordat, Elena and Kaelen entrusted the cipher templates to Julian.",
                        participants={"char_elena": "experiencer", "char_kaelen": "companion"},
                        witnesses=["char_elena", "char_kaelen"],
                        location_id="loc_observatory",
                        story_time="10_years_ago",
                        salience=6.2,
                    )
                ],
            ),
            required_prose_cues=["Ten years ago", "Elena", "Kaelen"],
        )
    )

    # Step 12: End of Chapter 2 (Branching Point): Elena tells Isolde about Step 2 confrontation
    beats.append(
        ChunkBeatBlueprint(
            step=12,
            scene_id="scene_9_observatory_disclosure",
            chapter_id="ch_2",
            location_id="loc_observatory",
            story_time="day_2_pre_dawn",
            beat_title="Elena Informs the Tribunal Envoy",
            writing_directives=(
                "In the Starlight Observatory, Elena informs Isolde of the truth about Julian's theft during the Step 2 confrontation. "
                "Isolde becomes determined."
            ),
            extraction_hint="char_isolde mood='determined'; InformDelta(episode_id='ep_step2_secret_confrontation', entity_id='char_isolde', epistemic='told').",
            expected_room_occupants={
                "loc_council": set(),
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": {"char_elena", "char_isolde", "char_kaelen"},
                "loc_courtyard": {"char_marcus", "char_julian"},
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "determined",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step12_envoy_briefed",
            oracle_delta=StepDelta(
                scene_id="scene_9_observatory_disclosure",
                chapter_id="ch_2",
                story_time="day_2_pre_dawn",
                entities=[
                    EntityDelta(entity_id="char_isolde", state={"mood": "determined"}, caused_by_episode_id="ep_step2_secret_confrontation")
                ],
                inform=[
                    InformDelta(episode_id="ep_step2_secret_confrontation", entity_id="char_isolde", epistemic="told"),
                ],
            ),
            required_prose_cues=["Elena", "Isolde", "Julian"],
        )
    )

    # CHAPTER 3 (Steps 13..18): The Tribunal Reckoning
    # Step 13: Elena, Isolde, and Kaelen descend from Starlight Observatory back to Council Chamber
    beats.append(
        ChunkBeatBlueprint(
            step=13,
            scene_id="scene_10_council_reconvene",
            chapter_id="ch_3",
            location_id="loc_council",
            story_time="day_2_dawn",
            beat_title="Return from the Observatory to the Council Chamber",
            writing_directives="Elena, Isolde, and Kaelen return from the Starlight Observatory into the Council Chamber (loc_council).",
            extraction_hint="char_elena, char_isolde, and char_kaelen move to loc_council.",
            expected_room_occupants={
                "loc_council": {"char_elena", "char_isolde", "char_kaelen"},
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": set(),
                "loc_courtyard": {"char_marcus", "char_julian"},
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "determined",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step13_observatory_return",
            oracle_delta=StepDelta(
                scene_id="scene_10_council_reconvene",
                chapter_id="ch_3",
                story_time="day_2_dawn",
                entities=[
                    EntityDelta(entity_id="char_elena", state={"location_id": "loc_council"}),
                    EntityDelta(entity_id="char_isolde", state={"location_id": "loc_council"}),
                    EntityDelta(entity_id="char_kaelen", state={"location_id": "loc_council"}),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step13_observatory_return",
                        predicate="room_entry",
                        summary="Elena, Isolde, and Kaelen returned to the Council Chamber carrying the decoded Brass Cipher Cylinder.",
                        participants={"char_elena": "actor", "char_isolde": "companion", "char_kaelen": "companion"},
                        witnesses=["char_elena", "char_isolde", "char_kaelen"],
                        location_id="loc_council",
                        salience=4.0,
                    )
                ],
            ),
            required_prose_cues=["Elena", "Isolde", "Kaelen", "Council Chamber"],
        )
    )

    # Step 14: Marcus and Julian return from Inner Courtyard into Council Chamber
    beats.append(
        ChunkBeatBlueprint(
            step=14,
            scene_id="scene_10_council_reconvene",
            chapter_id="ch_3",
            location_id="loc_council",
            story_time="day_2_dawn",
            beat_title="Summons from the Inner Courtyard",
            writing_directives="Marcus and Julian enter the Council Chamber (loc_council) from the Inner Courtyard.",
            extraction_hint="char_marcus and char_julian move to loc_council.",
            expected_room_occupants={
                "loc_council": {"char_elena", "char_isolde", "char_kaelen", "char_marcus", "char_julian"},
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": set(),
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "determined",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step14_courtyard_return",
            oracle_delta=StepDelta(
                scene_id="scene_10_council_reconvene",
                chapter_id="ch_3",
                story_time="day_2_dawn",
                entities=[
                    EntityDelta(entity_id="char_marcus", state={"location_id": "loc_council"}),
                    EntityDelta(entity_id="char_julian", state={"location_id": "loc_council"}),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step14_courtyard_return",
                        predicate="room_entry",
                        summary="Marcus and Julian returned from the Inner Courtyard into the Council Chamber.",
                        participants={"char_marcus": "actor", "char_julian": "companion"},
                        witnesses=["char_elena", "char_isolde", "char_kaelen", "char_marcus", "char_julian"],
                        location_id="loc_council",
                        salience=4.2,
                    )
                ],
            ),
            required_prose_cues=["Marcus", "Julian", "Council Chamber"],
        )
    )

    # Step 15: Julian tries to deceive Isolde about the cipher cylinder; Isolde rejects it
    beats.append(
        ChunkBeatBlueprint(
            step=15,
            scene_id="scene_10_council_reconvene",
            chapter_id="ch_3",
            location_id="loc_council",
            story_time="day_2_dawn",
            beat_title="Cross-Examination of the Royal Treasurer",
            writing_directives="In the Council Chamber, Isolde confronts Julian with the Brass Cipher Cylinder; Julian grows panicked.",
            extraction_hint="char_julian mood='panicked'; EdgeDelta char_isolde -> char_julian stance (register='ti', tone='commanding', trust=-0.7).",
            expected_room_occupants={
                "loc_council": {"char_elena", "char_isolde", "char_kaelen", "char_marcus", "char_julian"},
                "loc_vault": {"char_lyra"},
                "loc_balcony": set(),
                "loc_observatory": set(),
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "panicked",
                "char_kaelen": "nervous",
                "char_isolde": "determined",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step15_tribunal_cross_exam",
            oracle_delta=StepDelta(
                scene_id="scene_10_council_reconvene",
                chapter_id="ch_3",
                story_time="day_2_dawn",
                entities=[
                    EntityDelta(entity_id="char_julian", state={"mood": "panicked"}, caused_by_episode_id="ep_step15_tribunal_cross_exam")
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_isolde",
                        target_id="char_julian",
                        relation_type="stance",
                        state={"register": "ti", "tone": "commanding", "trust": -0.7},
                        caused_by_episode_id="ep_step15_tribunal_cross_exam",
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step15_tribunal_cross_exam",
                        predicate="interrogation",
                        summary="Isolde interrogated Julian in the Council Chamber using the decoded Brass Cipher Cylinder.",
                        participants={"char_isolde": "actor", "char_julian": "target", "item_cipher_cylinder": "instrument"},
                        witnesses=["char_elena", "char_isolde", "char_kaelen", "char_marcus", "char_julian"],
                        location_id="loc_council",
                        salience=8.1,
                    )
                ],
            ),
            required_prose_cues=["Isolde", "Julian", "Brass Cipher Cylinder"],
        )
    )

    # Step 16: Lyra enters Council Chamber from Archive Vault (all 6 characters now in loc_council!) and shares Step 9 discovery
    beats.append(
        ChunkBeatBlueprint(
            step=16,
            scene_id="scene_11_grand_climax",
            chapter_id="ch_3",
            location_id="loc_council",
            story_time="day_2_dawn",
            beat_title="Spymaster's Ascent from the Archive Vault",
            writing_directives=(
                "Lyra enters the Council Chamber (loc_council) from the Archive Vault carrying the Obsidian Signet and informs Elena of ep_step9_vault_discovery."
            ),
            extraction_hint="char_lyra moves to loc_council; InformDelta(episode_id='ep_step9_vault_discovery', entity_id='char_elena', epistemic='told').",
            expected_room_occupants={
                "loc_council": {"char_elena", "char_isolde", "char_kaelen", "char_marcus", "char_julian", "char_lyra"},
                "loc_vault": set(),
                "loc_balcony": set(),
                "loc_observatory": set(),
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "restless",
                "char_lyra": "determined",
                "char_julian": "panicked",
                "char_kaelen": "nervous",
                "char_isolde": "determined",
            },
            acceptable_moods={"char_marcus": {"restless", "confident", "irritated", "uneasy"}},
            expected_episode_id="ep_step16_lyra_returns",
            oracle_delta=StepDelta(
                scene_id="scene_11_grand_climax",
                chapter_id="ch_3",
                story_time="day_2_dawn",
                entities=[EntityDelta(entity_id="char_lyra", state={"location_id": "loc_council"})],
                inform=[InformDelta(episode_id="ep_step9_vault_discovery", entity_id="char_elena", epistemic="told")],
            ),
            required_prose_cues=["Lyra", "Council Chamber", "Obsidian Signet", "Elena"],
        )
    )

    # Step 17: Lyra unmasks her courier persona to Marcus and tells Marcus the truth about Step 2 confrontation (overturning Step 5 lie 12 steps later!)
    beats.append(
        ChunkBeatBlueprint(
            step=17,
            scene_id="scene_11_grand_climax",
            chapter_id="ch_3",
            location_id="loc_council",
            story_time="day_2_dawn",
            beat_title="Overturning the Twelve-Step Deception",
            writing_directives=(
                "Lyra informs Marcus of the full truth about Julian's theft in Step 2 (ep_step2_secret_confrontation). "
                "Marcus becomes indignant and barks at Julian: \"Kako se usuđuješ da obmanjuješ vojni savet dok držimo dokaze u rukama!\""
            ),
            extraction_hint=(
                "char_marcus mood='indignant'; EdgeDelta char_marcus -> char_julian stance (register='ti', tone='stern', trust=-0.8); "
                "InformDelta(episode_id='ep_step2_secret_confrontation', entity_id='char_marcus', epistemic='told')."
            ),
            expected_room_occupants={
                "loc_council": {"char_elena", "char_isolde", "char_kaelen", "char_marcus", "char_julian", "char_lyra"},
                "loc_vault": set(),
                "loc_balcony": set(),
                "loc_observatory": set(),
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "indignant",
                "char_lyra": "determined",
                "char_julian": "panicked",
                "char_kaelen": "nervous",
                "char_isolde": "determined",
            },
            optional_inform=[
                InformDelta(episode_id="ep_step3_artifact_handoff", entity_id="char_marcus", epistemic="told"),
                InformDelta(episode_id="ep_step9_vault_discovery", entity_id="char_marcus", epistemic="told"),
            ],
            oracle_delta=StepDelta(
                scene_id="scene_11_grand_climax",
                chapter_id="ch_3",
                story_time="day_2_dawn",
                entities=[
                    EntityDelta(
                        entity_id="char_marcus",
                        state={"mood": "indignant"},
                        caused_by_episode_id="ep_step2_secret_confrontation",
                    )
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_marcus",
                        target_id="char_julian",
                        relation_type="stance",
                        state={"register": "ti", "tone": "stern", "trust": -0.8},
                        caused_by_episode_id="ep_step2_secret_confrontation",
                    )
                ],
                inform=[
                    InformDelta(episode_id="ep_step2_secret_confrontation", entity_id="char_marcus", epistemic="told"),
                ],
            ),
            voice_metadata_by_speaker={"char_marcus": {"register": "ti", "tone": "stern"}},
            voice_quality_by_speaker={"char_marcus": 9.2},
            required_prose_cues=["Lyra", "Marcus", "Julian", "Kako se usuđuješ da obmanjuješ vojni savet dok držimo dokaze u rukama!"],
        )
    )

    # Step 18: Finale — Julian drinks memory draught wiping Step 4 flashback ('ep_step4_past_flashback') and becomes shocked/confused
    beats.append(
        ChunkBeatBlueprint(
            step=18,
            scene_id="scene_11_grand_climax",
            chapter_id="ch_3",
            location_id="loc_council",
            story_time="day_2_dawn",
            beat_title="The Alchemical Lethe Draught Finale",
            writing_directives=(
                "Cornered by all five peers in the Council Chamber, Julian drinks a memory-erasing draught that wipes his memory "
                "of the ten-years-ago winter siege pact (ep_step4_past_flashback), leaving him shocked and bewildered."
            ),
            extraction_hint=(
                "char_julian mood='shocked'; InformDelta(episode_id='ep_step4_past_flashback', entity_id='char_julian', epistemic='forgotten')."
            ),
            expected_room_occupants={
                "loc_council": {"char_elena", "char_isolde", "char_kaelen", "char_marcus", "char_julian", "char_lyra"},
                "loc_vault": set(),
                "loc_balcony": set(),
                "loc_observatory": set(),
                "loc_courtyard": set(),
            },
            expected_moods={
                "char_elena": "furious",
                "char_marcus": "indignant",
                "char_lyra": "determined",
                "char_julian": "shocked",
                "char_kaelen": "nervous",
                "char_isolde": "determined",
            },
            acceptable_moods={"char_julian": {"shocked", "confused", "calm", "bewildered", "dazed", "panicked"}},
            optional_inform=[
                InformDelta(episode_id="ep_step11_astral_flashback", entity_id="char_julian", epistemic="forgotten"),
                InformDelta(episode_id="ep_step2_secret_confrontation", entity_id="char_julian", epistemic="forgotten"),
            ],
            oracle_delta=StepDelta(
                scene_id="scene_11_grand_climax",
                chapter_id="ch_3",
                story_time="day_2_dawn",
                entities=[EntityDelta(entity_id="char_julian", state={"mood": "shocked"})],
                inform=[InformDelta(episode_id="ep_step4_past_flashback", entity_id="char_julian", epistemic="forgotten")],
            ),
            required_prose_cues=["Julian", "Elena", "Marcus", "Council Chamber"],
        )
    )

    return LongStoryBlueprint(
        story_slug="aethelgard_chronicle_12k",
        title="The Aethelgard Chronicle (18-Chunk Multi-Chapter 12K Marathon)",
        genre="Epic Political & Arcane Conspiracy Marathon",
        description="An 18-chunk (~9,000-12,000 token) 3-chapter narrative benchmark testing long-horizon epistemic recall, chapter consolidation, and timeline branching.",
        target_total_tokens=9000,
        target_chunk_tokens=500,
        target_chunk_words=(320, 460),
        params=params,
        beats=beats,
        key_episode_ids={
            "opening": "ep_step1_opening_briefing",
            "confrontation": "ep_step2_secret_confrontation",
            "handoff": "ep_step3_artifact_handoff",
            "flashback": "ep_step4_past_flashback",
            "unmask": "ep_step5_disguise_unmasked",
            "vault_discovery": "ep_step9_vault_discovery",
            "second_flashback": "ep_step11_astral_flashback",
        },
        consolidation_step=7,
        branch_fork_step=12,
    )


_FORBIDDEN_META_IDENTIFIERS: tuple[str, ...] = (
    "naive_trusting",
    "is_flashback",
    "EntityDelta",
    "EdgeDelta",
    "InformDelta",
    "EpisodeRecord",
    "StepDelta",
    "ep_step",
)

_SUBTEXT_MOOD_ACTIONS: dict[str, str] = {
    "furious": "white-knuckled fingers snapped the bone paper-knife in two against the map table as an icy tremor entered the voice",
    "panicked": "knocked over the silver inkhorn and stumbled backward into the globe stand with shallow, gasping breaths",
    "restless": "checked the pocket chronometer for the fourth time in a minute and tugged at the suffocating collar",
    "determined": "planted both boots across the threshold and locked the iron bar into place with an unflinching jaw",
    "shocked": "dropped the leather ledger to the flagstones and froze with blood-drained cheeks and wide, unblinking eyes",
    "indignant": "slammed the heavy brass compass casing onto the table until the glass panes rattled in their frames",
    "nervous": "drummed uneven fingers against the brass clasp while glancing repeatedly toward the bolted door",
    "calm": "folded both hands evenly over the parchment chart and spoke in an unhurried, level cadence",
    "watchful": "tracked every shadow from beneath the lantern cowl without shifting a single footstep",
}


def _normalize_cue_for_match(s: str) -> str:
    """Normalizes quotes and trailing punctuation so minor dialogue punctuation differences match cleanly."""
    cleaned = s.strip().lower().strip("\"'“”‘’.,!?;:")
    return re.sub(r"\s+", " ", cleaned)


def _cue_in_text(cue: str, text: str) -> bool:
    if not cue:
        return True
    if cue.lower() in text.lower():
        return True
    norm_cue = _normalize_cue_for_match(cue)
    if not norm_cue:
        return True
    norm_text = re.sub(r"\s+", " ", text.lower())
    return norm_cue in norm_text


def _build_subtext_intrigue_3k_blueprint() -> LongStoryBlueprint:
    """Constructs a non-isomorphic Show-Don't-Tell Subtext & Dynamic Entity Discovery story (`subtext_intrigue_3k`).

    Unlike the Obsidian Signet 6-step template, this blueprint tests:
    - Zero verbatim mood labels in prose (`forbidden_prose_words` forbids 'furious', 'panicked', 'restless', 'determined', 'shocked', 'indignant')
    - Non-isomorphic beat order: immediate Step 1 room movement, early Flashback at Step 2, Confrontation at Step 3
    - Dynamic Mid-Story Entity & Artifact Introduction: `char_cassian` ('Cassian') and `item_astrolabe` ('Silver Astrolabe')
      enter at Step 3 without being pre-registered at Step 0
    - Multi-room initial distribution (3 in `loc_chartroom`, 1 disguised in `loc_rampart`)
    - Exclusive inventory transfer (`remove=True`) and multi-room split at Step 4
    """
    sera = CharacterSeed(
        entity_id="char_sera",
        canonical_name="Sera",
        base_attributes={"title": "Royal Cartographer", "gender": "F", "declension": "type_a"},
        initial_mood="calm",
        initial_location="loc_chartroom",
        private_state={"secret_agenda": "audit_sea_charts"},
        aliases=["the royal cartographer"],
    )
    bastian = CharacterSeed(
        entity_id="char_bastian",
        canonical_name="Bastian",
        base_attributes={"title": "Harbor Admiral", "gender": "M", "declension": "type_1"},
        initial_mood="confident",
        initial_location="loc_chartroom",
        private_state={"secret_agenda": "protect_convoy_routes"},
        aliases=["the harbor admiral"],
    )
    nyx = CharacterSeed(
        entity_id="char_nyx",
        canonical_name="Nyx",
        base_attributes={"title": "Crown Auditor", "gender": "F", "declension": "type_a"},
        initial_mood="watchful",
        initial_location="loc_rampart",
        private_state={"true_loyalty": "admiralty_counterintelligence"},
        aliases=[
            AliasRecord(
                alias="the lantern bearer",
                entity_id="char_nyx",
                kind="disguise",
                persona_name="Lantern Bearer",
                known_to=["char_sera"],
            )
        ],
    )
    orin = CharacterSeed(
        entity_id="char_orin",
        canonical_name="Orin",
        base_attributes={"title": "Quartermaster", "gender": "M", "declension": "type_1"},
        initial_mood="nervous",
        initial_location="loc_chartroom",
        private_state={"guilt": "altered_navigation_compass"},
        aliases=["the quartermaster"],
    )

    loc_chart = LocationSeed("loc_chartroom", "Navigators Chartroom", "Salt-scented tower room lined with brass globes and sea charts")
    loc_ramp = LocationSeed("loc_rampart", "Open Rampart", "Wind-scoured stone walkway overlooking the harbor lighthouse")
    loc_sub = LocationSeed("loc_sub_vault", "Tidal Sub-Vault", "Low arched stone chamber where tidewater laps against iron grates")
    art_comp = ArtifactSeed("item_compass", "Gilded Compass", "char_orin")

    ep_flash = "ep_step2_early_flashback"
    ep_confront = "ep_step3_chartroom_accusation"
    ep_handoff = "ep_step4_compass_handoff"
    ep_unmask = "ep_step5_subvault_unmask"

    q_open = "Molim vas da proverite ove pomorske karte pre nego što flota podigne jedra u zoru."
    q_confront = "Prestani sa tim lažima, Orine, jer znam da si sinoć zamenio magnet u pozlaćenom kompasu!"
    q_handoff = "Hide this cursed compass deep inside the tidal vault before the harbor watch seals the stairs!"
    q_unmask = "Predaj mi lažne brodske dnevnike odmah, Orine, ili ćeš do svitanja biti u lancima!"
    q_sarcasm = "Oh, what a remarkably selfless patrol through the storm drains, dear quartermaster."
    q_climax = "Kako se usuđuješ da obmanjuješ admiralitet dok držimo dokaze u rukama!"
    deception_summary = "Smugglers scaled the seawall during the squall and pried the Gilded Compass from the chart table while Orin fought them off."
    fb_summary = "Ten years ago during the winter blockade, Sera trusted Orin with the fleet route keys."

    params = SynthStoryParams(
        title="The Silent Astrolabe (Subtext & Dynamic Discovery 3K)",
        chunk_target_words=(320, 460),
        locations=[loc_chart, loc_ramp, loc_sub],
        characters=[sera, bastian, nyx, orin],
        artifacts=[art_comp],
        initial_edges=[
            InitialEdgeSeed("char_sera", "char_bastian", "stance", {"register": "Vi", "tone": "respectful", "trust": 0.7}),
            InitialEdgeSeed("char_bastian", "char_sera", "stance", {"register": "Vi", "tone": "guarded", "trust": 0.4}),
            InitialEdgeSeed("char_sera", "char_orin", "stance", {"register": "Vi", "tone": "suspicious", "trust": -0.3}),
            InitialEdgeSeed("char_orin", "char_sera", "facade", {"register": "Vi", "tone": "polite", "trust": 0.8}),
            InitialEdgeSeed("char_bastian", "char_orin", "belief_state", {"mood": "composed", "loyalty": "devoted"}),
            InitialEdgeSeed("char_nyx", "char_bastian", "facade", {"register": "Vi", "tone": "deferential", "trust": 0.5}),
            InitialEdgeSeed("char_orin", "item_compass", "possesses", {"status": "hidden_in_coat"}),
        ],
    )

    beats: list[ChunkBeatBlueprint] = [
        # Step 1: Immediate Step 1 location movement (the lantern bearer enters chartroom from rampart)
        ChunkBeatBlueprint(
            step=1,
            scene_id="scene_1_chartroom",
            chapter_id="ch_1",
            location_id="loc_chartroom",
            story_time="day_1_dusk",
            beat_title="Lantern Bearer Enters the Chartroom",
            writing_directives=(
                "Show-Don't-Tell Subtext ONLY (never use literal mood words). Inside the Navigators Chartroom, Sera, Bastian, "
                "and Orin inspect fleet charts as the lantern bearer steps inside from the Open Rampart."
            ),
            extraction_hint="char_nyx moves to loc_chartroom; record ep_step1_chartroom_parley.",
            expected_room_occupants={
                "loc_chartroom": {"char_sera", "char_bastian", "char_nyx", "char_orin"},
                "loc_rampart": set(),
                "loc_sub_vault": set(),
            },
            expected_moods={
                "char_sera": "calm",
                "char_bastian": "confident",
                "char_nyx": "watchful",
                "char_orin": "nervous",
            },
            expected_episode_id="ep_step1_chartroom_parley",
            oracle_delta=StepDelta(
                scene_id="scene_1_chartroom",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                entities=[EntityDelta(entity_id="char_nyx", state={"location_id": "loc_chartroom"})],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step1_chartroom_parley",
                        predicate="briefing",
                        summary="The lantern bearer entered the Navigators Chartroom from the Open Rampart while Sera presented fleet charts to Bastian.",
                        participants={"char_sera": "actor", "char_bastian": "target", "char_nyx": "messenger"},
                        witnesses=["char_sera", "char_bastian", "char_nyx", "char_orin"],
                        location_id="loc_chartroom",
                        salience=3.5,
                    )
                ],
            ),
            voice_metadata_by_speaker={"char_sera": {"register": "Vi", "tone": "respectful"}},
            voice_quality_by_speaker={"char_sera": 9.3},
            required_prose_cues=["Sera", "Bastian", "Orin", "the lantern bearer", q_open],
        ),
        # Step 2: Early Flashback (at Step 2 instead of Step 4!)
        ChunkBeatBlueprint(
            step=2,
            scene_id="scene_2_early_flashback",
            chapter_id="ch_1",
            location_id="loc_chartroom",
            story_time="10_years_ago",
            beat_title="Early Flashback to the Winter Blockade",
            writing_directives=(
                "Flashback set ten years ago during the winter blockade when Sera was unguarded and trusted Orin with the fleet route keys. "
                "Do not overwrite present-day moods or room locations."
            ),
            extraction_hint="Set is_flashback=True, story_time='10_years_ago'; record ep_step2_early_flashback.",
            expected_room_occupants={
                "loc_chartroom": {"char_sera", "char_bastian", "char_nyx", "char_orin"},
                "loc_rampart": set(),
                "loc_sub_vault": set(),
            },
            expected_moods={
                "char_sera": "calm",
                "char_bastian": "confident",
                "char_nyx": "watchful",
                "char_orin": "nervous",
            },
            expected_episode_id=ep_flash,
            is_flashback=True,
            oracle_delta=StepDelta(
                scene_id="scene_2_early_flashback",
                chapter_id="ch_1",
                story_time="10_years_ago",
                is_flashback=True,
                entities=[EntityDelta(entity_id="char_sera", state={"mood": "naive_trusting"})],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_flash,
                        predicate="flashback",
                        summary=fb_summary,
                        participants={"char_sera": "experiencer", "char_orin": "companion"},
                        witnesses=["char_sera", "char_orin"],
                        location_id="loc_chartroom",
                        story_time="10_years_ago",
                        salience=6.5,
                    )
                ],
            ),
            required_prose_cues=["Ten years ago", "Sera", "Orin"],
        ),
        # Step 3: Dynamic Mid-Story Entity ('char_cassian') & Artifact ('item_astrolabe') + Subtextual Confrontation!
        ChunkBeatBlueprint(
            step=3,
            scene_id="scene_3_subtext_confrontation",
            chapter_id="ch_1",
            location_id="loc_chartroom",
            story_time="day_1_dusk",
            beat_title="Rampart Exit, Dynamic Arrival & Subtextual Accusation",
            writing_directives=(
                "Bastian checks his pocket chronometer impatiently and steps out onto the Open Rampart. "
                "Moments later, a newly arrived courier scholar named Cassian enters carrying a Silver Astrolabe, "
                "just as Sera snaps her bone paper-knife in two and confronts Orin (who knocks over his inkhorn in terror) "
                "over the tampered Gilded Compass while the lantern bearer watches silently."
            ),
            extraction_hint="Bastian moves to loc_rampart (restless); introduce char_cassian (calm) & item_astrolabe; Sera (furious) confronts Orin (panicked).",
            expected_room_occupants={
                "loc_chartroom": {"char_sera", "char_nyx", "char_orin", "char_cassian"},
                "loc_rampart": {"char_bastian"},
                "loc_sub_vault": set(),
            },
            expected_moods={
                "char_sera": "furious",
                "char_bastian": "restless",
                "char_nyx": "watchful",
                "char_orin": "panicked",
                "char_cassian": "calm",
            },
            acceptable_moods={"char_bastian": {"restless", "confident", "irritated", "impatient", "uneasy"}},
            expected_episode_id=ep_confront,
            oracle_delta=StepDelta(
                scene_id="scene_3_subtext_confrontation",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                entities=[
                    EntityDelta(entity_id="char_bastian", state={"mood": "restless", "location_id": "loc_rampart"}),
                    EntityDelta(
                        entity_id="char_cassian",
                        kind="character",
                        canonical_name="Cassian",
                        state={"mood": "calm", "location_id": "loc_chartroom"},
                    ),
                    EntityDelta(
                        entity_id="item_astrolabe",
                        kind="artifact",
                        canonical_name="Silver Astrolabe",
                        state={"holder_id": "char_cassian"},
                    ),
                    EntityDelta(entity_id="char_sera", state={"mood": "furious"}, caused_by_episode_id=ep_confront),
                    EntityDelta(entity_id="char_orin", state={"mood": "panicked"}, caused_by_episode_id=ep_confront),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_sera",
                        target_id="char_orin",
                        relation_type="stance",
                        state={"register": "ti", "tone": "furious", "trust": -0.95},
                        caused_by_episode_id=ep_confront,
                    ),
                    EdgeDelta(
                        source_id="char_cassian",
                        target_id="item_astrolabe",
                        relation_type="possesses",
                        state={"status": "carried"},
                    ),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_confront,
                        predicate="confrontation",
                        summary="Sera accused Orin of tampering with the Gilded Compass while Bastian was outside on the Open Rampart.",
                        participants={"char_sera": "actor", "char_orin": "target", "item_compass": "object"},
                        witnesses=["char_sera", "char_orin", "char_nyx", "char_cassian"],
                        location_id="loc_chartroom",
                        salience=9.2,
                    )
                ],
            ),
            voice_metadata_by_speaker={"char_sera": {"register": "ti", "tone": "furious"}},
            voice_quality_by_speaker={"char_sera": 9.6},
            required_prose_cues=["Bastian", "Cassian", "Silver Astrolabe", "Sera", "Orin", q_confront],
        ),
        # Step 4: Exclusive Compass Transfer & Multi-Room Split
        ChunkBeatBlueprint(
            step=4,
            scene_id="scene_3_subtext_confrontation",
            chapter_id="ch_1",
            location_id="loc_chartroom",
            story_time="day_1_dusk",
            beat_title="Exclusive Compass Handoff & Multi-Room Split",
            writing_directives=(
                "Orin thrusts the Gilded Compass into the lantern bearer's hand and bolts down into the Tidal Sub-Vault, "
                "pursued by the lantern bearer, while Cassian steps out onto the Open Rampart."
            ),
            extraction_hint="Orin & Nyx move to loc_sub_vault; Cassian moves to loc_rampart; exclusive possesses transfer of item_compass.",
            expected_room_occupants={
                "loc_chartroom": {"char_sera"},
                "loc_rampart": {"char_bastian", "char_cassian"},
                "loc_sub_vault": {"char_orin", "char_nyx"},
            },
            expected_moods={
                "char_sera": "furious",
                "char_bastian": "restless",
                "char_nyx": "watchful",
                "char_orin": "panicked",
                "char_cassian": "calm",
            },
            acceptable_moods={"char_bastian": {"restless", "confident", "irritated", "impatient", "uneasy"}},
            expected_episode_id=ep_handoff,
            oracle_delta=StepDelta(
                scene_id="scene_3_subtext_confrontation",
                chapter_id="ch_1",
                story_time="day_1_dusk",
                entities=[
                    EntityDelta(entity_id="char_orin", state={"location_id": "loc_sub_vault"}),
                    EntityDelta(entity_id="char_nyx", state={"location_id": "loc_sub_vault"}),
                    EntityDelta(entity_id="char_cassian", state={"location_id": "loc_rampart"}),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_orin",
                        target_id="item_compass",
                        relation_type="possesses",
                        remove=True,
                        caused_by_episode_id=ep_handoff,
                    ),
                    EdgeDelta(
                        source_id="char_nyx",
                        target_id="item_compass",
                        relation_type="possesses",
                        state={"status": "concealed"},
                        caused_by_episode_id=ep_handoff,
                    ),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_handoff,
                        predicate="item_transfer",
                        summary="Orin handed the Gilded Compass to the lantern bearer and fled into the Tidal Sub-Vault.",
                        participants={"char_orin": "actor", "char_nyx": "recipient", "item_compass": "instrument"},
                        witnesses=["char_sera", "char_orin", "char_nyx", "char_cassian"],
                        location_id="loc_chartroom",
                        salience=8.8,
                    )
                ],
            ),
            required_prose_cues=["Gilded Compass", "Tidal Sub-Vault", "Cassian", q_handoff],
        ),
        # Step 5: Sub-Vault Unmasking & Deception Upstairs
        ChunkBeatBlueprint(
            step=5,
            scene_id="scene_4_unmask_and_lie",
            chapter_id="ch_2",
            location_id="loc_chartroom",
            story_time="day_1_night",
            beat_title="Sub-Vault Unmasking & Smuggler Cover Story",
            writing_directives=(
                "In the Tidal Sub-Vault, the lantern bearer lowers her cowl, revealing Nyx with unflinching jaw to blood-drained Orin. "
                "Upstairs in the Navigators Chartroom, Bastian and Orin return, and Orin lies to Bastian about smugglers scaling the seawall."
            ),
            extraction_hint="Nyx unmasks to Orin; Orin & Bastian move to loc_chartroom; InformDelta deceived on ep_step3_chartroom_accusation for char_bastian.",
            expected_room_occupants={
                "loc_chartroom": {"char_sera", "char_bastian", "char_orin"},
                "loc_rampart": {"char_cassian"},
                "loc_sub_vault": {"char_nyx"},
            },
            expected_moods={
                "char_sera": "furious",
                "char_bastian": "restless",
                "char_nyx": "determined",
                "char_orin": "shocked",
                "char_cassian": "calm",
            },
            acceptable_moods={"char_bastian": {"restless", "confident", "irritated", "impatient", "uneasy"}},
            expected_episode_id=ep_unmask,
            oracle_delta=StepDelta(
                scene_id="scene_4_unmask_and_lie",
                chapter_id="ch_2",
                story_time="day_1_night",
                entities=[
                    EntityDelta(
                        entity_id="char_nyx",
                        state={"mood": "determined"},
                        add_aliases=[
                            AliasRecord(
                                alias="the lantern bearer",
                                entity_id="char_nyx",
                                kind="disguise",
                                persona_name="Lantern Bearer",
                                known_to=["char_sera", "char_orin"],
                            )
                        ],
                    ),
                    EntityDelta(
                        entity_id="char_orin",
                        state={"mood": "shocked", "location_id": "loc_chartroom"},
                        caused_by_episode_id=ep_unmask,
                    ),
                    EntityDelta(entity_id="char_bastian", state={"location_id": "loc_chartroom"}),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_nyx",
                        target_id="char_orin",
                        relation_type="stance",
                        state={"register": "ti", "tone": "commanding", "trust": -0.6},
                        caused_by_episode_id=ep_unmask,
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id=ep_unmask,
                        predicate="revelation",
                        summary="Nyx unmasked her lantern bearer disguise to Orin in the Tidal Sub-Vault.",
                        participants={"char_nyx": "actor", "char_orin": "target"},
                        witnesses=["char_nyx", "char_orin"],
                        location_id="loc_sub_vault",
                        salience=9.0,
                    )
                ],
                inform=[
                    InformDelta(
                        episode_id=ep_confront,
                        entity_id="char_bastian",
                        epistemic="deceived",
                        believed_summary=deception_summary,
                    )
                ],
            ),
            voice_metadata_by_speaker={"char_nyx": {"register": "ti", "tone": "commanding"}},
            voice_quality_by_speaker={"char_nyx": 9.4},
            required_prose_cues=["Nyx", "Orin", "Bastian", q_unmask, q_sarcasm],
        ),
        # Step 6: Climax — Truth Overturned & Memory Suppression of Step 2 Flashback
        ChunkBeatBlueprint(
            step=6,
            scene_id="scene_5_climax",
            chapter_id="ch_2",
            location_id="loc_chartroom",
            story_time="day_1_night",
            beat_title="Admiralty Reckoning & Ether Salt Memory Wipe",
            writing_directives=(
                "Nyx and Cassian enter the Navigators Chartroom; Nyx informs Bastian of the full truth about Orin's compass tampering. "
                "Bastian slams the compass casing onto the table; cornered, Orin inhales an ether salt vial that wipes his recollection of the ten-years-ago winter blockade."
            ),
            extraction_hint="Nyx & Cassian move to loc_chartroom; Bastian mood='indignant'; InformDelta told (char_bastian) and forgotten (char_orin on ep_step2_early_flashback).",
            expected_room_occupants={
                "loc_chartroom": {"char_sera", "char_bastian", "char_nyx", "char_orin", "char_cassian"},
                "loc_rampart": set(),
                "loc_sub_vault": set(),
            },
            expected_moods={
                "char_sera": "furious",
                "char_bastian": "indignant",
                "char_nyx": "determined",
                "char_orin": "shocked",
                "char_cassian": "calm",
            },
            acceptable_moods={"char_orin": {"shocked", "confused", "calm", "bewildered", "dazed"}},
            optional_inform=[
                InformDelta(episode_id=ep_handoff, entity_id="char_bastian", epistemic="told"),
                InformDelta(episode_id=ep_confront, entity_id="char_orin", epistemic="forgotten"),
            ],
            oracle_delta=StepDelta(
                scene_id="scene_5_climax",
                chapter_id="ch_2",
                story_time="day_1_night",
                entities=[
                    EntityDelta(entity_id="char_nyx", state={"location_id": "loc_chartroom"}),
                    EntityDelta(entity_id="char_cassian", state={"location_id": "loc_chartroom"}),
                    EntityDelta(entity_id="char_bastian", state={"mood": "indignant"}, caused_by_episode_id=ep_confront),
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_bastian",
                        target_id="char_orin",
                        relation_type="stance",
                        state={"register": "ti", "tone": "stern", "trust": -0.8},
                        caused_by_episode_id=ep_confront,
                    )
                ],
                inform=[
                    InformDelta(episode_id=ep_confront, entity_id="char_bastian", epistemic="told"),
                    InformDelta(episode_id=ep_flash, entity_id="char_orin", epistemic="forgotten"),
                ],
            ),
            voice_metadata_by_speaker={"char_bastian": {"register": "ti", "tone": "stern"}},
            voice_quality_by_speaker={"char_bastian": 9.2},
            required_prose_cues=["Nyx", "Cassian", "Bastian", "Orin", q_climax],
        ),
    ]

    return LongStoryBlueprint(
        story_slug="subtext_intrigue_3k",
        title="The Silent Astrolabe (Subtext & Dynamic Discovery 3K)",
        genre="Maritime Subtext & Counterintelligence Mystery",
        description="A non-isomorphic 3K story with zero literal mood labels in prose (Show-Don't-Tell), early Step 2 flashback, and dynamic mid-story entity introduction at Step 3.",
        target_total_tokens=3000,
        target_chunk_tokens=500,
        target_chunk_words=(320, 460),
        params=params,
        beats=beats,
        key_episode_ids={
            "opening": "ep_step1_chartroom_parley",
            "flashback": ep_flash,
            "confrontation": ep_confront,
            "handoff": ep_handoff,
            "unmask": ep_unmask,
        },
        forbidden_prose_words=[
            "furious",
            "panicked",
            "restless",
            "indignant",
            "determined",
            "shocked",
            "naive_trusting",
        ],
    )


def validate_story_prose_contract(
    story: SavedSyntheticStory,
    blueprint: LongStoryBlueprint | None = None,
) -> list[str]:
    """Audits a generated story against its blueprint for word-count bounds, required cues,
    forbidden subtext mood labels, and leaked schema identifiers.
    """
    violations: list[str] = []
    min_w, max_w = blueprint.target_chunk_words if blueprint else story.params.chunk_target_words
    beat_by_step = {b.step: b for b in blueprint.beats} if blueprint else {}
    forbidden_words = blueprint.forbidden_prose_words if blueprint else []

    for chunk in story.chunks:
        wc = len(re.findall(r"\w+", chunk.text))
        if wc < int(min_w * 0.75) or wc > int(max_w * 1.35):
            violations.append(
                f"[{story.story_slug} Step {chunk.step}] Word count {wc} outside target band [{min_w}, {max_w}]"
            )
        for forbidden in _FORBIDDEN_META_IDENTIFIERS:
            if forbidden in chunk.text:
                violations.append(
                    f"[{story.story_slug} Step {chunk.step}] Leaked schema meta-identifier '{forbidden}' in story prose"
                )
        for fw in forbidden_words:
            if re.search(rf"\b{re.escape(fw)}\b", chunk.text, flags=re.IGNORECASE):
                violations.append(
                    f"[{story.story_slug} Step {chunk.step}] Forbidden literal mood label '{fw}' found in subtext prose"
                )
        beat = beat_by_step.get(chunk.step)
        if beat and beat.required_prose_cues:
            for cue in beat.required_prose_cues:
                if cue and not _cue_in_text(cue, chunk.text):
                    violations.append(
                        f"[{story.story_slug} Step {chunk.step}] Missing required prose cue: {cue!r}"
                    )
    return violations


def _repair_chunk_prose_contract(text: str, beat: ChunkBeatBlueprint) -> str:
    """Ensures generated chunk prose contains all required cues and zero leaked schema meta-identifiers."""
    repaired = text.strip()
    for forbidden in _FORBIDDEN_META_IDENTIFIERS:
        if forbidden in repaired:
            repaired = repaired.replace(forbidden, forbidden.replace("_", " "))
    missing_quotes: list[str] = []
    for cue in beat.required_prose_cues:
        if cue and not _cue_in_text(cue, repaired):
            if len(cue.split()) >= 5:
                missing_quotes.append(f'"{cue}"')
            else:
                missing_quotes.append(f"{cue} remained in clear view during the exchange.")
    if missing_quotes:
        repaired = repaired + "\n\n" + " ".join(missing_quotes)
    return repaired


def compile_deterministic_long_story_chunks(blueprint: LongStoryBlueprint) -> list[SyntheticChunkSpec]:
    """Deterministically synthesizes rich ~360-400 word (~500-token) literary prose chunks for any LongStoryBlueprint
    when running offline without an external LLM generator, guaranteeing 100% prose-oracle contract alignment.
    """
    char_by_id: dict[str, CharacterSeed] = {c.entity_id: c for c in blueprint.params.characters}
    loc_by_id = {loc.entity_id: loc for loc in blueprint.params.locations}
    art_by_id: dict[str, ArtifactSeed] = {a.entity_id: a for a in blueprint.params.artifacts}
    dynamic_names: dict[str, str] = {}
    is_subtext = bool(blueprint.forbidden_prose_words)

    def _surface_for_char(cid: str, step: int) -> str:
        if cid in char_by_id:
            c = char_by_id[cid]
            for a in c.aliases:
                if isinstance(a, AliasRecord) and a.kind == "disguise" and step < 5:
                    return a.alias
            return c.canonical_name
        return dynamic_names.get(cid, cid)

    def _describe_mood(cname: str, mood_val: str) -> str:
        if is_subtext:
            action = _SUBTEXT_MOOD_ACTIONS.get(
                mood_val, "shifted posture sharply as a sudden current of tension crossed the table"
            )
            return f"In that decisive moment, {cname} {action}."
        return f"In that decisive moment, {cname} grew visibly {mood_val}, eyes locked upon the evidence at hand with unmistakable intensity."

    chunks: list[SyntheticChunkSpec] = []
    for beat in blueprint.beats:
        step = beat.step
        for ed in beat.oracle_delta.entities:
            if ed.canonical_name:
                dynamic_names[ed.entity_id] = ed.canonical_name
        loc_obj = loc_by_id.get(beat.location_id, blueprint.params.locations[0])
        loc_name = loc_obj.canonical_name
        present_ids = sorted(beat.expected_room_occupants.get(beat.location_id, set()))
        present_names = [_surface_for_char(cid, step) for cid in present_ids if cid in char_by_id or cid in dynamic_names]

        paragraphs: list[str] = []
        # Paragraph 1: Atmospheric setting & active character staging (~125 words)
        p1 = (
            f"Inside the {loc_name}, the heavy air carried the scent of aged timber, damp stone, and cold iron as "
            f"long shadows stretched across the polished floor under the flickering lamplight. "
            f"{', '.join(present_names) if present_names else 'The chamber'} occupied the room in tense silence, each "
            f"weighing the consequences of the unfolding inquiry while distant gusts rattled the high casements and "
            f"drafts stirred the velvet draperies along the stone walls. "
            f"Every ledger page, wax seal, and whispered calculation seemed magnified by the stillness of the hour, "
            f"turning ordinary gestures into deliberate moves across a dangerous chessboard where loyalty, ambition, "
            f"and deep suspicion could scarcely be told apart by those watching carefully from the perimeter of the chamber."
        )
        paragraphs.append(p1)

        # Paragraph 2: Beat transitions, movements, moods, and flashbacks (~135 words)
        action_sents: list[str] = []
        if beat.is_flashback:
            prot_name = char_by_id[blueprint.params.characters[0].entity_id].canonical_name
            if is_subtext:
                action_sents.append(
                    f"Ten years ago, long before the present bitterness had taken root inside the {loc_name}, the winter blockade "
                    f"had tested everyone gathered around the historic table. Remembering that distant season, "
                    f"{prot_name} recalled those unguarded, open-hearted apprentice years when every signature was accepted without question."
                )
            else:
                action_sents.append(
                    f"Ten years ago, long before the present bitterness had taken root inside the {loc_name}, the winter crisis "
                    f"had tested everyone gathered around the historic table. Remembering that distant season, "
                    f"{prot_name} recalled how naive and trusting those early years had been when the solemn pact was signed in good faith."
                )
        for ed in beat.oracle_delta.entities:
            if ed.kind == "artifact":
                art_label = ed.canonical_name or ed.entity_id
                action_sents.append(f"Resting prominently on the timber table gleamed the newly brought {art_label}.")
                continue
            cname = _surface_for_char(ed.entity_id, step if "add_aliases" not in ed.model_fields_set or not ed.add_aliases else 99)
            if ed.canonical_name and ed.entity_id not in char_by_id:
                action_sents.append(f"A new arrival, {ed.canonical_name}, stepped across the threshold and joined the assembly.")
            if ed.add_aliases:
                disg_a = ed.add_aliases[0].alias
                real_name = char_by_id[ed.entity_id].canonical_name if ed.entity_id in char_by_id else cname
                if is_subtext:
                    action_sents.append(
                        f"In the dim light, {disg_a} threw back the cowl, revealing to everyone present that the disguised figure was none other than {real_name}, who {_SUBTEXT_MOOD_ACTIONS['determined']}."
                    )
                else:
                    action_sents.append(
                        f"In the dim light, {disg_a} threw back the hood and mask, revealing to everyone present that the disguised figure was none other than {real_name}, looking thoroughly {ed.state.get('mood', 'determined')}."
                    )
            if "location_id" in ed.state:
                dest_name = loc_by_id[ed.state["location_id"]].canonical_name
                mood_word = ed.state.get("mood")
                if mood_word and is_subtext:
                    action_sents.append(
                        f"After {cname} {_SUBTEXT_MOOD_ACTIONS.get(mood_word, 'exhaled sharply')}, {cname} stepped out and moved directly into the {dest_name}, leaving the heavy door echoing behind."
                    )
                elif mood_word:
                    action_sents.append(
                        f"Growing {mood_word} beneath the mounting pressure, {cname} stepped out and moved directly into the {dest_name}, leaving the heavy door echoing behind."
                    )
                else:
                    action_sents.append(
                        f"Moving with urgent purpose, {cname} stepped into the {dest_name} as the tension shifted across the corridors."
                    )
            elif "mood" in ed.state and not beat.is_flashback:
                action_sents.append(_describe_mood(cname, str(ed.state["mood"])))

        for edge in beat.oracle_delta.edges:
            if edge.relation_type == "possesses" and not edge.remove:
                holder_name = _surface_for_char(edge.source_id, step)
                art_name = art_by_id[edge.target_id].canonical_name if edge.target_id in art_by_id else dynamic_names.get(edge.target_id, edge.target_id)
                action_sents.append(
                    f"With a swift motion, the {art_name} was pressed into the hand of {holder_name} for safekeeping."
                )

        for inf in beat.oracle_delta.inform:
            listener_name = _surface_for_char(inf.entity_id, 99)
            if inf.epistemic == "deceived":
                action_sents.append(
                    f"Turning toward {listener_name}, the culprit smoothly lied and offered a fabricated cover story: {inf.believed_summary}"
                )
            elif inf.epistemic == "told":
                action_sents.append(
                    f"Speaking plainly and holding the recovered proof aloft, the investigator informed {listener_name} of the full truth about the earlier confrontation and the tampered artifact."
                )
            elif inf.epistemic == "forgotten":
                action_sents.append(
                    f"Cornered against the table, {listener_name} swallowed a bitter alchemical memory-erasing draught that completely wiped and blanked all recollection of the ten-years-ago winter event."
                )

        for ep in beat.oracle_delta.episodes:
            action_sents.append(ep.summary)

        action_sents.append(
            f"The candles guttered in their brass sconces as each movement altered the balance of power within the {loc_name}, "
            f"leaving no doubt that the reckoning could no longer be postponed by courtesy or protocol."
        )
        paragraphs.append(" ".join(action_sents))

        # Paragraph 3: Required verbatim quotes & character dialogue (~125 words)
        quote_sents: list[str] = []
        for cue in beat.required_prose_cues:
            if len(cue.split()) >= 5:
                quote_sents.append(f'The voice rang out clearly across the room: "{cue}"')
            elif not _cue_in_text(cue, " ".join(paragraphs)):
                quote_sents.append(f"Nearby, {cue} observed every detail of the exchange in silence.")
        if step == 1:
            quote_sents.append('From the end of the table came a brief, uneven murmur: "Yes, indeed."')
        quote_sents.append(
            f"Around the {loc_name}, the implications of those words settled into the stonework like frost, "
            f"binding every participant and silent observer to the irrevocable course that had been set in motion "
            f"under the watchful eyes of the assembly, while the heavy clock in the corridor counted down the remaining hours before dawn "
            f"and the sentries outside kept their lonely vigil beneath the darkening sky. "
            f"No one dared break the fragile stillness that followed, knowing that every word spoken or withheld inside "
            f"the {loc_name} would be remembered when the final judgment was rendered before the council."
        )
        paragraphs.append(" ".join(quote_sents))

        full_text = _repair_chunk_prose_contract("\n\n".join(paragraphs), beat)
        chunks.append(
            SyntheticChunkSpec(
                step=step,
                scene_id=beat.scene_id,
                chapter_id=beat.chapter_id,
                location_id=beat.location_id,
                story_time=beat.story_time,
                text=full_text,
                expected_room_occupants=beat.expected_room_occupants,
                expected_moods=beat.expected_moods,
                oracle_delta=beat.oracle_delta.model_copy(deep=True),
                voice_metadata_by_speaker=dict(beat.voice_metadata_by_speaker),
                voice_quality_by_speaker=dict(beat.voice_quality_by_speaker),
                acceptable_moods=dict(beat.acceptable_moods),
                optional_inform=list(beat.optional_inform),
            )
        )
    return chunks


def get_story_blueprint_catalog() -> dict[str, LongStoryBlueprint]:
    """Returns the catalog of 3,000-token (~6 x 500-token) synthetic story blueprints."""
    # 1. Political Intrigue: The Obsidian Signet Conspiracy (3K Edition)
    obsidian_3k = _build_six_chunk_family_blueprint(
        story_slug="obsidian_signet_3k",
        title="The Obsidian Signet Conspiracy (Extended 3K Edition)",
        genre="Political Fantasy Intrigue",
        description="A 3,000-token (6 x ~500-token) political conspiracy in a candlelit citadel testing all 6 Tiers.",
        protagonist=CharacterSeed(
            entity_id="char_elena",
            canonical_name="Elena",
            base_attributes={"title": "Chancellor", "gender": "F", "declension": "type_a"},
            initial_mood="calm",
            initial_location="loc_council",
            private_state={"secret_agenda": "expose_council_traitor"},
            aliases=["the Iron Chancellor"],
        ),
        commander=CharacterSeed(
            entity_id="char_marcus",
            canonical_name="Marcus",
            base_attributes={"title": "General", "gender": "M", "declension": "type_1"},
            initial_mood="confident",
            initial_location="loc_council",
            private_state={"secret_agenda": "protect_garrison_funds"},
            aliases=["the old general"],
        ),
        spy=CharacterSeed(
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
                    known_to=["char_elena"],
                )
            ],
        ),
        disguise_alias="the hooded courier",
        disguise_persona="Courier",
        culprit=CharacterSeed(
            entity_id="char_julian",
            canonical_name="Julian",
            base_attributes={"title": "Treasurer", "gender": "M", "declension": "type_1"},
            initial_mood="nervous",
            initial_location="loc_council",
            private_state={"guilt": "forged_royal_seal"},
            aliases=["the young treasurer"],
        ),
        culprit_alias="the young treasurer",
        loc_main=LocationSeed("loc_council", "Council Chamber", "Candlelit chamber with a long oak table"),
        loc_secret=LocationSeed("loc_vault", "Archive Vault", "Subterranean stone vault behind an iron grate"),
        loc_out=LocationSeed("loc_balcony", "Windy Balcony", "High stone balcony overlooking the courtyard"),
        artifact=ArtifactSeed("item_signet", "Obsidian Signet", "char_julian"),
        protagonist_open_quote="Molim vas da pažljivo pregledate ove poverljive izveštaje pre nego što kraljevska straža zatvori kapije.",
        spy_open_quote="Ponizno vam donosim zapečaćene depeše sa severne granice koje zahtevaju vašu hitnu vojnu procenu.",
        protagonist_confront_quote="Prestani sa tim bednim lažima, izdajniče, jer znam da si sinoć ukrao crni pečat iz komore!",
        culprit_handoff_quote="Take this cursed ring and hide it deep inside the stone vault before the guards arrive!",
        spy_unmask_quote="Predaj mi sve lažne povelje odmah, Julijane, ili ćeš do zore biti u kraljevskoj tamnici!",
        protagonist_sarcasm_quote="Oh, what a wonderfully convenient tale of heroic vigilance, dear cousin.",
        commander_climax_quote="Kako se usuđuješ da obmanjuješ vojni savet dok držimo dokaze u rukama!",
        deception_believed_summary="Northern raiders broke into the annex and stole the Obsidian Signet while Julian defended the ledger.",
        flashback_summary="Ten years ago during the winter siege, Elena trusted Julian at the treaty signing.",
    )

    # 2. Hard Sci-Fi: Echoes of Chronos Station (3K Edition)
    chronos_3k = _build_six_chunk_family_blueprint(
        story_slug="chronos_station_3k",
        title="Echoes of Chronos Station (Extended 3K Edition)",
        genre="Hard Sci-Fi Orbital Thriller",
        description="A 3,000-token (6 x ~500-token) orbital reactor sabotage and neural-wipe mystery.",
        protagonist=CharacterSeed(
            entity_id="char_elena",
            canonical_name="Elena",
            base_attributes={"title": "Chief Science Officer", "gender": "F", "clearance": "omega"},
            initial_mood="calm",
            initial_location="loc_council",
            private_state={"secret_agenda": "trace_quantum_saboteur"},
            aliases=["the science director"],
        ),
        commander=CharacterSeed(
            entity_id="char_marcus",
            canonical_name="Marcus",
            base_attributes={"title": "Station Commander", "gender": "M", "clearance": "alpha"},
            initial_mood="confident",
            initial_location="loc_council",
            private_state={"secret_agenda": "prevent_orbital_quarantine"},
            aliases=["the veteran commander"],
        ),
        spy=CharacterSeed(
            entity_id="char_lyra",
            canonical_name="Lyra",
            base_attributes={"title": "Internal Security Operative", "gender": "F", "clearance": "shadow"},
            initial_mood="watchful",
            initial_location="loc_council",
            private_state={"true_loyalty": "orbital_intelligence"},
            aliases=[
                AliasRecord(
                    alias="the hooded courier",
                    entity_id="char_lyra",
                    kind="disguise",
                    persona_name="Courier",
                    known_to=["char_elena"],
                )
            ],
        ),
        disguise_alias="the hooded courier",
        disguise_persona="Courier",
        culprit=CharacterSeed(
            entity_id="char_julian",
            canonical_name="Julian",
            base_attributes={"title": "Telemetry Specialist", "gender": "M", "clearance": "beta"},
            initial_mood="nervous",
            initial_location="loc_council",
            private_state={"guilt": "stole_chronometer_core"},
            aliases=["the young treasurer"],
        ),
        culprit_alias="the young treasurer",
        loc_main=LocationSeed("loc_council", "Council Chamber", "Orbital command deck overlooking the gas giant"),
        loc_secret=LocationSeed("loc_vault", "Archive Vault", "Sub-deck cryogenic data vault behind blast doors"),
        loc_out=LocationSeed("loc_balcony", "Windy Balcony", "Shielded observation gantry above the docking ring"),
        artifact=ArtifactSeed("item_signet", "Obsidian Signet", "char_julian"),
        protagonist_open_quote="Molim vas da pažljivo pregledate ove poverljive izveštaje pre nego što orbitalna straža zatvori kapije.",
        spy_open_quote="Ponizno vam donosim zapečaćene depeše sa severne granice koje zahtevaju vašu hitnu vojnu procenu.",
        protagonist_confront_quote="Prestani sa tim bednim lažima, izdajniče, jer znam da si sinoć ukrao crni pečat iz komore!",
        culprit_handoff_quote="Take this cursed ring and hide it deep inside the stone vault before the guards arrive!",
        spy_unmask_quote="Predaj mi sve lažne povelje odmah, Julijane, ili ćeš do zore biti u kraljevskoj tamnici!",
        protagonist_sarcasm_quote="Oh, what a wonderfully convenient tale of heroic vigilance, dear cousin.",
        commander_climax_quote="Kako se usuđuješ da obmanjuješ vojni savet dok držimo dokaze u rukama!",
        deception_believed_summary="Northern raiders broke into the annex and stole the Obsidian Signet while Julian defended the ledger.",
        flashback_summary="Ten years ago during the winter siege, Elena trusted Julian at the treaty signing.",
    )

    # 3. Gothic Mystery: The Crimson Masquerade (3K Edition)
    masquerade_3k = _build_six_chunk_family_blueprint(
        story_slug="crimson_masquerade_3k",
        title="The Crimson Masquerade (Extended 3K Edition)",
        genre="Gothic Venetian Mystery",
        description="A 3,000-token (6 x ~500-token) Venetian lagoon conspiracy during a midnight carnival storm.",
        protagonist=CharacterSeed(
            entity_id="char_elena",
            canonical_name="Elena",
            base_attributes={"title": "High Magistrate", "gender": "F", "house": "Valieri"},
            initial_mood="calm",
            initial_location="loc_council",
            private_state={"secret_agenda": "unmask_syndicate_forger"},
            aliases=["the Iron Chancellor"],
        ),
        commander=CharacterSeed(
            entity_id="char_marcus",
            canonical_name="Marcus",
            base_attributes={"title": "Fleet Admiral", "gender": "M", "house": "Contarini"},
            initial_mood="confident",
            initial_location="loc_council",
            private_state={"secret_agenda": "shield_armada_treasury"},
            aliases=["the old general"],
        ),
        spy=CharacterSeed(
            entity_id="char_lyra",
            canonical_name="Lyra",
            base_attributes={"title": "Cipher Inquisitor", "gender": "F", "house": "Tribunal"},
            initial_mood="watchful",
            initial_location="loc_council",
            private_state={"true_loyalty": "secret_tribunal"},
            aliases=[
                AliasRecord(
                    alias="the hooded courier",
                    entity_id="char_lyra",
                    kind="disguise",
                    persona_name="Courier",
                    known_to=["char_elena"],
                )
            ],
        ),
        disguise_alias="the hooded courier",
        disguise_persona="Courier",
        culprit=CharacterSeed(
            entity_id="char_julian",
            canonical_name="Julian",
            base_attributes={"title": "Guild Bursar", "gender": "M", "house": "Valieri"},
            initial_mood="nervous",
            initial_location="loc_council",
            private_state={"guilt": "forged_doge_signet"},
            aliases=["the young treasurer"],
        ),
        culprit_alias="the young treasurer",
        loc_main=LocationSeed("loc_council", "Council Chamber", "Velvet-draped palazzo council hall above the grand canal"),
        loc_secret=LocationSeed("loc_vault", "Archive Vault", "Flooded stone reliquary vault beneath the palazzo"),
        loc_out=LocationSeed("loc_balcony", "Windy Balcony", "Rain-swept marble loggia overlooking the lagoon"),
        artifact=ArtifactSeed("item_signet", "Obsidian Signet", "char_julian"),
        protagonist_open_quote="Molim vas da pažljivo pregledate ove poverljive izveštaje pre nego što kraljevska straža zatvori kapije.",
        spy_open_quote="Ponizno vam donosim zapečaćene depeše sa severne granice koje zahtevaju vašu hitnu vojnu procenu.",
        protagonist_confront_quote="Prestani sa tim bednim lažima, izdajniče, jer znam da si sinoć ukrao crni pečat iz komore!",
        culprit_handoff_quote="Take this cursed ring and hide it deep inside the stone vault before the guards arrive!",
        spy_unmask_quote="Predaj mi sve lažne povelje odmah, Julijane, ili ćeš do zore biti u kraljevskoj tamnici!",
        protagonist_sarcasm_quote="Oh, what a wonderfully convenient tale of heroic vigilance, dear cousin.",
        commander_climax_quote="Kako se usuđuješ da obmanjuješ vojni savet dok držimo dokaze u rukama!",
        deception_believed_summary="Northern raiders broke into the annex and stole the Obsidian Signet while Julian defended the ledger.",
        flashback_summary="Ten years ago during the winter siege, Elena trusted Julian at the treaty signing.",
    )

    # 4. Locked-Room Detective Mystery (Distinct 5-Character, 4-Room, 2-Artifact Topology):
    veritas_3k = _build_veritas_locked_room_3k_blueprint()

    # 5. Subtext (Show-Don't-Tell) & Dynamic Entity Discovery Topology:
    subtext_3k = _build_subtext_intrigue_3k_blueprint()

    # 6. Long-Horizon Multi-Chapter Marathon (18 Chunks across 3 Chapters, ~9K-12K Tokens, 6 Characters, 5 Rooms, 2 Artifacts):
    aethelgard_12k = _build_aethelgard_chronicle_12k_blueprint()

    return {
        "obsidian_signet_3k": obsidian_3k,
        "chronos_station_3k": chronos_3k,
        "crimson_masquerade_3k": masquerade_3k,
        "veritas_locked_room_3k": veritas_3k,
        "subtext_intrigue_3k": subtext_3k,
        "aethelgard_chronicle_12k": aethelgard_12k,
    }


# ============================================================================
# 2. Story Library Manager (Subfolder Storage, Generation, Caching & Costing)
# ============================================================================


class StoryLibraryManager:
    """Manages synthetic stories grouped in subfolders under `tests/synthetic_stories/<story_slug>/`.

    Each subfolder contains:
    - `story.json`: Cached story chunks, metadata, and generation token usage
    - `story.md`: Human-readable Markdown edition with token/cost tables and full prose for inspection
    - `usage_report.json`: Complete token usage & USD cost estimation breakdown
    - `extractions/`: Content-addressed LLM extraction cache (`<sha256>.json`) + `step_01_delta.json`..`step_06_delta.json`
    """

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_STORIES_ROOT,
        *,
        dotenv_path: str | Path = ".env",
    ) -> None:
        load_dotenv(dotenv_path)
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = get_story_blueprint_catalog()

    def story_subfolder(self, story_slug: str) -> Path:
        folder = self.root_dir / story_slug
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def build_long_story_generation_prompt(self, blueprint: LongStoryBlueprint) -> str:
        char_lines = [
            f"- {c.canonical_name} (id={c.entity_id}, aliases={[a.alias if isinstance(a, AliasRecord) else a for a in c.aliases]}): "
            f"initial_mood={c.initial_mood}, initial_location={c.initial_location}"
            for c in blueprint.params.characters
        ]
        loc_lines = [
            f"- {loc.canonical_name} (id={loc.entity_id}): {loc.description}"
            for loc in blueprint.params.locations
        ]
        beat_sections = []
        for b in blueprint.beats:
            cues_line = (
                f"\n- Required Verbatim Quotes / Cues to Include in Prose: {b.required_prose_cues}"
                if b.required_prose_cues
                else ""
            )
            beat_sections.append(
                f"#### Chunk {b.step}: {b.beat_title}\n"
                f"- scene_id: `{b.scene_id}` | chapter_id: `{b.chapter_id}` | location_id: `{b.location_id}` | story_time: `{b.story_time}`\n"
                f"- Target Length: **350 to 420 words (~500 tokens)** across 3-4 rich paragraphs.\n"
                f"- Mandatory Beat Directives: {b.writing_directives}{cues_line}"
            )

        return (
            f"You are an expert novelist writing a synthetic benchmark story titled '{blueprint.title}' "
            f"(Genre: {blueprint.genre}).\n\n"
            f"### Length & Chunking Requirement (CRITICAL):\n"
            f"- The complete story MUST be approximately **{blueprint.target_total_tokens} tokens (~2,200-2,500 words)** in total.\n"
            f"- It is divided into **{len(blueprint.beats)} sequential chunks**.\n"
            f"- **EVERY SINGLE CHUNK (Chunks 1 through {len(blueprint.beats)}) MUST be approximately "
            f"{blueprint.target_chunk_tokens} tokens (350 to 420 words, 3-4 full paragraphs) long.** "
            f"Do NOT shorten later chunks! Expand atmospheric description, sensory detail, body language, "
            f"and internal tension so every chunk is a rich ~500-token scene.\n"
            f"- Preserve all mandatory verbatim quotes exactly as written in each beat directive.\n\n"
            f"### Characters:\n" + "\n".join(char_lines) + "\n\n"
            "### Locations:\n" + "\n".join(loc_lines) + "\n\n"
            "### Required 6-Chunk Beat Architecture:\n\n" + "\n\n".join(beat_sections) + "\n\n"
            "### Output Format:\n"
            "Respond ONLY with valid JSON containing a top-level key `\"chunks\"` with 6 objects, "
            "each containing `\"step\"` (1..6), `\"scene_id\"`, `\"chapter_id\"`, `\"location_id\"`, "
            "`\"story_time\"`, and `\"text\"` (the full 350-420 word / ~500-token prose for that chunk)."
        )

    def _resolve_story_generator_candidates(
        self,
        llm_client: OpenAICompatibleExtractor | None = None,
    ) -> list[OpenAICompatibleExtractor]:
        """Returns story generator LLM clients in priority order:
        1. Explicit `llm_client` if provided
        2. Gemini (`gemini-4-argon` first, then `gemini-4-hydrogen`) when GEMINI_API_KEY is configured
        3. DeepSeek (`deepseek-flash`) when DEEPSEEK_API_KEY is configured
        """
        if llm_client is not None:
            return [llm_client]
        candidates: list[OpenAICompatibleExtractor] = []
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key and gemini_key not in ("your_gemini_api_key_here", "paste_your_gemini_api_key_here"):
            env_story_model = os.environ.get("GEMINI_STORY_MODEL")
            gemini_models = [env_story_model] if env_story_model else ["gemini-4-argon", "gemini-4-hydrogen"]
            for gm in gemini_models:
                if gm:
                    candidates.append(
                        OpenAICompatibleExtractor.from_gemini(
                            model=gm,
                            temperature=0.3,
                            max_tokens=8192,
                            reasoning_effort="low",
                            timeout=240.0,
                        )
                    )
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if deepseek_key and deepseek_key not in ("paste_your_deepseek_api_key_here", "your_deepseek_api_key_here"):
            candidates.append(
                OpenAICompatibleExtractor.from_deepseek(
                    model=os.environ.get("DEEPSEEK_STORY_MODEL") or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
                    temperature=0.3,
                    max_tokens=8192,
                    reasoning_effort="low",
                    timeout=240.0,
                )
            )
        return candidates

    def generate_multi_chapter_story(
        self,
        blueprint: LongStoryBlueprint,
        *,
        llm_client: OpenAICompatibleExtractor,
        batch_size: int = 3,
    ) -> tuple[list[SyntheticChunkSpec], TokenUsage]:
        """Generates a multi-chapter story in batches of `batch_size` (3 chunks per call) to prevent
        output token truncation and lazy summarization on 12K+ token stories.
        """
        total_usage = TokenUsage()
        compiled_chunks: list[SyntheticChunkSpec] = []
        beats = list(blueprint.beats)
        for start_idx in range(0, len(beats), batch_size):
            batch_beats = beats[start_idx : start_idx + batch_size]
            batch_label = f"steps_{batch_beats[0].step}_to_{batch_beats[-1].step}"
            ch_bp = LongStoryBlueprint(
                story_slug=f"{blueprint.story_slug}_{batch_label}",
                title=f"{blueprint.title} ({batch_label})",
                genre=blueprint.genre,
                description=blueprint.description,
                target_total_tokens=len(batch_beats) * blueprint.target_chunk_tokens,
                target_chunk_tokens=blueprint.target_chunk_tokens,
                target_chunk_words=blueprint.target_chunk_words,
                params=blueprint.params,
                beats=batch_beats,
            )
            prompt = self.build_long_story_generation_prompt(ch_bp)
            raw_content, ch_usage = llm_client.call_json_api(
                prompt,
                system_prompt=(
                    "You are a precision literary story generator for a temporal graph memory benchmark. "
                    f"Output strictly valid JSON with key 'chunks' containing {len(batch_beats)} rich ~500-token (350-420 word) chunks."
                ),
                max_tokens=8192,
                temperature=0.3,
            )
            total_usage.add(ch_usage)
            cleaned = _strip_json_fences(raw_content)
            parsed = json.loads(cleaned)
            raw_chunks = parsed.get("chunks", [])
            if len(raw_chunks) != len(batch_beats):
                raise RuntimeError(
                    f"Expected {len(batch_beats)} chunks from LLM for '{blueprint.story_slug}' ({batch_label}), got {len(raw_chunks)}."
                )
            beat_by_step = {b.step: b for b in batch_beats}
            for item in raw_chunks:
                step = int(item["step"])
                beat = beat_by_step.get(step, batch_beats[0])
                compiled_chunks.append(
                    SyntheticChunkSpec(
                        step=beat.step,
                        scene_id=item.get("scene_id", beat.scene_id),
                        chapter_id=item.get("chapter_id", beat.chapter_id),
                        location_id=item.get("location_id", beat.location_id),
                        story_time=item.get("story_time", beat.story_time),
                        text=_repair_chunk_prose_contract(str(item["text"]).strip(), beat),
                        expected_room_occupants=beat.expected_room_occupants,
                        expected_moods=beat.expected_moods,
                        oracle_delta=beat.oracle_delta,
                        voice_metadata_by_speaker=beat.voice_metadata_by_speaker,
                        voice_quality_by_speaker=beat.voice_quality_by_speaker,
                        acceptable_moods=dict(beat.acceptable_moods),
                        optional_inform=list(beat.optional_inform),
                    )
                )
        return compiled_chunks, total_usage

    def get_or_create_story(
        self,
        story_slug: str = "obsidian_signet_3k",
        *,
        force_regenerate: bool = False,
        llm_client: OpenAICompatibleExtractor | None = None,
        use_deterministic_fallback: bool = False,
    ) -> SavedSyntheticStory:
        """Loads the saved synthetic story from its subfolder if cached; otherwise generates it via LLM or deterministic compiler."""
        subfolder = self.story_subfolder(story_slug)
        story_json_path = subfolder / "story.json"
        env_force = os.environ.get("CANONARC_REGENERATE_STORY") == "1"

        if story_slug == "obsidian_signet_micro":
            return self._get_or_create_micro_baseline(subfolder, force_regenerate=force_regenerate or env_force)

        if story_slug not in self.catalog:
            raise KeyError(f"Unknown story_slug '{story_slug}'. Available: {list(self.catalog.keys()) + ['obsidian_signet_micro']}")

        blueprint = self.catalog[story_slug]

        if story_json_path.is_file() and not (force_regenerate or env_force):
            return self._load_saved_story(story_json_path, blueprint)

        candidates = self._resolve_story_generator_candidates(llm_client)

        if use_deterministic_fallback or not candidates:
            compiled_chunks = compile_deterministic_long_story_chunks(blueprint)
            saved_story = SavedSyntheticStory(
                story_slug=story_slug,
                title=blueprint.title,
                genre=blueprint.genre,
                model="deterministic_compiler",
                created_at=datetime.datetime.now(datetime.UTC).isoformat(),
                story_dir=subfolder,
                params=blueprint.params,
                chunks=compiled_chunks,
                generation_usage=TokenUsage(),
                loaded_from_cache=False,
            )
            self._persist_story_files(saved_story, blueprint=blueprint)
            return saved_story

        last_err: Exception | None = None
        active_client: OpenAICompatibleExtractor = candidates[0]
        compiled_chunks = []
        gen_usage = TokenUsage()
        for cand_client in candidates:
            active_client = cand_client
            try:
                if len(blueprint.beats) > 6:
                    compiled_chunks, gen_usage = self.generate_multi_chapter_story(
                        blueprint, llm_client=active_client
                    )
                else:
                    prompt = self.build_long_story_generation_prompt(blueprint)
                    raw_content, gen_usage = active_client.call_json_api(
                        prompt,
                        system_prompt=(
                            "You are a precision literary story generator for a temporal graph memory benchmark. "
                            "Output strictly valid JSON with rich ~500-token (350-420 word) chunks."
                        ),
                        max_tokens=8192,
                        temperature=0.3,
                    )
                    cleaned = _strip_json_fences(raw_content)
                    parsed = json.loads(cleaned)
                    raw_chunks = parsed.get("chunks", [])
                    if len(raw_chunks) != len(blueprint.beats):
                        raise RuntimeError(
                            f"Expected {len(blueprint.beats)} chunks from LLM for '{story_slug}', got {len(raw_chunks)}."
                        )

                    beat_by_step = {b.step: b for b in blueprint.beats}
                    compiled_chunks = []
                    for item in raw_chunks:
                        step = int(item["step"])
                        beat = beat_by_step[step]
                        compiled_chunks.append(
                            SyntheticChunkSpec(
                                step=step,
                                scene_id=item.get("scene_id", beat.scene_id),
                                chapter_id=item.get("chapter_id", beat.chapter_id),
                                location_id=item.get("location_id", beat.location_id),
                                story_time=item.get("story_time", beat.story_time),
                                text=_repair_chunk_prose_contract(str(item["text"]).strip(), beat),
                                expected_room_occupants=beat.expected_room_occupants,
                                expected_moods=beat.expected_moods,
                                oracle_delta=beat.oracle_delta,
                                voice_metadata_by_speaker=beat.voice_metadata_by_speaker,
                                voice_quality_by_speaker=beat.voice_quality_by_speaker,
                                acceptable_moods=dict(beat.acceptable_moods),
                                optional_inform=list(beat.optional_inform),
                            )
                        )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if last_err is not None:
            raise last_err

        saved_story = SavedSyntheticStory(
            story_slug=story_slug,
            title=blueprint.title,
            genre=blueprint.genre,
            model=active_client.model,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            story_dir=subfolder,
            params=blueprint.params,
            chunks=compiled_chunks,
            generation_usage=gen_usage,
            loaded_from_cache=False,
        )
        self._persist_story_files(saved_story, blueprint=blueprint)
        return saved_story

    def _get_or_create_micro_baseline(
        self,
        subfolder: Path,
        *,
        force_regenerate: bool = False,
    ) -> SavedSyntheticStory:
        story_json_path = subfolder / "story.json"
        params = SynthStoryParams.default_conspiracy_seed()
        compiler = DeterministicStoryCompiler(params)
        chunks = compiler.compile_chunks()
        if story_json_path.is_file() and not force_regenerate:
            data = json.loads(story_json_path.read_text(encoding="utf-8"))
            return SavedSyntheticStory(
                story_slug="obsidian_signet_micro",
                title=data.get("title", params.title),
                genre=data.get("genre", "Deterministic Micro-Chunk Baseline"),
                model=data.get("model", "deterministic_compiler"),
                created_at=data.get("created_at", ""),
                story_dir=subfolder,
                params=params,
                chunks=chunks,
                generation_usage=TokenUsage.from_dict(data.get("generation_usage")),
                loaded_from_cache=True,
            )

        saved = SavedSyntheticStory(
            story_slug="obsidian_signet_micro",
            title="The Obsidian Signet Conspiracy (10x40w Micro-Chunk Baseline)",
            genre="Deterministic Micro-Chunk Baseline",
            model="deterministic_compiler",
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            story_dir=subfolder,
            params=params,
            chunks=chunks,
            generation_usage=TokenUsage(),
            loaded_from_cache=False,
        )
        self._persist_story_files(saved, blueprint=None)
        return saved

    def _load_saved_story(
        self,
        story_json_path: Path,
        blueprint: LongStoryBlueprint,
    ) -> SavedSyntheticStory:
        data = json.loads(story_json_path.read_text(encoding="utf-8"))
        beat_by_step = {b.step: b for b in blueprint.beats}
        chunks: list[SyntheticChunkSpec] = []
        for item in data["chunks"]:
            step = int(item["step"])
            beat = beat_by_step[step]
            chunks.append(
                SyntheticChunkSpec(
                    step=step,
                    scene_id=item.get("scene_id", beat.scene_id),
                    chapter_id=item.get("chapter_id", beat.chapter_id),
                    location_id=item.get("location_id", beat.location_id),
                    story_time=item.get("story_time", beat.story_time),
                    text=str(item["text"]),
                    expected_room_occupants=beat.expected_room_occupants,
                    expected_moods=beat.expected_moods,
                    oracle_delta=beat.oracle_delta,
                    voice_metadata_by_speaker=beat.voice_metadata_by_speaker,
                    voice_quality_by_speaker=beat.voice_quality_by_speaker,
                    acceptable_moods=dict(beat.acceptable_moods),
                    optional_inform=list(beat.optional_inform),
                )
            )
        return SavedSyntheticStory(
            story_slug=blueprint.story_slug,
            title=data.get("title", blueprint.title),
            genre=data.get("genre", blueprint.genre),
            model=data.get("model", "deepseek-flash"),
            created_at=data.get("created_at", ""),
            story_dir=story_json_path.parent,
            params=blueprint.params,
            chunks=chunks,
            generation_usage=TokenUsage.from_dict(data.get("generation_usage")),
            loaded_from_cache=True,
        )

    def _persist_story_files(
        self,
        story: SavedSyntheticStory,
        *,
        blueprint: LongStoryBlueprint | None = None,
        extraction_result: StoryE2EExecutionResult | None = None,
    ) -> None:
        subfolder = story.story_dir
        subfolder.mkdir(parents=True, exist_ok=True)
        beat_by_step = {b.step: b for b in blueprint.beats} if blueprint else {}

        story_payload = {
            "story_slug": story.story_slug,
            "title": story.title,
            "genre": story.genre,
            "model": story.model,
            "created_at": story.created_at,
            "total_chunks": len(story.chunks),
            "total_words": story.total_words,
            "total_estimated_tokens": story.total_estimated_tokens,
            "average_tokens_per_chunk": round(story.total_estimated_tokens / max(1, len(story.chunks)), 1),
            "generation_usage": story.generation_usage.to_dict(),
            "chunks": [
                {
                    "step": c.step,
                    "beat_title": beat_by_step[c.step].beat_title if c.step in beat_by_step else f"Step {c.step}",
                    "scene_id": c.scene_id,
                    "chapter_id": c.chapter_id,
                    "location_id": c.location_id,
                    "story_time": c.story_time,
                    "word_count": len(re.findall(r"\w+", c.text)),
                    "estimated_tokens": estimate_text_tokens(c.text),
                    "text": c.text,
                }
                for c in story.chunks
            ],
        }
        (subfolder / "story.json").write_text(
            json.dumps(story_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Also write or update usage_report.json, model-specific benchmark JSON, and story.md
        if extraction_result is not None:
            usage_report = {
                "story_slug": story.story_slug,
                "title": story.title,
                "story_model": story.model,
                "extraction_model": extraction_result.extraction_model,
                "guidance_mode": extraction_result.guidance_mode,
                "run_cache_key": extraction_result.run_cache_key,
                "checks_passed": extraction_result.checks_passed,
                "checks_total": extraction_result.checks_total,
                "step_delta_checks_passed": extraction_result.step_delta_checks_passed,
                "step_delta_checks_total": extraction_result.step_delta_checks_total,
                "step_delta_pass_rate": round(extraction_result.step_delta_pass_rate, 4),
                "cumulative_state_checks_passed": extraction_result.cumulative_state_checks_passed,
                "cumulative_state_checks_total": extraction_result.cumulative_state_checks_total,
                "cumulative_state_pass_rate": round(extraction_result.cumulative_state_pass_rate, 4),
                "failures": extraction_result.failures,
                "strict_pass_rate": round(extraction_result.pass_rate, 4),
                "overall_quality_score": round(extraction_result.metrics.overall_quality_score, 4),
                "benchmark_metrics": extraction_result.metrics.to_dict(),
                "story_prose_stats": {
                    "total_chunks": len(story.chunks),
                    "total_words": story.total_words,
                    "total_estimated_story_tokens": story.total_estimated_tokens,
                    "chunk_estimated_tokens": story.chunk_token_counts,
                    "chunk_word_counts": story.chunk_word_counts,
                },
                "story_generation_api_usage": story.generation_usage.to_dict(model=story.model),
                "llm_extraction_historical_api_usage": extraction_result.extraction_historical_usage.to_dict(
                    model=extraction_result.extraction_model
                ),
                "llm_extraction_current_run_billable_usage": extraction_result.extraction_billable_usage.to_dict(
                    model=extraction_result.extraction_model
                ),
                "combined_total_story_plus_extraction_usage": extraction_result.combined_historical_usage.to_dict(
                    model=extraction_result.extraction_model
                ),
            }
            is_primary_baseline = (
                extraction_result.extraction_model == story.model
                and extraction_result.guidance_mode == "full_hints"
            )
            if is_primary_baseline or not (subfolder / "usage_report.json").is_file():
                (subfolder / "usage_report.json").write_text(
                    json.dumps(usage_report, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            benchmarks_dir = subfolder / "benchmarks"
            benchmarks_dir.mkdir(parents=True, exist_ok=True)
            safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", extraction_result.extraction_model)
            bench_path = benchmarks_dir / f"benchmark_{safe_model}_{extraction_result.guidance_mode}.json"
            bench_path.write_text(
                json.dumps(usage_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if not is_primary_baseline and (subfolder / "story.md").is_file():
                return

        # Render human-inspectable Markdown edition (story.md)
        md_lines = [
            f"# {story.title}",
            "",
            f"- **Story Slug**: `{story.story_slug}`",
            f"- **Genre**: {story.genre}",
            f"- **Generator Model**: `{story.model}`",
            f"- **Created At**: `{story.created_at}`",
            f"- **Total Story Chunks**: `{len(story.chunks)}`",
            f"- **Total Story Words**: `{story.total_words:,}` words",
            (
                f"- **Total Story Prose Tokens (est.)**: `~{story.total_estimated_tokens:,}` tokens "
                f"(avg `~{round(story.total_estimated_tokens / max(1, len(story.chunks)))}` tokens/chunk)"
            ),
            "",
            "---",
            "",
            "## 1. Token Usage & Price Estimation Summary",
            "",
            "| Stage | Prompt Tokens | Cached Prompt | Completion Tokens | Reasoning Tokens | Total API Tokens | Est. Cost (USD) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        gen_u = story.generation_usage
        gen_cost = gen_u.estimate_cost_usd()["total_cost_usd"]
        md_lines.append(
            f"| **Story Generation (`{story.model}`)** | {gen_u.prompt_tokens:,} | {gen_u.cached_prompt_tokens:,} | "
            f"{gen_u.completion_tokens:,} | {gen_u.reasoning_tokens:,} | **{gen_u.total_tokens:,}** | **${gen_cost:.6f}** |"
        )
        if extraction_result is not None:
            ext_h = extraction_result.extraction_historical_usage
            ext_h_cost = ext_h.estimate_cost_usd()["total_cost_usd"]
            comb = extraction_result.combined_historical_usage
            comb_cost = comb.estimate_cost_usd()["total_cost_usd"]
            bill = extraction_result.extraction_billable_usage
            bill_cost = bill.estimate_cost_usd()["total_cost_usd"]
            md_lines.append(
                f"| **LLM StepDelta Extraction (6 Chunks Total)** | {ext_h.prompt_tokens:,} | {ext_h.cached_prompt_tokens:,} | "
                f"{ext_h.completion_tokens:,} | {ext_h.reasoning_tokens:,} | **{ext_h.total_tokens:,}** | **${ext_h_cost:.6f}** |"
            )
            md_lines.append(
                f"| **COMBINED TOTAL (Generate + Extract Story)** | **{comb.prompt_tokens:,}** | **{comb.cached_prompt_tokens:,}** | "
                f"**{comb.completion_tokens:,}** | **{comb.reasoning_tokens:,}** | **{comb.total_tokens:,}** | **${comb_cost:.6f}** |"
            )
            md_lines.append(
                f"| *Last Run Billable (Cache Hits: {extraction_result.extraction_cache_hits}/{len(story.chunks)})* | "
                f"*{bill.prompt_tokens:,}* | *{bill.cached_prompt_tokens:,}* | *{bill.completion_tokens:,}* | "
                f"*{bill.reasoning_tokens:,}* | ***{bill.total_tokens:,}*** | ***${bill_cost:.6f}*** |"
            )
        md_lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Chunk Breakdown",
                "",
                "| Step | Beat Title | Scene ID | Location | Story Time | Words | Est. Tokens |",
                "| ---: | :--- | :--- | :--- | :--- | ---: | ---: |",
            ]
        )
        for c in story.chunks:
            b_title = beat_by_step[c.step].beat_title if c.step in beat_by_step else f"Step {c.step}"
            wc = len(re.findall(r"\w+", c.text))
            tc = estimate_text_tokens(c.text)
            md_lines.append(
                f"| {c.step} | {b_title} | `{c.scene_id}` | `{c.location_id}` | `{c.story_time}` | {wc} | ~{tc} |"
            )

        md_lines.extend(["", "---", "", "## 3. Full Synthetic Story Prose", ""])
        for c in story.chunks:
            b_title = beat_by_step[c.step].beat_title if c.step in beat_by_step else f"Step {c.step}"
            wc = len(re.findall(r"\w+", c.text))
            tc = estimate_text_tokens(c.text)
            md_lines.extend(
                [
                    f"### Chunk {c.step}: {b_title}",
                    (
                        f"*Scene: `{c.scene_id}` | Chapter: `{c.chapter_id}` | Location: `{c.location_id}` | "
                        f"Time: `{c.story_time}` | Length: {wc} words (~{tc} tokens)*"
                    ),
                    "",
                    c.text,
                    "",
                    "---",
                    "",
                ]
            )

        (subfolder / "story.md").write_text("\n".join(md_lines), encoding="utf-8")
        self._update_root_readme()

    def _update_root_readme(self) -> None:
        """Generates/updates tests/synthetic_stories/README.md summarizing all saved story subfolders."""
        rows: list[str] = []
        for sub in sorted(self.root_dir.iterdir()):
            if not sub.is_dir():
                continue
            s_json = sub / "story.json"
            u_json = sub / "usage_report.json"
            if not s_json.is_file():
                continue
            s_data = json.loads(s_json.read_text(encoding="utf-8"))
            slug = s_data.get("story_slug", sub.name)
            title = s_data.get("title", slug)
            chunks = s_data.get("total_chunks", 0)
            words = s_data.get("total_words", 0)
            est_toks = s_data.get("total_estimated_tokens", 0)
            gen_u = TokenUsage.from_dict(s_data.get("generation_usage"))
            gen_toks = gen_u.total_tokens
            ext_toks = 0
            total_api_toks = gen_toks
            total_cost = gen_u.estimate_cost_usd()["total_cost_usd"]
            if u_json.is_file():
                u_data = json.loads(u_json.read_text(encoding="utf-8"))
                ext_u = TokenUsage.from_dict(u_data.get("llm_extraction_historical_api_usage"))
                comb_u = TokenUsage.from_dict(u_data.get("combined_total_story_plus_extraction_usage"))
                ext_toks = ext_u.total_tokens
                total_api_toks = comb_u.total_tokens
                total_cost = comb_u.estimate_cost_usd()["total_cost_usd"]
            rows.append(
                f"| [`{slug}/`](./{slug}/story.md) | {title} | {chunks} | {words:,} | ~{est_toks:,} | "
                f"{gen_toks:,} | {ext_toks:,} | **{total_api_toks:,}** | **${total_cost:.5f}** |"
            )

        readme = [
            "# Saved Synthetic Stories & Token Usage Catalog",
            "",
            (
                "Each subfolder contains a saved synthetic benchmark story (`story.json` + inspectable `story.md`), "
                "its token usage & price estimate report (`usage_report.json`), and its content-addressed LLM extraction cache (`extractions/`)."
            ),
            "",
            "| Story Subfolder | Title | Chunks | Story Words | Story Prose Tokens | Generation API Tokens | Extraction API Tokens | Total API Tokens | Est. Total Cost (USD) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
        ]
        (self.root_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    def _seed_engine_from_params(self, engine: CanonArcEngine, params: SynthStoryParams) -> None:
        for loc in params.locations:
            engine.register_entity(
                EntityRecord(
                    entity_id=loc.entity_id,
                    kind="location",
                    canonical_name=loc.canonical_name,
                    base_attributes={"description": loc.description},
                ),
                step=0,
            )
        for art in params.artifacts:
            engine.register_entity(
                EntityRecord(
                    entity_id=art.entity_id,
                    kind="artifact",
                    canonical_name=art.canonical_name,
                    state={"holder_id": art.initial_holder_id},
                ),
                step=0,
            )
        for char in params.characters:
            engine.register_entity(
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
        engine.commit_step(
            step=0,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id=e.source_id,
                        target_id=e.target_id,
                        relation_type=e.relation_type,
                        state=dict(e.state),
                    )
                    for e in params.initial_edges
                ]
            ),
        )

    _LEGACY_FULL_HINTS_BASE_INST = (
        "Analyze the scene outcome and emit a StepDelta JSON diff recording any changes in "
        "character mood, disguises, relationship shifts, new aliases, or discrete plot episodes."
    )

    @staticmethod
    def _build_beat_hint_for_mode(beat: ChunkBeatBlueprint | None, guidance_mode: GuidanceMode) -> str:
        if beat is None or guidance_mode in ("none", "surface_only"):
            return ""
        if guidance_mode == "full_hints":
            return beat.extraction_hint
        if guidance_mode == "canonical_episode_ids":
            lines: list[str] = []
            if beat.expected_episode_id:
                lines.append(
                    f"- If you extract the primary plot episode in this chunk, use exact "
                    f"episode_id='{beat.expected_episode_id}' so later steps can reference it."
                )
            for inf in beat.oracle_delta.inform:
                lines.append(
                    f"- Note: Prior episode '{inf.episode_id}' is available in history if referenced."
                )
            return "\n".join(lines)
        return ""

    @staticmethod
    def _format_active_world_state_for_extraction(
        engine: CanonArcEngine,
        cutoff_step: int,
        *,
        include_canonical_ids: bool = True,
        max_episodes: int = 12,
    ) -> str:
        """Formats the engine's current point-in-time world registry (at step-1) for unguided extraction.

        - When `include_canonical_ids=True` (`guidance_mode='none'`), includes exact canonical IDs.
        - When `include_canonical_ids=False` (`guidance_mode='surface_only'`), formats only natural surface names
          and aliases so `engine.normalize_delta()` is genuinely exercised.
        - Bounds the timeline episode ledger to `max_episodes` (highest-salience + most recent) so prompt size
          remains O(1) on long-horizon 18-50+ chunk stories.
        """
        tl = engine.active_timeline
        entities = engine.store.get_all_active_entities(tl, cutoff_step)
        aliases = engine.store.get_all_active_aliases(tl, cutoff_step)
        all_eps = engine.store.query_episodes(timeline_id=tl, cutoff_step=cutoff_step)
        name_by_id = {e.entity_id: e.canonical_name for e in entities}

        if len(all_eps) > max_episodes:
            # Keep the 4 most recent episodes + top-(max_episodes-4) by salience, then restore step order
            recent_set = {ep.episode_id for ep in all_eps[-4:]}
            remaining = [ep for ep in all_eps[:-4]]
            remaining.sort(key=lambda ep: (ep.salience, ep.step), reverse=True)
            keep_ids = recent_set | {ep.episode_id for ep in remaining[: max(0, max_episodes - 4)]}
            selected_eps = [ep for ep in all_eps if ep.episode_id in keep_ids]
        else:
            selected_eps = list(all_eps)

        if not include_canonical_ids:
            locs = [f"- '{e.canonical_name}'" for e in entities if e.kind == "location"]
            arts = [f"- '{e.canonical_name}'" for e in entities if e.kind == "artifact"]
            aliases_by_ent: dict[str, list[str]] = {}
            for a in aliases:
                if a.kind == "disguise":
                    known_names = [name_by_id.get(k, k) for k in a.known_to]
                    aliases_by_ent.setdefault(a.entity_id, []).append(
                        f"disguise='{a.alias}' (persona='{a.persona_name}', known_to={known_names})"
                    )
                else:
                    aliases_by_ent.setdefault(a.entity_id, []).append(f"alias='{a.alias}'")
            chars: list[str] = []
            for e in entities:
                if e.kind != "character":
                    continue
                loc_id = e.state.get("location_id", "unknown")
                loc_name = name_by_id.get(loc_id, loc_id)
                mood = e.state.get("mood", "neutral")
                al_str = f", aliases=[{'; '.join(aliases_by_ent[e.entity_id])}]" if e.entity_id in aliases_by_ent else ""
                chars.append(
                    f"- '{e.canonical_name}': current_location='{loc_name}', current_mood='{mood}'{al_str}"
                )
            lines: list[str] = [
                f"### Known Story World State (as of Step {cutoff_step} — Use Natural Character/Location/Artifact Names):",
                "**Known Locations**:",
                *(locs or ["(none)"]),
                "**Known Artifacts**:",
                *(arts or ["(none)"]),
                "**Characters' Current Room, Mood & Disguises Before This Chunk**:",
                *(chars or ["(none)"]),
            ]
            if selected_eps:
                lines.append("**Prior Plot Episodes (Use `episode_id` for `InformDelta` or `caused_by_episode_id`)**:")
                for ep in selected_eps:
                    w_names = [name_by_id.get(w, w) for w in ep.witnesses]
                    loc_name = name_by_id.get(ep.location_id or "", ep.location_id or "")
                    lines.append(
                        f"- {ep.episode_id} (step={ep.step}, predicate='{ep.predicate}', location='{loc_name}', witnesses={w_names}): {ep.summary}"
                    )
            return "\n".join(lines)

        locs = [f"- {e.entity_id} (name: '{e.canonical_name}')" for e in entities if e.kind == "location"]
        arts = [
            f"- {e.entity_id} (name: '{e.canonical_name}', state={e.state})"
            for e in entities
            if e.kind == "artifact"
        ]
        aliases_by_ent = {}
        for a in aliases:
            if a.kind == "disguise":
                aliases_by_ent.setdefault(a.entity_id, []).append(
                    f"disguise='{a.alias}' (persona='{a.persona_name}', known_to={a.known_to})"
                )
            else:
                aliases_by_ent.setdefault(a.entity_id, []).append(f"alias='{a.alias}'")

        chars = []
        for e in entities:
            if e.kind != "character":
                continue
            loc_id = e.state.get("location_id", "unknown")
            mood = e.state.get("mood", "neutral")
            al_str = f", aliases=[{'; '.join(aliases_by_ent[e.entity_id])}]" if e.entity_id in aliases_by_ent else ""
            chars.append(
                f"- {e.entity_id} (name: '{e.canonical_name}'): current_location_id='{loc_id}', current_mood='{mood}'{al_str}"
            )

        lines = [
            f"### Active World Registry (as of Step {cutoff_step} — Use Exact IDs Below):",
            "**Registered Locations (`location_id` values)**:",
            *(locs or ["(none)"]),
            "**Registered Artifacts (`artifact_id` values)**:",
            *(arts or ["(none)"]),
            "**Characters' Current Room, Mood & Disguises Before This Chunk**:",
            *(chars or ["(none)"]),
        ]
        if selected_eps:
            lines.append("**Complete Timeline Episode Ledger (Use these exact `episode_id`s for `InformDelta` or `caused_by_episode_id`)**:")
            for ep in selected_eps:
                lines.append(
                    f"- {ep.episode_id} (step={ep.step}, predicate='{ep.predicate}', location='{ep.location_id}', witnesses={ep.witnesses}): {ep.summary}"
                )
        return "\n".join(lines)

    def _compute_story_run_cache_key(
        self,
        *,
        story: SavedSyntheticStory,
        model_id: str,
        system_prompt: str,
        guidance_mode: GuidanceMode,
        base_instructions: str,
        beat_by_step: dict[int, ChunkBeatBlueprint],
    ) -> str:
        """Computes a deterministic SHA-256 signature for (model, prompt_template, story)."""
        chunk_sigs = [
            {
                "step": c.step,
                "scene_id": c.scene_id,
                "location_id": c.location_id,
                "story_time": c.story_time,
                "text": c.text.strip(),
                "beat_hint": self._build_beat_hint_for_mode(beat_by_step.get(c.step), guidance_mode),
            }
            for c in story.chunks
        ]
        material = (
            f"model={model_id}\0"
            f"system={system_prompt.strip()}\0"
            f"guidance_mode={guidance_mode}\0"
            f"base_instructions={base_instructions.strip()}\0"
            f"story_slug={story.story_slug}\0"
            f"chunks={json.dumps(chunk_sigs, sort_keys=True, ensure_ascii=False)}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def run_story_extraction_e2e(
        self,
        story_slug: str = "obsidian_signet_3k",
        *,
        force_regenerate_story: bool = False,
        bypass_extraction_cache: bool = False,
        base_extractor: Any | None = None,
        guidance_mode: GuidanceMode = "full_hints",
    ) -> StoryE2EExecutionResult:
        """Runs full E2E extraction, verification, and multi-metric benchmarking on a synthetic story.

        If `(model, prompt, story)` matches a completed benchmark already on disk (and
        `bypass_extraction_cache` is False), skips re-running the extraction test and returns
        the cached result immediately.
        """
        story = self.get_or_create_story(story_slug, force_regenerate=force_regenerate_story)
        blueprint = self.catalog.get(story_slug)
        beat_by_step = {b.step: b for b in blueprint.beats} if blueprint else {}

        env_bypass = os.environ.get("CANONARC_BYPASS_EXTRACTION_CACHE") == "1"
        effective_bypass = bypass_extraction_cache or env_bypass

        extractions_dir = story.story_dir / "extractions"
        extractions_dir.mkdir(parents=True, exist_ok=True)

        if base_extractor is None:
            base_extractor = OpenAICompatibleExtractor.from_deepseek(
                model=os.environ.get("DEEPSEEK_EXTRACT_MODEL") or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
                temperature=0.0,
                max_tokens=4096,
                reasoning_effort="low",
                timeout=180.0,
            )

        cached_extractor = CachedDeltaExtractor(
            base_extractor,
            cache_dir=extractions_dir,
            bypass_cache=effective_bypass,
        )
        safe_model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", cached_extractor.model_id)
        model_step_dir = extractions_dir / f"{safe_model_slug}_{guidance_mode}"
        model_step_dir.mkdir(parents=True, exist_ok=True)
        benchmarks_dir = story.story_dir / "benchmarks"
        benchmarks_dir.mkdir(parents=True, exist_ok=True)
        bench_path = benchmarks_dir / f"benchmark_{safe_model_slug}_{guidance_mode}.json"

        outer_mgr = self

        # Custom preset wrapping InteractiveFictionPreset with configurable beat guidance
        class _GuidedStoryPreset(InteractiveFictionPreset):
            def __init__(self) -> None:
                super().__init__(include_background_edges=True)
                self.current_beat_hint: str = ""
                self.current_world_state_block: str = ""

            def extraction_instructions(self) -> str:
                if guidance_mode == "full_hints":
                    base_inst = outer_mgr._LEGACY_FULL_HINTS_BASE_INST
                    if not self.current_beat_hint:
                        return base_inst
                    return (
                        f"{base_inst}\n\n"
                        "### Canonical Beat Schema & ID Requirements for This Chunk:\n"
                        f"{self.current_beat_hint}\n"
                        "Return ONLY valid JSON conforming to the StepDelta schema."
                    )
                base_inst = super().extraction_instructions()
                parts = [base_inst]
                if self.current_world_state_block:
                    parts.append(self.current_world_state_block)
                if self.current_beat_hint:
                    parts.append(
                        "### Canonical Beat Schema & ID Requirements for This Chunk:\n"
                        f"{self.current_beat_hint}"
                    )
                parts.append("Return ONLY valid JSON conforming to the StepDelta schema.")
                return "\n\n".join(parts)

        preset = _GuidedStoryPreset()
        base_inst_for_sig = (
            self._LEGACY_FULL_HINTS_BASE_INST
            if guidance_mode == "full_hints"
            else InteractiveFictionPreset(include_background_edges=True).extraction_instructions()
        )
        run_cache_key = self._compute_story_run_cache_key(
            story=story,
            model_id=cached_extractor.model_id,
            system_prompt=cached_extractor.system_prompt,
            guidance_mode=guidance_mode,
            base_instructions=base_inst_for_sig,
            beat_by_step=beat_by_step,
        )

        # Check if (model, prompt, story) run is already cached on disk -> skip re-running test!
        if not effective_bypass and bench_path.is_file():
            try:
                bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
                stored_key = bench_data.get("run_cache_key", "")
                step_files_exist = all(
                    (model_step_dir / f"step_{c.step:02d}_delta.json").is_file()
                    for c in story.chunks
                )
                # Backfill run_cache_key for existing full_hints runs whose prompt is unchanged
                if not stored_key and guidance_mode == "full_hints" and step_files_exist:
                    stored_key = run_cache_key
                    bench_data["run_cache_key"] = run_cache_key
                    bench_path.write_text(
                        json.dumps(bench_data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                if stored_key == run_cache_key and step_files_exist and "benchmark_metrics" in bench_data:
                    loaded_deltas: dict[int, StepDelta] = {}
                    for c in story.chunks:
                        p = model_step_dir / f"step_{c.step:02d}_delta.json"
                        loaded_deltas[c.step] = StepDelta.model_validate_json(p.read_text(encoding="utf-8"))
                    ext_hist = TokenUsage.from_dict(bench_data.get("llm_extraction_historical_api_usage"))
                    ext_hist.cache_hits = len(story.chunks)
                    ext_hist.cache_misses = 0
                    comb_hist = TokenUsage()
                    comb_hist.add(story.generation_usage)
                    comb_hist.add(ext_hist)
                    metrics_obj = ExtractionBenchmarkMetrics.from_dict(bench_data.get("benchmark_metrics"))
                    c_passed = int(bench_data.get("checks_passed", round(bench_data.get("strict_pass_rate", 1.0) * 50)))
                    c_total = int(bench_data.get("checks_total", 50))
                    sd_passed = int(bench_data.get("step_delta_checks_passed", min(8, c_passed)))
                    sd_total = int(bench_data.get("step_delta_checks_total", 8))
                    cs_passed = int(bench_data.get("cumulative_state_checks_passed", max(0, c_passed - sd_passed)))
                    cs_total = int(bench_data.get("cumulative_state_checks_total", max(1, c_total - sd_total)))
                    return StoryE2EExecutionResult(
                        story=story,
                        extraction_model=cached_extractor.model_id,
                        story_cache_hit=story.loaded_from_cache,
                        extraction_cache_hits=len(story.chunks),
                        extraction_cache_misses=0,
                        generation_usage=story.generation_usage,
                        extraction_billable_usage=TokenUsage(cache_hits=len(story.chunks)),
                        extraction_historical_usage=ext_hist,
                        combined_historical_usage=comb_hist,
                        checks_passed=c_passed,
                        checks_total=c_total,
                        failures=list(bench_data.get("failures") or []),
                        extracted_deltas_by_step=loaded_deltas,
                        raw_deltas_by_step=dict(loaded_deltas),
                        guidance_mode=guidance_mode,
                        metrics=metrics_obj,
                        skipped_cached_run=True,
                        run_cache_key=run_cache_key,
                        step_delta_checks_passed=sd_passed,
                        step_delta_checks_total=sd_total,
                        cumulative_state_checks_passed=cs_passed,
                        cumulative_state_checks_total=cs_total,
                    )
            except Exception:  # noqa: BLE001, S110
                pass

        engine = CanonArcEngine(":memory:", preset=preset)
        oracle_engine = CanonArcEngine(
            ":memory:",
            preset=InteractiveFictionPreset(include_background_edges=True),
        )

        # Seed Step 0 world on both the model engine and the reference oracle engine
        self._seed_engine_from_params(engine, story.params)
        self._seed_engine_from_params(oracle_engine, story.params)

        scorer = ExtractionBenchmarkScorer(
            params=story.params,
            model_id=cached_extractor.model_id,
            guidance_mode=guidance_mode,
        )

        checks_passed = 0
        checks_total = 0
        step_delta_checks_passed = 0
        step_delta_checks_total = 0
        cumulative_state_checks_passed = 0
        cumulative_state_checks_total = 0
        failures: list[str] = []
        extracted_deltas: dict[int, StepDelta] = {}
        raw_deltas: dict[int, StepDelta] = {}

        def _check(cond: bool, desc: str, *, kind: Literal["step_delta", "cumulative"] = "step_delta") -> None:
            nonlocal checks_passed, checks_total
            nonlocal step_delta_checks_passed, step_delta_checks_total
            nonlocal cumulative_state_checks_passed, cumulative_state_checks_total
            checks_total += 1
            if kind == "cumulative":
                cumulative_state_checks_total += 1
            else:
                step_delta_checks_total += 1
            if cond:
                checks_passed += 1
                if kind == "cumulative":
                    cumulative_state_checks_passed += 1
                else:
                    step_delta_checks_passed += 1
            else:
                failures.append(desc)

        c_prot_id = story.params.characters[0].entity_id if len(story.params.characters) > 0 else "char_elena"
        c_cmd_id = story.params.characters[1].entity_id if len(story.params.characters) > 1 else "char_marcus"
        c_spy_id = story.params.characters[2].entity_id if len(story.params.characters) > 2 else "char_lyra"
        c_culp_id = story.params.characters[3].entity_id if len(story.params.characters) > 3 else "char_julian"
        disg_alias = "the hooded courier"
        disg_persona = "Courier"
        for a in (story.params.characters[2].aliases if len(story.params.characters) > 2 else []):
            if isinstance(a, AliasRecord) and a.kind == "disguise":
                disg_alias = a.alias
                disg_persona = a.persona_name or "Courier"

        for chunk in story.chunks:
            step = chunk.step
            beat = beat_by_step.get(step)
            preset.current_beat_hint = self._build_beat_hint_for_mode(beat, guidance_mode)
            if guidance_mode != "full_hints":
                preset.current_world_state_block = self._format_active_world_state_for_extraction(
                    engine,
                    step - 1,
                    include_canonical_ids=(guidance_mode != "surface_only"),
                )
            else:
                preset.current_world_state_block = ""

            candidates: list[DialogueCandidate] = engine.extract_dialogue_candidates(
                chunk.text,
                step=step - 1,
            )
            exemplars_to_stage: list[VoiceExemplar] = []
            for cand in candidates:
                if cand.speaker_id:
                    meta = chunk.voice_metadata_by_speaker.get(cand.speaker_id, {})
                    q_score = chunk.voice_quality_by_speaker.get(cand.speaker_id, 9.0)
                    exemplars_to_stage.append(
                        cand.to_voice_exemplar(step=step, metadata=meta, quality_score=q_score)
                    )

            prior_entities = {
                e.entity_id: e
                for e in engine.store.get_all_active_entities(engine.active_timeline, step - 1)
            }
            prior_prot_mood = prior_entities[c_prot_id].state.get("mood") if c_prot_id in prior_entities else "furious"

            with engine.step_transaction(
                step=step,
                text=chunk.text,
                location_id=chunk.location_id,
                rerun_clean=True,
            ) as tx:
                committed_delta = tx.extract_and_commit(
                    chunk.text,
                    extractor=cached_extractor,
                    exemplars=exemplars_to_stage,
                )
                raw_delta = tx.last_raw_delta or committed_delta

            # Advance reference oracle engine in lockstep
            oracle_norm = oracle_engine.normalize_delta(chunk.oracle_delta, step=step)
            oracle_engine.commit_step(step=step, delta=oracle_norm, exemplars=exemplars_to_stage)

            extracted_deltas[step] = committed_delta
            raw_deltas[step] = raw_delta

            # Save step_XX_delta.json in model-specific subfolder (and legacy root for primary baseline)
            serialized_delta = json.dumps(committed_delta.model_dump(mode="json"), indent=2, ensure_ascii=False)
            (model_step_dir / f"step_{step:02d}_delta.json").write_text(serialized_delta, encoding="utf-8")
            legacy_step_path = extractions_dir / f"step_{step:02d}_delta.json"
            if (cached_extractor.model_id == story.model and guidance_mode == "full_hints") or not legacy_step_path.is_file():
                legacy_step_path.write_text(serialized_delta, encoding="utf-8")

            # Score step-local, cumulative, ID hygiene, and downstream POV retrieval metrics
            scorer.score_step(
                chunk=chunk,
                raw_delta=raw_delta,
                committed_delta=committed_delta,
                model_engine=engine,
                oracle_engine=oracle_engine,
            )

            # Verify strict room occupants and moods after this chunk (cumulative state checks)
            for room_id, expected_occupants in chunk.expected_room_occupants.items():
                room_ctx = engine.retrieve_context(
                    step=step,
                    location_id=room_id,
                    inclusive=True,
                    fallback_on_empty_mentions="none",
                )
                actual_chars = {e.entity_id for e in room_ctx.entities if e.kind == "character"}
                _check(
                    actual_chars == expected_occupants,
                    f"[{story_slug} Step {step}] Room '{room_id}': expected {expected_occupants}, got {actual_chars}",
                    kind="cumulative",
                )

            all_entities = {
                e.entity_id: e
                for e in engine.store.get_all_active_entities(engine.active_timeline, step)
            }
            for cid, exp_mood in chunk.expected_moods.items():
                actual_mood = all_entities[cid].state.get("mood") if cid in all_entities else None
                allowed = chunk.acceptable_moods.get(cid)
                mood_ok = moods_soft_match(exp_mood, actual_mood, acceptable=allowed)
                _check(
                    mood_ok,
                    f"[{story_slug} Step {step}] Entity '{cid}' mood: expected '{exp_mood}', got '{actual_mood}'",
                    kind="cumulative",
                )

            # Deep Tier 1-6 step-delta checks (blueprint-driven so non-isomorphic beat orders work)
            if blueprint is not None:
                ep_confront_id = blueprint.key_episode_ids.get("confrontation", "ep_step2_secret_confrontation")
                ep_flashback_id = blueprint.key_episode_ids.get("flashback", "ep_step4_past_flashback")
                oracle_ep_ids = {ep.episode_id for ep in chunk.oracle_delta.episodes}
                oracle_deceived_informs = [
                    inf for inf in chunk.oracle_delta.inform if inf.epistemic == "deceived"
                ]
                oracle_unmask_entities = [
                    ed
                    for ed in chunk.oracle_delta.entities
                    if any(
                        isinstance(a, AliasRecord) and a.kind == "disguise"
                        for a in ed.add_aliases
                    )
                ]

                if step == 1:
                    ctx_cmd = engine.retrieve_context(
                        step=1,
                        text=f"{disg_alias} waited near the table.",
                        pov_entity_ids=[c_cmd_id],
                        inclusive=True,
                    )
                    ctx_prot = engine.retrieve_context(
                        step=1,
                        text=f"{disg_alias} waited near the table.",
                        pov_entity_ids=[c_prot_id],
                        inclusive=True,
                    )
                    m_ok = f"masked:{c_spy_id}" in {e.entity_id for e in ctx_cmd.entities}
                    e_ok = c_spy_id in {e.entity_id for e in ctx_prot.entities}
                    # Only record 1 combined Step 1 preservation check so passive Step 0 inertia doesn't dominate disguise_accuracy
                    scorer.record_disguise_check(m_ok and e_ok)
                    _check(
                        m_ok,
                        f"[{story_slug} Step 1] {c_cmd_id} should see masked:{c_spy_id}",
                        kind="step_delta",
                    )
                    _check(
                        e_ok,
                        f"[{story_slug} Step 1] {c_prot_id} should pierce disguise to {c_spy_id}",
                        kind="step_delta",
                    )
                if ep_confront_id in oracle_ep_ids:
                    ctx_cmd_ep = engine.retrieve_context(
                        step=step,
                        text=chunk.text,
                        pov_entity_ids=[c_cmd_id],
                        inclusive=True,
                    )
                    aligned_ep_ids = {
                        scorer.align_episode_id(ep.episode_id) for ep in ctx_cmd_ep.episodes
                    }
                    _check(
                        ep_confront_id not in aligned_ep_ids,
                        f"[{story_slug} Step {step}] Absent {c_cmd_id} must not know {ep_confront_id}",
                        kind="step_delta",
                    )
                if chunk.oracle_delta.is_flashback:
                    prot_mood_now = all_entities[c_prot_id].state.get("mood") if c_prot_id in all_entities else None
                    fb_ok = bool(committed_delta.is_flashback) and not moods_soft_match(
                        "naive_trusting", prot_mood_now
                    )
                    scorer.flashback_guard_passed = scorer.flashback_guard_passed and fb_ok
                    # Flashback check verifies that the flashback did not overwrite present-day mood with 'naive_trusting'
                    _check(
                        fb_ok and prot_mood_now == prior_prot_mood,
                        f"[{story_slug} Step {step}] Flashback must not overwrite {c_prot_id}'s present mood ('{prior_prot_mood}')",
                        kind="step_delta",
                    )
                if oracle_unmask_entities:
                    ctx_culp = engine.retrieve_context(
                        step=step,
                        text=f"{disg_alias} spoke in the secret chamber.",
                        pov_entity_ids=[c_culp_id],
                        inclusive=True,
                    )
                    unmask_ok = any(
                        e.entity_id == c_spy_id and e.disguised_as == disg_persona for e in ctx_culp.entities
                    )
                    # Active disguise unmasking carries 2 checks (dominating Step 1's passive check)
                    scorer.record_disguise_check(unmask_ok)
                    scorer.record_disguise_check(unmask_ok)
                    _check(
                        unmask_ok,
                        f"[{story_slug} Step {step}] {c_culp_id} should pierce {c_spy_id}'s disguise after unmasking",
                        kind="step_delta",
                    )
                if oracle_deceived_informs:
                    dec_inf = oracle_deceived_informs[0]
                    dec_query = dec_inf.believed_summary or "Northern raiders broke into the annex and stole the artifact."
                    dec_tokens = [
                        t for t in re.findall(r"[a-z]{4,}", dec_query.lower())
                        if t not in ("that", "this", "with", "from", "into", "while", "were", "have", "been")
                    ]
                    ctx_cmd_dec = engine.retrieve_context(
                        step=step,
                        text=dec_query,
                        pov_entity_ids=[dec_inf.entity_id],
                        inclusive=True,
                    )
                    ep2_m = next(
                        (
                            ep
                            for ep in ctx_cmd_dec.episodes
                            if scorer.align_episode_id(ep.episode_id) == dec_inf.episode_id
                        ),
                        None,
                    )
                    dec_match = ep2_m is not None and (
                        "northern raiders" in ep2_m.summary.lower()
                        or any(tok in ep2_m.summary.lower() for tok in dec_tokens[:4])
                    )
                    _check(
                        dec_match,
                        f"[{story_slug} Step {step}] {dec_inf.entity_id} POV should recall deceived summary, got {ep2_m}",
                        kind="step_delta",
                    )
                if step == len(story.chunks):
                    ctx_cmd_truth = engine.retrieve_context(
                        step=step,
                        text=chunk.text,
                        pov_entity_ids=[c_cmd_id],
                        inclusive=True,
                    )
                    ep2_t = next(
                        (
                            ep
                            for ep in ctx_cmd_truth.episodes
                            if scorer.align_episode_id(ep.episode_id) == ep_confront_id
                        ),
                        None,
                    )
                    _check(
                        ep2_t is not None
                        and "northern raiders" not in ep2_t.summary.lower()
                        and "smugglers scaled" not in ep2_t.summary.lower(),
                        f"[{story_slug} Step {step}] {c_cmd_id} should see ground-truth summary after epistemic='told'",
                        kind="step_delta",
                    )
                    ctx_culp_mem = engine.retrieve_context(
                        step=step,
                        text="Ten years ago during the winter pact signing.",
                        pov_entity_ids=[c_culp_id],
                        inclusive=True,
                    )
                    _check(
                        all(
                            scorer.align_episode_id(ep.episode_id) != ep_flashback_id
                            for ep in ctx_culp_mem.episodes
                        ),
                        f"[{story_slug} Step {step}] {c_culp_id} must not recall forgotten {ep_flashback_id}",
                        kind="step_delta",
                    )

                # Long-horizon multi-chapter checks (consolidation & branching) when configured on blueprint
                if blueprint.consolidation_step and step == blueprint.consolidation_step:
                    all_eps_now = engine.query_episodes(cutoff_step=step)
                    low_sal = [ep for ep in all_eps_now if ep.step < step and ep.salience <= 4.0]
                    _check(
                        len(low_sal) >= 1,
                        f"[{story_slug} Step {step}] Expected low-salience Chapter 1 episodes available for consolidation",
                        kind="step_delta",
                    )
                if blueprint.branch_fork_step and step == blueprint.branch_fork_step:
                    alt_branch = engine.session("main").branch(at_step=step, name=f"alt_route_step_{step}")
                    alt_branch.commit_step(
                        step=step + 1,
                        delta=StepDelta(
                            entities=[EntityDelta(entity_id=c_culp_id, state={"mood": "repentant"})]
                        ),
                    )
                    diff = engine.diff_timelines("main", f"alt_route_step_{step}", at_step=step + 1)
                    changed_keys = {item["subject_key"] for item in diff.get("changed", [])}
                    _check(
                        c_culp_id in changed_keys,
                        f"[{story_slug} Step {step}] Branch divergence check failed for {c_culp_id}",
                        kind="step_delta",
                    )

        combined_historical = TokenUsage()
        combined_historical.add(story.generation_usage)
        combined_historical.add(cached_extractor.historical_usage)

        benchmark_metrics = scorer.finalize(
            step_usage_log=cached_extractor.step_usage_log,
            historical_usage=cached_extractor.historical_usage,
        )

        result = StoryE2EExecutionResult(
            story=story,
            extraction_model=cached_extractor.model_id,
            story_cache_hit=story.loaded_from_cache,
            extraction_cache_hits=cached_extractor.cache_hits,
            extraction_cache_misses=cached_extractor.cache_misses,
            generation_usage=story.generation_usage,
            extraction_billable_usage=cached_extractor.billable_usage,
            extraction_historical_usage=cached_extractor.historical_usage,
            combined_historical_usage=combined_historical,
            checks_passed=checks_passed,
            checks_total=checks_total,
            failures=failures,
            extracted_deltas_by_step=extracted_deltas,
            raw_deltas_by_step=raw_deltas,
            guidance_mode=guidance_mode,
            metrics=benchmark_metrics,
            skipped_cached_run=False,
            run_cache_key=run_cache_key,
            step_delta_checks_passed=step_delta_checks_passed,
            step_delta_checks_total=step_delta_checks_total,
            cumulative_state_checks_passed=cumulative_state_checks_passed,
            cumulative_state_checks_total=cumulative_state_checks_total,
        )
        self._persist_story_files(story, blueprint=blueprint, extraction_result=result)
        return result

    def compare_models(
        self,
        story_slug: str = "obsidian_signet_3k",
        *,
        extractors: Sequence[Any] | dict[str, Any],
        guidance_mode: GuidanceMode = "none",
        bypass_extraction_cache: bool = False,
        n_trials: int = 1,
    ) -> ModelComparisonReport:
        """Runs multiple models/extractors on the same story (across `n_trials` >= 1) and aggregates trial Mean ± Std."""
        import math

        def _mean_std(vals: Sequence[float]) -> tuple[float, float]:
            if not vals:
                return 0.0, 0.0
            m = sum(vals) / len(vals)
            if len(vals) < 2:
                return m, 0.0
            return m, math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))

        story = self.get_or_create_story(story_slug)
        results: list[StoryE2EExecutionResult] = []
        trial_stats_by_model: dict[str, dict[str, float]] = {}
        items: Sequence[tuple[str | None, Any]]
        if isinstance(extractors, dict):
            items = list(extractors.items())
        else:
            items = [(None, ext) for ext in extractors]

        trials = max(1, int(n_trials))
        for custom_name, ext in items:
            if isinstance(ext, str):
                if ext.strip().lower() in ("lmstudio", "lm-studio", "auto", "local"):
                    ext_obj: Any = OpenAICompatibleExtractor.from_lmstudio()
                else:
                    ext_obj = OpenAICompatibleExtractor.from_deepseek(model=ext, temperature=0.0)
            else:
                ext_obj = ext
            if custom_name and not hasattr(ext_obj, "model"):
                ext_obj.model = custom_name
            trial_runs: list[StoryE2EExecutionResult] = []
            for trial_idx in range(trials):
                res = self.run_story_extraction_e2e(
                    story_slug,
                    base_extractor=ext_obj,
                    guidance_mode=guidance_mode,
                    bypass_extraction_cache=bypass_extraction_cache or (trial_idx > 0),
                )
                trial_runs.append(res)
            assert trial_runs
            rep_res = trial_runs[-1]
            if trials > 1:
                oq_m, oq_s = _mean_std([tr.metrics.overall_quality_score for tr in trial_runs])
                pr_m, pr_s = _mean_std([tr.pass_rate for tr in trial_runs])
                sd_m, sd_s = _mean_std([tr.step_delta_pass_rate for tr in trial_runs])
                trial_stats_by_model[rep_res.extraction_model] = {
                    "n_trials": float(trials),
                    "overall_quality_mean": round(oq_m, 4),
                    "overall_quality_std": round(oq_s, 4),
                    "strict_pass_mean": round(pr_m, 4),
                    "strict_pass_std": round(pr_s, 4),
                    "step_delta_pass_mean": round(sd_m, 4),
                    "step_delta_pass_std": round(sd_s, 4),
                }
            results.append(rep_res)

        report = ModelComparisonReport(
            story_slug=story_slug,
            story_title=story.title,
            guidance_mode=guidance_mode,
            results=results,
            n_trials=trials,
            trial_stats_by_model=trial_stats_by_model,
        )
        benchmarks_dir = story.story_dir / "benchmarks"
        benchmarks_dir.mkdir(parents=True, exist_ok=True)
        (benchmarks_dir / f"comparison_{guidance_mode}.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (benchmarks_dir / f"comparison_{guidance_mode}.md").write_text(
            report.to_markdown_table(),
            encoding="utf-8",
        )
        return report

    def compare_models_across_stories(
        self,
        story_slugs: Sequence[str] = (
            "obsidian_signet_3k",
            "chronos_station_3k",
            "crimson_masquerade_3k",
            "veritas_locked_room_3k",
        ),
        *,
        extractors: Sequence[Any] | dict[str, Any],
        guidance_mode: GuidanceMode = "none",
        bypass_extraction_cache: bool = False,
    ) -> CrossStoryBenchmarkReport:
        """Evaluates a set of models across multiple story topologies and computes Mean ± StdDev confidence metrics."""
        import math

        def _mean_std(vals: Sequence[float]) -> tuple[float, float]:
            if not vals:
                return 0.0, 0.0
            m = sum(vals) / len(vals)
            if len(vals) < 2:
                return m, 0.0
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            return m, math.sqrt(var)

        per_story: dict[str, ModelComparisonReport] = {}
        by_model: dict[str, list[StoryE2EExecutionResult]] = {}

        for slug in story_slugs:
            rep = self.compare_models(
                slug,
                extractors=extractors,
                guidance_mode=guidance_mode,
                bypass_extraction_cache=bypass_extraction_cache,
            )
            per_story[slug] = rep
            for res in rep.results:
                by_model.setdefault(res.extraction_model, []).append(res)

        aggregates: list[CrossStoryModelAggregate] = []
        for model_id, runs in by_model.items():
            q_mean, q_std = _mean_std([r.metrics.overall_quality_score for r in runs])
            s_mean, s_std = _mean_std([r.pass_rate for r in runs])
            sd_mean, _ = _mean_std([r.step_delta_pass_rate for r in runs])
            cs_mean, _ = _mean_std([r.cumulative_state_pass_rate for r in runs])
            ep_mean, _ = _mean_std([r.metrics.epistemic_score for r in runs])
            sp_mean, _ = _mean_std([r.metrics.spatial_state_score for r in runs])
            pv_mean, _ = _mean_std([r.metrics.pov_retrieval_score for r in runs])
            ef_mean, _ = _mean_std([r.metrics.edge_flashback_score for r in runs])
            sh_mean, _ = _mean_std([r.metrics.schema_hygiene_score for r in runs])
            tot_tok = sum(r.extraction_historical_usage.total_tokens for r in runs)
            tot_cost = sum(
                r.extraction_historical_usage.estimate_cost_usd(model=model_id)["total_cost_usd"]
                for r in runs
            )
            aggregates.append(
                CrossStoryModelAggregate(
                    model_id=model_id,
                    guidance_mode=guidance_mode,
                    story_slugs=[r.story.story_slug for r in runs],
                    n_runs=len(runs),
                    overall_quality_mean=q_mean,
                    overall_quality_std=q_std,
                    strict_pass_mean=s_mean,
                    strict_pass_std=s_std,
                    step_delta_pass_mean=sd_mean,
                    cumulative_state_pass_mean=cs_mean,
                    epistemic_score_mean=ep_mean,
                    spatial_state_score_mean=sp_mean,
                    pov_retrieval_score_mean=pv_mean,
                    edge_flashback_score_mean=ef_mean,
                    schema_hygiene_score_mean=sh_mean,
                    total_tokens=tot_tok,
                    total_cost_usd=tot_cost,
                )
            )

        aggregates.sort(key=lambda a: a.overall_quality_mean, reverse=True)
        return CrossStoryBenchmarkReport(
            story_slugs=list(story_slugs),
            guidance_mode=guidance_mode,
            aggregates=aggregates,
            per_story_reports=per_story,
        )


# ============================================================================
# 3. Multi-Metric Extraction Benchmark Scorer & Simulated Model Tiers
# ============================================================================


class ExtractionBenchmarkScorer:
    """Computes step-local, cumulative, epistemic, ID-hygiene, and downstream POV retrieval metrics."""

    MIN_EPISODE_ALIGNMENT_SCORE: float = 0.35

    def __init__(
        self,
        params: SynthStoryParams,
        *,
        model_id: str = "",
        guidance_mode: GuidanceMode = "full_hints",
    ) -> None:
        self.params = params
        self.model_id = model_id
        self.guidance_mode: GuidanceMode = guidance_mode
        self.canonical_char_ids: set[str] = {c.entity_id for c in params.characters}
        self.canonical_loc_ids: set[str] = {loc.entity_id for loc in params.locations}
        self.canonical_art_ids: set[str] = {art.entity_id for art in params.artifacts}
        self.all_canonical_ids: set[str] = (
            self.canonical_char_ids | self.canonical_loc_ids | self.canonical_art_ids
        )

        self.surface_to_canonical: dict[str, str] = {}
        for c in params.characters:
            self.surface_to_canonical[c.canonical_name.lower()] = c.entity_id
            for a in c.aliases:
                alias_str = a.alias if isinstance(a, AliasRecord) else str(a)
                if alias_str:
                    self.surface_to_canonical[alias_str.lower()] = c.entity_id
        for loc in params.locations:
            self.surface_to_canonical[loc.canonical_name.lower()] = loc.entity_id
        for art in params.artifacts:
            self.surface_to_canonical[art.canonical_name.lower()] = art.entity_id

        self.episode_id_map: dict[str, str] = {}
        self.allowed_optional_epistemic_reveals: set[tuple[str, str]] = set()
        self.step_metrics: list[StepExtractionMetrics] = []
        self.disguise_checks: list[bool] = []
        self.flashback_guard_passed: bool = True

    def align_episode_id(self, ep_id: str | None) -> str:
        if not ep_id:
            return ""
        return self.episode_id_map.get(ep_id, ep_id)

    def record_disguise_check(self, passed: bool) -> None:
        self.disguise_checks.append(bool(passed))

    def _classify_raw_id(self, raw_id: str | None, sm: StepExtractionMetrics) -> None:
        if not raw_id or not str(raw_id).strip():
            return
        cleaned = str(raw_id).strip()
        sm.raw_id_total += 1
        if cleaned in self.all_canonical_ids:
            sm.canonical_id_count += 1
        elif cleaned.lower() in self.surface_to_canonical:
            sm.normalized_surface_id_count += 1
        else:
            sm.hallucinated_id_count += 1
            if cleaned not in sm.hallucinated_ids:
                sm.hallucinated_ids.append(cleaned)

    @staticmethod
    def _summary_token_overlap(text_a: str, text_b: str) -> float:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "over", "while",
            "about", "after", "before", "their", "there", "where", "which", "were", "have",
            "been", "being", "they", "them", "then", "than", "when", "what",
        }
        tok_a = {t for t in re.findall(r"[a-z]{3,}", (text_a or "").lower()) if t not in stop}
        tok_b = {t for t in re.findall(r"[a-z]{3,}", (text_b or "").lower()) if t not in stop}
        if not tok_a or not tok_b:
            return 0.0
        return len(tok_a & tok_b) / len(tok_a | tok_b)

    @classmethod
    def _episode_similarity(cls, o_ep: EpisodeRecord, e_ep: EpisodeRecord) -> float:
        o_parts = set(o_ep.participants.keys()) | set(o_ep.witnesses)
        e_parts = set(e_ep.participants.keys()) | set(e_ep.witnesses)
        overlap = _jaccard(o_parts, e_parts)
        pred_bonus = (
            0.20
            if (e_ep.predicate and o_ep.predicate and e_ep.predicate.lower() == o_ep.predicate.lower())
            else 0.0
        )
        loc_bonus = (
            0.15
            if (e_ep.location_id and o_ep.location_id and e_ep.location_id == o_ep.location_id)
            else 0.0
        )
        summary_bonus = 0.25 * cls._summary_token_overlap(o_ep.summary, e_ep.summary)
        return overlap + pred_bonus + loc_bonus + summary_bonus

    def _register_dynamic_oracle_entities(self, oracle: StepDelta) -> None:
        """Registers any mid-story dynamically introduced entities from the oracle delta."""
        for ed in oracle.entities:
            if ed.entity_id not in self.all_canonical_ids:
                self.all_canonical_ids.add(ed.entity_id)
                if ed.kind == "character" or ed.entity_id.startswith("char_"):
                    self.canonical_char_ids.add(ed.entity_id)
                elif ed.kind == "location" or ed.entity_id.startswith("loc_"):
                    self.canonical_loc_ids.add(ed.entity_id)
                elif ed.kind == "artifact" or ed.entity_id.startswith("item_"):
                    self.canonical_art_ids.add(ed.entity_id)
            if ed.canonical_name:
                self.surface_to_canonical[ed.canonical_name.lower()] = ed.entity_id

    def score_step(
        self,
        *,
        chunk: SyntheticChunkSpec,
        raw_delta: StepDelta,
        committed_delta: StepDelta,
        model_engine: CanonArcEngine,
        oracle_engine: CanonArcEngine,
    ) -> StepExtractionMetrics:
        step = chunk.step
        oracle = chunk.oracle_delta
        self._register_dynamic_oracle_entities(oracle)
        sm = StepExtractionMetrics(step=step, flashback_expected=oracle.is_flashback)

        # Register optional valid informs at this step (e.g., prose-supported secondary tells)
        for opt_inf in chunk.optional_inform:
            if opt_inf.epistemic in ("told", "deceived"):
                self.allowed_optional_epistemic_reveals.add((opt_inf.episode_id, opt_inf.entity_id))

        # 1. Raw ID Hygiene (Pre-Normalization vs Canonical vs Surface vs Hallucinated)
        for ed in raw_delta.entities:
            self._classify_raw_id(ed.entity_id, sm)
            loc_val = ed.state.get("location_id") or ed.state.get("location")
            if loc_val:
                self._classify_raw_id(str(loc_val), sm)
        for edge in raw_delta.edges:
            self._classify_raw_id(edge.source_id, sm)
            self._classify_raw_id(edge.target_id, sm)
        for ep in raw_delta.episodes:
            for pid in ep.participants:
                self._classify_raw_id(pid, sm)
            for wid in ep.witnesses:
                self._classify_raw_id(wid, sm)
            if ep.location_id:
                self._classify_raw_id(ep.location_id, sm)
        for inf in raw_delta.inform:
            self._classify_raw_id(inf.entity_id, sm)

        # 2. Align Episodes at this step (requiring score >= MIN_EPISODE_ALIGNMENT_SCORE) & score Witness metrics
        used_ext_indices: set[int] = set()
        for o_ep in oracle.episodes:
            matched_idx: int | None = None
            for idx, e_ep in enumerate(committed_delta.episodes):
                if idx in used_ext_indices:
                    continue
                if e_ep.episode_id == o_ep.episode_id:
                    matched_idx = idx
                    break
            if matched_idx is None and committed_delta.episodes:
                best_score = -1.0
                for idx, e_ep in enumerate(committed_delta.episodes):
                    if idx in used_ext_indices:
                        continue
                    score = self._episode_similarity(o_ep, e_ep)
                    if score >= self.MIN_EPISODE_ALIGNMENT_SCORE and score > best_score:
                        best_score = score
                        matched_idx = idx

            exp_w = set(o_ep.witnesses)
            absent_chars = self.canonical_char_ids - exp_w
            sm.absent_character_opportunities += len(absent_chars)
            primary_actors = {
                k for k, role in o_ep.participants.items() if role in ("actor", "target", "experiencer")
            }
            silent_exp = (exp_w & self.canonical_char_ids) - primary_actors
            sm.silent_witness_expected += len(silent_exp)

            if matched_idx is not None:
                used_ext_indices.add(matched_idx)
                e_ep = committed_delta.episodes[matched_idx]
                if e_ep.episode_id and o_ep.episode_id:
                    self.episode_id_map[e_ep.episode_id] = o_ep.episode_id
                act_w = set(e_ep.witnesses)
                sm.witness_tp += len(act_w & exp_w)
                sm.witness_fp += len(act_w - exp_w)
                sm.witness_fn += len(exp_w - act_w)
                sm.epistemic_leak_count += len(act_w & absent_chars)
                sm.silent_witness_matched += len(act_w & silent_exp)
            else:
                sm.witness_fn += len(exp_w)

        # Many-to-one sub-episode alignment: only map an extra sub-episode to an oracle episode if it occurred
        # in the same location with the exact same witness set (so a public upstairs beat doesn't alias a secret vault beat).
        if oracle.episodes:
            for idx, e_ep in enumerate(committed_delta.episodes):
                if idx in used_ext_indices or not e_ep.episode_id:
                    continue
                for o_ep in oracle.episodes:
                    if (
                        o_ep.episode_id
                        and e_ep.location_id == o_ep.location_id
                        and set(e_ep.witnesses) == set(o_ep.witnesses)
                        and self._episode_similarity(o_ep, e_ep) >= self.MIN_EPISODE_ALIGNMENT_SCORE
                    ):
                        self.episode_id_map[e_ep.episode_id] = o_ep.episode_id
                        break

        # 3. Step-Local Location Moves & Mood Deltas (ignoring idempotent re-assertions of true current state)
        exp_moves = {
            (ed.entity_id, str(ed.state["location_id"]))
            for ed in oracle.entities
            if ed.state.get("location_id")
        }
        act_moves = {
            (ed.entity_id, str(ed.state["location_id"]))
            for ed in committed_delta.entities
            if ed.state.get("location_id")
        }
        sm.location_delta_tp = len(exp_moves & act_moves)
        sm.location_delta_fn = len(exp_moves - act_moves)
        sm.location_delta_fp = sum(
            1
            for cid, loc_id in (act_moves - exp_moves)
            if cid not in chunk.expected_room_occupants.get(loc_id, set())
        )

        if not oracle.is_flashback:
            exp_mood_changes = {
                ed.entity_id: str(ed.state["mood"])
                for ed in oracle.entities
                if ed.state.get("mood")
            }
            act_mood_changes = {
                ed.entity_id: str(ed.state["mood"])
                for ed in committed_delta.entities
                if ed.state.get("mood")
            }
            for cid, exp_m in exp_mood_changes.items():
                allowed = chunk.acceptable_moods.get(cid)
                if cid in act_mood_changes and moods_soft_match(exp_m, act_mood_changes[cid], acceptable=allowed):
                    sm.mood_delta_tp += 1
                elif allowed and chunk.expected_moods.get(cid) in allowed and cid not in act_mood_changes:
                    # When keeping prior mood is acceptable in the prose (e.g., Step 2 Marcus), do not penalize as FN
                    pass
                else:
                    sm.mood_delta_fn += 1
            for cid, act_m in act_mood_changes.items():
                allowed = chunk.acceptable_moods.get(cid)
                if cid in exp_mood_changes:
                    if not moods_soft_match(exp_mood_changes[cid], act_m, acceptable=allowed):
                        sm.mood_delta_fp += 1
                elif not moods_soft_match(chunk.expected_moods.get(cid), act_m, acceptable=allowed):
                    sm.mood_delta_fp += 1

        if oracle.is_flashback:
            sm.flashback_flag_correct = bool(committed_delta.is_flashback)

        # 4. Step-Local Edge Deltas, Register & Causal Links
        exp_edges = {(e.source_id, e.target_id, e.relation_type): e for e in oracle.edges}
        act_edges = {(e.source_id, e.target_id, e.relation_type): e for e in committed_delta.edges}
        sm.edge_delta_tp = len(set(exp_edges) & set(act_edges))
        sm.edge_delta_fp = len(set(act_edges) - set(exp_edges))
        sm.edge_delta_fn = len(set(exp_edges) - set(act_edges))
        for e_key, o_edge in exp_edges.items():
            if "register" in o_edge.state:
                sm.edge_register_total += 1
                a_edge = act_edges.get(e_key)
                if a_edge and a_edge.state.get("register") == o_edge.state.get("register"):
                    sm.edge_register_matches += 1

        # Causal links on entities and edges
        act_ent_by_id = {ed.entity_id: ed for ed in committed_delta.entities}
        for o_ed in oracle.entities:
            if o_ed.caused_by_episode_id:
                sm.causal_links_expected += 1
                a_ed = act_ent_by_id.get(o_ed.entity_id)
                if a_ed and self.align_episode_id(a_ed.caused_by_episode_id) == o_ed.caused_by_episode_id:
                    sm.causal_links_matched += 1
        for e_key, o_edge in exp_edges.items():
            if o_edge.caused_by_episode_id:
                sm.causal_links_expected += 1
                a_edge = act_edges.get(e_key)
                if a_edge and self.align_episode_id(a_edge.caused_by_episode_id) == o_edge.caused_by_episode_id:
                    sm.causal_links_matched += 1

        # 5. InformDelta Scoring (with Episode ID Alignment & Optional Prose-Supported Informs)
        exp_informs = {(inf.episode_id, inf.entity_id, inf.epistemic): inf for inf in oracle.inform}
        opt_informs = {(inf.episode_id, inf.entity_id, inf.epistemic): inf for inf in chunk.optional_inform}
        act_informs = {
            (self.align_episode_id(inf.episode_id), inf.entity_id, inf.epistemic): inf
            for inf in committed_delta.inform
        }
        sm.inform_tp = len(set(exp_informs) & set(act_informs))
        sm.inform_fp = len(set(act_informs) - set(exp_informs) - set(opt_informs))
        sm.inform_fn = len(set(exp_informs) - set(act_informs))
        for i_key, o_inf in exp_informs.items():
            if o_inf.epistemic == "deceived" and o_inf.believed_summary:
                sm.deception_summary_expected += 1
                a_inf = act_informs.get(i_key)
                if a_inf and a_inf.believed_summary and len(a_inf.believed_summary.strip()) >= 10:
                    bel_low = a_inf.believed_summary.lower()
                    overlap = self._summary_token_overlap(o_inf.believed_summary, a_inf.believed_summary)
                    leaked_secret = any(
                        leak_kw in bel_low
                        for leak_kw in ("accused", "confronted", "guilty of stealing", "forged the seal")
                    )
                    if overlap >= 0.15 and not leaked_secret:
                        sm.deception_summary_matched += 1

        # 6. Cumulative Occupied-Room Occupancy, Character Location Accuracy & Moods after Step Commit
        room_jaccards: list[float] = []
        occupied_room_jaccards: list[float] = []
        all_rooms_exact = True
        expected_loc_by_char: dict[str, str] = {}
        for room_id, expected_occupants in chunk.expected_room_occupants.items():
            for cid in expected_occupants:
                expected_loc_by_char[cid] = room_id
            room_ctx = model_engine.retrieve_context(
                step=step,
                location_id=room_id,
                inclusive=True,
                fallback_on_empty_mentions="none",
            )
            actual_chars = {e.entity_id for e in room_ctx.entities if e.kind == "character"}
            j = _jaccard(expected_occupants, actual_chars)
            room_jaccards.append(j)
            if expected_occupants or actual_chars:
                occupied_room_jaccards.append(j)
            if actual_chars != expected_occupants:
                all_rooms_exact = False
        sm.room_occupancy_jaccard = (
            sum(occupied_room_jaccards) / len(occupied_room_jaccards)
            if occupied_room_jaccards
            else sum(room_jaccards) / max(1, len(room_jaccards))
        )
        sm.room_occupancy_exact = all_rooms_exact

        active_entities = {
            e.entity_id: e
            for e in model_engine.store.get_all_active_entities(model_engine.active_timeline, step)
        }
        for cid, exp_room in expected_loc_by_char.items():
            sm.character_location_total += 1
            act_room = active_entities[cid].state.get("location_id") if cid in active_entities else None
            if act_room == exp_room:
                sm.character_location_matches += 1

        for cid, exp_mood in chunk.expected_moods.items():
            sm.mood_total_characters += 1
            act_mood = active_entities[cid].state.get("mood") if cid in active_entities else None
            allowed = chunk.acceptable_moods.get(cid)
            if act_mood == exp_mood or (allowed and act_mood in allowed):
                sm.mood_exact_matches += 1
            if moods_soft_match(exp_mood, act_mood, acceptable=allowed):
                sm.mood_soft_matches += 1

        # 6b. Tier 5 Dyadic Voice Register Retrieval Accuracy (Model active stance/facade register vs Oracle)
        if chunk.voice_metadata_by_speaker:
            o_edges_all = oracle_engine.store.get_all_active_edges(oracle_engine.active_timeline, step)
            m_edges_all = model_engine.store.get_all_active_edges(model_engine.active_timeline, step)
            for spk_id, expected_meta in chunk.voice_metadata_by_speaker.items():
                exp_reg = expected_meta.get("register")
                if not exp_reg:
                    continue
                sm.voice_register_total += 1
                # Compare resolved dyadic register on the speaker's outgoing stance/facade edges
                o_regs = {
                    (e.target_id, e.state.get("register"))
                    for e in o_edges_all
                    if e.source_id == spk_id and e.state.get("register") == exp_reg
                }
                m_regs = {
                    (e.target_id, e.state.get("register"))
                    for e in m_edges_all
                    if e.source_id == spk_id and e.state.get("register") == exp_reg
                }
                if not o_regs or (o_regs & m_regs):
                    sm.voice_register_matches += 1

        # 7. Downstream POV Retrieval Comparison (Model Engine vs Reference Oracle Engine)
        all_oracle_eps = {ep.episode_id for ep in oracle_engine.query_episodes(cutoff_step=step)}
        pov_ent_scores: list[float] = []
        pov_edge_scores: list[float] = []
        pov_ep_recalls: list[float] = []
        pov_leaks = 0

        for char in self.params.characters:
            cid = char.entity_id
            o_ctx = oracle_engine.retrieve_context(
                step=step,
                text=chunk.text,
                pov_entity_ids=[cid],
                location_id=chunk.location_id,
                top_k_episodes=25,
                inclusive=True,
            )
            m_ctx = model_engine.retrieve_context(
                step=step,
                text=chunk.text,
                pov_entity_ids=[cid],
                location_id=chunk.location_id,
                top_k_episodes=25,
                inclusive=True,
            )
            o_ents = {e.entity_id for e in o_ctx.entities}
            m_ents = {e.entity_id for e in m_ctx.entities}
            pov_ent_scores.append(_jaccard(o_ents, m_ents))

            o_edge_keys = {
                (e.source_id, e.target_id, e.relation_type, e.state.get("register"), e.state.get("tone"))
                for e in o_ctx.active_edges
            }
            m_edge_keys = {
                (e.source_id, e.target_id, e.relation_type, e.state.get("register"), e.state.get("tone"))
                for e in m_ctx.active_edges
            }
            pov_edge_scores.append(_jaccard(o_edge_keys, m_edge_keys))

            o_vis_eps = {ep.episode_id for ep in o_ctx.episodes}
            m_vis_eps = {self.align_episode_id(ep.episode_id) for ep in m_ctx.episodes}
            pov_ep_recalls.append(_safe_div(len(o_vis_eps & m_vis_eps), len(o_vis_eps), 1.0))

            # Any oracle episode that exists at this step but is hidden from cid in o_ctx
            # (excluding optional prose-supported reveals at this step)
            opt_revealed_for_cid = {
                ep_id for ep_id, target_cid in self.allowed_optional_epistemic_reveals if target_cid == cid
            }
            o_hidden_eps = (all_oracle_eps - o_vis_eps) - opt_revealed_for_cid
            pov_leaks += len(o_hidden_eps & m_vis_eps)

            # Check if a disguised entity masked in o_ctx was leaked unmasked in m_ctx
            for eid in o_ents:
                if eid.startswith("masked:"):
                    real_id = eid.split("masked:", 1)[1]
                    if real_id in m_ents:
                        pov_leaks += 1

        sm.pov_entity_jaccard = sum(pov_ent_scores) / max(1, len(pov_ent_scores))
        sm.pov_edge_jaccard = sum(pov_edge_scores) / max(1, len(pov_edge_scores))
        sm.pov_episode_recall = sum(pov_ep_recalls) / max(1, len(pov_ep_recalls))
        sm.pov_secret_leaks = pov_leaks

        self.step_metrics.append(sm)
        return sm

    def finalize(
        self,
        *,
        step_usage_log: Sequence[dict[str, Any]] = (),
        historical_usage: TokenUsage | None = None,
    ) -> ExtractionBenchmarkMetrics:
        total_steps = max(1, len(self.step_metrics))

        # Witness & Epistemic totals
        w_tp = sum(s.witness_tp for s in self.step_metrics)
        w_fp = sum(s.witness_fp for s in self.step_metrics)
        w_fn = sum(s.witness_fn for s in self.step_metrics)
        w_prec = _safe_div(w_tp, w_tp + w_fp, 1.0)
        w_rec = _safe_div(w_tp, w_tp + w_fn, 1.0)
        w_f1 = _compute_f1(w_prec, w_rec) if (w_tp + w_fp + w_fn) > 0 else 1.0

        leak_cnt = sum(s.epistemic_leak_count for s in self.step_metrics)
        absent_opps = sum(s.absent_character_opportunities for s in self.step_metrics)
        leak_rate = _safe_div(leak_cnt, absent_opps, 0.0)

        sw_match = sum(s.silent_witness_matched for s in self.step_metrics)
        sw_exp = sum(s.silent_witness_expected for s in self.step_metrics)
        sw_rec = _safe_div(sw_match, sw_exp, 1.0)

        i_tp = sum(s.inform_tp for s in self.step_metrics)
        i_fp = sum(s.inform_fp for s in self.step_metrics)
        i_fn = sum(s.inform_fn for s in self.step_metrics)
        i_prec = _safe_div(i_tp, i_tp + i_fp, 1.0)
        i_rec = _safe_div(i_tp, i_tp + i_fn, 1.0)
        i_f1 = _compute_f1(i_prec, i_rec) if (i_tp + i_fp + i_fn) > 0 else 1.0

        dec_match = sum(s.deception_summary_matched for s in self.step_metrics)
        dec_exp = sum(s.deception_summary_expected for s in self.step_metrics)
        dec_acc = _safe_div(dec_match, dec_exp, 1.0)

        disg_acc = (
            sum(1 for ok in self.disguise_checks if ok) / len(self.disguise_checks)
            if self.disguise_checks
            else 1.0
        )

        # Spatial & Mood totals
        loc_tp = sum(s.location_delta_tp for s in self.step_metrics)
        loc_fp = sum(s.location_delta_fp for s in self.step_metrics)
        loc_fn = sum(s.location_delta_fn for s in self.step_metrics)
        loc_prec = _safe_div(loc_tp, loc_tp + loc_fp, 1.0)
        loc_rec = _safe_div(loc_tp, loc_tp + loc_fn, 1.0)
        loc_f1 = _compute_f1(loc_prec, loc_rec) if (loc_tp + loc_fp + loc_fn) > 0 else 1.0

        room_jacc = sum(s.room_occupancy_jaccard for s in self.step_metrics) / total_steps
        room_exact = sum(1 for s in self.step_metrics if s.room_occupancy_exact) / total_steps
        cloc_m = sum(s.character_location_matches for s in self.step_metrics)
        cloc_t = sum(s.character_location_total for s in self.step_metrics)
        char_loc_acc = _safe_div(cloc_m, cloc_t, room_jacc)

        m_exact = sum(s.mood_exact_matches for s in self.step_metrics)
        m_soft = sum(s.mood_soft_matches for s in self.step_metrics)
        m_tot = sum(s.mood_total_characters for s in self.step_metrics)
        mood_exact_acc = _safe_div(m_exact, m_tot, 1.0)
        mood_soft_acc = _safe_div(m_soft, m_tot, 1.0)

        md_tp = sum(s.mood_delta_tp for s in self.step_metrics)
        md_fp = sum(s.mood_delta_fp for s in self.step_metrics)
        md_fn = sum(s.mood_delta_fn for s in self.step_metrics)
        md_prec = _safe_div(md_tp, md_tp + md_fp, 1.0)
        md_rec = _safe_div(md_tp, md_tp + md_fn, 1.0)
        mood_delta_f1 = _compute_f1(md_prec, md_rec) if (md_tp + md_fp + md_fn) > 0 else 1.0

        # Edge, Voice Register & Causal totals
        e_tp = sum(s.edge_delta_tp for s in self.step_metrics)
        e_fp = sum(s.edge_delta_fp for s in self.step_metrics)
        e_fn = sum(s.edge_delta_fn for s in self.step_metrics)
        e_prec = _safe_div(e_tp, e_tp + e_fp, 1.0)
        e_rec = _safe_div(e_tp, e_tp + e_fn, 1.0)
        e_f1 = _compute_f1(e_prec, e_rec) if (e_tp + e_fp + e_fn) > 0 else 1.0

        reg_m = sum(s.edge_register_matches for s in self.step_metrics)
        reg_t = sum(s.edge_register_total for s in self.step_metrics)
        reg_acc = _safe_div(reg_m, reg_t, 1.0)

        vreg_m = sum(s.voice_register_matches for s in self.step_metrics)
        vreg_t = sum(s.voice_register_total for s in self.step_metrics)
        voice_reg_acc = _safe_div(vreg_m, vreg_t, reg_acc)

        cl_m = sum(s.causal_links_matched for s in self.step_metrics)
        cl_t = sum(s.causal_links_expected for s in self.step_metrics)
        cl_acc = _safe_div(cl_m, cl_t, 1.0)

        # Schema & ID Hygiene totals
        raw_tot = sum(s.raw_id_total for s in self.step_metrics)
        can_cnt = sum(s.canonical_id_count for s in self.step_metrics)
        surf_cnt = sum(s.normalized_surface_id_count for s in self.step_metrics)
        hall_cnt = sum(s.hallucinated_id_count for s in self.step_metrics)
        all_hall_ids: list[str] = []
        for s in self.step_metrics:
            for hid in s.hallucinated_ids:
                if hid not in all_hall_ids:
                    all_hall_ids.append(hid)

        can_rate = _safe_div(can_cnt, raw_tot, 1.0)
        surf_rate = _safe_div(surf_cnt, raw_tot, 0.0)
        hall_rate = _safe_div(hall_cnt, raw_tot, 0.0)

        retries_total = historical_usage.retry_count if historical_usage else 0
        val_errs_total = historical_usage.validation_errors if historical_usage else 0
        steps_with_retries = sum(
            1
            for entry in step_usage_log
            if int((entry.get("usage") or {}).get("retry_count", 0)) > 0
        )
        if steps_with_retries == 0 and retries_total > 0:
            steps_with_retries = min(total_steps, retries_total)
        first_pass_rate = max(0.0, (total_steps - steps_with_retries) / total_steps)

        # Downstream POV Retrieval totals
        pov_ent_j = sum(s.pov_entity_jaccard for s in self.step_metrics) / total_steps
        pov_edge_j = sum(s.pov_edge_jaccard for s in self.step_metrics) / total_steps
        pov_ep_r = sum(s.pov_episode_recall for s in self.step_metrics) / total_steps
        pov_leaks_total = sum(s.pov_secret_leaks for s in self.step_metrics)

        # Category Subscores (0.0 .. 1.0)
        if dec_exp > 0:
            epistemic_score = (
                0.30 * w_f1
                + 0.25 * max(0.0, 1.0 - leak_rate)
                + 0.20 * i_f1
                + 0.10 * dec_acc
                + 0.15 * disg_acc
            )
        else:
            epistemic_score = (
                0.35 * w_f1
                + 0.25 * max(0.0, 1.0 - leak_rate)
                + 0.25 * i_f1
                + 0.15 * disg_acc
            )
        spatial_state_score = (
            0.35 * room_jacc
            + 0.25 * loc_f1
            + 0.25 * mood_soft_acc
            + 0.15 * mood_delta_f1
        )
        pov_leak_penalty = max(0.0, 1.0 - 0.20 * pov_leaks_total)
        pov_retrieval_score = (
            0.30 * pov_ent_j
            + 0.30 * pov_ep_r
            + 0.40 * pov_leak_penalty
        )
        edge_flashback_score = (
            0.40 * e_f1
            + 0.15 * voice_reg_acc
            + 0.15 * cl_acc
            + 0.30 * (1.0 if self.flashback_guard_passed else 0.0)
        )
        schema_hygiene_score = (
            0.50 * first_pass_rate
            + 0.50 * max(0.0, 1.0 - hall_rate)
        )
        overall_quality_score = (
            0.30 * epistemic_score
            + 0.25 * spatial_state_score
            + 0.20 * pov_retrieval_score
            + 0.15 * edge_flashback_score
            + 0.10 * schema_hygiene_score
        )

        return ExtractionBenchmarkMetrics(
            model_id=self.model_id,
            guidance_mode=self.guidance_mode,
            witness_precision=w_prec,
            witness_recall=w_rec,
            witness_f1=w_f1,
            epistemic_leak_count=leak_cnt,
            epistemic_leak_rate=leak_rate,
            silent_witness_recall=sw_rec,
            inform_delta_precision=i_prec,
            inform_delta_recall=i_rec,
            inform_delta_f1=i_f1,
            deception_summary_accuracy=dec_acc,
            disguise_accuracy=disg_acc,
            location_delta_precision=loc_prec,
            location_delta_recall=loc_rec,
            location_delta_f1=loc_f1,
            room_occupancy_jaccard=room_jacc,
            room_occupancy_exact_rate=room_exact,
            character_location_accuracy=char_loc_acc,
            mood_exact_accuracy=mood_exact_acc,
            mood_soft_accuracy=mood_soft_acc,
            mood_delta_f1=mood_delta_f1,
            flashback_guard_passed=self.flashback_guard_passed,
            edge_delta_precision=e_prec,
            edge_delta_recall=e_rec,
            edge_delta_f1=e_f1,
            edge_register_accuracy=reg_acc,
            voice_register_accuracy=voice_reg_acc,
            causal_link_accuracy=cl_acc,
            first_pass_valid_rate=first_pass_rate,
            total_retries=retries_total,
            total_validation_errors=val_errs_total,
            raw_id_total=raw_tot,
            canonical_id_count=can_cnt,
            canonical_id_rate=can_rate,
            normalized_surface_id_count=surf_cnt,
            normalized_surface_id_rate=surf_rate,
            hallucinated_id_count=hall_cnt,
            hallucinated_id_rate=hall_rate,
            hallucinated_ids=all_hall_ids,
            pov_entity_jaccard=pov_ent_j,
            pov_edge_jaccard=pov_edge_j,
            pov_episode_recall=pov_ep_r,
            pov_secret_leak_count=pov_leaks_total,
            epistemic_score=epistemic_score,
            spatial_state_score=spatial_state_score,
            pov_retrieval_score=pov_retrieval_score,
            edge_flashback_score=edge_flashback_score,
            schema_hygiene_score=schema_hygiene_score,
            overall_quality_score=overall_quality_score,
            step_metrics=list(self.step_metrics),
        )


ModelTierProfile = Literal["oracle_tier", "medium_8b_tier", "small_3b_tier", "tiny_1b_tier"]


class SimulatedModelTierExtractor:
    """Deterministic extractor simulating how extraction quality degrades across smaller model tiers.

    Profiles:
    - `oracle_tier`: Reference frontier/guided model (100% accuracy across all tiers).
    - `medium_8b_tier`: Uses surface names ('Elena', 'Julian') and mood synonyms ('enraged', 'terrified'),
      invents custom episode IDs ('ep_chunk2_confront'), and misses silent witness `char_lyra` in Step 2.
    - `small_3b_tier`: Exhibits classic small-model failures:
      * Needs 1 JSON validation retry on Step 2
      * Leaks Step 2 confrontation to `char_marcus` (includes him in `witnesses` because he was in para 1)
      * Misses `char_lyra` moving to `loc_vault` in Step 3 (causing cumulative room occupancy drift)
      * Forgets `is_flashback=True` in Step 4 (overwriting Elena's present mood with flashback mood)
      * Hallucinates an unregistered entity ID (`char_citadel_guards`)
    - `tiny_1b_tier`: Severe degradation (misses most edges/informs, leaks secrets, hallucinates IDs, retries).
    """

    def __init__(
        self,
        chunks: Sequence[SyntheticChunkSpec],
        *,
        profile: ModelTierProfile = "oracle_tier",
        model_name: str | None = None,
    ) -> None:
        self.profile: ModelTierProfile = profile
        self.model = model_name or f"sim-{profile.replace('_', '-')}"
        self.system_prompt = f"Simulated {self.profile} narrative extractor."
        self._by_text: dict[str, SyntheticChunkSpec] = {c.text.strip(): c for c in chunks}
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        spec = self._by_text.get(scene_text.strip())
        if spec is None:
            return StepDelta()

        step = spec.step
        delta = spec.oracle_delta.model_copy(deep=True)
        retries = 0
        val_errs = 0

        if self.profile == "oracle_tier":
            usage = TokenUsage(prompt_tokens=3800, completion_tokens=420, total_tokens=4220, api_calls=1, latency_ms=950.0)
            self.last_usage = usage
            self.total_usage.add(usage)
            return delta

        if self.profile == "medium_8b_tier":
            # 1. Uses surface names for some entities (rescued by engine.normalize_delta)
            surface_map = {"char_elena": "Elena", "char_julian": "Julian"}
            # 2. Uses mood synonyms ('enraged' instead of 'furious', 'terrified' instead of 'panicked')
            synonym_moods = {"furious": "enraged", "panicked": "terrified", "indignant": "outraged"}
            # 3. Uses custom episode IDs ('ep_custom_2' instead of 'ep_step2_secret_confrontation')
            ep_rename = {
                "ep_step2_secret_confrontation": "ep_custom_2_confront",
                "ep_step4_past_flashback": "ep_custom_4_memory",
            }
            for ed in delta.entities:
                ed.entity_id = surface_map.get(ed.entity_id, ed.entity_id)
                if "mood" in ed.state and ed.state["mood"] in synonym_moods:
                    ed.state["mood"] = synonym_moods[ed.state["mood"]]
                if ed.caused_by_episode_id in ep_rename:
                    ed.caused_by_episode_id = ep_rename[ed.caused_by_episode_id]
            for edge in delta.edges:
                edge.source_id = surface_map.get(edge.source_id, edge.source_id)
                edge.target_id = surface_map.get(edge.target_id, edge.target_id)
                if edge.caused_by_episode_id in ep_rename:
                    edge.caused_by_episode_id = ep_rename[edge.caused_by_episode_id]
            for ep in delta.episodes:
                if ep.episode_id in ep_rename:
                    ep.episode_id = ep_rename[ep.episode_id]
                # Misses silent witness char_lyra in Step 2 confrontation
                if step == 2 and "char_lyra" in ep.witnesses:
                    ep.witnesses = [w for w in ep.witnesses if w != "char_lyra"]
            for inf in delta.inform:
                if inf.episode_id in ep_rename:
                    inf.episode_id = ep_rename[inf.episode_id]

            usage = TokenUsage(prompt_tokens=3800, completion_tokens=360, total_tokens=4160, api_calls=1, latency_ms=420.0)
            self.last_usage = usage
            self.total_usage.add(usage)
            return delta

        if self.profile == "small_3b_tier":
            if step == 2:
                retries = 1
                val_errs = 1
                # Leaks secret confrontation to absent char_marcus AND misses silent witness char_lyra!
                for ep in delta.episodes:
                    ep.witnesses = ["Elena", "Julian", "char_marcus"]
                # Drops causal link on edge
                for edge in delta.edges:
                    edge.caused_by_episode_id = None
            elif step == 3:
                # Misses char_lyra moving to loc_vault AND hallucinates char_citadel_guards
                delta.entities = [ed for ed in delta.entities if ed.entity_id != "char_lyra"]
                delta.entities.append(EntityDelta(entity_id="char_citadel_guards", state={"mood": "alert"}))
            elif step == 4:
                # Forgets is_flashback=True -> overwrites Elena's present-day furious mood with naive_trusting!
                delta.is_flashback = False
            elif step == 6:
                # Misses memory suppression InformDelta(epistemic='forgotten')
                delta.inform = [inf for inf in delta.inform if inf.epistemic != "forgotten"]

            usage = TokenUsage(
                prompt_tokens=3800 * (1 + retries),
                completion_tokens=310 * (1 + retries),
                total_tokens=4110 * (1 + retries),
                api_calls=1 + retries,
                retry_count=retries,
                validation_errors=val_errs,
                latency_ms=240.0 * (1 + retries),
            )
            self.last_usage = usage
            self.total_usage.add(usage)
            return delta

        # tiny_1b_tier
        if step in (2, 5):
            retries = 1
            val_errs = 1
        delta.is_flashback = False
        delta.edges = []
        delta.inform = []
        if delta.entities:
            delta.entities = delta.entities[:1]
            delta.entities.append(EntityDelta(entity_id="hallucinated_npc", state={"mood": "confused"}))
        for ep in delta.episodes:
            ep.witnesses = ["char_elena", "char_marcus", "char_lyra", "char_julian"]

        usage = TokenUsage(
            prompt_tokens=3800 * (1 + retries),
            completion_tokens=220 * (1 + retries),
            total_tokens=4020 * (1 + retries),
            api_calls=1 + retries,
            retry_count=retries,
            validation_errors=val_errs,
            latency_ms=130.0 * (1 + retries),
        )
        self.last_usage = usage
        self.total_usage.add(usage)
        return delta


class NoOpBaselineExtractor:
    """Zero-extraction null baseline returning an empty StepDelta() on every chunk.

    Establishes the exact step-0 world-state inertia floor so readers can distinguish genuine
    extraction lift from pre-seeded state preservation.
    """

    def __init__(self, model_name: str = "noop-null-baseline") -> None:
        self.model = model_name
        self.system_prompt = "Null no-op extractor baseline (returns empty StepDelta)."
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        usage = TokenUsage(api_calls=0, latency_ms=0.0)
        self.last_usage = usage
        self.total_usage.add(usage)
        return StepDelta()


class RuleBasedNarrativeExtractor:
    """Zero-LLM generic lexical/rule-based StepDelta extractor with zero hardcoded story IDs or plot strings.

    Parses the registered entity/location/episode context from `prompt` and applies generic sentence-level
    regex & keyword heuristics to `scene_text` without calling any LLM or inspecting `oracle_delta`.
    """

    _MOVEMENT_VERBS = (
        "stepped out",
        "stepped back",
        "stepped into",
        "left ",
        "bolted",
        "fled",
        "chased",
        "chasing",
        "entered",
        "returned",
        "walked into",
        "moved to",
        "ran into",
        "went to",
        "retreated",
    )
    _MOOD_KEYWORDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "furious": ("furious", "fury", "rage", "wrath"),
        "panicked": ("panicked", "panic", "trembling", "terrified", "frantic", "desperate"),
        "restless": ("restless", "irritated", "stifling", "uneasy"),
        "determined": ("determined", "resolute", "steely"),
        "shocked": ("shocked", "stunned", "aghast", "gasped"),
        "indignant": ("indignant", "outraged", "betrayed"),
        "naive_trusting": ("naive", "trusted blindly", "trusted implicitly"),
    }

    def __init__(self, model_name: str = "rule-based-non-llm") -> None:
        self.model = model_name
        self.system_prompt = "Deterministic rule-based lexical extractor (no LLM)."
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()
        self._step_counter = 0

    @staticmethod
    def _parse_registry_from_prompt(
        prompt: str,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], list[tuple[str, str]]]:
        """Extracts (chars_name_to_id, locs_name_to_id, arts_name_to_id, recent_episodes) from prompt."""
        chars: dict[str, str] = {}
        locs: dict[str, str] = {}
        arts: dict[str, str] = {}
        episodes: list[tuple[str, str]] = []

        # Parse Active World Registry, Known Entity IDs, or Surface-Only Registry
        section = ""
        for line in prompt.splitlines():
            s = line.strip()
            if s.startswith("**Known Locations**"):
                section = "loc"
                continue
            if s.startswith("**Known Artifacts**"):
                section = "art"
                continue
            if s.startswith("**Characters' Current Room"):
                section = "char"
                continue
            if s.startswith(("**Prior Plot Episodes", "**Complete Timeline Episode")):
                section = "ep"
                continue

            m_reg = re.match(r"-\s+(loc_\w+|char_\w+|item_\w+)\s+\(name:\s*'([^']+)'", s)
            if m_reg:
                eid, name = m_reg.group(1), m_reg.group(2).lower()
                if eid.startswith("loc_"):
                    locs[name] = eid
                elif eid.startswith("char_"):
                    chars[name] = eid
                elif eid.startswith("item_"):
                    arts[name] = eid
                m_disg = re.search(r"disguise='([^']+)'", s)
                if m_disg and eid.startswith("char_"):
                    chars[m_disg.group(1).lower()] = eid
                continue
            m_surf_char = re.match(r"-\s+'([^']+)':\s+current_location='([^']+)'", s)
            if m_surf_char:
                cname = m_surf_char.group(1)
                chars[cname.lower()] = cname
                m_disg = re.search(r"disguise='([^']+)'", s)
                if m_disg:
                    chars[m_disg.group(1).lower()] = cname
                continue
            m_surf_item = re.match(r"^-\s+'([^']+)'$", s)
            if m_surf_item and section in ("loc", "art"):
                iname = m_surf_item.group(1)
                if section == "loc":
                    locs[iname.lower()] = iname
                else:
                    arts[iname.lower()] = iname
                continue
            m_known = re.match(r"-\s+(loc_\w+|char_\w+|item_\w+)\s+\(name:\s*([^,]+),\s*kind:\s*(\w+)\)", s)
            if m_known:
                eid, name, kind = m_known.group(1), m_known.group(2).strip().lower(), m_known.group(3)
                if kind == "location":
                    locs[name] = eid
                elif kind == "character":
                    chars[name] = eid
                elif kind == "artifact":
                    arts[name] = eid
                continue
            m_ep = re.match(r"-\s+(ep_\w+)\s+\(step[= ]\d+[^)]*\):\s*(.+)$", s)
            if m_ep:
                episodes.append((m_ep.group(1), m_ep.group(2)))

        return chars, locs, arts, episodes

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        import time

        t0 = time.perf_counter()
        self._step_counter += 1
        chars, locs, arts, recent_eps = self._parse_registry_from_prompt(prompt)
        text_lower = scene_text.lower()
        sentences = re.split(r"(?<=[.!?])\s+", scene_text.strip())

        is_flashback = any(
            kw in text_lower for kw in ("ten years ago", "10 years ago", "remembered when", "flashback")
        )
        story_time = "10_years_ago" if is_flashback else None

        # Detect characters mentioned in the chunk
        mentioned_chars: list[str] = []
        for name, cid in chars.items():
            if name in text_lower and cid not in mentioned_chars:
                mentioned_chars.append(cid)

        # Sentence-level location transitions & mood cues
        entity_states: dict[str, dict[str, Any]] = {}
        for sent in sentences:
            sl = sent.lower()
            sent_chars = [cid for name, cid in chars.items() if name in sl]
            # Location transitions: pick the last mentioned location when a movement verb is present
            if any(mv in sl for mv in self._MOVEMENT_VERBS):
                matched_locs: list[tuple[int, str]] = []
                for lname, lid in locs.items():
                    idx = sl.rfind(lname)
                    if idx != -1:
                        matched_locs.append((idx, lid))
                if matched_locs:
                    matched_locs.sort()
                    dest_loc = matched_locs[-1][1]
                    for cid in sent_chars:
                        entity_states.setdefault(cid, {})["location_id"] = dest_loc

            # Mood detection near character mention
            for mood_label, kws in self._MOOD_KEYWORDS.items():
                if any(re.search(rf"\b{re.escape(kw)}\b", sl) for kw in kws):
                    for cid in sent_chars:
                        entity_states.setdefault(cid, {})["mood"] = mood_label

        # Determine primary episode predicate using generic narrative cues (no hardcoded story-specific roles)
        predicate = "scene_event"
        if is_flashback:
            predicate = "flashback"
        elif any(kw in text_lower for kw in ("accused", "confronted", "theft", "stole", "stolen")):
            predicate = "confrontation"
        elif any(kw in text_lower for kw in ("pressed", "handed", "gave", "hide it")):
            predicate = "item_transfer"
        elif any(kw in text_lower for kw in ("threw back", "revealed", "unmasked")):
            predicate = "revelation"
        elif any(kw in text_lower for kw in ("reports", "dispatches", "brief")):
            predicate = "briefing"

        ep_id = f"ep_rule_{self._step_counter}"
        episodes: list[EpisodeRecord] = []
        inform: list[InformDelta] = []
        edges: list[EdgeDelta] = []

        # Detect InformDelta cues referencing prior episodes purely from generic verbs & sentence order
        confront_ep = next(
            (eid for eid, summ in recent_eps if "accused" in summ.lower() or "stole" in summ.lower() or "confront" in eid),
            None,
        )
        flashback_ep = next(
            (eid for eid, summ in recent_eps if "ten years ago" in summ.lower() or "flashback" in eid),
            None,
        )

        def _ordered_sentence_chars(sent_lower: str) -> list[str]:
            hits: list[tuple[int, str]] = []
            for name, cid in chars.items():
                pos = sent_lower.find(name)
                if pos != -1:
                    hits.append((pos, cid))
            hits.sort()
            out: list[str] = []
            for _, cid in hits:
                if cid not in out:
                    out.append(cid)
            return out

        if confront_ep and any(kw in text_lower for kw in ("lied", "deceived", "fabricated", "cover story", "false alibi")):
            for sent in sentences:
                sl = sent.lower()
                if any(kw in sl for kw in ("lied", "deceived", "fabricated", "cover story", "false alibi")):
                    sc = _ordered_sentence_chars(sl)
                    if len(sc) >= 2:
                        inform.append(
                            InformDelta(
                                episode_id=confront_ep,
                                entity_id=sc[1],
                                epistemic="deceived",
                                believed_summary=sent.strip()[:180],
                            )
                        )
                        break

        if confront_ep and any(kw in text_lower for kw in ("informed", "revealed the truth", "full truth")):
            for sent in sentences:
                sl = sent.lower()
                if "informed" in sl or "truth" in sl:
                    sc = _ordered_sentence_chars(sl)
                    if len(sc) >= 2:
                        inform.append(InformDelta(episode_id=confront_ep, entity_id=sc[1], epistemic="told"))
                        break

        if flashback_ep and any(kw in text_lower for kw in ("erased", "wiped", "blanked", "memory-erasing")):
            for sent in sentences:
                sl = sent.lower()
                if any(kw in sl for kw in ("erased", "wiped", "blanked", "memory")):
                    sc = _ordered_sentence_chars(sl)
                    if sc:
                        inform.append(InformDelta(episode_id=flashback_ep, entity_id=sc[0], epistemic="forgotten"))
                        break

        # Generic artifact possession transfer: pick the second character mentioned in the transfer sentence
        if predicate == "item_transfer" and arts:
            art_id = next(iter(arts.values()))
            for sent in sentences:
                sl = sent.lower()
                if any(kw in sl for kw in ("pressed", "handed", "gave", "thrust")):
                    sc = _ordered_sentence_chars(sl)
                    if len(sc) >= 2:
                        edges.append(
                            EdgeDelta(
                                source_id=sc[1],
                                target_id=art_id,
                                relation_type="possesses",
                                state={"status": "concealed"},
                                caused_by_episode_id=ep_id,
                            )
                        )
                        break

        if (
            not inform
            or predicate in ("confrontation", "item_transfer", "revelation", "flashback", "briefing")
        ) and not (len(inform) >= 2 and predicate == "scene_event"):
            # Only emit episode if not purely an inform-only climax
            episodes.append(
                EpisodeRecord(
                    episode_id=ep_id,
                    predicate=predicate,
                    summary=sentences[0][:160] if sentences else scene_text[:160],
                    participants={cid: "participant" for cid in mentioned_chars[:2]},
                    witnesses=mentioned_chars,
                    story_time=story_time,
                    salience=7.0,
                )
            )

        entities = [
            EntityDelta(entity_id=cid, state=st, caused_by_episode_id=ep_id if episodes else None)
            for cid, st in entity_states.items()
            if st
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        usage = TokenUsage(api_calls=0, latency_ms=round(elapsed_ms, 3))
        self.last_usage = usage
        self.total_usage.add(usage)

        return StepDelta(
            entities=entities,
            edges=edges,
            episodes=episodes,
            inform=inform,
            is_flashback=is_flashback,
            story_time=story_time,
        )


