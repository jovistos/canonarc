from __future__ import annotations

from pathlib import Path
import os
import pytest

from canonarc import (
    CachedDeltaExtractor,
    OpenAICompatibleExtractor,
    StepDelta,
    TokenUsage,
    load_dotenv,
    resolve_lmstudio_model,
)
from synthetic_story_harness import (
    BlueprintOracleExtractor,
    DeterministicStoryCompiler,
    HeuristicNarrativeExtractor,
    LLMStoryClosedLoopHarness,
    SyntheticStoryE2ERunner,
    SynthStoryParams,
)
from synthetic_story_library import (
    RuleBasedNarrativeExtractor,
    SimulatedModelTierExtractor,
    StoryLibraryManager,
    moods_soft_match,
)


def test_mode1_deterministic_blueprint_oracle_e2e() -> None:
    """Mode 1A: Deterministic Blueprint E2E test from SynthStoryParams.

    Verifies:
    - Small chunk sizing (18-55 words per chunk across 10 steps)
    - Room occupancy changes across 3 rooms (4 -> 3 -> 1 -> 2 -> 4 characters)
    - Tier 1-4 entity moods, asymmetric stances, facade masking, belief_state overrides
    - Tier 4 scoped disguises ('the hooded courier') pierced by Elena/Julian, masked to Marcus
    - Tier 5 Gate 1 filler rejection ('Yes, cousin.') & Gate 2 low-quality rejection (Step 6)
    - Tier 5 register shift ('Vi' -> 'ti') and voice exemplar retrieval
    - Tier 6 POV epistemic isolation, deception ('believed_summary'), secondhand truth ('told'),
      memory suppression ('forgotten'), flashback protection ('is_flashback=True'), and consolidation
    - Causal time-travel branching ('julian_confesses_route'), diffing, and rollback
    - Diagnostic recording of all 5 uncovered/boundary scenarios
    """
    params = SynthStoryParams.default_conspiracy_seed()
    compiler = DeterministicStoryCompiler(params)
    chunks = compiler.compile_chunks()

    assert len(chunks) == 10

    runner = SyntheticStoryE2ERunner(params)
    oracle_extractor = BlueprintOracleExtractor(chunks)
    report = runner.run_pipeline(
        chunks,
        extractor=oracle_extractor,
        mode="deterministic_oracle",
    )

    # 1. Verify small chunk size constraint across the entire generated story
    min_words, max_words = params.chunk_target_words
    for idx, wc in enumerate(report.chunk_word_counts, start=1):
        assert min_words <= wc <= max_words, f"Chunk {idx} word count {wc} outside [{min_words}, {max_words}]"

    # 2. Verify 100% pass rate on all covered engine checks
    assert report.covered_failures == [], f"Covered checks failed: {report.covered_failures}"
    assert report.covered_pass_rate == 1.0
    assert report.covered_checks_total >= 45

    # 3. Verify Tier 5 Gate 1 & Gate 2 filtering statistics
    # - 6 rich dialogue lines pass Gate 1 (Steps 1, 2, 4, 6, 8, 10)
    # - 2 short dialogue lines ("Yes, cousin." at Step 1 and 11-word sarcastic remark at Step 9) are rejected by Gate 1
    # - 1 low-quality panicked line (Step 6, quality_score=5.5 < 8.0) is rejected by Gate 2
    assert report.gate1_accepted_total == 6
    assert report.gate1_rejected_total == 2
    assert report.gate2_accepted_total == 5
    assert report.gate2_rejected_total == 1

    # 4. Verify all 5 Uncovered / Boundary Diagnostics were captured and pinpointed
    diag_by_tag = {d.tag: d for d in report.boundary_diagnostics}
    assert set(diag_by_tag.keys()) == {
        "implicit_room_witness_unmentioned_in_text",
        "cross_chunk_pronoun_coreference",
        "intra_scene_same_mood_repeat",
        "exclusive_inventory_transfer",
        "sarcastic_surface_dialogue_vs_true_mood",
    }

    # Boundary 1: Lyra is present when location_id="loc_council" is passed, but absent from text-only scan
    b1 = diag_by_tag["implicit_room_witness_unmentioned_in_text"]
    assert "present_with_location_id=True" in b1.observed_behavior
    assert "present_from_text_alone=False" in b1.observed_behavior
    assert b1.passed_strict_ideal is False

    # Boundary 2: Pronoun-only Chunk 5 ("she", "he") does not resolve mentions via lexical resolver alone
    b2 = diag_by_tag["cross_chunk_pronoun_coreference"]
    assert "resolved_mentions_from_pronouns=[]" in b2.observed_behavior
    assert b2.passed_strict_ideal is False

    # Boundary 3: Re-emitting mood='furious' at Step 5 is deduplicated by the Idempotent State Guard,
    # preserving prior_mood='calm' (step 0) instead of overwriting it with 'furious' (step 4).
    b3 = diag_by_tag["intra_scene_same_mood_repeat"]
    assert "prior_mood='calm' at prior_step=0" in b3.observed_behavior
    assert "revision_count=2" in b3.observed_behavior
    assert b3.passed_strict_ideal is True

    # Boundary 4: Adding Lyra's possesses edge without remove=True on Julian's edge leaves both active
    b4 = diag_by_tag["exclusive_inventory_transfer"]
    assert "char_julian" in b4.observed_behavior and "char_lyra" in b4.observed_behavior
    assert b4.passed_strict_ideal is False

    # Boundary 5: Sarcastic surface praise in Chunk 9 does not corrupt Elena's true mood ('furious')
    b5 = diag_by_tag["sarcastic_surface_dialogue_vs_true_mood"]
    assert b5.passed_strict_ideal is True


def test_mode1_sliding_window_halo_resolves_cross_chunk_coreference() -> None:
    """Mode 1A with use_sliding_window=True: Read-Only Halo resolves cross-chunk pronouns & preserves trajectory."""
    params = SynthStoryParams.default_conspiracy_seed()
    chunks = DeterministicStoryCompiler(params).compile_chunks()
    runner = SyntheticStoryE2ERunner(params)
    report = runner.run_pipeline(
        chunks,
        extractor=BlueprintOracleExtractor(chunks),
        mode="deterministic_oracle",
        use_sliding_window=True,
    )

    assert report.covered_failures == [], f"Covered checks failed: {report.covered_failures}"
    assert report.covered_pass_rate == 1.0

    diag_by_tag = {d.tag: d for d in report.boundary_diagnostics}
    b2 = diag_by_tag["cross_chunk_pronoun_coreference"]
    assert "char_elena" in b2.observed_behavior and "char_julian" in b2.observed_behavior
    assert b2.passed_strict_ideal is True

    b3 = diag_by_tag["intra_scene_same_mood_repeat"]
    assert b3.passed_strict_ideal is True


def test_mode1_heuristic_text_only_extractor_e2e() -> None:
    """Mode 1B: Runs the small-chunk story through a purely text-driven HeuristicNarrativeExtractor.

    Demonstrates the difference between oracle extraction and single-chunk text extraction:
    - Surface names ('Elena', 'Marcus', 'Julian', 'Obsidian Signet') are normalized to canonical IDs
      by engine.normalize_delta() inside StepTransaction.extract_and_commit().
    - In Chunk 4, because Lyra sits silently in the Council Chamber and is not mentioned in Chunk 4's
      sentences, a text-only extractor omits her from ep_step4_seal_accusation.witnesses!
    """
    params = SynthStoryParams.default_conspiracy_seed()
    chunks = DeterministicStoryCompiler(params).compile_chunks()

    runner = SyntheticStoryE2ERunner(params)
    heuristic_extractor = HeuristicNarrativeExtractor()
    report = runner.run_pipeline(
        chunks,
        extractor=heuristic_extractor,
        mode="deterministic_heuristic",
    )

    assert heuristic_extractor.call_count == 10

    # Verify normalize_delta mapped surface names ("Elena", "Julian") to canonical IDs ("char_elena", "char_julian")
    ctx_final = runner.engine.retrieve_context(
        step=10,
        location_id="loc_council",
        inclusive=True,
    )
    mood_by_id = {e.entity_id: e.state.get("mood") for e in ctx_final.entities}
    assert mood_by_id["char_elena"] == "furious"
    assert mood_by_id["char_marcus"] == "indignant"
    assert mood_by_id["char_julian"] == "shocked"
    assert mood_by_id["char_lyra"] == "determined"

    # Verify the exact small-chunk witness blind spot exposed by Mode 1B:
    # In Chunk 4, Lyra was in loc_council (from Step 0), but Chunk 4's text only mentioned Elena and Julian.
    diag_by_tag = {d.tag: d for d in report.boundary_diagnostics}
    b1 = diag_by_tag["implicit_room_witness_unmentioned_in_text"]
    assert "present_with_location_id=True" in b1.observed_behavior
    assert "episode_witnessed=False" in b1.observed_behavior


def test_mode2_llm_closed_loop_e2e_with_retry_and_normalization() -> None:
    """Mode 2: Closed-Loop LLM Story Generation + Structured Extraction E2E Test.

    Tests the full LLM pipeline:
    1. Builds a story generation prompt from SynthStoryParams and parses Markdown-fenced JSON chunks.
    2. Runs StepTransaction.extract_and_commit() using CallableExtractor.
    3. Exercises automatic JSON validation retry when the LLM returns malformed JSON on Step 4 (attempt 1 -> 2).
    4. Exercises engine.normalize_delta() when the LLM uses canonical names ('Elena', 'Marcus') and
       epithet aliases ('the young treasurer') instead of internal entity_ids.
    5. Verifies 100% pass rate against the starting SynthStoryParams scorecard.
    """
    params = SynthStoryParams.default_conspiracy_seed()
    llm_harness = LLMStoryClosedLoopHarness(params, max_retries=2)

    # 1. Generate story chunks from SynthStoryParams via the LLM writer
    story_prompt = llm_harness.build_story_generation_prompt()
    assert "The Obsidian Signet Conspiracy" in story_prompt
    assert "char_elena" in story_prompt

    generated_chunks = llm_harness.generate_story_chunks()
    assert len(generated_chunks) == 10

    # 2. Run the full E2E pipeline using the LLM extractor
    runner = SyntheticStoryE2ERunner(params)
    report = runner.run_pipeline(
        generated_chunks,
        extractor=llm_harness.extractor,
        mode="live_llm_closed_loop",
    )

    # 3. Verify that Step 4 triggered a retry after malformed JSON and succeeded on attempt 2
    # (10 chunks + 1 retry on Step 4 = 11 total LLM extractor calls)
    assert len(llm_harness.raw_llm_calls) == 11
    step4_calls = [c for c in llm_harness.raw_llm_calls if "Elena confronted Julian" in c["scene_text"]]
    assert len(step4_calls) == 2
    assert step4_calls[0]["attempt"] == 1
    assert step4_calls[1]["attempt"] == 2
    assert "Previous output failed StepDelta JSON validation" in step4_calls[1]["prompt"]

    # 4. Verify 100% pass rate on all covered mechanics after LLM extraction & ID normalization
    assert report.covered_failures == [], f"Mode 2 LLM Closed-Loop failures: {report.covered_failures}"
    assert report.covered_pass_rate == 1.0


def test_cached_delta_extractor_keying_and_zero_redo(tmp_path: Path) -> None:
    """Verifies that CachedDeltaExtractor caches by (chunk, prompt, model) and never redoes completed extractions."""
    raw_calls: list[tuple[str, str]] = []

    class _CountingExtractor:
        def __init__(self, model: str = "deepseek-flash") -> None:
            self.model = model
            self.system_prompt = "Extract StepDelta JSON."
            self.total_usage = TokenUsage()

        def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
            raw_calls.append((prompt, scene_text))
            step_usage = TokenUsage(prompt_tokens=600, completion_tokens=200, total_tokens=800, api_calls=1)
            self.total_usage.add(step_usage)
            return StepDelta(scene_id="s1", story_time="day_1")

    inner = _CountingExtractor(model="deepseek-flash")
    cached = CachedDeltaExtractor(inner, cache_dir=tmp_path / "extractions")

    # 1st call: cache miss -> invokes inner extractor and writes cache file
    d1 = cached.extract_delta("Prompt A", "Chunk 1 text")
    assert d1.scene_id == "s1"
    assert len(raw_calls) == 1
    assert cached.cache_misses == 1
    assert cached.cache_hits == 0
    assert cached.billable_usage.total_tokens == 800

    # 2nd call with identical (chunk, prompt, model): cache hit -> zero inner calls, zero new billable tokens!
    d2 = cached.extract_delta("Prompt A", "Chunk 1 text")
    assert d2.scene_id == "s1"
    assert len(raw_calls) == 1
    assert cached.cache_misses == 1
    assert cached.cache_hits == 1
    assert cached.billable_usage.total_tokens == 800
    assert cached.historical_usage.total_tokens == 1600

    # 3rd call with different prompt: cache miss
    cached.extract_delta("Prompt B", "Chunk 1 text")
    assert len(raw_calls) == 2
    assert cached.cache_misses == 2

    # 4th call with different model on same cache_dir: cache miss
    cached_other_model = CachedDeltaExtractor(
        _CountingExtractor(model="deepseek-v4-pro"),
        cache_dir=tmp_path / "extractions",
    )
    cached_other_model.extract_delta("Prompt A", "Chunk 1 text")
    assert len(raw_calls) == 3
    assert cached_other_model.cache_misses == 1


@pytest.mark.parametrize(
    "story_slug",
    ["obsidian_signet_3k", "chronos_station_3k", "crimson_masquerade_3k"],
)
def test_deepseek_flash_3k_stories_cached_generation_and_extraction_e2e(story_slug: str) -> None:
    """End-to-End test for 3,000-token (~6 x 500-token) synthetic stories in subfolders with DeepSeek Flash.

    Verifies:
    1. Stories are saved in subfolders (`tests/synthetic_stories/<story_slug>/story.json`, `story.md`, `usage_report.json`).
    2. Each story has 6 rich chunks averaging ~500 tokens (~3,000 tokens total).
    3. Running the test uses the already cached/saved story (`story_cache_hit=True` once generated).
    4. Running extraction a second time with the same `(chunk, prompt, model)` hits the disk extraction cache
       100% (`extraction_cache_hits == 6`, `extraction_cache_misses == 0`, `billable_usage.total_tokens == 0`).
    5. All Tier 1-6 narrative memory checks pass 100% and token usage + USD cost estimates are written to disk.
    """
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "paste_your_deepseek_api_key_here":
        pytest.skip("DEEPSEEK_API_KEY not configured in .env")

    mgr = StoryLibraryManager()

    # Ensure micro baseline is also persisted in its own subfolder
    micro = mgr.get_or_create_story("obsidian_signet_micro")
    assert (micro.story_dir / "story.json").is_file()
    assert (micro.story_dir / "story.md").is_file()

    # Run 1: Generates (or loads cached) 3K story and runs (or loads cached) DeepSeek Flash extraction
    res1 = mgr.run_story_extraction_e2e(story_slug)
    assert len(res1.story.chunks) == 6
    assert 2200 <= res1.story.total_estimated_tokens <= 4200, (
        f"Expected ~3,000 tokens for {story_slug}, got {res1.story.total_estimated_tokens}"
    )
    for idx, tok_count in enumerate(res1.story.chunk_token_counts, start=1):
        assert 330 <= tok_count <= 750, f"Chunk {idx} token count {tok_count} outside expected ~500-token band"

    assert res1.failures == [], f"E2E verification failures for {story_slug}: {res1.failures}"
    assert res1.pass_rate == 1.0

    # Verify files saved for inspection in subfolder
    subfolder = res1.story.story_dir
    assert (subfolder / "story.json").is_file()
    assert (subfolder / "story.md").is_file()
    assert (subfolder / "usage_report.json").is_file()
    assert (subfolder / "extractions" / "step_01_delta.json").is_file()
    assert (subfolder / "extractions" / "step_06_delta.json").is_file()

    # Run 2: Must reuse the cached story AND hit 100% of cached LLM extractions with 0 new API calls!
    res2 = mgr.run_story_extraction_e2e(story_slug)
    assert res2.story_cache_hit is True
    assert res2.extraction_cache_hits == 6
    assert res2.extraction_cache_misses == 0
    assert res2.extraction_billable_usage.total_tokens == 0
    assert res2.combined_historical_usage.total_tokens > 0
    assert res2.pass_rate == 1.0
    assert res2.metrics.overall_quality_score >= 0.98
    assert res2.metrics.epistemic_leak_count == 0
    assert res2.metrics.pov_secret_leak_count == 0
    assert res2.metrics.flashback_guard_passed is True


def test_multi_model_extraction_degradation_metrics_on_3k_story(tmp_path: Path) -> None:
    """Verifies granular extraction & POV metrics across simulated model tiers (Oracle -> 8B -> 3B -> 1B)
    on a 3,000-token story under unguided extraction (`guidance_mode='none'`).
    """
    source_mgr = StoryLibraryManager()
    orig_story = source_mgr.get_or_create_story("obsidian_signet_3k")

    # Copy story.json into isolated tmp_path library root so test artifacts don't dirty repo cache
    tmp_story_dir = tmp_path / "obsidian_signet_3k"
    tmp_story_dir.mkdir(parents=True, exist_ok=True)
    (tmp_story_dir / "story.json").write_text(
        (orig_story.story_dir / "story.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    mgr = StoryLibraryManager(root_dir=tmp_path)
    story = mgr.get_or_create_story("obsidian_signet_3k")

    ext_oracle = SimulatedModelTierExtractor(story.chunks, profile="oracle_tier", model_name="tier1-frontier")
    ext_8b = SimulatedModelTierExtractor(story.chunks, profile="medium_8b_tier", model_name="tier2-qwen-8b")
    ext_3b = SimulatedModelTierExtractor(story.chunks, profile="small_3b_tier", model_name="tier3-llama-3b")
    ext_1b = SimulatedModelTierExtractor(story.chunks, profile="tiny_1b_tier", model_name="tier4-tiny-1b")

    comparison = mgr.compare_models(
        "obsidian_signet_3k",
        extractors=[ext_oracle, ext_8b, ext_3b, ext_1b],
        guidance_mode="none",
    )

    assert len(comparison.results) == 4
    r_oracle, r_8b, r_3b, r_1b = comparison.results
    m_oracle, m_8b, m_3b, m_1b = r_oracle.metrics, r_8b.metrics, r_3b.metrics, r_1b.metrics

    # 1. Overall quality score degrades monotonically with smaller model tiers
    assert m_oracle.overall_quality_score == pytest.approx(1.0)
    assert m_oracle.overall_quality_score > m_8b.overall_quality_score > m_3b.overall_quality_score > m_1b.overall_quality_score

    # 2. Medium 8B tier:
    # - Uses surface names ('Elena', 'Julian') rescued by normalize_delta -> 0 hallucinated IDs
    assert m_8b.normalized_surface_id_count > 0
    assert m_8b.hallucinated_id_count == 0
    # - Uses mood synonyms ('enraged' for 'furious', 'terrified' for 'panicked') -> soft mood 100%, exact mood < 100%
    assert m_8b.mood_soft_accuracy == pytest.approx(1.0)
    assert m_8b.mood_exact_accuracy < 1.0
    # - Custom episode IDs ('ep_custom_2_confront') are aligned automatically -> InformDelta F1 & Causal Link Acc = 100%
    assert m_8b.inform_delta_f1 == pytest.approx(1.0)
    assert m_8b.causal_link_accuracy == pytest.approx(1.0)
    # - Drops silent witness char_lyra in Step 2, but has 0 false-positive epistemic leaks
    assert m_8b.silent_witness_recall < 1.0
    assert m_8b.epistemic_leak_count == 0
    assert m_8b.pov_secret_leak_count == 0

    # 3. Small 3B tier:
    # - Leaks Step 2 confrontation to absent char_marcus -> epistemic_leak_count >= 1 & pov_secret_leak_count >= 1
    assert m_3b.epistemic_leak_count >= 1
    assert m_3b.epistemic_leak_rate > 0.0
    assert m_3b.pov_secret_leak_count >= 1
    # - Misses char_lyra moving to loc_vault in Step 3 -> step-local location F1 < 1.0 & cumulative room Jaccard < 1.0
    assert m_3b.location_delta_f1 < 1.0
    assert m_3b.room_occupancy_jaccard < 1.0
    # - Forgets is_flashback=True in Step 4 -> flashback_guard_passed is False
    assert m_3b.flashback_guard_passed is False
    # - Hallucinates 'char_citadel_guards' and requires a schema validation retry on Step 2
    assert m_3b.hallucinated_id_count >= 1
    assert "char_citadel_guards" in m_3b.hallucinated_ids
    assert m_3b.total_retries >= 1
    assert m_3b.first_pass_valid_rate < 1.0

    # 4. Verify comparison artifacts & Markdown table were persisted
    comp_json = tmp_story_dir / "benchmarks" / "comparison_none.json"
    comp_md = tmp_story_dir / "benchmarks" / "comparison_none.md"
    assert comp_json.is_file()
    assert comp_md.is_file()
    md_text = comp_md.read_text(encoding="utf-8")
    assert "tier1-frontier" in md_text
    assert "tier2-qwen-8b" in md_text
    assert "tier3-llama-3b" in md_text
    assert "tier4-tiny-1b" in md_text


def test_guidance_modes_and_mood_soft_matching(tmp_path: Path) -> None:
    """Verifies guidance_mode ('full_hints', 'canonical_episode_ids', 'none') and mood synonym matching."""
    assert moods_soft_match("furious", "enraged") is True
    assert moods_soft_match("panicked", "terrified") is True
    assert moods_soft_match("shocked", "stunned") is True
    assert moods_soft_match("furious", "calm") is False

    mgr = StoryLibraryManager(root_dir=tmp_path)
    bp = mgr.catalog["obsidian_signet_3k"]
    beat2 = bp.beats[1]

    hint_full = mgr._build_beat_hint_for_mode(beat2, "full_hints")
    hint_ids = mgr._build_beat_hint_for_mode(beat2, "canonical_episode_ids")
    hint_none = mgr._build_beat_hint_for_mode(beat2, "none")

    assert "EntityDelta" in hint_full and "ep_step2_secret_confrontation" in hint_full
    assert "ep_step2_secret_confrontation" in hint_ids and "EntityDelta" not in hint_ids
    assert hint_none == ""


def test_lmstudio_default_model_discovery_and_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies OpenAICompatibleExtractor.from_lmstudio() auto-selects the currently loaded model
    from LM Studio (/api/v0/models state=='loaded') without requiring a model argument.
    """
    import io
    import json
    import urllib.request

    fake_v0_models = {
        "data": [
            {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings", "state": "loaded"},
            {"id": "gemma-4-31b-en2sr-v4@q4_k_m", "type": "llm", "state": "not-loaded"},
            {"id": "qwen/qwen3.8-27b", "type": "vlm", "state": "loaded"},
        ]
    }
    fake_completion = {
        "model": "qwen/qwen3.8-27b",
        "choices": [
            {
                "message": {
                    "content": "<think>\nAnalyzing scene...\n</think>\n```json\n{\"scene_id\": \"s_lm\", \"story_time\": \"night\"}\n```"
                }
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }

    class _FakeResp:
        def __init__(self, payload: dict) -> None:
            self._bytes = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._bytes

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 5.0) -> _FakeResp:  # noqa: ARG001
        if req.full_url.endswith("/api/v0/models"):
            return _FakeResp(fake_v0_models)
        if req.full_url.endswith("/chat/completions"):
            sent = json.loads(req.data.decode("utf-8"))
            assert sent["model"] == "qwen/qwen3.8-27b"
            assert "response_format" not in sent
            return _FakeResp(fake_completion)
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    resolved = resolve_lmstudio_model("http://localhost:1234/v1")
    assert resolved == "qwen/qwen3.8-27b"

    ext = OpenAICompatibleExtractor.from_lmstudio()
    assert ext.model == "qwen/qwen3.8-27b"
    delta = ext.extract_delta("Extract scene delta", "The night was silent.")
    assert delta.scene_id == "s_lm"
    assert delta.story_time == "night"
    assert ext.total_usage.total_tokens == 150


def test_run_level_skip_when_model_prompt_and_story_unchanged(tmp_path: Path) -> None:
    """Verifies that run_story_extraction_e2e skips execution when (model, prompt, story) is unchanged,
    and re-runs only when the prompt, model, or story changes.
    """
    source_mgr = StoryLibraryManager()
    orig_story = source_mgr.get_or_create_story("obsidian_signet_3k")

    tmp_story_dir = tmp_path / "obsidian_signet_3k"
    tmp_story_dir.mkdir(parents=True, exist_ok=True)
    (tmp_story_dir / "story.json").write_text(
        (orig_story.story_dir / "story.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    mgr = StoryLibraryManager(root_dir=tmp_path)
    story = mgr.get_or_create_story("obsidian_signet_3k")
    ext = SimulatedModelTierExtractor(story.chunks, profile="oracle_tier", model_name="cached-test-model")

    # 1st run: executes and writes benchmark cache with run_cache_key
    r1 = mgr.run_story_extraction_e2e("obsidian_signet_3k", base_extractor=ext, guidance_mode="full_hints")
    assert r1.skipped_cached_run is False
    assert r1.run_cache_key != ""

    # 2nd run with identical (model, prompt, story): skipped immediately!
    r2 = mgr.run_story_extraction_e2e("obsidian_signet_3k", base_extractor=ext, guidance_mode="full_hints")
    assert r2.skipped_cached_run is True
    assert r2.run_cache_key == r1.run_cache_key
    assert r2.metrics.overall_quality_score == pytest.approx(r1.metrics.overall_quality_score)

    # 3rd run with updated prompt (guidance_mode='none'): cache key changes -> runs!
    r3 = mgr.run_story_extraction_e2e("obsidian_signet_3k", base_extractor=ext, guidance_mode="none")
    assert r3.skipped_cached_run is False
    assert r3.run_cache_key != r1.run_cache_key


def test_gemini_hydrogen_from_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies OpenAICompatibleExtractor.from_gemini() defaults to gemini-4-hydrogen and reads GEMINI_API_KEY."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_EXTRACT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    ext = OpenAICompatibleExtractor.from_gemini()
    assert ext.model == "gemini-4-hydrogen"
    assert ext.api_key == "test-gemini-key"
    assert ext.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_gemini_hydrogen_and_model_comparison_3k_e2e() -> None:
    """Live E2E benchmark & comparison for Gemini 4 Hydrogen (`gemini-4-hydrogen`) vs DeepSeek & local Qwen models."""
    load_dotenv()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key or gemini_key in ("your_gemini_api_key_here", "paste_your_gemini_api_key_here"):
        pytest.skip("GEMINI_API_KEY not configured in .env")

    mgr = StoryLibraryManager()
    extractors: list[OpenAICompatibleExtractor] = [OpenAICompatibleExtractor.from_gemini()]

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key and deepseek_key not in ("your_deepseek_api_key_here", "paste_your_deepseek_api_key_here"):
        extractors.append(OpenAICompatibleExtractor.from_deepseek())

    # Include cached local Qwen baselines if their benchmarks exist on disk
    story_dir = mgr.story_subfolder("obsidian_signet_3k")
    for qwen_model in ("qwen/qwen3.6-27b", "qwen/qwen3.5-9b"):
        safe_name = qwen_model.replace("/", "_")
        if (story_dir / "benchmarks" / f"benchmark_{safe_name}_none.json").is_file():
            extractors.append(OpenAICompatibleExtractor.from_lmstudio(model=qwen_model))

    for mode in ("full_hints", "none"):
        report = mgr.compare_models(
            "obsidian_signet_3k",
            extractors=extractors,
            guidance_mode=mode,
        )
        print(f"\n{report.to_markdown_table()}")
        gemini_res = next(r for r in report.results if r.extraction_model == OpenAICompatibleExtractor.from_gemini().model)
        assert len(gemini_res.story.chunks) == 6
        if mode == "full_hints":
            assert gemini_res.metrics.overall_quality_score >= 0.95
            assert gemini_res.pass_rate >= 0.85
        else:
            assert gemini_res.metrics.overall_quality_score > 0.5


def test_non_llm_rule_based_benchmark_across_3k_stories(tmp_path: Path) -> None:
    """Benchmarks the zero-LLM RuleBasedNarrativeExtractor across all three 3K-token stories."""
    source_mgr = StoryLibraryManager()
    mgr = StoryLibraryManager(root_dir=tmp_path)

    for slug in ("obsidian_signet_3k", "chronos_station_3k", "crimson_masquerade_3k"):
        orig_story = source_mgr.get_or_create_story(slug)
        sub = tmp_path / slug
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "story.json").write_text(
            (orig_story.story_dir / "story.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        res = mgr.run_story_extraction_e2e(
            slug,
            base_extractor=RuleBasedNarrativeExtractor(),
            guidance_mode="none",
        )
        assert res.extraction_historical_usage.total_tokens == 0
        assert res.metrics.flashback_guard_passed is True
        assert res.metrics.hallucinated_id_count == 0
        assert res.metrics.overall_quality_score > 0.5






