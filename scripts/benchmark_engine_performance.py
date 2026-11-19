from __future__ import annotations

import gc
import json
from pathlib import Path
import statistics
import tempfile
import time
import tracemalloc
from typing import Any

from canonarc import (
    CanonArcEngine,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    InteractiveFictionPreset,
    StepDelta,
)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _iqr(sorted_vals: list[float]) -> float:
    if len(sorted_vals) < 4:
        return 0.0
    return _percentile(sorted_vals, 75.0) - _percentile(sorted_vals, 25.0)


def run_engine_performance_benchmarks(
    *,
    step_scales: tuple[int, ...] = (100, 500, 1500),
    n_queries: int = 60,
    n_warmup: int = 5,
    n_op_trials: int = 7,
) -> dict[str, Any]:
    """Benchmarks CanonArcEngine algorithmic performance, memory/storage scaling, multi-step rollback,
    retrieval latency distributions (P50, IQR, P95, P99), episode consolidation, and adversarial deceived-POV
    vector isolation.
    """
    timeline_results: list[dict[str, Any]] = []
    retrieval_results: list[dict[str, Any]] = []
    storage_results: list[dict[str, Any]] = []

    for n_steps in step_scales:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / f"bench_{n_steps}.sqlite"
            snap_path = Path(tmp_dir) / f"snap_{n_steps}.sqlite"

            tracemalloc.start()
            with CanonArcEngine(
                db_path,
                preset=InteractiveFictionPreset(include_background_edges=True),
            ) as engine:
                n_chars = min(120, max(12, n_steps // 10))
                n_rooms = 6
                for r_idx in range(n_rooms):
                    engine.register_entity(
                        EntityRecord(
                            entity_id=f"loc_room_{r_idx}",
                            kind="location",
                            canonical_name=f"Chamber {r_idx}",
                        ),
                        step=0,
                    )
                for c_idx in range(n_chars):
                    engine.register_entity(
                        EntityRecord(
                            entity_id=f"char_{c_idx}",
                            kind="character",
                            canonical_name=f"Character_{c_idx}",
                            state={"mood": "calm", "location_id": f"loc_room_{c_idx % n_rooms}"},
                        ),
                        step=0,
                    )

                # Populate n_steps with state mutations, edges, and episodes
                gc_was_enabled = gc.isenabled()
                gc.collect()
                t_ingest_0 = time.perf_counter()
                for step in range(1, n_steps + 1):
                    actor = f"char_{step % n_chars}"
                    target = f"char_{(step + 1) % n_chars}"
                    room = f"loc_room_{step % n_rooms}"
                    sal = 9.0 if (step % 25 == 0) else 2.5
                    engine.commit_step(
                        step=step,
                        delta=StepDelta(
                            entities=[
                                EntityDelta(
                                    entity_id=actor,
                                    state={"mood": "focused" if step % 2 == 0 else "watchful", "location_id": room},
                                )
                            ],
                            edges=[
                                EdgeDelta(
                                    source_id=actor,
                                    target_id=target,
                                    relation_type="stance",
                                    state={"register": "Vi", "tone": "guarded", "trust": 0.2},
                                )
                            ],
                            episodes=[
                                EpisodeRecord(
                                    episode_id=f"ep_step_{step}",
                                    predicate="interaction",
                                    summary=f"Character_{step % n_chars} discussed garrison logistics with Character_{(step + 1) % n_chars} in Chamber {step % n_rooms}.",
                                    participants={actor: "actor", target: "target"},
                                    witnesses=[actor, target],
                                    location_id=room,
                                    salience=sal,
                                )
                            ],
                        ),
                    )
                ingest_ms = (time.perf_counter() - t_ingest_0) * 1000.0
                _, peak_ingest_bytes = tracemalloc.get_traced_memory()

                # Flush WAL so on-disk SQLite size is exact
                if hasattr(engine.store, "conn"):
                    engine.store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                pre_consolidate_db_kb = db_path.stat().st_size / 1024.0 if db_path.exists() else 0.0

                # 1. Warmup + N timed retrieve_context() queries before consolidation
                for w_idx in range(n_warmup):
                    c_w = f"char_{w_idx % n_chars}"
                    engine.retrieve_context(
                        step=n_steps,
                        text=f"Character_{w_idx % n_chars} inspected garrison logistics in Chamber {w_idx % n_rooms}.",
                        pov_entity_ids=[c_w],
                        location_id=f"loc_room_{w_idx % n_rooms}",
                        top_k_episodes=8,
                        inclusive=True,
                    )

                latencies_ms: list[float] = []
                for q_idx in range(n_queries):
                    c_a = f"char_{q_idx % n_chars}"
                    t0 = time.perf_counter()
                    engine.retrieve_context(
                        step=n_steps,
                        text=f"Character_{q_idx % n_chars} inspected garrison logistics in Chamber {q_idx % n_rooms}.",
                        pov_entity_ids=[c_a],
                        location_id=f"loc_room_{q_idx % n_rooms}",
                        top_k_episodes=8,
                        inclusive=True,
                    )
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                latencies_ms.sort()

                # 2. Multi-trial timing for branch(), diff_timelines(), 1-step branch undo, and deep multi-step rollback
                fork_step = max(1, n_steps // 2)
                branch_samples: list[float] = []
                diff_samples: list[float] = []
                undo_1step_samples: list[float] = []

                for trial in range(n_op_trials):
                    b_name = f"bench_alt_{trial}"
                    gc.disable()
                    try:
                        t_b0 = time.perf_counter()
                        branch_sess = engine.session("main").branch(at_step=fork_step, name=b_name)
                        branch_samples.append((time.perf_counter() - t_b0) * 1000.0)

                        branch_sess.commit_step(
                            step=fork_step + 1,
                            delta=StepDelta(
                                entities=[EntityDelta(entity_id="char_0", state={"mood": "furious"})]
                            ),
                        )

                        t_d0 = time.perf_counter()
                        diff = engine.diff_timelines("main", b_name, at_step=n_steps)
                        diff_samples.append((time.perf_counter() - t_d0) * 1000.0)
                        assert diff["changed"] or diff["only_in_a"]

                        t_r0 = time.perf_counter()
                        branch_sess.rollback_to(fork_step)
                        undo_1step_samples.append((time.perf_counter() - t_r0) * 1000.0)
                    finally:
                        if gc_was_enabled:
                            gc.enable()
                    engine.delete_timeline(b_name)

                branch_samples.sort()
                diff_samples.sort()
                undo_1step_samples.sort()

                # Export snapshot at n_steps
                t_s0 = time.perf_counter()
                engine.export_snapshot(snap_path, timeline_id="main", at_step=n_steps)
                snapshot_ms = (time.perf_counter() - t_s0) * 1000.0
                snapshot_kb = snap_path.stat().st_size / 1024.0 if snap_path.exists() else 0.0

                # Deep multi-step rollback benchmark: commit k_rollback_steps on a dedicated branch and roll them all back
                k_rollback_steps = max(10, n_steps // 2)
                deep_branch = engine.session("main").branch(at_step=fork_step, name="deep_rollback_branch")
                for r_step in range(fork_step + 1, fork_step + k_rollback_steps + 1):
                    actor = f"char_{r_step % n_chars}"
                    deep_branch.commit_step(
                        step=r_step,
                        delta=StepDelta(
                            entities=[EntityDelta(entity_id=actor, state={"mood": "panicked"})],
                            episodes=[
                                EpisodeRecord(
                                    episode_id=f"ep_deep_{r_step}",
                                    predicate="alert",
                                    summary=f"Emergency drill at step {r_step}.",
                                    participants={actor: "actor"},
                                    witnesses=[actor],
                                    salience=3.0,
                                )
                            ],
                        ),
                    )
                t_deep0 = time.perf_counter()
                deleted_rows = deep_branch.rollback_to(fork_step)
                deep_rollback_ms = (time.perf_counter() - t_deep0) * 1000.0
                assert deleted_rows >= k_rollback_steps
                engine.delete_timeline("deep_rollback_branch")

                # 3. Episode consolidation speed & post-consolidation retrieval speedup
                t_c0 = time.perf_counter()
                engine.consolidate_episodes(
                    step=n_steps + 1,
                    before_step=n_steps - 20,
                    max_salience=4.0,
                    summary="Consolidated historical low-salience logistics interactions.",
                )
                consolidate_ms = (time.perf_counter() - t_c0) * 1000.0

                if hasattr(engine.store, "conn"):
                    engine.store.conn.execute("VACUUM;")
                    engine.store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                post_consolidate_db_kb = db_path.stat().st_size / 1024.0 if db_path.exists() else 0.0

                post_latencies_ms: list[float] = []
                for q_idx in range(n_queries):
                    c_a = f"char_{q_idx % n_chars}"
                    t0 = time.perf_counter()
                    engine.retrieve_context(
                        step=n_steps + 1,
                        text=f"Character_{q_idx % n_chars} inspected garrison logistics.",
                        pov_entity_ids=[c_a],
                        location_id=f"loc_room_{q_idx % n_rooms}",
                        top_k_episodes=8,
                        inclusive=True,
                    )
                    post_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                post_latencies_ms.sort()

                _, peak_total_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                timeline_results.append(
                    {
                        "steps": n_steps,
                        "characters": n_chars,
                        "ingest_per_step_ms": round(ingest_ms / n_steps, 3),
                        "branch_ms": round(statistics.median(branch_samples), 3),
                        "rollback_ms": round(statistics.median(undo_1step_samples), 3),
                        "deep_rollback_steps": k_rollback_steps,
                        "deep_rollback_ms": round(deep_rollback_ms, 3),
                        "diff_timelines_ms": round(statistics.median(diff_samples), 3),
                        "export_snapshot_ms": round(snapshot_ms, 3),
                    }
                )
                retrieval_results.append(
                    {
                        "steps": n_steps,
                        "characters": n_chars,
                        "n_queries": n_queries,
                        "p50_retrieve_ms": round(statistics.median(latencies_ms), 3),
                        "iqr_retrieve_ms": round(_iqr(latencies_ms), 3),
                        "p95_retrieve_ms": round(_percentile(latencies_ms, 95.0), 3),
                        "p99_retrieve_ms": round(_percentile(latencies_ms, 99.0), 3),
                        "consolidate_ms": round(consolidate_ms, 3),
                        "post_consolidation_p50_ms": round(statistics.median(post_latencies_ms), 3),
                        "post_consolidation_p95_ms": round(_percentile(post_latencies_ms, 95.0), 3),
                    }
                )
                storage_results.append(
                    {
                        "steps": n_steps,
                        "characters": n_chars,
                        "sqlite_file_kb": round(pre_consolidate_db_kb, 1),
                        "post_consolidate_sqlite_kb": round(post_consolidate_db_kb, 1),
                        "snapshot_file_kb": round(snapshot_kb, 1),
                        "peak_heap_kb": round(max(peak_ingest_bytes, peak_total_bytes) / 1024.0, 1),
                    }
                )

    # 4. Adversarial Deceived-POV Vector Isolation & Side-Channel Leak Benchmark
    with CanonArcEngine(":memory:") as pov_engine:
        for cid, cname in (("char_elena", "Elena"), ("char_marcus", "Marcus"), ("char_julian", "Julian")):
            pov_engine.register_entity(
                EntityRecord(entity_id=cid, kind="character", canonical_name=cname),
                step=0,
            )

        # Include adversarial distractors that overlap partially with both mountain/raider and wine-cellar vocabulary
        adversarial_distractors = [
            EpisodeRecord(
                episode_id="ep_distractor_cellar_inventory",
                predicate="inspection",
                summary="The cellar steward counted oak casks in the subterranean wine cellar storage.",
                participants={"char_marcus": "actor"},
                witnesses=["char_marcus", "char_elena"],
                salience=3.5,
            ),
            EpisodeRecord(
                episode_id="ep_distractor_mountain_weather",
                predicate="report",
                summary="Scouts reported heavy snowfalls blocking the northern mountain pass trails.",
                participants={"char_marcus": "actor"},
                witnesses=["char_marcus", "char_elena"],
                salience=3.5,
            ),
            EpisodeRecord(
                episode_id="ep_distractor_signet_protocol",
                predicate="protocol",
                summary="Chancery clerks reviewed wax seal protocols for royal charters in the hall.",
                participants={"char_elena": "actor"},
                witnesses=["char_marcus", "char_elena"],
                salience=3.5,
            ),
            *[
                EpisodeRecord(
                    episode_id=f"ep_distractor_{i}",
                    predicate="routine",
                    summary=f"Routine patrol report number {i} along the eastern battlements and courtyard stables.",
                    participants={"char_marcus": "actor"},
                    witnesses=["char_marcus", "char_elena", "char_julian"],
                    salience=3.0,
                )
                for i in range(22)
            ],
        ]

        pov_engine.commit_step(
            step=1,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_secret_cellar_theft",
                        predicate="theft",
                        summary="Julian secretly hid the dragon signet inside the subterranean wine cellar.",
                        participants={"char_julian": "actor", "char_elena": "witness"},
                        witnesses=["char_elena", "char_julian"],
                        salience=9.0,
                    ),
                    *adversarial_distractors,
                ],
                inform=[
                    InformDelta(
                        episode_id="ep_secret_cellar_theft",
                        entity_id="char_marcus",
                        epistemic="deceived",
                        believed_summary="Northern mountain raiders ambushed the caravan on the high glacier pass.",
                    )
                ],
            ),
        )

        # Query 1: Deceived character queries by believed keywords ("caravan ambushed on high glacier pass")
        ctx_believed = pov_engine.retrieve_context(
            step=1,
            text="Northern mountain raiders ambushed the caravan on the high glacier pass.",
            pov_entity_ids=["char_marcus"],
            top_k_episodes=5,
            inclusive=True,
        )
        believed_top1 = ctx_believed.episodes[0].episode_id if ctx_believed.episodes else ""
        believed_summary_isolated = (
            bool(ctx_believed.episodes)
            and "wine cellar" not in ctx_believed.episodes[0].summary.lower()
            and "dragon signet" not in ctx_believed.episodes[0].summary.lower()
            and "glacier pass" in ctx_believed.episodes[0].summary.lower()
        )

        # Query 2: Witness queries by true keywords ("subterranean wine cellar dragon signet")
        ctx_true = pov_engine.retrieve_context(
            step=1,
            text="Julian hid the dragon signet in the subterranean wine cellar.",
            pov_entity_ids=["char_elena"],
            top_k_episodes=5,
            inclusive=True,
        )
        true_top1 = ctx_true.episodes[0].episode_id if ctx_true.episodes else ""

        # Query 3 (Adversarial Side-Channel Probe): Deceived character queries by GROUND-TRUTH secret keywords!
        # Because char_marcus is deceived (believing it was a glacier pass ambush), querying "subterranean wine cellar"
        # must rank the legitimate wine-cellar inspection episode ('ep_distractor_cellar_inventory') at #1,
        # NOT 'ep_secret_cellar_theft', and must never expose ground-truth words in any returned episode summary.
        ctx_adversarial = pov_engine.retrieve_context(
            step=1,
            text="Subterranean wine cellar oak casks and dragon signet.",
            pov_entity_ids=["char_marcus"],
            top_k_episodes=5,
            inclusive=True,
        )
        adv_top1 = ctx_adversarial.episodes[0].episode_id if ctx_adversarial.episodes else ""
        adv_summaries_clean = all(
            "dragon signet" not in ep.summary.lower() for ep in ctx_adversarial.episodes
        )
        side_channel_blocked = (adv_top1 == "ep_distractor_cellar_inventory") and adv_summaries_clean

    return {
        "timeline_operations": timeline_results,
        "retrieval_scaling": retrieval_results,
        "storage_and_memory_scaling": storage_results,
        "deceived_pov_vector_isolation": {
            "total_candidate_episodes": 26,
            "deceived_query_top1_episode": believed_top1,
            "witness_query_top1_episode": true_top1,
            "adversarial_secret_query_top1_episode": adv_top1,
            "believed_summary_vector_recall_at_1": 1.0 if believed_top1 == "ep_secret_cellar_theft" else 0.0,
            "witness_true_vector_recall_at_1": 1.0 if true_top1 == "ep_secret_cellar_theft" else 0.0,
            "zero_ground_truth_vector_leak": bool(believed_summary_isolated and side_channel_blocked),
            "adversarial_side_channel_blocked": side_channel_blocked,
        },
    }


def format_engine_performance_markdown(report: dict[str, Any]) -> str:
    lines = [
        "## Engine Algorithmic & Scaling Micro-Benchmarks",
        "",
        "### 1. Timeline Time-Travel, Multi-Step Rollback & Snapshot Latency",
        "",
        "| Timeline Steps | Active Characters | Commit / Step (ms) | Zero-Copy `branch()` (ms) | 1-Step Undo `rollback_to()` (ms) | Deep Rollback (50% History) (ms) | `diff_timelines()` (ms) | `export_snapshot()` (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["timeline_operations"]:
        deep_steps = row.get("deep_rollback_steps", row["steps"] // 2)
        deep_ms = row.get("deep_rollback_ms", row["rollback_ms"])
        lines.append(
            f"| {row['steps']:,} | {row['characters']} | {row['ingest_per_step_ms']:.3f} ms | "
            f"**{row['branch_ms']:.3f} ms** | **{row['rollback_ms']:.3f} ms** | "
            f"**{deep_ms:.3f} ms** (-{deep_steps} steps) | "
            f"{row['diff_timelines_ms']:.3f} ms | {row['export_snapshot_ms']:.3f} ms |"
        )
    lines.extend(
        [
            "",
            "### 2. Retrieval Scaling ($N=60$ Queries, 5 Warmup) & Episode Consolidation Speedup",
            "",
            "| Timeline Steps | Active Characters | `retrieve_context()` P50 (ms) | IQR (ms) | P95 (ms) | P99 (ms) | `consolidate_episodes()` (ms) | Post-Consolidation P50 (ms) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["retrieval_scaling"]:
        iqr_val = row.get("iqr_retrieve_ms", 0.0)
        p99_val = row.get("p99_retrieve_ms", row["p95_retrieve_ms"])
        lines.append(
            f"| {row['steps']:,} | {row['characters']} | **{row['p50_retrieve_ms']:.3f} ms** | "
            f"{iqr_val:.3f} ms | {row['p95_retrieve_ms']:.3f} ms | {p99_val:.3f} ms | "
            f"{row['consolidate_ms']:.3f} ms | **{row['post_consolidation_p50_ms']:.3f} ms** |"
        )
    if "storage_and_memory_scaling" in report:
        lines.extend(
            [
                "",
                "### 3. SQLite Save File & Memory Footprint Scaling",
                "",
                "| Timeline Steps | Active Characters | SQLite Save Size (KB) | Post-Consolidation SQLite (KB) | Standalone Snapshot Size (KB) | Peak Traced Heap (KB) |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["storage_and_memory_scaling"]:
            lines.append(
                f"| {row['steps']:,} | {row['characters']} | {row['sqlite_file_kb']:,.1f} KB | "
                f"**{row['post_consolidate_sqlite_kb']:,.1f} KB** | **{row['snapshot_file_kb']:,.1f} KB** | "
                f"{row['peak_heap_kb']:,.1f} KB |"
            )

    pov = report["deceived_pov_vector_isolation"]
    lines.extend(
        [
            "",
            "### 4. Adversarial Deceived-POV Vector Isolation Verification (26 Candidate Episodes)",
            "",
            f"- **Deceived POV Recall@1 (`believed_summary` query)**: `{pov['believed_summary_vector_recall_at_1']:.0%}`",
            f"- **Direct Witness Recall@1 (ground-truth query)**: `{pov['witness_true_vector_recall_at_1']:.0%}`",
            f"- **Adversarial Secret Query Side-Channel Blocked**: `{'PASS' if pov.get('adversarial_side_channel_blocked', True) else 'FAIL'}`",
            f"- **Zero Ground-Truth Vector Leakage**: `{'PASS' if pov['zero_ground_truth_vector_leak'] else 'FAIL'}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run_engine_performance_benchmarks()
    print(format_engine_performance_markdown(rep))
    out_path = Path(__file__).resolve().parent.parent / "tests" / "synthetic_stories" / "engine_performance_benchmark.json"
    out_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
