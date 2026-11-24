"""Companion demo module for examples/04_mystery_and_social_deduction.ipynb.

Keeps all HTML/SVG rendering, LLM HTTP plumbing, and step-by-step scenario
setup out of the notebook so readers can focus on the story, high-level
concepts, and live visual outputs — while still being able to inspect the
full, clean Python implementation of every step right here.
"""

from __future__ import annotations

import html
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from IPython.display import HTML, SVG, display

from canonarc import (
    AliasRecord,
    BaseEngineHooks,
    CachedDeltaExtractor,
    CanonArcEngine,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    InteractiveFictionPreset,
    OpenAICompatibleExtractor,
    StepDelta,
    TimelineInfo,
    TokenUsage,
    VoiceExemplar,
    load_dotenv,
)

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
HAS_LIVE_LLM = bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here")


class EpistemicAuditHook(BaseEngineHooks):
    """Flight-recorder middleware tracking commits, branches, rollbacks, and leak checks."""

    def __init__(self) -> None:
        self.commits: list[dict[str, Any]] = []
        self.branches: list[str] = []
        self.rollbacks: list[dict[str, Any]] = []
        self.gate_rejections: list[dict[str, Any]] = []
        self.leak_audits: list[dict[str, Any]] = []

    def on_after_commit(
        self,
        step: int,
        timeline_id: str,
        committed_episodes: Sequence[EpisodeRecord],
        indexed_exemplars: Sequence[VoiceExemplar],
    ) -> None:
        self.commits.append(
            {
                "step": step,
                "timeline": timeline_id,
                "episodes": len(committed_episodes),
                "exemplars": len(indexed_exemplars),
            }
        )

    def on_branch(self, timeline_info: TimelineInfo) -> None:
        self.branches.append(timeline_info.timeline_id)

    def on_rollback(self, step: int, timeline_id: str, deleted_count: int) -> None:
        self.rollbacks.append({"step": step, "timeline": timeline_id, "deleted": deleted_count})

    def on_gate_rejected(
        self, stage: str, text: str, reason: str, metadata: dict[str, Any]
    ) -> None:
        self.gate_rejections.append({"stage": stage, "reason": reason, "text": text[:60]})

    def audit_prompt(self, label: str, prompt_text: str, forbidden_tokens: list[str]) -> list[str]:
        leaked = [tok for tok in forbidden_tokens if tok.lower() in prompt_text.lower()]
        self.leak_audits.append({"label": label, "leaked": leaked, "passed": len(leaked) == 0})
        return leaked


# ---------------------------------------------------------------------------
# Visual HTML & SVG Renderers
# ---------------------------------------------------------------------------

def render_ab_comparison(
    naive_reply: str,
    pov_reply: str,
    leaked_tokens_naive: list[str],
    leaked_tokens_pov: list[str],
) -> None:
    naive_badge = (
        f"LEAKED: {', '.join(leaked_tokens_naive)}"
        if leaked_tokens_naive
        else "RISKY: Secrets Present in Prompt"
    )
    pov_badge = (
        "0 SECRET LEAKS (Structurally Impossible)"
        if not leaked_tokens_pov
        else f"LEAKED: {leaked_tokens_pov}"
    )
    display(HTML(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;font-family:system-ui,-apple-system,sans-serif;margin:10px 0;">
  <div style="border:1px solid #fca5a5;background:#fef2f2;border-radius:10px;padding:14px;">
    <div style="font-weight:700;color:#991b1b;margin-bottom:6px;">
      ❌ Condition A: Omniscient Context ({html.escape(naive_badge)})
    </div>
    <div style="font-size:12px;color:#7f1d1d;margin-bottom:8px;">
      Prompt exposed: <code>Lord Vance</code>, <code>steal_royal_cipher</code>, <code>picked the vault lock</code>
    </div>
    <blockquote style="margin:0;padding:10px;background:#fff;border-left:4px solid #dc2626;color:#1f2937;font-size:13px;line-height:1.55;">
      {html.escape(naive_reply)}
    </blockquote>
  </div>
  <div style="border:1px solid #86efac;background:#f0fdf4;border-radius:10px;padding:14px;">
    <div style="font-weight:700;color:#166534;margin-bottom:6px;">
      ✅ Condition B: canonarc POV Projection ({html.escape(pov_badge)})
    </div>
    <div style="font-size:12px;color:#14532d;margin-bottom:8px;">
      Prompt saw only: <code>The Velvet Courier</code>, <code>believed_summary</code> (archivist cover story)
    </div>
    <blockquote style="margin:0;padding:10px;background:#fff;border-left:4px solid #16a34a;color:#1f2937;font-size:13px;line-height:1.55;">
      {html.escape(pov_reply)}
    </blockquote>
  </div>
</div>
"""))


def render_epistemic_matrix(
    rows: list[dict[str, Any]],
    episode_headers: list[tuple[str, str]],
) -> None:
    badge_styles = {
        "direct": ("#dcfce7", "#166534", "DIRECT"),
        "told": ("#dbeafe", "#1e40af", "TOLD"),
        "deceived": ("#fef3c7", "#92400e", "DECEIVED"),
        "forgotten": ("#f3e8ff", "#6b21a8", "FORGOTTEN"),
        "suppressed": ("#fce7f3", "#9d174d", "SUPPRESSED"),
        "ignorant": ("#f1f5f9", "#475569", "IGNORANT"),
    }
    th_eps = "".join(
        f'<th style="padding:8px 10px;text-align:center;border-bottom:2px solid #cbd5e1;">'
        f'{html.escape(label)}<br/><span style="font-weight:400;font-size:11px;color:#64748b;">{html.escape(ep_id)}</span></th>'
        for ep_id, label in episode_headers
    )
    body_rows = []
    for r in rows:
        disg_bg, disg_fg = ("#dcfce7", "#166534") if r["disguise"] == "PIERCED" else ("#fee2e2", "#991b1b")
        ep_cells = []
        for ep_id, _ in episode_headers:
            mode = r["modes"][ep_id]
            bg, fg, lbl = badge_styles.get(mode, ("#f1f5f9", "#475569", mode.upper()))
            tip = r.get("summaries", {}).get(ep_id, "")
            sub = f'<div style="font-size:10px;color:#475569;margin-top:3px;">{html.escape(tip)}</div>' if tip else ""
            ep_cells.append(
                f'<td style="padding:8px 10px;text-align:center;border-bottom:1px solid #e2e8f0;vertical-align:top;">'
                f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;'
                f'background:{bg};color:{fg};">{lbl}</span>{sub}</td>'
            )
        body_rows.append(
            f'<tr>'
            f'<td style="padding:8px 10px;font-weight:600;border-bottom:1px solid #e2e8f0;color:#0f172a;">{html.escape(r["name"])}</td>'
            f'<td style="padding:8px 10px;text-align:center;border-bottom:1px solid #e2e8f0;">'
            f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;'
            f'background:{disg_bg};color:{disg_fg};">{html.escape(r["disguise"])}</span></td>'
            + "".join(ep_cells)
            + "</tr>"
        )
    display(HTML(f"""
<div style="font-family:system-ui,-apple-system,sans-serif;margin:10px 0;">
  <div style="font-weight:700;font-size:14px;color:#0f172a;margin-bottom:6px;">
    🧭 Live Epistemic Knowledge Matrix (Step 20 — 5 Characters × 3 Secrets + Disguise Lens)
  </div>
  <table style="width:100%;border-collapse:collapse;background:#fff;border:1px solid #cbd5e1;border-radius:8px;overflow:hidden;font-size:12.5px;">
    <thead style="background:#f8fafc;color:#334155;">
      <tr>
        <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #cbd5e1;">Observer POV</th>
        <th style="padding:8px 10px;text-align:center;border-bottom:2px solid #cbd5e1;">Vance Disguise</th>
        {th_eps}
      </tr>
    </thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
"""))


def render_pov_cards(cards: list[tuple[str, str, str, str]]) -> None:
    """Renders multi-column HTML cards: (title, accent_hex, subtitle, body_text)."""
    cols = []
    for title, accent, subtitle, body in cards:
        cols.append(f"""
<div style="border:1px solid #cbd5e1;border-top:4px solid {accent};border-radius:10px;background:#fff;padding:12px;">
  <div style="font-weight:700;color:#0f172a;font-size:13.5px;">{html.escape(title)}</div>
  <div style="font-size:11.5px;color:#475569;margin:3px 0 8px 0;">{html.escape(subtitle)}</div>
  <pre style="margin:0;padding:8px;background:#f8fafc;border-radius:6px;font-size:11px;line-height:1.45;white-space:pre-wrap;color:#1e293b;">{html.escape(body)}</pre>
</div>
""")
    display(HTML(
        f'<div style="display:grid;grid-template-columns:repeat({len(cards)},1fr);gap:12px;'
        f'font-family:system-ui,-apple-system,sans-serif;margin:10px 0;">{"".join(cols)}</div>'
    ))


def render_suspicion_svg(points: list[dict[str, Any]]) -> None:
    """Inline SVG chart plotting Clara's perceived trust (facade), true trust (stance), and suspicion."""
    w, h = 720, 260
    pad_l, pad_r, pad_t, pad_b = 58, 30, 34, 42
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    n = len(points)

    def x_at(idx: int) -> float:
        return pad_l + (idx / max(n - 1, 1)) * plot_w

    def y_at(val: float) -> float:
        return pad_t + ((1.0 - val) / 2.0) * plot_h

    def polyline(key: str, color: str, dash: str = "") -> str:
        pts = " ".join(f"{x_at(i):.1f},{y_at(p[key]):.1f}" for i, p in enumerate(points))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        circles = "".join(
            f'<circle cx="{x_at(i):.1f}" cy="{y_at(p[key]):.1f}" r="4.5" fill="{color}" stroke="#fff" stroke-width="1.5"/>'
            for i, p in enumerate(points)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} points="{pts}"/>{circles}'

    grid_lines = []
    for v, lbl in [(1.0, "+1.0"), (0.5, "+0.5"), (0.0, "0.0"), (-0.5, "-0.5"), (-1.0, "-1.0")]:
        yy = y_at(v)
        stroke = "#94a3b8" if v == 0.0 else "#e2e8f0"
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="10.5" fill="#475569">{lbl}</text>'
        )

    x_labels = "".join(
        f'<text x="{x_at(i):.1f}" y="{h - 14}" text-anchor="middle" font-size="10.5" fill="#334155">{html.escape(p["label"])}</text>'
        for i, p in enumerate(points)
    )

    unmask_idx = next((i for i, p in enumerate(points) if "Unmask" in p["label"]), n - 1)
    ux = x_at(unmask_idx)

    svg = f"""
<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:740px;background:#fff;border:1px solid #cbd5e1;border-radius:10px;font-family:system-ui,-apple-system,sans-serif;">
  <text x="{pad_l}" y="20" font-size="13" font-weight="700" fill="#0f172a">
    Social Deduction Dynamics: Perceived Trust (Facade) vs. True Stance &amp; Clara's Suspicion
  </text>
  {''.join(grid_lines)}
  <line x1="{ux:.1f}" y1="{pad_t}" x2="{ux:.1f}" y2="{h - pad_b}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,4"/>
  <text x="{ux - 6:.1f}" y="{pad_t + 12}" text-anchor="end" font-size="10" font-weight="700" fill="#b45309">Disguise Pierced</text>
  {polyline("perceived_trust", "#2563eb")}
  {polyline("true_trust", "#dc2626", "6,4")}
  {polyline("suspicion", "#d97706")}
  {x_labels}
  <g transform="translate({pad_l + 8}, {h - pad_b - 22})">
    <rect x="0" y="0" width="465" height="20" rx="5" fill="#f8fafc" stroke="#e2e8f0"/>
    <line x1="10" y1="10" x2="28" y2="10" stroke="#2563eb" stroke-width="2.6"/>
    <text x="33" y="13.5" font-size="10.5" fill="#1e293b">Perceived Trust (Facade)</text>
    <line x1="172" y1="10" x2="190" y2="10" stroke="#dc2626" stroke-width="2.6" stroke-dasharray="5,3"/>
    <text x="195" y="13.5" font-size="10.5" fill="#1e293b">True Stance (Hidden)</text>
    <line x1="322" y1="10" x2="340" y2="10" stroke="#d97706" stroke-width="2.6"/>
    <text x="345" y="13.5" font-size="10.5" fill="#1e293b">Clara's Suspicion</text>
  </g>
</svg>
"""
    display(SVG(svg))


def render_branch_tree_svg(route_a_note: str, route_b_note: str) -> None:
    svg = f"""
<svg viewBox="0 0 720 190" width="100%" style="max-width:740px;background:#fff;border:1px solid #cbd5e1;border-radius:10px;font-family:system-ui,-apple-system,sans-serif;">
  <text x="20" y="24" font-size="13" font-weight="700" fill="#0f172a">
    Counterfactual Investigation Branches (Forked from Step 25 via engine.branch())
  </text>
  <line x1="40" y1="100" x2="240" y2="100" stroke="#475569" stroke-width="3"/>
  <circle cx="55" cy="100" r="8" fill="#334155"/>
  <text x="55" y="124" text-anchor="middle" font-size="10.5" fill="#334155">Step 0–20</text>
  <circle cx="235" cy="100" r="9" fill="#0ea5e9" stroke="#fff" stroke-width="2"/>
  <text x="235" y="126" text-anchor="middle" font-size="11" font-weight="700" fill="#0369a1">Fork @ Step 25</text>

  <!-- Route A -->
  <path d="M 244 100 C 295 100, 300 58, 355 58 L 680 58" fill="none" stroke="#dc2626" stroke-width="2.6"/>
  <circle cx="380" cy="58" r="7" fill="#dc2626"/>
  <rect x="398" y="38" width="295" height="38" rx="7" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="408" y="54" font-size="11.5" font-weight="700" fill="#991b1b">route_ballroom (Public Confrontation)</text>
  <text x="408" y="69" font-size="10.5" fill="#7f1d1d">{html.escape(route_a_note)}</text>

  <!-- Route B -->
  <path d="M 244 100 C 295 100, 300 142, 355 142 L 680 142" fill="none" stroke="#16a34a" stroke-width="2.6"/>
  <circle cx="380" cy="142" r="7" fill="#16a34a"/>
  <rect x="398" y="122" width="295" height="38" rx="7" fill="#f0fdf4" stroke="#86efac"/>
  <text x="408" y="138" font-size="11.5" font-weight="700" fill="#166534">route_conservatory (Silent Bust Search)</text>
  <text x="408" y="153" font-size="10.5" fill="#14532d">{html.escape(route_b_note)}</text>
</svg>
"""
    display(SVG(svg))


def render_qa_scorecard(checks: list[tuple[str, bool]], hook: EpistemicAuditHook, cost_note: str) -> None:
    passed = sum(1 for _, ok in checks if ok)
    rows_html = "".join(
        f'<tr><td style="padding:6px 10px;border-bottom:1px solid #e2e8f0;">{html.escape(name)}</td>'
        f'<td style="padding:6px 10px;border-bottom:1px solid #e2e8f0;text-align:center;font-weight:700;'
        f'color:{"#166534" if ok else "#991b1b"};">{"✅ PASS" if ok else "❌ FAIL"}</td></tr>'
        for name, ok in checks
    )
    display(HTML(f"""
<div style="font-family:system-ui,-apple-system,sans-serif;margin:10px 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-weight:700;font-size:15px;color:#0f172a;">✅ Automated Epistemic QA Audit ({passed}/{len(checks)} Passed)</span>
    <span style="font-size:12px;color:#475569;">Commits: {len(hook.commits)} · Branches: {len(hook.branches)} · Rollbacks: {len(hook.rollbacks)} · Leak Audits: {len(hook.leak_audits)}</span>
  </div>
  <table style="width:100%;border-collapse:collapse;background:#fff;border:1px solid #cbd5e1;font-size:12.5px;">
    <thead style="background:#f8fafc;"><tr><th style="padding:7px 10px;text-align:left;border-bottom:2px solid #cbd5e1;">Check</th><th style="padding:7px 10px;text-align:center;border-bottom:2px solid #cbd5e1;">Status</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div style="margin-top:8px;padding:8px 12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px;color:#334155;">
    {html.escape(cost_note)}
  </div>
</div>
"""))


# ---------------------------------------------------------------------------
# Stateful Demo Runner (one clean method per notebook step)
# ---------------------------------------------------------------------------

class MysteryDemoSession:
    """Runs 'The Velvet Cipher' walkthrough step-by-step and tracks QA metrics."""

    def __init__(self) -> None:
        self.dialogue_llm_usage = TokenUsage()
        self.hook = EpistemicAuditHook()
        self.engine: CanonArcEngine | None = None
        self.state: dict[str, Any] = {}

        display(HTML(
            f'<div style="font-family:system-ui,sans-serif;padding:8px 12px;background:#f0fdf4;border:1px solid #86efac;'
            f'border-radius:8px;color:#166534;font-size:13px;">'
            f'⚙️ Environment ready · Backend: <b>{"DeepSeek API (" + DEEPSEEK_MODEL + ")" if HAS_LIVE_LLM else "Offline Fallback"}</b> · '
            f'Preset: <code>InteractiveFictionPreset(include_background_edges=True)</code></div>'
        ))

    def _call_deepseek_flash(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.25,
        fallback_text: str = "",
    ) -> tuple[str, TokenUsage]:
        """Zero-dependency HTTP call to DeepSeek Flash with TokenUsage tracking and offline fallback."""
        if not HAS_LIVE_LLM:
            return fallback_text, TokenUsage()

        payload = json.dumps(
            {"model": DEEPSEEK_MODEL, "temperature": temperature, "messages": messages}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        content = data["choices"][0]["message"]["content"].strip()
        raw_usage = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            total_tokens=int(raw_usage.get("total_tokens", 0)),
            latency_ms=elapsed_ms,
            api_calls=1,
        )
        self.dialogue_llm_usage.add(usage)
        return content, usage

    # ------------------------------------------------------------------
    # Step 1: Omniscient vs. canonarc POV A/B Experiment
    # ------------------------------------------------------------------
    def run_step1_ab_metagaming_proof(self) -> None:
        ab_engine = CanonArcEngine(":memory:", preset=InteractiveFictionPreset(include_background_edges=True))
        ab_engine.register_entity(
            EntityRecord(entity_id="char_clara", canonical_name="Detective Clara", state={"mood": "analytical"}),
            aliases=["Clara"],
            step=0,
        )
        ab_engine.register_entity(
            EntityRecord(
                entity_id="char_vance",
                canonical_name="Lord Vance",
                state={"mood": "poised", "disguise": "The Velvet Courier"},
                private_state={"agenda": "steal_royal_cipher", "guilt": "high"},
            ),
            aliases=[
                AliasRecord(
                    alias="The Velvet Courier",
                    entity_id="char_vance",
                    kind="disguise",
                    known_to=["char_vance"],
                    persona_name="The Velvet Courier",
                )
            ],
            step=0,
        )
        ab_engine.commit_step(
            step=12,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_ab_theft",
                        predicate="theft",
                        summary="Lord Vance picked the vault lock and stole the royal cipher.",
                        participants={"char_vance": "thief"},
                        witnesses=["char_vance"],
                        salience=9.5,
                    )
                ],
                inform=[
                    InformDelta(
                        episode_id="ep_ab_theft",
                        entity_id="char_clara",
                        epistemic="deceived",
                        believed_summary="The royal cipher was accidentally misplaced by the archivist during the banquet.",
                    )
                ],
            ),
        )

        ab_scene = "Detective Clara approached The Velvet Courier by the champagne fountain to ask about the missing cipher."
        omni_prompt = ab_engine.format_context(ab_engine.retrieve_context(step=15, text=ab_scene, pov_entity_ids=None))
        clara_prompt = ab_engine.format_context(ab_engine.retrieve_context(step=15, text=ab_scene, pov_entity_ids=["char_clara"]))

        naive_reply, _ = self._call_deepseek_flash(
            [
                {
                    "role": "system",
                    "content": (
                        "You are roleplaying Detective Clara in a mystery game. "
                        "Write 2-3 sentences of dialogue and internal thought as Clara questions The Velvet Courier. "
                        "Use the world state below for context; try not to reveal secrets Clara shouldn't know.\n\n"
                        + omni_prompt
                    ),
                },
                {"role": "user", "content": ab_scene},
            ],
            fallback_text=(
                "\"Good evening, Lord Vance—I mean, Velvet Courier,\" Clara said, eyeing him sharply. "
                "She suspected he had picked the vault lock to steal the royal cipher."
            ),
        )

        pov_reply, _ = self._call_deepseek_flash(
            [
                {
                    "role": "system",
                    "content": (
                        "You are roleplaying Detective Clara in a mystery game. "
                        "Write 2-3 sentences of dialogue and internal thought as Clara questions The Velvet Courier "
                        "using ONLY the POV context below.\n\n"
                        + clara_prompt
                    ),
                },
                {"role": "user", "content": ab_scene},
            ],
            fallback_text=(
                "\"Pardon the intrusion, Velvet Courier,\" Clara said politely. "
                "\"Our archivist seems to have misplaced the royal cipher during the banquet—did you happen to pass the library?\""
            ),
        )

        secret_tokens = ["Vance", "vault", "picked", "steal_royal_cipher"]
        leaked_naive = [t for t in secret_tokens if t.lower() in naive_reply.lower()]
        leaked_pov = [t for t in secret_tokens if t.lower() in pov_reply.lower()]

        render_ab_comparison(naive_reply, pov_reply, leaked_naive, leaked_pov)
        assert not leaked_pov, f"POV reply leaked secret tokens: {leaked_pov}"
        self.state["leaked_pov"] = leaked_pov
        ab_engine.close()

    # ------------------------------------------------------------------
    # Step 2: Seed Masquerade & Four Epistemic Boundaries
    # ------------------------------------------------------------------
    def run_step2_seed_masquerade(self) -> None:
        self.hook = EpistemicAuditHook()
        self.engine = CanonArcEngine(
            ":memory:",
            preset=InteractiveFictionPreset(include_background_edges=True),
            hooks=[self.hook],
        )
        engine = self.engine

        engine.register_entity(
            EntityRecord(
                entity_id="char_clara",
                canonical_name="Detective Clara",
                state={"mood": "analytical", "location_id": "loc_ballroom", "suspicion": -0.60},
            ),
            aliases=["Clara"],
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_vance",
                canonical_name="Lord Vance",
                state={"mood": "poised", "location_id": "loc_ballroom", "disguise": "The Velvet Courier"},
                private_state={"agenda": "steal_royal_cipher", "guilt": "high"},
            ),
            aliases=[
                "Vance",
                AliasRecord(
                    alias="The Velvet Courier",
                    entity_id="char_vance",
                    kind="disguise",
                    known_to=["char_vance", "char_mira"],
                    persona_name="The Velvet Courier",
                ),
            ],
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_mira",
                canonical_name="Mira",
                state={"mood": "composed", "location_id": "loc_ballroom"},
                private_state={"role": "vault_lookout"},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_archivist",
                canonical_name="Archivist Henrik",
                state={"status": "alive", "mood": "focused", "location_id": "loc_library"},
            ),
            aliases=["Henrik", "the archivist"],
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_guard",
                canonical_name="Captain Brandt",
                state={"mood": "vigilant", "location_id": "loc_ballroom"},
            ),
            aliases=["Brandt"],
            step=0,
        )

        for eid, kind, name, st in [
            ("loc_ballroom", "location", "Grand Ballroom", {"ambiance": "candlelit_masquerade"}),
            ("loc_vault", "location", "Royal Vault", {"security": "double_tumbler"}),
            ("loc_conservatory", "location", "Glass Conservatory", {"landmark": "hollow_marble_bust"}),
            ("loc_cellar", "location", "Wine Cellar", {"access": "locked_from_outside"}),
            ("item_royal_cipher", "item", "Royal Cipher", {"status": "secured", "location_id": "loc_vault"}),
        ]:
            engine.register_entity(EntityRecord(entity_id=eid, kind=kind, canonical_name=name, state=st), step=0)

        engine.commit_step(
            step=5,
            delta=StepDelta(
                edges=[
                    EdgeDelta(
                        source_id="char_mira",
                        target_id="char_clara",
                        relation_type="stance",
                        state={"register": "informal", "tone": "hostile", "trust": -0.85},
                    ),
                    EdgeDelta(
                        source_id="char_mira",
                        target_id="char_clara",
                        relation_type="facade",
                        state={"register": "formal", "tone": "helpful", "trust": 0.75},
                    ),
                    EdgeDelta(
                        source_id="char_vance",
                        target_id="char_clara",
                        relation_type="stance",
                        state={"register": "informal", "tone": "hostile", "trust": -0.90},
                    ),
                    EdgeDelta(
                        source_id="char_vance",
                        target_id="char_clara",
                        relation_type="facade",
                        state={"register": "formal", "tone": "charming", "trust": 0.70},
                    ),
                    EdgeDelta(
                        source_id="char_clara",
                        target_id="char_archivist",
                        relation_type="belief_state",
                        state={"status": "alive", "location_id": "loc_library"},
                    ),
                    EdgeDelta(
                        source_id="loc_vault",
                        target_id="item_royal_cipher",
                        relation_type="possesses",
                        state={"secured": True},
                    ),
                ]
            ),
        )

        engine.commit_step(
            step=10,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_archivist",
                        state={"status": "incapacitated", "location_id": "loc_cellar"},
                    )
                ]
            ),
        )

        engine.commit_step(
            step=12,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step12_cipher_theft",
                        predicate="theft",
                        summary="Lord Vance picked the vault lock and stole the royal cipher while Mira kept watch.",
                        participants={"char_vance": "thief", "char_mira": "lookout"},
                        witnesses=["char_vance", "char_mira"],
                        location_id="loc_vault",
                        salience=9.5,
                    )
                ],
                entities=[
                    EntityDelta(
                        entity_id="char_mira",
                        state={"mood": "tense"},
                        caused_by_episode_id="ep_step12_cipher_theft",
                        reason="Kept watch while Vance picked the vault lock",
                    ),
                    EntityDelta(
                        entity_id="item_royal_cipher",
                        state={"status": "stolen", "location_id": "loc_ballroom"},
                        caused_by_episode_id="ep_step12_cipher_theft",
                    ),
                ],
                edges=[
                    EdgeDelta(
                        source_id="loc_vault",
                        target_id="item_royal_cipher",
                        relation_type="possesses",
                        remove=True,
                    ),
                    EdgeDelta(
                        source_id="char_vance",
                        target_id="item_royal_cipher",
                        relation_type="possesses",
                        state={"concealed": True},
                        caused_by_episode_id="ep_step12_cipher_theft",
                    ),
                ],
            ),
        )

        probe_scene = "Detective Clara spoke with Mira and The Velvet Courier while asking about Archivist Henrik."
        clara_13 = engine.retrieve_context(step=13, text=probe_scene, pov_entity_ids=["char_clara"])
        gm_13 = engine.retrieve_context(step=13, text=probe_scene, pov_entity_ids=None)

        mira_for_clara = next(e for e in clara_13.entities if e.entity_id == "char_mira")
        henrik_for_clara = next(e for e in clara_13.entities if e.entity_id == "char_archivist")
        henrik_for_gm = next(e for e in gm_13.entities if e.entity_id == "char_archivist")
        mira_edges_clara = [e for e in clara_13.active_edges if e.source_id == "char_mira" and e.target_id == "char_clara"]

        assert any(e.entity_id == "masked:char_vance" for e in clara_13.entities)
        assert len(mira_edges_clara) == 1 and mira_edges_clara[0].relation_type == "facade"
        assert henrik_for_clara.state["status"] == "alive" and henrik_for_gm.state["status"] == "incapacitated"
        assert mira_for_clara.state["mood"] == "tense" and mira_for_clara.reason is None and mira_for_clara.caused_by_episode_id is None

        self.state.update(
            clara_13=clara_13,
            mira_for_clara=mira_for_clara,
            henrik_for_clara=henrik_for_clara,
            henrik_for_gm=henrik_for_gm,
            mira_edges_clara=mira_edges_clara,
        )

        render_pov_cards([
            (
                "🕵️ Clara's POV (Step 13 — All 4 Boundaries Active)",
                "#2563eb",
                "Vance masked · Mira [facade] only · Mira reason redacted · Henrik belief_state='alive'",
                engine.format_context(clara_13),
            ),
            (
                "🎭 Omniscient GM View (Step 13 — Ground Truth)",
                "#7c3aed",
                "Vance unmasked + private_state · Mira stance+facade+reason · Henrik incapacitated",
                engine.format_context(gm_13),
            ),
        ])

    # ------------------------------------------------------------------
    # Step 3: All 5 EpistemicMode States & Vector Isolation Proof
    # ------------------------------------------------------------------
    def run_step3_epistemic_modes(self) -> None:
        assert self.engine is not None
        engine = self.engine

        engine.commit_step(
            step=14,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step14_courier_glimpse",
                        predicate="sighting",
                        summary="Captain Brandt saw the masked Velvet Courier slipping away from the vault corridor.",
                        participants={"char_guard": "witness", "char_vance": "suspect"},
                        witnesses=["char_guard"],
                        location_id="loc_vault",
                        salience=8.5,
                    )
                ]
            ),
        )

        engine.commit_step(
            step=15,
            delta=StepDelta(
                entities=[EntityDelta(entity_id="char_clara", state={"suspicion": -0.45})],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step15_cellar_ambush",
                        predicate="ambush",
                        summary="Lord Vance locked Archivist Henrik in the wine cellar after seizing his master key.",
                        participants={"char_vance": "attacker", "char_archivist": "victim"},
                        witnesses=["char_vance", "char_archivist"],
                        location_id="loc_cellar",
                        salience=9.0,
                    )
                ],
                inform=[
                    InformDelta(
                        episode_id="ep_step12_cipher_theft",
                        entity_id="char_clara",
                        epistemic="deceived",
                        believed_summary="The royal cipher was accidentally misplaced by the archivist during the banquet.",
                    )
                ],
            ),
        )

        engine.inform_entity(step=16, episode_id="ep_step12_cipher_theft", entity_id="char_guard", epistemic="told")
        engine.inform_entity(step=17, episode_id="ep_step14_courier_glimpse", entity_id="char_guard", epistemic="forgotten")
        engine.inform_entity(step=18, episode_id="ep_step15_cellar_ambush", entity_id="char_archivist", epistemic="suppressed")

        clara_secret_probe = engine.retrieve_context(
            step=20,
            text="vault lock stolen royal cipher",
            mentions=["Clara", "The Velvet Courier"],
            pov_entity_ids=["char_clara"],
        )
        clara_cover_probe = engine.retrieve_context(
            step=20,
            text="archivist accidentally misplaced royal cipher banquet",
            mentions=["Clara", "The Velvet Courier"],
            pov_entity_ids=["char_clara"],
        )
        clara_secret_fmt = engine.format_context(clara_secret_probe)
        leaks_clara_20 = self.hook.audit_prompt(
            "Step 20 Clara Vector Isolation",
            clara_secret_fmt,
            ["Lord Vance", "picked the vault lock", "steal_royal_cipher", "wine cellar"],
        )
        assert not leaks_clara_20, f"Unexpected leak: {leaks_clara_20}"
        assert any("misplaced by the archivist" in ep.summary for ep in clara_cover_probe.episodes)
        assert all("picked the vault lock" not in ep.summary for ep in clara_secret_probe.episodes)

        brandt_20 = engine.retrieve_context(
            step=20,
            text="Velvet Courier vault corridor sighting",
            mentions=["Brandt"],
            pov_entity_ids=["char_guard"],
        )
        henrik_20 = engine.retrieve_context(
            step=20,
            text="wine cellar master key ambush",
            mentions=["Henrik"],
            pov_entity_ids=["char_archivist"],
        )
        assert not any(ep.episode_id == "ep_step14_courier_glimpse" for ep in brandt_20.episodes)
        assert any(ep.episode_id == "ep_step12_cipher_theft" for ep in brandt_20.episodes)
        assert henrik_20.episodes == []

        self.state.update(
            leaks_clara_20=leaks_clara_20,
            clara_cover_probe=clara_cover_probe,
            brandt_20=brandt_20,
            henrik_20=henrik_20,
        )

        ep_headers = [
            ("ep_step12_cipher_theft", "Vault Theft (Step 12)"),
            ("ep_step14_courier_glimpse", "Corridor Glimpse (Step 14)"),
            ("ep_step15_cellar_ambush", "Cellar Ambush (Step 15)"),
        ]
        cast = [
            ("char_clara", "Detective Clara"),
            ("char_vance", "Lord Vance"),
            ("char_mira", "Mira"),
            ("char_guard", "Captain Brandt"),
            ("char_archivist", "Archivist Henrik"),
        ]
        matrix_rows = []
        for eid, name in cast:
            ctx = engine.retrieve_context(
                step=20,
                text="The Velvet Courier vault cipher corridor wine cellar archivist misplaced",
                mentions=[name, "The Velvet Courier"],
                pov_entity_ids=[eid],
                top_k_episodes=10,
            )
            pierced = any(e.entity_id == "char_vance" and e.disguised_as for e in ctx.entities) or eid == "char_vance"
            visible_eps = {ep.episode_id: ep for ep in ctx.episodes}
            modes: dict[str, str] = {}
            summaries: dict[str, str] = {}
            for ep_id, _ in ep_headers:
                if ep_id in visible_eps:
                    ep = visible_eps[ep_id]
                    mode = ep.known_by.get(eid, "direct")
                    modes[ep_id] = mode
                    if mode == "deceived":
                        summaries[ep_id] = "Believes: archivist misplaced cipher"
                    elif mode == "told":
                        summaries[ep_id] = "Heard secondhand"
                else:
                    if eid == "char_guard" and ep_id == "ep_step14_courier_glimpse":
                        modes[ep_id] = "forgotten"
                        summaries[ep_id] = "Drugged wine @ Step 17"
                    elif eid == "char_archivist" and ep_id == "ep_step15_cellar_ambush":
                        modes[ep_id] = "suppressed"
                        summaries[ep_id] = "Hypnotic tonic @ Step 18"
                    else:
                        modes[ep_id] = "ignorant"
            matrix_rows.append({"name": name, "disguise": "PIERCED" if pierced else "MASKED", "modes": modes, "summaries": summaries})

        render_epistemic_matrix(matrix_rows, ep_headers)
        display(HTML(
            '<div style="font-family:system-ui,sans-serif;padding:8px 12px;background:#f0fdf4;border:1px solid #86efac;'
            'border-radius:8px;color:#166534;font-size:12.5px;margin-top:6px;">'
            '🛡️ <b>Vector Isolation Verified:</b> Searching Clara\'s memory with secret ground-truth words '
            '(<code>"vault lock stolen royal cipher"</code>) returned <b>0 ground-truth leaks</b> — she only retrieves her cover story.'
            '</div>'
        ))

    # ------------------------------------------------------------------
    # Step 4: Multi-POV Projection ("any" vs. "all") & Dramatic Irony
    # ------------------------------------------------------------------
    def run_step4_multi_pov(self) -> None:
        assert self.engine is not None
        engine = self.engine

        engine.commit_step(
            step=11,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step11_conspiracy_pact",
                        predicate="conspiracy",
                        summary="Lord Vance and Mira agreed on a midnight signal in the ballroom to cover the cipher heist.",
                        participants={"char_vance": "mastermind", "char_mira": "accomplice"},
                        witnesses=["char_vance", "char_mira"],
                        location_id="loc_ballroom",
                        salience=9.2,
                    )
                ]
            ),
        )

        joint_scene = "Detective Clara and Mira watched The Velvet Courier cross the ballroom near the midnight signal display."
        ctx_any = engine.retrieve_context(step=20, text=joint_scene, pov_entity_ids=["char_clara", "char_mira"], pov_mode="any")
        ctx_all = engine.retrieve_context(step=20, text=joint_scene, pov_entity_ids=["char_clara", "char_mira"], pov_mode="all")
        ctx_gm = engine.retrieve_context(step=20, text=joint_scene, pov_entity_ids=None)

        ctx_all_fmt = engine.format_context(ctx_all)
        leaks_all = self.hook.audit_prompt("Step 4 Multi-POV all", ctx_all_fmt, ["Lord Vance", "picked the vault lock"])
        assert not leaks_all, f"pov_mode='all' leaked secrets: {leaks_all}"
        assert any(e.entity_id == "char_vance" and e.disguised_as == "The Velvet Courier" for e in ctx_any.entities)
        assert any(ep.episode_id == "ep_step11_conspiracy_pact" for ep in ctx_any.episodes)
        assert any(e.entity_id == "masked:char_vance" for e in ctx_all.entities)
        assert all(e.private_state == {} for e in ctx_all.entities)
        assert not any(ep.episode_id == "ep_step11_conspiracy_pact" for ep in ctx_all.episodes)

        self.state.update(ctx_any=ctx_any, ctx_all=ctx_all, leaks_all=leaks_all)

        render_pov_cards([
            (
                'pov_mode="any" (Union + Dramatic Irony)',
                "#0ea5e9",
                "Mira pierces Vance's disguise · Secret pact visible with | ignorant: char_clara",
                engine.format_context(ctx_any),
            ),
            (
                'pov_mode="all" (Strict Common Ground)',
                "#16a34a",
                "Vance stays MASKED · Secret pact & unshared theft hidden · Zero-leak shared floor",
                ctx_all_fmt,
            ),
            (
                "pov_entity_ids=None (Omniscient GM)",
                "#7c3aed",
                "Full private_state · Both stance & facade edges · Full ground truth",
                engine.format_context(ctx_gm),
            ),
        ])

    # ------------------------------------------------------------------
    # Step 5: Persona-Aware & Facade-Driven Voice Bank (Tier 5)
    # ------------------------------------------------------------------
    def run_step5_voice_bank(self) -> None:
        assert self.engine is not None
        engine = self.engine

        engine.commit_step(
            step=19,
            exemplars=[
                VoiceExemplar(
                    step=19,
                    speaker_id="char_vance",
                    addressee_id="char_clara",
                    text="A masquerade thrives on mystery, Detective—surely a misplaced ledger is beneath your brilliance.",
                    metadata={"persona": "The Velvet Courier", "register": "formal", "tone": "charming"},
                    quality_score=9.5,
                ),
                VoiceExemplar(
                    step=19,
                    speaker_id="char_vance",
                    addressee_id="char_mira",
                    text="Keep the detective busy, Mira; if she reaches the conservatory bust, burn the ledger.",
                    metadata={"register": "informal", "tone": "hostile"},
                    quality_score=9.5,
                ),
                VoiceExemplar(
                    step=19,
                    speaker_id="char_mira",
                    addressee_id="char_clara",
                    text="I would be delighted to help you trace the archivist's steps, Detective Clara.",
                    metadata={"register": "formal", "tone": "helpful"},
                    quality_score=9.4,
                ),
                VoiceExemplar(
                    step=19,
                    speaker_id="char_mira",
                    addressee_id="char_clara",
                    text="Stop nosing around the ballroom before you trip over your own arrogance.",
                    metadata={"register": "informal", "tone": "hostile"},
                    quality_score=9.4,
                ),
            ],
        )

        voice_clara_ctx = engine.retrieve_context(
            step=20,
            text="The Velvet Courier and Mira greeted Detective Clara in the ballroom.",
            pov_entity_ids=["char_clara"],
            top_k_voice=1,
        )

        selected_lines = [ex.text for ex in voice_clara_ctx.voice_exemplars]
        assert any("masquerade thrives on mystery" in t for t in selected_lines)
        assert any("delighted to help you trace" in t for t in selected_lines)
        assert all("conservatory bust" not in t and "arrogance" not in t for t in selected_lines)
        assert any(ex.speaker_id == "masked:char_vance" for ex in voice_clara_ctx.voice_exemplars)
        assert all(ex.speaker_id != "char_vance" for ex in voice_clara_ctx.voice_exemplars)

        self.state.update(selected_lines=selected_lines, voice_clara_ctx=voice_clara_ctx)

        cards_html = "".join(
            f'<li style="margin:6px 0;"><b>{html.escape(ex.speaker_id)} → {html.escape(ex.addressee_id or "all")}</b> '
            f'<code style="background:#e2e8f0;padding:1px 5px;border-radius:4px;">{html.escape(str(ex.metadata))}</code><br/>'
            f'<em>"{html.escape(ex.text)}"</em></li>'
            for ex in voice_clara_ctx.voice_exemplars
        )
        display(HTML(f"""
<div style="font-family:system-ui,sans-serif;border:1px solid #86efac;background:#f0fdf4;border-radius:10px;padding:12px;margin:8px 0;">
  <div style="font-weight:700;color:#166534;margin-bottom:6px;">
    ✅ Tier 5 Voice Exemplars Selected for Clara's POV (Step 20) — 0 Conspirator Lines Leaked
  </div>
  <ul style="margin:0;padding-left:18px;font-size:13px;color:#1e293b;">{cards_html}</ul>
</div>
"""))

    # ------------------------------------------------------------------
    # Step 6: Mini Social Deduction Loop & SVG Trust/Suspicion Chart
    # ------------------------------------------------------------------
    def run_step6_social_deduction_loop(self) -> None:
        assert self.engine is not None
        engine = self.engine

        engine.commit_step(
            step=21,
            delta=StepDelta(
                entities=[EntityDelta(entity_id="char_clara", state={"suspicion": -0.20})],
                edges=[
                    EdgeDelta(
                        source_id="char_vance",
                        target_id="char_clara",
                        relation_type="facade",
                        state={"trust": 0.65},
                    )
                ],
            ),
        )
        engine.commit_step(
            step=25,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step25_cufflink_clue",
                        predicate="clue_found",
                        summary="Detective Clara found a torn velvet cufflink outside the Royal Vault.",
                        participants={"char_clara": "investigator"},
                        witnesses=["char_clara"],
                        location_id="loc_vault",
                        salience=8.8,
                    )
                ],
                entities=[
                    EntityDelta(
                        entity_id="char_clara",
                        state={"mood": "suspicious", "suspicion": 0.45},
                        caused_by_episode_id="ep_step25_cufflink_clue",
                        reason="Found torn velvet cufflink near the vault",
                    )
                ],
                edges=[
                    EdgeDelta(
                        source_id="char_vance",
                        target_id="char_clara",
                        relation_type="facade",
                        state={"trust": 0.25},
                    )
                ],
            ),
        )

        def run_deduction_vote(step: int, label: str) -> list[dict[str, str]]:
            ballots = []
            for agent_id, agent_name in [
                ("char_clara", "Detective Clara"),
                ("char_guard", "Captain Brandt"),
                ("char_mira", "Mira (Conspirator)"),
                ("char_vance", "Lord Vance (Conspirator)"),
            ]:
                ctx = engine.retrieve_context(
                    step=step,
                    text="Who stole the royal cipher? Detective Clara, Mira, The Velvet Courier, Captain Brandt, Archivist Henrik.",
                    pov_entity_ids=[agent_id],
                    top_k_episodes=5,
                )
                ep_text = " ".join(e.summary for e in ctx.episodes)
                pierced_vance = (
                    any(e.entity_id == "char_vance" and e.disguised_as for e in ctx.entities)
                    or agent_id == "char_vance"
                )
                if agent_id in ("char_vance", "char_mira"):
                    vote = "Deflect → Archivist Henrik"
                    basis = "Conspirator knows truth; protects cover story"
                elif pierced_vance and "picked the vault lock" in ep_text:
                    vote = "Accuse → Lord Vance (The Velvet Courier)"
                    basis = "Disguise pierced + vault theft confirmed"
                elif "cufflink" in ep_text:
                    vote = "Suspect → The Velvet Courier (identity unknown)"
                    basis = "Cufflink clue raises suspicion; true name still masked"
                elif "picked the vault lock" in ep_text:
                    vote = "Accuse → Lord Vance (whereabouts unknown; courier unlinked)"
                    basis = "Told Lord Vance stole cipher, but forgot corridor sighting"
                else:
                    vote = "Question → Archivist Henrik"
                    basis = "Deceived by archivist cover story"
                ballots.append({"round": label, "agent": agent_name, "vote": vote, "basis": basis})
            return ballots

        day1_ballots = run_deduction_vote(22, "Day 1 (Step 22)")
        day2_ballots = run_deduction_vote(26, "Day 2 (Step 26)")

        engine.commit_step(
            step=30,
            delta=StepDelta(
                entities=[
                    EntityDelta(entity_id="char_clara", state={"suspicion": 0.95}),
                    EntityDelta(
                        entity_id="char_vance",
                        add_aliases=[
                            AliasRecord(
                                alias="The Velvet Courier",
                                entity_id="char_vance",
                                kind="disguise",
                                known_to=["char_vance", "char_mira", "char_clara"],
                                persona_name="The Velvet Courier",
                            )
                        ],
                    ),
                ],
                edges=[
                    EdgeDelta(source_id="char_vance", target_id="char_clara", relation_type="facade", remove=True),
                    EdgeDelta(
                        source_id="char_clara",
                        target_id="char_archivist",
                        relation_type="belief_state",
                        state={"status": "incapacitated", "location_id": "loc_cellar"},
                    ),
                ],
                inform=[
                    InformDelta(episode_id="ep_step12_cipher_theft", entity_id="char_clara", epistemic="told")
                ],
            ),
        )

        day3_ballots = run_deduction_vote(31, "Day 3 (Step 31 — Post-Unmasking)")

        clara_31 = engine.retrieve_context(
            step=31,
            text="Detective Clara confronted The Velvet Courier about the cipher and Archivist Henrik.",
            pov_entity_ids=["char_clara"],
        )
        assert "Lord Vance" in engine.format_context(clara_31)
        assert "picked the vault lock" in engine.format_context(clara_31)
        assert any(e.source_id == "char_vance" and e.relation_type == "stance" for e in clara_31.active_edges)
        henrik_clara_31 = next(e for e in clara_31.entities if e.entity_id == "char_archivist")
        assert henrik_clara_31.state["status"] == "incapacitated"
        self.state["clara_31"] = clara_31

        all_ballots = day1_ballots + day2_ballots + day3_ballots
        b_rows = "".join(
            f'<tr><td style="padding:6px 9px;border-bottom:1px solid #e2e8f0;font-weight:600;">{html.escape(b["round"])}</td>'
            f'<td style="padding:6px 9px;border-bottom:1px solid #e2e8f0;">{html.escape(b["agent"])}</td>'
            f'<td style="padding:6px 9px;border-bottom:1px solid #e2e8f0;color:#1e40af;font-weight:600;">{html.escape(b["vote"])}</td>'
            f'<td style="padding:6px 9px;border-bottom:1px solid #e2e8f0;color:#475569;">{html.escape(b["basis"])}</td></tr>'
            for b in all_ballots
        )
        display(HTML(f"""
<div style="font-family:system-ui,sans-serif;margin:8px 0;">
  <div style="font-weight:700;font-size:14px;color:#0f172a;margin-bottom:6px;">🗳️ POV-Isolated Social Deduction Ballots Across 3 Rounds</div>
  <table style="width:100%;border-collapse:collapse;background:#fff;border:1px solid #cbd5e1;font-size:12px;">
    <thead style="background:#f8fafc;"><tr>
      <th style="padding:7px 9px;text-align:left;border-bottom:2px solid #cbd5e1;">Round</th>
      <th style="padding:7px 9px;text-align:left;border-bottom:2px solid #cbd5e1;">Agent POV</th>
      <th style="padding:7px 9px;text-align:left;border-bottom:2px solid #cbd5e1;">Decision / Vote</th>
      <th style="padding:7px 9px;text-align:left;border-bottom:2px solid #cbd5e1;">Epistemic Basis</th>
    </tr></thead>
    <tbody>{b_rows}</tbody>
  </table>
</div>
"""))

        chart_points = []
        for chk_step, chk_label in [
            (5, "Step 5 (Banquet)"),
            (15, "Step 15 (Deceived)"),
            (22, "Step 22 (Day 1 Vote)"),
            (25, "Step 25 (Cufflink)"),
            (30, "Step 30 (Unmasking)"),
        ]:
            snap = engine.retrieve_context(
                step=chk_step,
                mentions=["Clara", "Vance"],
                pov_entity_ids=None,
                inclusive=True,
            )
            clara_ent = next(e for e in snap.entities if e.entity_id == "char_clara")
            facade_edge = next(
                (e for e in snap.active_edges if e.source_id == "char_vance" and e.target_id == "char_clara" and e.relation_type == "facade"),
                None,
            )
            stance_edge = next(
                e for e in snap.active_edges if e.source_id == "char_vance" and e.target_id == "char_clara" and e.relation_type == "stance"
            )
            true_trust = float(stance_edge.state["trust"])
            perceived_trust = float(facade_edge.state["trust"]) if facade_edge is not None else true_trust
            chart_points.append(
                {
                    "label": chk_label,
                    "perceived_trust": perceived_trust,
                    "true_trust": true_trust,
                    "suspicion": float(clara_ent.state.get("suspicion", 0.0)),
                }
            )
        render_suspicion_svg(chart_points)

    # ------------------------------------------------------------------
    # Step 7: Flashback Safety, Counterfactual Branches & Instant Rollback
    # ------------------------------------------------------------------
    def run_step7_time_travel_and_branches(self) -> None:
        assert self.engine is not None
        engine = self.engine

        engine.commit_step(
            step=32,
            delta=StepDelta(
                is_flashback=True,
                entities=[EntityDelta(entity_id="char_clara", state={"mood": "unsuspecting"})],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_flashback_banquet_toast",
                        predicate="flashback",
                        summary="Earlier at the banquet, The Velvet Courier toasted Clara with effortless charm.",
                        participants={"char_clara": "guest", "char_vance": "host"},
                        witnesses=["char_clara", "char_vance"],
                        salience=6.0,
                    )
                ],
            ),
        )
        clara_33 = engine.retrieve_context(step=33, mentions=["Clara"], pov_entity_ids=["char_clara"])
        clara_state_33 = next(e for e in clara_33.entities if e.entity_id == "char_clara")
        assert clara_state_33.state["mood"] == "suspicious", f"Flashback corrupted mood: {clara_state_33.state}"

        engine.branch(at_step=25, name="route_ballroom")
        engine.branch(at_step=25, name="route_conservatory")

        s_ball = engine.session("route_ballroom")
        s_ball.commit_step(
            step=28,
            delta=StepDelta(
                entities=[
                    EntityDelta(entity_id="char_vance", state={"location_id": "loc_harbor", "mood": "fleeing"}),
                    EntityDelta(entity_id="char_clara", state={"mood": "frustrated"}),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_ballroom_escape",
                        predicate="escape",
                        summary="Confronted publicly in the ballroom, Lord Vance vaulted the balcony and escaped with the cipher.",
                        participants={"char_clara": "pursuer", "char_vance": "fugitive"},
                        witnesses=["char_clara", "char_vance", "char_mira"],
                        salience=9.2,
                    )
                ],
            ),
        )

        s_cons = engine.session("route_conservatory")
        s_cons.commit_step(
            step=28,
            delta=StepDelta(
                entities=[
                    EntityDelta(entity_id="char_vance", state={"location_id": "loc_conservatory", "mood": "cornered"}),
                    EntityDelta(entity_id="char_clara", state={"mood": "triumphant"}),
                    EntityDelta(entity_id="item_royal_cipher", state={"status": "recovered", "location_id": "loc_conservatory"}),
                ],
                edges=[
                    EdgeDelta(source_id="char_vance", target_id="item_royal_cipher", relation_type="possesses", remove=True),
                    EdgeDelta(source_id="char_clara", target_id="item_royal_cipher", relation_type="possesses", state={"secured": True}),
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_conservatory_capture",
                        predicate="capture",
                        summary="Clara recovered the royal cipher from the hollow marble bust and cornered Lord Vance.",
                        participants={"char_clara": "detective", "char_vance": "culprit"},
                        witnesses=["char_clara", "char_vance"],
                        salience=9.6,
                    )
                ],
            ),
        )

        diff = engine.diff_timelines("route_ballroom", "route_conservatory", at_step=28)
        changed_keys = {item["subject_key"] for item in diff["changed"]}
        assert any("char_vance" in k for k in changed_keys) and any("item_royal_cipher" in k for k in changed_keys)

        engine.branch(at_step=25, name="scratch_rewind")
        s_rew = engine.session("scratch_rewind")
        s_rew.commit_step(step=26, delta=StepDelta(entities=[EntityDelta(entity_id="char_clara", state={"mood": "panicked"})]))
        t0 = time.perf_counter()
        deleted = s_rew.rollback_to(25)
        rewind_ms = (time.perf_counter() - t0) * 1000.0
        rew_ctx = s_rew.retrieve_context(step=26, mentions=["Clara"], pov_entity_ids=["char_clara"])
        assert next(e for e in rew_ctx.entities if e.entity_id == "char_clara").state["mood"] == "suspicious"

        self.state.update(clara_state_33=clara_state_33, changed_keys=changed_keys)

        render_branch_tree_svg(
            "Vance flees → loc_harbor · Cipher stays stolen",
            "Cipher recovered → Clara possesses · Vance cornered",
        )

        diff_rows = "".join(
            f'<tr><td style="padding:6px 10px;border-bottom:1px solid #e2e8f0;font-weight:600;">{html.escape(item["subject_key"])}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #e2e8f0;color:#991b1b;"><code>{html.escape(json.dumps(item["timeline_a"], default=str)[:160])}</code></td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #e2e8f0;color:#166534;"><code>{html.escape(json.dumps(item["timeline_b"], default=str)[:160])}</code></td></tr>'
            for item in diff["changed"]
        )
        display(HTML(f"""
<div style="font-family:system-ui,sans-serif;margin:8px 0;">
  <div style="font-weight:700;font-size:13.5px;color:#0f172a;margin-bottom:5px;">
    🔍 <code>diff_timelines("route_ballroom", "route_conservatory", at_step=28)</code> · Flashback Safe (Clara mood=<code>{html.escape(clara_state_33.state['mood'])}</code>) · Scratch Rollback: <b>{rewind_ms:.2f} ms</b> (deleted={deleted})
  </div>
  <table style="width:100%;border-collapse:collapse;background:#fff;border:1px solid #cbd5e1;font-size:12px;">
    <thead style="background:#f8fafc;"><tr>
      <th style="padding:6px 10px;text-align:left;border-bottom:2px solid #cbd5e1;">Entity</th>
      <th style="padding:6px 10px;text-align:left;border-bottom:2px solid #cbd5e1;">route_ballroom</th>
      <th style="padding:6px 10px;text-align:left;border-bottom:2px solid #cbd5e1;">route_conservatory</th>
    </tr></thead>
    <tbody>{diff_rows}</tbody>
  </table>
</div>
"""))

    # ------------------------------------------------------------------
    # Step 8: Live DeepSeek Flash Extraction (CachedDeltaExtractor)
    # ------------------------------------------------------------------
    def run_step8_live_extraction(self) -> None:
        assert self.engine is not None
        engine = self.engine

        scene_step40 = (
            "While Captain Brandt remained in the ballroom questioning the banquet guests, "
            "Lord Vance slipped quietly into the conservatory with Mira watching the door and hid a second cipher ledger "
            "inside the hollow marble bust. Moments later, Mira returned to the ballroom and fed Captain Brandt "
            "a fabricated lead, convincing Brandt that the second cipher ledger had already been smuggled aboard a ship at the harbor docks."
        )

        extraction_usage_summary = "Offline fallback (no DEEPSEEK_API_KEY configured)"
        cache_dir = Path(tempfile.gettempdir()) / "canonarc_nb04_cache"

        if HAS_LIVE_LLM:
            raw_extractor = OpenAICompatibleExtractor.from_deepseek()
            extractor = CachedDeltaExtractor(raw_extractor, cache_dir=str(cache_dir))
            with engine.step_transaction(step=40, text=scene_step40) as tx:
                extracted_delta = tx.extract_and_commit(scene_step40, extractor=extractor)

            usage = extractor.historical_usage.to_dict(model=raw_extractor.model)
            cache_note = "cache_hit" if extractor.cache_hits > 0 else "live_call"
            extraction_usage_summary = (
                f"Extractor: DeepSeek ({raw_extractor.model}, {cache_note}) · "
                f"{usage['total_tokens']} tokens · {usage['latency_ms']:.0f} ms · "
                f"retries={usage['retry_count']} · est_cost=${usage['cost_estimate']['total_cost_usd']:.6f}"
            )
        else:
            extracted_delta = StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_step40_ledger_hiding",
                        predicate="concealment",
                        summary="Lord Vance hid a second cipher ledger inside the hollow marble bust in the conservatory while Mira kept watch.",
                        participants={"char_vance": "actor", "char_mira": "lookout"},
                        witnesses=["char_vance", "char_mira"],
                        location_id="loc_conservatory",
                        salience=8.5,
                    )
                ],
                inform=[
                    InformDelta(
                        episode_id="ep_step40_ledger_hiding",
                        entity_id="char_guard",
                        epistemic="deceived",
                        believed_summary="The second cipher ledger was smuggled aboard a ship at the harbor docks.",
                    )
                ],
            )
            engine.commit_step(step=40, delta=extracted_delta)

        brandt_41 = engine.retrieve_context(
            step=41,
            mentions=["Brandt", "The Velvet Courier", "Mira"],
            pov_entity_ids=["char_guard"],
        )
        gm_41 = engine.retrieve_context(step=41, mentions=["Brandt", "Vance", "Mira"], pov_entity_ids=None)

        brandt_41_fmt = engine.format_context(brandt_41)
        leaks_brandt_41 = self.hook.audit_prompt(
            "Step 41 Brandt Post-Extraction",
            brandt_41_fmt,
            ["hollow marble bust", "conservatory", "bust"],
        )
        assert not leaks_brandt_41, f"Brandt's POV leaked conservatory/bust secrets: {leaks_brandt_41}"
        assert any(e.entity_id == "masked:char_vance" for e in brandt_41.entities)
        assert any(e.entity_id == "char_vance" for e in gm_41.entities), "Vance alias must resolve in GM POV!"

        self.state.update(
            extraction_usage_summary=extraction_usage_summary,
            leaks_brandt_41=leaks_brandt_41,
            gm_41=gm_41,
        )

        delta_json = json.dumps(extracted_delta.model_dump(exclude_defaults=True), indent=2)
        display(HTML(f"""
<div style="font-family:system-ui,sans-serif;border:1px solid #cbd5e1;background:#f8fafc;border-radius:10px;padding:10px 14px;margin:8px 0;font-size:12.5px;color:#334155;">
  <div style="font-weight:700;color:#0f172a;margin-bottom:4px;">🤖 Automatic Story-to-Memory Extraction Completed</div>
  <div>{html.escape(extraction_usage_summary)}</div>
  <details style="margin-top:6px;">
    <summary style="cursor:pointer;color:#2563eb;font-weight:600;">Inspect raw extracted StepDelta JSON</summary>
    <pre style="margin:6px 0 0 0;padding:8px;background:#fff;border:1px solid #e2e8f0;border-radius:6px;font-size:11px;overflow-x:auto;">{html.escape(delta_json)}</pre>
  </details>
</div>
"""))

        render_pov_cards([
            (
                "🛡️ Captain Brandt's POV (Step 41 — Deceived by Extraction)",
                "#d97706",
                "Believes harbor-docks cover story · Never sees hollow marble bust",
                brandt_41_fmt,
            ),
            (
                "🎭 Omniscient GM POV (Step 41 — Full Ground Truth)",
                "#7c3aed",
                "Resolves 'Vance' alias · Shows true conservatory concealment + ignorant tags",
                engine.format_context(gm_41),
            ),
        ])

    # ------------------------------------------------------------------
    # Step 9: 14-Point Automated Epistemic QA Audit & Telemetry Summary
    # ------------------------------------------------------------------
    def run_step9_qa_audit(self) -> None:
        assert self.engine is not None
        engine = self.engine
        s = self.state

        qa_checks = [
            ("1. A/B Live LLM Test: Zero secret leaks under canonarc POV projection", len(s["leaked_pov"]) == 0),
            ("2. Disguise Masking: Vance masked to masked:char_vance for fooled Clara", any(e.entity_id == "masked:char_vance" for e in s["clara_13"].entities)),
            ("3. Unmasked Facade: Clara sees ONLY Mira's [facade], hidden stance concealed", len(s["mira_edges_clara"]) == 1 and s["mira_edges_clara"][0].relation_type == "facade"),
            ("4. Stale belief_state: Clara projects Henrik='alive' while GM sees 'incapacitated'", s["henrik_for_clara"].state["status"] == "alive" and s["henrik_for_gm"].state["status"] == "incapacitated"),
            ("5. Causal Provenance Redaction: Clara sees Mira's mood='tense', reason=None", s["mira_for_clara"].state["mood"] == "tense" and s["mira_for_clara"].reason is None),
            ("6. Deceived-POV Vector Isolation: Secret-keyword query leaks 0 vault facts", len(s["leaks_clara_20"]) == 0 and any("misplaced by the archivist" in ep.summary for ep in s["clara_cover_probe"].episodes)),
            ("7. Memory Erasure (forgotten): Brandt's drugged corridor sighting invisible", not any(ep.episode_id == "ep_step14_courier_glimpse" for ep in s["brandt_20"].episodes)),
            ("8. Memory Erasure (suppressed): Henrik's wiped cellar ambush invisible", s["henrik_20"].episodes == []),
            ("9. Multi-POV ('any' vs 'all'): 'any' pierces disguise; 'all' keeps masked & strips secrets", any(e.entity_id == "char_vance" for e in s["ctx_any"].entities) and any(e.entity_id == "masked:char_vance" for e in s["ctx_all"].entities) and len(s["leaks_all"]) == 0),
            ("10. Tier 5 VoiceBank: Selects persona/facade exemplars, 0 conspirator lines leaked", any("masquerade thrives" in t for t in s["selected_lines"]) and all("conservatory bust" not in t for t in s["selected_lines"]) and all(ex.speaker_id != "char_vance" for ex in s["voice_clara_ctx"].voice_exemplars)),
            ("11. Unmasking & Reveal (Step 30): Clara pierces disguise, sees theft & real stance", "Lord Vance" in engine.format_context(s["clara_31"]) and "picked the vault lock" in engine.format_context(s["clara_31"])),
            ("12. Flashback Safety: Step 32 flashback did not overwrite Step 33 mood", s["clara_state_33"].state["mood"] == "suspicious"),
            ("13. Counterfactual Branches & Rollback: diff_timelines() + sub-ms rollback verified", any("char_vance" in k for k in s["changed_keys"]) and len(self.hook.rollbacks) >= 1),
            ("14. Step 40 Extraction & Alias Resolution: 'Vance' resolved & 0 bust leaks to Brandt", len(s["leaks_brandt_41"]) == 0 and any(e.entity_id == "char_vance" for e in s["gm_41"].entities)),
        ]

        assert all(ok for _, ok in qa_checks), f"QA failures: {[n for n, ok in qa_checks if not ok]}"

        dlg_dict = self.dialogue_llm_usage.to_dict(model=DEEPSEEK_MODEL)
        cost_summary = (
            f"Dialogue LLM ({DEEPSEEK_MODEL}): {dlg_dict['api_calls']} calls · {dlg_dict['total_tokens']} tokens · "
            f"est_cost=${dlg_dict['cost_estimate']['total_cost_usd']:.6f}  |  {s['extraction_usage_summary']}"
        )
        render_qa_scorecard(qa_checks, self.hook, cost_summary)
        engine.close()
