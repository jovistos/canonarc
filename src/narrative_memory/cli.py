from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .engine import NarrativeMemoryEngine
from .models import TimelineInfo


def _render_tree(timelines: Sequence[TimelineInfo]) -> str:
    if not timelines:
        return "(no timelines)"
    children: dict[str | None, list[TimelineInfo]] = {}
    for tl in timelines:
        children.setdefault(tl.parent_id, []).append(tl)
    for group in children.values():
        group.sort(key=lambda t: (t.fork_step or 0, t.timeline_id))

    lines: list[str] = []

    def _walk(node: TimelineInfo, prefix: str, is_last: bool, is_root: bool) -> None:
        if is_root:
            connector = ""
            next_prefix = ""
        else:
            connector = "└─ " if is_last else "├─ "
            next_prefix = prefix + ("   " if is_last else "│  ")
        fork_info = f" (fork@{node.fork_step})" if node.fork_step is not None else ""
        lines.append(f"{prefix}{connector}{node.timeline_id}{fork_info} [head={node.head_step}]")
        kids = children.get(node.timeline_id, [])
        for idx, child in enumerate(kids):
            _walk(child, next_prefix, idx == len(kids) - 1, False)

    roots = children.get(None, [])
    for idx, root in enumerate(roots):
        _walk(root, "", idx == len(roots) - 1, True)
    return "\n".join(lines)


def _cmd_tree(args: argparse.Namespace) -> int:
    with NarrativeMemoryEngine(args.db_path, on_embedder_mismatch="ignore") as engine:
        timelines = engine.list_timelines()
    if args.json:
        print(json.dumps([t.model_dump() for t in timelines], indent=2))
    else:
        print(_render_tree(timelines))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    with NarrativeMemoryEngine(args.db_path, timeline_id=args.timeline) as engine:
        tl_info = engine.store.get_timeline(args.timeline)
        if tl_info is None:
            print(f"Error: timeline '{args.timeline}' not found.", file=sys.stderr)
            return 1
        target_step = args.step if args.step is not None else tl_info.head_step
        inclusive = args.step is None or bool(args.inclusive)
        ctx = engine.retrieve_context(
            target_step,
            text=args.text or "",
            pov_entity_ids=args.pov or None,
            inclusive=inclusive,
            timeline_id=args.timeline,
            fallback_on_empty_mentions="all" if not args.text else "pov_only",
        )
        if args.json:
            print(json.dumps(ctx.model_dump(), indent=2))
        else:
            print(engine.format_context(ctx))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    with NarrativeMemoryEngine(args.db_path, on_embedder_mismatch="ignore") as engine:
        diff = engine.diff_timelines(args.timeline_a, args.timeline_b, at_step=args.step)
    if args.json:
        print(json.dumps(diff, indent=2))
    else:
        print(f"Diff {args.timeline_a} vs {args.timeline_b} @ step {args.step}:")
        print(f"  Only in {args.timeline_a}: {diff['only_in_a']}")
        print(f"  Only in {args.timeline_b}: {diff['only_in_b']}")
        print(f"  Changed ({len(diff['changed'])}):")
        for item in diff["changed"]:
            print(f"    - {item['subject_key']}: {item['timeline_a']} vs {item['timeline_b']}")
        print(f"  Identical keys: {len(diff['identical'])}")
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    with NarrativeMemoryEngine(args.db_path, timeline_id=args.timeline, on_embedder_mismatch="ignore") as engine:
        hist = engine.get_history(args.subject_key, timeline_id=args.timeline, record_type=args.type)
    if args.json:
        print(json.dumps(hist, indent=2))
    else:
        if not hist:
            print(f"No revisions found for '{args.subject_key}' on '{args.timeline}'.")
            return 0
        for item in hist:
            end_step = item["valid_to_step"] if item["valid_to_step"] is not None else "HEAD"
            tomb = " [TOMBSTONE]" if item["is_tombstone"] else ""
            print(
                f"[{item['valid_from_step']}..{end_step}] ({item['timeline_id']} / {item['record_type']}){tomb}: "
                f"{json.dumps(item['payload'], ensure_ascii=False)}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narrative-memory",
        description="Inspect and diff narrative-memory SQLite timeline databases.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tree = sub.add_parser("tree", help="Render the branching timeline tree.")
    p_tree.add_argument("db_path", help="Path to SQLite database file.")
    p_tree.add_argument("--json", action="store_true", help="Output JSON.")
    p_tree.set_defaults(func=_cmd_tree)

    p_inspect = sub.add_parser("inspect", help="Inspect world state at a timeline step.")
    p_inspect.add_argument("db_path", help="Path to SQLite database file.")
    p_inspect.add_argument("--timeline", default="main", help="Timeline ID (default: main).")
    p_inspect.add_argument("--step", type=int, default=None, help="Step to inspect (default: head).")
    p_inspect.add_argument("--inclusive", action="store_true", help="Include changes at --step.")
    p_inspect.add_argument("--pov", action="append", default=[], help="POV entity_id (repeatable).")
    p_inspect.add_argument("--text", default="", help="Optional scene cue text.")
    p_inspect.add_argument("--json", action="store_true", help="Output JSON.")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_diff = sub.add_parser("diff", help="Diff active world state between two timelines.")
    p_diff.add_argument("db_path", help="Path to SQLite database file.")
    p_diff.add_argument("timeline_a", help="First timeline ID.")
    p_diff.add_argument("timeline_b", help="Second timeline ID.")
    p_diff.add_argument("--step", type=int, required=True, help="Step at which to compare.")
    p_diff.add_argument("--json", action="store_true", help="Output JSON.")
    p_diff.set_defaults(func=_cmd_diff)

    p_hist = sub.add_parser("history", help="Print revision trajectory for a subject_key.")
    p_hist.add_argument("db_path", help="Path to SQLite database file.")
    p_hist.add_argument("subject_key", help="Subject key (e.g. char_arthur).")
    p_hist.add_argument("--timeline", default="main", help="Timeline ID (default: main).")
    p_hist.add_argument("--type", default=None, help="Record type filter (entity|edge|alias).")
    p_hist.add_argument("--json", action="store_true", help="Output JSON.")
    p_hist.set_defaults(func=_cmd_history)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    func: Any = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
