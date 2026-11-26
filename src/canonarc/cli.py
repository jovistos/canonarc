from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bible import infer_export_format
from .engine import CanonArcEngine
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
    with CanonArcEngine(args.db_path, on_embedder_mismatch="ignore") as engine:
        timelines = engine.list_timelines()
    if args.json:
        print(json.dumps([t.model_dump() for t in timelines], indent=2))
    else:
        print(_render_tree(timelines))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    with CanonArcEngine(args.db_path, timeline_id=args.timeline) as engine:
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
    with CanonArcEngine(args.db_path, on_embedder_mismatch="ignore") as engine:
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
    with CanonArcEngine(args.db_path, timeline_id=args.timeline, on_embedder_mismatch="ignore") as engine:
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


def _cmd_export(args: argparse.Namespace) -> int:
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()] if args.kinds else None
    with CanonArcEngine(args.db_path, timeline_id=args.timeline, on_embedder_mismatch="ignore") as engine:
        if args.output_path == "-":
            fmt = infer_export_format(None, args.format)
            content, _ = engine.render_bible(
                format=fmt,
                at_step=args.step,
                timeline_id=args.timeline,
                kinds=kinds,
                include_state=bool(args.include_state),
                include_edges=not bool(args.no_edges),
                include_aliases=not bool(args.no_aliases),
                include_inflections=args.include_inflections,
                lang=args.lang,
            )
            sys.stdout.write(content)
            return 0
        result = engine.export_bible(
            args.output_path,
            format=args.format,
            at_step=args.step,
            timeline_id=args.timeline,
            kinds=kinds,
            include_state=bool(args.include_state),
            include_edges=not bool(args.no_edges),
            include_aliases=not bool(args.no_aliases),
            include_inflections=args.include_inflections,
            lang=args.lang,
            excel_bom=bool(args.excel_bom),
        )
    if args.json:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(
            f"Exported {result.entity_count} entities, {result.alias_count} aliases, "
            f"{result.edge_count} edges ({result.format}) to {result.path} "
            f"[timeline={result.timeline_id}, step={result.at_step}]."
        )
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    col_map: dict[str, str] | None = None
    if args.column_map:
        col_map = {}
        for pair in args.column_map:
            if "=" not in pair:
                print(
                    f"Error: --column-map expects 'SRC=DST' format, got {pair!r}.",
                    file=sys.stderr,
                )
                return 2
            src_col, dst_col = pair.split("=", 1)
            col_map[src_col.strip()] = dst_col.strip()
    with CanonArcEngine(args.db_path, timeline_id=args.timeline, on_embedder_mismatch="ignore") as engine:
        report = engine.sync_bible(
            args.input_path,
            format=args.format,
            step=args.step,
            timeline_id=args.timeline,
            mode=args.mode,
            target_lang=args.target_lang,
            auto_inflect=False,
            dry_run=bool(args.dry_run),
            column_map=col_map,
        )
    if args.json:
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0


def _load_pairs_jsonl(path: str) -> dict[int, tuple[str, str]]:
    pairs: dict[int, tuple[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if "step" not in obj:
                raise ValueError(f"{path}:{line_no}: missing required 'step' field.")
            step = int(obj["step"])
            src = str(obj.get("source", obj.get("source_text", "")) or "")
            tgt = str(obj.get("target", obj.get("target_text", "")) or "")
            pairs[step] = (src, tgt)
    return pairs


def _cmd_audit(args: argparse.Namespace) -> int:
    include_rules = (
        [r.strip() for r in args.rules.split(",") if r.strip()] if args.rules else None
    )
    exclude_rules = (
        [r.strip() for r in args.exclude_rules.split(",") if r.strip()]
        if args.exclude_rules
        else None
    )
    with CanonArcEngine(
        args.db_path, timeline_id=args.timeline, on_embedder_mismatch="ignore"
    ) as engine:
        tl_info = engine.store.get_timeline(args.timeline)
        if tl_info is None:
            print(f"Error: timeline '{args.timeline}' not found.", file=sys.stderr)
            return 1

        if args.step is not None:
            source_text = (
                Path(args.source).read_text(encoding="utf-8") if args.source else ""
            )
            target_text = (
                Path(args.target).read_text(encoding="utf-8") if args.target else ""
            )
            report = engine.audit_step(
                args.step,
                source_text=source_text,
                target_text=target_text,
                timeline_id=args.timeline,
                include_rules=include_rules,
                exclude_rules=exclude_rules,
                min_severity=args.min_severity,
                lang=args.lang,
            )
        else:
            texts: dict[int, tuple[str, str]] | None = None
            if args.pairs_jsonl:
                try:
                    texts = _load_pairs_jsonl(args.pairs_jsonl)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    print(f"Error reading --pairs-jsonl: {exc}", file=sys.stderr)
                    return 2
            report = engine.audit_timeline(
                timeline_id=args.timeline,
                from_step=args.from_step,
                to_step=args.to_step,
                texts=texts,
                include_rules=include_rules,
                exclude_rules=exclude_rules,
                min_severity=args.min_severity,
                lang=args.lang,
            )

    if args.json:
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())

    if args.fail_on != "none" and report.has_issues_at_or_above(args.fail_on):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canonarc",
        description="Inspect and diff canonarc SQLite timeline databases.",
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

    p_export = sub.add_parser("export", help="Export Series Bible to CSV, JSON, or Markdown.")
    p_export.add_argument("db_path", help="Path to SQLite database file.")
    p_export.add_argument("output_path", help="Output file path (or '-' for stdout).")
    p_export.add_argument(
        "--format",
        default="auto",
        choices=["auto", "csv", "json", "md"],
        help="Output format (default: auto-detect from extension).",
    )
    p_export.add_argument("--timeline", default="main", help="Timeline ID (default: main).")
    p_export.add_argument("--step", type=int, default=None, help="Step to export (default: timeline head).")
    p_export.add_argument("--kinds", default=None, help="Comma-separated entity kind filter (e.g. character,term).")
    p_export.add_argument("--lang", default=None, help="Filter aliases by ISO/BCP-47 language tag.")
    p_export.add_argument("--include-state", action="store_true", help="Include Tier 3 rolling state.")
    p_export.add_argument("--no-edges", action="store_true", help="Exclude directed edges (JSON/MD).")
    p_export.add_argument("--no-aliases", action="store_true", help="Exclude aliases.")
    inflect_grp = p_export.add_mutually_exclusive_group()
    inflect_grp.add_argument(
        "--include-inflections",
        dest="include_inflections",
        action="store_true",
        default=None,
        help="Include kind='inflection' aliases (default: true for JSON, false for CSV/MD).",
    )
    inflect_grp.add_argument(
        "--exclude-inflections",
        dest="include_inflections",
        action="store_false",
        help="Exclude kind='inflection' aliases even in JSON exports.",
    )
    p_export.add_argument("--excel-bom", action="store_true", help="Write UTF-8 BOM for Excel CSV compatibility.")
    p_export.add_argument("--json", action="store_true", help="Output export metadata as JSON.")
    p_export.set_defaults(func=_cmd_export)

    p_import = sub.add_parser("import", help="Sync a CSV or JSON Series Bible into the database.")
    p_import.add_argument("db_path", help="Path to SQLite database file.")
    p_import.add_argument("input_path", help="Path to CSV or JSON Bible file.")
    p_import.add_argument(
        "--format",
        default="auto",
        choices=["auto", "csv", "json"],
        help="Input format (default: auto-detect from extension).",
    )
    p_import.add_argument("--timeline", default="main", help="Timeline ID (default: main).")
    p_import.add_argument("--step", type=int, default=0, help="Causal step at which to apply changes (default: 0).")
    p_import.add_argument(
        "--mode",
        default="upsert",
        choices=["upsert", "merge", "replace"],
        help="Sync mode: upsert (default), merge (non-destructive), or replace (strict mirror).",
    )
    p_import.add_argument("--target-lang", default=None, help="Default target language tag for target_name aliases.")
    p_import.add_argument(
        "--column-map",
        action="append",
        default=[],
        help="Map CSV header to canonical column in 'SRC=DST' format (repeatable).",
    )
    p_import.add_argument("--dry-run", action="store_true", help="Preview diff without modifying the database.")
    p_import.add_argument("--json", action="store_true", help="Output sync report as JSON.")
    p_import.set_defaults(func=_cmd_import)

    p_audit = sub.add_parser("audit", help="Run continuity and localization QA audit rules.")
    p_audit.add_argument("db_path", help="Path to SQLite database file.")
    p_audit.add_argument("--timeline", default="main", help="Timeline ID (default: main).")
    p_audit.add_argument("--step", type=int, default=None, help="Audit a single step (omit to audit timeline range).")
    p_audit.add_argument("--from-step", type=int, default=0, help="Start step for timeline audit (default: 0).")
    p_audit.add_argument("--to-step", type=int, default=None, help="End step for timeline audit (default: head).")
    p_audit.add_argument("--source", default=None, help="Source text file path (used with --step).")
    p_audit.add_argument("--target", default=None, help="Target text file path (used with --step).")
    p_audit.add_argument(
        "--pairs-jsonl",
        default=None,
        help="JSONL file with {step, source, target} records for multi-step text auditing.",
    )
    p_audit.add_argument("--rules", default=None, help="Comma-separated rule names to include.")
    p_audit.add_argument("--exclude-rules", default=None, help="Comma-separated rule names to exclude.")
    p_audit.add_argument(
        "--min-severity",
        default="info",
        choices=["info", "warning", "error"],
        help="Minimum issue severity to report (default: info).",
    )
    p_audit.add_argument(
        "--fail-on",
        default="error",
        choices=["error", "warning", "none"],
        help="Exit non-zero if issues at or above this severity exist (default: error).",
    )
    p_audit.add_argument("--lang", default=None, help="Optional target language tag.")
    p_audit.add_argument("--json", action="store_true", help="Output audit report as JSON.")
    p_audit.set_defaults(func=_cmd_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    func: Any = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
