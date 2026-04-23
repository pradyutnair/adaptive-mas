#!/usr/bin/env python3
"""Summarise where sufficiency-controller tokens are actually spent."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary(predictions_path: Path, oracle_path: Path | None = None) -> dict:
    rows = _load_rows(predictions_path)
    route_counts: Counter[str] = Counter()
    route_tokens: defaultdict[str, list[float]] = defaultdict(list)
    action_tokens: defaultdict[str, list[float]] = defaultdict(list)
    total_tokens: list[float] = []

    recurse_route = "recurse_after_probe"
    recurse_slice_tokens: list[float] = []

    for row in rows:
        metadata = row.get("metadata", {})
        route = str(metadata.get("route_decision", "")).strip() or "unknown"
        route_counts[route] += 1
        total = float(metadata.get("total_tokens", 0.0))
        total_tokens.append(total)
        route_tokens[route].append(total)
        if route == recurse_route:
            recurse_slice_tokens.append(total)

        for step in metadata.get("step_trace", []):
            action = str(step.get("action", "")).strip() or "unknown"
            action_tokens[action].append(float(step.get("tokens", 0.0)))

    summary = {
        "n": len(rows),
        "mean_total_tokens": round(_mean(total_tokens), 1),
        "route_counts": dict(route_counts),
        "route_mean_tokens": {
            route: round(_mean(vals), 1) for route, vals in route_tokens.items()
        },
        "assess_overhead_per_question": round(_mean(action_tokens.get("assess", [])), 1),
        "assess_share_total_pct": round(
            100.0 * sum(action_tokens.get("assess", [])) / max(sum(total_tokens), 1.0),
            1,
        ),
        "spawn_overhead_per_question": round(_mean(action_tokens.get("spawn", [])), 1),
        "spawn_share_total_pct": round(
            100.0 * sum(action_tokens.get("spawn", [])) / max(sum(total_tokens), 1.0),
            1,
        ),
        "recurse_slice_n": len(recurse_slice_tokens),
        "recurse_slice_mean_tokens": round(_mean(recurse_slice_tokens), 1),
    }

    if oracle_path and oracle_path.exists():
        summary["oracle_probe_upper_bound"] = json.loads(
            oracle_path.read_text(encoding="utf-8")
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument(
        "--oracle",
        default="",
        help="Optional oracle_probe_upper_bound.json path",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path; defaults to stdout",
    )
    args = parser.parse_args()

    summary = build_summary(
        Path(args.predictions),
        Path(args.oracle) if args.oracle else None,
    )
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        return
    print(payload)


if __name__ == "__main__":
    main()
