#!/usr/bin/env python3
"""Compute a random-routing ablation between two prediction sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                rows[str(obj.get("id", ""))] = obj
    return rows


def _route_mix(path: Path) -> float:
    recurse = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {}) or {}
            decision = str(meta.get("route_decision", "")).strip()
            if not decision:
                trace = meta.get("step_trace", []) or []
                if trace:
                    decision = str(trace[0].get("route_decision", "")).strip()
            if not decision:
                continue
            total += 1
            recurse += int(decision == "recurse")
    return (recurse / total) if total else 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute random routing ablation.")
    parser.add_argument("--base", required=True, help="Base predictions.jsonl")
    parser.add_argument("--alt", required=True, help="Alt predictions.jsonl")
    parser.add_argument("--output", required=True, help="Output predictions.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--alt-prob",
        type=float,
        default=None,
        help="Probability of choosing alt rows; defaults to learned recurse rate",
    )
    parser.add_argument(
        "--match-variant",
        help="Predictions.jsonl whose recurse rate should define alt probability",
    )
    args = parser.parse_args()

    base = _load_predictions(Path(args.base))
    alt = _load_predictions(Path(args.alt))
    qids = sorted(set(base) & set(alt))
    alt_prob = args.alt_prob
    if alt_prob is None and args.match_variant:
        alt_prob = _route_mix(Path(args.match_variant))
    if alt_prob is None:
        alt_prob = 0.5

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in qids:
            choose_alt = rng.random() < alt_prob
            row = dict(alt[qid] if choose_alt else base[qid])
            row.setdefault("metadata", {})
            row["metadata"]["random_routing_choice"] = "alt" if choose_alt else "base"
            row["metadata"]["random_routing_alt_prob"] = round(float(alt_prob), 6)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
