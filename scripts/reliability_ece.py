"""Reliability diagram and ECE for the sufficiency score.

Reads a predictions.jsonl produced by the sufficiency controller (where each
row's ``metadata.step_trace`` contains an ``assess`` step with a
``sufficiency`` value) and a gold questions.json. For each question:

- x = sufficiency s computed at probe time.
- y = contain-correctness of the final answer.

Reports per-bin (count, mean_s, accuracy) and the Expected Calibration Error.
Optionally writes a CSV for plotting.

Usage:

    python3 scripts/reliability_ece.py \\
        --predictions results/<run>/predictions.jsonl \\
        --questions data/hotpotqa/questions_1000_seed42.json \\
        --output results/<run>/reliability.json \\
        --bins 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_offline import contain  # noqa: E402


def _extract_sufficiency(row: dict) -> float | None:
    metadata = row.get("metadata") or {}
    trace = metadata.get("step_trace") or []
    for entry in trace:
        if entry.get("action") == "assess":
            entry_metadata = entry.get("metadata") or {}
            value = entry_metadata.get("sufficiency")
            if value is None:
                value = entry.get("justification_confidence")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--csv", default="", help="Optional CSV path for plotting")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as handle:
        gold = {str(q.get("id", "")).strip(): str(q.get("answer", "")) for q in json.load(handle)}

    points: list[tuple[float, float]] = []
    with open(args.predictions, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("id", "")).strip()
            sufficiency = _extract_sufficiency(row)
            if sufficiency is None or qid not in gold:
                continue
            correct = contain(str(row.get("answer", "")), gold[qid])
            points.append((sufficiency, correct))

    if not points:
        raise SystemExit("No (sufficiency, correctness) pairs found in predictions.")

    bins = [[] for _ in range(args.bins)]
    for s, y in points:
        idx = min(int(s * args.bins), args.bins - 1)
        bins[idx].append((s, y))

    n = len(points)
    ece = 0.0
    bin_rows: list[dict] = []
    for i, members in enumerate(bins):
        if not members:
            bin_rows.append(
                {
                    "bin": i,
                    "lo": i / args.bins,
                    "hi": (i + 1) / args.bins,
                    "count": 0,
                    "mean_sufficiency": None,
                    "accuracy": None,
                }
            )
            continue
        mean_s = sum(s for s, _ in members) / len(members)
        acc = sum(y for _, y in members) / len(members)
        ece += (len(members) / n) * abs(acc - mean_s)
        bin_rows.append(
            {
                "bin": i,
                "lo": i / args.bins,
                "hi": (i + 1) / args.bins,
                "count": len(members),
                "mean_sufficiency": mean_s,
                "accuracy": acc,
            }
        )

    summary = {
        "n": n,
        "bins": args.bins,
        "ece": ece,
        "bin_table": bin_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["bin", "lo", "hi", "count", "mean_sufficiency", "accuracy"],
            )
            writer.writeheader()
            writer.writerows(bin_rows)

    print(json.dumps({"n": n, "ece": ece}, indent=2))


if __name__ == "__main__":
    main()
