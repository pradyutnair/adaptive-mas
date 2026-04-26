#!/usr/bin/env python3
"""Combine shard predictions, preserve question order, and write eval artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eval_offline import evaluate


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                rows[str(obj.get("id", ""))] = obj
    return rows


def _question_wallclock_stats(rows: dict[str, dict]) -> dict[str, float]:
    values = []
    for row in rows.values():
        meta = row.get("metadata", {}) or {}
        values.append(float(meta.get("wallclock_seconds", 0.0)))
    values.sort()
    if not values:
        return {
            "mean_question_wallclock_seconds": 0.0,
            "p50_question_wallclock_seconds": 0.0,
            "p95_question_wallclock_seconds": 0.0,
        }
    return {
        "mean_question_wallclock_seconds": round(sum(values) / len(values), 3),
        "p50_question_wallclock_seconds": round(values[len(values) // 2], 3),
        "p95_question_wallclock_seconds": round(
            values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
            3,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine shard outputs and evaluate.")
    parser.add_argument("--questions", required=True, help="Combined questions JSON")
    parser.add_argument("--output-prefix", required=True, help="Combined output prefix")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard predictions.jsonl")
    parser.add_argument(
        "--run-summaries",
        nargs="*",
        default=[],
        help="Optional shard run_summary.json files",
    )
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    gold_by_id = {str(q.get("id", "")): q for q in questions}

    combined: dict[str, dict] = {}
    for input_path in args.inputs:
        combined.update(_load_jsonl(Path(input_path)))

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    combined_path = output_prefix.with_name(output_prefix.name + "_combined.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for question in questions:
            qid = str(question.get("id", ""))
            if qid in combined:
                f.write(json.dumps(combined[qid], ensure_ascii=False) + "\n")

    eval_path = output_prefix.with_name(output_prefix.name + "_eval.json")
    evaluate(str(combined_path), args.questions, str(eval_path))

    if args.run_summaries:
        summaries = []
        for summary_path in args.run_summaries:
            path = Path(summary_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
        if summaries:
            total_rows = sum(int(s.get("total_output_rows", 0)) for s in summaries)
            wallclock_seconds = max(float(s.get("elapsed_seconds", 0.0)) for s in summaries)
            q_stats = _question_wallclock_stats(combined)
            merged = {
                "attempted": sum(int(s.get("attempted", 0)) for s in summaries),
                "succeeded": sum(int(s.get("succeeded", 0)) for s in summaries),
                "failed": sum(int(s.get("failed", 0)) for s in summaries),
                "total_output_rows": total_rows,
                "elapsed_seconds": round(wallclock_seconds, 3),
                "mean_wallclock_seconds": round(wallclock_seconds / total_rows, 3)
                if total_rows
                else 0.0,
                **q_stats,
                "questions_path": args.questions,
                "inputs": args.inputs,
            }
            with open(
                output_prefix.with_name(output_prefix.name + "_run_summary.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
