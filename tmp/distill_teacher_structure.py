#!/usr/bin/env python3
"""Extract structural teacher stats from matched-ID runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from compare_subset_runs import _load_jsonl, _load_questions  # noqa: E402
from eval_offline import contain  # noqa: E402


def _get_first(row: dict, *paths: tuple[str, ...]) -> object | None:
    for path in paths:
        cur = row
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            return cur
    return None


def _trace(row: dict) -> list[dict]:
    value = _get_first(row, ("metadata", "step_trace"), ("step_trace",))
    return value if isinstance(value, list) else []


def _intish(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _floatish(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _slot_count(row: dict) -> int:
    value = _get_first(
        row,
        ("metadata", "slot_count"),
        ("metadata", "planned_hop_count"),
        ("metadata", "expected_hop_count"),
        ("plan_steps",),
    )
    if value is not None:
        return _intish(value, 0)
    hops = _get_first(row, ("metadata", "route_required_hops"))
    return len(hops) if isinstance(hops, list) else 0


def _rewrite_count(row: dict) -> int:
    value = _get_first(row, ("metadata", "num_rewrites"), ("num_rewrites",))
    if value is not None:
        return _intish(value, 0)
    return sum(1 for step in _trace(row) if str(step.get("action", "")).strip() == "refine")


def _execution_depth(row: dict) -> int:
    value = _get_first(
        row,
        ("metadata", "num_plan_exec_steps"),
        ("metadata", "plan_steps"),
        ("plan_steps",),
    )
    if value is not None:
        return _intish(value, 0)
    trace = _trace(row)
    return sum(
        1
        for step in trace
        if str(step.get("action", "")).strip() in {"spawn", "refine"}
    )


def _question_type(row: dict) -> str:
    value = _get_first(
        row,
        ("question_type",),
        ("metadata", "question_type"),
        ("type",),
    )
    return str(value or "unknown").strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Subset questions JSON.")
    parser.add_argument("--run", required=True, help="Predictions JSONL.")
    parser.add_argument("--label", default="teacher", help="Run label.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    ids, gold_by_id = _load_questions(Path(args.questions))
    rows = _load_jsonl(Path(args.run))

    slot_counts: list[int] = []
    rewrite_counts: list[int] = []
    depths: list[int] = []
    step_costs: list[float] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    failures_by_type: dict[str, dict[str, float]] = {}

    for qid in ids:
        row = rows.get(qid)
        if row is None:
            continue
        gold = gold_by_id.get(qid, "")
        answer = str(row.get("answer", row.get("prediction", "")))
        ok = contain(answer, gold)
        qtype = _question_type(row)
        bucket = failures_by_type.setdefault(qtype, {"n": 0, "contain": 0.0})
        bucket["n"] += 1
        bucket["contain"] += ok

        slot_count = _slot_count(row)
        rewrite_count = _rewrite_count(row)
        depth = _execution_depth(row)
        total_tokens = _floatish(
            _get_first(row, ("metadata", "total_tokens"), ("total_tokens",)),
            0.0,
        )
        prompt_value = _floatish(
            _get_first(row, ("metadata", "prompt_tokens"), ("prompt_tokens",)),
            0.0,
        )
        completion_value = _floatish(
            _get_first(row, ("metadata", "completion_tokens"), ("completion_tokens",)),
            0.0,
        )

        slot_counts.append(slot_count)
        rewrite_counts.append(rewrite_count)
        depths.append(depth)
        if prompt_value > 0:
            prompt_tokens.append(prompt_value)
        if completion_value > 0:
            completion_tokens.append(completion_value)
        if depth > 0 and total_tokens > 0:
            step_costs.append(total_tokens / depth)

    coverage = round(len(depths) / len(ids), 4) if ids else 0.0
    summary = {
        args.label: {
            "n": len(depths),
            "coverage": coverage,
            "status": "complete" if coverage >= 1.0 else "partial",
            "mean_slot_count": round(mean(slot_counts), 3) if slot_counts else 0.0,
            "slot_count_histogram": {
                str(value): slot_counts.count(value) for value in sorted(set(slot_counts))
            },
            "mean_rewrite_count": round(mean(rewrite_counts), 3)
            if rewrite_counts
            else 0.0,
            "rewrite_rate": round(
                sum(1 for value in rewrite_counts if value > 0) / len(rewrite_counts),
                4,
            )
            if rewrite_counts
            else 0.0,
            "mean_execution_depth": round(mean(depths), 3) if depths else 0.0,
            "depth_histogram": {
                str(value): depths.count(value) for value in sorted(set(depths))
            },
            "mean_step_token_cost": round(mean(step_costs), 1) if step_costs else 0.0,
            "mean_prompt_tokens": round(mean(prompt_tokens), 1) if prompt_tokens else 0.0,
            "mean_completion_tokens": round(mean(completion_tokens), 1)
            if completion_tokens
            else 0.0,
            "failure_patterns": {
                key: {
                    "n": int(value["n"]),
                    "contain": round(value["contain"] / value["n"], 4)
                    if value["n"]
                    else 0.0,
                }
                for key, value in sorted(failures_by_type.items())
            },
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
