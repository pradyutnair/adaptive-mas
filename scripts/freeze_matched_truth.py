#!/usr/bin/env python3
"""Freeze matched-ID baseline tables across mixed result schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from compare_subset_runs import _load_jsonl, _load_questions  # noqa: E402
from eval_offline import contain, norm_em, token_f1  # noqa: E402


def _get_first(row: dict, *paths: tuple[str, ...]) -> float | int | str | None:
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


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _step_trace(row: dict) -> list[dict]:
    trace = _get_first(row, ("metadata", "step_trace"), ("step_trace",))
    return trace if isinstance(trace, list) else []


def _route_name(row: dict) -> str:
    value = _get_first(
        row,
        ("metadata", "route_decision"),
        ("route_decision",),
        ("metadata", "route"),
    )
    return str(value or "missing").strip() or "missing"


def _planned_hops(row: dict) -> int:
    value = _get_first(
        row,
        ("metadata", "planned_hop_count"),
        ("metadata", "expected_hop_count"),
        ("metadata", "plan_steps"),
        ("plan_steps",),
    )
    if value is None:
        required_hops = _get_first(row, ("metadata", "route_required_hops"))
        if isinstance(required_hops, list):
            return len(required_hops)
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _subagent_calls(row: dict) -> float | None:
    value = _get_first(
        row,
        ("metadata", "num_subagent_calls"),
        ("metadata", "llm_call_count"),
        ("num_subagent_calls",),
        ("llm_call_count",),
    )
    return _as_float(value)


def _step_counts(row: dict) -> dict[str, int]:
    trace = _step_trace(row)
    counts = {"spawn": 0, "refine": 0, "answer": 0, "assess": 0}
    for step in trace:
        action = str(step.get("action", "")).strip()
        if action in counts:
            counts[action] += 1
    return counts


def _aggregate(rows: dict[str, dict], ids: list[str], gold_by_id: dict[str, str]) -> dict:
    contains: list[float] = []
    f1s: list[float] = []
    ems: list[float] = []
    total_tokens: list[float] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    wallclock: list[float] = []
    llm_calls: list[float] = []
    planned_hops: list[int] = []
    route_counts: dict[str, int] = {}
    step_totals = {"spawn": 0, "refine": 0, "answer": 0, "assess": 0}

    for qid in ids:
        row = rows.get(qid)
        if row is None:
            continue
        gold = gold_by_id.get(qid, "")
        answer = str(_get_first(row, ("answer",), ("prediction",)) or "")
        contains.append(contain(answer, gold))
        f1s.append(token_f1(answer, gold))
        ems.append(norm_em(answer, gold))

        for store, raw in (
            (total_tokens, _get_first(row, ("metadata", "total_tokens"), ("total_tokens",))),
            (prompt_tokens, _get_first(row, ("metadata", "prompt_tokens"), ("prompt_tokens",))),
            (completion_tokens, _get_first(row, ("metadata", "completion_tokens"), ("completion_tokens",))),
            (wallclock, _get_first(row, ("metadata", "wallclock_seconds"), ("wallclock_seconds",))),
        ):
            value = _as_float(raw)
            if value is not None:
                store.append(value)

        call_value = _subagent_calls(row)
        if call_value is not None:
            llm_calls.append(call_value)

        hops = _planned_hops(row)
        if hops:
            planned_hops.append(hops)

        route = _route_name(row)
        route_counts[route] = route_counts.get(route, 0) + 1

        counts = _step_counts(row)
        for key, value in counts.items():
            step_totals[key] += value

    n = len(contains)
    return {
        "n": n,
        "coverage": round(n / len(ids), 4) if ids else 0.0,
        "contain": round(sum(contains) / n, 4) if n else 0.0,
        "token_f1": round(sum(f1s) / n, 4) if n else 0.0,
        "norm_em": round(sum(ems) / n, 4) if n else 0.0,
        "mean_total_tokens": round(mean(total_tokens), 1) if total_tokens else 0.0,
        "mean_prompt_tokens": round(mean(prompt_tokens), 1) if prompt_tokens else 0.0,
        "mean_completion_tokens": round(mean(completion_tokens), 1)
        if completion_tokens
        else 0.0,
        "mean_wallclock_seconds": round(mean(wallclock), 1) if wallclock else 0.0,
        "mean_llm_calls": round(mean(llm_calls), 3) if llm_calls else 0.0,
        "mean_planned_hops": round(mean(planned_hops), 3) if planned_hops else 0.0,
        "route_counts": dict(sorted(route_counts.items())),
        "step_totals": step_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Subset questions JSON.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec label=predictions.jsonl",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    if not args.run:
        raise SystemExit("At least one --run label=path is required.")

    ids, gold_by_id = _load_questions(Path(args.questions))
    summary: dict[str, dict] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run spec: {spec}")
        label, raw_path = spec.split("=", 1)
        summary[label] = _aggregate(_load_jsonl(Path(raw_path)), ids, gold_by_id)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
