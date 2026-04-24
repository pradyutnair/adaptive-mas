#!/usr/bin/env python3
"""Compare multiple prediction files on an exact question subset."""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_offline import contain, norm_em, token_f1  # noqa: E402


def _get_first(row: dict, *paths: tuple[str, ...]) -> object | None:
    """Return the first field found across alternate schemas."""
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


def _first_numeric(row: dict, *paths: tuple[str, ...]) -> float | None:
    """Return the first numeric field found across alternate schemas."""
    value = _get_first(row, *paths)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _step_trace(row: dict) -> list[dict]:
    trace = _get_first(row, ("metadata", "step_trace"), ("step_trace",))
    return trace if isinstance(trace, list) else []


def _route_name(row: dict) -> str:
    route = _get_first(
        row,
        ("metadata", "route_decision"),
        ("route_decision",),
        ("metadata", "route"),
    )
    return str(route or "missing").strip() or "missing"


def _planned_hops(row: dict) -> int:
    value = _get_first(
        row,
        ("metadata", "planned_hop_count"),
        ("metadata", "expected_hop_count"),
        ("metadata", "plan_steps"),
        ("plan_steps",),
    )
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    hops = _get_first(row, ("metadata", "route_required_hops"))
    return len(hops) if isinstance(hops, list) else 0


def _step_counts(row: dict) -> dict[str, int]:
    counts = {"spawn": 0, "refine": 0, "answer": 0, "assess": 0}
    for step in _step_trace(row):
        action = str(step.get("action", "")).strip()
        if action in counts:
            counts[action] += 1
    return counts
    return None


def _load_questions(path: Path) -> tuple[list[str], dict[str, str]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(q["id"]) for q in questions]
    gold = {str(q["id"]): str(q.get("answer", "")) for q in questions}
    return ids, gold


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[str(obj.get("id", ""))] = obj
    return rows


def _aggregate(rows: dict[str, dict], ids: list[str], gold_by_id: dict[str, str]) -> dict:
    contains: list[float] = []
    f1s: list[float] = []
    ems: list[float] = []
    toks: list[float] = []
    prompt_toks: list[float] = []
    completion_toks: list[float] = []
    walls: list[float] = []
    calls: list[float] = []
    planned_hops: list[int] = []
    routes: dict[str, int] = {}
    step_totals = {"spawn": 0, "refine": 0, "answer": 0, "assess": 0}

    for qid in ids:
        row = rows.get(qid)
        if row is None:
            continue
        gold = gold_by_id.get(qid, "")
        answer = str(row.get("answer", ""))
        contains.append(contain(answer, gold))
        f1s.append(token_f1(answer, gold))
        ems.append(norm_em(answer, gold))
        token_value = _first_numeric(
            row,
            ("metadata", "total_tokens"),
            ("total_tokens",),
        )
        wall_value = _first_numeric(
            row,
            ("metadata", "wallclock_seconds"),
            ("wallclock_seconds",),
        )
        prompt_value = _first_numeric(
            row,
            ("metadata", "prompt_tokens"),
            ("prompt_tokens",),
        )
        completion_value = _first_numeric(
            row,
            ("metadata", "completion_tokens"),
            ("completion_tokens",),
        )
        call_value = _first_numeric(
            row,
            ("metadata", "num_subagent_calls"),
            ("metadata", "llm_call_count"),
            ("llm_call_count",),
        )
        if token_value is not None:
            toks.append(token_value)
        if prompt_value is not None:
            prompt_toks.append(prompt_value)
        if completion_value is not None:
            completion_toks.append(completion_value)
        if wall_value is not None:
            walls.append(wall_value)
        if call_value is not None:
            calls.append(call_value)
        hops = _planned_hops(row)
        if hops:
            planned_hops.append(hops)
        route = _route_name(row)
        routes[route] = routes.get(route, 0) + 1
        counts = _step_counts(row)
        for key, value in counts.items():
            step_totals[key] += value

    n = len(contains)
    coverage = round(n / len(ids), 4) if ids else 0.0
    return {
        "n": n,
        "contain": round(sum(contains) / n, 4) if n else 0.0,
        "token_f1": round(sum(f1s) / n, 4) if n else 0.0,
        "norm_em": round(sum(ems) / n, 4) if n else 0.0,
        "mean_total_tokens": round(mean(toks), 1) if toks else 0.0,
        "mean_prompt_tokens": round(mean(prompt_toks), 1) if prompt_toks else 0.0,
        "mean_completion_tokens": round(mean(completion_toks), 1)
        if completion_toks
        else 0.0,
        "mean_wallclock_seconds": round(mean(walls), 1) if walls else 0.0,
        "mean_subagent_calls": round(mean(calls), 3) if calls else 0.0,
        "mean_planned_hops": round(mean(planned_hops), 3) if planned_hops else 0.0,
        "coverage": coverage,
        "status": "complete" if coverage >= 1.0 else "partial",
        "route_counts": routes,
        "step_totals": step_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Subset questions JSON")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec in the form label=predictions.jsonl",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    if not args.run:
        raise SystemExit("At least one --run label=path is required.")

    ids, gold_by_id = _load_questions(Path(args.questions))
    summary: dict[str, dict] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run spec: {spec}")
        label, raw_path = spec.split("=", 1)
        rows = _load_jsonl(Path(raw_path))
        summary[label] = _aggregate(rows, ids, gold_by_id)

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"{'run':<18}{'n':>6}{'cover':>8}{'contain':>10}{'tok_f1':>10}"
        f"{'em':>8}{'tokens':>12}{'wall':>10}{'calls':>9}"
    )
    for label, stats in summary.items():
        print(
            f"{label:<18}{stats['n']:>6}{stats['coverage']:>8.3f}"
            f"{stats['contain']:>10.4f}{stats['token_f1']:>10.4f}"
            f"{stats['norm_em']:>8.4f}{stats['mean_total_tokens']:>12.1f}"
            f"{stats['mean_wallclock_seconds']:>10.1f}{stats['mean_subagent_calls']:>9.3f}"
        )
        print(f"  routes: {json.dumps(stats['route_counts'], sort_keys=True)}")


if __name__ == "__main__":
    main()
