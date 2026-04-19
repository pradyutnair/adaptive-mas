"""Per-slice decomposition for the sufficiency-controlled run.

Splits the questions into:
- ``probe_sufficient``:  s >= tau   (answered from the probe).
- ``probe_insufficient``: s <  tau   (escalated to the recursive lane).

Reports per-slice contain, norm_em, token_f1, mean tokens, and slice size,
plus the overall numbers. This directly substantiates the "easy questions
stop early, hard questions get the budget" claim without invoking any
dataset name.

Usage:

    python3 scripts/slice_decomposition.py \\
        --predictions results/<m1_2_sufficiency>/predictions.jsonl \\
        --questions data/hotpotqa/questions_1000_seed42.json \\
        --tau 0.70 \\
        --output results/<m1_2_sufficiency>/slice_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_offline import contain, norm_em, token_f1  # noqa: E402


def _extract_sufficiency(row: dict) -> float | None:
    metadata = row.get("metadata") or {}
    trace = metadata.get("step_trace") or []
    for entry in trace:
        if entry.get("action") == "assess":
            entry_meta = entry.get("metadata") or {}
            value = entry_meta.get("sufficiency")
            if value is None:
                value = entry.get("justification_confidence")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def _aggregate(rows: list[dict], gold: dict[str, str]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "contain": 0.0,
            "norm_em": 0.0,
            "token_f1": 0.0,
            "mean_total_tokens": 0.0,
            "mean_subagent_calls": 0.0,
            "mean_wallclock_seconds": 0.0,
        }
    n = len(rows)
    contain_sum = 0.0
    em_sum = 0.0
    f1_sum = 0.0
    tokens_sum = 0.0
    subagents_sum = 0.0
    wall_sum = 0.0
    for row in rows:
        qid = str(row.get("id", "")).strip()
        ref = gold.get(qid, "")
        pred = str(row.get("answer", ""))
        contain_sum += contain(pred, ref)
        em_sum += norm_em(pred, ref)
        f1_sum += token_f1(pred, ref)
        meta = row.get("metadata") or {}
        tokens_sum += float(meta.get("total_tokens", 0))
        subagents_sum += float(meta.get("num_subagent_calls", 0))
        wall_sum += float(meta.get("wallclock_seconds", 0.0))
    return {
        "n": n,
        "contain": contain_sum / n,
        "norm_em": em_sum / n,
        "token_f1": f1_sum / n,
        "mean_total_tokens": tokens_sum / n,
        "mean_subagent_calls": subagents_sum / n,
        "mean_wallclock_seconds": wall_sum / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--tau", type=float, default=0.70)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as handle:
        gold = {str(q.get("id", "")).strip(): str(q.get("answer", "")) for q in json.load(handle)}

    sufficient: list[dict] = []
    insufficient: list[dict] = []
    missing: list[dict] = []
    with open(args.predictions, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = _extract_sufficiency(row)
            if s is None:
                missing.append(row)
                continue
            if s >= args.tau:
                sufficient.append(row)
            else:
                insufficient.append(row)

    summary = {
        "tau": args.tau,
        "overall": _aggregate(sufficient + insufficient + missing, gold),
        "probe_sufficient": _aggregate(sufficient, gold),
        "probe_insufficient": _aggregate(insufficient, gold),
        "no_sufficiency_logged": _aggregate(missing, gold),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
