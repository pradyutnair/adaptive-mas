"""Paired bootstrap and McNemar significance for two systems' predictions.

Both systems must have a predictions.jsonl file containing at least the keys
``id`` and ``answer``. Gold answers come from a questions.json file with
``id`` and ``answer``. The primary metric is contain (per the EMNLP plan).
Secondary: norm_em, token_f1.

Reports:
- Per-system point estimates on the intersected question set.
- Paired-bootstrap 95% CI on the contain delta (system_b - system_a).
- McNemar test on per-question contain-correctness.

Usage:

    python3 scripts/paired_bootstrap.py \\
        --system-a results/<baseline>/predictions.jsonl \\
        --system-b results/<method>/predictions.jsonl \\
        --questions data/hotpotqa/questions_1000_seed42.json \\
        --output results/<method>/significance_vs_<baseline>.json \\
        --bootstrap-iters 10000 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_offline import contain, norm_em, token_f1  # noqa: E402


def _load_predictions(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(obj.get("id") or obj.get("question_id") or "").strip()
            if qid:
                out[qid] = str(obj.get("answer", ""))
    return out


def _load_gold(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        questions = json.load(handle)
    return {str(q.get("id", "")).strip(): str(q.get("answer", "")) for q in questions}


def _per_question_scores(
    preds: dict[str, str],
    gold: dict[str, str],
    qids: list[str],
) -> dict[str, list[float]]:
    contain_scores = []
    em_scores = []
    f1_scores = []
    for qid in qids:
        pred = preds.get(qid, "")
        ref = gold.get(qid, "")
        contain_scores.append(contain(pred, ref))
        em_scores.append(norm_em(pred, ref))
        f1_scores.append(token_f1(pred, ref))
    return {"contain": contain_scores, "norm_em": em_scores, "token_f1": f1_scores}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _paired_bootstrap_ci(
    a_scores: list[float],
    b_scores: list[float],
    iters: int,
    seed: int,
) -> dict[str, float]:
    """Paired bootstrap 95% CI on (mean(b) - mean(a))."""
    n = len(a_scores)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iters):
        sample_sum_a = 0.0
        sample_sum_b = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            sample_sum_a += a_scores[idx]
            sample_sum_b += b_scores[idx]
        deltas.append((sample_sum_b - sample_sum_a) / n)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[int(0.975 * iters) - 1] if iters > 0 else 0.0
    point = _mean(b_scores) - _mean(a_scores)
    p_value = sum(1 for d in deltas if d <= 0) / iters if iters else 1.0
    p_value = 2 * min(p_value, 1.0 - p_value)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "p_value_two_sided": p_value}


def _mcnemar(
    a_scores: list[float],
    b_scores: list[float],
) -> dict[str, float]:
    """McNemar on binary contain-correctness."""
    b_only = sum(1 for a, b in zip(a_scores, b_scores) if b > a)
    a_only = sum(1 for a, b in zip(a_scores, b_scores) if a > b)
    n_disc = b_only + a_only
    if n_disc == 0:
        return {"a_only": 0, "b_only": 0, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b_only - a_only) - 1) ** 2 / n_disc
    # Approximate two-sided p-value from chi-square(1).
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "a_only": a_only,
        "b_only": b_only,
        "statistic": statistic,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-a", required=True, help="Baseline predictions.jsonl")
    parser.add_argument("--system-b", required=True, help="Method predictions.jsonl")
    parser.add_argument("--questions", required=True, help="Gold questions.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-a", default="system_a")
    parser.add_argument("--label-b", default="system_b")
    args = parser.parse_args()

    preds_a = _load_predictions(args.system_a)
    preds_b = _load_predictions(args.system_b)
    gold = _load_gold(args.questions)

    qids = sorted(set(preds_a) & set(preds_b) & set(gold))
    if not qids:
        raise SystemExit("No overlapping question ids across the three inputs.")

    scores_a = _per_question_scores(preds_a, gold, qids)
    scores_b = _per_question_scores(preds_b, gold, qids)

    summary = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n": len(qids),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "metrics": {},
    }

    for metric in ("contain", "norm_em", "token_f1"):
        a_vals = scores_a[metric]
        b_vals = scores_b[metric]
        ci = _paired_bootstrap_ci(a_vals, b_vals, args.bootstrap_iters, args.seed)
        block = {
            "mean_a": _mean(a_vals),
            "mean_b": _mean(b_vals),
            "delta_b_minus_a": ci["point"],
            "delta_ci95_lo": ci["ci_lo"],
            "delta_ci95_hi": ci["ci_hi"],
            "delta_p_two_sided": ci["p_value_two_sided"],
        }
        if metric == "contain":
            block["mcnemar"] = _mcnemar(a_vals, b_vals)
        summary["metrics"][metric] = block

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
