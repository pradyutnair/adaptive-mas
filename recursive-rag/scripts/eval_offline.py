"""Offline evaluation for Adaptive Recursive SAGE predictions.

Computes norm_em, token_f1, and contain metrics from a
predictions.jsonl file matched against the gold questions.

Usage::

    python3 scripts/eval_offline.py \\
        --predictions results/S4/predictions.jsonl \\
        --questions data/musique/questions.json \\
        --output results/S4/predictions_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Normalisation and metrics
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Normalize an answer string for metric computation.

    Steps: lowercase → strip articles (a, an, the) → strip punctuation
    → collapse whitespace → strip.
    """
    s = s.lower()
    s = _ARTICLES.sub("", s)
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = " ".join(s.split())  # collapse whitespace
    return s.strip()


def norm_em(pred: str, gold: str) -> float:
    """Compute normalized exact match (1.0 if equal, 0.0 otherwise)."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between normalized predictions and gold.

    Both strings are normalized, tokenized on whitespace, and compared
    as bags of tokens.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    # Use counts for proper precision/recall
    from collections import Counter

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    common_count = sum((pred_counts & gold_counts).values())

    precision = common_count / len(pred_tokens)
    recall = common_count / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def contain(pred: str, gold: str) -> float:
    """Compute containment.

    Returns 1.0 if the normalized gold answer is a substring of the
    normalized prediction; 0.0 otherwise.
    """
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)

    if not norm_pred or not norm_gold:
        return 0.0

    if norm_gold in norm_pred:
        return 1.0
    return 0.0


def contain_bi(pred: str, gold: str) -> float:
    """Backward-compatible alias for contain()."""
    return contain(pred, gold)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(
    predictions_path: str,
    questions_path: str,
    output_path: str,
) -> dict[str, Any]:
    """Load predictions and questions, compute metrics, write summary JSON.

    Parameters
    ----------
    predictions_path:
        Path to predictions.jsonl (one JSON object per line).
    questions_path:
        Path to questions.json (list of dicts with ``id`` and ``answer``).
    output_path:
        Path to write the summary JSON.

    Returns
    -------
    dict
        The computed summary: ``{norm_em, token_f1, contain, total, answered}``.
    """
    # Load questions → gold answers by ID
    with open(questions_path, "r", encoding="utf-8") as f:
        questions: list[dict] = json.load(f)
    gold_by_id: dict[str, str] = {}
    for q in questions:
        qid = str(q.get("id", ""))
        gold_answer = str(q.get("answer", ""))
        gold_by_id[qid] = gold_answer

    # Load predictions
    predictions: list[dict] = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Compute per-question metrics
    em_scores: list[float] = []
    f1_scores: list[float] = []
    contain_scores: list[float] = []
    answered = 0

    for pred in predictions:
        qid = str(pred.get("id", ""))
        pred_answer = str(pred.get("answer", ""))
        gold_answer = gold_by_id.get(qid, "")

        if pred_answer.strip():
            answered += 1

        em_scores.append(norm_em(pred_answer, gold_answer))
        f1_scores.append(token_f1(pred_answer, gold_answer))
        contain_scores.append(contain(pred_answer, gold_answer))

    total = len(predictions)

    # Aggregate
    summary = {
        "norm_em": round(sum(em_scores) / total, 4) if total > 0 else 0.0,
        "token_f1": round(sum(f1_scores) / total, 4) if total > 0 else 0.0,
        "contain": round(sum(contain_scores) / total, 4) if total > 0 else 0.0,
        "total": total,
        "answered": answered,
    }

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Eval summary: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline evaluation of Adaptive Recursive SAGE predictions."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions.json (gold answers)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write predictions_eval_summary.json",
    )
    args = parser.parse_args()

    evaluate(
        predictions_path=args.predictions,
        questions_path=args.questions,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
