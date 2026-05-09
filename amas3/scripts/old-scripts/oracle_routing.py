#!/usr/bin/env python3
"""Compute an oracle routing upper bound between two variants."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from eval_offline import contain, norm_em, token_f1

logger = logging.getLogger(__name__)


def _load_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[str(obj.get("id", ""))] = obj
    return rows


def _select_prediction(base: dict, alt: dict, gold_answer: str) -> tuple[str, dict]:
    base_answer = str(base.get("answer", ""))
    alt_answer = str(alt.get("answer", ""))
    base_score = (norm_em(base_answer, gold_answer), token_f1(base_answer, gold_answer))
    alt_score = (norm_em(alt_answer, gold_answer), token_f1(alt_answer, gold_answer))
    if alt_score > base_score:
        return "alt", alt
    if base_score > alt_score:
        return "base", base
    base_tokens = (base.get("metadata", {}) or {}).get("total_tokens", 0)
    alt_tokens = (alt.get("metadata", {}) or {}).get("total_tokens", 0)
    return ("alt", alt) if alt_tokens < base_tokens else ("base", base)


def _summary(predictions: list[dict]) -> dict:
    n = len(predictions)
    if n == 0:
        return {
            "norm_em": 0.0,
            "token_f1": 0.0,
            "contain": 0.0,
            "mean_tokens": 0.0,
            "count": 0,
        }

    em_scores = []
    f1_scores = []
    contain_scores = []
    total_tokens = []
    for pred in predictions:
        answer = str(pred.get("answer", ""))
        gold_answer = str(pred.get("gold_answer", ""))
        em_scores.append(norm_em(answer, gold_answer))
        f1_scores.append(token_f1(answer, gold_answer))
        contain_scores.append(contain(answer, gold_answer))
        total_tokens.append((pred.get("metadata", {}) or {}).get("total_tokens", 0))
    return {
        "norm_em": round(sum(em_scores) / n, 4),
        "token_f1": round(sum(f1_scores) / n, 4),
        "contain": round(sum(contain_scores) / n, 4),
        "mean_tokens": round(sum(total_tokens) / n, 1),
        "count": n,
    }


def run(config_path: str, results_dir: str, questions_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    oracle_cfg = config.get("oracle_routing", {})
    base_variant = oracle_cfg.get("base_variant", "S0")
    alt_variant = oracle_cfg.get("alt_variant", "M1")
    output_variant = oracle_cfg.get("output_variant", "D1")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    gold_by_id = {str(q.get("id", "")): str(q.get("answer", "")) for q in questions}

    rdir = Path(results_dir)
    out_dir = rdir / output_variant
    out_dir.mkdir(parents=True, exist_ok=True)

    base_preds = _load_predictions(rdir / base_variant / "predictions.jsonl")
    alt_preds = _load_predictions(rdir / alt_variant / "predictions.jsonl")
    qids = sorted(set(base_preds) & set(alt_preds))

    chosen: list[dict] = []
    disagreement_total = 0
    disagreement_gain = 0

    for qid in qids:
        gold_answer = gold_by_id.get(qid, str(base_preds[qid].get("gold_answer", "")))
        base = base_preds[qid]
        alt = alt_preds[qid]
        base_correct = norm_em(str(base.get("answer", "")), gold_answer) >= 1.0
        alt_correct = norm_em(str(alt.get("answer", "")), gold_answer) >= 1.0
        if base_correct != alt_correct:
            disagreement_total += 1
            disagreement_gain += 1

        selected_name, selected = _select_prediction(base, alt, gold_answer)
        record = dict(selected)
        record["gold_answer"] = gold_answer
        record.setdefault("metadata", {})
        record["metadata"]["oracle_choice"] = base_variant if selected_name == "base" else alt_variant
        chosen.append(record)

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summary(chosen)
    summary.update(
        {
            "base_variant": base_variant,
            "alt_variant": alt_variant,
            "output_variant": output_variant,
            "disagreement_count": disagreement_total,
            "disagreement_oracle_gain": disagreement_gain,
        }
    )
    with open(out_dir / "predictions_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "oracle_routing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Oracle routing summary written to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute oracle routing upper bound.")
    parser.add_argument("--config", required=True, help="Path to d1.yaml")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--questions", required=True, help="Path to questions.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args.config, args.results_dir, args.questions)


if __name__ == "__main__":
    main()
