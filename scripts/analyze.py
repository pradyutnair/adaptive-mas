"""Analysis script for Adaptive Recursive SAGE experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval_offline import contain, norm_em, token_f1

logger = logging.getLogger(__name__)

_VARIANT_ORDER = [
    "P0",
    "P1",
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "M1",
    "M1_1",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7S",
    "A7",
    "A7L",
    "A8",
    "A8L",
    "D1",
]
_HOP_PATTERNS = {"2hop": "2hop", "3hop": "3hop", "4hop": "4hop"}
_FAILURE_ORDER = [
    "no_final_answer",
    "fully_unresolved",
    "multi_step_wrong_chain",
    "single_step_semantic_mismatch",
]


def _variant_sort_key(variant: str) -> tuple[int, str]:
    try:
        return (_VARIANT_ORDER.index(variant), variant)
    except ValueError:
        return (len(_VARIANT_ORDER), variant)


def _detect_hop(question_id: str) -> str:
    qid_lower = question_id.lower()
    for key, pattern in _HOP_PATTERNS.items():
        if pattern in qid_lower:
            return key
    return "unknown"


def _load_questions(questions_path: str) -> list[dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gold_by_id(questions: list[dict]) -> dict[str, str]:
    return {str(q.get("id", "")): str(q.get("answer", "")) for q in questions}


def _hop_by_id(questions: list[dict]) -> dict[str, str]:
    return {str(q.get("id", "")): _detect_hop(str(q.get("id", ""))) for q in questions}


def _discover_variants(results_dir: Path) -> list[str]:
    variants: list[str] = []
    if not results_dir.exists():
        return variants
    for path in results_dir.iterdir():
        if not path.is_dir():
            continue
        if any(
            (path / name).exists()
            for name in ("predictions.jsonl", "predictions_eval_summary.json")
        ):
            variants.append(path.name)
    return sorted(set(variants), key=_variant_sort_key)


def _load_predictions(results_dir: Path, variant: str) -> list[dict]:
    preds_file = results_dir / variant / "predictions.jsonl"
    if not preds_file.exists():
        return []
    predictions: list[dict] = []
    with open(preds_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return predictions


def _load_eval_summary(results_dir: Path, variant: str) -> dict[str, Any]:
    summary_file = results_dir / variant / "predictions_eval_summary.json"
    if not summary_file.exists():
        return {}
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_summary(results_dir: Path, variant: str) -> dict[str, Any]:
    summary_file = results_dir / variant / "run_summary.json"
    if not summary_file.exists():
        return {}
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _meta(pred: dict) -> dict[str, Any]:
    return pred.get("metadata", {}) or {}


def _score_prediction(pred: dict, gold_answer: str) -> dict[str, float]:
    answer = str(pred.get("answer", ""))
    return {
        "norm_em": norm_em(answer, gold_answer),
        "token_f1": token_f1(answer, gold_answer),
        "contain": contain(answer, gold_answer),
    }


def _is_correct(pred: dict, gold_answer: str) -> bool:
    return _score_prediction(pred, gold_answer)["norm_em"] >= 1.0


def _aggregate_rows(
    variant: str,
    predictions: list[dict],
    gold_lookup: dict[str, str],
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not predictions:
        return {
            "variant": variant,
            "count": 0,
            "norm_em": 0.0,
            "token_f1": 0.0,
            "contain": 0.0,
            "mean_subagent_calls": 0.0,
            "mean_verify_calls": 0.0,
            "mean_tokens": 0.0,
            "mean_wallclock_seconds": 0.0,
            "p50_question_wallclock_seconds": 0.0,
            "p95_question_wallclock_seconds": 0.0,
        }

    em_scores = []
    f1_scores = []
    contain_scores = []
    subagent_calls = []
    verify_calls = []
    total_tokens = []

    for pred in predictions:
        qid = str(pred.get("id", ""))
        gold_answer = gold_lookup.get(qid, str(pred.get("gold_answer", "")))
        scores = _score_prediction(pred, gold_answer)
        meta = _meta(pred)
        em_scores.append(scores["norm_em"])
        f1_scores.append(scores["token_f1"])
        contain_scores.append(scores["contain"])
        subagent_calls.append(meta.get("num_subagent_calls", 0))
        verify_calls.append(meta.get("num_verify_calls", 0))
        total_tokens.append(meta.get("total_tokens", 0))

    n = len(predictions)
    return {
        "variant": variant,
        "count": n,
        "norm_em": round(sum(em_scores) / n, 4),
        "token_f1": round(sum(f1_scores) / n, 4),
        "contain": round(sum(contain_scores) / n, 4),
        "mean_subagent_calls": round(sum(subagent_calls) / n, 2),
        "mean_verify_calls": round(sum(verify_calls) / n, 2),
        "mean_tokens": round(sum(total_tokens) / n, 1),
        "mean_wallclock_seconds": round(
            float((run_summary or {}).get("mean_wallclock_seconds", 0.0)),
            3,
        ),
        "p50_question_wallclock_seconds": round(
            float((run_summary or {}).get("p50_question_wallclock_seconds", 0.0)),
            3,
        ),
        "p95_question_wallclock_seconds": round(
            float((run_summary or {}).get("p95_question_wallclock_seconds", 0.0)),
            3,
        ),
    }


def _write_json_md(
    rows: list[dict[str, Any]],
    json_path: Path,
    md_path: Path,
    title: str,
    columns: list[tuple[str, str]],
) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(label for _, label in columns) + " |\n")
        f.write("|" + "|".join("-" * (len(label) + 2) for _, label in columns) + "|\n")
        for row in rows:
            values: list[str] = []
            for key, _ in columns:
                value = row.get(key, "")
                if isinstance(value, float):
                    if key.endswith(("norm_em", "token_f1", "contain")):
                        values.append(f"{value:.4f}")
                    else:
                        values.append(f"{value:.2f}")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def _compute_main_results(
    results_dir: Path,
    variants: list[str],
    gold_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        rows.append(
            _aggregate_rows(
                variant,
                _load_predictions(results_dir, variant),
                gold_lookup,
                _load_run_summary(results_dir, variant),
            )
        )
    return rows


def _compute_per_hop_breakdown(
    results_dir: Path,
    variants: list[str],
    gold_lookup: dict[str, str],
    hop_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        preds = _load_predictions(results_dir, variant)
        run_summary = _load_run_summary(results_dir, variant)
        by_hop: dict[str, list[dict]] = defaultdict(list)
        for pred in preds:
            by_hop[hop_lookup.get(str(pred.get("id", "")), "unknown")].append(pred)
        for hop in ["2hop", "3hop", "4hop"]:
            rows.append(
                {
                    "variant": variant,
                    "hop": hop,
                    **_aggregate_rows(
                        f"{variant}:{hop}",
                        by_hop.get(hop, []),
                        gold_lookup,
                        run_summary,
                    ),
                }
            )
            rows[-1].pop("variant_ignored", None)
            rows[-1]["variant"] = variant
            rows[-1].pop("count", None)
            rows[-1]["count"] = len(by_hop.get(hop, []))
    return rows


def _baseline_subsets(
    results_dir: Path,
    gold_lookup: dict[str, str],
) -> tuple[set[str], set[str]]:
    easy_ids: set[str] = set()
    hard_ids: set[str] = set()
    baseline_variant = "s0_matched" if (results_dir / "s0_matched").exists() else "S0"
    for pred in _load_predictions(results_dir, baseline_variant):
        qid = str(pred.get("id", ""))
        gold_answer = gold_lookup.get(qid, str(pred.get("gold_answer", "")))
        if _is_correct(pred, gold_answer):
            easy_ids.add(qid)
        else:
            hard_ids.add(qid)
    return easy_ids, hard_ids


def _subset_rows(
    results_dir: Path,
    variants: list[str],
    gold_lookup: dict[str, str],
    subsets: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        preds = _load_predictions(results_dir, variant)
        run_summary = _load_run_summary(results_dir, variant)
        by_id = {str(pred.get("id", "")): pred for pred in preds}
        for subset_name, subset_ids in subsets.items():
            selected = [by_id[qid] for qid in subset_ids if qid in by_id]
            row = _aggregate_rows(variant, selected, gold_lookup, run_summary)
            row["subset"] = subset_name
            rows.append(row)
    return rows


def _classify_failure(pred: dict, gold_answer: str) -> str | None:
    if _is_correct(pred, gold_answer):
        return None
    answer = str(pred.get("answer", "")).strip()
    meta = _meta(pred)
    step_trace = meta.get("step_trace", []) or []
    fact_count = len(meta.get("facts_used", []) or [])
    subagents = int(meta.get("num_subagent_calls", 0) or 0)
    verify_calls = int(meta.get("num_verify_calls", 0) or 0)

    if not answer:
        return "no_final_answer"
    if fact_count == 0 and subagents == 0 and not step_trace:
        return "fully_unresolved"
    if subagents + verify_calls >= 2 or len(step_trace) >= 3:
        return "multi_step_wrong_chain"
    return "single_step_semantic_mismatch"


def _compute_failure_modes(
    results_dir: Path,
    variants: list[str],
    gold_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        preds = _load_predictions(results_dir, variant)
        counts = Counter()
        failures = 0
        for pred in preds:
            qid = str(pred.get("id", ""))
            gold_answer = gold_lookup.get(qid, str(pred.get("gold_answer", "")))
            category = _classify_failure(pred, gold_answer)
            if category is None:
                continue
            failures += 1
            counts[category] += 1
        for category in _FAILURE_ORDER:
            count = counts.get(category, 0)
            rows.append(
                {
                    "variant": variant,
                    "category": category,
                    "count": count,
                    "share_of_failures": round(count / failures, 4) if failures else 0.0,
                    "total_failures": failures,
                }
            )
    return rows


def _compute_mechanism_validation(
    subset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hard_rows = {
        row["variant"]: row for row in subset_rows if row["subset"] == "hard"
    }
    main_variant = "M1_1" if "M1_1" in hard_rows else "M1"
    if main_variant not in hard_rows:
        return rows

    main = hard_rows[main_variant]
    ablation_variants = [
        variant
        for variant in hard_rows
        if variant.startswith("A") and variant != "M1"
    ]
    for ablation in sorted(ablation_variants, key=_variant_sort_key):
        other = hard_rows[ablation]
        rows.append(
            {
                "variant": ablation,
                "subset": "hard",
                "m1_norm_em": main["norm_em"],
                "variant_norm_em": other["norm_em"],
                "delta_norm_em": round(main["norm_em"] - other["norm_em"], 4),
                "m1_token_f1": main["token_f1"],
                "variant_token_f1": other["token_f1"],
                "delta_token_f1": round(main["token_f1"] - other["token_f1"], 4),
                "m1_mean_tokens": main["mean_tokens"],
                "variant_mean_tokens": other["mean_tokens"],
                "delta_mean_tokens": round(main["mean_tokens"] - other["mean_tokens"], 1),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_frontiers(
    main_rows: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    scaling_rows = [
        {
            "variant": row["variant"],
            "mean_tokens": row["mean_tokens"],
            "norm_em": row["norm_em"],
            "token_f1": row["token_f1"],
        }
        for row in main_rows
    ]
    _write_csv(
        results_dir / "scaling_data.csv",
        scaling_rows,
        ["variant", "mean_tokens", "norm_em", "token_f1"],
    )

    frontier_rows = [
        {
            "variant": row["variant"],
            "subset": row["subset"],
            "mean_tokens": row["mean_tokens"],
            "norm_em": row["norm_em"],
            "token_f1": row["token_f1"],
        }
        for row in subset_rows
    ]
    _write_csv(
        results_dir / "efficiency_frontier.csv",
        frontier_rows,
        ["variant", "subset", "mean_tokens", "norm_em", "token_f1"],
    )


def _compute_route_decisions(
    results_dir: Path,
    variants: list[str],
    hop_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        preds = _load_predictions(results_dir, variant)
        counts: Counter[tuple[str, str]] = Counter()
        totals: Counter[str] = Counter()
        for pred in preds:
            qid = str(pred.get("id", ""))
            hop = hop_lookup.get(qid, "unknown")
            meta = _meta(pred)
            route_decision = str(meta.get("route_decision", "")).strip()
            if not route_decision:
                step_trace = meta.get("step_trace", []) or []
                if step_trace:
                    route_decision = str(step_trace[0].get("route_decision", "")).strip()
            if not route_decision:
                route_decision = "untracked"
            counts[(hop, route_decision)] += 1
            totals[hop] += 1

        for hop in ["2hop", "3hop", "4hop", "unknown"]:
            total = totals.get(hop, 0)
            decisions = {decision for (row_hop, decision) in counts if row_hop == hop}
            for decision in sorted(decisions):
                count = counts[(hop, decision)]
                rows.append(
                    {
                        "variant": variant,
                        "hop": hop,
                        "route_decision": decision,
                        "count": count,
                        "share": round(count / total, 4) if total else 0.0,
                    }
                )
    return rows


def _check_acceptance_criteria(
    main_rows: list[dict[str, Any]],
    per_hop_rows: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
) -> None:
    if any(row["variant"] == "M1_1" for row in main_rows):
        primary = "M1_1"
    elif any(row["variant"] == "M1" for row in main_rows):
        primary = "M1"
    else:
        primary = "S4"
    per_hop = {(row["variant"], row["hop"]): row for row in per_hop_rows}

    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA")
    print("=" * 60)

    criterion1 = False
    for hop in ["3hop", "4hop"]:
        base = per_hop.get(("S0", hop), {})
        test = per_hop.get((primary, hop), {})
        em_delta = (test.get("norm_em", 0.0) - base.get("norm_em", 0.0)) * 100
        f1_delta = (test.get("token_f1", 0.0) - base.get("token_f1", 0.0)) * 100
        criterion1 = criterion1 or em_delta >= 1.0 or f1_delta >= 1.0
        print(
            f"  {hop}: {primary} vs S0 -> EM {em_delta:+.1f}pp, "
            f"F1 {f1_delta:+.1f}pp"
        )
    print(
        f"  → {primary} beats S0 on 3-hop or 4-hop by ≥1pp: "
        f"{'PASS' if criterion1 else 'FAIL'}"
    )

    subset_map = {(row["variant"], row["subset"]): row for row in subset_rows}
    easy = subset_map.get((primary, "easy"), {})
    hard = subset_map.get((primary, "hard"), {})
    if easy or hard:
        print(
            f"  Mean subagent calls easy={easy.get('mean_subagent_calls', 0.0):.2f}, "
            f"hard={hard.get('mean_subagent_calls', 0.0):.2f}: "
            f"{'PASS' if hard.get('mean_subagent_calls', 0.0) > easy.get('mean_subagent_calls', 0.0) else 'FAIL'}"
        )
        print(
            f"  Mean subagent calls easy={easy.get('mean_subagent_calls', 0.0):.2f} < 1.5: "
            f"{'PASS' if easy.get('mean_subagent_calls', 0.0) < 1.5 else 'FAIL'}"
        )

    print("=" * 60)


def analyze(results_dir: str, questions_path: str) -> None:
    rdir = Path(results_dir)
    questions = _load_questions(questions_path)
    gold_lookup = _gold_by_id(questions)
    hop_lookup = _hop_by_id(questions)
    variants = _discover_variants(rdir)
    if not variants:
        raise FileNotFoundError(f"No experiment result directories found under {rdir}")

    main_rows = _compute_main_results(rdir, variants, gold_lookup)
    _write_json_md(
        main_rows,
        rdir / "main_results.json",
        rdir / "main_results.md",
        "Main Results",
        [
            ("variant", "Variant"),
            ("count", "Count"),
            ("norm_em", "Norm EM"),
            ("token_f1", "Token F1"),
            ("contain", "Contain"),
            ("mean_subagent_calls", "Mean Subagent Calls"),
            ("mean_verify_calls", "Mean Verify Calls"),
            ("mean_tokens", "Mean Tokens"),
            ("mean_wallclock_seconds", "Mean Wallclock Seconds"),
            ("p50_question_wallclock_seconds", "P50 Question Seconds"),
            ("p95_question_wallclock_seconds", "P95 Question Seconds"),
        ],
    )

    per_hop_rows = _compute_per_hop_breakdown(rdir, variants, gold_lookup, hop_lookup)
    _write_json_md(
        per_hop_rows,
        rdir / "per_hop_breakdown.json",
        rdir / "per_hop_breakdown.md",
        "Per-Hop Breakdown",
        [
            ("variant", "Variant"),
            ("hop", "Hop"),
            ("count", "Count"),
            ("norm_em", "Norm EM"),
            ("token_f1", "Token F1"),
            ("contain", "Contain"),
            ("mean_subagent_calls", "Mean Subagent Calls"),
            ("mean_tokens", "Mean Tokens"),
            ("mean_wallclock_seconds", "Mean Wallclock Seconds"),
        ],
    )

    easy_ids, hard_ids = _baseline_subsets(rdir, gold_lookup)
    subset_rows = _subset_rows(
        rdir,
        variants,
        gold_lookup,
        {"easy": easy_ids, "hard": hard_ids},
    )
    _write_json_md(
        subset_rows,
        rdir / "subset_breakdown.json",
        rdir / "subset_breakdown.md",
        "Subset Breakdown",
        [
            ("variant", "Variant"),
            ("subset", "Subset"),
            ("count", "Count"),
            ("norm_em", "Norm EM"),
            ("token_f1", "Token F1"),
            ("contain", "Contain"),
            ("mean_subagent_calls", "Mean Subagent Calls"),
            ("mean_tokens", "Mean Tokens"),
            ("mean_wallclock_seconds", "Mean Wallclock Seconds"),
        ],
    )

    failure_rows = _compute_failure_modes(rdir, variants, gold_lookup)
    _write_json_md(
        failure_rows,
        rdir / "failure_modes.json",
        rdir / "failure_modes.md",
        "Failure Modes",
        [
            ("variant", "Variant"),
            ("category", "Category"),
            ("count", "Count"),
            ("share_of_failures", "Share of Failures"),
            ("total_failures", "Total Failures"),
        ],
    )

    mechanism_rows = _compute_mechanism_validation(subset_rows)
    _write_json_md(
        mechanism_rows,
        rdir / "mechanism_validation.json",
        rdir / "mechanism_validation.md",
        "Mechanism Validation",
        [
            ("variant", "Variant"),
            ("subset", "Subset"),
            ("m1_norm_em", "M1 Norm EM"),
            ("variant_norm_em", "Variant Norm EM"),
            ("delta_norm_em", "Delta Norm EM"),
            ("m1_token_f1", "M1 Token F1"),
            ("variant_token_f1", "Variant Token F1"),
            ("delta_token_f1", "Delta Token F1"),
            ("m1_mean_tokens", "M1 Mean Tokens"),
            ("variant_mean_tokens", "Variant Mean Tokens"),
            ("delta_mean_tokens", "Delta Mean Tokens"),
        ],
    )

    route_rows = _compute_route_decisions(rdir, variants, hop_lookup)
    _write_json_md(
        route_rows,
        rdir / "route_decisions.json",
        rdir / "route_decisions.md",
        "Route Decisions",
        [
            ("variant", "Variant"),
            ("hop", "Hop"),
            ("route_decision", "Route Decision"),
            ("count", "Count"),
            ("share", "Share"),
        ],
    )

    _write_frontiers(main_rows, subset_rows, rdir)
    _check_acceptance_criteria(main_rows, per_hop_rows, subset_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Adaptive Recursive SAGE experiment results."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing variant result subdirectories",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analyze(results_dir=args.results_dir, questions_path=args.questions)


if __name__ == "__main__":
    main()
