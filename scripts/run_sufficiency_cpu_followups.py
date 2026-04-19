#!/usr/bin/env python3
"""CPU-only follow-up analyses for iter31 sufficiency results.

Produces:
- held-out Platt/isotonic calibration summaries
- conservative oracle probe upper-bound analysis
- cross-dataset tau-transfer analysis from the MuSiQue-200 ablation sweep
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_offline import contain, norm_em, token_f1  # noqa: E402

DATASETS = ("musique", "hotpotqa", "2wikimultihop")
QUESTION_FILES = {
    "musique": ROOT / "data/musique/questions_1000_seedfull_combined.json",
    "hotpotqa": ROOT / "data/hotpotqa/questions_1000_seed42.json",
    "2wikimultihop": ROOT / "data/2wikimultihop/questions_1000_seed42.json",
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(scores, labels, reg=1e-3, max_iter=100):
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    a = 1.0
    b = 0.0
    for _ in range(max_iter):
        z = a * x + b
        p = sigmoid(z)
        grad_a = np.sum((p - y) * x) + reg * a
        grad_b = np.sum(p - y) + reg * b
        w = p * (1.0 - p)
        h_aa = np.sum(w * x * x) + reg
        h_ab = np.sum(w * x)
        h_bb = np.sum(w) + reg
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        step_a = (h_bb * grad_a - h_ab * grad_b) / det
        step_b = (-h_ab * grad_a + h_aa * grad_b) / det
        a -= step_a
        b -= step_b
        if abs(step_a) + abs(step_b) < 1e-8:
            break
    return float(a), float(b)


def fit_isotonic(scores, labels):
    order = np.argsort(scores)
    x = np.asarray(scores, dtype=float)[order]
    y = np.asarray(labels, dtype=float)[order]
    blocks = []
    for xi, yi in zip(x, y):
        blocks.append({"start": xi, "end": xi, "sum": yi, "count": 1})
        while len(blocks) >= 2:
            prev = blocks[-2]["sum"] / blocks[-2]["count"]
            curr = blocks[-1]["sum"] / blocks[-1]["count"]
            if prev <= curr:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "sum": left["sum"] + right["sum"],
                    "count": left["count"] + right["count"],
                }
            )
    thresholds = []
    values = []
    for block in blocks:
        thresholds.append(block["end"])
        values.append(block["sum"] / block["count"])
    return np.asarray(thresholds, dtype=float), np.asarray(values, dtype=float)


def predict_isotonic(scores, thresholds, values):
    scores = np.asarray(scores, dtype=float)
    idx = np.searchsorted(thresholds, scores, side="left")
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def brier(probs, labels):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def nll(probs, labels):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def expected_calibration_error(probs, labels, bins=10):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probs)
    ece = 0.0
    table = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        mean_prob = float(np.mean(probs[idx]))
        acc = float(np.mean(labels[idx]))
        frac = len(idx) / total
        ece += frac * abs(mean_prob - acc)
        table.append(
            {
                "bin": i,
                "lo": round(float(lo), 3),
                "hi": round(float(hi), 3),
                "count": int(len(idx)),
                "mean_prob": round(mean_prob, 4),
                "accuracy": round(acc, 4),
            }
        )
    return round(float(ece), 4), table


def load_questions(dataset):
    with open(QUESTION_FILES[dataset], "r", encoding="utf-8") as handle:
        questions = json.load(handle)
    gold = {str(q["id"]): str(q.get("answer", "")) for q in questions}
    order = [str(q["id"]) for q in questions]
    return gold, order


def extract_record(row, gold_answer):
    steps = row["metadata"]["step_trace"]
    assess = None
    for step in steps:
        if step.get("action") == "assess":
            assess = step
            break
    if assess is None:
        raise ValueError("missing assess step for %s" % row["id"])
    final_step = steps[-1]
    score = float(row["metadata"].get("extras", {}).get("sufficiency_score", assess["metadata"]["sufficiency"]))
    proposed = str(assess["metadata"].get("proposed_answer", ""))
    final_answer = str(row.get("answer", ""))
    route = final_step.get("metadata", {}).get("route", "")
    prefix_tokens = sum(float((step.get("tokens") or 0.0)) for step in steps[:2])
    total_tokens = float(row["metadata"].get("total_tokens", 0.0) or 0.0)
    return {
        "id": str(row["id"]),
        "score": score,
        "proposed_answer": proposed,
        "final_answer": final_answer,
        "gold_answer": gold_answer,
        "probe_contain": contain(proposed, gold_answer),
        "probe_em": norm_em(proposed, gold_answer),
        "final_contain": contain(final_answer, gold_answer),
        "final_f1": token_f1(final_answer, gold_answer),
        "final_em": norm_em(final_answer, gold_answer),
        "route": route,
        "prefix_tokens": prefix_tokens,
        "total_tokens": total_tokens,
        "steps": steps,
    }


def load_dataset_records(dataset, results_root):
    gold, order = load_questions(dataset)
    rows = read_jsonl(results_root / dataset / "sufficiency" / "predictions.jsonl")
    by_id = {str(r["id"]): r for r in rows}
    records = []
    for qid in order:
        records.append(extract_record(by_id[qid], gold[qid]))
    return records


def calibration_analysis(records):
    calib = records[::2]
    eval_records = records[1::2]
    x_cal = np.asarray([r["score"] for r in calib], dtype=float)
    y_cal = np.asarray([r["probe_contain"] for r in calib], dtype=float)
    x_eval = np.asarray([r["score"] for r in eval_records], dtype=float)
    y_eval = np.asarray([r["probe_contain"] for r in eval_records], dtype=float)

    a, b = fit_platt(x_cal, y_cal)
    iso_thresholds, iso_values = fit_isotonic(x_cal, y_cal)

    raw_probs = x_eval
    platt_probs = sigmoid(a * x_eval + b)
    isotonic_probs = predict_isotonic(x_eval, iso_thresholds, iso_values)

    out = {
        "split": {"calibration_n": int(len(calib)), "eval_n": int(len(eval_records))},
        "platt": {"a": round(a, 6), "b": round(b, 6)},
        "metrics": {},
    }
    for name, probs in (
        ("raw", raw_probs),
        ("platt", platt_probs),
        ("isotonic", isotonic_probs),
    ):
        ece, table = expected_calibration_error(probs, y_eval)
        out["metrics"][name] = {
            "ece": ece,
            "brier": round(brier(probs, y_eval), 4),
            "nll": round(nll(probs, y_eval), 4),
            "bin_table": table,
        }
    return out


def oracle_analysis(records):
    probe_correct = [r for r in records if r["probe_contain"] == 1.0]
    current_probe_correct = [r for r in probe_correct if r["route"] == "answer_from_probe"]
    missed_probe_correct = [r for r in probe_correct if r["route"] != "answer_from_probe"]
    current_probe_incorrect = [r for r in records if r["route"] == "answer_from_probe" and r["probe_contain"] == 0.0]

    tail_tokens = [
        r["total_tokens"] - r["prefix_tokens"]
        for r in records
        if r["route"] == "answer_from_probe"
    ]
    median_probe_tail = float(np.median(tail_tokens)) if tail_tokens else 0.0

    oracle_tokens = []
    current_tokens = []
    oracle_contain = []
    current_contain = []
    avoidable_tokens = 0.0
    for r in records:
        current_tokens.append(r["total_tokens"])
        current_contain.append(r["final_contain"])
        if r["probe_contain"] == 1.0:
            oracle_contain.append(1.0)
            if r["route"] == "answer_from_probe":
                oracle_tokens.append(r["total_tokens"])
            else:
                probe_path_tokens = r["prefix_tokens"] + median_probe_tail
                oracle_tokens.append(probe_path_tokens)
                avoidable_tokens += max(0.0, r["total_tokens"] - probe_path_tokens)
        else:
            oracle_contain.append(r["final_contain"])
            oracle_tokens.append(r["total_tokens"])

    oracle_probe_count = len(probe_correct)
    current_probe_count = sum(1 for r in records if r["route"] == "answer_from_probe")
    ratio = (len(current_probe_correct) / oracle_probe_count) if oracle_probe_count else 0.0
    return {
        "n": len(records),
        "oracle_answerable_from_probe": oracle_probe_count,
        "current_answered_from_probe": current_probe_count,
        "current_probe_precision": round(
            (len(current_probe_correct) / current_probe_count) if current_probe_count else 0.0,
            4,
        ),
        "controller_recall_of_oracle_probe": round(ratio, 4),
        "current_missed_correct_probe": len(missed_probe_correct),
        "current_wrong_probe_answers": len(current_probe_incorrect),
        "current_mean_contain": round(float(np.mean(current_contain)), 4),
        "oracle_mean_contain": round(float(np.mean(oracle_contain)), 4),
        "current_mean_tokens": round(float(np.mean(current_tokens)), 1),
        "oracle_mean_tokens_approx": round(float(np.mean(oracle_tokens)), 1),
        "approx_avoidable_tokens_total": round(float(avoidable_tokens), 1),
        "approx_avoidable_tokens_per_question": round(float(avoidable_tokens / len(records)), 1),
        "median_probe_answer_tail_tokens": round(median_probe_tail, 1),
    }


def offline_tau_counterfactual(records, tau):
    tail_tokens = [
        r["total_tokens"] - r["prefix_tokens"]
        for r in records
        if r["route"] == "answer_from_probe"
    ]
    median_probe_tail = float(np.median(tail_tokens)) if tail_tokens else 0.0

    contains = []
    f1s = []
    ems = []
    tokens = []
    probe_routes = 0
    synthetic_probe_routes = 0
    for r in records:
        if r["score"] >= tau:
            answer = r["proposed_answer"]
            probe_routes += 1
            if r["route"] != "answer_from_probe":
                synthetic_probe_routes += 1
                tok = r["prefix_tokens"] + median_probe_tail
            else:
                tok = r["total_tokens"]
        else:
            answer = r["final_answer"]
            tok = r["total_tokens"]
        gold = r["gold_answer"]
        contains.append(contain(answer, gold))
        f1s.append(token_f1(answer, gold))
        ems.append(norm_em(answer, gold))
        tokens.append(tok)
    return {
        "tau": tau,
        "contain": round(float(np.mean(contains)), 4),
        "token_f1": round(float(np.mean(f1s)), 4),
        "norm_em": round(float(np.mean(ems)), 4),
        "mean_tokens_approx": round(float(np.mean(tokens)), 1),
        "probe_route_rate": round(probe_routes / len(records), 4),
        "synthetic_probe_routes": synthetic_probe_routes,
        "median_probe_answer_tail_tokens": round(median_probe_tail, 1),
    }


def choose_transfer_tau(ablation_summary):
    tau_rows = []
    for name, row in ablation_summary["variants"].items():
        if not name.startswith("abl_tau_"):
            continue
        tau_rows.append(
            {
                "name": name,
                "tau": float(name.split("_")[-1]) / 100.0,
                "contain": float(row["mean_contain"]),
                "mean_tokens": float(row["mean_tokens"]),
            }
        )
    tau_rows.sort(key=lambda x: (-x["contain"], x["mean_tokens"], x["tau"]))
    selected = tau_rows[0]
    return selected, sorted(tau_rows, key=lambda x: x["tau"])


def write_summary(output_dir, calibration, oracle, tau_transfer):
    lines = ["# CPU Follow-ups", ""]
    lines.append("## 7. Calibration")
    for ds in DATASETS:
        m = calibration[ds]["metrics"]
        lines.append(
            "- `%s`: raw ECE `%0.4f` -> Platt `%0.4f` -> isotonic `%0.4f`; "
            "raw Brier `%0.4f` -> isotonic `%0.4f`"
            % (
                ds,
                m["raw"]["ece"],
                m["platt"]["ece"],
                m["isotonic"]["ece"],
                m["raw"]["brier"],
                m["isotonic"]["brier"],
            )
        )
    lines.append("")
    lines.append("## 9. Oracle Probe Upper Bound")
    for ds in DATASETS:
        row = oracle[ds]
        lines.append(
            "- `%s`: oracle-answerable probes `%d`; controller recall of oracle probe `%0.4f`; "
            "contain `%0.4f` -> `%0.4f`; mean tokens `%0.1f` -> `%0.1f` (approx)"
            % (
                ds,
                row["oracle_answerable_from_probe"],
                row["controller_recall_of_oracle_probe"],
                row["current_mean_contain"],
                row["oracle_mean_contain"],
                row["current_mean_tokens"],
                row["oracle_mean_tokens_approx"],
            )
        )
    lines.append("")
    lines.append("## 11. Tau Transfer")
    lines.append(
        "- source sweep: MuSiQue-200 selects `tau=%0.2f` with contain `%0.3f` and mean tokens `%0.1f`"
        % (
            tau_transfer["source_selection"]["tau"],
            tau_transfer["source_selection"]["contain"],
            tau_transfer["source_selection"]["mean_tokens"],
        )
    )
    for ds in DATASETS:
        best = tau_transfer["transfer"][ds]["selected_tau"]
        default = tau_transfer["transfer"][ds]["default_tau"]
        lines.append(
            "- `%s`: transfer `%0.2f` gives contain `%0.4f`, F1 `%0.4f`, EM `%0.4f`, "
            "mean tokens `%0.1f` (approx) vs default `%0.2f` contain `%0.4f`, tokens `%0.1f`"
            % (
                ds,
                best["tau"],
                best["contain"],
                best["token_f1"],
                best["norm_em"],
                best["mean_tokens_approx"],
                default["tau"],
                default["contain"],
                default["mean_tokens_approx"],
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    latest_rel = (ROOT / "results/.sufficiency_1000q_latest").read_text(encoding="utf-8").strip()
    results_root = ROOT / latest_rel
    output_dir = results_root / "cpu_followups"
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_dataset = {
        dataset: load_dataset_records(dataset, results_root) for dataset in DATASETS
    }

    calibration = {}
    oracle = {}
    for dataset, records in records_by_dataset.items():
        calibration[dataset] = calibration_analysis(records)
        oracle[dataset] = oracle_analysis(records)

    ablation_summary = json.loads(
        (ROOT / "results/abl_musique200_20260419_073534/ablation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    selected_tau, source_rows = choose_transfer_tau(ablation_summary)
    transfer = {}
    for dataset, records in records_by_dataset.items():
        rows = [offline_tau_counterfactual(records, tau) for tau in (0.5, 0.6, 0.7)]
        by_tau = {row["tau"]: row for row in rows}
        transfer[dataset] = {
            "grid": rows,
            "selected_tau": by_tau[selected_tau["tau"]],
            "default_tau": by_tau[0.7],
        }

    tau_transfer = {
        "source_dataset": "musique_200_ablation",
        "source_selection": selected_tau,
        "source_grid": source_rows,
        "transfer": transfer,
        "notes": {
            "quality": "exact post-hoc for tau <= 0.7 using logged proposed probe answers and logged final answers",
            "tokens": "approximate only when lowering tau converts a recurse_after_probe example into answer_from_probe; uses dataset median probe-answer tail tokens",
        },
    }

    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (output_dir / "oracle_probe_upper_bound.json").write_text(
        json.dumps(oracle, indent=2), encoding="utf-8"
    )
    (output_dir / "tau_transfer.json").write_text(
        json.dumps(tau_transfer, indent=2), encoding="utf-8"
    )
    write_summary(output_dir, calibration, oracle, tau_transfer)

    print("wrote", output_dir)


if __name__ == "__main__":
    main()
