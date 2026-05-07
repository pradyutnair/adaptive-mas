"""5-way matched-subset comparison: HERA-repro / SAS-matched / AMAS-{off,bay,conf}.

Joins predictions by qid. Outputs paired EM/F1/tokens table + Pareto plot data.

Usage:
  python scripts/compare_5way.py --dataset musique --n 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


HERA_REPRO_PRED = {
    "musique": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_musique.jsonl",
    "hotpotqa": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_hotpotqa.jsonl",
    "2wikimultihop": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_2wikimultihop.jsonl",
    "bamboogle": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_bamboogle.jsonl",
}


def load_jsonl(p: str | Path) -> list[dict]:
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def index_by(rows: list[dict], key: str = "qid") -> dict[str, dict]:
    out = {}
    for r in rows:
        k = str(r.get(key, "")) or str(r.get("id", ""))
        if k:
            out[k] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--amas-root", default="results/p1_200")
    ap.add_argument("--sas-root", default="results/sas_matched")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = args.dataset
    amas_runs = {}
    for gate in ("off", "bayesian", "conformal"):
        path = f"{args.amas_root}/{ds}_{gate}/predictions.jsonl"
        amas_runs[gate] = index_by(load_jsonl(path))
    sas_path = f"{args.sas_root}/{ds}_200/predictions.jsonl"
    sas = index_by(load_jsonl(sas_path))
    hera = index_by(load_jsonl(HERA_REPRO_PRED.get(ds, "")), key="id")

    # qids = intersection of AMAS-off and HERA (HERA has all 1000q; AMAS has 200q matched-seed).
    qids = sorted(set(amas_runs["off"].keys()))
    n = len(qids)
    if not n:
        print("no AMAS qids found; ensure runs completed at", args.amas_root)
        return

    print(f"Dataset: {ds}  matched n={n}\n")

    rows = []
    methods = [
        ("HERA-repro", hera, "tokens"),
        ("SAS-matched", sas, "tokens"),
        ("AMAS-off", amas_runs["off"], "total_tokens"),
        ("AMAS-bayesian", amas_runs["bayesian"], "total_tokens"),
        ("AMAS-conformal", amas_runs["conformal"], "total_tokens"),
    ]

    print(f"{'method':<18} {'n':>4} {'EM':>6} {'F1':>6} {'Acc':>6} {'tokens':>8}")
    print("-" * 60)
    for name, src, tk in methods:
        present = [src[q] for q in qids if q in src]
        if not present:
            print(f"{name:<18} {'n/a':>4}")
            continue
        em = mean(float(p.get("em", 0)) for p in present)
        f1 = mean(float(p.get("f1", 0)) for p in present)
        ac = mean(float(p.get("acc", p.get("contain", 0))) for p in present)
        tok = mean(float(p.get(tk, 0)) for p in present) or 0.0
        n_p = len(present)
        rows.append({"method": name, "n": n_p, "em": em, "f1": f1, "acc": ac, "tokens": tok})
        print(f"{name:<18} {n_p:>4} {em:.3f} {f1:.3f} {ac:.3f} {tok:>8.0f}")

    # Pareto: sort by tokens, keep dominant points
    print("\nPareto (sorted by tokens asc):")
    for r in sorted(rows, key=lambda x: x["tokens"]):
        print(f"  {r['method']:<18} tokens={r['tokens']:>7.0f}  EM={r['em']:.3f}  F1={r['f1']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
