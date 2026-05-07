"""Full 1000q × 4 datasets matched-subset aggregation + paired bootstrap CIs.

Joins HERA-repro / SAS-matched / AMAS predictions by qid for each dataset.
Outputs:
  results/p1_full/aggregate.json     full table per dataset
  results/p1_full/aggregate.md       markdown headline
  results/p1_full/pareto.png         4-panel Pareto plot
  results/p1_full/per_profile.json   per-profile breakdown using annotations
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any


HERA_PRED = {
    "musique": "/local/yzheng/pnair/workspace/adaptive-mas/amas/results/hera_renormalized/predictions_musique.jsonl",
    "hotpotqa": "/local/yzheng/pnair/workspace/adaptive-mas/amas/results/hera_renormalized/predictions_hotpotqa.jsonl",
    "2wikimultihop": "/local/yzheng/pnair/workspace/adaptive-mas/amas/results/hera_renormalized/predictions_2wikimultihop.jsonl",
    "bamboogle": "/local/yzheng/pnair/workspace/adaptive-mas/amas/results/hera_renormalized/predictions_bamboogle.jsonl",
}

ANNOT_PATH = {
    "musique": "/local/yzheng/pnair/workspace/reproduction/hera/data/annotations/annot_test_musique.jsonl",
    "hotpotqa": "/local/yzheng/pnair/workspace/reproduction/hera/data/annotations/annot_test_hotpotqa.jsonl",
    "2wikimultihop": "/local/yzheng/pnair/workspace/reproduction/hera/data/annotations/annot_test_2wikimultihop.jsonl",
    "bamboogle": "/local/yzheng/pnair/workspace/reproduction/hera/data/annotations/annot_test_bamboogle.jsonl",
}


def load_jsonl(p: str | Path) -> list[dict]:
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def index_by(rows: list[dict], key: str = "qid", fallback_keys: list[str] = ("id",)) -> dict[str, dict]:
    out = {}
    for r in rows:
        k = str(r.get(key, ""))
        if not k:
            for fb in fallback_keys:
                k = str(r.get(fb, ""))
                if k:
                    break
        if k:
            out[k] = r
    return out


def paired_bootstrap_diff(a: list[float], b: list[float], n_resamples: int = 1000,
                          seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean_diff, ci_lo, ci_hi) for a-b paired bootstrap."""
    rng = random.Random(seed)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = [a[i] - b[i] for i in range(n)]
    boot = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    return float(mean(diffs)), float(boot[int(0.025 * n_resamples)]), float(boot[int(0.975 * n_resamples)])


def run_dataset(ds: str, *, amas_root: str, sas_root: str) -> dict[str, Any]:
    amas_runs = {}
    for gate in ("off", "bayesian", "conformal"):
        path = f"{amas_root}/{ds}_{gate}/predictions.jsonl"
        amas_runs[gate] = index_by(load_jsonl(path))
    sas = index_by(load_jsonl(f"{sas_root}/{ds}_full/predictions.jsonl"))
    hera = index_by(load_jsonl(HERA_PRED.get(ds, "")), key="id")

    qids = sorted(set(amas_runs["off"].keys()))
    if not qids:
        return {}

    methods = [
        ("HERA-repro", hera, "tokens"),
        ("SAS-matched", sas, "tokens"),
        ("AMAS-off", amas_runs["off"], "total_tokens"),
        ("AMAS-bayesian", amas_runs["bayesian"], "total_tokens"),
        ("AMAS-conformal", amas_runs["conformal"], "total_tokens"),
    ]
    out_rows = []
    per_q = {}
    for name, src, tk in methods:
        rows_em, rows_f1, rows_acc, rows_tok = [], [], [], []
        for q in qids:
            r = src.get(q, {})
            if not r:
                rows_em.append(0); rows_f1.append(0); rows_acc.append(0); rows_tok.append(0)
                continue
            rows_em.append(float(r.get("em", 0)))
            rows_f1.append(float(r.get("f1", 0)))
            rows_acc.append(float(r.get("acc", r.get("contain", 0))))
            rows_tok.append(float(r.get(tk, 0)))
        out_rows.append({
            "method": name, "n": len(qids),
            "em": mean(rows_em), "f1": mean(rows_f1), "acc": mean(rows_acc),
            "tokens": mean(rows_tok),
        })
        per_q[name] = {"em": rows_em, "f1": rows_f1, "acc": rows_acc, "tokens": rows_tok}

    # Paired bootstrap CIs vs HERA-repro for each AMAS variant
    base = per_q.get("HERA-repro", {})
    ci_table = {}
    if base:
        for name in ("AMAS-off", "AMAS-bayesian", "AMAS-conformal", "SAS-matched"):
            if name not in per_q:
                continue
            for metric in ("em", "f1", "acc"):
                d, lo, hi = paired_bootstrap_diff(per_q[name][metric], base[metric])
                ci_table[f"{name}_vs_HERA_{metric}"] = {"diff": d, "ci_lo": lo, "ci_hi": hi}

    return {"dataset": ds, "n": len(qids), "rows": out_rows, "ci": ci_table,
            "per_q": per_q, "qids": qids}


def per_profile_breakdown(ds: str, ds_data: dict, profile_path: str) -> dict:
    """Group EM/Acc by reasoning_type from annotations."""
    annots = {}
    if Path(profile_path).exists():
        for line in Path(profile_path).read_text().splitlines():
            try:
                d = json.loads(line)
                annots[str(d["id"])] = d.get("reasoning_type", "bridge")
            except Exception:
                pass
    qids = ds_data.get("qids", [])
    per_q = ds_data.get("per_q", {})
    by_profile = {}
    for i, q in enumerate(qids):
        prof = annots.get(q, "bridge")
        by_profile.setdefault(prof, []).append(i)

    profile_table = {}
    for prof, idxs in by_profile.items():
        if not idxs:
            continue
        profile_table[prof] = {"n": len(idxs)}
        for method in per_q:
            em_arr = per_q[method]["em"]
            acc_arr = per_q[method]["acc"]
            tok_arr = per_q[method]["tokens"]
            profile_table[prof][method] = {
                "em": mean(em_arr[i] for i in idxs),
                "acc": mean(acc_arr[i] for i in idxs),
                "tokens": mean(tok_arr[i] for i in idxs),
            }
    return profile_table


def write_md(all_data: dict, out_path: str | Path) -> None:
    lines = ["# AMAS Headline Results — Full 1000q × 4 datasets\n",
             "Matched-subset comparison (qid-joined). HERA-repro is the paper-faithful reproduction.\n"]
    for ds in ("musique", "hotpotqa", "2wikimultihop", "bamboogle"):
        d = all_data.get(ds, {})
        if not d:
            continue
        n = d["n"]
        lines.append(f"\n## {ds} (n={n})\n")
        lines.append("| Method | EM | F1 | Acc | Tokens |")
        lines.append("|---|---|---|---|---|")
        for r in d["rows"]:
            lines.append(f"| {r['method']} | {r['em']:.3f} | {r['f1']:.3f} | "
                         f"{r['acc']:.3f} | {r['tokens']:.0f} |")
        lines.append("\n**Paired bootstrap CIs (95%) vs HERA-repro:**\n")
        lines.append("| Comparison | EM Δ | 95% CI | F1 Δ | 95% CI | Acc Δ | 95% CI |")
        lines.append("|---|---|---|---|---|---|---|")
        for cmp_name in ("AMAS-off", "AMAS-bayesian", "AMAS-conformal", "SAS-matched"):
            row = [cmp_name + " - HERA"]
            for metric in ("em", "f1", "acc"):
                k = f"{cmp_name}_vs_HERA_{metric}"
                if k in d["ci"]:
                    c = d["ci"][k]
                    row.append(f"{c['diff']:+.3f}")
                    row.append(f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]")
                else:
                    row.append("n/a"); row.append("")
            lines.append("| " + " | ".join(row) + " |")
    Path(out_path).write_text("\n".join(lines))
    print("\n".join(lines))


def plot_pareto(all_data: dict, out_path: str | Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    datasets = [ds for ds in ("musique", "hotpotqa", "2wikimultihop", "bamboogle") if ds in all_data]
    n_ds = len(datasets)
    if not n_ds:
        return
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 4.0), sharey=False)
    if n_ds == 1:
        axes = [axes]
    color_map = {"HERA-repro": "tab:red", "SAS-matched": "tab:gray",
                 "AMAS-off": "tab:blue", "AMAS-bayesian": "tab:green",
                 "AMAS-conformal": "tab:orange"}
    for ax, ds in zip(axes, datasets):
        rows = all_data[ds]["rows"]
        for r in rows:
            ax.scatter(r["tokens"], r["em"], s=140,
                       color=color_map.get(r["method"], "k"),
                       label=r["method"], edgecolors="black", linewidths=0.5)
            ax.annotate(r["method"], (r["tokens"], r["em"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_xlabel("avg tokens / question")
        ax.set_ylabel("EM")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set(); uh, ul = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uh.append(h); ul.append(l)
    fig.legend(uh, ul, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amas-root", default="results/p1_full")
    ap.add_argument("--sas-root", default="results/sas_matched")
    ap.add_argument("--out-dir", default="results/p1_full")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    profile_data = {}
    for ds in ("musique", "hotpotqa", "2wikimultihop", "bamboogle"):
        d = run_dataset(ds, amas_root=args.amas_root, sas_root=args.sas_root)
        if d:
            all_data[ds] = d
            profile_data[ds] = per_profile_breakdown(ds, d, ANNOT_PATH.get(ds, ""))

    # strip per_q from json output (large)
    aggregate_json = {ds: {k: v for k, v in d.items() if k != "per_q"}
                      for ds, d in all_data.items()}
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate_json, indent=2))
    (out_dir / "per_profile.json").write_text(json.dumps(profile_data, indent=2))
    write_md(all_data, out_dir / "aggregate.md")
    plot_pareto(all_data, out_dir / "pareto.png")


if __name__ == "__main__":
    main()
