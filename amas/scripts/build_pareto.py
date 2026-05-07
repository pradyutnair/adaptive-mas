"""Cross-dataset Pareto + per-dataset matched-subset comparison.

Joins HERA-repro / SAS-matched / AMAS predictions by qid for each dataset.
Outputs:
  - results/p1_full/headline_table.md    (per-dataset 5-way table)
  - results/p1_full/pareto_data.json     (rows for plotting)
  - results/p1_full/pareto.png           (matplotlib Pareto plot)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


HERA_PRED = {
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


def build_rows(ds: str, *, amas_root: str, sas_root: str) -> list[dict]:
    amas_runs = {}
    for gate in ("off", "bayesian", "conformal"):
        amas_runs[gate] = index_by(load_jsonl(f"{amas_root}/{ds}_{gate}/predictions.jsonl"))
    sas = index_by(load_jsonl(f"{sas_root}/{ds}_full/predictions.jsonl"))
    if not sas:
        sas = index_by(load_jsonl(f"{sas_root}/{ds}_200/predictions.jsonl"))
    hera = index_by(load_jsonl(HERA_PRED.get(ds, "")), key="id")

    qids = sorted(set(amas_runs["off"].keys()))
    if not qids:
        return []

    methods = [
        ("HERA-repro", hera, "tokens"),
        ("SAS-matched", sas, "tokens"),
        ("AMAS-off", amas_runs["off"], "total_tokens"),
        ("AMAS-bayesian", amas_runs["bayesian"], "total_tokens"),
        ("AMAS-conformal", amas_runs["conformal"], "total_tokens"),
    ]
    rows = []
    for name, src, tk in methods:
        present = [src[q] for q in qids if q in src]
        if not present:
            continue
        em = mean(float(p.get("em", 0)) for p in present)
        f1 = mean(float(p.get("f1", 0)) for p in present)
        ac = mean(float(p.get("acc", p.get("contain", 0))) for p in present)
        tok = mean(float(p.get(tk, 0)) for p in present)
        rows.append({"dataset": ds, "method": name, "n": len(present),
                     "em": em, "f1": f1, "acc": ac, "tokens": tok})
    return rows


def is_pareto(rows: list[dict], metric_key: str = "em") -> set[int]:
    """Indices on Pareto frontier (max metric, min tokens)."""
    pareto = set()
    for i, r in enumerate(rows):
        dominated = False
        for j, s in enumerate(rows):
            if i == j:
                continue
            if s["tokens"] <= r["tokens"] and s[metric_key] >= r[metric_key] and (
                s["tokens"] < r["tokens"] or s[metric_key] > r[metric_key]
            ):
                dominated = True
                break
        if not dominated:
            pareto.add(i)
    return pareto


def write_md(all_rows: list[dict], out_path: str | Path) -> None:
    by_ds = {}
    for r in all_rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    lines = ["# AMAS vs Baselines — Matched-Subset Comparison\n"]
    for ds in sorted(by_ds.keys()):
        rows = by_ds[ds]
        n = rows[0]["n"]
        lines.append(f"\n## {ds} (n={n})\n")
        lines.append("| Method | EM | F1 | Acc | Tokens | Pareto |")
        lines.append("|---|---|---|---|---|---|")
        pareto = is_pareto(rows, "em")
        for i, r in enumerate(rows):
            mk = "★" if i in pareto else ""
            lines.append(
                f"| {r['method']} | {r['em']:.3f} | {r['f1']:.3f} | "
                f"{r['acc']:.3f} | {r['tokens']:.0f} | {mk} |"
            )
        # AMAS Pareto wins highlighted
        amas_pareto = [rows[i]["method"] for i in pareto if rows[i]["method"].startswith("AMAS")]
        if amas_pareto:
            lines.append(f"\n*Pareto: {', '.join(amas_pareto)}*")
    Path(out_path).write_text("\n".join(lines))
    print("\n".join(lines))


def plot_pareto(all_rows: list[dict], out_path: str | Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping plot", file=sys.stderr)
        return

    by_ds = {}
    for r in all_rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    n_ds = len(by_ds)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=False)
    if n_ds == 1:
        axes = [axes]
    color_map = {"HERA-repro": "tab:red", "SAS-matched": "tab:gray",
                 "AMAS-off": "tab:blue", "AMAS-bayesian": "tab:green",
                 "AMAS-conformal": "tab:orange"}
    for ax, (ds, rows) in zip(axes, sorted(by_ds.items())):
        for r in rows:
            ax.scatter(r["tokens"], r["em"], s=130,
                       color=color_map.get(r["method"], "k"),
                       label=r["method"])
            ax.annotate(r["method"], (r["tokens"], r["em"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("avg tokens / question")
        ax.set_ylabel("EM")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"plot saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amas-root", default="results/p1_full")
    ap.add_argument("--sas-root", default="results/sas_matched")
    ap.add_argument("--datasets", nargs="+",
                     default=["musique", "hotpotqa", "2wikimultihop", "bamboogle"])
    ap.add_argument("--out-md", default="results/p1_full/headline_table.md")
    ap.add_argument("--out-png", default="results/p1_full/pareto.png")
    ap.add_argument("--out-json", default="results/p1_full/pareto_data.json")
    args = ap.parse_args()

    all_rows = []
    for ds in args.datasets:
        rows = build_rows(ds, amas_root=args.amas_root, sas_root=args.sas_root)
        all_rows.extend(rows)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(all_rows, indent=2))
    write_md(all_rows, args.out_md)
    plot_pareto(all_rows, args.out_png)


if __name__ == "__main__":
    main()
