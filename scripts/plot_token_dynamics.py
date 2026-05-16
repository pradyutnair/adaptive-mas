#!/usr/bin/env python3
"""Render per-dataset LOWESS plots from `token_dynamics.jsonl`.

This reproduces the HERA Section 5.3.1 token-consumption-dynamics figure:
a LOWESS-smoothed trajectory of `avg_tokens` over the TF-GRPO learning
step, per dataset, so you can see the exploration-spike -> exploitation-
decline -> plateau curve as the experience library matures.

The training script (`scripts/run_hera_train.py`) writes the source jsonl
to `<output_dir>/token_dynamics.jsonl`. Each line is one record:
  {"epoch": int, "learning_step": int, "scope": "all"|"dataset",
   "dataset": str (only when scope=="dataset"), "rows": int,
   "library_size": int, "avg_tokens": float, "avg_f1": float,
   "avg_contain": float, "avg_em": float, "avg_token_efficiency": float}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def lowess(ys: np.ndarray, xs: np.ndarray, frac: float = 0.4, it: int = 1) -> np.ndarray:
    """Minimal LOWESS (Cleveland, 1979) using only numpy.

    Returns an (n, 2) array sorted by x: column 0 is x, column 1 is smoothed y.
    Matches the surface of statsmodels.nonparametric.smoothers_lowess.lowess
    with return_sorted=True for our plotting needs.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    n = xs.size
    if n == 0:
        return np.empty((0, 2))
    r = int(np.ceil(frac * n))
    r = max(2, min(r, n))
    weights = np.ones(n)
    smoothed = np.empty(n)
    for _ in range(max(it, 0) + 1):
        for i in range(n):
            d = np.abs(xs - xs[i])
            idx = np.argpartition(d, r - 1)[:r]
            dmax = d[idx].max() or 1.0
            w = np.clip(1.0 - (d[idx] / dmax) ** 3, 0.0, None) ** 3
            w = w * weights[idx]
            x_w, y_w = xs[idx], ys[idx]
            sw = w.sum() or 1.0
            mx = (w * x_w).sum() / sw
            my = (w * y_w).sum() / sw
            num = (w * (x_w - mx) * (y_w - my)).sum()
            den = (w * (x_w - mx) ** 2).sum() or 1.0
            b = num / den
            a = my - b * mx
            smoothed[i] = a + b * xs[i]
        residuals = ys - smoothed
        s = np.median(np.abs(residuals)) or 1.0
        weights = np.clip(1.0 - (residuals / (6.0 * s)) ** 2, 0.0, None) ** 2
    return np.column_stack((xs, smoothed))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "compiled" / "hera" / "token_dynamics.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "compiled" / "hera" / "plots"

# Aliases used in the eval/baseline naming convention.
DATASET_ALIASES = {
    "hotpotqa": "hotpotqa",
    "hotpot": "hotpotqa",
    "2wiki": "2wikimultihop",
    "2wikimultihop": "2wikimultihop",
    "musique": "musique",
    "bamboogle": "bamboogle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOWESS plots of TF-GRPO token dynamics.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to token_dynamics.jsonl from a training run.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to write PNGs into.")
    parser.add_argument("--frac", type=float, default=0.4,
                        help="LOWESS smoothing fraction (0<frac<=1). Lower -> wigglier curve.")
    parser.add_argument("--metric", default="avg_tokens",
                        choices=("avg_tokens", "avg_f1", "avg_contain", "avg_em",
                                 "avg_token_efficiency", "library_size"),
                        help="Metric to smooth and plot.")
    parser.add_argument("--per-epoch", action="store_true",
                        help="Aggregate to one point per (dataset, epoch) instead of per learning step.")
    return parser.parse_args()


def canonical_dataset(name: str) -> str:
    return DATASET_ALIASES.get((name or "").strip().lower(), name)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing token dynamics file: {path}")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_dataset_rows(rows: list[dict]) -> dict[str, list[dict]]:
    by_ds: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("scope") != "dataset":
            continue
        ds = canonical_dataset(str(row.get("dataset", "")))
        if not ds:
            continue
        by_ds.setdefault(ds, []).append(row)
    for ds in by_ds:
        by_ds[ds].sort(key=lambda r: (int(r.get("learning_step", 0)), int(r.get("epoch", 0))))
    return by_ds


def aggregate_per_epoch(records: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    by_epoch: dict[int, list[float]] = {}
    for r in records:
        epoch = int(r.get("epoch", 0))
        val = r.get(metric)
        if val is None:
            continue
        by_epoch.setdefault(epoch, []).append(float(val))
    epochs = sorted(by_epoch)
    xs = np.asarray(epochs, dtype=float)
    ys = np.asarray([float(np.mean(by_epoch[e])) for e in epochs])
    return xs, ys


def fit_lowess(xs: np.ndarray, ys: np.ndarray, frac: float) -> np.ndarray:
    if len(xs) < 2:
        return np.column_stack((xs, ys))
    frac = max(min(frac, 1.0), 2.0 / max(len(xs), 2))
    return lowess(ys, xs, frac=frac, it=1)


def plot_per_dataset(
    by_ds: dict[str, list[dict]],
    output_dir: Path,
    frac: float,
    metric: str,
    per_epoch: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ds, records in by_ds.items():
        if per_epoch:
            xs, ys = aggregate_per_epoch(records, metric)
            xlabel = "epoch"
        else:
            xs = np.asarray([float(r.get("learning_step", 0)) for r in records])
            ys = np.asarray([float(r.get(metric, 0.0) or 0.0) for r in records])
            xlabel = "learning step"
        if not len(xs):
            continue
        smoothed = fit_lowess(xs, ys, frac=frac)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(xs, ys, s=14, alpha=0.35, label="raw")
        ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.0,
                label=f"LOWESS (frac={frac:.2f})")
        ax.set_title(f"TF-GRPO token dynamics: {ds} ({metric})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        out_path = output_dir / f"token_dynamics_{ds}_{metric}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        written.append(out_path)
    return written


def plot_combined(
    by_ds: dict[str, list[dict]],
    output_dir: Path,
    frac: float,
    metric: str,
    per_epoch: bool,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for ds, records in by_ds.items():
        if per_epoch:
            xs, ys = aggregate_per_epoch(records, metric)
            xlabel = "epoch"
        else:
            xs = np.asarray([float(r.get("learning_step", 0)) for r in records])
            ys = np.asarray([float(r.get(metric, 0.0) or 0.0) for r in records])
            xlabel = "learning step"
        if not len(xs):
            continue
        smoothed = fit_lowess(xs, ys, frac=frac)
        ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.0, label=ds)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title(f"TF-GRPO {metric} dynamics (LOWESS, frac={frac:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = output_dir / f"token_dynamics_combined_{metric}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    by_ds = group_dataset_rows(rows)
    if not by_ds:
        raise SystemExit("No per-dataset rows in input. Re-run training so scope='dataset' rows are emitted.")

    output_dir = Path(args.output_dir)
    written = plot_per_dataset(by_ds, output_dir, args.frac, args.metric, args.per_epoch)
    combined = plot_combined(by_ds, output_dir, args.frac, args.metric, args.per_epoch)

    print(f"Wrote {len(written)} per-dataset plot(s) to {output_dir}:")
    for p in written:
        print(f"  {p}")
    if combined is not None:
        print(f"Combined overlay: {combined}")


if __name__ == "__main__":
    main()
