"""Build AMAS thesis report: plots + master doc.

Run on node409 inside amas/. Outputs to reports/.
Aesthetic upgrade: Okabe-Ito palette, clean typography, Pareto frontier overlay.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------- Style ----------
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#222",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#222",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#444",
    "ytick.color": "#444",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

ROOT = Path(".")
REPORTS = ROOT / "reports"
PLOTS = REPORTS / "plots"
DATA = REPORTS / "data"
PLOTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# ---------- Load all data ----------
agg = json.loads((ROOT / "results/p1_full/aggregate.json").read_text())
prof = json.loads((ROOT / "results/p1_full/per_profile.json").read_text())
alpha_sw = json.loads((ROOT / "results/alpha_sweep/alpha_sweep.json").read_text())
tau_b_sw = json.loads((ROOT / "results/tau_b_sweep_v2/tau_b_sweep.json").read_text())
lam_sw = json.loads((ROOT / "results/lambda_sweep/lambda_sweep.json").read_text())
calib = json.loads((ROOT / "results/route_a_calibration.json").read_text())
hera_renorm = json.loads((ROOT / "results/hera_renormalized/summary_all.json").read_text())

for src in [
    "results/p1_full/aggregate.json",
    "results/p1_full/aggregate.md",
    "results/p1_full/per_profile.json",
    "results/alpha_sweep/alpha_sweep.json",
    "results/tau_b_sweep_v2/tau_b_sweep.json",
    "results/lambda_sweep/lambda_sweep.json",
    "results/route_a_calibration.json",
    "results/hera_renormalized/summary_all.json",
]:
    p = ROOT / src
    if p.exists():
        shutil.copy2(p, DATA / p.name)

DATASETS = ["musique", "hotpotqa", "2wikimultihop", "bamboogle"]
DATASET_LABELS = {
    "musique": "MuSiQue",
    "hotpotqa": "HotpotQA",
    "2wikimultihop": "2WikiMultihop",
    "bamboogle": "Bamboogle",
}
METHODS = ["HERA-repro", "SAS-matched", "AMAS-off", "AMAS-bayesian", "AMAS-conformal"]

# Okabe-Ito accessible palette
PAL = {
    "HERA-repro":     "#444444",  # dark gray (baseline)
    "SAS-matched":    "#999999",  # light gray (baseline)
    "AMAS-off":       "#0072B2",  # blue
    "AMAS-bayesian":  "#D55E00",  # vermillion
    "AMAS-conformal": "#009E73",  # bluish green (hero)
}
MARK = {
    "HERA-repro":     "o",
    "SAS-matched":    "s",
    "AMAS-off":       "D",
    "AMAS-bayesian":  "v",
    "AMAS-conformal": "*",
}
SIZE = {
    "HERA-repro":     130,
    "SAS-matched":    110,
    "AMAS-off":       130,
    "AMAS-bayesian":  130,
    "AMAS-conformal": 260,  # hero method, larger star
}


def _row(ds, method):
    for r in agg[ds]["rows"]:
        if r["method"] == method:
            return r
    return None


def _pareto_front(points):
    """Return non-dominated subset (max y, min x). points: list of (x, y, label)."""
    pts = sorted(points, key=lambda t: (t[0], -t[1]))
    front = []
    best_y = -np.inf
    for x, y, lbl in pts:
        if y > best_y:
            front.append((x, y, lbl))
            best_y = y
    return front


def _scatter_methods(ax, ds, ymetric):
    pts = []
    for m in METHODS:
        r = _row(ds, m)
        if r is None:
            continue
        x, y = r["tokens"], r[ymetric]
        ax.scatter(x, y, c=PAL[m], marker=MARK[m], s=SIZE[m],
                   edgecolor="white", linewidth=1.2, zorder=4, label=m)
        pts.append((x, y, m))
    # Pareto frontier (min tokens, max y)
    front = _pareto_front(pts)
    if len(front) >= 2:
        fx = [p[0] for p in front]; fy = [p[1] for p in front]
        ax.plot(fx, fy, ls="--", color="#888", lw=1.0, alpha=0.7, zorder=2,
                label="Pareto frontier" if ymetric == "em" and ds == DATASETS[0] else None)


# ---------- Plot 1+2: Pareto EM/Acc ----------
def pareto_panel(ymetric, ylabel, fname, title):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), sharey=False)
    for i, ds in enumerate(DATASETS):
        ax = axes[i // 2, i % 2]
        _scatter_methods(ax, ds, ymetric)
        n = agg[ds]["n"]
        ax.set_xlabel("Avg tokens per question")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{DATASET_LABELS[ds]}  (n={n})")
        ax.grid(axis="both", alpha=0.45)
        ax.set_axisbelow(True)
        # x in thousands
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    # single legend on figure
    handles = [plt.scatter([], [], c=PAL[m], marker=MARK[m], s=SIZE[m],
                           edgecolor="white", linewidth=1.0, label=m) for m in METHODS]
    handles.append(plt.Line2D([0], [0], ls="--", color="#888", label="Pareto frontier"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle(title, fontsize=13.5, y=0.995, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(PLOTS / fname)
    plt.close(fig)


pareto_panel("em", "Exact Match",
             "pareto_em.png",
             "AMAS Pareto: EM vs Tokens (1000q × 4 datasets, Het regime)")
pareto_panel("acc", "Accuracy (contain)",
             "pareto_acc.png",
             "AMAS Pareto: Accuracy vs Tokens (1000q × 4 datasets, Het regime)")
pareto_panel("f1", "F1",
             "pareto_f1.png",
             "AMAS Pareto: F1 vs Tokens (1000q × 4 datasets, Het regime)")

# ---------- Plot 3: F1 grouped bar ----------
fig, ax = plt.subplots(figsize=(10.5, 5.0))
x = np.arange(len(DATASETS))
w = 0.16
for j, m in enumerate(METHODS):
    vals = [_row(ds, m)["f1"] if _row(ds, m) else 0 for ds in DATASETS]
    bars = ax.bar(x + (j - 2) * w, vals, w, color=PAL[m],
                  edgecolor="white", linewidth=0.8, label=m, zorder=3)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#222")
ax.set_xticks(x); ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
ax.set_ylabel("F1")
ax.set_title("F1 by dataset and method", loc="left", pad=12)
ax.grid(axis="y", alpha=0.45); ax.set_axisbelow(True)
ax.legend(ncol=5, fontsize=9.0, loc="upper center",
          bbox_to_anchor=(0.5, -0.10))
ax.set_ylim(0, max(_row(d, m)["f1"] for d in DATASETS for m in METHODS if _row(d, m)) * 1.18)
fig.tight_layout()
fig.savefig(PLOTS / "f1_bars.png")
plt.close(fig)

# ---------- Plot 4: alpha sweep ----------
def styled_sweep(rows, xkey, xlabel, fname, title, logx=False, mark_best=None):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    xs = [r[xkey] for r in rows]
    plot_fn = lambda ax: ax.semilogx if logx else ax.plot
    # Panel 1: SAS rate
    ax = axes[0]
    (ax.semilogx if logx else ax.plot)(xs, [r["sas_rate"] for r in rows],
            marker="o", lw=1.6, ms=8, color="#009E73",
            markeredgecolor="white", markeredgewidth=1.2)
    ax.set_xlabel(xlabel); ax.set_ylabel("SAS-commit rate")
    ax.set_title("SAS rate", loc="left"); ax.grid(alpha=0.45); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)
    # Panel 2: EM/Acc/F1
    ax = axes[1]
    (ax.semilogx if logx else ax.plot)(xs, [r["em"] for r in rows],
            marker="o", lw=1.6, ms=8, color="#0072B2", label="EM",
            markeredgecolor="white", markeredgewidth=1.2)
    if "f1" in rows[0]:
        (ax.semilogx if logx else ax.plot)(xs, [r["f1"] for r in rows],
                marker="s", lw=1.6, ms=7, color="#D55E00", label="F1",
                markeredgecolor="white", markeredgewidth=1.2)
    if "acc" in rows[0]:
        (ax.semilogx if logx else ax.plot)(xs, [r["acc"] for r in rows],
                marker="^", lw=1.6, ms=7, color="#444", label="Acc",
                markeredgecolor="white", markeredgewidth=1.2)
    ax.set_xlabel(xlabel); ax.set_ylabel("Score")
    ax.set_title("Quality", loc="left"); ax.grid(alpha=0.45); ax.set_axisbelow(True)
    ax.legend()
    # Panel 3: tokens or sas-error
    ax = axes[2]
    if "sas_error_rate" in rows[0]:
        (ax.semilogx if logx else ax.plot)(xs, [r["sas_error_rate"] for r in rows],
                marker="o", lw=1.6, ms=8, color="#D55E00",
                markeredgecolor="white", markeredgewidth=1.2)
        ax.set_ylabel("SAS-error rate")
        ax.set_title("SAS-error (lower = better)", loc="left")
        ax.set_ylim(-0.05, 1.05)
    else:
        (ax.semilogx if logx else ax.plot)(xs, [r["avg_tokens"] for r in rows],
                marker="o", lw=1.6, ms=8, color="#D55E00",
                markeredgecolor="white", markeredgewidth=1.2)
        ax.set_ylabel("Avg tokens")
        ax.set_title("Cost", loc="left")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
    ax.set_xlabel(xlabel); ax.grid(alpha=0.45); ax.set_axisbelow(True)
    if mark_best is not None and mark_best in xs:
        idx = xs.index(mark_best)
        for a in axes:
            a.axvline(mark_best, color="#888", ls=":", lw=1.0, alpha=0.7)
    fig.suptitle(title, fontsize=12.5, fontweight="semibold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / fname)
    plt.close(fig)


styled_sweep(alpha_sw, "alpha", r"$\alpha$ (target SAS-error)",
             "alpha_sweep.png",
             "Conformal Gate (Route A) — α-sweep on val 100q MuSiQue",
             mark_best=0.05)

styled_sweep(tau_b_sw, "tau_b", r"$\tau_b$ (commit threshold)",
             "tau_b_sweep_v2.png",
             "Bayesian Gate v2 (top-score + entropy) — bimodal, no usable knee")

styled_sweep(lam_sw, "lambda", r"$\lambda$",
             "lambda_sweep.png",
             "Bayesian Gate v1 (entropy-only) — degenerate at G=1 probe (H ≡ 0)",
             logx=True)

# ---------- Plot 5: per-profile heatmap ----------
prof_ds = "musique"
all_profs = list(prof[prof_ds].keys())
profiles = [p for p in all_profs if prof[prof_ds][p]["n"] >= 5]
profiles.sort(key=lambda p: -prof[prof_ds][p]["n"])
methods_p = METHODS
mat_em = np.zeros((len(methods_p), len(profiles)))
mat_acc = np.zeros((len(methods_p), len(profiles)))
ns = [prof[prof_ds][p]["n"] for p in profiles]
for i, m in enumerate(methods_p):
    for j, p in enumerate(profiles):
        mat_em[i, j] = prof[prof_ds][p][m]["em"]
        mat_acc[i, j] = prof[prof_ds][p][m]["acc"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
xtick_lbls = [f"{p}\n(n={n})" for p, n in zip(profiles, ns)]
for ax, mat, title in [(axes[0], mat_em, "EM"), (axes[1], mat_acc, "Accuracy")]:
    vmax = max(mat.max(), 0.01)
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(profiles))); ax.set_xticklabels(xtick_lbls, rotation=0, fontsize=8.5)
    ax.set_yticks(np.arange(len(methods_p))); ax.set_yticklabels(methods_p)
    for i in range(len(methods_p)):
        for j in range(len(profiles)):
            v = mat[i, j]
            color = "white" if v > vmax * 0.55 else "#222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=11.5, pad=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
fig.suptitle(f"MuSiQue per-profile breakdown (1000q)", fontsize=13, fontweight="semibold", y=1.02)
fig.tight_layout()
fig.savefig(PLOTS / "profile_heatmap_musique.png")
plt.close(fig)

# ---------- Plot 6: Token savings vs HERA (bar) ----------
fig, ax = plt.subplots(figsize=(10.5, 4.5))
amas_methods = ["AMAS-off", "AMAS-bayesian", "AMAS-conformal"]
x = np.arange(len(DATASETS))
w = 0.26
for j, m in enumerate(amas_methods):
    savings = []
    for ds in DATASETS:
        hera_t = _row(ds, "HERA-repro")["tokens"]
        m_t = _row(ds, m)["tokens"]
        savings.append((hera_t - m_t) / hera_t * 100)
    bars = ax.bar(x + (j - 1) * w, savings, w, color=PAL[m],
                  edgecolor="white", linewidth=0.8, label=m, zorder=3)
    for b, v in zip(bars, savings):
        ax.text(b.get_x() + b.get_width()/2,
                v + (1.5 if v >= 0 else -3.5),
                f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5, color="#222")
ax.axhline(0, color="#222", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
ax.set_ylabel("Token savings vs HERA (%)")
ax.set_title("Cost reduction relative to HERA-repro", loc="left", pad=10)
ax.grid(axis="y", alpha=0.45); ax.set_axisbelow(True)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS / "token_savings.png")
plt.close(fig)

# ---------- Plot 7: Δ vs HERA forest plot (paired bootstrap) ----------
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
amas_compare = ["AMAS-off", "AMAS-bayesian", "AMAS-conformal", "SAS-matched"]
ypos = np.arange(len(amas_compare))
for ax_i, ds in enumerate(DATASETS):
    ax = axes[ax_i]
    ci_block = agg[ds]["ci"]
    diffs, los, his, colors = [], [], [], []
    for m in amas_compare:
        key = f"{m}_vs_HERA_f1"
        if key in ci_block:
            diffs.append(ci_block[key]["diff"])
            los.append(ci_block[key]["ci_lo"])
            his.append(ci_block[key]["ci_hi"])
            colors.append(PAL.get(m, "#444"))
        else:
            diffs.append(0); los.append(0); his.append(0); colors.append("#888")
    diffs = np.array(diffs); los = np.array(los); his = np.array(his)
    err_low = diffs - los; err_high = his - diffs
    ax.errorbar(diffs, ypos, xerr=[err_low, err_high], fmt="none",
                ecolor="#444", capsize=4, lw=1.2, zorder=2)
    for i, (d, c) in enumerate(zip(diffs, colors)):
        ax.scatter(d, ypos[i], s=110, color=c, edgecolor="white", linewidth=1.2, zorder=3)
    ax.axvline(0, color="#222", ls="-", lw=0.7, alpha=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(amas_compare if ax_i == 0 else [])
    ax.set_xlabel("Δ F1 vs HERA-repro")
    ax.set_title(DATASET_LABELS[ds], loc="left")
    ax.grid(axis="x", alpha=0.45); ax.set_axisbelow(True)
fig.suptitle("Forest plot: F1 difference vs HERA-repro (95% paired bootstrap CI)",
             fontsize=12.5, fontweight="semibold", y=1.02)
fig.tight_layout()
fig.savefig(PLOTS / "forest_f1.png")
plt.close(fig)

print("Plots:")
for p in sorted(PLOTS.glob("*.png")):
    print(" ", p, p.stat().st_size, "bytes")
print("Data:")
for p in sorted(DATA.glob("*")):
    print(" ", p)
