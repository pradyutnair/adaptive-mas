"""Compare 1000q sufficiency vs 1000q baselines (s0_matched, iter30_think) on
musique + hotpotqa + 2wikimultihop. Reports contain / token_f1 / norm_em,
mean tokens per question, and a paired bootstrap 95% CI on the contain delta
(sufficiency - s0_matched, sufficiency - iter30_think).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_offline import contain, norm_em, token_f1  # noqa: E402

SYSTEMS = ("s0_matched", "iter30_think", "sufficiency")
DATASETS = ("musique", "hotpotqa", "2wikimultihop")
QFILES = {
    "musique": "data/musique/questions_1000_seedfull_combined.json",
    "hotpotqa": "data/hotpotqa/questions_1000_seed42.json",
    "2wikimultihop": "data/2wikimultihop/questions_1000_seed42.json",
}


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def per_q_scores(preds: list[dict], gold_by_id: dict[str, str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in preds:
        qid = str(p["id"])
        gold = gold_by_id.get(qid, "")
        ans = str(p.get("answer", ""))
        md = p.get("metadata") or {}
        out[qid] = {
            "contain": contain(ans, gold),
            "token_f1": token_f1(ans, gold),
            "norm_em": norm_em(ans, gold),
            "tokens": md.get("total_tokens"),
            "subagent_calls": md.get("num_subagent_calls"),
        }
    return out


def aggregate(scored: dict[str, dict], ids: list[str]) -> dict:
    cs = [scored[i]["contain"] for i in ids if i in scored]
    fs = [scored[i]["token_f1"] for i in ids if i in scored]
    es = [scored[i]["norm_em"] for i in ids if i in scored]
    toks = [scored[i]["tokens"] for i in ids if i in scored and scored[i]["tokens"] is not None]
    subs = [scored[i]["subagent_calls"] for i in ids if i in scored and scored[i]["subagent_calls"] is not None]
    n = len(cs)
    out = {
        "n": n,
        "contain": round(sum(cs) / n, 4) if n else 0.0,
        "token_f1": round(sum(fs) / n, 4) if n else 0.0,
        "norm_em": round(sum(es) / n, 4) if n else 0.0,
    }
    if toks:
        out["mean_total_tokens"] = round(mean(toks), 1)
        out["sum_total_tokens"] = int(sum(toks))
    if subs:
        out["mean_subagent_calls"] = round(mean(subs), 3)
    return out


def paired_bootstrap_ci(
    contain_a: list[int], contain_b: list[int], n_boot: int = 10000, seed: int = 0
) -> dict:
    """95% paired bootstrap CI on mean(a) - mean(b)."""
    a = np.asarray(contain_a, dtype=float)
    b = np.asarray(contain_b, dtype=float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = a[idx].mean() - b[idx].mean()
    obs = float(a.mean() - b.mean())
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    p_two_sided = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "delta": round(obs, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "p_two_sided": round(float(p_two_sided), 4),
    }


def main() -> None:
    latest = (ROOT / "results" / ".sufficiency_1000q_latest").read_text().strip()
    suff_root = ROOT / latest

    table: dict[str, dict] = {}
    for ds in DATASETS:
        questions = json.load(open(ROOT / QFILES[ds]))
        gold_by_id = {str(q["id"]): str(q.get("answer", "")) for q in questions}
        ids = [str(q["id"]) for q in questions]

        per_sys_scored: dict[str, dict] = {}
        per_sys_agg: dict[str, dict] = {}
        for s in SYSTEMS:
            if s == "sufficiency":
                pred_path = suff_root / f"{ds}/sufficiency/predictions.jsonl"
            else:
                pred_path = ROOT / f"paper_results/latest/{ds}/{s}/predictions.jsonl"
            preds = load_jsonl(pred_path)
            scored = per_q_scores(preds, gold_by_id)
            per_sys_scored[s] = scored
            per_sys_agg[s] = aggregate(scored, ids)

        common_ids = [i for i in ids if all(i in per_sys_scored[s] for s in SYSTEMS)]
        suff_c = [per_sys_scored["sufficiency"][i]["contain"] for i in common_ids]
        s0_c = [per_sys_scored["s0_matched"][i]["contain"] for i in common_ids]
        it_c = [per_sys_scored["iter30_think"][i]["contain"] for i in common_ids]

        table[ds] = {
            "n_common": len(common_ids),
            "agg": per_sys_agg,
            "ci": {
                "sufficiency_vs_s0_matched": paired_bootstrap_ci(suff_c, s0_c),
                "sufficiency_vs_iter30_think": paired_bootstrap_ci(suff_c, it_c),
            },
        }

    print(json.dumps(table, indent=2))

    print("\n=== contain (1000q) ===")
    print(f"{'system':<14}" + "".join(f"{ds:>15}" for ds in DATASETS))
    for s in SYSTEMS:
        row = f"{s:<14}"
        for ds in DATASETS:
            row += f"{table[ds]['agg'][s]['contain']:>15.4f}"
        print(row)

    print("\n=== mean total tokens / question ===")
    print(f"{'system':<14}" + "".join(f"{ds:>15}" for ds in DATASETS))
    for s in SYSTEMS:
        row = f"{s:<14}"
        for ds in DATASETS:
            v = table[ds]["agg"][s].get("mean_total_tokens", float("nan"))
            row += f"{v:>15.1f}"
        print(row)

    print("\n=== paired bootstrap 95% CI on contain delta (sufficiency - baseline) ===")
    for ds in DATASETS:
        print(f"\n[{ds}]  n={table[ds]['n_common']}")
        for k, v in table[ds]["ci"].items():
            sig = "*" if v["ci95_lo"] > 0 or v["ci95_hi"] < 0 else " "
            print(
                f"  {sig} {k:<32} delta={v['delta']:+.4f}  "
                f"95%CI=[{v['ci95_lo']:+.4f}, {v['ci95_hi']:+.4f}]  p={v['p_two_sided']:.4f}"
            )

    out = suff_root / "compare_1000q.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
