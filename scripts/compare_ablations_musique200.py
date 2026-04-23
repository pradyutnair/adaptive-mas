"""Aggregate musique-200q ablation results into a single comparison table.

Reads predictions from results/<latest abl_musique200>/<variant>/ and the
canonical 200q gold from data/musique/questions_200_seedfull_first.json.
Reports per-variant: n, contain, token_f1, norm_em, mean_total_tokens,
mean_subagent_calls. Also computes paired-bootstrap 95% CI on contain
delta vs canonical sufficiency for every ablation.
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

QFILE = ROOT / "data/musique/questions_200_seedfull_first.json"


def _first_numeric(row: dict, *paths: tuple[str, ...]) -> float | None:
    """Return the first numeric field found across alternate schemas."""
    for path in paths:
        cur = row
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if not ok or cur is None:
            continue
        try:
            return float(cur)
        except (TypeError, ValueError):
            continue
    return None


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
            "tokens": _first_numeric(
                p,
                ("metadata", "total_tokens"),
                ("total_tokens",),
            ),
            "subagent_calls": _first_numeric(
                p,
                ("metadata", "num_subagent_calls"),
                ("metadata", "llm_call_count"),
                ("llm_call_count",),
            ),
        }
    return out


def aggregate(scored: dict[str, dict], ids: list[str]) -> dict:
    cs, fs, es = [], [], []
    toks, subs = [], []
    for i in ids:
        if i not in scored:
            continue
        cs.append(scored[i]["contain"])
        fs.append(scored[i]["token_f1"])
        es.append(scored[i]["norm_em"])
        if scored[i]["tokens"] is not None:
            toks.append(scored[i]["tokens"])
        if scored[i]["subagent_calls"] is not None:
            subs.append(scored[i]["subagent_calls"])
    n = len(cs)
    out = {
        "n": n,
        "contain": round(sum(cs) / n, 4) if n else 0.0,
        "token_f1": round(sum(fs) / n, 4) if n else 0.0,
        "norm_em": round(sum(es) / n, 4) if n else 0.0,
    }
    if toks:
        out["mean_total_tokens"] = round(mean(toks), 1)
    if subs:
        out["mean_subagent_calls"] = round(mean(subs), 3)
    return out


def paired_bootstrap_ci(
    a: list[int], b: list[int], n_boot: int = 10000, seed: int = 0
) -> dict:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(aa)
    diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = aa[idx].mean() - bb[idx].mean()
    obs = float(aa.mean() - bb.mean())
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "delta": round(obs, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "p_two_sided": round(float(p_two), 4),
    }


def main() -> None:
    latest = (ROOT / "results" / ".abl_musique200_latest").read_text().strip()
    abl_root = ROOT / latest

    questions = json.loads(QFILE.read_text())
    ids = [str(q["id"]) for q in questions]
    gold = {str(q["id"]): str(q.get("answer", "")) for q in questions}

    variants = [
        d.name for d in sorted(abl_root.iterdir()) if d.is_dir() and (d / "predictions.jsonl").exists()
    ]
    canonical = "sufficiency"

    per_var_scored: dict[str, dict] = {}
    per_var_agg: dict[str, dict] = {}
    for v in variants:
        preds = load_jsonl(abl_root / v / "predictions.jsonl")
        scored = per_q_scores(preds, gold)
        per_var_scored[v] = scored
        per_var_agg[v] = aggregate(scored, ids)

    cis: dict[str, dict] = {}
    if canonical in per_var_scored:
        common = [i for i in ids if i in per_var_scored[canonical]]
        suff_c = [per_var_scored[canonical][i]["contain"] for i in common]
        for v in variants:
            if v == canonical:
                continue
            common_v = [i for i in common if i in per_var_scored[v]]
            a = [per_var_scored[canonical][i]["contain"] for i in common_v]
            b = [per_var_scored[v][i]["contain"] for i in common_v]
            cis[v] = paired_bootstrap_ci(a, b)

    table = {"variants": per_var_agg, "ci_vs_sufficiency": cis}
    out = abl_root / "ablation_summary.json"
    out.write_text(json.dumps(table, indent=2))

    print("\n=== musique 200q ablations ===")
    print(f"{'variant':<22}{'n':>5}{'contain':>10}{'tok_f1':>10}{'norm_em':>10}{'mean_tok':>14}{'subcalls':>11}")
    for v in variants:
        a = per_var_agg[v]
        print(
            f"{v:<22}{a['n']:>5}{a['contain']:>10.4f}{a['token_f1']:>10.4f}"
            f"{a['norm_em']:>10.4f}{a.get('mean_total_tokens', 0.0):>14.1f}"
            f"{a.get('mean_subagent_calls', 0.0):>11.3f}"
        )

    print("\n=== paired bootstrap 95% CI on contain delta (sufficiency - variant) ===")
    for v, ci in cis.items():
        sig = "*" if ci["ci95_lo"] > 0 or ci["ci95_hi"] < 0 else " "
        print(
            f"  {sig} sufficiency vs {v:<20} delta={ci['delta']:+.4f}  "
            f"95%CI=[{ci['ci95_lo']:+.4f}, {ci['ci95_hi']:+.4f}]  p={ci['p_two_sided']:.4f}"
        )

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
