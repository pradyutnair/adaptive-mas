"""Smoke50 comparison: sufficiency vs s0_matched vs iter30_think on the same 50 IDs.

For each dataset, slices the existing 1000q baseline predictions to the smoke50
ID set and reports contain / token_f1 / norm_em plus mean total_tokens (efficiency)
and mean orchestrator+subagent steps where available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_offline import contain, norm_em, token_f1  # noqa: E402


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


def slice_to_ids(preds: list[dict], ids: set[str]) -> list[dict]:
    by_id = {str(p["id"]): p for p in preds}
    return [by_id[i] for i in ids if i in by_id]


def score(preds: list[dict], gold_by_id: dict[str, str]) -> dict:
    cs, fs, es = [], [], []
    toks = []
    n_sub = []
    for p in preds:
        qid = str(p["id"])
        gold = gold_by_id.get(qid, "")
        ans = str(p.get("answer", ""))
        cs.append(contain(ans, gold))
        fs.append(token_f1(ans, gold))
        es.append(norm_em(ans, gold))
        token_value = _first_numeric(
            p,
            ("metadata", "total_tokens"),
            ("total_tokens",),
        )
        subagent_value = _first_numeric(
            p,
            ("metadata", "num_subagent_calls"),
            ("metadata", "llm_call_count"),
            ("llm_call_count",),
        )
        if token_value is not None:
            toks.append(token_value)
        if subagent_value is not None:
            n_sub.append(subagent_value)
    n = len(preds)
    out = {
        "n": n,
        "contain": round(sum(cs) / n, 4) if n else 0.0,
        "token_f1": round(sum(fs) / n, 4) if n else 0.0,
        "norm_em": round(sum(es) / n, 4) if n else 0.0,
    }
    if toks:
        out["mean_total_tokens"] = round(mean(toks), 1)
        out["sum_total_tokens"] = sum(toks)
    if n_sub:
        out["mean_subagent_calls"] = round(mean(n_sub), 3)
    return out


def main() -> None:
    latest = (ROOT / "results" / ".smoke50_latest").read_text().strip()
    suff_root = ROOT / latest

    table: dict[str, dict[str, dict]] = {}
    for ds in ("musique", "hotpotqa"):
        questions = json.load(open(ROOT / f"data/{ds}/questions_smoke50_seed42.json"))
        ids = {str(q["id"]) for q in questions}
        gold_by_id = {str(q["id"]): str(q.get("answer", "")) for q in questions}

        runs = {
            "s0_matched": ROOT / f"paper_results/latest/{ds}/s0_matched/predictions.jsonl",
            "iter30_think": ROOT / f"paper_results/latest/{ds}/iter30_think/predictions.jsonl",
            "sufficiency": suff_root / f"{ds}/sufficiency/predictions.jsonl",
        }
        per_sys = {}
        for sys_name, pred_path in runs.items():
            preds = load_jsonl(pred_path)
            sliced = slice_to_ids(preds, ids)
            assert len(sliced) == 50, f"{ds}/{sys_name}: got {len(sliced)} preds for 50 IDs"
            per_sys[sys_name] = score(sliced, gold_by_id)
        table[ds] = per_sys

    print(json.dumps(table, indent=2))

    print("\n=== headline (contain) ===")
    print(f"{'system':<14}{'musique':>12}{'hotpotqa':>12}")
    for s in ("s0_matched", "iter30_think", "sufficiency"):
        print(f"{s:<14}{table['musique'][s]['contain']:>12.4f}{table['hotpotqa'][s]['contain']:>12.4f}")

    print("\n=== efficiency (mean total_tokens / question) ===")
    print(f"{'system':<14}{'musique':>14}{'hotpotqa':>14}")
    for s in ("s0_matched", "iter30_think", "sufficiency"):
        m = table["musique"][s].get("mean_total_tokens", float("nan"))
        h = table["hotpotqa"][s].get("mean_total_tokens", float("nan"))
        print(f"{s:<14}{m:>14.1f}{h:>14.1f}")

    out = ROOT / latest / "smoke50_compare.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
