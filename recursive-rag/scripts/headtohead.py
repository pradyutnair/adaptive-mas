from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.contracts import normalize_answer


def norm_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    from collections import Counter
    pt = normalize_answer(pred).split(); gt = normalize_answer(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    common = sum((Counter(pt) & Counter(gt)).values())
    if not common: return 0.0
    p = common / len(pt); r = common / len(gt)
    return 2 * p * r / (p + r)


def contain(pred: str, gold: str) -> float:
    p = normalize_answer(pred); g = normalize_answer(gold)
    return 1.0 if p and g and g in p else 0.0


def load_preds(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_run(predictions: Path, questions: Path) -> dict:
    gold = {str(q.get("id")): str(q.get("answer", "")) for q in json.loads(questions.read_text(encoding="utf-8"))}
    rows = load_preds(predictions)
    em = []; f1 = []; con = []; toks = []; hops = []; retries = []
    for r in rows:
        ga = gold.get(str(r.get("id", "")), "")
        pa = str(r.get("answer", r.get("predicted_answer", "")))
        em.append(norm_em(pa, ga)); f1.append(token_f1(pa, ga)); con.append(contain(pa, ga))
        m = r.get("metadata", {}) or {}
        toks.append(int(m.get("total_tokens", 0))); hops.append(int(m.get("hops", 0))); retries.append(int(m.get("retries", 0)))
    n = len(rows) or 1
    return {"norm_em": sum(em)/n, "token_f1": sum(f1)/n, "contain": sum(con)/n, "mean_tokens": sum(toks)/n, "mean_hops": sum(hops)/n, "mean_retries": sum(retries)/n, "total": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="name=predictions.jsonl")
    p.add_argument("--output")
    args = p.parse_args()
    matrix = {}
    for spec in args.runs:
        name, path = spec.split("=", 1) if "=" in spec else (Path(spec).parent.name, spec)
        matrix[name] = score_run(Path(path), Path(args.questions))
    text = json.dumps(matrix, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
