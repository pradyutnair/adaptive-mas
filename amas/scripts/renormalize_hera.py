"""Apply AMAS normalize_answer_span to HERA-repro stored preds for fair Acc comparison.

HERA-repro v1_reference predictions are 25-word avg verbose answers (untruncated).
AMAS predictions are clean 8-word spans. Apples-to-oranges on contain.
This script renormalizes HERA preds and rewrites a 'normalized' predictions copy.

Output: results/hera_renormalized/predictions_<ds>.jsonl  with em/f1/contain/acc recomputed
        results/hera_renormalized/summary_<ds>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from amas.orchestrator import normalize_answer_span
from amas.metric import accuracy, contain, exact_match, f1_score


HERA_PRED = {
    "musique": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_musique.jsonl",
    "hotpotqa": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_hotpotqa.jsonl",
    "2wikimultihop": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_2wikimultihop.jsonl",
    "bamboogle": "/local/yzheng/pnair/workspace/reproduction/hera/results/v1_reference/run01_eval/predictions_bamboogle.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/hera_renormalized")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_all = []
    for ds, src in HERA_PRED.items():
        if not Path(src).exists():
            print(f"skip {ds}: missing {src}", file=sys.stderr)
            continue
        rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
        out_path = out_dir / f"predictions_{ds}.jsonl"
        out_fh = open(out_path, "w")
        em_list, f1_list, cont_list, acc_list = [], [], [], []
        for r in rows:
            raw_pred = r.get("pred", "") or ""
            q = r.get("question", "")
            gold = r.get("gold", r.get("answer", ""))
            norm = normalize_answer_span(raw_pred, question=q)
            em = exact_match(norm, gold)
            f1 = f1_score(norm, gold)
            cont = contain(norm, gold)
            acc = accuracy(norm, gold)
            em_list.append(em); f1_list.append(f1); cont_list.append(cont); acc_list.append(acc)
            out_fh.write(json.dumps({
                "id": r.get("id", ""), "qid": r.get("id", ""),
                "question": q, "gold": gold,
                "pred_raw": raw_pred,
                "pred": norm,
                "em": em, "f1": f1, "contain": cont, "acc": acc,
                "tokens": r.get("tokens", 0),
            }, ensure_ascii=False) + "\n")
        out_fh.close()
        n = len(rows)
        em = mean(em_list); f1 = mean(f1_list); cont = mean(cont_list); acc = mean(acc_list)
        tok = mean(r.get("tokens", 0) for r in rows)
        summary = {"dataset": ds, "n": n, "em": em, "f1": f1, "contain": cont, "acc": acc,
                   "avg_tokens": tok, "method": "HERA-repro-renormalized"}
        (out_dir / f"summary_{ds}.json").write_text(json.dumps(summary, indent=2))
        summary_all.append(summary)
        print(f"{ds}: n={n} EM={em:.3f} F1={f1:.3f} Acc={acc:.3f} tok={tok:.0f}")

    (out_dir / "summary_all.json").write_text(json.dumps(summary_all, indent=2))


if __name__ == "__main__":
    main()
