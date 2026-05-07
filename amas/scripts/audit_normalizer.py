"""Phase 0 audit: how much do AMAS headline gains depend on normalize_answer_span?

For each (dataset, method) pair score under three regimes:
  R1 raw           : skip normalize_answer_span entirely. Score raw text vs gold via metric.normalize_answer.
  R2 cap8          : apply normalize_answer_span(max_words=8). Current headline behaviour.
  R3 cap64         : apply normalize_answer_span(max_words=64). Disables the 8-word cap. Tells us how much of the AMAS-vs-HERA gap is the cap penalising HERA verbose prose.

AMAS preds: results/p1_full/<ds>_<gate>/predictions.jsonl. Stored final_answer is post-cap8.
            Recover raw from invocations: ConcludeAgent > ReflectAgent > AnswerGenerator.answer
            For SAS-committed rows, raw = Probe inv answer (already short).
HERA preds: reproduction/hera/results/run02_eval_verbose/predictions_<ds>.jsonl  (read-only).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean

ROOT = Path("/local/yzheng/pnair/workspace/adaptive-mas/amas")
sys.path.insert(0, str(ROOT / "src"))

from amas.metric import exact_match, f1_score, contain, accuracy
from amas.orchestrator import normalize_answer_span

DATASETS = ["musique", "hotpotqa", "2wikimultihop", "bamboogle"]
GATES    = ["off", "conformal", "bayesian"]

HERA_PRED = {
    "musique": "/local/yzheng/pnair/workspace/reproduction/hera/results/run02_eval_verbose/predictions_musique.jsonl",
    "hotpotqa": "/local/yzheng/pnair/workspace/reproduction/hera/results/run02_eval_verbose/predictions_hotpotqa.jsonl",
    "2wikimultihop": "/local/yzheng/pnair/workspace/reproduction/hera/results/run02_eval_verbose/predictions_2wikimultihop.jsonl",
    "bamboogle": "/local/yzheng/pnair/workspace/reproduction/hera/results/run02_eval_verbose/predictions_bamboogle.jsonl",
}


def _inv_answer(inv: dict) -> str:
    out = inv.get("output")
    if isinstance(out, dict):
        a = out.get("answer", "")
        if a:
            return a
    return inv.get("answer", "") or ""


def amas_raw_answer(row: dict) -> str:
    turns = row.get("turns") or []
    sas = bool(row.get("sas_committed"))
    if sas and turns:
        for inv in turns[0].get("invocations", []):
            if inv.get("name") == "Probe":
                a = _inv_answer(inv)
                if a:
                    return a
        return turns[0].get("answer", "") or ""
    if turns:
        last = turns[-1]
        by = {inv["name"]: inv for inv in last.get("invocations", [])}
        for name in ("ConcludeAgent", "ReflectAgent", "AnswerGenerator"):
            inv = by.get(name)
            if inv:
                a = _inv_answer(inv)
                if a:
                    return a
        for inv in reversed(last.get("invocations", [])):
            a = _inv_answer(inv)
            if a:
                return a
        return last.get("answer", "") or ""
    return row.get("final_answer", "") or ""


def score(pred: str, gold) -> dict:
    return {
        "em":      exact_match(pred, gold),
        "f1":      f1_score(pred, gold),
        "contain": contain(pred, gold),
        "acc":     accuracy(pred, gold),
    }


def score_corpus(rows, get_pred):
    em, f1, ct, ac = [], [], [], []
    for r in rows:
        gold = r.get("gold") or r.get("answer") or ""
        p = get_pred(r) or ""
        s = score(p, gold)
        em.append(s["em"]); f1.append(s["f1"]); ct.append(s["contain"]); ac.append(s["acc"])
    n = len(rows)
    return {"n": n, "em": mean(em) if n else 0.0, "f1": mean(f1) if n else 0.0,
            "contain": mean(ct) if n else 0.0, "acc": mean(ac) if n else 0.0}


def audit_amas(ds: str, gate: str) -> dict | None:
    p = ROOT / f"results/p1_full/{ds}_{gate}/predictions.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    raw_ans = [amas_raw_answer(r) for r in rows]
    avg_tok = mean(r.get("total_tokens", 0) for r in rows) if rows else 0
    return {
        "method": f"AMAS-{gate}",
        "dataset": ds,
        "avg_tokens": avg_tok,
        "R1_raw":   score_corpus(rows, lambda r: amas_raw_answer(r)),
        "R2_cap8":  score_corpus(rows, lambda r: normalize_answer_span(amas_raw_answer(r), question=r.get("question",""), max_words=8)),
        "R3_cap64": score_corpus(rows, lambda r: normalize_answer_span(amas_raw_answer(r), question=r.get("question",""), max_words=64)),
    }


def audit_hera(ds: str) -> dict | None:
    p = Path(HERA_PRED[ds])
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    avg_tok = mean(r.get("tokens", 0) for r in rows) if rows else 0
    return {
        "method": "HERA-repro",
        "dataset": ds,
        "avg_tokens": avg_tok,
        "R1_raw":   score_corpus(rows, lambda r: r.get("pred","")),
        "R2_cap8":  score_corpus(rows, lambda r: normalize_answer_span(r.get("pred",""), question=r.get("question",""), max_words=8)),
        "R3_cap64": score_corpus(rows, lambda r: normalize_answer_span(r.get("pred",""), question=r.get("question",""), max_words=64)),
    }


def main():
    out = {"datasets": {}}
    for ds in DATASETS:
        block = {"HERA-repro": audit_hera(ds)}
        for gate in GATES:
            r = audit_amas(ds, gate)
            if r is not None:
                block[f"AMAS-{gate}"] = r
        out["datasets"][ds] = block
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
