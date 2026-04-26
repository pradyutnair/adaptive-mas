"""Classify iter27_think contain=0 failures into retrieval/distill/synthesis buckets."""

from __future__ import annotations

import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_offline import contain, normalize_answer  # noqa: E402

PREDS = "results/iter27_think_full1000_c24_combined.jsonl"
CHUNKS = "data/musique/chunks.json"
BUCKETS = ["BUCKET_RETRIEVAL_FAIL", "BUCKET_DISTILL_FAIL", "BUCKET_SYNTHESIS_FAIL", "BUCKET_OTHER"]


def hop_of(qid: str) -> str:
    qid = qid.lower()
    return "2hop" if "2hop" in qid else "3hop" if "3hop" in qid else "4hop" if "4hop" in qid else "unknown"


def norm_has(blob: str, gold: str) -> bool:
    nblob, ngold = normalize_answer(blob), normalize_answer(gold)
    return bool(nblob and ngold and ngold in nblob)


def load_chunks(path: str) -> dict[str, str]:
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            out[str(k)] = str(v.get("text", v.get("contents", v.get("content", v)))) if isinstance(v, dict) else str(v)
        return out
    out = {}
    for row in data:
        if isinstance(row, str):
            cid, _, text = row.partition(":")
            out[str(cid)] = text
            continue
        if not isinstance(row, dict):
            continue
        cid = row.get("id", row.get("chunk_id", row.get("_id", "")))
        text = row.get("text", row.get("contents", row.get("content", "")))
        out[str(cid)] = str(text)
    return out


def main() -> None:
    preds = [json.loads(l) for l in open(PREDS, encoding="utf-8") if l.strip()]
    chunks = load_chunks(CHUNKS)
    failures, overall, per_hop = [], Counter(), defaultdict(Counter)
    for row in preds:
        gold = row.get("gold_answer", "")
        if contain(row.get("answer", ""), gold) != 0.0:
            continue
        meta, facts = row.get("metadata", {}) or {}, (row.get("metadata", {}) or {}).get("facts_used", [])
        passages = " ".join(chunks.get(str(cid), "") for fact in facts for cid in fact.get("support_ids", []))
        facts_blob = " ".join(str(f.get("text", "")) for f in facts)
        answer_blob = str(row.get("answer", ""))
        if not norm_has(passages, gold):
            bucket = "BUCKET_RETRIEVAL_FAIL"
        elif not norm_has(facts_blob, gold):
            bucket = "BUCKET_DISTILL_FAIL"
        elif not norm_has(answer_blob, gold):
            bucket = "BUCKET_SYNTHESIS_FAIL"
        else:
            bucket = "BUCKET_OTHER"
        rec = {
            "bucket": bucket, "hop": hop_of(str(row.get("id", ""))), "question": row.get("question", ""),
            "gold": gold, "pred": answer_blob,
            "top_fact": max(facts, key=lambda f: float(f.get("confidence", 0.0)), default={}).get("text", ""),
        }
        failures.append(rec); overall[bucket] += 1; per_hop[rec["hop"]][bucket] += 1
    n = len(failures); rng = random.Random(42)
    print(f"Total failures N = {n}\n")
    print("Overall breakdown:")
    for b in BUCKETS:
        c = overall[b]; print(f"- {b}: {c} ({(100*c/n if n else 0):.1f}%)")
    print("\nPer-hop breakdown:")
    for hop in ["2hop", "3hop", "4hop"]:
        total = sum(per_hop[hop].values()); print(f"- {hop}: {total}")
        for b in BUCKETS:
            c = per_hop[hop][b]; print(f"  - {b}: {c} ({(100*c/total if total else 0):.1f}%)")
    print("\nExamples:")
    for b in BUCKETS:
        rows = [r for r in failures if r["bucket"] == b]
        sample = rng.sample(rows, min(5, len(rows)))
        print(f"\n{b} ({len(rows)} total)")
        for i, r in enumerate(sample, 1):
            print(f"{i}. hop={r['hop']} | q={r['question']}\n   gold={r['gold']}\n   pred={r['pred']}\n   top_fact={r['top_fact']}")


if __name__ == "__main__":
    main()
