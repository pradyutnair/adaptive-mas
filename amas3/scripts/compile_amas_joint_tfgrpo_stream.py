#!/usr/bin/env python3
"""Streaming corrective TF-GRPO compiler for AMAS3."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from amas3.lm import LMConfig, make_qwen14b_nothink_lm
from amas3.pipeline import AmasPipeline
from amas3.retriever import Retriever
from compile_amas_joint_tfgrpo import POLICIES, contain, load_train, reward, token_f1


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/local/yzheng/pnair/workspace/reproduction/hera/data/train_240_v2.jsonl")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--retriever-url", default="http://node408:8003")
    ap.add_argument("--token-target", type=float, default=8000.0)
    ap.add_argument("--token-weight", type=float, default=0.08)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout-seconds", type=float, default=120.0)
    ap.add_argument("--policy-names", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--source", default="", help="optional train source filter, e.g. musique")
    args = ap.parse_args()

    selected = {x.strip() for x in args.policy_names.split(",") if x.strip()}
    policies = [p for p in POLICIES if p.name in selected]
    if len(policies) != len(selected):
        found = {p.name for p in policies}
        raise ValueError("unknown policies: " + str(sorted(selected - found)))

    rows = load_train(args.train)
    if args.source:
        rows = [r for r in rows if r.get("source", "") == args.source]
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.n_train]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "rollouts.jsonl"
    summary_path = out / "summary.json"
    policy_path = out / "policy.json"

    cfg = LMConfig(qwen_nothink_max_tokens=768)
    retriever = Retriever(base_url=args.retriever_url)
    sem = asyncio.Semaphore(args.concurrency)
    n_replicas = 3

    async def run_one(qidx: int, q: dict, pidx: int, policy) -> dict:
        async with sem:
            async def inner() -> dict:
                lm_idx = (qidx + pidx) % n_replicas
                planner = make_qwen14b_nothink_lm(cfg, replica_idx=lm_idx, max_tokens=768)
                worker = make_qwen14b_nothink_lm(cfg, replica_idx=(lm_idx + 1) % n_replicas, max_tokens=768)
                synth = make_qwen14b_nothink_lm(cfg, replica_idx=(lm_idx + 2) % n_replicas, max_tokens=768)
                sas = make_qwen14b_nothink_lm(cfg, replica_idx=lm_idx, max_tokens=384)
                pipe = AmasPipeline(
                    planner_lm=planner,
                    worker_lm=worker,
                    synth_lm=synth,
                    sas_lm=sas,
                    retriever=retriever,
                    config=policy.to_config(),
                )
                t0 = time.time()
                r = await pipe.run(q["question"], qid=str(q.get("id", "")))
                return {
                    "id": q.get("id", ""),
                    "source": q.get("source", ""),
                    "policy": policy.name,
                    "answer": r.answer,
                    "gold": q["answer"],
                    "f1": token_f1(r.answer, q["answer"]),
                    "contain": contain(r.answer, q["answer"]),
                    "tokens": r.total_tokens,
                    "topology": r.topology,
                    "plan_subgoals": r.plan_subgoals,
                    "n_retrieval_calls": r.n_retrieval_calls,
                    "n_solvers_invoked": r.n_solvers_invoked,
                    "wallclock": round(time.time() - t0, 3),
                }

            try:
                row = await asyncio.wait_for(inner(), timeout=args.timeout_seconds)
                row["reward"] = reward(row["answer"], q["answer"], row["tokens"], args.token_target, args.token_weight)
                return row
            except Exception as e:
                return {
                    "id": q.get("id", ""),
                    "source": q.get("source", ""),
                    "policy": policy.name,
                    "answer": "",
                    "gold": q.get("answer", ""),
                    "f1": 0.0,
                    "contain": 0.0,
                    "reward": -1.0,
                    "tokens": 0,
                    "error": str(e)[:300] or type(e).__name__,
                }

    tasks = [
        asyncio.create_task(run_one(qidx, q, pidx, policy))
        for qidx, q in enumerate(rows)
        for pidx, policy in enumerate(policies)
    ]

    print(f"streaming {len(tasks)} train rollouts over {len(rows)} questions and {len(policies)} policies", flush=True)
    results = []
    with trace_path.open("w") as f:
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            row = await fut
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(tasks):
                print(f"completed {i}/{len(tasks)} rollouts", flush=True)

    stats = {}
    source_stats = {}
    winner_counts = Counter()
    by_q = defaultdict(list)
    for row in results:
        by_q[str(row["id"])].append(row)
    for qid, rs in by_q.items():
        winner_counts[max(rs, key=lambda r: r["reward"])["policy"]] += 1

    for policy in policies:
        rs = [r for r in results if r["policy"] == policy.name]
        n = max(1, len(rs))
        stats[policy.name] = {
            "n": len(rs),
            "mean_reward": sum(r["reward"] for r in rs) / n,
            "mean_f1": sum(r["f1"] for r in rs) / n,
            "mean_contain": sum(r["contain"] for r in rs) / n,
            "mean_tokens": sum(r["tokens"] for r in rs) / n,
            "timeouts_or_errors": sum(1 for r in rs if r.get("error")),
            "wins": winner_counts[policy.name],
        }
    for source in sorted({r.get("source", "") for r in results}):
        for policy in policies:
            rs = [r for r in results if r.get("source", "") == source and r["policy"] == policy.name]
            if not rs:
                continue
            n = len(rs)
            source_stats[f"{source}:{policy.name}"] = {
                "n": n,
                "mean_reward": sum(r["reward"] for r in rs) / n,
                "mean_f1": sum(r["f1"] for r in rs) / n,
                "mean_contain": sum(r["contain"] for r in rs) / n,
                "mean_tokens": sum(r["tokens"] for r in rs) / n,
            }

    best_policy_name = max(stats, key=lambda k: stats[k]["mean_reward"])
    best_policy = next(p for p in policies if p.name == best_policy_name)
    summary = {
        "train_path": args.train,
        "n_train": len(rows),
        "source_filter": args.source,
        "token_target": args.token_target,
        "token_weight": args.token_weight,
        "policy_names": [p.name for p in policies],
        "stats": stats,
        "source_stats": source_stats,
        "winner_counts": dict(winner_counts),
        "selected_policy_by_mean_reward": best_policy_name,
        "rollouts": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    policy_path.write_text(json.dumps(asdict(best_policy), indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
