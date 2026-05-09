#!/usr/bin/env python3
"""Joint training-free TF-GRPO compiler for AMAS3.

Uses only IRCoT-style train questions. It compares a small, fixed group of
SAS+MAS policies on each train question, rewards quality and token efficiency,
then writes one deployment policy plus a compact RoPE/GEPA-style experience
library. No test labels or baseline outputs are used.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from amas3.lm import LMConfig, make_qwen14b_nothink_lm
from amas3.pipeline import AmasPipeline, AmasPipelineConfig
from amas3.retriever import Retriever


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", "", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split()).strip()


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    from collections import Counter as C
    common = sum((C(p) & C(g)).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def contain(pred: str, gold: str) -> float:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    return float(bool(p and g and g in p))


def load_train(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    obj = json.loads(p.read_text())
    if not isinstance(obj, list):
        raise ValueError(f"expected list/jsonl at {path}")
    return obj


@dataclass(frozen=True)
class Policy:
    name: str
    max_retrievals: int
    max_plan_subgoals: int
    synth_max_chunks: int
    synth_excerpt_chars: int
    synth_chain_of_thought: bool
    skip_synth_on_final_ok: bool
    skip_synth_confidence: float
    repair_enabled: bool
    use_sas_collapse: bool = True
    tau_sas_g: float = 0.65
    tau_sas_conf: float = 0.75
    adaptive_solver_budget: bool = True
    min_retrievals_per_solver: int = 1
    medium_retrievals_per_solver: int = 2
    max_repairs: int = 0
    use_bridge_resolver: bool = False
    bridge_g_threshold: float = 0.45
    use_multi_plan: bool = False
    K_plans: int = 3

    def to_config(self, experience: str = "") -> AmasPipelineConfig:
        return AmasPipelineConfig(
            max_retrievals_per_solver=self.max_retrievals,
            repair_enabled=self.repair_enabled,
            experience_library=experience,
            use_sas_collapse=self.use_sas_collapse,
            tau_sas_g=self.tau_sas_g,
            tau_sas_conf=self.tau_sas_conf,
            adaptive_solver_budget=self.adaptive_solver_budget,
            min_retrievals_per_solver=self.min_retrievals_per_solver,
            medium_retrievals_per_solver=self.medium_retrievals_per_solver,
            max_repairs=self.max_repairs,
            max_plan_subgoals=self.max_plan_subgoals,
            synth_max_chunks=self.synth_max_chunks,
            synth_excerpt_chars=self.synth_excerpt_chars,
            synth_chain_of_thought=self.synth_chain_of_thought,
            skip_synth_on_final_ok=self.skip_synth_on_final_ok,
            skip_synth_confidence=self.skip_synth_confidence,
            use_bridge_resolver=self.use_bridge_resolver,
            bridge_g_threshold=self.bridge_g_threshold,
            use_multi_plan=self.use_multi_plan,
            K_plans=self.K_plans,
        )


POLICIES = [
    Policy(
        name="full_reference",
        max_retrievals=3,
        max_plan_subgoals=6,
        synth_max_chunks=20,
        synth_excerpt_chars=700,
        synth_chain_of_thought=True,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=True,
        max_repairs=1,
    ),
    Policy(
        name="compact_synth",
        max_retrievals=2,
        max_plan_subgoals=3,
        synth_max_chunks=8,
        synth_excerpt_chars=420,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="compact5_synth",
        max_retrievals=2,
        max_plan_subgoals=3,
        synth_max_chunks=5,
        synth_excerpt_chars=300,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="compact_fastfinal",
        max_retrievals=2,
        max_plan_subgoals=3,
        synth_max_chunks=6,
        synth_excerpt_chars=360,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=True,
        skip_synth_confidence=0.82,
        repair_enabled=False,
    ),
    Policy(
        name="retrieval3_compact8",
        max_retrievals=3,
        max_plan_subgoals=3,
        synth_max_chunks=8,
        synth_excerpt_chars=420,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="plan4_compact8",
        max_retrievals=2,
        max_plan_subgoals=4,
        synth_max_chunks=8,
        synth_excerpt_chars=420,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="quality8_repair",
        max_retrievals=3,
        max_plan_subgoals=4,
        synth_max_chunks=8,
        synth_excerpt_chars=420,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=True,
        max_repairs=1,
    ),
    Policy(
        name="synth12_retrieval2",
        max_retrievals=2,
        max_plan_subgoals=3,
        synth_max_chunks=12,
        synth_excerpt_chars=420,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="bridge_retrieval3_compact8",
        max_retrievals=3,
        max_plan_subgoals=3,
        synth_max_chunks=8,
        synth_excerpt_chars=420,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
        use_bridge_resolver=True,
        bridge_g_threshold=0.45,
    ),
    Policy(
        name="cot_retrieval3_compact8",
        max_retrievals=3,
        max_plan_subgoals=3,
        synth_max_chunks=8,
        synth_excerpt_chars=420,
        synth_chain_of_thought=True,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="quality12_retrieval3",
        max_retrievals=3,
        max_plan_subgoals=4,
        synth_max_chunks=12,
        synth_excerpt_chars=500,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="plan6_retrieval3_compact8",
        max_retrievals=3,
        max_plan_subgoals=6,
        synth_max_chunks=8,
        synth_excerpt_chars=420,
        synth_chain_of_thought=False,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=False,
    ),
    Policy(
        name="plan6_retrieval3_oldlike",
        max_retrievals=3,
        max_plan_subgoals=6,
        synth_max_chunks=20,
        synth_excerpt_chars=700,
        synth_chain_of_thought=True,
        skip_synth_on_final_ok=False,
        skip_synth_confidence=0.90,
        repair_enabled=True,
        max_repairs=1,
    ),
]


def reward(answer: str, gold: str, tokens: int, token_target: float, token_weight: float) -> float:
    quality = 0.75 * token_f1(answer, gold) + 0.25 * contain(answer, gold)
    cost = min(max(tokens, 0) / token_target, 2.0)
    return quality - token_weight * cost


def build_experience(winner_counts: Counter, stats: dict[str, dict], failure_modes: Counter) -> str:
    entries = [
        "For SAS, accept only direct wh-target answers that are type-safe, verifier-passed, and explicitly supported by retrieved text; otherwise escalate.",
        "For MAS, prefer the minimum sufficient decomposition and cap bridge questions at three subgoals unless the question explicitly lists independent comparisons.",
        "For final synthesis, use only the final-hop evidence plus the strongest supporting solver chunks; avoid re-reading every probe chunk.",
        "When a final solver answer is high-confidence, grounded, and type-matched, commit it directly instead of running a broad synthesis pass.",
        "For bridge questions, phrase each dependent retrieval query with the resolved bridge answer and the final requested relation.",
    ]
    if failure_modes.get("over_budget", 0) > failure_modes.get("wrong_answer", 0):
        entries.append("When token cost rises above budget, reduce retrieval retries before shortening answer-span constraints.")
    if winner_counts:
        best = winner_counts.most_common(1)[0][0]
        entries.append(f"Deployment policy selected by train reward: {best}; keep this as a fixed policy at inference.")
    return "\n".join(f"- {e}" for e in entries)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/local/yzheng/pnair/workspace/reproduction/hera/data/train_240_v2.jsonl")
    ap.add_argument("--n-train", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retriever-url", default="http://node408:8003")
    ap.add_argument("--token-target", type=float, default=7000.0)
    ap.add_argument("--token-weight", type=float, default=0.25)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--output-dir", default="compiled/amas3_joint_tfgrpo")
    ap.add_argument("--policy-names", default="", help="comma-separated policy names to run")
    args = ap.parse_args()

    selected_names = {x.strip() for x in args.policy_names.split(",") if x.strip()}
    policies = [p for p in POLICIES if not selected_names or p.name in selected_names]
    if selected_names and len(policies) != len(selected_names):
        found = {p.name for p in policies}
        raise ValueError("unknown policies: " + str(sorted(selected_names - found)))

    rows = load_train(args.train)
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.n_train]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "rollouts.jsonl"
    summary_path = out / "summary.json"
    library_path = out / "experience.txt"
    policy_path = out / "policy.json"

    cfg = LMConfig(qwen_nothink_max_tokens=768)
    n_replicas = 3
    retriever = Retriever(base_url=args.retriever_url)
    sem = asyncio.Semaphore(args.concurrency)

    async def run_one(qidx: int, q: dict, pidx: int, policy: Policy) -> dict:
        async with sem:
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
            try:
                r = await pipe.run(q["question"], qid=str(q.get("id", "")))
                rew = reward(r.answer, q["answer"], r.total_tokens, args.token_target, args.token_weight)
                return {
                    "id": q.get("id", ""),
                    "source": q.get("source", ""),
                    "policy": policy.name,
                    "answer": r.answer,
                    "gold": q["answer"],
                    "f1": token_f1(r.answer, q["answer"]),
                    "contain": contain(r.answer, q["answer"]),
                    "reward": rew,
                    "tokens": r.total_tokens,
                    "topology": r.topology,
                    "plan_subgoals": r.plan_subgoals,
                    "n_retrieval_calls": r.n_retrieval_calls,
                    "n_solvers_invoked": r.n_solvers_invoked,
                    "wallclock": round(time.time() - t0, 3),
                }
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
                    "error": str(e)[:300],
                }

    tasks = []
    for qidx, q in enumerate(rows):
        for pidx, policy in enumerate(policies):
            tasks.append(run_one(qidx, q, pidx, policy))

    print(f"running {len(tasks)} train rollouts over {len(rows)} IRCoT-train questions")
    results = await asyncio.gather(*tasks)
    with trace_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_q[str(r["id"])].append(r)

    winner_counts: Counter = Counter()
    failure_modes: Counter = Counter()
    for qid, rs in by_q.items():
        best = max(rs, key=lambda r: r["reward"])
        winner_counts[best["policy"]] += 1
        for r in rs:
            if r["tokens"] > args.token_target:
                failure_modes["over_budget"] += 1
            if r["f1"] == 0.0:
                failure_modes["wrong_answer"] += 1

    stats: dict[str, dict] = {}
    for policy in policies:
        rs = [r for r in results if r["policy"] == policy.name]
        n = max(1, len(rs))
        stats[policy.name] = {
            "n": len(rs),
            "mean_reward": sum(r["reward"] for r in rs) / n,
            "mean_f1": sum(r["f1"] for r in rs) / n,
            "mean_contain": sum(r["contain"] for r in rs) / n,
            "mean_tokens": sum(r["tokens"] for r in rs) / n,
            "wins": winner_counts[policy.name],
        }

    best_policy_name = max(stats, key=lambda k: (stats[k]["wins"], stats[k]["mean_reward"]))
    best_policy = next(p for p in policies if p.name == best_policy_name)
    library = build_experience(winner_counts, stats, failure_modes)
    library_path.write_text(library)
    policy_path.write_text(json.dumps(asdict(best_policy), indent=2))
    summary = {
        "train_path": args.train,
        "n_train": len(rows),
        "token_target": args.token_target,
        "token_weight": args.token_weight,
        "policy_names": [p.name for p in policies],
        "stats": stats,
        "winner_counts": dict(winner_counts),
        "failure_modes": dict(failure_modes),
        "selected_policy": best_policy_name,
        "experience_file": str(library_path),
        "policy_file": str(policy_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
