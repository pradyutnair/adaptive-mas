#!/usr/bin/env python3
"""SAGE runner: Plan -> Probe -> Topology -> Solve -> Synth.

Reads a questions JSON file (list of {id, question, answer}), runs the SAGE
pipeline per question, writes predictions.jsonl in the format expected by
scripts/eval_offline.py.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

import dspy
from amas3.lm import LMConfig, make_qwen_think_lm, make_qwen_nothink_lm, make_mini_lm
from amas3.pipeline import AmasPipeline, AmasPipelineConfig
from amas3.retriever import Retriever


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True, help='Path to questions JSON list')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--retriever-url', default='http://node408:8003')
    ap.add_argument('--max-retrievals', type=int, default=3)
    ap.add_argument('--repair', dest='repair', action='store_true', default=True)
    ap.add_argument('--no-repair', dest='repair', action='store_false')
    ap.add_argument('--worker', choices=['mini', 'qwen_nothink'], default='mini')
    ap.add_argument('--planner-replica', type=int, default=0)
    ap.add_argument('--planner-model', choices=['qwen3-8b', 'qwen3-14b'], default='qwen3-8b')
    ap.add_argument('--limit', type=int, default=0, help='if > 0, only run first N questions')
    ap.add_argument('--qwen-think-budget', type=int, default=4096)
    ap.add_argument('--experience-file', default=None, help='path to GRPO-compiled experience library txt')
    ap.add_argument('--use-multi-plan', action='store_true', default=False, help='enable plan-level GRPO (K plans, pick by probe groundedness)')
    ap.add_argument('--K-plans', type=int, default=3)
    ap.add_argument('--use-bridge-resolver', action='store_true', default=False, help='enable bridge-resolution preprocessor')
    ap.add_argument('--bridge-g-threshold', type=float, default=0.45)
    ap.add_argument('--concurrency', type=int, default=4)
    return ap.parse_args()


def load_questions(path: str) -> list[dict]:
    obj = json.load(open(path))
    if isinstance(obj, list):
        return obj
    raise ValueError(f'expected list, got {type(obj)}')


async def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s | %(message)s')

    cfg = LMConfig(qwen_think_max_tokens=args.qwen_think_budget)
    # Round-robin across 3 vLLM replicas to use all 3 GPUs.
    n_replicas = 3
    if args.planner_model == 'qwen3-14b':
        from amas3.lm import make_qwen14b_think_lm
        # Single 14B endpoint, replicate the LM reference for round-robin (all hit port 8003)
        planner_lms = [make_qwen14b_think_lm(cfg, replica_idx=i) for i in range(n_replicas)]
    else:
        planner_lms = [make_qwen_think_lm(cfg, replica_idx=i) for i in range(n_replicas)]
    if args.worker == 'mini':
        worker_lms = [make_mini_lm(cfg) for _ in range(n_replicas)]
        synth_lms = [make_mini_lm(cfg) for _ in range(n_replicas)]
    else:
        # All-Qwen: stagger replicas to avoid hammering one GPU.
        worker_lms = [make_qwen_nothink_lm(cfg, replica_idx=(i + 1) % n_replicas) for i in range(n_replicas)]
        synth_lms = [make_qwen_nothink_lm(cfg, replica_idx=(i + 2) % n_replicas) for i in range(n_replicas)]

    retriever = Retriever(base_url=args.retriever_url)
    experience_text = ''
    if args.experience_file:
        from pathlib import Path as _P
        if _P(args.experience_file).exists():
            experience_text = _P(args.experience_file).read_text().strip()
            print(f'loaded experience library: {len(experience_text)} chars from {args.experience_file}')

    pipelines = [
        AmasPipeline(
            planner_lm=planner_lms[i],
            worker_lm=worker_lms[i],
            synth_lm=synth_lms[i],
            retriever=retriever,
            config=AmasPipelineConfig(
                max_retrievals_per_solver=args.max_retrievals,
                repair_enabled=args.repair,
                experience_library=experience_text,
                use_multi_plan=args.use_multi_plan,
                K_plans=args.K_plans,
                use_bridge_resolver=args.use_bridge_resolver,
                bridge_g_threshold=args.bridge_g_threshold,
            ),
        )
        for i in range(n_replicas)
    ]

    questions = load_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    pred_path = out_dir / 'predictions.jsonl'
    config_path = out_dir / 'run_config.json'
    config_path.write_text(json.dumps({
        'questions_path': args.questions,
        'n_questions': len(questions),
        'planner': args.planner_model + '+think',
        'worker': args.worker,
        'synth': 'gpt-4o-mini',
        'retriever_url': args.retriever_url,
        'max_retrievals_per_solver': args.max_retrievals,
        'repair_enabled': args.repair,
        'qwen_think_budget': args.qwen_think_budget,
        'concurrency': args.concurrency,
        'experience_file': args.experience_file,
        'use_multi_plan': args.use_multi_plan,
        'K_plans': args.K_plans,
        'use_bridge_resolver': args.use_bridge_resolver,
        'bridge_g_threshold': args.bridge_g_threshold,
    }, indent=2))

    sem = asyncio.Semaphore(args.concurrency)
    rows_written = 0
    t0 = time.time()

    async def process_one(idx: int, q: dict) -> dict:
        async with sem:
            qid = q.get('id', f'q{idx}')
            question = q.get('question', '')
            gold = q.get('answer', '')
            try:
                r = await pipelines[idx % n_replicas].run(question=question, qid=qid)
                row = {
                    'id': qid,
                    'question': question,
                    'gold_answer': gold,
                    'answer': r.answer,
                    'prediction': r.answer,
                    'metadata': {
                        'total_tokens': r.total_tokens,
                        'planner_tokens': r.planner_tokens,
                        'solver_tokens': r.solver_tokens,
                        'synth_tokens': r.synth_tokens,
                        'rewrite_tokens': r.rewrite_tokens,
                        'plan_subgoals': r.plan_subgoals,
                        'topology': r.topology,
                        'topology_rationale': r.topology_rationale,
                        'probe_groundedness': r.probe_groundedness,
                        'n_retrieval_calls': r.n_retrieval_calls,
                        'n_solvers_invoked': r.n_solvers_invoked,
                        'repair_invoked': r.repair_invoked,
                        'wallclock_seconds': r.wallclock_seconds,
                        'answer_type': r.answer_type,
                        'support_ids': r.support_ids,
                        'justification': r.justification,
                        'findings': r.findings,
                        'bridge_resolved': r.bridge_resolved,
                        'bridge_resolver_tokens': r.bridge_resolver_tokens,
                        'multi_plan_rewards': r.multi_plan_rewards,
                        'multi_plan_subgoal_counts': r.multi_plan_subgoal_counts,
                        'multi_plan_temperatures': r.multi_plan_temperatures,
                    },
                }
            except Exception as e:
                logging.exception('error on %s', qid)
                row = {
                    'id': qid,
                    'question': question,
                    'gold_answer': gold,
                    'answer': '',
                    'prediction': '',
                    'metadata': {'total_tokens': 0, 'error': str(e)[:300]},
                }
            return row

    with open(pred_path, 'w') as f:
        tasks = [asyncio.create_task(process_one(i, q)) for i, q in enumerate(questions)]
        for fut in asyncio.as_completed(tasks):
            row = await fut
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            rows_written += 1
            if rows_written % 5 == 0:
                elapsed = time.time() - t0
                logging.info('progress: %d/%d (%.1fs, %.1fs/q)', rows_written, len(questions), elapsed, elapsed / max(rows_written, 1))

    logging.info('done: %d predictions written to %s', rows_written, pred_path)


if __name__ == '__main__':
    asyncio.run(main())
