#!/usr/bin/env python3
"""Run a GEPA-compiled AmasProgram over a questions JSON file and write
predictions.jsonl in the eval-compatible schema.

This uses the simplified 3-step pipeline (Plan -> Chain Solve -> Synth)
embedded inside scripts/compile_amas_gepa.py::AmasProgram. No probe layer
or topology selector, just the prompts GEPA optimised.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import dspy
from amas3.lm import LMConfig, make_qwen_think_lm, make_mini_lm
from amas3.retriever import Retriever
from compile_amas_gepa import AmasProgram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--compiled', required=True)
    ap.add_argument('--retriever-url', default='http://node408:8003')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--concurrency', type=int, default=4)
    args = ap.parse_args()

    os.environ.setdefault('DSPY_CACHEDIR', '/local/yzheng/pnair/.dspy_cache')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LMConfig()
    planner_lm = make_qwen_think_lm(cfg, replica_idx=0)
    worker_lm = make_mini_lm(cfg)
    synth_lm = make_mini_lm(cfg)
    retriever = Retriever(base_url=args.retriever_url)

    dspy.settings.configure(lm=worker_lm)
    dspy.settings.amas_planner_lm = planner_lm
    dspy.settings.amas_worker_lm = worker_lm
    dspy.settings.amas_synth_lm = synth_lm
    dspy.settings.amas_retriever = retriever

    program = AmasProgram()
    program.load(args.compiled)
    print(f'loaded compiled program from {args.compiled}')

    print('compiled instructions:')
    for name, p in program.named_predictors():
        instr = getattr(p.signature, 'instructions', '') or ''
        print(f'  {name}: {len(instr)} chars')

    questions = json.load(open(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]

    pred_path = out_dir / 'predictions.jsonl'
    config_path = out_dir / 'run_config.json'
    config_path.write_text(json.dumps({
        'compiled_path': args.compiled,
        'questions_path': args.questions,
        'n_questions': len(questions),
        'planner': 'Qwen3-8B+think (compiled)',
        'worker': 'gpt-4o-mini (compiled)',
        'synth': 'gpt-4o-mini (compiled)',
    }, indent=2))

    sem = asyncio.Semaphore(args.concurrency)
    rows_done = 0
    t0 = time.time()

    async def process_one(q):
        async with sem:
            qid = q.get('id', '')
            question = q.get('question', '')
            gold = q.get('answer', '')
            try:
                pred = await asyncio.to_thread(program, question=question)
                ans = str(getattr(pred, 'answer', '')).strip()
                row = {
                    'id': qid,
                    'question': question,
                    'gold_answer': gold,
                    'answer': ans,
                    'prediction': ans,
                    'metadata': {},
                }
            except Exception as e:
                row = {
                    'id': qid,
                    'question': question,
                    'gold_answer': gold,
                    'answer': '',
                    'prediction': '',
                    'metadata': {'error': str(e)[:300]},
                }
            return row

    async def main_async():
        nonlocal rows_done
        with open(pred_path, 'w') as f:
            tasks = [asyncio.create_task(process_one(q)) for q in questions]
            for fut in asyncio.as_completed(tasks):
                row = await fut
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                rows_done += 1
                if rows_done % 5 == 0:
                    elapsed = time.time() - t0
                    print(f'progress: {rows_done}/{len(questions)} ({elapsed:.0f}s, {elapsed/max(rows_done,1):.1f}s/q)', flush=True)

    asyncio.run(main_async())
    print(f'wrote {rows_done} predictions to {pred_path}')


if __name__ == '__main__':
    main()
