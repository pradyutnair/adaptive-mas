#!/usr/bin/env python3
"""Training-Free GRPO compile loop for AMAS (faithful to arXiv 2510.08191).

Implements the paper's three-stage learning step on the AMAS multi-hop QA
pipeline:

  1. Rollout & Reward
     - For each query in the current batch, run G rollouts of the AMAS
       pipeline conditioned on (q, E) and score with partial-EM.
     - Skip groups with std=0 (no winner-vs-loser signal).

  2. Group Advantage (Fig. 11 + Fig. 12 of the paper)
     - Summarise each rollout (keeping retrieval queries as the
       transferable signal) using SummarizeRollout.
     - From the G summaries + scores + current library E, extract a small
       list of OP PROPOSALS (ADD / MODIFY / DELETE / KEEP) per group.

  3. Optimization (Fig. 13 of the paper)
     - Once per BATCH, consolidate ALL group proposals through OptimizeBatch
       which can MERGE redundant entries, MODIFY, ADD, or DELETE.
     - Apply ops to the experience library E.

Notable improvements over the per-question version:
  - True batch-level optimization (paper-faithful).
  - Stable experience IDs (E1..En) so MERGE/MODIFY/DELETE survive across steps.
  - Larger default group size (G=5).
  - Rollout-temperature knob is actually wired into all three pipeline LMs.
  - Replica round-robin only distributes load across vLLM endpoints; the
    underlying frozen policy stays identical.
  - Retrieval sub-queries are preserved in summaries.
  - Per-batch JSONL log of the full update trace for reproducibility.

Inputs : MuSiQue training subset (default 100 questions, 3 epochs, G=5).
Output : compiled/amas_grpo_E.txt (final library) + per-batch JSONL log.
"""
import argparse
import asyncio
import json
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

import dspy
from amas3.lm import LMConfig, make_qwen_think_lm, make_mini_lm
from amas3.pipeline import AmasPipeline, AmasPipelineConfig
from amas3.retriever import Retriever
from amas3.grpo_signatures import (
    SummarizeRollout,
    ExtractGroupOps,
    OptimizeBatch,
)


def normalize_answer(s):
    s = (s or '').lower()
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split()).strip()


def score_em_partial(pred, gold):
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not g or not p:
        return 0.0
    if p == g:
        return 1.0
    if g in p or p in g:
        return 0.5
    return 0.0


def parse_json_obj(raw: str) -> dict:
    text = (raw or '').strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass
class Experience:
    eid: str
    text: str
    born_step: int = 0


@dataclass
class ExperienceLibrary:
    """Mutable library of natural-language experiences with stable IDs.

    Format used in prompts: "E1: <text>\\nE2: <text>\\n..." (so the LLM can
    reference entries by id when proposing MODIFY/MERGE/DELETE ops).

    Format passed to the AMAS pipeline: bullet list with no IDs.
    """
    entries: list[Experience] = field(default_factory=list)
    counter: int = 0

    def next_id(self) -> str:
        self.counter += 1
        return f"E{self.counter}"

    def format_with_ids(self) -> str:
        if not self.entries:
            return ""
        return "\n".join(f"{e.eid}: {e.text}" for e in self.entries)

    def format_for_pipeline(self) -> str:
        if not self.entries:
            return ""
        return "\n".join(f"- {e.text}" for e in self.entries)

    def to_serialisable(self) -> list[dict]:
        return [{"id": e.eid, "text": e.text, "born_step": e.born_step} for e in self.entries]

    def find(self, eid: str) -> Experience | None:
        for e in self.entries:
            if e.eid == eid:
                return e
        return None

    def apply_ops(self, ops: list[dict], step: int) -> list[str]:
        """Apply a list of update ops; return a list of human-readable change descriptions."""
        changes: list[str] = []
        for op in ops:
            if not isinstance(op, dict):
                continue
            kind = str(op.get('op', '')).upper()
            if kind == 'ADD':
                t = str(op.get('text', '')).strip()
                if t:
                    eid = self.next_id()
                    self.entries.append(Experience(eid=eid, text=t, born_step=step))
                    changes.append(f"+{eid}: {t}")
            elif kind == 'MODIFY':
                eid = str(op.get('id', '') or op.get('modified_from', '')).strip()
                t = str(op.get('text', '')).strip()
                e = self.find(eid) if eid else None
                if e and t:
                    old = e.text
                    e.text = t
                    changes.append(f"~{eid}: {old!r} -> {t!r}")
            elif kind == 'DELETE':
                eid = str(op.get('id', '') or op.get('delete_id', '')).strip()
                e = self.find(eid) if eid else None
                if e:
                    self.entries = [x for x in self.entries if x.eid != eid]
                    changes.append(f"-{eid}: {e.text}")
            elif kind == 'MERGE':
                ids = op.get('ids') or op.get('merged_from') or []
                t = str(op.get('text', '')).strip()
                if not isinstance(ids, list):
                    continue
                ids = [str(x).strip() for x in ids if str(x).strip()]
                if t and len(ids) >= 2:
                    targets = [self.find(i) for i in ids]
                    targets = [x for x in targets if x is not None]
                    if len(targets) >= 2:
                        keep = set(ids)
                        self.entries = [x for x in self.entries if x.eid not in keep]
                        new_eid = self.next_id()
                        self.entries.append(Experience(eid=new_eid, text=t, born_step=step))
                        changes.append(f"M{','.join(ids)}->{new_eid}: {t}")
            elif kind == 'KEEP':
                continue
        return changes


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/musique/musique_train_500.json')
    ap.add_argument('--n-train', type=int, default=100, help='Paper uses 100 examples.')
    ap.add_argument('--epochs', type=int, default=3, help='Paper uses 3 epochs.')
    ap.add_argument('--group-size', type=int, default=5, help='G; paper uses 5 (math) / 3 (web).')
    ap.add_argument('--batch-size', type=int, default=10,
                    help='Number of queries whose group advantages are pooled into one optimization step.')
    ap.add_argument('--rollout-temperature', type=float, default=0.7,
                    help='Temperature for ALL rollout LMs (planner/worker/synth) to ensure intra-group diversity.')
    ap.add_argument('--retriever-url', default='http://node408:8003')
    ap.add_argument('--max-retrievals', type=int, default=2)
    ap.add_argument('--concurrency', type=int, default=8, help='Max concurrent rollouts.')
    ap.add_argument('--n-replicas', type=int, default=3, help='Number of vLLM endpoints to round-robin across.')
    ap.add_argument('--no-ground-truth', action='store_true',
                    help='Robustness ablation: hide gold answers during semantic-advantage extraction.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='compiled/amas_grpo_E.txt')
    ap.add_argument('--log', default='results/grpo_logs/compile.log')
    ap.add_argument('--trace', default='results/grpo_logs/compile_trace.jsonl')
    args = ap.parse_args()

    os.environ.setdefault('DSPY_CACHEDIR', '/local/yzheng/pnair/.dspy_cache')
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
    logf = open(args.log, 'w')
    tracef = open(args.trace, 'w')

    def log(msg):
        ts = time.strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        logf.write(line + '\n')
        logf.flush()

    def trace(obj):
        tracef.write(json.dumps(obj, ensure_ascii=False) + '\n')
        tracef.flush()

    cfg = LMConfig()
    n_replicas = max(1, args.n_replicas)
    rollout_planner_lms = [
        make_qwen_think_lm(cfg, replica_idx=i, temperature=args.rollout_temperature)
        for i in range(n_replicas)
    ]
    rollout_worker_lms = [
        make_mini_lm(cfg, temperature=args.rollout_temperature) for _ in range(n_replicas)
    ]
    rollout_synth_lms = [
        make_mini_lm(cfg, temperature=args.rollout_temperature) for _ in range(n_replicas)
    ]
    reflection_lm = make_mini_lm(cfg, temperature=0.0, max_tokens=2048)
    retriever = Retriever(base_url=args.retriever_url)

    train_all = json.load(open(args.train))
    random.seed(args.seed)
    random.shuffle(train_all)
    train = train_all[: args.n_train]
    log(
        f'TRAIN: {len(train)} questions | epochs={args.epochs} | G={args.group_size} | '
        f'batch={args.batch_size} | rollout_T={args.rollout_temperature} | '
        f'GT={"off" if args.no_ground_truth else "on"}'
    )

    library = ExperienceLibrary()
    sem = asyncio.Semaphore(args.concurrency)
    step_idx = 0

    async def one_rollout(replica_idx: int, q: dict, exp_text: str) -> dict:
        async with sem:
            cfg_pipe = AmasPipelineConfig(
                max_retrievals_per_solver=args.max_retrievals,
                repair_enabled=False,
                experience_library=exp_text,
            )
            pipe = AmasPipeline(
                planner_lm=rollout_planner_lms[replica_idx],
                worker_lm=rollout_worker_lms[replica_idx],
                synth_lm=rollout_synth_lms[replica_idx],
                retriever=retriever,
                config=cfg_pipe,
            )
            try:
                r = await pipe.run(question=q['question'], qid=q.get('id', ''))
                ans = r.answer
                plan_subgoals = json.dumps([
                    {'node_id': f['node_id'], 'sub_question': f['sub_question']}
                    for f in r.findings
                ])
                findings_lite = json.dumps([
                    {
                        'node_id': f['node_id'],
                        'sub_question': f['sub_question'],
                        'answer': f['answer'],
                        'status': f['status'],
                        'confidence': round(float(f.get('confidence', 0.0)), 3),
                    }
                    for f in r.findings
                ])
                return {
                    'answer': ans,
                    'plan_subgoals': plan_subgoals,
                    'findings': findings_lite,
                    'score': score_em_partial(ans, q['answer']),
                }
            except (RuntimeError, ValueError) as e:
                return {
                    'answer': '', 'plan_subgoals': '[]', 'findings': '[]',
                    'score': 0.0, 'error': str(e)[:200],
                }

    def summarize_one(q: dict, rollout: dict, current_e: str) -> str:
        gold = q['answer'] if not args.no_ground_truth else ''
        with dspy.context(lm=reflection_lm):
            mod = dspy.Predict(SummarizeRollout)
            pred = mod(
                question=q['question'],
                gold_answer=gold,
                plan_subgoals=rollout['plan_subgoals'],
                findings=rollout['findings'],
                final_answer=rollout['answer'],
                score=rollout['score'],
                current_experience_library=current_e,
            )
        return str(getattr(pred, 'summary', '')).strip()[:1200]

    def extract_group_ops(q: dict, rollouts: list[dict], current_e: str) -> dict:
        summaries = [
            {'summary': r.get('_summary', ''), 'score': round(r['score'], 3)}
            for r in rollouts
        ]
        with dspy.context(lm=reflection_lm):
            mod = dspy.Predict(ExtractGroupOps)
            pred = mod(
                question=q['question'],
                summaries_with_scores=json.dumps(summaries, ensure_ascii=False),
                current_experience_library=current_e,
            )
        obj = parse_json_obj(getattr(pred, 'operations_json', ''))
        ops = obj.get('operations', []) if isinstance(obj, dict) else []
        return {
            'reasoning': obj.get('reasoning', '') if isinstance(obj, dict) else '',
            'operations': ops if isinstance(ops, list) else [],
        }

    def optimize_batch(current_e: str, batch_proposals: list[dict]) -> dict:
        if not batch_proposals:
            return {'reasoning': '', 'operations': []}
        with dspy.context(lm=reflection_lm):
            mod = dspy.Predict(OptimizeBatch)
            pred = mod(
                current_experience_library=current_e,
                batch_proposals_json=json.dumps(batch_proposals, ensure_ascii=False),
            )
        obj = parse_json_obj(getattr(pred, 'operations_json', ''))
        ops = obj.get('operations', []) if isinstance(obj, dict) else []
        return {
            'reasoning': obj.get('reasoning', '') if isinstance(obj, dict) else '',
            'operations': ops if isinstance(ops, list) else [],
        }

    total_rollouts = 0
    total_groups = 0
    total_active_groups = 0
    total_batches = 0
    total_ops_applied = 0

    for epoch in range(args.epochs):
        random.shuffle(train)
        log(f'=== epoch {epoch + 1}/{args.epochs} ===')

        for batch_start in range(0, len(train), args.batch_size):
            batch = train[batch_start: batch_start + args.batch_size]
            current_e_text = library.format_with_ids()
            pipeline_e_text = library.format_for_pipeline()
            step_idx += 1
            total_batches += 1

            roll_tasks = []
            for q in batch:
                for g in range(args.group_size):
                    roll_tasks.append(one_rollout(g % n_replicas, q, pipeline_e_text))
            t0 = time.time()
            flat_rollouts = await asyncio.gather(*roll_tasks)
            roll_dt = time.time() - t0
            total_rollouts += len(flat_rollouts)

            grouped = []
            for qi, q in enumerate(batch):
                rs = flat_rollouts[qi * args.group_size: (qi + 1) * args.group_size]
                grouped.append((q, rs))
                total_groups += 1

            active = []
            for q, rs in grouped:
                scores = [r['score'] for r in rs]
                if max(scores) - min(scores) > 1e-9:
                    active.append((q, rs, scores))
            total_active_groups += len(active)

            if not active:
                log(
                    f'epoch{epoch + 1} step{step_idx} | batch={len(batch)} '
                    f'rollouts={len(flat_rollouts)} ({roll_dt:.1f}s) | active=0 | skip'
                )
                trace({'epoch': epoch + 1, 'step': step_idx, 'reason': 'no_active_groups'})
                continue

            for q, rs, _ in active:
                for r in rs:
                    r['_summary'] = summarize_one(q, r, current_e_text)

            batch_proposals = []
            for q, rs, scores in active:
                got = extract_group_ops(q, rs, current_e_text)
                if got['operations']:
                    batch_proposals.append({
                        'question': q['question'],
                        'gold_answer': '' if args.no_ground_truth else q['answer'],
                        'scores': [round(s, 3) for s in scores],
                        'reasoning': got['reasoning'],
                        'ops': got['operations'],
                    })

            if not batch_proposals:
                log(
                    f'epoch{epoch + 1} step{step_idx} | batch={len(batch)} '
                    f'active={len(active)} | proposals=0 | skip'
                )
                trace({'epoch': epoch + 1, 'step': step_idx, 'reason': 'no_proposals',
                       'n_active': len(active)})
                continue

            optim = optimize_batch(current_e_text, batch_proposals)
            changes = library.apply_ops(optim['operations'], step=step_idx)
            total_ops_applied += len(changes)

            log(
                f'epoch{epoch + 1} step{step_idx} | batch={len(batch)} '
                f'rollouts={len(flat_rollouts)} ({roll_dt:.1f}s) | '
                f'active={len(active)} props={len(batch_proposals)} '
                f'ops={len(optim["operations"])} applied={len(changes)} '
                f'|E|={len(library.entries)}'
            )
            for c in changes:
                log(f'    {c}')

            trace({
                'epoch': epoch + 1,
                'step': step_idx,
                'batch_size': len(batch),
                'n_active': len(active),
                'group_proposals': batch_proposals,
                'optimizer_reasoning': optim['reasoning'],
                'optimizer_ops': optim['operations'],
                'changes': changes,
                'library_after': library.to_serialisable(),
            })

    final_text = library.format_for_pipeline()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final_text)
    json_out = out.with_suffix('.json')
    json_out.write_text(json.dumps(library.to_serialisable(), ensure_ascii=False, indent=2))

    log('=== DONE ===')
    log(
        f'rollouts={total_rollouts} groups={total_groups} active_groups={total_active_groups} '
        f'batches={total_batches} ops_applied={total_ops_applied} |E|={len(library.entries)}'
    )
    log(f'wrote E -> {out}  (+ structured: {json_out})')

    print('\n=== Final Experience Library ===')
    print(final_text or '(empty)')

    logf.close()
    tracef.close()


if __name__ == '__main__':
    asyncio.run(main())
