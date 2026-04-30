#!/usr/bin/env python3
"""Compile the three AMAS DSPy signatures (DecomposeMultiHop,
ExtractAnswerSpan, WhTargetAlignedSynthesis) with GEPA reflective prompt
evolution.

Approach: standalone synchronous AmasProgram dspy.Module that wires the
same signatures into a Plan -> per-subgoal Retrieve+Extract -> Synth chain.
GEPA mutates the signature instructions; we save the compiled program and
later transfer its instruction strings into the production AmasPipeline
predictors.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

import dspy
from amas3.lm import LMConfig, make_qwen_think_lm, make_mini_lm
from amas3.retriever import Retriever
from amas3.planner import DecomposeMultiHop, _parse_plan
from amas3.solver import ExtractAnswerSpan, _parse_extraction, _format_chunks
from amas3.synthesizer import WhTargetAlignedSynthesis, _format_evidence, _parse_synth
from amas3.types import RetrievedChunk


def normalize_answer(s: str) -> str:
    import string
    s = (s or '').lower()
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split()).strip()


def metric_em(gold, pred, trace=None, pred_name=None, pred_trace=None):
    g = normalize_answer(getattr(gold, 'answer', ''))
    p = normalize_answer(getattr(pred, 'answer', ''))
    if not g or not p:
        return 0.0
    if g == p:
        return 1.0
    if g in p:
        return 0.5
    return 0.0


class AmasProgram(dspy.Module):
    """End-to-end AMAS as a sync dspy.Module GEPA can introspect.

    Uses three predictor attributes that GEPA can find: planner_pred,
    extract_pred, synth_pred. Pipeline logic (plan -> chain solve -> synth)
    lives in forward().

    Important: the retriever lives in forward() args, not as a module
    attribute, so dspy deep_copy works without serialising it.
    """

    def __init__(self):
        super().__init__()
        self.planner_pred = dspy.ChainOfThought(DecomposeMultiHop)
        self.extract_pred = dspy.Predict(ExtractAnswerSpan)
        self.synth_pred = dspy.ChainOfThought(WhTargetAlignedSynthesis)

    def forward(self, question: str) -> dspy.Prediction:
        retriever: Retriever = dspy.settings.amas_retriever
        planner_lm: dspy.LM = dspy.settings.amas_planner_lm
        worker_lm: dspy.LM = dspy.settings.amas_worker_lm
        synth_lm: dspy.LM = dspy.settings.amas_synth_lm

        with dspy.context(lm=planner_lm):
            plan_pred = self.planner_pred(question=question)
        plan, _ = _parse_plan(getattr(plan_pred, 'plan_json', ''), question)

        findings = {}
        chunks_per_node = {}
        for node in plan.subgoals:
            sub_q = node.question
            for nid, fans in findings.items():
                sub_q = sub_q.replace(f'<A.{nid}>', fans)
            chunk_lists = asyncio.run(retriever.retrieve_batch([sub_q]))
            chunks = chunk_lists[0] if chunk_lists else []
            chunks_per_node[node.id] = chunks
            chunks_json = _format_chunks(chunks)
            with dspy.context(lm=worker_lm):
                ex_pred = self.extract_pred(
                    sub_question=sub_q,
                    expected_answer_type=node.expected_answer_type,
                    chunks_json=chunks_json,
                )
            ex_obj = _parse_extraction(getattr(ex_pred, 'extraction_json', ''))
            findings[node.id] = str(ex_obj.get('answer_span', '')).strip()

        final_node = next((n for n in plan.subgoals if n.is_final), None) or plan.subgoals[-1]
        final_chunks = chunks_per_node.get(final_node.id, [])

        findings_summary = json.dumps([
            {'node_id': nid, 'sub_question': '', 'answer': a, 'status': 'ok' if a else 'no_evidence'}
            for nid, a in findings.items()
        ])
        final_evidence_json = _format_evidence(final_chunks)
        with dspy.context(lm=synth_lm):
            synth_pred = self.synth_pred(
                original_question=question,
                findings_summary=findings_summary,
                final_evidence_json=final_evidence_json,
            )
        obj = _parse_synth(getattr(synth_pred, 'final_json', ''))
        answer = str(obj.get('answer', '')).strip()
        return dspy.Prediction(answer=answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/musique/musique_train_500.json')
    ap.add_argument('--n-train', type=int, default=80)
    ap.add_argument('--n-val', type=int, default=40)
    ap.add_argument('--auto', choices=['light', 'medium', 'heavy'], default=None)
    ap.add_argument('--max-metric-calls', type=int, default=0)
    ap.add_argument('--num-threads', type=int, default=4)
    ap.add_argument('--output', default='compiled/amas_gepa.json')
    ap.add_argument('--log-dir', default='results/gepa_logs')
    args = ap.parse_args()

    os.environ.setdefault('DSPY_CACHEDIR', '/local/yzheng/pnair/.dspy_cache')

    cfg = LMConfig()
    planner_lm = make_qwen_think_lm(cfg, replica_idx=0)
    worker_lm = make_mini_lm(cfg)
    synth_lm = make_mini_lm(cfg)
    retriever = Retriever(base_url='http://node408:8003')

    dspy.settings.configure(lm=worker_lm)
    dspy.settings.amas_planner_lm = planner_lm
    dspy.settings.amas_worker_lm = worker_lm
    dspy.settings.amas_synth_lm = synth_lm
    dspy.settings.amas_retriever = retriever

    train_all = json.load(open(args.train))
    train = train_all[: args.n_train]
    val = train_all[args.n_train : args.n_train + args.n_val]

    trainset = [dspy.Example(question=q['question'], answer=q['answer']).with_inputs('question') for q in train]
    valset = [dspy.Example(question=q['question'], answer=q['answer']).with_inputs('question') for q in val]
    print(f'train: {len(trainset)}  val: {len(valset)}')

    program = AmasProgram()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    gepa_kwargs = dict(
        metric=metric_em,
        reflection_lm=dspy.LM(model='openai/gpt-4o-mini', max_tokens=2048, temperature=0.7),
        num_threads=args.num_threads,
        log_dir=args.log_dir,
        track_stats=True,
        seed=42,
    )
    if args.max_metric_calls and args.max_metric_calls > 0:
        gepa_kwargs['max_metric_calls'] = args.max_metric_calls
    elif args.auto:
        gepa_kwargs['auto'] = args.auto
    else:
        gepa_kwargs['auto'] = 'light'
    optimiser = dspy.GEPA(**gepa_kwargs)
    compiled = optimiser.compile(student=program, trainset=trainset, valset=valset)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out))
    print(f'saved compiled program to {out}')

    print('--- final signature instructions ---')
    for name, p in compiled.named_predictors():
        sig = p.signature
        instr = getattr(sig, 'instructions', '') or ''
        print(f'{name}: {len(instr)} chars')
        print((instr or '')[:500])
        print()


if __name__ == '__main__':
    main()
