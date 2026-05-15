"""Planner: Qwen3-8B + thinking, dspy.ChainOfThought.

Produces an atomic decomposition of the multi-hop question.
- Each sub-question is single-hop and answerable by one retrieval call.
- Dependencies are encoded as <A.I> tags inside child sub-question text:
  e.g. "What year was <A.1> born?" depends on the answer of node 1.
- One subgoal is tagged is_final=true; its answer is the final answer.

We use dspy.ChainOfThought because thinking-mode Qwen3 produces strong
multi-step reasoning over decomposition. Output is a JSON-shaped Plan.
"""
from __future__ import annotations
import json
import re
from typing import Any
import dspy
from .types import Plan, SubgoalNode


class DecomposeMultiHop(dspy.Signature):
    """Decompose a multi-hop question into atomic single-hop sub-questions.

Each sub-question must be answerable by ONE retrieval over Wikipedia plus
ONE short reading step. If a sub-question depends on an earlier answer,
reference it inline using <A.I> where I is the earlier subgoal id.

Output STRICT JSON with keys:
- subgoals: list of {id, question, depends_on, expected_answer_type, is_final, rationale}
- final_id: id of the subgoal whose answer is the final answer
- reasoning: one-paragraph plan summary

ids start at 1 and are sequential. depends_on is a list of earlier ids.
expected_answer_type in {person, place, date, number, yes_no, entity, other}.
Exactly one subgoal has is_final=true.

Decomposition rules:
- Preserve the answer category named by the original wh-phrase. If the
  question asks "what rocket/company/source", the final subgoal must ask for
  that category, not for an intermediate spacecraft, founder, or location.
- Preserve descriptor head nouns. For "the company founded as X", first ask
  which company matches that descriptor, then ask the final attribute of that
  company.
- For comparisons and superlatives, retrieve each candidate's comparable
  attribute before the final decision subgoal.
- The final subgoal should restate the original wh-target with resolved
  <A.I> placeholders; it must not ask for a bridge entity unless the original
  question asks for that bridge.
"""
    question: str = dspy.InputField()
    plan_json: str = dspy.OutputField(desc='Strict JSON object with subgoals, final_id, reasoning')


import os as _os
_MAX_SUBGOALS = int(_os.environ.get('AMAS_MAX_SUBGOALS', 6))


def _parse_plan(raw: str, original_question: str, max_subgoals: int | None = None) -> tuple[Plan, str]:
    """Parse the planner's JSON output. Falls back to a single-node plan on errors."""
    cap = max(1, int(max_subgoals or _MAX_SUBGOALS))
    text = raw.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
    except Exception:
        node = SubgoalNode(id=1, question=original_question, is_final=True, rationale='parse_fallback')
        return Plan(subgoals=[node], final_id=1, raw={}), 'parse_failed'

    raw_subgoals = obj.get('subgoals', [])
    subgoals: list[SubgoalNode] = []
    seen_ids = set()
    for i, item in enumerate(raw_subgoals[:cap], start=1):
        if not isinstance(item, dict):
            continue
        try:
            nid = int(item.get('id', i))
        except (TypeError, ValueError):
            nid = i
        if nid in seen_ids:
            nid = max(seen_ids, default=0) + 1
        seen_ids.add(nid)
        question_str = str(item.get('question', '')).strip()
        if not question_str:
            continue
        deps_raw = item.get('depends_on', []) or []
        deps: list[int] = []
        if isinstance(deps_raw, list):
            for d in deps_raw:
                try:
                    deps.append(int(d))
                except (TypeError, ValueError):
                    pass
        node = SubgoalNode(
            id=nid,
            question=question_str,
            depends_on=[d for d in deps if d != nid],
            expected_answer_type=str(item.get('expected_answer_type', 'entity')),
            is_final=bool(item.get('is_final', False)),
            rationale=str(item.get('rationale', '')),
        )
        subgoals.append(node)

    if not subgoals:
        subgoals = [SubgoalNode(id=1, question=original_question, is_final=True, rationale='empty_plan_fallback')]

    valid_ids = {n.id for n in subgoals}
    for n in subgoals:
        n.depends_on = [d for d in n.depends_on if d in valid_ids and d != n.id]

    finals = [n for n in subgoals if n.is_final]
    if not finals:
        subgoals[-1].is_final = True
        final_id = subgoals[-1].id
    else:
        final_id = finals[0].id
        for n in subgoals:
            n.is_final = (n.id == final_id)

    reasoning = str(obj.get('reasoning', ''))[:500]
    plan = Plan(subgoals=subgoals, final_id=final_id, raw=obj if isinstance(obj, dict) else {}, reasoning=reasoning)
    return plan, 'ok'


def _maybe_with_experience(sig_class, experience: str):
    if not experience:
        return sig_class
    base_instr = sig_class.instructions or sig_class.__doc__ or ""
    new_instr = f"Prior experiential knowledge from past attempts:\n{experience}\n\n" + base_instr
    return sig_class.with_instructions(new_instr)


def run_planner(
    planner_lm: dspy.LM,
    question: str,
    experience: str = "",
    max_subgoals: int | None = None,
) -> Plan:
    """Run the Planner once. Returns a Plan with planner_tokens populated."""
    sig = _maybe_with_experience(DecomposeMultiHop, experience)
    if max_subgoals:
        base_instr = sig.instructions or sig.__doc__ or ""
        sig = sig.with_instructions(
            f"Efficiency constraint: produce at most {max_subgoals} subgoals. "
            "Combine bridge lookup and attribute lookup when one retrieval can answer both. "
            "For comparisons, use one subgoal per candidate attribute plus one final decision at most.\n\n"
            + base_instr
        )
    with dspy.context(lm=planner_lm):
        cot = dspy.ChainOfThought(sig)
        pred = cot(question=question)
    plan, status = _parse_plan(pred.plan_json, question, max_subgoals=max_subgoals)
    try:
        history = planner_lm.history[-1] if planner_lm.history else None
        usage = (history or {}).get('usage') or {}
        plan.planner_tokens = int(usage.get('total_tokens', 0))
    except Exception:
        plan.planner_tokens = 0
    plan.raw['_parse_status'] = status
    return plan
