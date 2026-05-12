"""Multi-plan GRPO at the plan/topology level (per-query, training-free).

Inspired by Training-Free GRPO (Cai et al., 2510.08191) and HERA's topology
optimization, but operates PER QUERY at inference time with no experience
library and no cross-query learning. The reward signal is retrieval grounding
(probe layer), not LLM-judgment or post-hoc EM.

Algorithm:
1. Sample K candidate plans from the planner LM at different temperatures.
2. Probe each plan's sub-questions in parallel (zero-LLM operation).
3. Compute aggregate plan reward = mean groundedness of probed sub-Qs.
4. Return the plan with highest reward and its probe results.

Single execution downstream (no answer ensembling).
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any
import dspy
from .planner import run_planner
from .probe import probe_all
from .retriever import Retriever
from .types import Plan, ProbeResult


@dataclass
class MultiPlanResult:
    chosen_plan: Plan
    chosen_probes: list[ProbeResult]
    chosen_reward: float
    candidate_rewards: list[float]
    candidate_plans_subgoals: list[int]
    candidate_temperatures: list[float]
    planner_tokens: int


def _aggregate_reward(probes: list[ProbeResult]) -> float:
    if not probes:
        return 0.0
    sub = probes[1:] if len(probes) > 1 else probes
    if not sub:
        return probes[0].groundedness
    grounds = [p.groundedness for p in sub]
    return sum(grounds) / max(len(grounds), 1)


def _make_temped_lm(base_lm: dspy.LM, temperature: float) -> dspy.LM:
    """Return a copy of base_lm with overridden temperature."""
    new_lm = dspy.LM(
        model=getattr(base_lm, 'model', base_lm.kwargs.get('model', '')),
        api_base=base_lm.kwargs.get('api_base'),
        api_key=base_lm.kwargs.get('api_key'),
        max_tokens=base_lm.kwargs.get('max_tokens', 4096),
        temperature=temperature,
        extra_body=base_lm.kwargs.get('extra_body', {}),
        cache=False,
    )
    return new_lm


async def run_multi_plan_grpo(
    *,
    planner_lm: dspy.LM,
    retriever: Retriever,
    question: str,
    experience: str = '',
    K: int = 3,
    temperatures: tuple[float, ...] = (0.4, 0.7, 0.9),
) -> MultiPlanResult:
    """Sample K plans, probe each, pick the best by aggregate groundedness."""
    if len(temperatures) < K:
        temperatures = tuple(list(temperatures) + [0.7] * (K - len(temperatures)))
    temps = list(temperatures[:K])

    async def _gen_plan(temp: float) -> tuple[Plan, float, int]:
        lm = _make_temped_lm(planner_lm, temp)
        plan = await asyncio.to_thread(run_planner, lm, question, experience)
        return plan, temp, plan.planner_tokens

    plans_with_meta = await asyncio.gather(*[_gen_plan(t) for t in temps])

    async def _probe_plan(plan: Plan) -> tuple[Plan, list[ProbeResult], float]:
        from .working_memory import FindingsBus
        bus = FindingsBus()
        sub_qs = [bus.interpolate(n.question) for n in plan.subgoals]
        probes = await probe_all(retriever=retriever, original_question=question, sub_questions=sub_qs)
        return plan, probes, _aggregate_reward(probes)

    probed = await asyncio.gather(*[_probe_plan(p) for p, _, _ in plans_with_meta])
    rewards = [r for _, _, r in probed]
    plan_subgoal_counts = [len(p.subgoals) for p, _, _ in probed]
    best_idx = max(range(len(probed)), key=lambda i: rewards[i])
    chosen_plan, chosen_probes, chosen_reward = probed[best_idx]
    total_planner_tokens = sum(t for _, _, t in plans_with_meta)
    return MultiPlanResult(
        chosen_plan=chosen_plan,
        chosen_probes=chosen_probes,
        chosen_reward=chosen_reward,
        candidate_rewards=rewards,
        candidate_plans_subgoals=plan_subgoal_counts,
        candidate_temperatures=temps,
        planner_tokens=total_planner_tokens,
    )
