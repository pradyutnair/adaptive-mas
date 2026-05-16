"""Single and group rollout drivers for TF-GRPO."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

from .experience_library import ExperienceLibrary
from .metrics import compute_contain, compute_em, compute_f1, compute_task_reward
from .profiles import characterize_query_profile
from .rewards import (
    TOKEN_BUDGET_BASELINES,
    compute_dual_reward,
    compute_token_efficiency_reward,
)
from .rollout import GroupResult, Rollout
from .topology import (
    config_from_topology,
    sample_topology,
    topology_mutations,
    topology_signature,
)
from ..lm import make_qwen14b_nothink_lm
from ..pipeline import AmasPipeline, AmasPipelineConfig
from ..retriever import Retriever

logger = logging.getLogger(__name__)


async def run_single_rollout(
    pipeline: AmasPipeline,
    question: str,
    qid: str,
    gold_answer: str,
    temperature: float,
    policy_name: str = "",
    sampled_topology: dict | None = None,
    dataset: str = "default",
    reward_alpha: float = 0.7,
    deployment_budget: int | None = None,
) -> Rollout:
    """Execute one pipeline run and score it with the dual reward."""
    result = await pipeline.run(question, qid)
    pred = result.answer or ""
    em = compute_em(pred, gold_answer)
    f1 = compute_f1(pred, gold_answer)
    contain = compute_contain(pred, gold_answer)

    sampler_tokens = int((sampled_topology or {}).get("_sampler_tokens", 0))
    scored_total_tokens = result.total_tokens + sampler_tokens

    answered = bool(pred.strip())
    token_eff = compute_token_efficiency_reward(
        scored_total_tokens, dataset, deployment_budget=deployment_budget,
    )
    dual_r = compute_dual_reward(
        em, f1, scored_total_tokens, dataset, alpha=reward_alpha,
        answered=answered, contain=contain,
        deployment_budget=deployment_budget,
    )

    result_dict = asdict(result) if hasattr(result, "__dataclass_fields__") else {}

    return Rollout(
        question_id=qid,
        question=question,
        gold_answer=gold_answer,
        temperature=temperature,
        policy_name=policy_name,
        predicted_answer=pred,
        em=em,
        f1=f1,
        contain=contain,
        total_tokens=scored_total_tokens,
        topology=result.topology,
        sampled_topology=sampled_topology or {},
        plan_subgoals=result.plan_subgoals,
        findings=result.findings,
        wallclock_seconds=result.wallclock_seconds,
        result=result_dict,
        dual_reward=dual_r,
        token_efficiency=token_eff,
        dataset=dataset,
    )


def needs_topology_mutation(group: GroupResult, dataset: str = "default") -> bool:
    if not group.rollouts:
        return False
    if max(r.f1 for r in group.rollouts) > 0.05:
        return False
    all_failed = all(not (r.predicted_answer or "").strip() or r.f1 == 0.0 for r in group.rollouts)
    if not all_failed:
        return False
    baseline = TOKEN_BUDGET_BASELINES.get(dataset, TOKEN_BUDGET_BASELINES["default"])
    avg_tokens = sum(max(0, int(r.total_tokens)) for r in group.rollouts) / max(1, len(group.rollouts))
    min_tokens = min(max(0, int(r.total_tokens)) for r in group.rollouts)
    # Structural fallback is useful only for cheap failures. If the group
    # already spent near-budget tokens and still failed, extra mutation
    # rollouts are pure waste.
    return avg_tokens <= 0.9 * baseline and min_tokens <= 0.75 * baseline


async def run_group_rollouts(
    question: str,
    qid: str,
    gold_answer: str,
    retriever: Retriever,
    config: AmasPipelineConfig,
    temperatures: tuple[float, ...] = (0.4, 0.7, 0.9),
    library: ExperienceLibrary | None = None,
    dataset: str = "default",
    reward_alpha: float = 0.7,
    deployment_budget: int | None = None,
    forced_diversity: bool = True,
) -> GroupResult:
    """Run K same-query rollouts, score with dual reward, rank by task then efficiency.

    When ``deployment_budget`` is set (budget-conditioned policy), every
    rollout in this group shares the same B for both topology sampling and
    reward shaping. The training loop is expected to sample a fresh B per
    group so the policy learns to adapt across budgets.
    """
    group = GroupResult(question_id=qid, question=question, gold_answer=gold_answer)
    group.deployment_budget = int(deployment_budget) if deployment_budget else 0

    async def execute_topology(idx: int, temp: float, sampled_topology: dict[str, Any]) -> Rollout:
        policy_config = config_from_topology(config, sampled_topology)
        # Hard runtime budget enforcement: the executor exits gracefully the
        # moment tokens_spent >= B. No per-strategy ceilings, just the same B
        # that pi_O conditioned on for this group.
        policy_config.deployment_budget = int(deployment_budget) if deployment_budget else 0
        planner_lm = make_qwen14b_nothink_lm(replica_idx=idx, max_tokens=768)
        worker_lm = make_qwen14b_nothink_lm(replica_idx=idx + 1, max_tokens=768)
        synth_lm = make_qwen14b_nothink_lm(replica_idx=idx + 2, max_tokens=768)
        sas_lm = make_qwen14b_nothink_lm(replica_idx=idx, max_tokens=384)
        pipeline = AmasPipeline(
            planner_lm=planner_lm, worker_lm=worker_lm, synth_lm=synth_lm,
            sas_lm=sas_lm, retriever=retriever, config=policy_config,
        )
        profile = str(sampled_topology.get("query_profile", "semantic_topology")).strip()
        strategy = str(sampled_topology.get("routing_strategy", "unknown")).strip()
        policy_name = f"piO_sample_{idx + 1}:{strategy}:{profile[:50]}"
        return await run_single_rollout(
            pipeline, question, qid, gold_answer, temp,
            policy_name=policy_name, sampled_topology=sampled_topology,
            dataset=dataset, reward_alpha=reward_alpha,
            deployment_budget=deployment_budget,
        )

    async def sample_one_topology(idx: int, temp: float, prior_samples: list[dict[str, Any]]) -> dict[str, Any]:
        sampler_lm = make_qwen14b_nothink_lm(replica_idx=idx, max_tokens=900, temperature=max(0.2, temp))
        return await asyncio.to_thread(
            sample_topology,
            question=question, qid=qid, library=library,
            sampler_lm=sampler_lm, dataset=dataset,
            avoid_topologies=prior_samples,
            deployment_budget=deployment_budget,
        )

    sampled_topologies: list[dict[str, Any]] = []
    seen_signatures: set[tuple] = set()
    forced_templates: list[tuple[str, int]] = [
        ("sas_first", 2),
        ("direct_mas", 2),
    ]
    query_profile = characterize_query_profile(question, dataset, qid=qid)
    for idx, temp in enumerate(temperatures):
        if forced_diversity and idx < len(forced_templates):
            strategy, budget = forced_templates[idx]
            sampled = {
                "query_profile": f"forced_diversity_{strategy}:{query_profile}",
                "routing_strategy": strategy,
                "retrieval_budget": budget,
                "repair": False,
                "rationale": "Forced structural exploration for TF-GRPO routing contrast.",
                "_sampler_tokens": 0,
                "_query_profile": query_profile,
                "_experience_entry_ids": [],
                "_deployment_budget": deployment_budget if deployment_budget is not None else 0,
                "_forced_diversity": True,
            }
        else:
            sampled = await sample_one_topology(idx, temp, sampled_topologies)
        sig = topology_signature(sampled)
        sampled["_topology_signature"] = list(sig)
        sampled["_duplicate_retry"] = False
        sampled["_duplicate_retry_changed"] = False
        if sig in seen_signatures and not sampled.get("_forced_diversity"):
            sampled["_duplicate_retry"] = True
            sampler_lm = make_qwen14b_nothink_lm(
                replica_idx=idx, max_tokens=900, temperature=min(1.15, max(0.45, temp + 0.25))
            )
            sampled_retry = await asyncio.to_thread(
                sample_topology,
                question=question, qid=qid, library=library,
                sampler_lm=sampler_lm, dataset=dataset,
                avoid_topologies=sampled_topologies,
                deployment_budget=deployment_budget,
            )
            retry_sig = topology_signature(sampled_retry)
            if retry_sig != sig:
                sampled = sampled_retry
                sig = retry_sig
                sampled["_duplicate_retry"] = True
                sampled["_duplicate_retry_changed"] = True
            else:
                sampled["_duplicate_retry"] = True
                sampled["_duplicate_retry_changed"] = False
            sampled["_topology_signature"] = list(sig)
        sampled_topologies.append(sampled)
        seen_signatures.add(sig)

    group.rollouts = list(await asyncio.gather(
        *[execute_topology(idx, temp, sampled_topologies[idx]) for idx, temp in enumerate(temperatures)]
    ))

    if needs_topology_mutation(group, dataset):
        mutation_lm = make_qwen14b_nothink_lm(replica_idx=len(group.rollouts), max_tokens=900, temperature=0.45)
        mutations = topology_mutations(group.rollouts, mutation_lm=mutation_lm, max_candidates=1)
        start = len(group.rollouts)
        mutated_rollouts = await asyncio.gather(*[
            execute_topology(start + idx, temperatures[-1] if temperatures else 0.9, topo)
            for idx, topo in enumerate(mutations)
        ])
        group.rollouts.extend(mutated_rollouts)

    # HERA-style ranking: task performance first, token efficiency second.
    def task_score(r: Rollout) -> float:
        return compute_task_reward(float(r.em), float(r.f1), float(r.contain))

    ranked = sorted(group.rollouts, key=lambda r: (-task_score(r), int(r.total_tokens)))
    if ranked:
        winner = ranked[0]
        loser = sorted(group.rollouts, key=lambda r: (task_score(r), -int(r.total_tokens)))[0]
        group.winners = [winner]
        if loser is not winner:
            group.losers = [loser]

        winner_strategy = str((winner.sampled_topology or {}).get("routing_strategy", ""))
        loser_strategy = str((loser.sampled_topology or {}).get("routing_strategy", ""))
        score_gap = abs(task_score(winner) - task_score(loser))
        token_gap = max(winner.total_tokens, loser.total_tokens) >= 1.10 * max(1, min(winner.total_tokens, loser.total_tokens))
        group.has_mixed_outcomes = bool(group.losers) and (
            winner_strategy != loser_strategy or score_gap > 0.02 or token_gap
        )
    else:
        group.has_mixed_outcomes = False
    return group
