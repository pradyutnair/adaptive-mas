"""Topology sampling for pi_O(Gamma | q, E, N, B).

Contains the full sampling stack:
  * exploration axis selection (group-local diversity in semantic terms),
  * topology signature + pipeline-config translation,
  * Algorithm 6 structural mutation fallback,
  * fallback topology when the LM output is unparseable.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import dspy

from .experience_library import (
    ExperienceEntry,
    ExperienceLibrary,
    format_for_orchestrator,
)
from .parsing import parse_json_object
from .profiles import characterize_query_profile
from .prompts import AGENT_DESCRIPTIONS, TOPOLOGY_MUTATION_PROMPT, TOPOLOGY_SAMPLING_PROMPT
from .rollout import Rollout
from .metrics import compute_task_reward
from ..pipeline import AmasPipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exploration axes
# ---------------------------------------------------------------------------



def format_avoid_topologies(topologies: list[dict[str, Any]] | None) -> str:
    """Compact group-local diversity context for pi_O sampling."""
    if not topologies:
        return "(none)"
    rows = []
    for idx, topo in enumerate(topologies[-4:], start=1):
        rows.append(
            "{}. profile={}; strategy={}; budget={}; repair={}".format(
                idx,
                str(topo.get("query_profile", ""))[:70],
                topo.get("routing_strategy"),
                topo.get("retrieval_budget"),
                topo.get("repair"),
            )
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Topology sampling
# ---------------------------------------------------------------------------

def normalize_strategy(strategy: str) -> str:
    """Map legacy labels onto the two valid deployment strategies."""
    value = (strategy or "").strip().lower()
    if value in {"sas", "sas_then_mas", "sas_first"}:
        return "sas_first"
    if value in {"mas", "full_mas", "direct_mas"}:
        return "direct_mas"
    return "sas_first"

def budget_block_text(deployment_budget: int | None) -> str:
    """Render the deployment-budget context block for the orchestrator prompt."""
    if deployment_budget is None or deployment_budget <= 0:
        return "(no explicit budget; optimize for cost while preserving quality)"
    if deployment_budget < 4000:
        regime = "tight"
    elif deployment_budget < 7000:
        regime = "moderate"
    else:
        regime = "generous"
    return (
        f"B = {int(deployment_budget)} tokens ({regime} regime). "
        f"Exceeding B incurs an over-budget penalty in the reward; "
        f"staying well below B is rewarded as efficiency."
    )


def sample_topology(
    question: str,
    qid: str,
    library: ExperienceLibrary | None,
    sampler_lm: dspy.LM,
    sample_index: int = 1,
    dataset: str = "default",
    avoid_topologies: list[dict[str, Any]] | None = None,
    deployment_budget: int | None = None,
) -> dict[str, Any]:
    """Sample a topology from the budget-conditioned orchestrator policy
    pi_O(Gamma | q, E, N, B).

    No per-query-type tables and no flat threshold priors are injected. The
    policy reasons over the question, the retrieved experience entries, the
    agent pool, group-local diversity context, and (optionally) the runtime
    deployment budget B. Passing ``deployment_budget=None`` recovers HERA's
    policy pi_O(Gamma | q, E, N).

    ``dataset`` is retained as a passthrough tag for downstream logging and
    reward shaping (training-time only). It is NOT consumed by the
    orchestrator prompt or the sampled topology.
    """
    del sample_index
    experience_text = "(no prior experiences)"
    entries: list[ExperienceEntry] = []
    if library and library.size() > 0:
        # First: experiences tagged for the GRPO orchestrator (pi_O itself).
        entries = library.retrieve_for_orchestrator(question, limit=3)
        if not entries:
            entries = library.retrieve(question, role="orchestrator", limit=3)
        if entries:
            experience_text = format_for_orchestrator(entries, max_entries=3, max_insight_chars=180)

    query_profile = characterize_query_profile(question, dataset, qid=qid)
    budget_text = budget_block_text(deployment_budget)

    prompt = TOPOLOGY_SAMPLING_PROMPT.format(
        agent_descriptions=AGENT_DESCRIPTIONS,
        experience_text=experience_text or "(no prior experiences)",
        query_profile=query_profile,
        budget_block=budget_text,
        avoid_topologies_text=format_avoid_topologies(avoid_topologies),
        question=question,
    )

    try:
        with dspy.context(lm=sampler_lm):
            response = sampler_lm(prompt)
        raw = response[0] if isinstance(response, list) else str(response)
        try:
            usage = sampler_lm.history[-1].get("usage", {}) if sampler_lm.history else {}
            sampler_tokens = int(usage.get("total_tokens", 0))
        except Exception:
            sampler_tokens = 0
        obj = parse_json_object(raw)
        if obj:
            obj["_sampler_tokens"] = sampler_tokens
            obj["_query_profile"] = query_profile
            obj["_experience_entry_ids"] = [e.id for e in entries]
            obj["_deployment_budget"] = deployment_budget if deployment_budget is not None else 0
            obj["routing_strategy"] = normalize_strategy(str(obj.get("routing_strategy", "")))
            return obj
    except Exception as e:
        logger.warning("Topology sampling failed: %s", e)

    fallback = {
        "query_profile": "fallback_conservative_sas_first",
        "routing_strategy": "sas_first",
        "retrieval_budget": 2,
        "repair": False,
        "_sampler_tokens": 0,
        "_query_profile": query_profile,
        "_experience_entry_ids": [e.id for e in entries],
        "_deployment_budget": deployment_budget if deployment_budget is not None else 0,
    }
    return fallback


# ---------------------------------------------------------------------------
# Topology signature + pipeline config mapping
# ---------------------------------------------------------------------------

def bounded_int(val, default, lo, hi):
    try:
        v = int(val)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def topology_signature(topology: dict[str, Any]) -> tuple:
    return (
        str(topology.get("routing_strategy", "")),
        int(bounded_int(topology.get("retrieval_budget"), 2, 1, 4)),
        bool(topology.get("repair", False)),
    )


def config_from_topology(config: AmasPipelineConfig, topology: dict) -> AmasPipelineConfig:
    """Translate a sampled topology into a concrete AmasPipelineConfig.

    The two supported routing_strategy values map to disjoint executor modes:
    ``sas_first`` probes with SAS and honors verifier-driven escalation, while
    ``direct_mas`` skips SAS entirely.
    """
    c = replace(config)
    strategy = normalize_strategy(str(topology.get("routing_strategy", "")))
    budget = bounded_int(topology.get("retrieval_budget"), 2, 1, 4)
    learned_cap = max(1, int(c.max_retrievals_per_solver))
    repair = bool(topology.get("repair", False))

    # The sampler chooses a topology-level retrieval budget, clamped to the
    # learned per-query-type cap. We never silently expand it.
    c.max_retrievals_per_solver = max(1, min(budget, learned_cap))
    c.medium_retrievals_per_solver = min(c.medium_retrievals_per_solver, c.max_retrievals_per_solver)
    c.min_retrievals_per_solver = min(c.min_retrievals_per_solver, c.medium_retrievals_per_solver)
    c.repair_enabled = repair
    c.deployment_budget = int(topology.get("_deployment_budget", 0) or 0)

    if strategy == "sas_first":
        # Probe with sas_solver; accept only confident answers. Otherwise
        # the executor naturally escalates to full MAS. Keep SAS cheap:
        # max 1 followup, small excerpts, no verifier (matches baseline).
        c.use_sas_solver = True
        c.sas_strict_single_pass = False
        c.sas_min_confidence = float(topology.get("sas_confidence", c.sas_min_confidence))
        c.sas_max_followups = 1
    elif strategy == "direct_mas":
        # Pure planner -> solver -> synth. Sas_solver is skipped.
        c.use_sas_solver = False
        c.sas_strict_single_pass = False
    else:
        # Unknown strategy: refuse to silently inherit base defaults.
        c.use_sas_solver = False
        c.sas_strict_single_pass = False

    return c


# ---------------------------------------------------------------------------
# Algorithm 6: structural mutation when rollouts consistently fail
# ---------------------------------------------------------------------------

def format_rollout_for_reflection(r: Rollout) -> str:
    """Format a rollout for reflection / mutation prompts.

    Exposes the per-agent token breakdown + budget-exit signal so the
    reflection LM can attribute cost to specific stages and write actionable
    insights ('synth wasted N tokens for entity questions', 'planner overran
    B on comparison queries', etc.) rather than vague total-token guidance.
    """
    topo = r.sampled_topology or {}
    res = r.result if isinstance(r.result, dict) else {}
    planner_t = int(res.get("planner_tokens", 0) or 0)
    solver_t = int(res.get("solver_tokens", 0) or 0)
    synth_t = int(res.get("synth_tokens", 0) or 0)
    rewrite_t = int(res.get("rewrite_tokens", 0) or 0)
    sas_t = int(res.get("sas_attempt_tokens", 0) or 0)
    sas_v_t = int(res.get("sas_verifier_tokens", 0) or 0)
    B = int(topo.get("_deployment_budget", 0) or 0)
    lines = [
        f"Policy: {r.policy_name}",
        f"Profile: {topo.get('query_profile', 'unknown')}",
        f"Strategy: {topo.get('routing_strategy', 'unknown')}",
        f"Retrieval budget: {topo.get('retrieval_budget', '?')}",
        f"Deployment budget B: {B if B else 'unset'}",
        f"EM: {r.em}, F1: {r.f1:.3f}, Contain: {r.contain:.3f}",
        f"Total tokens: {r.total_tokens} (efficiency: {r.token_efficiency:.3f})",
        (
            f"Per-agent tokens: sas={sas_t}, sas_verifier={sas_v_t}, "
            f"planner={planner_t}, solver={solver_t} (rewrite={rewrite_t}), synth={synth_t}"
        ),
        f"Dual reward: {r.dual_reward:.3f}",
        f"Topology: {r.topology}",
        f"Plan subgoals: {r.plan_subgoals}",
        f"Answer: {(r.predicted_answer or '')[:80]}",
    ]
    return "\n".join(lines)


def topology_mutations(
    rollouts: list[Rollout],
    mutation_lm: dspy.LM | None = None,
    max_candidates: int = 1,
) -> list[dict[str, Any]]:
    """Algorithm 6: semantically justified structural mutation, not a fixed menu."""
    if not rollouts or mutation_lm is None:
        return []
    ranked = sorted(
        rollouts,
        key=lambda r: (compute_task_reward(float(r.em), float(r.f1), float(r.contain)), -int(r.total_tokens)),
    )
    failed_text = "\n---\n".join(format_rollout_for_reflection(r) for r in ranked[:4])
    prompt = TOPOLOGY_MUTATION_PROMPT.format(
        agent_descriptions=AGENT_DESCRIPTIONS,
        question=rollouts[0].question,
        failed_trajectories=failed_text,
    )
    try:
        with dspy.context(lm=mutation_lm):
            response = mutation_lm(prompt)
        raw = response[0] if isinstance(response, list) else str(response)
        obj = parse_json_object(raw)
        if not obj:
            return []
        obj["topology_mutation"] = "semantic_orchestrator_mutation"
        try:
            usage = mutation_lm.history[-1].get("usage", {}) if mutation_lm.history else {}
            obj["_sampler_tokens"] = int(usage.get("total_tokens", 0))
        except Exception:
            obj["_sampler_tokens"] = 0
        return [obj][:max_candidates]
    except Exception as exc:
        logger.warning("Topology mutation sampling failed: %s", exc)
        return []
