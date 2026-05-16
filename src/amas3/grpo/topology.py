"""Topology sampling for pi_O(Gamma | q, E, N, B).

Contains the full sampling stack:
  * exploration axis selection (group-local diversity in semantic terms),
  * topology signature + pipeline-config translation,
  * SAS coercion gate (data-driven, no per-query-type table),
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
# SAS coercion gate (data-driven, learned-experience signal only)
# ---------------------------------------------------------------------------

SAS_ENDORSEMENT_KEYWORDS: tuple[str, ...] = (
    "prefer sas",
    "sas shortcut",
    "sas lane",
    "single agent",
    "single-agent",
    "no decomposition",
    "skip decomposition",
    "skip planner",
    "direct answer",
    "answer directly",
    "avoid full mas",
    "avoid decomposition",
    "full mas wastes",
    "shortcut when",
)


def coerce_sas_if_supported(
    topology: dict[str, Any],
    retrieved_entries: list[ExperienceEntry] | None,
    min_utility: float = 0.7,
    min_supporting: int = 1,
) -> dict[str, Any]:
    """Revert pi_O's sampled topology to SAS when E says it suffices.

    Triggers ONLY when ``min_supporting`` retrieved entries with utility >=
    ``min_utility`` explicitly endorse the SAS lane via the keyword set above.
    No per-query-type table, no hardcoded threshold map: the signal is the
    learned experience library itself.
    """
    if not retrieved_entries:
        return topology
    strategy = str(topology.get("routing_strategy", "")).lower()
    if strategy == "sas":
        return topology

    supporting_ids: list[str] = []
    for entry in retrieved_entries:
        if float(getattr(entry, "utility", 0.0)) < min_utility:
            continue
        text = " ".join([
            getattr(entry, "insight", "") or "",
            getattr(entry, "applies_when", "") or "",
        ]).lower()
        if any(k in text for k in SAS_ENDORSEMENT_KEYWORDS):
            supporting_ids.append(entry.id)
    if len(supporting_ids) < min_supporting:
        return topology

    coerced = dict(topology)
    coerced["_pre_coercion_strategy"] = strategy or "unknown"
    coerced["_pre_coercion_budget"] = topology.get("retrieval_budget")
    coerced["_coerced_to_sas"] = True
    coerced["_sas_coercion_support"] = supporting_ids
    coerced["routing_strategy"] = "sas"
    coerced["retrieval_budget"] = 1
    coerced["repair"] = False
    return coerced


# ---------------------------------------------------------------------------
# Topology sampling
# ---------------------------------------------------------------------------

def _budget_block(deployment_budget: int | None) -> str:
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
    dataset: str = "default",
    avoid_topologies: list[dict[str, Any]] | None = None,
    sas_coercion_min_utility: float = 0.5,
    sas_coercion_min_supporting: int = 1,
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
    budget_text = _budget_block(deployment_budget)

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
            effective_min_utility = sas_coercion_min_utility
            effective_min_supporting = sas_coercion_min_supporting
            if deployment_budget is not None and 0 < deployment_budget < 4000:
                # Tight budget tightens SAS coercion.
                effective_min_utility = max(0.5, sas_coercion_min_utility - 0.2)
                effective_min_supporting = max(1, sas_coercion_min_supporting - 0)
            obj = coerce_sas_if_supported(
                obj, entries,
                min_utility=effective_min_utility,
                min_supporting=effective_min_supporting,
            )
            return obj
    except Exception as e:
        logger.warning("Topology sampling failed: %s", e)

    fallback = {
        "query_profile": "fallback_conservative_sas_then_mas",
        "routing_strategy": "sas_then_mas",
        "retrieval_budget": 2,
        "repair": False,
        "_sampler_tokens": 0,
        "_query_profile": query_profile,
        "_experience_entry_ids": [e.id for e in entries],
        "_deployment_budget": deployment_budget if deployment_budget is not None else 0,
    }
    effective_min_utility = sas_coercion_min_utility
    effective_min_supporting = sas_coercion_min_supporting
    if deployment_budget is not None and 0 < deployment_budget < 4000:
        effective_min_utility = max(0.5, sas_coercion_min_utility - 0.2)
    return coerce_sas_if_supported(
        fallback, entries,
        min_utility=effective_min_utility,
        min_supporting=effective_min_supporting,
    )


# ---------------------------------------------------------------------------
# Topology signature + pipeline config mapping
# ---------------------------------------------------------------------------

def _bounded_int(val, default, lo, hi):
    try:
        v = int(val)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def topology_signature(topology: dict[str, Any]) -> tuple:
    return (
        str(topology.get("routing_strategy", "")),
        int(_bounded_int(topology.get("retrieval_budget"), 2, 1, 4)),
        bool(topology.get("repair", False)),
    )


def config_from_topology(config: AmasPipelineConfig, topology: dict) -> AmasPipelineConfig:
    """Translate a sampled topology into a concrete AmasPipelineConfig.

    The three supported routing_strategy values map to disjoint executor
    modes. There are no leaky base-config defaults: every relevant field is
    set explicitly per strategy, so the sticky ``use_sas_solver=True`` from a
    base config can never override a ``full_mas`` choice (which was the bug
    that caused 14k-token full_mas rollouts under tight B).
    """
    c = replace(config)
    strategy = str(topology.get("routing_strategy", "")).lower()
    budget = _bounded_int(topology.get("retrieval_budget"), 2, 1, 4)
    learned_cap = max(1, int(c.max_retrievals_per_solver))
    repair = bool(topology.get("repair", False))

    # The sampler chooses a topology-level retrieval budget, clamped to the
    # learned per-query-type cap. We never silently expand it.
    c.max_retrievals_per_solver = max(1, min(budget, learned_cap))
    c.medium_retrievals_per_solver = min(c.medium_retrievals_per_solver, c.max_retrievals_per_solver)
    c.min_retrievals_per_solver = min(c.min_retrievals_per_solver, c.medium_retrievals_per_solver)
    c.repair_enabled = repair
    c.deployment_budget = int(topology.get("_deployment_budget", 0) or 0)

    if strategy == "sas":
        # Strict SAS: planner/solver/synth never run, but the sas_solver
        # itself is allowed multiple retrieval+LLM steps so bridge / 2-hop
        # questions are actually solvable on the SAS lane. The sampled
        # retrieval_budget controls how many followup retrievals beyond the
        # initial probe the sas_solver may issue (clamped 1..3 so SAS is
        # never crippled to a single retrieval).
        c.use_sas_solver = True
        c.sas_strict_single_pass = True
        c.sas_min_confidence = float(topology.get("sas_confidence", c.sas_min_confidence))
        c.sas_max_followups = _bounded_int(topology.get("retrieval_budget"), 2, 1, 3)
    elif strategy == "sas_then_mas":
        # Probe with sas_solver; on low confidence escalate to full MAS. The
        # sas_solver may still chain a bridge hop on its own before deciding
        # to escalate.
        c.use_sas_solver = True
        c.sas_strict_single_pass = False
        c.sas_min_confidence = float(topology.get("sas_confidence", c.sas_min_confidence))
        c.sas_max_followups = _bounded_int(topology.get("retrieval_budget"), 2, 1, 3)
    elif strategy == "full_mas":
        # Pure planner -> solver -> synth. Sas_solver MUST be off.
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
