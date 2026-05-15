"""Canonical token-cost-aware Training-Free GRPO with HERA semantic orchestration.

Design notes:
1. Dual reward: r = alpha*r_task + (1-alpha)*r_efficiency with explicit token penalty
2. Token-cost-aware group ranking: rank by (task_perf DESC, token_cost ASC)
3. Tighter experience consolidation with aggressive pruning (max 40 entries)
4. Insight extraction explicitly includes token efficiency analysis
5. Role-specific experience retrieval instead of broadcasting all entries
6. Token budget tracking in all reflections
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import string
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import dspy

from .experience_library import (
    ExperienceEntry,
    ExperienceLibrary,
    format_for_orchestrator,
    format_for_prompt,
)
from .lm import make_qwen14b_nothink_lm, make_qwen14b_think_lm, make_mini_lm
from .pipeline import AmasPipeline, AmasPipelineConfig, AmasResult
from .retriever import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer normalization and scoring
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _ARTICLES.sub("", s)
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = " ".join(s.split())
    return s.strip()


def compute_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum(1 for t in pred_tokens if t in gold_tokens)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_contain(pred: str, gold: str) -> float:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    return 1.0 if gold_norm in pred_norm else 0.0


def compute_task_reward(em: float, f1: float, contain: float) -> float:
    """Task reward aligned to eval: contain is primary, F1/EM preserved."""
    return 0.45 * contain + 0.35 * f1 + 0.20 * em


# ---------------------------------------------------------------------------
# Token efficiency scoring
# ---------------------------------------------------------------------------

# Baseline token budgets per dataset (from HANDOFF best runs)
TOKEN_BUDGET_BASELINES = {
    "hotpotqa": 7050,
    "2wiki": 7240,
    "musique": 7480,
    "bamboogle": 5100,
    "default": 7000,
}


def compute_token_efficiency_reward(total_tokens: int, dataset: str = "default") -> float:
    """Compute efficiency reward in [0, 1]. Higher = fewer tokens relative to baseline.
    
    Uses a sigmoid-like function centered on the baseline budget.
    Tokens < baseline get reward > 0.5, tokens > baseline get reward < 0.5.
    """
    baseline = TOKEN_BUDGET_BASELINES.get(dataset, TOKEN_BUDGET_BASELINES["default"])
    # Normalize: ratio of tokens used vs baseline
    ratio = total_tokens / max(baseline, 1)
    # Sigmoid: reward = 1 / (1 + exp(2*(ratio - 1)))
    # At ratio=1.0: reward=0.5, at ratio=0.5: reward~0.88, at ratio=1.5: reward~0.12
    import math
    reward = 1.0 / (1.0 + math.exp(2.0 * (ratio - 1.0)))
    return round(max(0.0, min(1.0, reward)), 4)


def compute_dual_reward(
    em: float, f1: float, total_tokens: int,
    dataset: str = "default", alpha: float = 0.7,
    answered: bool = True,
    contain: float = 0.0,
) -> float:
    """Compute combined task + efficiency reward.
    
    r = alpha * r_task + (1-alpha) * r_eff
    alpha=0.7 means task quality is primary, efficiency is secondary.
    Blank answers get a hard penalty.
    """
    r_task = compute_task_reward(em, f1, contain)
    r_eff = compute_token_efficiency_reward(total_tokens, dataset)
    r = alpha * r_task + (1 - alpha) * r_eff
    if not answered:
        r -= 0.15  # Hard penalty for blank answers
    return round(r, 4)


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """Single pipeline execution result for one question at one temperature."""
    question_id: str
    question: str
    gold_answer: str
    temperature: float
    policy_name: str = ""
    predicted_answer: str = ""
    em: float = 0.0
    f1: float = 0.0
    contain: float = 0.0
    total_tokens: int = 0
    topology: str = ""
    sampled_topology: dict[str, Any] = field(default_factory=dict)
    plan_subgoals: int = 0
    findings: list[dict] = field(default_factory=list)
    wallclock_seconds: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    dual_reward: float = 0.0
    token_efficiency: float = 0.0
    dataset: str = "default"


@dataclass
class GroupResult:
    """K rollouts for a single question."""
    question_id: str
    question: str
    gold_answer: str
    rollouts: list[Rollout] = field(default_factory=list)
    has_mixed_outcomes: bool = False
    winners: list[Rollout] = field(default_factory=list)
    losers: list[Rollout] = field(default_factory=list)


def characterize_query_profile(question: str, dataset: str = "default") -> str:
    """Lightweight query profile used for semantic routing and library updates.

    This is deliberately based only on the question text and dataset name. It is
    not a label leak and does not look at gold/baseline outputs.
    """
    q = (question or "").lower()
    wh_match = re.match(r"\s*(who|what|when|where|which|whose|how many|how much|how old)\b", q)
    wh = wh_match.group(1) if wh_match else "unknown"
    cues: list[str] = []
    if any(x in q for x in ("compare", "which of", "which one", "larger", "smaller", "older", "younger", "more", "less")):
        cues.append("comparison")
    if any(x in q for x in ("both", "and ", " or ", "same ", "different")):
        cues.append("set_or_boolean")
    if any(x in q for x in ("the ", " of ", " by ", " in ")) and len(q.split()) >= 12:
        cues.append("bridge")
    if any(x in q for x in ("film", "album", "book", "song", "series", "team", "city", "country")):
        cues.append("entity_linking")
    if wh in ("when",):
        cues.append("date")
    if wh in ("how many", "how much", "how old"):
        cues.append("numeric")
    if not cues:
        cues.append("factoid")
    return f"{dataset}:{'+'.join(dict.fromkeys(cues))}; wh={wh}; len={len(q.split())}"


# ---------------------------------------------------------------------------
# Token-cost-aware reflection prompts
# ---------------------------------------------------------------------------

TRAJECTORY_SUMMARY_PROMPT = """\
An agent system was given experiences and produced this multi-hop QA trajectory.

Question: {question}
Gold answer: {gold_answer}
Evaluation: EM={em:.1f}, F1={f1:.2f}, Contain={contain:.1f}, tokens={tokens}

Trajectory:
{trajectory}

Summarize step by step:
1. Topology and key routing decisions.
2. Which experiences appear to have influenced decisions, if any.
3. Detours, wasteful retrieval, missing evidence, wrong bridge/entity choices, or efficient shortcuts.
4. Final outcome and why it succeeded or failed.

Return only the numbered summary."""


SEMANTIC_ADVANTAGE_PROMPT = """\
You are analyzing multi-hop QA pipeline trajectories to extract reusable insights.
Focus on BOTH quality AND token efficiency.

Query: {question}
Query type: {query_type}

Current experience library:
{library_text}

The following trajectory summaries were executed for the SAME query, ranked by \
task performance (F1 desc) then efficiency (tokens asc):
{trajectory_summaries}

Compare high-advantage and low-advantage trajectories. If all trajectories
failed, compare the least-wasteful failure against the most-wasteful failure
and extract what to avoid or what missing routing/agent behavior is needed.
Focus on:
1. Planning: What decomposition strategies saved tokens while maintaining quality?
2. Retrieval: Which retrieval patterns were wasteful vs. efficient?
3. Routing: Did SAS shortcuts work well? Was orchestrator escalation justified?
4. Synthesis: Did concise findings improve or hurt final answer quality?
5. Token hotspots: Which agent consumed the most tokens unnecessarily?

Extract 1-3 actionable insights. EACH insight MUST address token efficiency and preserve/improve Contain:
- How to achieve same/better quality with fewer tokens
- Which steps to skip, combine, or shorten
- When to use cheap path (SAS) vs expensive path (full MAS)

Output STRICT JSON:
{{"success_factors":["<factor>"],"failure_modes":["<mode>"],"insights":[{{"query_type":"<type>","insight":"<actionable insight <=32 words>","target_roles":["planner|solver|synth|orchestrator"],"token_impact":"saves|costs|neutral","estimated_token_savings":"<rough estimate>"}}]}}

Return ONLY the JSON object. Keep each insight <=32 words."""

EXPERIENCE_UPDATE_PROMPT = """\
You are managing a compact experience library for a multi-hop QA system.
The library MUST stay under {max_entries} entries. Currently: {n_entries} entries.

## Current Experience Library:
{library_text}

## New Insights from Training:
{new_insights_text}

Rules:
- PRUNE low-utility entries (utility < 0.3) aggressively
- MERGE similar insights instead of adding duplicates
- PRUNE entries that increased tokens without improving quality
- Each insight text MUST be <=32 words
- Prefer insights about token efficiency and routing shortcuts

For each new insight, decide one action:
- ADD: novel insight not covered (only if library < {max_entries} entries)
- MERGE <target_id>: combine with existing complementary entry
- PRUNE <target_id>: remove stale/low-utility or conflicting entry
- KEEP: already well-covered

Output JSON array:
[{{"operation":"ADD|MERGE|PRUNE|KEEP","new_insight":"<text>","target_entry_ids":["<id>"],"merged_insight":"<text or null>","rationale":"<brief reason>","insight":{{"profile":"<query type>","insight":"<<=32 words>","target_roles":["<roles>"],"applies_when":"<condition>","avoid_when":"<condition>"}}}}]

Return ONLY the JSON array."""


AGENT_DESCRIPTIONS = """\
- orchestrator: routes to SAS shortcut or full MAS decomposition based on evidence quality.
- planner: decomposes multi-hop questions into ordered subgoals.
- solver: retrieves evidence and extracts grounded answer spans.
- synthesizer: aligns findings to the original wh-target and produces final answer.
- repair: retries when evidence is insufficient.
- bridge_resolver: resolves ambiguous bridge phrases."""


# ---------------------------------------------------------------------------
# Topology sampling (reused from original with token-cost context)
# ---------------------------------------------------------------------------

TOPOLOGY_SAMPLING_PROMPT = """\
You are an orchestrator for multi-hop QA. Design a minimal, efficient topology.

Agents: {agent_descriptions}

Past experiences (ranked by utility):
{experience_text}

Semantic query profile:
{query_profile}

Already sampled topologies for this rollout group:
{avoid_topologies_text}

Current rollout exploration axis:
{exploration_axis}

Query: {question}

Design a MINIMAL topology by reasoning about q, the retrieved experiences, and
the available agent pool (no fixed thresholds or per-type tables):
1. Select ONLY the agents needed. Fewer agents = fewer tokens.
2. Decide routing_strategy from query semantics and the retrieved insights,
   not from a hardcoded rubric.
3. Set retrieval_budget from the smallest budget supported by the evidence.
4. Use the exploration axis to sample a semantically justified alternative,
   not a fixed template.
5. Do not duplicate an already sampled topology unless query semantics leave
   no safe alternative.
6. Prefer fewer agents and fewer retrievals; every agent call and retrieval
   adds tokens, so justify each one.

Return STRICT JSON:
{{"query_profile":"<one sentence>","selected_agents":["<agent>"],"execution_order":[{{"step":1,"agent":"<agent>","depends_on":[],"mode":"sequential|parallel"}}],"routing_strategy":"sas|full_mas|orchestrator_then_mas","retrieval_budget":<int 1-3>,"repair":false,"rationale":"<why this topology is efficient>"}}"""


TOPOLOGY_MUTATION_PROMPT = """\
You are the orchestrator repairing failed multi-hop QA trajectories.

Agents: {agent_descriptions}

Question: {question}

Failed trajectories:
{failed_trajectories}

Propose ONE semantically justified structural mutation. Do not choose from a
fixed menu. You may replace, remove, reorder, or augment agents only if the
failure analysis supports it. Keep it minimal and token-aware.

Return STRICT JSON:
{{"query_profile":"<failure-aware profile>","selected_agents":["<agent>"],"execution_order":[{{"step":1,"agent":"<agent>","depends_on":[],"mode":"sequential|parallel"}}],"routing_strategy":"sas|full_mas|orchestrator_then_mas","retrieval_budget":<int 1-3>,"repair":false,"bridge_resolver":false,"rationale":"<why this mutation addresses the observed failure>"}}"""


def format_avoid_topologies(topologies: list[dict[str, Any]] | None) -> str:
    """Compact group-local diversity context for pi_O sampling."""
    if not topologies:
        return "(none)"
    rows = []
    for idx, topo in enumerate(topologies[-4:], start=1):
        agents = ",".join(str(a) for a in topo.get("selected_agents", [])[:6])
        rows.append(
            "{}. profile={}; strategy={}; budget={}; repair={}; agents={}".format(
                idx,
                str(topo.get("query_profile", ""))[:70],
                topo.get("routing_strategy"),
                topo.get("retrieval_budget"),
                topo.get("repair"),
                agents,
            )
        )
    return "\n".join(rows)


EXPLORATION_AXES = {
    "exploit": "exploit: choose the highest-utility learned route with the smallest safe budget",
    "frugal": "frugal: prefer direct orchestration or one-hop solving only when evidence supports the original wh-target",
    "robust": "robust: use decomposition for ambiguous bridges/comparisons but keep retrieval budget minimal",
    "risk_sensitive": "risk_sensitive: avoid direct answers unless the evidence states the exact final relation",
    "repair_aware": "repair_aware: add repair/bridge resolution only when prior experience indicates missing evidence or bridge drift",
}


def exploration_axis_for_sample(
    sample_index: int,
    query_profile: str,
    retrieved_entries: list[ExperienceEntry] | None = None,
) -> str:
    """Choose a semantic exploration axis conditioned on query/library state.

    The sample index selects among a query-ranked axis list. It does not map to a
    fixed topology or fixed config, so group diversity remains pi_O sampling.
    """
    profile = (query_profile or "").lower()
    text = " ".join(
        [profile]
        + [getattr(e, "insight", "") + " " + getattr(e, "applies_when", "") + " " + getattr(e, "avoid_when", "")
           for e in (retrieved_entries or [])]
    ).lower()
    scores = {name: 0.0 for name in EXPLORATION_AXES}
    scores["exploit"] += 0.4
    if "factoid" in profile or "date" in profile or "numeric" in profile:
        scores["frugal"] += 1.1
        scores["exploit"] += 0.5
    if "bridge" in profile or "comparison" in profile or "set_or_boolean" in profile:
        scores["robust"] += 1.0
        scores["risk_sensitive"] += 0.7
    if any(x in text for x in ("missing", "insufficient", "repair", "retry", "failed", "bridge drift")):
        scores["repair_aware"] += 1.0
    if any(x in text for x in ("skip", "shortcut", "sas", "direct", "fewer", "budget")):
        scores["frugal"] += 0.6
        scores["exploit"] += 0.2
    ordered = sorted(EXPLORATION_AXES, key=lambda name: (-scores[name], name))
    chosen = ordered[max(0, sample_index - 1) % len(ordered)]
    return f"{EXPLORATION_AXES[chosen]} | query_conditioned_rank={ordered}"


def sample_topology(
    question: str,
    qid: str,
    library: ExperienceLibrary | None,
    sampler_lm: dspy.LM,
    sample_index: int = 1,
    dataset: str = "default",
    avoid_topologies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sample a topology from the orchestrator policy conditioned on (q, E, N).

    No per-query-type tables, no per-dataset token-budget tables, and no flat
    threshold priors are injected into the orchestrator prompt. The policy
    pi_O reasons over the question, the retrieved experience entries, the
    agent pool, and group-local diversity context only.

    `dataset` is retained as a passthrough tag for downstream logging and
    reward shaping (training-time only). It is NOT consumed by the
    orchestrator prompt or the sampled topology.
    """
    experience_text = "(no prior experiences)"
    entries: list[ExperienceEntry] = []
    if library and library.size() > 0:
        # HERA Algorithm 4: utility-prioritized retrieval with similarity gate
        # and diversity filter; falls back to the dense retriever's MMR scorer
        # when the orchestrator-specific retriever returns nothing.
        entries = library.retrieve_for_orchestrator(question, limit=3)
        if not entries:
            entries = library.retrieve(question, role="orchestrator", limit=3)
        if entries:
            experience_text = format_for_orchestrator(entries, max_entries=3, max_insight_chars=180)

    query_profile = characterize_query_profile(question, dataset)
    exploration_axis = exploration_axis_for_sample(sample_index, query_profile, entries)

    prompt = TOPOLOGY_SAMPLING_PROMPT.format(
        agent_descriptions=AGENT_DESCRIPTIONS,
        experience_text=experience_text or "(no prior experiences)",
        query_profile=query_profile,
        avoid_topologies_text=format_avoid_topologies(avoid_topologies),
        exploration_axis=exploration_axis,
        question=question,
    )

    try:
        with dspy.context(lm=sampler_lm):
            response = sampler_lm(prompt)
        raw = response[0] if isinstance(response, list) else str(response)
        # Track sampler tokens
        try:
            usage = sampler_lm.history[-1].get("usage", {}) if sampler_lm.history else {}
            sampler_tokens = int(usage.get("total_tokens", 0))
        except Exception:
            sampler_tokens = 0
        obj = parse_json_object(raw)
        if obj:
            obj["_sampler_tokens"] = sampler_tokens
            obj["_query_profile"] = query_profile
            obj["_exploration_axis"] = exploration_axis
            obj["_experience_entry_ids"] = [e.id for e in entries]
            return obj
    except Exception as e:
        logger.warning("Topology sampling failed: %s", e)

    return {
        "query_profile": "fallback_conservative_mas",
        "selected_agents": ["orchestrator", "planner", "solver", "synthesizer"],
        "execution_order": [
            {"step": 1, "agent": "orchestrator", "depends_on": [], "mode": "sequential"},
            {"step": 2, "agent": "planner", "depends_on": ["orchestrator"], "mode": "sequential"},
            {"step": 3, "agent": "solver", "depends_on": ["planner"], "mode": "parallel"},
            {"step": 4, "agent": "synthesizer", "depends_on": ["solver"], "mode": "sequential"},
        ],
        "routing_strategy": "orchestrator_then_mas",
        "retrieval_budget": 2,
        "repair": False,
        "_sampler_tokens": 0,
        "_query_profile": query_profile,
        "_exploration_axis": exploration_axis,
        "_experience_entry_ids": [e.id for e in entries],
    }


def _bounded_int(val, default, lo, hi):
    try:
        v = int(val)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def topology_signature(topology: dict[str, Any]) -> tuple:
    agents = tuple(str(a) for a in topology.get("selected_agents", []))
    return (
        str(topology.get("routing_strategy", "")),
        agents,
        int(_bounded_int(topology.get("retrieval_budget"), 2, 1, 4)),
        bool(topology.get("repair", False)),
        bool(topology.get("bridge_resolver", False)),
    )


def config_from_topology(config: AmasPipelineConfig, topology: dict) -> AmasPipelineConfig:
    """Apply sampled topology to pipeline config."""
    c = replace(config)
    strategy = str(topology.get("routing_strategy", "")).lower()
    budget = _bounded_int(topology.get("retrieval_budget"), 2, 1, 4)
    learned_cap = max(1, int(c.max_retrievals_per_solver))
    repair = bool(topology.get("repair", False))

    # The sampler may choose a smaller topology-specific budget, but should not
    # silently expand the compiled TF-GRPO budget learned for this query type.
    c.max_retrievals_per_solver = max(1, min(budget, learned_cap))
    c.medium_retrievals_per_solver = min(c.medium_retrievals_per_solver, c.max_retrievals_per_solver)
    c.min_retrievals_per_solver = min(c.min_retrievals_per_solver, c.medium_retrievals_per_solver)
    c.repair_enabled = repair

    if strategy == "sas":
        c.use_orchestrator = True
        c.orch_min_confidence = float(topology.get("orchestrator_confidence", c.orch_min_confidence))
        c.orch_max_followups = 1
        # A sampled SAS topology is a cheap first attempt, not permission to
        # cripple the fallback. If the orchestrator escalates, run a real MAS
        # pass within the learned per-query-type cap.
        fallback_floor = min(2, learned_cap)
        c.max_retrievals_per_solver = max(c.max_retrievals_per_solver, fallback_floor)
        c.medium_retrievals_per_solver = min(
            max(c.medium_retrievals_per_solver, fallback_floor),
            c.max_retrievals_per_solver,
        )
    elif strategy == "full_mas":
        c.use_orchestrator = False
    elif strategy == "orchestrator_then_mas":
        c.use_orchestrator = True
        c.orch_min_confidence = float(topology.get("orchestrator_confidence", c.orch_min_confidence))
        c.orch_max_followups = _bounded_int(topology.get("max_followups"), 1, 0, 2)

    if topology.get("bridge_resolver"):
        c.use_bridge_resolver = True

    return c


# ---------------------------------------------------------------------------
# Group rollouts with token-cost-aware scoring
# ---------------------------------------------------------------------------

def format_rollout_for_reflection(r: Rollout) -> str:
    """Format a rollout for reflection, including token cost info."""
    topo = r.sampled_topology or {}
    lines = [
        f"Policy: {r.policy_name}",
        f"Profile: {topo.get('query_profile', 'unknown')}",
        f"Agents: {topo.get('selected_agents', [])}",
        f"Strategy: {topo.get('routing_strategy', 'unknown')}",
        f"Retrieval budget: {topo.get('retrieval_budget', '?')}",
        f"EM: {r.em}, F1: {r.f1:.3f}, Contain: {r.contain:.3f}",
        f"Total tokens: {r.total_tokens} (efficiency: {r.token_efficiency:.3f})",
        f"Dual reward: {r.dual_reward:.3f}",
        f"Topology: {r.topology}",
        f"Plan subgoals: {r.plan_subgoals}",
        f"Answer: {(r.predicted_answer or '')[:80]}",
    ]
    return "\n".join(lines)


def summarize_rollout(r: Rollout, reflection_lm: dspy.LM) -> str:
    """TF-GRPO per-rollout summarization before group advantage extraction."""
    prompt = TRAJECTORY_SUMMARY_PROMPT.format(
        question=r.question,
        gold_answer=r.gold_answer,
        em=r.em,
        f1=r.f1,
        contain=r.contain,
        tokens=r.total_tokens,
        trajectory=format_rollout_for_reflection(r),
    )
    with dspy.context(lm=reflection_lm):
        response = reflection_lm(prompt)
    text = response[0] if isinstance(response, list) else str(response)
    return text.strip()


def format_library_for_reflection(library: ExperienceLibrary | None, question: str = "", limit: int = 12) -> str:
    if library is None or not library.entries:
        return "(empty)"
    entries = library.retrieve(question, limit=limit) if question else list(library.entries.values())[:limit]
    return "\n".join(
        f"[{entry.id}] profile={entry.profile}, utility={entry.utility:.2f}, roles={list(entry.target_roles)}: {entry.insight}"
        for entry in entries
    ) or "(empty)"


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
) -> Rollout:
    """Execute one pipeline run and score it."""
    result = await pipeline.run(question, qid)
    pred = result.answer or ""
    em = compute_em(pred, gold_answer)
    f1 = compute_f1(pred, gold_answer)
    contain = compute_contain(pred, gold_answer)
    
    # Include sampler tokens in total
    sampler_tokens = int((sampled_topology or {}).get("_sampler_tokens", 0))
    scored_total_tokens = result.total_tokens + sampler_tokens
    
    # Compute dual reward
    answered = bool(pred.strip())
    token_eff = compute_token_efficiency_reward(scored_total_tokens, dataset)
    dual_r = compute_dual_reward(
        em, f1, scored_total_tokens, dataset, alpha=reward_alpha,
        answered=answered, contain=contain,
    )
    
    result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else {}

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
) -> GroupResult:
    """Run K same-query rollouts, score with dual reward, rank by task then efficiency."""
    group = GroupResult(question_id=qid, question=question, gold_answer=gold_answer)

    async def _execute_topology(idx: int, temp: float, sampled_topology: dict[str, Any]) -> Rollout:
        policy_config = config_from_topology(config, sampled_topology)
        planner_lm = make_qwen14b_nothink_lm(replica_idx=idx, max_tokens=768)
        worker_lm = make_qwen14b_nothink_lm(replica_idx=idx + 1, max_tokens=768)
        synth_lm = make_qwen14b_nothink_lm(replica_idx=idx + 2, max_tokens=768)
        sas_lm = make_qwen14b_nothink_lm(replica_idx=idx, max_tokens=384)
        pipeline = AmasPipeline(
            planner_lm=planner_lm, worker_lm=worker_lm, synth_lm=synth_lm,
            sas_lm=sas_lm, retriever=retriever, config=policy_config,
        )
        profile = str(sampled_topology.get("query_profile", "semantic_topology")).strip()
        policy_name = f"piO_sample_{idx + 1}:{profile[:60]}"
        rollout = await run_single_rollout(
            pipeline, question, qid, gold_answer, temp,
            policy_name=policy_name, sampled_topology=sampled_topology,
            dataset=dataset, reward_alpha=reward_alpha,
        )
        return rollout

    async def _sample_one_topology(idx: int, temp: float, prior_samples: list[dict[str, Any]]) -> dict[str, Any]:
        sampler_lm = make_qwen14b_nothink_lm(replica_idx=idx, max_tokens=900, temperature=max(0.2, temp))
        return await asyncio.to_thread(
            sample_topology,
            question=question, qid=qid, library=library,
            sampler_lm=sampler_lm, sample_index=idx + 1, dataset=dataset,
            avoid_topologies=prior_samples,
        )

    sampled_topologies: list[dict[str, Any]] = []
    seen_signatures: set[tuple] = set()
    for idx, temp in enumerate(temperatures):
        sampled = await _sample_one_topology(idx, temp, sampled_topologies)
        sig = topology_signature(sampled)
        sampled["_topology_signature"] = list(sig)
        sampled["_duplicate_retry"] = False
        sampled["_duplicate_retry_changed"] = False
        if sig in seen_signatures:
            sampled["_duplicate_retry"] = True
            sampler_lm = make_qwen14b_nothink_lm(
                replica_idx=idx, max_tokens=900, temperature=min(1.15, max(0.45, temp + 0.25))
            )
            sampled_retry = await asyncio.to_thread(
                sample_topology,
                question=question, qid=qid, library=library,
                sampler_lm=sampler_lm, sample_index=idx + 11, dataset=dataset,
                avoid_topologies=sampled_topologies,
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
        *[_execute_topology(idx, temp, sampled_topologies[idx]) for idx, temp in enumerate(temperatures)]
    ))

    if _needs_topology_mutation(group, dataset):
        mutation_lm = make_qwen14b_nothink_lm(replica_idx=len(group.rollouts), max_tokens=900, temperature=0.45)
        mutations = topology_mutations(group.rollouts, mutation_lm=mutation_lm, max_candidates=1)
        start = len(group.rollouts)
        mutated_rollouts = await asyncio.gather(*[
            _execute_topology(start + idx, temperatures[-1] if temperatures else 0.9, topo)
            for idx, topo in enumerate(mutations)
        ])
        group.rollouts.extend(mutated_rollouts)

    # HERA-style ranking: task performance first, token efficiency second.
    def task_score(r: Rollout) -> float:
        return compute_task_reward(float(r.em), float(r.f1), float(r.contain))

    ranked = sorted(group.rollouts, key=lambda r: (-task_score(r), int(r.total_tokens)))
    if ranked:
        best_task = task_score(ranked[0])
        cheapest_tokens = min(max(1, int(r.total_tokens)) for r in group.rollouts)
        for r in group.rollouts:
            score = task_score(r)
            if score >= best_task - 0.05 and r.total_tokens <= 1.25 * max(cheapest_tokens, ranked[0].total_tokens):
                group.winners.append(r)
            elif score <= best_task - 0.12 or r.total_tokens > 1.4 * cheapest_tokens:
                group.losers.append(r)

    group.has_mixed_outcomes = len(group.winners) > 0 and len(group.losers) > 0
    return group


def _needs_topology_mutation(group: GroupResult, dataset: str = "default") -> bool:
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
    # Structural fallback is useful only for cheap failures. If the group already
    # spent near-budget tokens and still failed, extra mutation rollouts are waste.
    return avg_tokens <= 0.9 * baseline and min_tokens <= 0.75 * baseline


def topology_mutations(
    rollouts: list[Rollout],
    mutation_lm: dspy.LM | None = None,
    max_candidates: int = 1,
) -> list[dict[str, Any]]:
    """Algorithm 6 structural fallback via semantic mutation, not fixed templates."""
    if not rollouts:
        return []
    if mutation_lm is None:
        return []
    ranked = sorted(rollouts, key=lambda r: (compute_task_reward(float(r.em), float(r.f1), float(r.contain)), -int(r.total_tokens)))
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


# ---------------------------------------------------------------------------
# Token-cost-aware semantic advantage extraction
# ---------------------------------------------------------------------------

def extract_semantic_advantages(
    group: GroupResult,
    reflection_lm: dspy.LM,
    query_type: str | None = None,
    library: ExperienceLibrary | None = None,
) -> list[dict]:
    """Extract insights from mixed-outcome groups, emphasizing token efficiency."""
    if not group.has_mixed_outcomes:
        return []

    # Rank by task quality first, then efficiency, matching HERA Algorithm 2.
    ranked = sorted(group.rollouts, key=lambda r: (-compute_task_reward(float(r.em), float(r.f1), float(r.contain)), int(r.total_tokens)))
    summary_blocks = []
    for idx, rollout in enumerate(ranked, start=1):
        summary = summarize_rollout(rollout, reflection_lm)
        summary_blocks.append(
            f"Rollout {idx}: EM={rollout.em:.1f}, F1={rollout.f1:.2f}, Contain={rollout.contain:.1f}, tokens={rollout.total_tokens}\n{summary}"
        )
    trajectory_summaries = "\n---\n".join(summary_blocks)
    
    prompt = SEMANTIC_ADVANTAGE_PROMPT.format(
        question=group.question,
        query_type=query_type or characterize_query_profile(
            group.question,
            group.rollouts[0].dataset if group.rollouts else "default",
        ),
        library_text=format_library_for_reflection(library, group.question),
        trajectory_summaries=trajectory_summaries,
    )

    with dspy.context(lm=reflection_lm):
        response = reflection_lm(prompt)

    raw_text = response[0] if isinstance(response, list) else str(response)
    obj = parse_json_object(raw_text)
    if obj and isinstance(obj.get("insights"), list):
        insights = obj.get("insights", [])
        # Enforce the TF-GRPO short-experience constraint.
        for ins in insights:
            if isinstance(ins, dict) and "insight" in ins:
                ins["insight"] = " ".join(str(ins["insight"]).split()[:32])
        return insights
    return parse_json_array(raw_text)


def update_experience_credit_from_group(
    library: ExperienceLibrary,
    group: GroupResult,
) -> None:
    """Credit only experience entries actually injected into sampled topologies."""
    if not library.entries:
        return
    best_task = max(
        (compute_task_reward(float(r.em), float(r.f1), float(r.contain)) for r in group.rollouts),
        default=0.0,
    )
    touched: set[str] = set()
    for rollout in group.rollouts:
        ids = (rollout.sampled_topology or {}).get("_experience_entry_ids") or []
        if not ids:
            continue
        score = compute_task_reward(float(rollout.em), float(rollout.f1), float(rollout.contain))
        success = (
            score >= max(0.45, best_task - 0.05)
            and bool((rollout.predicted_answer or "").strip())
            and rollout.token_efficiency >= 0.25
        )
        for eid in ids:
            if eid in library.entries:
                library.update_utility(eid, success=success)
                touched.add(eid)

    if touched:
        return

    # Cold-start fallback: if no entries were injected into pi_O, credit only
    # the most semantically similar entries, not the whole retrieved set.
    dataset = group.rollouts[0].dataset if group.rollouts else "default"
    entries = library.retrieve(characterize_query_profile(group.question, dataset) + " " + group.question, limit=3)
    any_success = any(
        compute_task_reward(float(r.em), float(r.f1), float(r.contain)) >= 0.45
        and bool((r.predicted_answer or "").strip())
        for r in group.rollouts
    )
    for entry in entries:
        library.update_utility(entry.id, success=any_success)


def _has_efficiency_contrast(group: GroupResult) -> bool:
    """Allow HERA reflection on same-quality groups with meaningful token spread.

    HERA reflects on ranked group trajectories. In practice, local Qwen often
    gives identical correctness across samples; ignoring those groups starves
    the experience library. This keeps the group-relative update but admits
    groups where task quality is tied and efficiency differs materially.
    """
    if len(group.rollouts) < 2:
        return False
    scores = [compute_task_reward(float(r.em), float(r.f1), float(r.contain)) for r in group.rollouts]
    tokens = [max(1, int(r.total_tokens)) for r in group.rollouts]
    baseline = TOKEN_BUDGET_BASELINES.get(group.rollouts[0].dataset, TOKEN_BUDGET_BASELINES["default"])
    avg_tokens = sum(tokens) / len(tokens)
    if max(scores) < 0.05:
        # All failed high-cost groups are informative: the library should learn
        # avoid/repair guidance instead of waiting for a successful rollout.
        return avg_tokens >= 1.25 * baseline or max(tokens) >= 1.8 * min(tokens)
    tied_quality = max(scores) - min(scores) <= 0.08
    token_spread = max(tokens) >= 1.35 * min(tokens)
    high_cost_plateau = avg_tokens >= 1.25 * baseline
    return tied_quality and (token_spread or high_cost_plateau)


def apply_experience_updates(
    library: ExperienceLibrary,
    new_insights: list[dict],
    reflection_lm: dspy.LM,
    max_library_size: int = 40,
) -> ExperienceLibrary:
    """Update experience library with aggressive consolidation."""
    if not new_insights:
        return library

    # Pre-prune: remove entries with utility < 0.2 and usage > 3
    stale_ids = [
        eid for eid, entry in library.entries.items()
        if entry.utility < 0.2 and entry.usage_count > 3
    ]
    for eid in stale_ids:
        library.prune(eid)
        logger.info("Pre-pruned stale entry %s (utility=%.2f)", eid,
                     library.entries.get(eid, ExperienceEntry(id="", profile="", insight="")).utility)

    lib_lines = []
    for entry in library.entries.values():
        lib_lines.append(
            f"  [{entry.id}] profile={entry.profile}, utility={entry.utility:.2f}, "
            f"usage={entry.usage_count}, roles={list(entry.target_roles)}: {entry.insight[:100]}"
        )
    library_text = "\n".join(lib_lines) if lib_lines else "(empty library)"

    # Truncate insight texts for the prompt
    truncated_insights = []
    for ins in new_insights:
        if isinstance(ins, dict):
            ins_copy = dict(ins)
            if "insight" in ins_copy:
                ins_copy["insight"] = " ".join(str(ins_copy["insight"]).split()[:32])
            truncated_insights.append(ins_copy)
    insights_text = json.dumps(truncated_insights, indent=2)

    prompt = EXPERIENCE_UPDATE_PROMPT.format(
        max_entries=max_library_size,
        n_entries=library.size(),
        library_text=library_text,
        new_insights_text=insights_text,
    )

    with dspy.context(lm=reflection_lm):
        response = reflection_lm(prompt)

    raw_text = response[0] if isinstance(response, list) else str(response)
    updates = parse_json_array(raw_text)
    if not updates:
        updates = []
        for ins in truncated_insights[: max(0, max_library_size - library.size())]:
            if not isinstance(ins, dict) or not ins.get("insight"):
                continue
            updates.append({
                "action": "ADD",
                "target_id": "",
                "insight": {
                    "profile": ins.get("query_type") or ins.get("profile") or "train_group",
                    "insight": " ".join(str(ins.get("insight", "")).split()[:32]),
                    "target_roles": ins.get("target_roles", ["orchestrator", "planner", "solver"]),
                    "applies_when": ins.get("applies_when", "similar query profile"),
                    "avoid_when": ins.get("avoid_when", ""),
                },
            })

    for update in updates:
        action = str(update.get("operation", update.get("action", update.get("option", "KEEP")))).upper()
        target_ids = update.get("target_entry_ids") or []
        if isinstance(target_ids, str):
            target_ids = [target_ids]
        target_id = str(update.get("target_id", "") or update.get("modified_from", "") or (target_ids[0] if target_ids else ""))
        insight_data = update.get("insight", {}) or {}
        if not insight_data and (update.get("merged_insight") or update.get("new_insight")):
            insight_data = {
                "profile": "train_group",
                "insight": update.get("merged_insight") or update.get("new_insight"),
                "target_roles": ["orchestrator", "planner", "solver"],
                "applies_when": "similar query profile",
                "avoid_when": "",
            }

        if action == "KEEP":
            continue
        if action in ("DELETE", "PRUNE") and target_id:
            library.prune(target_id)
            if action == "DELETE" or not insight_data or not insight_data.get("insight"):
                continue

        if not insight_data or not insight_data.get("insight"):
            continue

        # Enforce 32-word cap
        insight_text = " ".join(str(insight_data["insight"]).split()[:32])
        roles = insight_data.get("target_roles", ["planner", "solver", "synthesizer"])
        if isinstance(roles, str):
            roles = [roles]

        new_entry = ExperienceEntry(
            id="",
            profile=insight_data.get("profile", "general"),
            insight=insight_text,
            utility=0.5,
            target_roles=tuple(roles),
            applies_when=insight_data.get("applies_when", ""),
            avoid_when=insight_data.get("avoid_when", ""),
        )

        if action == "ADD" and library.size() < max_library_size:
            library.add(new_entry)
        elif action == "MODIFY" and target_id:
            library.modify(target_id, new_entry)
        elif action == "MERGE" and target_id:
            library.merge(target_id, new_entry)
        elif action == "PRUNE" and library.size() < max_library_size:
            library.add(new_entry)

    # Post-prune if over limit
    while library.size() > max_library_size:
        lowest = min(library.entries.values(), key=lambda e: (e.utility, -e.usage_count))
        library.prune(lowest.id)
        logger.info("Post-pruned entry %s to stay under %d", lowest.id, max_library_size)

    return library


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------

def parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start:end + 1])
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return []


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

async def train_epoch(
    train_data: list[dict],
    library: ExperienceLibrary,
    retriever: Retriever,
    reflection_lm: dspy.LM,
    epoch: int,
    K: int = 3,
    temperatures: tuple[float, ...] = (0.4, 0.7, 0.9),
    output_dir: Path | None = None,
) -> tuple[ExperienceLibrary, dict]:
    logger.info("=== Epoch %d: %d questions, K=%d ===", epoch, len(train_data), K)
    if K <= len(temperatures):
        temps = temperatures[:K]
    else:
        temps = temperatures + tuple(temperatures[-1] for _ in range(K - len(temperatures)))

    epoch_stats = {
        "epoch": epoch,
        "n_questions": len(train_data),
        "n_mixed_groups": 0,
        "n_all_correct": 0,
        "n_all_wrong": 0,
        "total_insights": 0,
        "avg_em": 0.0,
        "avg_f1": 0.0,
        "avg_contain": 0.0,
        "avg_tokens": 0.0,
        "avg_dual_reward": 0.0,
    }

    config = AmasPipelineConfig(
        max_retrievals_per_solver=3,
        repair_enabled=True,
        experience_library=library.to_text() if library.size() > 0 else "",
    )

    all_em, all_f1, all_contain, all_tokens, all_rewards = [], [], [], [], []
    all_insights: list[dict] = []

    for idx, item in enumerate(train_data):
        qid = str(item["id"])
        question = item["question"]
        gold = item["answer"]
        dataset = item.get("dataset", "default")
        logger.info("[Epoch %d][%d/%d] QID=%s", epoch, idx + 1, len(train_data), qid)

        group = await run_group_rollouts(
            question=question, qid=qid, gold_answer=gold,
            retriever=retriever, config=config, temperatures=temps,
            library=library, dataset=dataset,
        )

        for r in group.rollouts:
            all_em.append(r.em)
            all_f1.append(r.f1)
            all_contain.append(r.contain)
            all_tokens.append(r.total_tokens)
            all_rewards.append(r.dual_reward)

        if group.has_mixed_outcomes:
            epoch_stats["n_mixed_groups"] += 1
            insights = extract_semantic_advantages(
                group,
                reflection_lm,
                query_type=characterize_query_profile(question, dataset),
            )
            all_insights.extend(insights)
            epoch_stats["total_insights"] += len(insights)
        elif len(group.winners) == len(group.rollouts):
            epoch_stats["n_all_correct"] += 1
        else:
            epoch_stats["n_all_wrong"] += 1

        update_experience_credit_from_group(library, group)

    if all_insights:
        logger.info("Applying %d insights to experience library...", len(all_insights))
        library = apply_experience_updates(library, all_insights, reflection_lm, max_library_size=40)

    epoch_stats["avg_em"] = sum(all_em) / max(1, len(all_em))
    epoch_stats["avg_f1"] = sum(all_f1) / max(1, len(all_f1))
    epoch_stats["avg_contain"] = sum(all_contain) / max(1, len(all_contain))
    epoch_stats["avg_tokens"] = sum(all_tokens) / max(1, len(all_tokens))
    epoch_stats["avg_dual_reward"] = sum(all_rewards) / max(1, len(all_rewards))
    epoch_stats["library_size"] = library.size()

    logger.info(
        "Epoch %d done: avg_em=%.3f, avg_f1=%.3f, avg_tokens=%.0f, avg_reward=%.3f, library=%d",
        epoch, epoch_stats["avg_em"], epoch_stats["avg_f1"],
        epoch_stats["avg_tokens"], epoch_stats["avg_dual_reward"], library.size(),
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        library.save(output_dir / f"experience_library_epoch{epoch}.json")
        with open(output_dir / f"epoch{epoch}_stats.json", "w") as f:
            json.dump(epoch_stats, f, indent=2)

    return library, epoch_stats
