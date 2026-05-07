"""TF-GRPO: group rollouts + semantic advantage extraction + library ADD/MERGE/PRUNE/KEEP."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .library import ExperienceLibrary, profile_question
from .lm import VLLMClient, parse_json_lenient
from .orchestrator import Orchestrator, Trajectory

logger = logging.getLogger(__name__)


# Verbatim Appendix B — "Orchestrator: Semantic Advantage Extraction".
SEMANTIC_ADVANTAGE_SYSTEM = (
    "You are an orchestrator evaluating a group of multi-agent execution trajectories for the same "
    "query. Your goal is to identify why some trajectories succeeded and others failed, and extract "
    "generalizable insights to guide future orchestration."
)


def _trajectory_block(t: "Trajectory") -> str:
    """Per Appendix B: include topology used, per-agent actions/inputs/outputs, final answer, F1, tokens."""
    agents = [s["agent"] for s in t.topology["execution_order"]]
    per_agent = []
    for inv in t.invocations:
        out_repr = (inv.output.get("answer") if isinstance(inv.output, dict) and "answer" in inv.output
                    else inv.output)
        per_agent.append(f"  - {inv.name}: input=[upstream] output={json.dumps(out_repr, ensure_ascii=False)[:200]}")
    return (
        f"  Topology: {agents}\n"
        f"  Per-agent actions:\n" + "\n".join(per_agent) + "\n"
        f"  Final answer: {t.answer or '(empty)'}\n"
        f"  F1 score: {t.f1:.3f}\n"
        f"  Total tokens consumed: {t.total_tokens}"
    )


def build_semantic_advantage_user(query: str, profile: str, trajectories: list["Trajectory"]) -> str:
    rows = []
    for i, t in enumerate(trajectories, 1):
        rows.append(f"--- Trajectory {i} ---\n{_trajectory_block(t)}")
    G = len(trajectories)
    return (
        f"Query: {query}\n"
        f"Query type: {profile}\n\n"
        f"The following {G} trajectories were executed, ranked by task performance (F1) and "
        f"then by efficiency (total tokens consumed):\n\n"
        f"{chr(10).join(rows)}\n\n"
        "Each trajectory entry includes:\n"
        "• Topology used (agents selected, execution order)\n"
        "• Per-agent actions, inputs and outputs\n"
        "• Final answer produced\n"
        "• F1 score: {f1}\n"
        "• Total tokens consumed: {tokens}\n\n"
        "Perform a comparative analysis across the group:\n"
        "1. Identify what the successful trajectories did differently from the failed ones.\n"
        "2. Identify which agents, orderings, or dependencies contributed to success or failure.\n"
        "3. Identify recurring failure patterns (e.g., incorrect agent selection, redundant steps, "
        "missing retrieval before reasoning).\n"
        "4. Distill findings into concise, actionable insights applicable to future queries of the "
        "same type.\n\n"
        "Respond in the following JSON format:\n"
        "{\n"
        '  "success_factors": ["<factor>", ...],\n'
        '  "failure_modes": ["<failure_mode>", ...],\n'
        '  "insights": [\n'
        "    {\n"
        '      "query_type": "<type this insight applies to>",\n'
        '      "insight": "<actionable natural language insight>"\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}"
    )


# Auxiliary call (paper §3.4 requires "agent identified as primary failure source").
# Kept as a separate small prompt so the verbatim Appendix B SA extraction is preserved.
FAILURE_ATTRIBUTION_SYSTEM = (
    "You attribute failure of a multi-agent QA trajectory to a single primary agent. "
    "Always respond with valid JSON: {\"failed_agent\": \"<agent_name or null>\"}."
)


def build_failure_attribution_user(query: str, failure_modes: list[str], trajectories: list["Trajectory"]) -> str:
    seqs = []
    for t in trajectories:
        if t.f1 == 0:
            seq = [inv.name for inv in t.invocations]
            seqs.append(f"  - {seq} -> answer: {t.answer!r}")
    fm = "\n".join(f"  - {m}" for m in (failure_modes or []))
    return (
        f"Query: {query}\n\n"
        f"Failed trajectories:\n" + "\n".join(seqs) + "\n\n"
        f"Failure modes identified:\n{fm or '  (none)'}\n\n"
        "Identify the SINGLE agent most responsible for these failures. "
        "Respond with JSON: {\"failed_agent\": \"<agent_name>\"}."
    )


# Verbatim Appendix B — "Experience Library Operations".
LIBRARY_OPS_SYSTEM = (
    "You are managing an experience library for a multi-agent question answering system.\n"
    "The library stores insights as structured entries of the form:\n"
    "(query profile c, insight z, utility u)\n"
    "where utility u reflects how often insight z has led to successful agent orchestration."
)


def build_library_ops_user(current_lib: str, new_insights: list[dict[str, str]]) -> str:
    return (
        "Current experience library:\n"
        f"{current_lib or '(empty)'}\n\n"
        "New insights extracted from the latest execution group:\n"
        f"{json.dumps(new_insights, ensure_ascii=False, indent=2)}\n\n"
        "For each new insight, decide the appropriate consolidation operation:\n"
        "• ADD: Insert a distinct insight not covered by existing entries.\n"
        "• MERGE: Combine semantically similar or complementary insights into a single, more "
        "complete entry.\n"
        "• PRUNE: Remove a lower-utility entry that conflicts with a higher-utility one.\n"
        "• KEEP: Retain the current library without modification.\n\n"
        "Guidelines:\n"
        "• Prefer MERGE over ADD when insights share the same query type and recommend "
        "compatible strategies.\n"
        "• Prefer PRUNE when entries provide contradictory guidance for the same query type "
        "and differ substantially in utility.\n"
        "• Avoid unbounded growth—consolidate aggressively to maintain generalizability.\n\n"
        "Respond in the following JSON format:\n"
        "{\n"
        '  "operations": [\n'
        "    {\n"
        '      "operation": "ADD|MERGE|PRUNE|KEEP",\n'
        '      "new_insight": "<text of new insight>",\n'
        '      "target_entry_ids": ["<id of existing entry if MERGE or PRUNE>"],\n'
        '      "merged_insight": "<combined insight text if MERGE, else null>",\n'
        '      "rationale": "<one sentence explaining the decision>"\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}"
    )


@dataclass
class GroupResult:
    query: str
    profile: str
    trajectories: list[Trajectory]
    insights: list[dict[str, str]] = field(default_factory=list)
    failed_agents: list[str] = field(default_factory=list)
    success_factors: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)


def rank_group(trajs: list[Trajectory]) -> list[Trajectory]:
    """Rank by F1 desc, then total_tokens asc."""
    return sorted(trajs, key=lambda t: (-t.f1, t.total_tokens))


async def run_group_rollout(orchestrator: Orchestrator, query: str, gold: str | list[str],
                             profile: str, group_size: int, temperature: float = 0.9) -> list[Trajectory]:
    """Sample G topologies in parallel and execute each."""
    sampling = [orchestrator.sample_topology(query, profile, temperature=temperature) for _ in range(group_size)]
    sampled = await asyncio.gather(*sampling)
    exec_tasks = []
    for topo, ids, res in sampled:
        exec_tasks.append(orchestrator.execute(
            query, gold, topo, ids, orch_tokens=res.prompt_tokens + res.completion_tokens
        ))
    trajs = await asyncio.gather(*exec_tasks, return_exceptions=False)
    return list(trajs)


async def extract_semantic_advantage(vllm: VLLMClient, query: str, profile: str,
                                       trajectories: list[Trajectory]) -> dict[str, Any]:
    """Paper Algorithm 2 line 12-13: ReflectOnGroup. Pure verbatim Appendix B SA prompt."""
    user = build_semantic_advantage_user(query, profile, trajectories)
    res = await vllm.chat(SEMANTIC_ADVANTAGE_SYSTEM, user, temperature=0.4, max_tokens=1100)
    parsed = parse_json_lenient(res.text)
    if not isinstance(parsed, dict):
        parsed = {}
    parsed.setdefault("insights", [])
    parsed.setdefault("success_factors", [])
    parsed.setdefault("failure_modes", [])
    return parsed


async def attribute_failed_agent(vllm: VLLMClient, query: str, failure_modes: list[str],
                                   trajectories: list[Trajectory]) -> str | None:
    """Paper Algorithm 1 line 8: π_O.IDENTIFYFAILEDAGENTS(τ). Auxiliary call for RoPE."""
    if not any(t.f1 == 0 for t in trajectories):
        return None
    try:
        fa_user = build_failure_attribution_user(query, failure_modes or [], trajectories)
        fa_res = await vllm.chat(FAILURE_ATTRIBUTION_SYSTEM, fa_user, temperature=0.0, max_tokens=80)
        fa_parsed = parse_json_lenient(fa_res.text)
        if isinstance(fa_parsed, dict):
            fa = fa_parsed.get("failed_agent")
            if isinstance(fa, str) and fa:
                return fa
    except Exception:
        pass
    return None


async def apply_library_update(vllm: VLLMClient, library: ExperienceLibrary,
                                 new_insights: list[dict[str, str]]) -> None:
    if not new_insights:
        return
    cur = library.to_text() if library.entries else ""
    user = build_library_ops_user(cur, new_insights)
    res = await vllm.chat(LIBRARY_OPS_SYSTEM, user, temperature=0.3, max_tokens=900)
    parsed = parse_json_lenient(res.text)
    if isinstance(parsed, dict):
        ops = parsed.get("operations", [])
    elif isinstance(parsed, list):
        ops = parsed
    else:
        ops = []
    library.apply_ops(ops)


async def grpo_step(orchestrator: Orchestrator, vllm: VLLMClient, library: ExperienceLibrary,
                     query: str, gold: str | list[str], group_size: int = 4,
                     temperature: float = 0.9, enable_mutation: bool = True) -> GroupResult:
    profile = profile_question(query, qid=getattr(orchestrator, "_current_qid", None))
    trajs = await run_group_rollout(orchestrator, query, gold, profile, group_size, temperature)
    ranked = rank_group(trajs)
    f1s = [t.f1 for t in ranked]
    has_success = any(f > 0 for f in f1s)
    has_failure = any(f == 0 for f in f1s)

    gr = GroupResult(query=query, profile=profile, trajectories=ranked)
    # Reward used insights for successful trajectories.
    for t in ranked:
        for eid in t.used_insight_ids:
            library.reward(eid, hit=t.f1 > 0)

    # Paper Algorithm 2 lines 9-11: reflection only on mixed-outcome groups.
    if has_success and has_failure:
        sa = await extract_semantic_advantage(vllm, query, profile, ranked)
        gr.insights = sa.get("insights", []) or []
        gr.failure_modes = sa.get("failure_modes", []) or []
        gr.success_factors = sa.get("success_factors", []) or []
        await apply_library_update(vllm, library, gr.insights)
        # Algorithm 1 line 8: identify failed agents for RoPE handoff.
        fa = await attribute_failed_agent(vllm, query, gr.failure_modes, ranked)
        if fa:
            gr.failed_agents.append(fa)
            for t in ranked:
                if t.f1 == 0:
                    t.failed_agent = fa
        return gr
    if not has_success:
        # All-failure group: skip SA extraction (paper Algorithm 2). Identify failed agent for RoPE
        # and trigger TopologyMutation (Algorithm 6).
        fa = await attribute_failed_agent(vllm, query, [], ranked)
        if fa:
            gr.failed_agents.append(fa)
            for t in ranked:
                t.failed_agent = fa
        if enable_mutation:
            mutated = await topology_mutation_round(orchestrator, vllm, library, query, gold,
                                                     profile, ranked, group_size=group_size,
                                                     temperature=temperature, failed_agent=fa)
            if mutated:
                # Algorithm 6 line 13: feed mutated topologies back into orchestrator update loop.
                combined = ranked + mutated
                ranked2 = rank_group(combined)
                gr.trajectories = ranked2
                f1s2 = [t.f1 for t in ranked2]
                if any(f > 0 for f in f1s2) and any(f == 0 for f in f1s2):
                    sa2 = await extract_semantic_advantage(vllm, query, profile, ranked2)
                    new_ins = sa2.get("insights", []) or []
                    if new_ins:
                        gr.insights = new_ins
                        await apply_library_update(vllm, library, new_ins)
    # else: all-success group — skip SA + RoPE per Algorithm 2.
    return gr


def _build_replacement_topology(base_topo: dict, failed_agent: str, agent_pool: list[str]) -> dict | None:
    """Algorithm 6 Option A: SelectAlternative(N, exclude=N_i) and replace N_i with N_i'."""
    import copy
    candidates = [a for a in agent_pool if a != failed_agent]
    if not candidates:
        return None
    # Heuristic substitutions per agent role.
    sub_map = {
        "QueryDecomposer": "QueryRewriter",
        "QueryRewriter": "QueryDecomposer",
        "EvidenceSelector": "ContextValidator",
        "ContextValidator": "EvidenceSelector",
        "AnswerGenerator": "ConcludeAgent",
        "ConcludeAgent": "AnswerGenerator",
        "ReflectAgent": "ContextValidator",
    }
    repl = sub_map.get(failed_agent)
    if not repl or repl not in agent_pool:
        return None
    new_topo = copy.deepcopy(base_topo)
    for step in new_topo["execution_order"]:
        if step["agent"] == failed_agent:
            step["agent"] = repl
        # Update deps that referenced the failed agent.
        step["depends_on"] = [repl if d == failed_agent else d for d in step.get("depends_on", [])]
    if failed_agent in new_topo.get("selected_agents", []):
        new_topo["selected_agents"] = [
            repl if a == failed_agent else a for a in new_topo["selected_agents"]
        ]
    return new_topo


def _build_augmentation_topology(base_topo: dict, failed_agent: str, agent_pool: list[str]) -> dict | None:
    """Algorithm 6 Option B: CreateNewAgent + insert after failed N_i."""
    import copy
    # Reasonable augmentation: insert ReflectAgent or ContextValidator after the failure point
    # to add a verification step.
    aug_agent = None
    for cand in ("ReflectAgent", "ContextValidator", "EvidenceSelector"):
        if cand in agent_pool and cand != failed_agent:
            # Don't double-insert if already present.
            existing = {s["agent"] for s in base_topo["execution_order"]}
            if cand not in existing:
                aug_agent = cand
                break
    if aug_agent is None:
        return None
    new_topo = copy.deepcopy(base_topo)
    insert_after_step = None
    for s in new_topo["execution_order"]:
        if s["agent"] == failed_agent:
            insert_after_step = s["step"]
            break
    if insert_after_step is None:
        return None
    # Bump steps that come strictly after.
    for s in new_topo["execution_order"]:
        if s["step"] > insert_after_step:
            s["step"] += 1
    new_step = {"step": insert_after_step + 1, "agent": aug_agent,
                "depends_on": [failed_agent], "mode": "sequential"}
    new_topo["execution_order"].append(new_step)
    new_topo["execution_order"].sort(key=lambda x: x["step"])
    if aug_agent not in new_topo.get("selected_agents", []):
        new_topo.setdefault("selected_agents", []).append(aug_agent)
    return new_topo


async def topology_mutation_round(orchestrator: Orchestrator, vllm: VLLMClient,
                                    library: ExperienceLibrary,
                                    query: str, gold: str | list[str], profile: str,
                                    failed_trajs: list[Trajectory], group_size: int = 2,
                                    temperature: float = 0.9, failed_agent: str | None = None
                                    ) -> list[Trajectory]:
    """Paper Algorithm 6 — TopologyMutation (structural fallback).

    For each failed agent N_i:
      Option A: replace N_i with alternative N_i' (SelectAlternative).
      Option B: augment topology by inserting a new agent after N_i (CreateNewAgent).
    Candidate topologies fed back into orchestrator update (GRPO loop).
    """
    from .agents import AGENT_NAMES
    if not failed_trajs or not failed_agent:
        return []
    base_topo = failed_trajs[0].topology  # use the topmost failed topology as base
    candidates: list[dict] = []
    repl = _build_replacement_topology(base_topo, failed_agent, list(AGENT_NAMES))
    if repl:
        candidates.append(repl)
    aug = _build_augmentation_topology(base_topo, failed_agent, list(AGENT_NAMES))
    if aug:
        candidates.append(aug)

    if not candidates:
        # Fall back to mutation-hint resampling so the orchestrator can still propose alternatives.
        prior = [f"  - topology: {[s['agent'] for s in t.topology['execution_order']]}"
                 f" -> answer: {t.answer!r} (F1=0)" for t in failed_trajs[:3]]
        hint = "\n".join(prior + [f"  - identified failure source: {failed_agent}"])
        sampled = await asyncio.gather(*[
            orchestrator.sample_topology(query, profile, temperature=max(temperature, 0.95),
                                           mutation_hint=hint)
            for _ in range(group_size)
        ])
        exec_tasks = []
        for topo, ids, res in sampled:
            exec_tasks.append(orchestrator.execute(
                query, gold, topo, ids, orch_tokens=res.prompt_tokens + res.completion_tokens
            ))
        return list(await asyncio.gather(*exec_tasks, return_exceptions=False))

    # Execute Option A + Option B candidates.
    exec_tasks = [
        orchestrator.execute(query, gold, c, [], orch_tokens=0) for c in candidates
    ]
    return list(await asyncio.gather(*exec_tasks, return_exceptions=False))
