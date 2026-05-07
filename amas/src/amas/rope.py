"""Role-aware Prompt Evolution (RoPE): full version with variant generation, re-execution, contrastive analysis.

Pipeline (per under-performing agent):
1. Pull recent failed trajectories from agent's failure buffer.
2. Generate K variant prompts along behavioral axes (thoroughness, risk sensitivity, error correction, heuristic injection).
3. Re-execute each failed trajectory's topology with the variant prompt swapped in for the failing agent.
4. Contrastive analysis: best variant vs original failed run -> operational rules + behavioral principles.
5. Consolidate updated prompt (max length / coherence constraints).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .agents import AgentPrompt
from .library import ExperienceLibrary
from .lm import VLLMClient, parse_json_lenient
from .orchestrator import Orchestrator, Trajectory

logger = logging.getLogger(__name__)


# Paper Algorithm 5 line 3: axis ∈ {efficiency, thoroughness, risk_sensitivity}.
# (Paper §3.4.1 main text additionally mentions error_correction + heuristic_injection,
#  retained as descriptive guidance within thoroughness/risk_sensitivity prompts.)
VARIANT_AXES = [
    ("efficiency", "Reduce redundant steps; minimize tokens; prefer fewest sub-questions sufficient to answer."),
    ("thoroughness", "Extract more sub-questions; keep richer context; consider every entity; check granularity, units, and entity disambiguation."),
    ("risk_sensitivity", "Be conservative: only act when evidence is explicit; flag ambiguity; inject heuristics (year-only for year questions, ISO dates, coreference resolution)."),
]


VARIANT_SYSTEM = (
    "You are refining the prompt for a specific agent in a multi-agent QA system. Generate K prompt variants "
    "that explore different behavioral biases. Always respond with valid JSON."
)


def build_variant_user(prompt: AgentPrompt, failure_examples: list[dict], num_variants: int) -> str:
    op = "\n".join(f"  - {r}" for r in prompt.operational_rules) or "  (none)"
    bp = "\n".join(f"  - {r}" for r in prompt.behavioral_principles) or "  (none)"
    fail_blob = "\n".join(
        f"- query: {ex['query']!r}\n  agent_output: {json.dumps(ex.get('output', {}))[:300]}\n  trajectory: {ex.get('summary','')[:300]}\n  gold: {ex.get('gold','')}"
        for ex in failure_examples
    )
    return f"""Agent name: {prompt.name}
Role: {prompt.role}
Current operational rules:
{op}
Current behavioral principles:
{bp}

Recent failures by this agent:
{fail_blob}

Generate {num_variants} prompt variants. Each variant must:
- Inherit the role and instructions.
- Add or rewrite operational rules / behavioral principles to explore a different bias.
- Stay short (<=6 operational rules, <=4 behavioral principles).

Behavioral axes to consider (choose one per variant):
{chr(10).join(f'  - {ax[0]}: {ax[1]}' for ax in VARIANT_AXES)}

Respond in JSON:
{{
  "variants": [
    {{"axis": "<axis>", "operational_rules": ["..."], "behavioral_principles": ["..."]}},
    ...
  ]
}}
"""


# Verbatim Appendix B — "RoPE: contrastive analysis".
CONTRASTIVE_SYSTEM = (
    "You are analyzing the results of prompt variant re-executions for a failing agent to "
    "extract concrete prompt improvements."
)


def build_contrastive_user(prompt: AgentPrompt, original_prompt_text: str,
                            variant_results_blob: str) -> str:
    return (
        f"Agent name: {prompt.name}\n"
        f"Agent role: {prompt.role}\n"
        f"Original prompt: {original_prompt_text}\n\n"
        f"Variant execution results:\n{variant_results_blob}\n\n"
        "Each result entry contains:\n"
        "• Variant prompt: {variant prompt}\n"
        "• Re-executed trajectory: {trajectory}\n"
        "• F1 score: {f1}\n"
        "• Tokens used: {tokens}\n\n"
        "Perform a contrastive analysis between successful and failed variants:\n"
        "1. Operational rules (Δρ_op_i): Extract short-term corrective behaviors — specific, "
        "concrete instructions that directly address the observed failure pattern. These should "
        "be actionable in the agent's very next execution.\n"
        "2. Behavioral principles (Δρ_bp_i): Extract long-term strategic generalizations — "
        "higher-level guidance distilled from patterns across multiple trajectories. These shape "
        "the agent's overall approach rather than fixing a single failure.\n\n"
        "Respond in the following JSON format:\n"
        "{\n"
        '  "operational_rules": [\n'
        "    {\n"
        '      "rule": "<concrete instruction>",\n'
        '      "derived_from": "<which variant comparison motivated this rule>"\n'
        "    },\n"
        "    ...\n"
        "  ],\n"
        '  "behavioral_principles": [\n'
        "    {\n"
        '      "principle": "<strategic guidance>",\n'
        '      "derived_from": "<pattern observed across which variants>"\n'
        "    },\n"
        "    ...\n"
        "  ],\n"
        '  "updated_prompt": "<full revised prompt integrating all rules and principles into '
        'the original, with redundant instructions pruned>"\n'
        "}"
    )


# Verbatim Appendix B — "Prompt for Agent Prompt Integration".
PROMPT_INTEGRATION_SYSTEM = (
    "You are responsible for managing and evolving the prompt of a single agent under strict "
    "length and structural constraints. Your task is to refine, optimize, and consolidate the "
    "agent's prompt to ensure clarity, consistency, and adherence to operational and behavioral "
    "requirements."
)


def build_prompt_integration_user(current_prompt: str, op_rules: list[str],
                                    behavioral_principles: list[str], K: int = 6) -> str:
    return (
        f"Current Agent Prompt:\n{current_prompt}\n\n"
        f"Proposed Operational Rules:\n" + "\n".join(f"  - {r}" for r in op_rules) + "\n\n"
        f"Proposed Behavioral Principles:\n" + "\n".join(f"  - {p}" for p in behavioral_principles) + "\n\n"
        "Constraints:\n"
        f"1. Limit the number of tactical rules to a maximum of {K}.\n"
        "2. All instructions must be internally consistent — no contradictions.\n"
        "3. Preserve the agent's core role definition and tool usage instructions.\n"
        "4. Remove redundant, contradictory, or ambiguous instructions.\n"
        "5. Preserve essential operational and behavioral requirements.\n"
        "6. Ensure the updated prompt is concise, coherent, and actionable.\n\n"
        "Task:\n"
        "Produce a prompt diff that clearly indicates the modifications required to integrate "
        "the proposed rules and principles into the current agent prompt while satisfying the "
        "constraints above. Highlight additions, deletions, and replacements in a structured "
        "format.\n\n"
        "Respond in the following JSON format:\n"
        "{\n"
        '  "additions": ["<rule or principle to add>"],\n'
        '  "deletions": ["<rule or principle to remove>"],\n'
        '  "replacements": [{"old": "<text>", "new": "<text>"}],\n'
        '  "final_operational_rules": ["..."],\n'
        '  "final_behavioral_principles": ["..."]\n'
        "}"
    )


@dataclass
class FailureRecord:
    query: str
    gold: Any
    profile: str
    topology: dict[str, Any]
    agent_output: dict[str, Any]
    f1: float
    summary: str  # short trajectory summary


@dataclass
class FailureBuffer:
    by_agent: dict[str, deque] = field(default_factory=dict)
    capacity: int = 8

    def add(self, agent_name: str, rec: FailureRecord) -> None:
        if agent_name not in self.by_agent:
            self.by_agent[agent_name] = deque(maxlen=self.capacity)
        self.by_agent[agent_name].append(rec)

    def take(self, agent_name: str, k: int = 4) -> list[FailureRecord]:
        return list(self.by_agent.get(agent_name, deque()))[-k:]

    def size(self, agent_name: str) -> int:
        return len(self.by_agent.get(agent_name, deque()))


def add_traj_to_buffer(buffer: FailureBuffer, traj: Trajectory) -> None:
    """Add failed trajectory to the buffer for the agent identified as failure source."""
    if traj.f1 > 0:
        return
    fa = traj.failed_agent
    if not fa:
        return
    inv = next((i for i in traj.invocations if i.name == fa), None)
    if inv is None:
        return
    summary = " -> ".join(
        f"{i.name}:{(i.output.get('answer','') or json.dumps(i.output)[:40])[:40]}" for i in traj.invocations
    )
    buffer.add(fa, FailureRecord(
        query=traj.query,
        gold=traj.gold,
        profile=traj.profile,
        topology=traj.topology,
        agent_output=inv.output,
        f1=traj.f1,
        summary=summary[:600],
    ))


async def generate_variants(vllm: VLLMClient, prompt: AgentPrompt, failures: list[FailureRecord],
                              num_variants: int = 3) -> list[dict[str, Any]]:
    examples = [
        {"query": f.query, "output": f.agent_output, "summary": f.summary, "gold": str(f.gold)[:80]}
        for f in failures[:4]
    ]
    user = build_variant_user(prompt, examples, num_variants)
    res = await vllm.chat(VARIANT_SYSTEM, user, temperature=0.7, max_tokens=900)
    parsed = parse_json_lenient(res.text)
    variants = parsed.get("variants", []) if isinstance(parsed, dict) else []
    # Sanity filter
    cleaned = []
    for v in variants:
        if not isinstance(v, dict):
            continue
        op = v.get("operational_rules", [])
        bp = v.get("behavioral_principles", [])
        if not isinstance(op, list) or not isinstance(bp, list):
            continue
        cleaned.append({
            "axis": str(v.get("axis", "unknown")),
            "operational_rules": [str(x) for x in op[:6]],
            "behavioral_principles": [str(x) for x in bp[:4]],
        })
    return cleaned[:num_variants]


def make_variant_prompt(base: AgentPrompt, variant: dict[str, Any]) -> AgentPrompt:
    return AgentPrompt(
        name=base.name,
        role=base.role,
        instructions=base.instructions,
        operational_rules=list(variant.get("operational_rules", [])),
        behavioral_principles=list(variant.get("behavioral_principles", [])),
        output_schema=base.output_schema,
    )


async def reexecute_with_variant(orchestrator: Orchestrator, base_prompts: dict[str, AgentPrompt],
                                   agent_name: str, variant_prompt: AgentPrompt,
                                   failures: list[FailureRecord]) -> list[Trajectory]:
    """Run each failed trajectory's topology with the variant prompt for the agent."""
    swapped = dict(base_prompts)
    swapped[agent_name] = variant_prompt
    # Temporarily swap orchestrator's prompts.
    orig = orchestrator.prompts
    orchestrator.prompts = swapped
    try:
        tasks = [
            orchestrator.execute(f.query, f.gold, f.topology, [], orch_tokens=0)
            for f in failures
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)
    finally:
        orchestrator.prompts = orig


def _coerce_rule_list(items, key: str) -> list[str]:
    """Variant items in paper schema are dicts {rule, derived_from}; flatten to strings."""
    out: list[str] = []
    for x in items or []:
        if isinstance(x, dict):
            v = x.get(key) or next(iter(x.values()), "")
            if v:
                out.append(str(v))
        elif isinstance(x, str):
            out.append(x)
    return out


async def contrastive_update(vllm: VLLMClient, prompt: AgentPrompt,
                              failures: list[FailureRecord],
                              variant_results: list[tuple[dict[str, Any], list[Trajectory]]]
                              ) -> tuple[list[str], list[str]]:
    """Run paper Appendix B 'RoPE: contrastive analysis' verbatim.

    Returns (proposed_op_rules, proposed_behavioral_principles). Caller must pass these
    through `prompt_integration` to produce the final prompt under Π_C constraints.
    """
    # Render variant_results in paper format.
    variant_blob_parts = []
    for v, trajs in variant_results:
        avg_f1 = sum(t.f1 for t in trajs) / max(1, len(trajs))
        avg_tok = sum(t.total_tokens for t in trajs) / max(1, len(trajs))
        traj_summary = []
        for t in trajs[:3]:
            inv = next((i for i in t.invocations if i.name == prompt.name), None)
            traj_summary.append(
                f"      • final_answer={t.answer!r}, agent_output={(json.dumps(inv.output) if inv else '')[:200]}"
            )
        variant_blob_parts.append(
            f"  Variant axis: {v.get('axis','?')}\n"
            f"    Variant prompt operational_rules: {v.get('operational_rules',[])}\n"
            f"    Variant prompt behavioral_principles: {v.get('behavioral_principles',[])}\n"
            f"    Re-executed trajectories ({len(trajs)} runs):\n" + "\n".join(traj_summary) + "\n"
            f"    F1 score (mean over trajectories): {avg_f1:.3f}\n"
            f"    Tokens used (mean): {avg_tok:.0f}"
        )
    variant_blob = "\n\n".join(variant_blob_parts)

    # Render original prompt as a single text block (system_prompt is canonical).
    original_prompt_text = prompt.system_prompt()
    user = build_contrastive_user(prompt, original_prompt_text, variant_blob)

    res = await vllm.chat(CONTRASTIVE_SYSTEM, user, temperature=0.3, max_tokens=1200)
    parsed = parse_json_lenient(res.text)
    if not isinstance(parsed, dict):
        return ([], [])

    new_op = _coerce_rule_list(parsed.get("operational_rules", []), "rule")
    new_bp = _coerce_rule_list(parsed.get("behavioral_principles", []), "principle")
    return (new_op, new_bp)


async def prompt_integration(vllm: VLLMClient, prompt: AgentPrompt,
                              proposed_op: list[str], proposed_bp: list[str],
                              max_op_rules: int = 6, max_principles: int = 4) -> AgentPrompt:
    """Run paper Appendix B 'Prompt for Agent Prompt Integration' verbatim.

    Π_C projection: enforces length cap K and consistency. Applied AFTER contrastive_update.
    """
    user = build_prompt_integration_user(
        current_prompt=prompt.system_prompt(),
        op_rules=proposed_op,
        behavioral_principles=proposed_bp,
        K=max_op_rules,
    )
    res = await vllm.chat(PROMPT_INTEGRATION_SYSTEM, user, temperature=0.2, max_tokens=900)
    parsed = parse_json_lenient(res.text)

    final_op = parsed.get("final_operational_rules") if isinstance(parsed, dict) else None
    final_bp = parsed.get("final_behavioral_principles") if isinstance(parsed, dict) else None

    if not isinstance(final_op, list) or not final_op:
        # Fallback: deduplicate proposed rules vs current and cap.
        seen = set(prompt.operational_rules)
        final_op = list(prompt.operational_rules)
        for r in proposed_op:
            if r and r not in seen:
                final_op.append(r)
                seen.add(r)
    if not isinstance(final_bp, list) or not final_bp:
        seen = set(prompt.behavioral_principles)
        final_bp = list(prompt.behavioral_principles)
        for p in proposed_bp:
            if p and p not in seen:
                final_bp.append(p)
                seen.add(p)

    return AgentPrompt(
        name=prompt.name,
        role=prompt.role,
        instructions=prompt.instructions,
        operational_rules=[str(x) for x in final_op[:max_op_rules]],
        behavioral_principles=[str(x) for x in final_bp[:max_principles]],
        output_schema=prompt.output_schema,
    )


async def rope_update_agent(orchestrator: Orchestrator, vllm: VLLMClient,
                             prompts: dict[str, AgentPrompt], buffer: FailureBuffer,
                             agent_name: str, num_variants: int = 3, max_failures: int = 4
                             ) -> AgentPrompt | None:
    failures = buffer.take(agent_name, max_failures)
    if len(failures) < 2 or agent_name == "Retriever":
        return None
    base = prompts[agent_name]
    variants = await generate_variants(vllm, base, failures, num_variants=num_variants)
    if not variants:
        return None
    variant_results: list[tuple[dict[str, Any], list[Trajectory]]] = []
    for v in variants:
        vp = make_variant_prompt(base, v)
        try:
            trajs = await reexecute_with_variant(orchestrator, prompts, agent_name, vp, failures)
            variant_results.append((v, trajs))
        except Exception as e:
            logger.warning("Variant reexec failed for %s: %s", agent_name, str(e)[:200])
            continue
    if not variant_results:
        return None
    proposed_op, proposed_bp = await contrastive_update(vllm, base, failures, variant_results)
    if not proposed_op and not proposed_bp:
        return None
    new_prompt = await prompt_integration(vllm, base, proposed_op, proposed_bp,
                                            max_op_rules=6, max_principles=4)
    return new_prompt
