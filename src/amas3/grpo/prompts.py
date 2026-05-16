"""Prompt templates for TF-GRPO orchestration, reflection, and library ops.

All prompts live here so they can be edited without touching the runtime
logic in topology.py / reflection.py / library_update.py.
"""
from __future__ import annotations

AGENT_DESCRIPTIONS = """\
- orchestrator: routes to SAS shortcut or full MAS decomposition based on evidence quality.
- planner: decomposes multi-hop questions into ordered subgoals.
- solver: retrieves evidence and extracts grounded answer spans.
- synthesizer: aligns findings to the original wh-target and produces final answer.
- repair: retries when evidence is insufficient."""


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
You are managing a compact experience library E for a multi-hop QA system,
following Training-Free GRPO (arXiv:2510.08191). Whereas vanilla GRPO updates
model parameters theta via gradient ascent on JGRPO(theta), we update E using
all semantic advantages A_text from the current batch. Choose ONE action per
new insight from the 5-operation menu below.

The library MUST stay under {max_entries} entries. Currently: {n_entries} entries.

## Current Experience Library:
{library_text}

## New Insights from this Group's Semantic Advantages:
{new_insights_text}

## Operation menu (TF-GRPO + MERGE consolidation)
- ADD: Directly append a NOVEL experience not covered by any existing entry.
  Use only when no existing entry overlaps with the new insight. Requires
  library size < {max_entries}.
- MERGE <target_id>: Combine the new insight with ONE existing complementary
  entry into a single richer entry. Use when two entries share the same query
  type and recommend compatible strategies; reduces redundancy.
- DELETE <target_id>: Remove a low-quality experience based on A_text. Use
  when the new evidence shows an existing entry is wrong, conflicting with
  higher-utility guidance, or has utility < 0.3 with usage > 3.
- MODIFY <target_id>: Refine or improve an existing experience in place using
  insights from A_text. Use when ONE existing entry is partially correct but
  needs its insight / applies_when / avoid_when sharpened. Preserves the
  entry's id and history; do NOT use MODIFY when a second redundant entry
  exists (use MERGE instead).
- KEEP: E remains unchanged for this insight (already well-covered).

## Constraints
- Each insight text MUST be <=32 words.
- Prefer MODIFY over (DELETE + ADD) when only one entry needs to change.
- Prefer MERGE over MODIFY when consolidation reduces redundancy.
- Prefer DELETE when an entry contradicts a higher-utility entry.
- Prefer ADD only when no existing entry could be MERGEd or MODIFIed.
- Always favor insights about token efficiency and routing shortcuts (SAS
  lane, fewer agents, smaller retrieval budget) since the reward explicitly
  penalizes exceeding the learned token envelope.

## Output: JSON array, one object per new insight
[{{"operation":"ADD|MERGE|DELETE|MODIFY|KEEP","new_insight":"<text>","target_entry_ids":["<id>"],"merged_insight":"<text or null>","rationale":"<one sentence>","insight":{{"profile":"<query type>","insight":"<<=32 words>","target_roles":["planner|solver|synth|orchestrator"],"applies_when":"<condition>","avoid_when":"<condition>"}}}}]

Return ONLY the JSON array."""


TOPOLOGY_SAMPLING_PROMPT = """\
You are an orchestrator for multi-hop QA. Design a minimal, efficient topology.

Agents: {agent_descriptions}

Past experiences (ranked by utility):
{experience_text}

Semantic query profile:
{query_profile}

Deployment token budget B:
{budget_block}

Already sampled topologies for this rollout group:
{avoid_topologies_text}

Current rollout exploration axis:
{exploration_axis}

Query: {question}

Design a MINIMAL topology by reasoning about q, the retrieved experiences, the
deployment budget B, and the available agent pool (no fixed thresholds or
per-type tables):
1. Select ONLY the agents needed. Fewer agents = fewer tokens.
2. Decide routing_strategy from query semantics, retrieved insights, and B.
   When B is tight, strongly prefer routing_strategy="sas" and the smallest
   retrieval_budget. When B is generous, full_mas is acceptable for hard
   queries.
3. Set retrieval_budget from the smallest budget supported by the evidence
   and that fits comfortably within B.
4. Use the exploration axis to sample a semantically justified alternative,
   not a fixed template.
5. Do not duplicate an already sampled topology unless query semantics leave
   no safe alternative.
6. Prefer fewer agents and fewer retrievals; every agent call and retrieval
   adds tokens, so justify each one.

Return STRICT JSON:
{{"query_profile":"<one sentence>","selected_agents":["<agent>"],"execution_order":[{{"step":1,"agent":"<agent>","depends_on":[],"mode":"sequential|parallel"}}],"routing_strategy":"sas|full_mas|orchestrator_then_mas","retrieval_budget":<int 1-3>,"repair":false,"rationale":"<why this topology is efficient given B>"}}"""


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
{{"query_profile":"<failure-aware profile>","selected_agents":["<agent>"],"execution_order":[{{"step":1,"agent":"<agent>","depends_on":[],"mode":"sequential|parallel"}}],"routing_strategy":"sas|full_mas|orchestrator_then_mas","retrieval_budget":<int 1-3>,"repair":false,"rationale":"<why this mutation addresses the observed failure>"}}"""
