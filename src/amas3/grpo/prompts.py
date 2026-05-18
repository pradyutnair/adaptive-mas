"""Prompt templates for TF-GRPO orchestration, reflection, and library ops.

All prompts live here so they can be edited without touching the runtime
logic in topology.py / reflection.py / library_update.py.
"""
from __future__ import annotations

AGENT_DESCRIPTIONS = """\
Note: the "orchestrator" in this codebase is pi_O (this topology sampler).
The executor supports exactly two routing strategies. You do NOT choose
individual agents or execution orders; the strategy fully determines the
downstream agent set. Choose the cheapest strategy that the query
semantics + retrieved experiences support.

routing_strategy semantics:
- "sas_first"  = sas_solver probes first and may chain up to retrieval_budget
                 followup retrievals. Accept the SAS answer only when it is
                 confident and verifier-approved; otherwise the executor
                 naturally escalates to planner -> solver -> synth. Use for
                 simple factoids and bridge questions where the intermediate
                 entity is likely recoverable from retrieval.
- "direct_mas" = skip SAS entirely and run planner -> solver -> synth. Use
                 when the question itself clearly requires decomposition,
                 intersection, comparison, temporal chaining, or multiple
                 constraints where a cheap probe is likely to waste tokens."""


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
You are analyzing same-query AMAS rollouts to extract actionable insights.
The system has two routing strategies (sas_first, direct_mas) and multiple
agents (planner, solver, synth, sas_solver). Learn what worked and why.

Query: {question}
Query type: {query_type}

Current experience library:
{library_text}

The following trajectory summaries were executed for the SAME query, ranked by
task performance first and token cost second:
{trajectory_summaries}

Compare high-advantage and low-advantage trajectories. Identify:
1. Routing: did sas_first save tokens on a simple query, or did direct_mas
   avoid a wasteful SAS probe on a complex one?
2. Decomposition: did a shallower/deeper plan lead to a better answer?
3. Retrieval: did more/fewer retrievals help find the right evidence?
4. Synthesis: did the synth correctly combine sub-answers?

Extract 1-3 actionable insights. Each insight must be <=32 words and target
the SPECIFIC role that should act on it:
- "orchestrator" for routing decisions (sas_first vs direct_mas)
- "planner" for decomposition strategy
- "solver" for retrieval and evidence extraction
- "synth" for answer synthesis

Output STRICT JSON:
{{"success_factors":["<factor>"],"failure_modes":["<mode>"],"insights":[{{"query_type":"<type>","insight":"<<=32 words>","target_roles":["orchestrator|planner|solver|synth"],"token_impact":"saves|costs|neutral","estimated_token_savings":"<rough estimate>"}}]}}

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
- Favor insights that are role-specific and actionable: routing rules for
  orchestrator, decomposition guidance for planner, retrieval strategies for
  solver, synthesis rules for synth.

## Output: JSON array, one object per new insight.
## target_roles values must come from:
##   {{"orchestrator","sas_solver","planner","solver","synth"}}
## where "orchestrator" = the GRPO topology sampler pi_O (this policy).
[{{"operation":"ADD|MERGE|DELETE|MODIFY|KEEP","new_insight":"<text>","target_entry_ids":["<id>"],"merged_insight":"<text or null>","rationale":"<one sentence>","insight":{{"profile":"<query type>","insight":"<<=32 words>","target_roles":["orchestrator|sas_solver|planner|solver|synth"],"applies_when":"<condition>","avoid_when":"<condition>"}}}}]

Return ONLY the JSON array."""


TOPOLOGY_SAMPLING_PROMPT = """\
You are pi_O, the GRPO orchestrator for multi-hop QA. Your only job is to
emit a topology specification; you do NOT run any agent yourself.

Strategies (the executor supports EXACTLY these two):
{agent_descriptions}

Past experiences (ranked by utility):
{experience_text}

Semantic query profile:
{query_profile}

Deployment token budget B:
{budget_block}

Already sampled topologies for this rollout group:
{avoid_topologies_text}

Sampling directive:
{sampling_directive}

Query: {question}

Choose ONE routing_strategy from {{sas_first, direct_mas}} based on the query,
the retrieved experiences, and B. Pick retrieval_budget in [1,3]: it caps how
many followup retrievals sas_first may issue beyond its initial probe.
For direct_mas, still emit a retrieval_budget for downstream solver caps.
If this group already has a sampled topology, do not copy it unless the query
semantics and experiences make every alternative indefensible. Otherwise sample
the nearest meaningful counterfactual: cheaper SAS-first probing when the first
sample may over-allocate effort, or direct MAS when the first sample may miss
bridge/intersection evidence. Duplicate signatures weaken GRPO credit.
The reward penalizes wrong answers first; token efficiency matters only after
answer quality is preserved.

Return STRICT JSON (no other keys):
{{"query_profile":"<one sentence>","routing_strategy":"sas_first|direct_mas","retrieval_budget":<int 1-3>,"repair":false,"rationale":"<one sentence>"}}"""


TOPOLOGY_MUTATION_PROMPT = """\
You are pi_O, the GRPO orchestrator, repairing failed multi-hop QA trajectories.

Strategies (the executor supports EXACTLY these two):
{agent_descriptions}

Question: {question}

Failed trajectories:
{failed_trajectories}

Propose ONE semantically justified routing change. Keep it minimal and
token-aware. You may only switch between the two routing strategies and
tune retrieval_budget / repair.

Return STRICT JSON (no other keys):
{{"query_profile":"<failure-aware profile>","routing_strategy":"sas_first|direct_mas","retrieval_budget":<int 1-3>,"repair":false,"rationale":"<why this mutation addresses the observed failure>"}}"""
