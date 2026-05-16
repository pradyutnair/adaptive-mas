"""Prompt templates for TF-GRPO orchestration, reflection, and library ops.

All prompts live here so they can be edited without touching the runtime
logic in topology.py / reflection.py / library_update.py.
"""
from __future__ import annotations

AGENT_DESCRIPTIONS = """\
Note: the "orchestrator" in this codebase is pi_O (this topology sampler).
The executor supports exactly three routing strategies. You do NOT choose
individual agents or execution orders; the strategy fully determines the
downstream agent set. Choose the cheapest strategy that the query
semantics + retrieved experiences support.

routing_strategy semantics:
- "sas"          = sas_solver only (no planner/solver/synth fallback). The
                   sas_solver itself loops: probe -> [retrieve -> reason]*
                   for up to retrieval_budget followup retrievals (1-3),
                   then answers. ~500-3500 tokens. Use for simple factoid
                   lookups, single-entity attribute queries, yes/no checks,
                   AND any 2-hop bridge whose intermediate entity is easy
                   to extract from the probe (the sas_solver can chain a
                   second retrieval on its own).
- "sas_then_mas" = sas_solver probes first; on low confidence the executor
                   escalates to planner -> solver -> synth.
                   ~3000-8000 tokens. Use when the query needs verification
                   or you suspect a bridge step the sas_solver may not
                   handle alone.
- "full_mas"     = planner -> solver -> synth (sas_solver disabled).
                   ~5000-12000 tokens. Use only when query semantics clearly
                   need multi-hop decomposition (compound bridge with
                   synthesis, intersection across multiple distinct
                   entities, multi-step temporal reasoning)."""


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

Each trajectory exposes a per-agent token breakdown (sas, sas_verifier,
planner, solver, synth) plus the deployment budget B and whether the
executor truncated the rollout via a 'Budget exit'. Attribute cost to
specific stages: do not write vague 'use fewer tokens' insights. Write
agent-specific insights tied to the observed breakdown.

Focus on:
1. Per-agent cost: which agent(s) actually consumed the budget on this group?
   Was synth the hotspot (too many evidence chunks)? Was solver the hotspot
   (too many retrieval hops)? Was planner over-decomposing?
2. Budget exits: if a trajectory hit Budget exit: yes@<stage>, what stage
   would pi_O have skipped to keep B? Did the truncation still answer or
   did it leave a blank?
3. Routing: did SAS suffice given the per-stage breakdown of the MAS
   alternatives? Was escalation to full_mas justified by the marginal F1?
4. Decomposition: did fewer subgoals yield equal F1 with lower planner+
   solver cost?
5. Synthesis: did slim/short evidence give equal F1 with lower synth cost?

Extract 1-3 actionable insights. EACH insight MUST address token efficiency and preserve/improve Contain:
- How to achieve same/better quality with fewer tokens
- Which steps to skip, combine, or shorten
- When to use cheap path (SAS) vs expensive path (full MAS)

Output STRICT JSON. target_roles values must come from:
{{"orchestrator","sas_solver","planner","solver","synth"}}
where "orchestrator" refers to the GRPO topology sampler pi_O (this policy),
NOT to a downstream executor.

{{"success_factors":["<factor>"],"failure_modes":["<mode>"],"insights":[{{"query_type":"<type>","insight":"<actionable insight <=32 words>","target_roles":["orchestrator|sas_solver|planner|solver|synth"],"token_impact":"saves|costs|neutral","estimated_token_savings":"<rough estimate>"}}]}}

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

## Output: JSON array, one object per new insight.
## target_roles values must come from:
##   {{"orchestrator","sas_solver","planner","solver","synth"}}
## where "orchestrator" = the GRPO topology sampler pi_O (this policy).
[{{"operation":"ADD|MERGE|DELETE|MODIFY|KEEP","new_insight":"<text>","target_entry_ids":["<id>"],"merged_insight":"<text or null>","rationale":"<one sentence>","insight":{{"profile":"<query type>","insight":"<<=32 words>","target_roles":["orchestrator|sas_solver|planner|solver|synth"],"applies_when":"<condition>","avoid_when":"<condition>"}}}}]

Return ONLY the JSON array."""


TOPOLOGY_SAMPLING_PROMPT = """\
You are pi_O, the GRPO orchestrator for multi-hop QA. Your only job is to
emit a topology specification; you do NOT run any agent yourself.

Strategies (the executor supports EXACTLY these three):
{agent_descriptions}

Past experiences (ranked by utility):
{experience_text}

Semantic query profile:
{query_profile}

Deployment token budget B:
{budget_block}

Already sampled topologies for this rollout group:
{avoid_topologies_text}

Query: {question}

Choose ONE routing_strategy from {{sas, sas_then_mas, full_mas}} based on the
query, the retrieved experiences, and B. Pick retrieval_budget in [1,3]: it
caps how many followup retrievals the sas_solver may issue beyond its
initial probe. The reward penalizes both incorrect/blank answers and tokens
over B; there is no fixed default strategy.

Return STRICT JSON (no other keys):
{{"query_profile":"<one sentence>","routing_strategy":"sas|sas_then_mas|full_mas","retrieval_budget":<int 1-3>,"repair":false,"rationale":"<one sentence>"}}"""


TOPOLOGY_MUTATION_PROMPT = """\
You are pi_O, the GRPO orchestrator, repairing failed multi-hop QA trajectories.

Strategies (the executor supports EXACTLY these three):
{agent_descriptions}

Question: {question}

Failed trajectories:
{failed_trajectories}

Propose ONE semantically justified routing change. Keep it minimal and
token-aware. You may only switch between the three routing strategies and
tune retrieval_budget / repair.

Return STRICT JSON (no other keys):
{{"query_profile":"<failure-aware profile>","routing_strategy":"sas|sas_then_mas|full_mas","retrieval_budget":<int 1-3>,"repair":false,"rationale":"<why this mutation addresses the observed failure>"}}"""
