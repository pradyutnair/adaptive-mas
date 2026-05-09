"""DSPy signatures for Training-Free GRPO compile loop (arXiv 2510.08191).

Three roles, faithful to the paper's Figures 11/12/13:

1. SummarizeRollout (Fig. 11): turn one rollout trajectory into a structured
   step-by-step summary, calling out detours/errors against the ground truth.
   Retains retrieval queries because what to retrieve is the transferable signal.

2. ExtractGroupOps (Fig. 12): given G summarised rollouts (with scores) for the
   same query and the current experience library, propose a small list of
   operations (ADD / MODIFY / DELETE / KEEP) that would improve the library.
   This is the per-group semantic advantage. Inputs include both winners and
   losers so the LLM can articulate why winners beat losers.

3. OptimizeBatch (Fig. 13): consolidate ALL per-group proposals from the
   current batch into a single, deduplicated update plan. Adds MERGE to fold
   redundant entries together. This is the optimization step that mirrors a
   GRPO gradient update: one update per batch, not per query.
"""
from __future__ import annotations
import dspy


class SummarizeRollout(dspy.Signature):
    """Summarise one rollout of a multi-hop QA agent (planner -> retrieval ->
solver -> synthesiser). Produce a structured step-by-step trace.

For each step describe (a) what the agent did, (b) which retrieval queries
were issued, (c) the per-hop answer, (d) which existing experience (if any)
plausibly drove the decision. Then, given the evaluation and ground truth,
flag detours, wrong sub-questions, bad retrieval queries, or wrong final
answers and explain WHY they likely happened.

Keep retrieval QUERIES (they are the transferable signal). Do NOT include
retrieved chunk text. Keep it concise: 4-8 short bullets total."""
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField(desc='Ground-truth answer or empty string if unavailable')
    plan_subgoals: str = dspy.InputField(desc='JSON list of {id, sub_question, depends_on}')
    findings: str = dspy.InputField(desc='JSON list of {node_id, sub_question, answer, status, confidence, queries_issued}')
    final_answer: str = dspy.InputField()
    score: float = dspy.InputField(desc='Reward in [0,1]; 1.0 = exact match')
    current_experience_library: str = dspy.InputField(desc='Numbered library that conditioned the rollout (may be empty)')
    summary: str = dspy.OutputField(desc='4-8 bullet step-by-step trace, ending with explicit failure-mode notes')


class ExtractGroupOps(dspy.Signature):
    """You are reviewing a group of G rollouts on the SAME multi-hop QA query.
Some scored higher than others. Compare WINNERS vs LOSERS and extract
TRANSFERABLE experiences that would steer future rollouts toward the winning
behaviour. This is the GRPO group-relative semantic advantage in operation form.

Allowed ops (return strict JSON):
- {"op": "ADD", "text": "<new experience>"}
- {"op": "MODIFY", "id": "<existing E-id>", "text": "<refined experience>"}
- {"op": "DELETE", "id": "<existing E-id>"}
- {"op": "KEEP"}

Hard rules:
- Each experience: 1 sentence, <=32 words, starts with the situation it applies
  to ("When the question requires...", "For bridge-entity sub-questions...").
- Strategic, transferable patterns ONLY: planning style, sub-question phrasing,
  retrieval query patterns, when to chain hops vs go single-shot, answer-span
  formatting, wh-target alignment. NEVER mention specific entities/numbers.
- If proposing ADD, ensure it is not already covered by an existing entry; else
  prefer MODIFY over ADD.
- Propose AT MOST 2 operations. KEEP is valid if no clear winner-vs-loser
  signal exists.

Output STRICT JSON: {"reasoning": "<short>", "operations": [...]}."""
    question: str = dspy.InputField()
    summaries_with_scores: str = dspy.InputField(desc='JSON list of {summary, score} for the G rollouts in this group')
    current_experience_library: str = dspy.InputField(desc='Library entries as "E1: <text>\\nE2: <text>\\n..." (may be empty)')
    operations_json: str = dspy.OutputField(desc='Strict JSON {"reasoning": str, "operations": [...]}')


class OptimizeBatch(dspy.Signature):
    """You are the BATCH OPTIMIZER for Training-Free GRPO. You receive the
current experience library and a list of per-group operation proposals
collected over the current batch. Your job is to consolidate them into a
single, coherent update plan that improves the library WITHOUT bloat.

Allowed ops (return strict JSON):
- {"op": "ADD", "text": "<new experience>"}
- {"op": "MODIFY", "id": "<existing E-id>", "text": "<refined experience>"}
- {"op": "MERGE", "ids": ["E3", "E7"], "text": "<consolidated experience>"}
- {"op": "DELETE", "id": "<existing E-id>"}

Hard rules:
- Drop near-duplicate proposals; if multiple proposals say the same thing,
  emit ONE op (preferably MODIFY of the most relevant existing entry).
- Use MERGE to fold 2+ existing entries that overlap into a single, more
  general entry; prefer MERGE over ADD when overlap is clear.
- Each entry: 1 sentence, <=32 words, starts with the situation it applies to.
- Strategic transferable patterns ONLY (no entities/numbers).
- Keep the library small and high-signal. Prefer DELETE on entries that
  contradict the proposals or were never useful.

Output STRICT JSON: {"reasoning": "<short>", "operations": [...]}."""
    current_experience_library: str = dspy.InputField(desc='Library entries as "E1: <text>\\nE2: <text>\\n..." (may be empty)')
    batch_proposals_json: str = dspy.InputField(desc='JSON list of {question, scores, ops:[...]} aggregated from all groups in the batch')
    operations_json: str = dspy.OutputField(desc='Strict JSON {"reasoning": str, "operations": [...]}')
