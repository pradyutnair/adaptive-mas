from __future__ import annotations

import dspy


class SummarizeRollout(dspy.Signature):
    """Analyze one ReAct rollout and extract transferable strategy lessons.

    The trajectory shows each step: THOUGHT, ACTION (hop/hop_batch/submit/finish), and RESULT
    (answer, confidence, chunk_id). A DIAGNOSIS section at the end flags issues like low confidence,
    conflicting answers, missed parallelism, and submit rejections.

    Focus on:
    - What decomposition strategy was used (single hop, bridge, parallel)?
    - Did the agent pick the right topology for this question type?
    - Were there avoidable errors (repeated queries, ignoring low confidence, not re-searching)?
    - What would a better strategy look like for this question type?
    - What is transferable to OTHER questions of a similar structure?

    Output a 2-4 sentence summary of reusable strategy insights. Do NOT mention specific entity
    names or question details; generalize to question patterns.
    """

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    trajectory: str = dspy.InputField(desc="Structured trace: steps with THOUGHT/ACTION/RESULT + DIAGNOSIS")
    score: float = dspy.InputField(desc="1.0 = exact match, 0.0 = wrong")
    stats: str = dspy.InputField(desc="JSON: topology, hops, retries, tokens, low_confidence_hops, submit_rejections, used_hop_batch")
    summary: str = dspy.OutputField(desc="2-4 sentence transferable strategy insight")


class ExtractGroupOps(dspy.Signature):
    """Given rollout summaries for one question (G rollouts at different temperatures),
    extract experience library operations.

    Compare winning rollouts (score=1.0) against losing ones (score=0.0).
    What did winners do differently? What patterns should be added to the experience library?

    Output a JSON list of ops: [{"op": "ADD", "text": "...", "rationale": "..."}, {"op": "MODIFY", "id": "E-001", "text": "...", "rationale": "..."}, {"op": "DELETE", "id": "E-003"}, {"op": "KEEP", "id": "E-002"}]
    Only ADD entries that are genuinely transferable to new questions. Do not add question-specific facts.
    """

    summaries: str = dspy.InputField(desc="Newline-separated rollout summaries for one question, each with score")
    current_library: str = dspy.InputField(desc="Current experience library entries (E-001: ..., E-002: ...)")
    ops_json: str = dspy.OutputField(desc="JSON list of ADD/MODIFY/DELETE/KEEP operations")


class OptimizeBatch(dspy.Signature):
    """Consolidate experience libraries from a batch of questions into one concise library.

    Merge duplicate or overlapping entries. Remove entries that contradict each other.
    Keep the library concise (max ~15 entries). Each entry should be a transferable strategy
    principle, not a question-specific fact.
    """

    batch_proposals: str = dspy.InputField(desc="Newline-separated library snapshots from each question in the batch")
    current_library: str = dspy.InputField()
    merged_library: str = dspy.OutputField(desc="One entry per line: E-NNN: strategy text")
