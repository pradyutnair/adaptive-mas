"""Synth recursion: 2-round refinement (RecursiveMAS spirit, training-free).

Round 1: standard Synth produces answer A1.
Round 2: Synth re-reads (Q, A1, justification, support_ids, evidence) and
         outputs a refined answer A2. Trust A2 (no voting, no ensembling).
         This is the same agent recursively refining its own output.

Cost: ~1 extra synth call (~1k tokens) per question.
"""
from __future__ import annotations
import json
import re
from typing import Any
import dspy
from .types import RetrievedChunk
from .synthesizer import _format_findings, _format_evidence, _parse_synth, run_synthesizer
from .working_memory import FindingsBus


class WhTargetSelfRefine(dspy.Signature):
    """Re-examine your own previous answer and refine if needed.

You are the same synthesizer that produced answer_v1. You have a chance to
reconsider with the same evidence. Apply rigorous self-criticism:

1. Does answer_v1 directly answer the original question's wh-target?
   If question asks "where", answer must be a place; "when", must be a date; etc.
2. Does the evidence clearly support answer_v1, or is it a stretch?
3. Could the question be asking about a different entity than answer_v1
   captures? (E.g., bridge entity vs final entity confusion.)
4. Is the form complete? (E.g., "Mississippi" vs "University of Mississippi"
   when question asks for the university.)

If answer_v1 is correct and well-grounded, KEEP IT (return same answer).
If you spot a clear error, FIX IT.
Do NOT change the answer if you're uncertain - prefer the original.

Output STRICT JSON: {"answer": <str>, "answer_type": <str>, "justification": <str>, "support_ids": [<chunk_id>, ...], "changed": <bool>}.
"""
    original_question: str = dspy.InputField()
    answer_v1: str = dspy.InputField(desc='The first-round answer to reconsider.')
    answer_type_v1: str = dspy.InputField()
    justification_v1: str = dspy.InputField()
    findings_summary: str = dspy.InputField()
    final_evidence_json: str = dspy.InputField()
    final_json: str = dspy.OutputField()


def run_synth_recursion(
    *,
    synth_lm: dspy.LM,
    original_question: str,
    bus: FindingsBus,
    final_evidence: list[RetrievedChunk],
    experience: str = "",
    rounds: int = 2,
) -> tuple[dict[str, Any], int]:
    """Run synth with recursive refinement. Returns (final_obj, total_tokens).

    rounds=1: equivalent to plain run_synthesizer.
    rounds=2: one refinement pass after the initial synth.
    """
    obj, tokens = run_synthesizer(
        synth_lm=synth_lm,
        original_question=original_question,
        bus=bus,
        final_evidence=final_evidence,
        experience=experience,
    )
    if rounds <= 1 or not obj.get('answer'):
        return obj, tokens

    findings_summary = _format_findings(bus)
    final_evidence_json = _format_evidence(final_evidence)
    sig = WhTargetSelfRefine
    if experience:
        base = sig.instructions or sig.__doc__ or ""
        sig = sig.with_instructions("Prior experiential knowledge:\n" + experience + "\n\n" + base)
    with dspy.context(lm=synth_lm):
        cot = dspy.ChainOfThought(sig)
        try:
            pred = cot(
                original_question=original_question,
                answer_v1=str(obj.get('answer', '')),
                answer_type_v1=str(obj.get('answer_type', 'other')),
                justification_v1=str(obj.get('justification', ''))[:500],
                findings_summary=findings_summary,
                final_evidence_json=final_evidence_json,
            )
        except Exception:
            return obj, tokens
    try:
        history = synth_lm.history[-1] if synth_lm.history else None
        usage = (history or {}).get('usage') or {}
        tokens += int(usage.get('total_tokens', 0))
    except Exception:
        pass
    obj2 = _parse_synth(getattr(pred, 'final_json', ''))
    if not obj2 or not obj2.get('answer'):
        return obj, tokens
    return obj2, tokens
