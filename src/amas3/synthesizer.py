"""Synthesizer: wh-target alignment (GPT-4o-mini, dspy.ChainOfThought).

Reads:
- original question (the wh-target lives here)
- all Findings on the FindingsBus
- the final node's evidence chunks

Emits the final answer span (1-6 words). Enforces wh-target alignment so
the answer reflects what the original question asks (not a bridge entity).

v3: keep strong prompt, no post-processing rejection (was rejecting too many
correct answers because final_evidence chunks differ from where the
information actually lived).
"""
from __future__ import annotations
import json
import re
from typing import Any
import dspy
from .types import Finding, RetrievedChunk
from .working_memory import FindingsBus


class WhTargetAlignedSynthesis(dspy.Signature):
    """Given the original question, prior atomic Findings, and the final-hop
evidence chunks, emit the SHORT answer span (1-6 words) that DIRECTLY
answers the original question's wh-target.

GROUNDING RULES:
1. Prefer answers that appear verbatim in either the final_evidence
   chunks OR in the Findings (with their evidence already validated).
2. The original question is multi-hop. Earlier sub-answers are bridge
   entities; do NOT return them as the final answer unless the original
   question explicitly asks for them.
3. wh-target type matching:
   - when / what year / what date / what month -> date or year span
   - where / which city / which place / which county -> place name
   - who -> person's full name
   - how many / how much / number of -> a number
   - which / what (entity) -> entity name
4. If multiple Findings are inconsistent, prefer what the final_evidence
   chunks state over what intermediate Findings claim.
5. Use the FULL form as it appears in evidence (do not abbreviate names,
   dates, or place identifiers).

Output STRICT JSON: {"answer": <str>, "answer_type": <str>, "justification": <str>, "support_ids": [<chunk_id>, ...]}.
"""
    original_question: str = dspy.InputField()
    findings_summary: str = dspy.InputField(desc='List of {node_id, sub_question, answer, status, confidence}')
    final_evidence_json: str = dspy.InputField(desc='Top-k chunks supporting the final hop, as JSON')
    final_json: str = dspy.OutputField()


def _format_findings(bus: FindingsBus) -> str:
    rows = []
    for f in bus.all():
        rows.append({
            'node_id': f.node_id,
            'sub_question': f.sub_question,
            'answer': f.answer,
            'confidence': round(f.confidence, 3),
            'status': f.status.value,
        })
    return json.dumps(rows, ensure_ascii=False)


def _format_evidence(chunks: list[RetrievedChunk], excerpt_chars: int = 700) -> str:
    return json.dumps([
        {'chunk_id': c.chunk_id, 'text': c.text[:excerpt_chars]}
        for c in chunks
    ], ensure_ascii=False)


def _parse_synth(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def run_synthesizer(
    *,
    synth_lm: dspy.LM,
    original_question: str,
    bus: FindingsBus,
    final_evidence: list[RetrievedChunk],
    experience: str = "",
) -> tuple[dict[str, Any], int]:
    findings_summary = _format_findings(bus)
    final_evidence_json = _format_evidence(final_evidence)
    sig = WhTargetAlignedSynthesis
    if experience:
        base = sig.instructions or sig.__doc__ or ""
        sig = sig.with_instructions("Prior experiential knowledge from past attempts:\n" + experience + "\n\n" + base)
    with dspy.context(lm=synth_lm):
        cot = dspy.ChainOfThought(sig)
        pred = cot(
            original_question=original_question,
            findings_summary=findings_summary,
            final_evidence_json=final_evidence_json,
        )
    try:
        history = synth_lm.history[-1] if synth_lm.history else None
        usage = (history or {}).get('usage') or {}
        tokens = int(usage.get('total_tokens', 0))
    except Exception:
        tokens = 0
    obj = _parse_synth(getattr(pred, 'final_json', ''))
    if not obj:
        obj = {'answer': '', 'answer_type': 'other', 'justification': 'parse_failed', 'support_ids': []}
    return obj, tokens
