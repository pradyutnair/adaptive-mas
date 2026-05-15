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

FINAL-ANSWER CONTRACT (read carefully, this is the most common failure mode):

Step 1: Identify the wh-target of the ORIGINAL question. The wh-target tells
        you what TYPE the final answer must be:
        - when / what year / what date / what month -> date or year
        - where / which city / which place / which county -> place name
        - who -> person's full name
        - how many / how much / number of -> a number
        - which X / what X -> a specific entity matching X

Step 2: Identify FORBIDDEN BRIDGE ANSWERS. List the answers from intermediate
        Findings (node_id < final_node). These are bridge entities that helped
        you reach the final answer; they are NOT the final answer. Example:
        if Q1 asks "who founded BAND" and Q2 asks "where was that founder
        born", then BAND is a bridge and the Q1 answer is FORBIDDEN as final.

Step 3: Identify the REQUIRED RELATION between the final entity and bridges.
        Example: original Q "where was the founder of Steely Dan born",
        required relation is "birthplace of <bridge_entity>".

Step 4: Search the final_evidence chunks for a span that:
        (a) matches the wh-target type from Step 1,
        (b) is NOT in the forbidden-bridge list from Step 2 (unless the
            original question explicitly asks for one of them),
        (c) stands in the required relation from Step 3 with the bridge.

Step 5: Enforce the explicit answer category after the wh-word. If the
        original asks for a rocket/company/source/film/person/date/number,
        the final answer must be that category, not a related bridge.

Step 6: For comparisons/superlatives, use the candidate facts to decide the
        winner; do not treat an intermediate candidate as final unless the
        comparison relation has been resolved.

Step 7: Output the FULL form as it appears in evidence. Do not abbreviate.

GROUNDING RULES:
- Prefer answers that appear verbatim in final_evidence chunks.
- If multiple Findings are inconsistent, prefer what final_evidence states.
- If no final_evidence span satisfies (a)(b)(c), return the best approximation
  but lower confidence; do NOT default to a bridge answer.

Output STRICT JSON on ONE LINE (no surrounding prose):
{"answer": <str>, "answer_type": <str>, "justification": <str>, "support_ids": [<chunk_id>], "wh_target": <str>, "forbidden_bridges": [<str>]}.
"""
    original_question: str = dspy.InputField()
    findings_summary: str = dspy.InputField(desc='List of {node_id, sub_question, answer, status, confidence}. Earlier nodes are bridges.')
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


def _balanced_json_object(s: str) -> str | None:
    """Find the first balanced top-level JSON object in s, ignoring braces inside strings.

    Robust to extra braces in the rationale/preamble (the greedy regex
    {.*} approach was failing on Qwen3-14B think outputs that include
    braces in CoT before the final JSON line).
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return s[start:i + 1]
    return None


def _parse_synth(raw: str) -> dict[str, Any]:
    text = raw.strip()
    obj_str = _balanced_json_object(text)
    if obj_str:
        try:
            obj = json.loads(obj_str)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    # Try last balanced object (model may emit multiple)
    matches = list(re.finditer(r'\{', text))
    for m in reversed(matches):
        candidate = _balanced_json_object(text[m.start():])
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
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
