"""Per-hop LLM verifier for DAG solver extractions.

Called after each extraction attempt in best-of-N selection.
Returns accept/reject + reason. Verifier uses the same model as solver
(homogeneous setup).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
import dspy


class VerifyExtraction(dspy.Signature):
    """Verify whether the extracted answer is correct and well-supported.

Return STRICT JSON: {"accept": true|false, "reason": "<short explanation>"}

ACCEPT only if ALL conditions hold:
1. The answer_span directly answers the sub_question (not a related but wrong fact).
2. The answer_span is explicitly supported by at least one evidence chunk.
3. The answer matches the expected_answer_type (date for date questions, etc.).
4. The answer is complete (full name, not fragment; full date, not partial).

REJECT if the answer is a bridge entity, a partial match, unsupported, or wrong type.
Do not use your own knowledge — judge only from the evidence provided.
"""
    sub_question: str = dspy.InputField()
    extracted_answer: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField()
    evidence_json: str = dspy.InputField(desc="Retrieved chunks used for extraction")
    verdict_json: str = dspy.OutputField()


@dataclass
class VerificationResult:
    accept: bool
    reason: str
    tokens: int = 0


def _parse_verdict(raw: str) -> dict:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def verify_extraction(
    *,
    verifier_lm: dspy.LM,
    sub_question: str,
    extracted_answer: str,
    expected_answer_type: str,
    evidence_chunks: list,
    excerpt_chars: int = 700,
) -> VerificationResult:
    if not extracted_answer:
        return VerificationResult(accept=False, reason="empty answer", tokens=0)

    chunks_repr = [
        {"chunk_id": getattr(c, "chunk_id", ""), "text": (getattr(c, "text", "") or "")[:excerpt_chars]}
        for c in evidence_chunks
    ]

    try:
        with dspy.context(lm=verifier_lm):
            pred = dspy.Predict(VerifyExtraction)(
                sub_question=sub_question,
                extracted_answer=extracted_answer,
                expected_answer_type=expected_answer_type,
                evidence_json=json.dumps(chunks_repr, ensure_ascii=False),
            )
        try:
            history = verifier_lm.history[-1] if verifier_lm.history else None
            usage = (history or {}).get("usage") or {}
            tokens = int(usage.get("total_tokens", 0))
        except Exception:
            tokens = 0
    except Exception as e:
        return VerificationResult(accept=True, reason=f"verifier_error:{type(e).__name__}", tokens=0)

    obj = _parse_verdict(getattr(pred, "verdict_json", ""))
    accept = bool(obj.get("accept", False))
    reason = str(obj.get("reason", ""))[:200]

    return VerificationResult(accept=accept, reason=reason, tokens=tokens)
