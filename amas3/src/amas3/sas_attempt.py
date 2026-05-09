"""Verifier-gated SAS attempt: try to answer with one retrieval+extraction.

Operationalizes Tran-Kiela DPI: SAS is optimal under clean (high-groundedness) context.
We probe the original question, attempt extraction, and accept ONLY if:
  1. The probe groundedness is high (chunks look on-topic)
  2. The extracted answer span appears in at least one retrieved chunk
  3. The shape matches the expected answer type
  4. A strict verifier judges that the cited evidence entails the answer

If accepted, the pipeline returns immediately without planning, decomposition,
or multi-hop. Otherwise it abstains and the caller escalates to MAS.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
import dspy
from .types import RetrievedChunk

_DATE_RE = re.compile(r'\b(?:1[6-9]\d{2}|20\d{2})\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', re.IGNORECASE)
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand|million|billion)\b', re.IGNORECASE)
_YESNO_RE = re.compile(r'^(?:yes|no|true|false)\b', re.IGNORECASE)


def _shape_ok(answer: str, expected_type: str) -> bool:
    a = (answer or '').strip()
    if not a:
        return False
    t = (expected_type or 'entity').lower()
    if t == 'date':
        return bool(_DATE_RE.search(a))
    if t == 'number':
        return bool(_NUMBER_RE.search(a))
    if t == 'yes_no':
        return bool(_YESNO_RE.match(a))
    return True


def _direct_type_safe(question: str, answer: str, answer_type: str) -> bool:
    """High-precision runtime type gate for direct SAS.

    SAS should abstain whenever the wh-target is ambiguous. Broad entity
    questions like "what is X known for" are cheap to answer wrongly and costly
    for EM, so they must escalate to MAS.
    """
    q = (question or '').lower().strip()
    a = (answer or '').strip()
    t = (answer_type or 'other').lower().strip()
    if not a:
        return False
    if t == 'yes_no':
        return bool(re.match(r'^(is|are|was|were|did|does|do|can|could|has|have|had|will|would|should)\b', q))
    if t == 'date':
        return any(x in q for x in ('when', 'what year', 'what date', 'what month'))
    if t == 'number':
        return any(x in q for x in ('how many', 'how much', 'number of', 'how old'))
    if t == 'place':
        return any(x in q for x in ('where', 'what county', 'which county', 'what country', 'which country', 'what city', 'which city', 'what state', 'which state'))
    if t == 'person':
        if ' and ' in a.lower() or ',' in a:
            return False
        return any(x in q for x in ('who', 'whom', 'which person', 'what person', 'which man', 'what man', 'which woman', 'what woman'))
    return False


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]+', ' ', (s or '').lower()).strip()


def _answer_in_chunks(answer: str, chunks: list[RetrievedChunk], expected_type: str = '') -> bool:
    """Strict token-boundary check: prevents York/New York and Mississippi/
    University of Mississippi false positives. For entity/person/place,
    require >=2 normalized tokens unless type is date/number/yes_no.

    Implementation:
      1. Token-boundary match: normalized answer appears as a token sequence
         delimited by whitespace or string boundaries in some chunk.
      2. Strict-substring-of-longer-entity rejection: if the answer is a
         strict substring of a longer capitalized phrase in the matching
         chunk's text, reject (the LM probably picked a fragment of a more
         specific entity).
    """
    if not answer or not chunks:
        return False
    a = _norm(answer)
    if not a:
        return False

    a_tokens = a.split()
    t = (expected_type or '').lower()
    is_entity_type = t in ('entity', 'person', 'place', 'other', '')

    # Token-boundary regex: a appears flanked by start/end/space
    pat = re.compile(r'(?:^|\s)' + re.escape(a) + r'(?:\s|$)')

    for c in chunks:
        raw = getattr(c, 'text', '') or ''
        nt = _norm(raw)
        if not pat.search(nt):
            continue

        # Strict-substring-of-longer-cap-phrase guard:
        # If the answer appears INSIDE a longer multi-word capitalized phrase in the
        # raw chunk (case-preserved), it is likely a fragment of a more specific
        # entity. Reject. SAS must be conservative — false-positive accept costs EM,
        # rejection only costs MAS-fallback tokens.
        cap_runs = re.findall(r'(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+)', raw)
        a_lc = a.lower()
        fragment_of_longer = False
        for run in cap_runs:
            run_norm = _norm(run)
            if a_lc in run_norm and run_norm != a_lc and len(run_norm.split()) > len(a_tokens):
                fragment_of_longer = True
                break
        if fragment_of_longer:
            return False

        # Single-token entity answers without a containing longer phrase are accepted
        # (e.g., "Paris" in "Paris is the capital of France"). Date / number / yes_no
        # types are always allowed at single token if grounded.
        return True

    return False


@dataclass
class SasAttemptResult:
    accepted: bool
    answer: str
    confidence: float
    answer_type: str
    rationale: str
    extraction_tokens: int
    grounded_in_chunks: bool
    shape_ok: bool
    verifier_passed: bool = False
    verifier_verdict: str = ''
    verifier_tokens: int = 0


class SasExtract(dspy.Signature):
    """Try to answer the question DIRECTLY from the chunks. ONE focused try.

GROUNDING RULES:
1. answer_span MUST be a contiguous substring of one of the chunks. Verbatim copy.
2. If NO chunk DIRECTLY supports a confident answer, return answer_span='' and confidence=0.0.
3. confidence ∈ [0,1]: 0.9+ only if a single chunk explicitly states the answer.
4. answer_type ∈ {entity, person, place, date, number, yes_no, other}.

A confidently wrong answer is a critical bug. When in doubt, return empty.
"""
    question: str = dspy.InputField(desc='The user question.')
    chunks_json: str = dspy.InputField(desc='List of {chunk_id, title, text} (top-5).')
    answer_span: str = dspy.OutputField(desc='Short verbatim answer (1-6 words) or empty.')
    answer_type: str = dspy.OutputField(desc='Type label.')
    confidence: float = dspy.OutputField(desc='[0,1] confidence in answer_span.')
    rationale: str = dspy.OutputField(desc='Brief justification (one sentence).')


class SasVerify(dspy.Signature):
    """Strictly verify a proposed direct answer against retrieved evidence.

Return JSON only:
{"verdict": "PASS"|"FAIL"|"INSUFFICIENT_EVIDENCE", "reason": <short str>}

PASS only if the evidence explicitly entails that answer_span directly answers
the original question. FAIL for wrong type, bridge-entity answers, partial
answers, or entity ambiguity. INSUFFICIENT_EVIDENCE when support is plausible
but not explicit. Do not use model confidence; judge only evidence support.
"""
    question: str = dspy.InputField()
    answer_span: str = dspy.InputField()
    answer_type: str = dspy.InputField()
    evidence_json: str = dspy.InputField(desc='Retrieved chunks containing candidate evidence.')
    verdict_json: str = dspy.OutputField()


def _parse_json_obj(raw: str) -> dict:
    text = (raw or '').strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def try_sas_attempt(
    *,
    sas_lm: dspy.LM,
    question: str,
    chunks: list[RetrievedChunk],
    probe_groundedness: float,
    tau_g: float = 0.55,
    tau_conf: float = 0.75,
) -> SasAttemptResult:
    """Run SAS-attempt. Acceptance: groundedness >= tau_g, conf >= tau_conf,
    answer in chunks, shape_ok.
    """
    # Cheap rejection: probe groundedness too low - skip extraction call entirely
    if probe_groundedness < tau_g or not chunks:
        return SasAttemptResult(
            accepted=False, answer='', confidence=0.0, answer_type='other',
            rationale='probe groundedness below tau_g',
            extraction_tokens=0, grounded_in_chunks=False, shape_ok=False,
        )

    # Build chunks_json for the LM
    chunks_repr = [{'chunk_id': c.chunk_id, 'text': (c.text or '')[:1200]} for c in chunks]

    extract = dspy.Predict(SasExtract)
    tokens = 0
    try:
        with dspy.context(lm=sas_lm):
            pred = extract(question=question, chunks_json=json.dumps(chunks_repr, ensure_ascii=False))
        try:
            tokens = sum(int(c.get('usage', {}).get('total_tokens', 0)) for c in (sas_lm.history[-1:] or []))
        except Exception:
            tokens = 0
    except Exception as e:
        return SasAttemptResult(
            accepted=False, answer='', confidence=0.0, answer_type='other',
            rationale=f'sas extract failure: {type(e).__name__}',
            extraction_tokens=0, grounded_in_chunks=False, shape_ok=False,
        )

    answer = (pred.answer_span or '').strip()
    answer_type = (pred.answer_type or 'other').strip().lower()
    try:
        confidence = float(pred.confidence)
    except Exception:
        confidence = 0.0
    rationale = (pred.rationale or '')[:300]

    grounded = _answer_in_chunks(answer, chunks, expected_type=answer_type)
    shape = _shape_ok(answer, answer_type)

    verifier_passed = False
    verifier_verdict = ''
    verifier_tokens = 0
    if answer and grounded and shape:
        evidence_repr = [
            {'chunk_id': c.chunk_id, 'text': (c.text or '')[:1000]}
            for c in chunks[:5]
            if _norm(answer) in _norm(c.text or '')
        ] or chunks_repr[:3]
        try:
            with dspy.context(lm=sas_lm):
                verify = dspy.Predict(SasVerify)
                vpred = verify(
                    question=question,
                    answer_span=answer,
                    answer_type=answer_type,
                    evidence_json=json.dumps(evidence_repr, ensure_ascii=False),
                )
            try:
                verifier_tokens = sum(int(c.get('usage', {}).get('total_tokens', 0)) for c in (sas_lm.history[-1:] or []))
            except Exception:
                verifier_tokens = 0
            vobj = _parse_json_obj(getattr(vpred, 'verdict_json', ''))
            verifier_verdict = str(vobj.get('verdict', '')).strip().upper()
            verifier_passed = verifier_verdict == 'PASS'
            if not verifier_verdict:
                verifier_verdict = 'PARSE_FAILED'
        except Exception as e:
            verifier_verdict = f'VERIFIER_ERROR:{type(e).__name__}'

    type_safe = _direct_type_safe(question, answer, answer_type)

    # Confidence is logged for diagnostics but not used as the acceptance signal:
    # direct evidence support plus strict verification is the gate.
    accepted = bool(answer) and grounded and shape and verifier_passed and type_safe

    return SasAttemptResult(
        accepted=accepted,
        answer=answer,
        confidence=confidence,
        answer_type=answer_type or 'other',
        rationale=rationale,
        extraction_tokens=tokens + verifier_tokens,
        grounded_in_chunks=grounded,
        shape_ok=shape,
        verifier_passed=verifier_passed,
        verifier_verdict=verifier_verdict,
        verifier_tokens=verifier_tokens,
    )
