"""ISA: Iterative Single Agent for multi-hop QA.

Unlike the one-shot SAS (which retrieves once and accepts/rejects), ISA does
iterative retrieve-reason-rewrite cycles on the FULL question. Each round
retrieves new evidence, adds it to a growing evidence pool, and attempts
extraction from the entire pool. Query rewriting is informed by what was
found so far and what is still missing.

This is the cheap lane in the adaptive pipeline. It handles questions that
are answerable with 1-3 retrieval rounds by a single agent, without needing
decomposition or multi-agent coordination.
"""
from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass, field
import dspy
from .retriever import Retriever
from .types import RetrievedChunk


class ISAExtract(dspy.Signature):
    """Extract a SHORT answer span (1-6 words) for the question from the
accumulated evidence pool.

GROUNDING RULES:
1. answer_span MUST be a contiguous substring copied VERBATIM from one of
   the evidence chunks. Do not paraphrase or invent.
2. If the evidence does NOT directly support a confident answer, return
   answer_span='' and confidence=0.0.
3. confidence in [0,1]: use 0.8+ only when a chunk explicitly and
   unambiguously states the answer. Use 0.4-0.7 for partial or indirect
   support. Use 0.0-0.3 when guessing.
4. missing_info: if you cannot answer confidently, describe what specific
   information is still needed (e.g. "need the birth year of X" or
   "need to know which city Y is located in"). This guides the next
   retrieval round.
5. For comparison questions, return the entity that satisfies the condition,
   not the compared value.

Output STRICT JSON:
{"answer_span": <str>, "evidence_chunk_id": <str>, "confidence": <float>,
 "answer_type": <str>, "missing_info": <str>}
"""
    question: str = dspy.InputField(desc='The full original question.')
    evidence_json: str = dspy.InputField(desc='Accumulated evidence chunks from all retrieval rounds.')
    extraction_json: str = dspy.OutputField()


class ISAFollowUp(dspy.Signature):
    """Given the question and what we found so far, formulate a follow-up
search query to retrieve the missing information.

Guidelines:
- Use named entities and key relation words from the question.
- Focus on the MISSING piece identified in the previous extraction.
- Do NOT repeat a previous query verbatim.
- Return a single short query string (no JSON).
"""
    question: str = dspy.InputField(desc='The full original question.')
    current_answer: str = dspy.InputField(desc='Best answer so far (may be empty).')
    missing_info: str = dspy.InputField(desc='What information is still needed.')
    previous_queries: str = dspy.InputField(desc='Newline-separated queries already tried.')
    followup_query: str = dspy.OutputField()


_NE_RE = re.compile(r'(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)|(?:\b\d{4}\b)')


def _norm_text(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]+', ' ', (s or '').lower()).strip()


def _answer_grounded(answer: str, chunks: list[RetrievedChunk]) -> bool:
    if not answer or not chunks:
        return False
    a = _norm_text(answer)
    if not a:
        return False
    for c in chunks:
        if a in _norm_text(c.text):
            return True
    return False


_YESNO_Q = re.compile(r'^(is|are|was|were|did|does|do|can|could|has|have|had|will|would|should)\b', re.I)
_YESNO_A = re.compile(r'^(yes|no|true|false)\b', re.I)
_WHEN_Q = re.compile(r'\b(when|what year|what date|what month)\b', re.I)
_WHERE_Q = re.compile(r'\b(where|which city|which country|which state|which county)\b', re.I)
_WHO_Q = re.compile(r'\b(who|whom)\b', re.I)
_HOWMANY_Q = re.compile(r'\b(how many|how much|number of|how old)\b', re.I)


def _type_aligned(question: str, answer: str) -> bool:
    """Check if the answer type roughly matches what the question asks for.
    Returns False for clear mismatches (e.g. entity answer for yes/no question).
    """
    q = (question or '').strip()
    a = (answer or '').strip()
    if not q or not a:
        return False
    if _YESNO_Q.match(q) and not _YESNO_A.match(a):
        if ' or ' not in q.lower() and 'which' not in q.lower():
            return False
    return True


def _parse_json(raw: str) -> dict:
    text = (raw or '').strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _format_evidence(chunks: list[RetrievedChunk], max_chars: int = 800) -> str:
    return json.dumps([
        {'chunk_id': c.chunk_id, 'text': c.text[:max_chars]}
        for c in chunks
    ], ensure_ascii=False)


@dataclass
class ISAResult:
    accepted: bool
    answer: str
    answer_type: str = 'other'
    confidence: float = 0.0
    rounds_used: int = 0
    evidence_count: int = 0
    extraction_tokens: int = 0
    rewrite_tokens: int = 0
    total_tokens: int = 0
    queries_issued: list[str] = field(default_factory=list)
    grounded: bool = False
    missing_info: str = ''


async def run_isa(
    *,
    isa_lm: dspy.LM,
    retriever: Retriever,
    question: str,
    initial_chunks: list[RetrievedChunk],
    max_rounds: int = 3,
    accept_threshold: float = 0.7,
) -> ISAResult:
    """Run the Iterative Single Agent.

    Accumulates evidence across rounds. Each round: extract from full pool,
    if not confident enough, formulate follow-up query, retrieve, repeat.
    """
    evidence_pool: list[RetrievedChunk] = list(initial_chunks)
    seen_ids: set[str] = {c.chunk_id for c in evidence_pool}
    queries_issued: list[str] = [question]
    extraction_tokens = 0
    rewrite_tokens = 0
    best_answer = ''
    best_confidence = 0.0
    best_type = 'other'
    best_evidence_id = ''
    best_missing = ''

    for round_idx in range(max_rounds + 1):
        if round_idx > 0:
            followup_query = await _get_followup(
                isa_lm, question, best_answer, best_missing,
                queries_issued,
            )
            rw_tokens = _last_tokens(isa_lm)
            rewrite_tokens += rw_tokens

            if followup_query in queries_issued:
                followup_query = followup_query + ' ' + best_type
            queries_issued.append(followup_query)

            new_chunks = await retriever.retrieve(followup_query)
            for c in new_chunks:
                if c.chunk_id not in seen_ids:
                    evidence_pool.append(c)
                    seen_ids.add(c.chunk_id)

        evidence_json = _format_evidence(evidence_pool)
        try:
            with dspy.context(lm=isa_lm):
                pred = await asyncio.to_thread(
                    dspy.Predict(ISAExtract),
                    question=question,
                    evidence_json=evidence_json,
                )
        except Exception:
            continue
        extraction_tokens += _last_tokens(isa_lm)

        obj = _parse_json(getattr(pred, 'extraction_json', ''))
        answer = str(obj.get('answer_span', '')).strip()
        try:
            confidence = max(0.0, min(1.0, float(obj.get('confidence', 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        answer_type = str(obj.get('answer_type', 'other')).strip()
        evidence_id = str(obj.get('evidence_chunk_id', '')).strip()
        missing = str(obj.get('missing_info', '')).strip()

        if answer and confidence > 0.5 and not _answer_grounded(answer, evidence_pool):
            confidence = min(confidence, 0.45)

        if confidence > best_confidence:
            best_answer = answer
            best_confidence = confidence
            best_type = answer_type
            best_evidence_id = evidence_id
            best_missing = missing

        grounded_now = _answer_grounded(best_answer, evidence_pool)
        type_ok = _type_aligned(question, best_answer)
        if best_confidence >= accept_threshold and grounded_now and type_ok:
            total = extraction_tokens + rewrite_tokens
            return ISAResult(
                accepted=True,
                answer=best_answer,
                answer_type=best_type,
                confidence=best_confidence,
                rounds_used=round_idx + 1,
                evidence_count=len(evidence_pool),
                extraction_tokens=extraction_tokens,
                rewrite_tokens=rewrite_tokens,
                total_tokens=total,
                queries_issued=queries_issued,
                grounded=True,
                missing_info='',
            )

        if not missing:
            best_missing = f'Cannot find direct answer to: {question}'

    grounded = _answer_grounded(best_answer, evidence_pool) if best_answer else False
    type_ok = _type_aligned(question, best_answer) if best_answer else False
    accepted = best_confidence >= accept_threshold and grounded and type_ok
    total = extraction_tokens + rewrite_tokens
    return ISAResult(
        accepted=accepted,
        answer=best_answer if accepted else best_answer,
        answer_type=best_type,
        confidence=best_confidence,
        rounds_used=max_rounds + 1,
        evidence_count=len(evidence_pool),
        extraction_tokens=extraction_tokens,
        rewrite_tokens=rewrite_tokens,
        total_tokens=total,
        queries_issued=queries_issued,
        grounded=grounded,
        missing_info=best_missing,
    )


async def _get_followup(
    lm: dspy.LM,
    question: str,
    current_answer: str,
    missing_info: str,
    previous_queries: list[str],
) -> str:
    with dspy.context(lm=lm):
        pred = await asyncio.to_thread(
            dspy.Predict(ISAFollowUp),
            question=question,
            current_answer=current_answer or '(none yet)',
            missing_info=missing_info or 'need more evidence',
            previous_queries='\n'.join(previous_queries),
        )
    return str(getattr(pred, 'followup_query', '')).strip() or question


def _last_tokens(lm: dspy.LM) -> int:
    try:
        history = lm.history[-1] if lm.history else None
        usage = (history or {}).get('usage') or {}
        return int(usage.get('total_tokens', 0))
    except Exception:
        return 0
