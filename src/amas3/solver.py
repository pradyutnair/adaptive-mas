"""Solver: per-node extraction worker (GPT-4o-mini by default).

Receives:
- sub_question (parent answers already interpolated by FindingsBus)
- starting_chunks: chunks from the probe layer (may be empty)
- expected_answer_type: for extraction guidance
- max_retrievals: total retrieval budget (default 3)

Returns a Finding pushed by the caller into the FindingsBus.

Each retrieval is top-k=5. Re-retrieval = a new query (refined) at top-k=5.

v3: keep strengthened prompt and minimal post-extraction sanity check, but
trust the LLM's confidence (do not aggressively downgrade on substring
mismatch since paraphrasing is common in extracted spans).
"""
from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any
import dspy
from .retriever import Retriever
from .types import Finding, FindingStatus, RetrievedChunk

_DATE_RE = re.compile(r'\b(?:1[6-9]\d{2}|20\d{2})\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', re.IGNORECASE)
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand|million|billion)\b', re.IGNORECASE)
_YESNO_RE = re.compile(r'^(?:yes|no|true|false)\b', re.IGNORECASE)


def _shape_ok(answer: str, expected_answer_type: str) -> bool:
    a = (answer or '').strip()
    if not a:
        return False
    t = (expected_answer_type or 'entity').lower()
    if t == 'date':
        return bool(_DATE_RE.search(a))
    if t == 'number':
        return bool(_NUMBER_RE.search(a))
    if t == 'yes_no':
        return bool(_YESNO_RE.match(a))
    return True


class ExtractAnswerSpan(dspy.Signature):
    """Extract a SHORT answer span (1-6 words) for the sub-question.

GROUNDING RULES:
1. The answer_span MUST be copied VERBATIM as a contiguous substring of
   ONE of the provided chunks. Do not paraphrase, do not invent.
2. If NO chunk DIRECTLY supports the answer, return answer_span='' and
   confidence=0.0. A confidently wrong answer is a critical bug.
3. The answer must MATCH the expected_answer_type:
   - 'date': year, month, or specific date span
   - 'number': numeric or number-word
   - 'place': place name appearing in the chunk
   - 'person': full name as written in the chunk
   - 'yes_no': yes/no
4. evidence_chunk_id must be a chunk_id present in chunks_json.
5. confidence reflects the directness of support; do not default to 0.9
   on weak evidence. Use 0.3 to 0.6 for partial support, 0.7 to 1.0 only
   when the chunk explicitly states the answer.

Output STRICT JSON: {"answer_span": <str>, "evidence_chunk_id": <str>, "confidence": <float>}.
If you cannot ground the answer, output {"answer_span": "", "evidence_chunk_id": "", "confidence": 0.0}.
"""
    sub_question: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField()
    chunks_json: str = dspy.InputField(desc='Top-k chunks: list of {chunk_id, text}')
    extraction_json: str = dspy.OutputField()


class ProposeQueryRewrite(dspy.Signature):
    """Propose a refined search query when previous retrieval returned no useful evidence.

Use named entities and key relation words from the sub-question. Add
disambiguating context if you can infer it. Return a single short query
string (no JSON).
"""
    sub_question: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField()
    previous_queries: str = dspy.InputField(desc='Newline-separated previous queries that did not work')
    rewritten_query: str = dspy.OutputField()


def _parse_extraction(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return {}
        return obj
    except Exception:
        return {}


def _format_chunks(chunks: list[RetrievedChunk], excerpt_chars: int = 700) -> str:
    return json.dumps([
        {'chunk_id': c.chunk_id, 'text': c.text[:excerpt_chars]}
        for c in chunks
    ], ensure_ascii=False)


@dataclass
class SolverResult:
    finding: Finding
    chunks_used: list[RetrievedChunk]
    queries_issued: list[str]
    extraction_tokens: int
    rewrite_tokens: int


async def run_solver(
    *,
    solver_lm: dspy.LM,
    rewrite_lm: dspy.LM | None,
    retriever: Retriever,
    sub_question: str,
    expected_answer_type: str,
    starting_chunks: list[RetrievedChunk] | None,
    node_id: int,
    hop_idx: int,
    max_retrievals: int = 3,
    min_confidence: float = 0.3,
    experience: str = "",
) -> SolverResult:
    rewrite_lm = rewrite_lm or solver_lm
    extract_sig = ExtractAnswerSpan
    rewrite_sig = ProposeQueryRewrite
    if experience:
        base_x = (extract_sig.instructions or extract_sig.__doc__ or "")
        extract_sig = extract_sig.with_instructions("Prior experiential knowledge from past attempts:\n" + experience + "\n\n" + base_x)
        base_r = (rewrite_sig.instructions or rewrite_sig.__doc__ or "")
        rewrite_sig = rewrite_sig.with_instructions("Prior experiential knowledge from past attempts:\n" + experience + "\n\n" + base_r)
    queries_issued: list[str] = []
    chunks_used: list[RetrievedChunk] = []
    extraction_tokens = 0
    rewrite_tokens = 0
    best_finding: Finding | None = None

    async def _extract(chunks: list[RetrievedChunk]) -> tuple[Finding, int]:
        chunks_json = _format_chunks(chunks)
        with dspy.context(lm=solver_lm):
            mod = dspy.Predict(extract_sig)
            pred = await asyncio.to_thread(
                mod,
                sub_question=sub_question,
                expected_answer_type=expected_answer_type,
                chunks_json=chunks_json,
            )
        try:
            history = solver_lm.history[-1] if solver_lm.history else None
            usage = (history or {}).get('usage') or {}
            tokens = int(usage.get('total_tokens', 0))
        except Exception:
            tokens = 0
        obj = _parse_extraction(getattr(pred, 'extraction_json', ''))
        answer = str(obj.get('answer_span', '')).strip()
        ev = str(obj.get('evidence_chunk_id', '')).strip()
        try:
            conf = float(obj.get('confidence', 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if answer and not _shape_ok(answer, expected_answer_type):
            conf = min(conf, 0.4)
        if answer and conf >= min_confidence:
            status = FindingStatus.OK
        elif answer:
            status = FindingStatus.LOW_CONFIDENCE
        else:
            status = FindingStatus.NO_EVIDENCE
        f = Finding(
            sub_question=sub_question,
            answer=answer,
            evidence_ids=[ev] if ev else [],
            confidence=conf,
            status=status,
            hop_idx=hop_idx,
            node_id=node_id,
            tokens=tokens,
        )
        return f, tokens

    if starting_chunks:
        chunks_used = list(starting_chunks)
        queries_issued.append('(probe_layer)')
        f, tk = await _extract(chunks_used)
        extraction_tokens += tk
        best_finding = f
        if f.status == FindingStatus.OK:
            f.rewrites_used = 0
            return SolverResult(f, chunks_used, queries_issued, extraction_tokens, rewrite_tokens)

    fresh_query = sub_question
    for attempt in range(max_retrievals):
        if attempt > 0:
            with dspy.context(lm=rewrite_lm):
                rmod = dspy.Predict(rewrite_sig)
                rpred = await asyncio.to_thread(
                    rmod,
                    sub_question=sub_question,
                    expected_answer_type=expected_answer_type,
                    previous_queries='\n'.join(queries_issued),
                )
            try:
                history = rewrite_lm.history[-1] if rewrite_lm.history else None
                usage = (history or {}).get('usage') or {}
                rewrite_tokens += int(usage.get('total_tokens', 0))
            except Exception:
                pass
            fresh_query = str(getattr(rpred, 'rewritten_query', '')).strip() or sub_question
        if fresh_query in queries_issued:
            fresh_query = fresh_query + ' ' + expected_answer_type
        queries_issued.append(fresh_query)
        new_chunks = await retriever.retrieve(fresh_query)
        if new_chunks:
            chunks_used = new_chunks
            f, tk = await _extract(chunks_used)
            extraction_tokens += tk
            f.rewrites_used = attempt
            if f.status == FindingStatus.OK:
                return SolverResult(f, chunks_used, queries_issued, extraction_tokens, rewrite_tokens)
            if best_finding is None or f.confidence > best_finding.confidence:
                best_finding = f

    if best_finding is None:
        best_finding = Finding(
            sub_question=sub_question,
            answer='',
            evidence_ids=[],
            confidence=0.0,
            status=FindingStatus.NO_EVIDENCE,
            hop_idx=hop_idx,
            node_id=node_id,
        )
    best_finding.rewrites_used = max(0, len(queries_issued) - 1)
    return SolverResult(best_finding, chunks_used, queries_issued, extraction_tokens, rewrite_tokens)
