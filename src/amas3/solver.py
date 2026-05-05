"""Solver: per-node extraction worker with best-of-N enforced budget.

Performs max_retrievals extraction attempts with different query formulations.
Each attempt: retrieve -> extract -> (optionally verify). Selects best answer
across all attempts by verifier accept status + confidence.

v4: best-of-N enforced budget replaces retry-on-low-confidence.
"""
from __future__ import annotations
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
import dspy
from .retriever import Retriever
from .types import Finding, FindingStatus, RetrievedChunk, EvidenceCapsule

_DATE_RE = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2})\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand|million|billion)\b", re.IGNORECASE)
_YESNO_RE = re.compile(r"^(?:yes|no|true|false)\b", re.IGNORECASE)


def _shape_ok(answer: str, expected_answer_type: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    t = (expected_answer_type or "entity").lower()
    if t == "date":
        return bool(_DATE_RE.search(a))
    if t == "number":
        return bool(_NUMBER_RE.search(a))
    if t == "yes_no":
        return bool(_YESNO_RE.match(a))
    return True


_NE_RE = re.compile(r"(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)|(?:\b\d{4}\b)|(?:\b\d+(?:\.\d+)?\b)")


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _main_ne(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return ""
    matches = _NE_RE.findall(a)
    if not matches:
        return a
    return max(matches, key=len)


def _answer_grounded(answer: str, chunks: list) -> bool:
    if not answer or not chunks:
        return False
    a_norm = _norm_text(answer)
    ne = _norm_text(_main_ne(answer))
    if not a_norm and not ne:
        return False
    for c in chunks:
        text_n = _norm_text(getattr(c, "text", "") or "")
        title_n = _norm_text(getattr(c, "title", "") or "")
        if a_norm and (a_norm in text_n or a_norm in title_n):
            return True
        if ne and len(ne) >= 3 and (ne in text_n or ne in title_n):
            return True
    return False


class ExtractAnswerSpan(dspy.Signature):
    """Extract a SHORT answer span (1-6 words) for the sub-question.

GROUNDING RULES:
1. The answer_span MUST be copied VERBATIM as a contiguous substring of
   ONE of the provided chunks. Do not paraphrase, do not invent.
2. If NO chunk DIRECTLY supports the answer, return answer_span="" and
   confidence=0.0. A confidently wrong answer is a critical bug.
3. The answer must MATCH the expected_answer_type:
   - "date": year, month, or specific date span
   - "number": numeric or number-word
   - "place": place name appearing in the chunk
   - "person": full name as written in the chunk
   - "yes_no": yes/no
4. evidence_chunk_id must be a chunk_id present in chunks_json.
5. confidence reflects the directness of support; do not default to 0.9
   on weak evidence. Use 0.3 to 0.6 for partial support, 0.7 to 1.0 only
   when the chunk explicitly states the answer.

Output STRICT JSON: {"answer_span": <str>, "evidence_chunk_id": <str>, "confidence": <float>}.
If you cannot ground the answer, output {"answer_span": "", "evidence_chunk_id": "", "confidence": 0.0}.
"""
    sub_question: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField()
    chunks_json: str = dspy.InputField(desc="Top-k chunks: list of {chunk_id, text}")
    extraction_json: str = dspy.OutputField()


class ProposeQueryRewrite(dspy.Signature):
    """Propose a refined search query when previous retrieval returned no useful evidence.

Use named entities and key relation words from the sub-question. Add
disambiguating context if you can infer it. Return a single short query
string (no JSON).
"""
    sub_question: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField()
    previous_queries: str = dspy.InputField(desc="Newline-separated previous queries that did not work")
    rewritten_query: str = dspy.OutputField()


def _parse_extraction(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
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
        {"chunk_id": c.chunk_id, "text": c.text[:excerpt_chars]}
        for c in chunks
    ], ensure_ascii=False)


@dataclass
class ExtractionAttempt:
    """One extraction attempt within best-of-N."""
    finding: Finding
    chunks: list[RetrievedChunk]
    query: str
    verified: bool | None = None
    verify_reason: str = ""
    verify_tokens: int = 0


@dataclass
class SolverResult:
    finding: Finding
    chunks_used: list[RetrievedChunk]
    queries_issued: list[str]
    extraction_tokens: int
    rewrite_tokens: int
    verifier_tokens: int = 0
    verifier_calls: int = 0
    verifier_accepts: int = 0
    verifier_rejects: int = 0
    attempts: list[ExtractionAttempt] = field(default_factory=list)
    capsule: EvidenceCapsule | None = None


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
    use_verifier: bool = False,
    verifier_lm: dspy.LM | None = None,
    parent_ids: list[int] | None = None,
) -> SolverResult:
    t0 = time.time()
    rewrite_lm = rewrite_lm or solver_lm
    extract_sig = ExtractAnswerSpan
    rewrite_sig = ProposeQueryRewrite
    if experience:
        base_x = (extract_sig.instructions or extract_sig.__doc__ or "")
        extract_sig = extract_sig.with_instructions("Prior experiential knowledge from past attempts:\n" + experience + "\n\n" + base_x)
        base_r = (rewrite_sig.instructions or rewrite_sig.__doc__ or "")
        rewrite_sig = rewrite_sig.with_instructions("Prior experiential knowledge from past attempts:\n" + experience + "\n\n" + base_r)

    queries_issued: list[str] = []
    extraction_tokens = 0
    rewrite_tokens = 0
    verifier_tokens = 0
    verifier_calls = 0
    verifier_accepts = 0
    verifier_rejects = 0
    attempts: list[ExtractionAttempt] = []

    async def _extract(chunks: list[RetrievedChunk]) -> tuple[Finding, int]:
        chunks_json = _format_chunks(chunks)
        try:
            with dspy.context(lm=solver_lm):
                mod = dspy.Predict(extract_sig)
                pred = await asyncio.to_thread(
                    mod,
                    sub_question=sub_question,
                    expected_answer_type=expected_answer_type,
                    chunks_json=chunks_json,
                )
        except Exception:
            return Finding(
                sub_question=sub_question, answer="", evidence_ids=[],
                confidence=0.0, status=FindingStatus.NO_EVIDENCE,
                hop_idx=hop_idx, node_id=node_id,
            ), 0
        try:
            history = solver_lm.history[-1] if solver_lm.history else None
            usage = (history or {}).get("usage") or {}
            tokens = int(usage.get("total_tokens", 0))
        except Exception:
            tokens = 0
        obj = _parse_extraction(getattr(pred, "extraction_json", ""))
        answer = str(obj.get("answer_span", "")).strip()
        ev = str(obj.get("evidence_chunk_id", "")).strip()
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if answer and not _shape_ok(answer, expected_answer_type):
            conf = min(conf, 0.4)
        if answer and conf > 0.5 and not _answer_grounded(answer, chunks):
            conf = min(conf, 0.45)
        if answer and conf >= min_confidence:
            status = FindingStatus.OK
        elif answer:
            status = FindingStatus.LOW_CONFIDENCE
        else:
            status = FindingStatus.NO_EVIDENCE
        f = Finding(
            sub_question=sub_question, answer=answer,
            evidence_ids=[ev] if ev else [], confidence=conf,
            status=status, hop_idx=hop_idx, node_id=node_id, tokens=tokens,
        )
        return f, tokens

    # Best-of-N: always perform max_retrievals attempts
    # Attempt 0: use starting_chunks from probe (if available)
    # Attempts 1..N-1: rewrite query + fresh retrieval
    total_attempts = max(1, max_retrievals)

    for attempt_idx in range(total_attempts):
        if attempt_idx == 0 and starting_chunks:
            current_chunks = list(starting_chunks)
            queries_issued.append("(probe_layer)")
        else:
            if attempt_idx == 0:
                fresh_query = sub_question
            else:
                with dspy.context(lm=rewrite_lm):
                    rmod = dspy.Predict(rewrite_sig)
                    rpred = await asyncio.to_thread(
                        rmod,
                        sub_question=sub_question,
                        expected_answer_type=expected_answer_type,
                        previous_queries="\n".join(queries_issued),
                    )
                try:
                    history = rewrite_lm.history[-1] if rewrite_lm.history else None
                    usage = (history or {}).get("usage") or {}
                    rewrite_tokens += int(usage.get("total_tokens", 0))
                except Exception:
                    pass
                fresh_query = str(getattr(rpred, "rewritten_query", "")).strip() or sub_question

            if fresh_query in queries_issued:
                fresh_query = fresh_query + " " + expected_answer_type
            queries_issued.append(fresh_query)
            current_chunks = await retriever.retrieve(fresh_query)
            if not current_chunks:
                continue

        f, tk = await _extract(current_chunks)
        extraction_tokens += tk
        f.rewrites_used = attempt_idx

        att = ExtractionAttempt(finding=f, chunks=current_chunks, query=queries_issued[-1])

        # Optional per-hop verification
        if use_verifier and verifier_lm and f.answer:
            from .verifier import verify_extraction
            vr = verify_extraction(
                verifier_lm=verifier_lm,
                sub_question=sub_question,
                extracted_answer=f.answer,
                expected_answer_type=expected_answer_type,
                evidence_chunks=current_chunks,
            )
            att.verified = vr.accept
            att.verify_reason = vr.reason
            att.verify_tokens = vr.tokens
            verifier_tokens += vr.tokens
            verifier_calls += 1
            if vr.accept:
                verifier_accepts += 1
            else:
                verifier_rejects += 1

        attempts.append(att)

    # Select best attempt: prefer verified-accepted, then highest confidence
    best: ExtractionAttempt | None = None
    for att in attempts:
        if not att.finding.answer:
            continue
        if best is None:
            best = att
            continue
        # Verified-accepted beats non-verified or rejected
        if att.verified is True and best.verified is not True:
            best = att
        elif att.verified == best.verified and att.finding.confidence > best.finding.confidence:
            best = att

    if best is None:
        best_finding = Finding(
            sub_question=sub_question, answer="", evidence_ids=[],
            confidence=0.0, status=FindingStatus.NO_EVIDENCE,
            hop_idx=hop_idx, node_id=node_id,
        )
        best_chunks: list[RetrievedChunk] = []
    else:
        best_finding = best.finding
        best_chunks = best.chunks

    # Build EvidenceCapsule for working memory
    capsule = EvidenceCapsule(
        node_id=node_id,
        sub_question=sub_question,
        answer=best_finding.answer,
        confidence=best_finding.confidence,
        status=best_finding.status.value,
        evidence_ids=best_finding.evidence_ids,
        evidence_excerpts=[c.text[:200] for c in best_chunks[:3]],
        query_rewrites=queries_issued,
        verification={"accept": best.verified, "reason": best.verify_reason} if best and best.verified is not None else None,
        parent_ids=parent_ids or [],
        retrievals_used=len(attempts),
        retrievals_budget=total_attempts,
        latency_seconds=round(time.time() - t0, 3),
    )

    return SolverResult(
        finding=best_finding,
        chunks_used=best_chunks,
        queries_issued=queries_issued,
        extraction_tokens=extraction_tokens,
        rewrite_tokens=rewrite_tokens,
        verifier_tokens=verifier_tokens,
        verifier_calls=verifier_calls,
        verifier_accepts=verifier_accepts,
        verifier_rejects=verifier_rejects,
        attempts=attempts,
        capsule=capsule,
    )
