from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import dspy

from .contracts import CitationCheck, HopFinding, RetrievedChunk, normalize_answer
from .retriever import Retriever

CONF_THRESHOLD = 0.65
MAX_ATTEMPTS = 3
MAX_SPAN_WORDS = 8

_DATE_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\b\d{1,2},\s*\d{4}\b|\b(?:1[6-9]\d{2}|20\d{2})\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b", re.IGNORECASE)
_YESNO_RE = re.compile(r"^(?:yes|no|true|false)\b", re.IGNORECASE)
VALID_ANSWER_TYPES = {"auto", "person", "place", "date", "number", "title", "organization", "yes_no", "entity"}


def _clean_answer_type(expected_answer_type: str | None) -> str:
    value = str(expected_answer_type or "auto").strip().lower().replace("-", "_")
    return value if value in VALID_ANSWER_TYPES else "auto"




def _expected_shape(question: str, expected_answer_type: str = "auto") -> str:
    explicit = _clean_answer_type(expected_answer_type)
    if explicit != "auto":
        return explicit
    q = question.lower()
    if any(x in q for x in ["when", "what date", "date", "year", "month"]):
        return "date"
    if any(x in q for x in ["how many", "how much", "rank", "number"]):
        return "number"
    if q.startswith("who"):
        return "person"
    if q.startswith("where"):
        return "place"
    return "entity"


def _shape_ok(answer: str, question: str, expected_answer_type: str = "auto") -> bool:
    shape = _expected_shape(question, expected_answer_type)
    if not answer.strip():
        return False
    if shape == "date":
        return bool(_DATE_RE.search(answer))
    if shape == "number":
        return bool(_NUMBER_RE.search(answer))
    if shape == "yes_no":
        return bool(_YESNO_RE.match(answer.strip()))
    return True


class ExtractAnswerSpan(dspy.Signature):
    """Extract the SHORTEST possible answer span (1-6 words) from the chunks.

    CRITICAL rules:
    1. answer_span must be 1-6 words. Never a sentence. Never include verbs or filler.
       GOOD: "Shaun Tan", "450", "October 8, 2009", "Pakistan"
       BAD: "written and illustrated by Shaun Tan", "more than 450 books", "northern Pakistan."
    2. Strip qualifiers: drop "more than", "approximately", "around", trailing periods.
       If the chunk says "more than 450 books", answer_span is "450".
    3. For person questions: just the name. For number: just the number. For date: just the date.
    4. Copy from one chunk. If no chunk answers the question, return empty answer and confidence 0.0.
    5. evidence_chunk_id must be one of the chunk_ids in chunks_json.
    6. confidence >= 0.7 only when the span directly and precisely answers the question.
    7. confidence < 0.5 when the extracted span is indirect or only partially relevant.

    Output strict JSON: {"answer_span": str, "evidence_chunk_id": str, "confidence": float}.
    """

    question: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField(desc="Allowed: auto, person, place, date, number, title, organization, yes_no, entity")
    chunks_json: str = dspy.InputField(desc="Top-k chunks: list of {chunk_id, text}")
    extraction_json: str = dspy.OutputField()


class ProposeQueryRewrite(dspy.Signature):
    """Rewrite the retrieval query when evidence is missing or weak.

    Use named entities and relation words from the question. Return one short query, not JSON.
    """

    question: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField(desc="Expected answer type for the current sub-question")
    previous_queries: str = dspy.InputField(desc="Newline-separated failed or weak queries")
    best_answer_so_far: str = dspy.InputField()
    rewritten_query: str = dspy.OutputField()


@dataclass
class ToolRuntime:
    findings: list[HopFinding] = field(default_factory=list)
    chunks_by_id: dict[str, RetrievedChunk] = field(default_factory=dict)
    total_hops: int = 0
    total_retries: int = 0
    tool_errors: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.findings.clear()
        self.chunks_by_id.clear()
        self.total_hops = 0
        self.total_retries = 0
        self.tool_errors.clear()


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _parse_jsonish(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _format_chunks(chunks: list[RetrievedChunk], excerpt_chars: int = 900) -> str:
    return json.dumps(
        [{"chunk_id": c.chunk_id, "text": c.text[:excerpt_chars]} for c in chunks],
        ensure_ascii=False,
    )


def _model_tokens(lm: dspy.LM, start_idx: int) -> int:
    total = 0
    for item in getattr(lm, "history", [])[start_idx:]:
        usage = (item or {}).get("usage") or {}
        try:
            total += int(usage.get("total_tokens", 0))
        except Exception:
            pass
    return total


_FILLER_PREFIX = re.compile(
    r"^(?:(?:more|less|fewer|approximately|about|around|roughly|over|under|nearly|at least|at most|up to)\s+(?:than\s+)?)+",
    re.IGNORECASE,
)
_TRAILING_JUNK = re.compile(r"[.,;:!?]+$")
_SENTENCE_LIKE = re.compile(r"\b(?:is|was|were|are|has|had|have|who|which|that|the .+ of)\b", re.IGNORECASE)


def _trim_answer(answer: str, expected_type: str) -> str:
    """Strip qualifiers, trailing punctuation, and filler from extracted spans."""
    s = answer.strip()
    s = _TRAILING_JUNK.sub("", s).strip()
    if expected_type in ("number", "date"):
        s = _FILLER_PREFIX.sub("", s).strip()
    if s.endswith(" books") or s.endswith(" people") or s.endswith(" personnel"):
        s = s.rsplit(" ", 1)[0].strip()
    return s


def _grounded(answer: str, evidence_id: str, chunks: list[RetrievedChunk]) -> bool:
    if not answer or not evidence_id:
        return False
    needle = normalize_answer(answer)
    if not needle:
        return False
    for c in chunks:
        if c.chunk_id == evidence_id and needle in normalize_answer(c.text):
            return True
    return False


async def _extract(sub_lm: dspy.LM, question: str, expected_answer_type: str, chunks: list[RetrievedChunk]) -> tuple[HopFinding, int]:
    start = len(getattr(sub_lm, "history", []))
    try:
        with dspy.context(lm=sub_lm):
            pred = await asyncio.to_thread(
                dspy.Predict(ExtractAnswerSpan),
                question=question,
                expected_answer_type=_clean_answer_type(expected_answer_type),
                chunks_json=_format_chunks(chunks),
            )
    except Exception:
        return HopFinding(confidence=0.0), 0

    tokens = _model_tokens(sub_lm, start)
    obj = _parse_jsonish(getattr(pred, "extraction_json", ""))
    raw_answer = str(obj.get("answer_span", "")).strip()
    ev = str(obj.get("evidence_chunk_id", "")).strip()
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    answer = _trim_answer(raw_answer, _clean_answer_type(expected_answer_type))

    word_count = len(answer.split())
    if answer and word_count > MAX_SPAN_WORDS:
        conf = min(conf, 0.45)
    elif answer and word_count > 5:
        conf = min(conf, 0.6)
    if answer and _SENTENCE_LIKE.search(answer):
        conf = min(conf, 0.4)
    if answer and not _shape_ok(answer, question, expected_answer_type):
        conf = min(conf, 0.35)
    if answer and conf > 0.4 and not _grounded(answer, ev, chunks):
        conf = min(conf, 0.4)
    return HopFinding(answer=answer, evidence_chunk_id=ev, confidence=conf, expected_answer_type=_clean_answer_type(expected_answer_type)), tokens


async def _rewrite(sub_lm: dspy.LM, question: str, expected_answer_type: str, queries: list[str], best: HopFinding) -> tuple[str, int]:
    start = len(getattr(sub_lm, "history", []))
    try:
        with dspy.context(lm=sub_lm):
            pred = await asyncio.to_thread(
                dspy.Predict(ProposeQueryRewrite),
                question=question,
                expected_answer_type=_clean_answer_type(expected_answer_type),
                previous_queries="\n".join(queries),
                best_answer_so_far=best.answer,
            )
        rewritten = str(getattr(pred, "rewritten_query", "")).strip()
    except Exception:
        rewritten = ""
    return rewritten or f"{question} answer", _model_tokens(sub_lm, start)


async def _hop_async(question: str, expected_answer_type: str, retriever: Retriever, sub_lm: dspy.LM, state: ToolRuntime) -> str:
    question = str(question or "").strip()
    expected_answer_type = _clean_answer_type(expected_answer_type)
    if not question:
        finding = HopFinding(confidence=0.0)
        state.findings.append(finding)
        return finding.to_json()

    queries: list[str] = []
    seen_chunk_sets: list[set[str]] = []
    best = HopFinding(confidence=0.0)
    for attempt in range(MAX_ATTEMPTS):
        query = question if attempt == 0 else (await _rewrite(sub_lm, question, expected_answer_type, queries, best))[0]
        if query in queries:
            query = f"{query} evidence"
        queries.append(query)
        chunks = await retriever.retrieve(query, k=5)
        chunk_ids = {c.chunk_id for c in chunks}
        for chunk in chunks:
            state.chunks_by_id[chunk.chunk_id] = chunk
        if seen_chunk_sets and chunk_ids == seen_chunk_sets[-1]:
            if attempt < MAX_ATTEMPTS - 1:
                continue
        seen_chunk_sets.append(chunk_ids)
        finding, _tokens = await _extract(sub_lm, question, expected_answer_type, chunks)
        finding.queries_used = list(queries)
        if finding.confidence > best.confidence:
            best = finding
        if finding.confidence >= CONF_THRESHOLD:
            break

    best.queries_used = list(queries)
    state.total_hops += 1
    state.total_retries += max(0, len(queries) - 1)
    state.findings.append(best)
    return best.to_json()


def _parse_questions_arg(questions: str) -> list[tuple[str, str]]:
    text = str(questions or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            out: list[tuple[str, str]] = []
            for item in obj:
                if isinstance(item, dict):
                    q = str(item.get("question", "")).strip()
                    t = _clean_answer_type(item.get("expected_answer_type", "auto"))
                else:
                    q = str(item).strip()
                    t = "auto"
                if q:
                    out.append((q, t))
            return out
    except Exception:
        pass
    return [(x.strip(" -\t"), "auto") for x in text.splitlines() if x.strip(" -\t")]


def _parse_support_ids(support_ids: str) -> list[str]:
    text = str(support_ids or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"[,\n]", text) if x.strip()]


def make_tools(retriever: Retriever, sub_lm: dspy.LM, state: ToolRuntime | None = None) -> list[dspy.Tool]:
    state = state or ToolRuntime()

    def hop(question: str, expected_answer_type: str = "auto") -> str:
        """Run one autonomous retrieval agent. expected_answer_type is optional: auto/person/place/date/number/title/organization/yes_no/entity."""
        try:
            return _run_coro_sync(_hop_async(question, expected_answer_type, retriever, sub_lm, state))
        except Exception as exc:
            state.tool_errors.append(f"hop error: {exc}")
            return HopFinding(confidence=0.0).to_json()

    def hop_batch(questions: str | list | Any = "") -> str:
        """Run hop() concurrently for a JSON list of sub-questions. Returns JSON list."""
        if not isinstance(questions, str):
            questions = json.dumps(questions, ensure_ascii=False)
        qs = _parse_questions_arg(questions)

        async def run_all() -> list[str]:
            return await asyncio.gather(*[_hop_async(q, t, retriever, sub_lm, state) for q, t in qs])

        try:
            return json.dumps([json.loads(x) for x in _run_coro_sync(run_all())], ensure_ascii=False)
        except Exception as exc:
            state.tool_errors.append(f"hop_batch error: {exc}")
            return "[]"

    def submit(answer: str, support_ids: str) -> str:
        """Citation gate. Call before finish(answer=...). Returns ACCEPTED or a rejection reason."""
        ids = _parse_support_ids(support_ids)
        if not ids:
            return CitationCheck(cited_ids=[], answer_grounded=False, reason="FAIL: provide at least one support_id").to_json()
        known = {f.evidence_chunk_id for f in state.findings if f.evidence_chunk_id}
        missing = [x for x in ids if x not in known]
        if missing:
            return CitationCheck(cited_ids=ids, answer_grounded=False, reason=f"FAIL: support_ids not returned by hop: {missing}").to_json()
        ans_norm = normalize_answer(answer)
        answer_seen = False
        for f in state.findings:
            if f.evidence_chunk_id not in ids:
                continue
            f_norm = normalize_answer(f.answer)
            if not ans_norm or not f_norm:
                continue
            if ans_norm in f_norm or f_norm in ans_norm:
                answer_seen = True
                break
            ans_tokens = set(ans_norm.split())
            f_tokens = set(f_norm.split())
            if ans_tokens and f_tokens and len(ans_tokens & f_tokens) / len(ans_tokens) >= 0.5:
                answer_seen = True
                break
            chunk = state.chunks_by_id.get(f.evidence_chunk_id)
            if chunk and ans_norm in normalize_answer(chunk.text):
                answer_seen = True
                break
        if not answer_seen:
            return CitationCheck(cited_ids=ids, answer_grounded=False, reason="FAIL: answer not grounded in cited hop findings or their source chunks").to_json()
        return "ACCEPTED"

    return [
        dspy.Tool(hop, name="hop", desc="Autonomous retrieval sub-agent. Optional expected_answer_type: auto/person/place/date/number/title/organization/yes_no/entity", arg_types={"question": str, "expected_answer_type": str}),
        dspy.Tool(hop_batch, name="hop_batch", desc="Parallel retrieval for JSON list of strings or objects {question, expected_answer_type}"),
        dspy.Tool(submit, name="submit", desc="Programmatic citation gate; call before finish", arg_types={"answer": str, "support_ids": str}),
    ]
