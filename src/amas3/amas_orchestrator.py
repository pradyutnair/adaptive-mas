"""AMAS Orchestrator: one cheap LLM agent for the single-pass / SAS lane.

NOT to be confused with the GRPO orchestrator (pi_O in `grpo.topology`),
which is the topology sampler that decides whether to invoke this agent in
the first place. `AmasOrch` is the runtime executor agent; pi_O is the
learned routing policy.

One LLM call per iteration. The agent receives:
- original question
- top-k chunks from current retrieval (probe top-5 on first step, followups after)
- short search history (which queries already issued + their best evidence ids)
- remaining_followups budget

The agent emits strict JSON:
    {"action":"answer|retrieve|escalate", "answer":"...", "answer_type":"...",
     "justification":"...", "support_ids":[...], "confidence":0.0,
     "next_query":"..."}

`answer`     -> AmasOrch returns; pipeline emits the answer directly.
`retrieve`   -> AmasOrch issues `next_query`, appends new chunks, loops.
`escalate`   -> hands off to the full MAS pipeline (only honored when the
                pipeline config does NOT set sas_strict_single_pass).

AmasOrch is intentionally cheaper than full MAS: bounded retrieval
calls (1 + followups), no per-hop solver overhead, no synth call.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import dspy
import httpx

from .retriever import Retriever
from .types import RetrievedChunk


_PROMPT = """You are the AMAS Orchestrator (single-agent lane) for a multi-hop QA system. You see the original \
question and Wikipedia evidence chunks from the current retrieval. Decide one \
action per step.

Actions:
- "answer": evidence directly supports a confident answer to the ORIGINAL \
question (not a bridge entity). Only use answer when the wh-target of the \
original question is satisfied by an explicit span in the evidence.
- "retrieve": one more targeted retrieval will resolve the question. Emit \
`next_query` with a SPECIFIC follow-up query (a missing bridge fact, an entity \
attribute, a comparison side). Do NOT repeat prior queries. Use only if \
remaining_followups > 0.
- "escalate": needs decomposition: comparison across entities, set/count, \
multiple constraints, ambiguous bridge, or evidence too sparse for cheap path.

Rules:
- Never answer with a bridge entity. The answer must match the wh-target of \
the original question.
- Extract the required answer category from the original wh-phrase. For \
questions like "what rocket/company/source" or "which film/person", the \
answer must be that category, not a related entity mentioned in evidence.
- For questions containing a descriptor such as "the X that/which/who/where", \
evidence that only identifies X is not enough; answer only after evidence \
states the requested attribute of that X.
- `confidence` is your calibrated probability that the answer is exactly \
correct given the evidence. Be conservative: 0.85+ only if the evidence states \
the exact span verbatim.
- When `remaining_followups == 0` you MUST set action="answer" and provide \
your best-effort answer span (do NOT escalate, do NOT leave the answer field \
empty). If the evidence is weak, lower `confidence` accordingly but still \
commit to the most-supported span.
- Use action="escalate" only when remaining_followups > 0 AND the query \
clearly needs multi-agent decomposition (comparison across entities, \
set/count, multiple constraints, ambiguous bridge).

Return STRICT JSON on ONE LINE, no prose:
{"action":"answer|retrieve|escalate","answer":"short span or empty",\
"answer_type":"person|place|date|number|yes_no|entity|other",\
"justification":"short evidence relation","support_ids":["chunk_id"],\
"confidence":0.0,"next_query":"query or empty"}
"""


@dataclass
class AmasOrch:
    action: str = "escalate"
    answer: str = ""
    answer_type: str = "other"
    justification: str = ""
    support_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    chunks_used: list[RetrievedChunk] = field(default_factory=list)
    search_history: list[dict[str, Any]] = field(default_factory=list)
    retrieval_calls: int = 0
    tokens: int = 0


def _parse_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen, out = set(), []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out


def _chunks_payload(chunks: list[RetrievedChunk], max_chunks: int = 5, excerpt_chars: int = 320) -> list[dict[str, Any]]:
    return [
        {"chunk_id": c.chunk_id, "text": c.text[:excerpt_chars]}
        for c in chunks[:max_chunks]
    ]


def _clean_answer(a: str) -> str:
    s = (a or "").strip()
    low = s.lower()
    if low in {"n/a", "na", "none", "unknown", "not found", "not available", ""}:
        return ""
    if "not explicitly" in low or "further search" in low:
        return ""
    return s


async def _chat_json(
    lm: dspy.LM,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout: float = 30.0,
) -> tuple[dict[str, Any], int]:
    kwargs = getattr(lm, "kwargs", {}) or {}
    api_base = kwargs.get("api_base", "").rstrip("/")
    api_key = kwargs.get("api_key", "EMPTY")
    max_tokens = int(kwargs.get("max_tokens", 384))
    temperature = float(kwargs.get("temperature", 0.0))
    extra_body = kwargs.get("extra_body", {}) or {}
    model = str(getattr(lm, "model", "")).replace("hosted_vllm/", "")
    if not api_base or not model:
        return {}, 0
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(extra_body)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return {}, 0
    text = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
    usage = data.get("usage") or {}
    return _parse_json(text), int(usage.get("total_tokens", 0) or 0)


async def run_amas_orchestrator(
    *,
    sas_lm: dspy.LM,
    retriever: Retriever,
    question: str,
    probe_chunks: list[RetrievedChunk],
    max_followups: int = 2,
    min_answer_confidence: float = 0.65,
    chunk_excerpt_chars: int = 320,
    max_chunks_per_step: int = 5,
    experience: str = "",
) -> AmasOrch:
    """Run the SAS solver loop.

    Iteration 0 sees probe_chunks. If action='retrieve', we retrieve next_query
    and loop. At most `max_followups` retrievals beyond the probe. If the loop
    exits without 'answer', the final action is 'escalate'.
    """
    all_chunks = list(probe_chunks)
    history: list[dict[str, Any]] = []
    tokens_total = 0
    retrieval_calls = 1 if probe_chunks else 0
    last_result = AmasOrch(
        action="escalate",
        chunks_used=_dedupe(all_chunks),
        search_history=history,
        retrieval_calls=retrieval_calls,
    )
    best_answer: dict[str, Any] | None = None

    steps = max_followups + 1  # 1 initial step + N retrieve+answer steps
    for step_idx in range(steps):
        remaining = max_followups - step_idx
        chunks_now = _dedupe(all_chunks)
        user_payload = {
            "question": question,
            "remaining_followups": max(0, remaining),
            "search_history": history[-3:],
            "evidence": _chunks_payload(chunks_now, max_chunks=max_chunks_per_step, excerpt_chars=chunk_excerpt_chars),
        }
        system_prompt = _PROMPT
        if experience:
            system_prompt = "Prior experiential knowledge from past attempts:\n" + experience + "\n\n" + _PROMPT
        obj, tok = await _chat_json(sas_lm, system_prompt, user_payload)
        tokens_total += tok
        action = str(obj.get("action", "")).strip().lower()
        if action not in {"answer", "retrieve", "escalate"}:
            action = "escalate"
        answer = _clean_answer(str(obj.get("answer", "")))
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        justification = str(obj.get("justification", "")).strip()
        support_raw = obj.get("support_ids", [])
        if isinstance(support_raw, str):
            support_raw = _parse_json('{"x":' + support_raw + "}").get("x", [])
        support_ids = [str(x) for x in support_raw if x] if isinstance(support_raw, list) else []
        next_query = str(obj.get("next_query", "")).strip()
        history.append({
            "step": step_idx,
            "action": action,
            "answer": answer,
            "confidence": conf,
            "next_query": next_query,
            "support_ids": support_ids,
        })

        if action == "answer" and answer:
            if best_answer is None or conf > best_answer["confidence"]:
                best_answer = {
                    "answer": answer,
                    "answer_type": str(obj.get("answer_type", "other")),
                    "justification": justification,
                    "support_ids": support_ids,
                    "confidence": conf,
                    "chunks_used": chunks_now,
                }
            if conf >= min_answer_confidence:
                return AmasOrch(
                    action="answer",
                    answer=answer,
                    answer_type=str(obj.get("answer_type", "other")),
                    justification=justification,
                    support_ids=support_ids,
                    confidence=conf,
                    chunks_used=chunks_now,
                    search_history=history,
                    retrieval_calls=retrieval_calls,
                    tokens=tokens_total,
                )
        if action == "escalate":
            if answer and (best_answer is None or conf > best_answer["confidence"]):
                best_answer = {
                    "answer": answer,
                    "answer_type": str(obj.get("answer_type", "other")),
                    "justification": justification,
                    "support_ids": support_ids,
                    "confidence": conf,
                    "chunks_used": chunks_now,
                }
            last_result = AmasOrch(
                action="escalate",
                answer=answer,
                answer_type=str(obj.get("answer_type", "other")),
                justification=justification or "sas_escalate",
                support_ids=support_ids,
                confidence=conf,
                chunks_used=chunks_now,
                search_history=history,
                retrieval_calls=retrieval_calls,
                tokens=tokens_total,
            )
            return last_result
        # action == 'retrieve' (or invalid -> coerced earlier; but answer < min_conf falls through)
        if remaining <= 0 or not next_query:
            break
        previous = {h.get("next_query", "") for h in history[:-1]} | {question}
        if next_query in previous:
            break
        new_chunks = await retriever.retrieve(next_query)
        retrieval_calls += 1
        all_chunks.extend(new_chunks)
        last_result = AmasOrch(
            action="retrieve",
            answer=answer,
            answer_type=str(obj.get("answer_type", "other")),
            justification=justification,
            support_ids=support_ids,
            confidence=conf,
            chunks_used=_dedupe(all_chunks),
            search_history=history,
            retrieval_calls=retrieval_calls,
            tokens=tokens_total,
        )

    if best_answer is not None:
        return AmasOrch(
            action="answer",
            answer=best_answer["answer"],
            answer_type=best_answer["answer_type"],
            justification=best_answer["justification"],
            support_ids=best_answer["support_ids"],
            confidence=best_answer["confidence"],
            chunks_used=best_answer["chunks_used"] or _dedupe(all_chunks),
            search_history=history,
            retrieval_calls=retrieval_calls,
            tokens=tokens_total,
        )
    last_result.action = "escalate"
    last_result.tokens = tokens_total
    last_result.retrieval_calls = retrieval_calls
    last_result.chunks_used = _dedupe(all_chunks)
    last_result.search_history = history
    return last_result


_VERIFIER_PROMPT = """You verify whether a candidate answer is correct given the original question and the evidence excerpts the AMAS Orchestrator (single-agent lane) used. Be strict.

Accept ONLY if:
- the answer is a direct, verbatim or near-verbatim span supported by the evidence
- the answer matches the wh-target and explicit answer category of the original question (not a bridge entity)
- the relation between the answer and the question is explicit in the evidence
- evidence identifies the requested attribute of the described entity, not only the described entity itself

Otherwise escalate.

Return STRICT JSON, one line:
{"decision":"accept|escalate","confidence":0.0,"failure_reason":"short reason or empty"}
"""


@dataclass
class VerifierResult:
    decision: str = "escalate"
    confidence: float = 0.0
    failure_reason: str = ""
    tokens: int = 0


async def run_amas_verifier(
    *,
    verifier_lm: dspy.LM,
    question: str,
    answer: str,
    justification: str,
    chunks: list[RetrievedChunk],
    support_ids: list[str],
    excerpt_chars: int = 320,
    max_chunks: int = 5,
) -> VerifierResult:
    if not answer:
        return VerifierResult(decision="escalate", confidence=1.0, failure_reason="empty_answer", tokens=0)
    support_set = set(support_ids or [])
    # Prefer the chunks the SAS solver cited; otherwise top-k of what it saw
    cited = [c for c in chunks if c.chunk_id in support_set] if support_set else []
    extras = [c for c in chunks if c.chunk_id not in support_set]
    evidence = (cited + extras)[:max_chunks]
    payload = {
        "question": question,
        "candidate_answer": answer,
        "justification": justification[:300],
        "evidence": [{"chunk_id": c.chunk_id, "text": c.text[:excerpt_chars]} for c in evidence],
    }
    obj, tok = await _chat_json(verifier_lm, _VERIFIER_PROMPT, payload)
    decision = str(obj.get("decision", "escalate")).strip().lower()
    if decision not in ("accept", "escalate"):
        decision = "escalate"
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return VerifierResult(
        decision=decision,
        confidence=conf,
        failure_reason=str(obj.get("failure_reason", ""))[:240],
        tokens=tok,
    )
