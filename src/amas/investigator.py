"""Investigator subagent: ReAct-style agent that retrieves its own evidence.

Critical context-isolation property: chunks NEVER appear in any prompt template.
They only enter the conversation as ``role: user`` messages whose body is the
JSON tool-result of the agent's own ``search`` action. The orchestrator only
ever sees the resulting :class:`EvidenceCapsule` (answer span + justification
+ confidence + support_ids), never raw passage text.

Why ReAct (not OpenAI tools): the loop is identical in semantics but works on
ANY OpenAI-compatible endpoint without server-side tool-calling support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .llm import LLMClient, parse_json_object, strip_thinking
from .retriever import RetrievalHit, Retriever
from .types import AnswerType, EvidenceCapsule, Fact

logger = logging.getLogger(__name__)


class Investigator:
    """ReAct-style subagent. One investigator per sub-question."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        top_k: int = 10,
        min_confidence: float = 0.3,
        max_searches: int = 3,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.top_k = int(top_k)
        self.min_confidence = float(min_confidence)
        self.max_searches = int(max_searches)
        self._user_template = (
            Path(__file__).parent / "prompts" / "investigate.txt"
        ).read_text(encoding="utf-8")
        self.last_searches_used = 0

    async def investigate(
        self,
        sub_question: str,
        retrieval_query: str,
        expected_answer_type: str,
        slot_name: str,
        top_k_override: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Run the agent loop. Returns ``(capsule, total_tokens_used)``."""
        del retrieval_query  # The agent decides its own query each turn.
        ans_type = AnswerType.coerce(expected_answer_type)
        prompt = self._user_template.format(
            sub_question=sub_question.strip(),
            expected_answer_type=ans_type.value,
            max_searches=self.max_searches,
        )
        messages: list[dict] = [{"role": "user", "content": prompt}]
        retrieved_ids: list[str] = []
        total_tokens = 0
        searches_used = 0
        empty_answer_text = ""

        # Cap turns to max_searches + 2 (final assistant turn).
        for _turn in range(self.max_searches + 2):
            resp = await self.llm.chat(messages=messages)
            total_tokens += resp.total_tokens
            content = strip_thinking(resp.content)
            empty_answer_text = content
            parsed = parse_json_object(content)
            action = str(parsed.get("action", "")).strip().lower()

            if action == "search" and searches_used < self.max_searches:
                # Append the assistant's literal action (so the conversation reflects it).
                messages.append({"role": "assistant", "content": content})
                hits, ids = await self._run_search(parsed, top_k_override)
                for cid in ids:
                    if cid not in retrieved_ids:
                        retrieved_ids.append(cid)
                # Tool result = role:user (private to this agent's history).
                messages.append({
                    "role": "user",
                    "content": json.dumps({"search_result": self._format_hits(hits)}),
                })
                searches_used += 1
                continue

            if action == "final":
                self.last_searches_used = searches_used
                capsule = self._build_capsule(parsed, retrieved_ids, slot_name, ans_type)
                return capsule, total_tokens

            # Bad / missing action: nudge the model once.
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "system_note": (
                        "Your previous response was not valid. Output exactly one "
                        "JSON object with action 'search' or 'final'. "
                        f"You have used {searches_used}/{self.max_searches} searches."
                    ),
                }),
            })

        # Loop exhausted. Try to salvage anything from the last attempt.
        self.last_searches_used = searches_used
        salvage = parse_json_object(empty_answer_text)
        if salvage.get("action") == "final":
            return self._build_capsule(salvage, retrieved_ids, slot_name, ans_type), total_tokens
        empty_fact = Fact(text="", confidence=0.0, slot_name=slot_name)
        return (
            EvidenceCapsule(
                answer="", fact=empty_fact,
                retrieved_doc_ids=retrieved_ids,
                retrieved_docs_total=len(retrieved_ids),
            ),
            total_tokens,
        )

    # ------------------------------------------------------------------
    # Search execution
    # ------------------------------------------------------------------

    async def _run_search(
        self, parsed: dict, top_k_override: int | None,
    ) -> tuple[list[RetrievalHit], list[str]]:
        query = str(parsed.get("query", "")).strip()
        if not query:
            return [], []
        requested_k = parsed.get("top_k") or top_k_override or self.top_k
        try:
            top_k = max(1, min(int(requested_k), 20))
        except (TypeError, ValueError):
            top_k = self.top_k
        hits = await self.retriever.retrieve(query, top_k=top_k)
        return hits, [h.chunk_id for h in hits]

    @staticmethod
    def _format_hits(hits: list[RetrievalHit]) -> list[dict]:
        return [
            {"chunk_id": h.chunk_id, "score": round(h.score, 4), "text": h.text}
            for h in hits
        ]

    # ------------------------------------------------------------------
    # Capsule construction
    # ------------------------------------------------------------------

    def _build_capsule(
        self, parsed: dict, retrieved_ids: list[str], slot_name: str,
        ans_type: AnswerType,
    ) -> EvidenceCapsule:
        answer = str(parsed.get("answer_span", "")).strip()
        justification = str(parsed.get("justification", "")).strip()
        support_ids_raw = parsed.get("support_ids") or []
        support_ids = [str(s).strip() for s in support_ids_raw if str(s).strip()]
        rid_set = set(retrieved_ids)
        support_ids = [s for s in support_ids if s in rid_set]
        confidence_self = self._bounded_float(parsed.get("confidence", 0.0))

        if not answer:
            confidence_self = 0.0
            support_ids = []
        type_ok = bool(answer) and ans_type.validate_span(answer)
        if not type_ok:
            confidence_self *= 0.5

        confidence_retrieval = 1.0 if support_ids else 0.0
        slot_filled = bool(answer and justification and support_ids and type_ok)
        confidence = (
            0.4 * confidence_retrieval
            + 0.4 * confidence_self
            + 0.2 * float(slot_filled)
        )
        if confidence < self.min_confidence:
            slot_filled = False

        fact = Fact(
            text=justification,
            answer_span=answer,
            confidence=confidence,
            confidence_self=confidence_self,
            confidence_retrieval=confidence_retrieval,
            slot_filled=slot_filled,
            slot_name=slot_name,
            support_ids=support_ids,
        )
        return EvidenceCapsule(
            answer=answer,
            fact=fact,
            retrieved_doc_ids=retrieved_ids,
            retrieved_docs_total=len(retrieved_ids),
        )

    @staticmethod
    def _bounded_float(v) -> float:
        try:
            return max(0.0, min(float(v), 1.0))
        except (TypeError, ValueError):
            return 0.0
