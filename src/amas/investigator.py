"""Investigator subagent: ReAct-style isolated agent.

Critical context-isolation property: chunks NEVER appear in the orchestrator's
view. They only enter THIS agent's private message history as the JSON payload
returned by ``search`` actions. The orchestrator only ever receives the
:class:`EvidenceCapsule` (answer_span + justification + confidence +
support_ids).
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
    """One isolated subagent per sub-question."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        top_k: int = 10,
        min_confidence: float = 0.3,
        max_searches: int = 3,
        max_answer_words: int = 8,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.top_k = int(top_k)
        self.min_confidence = float(min_confidence)
        self.max_searches = int(max_searches)
        self.max_answer_words = int(max_answer_words)
        self._user_template = (
            Path(__file__).parent / "prompts" / "investigate.txt"
        ).read_text(encoding="utf-8")
        self.last_searches_used = 0

    async def investigate(
        self,
        sub_question: str,
        expected_answer_type: str,
        hint: str = "",
        slot_name: str = "",
        top_k_override: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Run the agent loop. Returns ``(capsule, total_tokens_used)``."""
        ans_type = AnswerType.coerce(expected_answer_type)
        prompt = self._user_template.format(
            sub_question=sub_question.strip(),
            expected_answer_type=ans_type.value,
            max_searches=self.max_searches,
            hint=hint.strip() or "(none)",
        )
        messages: list[dict] = [{"role": "user", "content": prompt}]
        retrieved_ids: list[str] = []
        total_tokens = 0
        searches_used = 0
        last_content = ""

        for _turn in range(self.max_searches + 2):
            resp = await self.llm.chat(messages=messages)
            total_tokens += resp.total_tokens
            content = strip_thinking(resp.content)
            last_content = content
            parsed = parse_json_object(content)
            action = str(parsed.get("action", "")).strip().lower()

            if action == "search" and searches_used < self.max_searches:
                messages.append({"role": "assistant", "content": content})
                hits, ids = await self._run_search(parsed, top_k_override)
                for cid in ids:
                    if cid not in retrieved_ids:
                        retrieved_ids.append(cid)
                messages.append({
                    "role": "user",
                    "content": json.dumps({"search_result": [
                        {"chunk_id": h.chunk_id, "score": round(h.score, 4), "text": h.text}
                        for h in hits
                    ]}),
                })
                searches_used += 1
                continue

            if action == "final":
                self.last_searches_used = searches_used
                return self._build_capsule(parsed, retrieved_ids, slot_name, ans_type), total_tokens

            # Bad turn: nudge.
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "system_note": (
                        "Output exactly one JSON object with action 'search' or 'final'. "
                        f"You have used {searches_used}/{self.max_searches} searches."
                    ),
                }),
            })

        # Loop exhausted: try to salvage.
        self.last_searches_used = searches_used
        salvage = parse_json_object(last_content)
        if salvage.get("action") == "final":
            return self._build_capsule(salvage, retrieved_ids, slot_name, ans_type), total_tokens
        empty = Fact(text="", confidence=0.0, slot_name=slot_name)
        return (
            EvidenceCapsule(
                answer="", fact=empty,
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
        try:
            top_k = max(1, min(int(parsed.get("top_k") or top_k_override or self.top_k), 20))
        except (TypeError, ValueError):
            top_k = self.top_k
        hits = await self.retriever.retrieve(query, top_k=top_k)
        return hits, [h.chunk_id for h in hits]

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

        # Length enforcement: penalise enumerations / long answers.
        too_long = len(answer.split()) > self.max_answer_words
        if too_long:
            confidence_self *= 0.4

        if not answer:
            confidence_self = 0.0
            support_ids = []
        type_ok = bool(answer) and ans_type.validate_span(answer)
        if not type_ok:
            confidence_self *= 0.5

        confidence_retrieval = 1.0 if support_ids else 0.0
        slot_filled = bool(answer and justification and support_ids and type_ok and not too_long)
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
