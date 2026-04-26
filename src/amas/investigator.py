"""Context-isolated investigator with a private search loop.

Raw chunks are never accepted as caller input and never leave this class. They
only appear as private search-tool results inside the investigator's own message
history before it returns a distilled evidence capsule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import tiktoken

from .llm import LLMClient, parse_json_object, strip_thinking
from .retriever import RetrievalHit, Retriever
from .types import AnswerType, EvidenceCapsule, Fact, SubgoalNode

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

logger = logging.getLogger(__name__)


class Investigator:
    """Resolve one subgoal with bounded test-time scaling."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        top_k: int = 10,
        min_confidence: float = 0.3,
        max_searches: int = 2,
        max_answer_words: int = 8,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.top_k = int(top_k)
        self.min_confidence = float(min_confidence)
        self.max_searches = max(1, int(max_searches))
        self.max_answer_words = int(max_answer_words)
        prompt_dir = Path(__file__).parent / "prompts"
        self._template = (prompt_dir / "investigate.txt").read_text(encoding="utf-8")
        self.last_searches_used = 0

    async def investigate(
        self,
        sub_question: str,
        expected_answer_type: str,
        hint: str = "",
        slot_name: str = "",
        top_k_override: int | None = None,
        subgoal_id: int = 0,
    ) -> tuple[EvidenceCapsule, int]:
        node = SubgoalNode(
            id=subgoal_id,
            question=sub_question,
            answer_type=AnswerType.coerce(expected_answer_type),
        )
        return await self.investigate_node(
            node=node,
            hint=hint,
            slot_name=slot_name,
            top_k_override=top_k_override,
        )

    async def investigate_node(
        self,
        node: SubgoalNode,
        hint: str = "",
        slot_name: str = "",
        top_k_override: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Let the investigator choose private searches, then emit a capsule."""
        prompt = self._template.format(
            sub_question=node.question.strip(),
            expected_answer_type=node.answer_type.value,
            max_searches=self.max_searches,
            hint=hint.strip() or "(none)",
        )
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        total_tokens = 0
        chunk_tokens = 0
        searches_used = 0
        retrieved_ids: list[str] = []
        last_parsed: dict[str, Any] = {}

        for _turn in range(self.max_searches + 2):
            chunk_tokens += self._chunk_tokens_in_messages(messages)
            resp = await self.llm.chat(messages=messages)
            total_tokens += resp.total_tokens
            content = strip_thinking(resp.content)
            parsed = parse_json_object(content)
            last_parsed = parsed
            action = str(parsed.get("action", "")).strip().lower()

            if action == "search" and searches_used < self.max_searches:
                messages.append({"role": "assistant", "content": content})
                hits = await self._run_search(parsed, top_k_override)
                for hit in hits:
                    if hit.chunk_id not in retrieved_ids:
                        retrieved_ids.append(hit.chunk_id)
                messages.append({
                    "role": "user",
                    "content": json.dumps({
                        "search_result": [
                            {
                                "chunk_id": h.chunk_id,
                                "score": round(h.score, 4),
                                "text": h.text,
                            }
                            for h in hits
                        ]
                    }, ensure_ascii=False),
                })
                searches_used += 1
                continue

            if action == "final":
                self.last_searches_used = searches_used
                capsule = self._build_capsule(parsed, node, retrieved_ids, slot_name)
                capsule.chunk_tokens = chunk_tokens
                return capsule, total_tokens

            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "system_note": (
                        "Output exactly one JSON object with action 'search' or 'final'. "
                        f"You have used {searches_used}/{self.max_searches} searches."
                    )
                }),
            })

        self.last_searches_used = searches_used
        capsule = self._build_capsule(last_parsed, node, retrieved_ids, slot_name)
        capsule.chunk_tokens = chunk_tokens
        return capsule, total_tokens

    async def _run_search(
        self,
        parsed: dict[str, Any],
        top_k_override: int | None,
    ) -> list[RetrievalHit]:
        query = str(parsed.get("query", "")).strip()
        if not query:
            return []
        try:
            top_k = max(1, min(int(parsed.get("top_k") or top_k_override or self.top_k), 20))
        except (TypeError, ValueError):
            top_k = self.top_k
        return await self.retriever.retrieve(query, top_k=top_k)

    def _build_capsule(
        self,
        parsed: dict[str, Any],
        node: SubgoalNode,
        retrieved_ids: list[str],
        slot_name: str,
    ) -> EvidenceCapsule:
        answer = str(parsed.get("answer_span", "")).strip()
        justification = str(parsed.get("justification", "")).strip()
        support_ids_raw = parsed.get("support_ids") or []
        support_ids = [str(s).strip() for s in support_ids_raw if str(s).strip()]
        support_ids = [s for s in support_ids if s in set(retrieved_ids)]
        confidence_self = self._bounded_float(parsed.get("confidence", 0.0))

        too_long = len(answer.split()) > self.max_answer_words
        if too_long:
            confidence_self *= 0.4

        type_ok = bool(answer) and node.answer_type.validate_span(answer)
        if not type_ok:
            confidence_self *= 0.5

        if not answer:
            confidence_self = 0.0
            support_ids = []

        confidence_retrieval = 1.0 if support_ids else 0.0
        slot_filled = bool(answer and justification and support_ids and type_ok and not too_long)
        confidence = (
            0.4 * confidence_retrieval
            + 0.4 * confidence_self
            + 0.2 * float(slot_filled)
        )
        if confidence < self.min_confidence:
            slot_filled = False

        fact_text = justification if slot_filled else ""
        fact = Fact(
            text=fact_text,
            confidence=confidence if slot_filled else 0.0,
            confidence_self=confidence_self,
            confidence_retrieval=confidence_retrieval,
            slot_filled=slot_filled,
            slot_name=slot_name,
            answer_span=answer if slot_filled else "",
            support_ids=support_ids if slot_filled else [],
        )
        return EvidenceCapsule(
            answer=answer if slot_filled else "",
            fact=fact,
            subgoal_id=node.id,
            sub_question=node.question,
            answer_type=node.answer_type,
            retrieved_doc_ids=retrieved_ids,
            retrieved_docs_total=len(retrieved_ids),
        )

    @staticmethod
    def _merge_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        by_id: dict[str, RetrievalHit] = {}
        for hit in hits:
            existing = by_id.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                by_id[hit.chunk_id] = hit
        return sorted(by_id.values(), key=lambda h: h.score, reverse=True)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(_TOKENIZER.encode(text or ""))

    @classmethod
    def _chunk_tokens_in_messages(cls, messages: list[dict[str, str]]) -> int:
        total = 0
        for message in messages:
            content = message.get("content", "")
            if '"search_result"' in content:
                total += cls._estimate_tokens(content)
        return total

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0
