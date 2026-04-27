"""Deterministic investigator with bounded stateless retrieval rounds."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import tiktoken

from .llm import LLMClient, parse_json_object
from .retriever import RetrievalHit, Retriever
from .types import AnswerType, EvidenceCapsule, Fact, SubgoalNode

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
logger = logging.getLogger(__name__)


class Investigator:
    """Resolve one subgoal with a deterministic 1-2 round pipeline."""

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
        self._analyze_template = (prompt_dir / "analyze.txt").read_text(encoding="utf-8")
        self._rewrite_template = (prompt_dir / "rewrite.txt").read_text(encoding="utf-8")
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
        total_tokens = 0
        chunk_tokens = 0
        top_k = int(top_k_override or self.top_k)

        query_1 = node.question.strip()
        hits_1 = await self.retriever.retrieve(query_1, top_k=top_k)
        retrieved_ids = [hit.chunk_id for hit in hits_1]

        analyze_1, tokens_1, used_chunk_tokens_1 = await self._analyze(node, hint, hits_1)
        total_tokens += tokens_1
        chunk_tokens += used_chunk_tokens_1
        capsule_1, rejection_1 = self._build_capsule(
            analyze_1,
            node,
            retrieved_ids,
            slot_name,
        )
        if capsule_1.fact.slot_filled or self.max_searches <= 1:
            self.last_searches_used = 1 if hits_1 else 0
            capsule_1.chunk_tokens = chunk_tokens
            return capsule_1, total_tokens

        rewrite_obj, rewrite_tokens, rewrite_chunk_tokens = await self._rewrite(
            node=node,
            hint=hint,
            previous_query=query_1,
            previous_answer=str(analyze_1.get("answer_span", "")).strip(),
            previous_justification=str(analyze_1.get("justification", "")).strip(),
            hits=hits_1,
            rejection_reason=rejection_1,
        )
        total_tokens += rewrite_tokens
        chunk_tokens += rewrite_chunk_tokens

        query_2 = str(rewrite_obj.get("query", "")).strip()
        if not query_2 or query_2 == query_1:
            query_2 = query_1

        hits_2 = await self.retriever.retrieve(query_2, top_k=top_k)
        merged_hits = self._merge_hits([*hits_1, *hits_2])
        for hit in hits_2:
            if hit.chunk_id not in retrieved_ids:
                retrieved_ids.append(hit.chunk_id)

        analyze_2, tokens_2, used_chunk_tokens_2 = await self._analyze(node, hint, merged_hits)
        total_tokens += tokens_2
        chunk_tokens += used_chunk_tokens_2
        capsule_2, _ = self._build_capsule(
            analyze_2,
            node,
            retrieved_ids,
            slot_name,
        )
        self.last_searches_used = 2 if hits_2 else 1
        capsule_2.chunk_tokens = chunk_tokens
        return capsule_2, total_tokens

    async def _analyze(
        self,
        node: SubgoalNode,
        hint: str,
        hits: list[RetrievalHit],
    ) -> tuple[dict[str, Any], int, int]:
        chunk_blob = self._format_chunks(hits)
        prompt = self._analyze_template.format(
            sub_question=node.question.strip(),
            expected_answer_type=node.answer_type.value,
            hint=hint.strip() or "(none)",
            chunks=chunk_blob,
        )
        resp = await self.llm.chat(messages=[{"role": "user", "content": prompt}])
        return parse_json_object(resp.content), resp.total_tokens, self._estimate_tokens(chunk_blob)

    async def _rewrite(
        self,
        node: SubgoalNode,
        hint: str,
        previous_query: str,
        previous_answer: str,
        previous_justification: str,
        hits: list[RetrievalHit],
        rejection_reason: str,
    ) -> tuple[dict[str, Any], int, int]:
        chunk_blob = self._format_chunks(hits)
        prompt = self._rewrite_template.format(
            sub_question=node.question.strip(),
            expected_answer_type=node.answer_type.value,
            hint=(hint.strip() or "(none)") + f" Rejection reason: {rejection_reason}.",
            previous_query=previous_query.strip(),
            previous_answer=previous_answer,
            previous_justification=previous_justification,
            chunks=chunk_blob,
        )
        resp = await self.llm.chat(messages=[{"role": "user", "content": prompt}])
        parsed = parse_json_object(resp.content)
        if "query" not in parsed:
            parsed = {"query": previous_query}
        return parsed, resp.total_tokens, self._estimate_tokens(chunk_blob)

    def _build_capsule(
        self,
        parsed: dict[str, Any],
        node: SubgoalNode,
        retrieved_ids: list[str],
        slot_name: str,
    ) -> tuple[EvidenceCapsule, str]:
        status = str(parsed.get("status", "")).strip().lower()
        answer = self._normalize_answer(node.question, str(parsed.get("answer_span", "")).strip())
        justification = str(parsed.get("justification", "")).strip()
        failure_reason = str(parsed.get("failure_reason", "")).strip()
        support_ids_raw = parsed.get("support_ids") or []
        support_ids = [str(s).strip() for s in support_ids_raw if str(s).strip()]
        support_ids = [s for s in support_ids if s in set(retrieved_ids)]
        confidence_self = self._bounded_float(parsed.get("confidence", 0.0))

        explicitly_sufficient = status == "sufficient"
        explicitly_insufficient = status == "insufficient"

        too_long = len(answer.split()) > self.max_answer_words
        if too_long:
            confidence_self *= 0.4

        type_ok = bool(answer) and node.answer_type.validate_span(answer)
        if not type_ok:
            confidence_self *= 0.5

        if not answer:
            confidence_self = 0.0
            support_ids = []

        if explicitly_insufficient:
            answer = ""
            justification = ""
            support_ids = []
            confidence_self = 0.0

        confidence_retrieval = 1.0 if support_ids else 0.0
        slot_filled = bool(
            answer and justification and support_ids and type_ok and not too_long
        )
        if not explicitly_sufficient:
            slot_filled = False
        confidence = 0.4 * confidence_retrieval + 0.4 * confidence_self + 0.2 * float(slot_filled)
        if confidence < self.min_confidence:
            slot_filled = False

        fact = Fact(
            text=justification if slot_filled else "",
            confidence=confidence if slot_filled else 0.0,
            confidence_self=confidence_self,
            confidence_retrieval=confidence_retrieval,
            slot_filled=slot_filled,
            slot_name=slot_name,
            answer_span=answer if slot_filled else "",
            support_ids=support_ids if slot_filled else [],
        )
        rejection_reason = failure_reason or self._rejection_reason(
            answer=answer,
            support_ids=support_ids,
            type_ok=type_ok,
            too_long=too_long,
        )
        return EvidenceCapsule(
            answer=answer if slot_filled else "",
            fact=fact,
            subgoal_id=node.id,
            sub_question=node.question,
            answer_type=node.answer_type,
            retrieved_doc_ids=retrieved_ids,
            retrieved_docs_total=len(retrieved_ids),
        ), rejection_reason

    @staticmethod
    def _format_chunks(hits: list[RetrievalHit]) -> str:
        rows = [
            {
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
                "text": hit.text,
            }
            for hit in hits
        ]
        return json.dumps(rows, ensure_ascii=False)

    @staticmethod
    def _normalize_answer(question: str, answer: str) -> str:
        return answer

    @staticmethod
    def _rejection_reason(
        answer: str,
        support_ids: list[str],
        type_ok: bool,
        too_long: bool,
    ) -> str:
        reasons: list[str] = []
        if not answer:
            reasons.append("no answer span was grounded")
        if not support_ids:
            reasons.append("no cited support chunk was retained")
        if not type_ok:
            reasons.append("the answer type did not match the sub-question")
        if too_long:
            reasons.append("the answer span was too long")
        return "; ".join(reasons) if reasons else "the evidence was insufficient"

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

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0
