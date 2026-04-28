"""Deterministic investigator with private retrieval-reading rounds."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .llm import LLMClient, parse_json_object
from .retriever import RetrievalHit, Retriever
from .types import AnswerType, EvidenceCapsule, Fact, SubgoalNode

logger = logging.getLogger(__name__)


class Investigator:
    """Resolve one subgoal and return only a compact evidence capsule."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        top_k: int = 10,
        min_confidence: float = 0.3,
        max_searches: int = 2,
        max_answer_words: int = 8,
        max_evidence_hits: int = 6,
        max_excerpt_chars: int = 600,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.top_k = int(top_k)
        self.min_confidence = float(min_confidence)
        self.max_searches = max(1, int(max_searches))
        self.max_answer_words = int(max_answer_words)
        self.max_evidence_hits = max(1, int(max_evidence_hits))
        self.max_excerpt_chars = max(120, int(max_excerpt_chars))
        prompt_dir = Path(__file__).parent / "prompts"
        self._read_template = (prompt_dir / "analyze.txt").read_text(encoding="utf-8")
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
        parent_question: str = "",
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
            parent_question=parent_question,
        )

    async def investigate_node(
        self,
        node: SubgoalNode,
        hint: str = "",
        slot_name: str = "",
        top_k_override: int | None = None,
        query_override: str | None = None,
        parent_question: str = "",
    ) -> tuple[EvidenceCapsule, int]:
        top_k = int(top_k_override or self.top_k)

        query = str(query_override or node.question).strip()
        retrieved_ids: list[str] = []
        rejection_reason = "initial search"

        queries = self._query_variants(
            query=query,
            node_question=node.question,
            hint=hint,
            parent_question=parent_question,
        )[: self.max_searches]
        all_hits: list[RetrievalHit] = []
        for search_query in queries:
            query_hits = await self.retriever.retrieve(search_query, top_k=top_k)
            all_hits.extend(query_hits)
            for hit in query_hits:
                if hit.chunk_id not in retrieved_ids:
                    retrieved_ids.append(hit.chunk_id)

        hits = self._merge_hits(all_hits)[:top_k]
        read_obj, read_tokens = await self._read_evidence(node, hint, hits)
        capsule, _ = self._build_capsule(
            read_obj,
            node,
            retrieved_ids,
            hits,
            slot_name,
            failure_reason=rejection_reason,
            search_queries=queries,
        )
        self.last_searches_used = len(queries)
        return capsule, read_tokens

    async def rewrite_query(
        self,
        node: SubgoalNode,
        hint: str,
        previous_query: str,
        previous_answer: str = "",
        previous_justification: str = "",
        rejection_reason: str = "",
    ) -> tuple[str, int]:
        rewrite_obj, tokens = await self._rewrite(
            node=node,
            hint=hint,
            previous_query=previous_query,
            previous_answer=previous_answer,
            previous_justification=previous_justification,
            rejection_reason=rejection_reason or "evidence was insufficient",
        )
        query = str(rewrite_obj.get("query", "")).strip() or previous_query
        return query, tokens

    async def _read_evidence(
        self,
        node: SubgoalNode,
        hint: str,
        hits: list[RetrievalHit],
    ) -> tuple[dict[str, Any], int]:
        evidence_blob = self._format_evidence(hits)
        prompt = self._read_template.format(
            sub_question=node.question.strip(),
            expected_answer_type=node.answer_type.value,
            hint=hint.strip() or "(none)",
            evidence=evidence_blob,
        )
        resp = await self.llm.chat(messages=[{"role": "user", "content": prompt}])
        return parse_json_object(resp.content), resp.total_tokens

    async def _rewrite(
        self,
        node: SubgoalNode,
        hint: str,
        previous_query: str,
        previous_answer: str,
        previous_justification: str,
        rejection_reason: str,
    ) -> tuple[dict[str, Any], int]:
        prompt = self._rewrite_template.format(
            sub_question=node.question.strip(),
            expected_answer_type=node.answer_type.value,
            hint=(hint.strip() or "(none)") + f" Rejection reason: {rejection_reason}.",
            previous_query=previous_query.strip(),
            previous_answer=previous_answer,
            previous_justification=previous_justification,
        )
        resp = await self.llm.chat(messages=[{"role": "user", "content": prompt}])
        parsed = parse_json_object(resp.content)
        if "query" not in parsed:
            parsed = {"query": previous_query}
        return parsed, resp.total_tokens

    def _build_capsule(
        self,
        parsed: dict[str, Any],
        node: SubgoalNode,
        retrieved_ids: list[str],
        evidence_hits: list[RetrievalHit],
        slot_name: str,
        failure_reason: str,
        search_queries: list[str],
    ) -> tuple[EvidenceCapsule, str]:
        status = str(parsed.get("status", "")).strip().lower()
        answer = self._normalize_answer(node.question, str(parsed.get("answer_span", "")).strip())
        justification = str(parsed.get("justification", "")).strip()
        parsed_failure_reason = str(parsed.get("failure_reason", "")).strip()
        support_ids_raw = parsed.get("support_ids") or []
        support_ids = [str(s).strip() for s in support_ids_raw if str(s).strip()]
        support_ids = [s for s in support_ids if s in set(retrieved_ids)]
        confidence_self = self._bounded_float(parsed.get("confidence", 0.0))

        explicitly_insufficient = status == "insufficient"

        if not answer:
            confidence_self = 0.0
            support_ids = []

        # Keep this gate deliberately simple. The investigator may reason over
        # ambiguous evidence; we only require a non-empty answer, a grounded
        # justification, and a cited retrieved id. Do not reject by answer type
        # or span length here.
        confidence_retrieval = 1.0 if support_ids else 0.0
        slot_filled = bool(answer and justification and support_ids)
        if explicitly_insufficient and not slot_filled:
            confidence_self = 0.0
        confidence = 0.5 * confidence_retrieval + 0.4 * confidence_self + 0.1 * float(slot_filled)
        if not slot_filled:
            confidence = 0.0

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
        rejection_reason = parsed_failure_reason or failure_reason or self._rejection_reason(
            answer=answer,
            support_ids=support_ids,
            type_ok=True,
            too_long=False,
        )
        capsule_failure_reason = "" if slot_filled else rejection_reason
        return EvidenceCapsule(
            answer=answer if slot_filled else "",
            fact=fact,
            subgoal_id=node.id,
            sub_question=node.question,
            answer_type=node.answer_type,
            evidence_snippets=self._support_snippets(support_ids, evidence_hits) if slot_filled else [],
            retrieved_doc_ids=retrieved_ids,
            retrieved_docs_total=len(retrieved_ids),
            failure_reason=capsule_failure_reason,
            search_queries=list(search_queries),
        ), rejection_reason

    def _format_evidence(self, hits: list[RetrievalHit]) -> str:
        rows = []
        for hit in hits[: self.max_evidence_hits]:
            snippets = [s for s in (hit.snippets or []) if str(s).strip()]
            if snippets:
                evidence_text = " ".join(snippets[:3])
            else:
                evidence_text = self._hit_excerpt(hit)
            rows.append({
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
                "evidence": self._excerpt(evidence_text),
            })
        return json.dumps(rows, ensure_ascii=False)

    def _support_snippets(
        self,
        support_ids: list[str],
        hits: list[RetrievalHit],
    ) -> list[dict[str, str]]:
        by_id = {hit.chunk_id: hit for hit in hits}
        snippets: list[dict[str, str]] = []
        for support_id in support_ids:
            hit = by_id.get(support_id)
            if hit is None:
                continue
            snippets.append({
                "chunk_id": support_id,
                "excerpt": self._hit_excerpt(hit),
            })
        return snippets

    def _hit_excerpt(self, hit: RetrievalHit) -> str:
        if hit.snippets:
            return self._excerpt(" ".join(hit.snippets))
        return self._excerpt(hit.text)

    def _excerpt(self, text: str) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= self.max_excerpt_chars:
            return cleaned
        return cleaned[: self.max_excerpt_chars].rstrip() + "..."

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
            reasons.append("no cited support id was retained")
        if not type_ok:
            reasons.append("the answer type did not match the sub-question")
        if too_long:
            reasons.append("the answer span was too long")
        return "; ".join(reasons) if reasons else "the evidence was insufficient"

    @classmethod
    def _query_variants(
        cls,
        query: str,
        node_question: str,
        hint: str,
        parent_question: str,
    ) -> list[str]:
        base = " ".join(str(query or node_question or "").split())
        variants: list[str] = []
        cls._append_unique_query(variants, base)

        context_terms = cls._context_terms(parent_question or hint, limit=10)
        entity_query = cls._entity_focused_query(base, context_terms)
        cls._append_unique_query(variants, entity_query)

        dense_query = cls._keyword_dense_query(base, context_terms)
        cls._append_unique_query(variants, dense_query)
        return variants or [base]

    @staticmethod
    def _append_unique_query(queries: list[str], query: str) -> None:
        cleaned = " ".join(str(query or "").split()).strip(" ?")
        if cleaned and cleaned.lower() not in {q.lower() for q in queries}:
            queries.append(cleaned[:240])

    @classmethod
    def _entity_focused_query(cls, query: str, context_terms: str) -> str:
        quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", query)
        proper = cls._proper_terms(query)
        relation = cls._content_terms(query, limit=6, keep_case=False)
        parts = quoted + proper + relation
        if context_terms:
            parts.append(context_terms)
        return " ".join(cls._dedupe_terms(parts))

    @classmethod
    def _keyword_dense_query(cls, query: str, context_terms: str) -> str:
        parts = cls._content_terms(query, limit=12, keep_case=True)
        if context_terms:
            parts.append(context_terms)
        return " ".join(cls._dedupe_terms(parts))

    @classmethod
    def _context_terms(cls, text: str, limit: int = 10) -> str:
        proper = cls._proper_terms(text or "")
        dates = re.findall(r"\b(?:1[0-9]|20)\d{2}\b", text or "")
        terms = proper + dates + cls._content_terms(text, limit=limit, keep_case=True)
        return " ".join(cls._dedupe_terms(terms)[:limit])

    @staticmethod
    def _proper_terms(text: str) -> list[str]:
        skip = {
            "what", "which", "who", "whom", "whose", "when", "where",
            "why", "how", "the", "a", "an", "if", "hint", "original",
            "subgoal", "question", "answer", "next", "step",
        }
        out: list[str] = []
        for phrase in re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", text or ""):
            first = phrase.split()[0].lower()
            if first not in skip:
                out.append(phrase)
        return out

    @staticmethod
    def _content_terms(text: str, limit: int = 10, keep_case: bool = True) -> list[str]:
        stop = {
            "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "from",
            "with", "by", "as", "at", "is", "are", "was", "were", "be", "been",
            "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
            "did", "does", "do", "this", "that", "these", "those", "it", "its", "their",
            "his", "her", "he", "she", "they", "them", "there", "then", "next", "step",
            "asks", "answer", "subgoal", "result", "question", "original",
        }
        terms: list[str] = []
        for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text or ""):
            low = raw.lower().strip("_-")
            if len(low) < 3 or low in stop:
                continue
            terms.append(raw if keep_case else low)
            if len(terms) >= limit:
                break
        return terms

    @staticmethod
    def _dedupe_terms(terms: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = " ".join(str(term or "").split()).strip(" ,.;:()[]")
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                out.append(cleaned)
        return out

    @staticmethod
    def _merge_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        by_id: dict[str, RetrievalHit] = {}
        for hit in hits:
            existing = by_id.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                by_id[hit.chunk_id] = hit
        return sorted(by_id.values(), key=lambda h: h.score, reverse=True)

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0
