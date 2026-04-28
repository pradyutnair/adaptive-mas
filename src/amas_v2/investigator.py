"""Investigator: resolve one subgoal via private retrieval + evidence reading."""

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
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        top_k: int = 5,
        max_searches: int = 3,
        max_evidence_hits: int = 6,
        max_excerpt_chars: int = 800,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.top_k = int(top_k)
        self.max_searches = max(1, int(max_searches))
        self.max_evidence_hits = max(1, int(max_evidence_hits))
        self.max_excerpt_chars = max(120, int(max_excerpt_chars))
        prompt_dir = Path(__file__).parent / "prompts"
        self._read_template = (prompt_dir / "analyze.txt").read_text(encoding="utf-8")
        self._rewrite_template = (prompt_dir / "rewrite.txt").read_text(encoding="utf-8")
        self.last_searches_used = 0

    async def investigate_node(
        self,
        node: SubgoalNode,
        hint: str = "",
        query_override: str | None = None,
        parent_question: str = "",
    ) -> tuple[EvidenceCapsule, int]:
        query = str(query_override or node.question).strip()
        retrieved_ids: list[str] = []

        queries = self._query_variants(query, node.question, hint, parent_question)[
            : self.max_searches
        ]
        all_hits: list[RetrievalHit] = []
        for sq in queries:
            hits = await self.retriever.retrieve(sq, top_k=self.top_k)
            all_hits.extend(hits)
            for h in hits:
                if h.chunk_id not in retrieved_ids:
                    retrieved_ids.append(h.chunk_id)

        merged = self._merge_hits(all_hits)[: self.top_k]
        read_obj, read_tokens = await self._read_evidence(node, hint, merged)
        capsule = self._build_capsule(read_obj, node, retrieved_ids, merged, queries)
        self.last_searches_used = len(queries)
        return capsule, read_tokens

    async def rewrite_query(
        self,
        node: SubgoalNode,
        hint: str,
        previous_query: str,
        previous_answer: str = "",
        previous_justification: str = "",
    ) -> tuple[str, int]:
        prompt = self._rewrite_template.format(
            sub_question=node.question.strip(),
            expected_answer_type=node.answer_type.value,
            hint=hint.strip() or "(none)",
            previous_query=previous_query,
            previous_answer=previous_answer,
            previous_justification=previous_justification,
        )
        resp = await self.llm.chat(messages=[{"role": "user", "content": prompt}], max_tokens=256)
        parsed = parse_json_object(resp.content)
        q = str(parsed.get("query", "")).strip() or previous_query
        return q, resp.total_tokens

    async def _read_evidence(
        self, node: SubgoalNode, hint: str, hits: list[RetrievalHit],
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

    def _build_capsule(
        self,
        parsed: dict[str, Any],
        node: SubgoalNode,
        retrieved_ids: list[str],
        evidence_hits: list[RetrievalHit],
        search_queries: list[str],
    ) -> EvidenceCapsule:
        status = str(parsed.get("status", "")).strip().lower()
        answer = str(parsed.get("answer_span", "")).strip()
        justification = str(parsed.get("justification", "")).strip()
        failure = str(parsed.get("failure_reason", "")).strip()
        support_raw = parsed.get("support_ids") or []
        support_ids = [str(s).strip() for s in support_raw if str(s).strip()]
        support_ids = [s for s in support_ids if s in set(retrieved_ids)]
        conf = self._bounded_float(parsed.get("confidence", 0.0))

        explicitly_insufficient = status == "insufficient"
        if not answer:
            conf = 0.0
            support_ids = []

        slot_filled = bool(answer and justification and support_ids)
        if explicitly_insufficient and not slot_filled:
            conf = 0.0
        confidence = conf if slot_filled else 0.0

        fact = Fact(
            text=justification if slot_filled else "",
            confidence=confidence,
            slot_filled=slot_filled,
            slot_name=f"subgoal_{node.id}",
            answer_span=answer if slot_filled else "",
            support_ids=support_ids if slot_filled else [],
        )
        capsule_failure = "" if slot_filled else (failure or "evidence insufficient")
        snippets = self._support_snippets(support_ids, evidence_hits) if slot_filled else []
        return EvidenceCapsule(
            answer=answer if slot_filled else "",
            fact=fact,
            subgoal_id=node.id,
            sub_question=node.question,
            answer_type=node.answer_type,
            evidence_snippets=snippets,
            retrieved_doc_ids=retrieved_ids,
            retrieved_docs_total=len(retrieved_ids),
            failure_reason=capsule_failure,
            search_queries=list(search_queries),
        )

    def _format_evidence(self, hits: list[RetrievalHit]) -> str:
        rows = []
        for hit in hits[: self.max_evidence_hits]:
            text = self._hit_excerpt(hit)
            rows.append({
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
                "evidence": text,
            })
        return json.dumps(rows, ensure_ascii=False)

    def _support_snippets(self, support_ids: list[str], hits: list[RetrievalHit]) -> list[dict]:
        by_id = {h.chunk_id: h for h in hits}
        return [
            {"chunk_id": sid, "excerpt": self._hit_excerpt(by_id[sid])}
            for sid in support_ids if sid in by_id
        ]

    def _hit_excerpt(self, hit: RetrievalHit) -> str:
        text = " ".join(hit.snippets) if hit.snippets else hit.text
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= self.max_excerpt_chars:
            return cleaned
        return cleaned[: self.max_excerpt_chars].rstrip() + "..."

    @classmethod
    def _query_variants(cls, query: str, node_q: str, hint: str, parent_q: str) -> list[str]:
        base = " ".join(str(query or node_q or "").split())
        variants: list[str] = []
        cls._add(variants, base)
        ctx = cls._context_terms(parent_q or hint, limit=10)
        entity_q = cls._entity_focused(base, ctx)
        cls._add(variants, entity_q)
        kw_q = cls._keyword_dense(base, ctx)
        cls._add(variants, kw_q)
        return variants or [base]

    @staticmethod
    def _add(lst: list[str], q: str) -> None:
        c = " ".join(str(q or "").split()).strip(" ?")
        if c and c.lower() not in {x.lower() for x in lst}:
            lst.append(c[:240])

    @classmethod
    def _entity_focused(cls, query: str, ctx: str) -> str:
        quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", query)
        proper = cls._proper_terms(query)
        rel = cls._content_terms(query, 6, False)
        parts = quoted + proper + rel
        if ctx:
            parts.append(ctx)
        return " ".join(cls._dedupe(parts))

    @classmethod
    def _keyword_dense(cls, query: str, ctx: str) -> str:
        parts = cls._content_terms(query, 12, True)
        if ctx:
            parts.append(ctx)
        return " ".join(cls._dedupe(parts))

    @classmethod
    def _context_terms(cls, text: str, limit: int = 10) -> str:
        proper = cls._proper_terms(text or "")
        dates = re.findall(r"\b(?:1[0-9]|20)\d{2}\b", text or "")
        terms = proper + dates + cls._content_terms(text, limit, True)
        return " ".join(cls._dedupe(terms)[:limit])

    @staticmethod
    def _proper_terms(text: str) -> list[str]:
        skip = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how",
                "the", "a", "an", "if", "hint", "original", "subgoal", "question", "answer"}
        out: list[str] = []
        for phrase in re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", text or ""):
            if phrase.split()[0].lower() not in skip:
                out.append(phrase)
        return out

    @staticmethod
    def _content_terms(text: str, limit: int = 10, keep_case: bool = True) -> list[str]:
        stop = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "from",
                "with", "by", "as", "at", "is", "are", "was", "were", "be", "been",
                "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
                "did", "does", "do", "this", "that", "these", "those", "it", "its", "their",
                "his", "her", "he", "she", "they", "them", "there", "then", "next", "step",
                "asks", "answer", "subgoal", "result", "question", "original"}
        terms: list[str] = []
        for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text or ""):
            low = raw.lower().strip("_-")
            if len(low) >= 3 and low not in stop:
                terms.append(raw if keep_case else low)
                if len(terms) >= limit:
                    break
        return terms

    @staticmethod
    def _dedupe(terms: list[str]) -> list[str]:
        out, seen = [], set()
        for t in terms:
            c = " ".join(str(t or "").split()).strip(" ,.;:()[]")
            k = c.lower()
            if c and k not in seen:
                seen.add(k)
                out.append(c)
        return out

    @staticmethod
    def _merge_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        by_id: dict[str, RetrievalHit] = {}
        for h in hits:
            if h.chunk_id not in by_id or h.score > by_id[h.chunk_id].score:
                by_id[h.chunk_id] = h
        return sorted(by_id.values(), key=lambda x: x.score, reverse=True)

    @staticmethod
    def _bounded_float(v: Any) -> float:
        try:
            return max(0.0, min(float(v), 1.0))
        except (TypeError, ValueError):
            return 0.0
