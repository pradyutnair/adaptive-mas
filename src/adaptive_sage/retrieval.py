"""Shared semantic retrieval and evidence distillation.

All retrieval goes through the retriever server. The server returns chunk-level
hits; this module compresses those chunks into short excerpts before any LLM
distillation so full passages are not passed around the MAS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient

from .types import EvidenceCapsule, Fact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalHit:
    """One result returned by the semantic retriever."""

    chunk_id: str
    text: str
    score: float


class RetrieverClient:
    """Async wrapper around the retriever server REST API."""

    def __init__(self, config: Config) -> None:
        self.base_url = str(
            config.get("retriever.base_url", "http://localhost:8003")
        ).rstrip("/")
        self.default_top_k = int(config.get("retriever.topk", 5))
        self.timeout_seconds = float(config.get("retriever.timeout_seconds", 30))

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return await asyncio.to_thread(self._retrieve_sync, query, top_k)

    def _retrieve_sync(self, query: str, top_k: int | None) -> list[RetrievalHit]:
        payload = {
            "queries": [query],
            "topk": int(top_k or self.default_top_k),
            "mode": "text",
        }
        request = urllib.request.Request(
            f"{self.base_url}/retrieve",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"retriever request failed for query={query!r}") from exc

        if not raw.get("success", False):
            raise RuntimeError(f"retriever returned failure: {raw}")

        result_obj = raw.get("results") or {}
        result_rows = result_obj.get("results") or []
        first_row = result_rows[0] if result_rows else []
        hits: list[RetrievalHit] = []
        for item in first_row:
            hits.append(
                RetrievalHit(
                    chunk_id=str(item.get("chunk_id", "")),
                    text=str(item.get("text", "")),
                    score=float(item.get("score", 0.0) or 0.0),
                )
            )
        return hits


class EvidenceReader:
    """Retrieve chunk-level hits, compress them, and distill a capsule."""

    def __init__(self, config: Config, llm_client: LLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self.retriever = RetrieverClient(config)
        self.evidence_capsule_limit = int(
            config.get("investigator.evidence_capsule_limit", 3)
        )
        self.min_fact_confidence = float(
            config.get("investigator.min_fact_confidence", 0.3)
        )
        self.excerpt_chars_per_hit = int(
            config.get("retriever.excerpt_chars_per_hit", 900)
        )
        self.distill_max_tokens = int(config.get("retriever.distill_max_tokens", 512))

        prompts_dir = Path(__file__).parent / "prompts"
        self._system_template = (
            prompts_dir / "retrieval_distill_system.txt"
        ).read_text(encoding="utf-8")
        self._user_template = (
            prompts_dir / "retrieval_distill_user.txt"
        ).read_text(encoding="utf-8")

    async def retrieve_and_distill(
        self,
        *,
        sub_question: str,
        retrieval_query: str,
        goal: str,
        slot_name: str,
        slot_hint: str,
        top_k: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Return a compact capsule from private semantic retrieval."""
        query = retrieval_query.strip() or sub_question.strip()
        hits = await self.retriever.retrieve(query, top_k=top_k)
        evidence = self._format_evidence(query, hits)
        user_prompt = self._user_template.format(
            sub_question=sub_question,
            goal=goal,
            slot_name=slot_name or "final_answer",
            slot_hint=slot_hint or "No extra slot hint available.",
            retrieval_query=query,
            evidence=evidence or "No relevant evidence found.",
        )
        response = await self.llm_client.async_chat(
            messages=[
                {"role": "system", "content": self._system_template},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.distill_max_tokens,
        )
        tokens = self._extract_total_tokens(response)
        content = self._strip_thinking(response["message"].get("content", ""))
        parsed = self._parse_json(content)
        capsule = self._build_capsule(parsed, hits, slot_name)
        return capsule, tokens

    def _format_evidence(self, query: str, hits: list[RetrievalHit]) -> str:
        lines: list[str] = []
        for hit in hits:
            excerpt = self._compact_chunk(query, hit.text)
            if not excerpt:
                continue
            lines.append(
                f"[{hit.chunk_id}] score={hit.score:.3f}\n{excerpt}"
            )
        return "\n\n".join(lines)

    def _compact_chunk(self, query: str, text: str) -> str:
        """Extract small query-relevant excerpts from one retrieved chunk."""
        cleaned = self._clean_text(text)
        if len(cleaned) <= self.excerpt_chars_per_hit:
            return cleaned

        terms = self._query_terms(query)
        sentences = self._split_sentences(cleaned)
        selected: list[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in terms):
                selected.append(sentence)
            if len(" ".join(selected)) >= self.excerpt_chars_per_hit:
                break

        if not selected:
            selected = sentences[:3]

        excerpt = " ".join(selected).strip()
        return excerpt[: self.excerpt_chars_per_hit].strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", query.lower())
        stop_words = {
            "what", "which", "where", "when", "who", "whom", "whose", "how",
            "the", "and", "or", "but", "that", "this", "with", "from", "into",
            "about", "called", "known", "answer", "question", "entity",
        }
        return [t for t in tokens if len(t) > 2 and t not in stop_words][:12]

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _build_capsule(
        self,
        parsed: dict[str, Any],
        hits: list[RetrievalHit],
        slot_name: str,
    ) -> EvidenceCapsule:
        answer = str(parsed.get("answer_span", parsed.get("answer", ""))).strip()
        justification = str(parsed.get("justification", "")).strip()
        support_ids = [
            str(sid).strip()
            for sid in list(parsed.get("support_ids") or [])[: self.evidence_capsule_limit]
            if str(sid).strip()
        ]
        retrieved_ids = [hit.chunk_id for hit in hits]
        support_ids = [sid for sid in support_ids if sid in set(retrieved_ids)]
        confidence_self = self._bounded_float(parsed.get("confidence", 0.0))
        type_valid = bool(parsed.get("answer_type_valid", True))

        if not answer or not type_valid:
            support_ids = []
            justification = ""
            confidence_self = 0.0

        confidence_retrieval = 1.0 if support_ids else 0.0
        slot_filled = bool(answer and justification and support_ids)
        confidence = (
            0.4 * confidence_retrieval
            + 0.4 * confidence_self
            + 0.2 * float(slot_filled)
        )
        if confidence < self.min_fact_confidence:
            slot_filled = False

        if not slot_filled:
            answer = ""
            justification = ""
            support_ids = []
            confidence = 0.0
            confidence_self = 0.0
            confidence_retrieval = 0.0

        fact = Fact(
            text=justification,
            confidence=confidence,
            confidence_self=confidence_self,
            confidence_retrieval=confidence_retrieval,
            slot_filled=slot_filled,
            slot_name=slot_name,
            answer_span=answer,
            support_ids=support_ids,
            support_snippets=[],
            source_step=0,
        )
        return EvidenceCapsule(
            answer=answer,
            fact=fact,
            support_snippets=[],
            retrieved_doc_ids=retrieved_ids,
            retrieved_docs_total=len(retrieved_ids),
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            result = json.loads(match.group())
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _strip_thinking(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned if cleaned else text.strip()

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_total_tokens(response: dict[str, Any]) -> int:
        raw_usage = response.get("raw_response", {}).get("usage", {}) or {}
        total_tokens = raw_usage.get("total_tokens")
        if total_tokens is not None:
            return int(total_tokens)
        return int(response.get("input_tokens", 0)) + int(response.get("output_tokens", 0))
