"""Focused investigator subagent for Adaptive Recursive SAGE.

Performs targeted retrieval (keyword + semantic search) and distills
the results into a bounded :class:`EvidenceCapsule` — a concise answer,
a single distilled fact, and a limited number of supporting snippet IDs.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import requests

from arag.core.config import Config
from arag.core.llm import LLMClient

from .types import EvidenceCapsule, Fact

logger = logging.getLogger(__name__)

# Maximum retries when LLM returns malformed JSON
_MAX_JSON_RETRIES = 2
_REPAIR_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return ONLY one JSON object matching the requested schema. "
    "No markdown, no commentary, no extra text."
)


class Investigator:
    """Focused subagent that retrieves and distills evidence for a sub-question.

    Parameters
    ----------
    config:
        Application configuration.  Must contain ``data.chunks_file``,
        ``data.index_dir``, ``data.embedding_model``, and optionally
        ``investigator.evidence_capsule_limit`` and
        ``investigator.search_top_k``.
    llm_client:
        An initialised :class:`LLMClient` used for distillation.
    """

    def __init__(self, config: Config, llm_client: LLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

        # Read investigator-specific settings
        self.evidence_capsule_limit: int = config.get(
            "investigator.evidence_capsule_limit", 2
        )
        self.search_top_k: int = config.get("investigator.search_top_k", 5)
        self.min_fact_confidence: float = float(
            config.get("investigator.min_fact_confidence", 0.6)
        )
        self.raw_snippets: bool = bool(config.get("ablation.raw_snippets", False))
        self.blind_subagent: bool = bool(config.get("ablation.blind_subagent", False))

        self.retriever_url: str = str(
            config.get("data.retriever_url", "http://node408:8003/retrieve")
        ).strip()
        self.max_passage_chars: int = int(
            config.get("investigator.max_passage_chars", 1200)
        )
        self.max_query_variants: int = int(
            config.get("investigator.max_query_variants", 2)
        )
        self.max_search_rounds: int = int(
            config.get("investigator.max_search_rounds", 3)
        )

        # Load distillation prompt template
        prompt_name = str(config.get("investigator.prompt_file", "")).strip()
        if not prompt_name:
            prompt_name = (
                "investigator_distill_strict.txt"
                if bool(config.get("investigator.use_strict_distill", True))
                else "investigator_distill.txt"
            )
        prompt_path = Path(__file__).parent / "prompts" / prompt_name
        self._distill_template = prompt_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def investigate(
        self,
        sub_question: str,
        goal: str,
        prior_facts: list[Fact],
        retrieval_query: str | None = None,
        slot_name: str = "",
        slot_hint: str = "",
        search_top_k_override: int | None = None,
        max_read_override: int | None = None,
    ) -> EvidenceCapsule:
        """Retrieve evidence for *sub_question* and distil into a capsule.

        Steps
        -----
        1. Extract keyword terms and a semantic query from *sub_question*.
        2. Run keyword search and semantic search in parallel (both
           return chunk IDs + abbreviated snippets).
        3. Merge and de-duplicate chunk IDs, keep top-scoring ones.
        4. Read the full text of the top chunks.
        5. Build a distillation prompt with the sub-question, goal,
           prior facts, and retrieved passage texts.
        6. Call the LLM to extract answer / fact / confidence /
           support_ids.
        7. Parse the JSON response (with retry / fallback on malformed
           JSON).
        8. Enforce *evidence_capsule_limit* on ``support_ids``.
        9. Return an :class:`EvidenceCapsule`.
        """
        capsule, _ = await self.investigate_with_usage(
            sub_question,
            goal,
            prior_facts,
            retrieval_query=retrieval_query,
            slot_name=slot_name,
            slot_hint=slot_hint,
            search_top_k_override=search_top_k_override,
            max_read_override=max_read_override,
        )
        return capsule

    async def investigate_with_usage(
        self,
        sub_question: str,
        goal: str,
        prior_facts: list[Fact],
        retrieval_query: str | None = None,
        slot_name: str = "",
        slot_hint: str = "",
        search_top_k_override: int | None = None,
        max_read_override: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Like :meth:`investigate`, but also returns token usage."""
        return await self._investigate_private_search_loop(
            sub_question=sub_question,
            goal=goal,
            prior_facts=prior_facts,
            retrieval_query=retrieval_query,
            slot_name=slot_name,
            slot_hint=slot_hint,
            search_top_k_override=search_top_k_override,
            max_read_override=max_read_override,
        )

        total_tokens = 0
        if self.blind_subagent:
            goal = ""
            prior_facts = []
        effective_top_k = (
            int(search_top_k_override)
            if search_top_k_override is not None
            else self.search_top_k
        )
        max_read = (
            int(max_read_override)
            if max_read_override is not None
            else effective_top_k * 2
        )

        # 1. Generate retrieval queries
        effective_query = retrieval_query.strip() if retrieval_query and retrieval_query.strip() else sub_question
        semantic_query = self._build_semantic_query(
            effective_query,
            goal,
            prior_facts,
            slot_name=slot_name,
            slot_hint=slot_hint,
        )
        query_candidates = self._build_query_candidates(
            effective_query=effective_query,
            semantic_query=semantic_query,
            sub_question=sub_question,
            goal=goal,
        )

        # 2. Retrieve from node408 only; keep investigators stateless and isolated.
        retrieved_docs = await self._retrieve_docs(
            query_candidates=query_candidates,
            top_k=effective_top_k,
            max_docs=max_read,
        )
        docs_by_id = {
            str(doc["chunk_id"]): str(doc["text"])
            for doc in retrieved_docs
            if str(doc.get("chunk_id", "")).strip() and str(doc.get("text", "")).strip()
        }
        all_chunk_ids = [str(doc["chunk_id"]) for doc in retrieved_docs]
        if retrieved_docs:
            retrieval_candidates = self._format_retrieval_candidates(query_candidates=query_candidates, docs=retrieved_docs)
        else:
            retrieval_candidates = "No relevant retrieval candidates found."
            logger.warning("No chunks found for sub-question: %s", sub_question)

        support_ids = all_chunk_ids[: self.evidence_capsule_limit]
        support_snippets = [docs_by_id[sid] for sid in support_ids if sid in docs_by_id]

        if self.raw_snippets:
            capsule = EvidenceCapsule(
                answer="",
                fact=Fact(
                    text=retrieval_candidates if retrieved_docs else "",
                    confidence=0.5 if retrieved_docs else 0.0,
                    confidence_self=0.0,
                    confidence_retrieval=0.5 if retrieved_docs else 0.0,
                    slot_filled=False,
                    support_ids=support_ids,
                    source_step=0,
                ),
                support_snippets=support_snippets,
                retrieved_doc_ids=all_chunk_ids,
                retrieved_docs_total=len(all_chunk_ids),
            )
            return capsule, total_tokens

        # 5. Build distillation prompt
        prior_facts_text = (
            "\n".join(f"- {f.text}" for f in prior_facts)
            if prior_facts
            else "None"
        )
        user_prompt = self._distill_template.format(
            sub_question=sub_question,
            goal=goal,
            target_slot=slot_name or "final_answer",
            target_slot_hint=slot_hint or "No extra slot hint available.",
            prior_facts=prior_facts_text,
            retrieval_candidates=retrieval_candidates,
            capsule_limit=self.evidence_capsule_limit,
        )

        # 6–7. Call LLM with retry logic for malformed JSON
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise fact extraction assistant. "
                    "Always respond with valid JSON."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        parsed: Optional[dict] = None
        for attempt in range(_MAX_JSON_RETRIES + 1):
            try:
                response = await self.llm_client.async_chat(messages)
                total_tokens += self._extract_total_tokens(response)
                content: str = response["message"].get("content", "")
                content = self._strip_thinking(content)
                parsed = self._parse_json_response(content)
                if parsed is not None:
                    break
                if attempt < _MAX_JSON_RETRIES:
                    messages = [
                        messages[0],
                        messages[1],
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _REPAIR_PROMPT},
                    ]
            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_JSON_RETRIES + 1,
                    exc,
                )

        # 8. Build EvidenceCapsule (with fallback on total failure)
        if parsed is not None:
            # Normalise support_ids: the LLM sometimes prepends "Chunk "
            # to the numeric ID (e.g. "Chunk 354" instead of "354").
            raw_ids = parsed.get("support_ids", [])[: self.evidence_capsule_limit]
            support_ids = [
                self._normalise_chunk_id(str(sid))
                for sid in raw_ids
            ]
            answer = str(
                parsed.get("answer_span", parsed.get("answer", ""))
            ).strip()
            fact_text = parsed.get("fact", "").strip()
            confidence_self = float(parsed.get("confidence", 0.0))
            slot_filled = bool(answer and fact_text and support_ids)
            confidence_retrieval = self._compute_retrieval_confidence(
                support_ids=support_ids,
                fallback_ids=all_chunk_ids,
                retrieved_docs=retrieved_docs,
            )
            confidence = (
                0.4 * confidence_retrieval
                + 0.4 * max(min(confidence_self, 1.0), 0.0)
                + 0.2 * float(slot_filled)
            )

            if not answer or not fact_text or not support_ids:
                answer = ""
                fact_text = ""
                confidence = 0.0
                confidence_self = 0.0
                confidence_retrieval = 0.0
                slot_filled = False
                support_ids = []
            elif confidence < self.min_fact_confidence:
                # Preserve weak grounded hypotheses for one-step recursive
                # refinement without marking the slot as solved.
                slot_filled = False

            # Populate support_snippets from actual chunk texts
            support_snippets = []
            for sid in support_ids:
                if sid in docs_by_id:
                    support_snippets.append(docs_by_id[sid])

            fact = Fact(
                text=fact_text,
                confidence=confidence,
                confidence_self=confidence_self,
                confidence_retrieval=confidence_retrieval,
                slot_filled=slot_filled,
                answer_span=answer,
                support_ids=support_ids,
                source_step=0,  # set by the pipeline
            )
            capsule = EvidenceCapsule(
                answer=answer,
                fact=fact,
                support_snippets=support_snippets,
                retrieved_doc_ids=all_chunk_ids,
                retrieved_docs_total=len(all_chunk_ids),
            )
        else:
            # Fallback: empty capsule
            logger.error(
                "All JSON parse attempts failed for sub-question: %s",
                sub_question,
            )
            capsule = EvidenceCapsule(
                answer="",
                fact=Fact(
                    text="",
                    confidence=0.0,
                    confidence_self=0.0,
                    confidence_retrieval=0.0,
                    slot_filled=False,
                    answer_span="",
                    support_ids=[],
                    source_step=0,
                ),
                support_snippets=[],
                retrieved_doc_ids=all_chunk_ids,
                retrieved_docs_total=len(all_chunk_ids),
            )

        logger.debug(
            "Investigator capsule: answer=%r, confidence=%.2f, "
            "support_ids=%s",
            capsule.answer,
            capsule.fact.confidence,
            capsule.fact.support_ids,
        )

        return capsule, total_tokens

    async def _investigate_private_search_loop(
        self,
        *,
        sub_question: str,
        goal: str,
        prior_facts: list[Fact],
        retrieval_query: str | None = None,
        slot_name: str = "",
        slot_hint: str = "",
        search_top_k_override: int | None = None,
        max_read_override: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        total_tokens = 0
        if self.blind_subagent:
            goal = ""
            prior_facts = []
        effective_top_k = (
            int(search_top_k_override)
            if search_top_k_override is not None
            else self.search_top_k
        )
        max_read = (
            int(max_read_override)
            if max_read_override is not None
            else effective_top_k * 2
        )
        effective_query = retrieval_query.strip() if retrieval_query and retrieval_query.strip() else sub_question
        semantic_query = self._build_semantic_query(
            effective_query,
            goal,
            prior_facts,
            slot_name=slot_name,
            slot_hint=slot_hint,
        )
        query_history = self._build_query_candidates(
            effective_query=effective_query,
            semantic_query=semantic_query,
            sub_question=sub_question,
            goal=goal,
        )[:1]
        current_query = query_history[0] if query_history else sub_question.strip()
        merged_docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        parsed: Optional[dict] = None

        for round_idx in range(max(1, self.max_search_rounds)):
            docs = await asyncio.to_thread(self._retrieve_once, current_query, effective_top_k)
            for doc in docs:
                cid = str(doc.get("chunk_id", "")).strip()
                text = str(doc.get("text", "")).strip()
                if not cid or not text or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                merged_docs.append(doc)
                if len(merged_docs) >= max_read:
                    break

            docs_by_id = {
                str(doc["chunk_id"]): str(doc["text"])
                for doc in merged_docs
                if str(doc.get("chunk_id", "")).strip() and str(doc.get("text", "")).strip()
            }
            all_chunk_ids = [str(doc["chunk_id"]) for doc in merged_docs]
            chunk_result = self._format_retrieved_passages(merged_docs)
            retrieval_candidates = self._format_retrieval_candidates(
                query_candidates=query_history,
                docs=merged_docs,
            ) if merged_docs else "No relevant retrieval candidates found."

            if self.raw_snippets:
                capsule = EvidenceCapsule(
                    answer="",
                    fact=Fact(
                        text=chunk_result if merged_docs else "",
                        confidence=0.5 if merged_docs else 0.0,
                        confidence_self=0.0,
                        confidence_retrieval=0.5 if merged_docs else 0.0,
                        slot_filled=False,
                        support_ids=all_chunk_ids[: self.evidence_capsule_limit],
                        source_step=0,
                    ),
                    support_snippets=[],
                    retrieved_doc_ids=all_chunk_ids,
                    retrieved_docs_total=len(all_chunk_ids),
                )
                return capsule, total_tokens

            prior_facts_text = (
                "\n".join(f"- {f.text}" for f in prior_facts)
                if prior_facts
                else "None"
            )
            user_prompt = self._distill_template.format(
                sub_question=sub_question,
                goal=goal,
                target_slot=slot_name or "final_answer",
                target_slot_hint=slot_hint or "No extra slot hint available.",
                prior_facts=prior_facts_text,
                retrieved_passages=chunk_result,
                capsule_limit=self.evidence_capsule_limit,
            )
            parsed, prompt_tokens = await self._call_and_parse_json([
                {
                    "role": "system",
                    "content": (
                        "You are a precise fact extraction assistant. "
                        "Always respond with valid JSON."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ])
            total_tokens += prompt_tokens

            if parsed is not None:
                raw_ids = parsed.get("support_ids", [])[: self.evidence_capsule_limit]
                support_ids = [self._normalise_chunk_id(str(sid)) for sid in raw_ids]
                answer = str(parsed.get("answer_span", parsed.get("answer", ""))).strip()
                fact_text = str(parsed.get("fact", "")).strip()
                confidence_self = float(parsed.get("confidence", 0.0) or 0.0)
                if answer and fact_text and support_ids and confidence_self >= self.min_fact_confidence:
                    confidence_retrieval = self._compute_retrieval_confidence(
                        support_ids=support_ids,
                        fallback_ids=all_chunk_ids,
                        retrieved_docs=merged_docs,
                    )
                    confidence = (
                        0.4 * confidence_retrieval
                        + 0.4 * max(min(confidence_self, 1.0), 0.0)
                        + 0.2
                    )
                    fact = Fact(
                        text=fact_text,
                        confidence=confidence,
                        confidence_self=confidence_self,
                        confidence_retrieval=confidence_retrieval,
                        slot_filled=True,
                        answer_span=answer,
                        support_ids=support_ids,
                        source_step=0,
                    )
                    return EvidenceCapsule(
                        answer=answer,
                        fact=fact,
                        support_snippets=[],
                        retrieved_doc_ids=all_chunk_ids,
                        retrieved_docs_total=len(all_chunk_ids),
                    ), total_tokens

            if round_idx >= max(1, self.max_search_rounds) - 1:
                break

            refined_query, refine_tokens = await self._propose_followup_query(
                sub_question=sub_question,
                goal=goal,
                prior_facts=prior_facts,
                slot_name=slot_name,
                slot_hint=slot_hint,
                retrieval_candidates=retrieval_candidates,
                weak_answer="" if parsed is None else str(parsed.get("answer_span", parsed.get("answer", ""))).strip(),
                weak_fact="" if parsed is None else str(parsed.get("fact", "")).strip(),
            )
            total_tokens += refine_tokens
            refined_query = refined_query.strip()
            if not refined_query or refined_query in query_history:
                break
            query_history.append(refined_query)
            current_query = refined_query

        docs_by_id = {
            str(doc["chunk_id"]): str(doc["text"])
            for doc in merged_docs
            if str(doc.get("chunk_id", "")).strip() and str(doc.get("text", "")).strip()
        }
        all_chunk_ids = [str(doc["chunk_id"]) for doc in merged_docs]
        if parsed is not None:
            raw_ids = parsed.get("support_ids", [])[: self.evidence_capsule_limit]
            support_ids = [self._normalise_chunk_id(str(sid)) for sid in raw_ids]
            answer = str(parsed.get("answer_span", parsed.get("answer", ""))).strip()
            fact_text = str(parsed.get("fact", "")).strip()
            confidence_self = float(parsed.get("confidence", 0.0) or 0.0)
            slot_filled = bool(answer and fact_text and support_ids)
            confidence_retrieval = self._compute_retrieval_confidence(
                support_ids=support_ids,
                fallback_ids=all_chunk_ids,
                retrieved_docs=merged_docs,
            )
            confidence = (
                0.4 * confidence_retrieval
                + 0.4 * max(min(confidence_self, 1.0), 0.0)
                + 0.2 * float(slot_filled)
            )
            if not answer or not fact_text or not support_ids:
                answer = ""
                fact_text = ""
                confidence = 0.0
                confidence_self = 0.0
                confidence_retrieval = 0.0
                slot_filled = False
                support_ids = []
            elif confidence < self.min_fact_confidence:
                slot_filled = False
            fact = Fact(
                text=fact_text,
                confidence=confidence,
                confidence_self=confidence_self,
                confidence_retrieval=confidence_retrieval,
                slot_filled=slot_filled,
                answer_span=answer,
                support_ids=support_ids,
                source_step=0,
            )
            return EvidenceCapsule(
                answer=answer,
                fact=fact,
                support_snippets=[],
                retrieved_doc_ids=all_chunk_ids,
                retrieved_docs_total=len(all_chunk_ids),
            ), total_tokens

        return EvidenceCapsule(
            answer="",
            fact=Fact(
                text="",
                confidence=0.0,
                confidence_self=0.0,
                confidence_retrieval=0.0,
                slot_filled=False,
                answer_span="",
                support_ids=[],
                source_step=0,
            ),
            support_snippets=[],
            retrieved_doc_ids=all_chunk_ids,
            retrieved_docs_total=len(all_chunk_ids),
        ), total_tokens

    async def _call_and_parse_json(self, messages: list[dict[str, str]]) -> tuple[Optional[dict], int]:
        total_tokens = 0
        parsed: Optional[dict] = None
        current_messages = list(messages)
        for attempt in range(_MAX_JSON_RETRIES + 1):
            try:
                response = await self.llm_client.async_chat(current_messages)
                total_tokens += self._extract_total_tokens(response)
                content = self._strip_thinking(response["message"].get("content", ""))
                parsed = self._parse_json_response(content)
                if parsed is not None:
                    break
                if attempt < _MAX_JSON_RETRIES:
                    current_messages = [
                        current_messages[0],
                        current_messages[1],
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _REPAIR_PROMPT},
                    ]
            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_JSON_RETRIES + 1,
                    exc,
                )
        return parsed, total_tokens

    async def _propose_followup_query(
        self,
        *,
        sub_question: str,
        goal: str,
        prior_facts: list[Fact],
        slot_name: str,
        slot_hint: str,
        retrieval_candidates: str,
        weak_answer: str,
        weak_fact: str,
    ) -> tuple[str, int]:
        prior_facts_text = "\n".join(f"- {f.text}" for f in prior_facts) if prior_facts else "None"
        prompt = f"""You are refining a private retrieval query for a sub-question.
Return only JSON: {{"query": "..."}}

Sub-question: {sub_question}
Goal: {goal}
Target slot: {slot_name or 'final_answer'}
Target slot hint: {slot_hint or 'none'}
Prior facts:
{prior_facts_text}
Weak answer: {weak_answer or '(none)'}
Weak fact: {weak_fact or '(none)'}
Retrieval candidates:
{retrieval_candidates}

Rules:
- Propose one sharper search query that resolves the missing bridge or target relation.
- Use entities already uncovered when helpful.
- Keep it short.
- If no better query exists, return {{"query": ""}}."""
        parsed, tokens = await self._call_and_parse_json([
            {"role": "system", "content": "You refine retrieval queries. Return only JSON."},
            {"role": "user", "content": prompt},
        ])
        if not parsed:
            return "", tokens
        return str(parsed.get("query", "")).strip(), tokens


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        """Extract short keyword terms from a natural-language question.

        Removes common English stop-words and question words, then
        returns the remaining terms (each 1–3 words).
        """
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "either", "neither", "each", "every", "all", "any", "few",
            "more", "most", "other", "some", "such", "no", "only",
            "own", "same", "than", "too", "very", "just", "because",
            "what", "which", "who", "whom", "when", "where", "why",
            "how", "that", "this", "these", "those", "it", "its",
        }

        # Tokenise on whitespace / punctuation
        tokens = re.findall(r"[A-Za-z0-9]+(?:[-][A-Za-z0-9]+)*", question)

        keywords: list[str] = []
        i = 0
        while i < len(tokens):
            token_lower = tokens[i].lower()
            if token_lower in stop_words:
                i += 1
                continue

            # Try to form a 2-word phrase if the next token is also content
            if i + 1 < len(tokens) and tokens[i + 1].lower() not in stop_words:
                phrase = f"{tokens[i]} {tokens[i + 1]}"
                keywords.append(phrase)
                i += 2
            else:
                keywords.append(tokens[i])
                i += 1

        # Fallback: if we extracted nothing, use the full question
        return keywords if keywords else [question]

    @staticmethod
    def _extract_chunk_ids(search_result: str) -> list[str]:
        """Extract chunk IDs from the formatted text output of ARAG search tools.

        Matches patterns like ``Chunk ID: 42`` or ``Chunk ID: abc123``.
        """
        raw_ids = re.findall(r"Chunk ID:\s*(\S+)", search_result)
        # Strip trailing punctuation that is not part of the ID
        return [cid.rstrip(",.;:") for cid in raw_ids]

    @staticmethod
    def _build_semantic_query(
        sub_question: str,
        goal: str,
        prior_facts: list[Fact],
        slot_name: str = "",
        slot_hint: str = "",
    ) -> str:
        """Augment semantic retrieval with the immediate goal and grounded facts."""
        parts = [sub_question.strip()]
        if slot_name.strip():
            parts.append(f"Target slot: {slot_name.strip()}")
        if slot_hint.strip():
            parts.append(f"Slot hint: {slot_hint.strip()}")
        if goal.strip():
            parts.append(f"Goal: {goal.strip()}")

        grounded_facts = [
            (fact.answer_span.strip() or fact.text.strip())
            for fact in sorted(prior_facts, key=lambda fact: fact.confidence, reverse=True)
            if fact.text.strip() and fact.confidence >= 0.6
        ][:2]
        if grounded_facts:
            parts.append("Known facts: " + " ".join(grounded_facts))

        return "\n".join(part for part in parts if part)

    def _build_query_candidates(
        self,
        *,
        effective_query: str,
        semantic_query: str,
        sub_question: str,
        goal: str,
    ) -> list[str]:
        candidates: list[str] = []
        for item in (
            effective_query.strip(),
            semantic_query.strip(),
            sub_question.strip(),
            f"{sub_question.strip()} Goal: {goal.strip()}".strip(),
        ):
            if not item:
                continue
            if item not in candidates:
                candidates.append(item)
        return candidates[: max(1, self.max_query_variants)]

    async def _retrieve_docs(
        self,
        *,
        query_candidates: list[str],
        top_k: int,
        max_docs: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for query in query_candidates:
            docs = await asyncio.to_thread(self._retrieve_once, query, top_k)
            for doc in docs:
                cid = str(doc.get("chunk_id", "")).strip()
                text = str(doc.get("text", "")).strip()
                if not cid or not text or cid in seen:
                    continue
                seen.add(cid)
                merged.append({
                    "chunk_id": cid,
                    "text": text,
                    "score": float(doc.get("score", 0.0) or 0.0),
                })
                if len(merged) >= max_docs:
                    return merged
        return merged

    def _retrieve_once(self, query: str, top_k: int) -> list[dict[str, Any]]:
        payload = {
            "queries": [query],
            "topk": min(max(int(top_k), 1), 10),
            "mode": "text",
        }
        response = requests.post(self.retriever_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        if not results:
            return []
        docs = results[0] or []
        normalised: list[dict[str, Any]] = []
        for doc in docs:
            cid = str(doc.get("chunk_id") or doc.get("id") or "").strip()
            text = str(doc.get("text") or doc.get("content") or "").strip()
            if not cid or not text:
                continue
            normalised.append({
                "chunk_id": cid,
                "text": text,
                "score": float(doc.get("score", 0.0) or 0.0),
            })
        return normalised

    def _format_retrieval_candidates(
        self,
        *,
        query_candidates: list[str],
        docs: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        if query_candidates:
            parts.append(
                "Queries used:\n" + "\n".join(f"- {query}" for query in query_candidates)
            )
        for doc in docs:
            cid = str(doc.get("chunk_id", "")).strip()
            text = str(doc.get("text", "")).strip()
            score = float(doc.get("score", 0.0) or 0.0)
            title = self._extract_doc_title(text)
            candidate_spans = self._extract_candidate_spans(text)[:6]
            span_text = ", ".join(candidate_spans) if candidate_spans else "(none)"
            parts.append(
                f"Chunk {cid} | score={score:.3f} | title={title} | candidates={span_text}"
            )
        return "\n".join(parts)

    def _format_retrieved_passages(self, docs: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for doc in docs:
            cid = str(doc.get("chunk_id", "")).strip()
            text = str(doc.get("text", "")).strip()
            clipped = text[: self.max_passage_chars]
            parts.append(f"Chunk {cid}:\n{clipped}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_doc_title(text: str) -> str:
        first_line = text.splitlines()[0].strip().strip('"').strip()
        return first_line[:120] if first_line else "unknown"

    @staticmethod
    def _extract_candidate_spans(text: str) -> list[str]:
        patterns = [
            r"\b(?:[0-3]?\d\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-3]?\d,\s+\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            r"\b(?:1[0-9]{3}|20[0-9]{2}|[0-9]{1,4})\b",
            r"\b[A-Z][A-Za-z0-9.'-]+(?:\s+[A-Z][A-Za-z0-9.'-]+){0,3}\b",
        ]
        seen: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, text):
                value = str(match).strip().strip(",.;:")
                if len(value) < 2 or len(value) > 80:
                    continue
                if value not in seen:
                    seen.append(value)
                if len(seen) >= 8:
                    return seen
        return seen


    @staticmethod
    def _compute_retrieval_confidence(
        support_ids: list[str],
        fallback_ids: list[str],
        retrieved_docs: list[dict[str, Any]],
    ) -> float:
        """Estimate retrieval confidence from node408 similarity scores."""
        score_by_chunk = {
            str(doc.get("chunk_id", "")): float(doc.get("score", 0.0) or 0.0)
            for doc in retrieved_docs
        }
        candidate_ids = support_ids or fallback_ids
        scores = [score_by_chunk[cid] for cid in candidate_ids if cid in score_by_chunk]
        if not scores:
            return 0.0
        top_scores = sorted(scores, reverse=True)[:2]
        avg_score = sum(top_scores) / len(top_scores)
        return max(0.0, min(avg_score, 1.0))

    @staticmethod
    def _extract_similarity_scores(search_result: str) -> dict[str, float]:
        """Extract semantic-search similarity scores keyed by chunk ID."""
        matches = re.findall(
            r"Chunk ID:\s*(\S+)\s+\(Similarity:\s*([0-9.]+)\)",
            search_result,
        )
        return {
            chunk_id.rstrip(",.;:"): float(score)
            for chunk_id, score in matches
        }

    @staticmethod
    def _normalise_chunk_id(raw_id: str) -> str:
        """Normalise a chunk ID returned by the LLM.

        The LLM sometimes prefixes numeric IDs with ``"Chunk "`` or
        ``"chunk "`` (e.g. ``"Chunk 354"`` instead of ``"354"``).
        Strip such prefixes so the ID matches the keys in
        :attr:`read_chunk.chunks_dict`.
        """
        cleaned = re.sub(r"^[Cc]hunk\s+", "", raw_id).strip()
        return cleaned

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3-style ``<think>...</think>`` blocks from *text*."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """Attempt to parse a JSON object from *text*.

        Tries, in order:
        1. Direct parse of the whole text and common fenced variants.
        2. Extract the first balanced ``{`` … ``}`` block and parse that.
        3. Apply light cleanup (trailing commas / smart quotes).
        4. Fall back to ``ast.literal_eval`` for Python-style dicts.
        Returns ``None`` if parsing fails.
        """
        candidates = [text.strip()]

        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            candidates.append(fence_match.group(1).strip())

        balanced = Investigator._extract_balanced_json_object(text)
        if balanced:
            candidates.append(balanced)

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)

            parsed = Investigator._parse_json_candidate(candidate)
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _extract_balanced_json_object(text: str) -> Optional[str]:
        """Return the first balanced JSON-like object found in *text*."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    @staticmethod
    def _parse_json_candidate(candidate: str) -> Optional[dict]:
        """Parse one JSON-like candidate with progressively looser rules."""
        normalised = (
            candidate.strip()
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
        )

        for payload in (
            normalised,
            re.sub(r",\s*([}\]])", r"\1", normalised),
        ):
            try:
                result = json.loads(payload)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

        try:
            result = ast.literal_eval(normalised)
            if isinstance(result, dict):
                return result
        except (SyntaxError, ValueError):
            return None

        return None

    @staticmethod
    def _extract_total_tokens(response: dict[str, Any]) -> int:
        """Extract total tokens from a chat response."""
        raw_usage = response.get("raw_response", {}).get("usage", {}) or {}
        total_tokens = raw_usage.get("total_tokens")
        if total_tokens is not None:
            return int(total_tokens)
        return int(response.get("input_tokens", 0)) + int(response.get("output_tokens", 0))
