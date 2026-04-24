"""Focused investigator subagent for Adaptive Recursive SAGE.

Performs targeted retrieval (keyword + semantic search) and distills
the results into a bounded :class:`EvidenceCapsule` — a concise answer,
a single distilled fact, and a limited number of supporting snippet IDs.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from arag.core.config import Config
from arag.core.context import AgentContext
from arag.core.llm import LLMClient, TokenBudgetExceededError
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.semantic_search import SemanticSearchTool

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
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0

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

        # Data paths
        chunks_file: str = config.get("data.chunks_file")
        index_dir: str = config.get("data.index_dir")
        embedding_model: str = os.environ.get(
            "ARAG_EMBEDDING_MODEL",
            config.get("data.embedding_model", "intfloat/e5-base-v2"),
        )

        # Initialise ARAG retrieval tools
        self.keyword_search = KeywordSearchTool(chunks_file)
        self.semantic_search = SemanticSearchTool(
            chunks_file, index_dir, embedding_model
        )
        self.read_chunk = ReadChunkTool(chunks_file)

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

    def reset_usage_totals(self) -> None:
        """Reset per-run prompt/completion token counters."""
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0

    def get_usage_totals(self) -> dict[str, int]:
        """Return per-run prompt/completion token totals."""
        return {
            "prompt_tokens": self._prompt_tokens_total,
            "completion_tokens": self._completion_tokens_total,
        }

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
        remaining_total_tokens: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Like :meth:`investigate`, but also returns token usage."""
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

        # 1. Generate search queries
        effective_query = retrieval_query.strip() if retrieval_query and retrieval_query.strip() else sub_question
        keywords = self._extract_keywords(effective_query)
        semantic_query = self._build_semantic_query(
            effective_query,
            goal,
            prior_facts,
            slot_name=slot_name,
            slot_hint=slot_hint,
        )

        # 2–3. Run both searches and collect chunk IDs
        ctx = AgentContext()

        kw_result, kw_log = self.keyword_search.execute(
            ctx, keywords=keywords, top_k=effective_top_k
        )
        sem_result, sem_log = self.semantic_search.execute(
            ctx, query=semantic_query, top_k=effective_top_k
        )

        kw_chunk_ids = self._extract_chunk_ids(kw_result)
        sem_chunk_ids = self._extract_chunk_ids(sem_result)

        # Merge preserving order (keyword first), de-duplicate
        seen: set[str] = set()
        all_chunk_ids: list[str] = []
        for cid in kw_chunk_ids + sem_chunk_ids:
            if cid not in seen:
                seen.add(cid)
                all_chunk_ids.append(cid)

        # 4. Read top chunks
        ids_to_read = all_chunk_ids[:max_read]
        if ids_to_read:
            chunk_result, chunk_log = self.read_chunk.execute(
                ctx, chunk_ids=ids_to_read
            )
        else:
            chunk_result = "No relevant passages found."
            logger.warning("No chunks found for sub-question: %s", sub_question)

        return await self._distill_from_chunk_result_with_usage(
            sub_question=sub_question,
            goal=goal,
            prior_facts=prior_facts,
            slot_name=slot_name,
            slot_hint=slot_hint,
            chunk_result=chunk_result,
            all_chunk_ids=all_chunk_ids,
            semantic_result=sem_result,
            remaining_total_tokens=remaining_total_tokens,
            tokens_spent=total_tokens,
        )

    async def distill_from_chunk_ids_with_usage(
        self,
        sub_question: str,
        goal: str,
        prior_facts: list[Fact],
        chunk_ids: list[str],
        slot_name: str = "",
        slot_hint: str = "",
        remaining_total_tokens: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Distil an evidence capsule from a fixed chunk-id set without new retrieval."""
        seen: set[str] = set()
        all_chunk_ids: list[str] = []
        for chunk_id in chunk_ids:
            cleaned = str(chunk_id).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                all_chunk_ids.append(cleaned)

        ctx = AgentContext()
        if all_chunk_ids:
            chunk_result, _ = self.read_chunk.execute(ctx, chunk_ids=all_chunk_ids)
        else:
            chunk_result = "No relevant passages found."
            logger.warning("No cached chunks provided for sub-question: %s", sub_question)

        return await self._distill_from_chunk_result_with_usage(
            sub_question=sub_question,
            goal=goal,
            prior_facts=prior_facts,
            slot_name=slot_name,
            slot_hint=slot_hint,
            chunk_result=chunk_result,
            all_chunk_ids=all_chunk_ids,
            semantic_result="",
            remaining_total_tokens=remaining_total_tokens,
            tokens_spent=0,
        )

    async def _distill_from_chunk_result_with_usage(
        self,
        *,
        sub_question: str,
        goal: str,
        prior_facts: list[Fact],
        slot_name: str,
        slot_hint: str,
        chunk_result: str,
        all_chunk_ids: list[str],
        semantic_result: str,
        remaining_total_tokens: int | None,
        tokens_spent: int,
    ) -> tuple[EvidenceCapsule, int]:
        """Distil a capsule from already selected passages."""
        total_tokens = int(tokens_spent)

        support_ids = all_chunk_ids[: self.evidence_capsule_limit]
        support_snippets = [
            self.read_chunk.chunks_dict[sid]
            for sid in support_ids
            if sid in self.read_chunk.chunks_dict
        ]

        if self.raw_snippets:
            has_support = bool(all_chunk_ids)
            capsule = EvidenceCapsule(
                answer="",
                fact=Fact(
                    text=chunk_result if has_support else "",
                    confidence=0.5 if has_support else 0.0,
                    confidence_self=0.0,
                    confidence_retrieval=0.5 if has_support else 0.0,
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
            retrieved_passages=chunk_result,
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
                response = await self.llm_client.async_chat(
                    messages,
                    remaining_total_tokens=(
                        None
                        if remaining_total_tokens is None
                        else max(int(remaining_total_tokens) - total_tokens, 0)
                    ),
                )
                self._record_usage(response)
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
            except TokenBudgetExceededError:
                raise
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
                semantic_result=semantic_result,
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
                if sid in self.read_chunk.chunks_dict:
                    support_snippets.append(self.read_chunk.chunks_dict[sid])

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

    @staticmethod
    def _compute_retrieval_confidence(
        support_ids: list[str],
        fallback_ids: list[str],
        semantic_result: str,
    ) -> float:
        """Estimate retrieval confidence from semantic-search scores."""
        similarity_by_chunk = Investigator._extract_similarity_scores(semantic_result)
        candidate_ids = support_ids or fallback_ids
        scores = [
            similarity_by_chunk[cid]
            for cid in candidate_ids
            if cid in similarity_by_chunk
        ]
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

    def _record_usage(self, response: dict[str, Any]) -> None:
        """Accumulate prompt/completion usage from one LLM response."""
        self._prompt_tokens_total += int(response.get("input_tokens", 0) or 0)
        self._completion_tokens_total += int(response.get("output_tokens", 0) or 0)
