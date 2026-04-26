"""Focused investigator subagent for Adaptive Recursive SAGE.

Runs a focused tool-using retrieval sub-agent and returns a bounded
:class:`EvidenceCapsule`: answer, justification/fact, confidence, and
support IDs. Raw retrieved passages are never passed back to the orchestrator.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

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
    """Focused tool-using subagent for one sub-question.

    Parameters
    ----------
    config:
        Application configuration.  Must contain ``data.chunks_file``,
        ``data.index_dir``, ``data.embedding_model``, and optionally
        ``investigator.evidence_capsule_limit`` and
        ``investigator.search_top_k``.
    llm_client:
        An initialised :class:`LLMClient` used by the isolated tool agent.
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
        self.subagent_max_loops: int = int(config.get("investigator.subagent_max_loops", 4))
        self.subagent_token_budget: int = int(
            config.get("investigator.subagent_token_budget", 24000)
        )

        self.include_prior_support_snippets: bool = bool(
            config.get("investigator.include_prior_support_snippets", True)
        )

        # Data paths
        chunks_file: str = config.get("data.chunks_file")
        index_dir: str = config.get("data.index_dir")
        embedding_model: str = os.environ.get(
            "ARAG_EMBEDDING_MODEL",
            config.get("data.embedding_model", "intfloat/e5-base-v2"),
        )

        # Single retrieval surface for sub-agents: E5 top-k semantic chunk search,
        # with an internal lexical fallback for exact-entity recall.
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model
        with open(Path(index_dir) / "sentence_index.pkl", "rb") as f:
            index_data = pickle.load(f)
        self.sentences = index_data["sentences"]
        self.embeddings = index_data["embeddings"]
        self.sentence_to_chunk = index_data["sentence_to_chunk"]
        self.chunks = index_data["chunks"]
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunk_texts = [
            (chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk))
            for chunk in self.chunks
        ] if isinstance(self.chunks, list) else {
            str(cid): (chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk))
            for cid, chunk in self.chunks.items()
        }

        # Prompt construction is inline so no retrieved passage text can be
        # accidentally templated in by the orchestrator.

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
        """Run an isolated retrieval sub-agent for *sub_question*.

        The prompt contains only the task and prior capsule facts. Any passage
        text seen by the sub-agent must come from its own tool calls.
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
        """Run an isolated tool-using sub-agent and return its capsule.

        The orchestrator never passes retrieved passages into this prompt. The
        sub-agent receives only the task, optional prior capsule facts, and
        tool access. It must retrieve/read evidence inside its own context and
        return a final JSON answer, justification, confidence, and support IDs.
        """
        if self.blind_subagent:
            goal = ""
            prior_facts = []

        top_k = (
            int(search_top_k_override)
            if search_top_k_override is not None
            else self.search_top_k
        )
        max_read = (
            int(max_read_override)
            if max_read_override is not None
            else max(top_k, self.evidence_capsule_limit)
        )
        prior_facts_text = (
            self._format_prior_capsules(
                prior_facts, include_support_snippets=self.include_prior_support_snippets
            )
            if prior_facts
            else "None"
        )
        seed_query = (
            retrieval_query.strip()
            if retrieval_query and retrieval_query.strip()
            else sub_question.strip()
        )

        system_prompt = (
            "You are an isolated retrieval sub-agent. You are not given evidence. "
            "Use the semantic_topk retrieval tool to find evidence yourself. "
            "Do not answer from memory. Read chunks before producing the final JSON. "
            "When you have enough evidence, stop calling tools and output only JSON. "
            "Never include raw passage text in the final answer or justification."
        )
        user_prompt = f"""Sub-question: {sub_question}
Goal: {goal}
Target slot: {slot_name or 'final_answer'}
Target slot hint: {slot_hint or 'No extra slot hint available.'}
Suggested retrieval query: {seed_query}
Prior capsule facts:
{prior_facts_text}

Constraints:
- Retrieve inside this sub-agent using tools.
- Use search top_k <= {top_k}.
- Read at most {max_read} chunks unless needed to disambiguate.
- Final output must be exactly one JSON object:
  {{"answer_span": "...", "justification": "...", "confidence": 0.0, "support_ids": ["..."]}}
- `answer_span` is the minimal answer span for the sub-question.
- `justification` should be 1-3 short sentences, not copied passages.
- answer_span MUST be non-empty whenever any retrieved chunk is even partially relevant. Use low confidence for weak evidence; empty output is valid only when every retrieved chunk is irrelevant.
"""

        try:
            result = await self._run_isolated_tool_loop(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_read=max_read,
            )
        except Exception as exc:
            logger.warning("Sub-agent failed for %s: %s", sub_question, exc)
            result = {"answer": "", "trajectory": [], "total_tokens": 0}

        total_tokens = int(result.get("total_tokens", 0) or 0)
        parsed = self._normalise_agent_result(result)
        answer = parsed["answer"]
        fact_text = parsed["fact"] or parsed["answer"]
        hits = result.get("hits", []) or []
        if not answer and hits:
            fallback_hit = hits[0]
            fallback_text = self._clean_snippet_text(str(fallback_hit.get("text", "")))[:420]
            answer = fallback_text
            fact_text = fallback_text
            parsed["support_ids"] = [fallback_hit.get("chunk_id", "")]
            parsed["confidence"] = min(float(parsed.get("confidence") or 0.25), 0.35)
        support_ids = [
            self._normalise_chunk_id(str(sid))
            for sid in parsed["support_ids"][: self.evidence_capsule_limit]
        ]
        support_ids = [sid for sid in support_ids if sid]
        confidence_self = max(0.0, min(float(parsed["confidence"]), 1.0))
        retrieved_doc_ids = self._extract_trajectory_chunk_ids(result.get("trajectory", []))
        if not support_ids:
            support_ids = retrieved_doc_ids[: self.evidence_capsule_limit]
        confidence_retrieval = 1.0 if support_ids else 0.0
        slot_filled = bool(answer and fact_text and support_ids)
        confidence = (
            0.4 * confidence_retrieval
            + 0.4 * confidence_self
            + 0.2 * float(slot_filled)
        )

        support_snippets = self._build_support_snippets(
            hits, support_ids, answer, fact_text
        )

        if not answer or not fact_text or not support_ids:
            answer = ""
            fact_text = ""
            confidence = 0.0
            confidence_self = 0.0
            confidence_retrieval = 0.0
            slot_filled = False
            support_ids = []
            support_snippets = []
        elif confidence < self.min_fact_confidence:
            slot_filled = False

        fact = Fact(
            text=fact_text,
            confidence=confidence,
            confidence_self=confidence_self,
            confidence_retrieval=confidence_retrieval,
            slot_filled=slot_filled,
            slot_name=slot_name,
            answer_span=answer,
            support_ids=support_ids,
            support_snippets=support_snippets,
            source_step=0,
        )
        capsule = EvidenceCapsule(
            answer=answer,
            fact=fact,
            support_snippets=support_snippets,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=len(retrieved_doc_ids),
        )

        logger.debug(
            "Investigator capsule: answer=%r, confidence=%.2f, support_ids=%s",
            capsule.answer,
            capsule.fact.confidence,
            capsule.fact.support_ids,
        )
        return capsule, total_tokens

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------



    async def _run_isolated_tool_loop(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_read: int,
    ) -> dict[str, Any]:
        query_match = re.search(r"Suggested retrieval query:\s*(.+)", user_prompt)
        query = query_match.group(1).strip() if query_match else user_prompt[:240]
        hits = self._semantic_retrieve(query, top_k=max_read)
        retrieval_text = "\n\n".join(
            f"[Chunk {hit['chunk_id']}] score={hit['score']:.3f}\n{hit['text']}"
            for hit in hits
        ) or "No relevant chunks found."
        trajectory = [{
            "tool_name": "semantic_topk",
            "arguments": {"query": query, "top_k": max_read},
            "tool_result": retrieval_text,
            "chunk_ids": [hit["chunk_id"] for hit in hits],
            "chunks_found": len(hits),
        }]
        prompt = user_prompt + """

Retrieved evidence from your semantic_topk tool:
""" + retrieval_text + """

Extract the best grounded answer from the retrieved evidence.
Return exactly one JSON object:
{"answer_span":"<minimal answer span>","justification":"<short fact directly supporting the answer>","confidence":<float 0.0-1.0>,"support_ids":["<chunk_id>"]}
Rules:
- Use only the retrieved evidence.
- Answer the sub-question exactly, not a broader description or upstream bridge entity.
- answer_span must be the shortest copied span that best resolves the target slot.
- answer_span MUST be non-empty whenever any retrieved chunk is even partially relevant. Use low confidence for weak evidence; empty output is valid only when every retrieved chunk is irrelevant.
- Do not output text outside JSON.
"""
        response = await self.llm_client.async_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        total_tokens = self._extract_total_tokens(response)
        content = self._strip_thinking(response["message"].get("content", ""))
        return {
            "answer": content,
            "trajectory": trajectory,
            "total_tokens": total_tokens,
            "hits": hits,
        }


    def _build_support_snippets(
        self,
        hits: list[dict[str, Any]],
        support_ids: list[str],
        answer: str,
        justification: str,
    ) -> list[str]:
        """Return tiny capsule evidence snippets from supported chunks only."""
        wanted = {self._normalise_chunk_id(str(cid)) for cid in support_ids}
        snippets: list[str] = []
        for hit in hits:
            chunk_id = self._normalise_chunk_id(str(hit.get("chunk_id", "")))
            if wanted and chunk_id not in wanted:
                continue
            text = self._clean_snippet_text(str(hit.get("text", "")))
            snippet = self._window_around_answer(text, answer)
            if not snippet and justification:
                snippet = self._window_around_answer(text, justification[:80])
            if not snippet:
                snippet = text[:260].strip()
            if snippet:
                snippets.append(f"[{chunk_id}] {snippet}")
            if len(snippets) >= min(2, self.evidence_capsule_limit):
                break
        return snippets

    @staticmethod
    def _clean_snippet_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _window_around_answer(text: str, answer: str) -> str:
        answer = answer.strip()
        if not text or not answer:
            return ""
        match = re.search(re.escape(answer), text, flags=re.IGNORECASE)
        if not match:
            return ""
        start = max(0, match.start() - 120)
        end = min(len(text), match.end() + 120)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        return snippet[:320].strip()

    @staticmethod
    def _format_prior_capsules(
        prior_facts: list[Fact], include_support_snippets: bool = True
    ) -> str:
        if not prior_facts:
            return "None"
        lines: list[str] = []
        for fact in prior_facts:
            answer = fact.answer_span.strip() or fact.text.strip()
            line = f"- {fact.slot_name or 'fact'}: {answer}"
            if fact.text.strip() and fact.text.strip() != answer:
                line += f"; {fact.text.strip()}"
            if fact.support_ids:
                line += f" (chunks: {', '.join(fact.support_ids[:3])})"
            lines.append(line)
            if include_support_snippets:
                for snippet in fact.support_snippets[:2]:
                    lines.append(f"  evidence: {snippet}")
        return "\n".join(lines)


    def _merge_keyword_fallback(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Add exact lexical matches without exposing a second tool."""
        seen = {str(hit.get("chunk_id", "")) for hit in hits}
        terms = self._query_terms(query)
        if not terms:
            return hits
        scored: list[tuple[float, str, str]] = []
        items = enumerate(self.chunk_texts) if isinstance(self.chunk_texts, list) else self.chunk_texts.items()
        for cid, text in items:
            text_str = str(text)
            text_norm = text_str.lower()
            score = 0.0
            for term in terms:
                if term in text_norm:
                    score += 3.0 if " " in term else 1.0
            if score > 0:
                scored.append((score, str(cid), text_str))
        scored.sort(key=lambda item: item[0], reverse=True)
        merged = list(hits)
        for score, cid, text in scored:
            if cid in seen:
                continue
            merged.append({"chunk_id": cid, "score": score / 10.0, "text": text})
            seen.add(cid)
            if len(merged) >= top_k + 3:
                break
        return merged

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        quoted = re.findall(r'[\'"]([^\'"]{3,80})[\'"]', query)
        caps = re.findall(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,4}\b", query)
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", query)]
        stop = {"what", "which", "where", "when", "who", "whose", "that", "this", "with", "from", "were", "was", "are", "the", "and", "for", "did", "does", "has", "had"}
        terms = [t.strip().lower() for t in quoted + caps if len(t.strip()) >= 3]
        terms.extend(t for t in tokens if t not in stop and len(t) >= 4)
        dedup: list[str] = []
        seen: set[str] = set()
        for term in terms:
            term = re.sub(r"\s+", " ", term).strip()
            if term and term not in seen:
                dedup.append(term)
                seen.add(term)
        return dedup[:12]

    def _semantic_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        top_k = max(1, min(int(top_k), 20))
        try:
            query_embedding = self.embedding_model.encode(
                [query],
                prompt_name=os.getenv("ARAG_QUERY_PROMPT_NAME", "query"),
                normalize_embeddings=True,
            )[0]
        except TypeError:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][: top_k * 4]
        by_chunk: dict[str, float] = {}
        for idx in top_indices:
            chunk_id = str(self.sentence_to_chunk[idx])
            score = float(similarities[idx])
            by_chunk[chunk_id] = max(score, by_chunk.get(chunk_id, -1.0))
        ranked = sorted(by_chunk.items(), key=lambda x: x[1], reverse=True)[:top_k]
        hits = []
        for chunk_id, score in ranked:
            chunk = self.chunks[int(chunk_id)] if isinstance(self.chunks, list) else self.chunks[chunk_id]
            text = chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk)
            hits.append({"chunk_id": chunk_id, "score": score, "text": text})
        return hits

    @staticmethod
    def _normalise_agent_result(result: dict[str, Any]) -> dict[str, Any]:
        raw_answer = str(result.get("answer", "") or "").strip()
        parsed: dict[str, Any] = {}
        parsed_obj = Investigator._parse_json_response(raw_answer)
        if isinstance(parsed_obj, dict):
            parsed = parsed_obj

        answer = str(
            parsed.get("answer_span", parsed.get("answer", ""))
        ).strip()
        justification = str(
            result.get("justification")
            or parsed.get("justification")
            or parsed.get("fact")
            or ""
        ).strip()
        confidence = result.get("confidence", parsed.get("confidence", 0.0))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        support_ids = (
            parsed.get("support_ids")
            or []
        )
        if not isinstance(support_ids, list):
            support_ids = [support_ids]
        return {
            "answer": answer,
            "fact": justification,
            "confidence": confidence,
            "support_ids": support_ids,
        }

    @classmethod
    def _extract_trajectory_chunk_ids(cls, trajectory: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for entry in trajectory:
            args = entry.get("arguments", {}) if isinstance(entry, dict) else {}
            candidates: list[Any] = []
            if isinstance(args, dict):
                if "chunk_id" in args:
                    candidates.append(args["chunk_id"])
                if isinstance(args.get("chunk_ids"), list):
                    candidates.extend(args["chunk_ids"])
            if isinstance(entry.get("chunk_ids"), list):
                candidates.extend(entry["chunk_ids"])
            tool_result = str(entry.get("tool_result", "")) if isinstance(entry, dict) else ""
            candidates.extend(re.findall(r"Chunk(?: ID)?:\s*([A-Za-z0-9_.:-]+)", tool_result))
            for cid in candidates:
                norm = cls._normalise_chunk_id(str(cid))
                if norm and norm not in seen:
                    seen.add(norm)
                    ids.append(norm)
        return ids

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
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in text:
            json_start = text.find("{")
            text = text[json_start:] if json_start >= 0 else re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

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
