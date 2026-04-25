"""Isolated investigator subagent backed by semantic top-k retrieval."""

from __future__ import annotations

import logging

from arag.core.config import Config
from arag.core.llm import LLMClient

from .retrieval import EvidenceReader
from .types import EvidenceCapsule, Fact

logger = logging.getLogger(__name__)


class Investigator:
    """Resolve one sub-question using private semantic retrieval.

    The orchestrator passes only task metadata: sub-question, retrieval query,
    slot name, and expected answer hint. The investigator retrieves evidence
    itself and returns only a compact capsule.
    """

    def __init__(self, config: Config, llm_client: LLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self.evidence_capsule_limit = int(
            config.get("investigator.evidence_capsule_limit", 3)
        )
        self.search_top_k = int(config.get("investigator.search_top_k", 5))
        self.reader = EvidenceReader(config, llm_client)

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
        capsule, _ = await self.investigate_with_usage(
            sub_question=sub_question,
            goal=goal,
            prior_facts=prior_facts,
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
        """Run retrieval and distillation inside this subagent."""
        del prior_facts  # Subagents intentionally do not receive orchestrator evidence.
        top_k = int(
            max_read_override
            if max_read_override is not None
            else search_top_k_override
            if search_top_k_override is not None
            else self.search_top_k
        )
        query = (retrieval_query or sub_question).strip()
        try:
            capsule, tokens = await self.reader.retrieve_and_distill(
                sub_question=sub_question,
                retrieval_query=query,
                goal=goal,
                slot_name=slot_name,
                slot_hint=slot_hint,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("investigator retrieval failed for %r: %s", sub_question, exc)
            fact = Fact(text="", confidence=0.0, slot_name=slot_name)
            return EvidenceCapsule(answer="", fact=fact), 0

        logger.debug(
            "Investigator capsule: answer=%r confidence=%.2f support_ids=%s",
            capsule.answer,
            capsule.fact.confidence,
            capsule.fact.support_ids,
        )
        return capsule, tokens
