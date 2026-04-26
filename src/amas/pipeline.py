"""AMAS pipeline: thin wrapper around the Cursor-style orchestrator.

The orchestrator is a single tool-using agent. Topology emerges from its
tool choices on each turn (search / spawn / final). This pipeline simply
constructs the agent + retriever + investigator from config and exposes a
``run(question, qid)`` method that returns a :class:`PipelineResult`.
"""

from __future__ import annotations

import logging

from .config import Config
from .investigator import Investigator
from .llm import LLMClient
from .orchestrator import Orchestrator
from .retriever import Retriever
from .types import PipelineResult

logger = logging.getLogger(__name__)


class AMASPipeline:
    """Adaptive Multi-Agent System pipeline (Cursor-style emergent topology)."""

    def __init__(self, config: Config) -> None:
        self.config = config

        orch_llm = LLMClient.from_config(config.agent_llm("orchestrator"))
        inv_llm = LLMClient.from_config(config.agent_llm("investigator"))

        ret_cfg = config.raw().get("retriever", {}) or {}
        self.retriever = Retriever(
            base_url=ret_cfg.get("base_url", "http://node408:8003"),
            default_top_k=int(ret_cfg.get("top_k", 10)),
            timeout_seconds=float(ret_cfg.get("timeout_seconds", 30)),
        )

        self.investigator = Investigator(
            llm=inv_llm,
            retriever=self.retriever,
            top_k=int(ret_cfg.get("top_k", 10)),
            min_confidence=float(config.get("pipeline.min_fact_confidence", 0.3)),
            max_searches=int(config.get("pipeline.max_searches_per_subagent", 3)),
            max_answer_words=int(config.get("pipeline.max_answer_words", 8)),
        )
        self.orchestrator = Orchestrator(
            llm=orch_llm,
            investigator=self.investigator,
            retriever=self.retriever,
            max_turns=int(config.get("pipeline.max_turns", 8)),
            default_top_k=int(ret_cfg.get("top_k", 10)),
            context_token_budget=int(config.get("pipeline.context_token_budget", 28000)),
            max_response_tokens=int(orch_llm.max_tokens),
        )

    async def run(self, question: str, question_id: str) -> PipelineResult:
        logger.info("AMAS start: qid=%s", question_id)
        result = await self.orchestrator.solve(question)

        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=result.answer,
            step_trace=result.trace,
            num_subagent_calls=result.n_subagents,
            total_tokens=result.total_tokens,
            orchestrator_tokens=result.orchestrator_tokens,
            subagent_tokens=result.subagent_tokens,
            facts_used=[c.fact for c in result.capsules],
            retrieved_doc_ids=result.retrieved_ids,
            retrieved_docs_total=result.retrieved_total,
            route_decision=result.route,
            route_confidence=result.confidence,
            extras={
                "answer_type": result.answer_type,
                "support_ids": result.support_ids,
                "justification": result.justification,
                "n_searches": result.n_searches,
                "n_spawn_turns": result.n_spawn_turns,
                "n_subagents": result.n_subagents,
                "chunk_tokens": result.chunk_tokens,
                "reasoning_tokens": result.reasoning_tokens,
            },
        )
