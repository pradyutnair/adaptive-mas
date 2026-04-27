"""AMAS v2: structure-aware adaptive topology RAG."""

from __future__ import annotations

import logging
from typing import Any

from .config import Config
from .dag_executor import DAGExecutor
from .investigator import Investigator
from .llm import LLMClient
from .planner import Planner
from .retriever import Retriever
from .synthesizer import Synthesizer
from .types import AnswerType, EvidenceCapsule, PipelineResult, StepTrace

logger = logging.getLogger(__name__)


class AMASPipeline:
    """Plan -> execute DAG -> synthesize pipeline."""

    def __init__(self, config: Config) -> None:
        self.config = config

        planner_llm = LLMClient.from_config(_agent_llm(config, "planner", "orchestrator"))
        inv_llm = LLMClient.from_config(config.agent_llm("investigator"))
        synth_llm = LLMClient.from_config(_agent_llm(config, "synthesizer", "orchestrator"))

        ret_cfg = config.raw().get("retriever", {}) or {}
        self.retriever = Retriever(
            base_url=ret_cfg.get("base_url", "http://node408:8003"),
            default_top_k=int(ret_cfg.get("top_k", 10)),
            timeout_seconds=float(ret_cfg.get("timeout_seconds", 30)),
        )

        self.planner = Planner(
            llm=planner_llm,
            max_subgoals=int(config.get("pipeline.max_subgoals", 6)),
        )
        self.investigator = Investigator(
            llm=inv_llm,
            retriever=self.retriever,
            top_k=int(ret_cfg.get("top_k", 10)),
            min_confidence=float(config.get("pipeline.min_fact_confidence", 0.3)),
            max_searches=int(config.get("pipeline.max_searches_per_subagent", 2)),
            max_answer_words=int(config.get("pipeline.max_answer_words", 8)),
        )
        self.dag_executor = DAGExecutor(self.investigator)
        self.synthesizer = Synthesizer(
            llm=synth_llm,
            max_answer_words=int(config.get("pipeline.max_answer_words", 8)),
        )

    async def run(self, question: str, question_id: str) -> PipelineResult:
        logger.info("AMAS v2 start: qid=%s", question_id)
        plan, planner_tokens = await self.planner.plan(question)

        trace = [
            StepTrace(
                step=0,
                action="plan",
                tokens=planner_tokens,
                route_decision=plan.complexity,
                route_confidence=plan.confidence,
                metadata={
                    "plan": plan.to_dict(),
                    "topology_shape": _topology_shape(plan.subgoals),
                },
            )
        ]

        if plan.complexity == "simple" or len(plan.subgoals) == 1:
            return await self._run_direct(question_id, question, plan, trace, planner_tokens)

        exec_result = await self.dag_executor.execute(plan)
        trace.extend(_renumber_trace(exec_result.trace, start=1))

        unresolved_node_ids = [
            node.id for node in plan.subgoals
            if exec_result.node_statuses.get(node.id) != "verified"
        ]
        if unresolved_node_ids:
            trace.append(
                StepTrace(
                    step=len(trace) + 1,
                    action="answer_blocked_pending_slots",
                    route_decision=plan.complexity,
                    metadata={
                        "unresolved_node_ids": unresolved_node_ids,
                        "node_statuses": exec_result.node_statuses,
                    },
                )
            )
            return self._build_result(
                question_id=question_id,
                question=question,
                answer="",
                trace=trace,
                planner_tokens=planner_tokens,
                exec_result=exec_result,
                route_decision=plan.complexity,
                route_confidence=plan.confidence,
                plan=plan,
                answer_type=plan.final_answer_type,
                justification="one or more slots remained unresolved",
            )

        terminal_capsules = [
            capsule
            for capsule in exec_result.capsules
            if capsule.subgoal_id in set(exec_result.terminal_ids)
        ]
        verified_terminal_capsules = [
            capsule for capsule in terminal_capsules
            if capsule.fact.slot_filled and capsule.answer and plan.final_answer_type.validate_span(capsule.answer)
        ]
        unresolved_terminal_ids = [
            terminal_id
            for terminal_id in exec_result.terminal_ids
            if exec_result.node_statuses.get(terminal_id) != "verified"
        ]

        if unresolved_terminal_ids:
            trace.append(
                StepTrace(
                    step=len(trace) + 1,
                    action="answer_blocked_pending_slots",
                    route_decision=plan.complexity,
                    metadata={
                        "unresolved_terminal_ids": unresolved_terminal_ids,
                        "node_statuses": exec_result.node_statuses,
                    },
                )
            )
            return self._build_result(
                question_id=question_id,
                question=question,
                answer="",
                trace=trace,
                planner_tokens=planner_tokens,
                exec_result=exec_result,
                route_decision=plan.complexity,
                route_confidence=plan.confidence,
                plan=plan,
                answer_type=plan.final_answer_type,
                justification="terminal slot unresolved",
            )

        if len(exec_result.terminal_ids) == 1 and len(verified_terminal_capsules) == 1:
            terminal = verified_terminal_capsules[0]
            trace.append(
                StepTrace(
                    step=len(trace) + 1,
                    action="answer",
                    route_decision=plan.complexity,
                    justification_confidence=terminal.fact.confidence,
                    metadata={
                        "answer": terminal.answer,
                        "source_subgoal_id": terminal.subgoal_id,
                        "node_statuses": exec_result.node_statuses,
                    },
                )
            )
            return self._build_result(
                question_id=question_id,
                question=question,
                answer=terminal.answer,
                trace=trace,
                planner_tokens=planner_tokens,
                exec_result=exec_result,
                route_decision=plan.complexity,
                route_confidence=plan.confidence,
                plan=plan,
                answer_type=plan.final_answer_type,
                justification=terminal.fact.text,
            )

        synth_obj, synth_tokens = await self.synthesizer.synthesize(
            question=question,
            capsules=exec_result.capsules,
            answer_type=plan.final_answer_type,
        )
        answer = _final_answer_from_synthesis(synth_obj, plan.final_answer_type)
        trace.append(
            StepTrace(
                step=len(trace) + 1,
                action="synthesize",
                tokens=synth_tokens,
                route_decision=plan.complexity,
                justification_confidence=_bounded_float(synth_obj.get("confidence", 0.0)),
                metadata={
                    "answer": answer,
                    "synthesis": synth_obj,
                    "node_statuses": exec_result.node_statuses,
                    "reasoning_tokens_excluding_chunks": (
                        planner_tokens + exec_result.subagent_tokens + synth_tokens
                        - exec_result.chunk_tokens
                    ),
                },
            )
        )

        return self._build_result(
            question_id=question_id,
            question=question,
            answer=answer,
            trace=trace,
            planner_tokens=planner_tokens + synth_tokens,
            exec_result=exec_result,
            route_decision=plan.complexity,
            route_confidence=plan.confidence,
            plan=plan,
            answer_type=plan.final_answer_type,
            justification=str(synth_obj.get("justification", "")),
        )

    async def _run_direct(
        self,
        question_id: str,
        question: str,
        plan,
        trace: list[StepTrace],
        planner_tokens: int,
    ) -> PipelineResult:
        node = plan.subgoals[0]
        capsule, subagent_tokens = await self.investigator.investigate_node(node)
        trace.append(
            StepTrace(
                step=1,
                action="direct",
                sub_question=node.question,
                fact_added=capsule.fact.slot_filled,
                tokens=subagent_tokens,
                slot_name=f"subgoal_{node.id}",
                route_decision="simple",
                justification_confidence=capsule.fact.confidence,
                metadata={
                    "answer": capsule.answer,
                    "support_ids": capsule.fact.support_ids,
                    "chunk_tokens": capsule.chunk_tokens,
                },
            )
        )
        total_tokens = planner_tokens + subagent_tokens
        reasoning_tokens = max(0, total_tokens - capsule.chunk_tokens)
        answer = capsule.answer or capsule.fact.answer_span
        if answer and not node.answer_type.validate_span(answer):
            answer = ""
        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=trace,
            num_subagent_calls=1,
            total_tokens=total_tokens,
            orchestrator_tokens=planner_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=[capsule.fact],
            retrieved_doc_ids=capsule.retrieved_doc_ids,
            retrieved_docs_total=capsule.retrieved_docs_total,
            route_decision="simple",
            route_confidence=plan.confidence,
            extras={
                "architecture": "saat_dag_v2",
                "answer_type": node.answer_type.value,
                "support_ids": capsule.fact.support_ids,
                "justification": capsule.fact.text,
                "n_searches": self.investigator.last_searches_used,
                "n_subagents": 1,
                "chunk_tokens": capsule.chunk_tokens,
                "reasoning_tokens": reasoning_tokens,
                "topology_shape": "single",
                "dag_levels": [[node.id]],
                "plan": plan.to_dict(),
            },
        )

    def _build_result(
        self,
        question_id: str,
        question: str,
        answer: str,
        trace: list[StepTrace],
        planner_tokens: int,
        exec_result,
        route_decision: str,
        route_confidence: float,
        plan,
        answer_type: AnswerType,
        justification: str,
    ) -> PipelineResult:
        total_tokens = planner_tokens + exec_result.subagent_tokens
        reasoning_tokens = max(0, total_tokens - exec_result.chunk_tokens)
        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=trace,
            num_subagent_calls=exec_result.n_subagents,
            total_tokens=total_tokens,
            orchestrator_tokens=planner_tokens,
            subagent_tokens=exec_result.subagent_tokens,
            facts_used=[capsule.fact for capsule in exec_result.capsules],
            retrieved_doc_ids=exec_result.retrieved_doc_ids,
            retrieved_docs_total=exec_result.retrieved_docs_total,
            route_decision=route_decision,
            route_confidence=route_confidence,
            extras={
                "architecture": "saat_dag_v2",
                "answer_type": answer_type.value,
                "support_ids": _support_ids(exec_result.capsules),
                "justification": justification,
                "n_searches": exec_result.n_searches,
                "n_subagents": exec_result.n_subagents,
                "chunk_tokens": exec_result.chunk_tokens,
                "reasoning_tokens": reasoning_tokens,
                "topology_shape": _topology_shape(plan.subgoals),
                "dag_levels": exec_result.levels,
                "plan": plan.to_dict(),
                "node_statuses": exec_result.node_statuses,
                "terminal_ids": exec_result.terminal_ids,
                "slot_outputs": [
                    {
                        "subgoal_id": capsule.subgoal_id,
                        "sub_question": capsule.sub_question,
                        "answer": capsule.answer,
                        "justification": capsule.fact.text,
                        "support_ids": capsule.fact.support_ids,
                        "status": exec_result.node_statuses.get(capsule.subgoal_id, "unknown"),
                    }
                    for capsule in exec_result.capsules
                ],
            },
        )


def _agent_llm(config: Config, agent: str, fallback: str) -> dict[str, Any]:
    raw = config.raw()
    agents = raw.get("agents", {}) or {}
    if agent in agents:
        return config.agent_llm(agent)
    merged = dict(raw.get("llm_defaults", {}) or {})
    merged.update(agents.get(fallback, {}) or {})
    return merged


def _final_answer_from_synthesis(
    synth_obj: dict[str, Any],
    answer_type: AnswerType,
) -> str:
    answer = str(synth_obj.get("answer_span", "")).strip()
    if answer and answer_type.validate_span(answer):
        return answer
    return ""


def _support_ids(capsules: list[EvidenceCapsule]) -> list[str]:
    ids: list[str] = []
    for capsule in capsules:
        for support_id in capsule.fact.support_ids:
            if support_id not in ids:
                ids.append(support_id)
    return ids


def _topology_shape(subgoals) -> str:
    if len(subgoals) <= 1:
        return "single"
    indep = sum(1 for node in subgoals if not node.depends_on)
    max_deps = max((len(node.depends_on) for node in subgoals), default=0)
    if indep == 1 and max_deps <= 1:
        return "chain"
    if indep > 1 and max_deps > 1:
        return "fan_in"
    if indep > 1:
        return "parallel"
    return "dag"


def _renumber_trace(trace: list[StepTrace], start: int) -> list[StepTrace]:
    for offset, item in enumerate(trace):
        item.step = start + offset
    return trace


def _bounded_float(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
