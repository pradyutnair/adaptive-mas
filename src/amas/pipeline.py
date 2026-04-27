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
from .types import AnswerType, EvidenceCapsule, PipelineResult, StepTrace, SubgoalNode

logger = logging.getLogger(__name__)


class AMASPipeline:
    """Plan -> execute DAG -> return the final planned slot."""

    def __init__(self, config: Config) -> None:
        self.config = config

        planner_llm = LLMClient.from_config(_agent_llm(config, "planner", "orchestrator"))
        inv_llm = LLMClient.from_config(config.agent_llm("investigator"))

        ret_cfg = config.raw().get("retriever", {}) or {}
        self.retriever = Retriever(
            base_url=ret_cfg.get("base_url", "http://node408:8003"),
            default_top_k=int(ret_cfg.get("top_k", 10)),
            timeout_seconds=float(ret_cfg.get("timeout_seconds", 30)),
            request_format=str(ret_cfg.get("request_format", "batch")),
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
            max_evidence_hits=int(config.get("pipeline.max_evidence_hits", 6)),
            max_excerpt_chars=int(config.get("pipeline.max_excerpt_chars", 600)),
        )
        self.dag_executor = DAGExecutor(
            self.investigator,
            max_hop_attempts=int(config.get("pipeline.max_hop_attempts", 2)),
        )
        self.final_recovery_attempts = int(config.get("pipeline.final_recovery_attempts", 2))
        self.final_recovery_top_k = int(
            config.get("pipeline.final_recovery_top_k", ret_cfg.get("top_k", 10))
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
            recovery = await self._try_direct_recovery(question, plan.final_answer_type, len(trace) + 1)
            if recovery is not None:
                capsule, recovery_tokens, recovery_trace = recovery
                trace.extend(recovery_trace)
                self._merge_recovery(exec_result, capsule, recovery_tokens)
                return self._build_result(
                    question_id=question_id,
                    question=question,
                    answer=capsule.answer,
                    trace=trace,
                    planner_tokens=planner_tokens,
                    exec_result=exec_result,
                    route_decision="direct_recovery",
                    route_confidence=plan.confidence,
                    plan=plan,
                    answer_type=plan.final_answer_type,
                    justification=capsule.fact.text,
                )
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

        final_subgoal_id = plan.subgoals[-1].id
        final_capsule = _capsule_by_subgoal_id(exec_result.capsules, final_subgoal_id)
        if (
            final_capsule is None
            or exec_result.node_statuses.get(final_subgoal_id) != "verified"
            or not final_capsule.answer
        ):
            trace.append(
                StepTrace(
                    step=len(trace) + 1,
                    action="answer_blocked_pending_slots",
                    route_decision=plan.complexity,
                    metadata={
                        "unresolved_final_slot_id": final_subgoal_id,
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
                justification="final slot unresolved",
            )

        trace.append(
            StepTrace(
                step=len(trace) + 1,
                action="answer",
                route_decision=plan.complexity,
                justification_confidence=final_capsule.fact.confidence,
                metadata={
                    "answer": final_capsule.answer,
                    "source_subgoal_id": final_capsule.subgoal_id,
                    "support_ids": final_capsule.fact.support_ids,
                    "evidence_snippets": final_capsule.evidence_snippets,
                    "node_statuses": exec_result.node_statuses,
                },
            )
        )

        return self._build_result(
            question_id=question_id,
            question=question,
            answer=final_capsule.answer,
            trace=trace,
            planner_tokens=planner_tokens,
            exec_result=exec_result,
            route_decision=plan.complexity,
            route_confidence=plan.confidence,
            plan=plan,
            answer_type=plan.final_answer_type,
            justification=final_capsule.fact.text,
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
                    "evidence_snippets": capsule.evidence_snippets,
                    "chunk_tokens": capsule.chunk_tokens,
                },
            )
        )
        total_tokens = planner_tokens + subagent_tokens
        reasoning_tokens = max(0, total_tokens - capsule.chunk_tokens)
        answer = capsule.answer or capsule.fact.answer_span
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
                "evidence_snippets": capsule.evidence_snippets,
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

    async def _try_direct_recovery(
        self,
        question: str,
        answer_type: AnswerType,
        start_step: int,
    ) -> tuple[EvidenceCapsule, int, list[StepTrace]] | None:
        if self.final_recovery_attempts <= 0:
            return None

        node = SubgoalNode(
            id=0,
            question=question,
            depends_on=[],
            answer_type=answer_type,
            rationale="Direct recovery on the original question after planned hops failed.",
        )
        query = question
        total_tokens = 0
        trace: list[StepTrace] = []
        last_capsule: EvidenceCapsule | None = None
        step = start_step

        for attempt in range(1, self.final_recovery_attempts + 1):
            capsule, tokens = await self.investigator.investigate_node(
                node,
                hint="Answer the original question directly. Use the retrieved evidence to resolve missing bridge facts internally.",
                slot_name="direct_recovery",
                top_k_override=self.final_recovery_top_k,
                query_override=query,
            )
            total_tokens += tokens
            last_capsule = capsule
            trace.append(
                StepTrace(
                    step=step,
                    action="direct",
                    sub_question=question,
                    fact_added=capsule.fact.slot_filled,
                    tokens=tokens,
                    slot_name="direct_recovery",
                    route_decision="direct_recovery",
                    justification_confidence=capsule.fact.confidence,
                    metadata={
                        "attempt": attempt,
                        "query": query,
                        "answer": capsule.answer,
                        "support_ids": capsule.fact.support_ids,
                        "failure_reason": capsule.failure_reason,
                        "evidence_snippets": capsule.evidence_snippets,
                    },
                )
            )
            step += 1
            if capsule.fact.slot_filled and capsule.answer:
                return capsule, total_tokens, trace
            if attempt >= self.final_recovery_attempts:
                break
            query, rewrite_tokens = await self.investigator.rewrite_query(
                node=node,
                hint="Direct recovery for the original question after planned decomposition failed.",
                previous_query=query,
                previous_answer=capsule.answer if capsule else "",
                previous_justification=capsule.fact.text if capsule else "",
                rejection_reason=capsule.failure_reason if capsule else "evidence was insufficient",
            )
            total_tokens += rewrite_tokens
            trace.append(
                StepTrace(
                    step=step,
                    action="rewrite",
                    sub_question=question,
                    tokens=rewrite_tokens,
                    slot_name="direct_recovery",
                    route_decision="direct_recovery_retry",
                    metadata={
                        "attempt": attempt,
                        "next_query": query,
                        "failure_reason": capsule.failure_reason if capsule else "",
                    },
                )
            )
            step += 1

        return None

    @staticmethod
    def _merge_recovery(exec_result, capsule: EvidenceCapsule, tokens: int) -> None:
        exec_result.capsules.append(capsule)
        exec_result.subagent_tokens += tokens
        exec_result.n_subagents += 1
        exec_result.n_searches += len(capsule.search_queries)
        exec_result.node_statuses[capsule.subgoal_id] = "verified"
        for doc_id in capsule.retrieved_doc_ids:
            if doc_id not in exec_result.retrieved_doc_ids:
                exec_result.retrieved_doc_ids.append(doc_id)
        exec_result.retrieved_docs_total += capsule.retrieved_docs_total

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
                "retry_count": exec_result.retry_count,
                "blackboard_state": exec_result.blackboard_state,
                "slot_outputs": [
                    {
                        "subgoal_id": capsule.subgoal_id,
                        "sub_question": capsule.sub_question,
                        "answer": capsule.answer,
                        "justification": capsule.fact.text,
                        "support_ids": capsule.fact.support_ids,
                        "evidence_snippets": capsule.evidence_snippets,
                        "failure_reason": capsule.failure_reason,
                        "search_queries": capsule.search_queries,
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


def _support_ids(capsules: list[EvidenceCapsule]) -> list[str]:
    ids: list[str] = []
    for capsule in capsules:
        for support_id in capsule.fact.support_ids:
            if support_id not in ids:
                ids.append(support_id)
    return ids


def _capsule_by_subgoal_id(
    capsules: list[EvidenceCapsule],
    subgoal_id: int,
) -> EvidenceCapsule | None:
    for capsule in reversed(capsules):
        if capsule.subgoal_id == subgoal_id:
            return capsule
    return None


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


