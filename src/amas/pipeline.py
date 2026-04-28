"""AMAS: structure-aware adaptive topology RAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .dag_executor import DAGExecutor, DAGExecutionResult
from .investigator import Investigator
from .llm import LLMClient, parse_json_object
from .planner import Planner
from .router import DifficultyRouter, RouterDecision
from .retriever import Retriever
from .types import AnswerType, EvidenceCapsule, ExecutionPlan, PipelineResult, StepTrace, SubgoalNode

logger = logging.getLogger(__name__)


class AMASPipeline:
    """Plan -> execute DAG -> strategist review loop -> return answer."""

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
        self.router_enabled = bool(config.get("pipeline.router_enabled", True))
        if self.router_enabled:
            self.router = DifficultyRouter(llm=planner_llm)
        else:
            self.router = None
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
            max_hop_attempts=int(config.get("pipeline.max_hop_attempts", 3)),
        )
        self.max_review_rounds = int(config.get("pipeline.max_review_rounds", 2))
        self.planner_llm = planner_llm
        self._review_template = (
            Path(__file__).parent / "prompts" / "strategist_review.txt"
        ).read_text(encoding="utf-8")

    async def run(self, question: str, question_id: str) -> PipelineResult:
        logger.info("AMAS start: qid=%s", question_id)

        router_tokens = 0
        router_decision: RouterDecision | None = None
        if self.router_enabled and self.router is not None:
            router_decision, router_tokens = await self.router.classify(question)
            if router_decision.complexity == "easy":
                return await self._run_router_easy(
                    question_id=question_id,
                    question=question,
                    router_decision=router_decision,
                    router_tokens=router_tokens,
                )

        plan, planner_tokens = await self.planner.plan(question)
        planner_tokens = planner_tokens + router_tokens

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

        exec_result = await self.dag_executor.execute(plan, original_question=question)
        trace.extend(_renumber_trace(exec_result.trace, start=1))

        # Strategist review loop: only trigger when the FINAL answer is missing.
        # Preserves already-verified capsules across re-executions.
        final_subgoal_id = plan.subgoals[-1].id
        for review_round in range(self.max_review_rounds):
            final_cap = _capsule_by_subgoal_id(exec_result.capsules, final_subgoal_id)
            final_ok = (
                final_cap is not None
                and exec_result.node_statuses.get(final_subgoal_id) == "verified"
                and final_cap.answer
            )
            if final_ok:
                break

            review_action, review_tokens = await self._strategist_review(
                question, plan, exec_result,
            )
            planner_tokens += review_tokens
            action = review_action.get("action", "accept")

            trace.append(
                StepTrace(
                    step=len(trace) + 1,
                    action="review",
                    tokens=review_tokens,
                    route_decision=f"review_round_{review_round}",
                    metadata=review_action,
                )
            )

            if action == "accept":
                answer = str(review_action.get("answer", "")).strip()
                if answer:
                    return self._build_result(
                        question_id=question_id,
                        question=question,
                        answer=answer,
                        trace=trace,
                        planner_tokens=planner_tokens,
                        exec_result=exec_result,
                        route_decision="review_accept",
                        route_confidence=plan.confidence,
                        plan=plan,
                        answer_type=plan.final_answer_type,
                        justification=review_action.get("reasoning", ""),
                    )
                break

            # Collect verified capsules to preserve across re-execution
            verified = {
                nid: _capsule_by_subgoal_id(exec_result.capsules, nid)
                for nid, status in exec_result.node_statuses.items()
                if status == "verified" and _capsule_by_subgoal_id(exec_result.capsules, nid)
            }

            if action == "revise_hop":
                hop_id = int(review_action.get("hop_id", -1))
                new_q = str(review_action.get("new_question", "")).strip()
                if 0 <= hop_id < len(plan.subgoals) and new_q:
                    plan.subgoals[hop_id] = SubgoalNode(
                        id=plan.subgoals[hop_id].id,
                        question=new_q,
                        depends_on=plan.subgoals[hop_id].depends_on,
                        answer_type=plan.subgoals[hop_id].answer_type,
                        rationale=plan.subgoals[hop_id].rationale,
                    )
                    verified.pop(hop_id, None)
                    # Also invalidate downstream hops that depend on the revised one
                    for node in plan.subgoals:
                        if hop_id in node.depends_on:
                            verified.pop(node.id, None)
                    exec_result = await self.dag_executor.execute(plan, prior_capsules=verified, original_question=question)
                    trace.extend(_renumber_trace(exec_result.trace, start=len(trace) + 1))
                    final_subgoal_id = plan.subgoals[-1].id
                    continue

            if action == "add_hop":
                new_q = str(review_action.get("question", "")).strip()
                deps = review_action.get("depends_on", [])
                at = str(review_action.get("answer_type", "entity"))
                if new_q:
                    new_id = max(n.id for n in plan.subgoals) + 1
                    plan.subgoals.append(SubgoalNode(
                        id=new_id,
                        question=new_q,
                        depends_on=[int(d) for d in deps if isinstance(d, (int, float))],
                        answer_type=AnswerType.coerce(at),
                        rationale="Added by strategist review.",
                    ))
                    exec_result = await self.dag_executor.execute(plan, prior_capsules=verified, original_question=question)
                    trace.extend(_renumber_trace(exec_result.trace, start=len(trace) + 1))
                    final_subgoal_id = plan.subgoals[-1].id
                    continue

            break

        # Extract final answer from the last subgoal
        final_subgoal_id = plan.subgoals[-1].id
        final_capsule = _capsule_by_subgoal_id(exec_result.capsules, final_subgoal_id)

        if (
            final_capsule is None
            or exec_result.node_statuses.get(final_subgoal_id) != "verified"
            or not final_capsule.answer
        ):
            # Try to find ANY verified capsule as a fallback
            best = _best_verified_capsule(exec_result, plan)
            if best and best.answer:
                trace.append(
                    StepTrace(
                        step=len(trace) + 1,
                        action="answer_fallback",
                        route_decision=plan.complexity,
                        metadata={
                            "source_subgoal_id": best.subgoal_id,
                            "node_statuses": exec_result.node_statuses,
                        },
                    )
                )
                return self._build_result(
                    question_id=question_id,
                    question=question,
                    answer=best.answer,
                    trace=trace,
                    planner_tokens=planner_tokens,
                    exec_result=exec_result,
                    route_decision="fallback_best_capsule",
                    route_confidence=plan.confidence,
                    plan=plan,
                    answer_type=plan.final_answer_type,
                    justification=best.fact.text,
                )

            trace.append(
                StepTrace(
                    step=len(trace) + 1,
                    action="answer_blocked",
                    route_decision=plan.complexity,
                    metadata={"node_statuses": exec_result.node_statuses},
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
                justification="all recovery attempts exhausted",
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

    async def _strategist_review(
        self,
        question: str,
        plan: ExecutionPlan,
        exec_result: DAGExecutionResult,
    ) -> tuple[dict, int]:
        """Ask the planner LLM to review the DAG results and decide a recovery action."""
        summary_parts = []
        for node in plan.subgoals:
            status = exec_result.node_statuses.get(node.id, "unknown")
            capsule = _capsule_by_subgoal_id(exec_result.capsules, node.id)
            answer = capsule.answer if capsule else ""
            failure = capsule.failure_reason if capsule else "no capsule"
            justification = capsule.fact.text if capsule else ""
            evidence_preview = ""
            if capsule and capsule.evidence_snippets:
                evidence_preview = " | ".join(
                    s.get("excerpt", "")[:120] for s in capsule.evidence_snippets[:2]
                )
            summary_parts.append(
                f"Hop {node.id} [{status}]: {node.question}\n"
                f"  Answer: {answer or '(empty)'}\n"
                f"  Justification: {justification[:150] or '(none)'}\n"
                f"  Failure reason: {failure or '(none)'}\n"
                f"  Evidence preview: {evidence_preview[:250] or '(none)'}"
            )

        prompt = self._review_template.format(
            question=question.strip(),
            execution_summary="\n\n".join(summary_parts),
        )
        resp = await self.planner_llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
        )
        parsed = parse_json_object(resp.content)
        return parsed, resp.total_tokens

    async def _run_direct(
        self,
        question_id: str,
        question: str,
        plan,
        trace: list[StepTrace],
        planner_tokens: int,
    ) -> PipelineResult:
        node = plan.subgoals[0]
        capsule, subagent_tokens = await self.investigator.investigate_node(
            node,
            parent_question=question,
        )
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

    async def _run_router_easy(
        self,
        question_id: str,
        question: str,
        router_decision: RouterDecision,
        router_tokens: int,
    ) -> PipelineResult:
        """Single-agent path: investigator answers the original question directly."""
        capsule, subagent_tokens = await self.investigator.investigate(
            sub_question=question,
            expected_answer_type="entity",
            slot_name="router_easy",
            parent_question=question,
        )
        total_tokens = router_tokens + subagent_tokens
        reasoning_tokens = max(0, subagent_tokens - capsule.chunk_tokens)
        trace: list[StepTrace] = [
            StepTrace(
                step=0,
                action="route",
                tokens=router_tokens,
                route_decision="easy",
                route_confidence=router_decision.confidence,
                metadata={"router": router_decision.raw, "reasoning": router_decision.reasoning},
            ),
            StepTrace(
                step=1,
                action="direct",
                sub_question=question,
                fact_added=capsule.fact.slot_filled,
                tokens=subagent_tokens,
                slot_name="router_easy",
                route_decision="simple",
                justification_confidence=capsule.fact.confidence,
                metadata={
                    "answer": capsule.answer,
                    "support_ids": capsule.fact.support_ids,
                    "evidence_snippets": capsule.evidence_snippets,
                    "chunk_tokens": capsule.chunk_tokens,
                },
            ),
        ]
        answer = capsule.answer or capsule.fact.answer_span
        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=trace,
            num_subagent_calls=1,
            total_tokens=total_tokens,
            orchestrator_tokens=router_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=[capsule.fact],
            retrieved_doc_ids=capsule.retrieved_doc_ids,
            retrieved_docs_total=capsule.retrieved_docs_total,
            route_decision="simple",
            route_confidence=router_decision.confidence,
            extras={
                "architecture": "saat_dag_v2_router",
                "router_decision": router_decision.complexity,
                "router_reasoning": router_decision.reasoning,
                "router_tokens": router_tokens,
                "answer_type": "entity",
                "support_ids": capsule.fact.support_ids,
                "evidence_snippets": capsule.evidence_snippets,
                "justification": capsule.fact.text,
                "n_searches": self.investigator.last_searches_used,
                "n_subagents": 1,
                "chunk_tokens": capsule.chunk_tokens,
                "reasoning_tokens": reasoning_tokens,
                "topology_shape": "single",
                "dag_levels": [[0]],
                "plan": {"complexity": "easy", "subgoals": []},
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


def _best_verified_capsule(
    exec_result: DAGExecutionResult,
    plan: ExecutionPlan,
) -> EvidenceCapsule | None:
    """Find the best verified capsule, preferring later subgoals."""
    for node in reversed(plan.subgoals):
        if exec_result.node_statuses.get(node.id) == "verified":
            cap = _capsule_by_subgoal_id(exec_result.capsules, node.id)
            if cap and cap.answer:
                return cap
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
