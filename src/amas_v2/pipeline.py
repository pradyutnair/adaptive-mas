"""AMAS v2 pipeline: Plan -> DAG Execute -> Synthesize with fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .dag_executor import DAGExecutor, DAGResult
from .investigator import Investigator
from .llm import LLMClient, parse_json_object
from .planner import Planner
from .retriever import Retriever
from .synthesizer import Synthesizer
from .types import (
    AnswerType, EvidenceCapsule, ExecutionPlan,
    PipelineResult, StepTrace, SubgoalNode,
)

logger = logging.getLogger(__name__)


class AMASv2Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config

        planner_llm = LLMClient.from_config(config.agent_llm("planner"))
        inv_llm = LLMClient.from_config(config.agent_llm("investigator"))
        synth_llm = LLMClient.from_config(config.agent_llm("synthesizer"))

        ret_cfg = config.raw().get("retriever", {}) or {}
        self.retriever = Retriever(
            base_url=ret_cfg.get("base_url", "http://node408:8003"),
            default_top_k=int(ret_cfg.get("top_k", 5)),
            timeout_seconds=float(ret_cfg.get("timeout_seconds", 30)),
            request_format=str(ret_cfg.get("request_format", "batch")),
        )

        self.planner = Planner(
            llm=planner_llm,
            max_subgoals=int(config.get("pipeline.max_subgoals", 5)),
        )
        self.investigator = Investigator(
            llm=inv_llm,
            retriever=self.retriever,
            top_k=int(ret_cfg.get("top_k", 7)),
            max_searches=int(config.get("pipeline.max_searches_per_subagent", 3)),
            max_evidence_hits=int(config.get("pipeline.max_evidence_hits", 5)),
            max_excerpt_chars=int(config.get("pipeline.max_excerpt_chars", 500)),
        )
        self.dag_executor = DAGExecutor(
            self.investigator,
            max_hop_attempts=int(config.get("pipeline.max_hop_attempts", 3)),
        )
        self.synthesizer = Synthesizer(llm=synth_llm)
        self.planner_llm = planner_llm
        self.max_review_rounds = int(config.get("pipeline.max_review_rounds", 1))
        self._review_template = (
            Path(__file__).parent / "prompts" / "strategist_review.txt"
        ).read_text(encoding="utf-8")

    async def run(self, question: str, question_id: str) -> PipelineResult:
        logger.info("AMAS-v2 start: qid=%s", question_id)

        plan, planner_tokens = await self.planner.plan(question)
        trace = [StepTrace(
            step=0, action="plan", tokens=planner_tokens,
            route_decision=plan.complexity,
            metadata={"plan": plan.to_dict()},
        )]

        if plan.complexity == "simple" or len(plan.subgoals) == 1:
            return await self._run_direct(question_id, question, plan, trace, planner_tokens)

        exec_result = await self.dag_executor.execute(plan, original_question=question)
        trace.extend(self._renumber(exec_result.trace, len(trace)))
        total_sub_tokens = exec_result.subagent_tokens

        final_cap = exec_result.capsules_by_id.get(plan.subgoals[-1].id)
        final_ok = (
            final_cap is not None
            and exec_result.node_statuses.get(plan.subgoals[-1].id) == "verified"
            and final_cap.answer
        )

        if not final_ok and self.max_review_rounds > 0:
            review_action, rev_tokens = await self._strategist_review(question, plan, exec_result)
            planner_tokens += rev_tokens
            trace.append(StepTrace(
                step=len(trace), action="review", tokens=rev_tokens,
                metadata=review_action,
            ))
            action = review_action.get("action", "accept")
            if action == "accept" and review_action.get("answer", "").strip():
                return self._build_result(
                    question_id, question, review_action["answer"].strip(),
                    trace, planner_tokens, exec_result, plan, "review_accept",
                )
            elif action == "revise_hop":
                hop_id = int(review_action.get("hop_id", -1))
                new_q = str(review_action.get("new_question", "")).strip()
                if 0 <= hop_id < len(plan.subgoals) and new_q:
                    plan.subgoals[hop_id] = SubgoalNode(
                        id=plan.subgoals[hop_id].id, question=new_q,
                        depends_on=plan.subgoals[hop_id].depends_on,
                        answer_type=plan.subgoals[hop_id].answer_type,
                        rationale=plan.subgoals[hop_id].rationale,
                    )
                    verified = {
                        nid: cap for nid, cap in exec_result.capsules_by_id.items()
                        if exec_result.node_statuses.get(nid) == "verified" and nid != hop_id
                    }
                    for n in plan.subgoals:
                        if hop_id in n.depends_on:
                            verified.pop(n.id, None)
                    exec2 = await self.dag_executor.execute(
                        plan, original_question=question, prior_capsules=verified,
                    )
                    trace.extend(self._renumber(exec2.trace, len(trace)))
                    total_sub_tokens += exec2.subagent_tokens
                    exec_result = exec2
                    final_cap = exec_result.capsules_by_id.get(plan.subgoals[-1].id)
                    final_ok = (
                        final_cap is not None
                        and exec_result.node_statuses.get(plan.subgoals[-1].id) == "verified"
                        and final_cap.answer
                    )

        facts = self._collect_facts(plan, exec_result)
        last_ev = ""
        if final_cap and final_cap.evidence_snippets:
            last_ev = " | ".join(s.get("excerpt", "")[:200] for s in final_cap.evidence_snippets[:2])

        if final_ok and final_cap:
            synth_answer, synth_tokens = await self.synthesizer.synthesize(
                question, facts, last_ev,
            )
            total_sub_tokens += synth_tokens
            trace.append(StepTrace(
                step=len(trace), action="synthesize", tokens=synth_tokens,
                metadata={"raw_answer": final_cap.answer, "synth_answer": synth_answer},
            ))
            answer = synth_answer or final_cap.answer
        elif facts:
            synth_answer, synth_tokens = await self.synthesizer.synthesize(
                question, facts, last_ev,
            )
            total_sub_tokens += synth_tokens
            trace.append(StepTrace(
                step=len(trace), action="synthesize_partial", tokens=synth_tokens,
                metadata={"synth_answer": synth_answer},
            ))
            answer = synth_answer
        else:
            answer = ""

        if not answer:
            fallback_answer, fb_tokens = await self._fallback_direct(question)
            total_sub_tokens += fb_tokens
            trace.append(StepTrace(
                step=len(trace), action="fallback_direct", tokens=fb_tokens,
                metadata={"fallback_answer": fallback_answer},
            ))
            answer = fallback_answer

        return self._build_result(
            question_id, question, answer, trace,
            planner_tokens, exec_result, plan, plan.complexity,
        )

    async def _run_direct(
        self, qid: str, question: str, plan: ExecutionPlan,
        trace: list[StepTrace], planner_tokens: int,
    ) -> PipelineResult:
        node = plan.subgoals[0]
        capsule, sub_tokens = await self.investigator.investigate_node(
            node, parent_question=question,
        )
        trace.append(StepTrace(
            step=1, action="direct", sub_question=node.question,
            fact_added=capsule.fact.slot_filled, tokens=sub_tokens,
            metadata={"answer": capsule.answer, "support_ids": capsule.fact.support_ids},
        ))
        answer = capsule.answer or capsule.fact.answer_span

        if not answer:
            fallback_answer, fb_tokens = await self._fallback_direct(question)
            sub_tokens += fb_tokens
            trace.append(StepTrace(
                step=2, action="fallback_direct", tokens=fb_tokens,
                metadata={"fallback_answer": fallback_answer},
            ))
            answer = fallback_answer

        return PipelineResult(
            question_id=qid, question=question, answer=answer,
            step_trace=trace, num_subagent_calls=1,
            total_tokens=planner_tokens + sub_tokens,
            orchestrator_tokens=planner_tokens,
            subagent_tokens=sub_tokens,
            facts_used=[capsule.fact],
            retrieved_doc_ids=capsule.retrieved_doc_ids,
            retrieved_docs_total=capsule.retrieved_docs_total,
            route_decision="simple",
            extras={"plan": plan.to_dict()},
        )

    async def _fallback_direct(self, question: str) -> tuple[str, int]:
        node = SubgoalNode(id=0, question=question, answer_type=AnswerType.ENTITY)
        capsule, tokens = await self.investigator.investigate_node(
            node, hint="Answer the original question directly.",
            parent_question=question,
        )
        return capsule.answer or capsule.fact.answer_span, tokens

    async def _strategist_review(
        self, question: str, plan: ExecutionPlan, exec_result: DAGResult,
    ) -> tuple[dict, int]:
        summary_parts = []
        for node in plan.subgoals:
            status = exec_result.node_statuses.get(node.id, "unknown")
            cap = exec_result.capsules_by_id.get(node.id)
            answer = cap.answer if cap else ""
            failure = cap.failure_reason if cap else "no capsule"
            justification = cap.fact.text if cap else ""
            summary_parts.append(
                f"Hop {node.id} [{status}]: {node.question}\n"
                f"  Answer: {answer or '(empty)'}\n"
                f"  Justification: {justification[:150] or '(none)'}\n"
                f"  Failure: {failure or '(none)'}"
            )
        prompt = self._review_template.format(
            question=question.strip(),
            execution_summary="\n\n".join(summary_parts),
        )
        resp = await self.planner_llm.chat(
            messages=[{"role": "user", "content": prompt}], max_tokens=400,
        )
        return parse_json_object(resp.content), resp.total_tokens

    def _collect_facts(self, plan: ExecutionPlan, exec_result: DAGResult) -> list[dict]:
        facts = []
        for node in plan.subgoals:
            cap = exec_result.capsules_by_id.get(node.id)
            if cap and cap.answer:
                facts.append({
                    "step": node.id,
                    "question": node.question,
                    "answer": cap.answer,
                    "justification": cap.fact.text,
                })
        return facts

    def _build_result(
        self, qid: str, question: str, answer: str,
        trace: list[StepTrace], planner_tokens: int,
        exec_result: DAGResult, plan: ExecutionPlan, route: str,
    ) -> PipelineResult:
        total = planner_tokens + exec_result.subagent_tokens
        extra_tokens = sum(t.tokens for t in trace if "synth" in t.action or "fallback" in t.action)
        total += extra_tokens
        return PipelineResult(
            question_id=qid, question=question, answer=answer,
            step_trace=trace,
            num_subagent_calls=exec_result.n_subagents,
            total_tokens=total,
            orchestrator_tokens=planner_tokens,
            subagent_tokens=exec_result.subagent_tokens + extra_tokens,
            facts_used=[c.fact for c in exec_result.capsules if c.fact.slot_filled],
            retrieved_doc_ids=exec_result.retrieved_doc_ids,
            retrieved_docs_total=exec_result.retrieved_docs_total,
            route_decision=route,
            extras={
                "architecture": "amas_v2",
                "plan": plan.to_dict(),
                "node_statuses": exec_result.node_statuses,
            },
        )

    @staticmethod
    def _renumber(trace: list[StepTrace], offset: int) -> list[StepTrace]:
        for i, t in enumerate(trace):
            t.step = offset + i
        return trace
