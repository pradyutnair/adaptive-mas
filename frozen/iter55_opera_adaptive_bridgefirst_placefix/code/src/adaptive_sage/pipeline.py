"""Adaptive Recursive SAGE pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient

from .fact_memory import FactMemory
from .investigator import Investigator
from .orchestrator import Orchestrator
from .types import EvidenceCapsule, PipelineResult, StepTrace

logger = logging.getLogger(__name__)


class AdaptiveRecursivePipeline:
    """Main pipeline for Adaptive Recursive SAGE."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.variant: str = str(config.get("variant", ""))

        llm_cfg = config.get("llm", {})
        self.llm_client = LLMClient(
            model=llm_cfg.get("model", "Qwen/Qwen3-8B"),
            api_key=llm_cfg.get("api_key", "EMPTY"),
            base_url=llm_cfg.get("base_url", "http://localhost:8001/v1"),
            temperature=llm_cfg.get("temperature", 0.6),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            chat_template_kwargs=llm_cfg.get("chat_template_kwargs"),
        )

        self.orchestrator = Orchestrator(config, self.llm_client)
        self.investigator = Investigator(config, self.llm_client)

        self.max_steps: int = config.get("orchestrator.max_steps", 4)
        self.max_verify_calls: int = config.get("orchestrator.max_verify_calls", 1)
        self.fact_memory_capacity: int = config.get("fact_memory.capacity", 4)
        self.fact_memory_strategy: str = str(
            config.get("fact_memory.strategy", "fifo")
        )
        self.direct_answer_threshold: float = float(
            config.get("adaptive.direct_answer_threshold", 0.9)
        )
        self.max_total_tokens: int = int(config.get("budget.max_total_tokens", 0) or 0)
        self.final_answer_threshold: float = float(
            config.get("adaptive.final_answer_threshold", 0.75)
        )
        self.use_routed_controller: bool = bool(
            config.get("adaptive.use_routed_controller", False)
        )
        self.use_hybrid_chain_exec: bool = bool(
            config.get("adaptive.use_hybrid_chain_exec", False)
        )
        self.use_opera_adaptive: bool = bool(
            config.get("adaptive.use_opera_adaptive", False)
        )
        self.route_direct_threshold: float = float(
            config.get("adaptive.route_direct_threshold", 0.85)
        )
        self.route_probe_threshold: float = float(
            config.get("adaptive.route_probe_threshold", 0.55)
        )
        self.require_support_for_final_answer: bool = bool(
            config.get("adaptive.require_support_for_final_answer", True)
        )
        self.answer_justification_threshold: float = float(
            config.get("adaptive.answer_justification_threshold", 0.6)
        )
        self.recurse_min_depth_conf_threshold: float = float(
            config.get("adaptive.recurse_min_depth_confidence_threshold", 0.7)
        )
        self.auto_verify_threshold: float = float(
            config.get("adaptive.auto_verify_threshold", 0.7)
        )
        self.enable_bootstrap_short_circuit: bool = bool(
            config.get("adaptive.enable_bootstrap_short_circuit", False)
        )
        self.enable_parallel_independent_hops: bool = bool(
            config.get("adaptive.enable_parallel_independent_hops", True)
        )
        self.force_planned_recurse_execution: bool = bool(
            config.get("adaptive.force_planned_recurse_execution", False)
        )
        self.use_deterministic_plan_queries: bool = bool(
            config.get("adaptive.use_deterministic_plan_queries", False)
        )
        self.enable_answer_verification: bool = bool(
            config.get("adaptive.enable_answer_verification", False)
        )
        self.max_parallel_hops: int = int(
            config.get("adaptive.max_parallel_hops", 2)
        )
        self.max_trace_actions: int = int(
            config.get("adaptive.max_trace_actions", 0)
        )
        self.refine_budget_per_slot: int = int(
            config.get("adaptive.refine_budget_per_slot", 1)
        )
        self.min_fact_confidence: float = float(
            config.get("investigator.min_fact_confidence", 0.6)
        )

        self.ablation_force_spawn: bool = bool(config.get("ablation.force_spawn", False))
        self.ablation_upfront_decomposition: bool = bool(
            config.get("ablation.upfront_decomposition", False)
        )
        self.ablation_no_verify: bool = bool(config.get("ablation.no_verify", False))
        self.ablation_always_verify: bool = bool(
            config.get("ablation.always_verify", False)
        )

    async def run(self, question: str, question_id: str) -> PipelineResult:
        """Execute the adaptive recursive pipeline on a question."""
        logger.info("Pipeline start: question_id=%s, max_steps=%d", question_id, self.max_steps)

        if self.max_steps == 0:
            result = await self._run_s0(question, question_id)
        elif self.ablation_upfront_decomposition:
            result = await self._run_upfront_decomposition(question, question_id)
        elif self.use_opera_adaptive:
            result = await self._run_adaptive_opera_plan_exec(question, question_id)
        elif self.use_hybrid_chain_exec:
            result = await self._run_hybrid_chain_exec(question, question_id)
        elif self.use_routed_controller:
            result = await self._run_adaptive_m11(question, question_id)
        else:
            result = await self._run_adaptive(question, question_id)

        logger.info(
            "Pipeline done: question_id=%s, answer=%r, subagents=%d, verify=%d, tokens=%d",
            question_id,
            result.answer[:80],
            result.num_subagent_calls,
            result.num_verify_calls,
            result.total_tokens,
        )
        return result

    async def _run_routed_iter26(self, question: str, question_id: str) -> PipelineResult:
        """Backward-compatible entrypoint for iter26."""
        return await self._run_adaptive_m11(question, question_id)

    async def _run_hybrid_chain_exec(self, question: str, question_id: str) -> PipelineResult:
        """Adaptive outer route with strict OPERA-style chain execution inside recurse."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        expected_answer_type = self._expected_answer_type(question)
        target_profile = self._target_profile(question)
        step_trace: list[StepTrace] = []
        total_tokens = 0
        orchestrator_tokens = 0
        subagent_tokens = 0
        subagent_calls = 0
        verify_count = 0
        duplicate_subquestion_count = 0
        retrieved_doc_ids: list[str] = []
        retrieved_docs_total = 0

        route, route_tokens = await self.orchestrator.route_with_usage(
            question=question,
            target_profile=target_profile,
        )
        if self._should_force_bridge_first_route(question, route):
            route = self._force_bridge_first_route(question, route, target_profile)
        total_tokens += route_tokens
        orchestrator_tokens += route_tokens

        slot_state = self._initialise_slot_state(route, target_profile)
        step_trace.append(
            StepTrace(
                step=0,
                action="route",
                tokens=route_tokens,
                route_decision=route["action"],
                route_confidence=route["confidence"],
                route_draft_answer=route["draft_answer"],
                metadata={
                    "expected_answer_type": expected_answer_type,
                    "answer_type": route["answer_type"],
                    "target_slot": route["target_slot"],
                    "retrieval_query": route.get("retrieval_query", ""),
                    "pending_slots": self._slot_snapshot(slot_state),
                    "hybrid_chain_exec": True,
                },
            )
        )

        step = 1
        final_slot = self._final_slot_name(slot_state)
        route_action = str(route.get("action", "")).strip().lower()

        if (
            route_action == "single_probe"
            and len(slot_state) <= 1
            and float(route.get("confidence", 0.0)) >= self.route_probe_threshold
        ):
            probe_query = question.strip()
            retrieval_query = str(route.get("retrieval_query", "")).strip() or probe_query
            capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                sub_question=probe_query,
                goal=str(route.get("goal", "")).strip() or f"Answer the question directly. {target_profile}",
                prior_facts=[],
                retrieval_query=retrieval_query or None,
                slot_name=final_slot,
                slot_hint=self._slot_hint(slot_state, final_slot),
            )
            total_tokens += investigate_tokens
            subagent_tokens += investigate_tokens
            subagent_calls += 1
            retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                retrieved_doc_ids,
                retrieved_docs_total,
                capsule,
            )
            fact_added = self._add_fact(memory, capsule, step=step, slot_name=final_slot)
            slot_idx, slot_value, slot_confidence = self._best_slot_value_with_index(
                memory.get_all(),
                final_slot,
                expected_answer_type=expected_answer_type,
            )
            for slot in slot_state:
                if str(slot.get("slot_name", "")).strip() == final_slot:
                    slot["resolved"] = bool(slot_value)
                    break
            step_trace.append(
                StepTrace(
                    step=step,
                    action="spawn",
                    sub_question=probe_query,
                    fact_added=fact_added,
                    tokens=investigate_tokens,
                    slot_name=final_slot,
                    metadata={
                        "goal": str(route.get("goal", "")).strip(),
                        "retrieval_query": retrieval_query,
                        "slot_expected_answer_type": expected_answer_type,
                    },
                )
            )
            step += 1
            if slot_value:
                answer_obj = self._strict_terminal_slot_answer_object(
                    facts=memory.get_all(),
                    slot_state=slot_state,
                    expected_answer_type=expected_answer_type,
                )
                step_trace.append(
                    StepTrace(
                        step=step,
                        action="answer",
                        tokens=0,
                        cited_fact_ids=answer_obj["cited_fact_ids"],
                        justification_confidence=answer_obj["justification_confidence"],
                        metadata={
                            "justification": answer_obj["justification"],
                            "missing_slot": answer_obj["missing_slot"],
                            "fallback_source": answer_obj.get("fallback_source", ""),
                            "direct_probe_terminal_answer": True,
                        },
                    )
                )
                return PipelineResult(
                    question_id=question_id,
                    question=question,
                    answer=answer_obj["answer"],
                    step_trace=step_trace,
                    num_subagent_calls=subagent_calls,
                    num_verify_calls=verify_count,
                    total_tokens=total_tokens,
                    orchestrator_tokens=orchestrator_tokens,
                    subagent_tokens=subagent_tokens,
                    facts_used=memory.get_all(),
                    retrieved_doc_ids=retrieved_doc_ids,
                    retrieved_docs_total=retrieved_docs_total,
                    evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                    fact_memory_capacity=self.fact_memory_capacity,
                    duplicate_subquestion_count=duplicate_subquestion_count,
                    route_decision=route["action"],
                    route_confidence=route["confidence"],
                    route_draft_answer=route["draft_answer"],
                    slot_resolution=self._slot_resolution_map(slot_state),
                )

        while self._pending_slots(slot_state):
            slot_name = self._first_pending_slot(slot_state)
            slot_expected_type = self._slot_expected_answer_type(
                slot_state,
                slot_name,
                expected_answer_type,
            )
            resolved = False
            max_attempts = 2 if slot_name == final_slot else 1

            for attempt in range(max_attempts):
                prompt_tokens = 0
                action_name = "spawn" if attempt == 0 else "refine"
                if attempt == 0:
                    sub_question, retrieval_query = self._hybrid_slot_query(
                        question=question,
                        route=route,
                        slot_state=slot_state,
                        memory=memory,
                        slot_name=slot_name,
                        expected_answer_type=slot_expected_type,
                    )
                    goal = f"Resolve slot '{slot_name}' exactly."
                else:
                    sub_question = question.strip()
                    retrieval_query = str(route.get("retrieval_query", "")).strip() or question.strip()
                    goal = (
                        "Answer the original question directly using the already resolved bridge facts. "
                        f"Target slot: {slot_name}."
                    )

                duplicate_subquestion_count += int(
                    self._is_duplicate_subquestion(sub_question, step_trace)
                )
                capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                    sub_question=sub_question,
                    goal=goal,
                    prior_facts=memory.get_all(),
                    retrieval_query=retrieval_query or None,
                    slot_name=slot_name,
                    slot_hint=self._slot_hint(slot_state, slot_name),
                )
                total_tokens += investigate_tokens
                subagent_tokens += investigate_tokens
                subagent_calls += 1
                retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                    retrieved_doc_ids,
                    retrieved_docs_total,
                    capsule,
                )
                if attempt == 0:
                    fact_added = self._add_fact(memory, capsule, step=step, slot_name=slot_name)
                else:
                    fact_added = self._replace_fact(memory, capsule, step=step, slot_name=slot_name)

                slot_idx, slot_value, slot_confidence = self._best_slot_value_with_index(
                    memory.get_all(),
                    slot_name,
                    expected_answer_type=slot_expected_type,
                )
                for slot in slot_state:
                    if str(slot.get("slot_name", "")).strip() == slot_name:
                        slot["resolved"] = bool(slot_value)
                        break

                step_trace.append(
                    StepTrace(
                        step=step,
                        action=action_name,
                        sub_question=sub_question,
                        fact_added=fact_added,
                        tokens=prompt_tokens + investigate_tokens,
                        slot_name=slot_name,
                        metadata={
                            "goal": goal,
                            "retrieval_query": retrieval_query,
                            "slot_expected_answer_type": slot_expected_type,
                            "resolved_value": slot_value,
                            "resolved_confidence": slot_confidence,
                            "retry_attempt": attempt,
                        },
                    )
                )
                step += 1

                if self._should_force_budget_answer(total_tokens):
                    answer_obj = self._strict_terminal_slot_answer_object(
                        facts=memory.get_all(),
                        slot_state=slot_state,
                        expected_answer_type=expected_answer_type,
                    )
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="answer",
                            tokens=0,
                            cited_fact_ids=answer_obj["cited_fact_ids"],
                            justification_confidence=answer_obj["justification_confidence"],
                            metadata={
                                "justification": answer_obj["justification"],
                                "missing_slot": answer_obj["missing_slot"],
                                "fallback_source": answer_obj.get("fallback_source", ""),
                                "budget_exhausted": True,
                            },
                        )
                    )
                    return PipelineResult(
                        question_id=question_id,
                        question=question,
                        answer=answer_obj["answer"],
                        step_trace=step_trace,
                        num_subagent_calls=subagent_calls,
                        num_verify_calls=verify_count,
                        total_tokens=total_tokens,
                        orchestrator_tokens=orchestrator_tokens,
                        subagent_tokens=subagent_tokens,
                        facts_used=memory.get_all(),
                        retrieved_doc_ids=retrieved_doc_ids,
                        retrieved_docs_total=retrieved_docs_total,
                        evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                        fact_memory_capacity=self.fact_memory_capacity,
                        duplicate_subquestion_count=duplicate_subquestion_count,
                        route_decision=route["action"],
                        route_confidence=route["confidence"],
                        route_draft_answer=route["draft_answer"],
                        slot_resolution=self._slot_resolution_map(slot_state),
                    )

                if slot_value:
                    resolved = True
                    break

            if not resolved:
                if slot_name != final_slot:
                    break
                if len(slot_state) <= 1 or any(slot.get("resolved", False) for slot in slot_state[:-1]):
                    fallback_slot_name = self._final_slot_name(slot_state)
                    fallback_capsule, fallback_tokens = await self.investigator.investigate_with_usage(
                        sub_question=question,
                        goal=f"Answer the original question directly. {target_profile}",
                        prior_facts=memory.get_all(),
                        retrieval_query=str(route.get("retrieval_query", "")).strip() or question,
                        slot_name=fallback_slot_name,
                        slot_hint=self._slot_hint(slot_state, fallback_slot_name),
                    )
                    total_tokens += fallback_tokens
                    subagent_tokens += fallback_tokens
                    subagent_calls += 1
                    retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                        retrieved_doc_ids,
                        retrieved_docs_total,
                        fallback_capsule,
                    )
                    fallback_added = self._add_fact(
                        memory,
                        fallback_capsule,
                        step=step,
                        slot_name=fallback_slot_name,
                    )
                    _, fallback_value, fallback_confidence = self._best_slot_value_with_index(
                        memory.get_all(),
                        fallback_slot_name,
                        expected_answer_type=expected_answer_type,
                    )
                    for slot in slot_state:
                        if str(slot.get("slot_name", "")).strip() == fallback_slot_name:
                            slot["resolved"] = bool(fallback_value)
                            break
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="spawn",
                            sub_question=question,
                            fact_added=fallback_added,
                            tokens=fallback_tokens,
                            slot_name=fallback_slot_name,
                            metadata={
                                "goal": f"Answer the original question directly. {target_profile}",
                                "retrieval_query": str(route.get("retrieval_query", "")).strip() or question,
                                "slot_expected_answer_type": expected_answer_type,
                                "resolved_value": fallback_value,
                                "resolved_confidence": fallback_confidence,
                                "fallback_original_question_probe": True,
                            },
                        )
                    )
                    step += 1
                    if fallback_value:
                        answer_obj = self._strict_terminal_slot_answer_object(
                            facts=memory.get_all(),
                            slot_state=slot_state,
                            expected_answer_type=expected_answer_type,
                        )
                        step_trace.append(
                            StepTrace(
                                step=step,
                                action="answer",
                                tokens=0,
                                cited_fact_ids=answer_obj["cited_fact_ids"],
                                justification_confidence=answer_obj["justification_confidence"],
                                metadata={
                                    "justification": answer_obj["justification"],
                                    "missing_slot": answer_obj["missing_slot"],
                                    "fallback_source": answer_obj.get("fallback_source", ""),
                                    "direct_terminal_slot_answer": True,
                                    "fallback_original_question_probe": True,
                                },
                            )
                        )
                        return PipelineResult(
                            question_id=question_id,
                            question=question,
                            answer=answer_obj["answer"],
                            step_trace=step_trace,
                            num_subagent_calls=subagent_calls,
                            num_verify_calls=verify_count,
                            total_tokens=total_tokens,
                            orchestrator_tokens=orchestrator_tokens,
                            subagent_tokens=subagent_tokens,
                            facts_used=memory.get_all(),
                            retrieved_doc_ids=retrieved_doc_ids,
                            retrieved_docs_total=retrieved_docs_total,
                            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                            fact_memory_capacity=self.fact_memory_capacity,
                            duplicate_subquestion_count=duplicate_subquestion_count,
                            route_decision=route["action"],
                            route_confidence=route["confidence"],
                            route_draft_answer=route["draft_answer"],
                            slot_resolution=self._slot_resolution_map(slot_state),
                        )
                break

        answer_obj = self._strict_terminal_slot_answer_object(
            facts=memory.get_all(),
            slot_state=slot_state,
            expected_answer_type=expected_answer_type,
        )
        step_trace.append(
            StepTrace(
                step=step,
                action="answer",
                tokens=0,
                cited_fact_ids=answer_obj["cited_fact_ids"],
                justification_confidence=answer_obj["justification_confidence"],
                metadata={
                    "justification": answer_obj["justification"],
                    "missing_slot": answer_obj["missing_slot"],
                    "fallback_source": answer_obj.get("fallback_source", ""),
                    "direct_terminal_slot_answer": True,
                },
            )
        )
        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer_obj["answer"],
            step_trace=step_trace,
            num_subagent_calls=subagent_calls,
            num_verify_calls=verify_count,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=memory.get_all(),
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
            fact_memory_capacity=self.fact_memory_capacity,
            duplicate_subquestion_count=duplicate_subquestion_count,
            route_decision=route["action"],
            route_confidence=route["confidence"],
            route_draft_answer=route["draft_answer"],
            slot_resolution=self._slot_resolution_map(slot_state),
        )

    async def _run_adaptive_opera_plan_exec(
        self, question: str, question_id: str
    ) -> PipelineResult:
        """Adaptive outer route with OPERA-style plan/placeholder execution inside recurse."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        expected_answer_type = self._expected_answer_type(question)
        target_profile = self._target_profile(question)
        step_trace: list[StepTrace] = []
        total_tokens = 0
        orchestrator_tokens = 0
        subagent_tokens = 0
        subagent_calls = 0
        verify_count = 0
        duplicate_subquestion_count = 0
        retrieved_doc_ids: list[str] = []
        retrieved_docs_total = 0

        route, route_tokens = await self.orchestrator.route_with_usage(
            question=question,
            target_profile=target_profile,
        )
        if self._should_force_bridge_first_route(question, route):
            route = self._force_bridge_first_route(question, route, target_profile)
        total_tokens += route_tokens
        orchestrator_tokens += route_tokens
        final_slot = str(route.get("target_slot", "")).strip() or "final_answer"
        slot_state = [{
            "slot_name": final_slot,
            "hint": target_profile,
            "resolved": False,
            "dependency_group": 0,
        }]
        step_trace.append(
            StepTrace(
                step=0,
                action="route",
                tokens=route_tokens,
                route_decision=route["action"],
                route_confidence=route["confidence"],
                route_draft_answer=route["draft_answer"],
                metadata={
                    "expected_answer_type": expected_answer_type,
                    "answer_type": route["answer_type"],
                    "target_slot": final_slot,
                    "retrieval_query": route.get("retrieval_query", ""),
                    "adaptive_opera_plan_exec": True,
                },
            )
        )

        if (
            str(route.get("action", "")).strip().lower() == "single_probe"
            and float(route.get("confidence", 0.0)) >= self.route_direct_threshold
        ):
            capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                sub_question=question.strip(),
                goal=str(route.get("goal", "")).strip() or f"Answer the question directly. {target_profile}",
                prior_facts=[],
                retrieval_query=str(route.get("retrieval_query", "")).strip() or question.strip(),
                slot_name=final_slot,
                slot_hint=target_profile,
            )
            total_tokens += investigate_tokens
            subagent_tokens += investigate_tokens
            subagent_calls += 1
            retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                retrieved_doc_ids,
                retrieved_docs_total,
                capsule,
            )
            fact_added = self._add_fact(memory, capsule, step=1, slot_name=final_slot)
            step_trace.append(
                StepTrace(
                    step=1,
                    action="spawn",
                    sub_question=question.strip(),
                    fact_added=fact_added,
                    tokens=investigate_tokens,
                    slot_name=final_slot,
                    metadata={
                        "goal": str(route.get("goal", "")).strip(),
                        "retrieval_query": str(route.get("retrieval_query", "")).strip() or question.strip(),
                        "slot_expected_answer_type": expected_answer_type,
                        "direct_probe": True,
                    },
                )
            )
            answer_obj = self._strict_terminal_slot_answer_object(
                facts=memory.get_all(),
                slot_state=slot_state,
                expected_answer_type=expected_answer_type,
            )
            if answer_obj["answer"]:
                step_trace.append(
                    StepTrace(
                        step=2,
                        action="answer",
                        tokens=0,
                        cited_fact_ids=answer_obj["cited_fact_ids"],
                        justification_confidence=answer_obj["justification_confidence"],
                        metadata={
                            "justification": answer_obj["justification"],
                            "missing_slot": answer_obj["missing_slot"],
                            "fallback_source": answer_obj.get("fallback_source", ""),
                            "direct_terminal_slot_answer": True,
                        },
                    )
                )
                return PipelineResult(
                    question_id=question_id,
                    question=question,
                    answer=answer_obj["answer"],
                    step_trace=step_trace,
                    num_subagent_calls=subagent_calls,
                    num_verify_calls=verify_count,
                    total_tokens=total_tokens,
                    orchestrator_tokens=orchestrator_tokens,
                    subagent_tokens=subagent_tokens,
                    facts_used=memory.get_all(),
                    retrieved_doc_ids=retrieved_doc_ids,
                    retrieved_docs_total=retrieved_docs_total,
                    evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                    fact_memory_capacity=self.fact_memory_capacity,
                    duplicate_subquestion_count=duplicate_subquestion_count,
                    route_decision=route["action"],
                    route_confidence=route["confidence"],
                    route_draft_answer=route["draft_answer"],
                    slot_resolution=self._slot_resolution_map(slot_state),
                )

        plan, plan_tokens = await self.orchestrator.decompose_upfront(
            question,
            self.max_steps,
            target_profile,
        )
        total_tokens += plan_tokens
        orchestrator_tokens += plan_tokens
        collected_answers: dict[int, str] = {}
        if not plan:
            answer_obj = self._strict_terminal_slot_answer_object(
                facts=memory.get_all(),
                slot_state=slot_state,
                expected_answer_type=expected_answer_type,
            )
            return PipelineResult(
                question_id=question_id,
                question=question,
                answer=answer_obj["answer"],
                step_trace=step_trace,
                num_subagent_calls=subagent_calls,
                num_verify_calls=verify_count,
                total_tokens=total_tokens,
                orchestrator_tokens=orchestrator_tokens,
                subagent_tokens=subagent_tokens,
                facts_used=memory.get_all(),
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_docs_total=retrieved_docs_total,
                evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                fact_memory_capacity=self.fact_memory_capacity,
                duplicate_subquestion_count=duplicate_subquestion_count,
                route_decision=route["action"],
                route_confidence=route["confidence"],
                route_draft_answer=route["draft_answer"],
                slot_resolution=self._slot_resolution_map(slot_state),
            )

        plan_steps = plan[: self.max_steps]
        for idx, item in enumerate(plan_steps, start=1):
            template = str(item.get("sub_question", "")).strip() or question.strip()
            filled_question = self._fill_plan_placeholders(template, collected_answers)
            goal = str(item.get("goal", "")).strip() or filled_question
            is_last = idx == len(plan_steps)
            slot_name = final_slot if is_last else f"plan_step_{idx}"
            slot_expected_type = expected_answer_type if is_last else self._expected_answer_type(filled_question)
            variants = self._generate_query_variants(filled_question)
            resolved_answer = ""
            for attempt, retrieval_query in enumerate(variants[:2]):
                duplicate_subquestion_count += int(
                    self._is_duplicate_subquestion(filled_question, step_trace)
                )
                capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                    sub_question=filled_question,
                    goal=goal,
                    prior_facts=memory.get_all(),
                    retrieval_query=retrieval_query or None,
                    slot_name=slot_name,
                    slot_hint=target_profile if is_last else filled_question,
                )
                total_tokens += investigate_tokens
                subagent_tokens += investigate_tokens
                subagent_calls += 1
                retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                    retrieved_doc_ids,
                    retrieved_docs_total,
                    capsule,
                )
                fact_added = (
                    self._add_fact(memory, capsule, step=idx, slot_name=slot_name)
                    if attempt == 0
                    else self._replace_fact(memory, capsule, step=idx, slot_name=slot_name)
                )
                resolved_answer = self._extract_plan_step_answer(
                    capsule,
                    expected_answer_type=slot_expected_type,
                )
                step_trace.append(
                    StepTrace(
                        step=idx,
                        action="spawn" if attempt == 0 else "refine",
                        sub_question=filled_question,
                        fact_added=fact_added,
                        tokens=investigate_tokens,
                        slot_name=slot_name,
                        metadata={
                            "goal": goal,
                            "retrieval_query": retrieval_query,
                            "template_sub_question": template,
                            "filled_sub_question": filled_question,
                            "resolved_answer": resolved_answer,
                            "slot_expected_answer_type": slot_expected_type,
                            "plan_step_id": idx,
                            "rewrite_retry": attempt == 1,
                        },
                    )
                )
                if resolved_answer:
                    collected_answers[idx] = resolved_answer
                    if is_last:
                        slot_state[0]["resolved"] = True
                    break
            if not resolved_answer:
                break

        answer_obj = self._strict_terminal_slot_answer_object(
            facts=memory.get_all(),
            slot_state=slot_state,
            expected_answer_type=expected_answer_type,
        )
        if not answer_obj["answer"] and plan_steps:
            last_answer = collected_answers.get(len(plan_steps), "")
            if self._answer_matches_expected_type(last_answer, expected_answer_type):
                answer_obj["answer"] = last_answer
                answer_obj["fallback_source"] = "last_plan_step"
        step_trace.append(
            StepTrace(
                step=len(step_trace),
                action="answer",
                tokens=0,
                cited_fact_ids=answer_obj.get("cited_fact_ids", []),
                justification_confidence=answer_obj.get("justification_confidence", 0.0),
                metadata={
                    "justification": answer_obj.get("justification", ""),
                    "missing_slot": answer_obj.get("missing_slot", ""),
                    "fallback_source": answer_obj.get("fallback_source", ""),
                    "last_subgoal_is_answer": True,
                },
            )
        )
        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer_obj["answer"],
            step_trace=step_trace,
            num_subagent_calls=subagent_calls,
            num_verify_calls=verify_count,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=memory.get_all(),
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
            fact_memory_capacity=self.fact_memory_capacity,
            duplicate_subquestion_count=duplicate_subquestion_count,
            route_decision=route["action"],
            route_confidence=route["confidence"],
            route_draft_answer=route["draft_answer"],
            slot_resolution=self._slot_resolution_map(slot_state),
        )

    async def _run_s0(self, question: str, question_id: str) -> PipelineResult:
        """S0 mode: one retrieval pass, then answer."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        expected_answer_type = self._expected_answer_type(question)
        target_profile = self._target_profile(question)
        step_trace: list[StepTrace] = []
        total_tokens = 0
        orchestrator_tokens = 0
        subagent_tokens = 0
        retrieved_doc_ids: list[str] = []
        retrieved_docs_total = 0

        capsule, investigate_tokens = await self.investigator.investigate_with_usage(
            sub_question=question,
            goal=f"Answer this question directly. {target_profile}",
            prior_facts=[],
        )
        total_tokens += investigate_tokens
        subagent_tokens += investigate_tokens
        retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
            retrieved_doc_ids,
            retrieved_docs_total,
            capsule,
        )

        fact_added = self._add_fact(memory, capsule, step=0)
        step_trace.append(
            StepTrace(
                step=0,
                action="spawn",
                sub_question=question,
                claim=None,
                fact_added=fact_added,
                tokens=investigate_tokens,
            )
        )

        answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
            question,
            memory.get_all(),
            target_profile,
            trace=step_trace,
        )
        answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
        total_tokens += answer_tokens
        orchestrator_tokens += answer_tokens
        step_trace.append(
            StepTrace(
                step=1,
                action="answer",
                sub_question=None,
                claim=None,
                fact_added=False,
                tokens=answer_tokens,
            )
        )

        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=step_trace,
            num_subagent_calls=1,
            num_verify_calls=0,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=memory.get_all(),
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
            fact_memory_capacity=self.fact_memory_capacity,
            duplicate_subquestion_count=0,
        )

    async def _run_upfront_decomposition(
        self, question: str, question_id: str
    ) -> PipelineResult:
        """A2: upfront decomposition, sequential execution, final answer."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        expected_answer_type = self._expected_answer_type(question)
        target_profile = self._target_profile(question)
        step_trace: list[StepTrace] = []
        total_tokens = 0
        orchestrator_tokens = 0
        subagent_tokens = 0
        verify_count = 0
        subagent_calls = 0
        duplicate_subquestion_count = 0
        retrieved_doc_ids: list[str] = []
        retrieved_docs_total = 0

        plan, plan_tokens = await self.orchestrator.decompose_upfront(
            question, self.max_steps, target_profile
        )
        total_tokens += plan_tokens
        orchestrator_tokens += plan_tokens

        for step, item in enumerate(plan[: self.max_steps]):
            duplicate_subquestion_count += int(
                self._is_duplicate_subquestion(item["sub_question"], step_trace)
            )
            capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                sub_question=item["sub_question"],
                goal=item["goal"],
                prior_facts=memory.get_all(),
            )
            total_tokens += investigate_tokens
            subagent_tokens += investigate_tokens
            subagent_calls += 1
            retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                retrieved_doc_ids,
                retrieved_docs_total,
                capsule,
            )

            fact_added = self._add_fact(memory, capsule, step=step)
            step_trace.append(
                StepTrace(
                    step=step,
                    action="spawn",
                    sub_question=item["sub_question"],
                    claim=None,
                    fact_added=fact_added,
                    tokens=investigate_tokens,
                )
            )

            if self.ablation_always_verify and capsule.fact.text:
                verify_tokens = await self._auto_verify(step, capsule, step_trace)
                total_tokens += verify_tokens
                orchestrator_tokens += verify_tokens
                verify_count += 1

        answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
            question,
            memory.get_all(),
            target_profile,
            trace=step_trace,
        )
        answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
        total_tokens += answer_tokens
        orchestrator_tokens += answer_tokens
        step_trace.append(
            StepTrace(
                step=len(plan[: self.max_steps]),
                action="answer",
                sub_question=None,
                claim=None,
                fact_added=False,
                tokens=answer_tokens,
            )
        )

        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=step_trace,
            num_subagent_calls=subagent_calls,
            num_verify_calls=verify_count,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=memory.get_all(),
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
            fact_memory_capacity=self.fact_memory_capacity,
            duplicate_subquestion_count=duplicate_subquestion_count,
        )

    async def _run_adaptive_m11(self, question: str, question_id: str) -> PipelineResult:
        """Legacy routed controller with slot state and answer escalation."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        expected_answer_type = self._expected_answer_type(question)
        target_profile = self._target_profile(question)
        step_trace: list[StepTrace] = []
        total_tokens = 0
        orchestrator_tokens = 0
        subagent_tokens = 0
        subagent_calls = 0
        verify_count = 0
        auto_verify_calls = 0
        answer_rejection_count = 0
        duplicate_subquestion_count = 0
        refine_count_by_slot: dict[str, int] = {}
        retrieved_doc_ids: list[str] = []
        retrieved_docs_total = 0
        answer = ""

        route, route_tokens = await self.orchestrator.route_with_usage(
            question=question,
            target_profile=target_profile,
        )
        if self._should_force_bridge_first_route(question, route):
            route = self._force_bridge_first_route(question, route, target_profile)
        total_tokens += route_tokens
        orchestrator_tokens += route_tokens

        slot_state = self._initialise_slot_state(route, target_profile)
        force_planned_recurse = self._should_force_planned_recurse_execution(
            route,
            slot_state,
        )
        step_trace.append(
            StepTrace(
                step=0,
                action="route",
                tokens=route_tokens,
                route_decision=route["action"],
                route_confidence=route["confidence"],
                route_draft_answer=route["draft_answer"],
                metadata={
                    "expected_answer_type": expected_answer_type,
                    "answer_type": route["answer_type"],
                    "target_slot": route["target_slot"],
                    "retrieval_query": route.get("retrieval_query", ""),
                    "pending_slots": self._slot_snapshot(slot_state),
                    "forced_bridge_first": bool(route.get("forced_bridge_first", False)),
                    "force_planned_recurse_execution": force_planned_recurse,
                },
            )
        )

        next_step = 1
        min_subagent_calls_before_answer = 1

        initial_sub_question = (route["sub_question"] or question).strip() or question
        initial_retrieval_query = str(route.get("retrieval_query", "")).strip()
        initial_goal = route["goal"]
        pending_slots = self._pending_slots(slot_state)

        if route["action"] == "recurse":
            min_subagent_calls_before_answer = (
                2 if route["confidence"] >= self.recurse_min_depth_conf_threshold else 1
            )
            if initial_sub_question.lower() == question.strip().lower():
                refine_decision, refine_tokens = await self.orchestrator.propose_spawn(
                    question=question,
                    facts=[],
                    trace=step_trace,
                    target_profile=target_profile,
                    pending_slots=pending_slots,
                    missing_reason=(
                        "Recurse requires a narrower first missing-fact probe than the original question. "
                        "Produce the best first focused sub-question."
                    ),
                )
                total_tokens += refine_tokens
                orchestrator_tokens += refine_tokens
                initial_sub_question = refine_decision["sub_question"]
                initial_retrieval_query = str(
                    refine_decision.get("retrieval_query", "")
                ).strip()
                initial_goal = refine_decision["goal"]
                step_trace[0].metadata["refined_bootstrap_sub_question"] = initial_sub_question
                step_trace[0].metadata["refined_bootstrap_retrieval_query"] = (
                    initial_retrieval_query or initial_sub_question
                )
                step_trace[0].metadata["refined_bootstrap_goal"] = initial_goal
                step_trace[0].tokens += refine_tokens
                if initial_sub_question.lower() == question.strip().lower() and pending_slots:
                    focused_slots = [pending_slots[0]]
                    refine_decision, refine_tokens = await self.orchestrator.propose_spawn(
                        question=question,
                        facts=[],
                        trace=step_trace,
                        target_profile=target_profile,
                        pending_slots=focused_slots,
                        missing_reason=(
                            "The first recurse probe must target only the first unresolved slot in dependency order. "
                            "Do not repeat the original question."
                        ),
                    )
                    total_tokens += refine_tokens
                    orchestrator_tokens += refine_tokens
                    initial_sub_question = refine_decision["sub_question"]
                    initial_retrieval_query = str(
                        refine_decision.get("retrieval_query", "")
                    ).strip()
                    initial_goal = refine_decision["goal"]
                    step_trace[0].metadata["slot_focused_bootstrap_sub_question"] = initial_sub_question
                    step_trace[0].metadata["slot_focused_bootstrap_retrieval_query"] = (
                        initial_retrieval_query or initial_sub_question
                    )
                    step_trace[0].metadata["slot_focused_bootstrap_goal"] = initial_goal
                    step_trace[0].tokens += refine_tokens

        next_step = 1
        if route["action"] in {"single_probe", "recurse"}:
            probe_slot = self._first_pending_slot(slot_state)
            capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                sub_question=initial_sub_question,
                goal=initial_goal,
                prior_facts=[],
                retrieval_query=initial_retrieval_query or None,
                slot_name=probe_slot,
                slot_hint=self._slot_hint(slot_state, probe_slot),
            )
            total_tokens += investigate_tokens
            subagent_tokens += investigate_tokens
            subagent_calls += 1
            retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                retrieved_doc_ids,
                retrieved_docs_total,
                capsule,
            )
            fact_added = self._add_fact(memory, capsule, step=1, slot_name=probe_slot)
            self._update_slot_resolution(slot_state, probe_slot, capsule)
            step_trace.append(
                StepTrace(
                    step=1,
                    action="spawn",
                    sub_question=initial_sub_question,
                    fact_added=fact_added,
                    tokens=investigate_tokens,
                    slot_name=probe_slot,
                    route_decision=route["action"],
                    route_confidence=route["confidence"],
                    route_draft_answer=route["draft_answer"],
                    metadata={
                        "goal": initial_goal,
                        "retrieval_query": initial_retrieval_query or initial_sub_question,
                    },
                    )
                )
            (
                verify_tokens,
                verify_delta,
                auto_verify_delta,
            ) = await self._maybe_verify_fact(
                question=question,
                step=1,
                slot_name=probe_slot,
                sub_question=initial_sub_question,
                capsule=capsule,
                memory=memory,
                slot_state=slot_state,
                step_trace=step_trace,
            )
            total_tokens += verify_tokens
            subagent_tokens += verify_tokens
            verify_count += verify_delta
            auto_verify_calls += auto_verify_delta
            next_step = 2
            if self._should_force_budget_answer(total_tokens):
                answer_obj, total_tokens, orchestrator_tokens = await self._force_budget_answer_object(
                    question=question,
                    facts=memory.get_all(),
                    target_profile=target_profile,
                    pending_slots=self._pending_slots(slot_state),
                    step_trace=step_trace,
                    step=2,
                    total_tokens=total_tokens,
                    orchestrator_tokens=orchestrator_tokens,
                    route_draft_answer=route["draft_answer"],
                    metadata={"budget_stage": "post_bootstrap"},
                )
                answer = answer_obj["answer"]
                return PipelineResult(
                    question_id=question_id,
                    question=question,
                    answer=answer,
                    step_trace=step_trace,
                    num_subagent_calls=subagent_calls,
                    num_verify_calls=verify_count,
                    total_tokens=total_tokens,
                    orchestrator_tokens=orchestrator_tokens,
                    subagent_tokens=subagent_tokens,
                    facts_used=memory.get_all(),
                    retrieved_doc_ids=retrieved_doc_ids,
                    retrieved_docs_total=retrieved_docs_total,
                    evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                    fact_memory_capacity=self.fact_memory_capacity,
                    duplicate_subquestion_count=duplicate_subquestion_count,
                    route_decision=route["action"],
                    route_confidence=route["confidence"],
                    route_draft_answer=route["draft_answer"],
                    slot_resolution=self._slot_resolution_map(slot_state),
                    auto_verify_calls=auto_verify_calls,
                    answer_rejection_count=answer_rejection_count,
                )

            if route["action"] == "single_probe":
                pending_slots = self._pending_slots(slot_state)
                answer_obj, answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
                    question=question,
                    facts=memory.get_all(),
                    target_profile=target_profile,
                    pending_slots=pending_slots,
                    trace=step_trace,
                    route_draft_answer=route["draft_answer"],
                )
                answer_obj = self._apply_answer_object_fallback(
                    answer_obj,
                    memory.get_all(),
                    route["draft_answer"],
                    expected_answer_type=expected_answer_type,
                )
                answer_obj = self._prefer_terminal_slot_answer_object(
                    answer_obj,
                    memory.get_all(),
                    slot_state,
                    expected_answer_type=expected_answer_type,
                )
                total_tokens += answer_tokens
                orchestrator_tokens += answer_tokens

                single_probe_verified_accept = False
                if not self._should_escalate_answer(answer_obj, pending_slots):
                    answer_verify_result, answer_verify_tokens = await self._verify_answer_attempt(
                        question=question,
                        step=2,
                        answer_obj=answer_obj,
                        facts=memory.get_all(),
                        target_profile=target_profile,
                        expected_answer_type=expected_answer_type,
                        step_trace=step_trace,
                        reason_tag="single_probe_final",
                    )
                    total_tokens += answer_verify_tokens
                    orchestrator_tokens += answer_verify_tokens
                    verify_count += int(answer_verify_tokens > 0)
                    if (
                        answer_verify_result.get("decision") == "reject"
                        and next_step <= self.max_steps
                    ):
                        answer_rejection_count += 1
                        step_trace.append(
                            StepTrace(
                                step=2,
                                action="answer_rejected_escalate",
                                tokens=answer_tokens + answer_verify_tokens,
                                cited_fact_ids=answer_obj["cited_fact_ids"],
                                justification_confidence=answer_obj["justification_confidence"],
                                metadata={
                                    "justification": answer_obj["justification"],
                                    "missing_slot": answer_obj["missing_slot"],
                                    "route_lane": route["action"],
                                    "fallback_source": answer_obj.get("fallback_source", ""),
                                    "verify_reason": answer_verify_result.get("reason", ""),
                                    "failure_mode": answer_verify_result.get("failure_mode", ""),
                                },
                            )
                        )
                        recovery = await self._targeted_answer_recovery(
                            question=question,
                            step=2,
                            route_draft_answer=route["draft_answer"],
                            target_profile=target_profile,
                            expected_answer_type=expected_answer_type,
                            memory=memory,
                            slot_state=slot_state,
                            step_trace=step_trace,
                            slot_name=self._final_slot_name(slot_state),
                            missing_reason=(
                                "The candidate answer failed verification. "
                                f"Failure mode: {answer_verify_result.get('failure_mode', 'other')}. "
                                f"Reason: {answer_verify_result.get('reason', '')}"
                            ),
                            retrieved_doc_ids=retrieved_doc_ids,
                            retrieved_docs_total=retrieved_docs_total,
                        )
                        total_tokens += recovery["orchestrator_tokens"] + recovery["subagent_tokens"]
                        orchestrator_tokens += recovery["orchestrator_tokens"]
                        subagent_tokens += recovery["subagent_tokens"]
                        subagent_calls += recovery["subagent_calls"]
                        verify_count += recovery["verify_calls"]
                        auto_verify_calls += recovery["auto_verify_calls"]
                        duplicate_subquestion_count += recovery["duplicate_subquestion_count"]
                        retrieved_doc_ids = recovery["retrieved_doc_ids"]
                        retrieved_docs_total = recovery["retrieved_docs_total"]
                        answer_obj = recovery["answer_obj"]
                        pending_slots = self._pending_slots(slot_state)
                        retry_verify_result, retry_verify_tokens = await self._verify_answer_attempt(
                            question=question,
                            step=4,
                            answer_obj=answer_obj,
                            facts=memory.get_all(),
                            target_profile=target_profile,
                            expected_answer_type=expected_answer_type,
                            step_trace=step_trace,
                            reason_tag="single_probe_retry_final",
                        )
                        total_tokens += retry_verify_tokens
                        orchestrator_tokens += retry_verify_tokens
                        verify_count += int(retry_verify_tokens > 0)
                        if (
                            not self._should_escalate_answer(answer_obj, pending_slots)
                            and retry_verify_result.get("decision") == "accept"
                        ):
                            answer = answer_obj["answer"]
                            step_trace.append(
                                StepTrace(
                                    step=4,
                                    action="answer",
                                    tokens=recovery["answer_tokens"] + retry_verify_tokens,
                                    cited_fact_ids=answer_obj["cited_fact_ids"],
                                    justification_confidence=answer_obj["justification_confidence"],
                                    metadata={
                                        "justification": answer_obj["justification"],
                                        "missing_slot": answer_obj["missing_slot"],
                                        "fallback_source": answer_obj.get("fallback_source", ""),
                                        "recovery_mode": "answer_verification_retry",
                                    },
                                )
                            )
                            return PipelineResult(
                                question_id=question_id,
                                question=question,
                                answer=answer,
                                step_trace=step_trace,
                                num_subagent_calls=subagent_calls,
                                num_verify_calls=verify_count,
                                total_tokens=total_tokens,
                                orchestrator_tokens=orchestrator_tokens,
                                subagent_tokens=subagent_tokens,
                                facts_used=memory.get_all(),
                                retrieved_doc_ids=retrieved_doc_ids,
                                retrieved_docs_total=retrieved_docs_total,
                                evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                                fact_memory_capacity=self.fact_memory_capacity,
                                duplicate_subquestion_count=duplicate_subquestion_count,
                                route_decision=route["action"],
                                route_confidence=route["confidence"],
                                route_draft_answer=route["draft_answer"],
                                slot_resolution=self._slot_resolution_map(slot_state),
                                auto_verify_calls=auto_verify_calls,
                                answer_rejection_count=answer_rejection_count,
                            )
                    elif answer_verify_result.get("decision") == "accept":
                        single_probe_verified_accept = True
                    if single_probe_verified_accept:
                        answer = answer_obj["answer"]
                        step_trace.append(
                            StepTrace(
                                step=2,
                                action="answer",
                                tokens=answer_tokens + answer_verify_tokens,
                                cited_fact_ids=answer_obj["cited_fact_ids"],
                                justification_confidence=answer_obj["justification_confidence"],
                                metadata={
                                    "justification": answer_obj["justification"],
                                    "missing_slot": answer_obj["missing_slot"],
                                    "fallback_source": answer_obj.get("fallback_source", ""),
                                },
                            )
                        )
                        return PipelineResult(
                            question_id=question_id,
                            question=question,
                            answer=answer,
                            step_trace=step_trace,
                            num_subagent_calls=subagent_calls,
                            num_verify_calls=verify_count,
                            total_tokens=total_tokens,
                            orchestrator_tokens=orchestrator_tokens,
                            subagent_tokens=subagent_tokens,
                            facts_used=memory.get_all(),
                            retrieved_doc_ids=retrieved_doc_ids,
                            retrieved_docs_total=retrieved_docs_total,
                            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                            fact_memory_capacity=self.fact_memory_capacity,
                            duplicate_subquestion_count=duplicate_subquestion_count,
                            route_decision=route["action"],
                            route_confidence=route["confidence"],
                            route_draft_answer=route["draft_answer"],
                            slot_resolution=self._slot_resolution_map(slot_state),
                            auto_verify_calls=auto_verify_calls,
                            answer_rejection_count=answer_rejection_count,
                        )
                if next_step <= self.max_steps and not single_probe_verified_accept:
                    answer_rejection_count += 1
                    step_trace.append(
                        StepTrace(
                            step=2,
                            action="answer_rejected_escalate",
                            tokens=answer_tokens,
                            cited_fact_ids=answer_obj["cited_fact_ids"],
                            justification_confidence=answer_obj["justification_confidence"],
                            metadata={
                                "justification": answer_obj["justification"],
                                "missing_slot": answer_obj["missing_slot"],
                                "route_lane": route["action"],
                                "fallback_source": answer_obj.get("fallback_source", ""),
                            },
                        )
                    )

        for step in range(next_step, self.max_steps + 1):
            if self._should_force_budget_answer(total_tokens):
                answer_obj, total_tokens, orchestrator_tokens = await self._force_budget_answer_object(
                    question=question,
                    facts=memory.get_all(),
                    target_profile=target_profile,
                    pending_slots=self._pending_slots(slot_state),
                    step_trace=step_trace,
                    step=step,
                    total_tokens=total_tokens,
                    orchestrator_tokens=orchestrator_tokens,
                    route_draft_answer=route["draft_answer"],
                    metadata={"budget_stage": "loop_start"},
                )
                answer = answer_obj["answer"]
                break
            pending_slots = self._pending_slots(slot_state)
            ready_parallel_slots = self._parallel_ready_slots(slot_state)
            if (
                force_planned_recurse
                and pending_slots
                and step < self.max_steps
            ):
                decision, decide_tokens = await self._propose_planned_spawn(
                    question=question,
                    target_profile=target_profile,
                    memory=memory,
                    slot_state=slot_state,
                    step_trace=step_trace,
                )
                action = "spawn"
            elif (
                self.enable_parallel_independent_hops
                and len(ready_parallel_slots) > 1
                and step < self.max_steps
            ):
                parallel_result = await self._run_parallel_slot_batch(
                    question=question,
                    step=step,
                    route=route,
                    target_profile=target_profile,
                    memory=memory,
                    slot_state=slot_state,
                    step_trace=step_trace,
                )
                total_tokens += parallel_result["total_tokens"]
                orchestrator_tokens += parallel_result["orchestrator_tokens"]
                subagent_tokens += parallel_result["subagent_tokens"]
                subagent_calls += parallel_result["subagent_calls"]
                verify_count += parallel_result["verify_count"]
                auto_verify_calls += parallel_result["auto_verify_calls"]
                duplicate_subquestion_count += parallel_result["duplicate_subquestion_count"]
                retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_ids(
                    retrieved_doc_ids,
                    retrieved_docs_total,
                    parallel_result["retrieved_doc_ids"],
                    parallel_result["retrieved_docs_total"],
                )
                continue
            elif self.ablation_force_spawn and step < self.max_steps:
                decision, decide_tokens = await self.orchestrator.propose_spawn(
                    question=question,
                    facts=memory.get_all(),
                    trace=step_trace,
                    target_profile=target_profile,
                    pending_slots=pending_slots,
                    missing_reason="Forced spawn ablation.",
                )
                action = "spawn"
            else:
                decision, decide_tokens = await self.orchestrator.decide_with_usage(
                    question=question,
                    facts=memory.get_all(),
                    trace=step_trace,
                    step=step,
                    target_profile=target_profile,
                    pending_slots=pending_slots,
                )
                action = decision["action"]

            total_tokens += decide_tokens
            orchestrator_tokens += decide_tokens
            if action != "answer" and self._should_force_budget_answer(total_tokens):
                answer_obj, total_tokens, orchestrator_tokens = await self._force_budget_answer_object(
                    question=question,
                    facts=memory.get_all(),
                    target_profile=target_profile,
                    pending_slots=self._pending_slots(slot_state),
                    step_trace=step_trace,
                    step=step,
                    total_tokens=total_tokens,
                    orchestrator_tokens=orchestrator_tokens,
                    route_draft_answer=route["draft_answer"],
                    trace_tokens=decide_tokens,
                    metadata={"budget_stage": "post_decide"},
                )
                answer = answer_obj["answer"]
                break

            if action == "answer":
                if pending_slots and step < self.max_steps:
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="answer_blocked_pending_slots",
                            tokens=decide_tokens,
                            metadata={
                                "pending_slots": pending_slots,
                            },
                        )
                    )
                    decision, spawn_tokens = await self.orchestrator.propose_spawn(
                        question=question,
                        facts=memory.get_all(),
                        trace=step_trace,
                        target_profile=target_profile,
                        pending_slots=pending_slots,
                        missing_reason="Pending slots remain unresolved. Retrieve the next missing fact instead of answering.",
                    )
                    total_tokens += spawn_tokens
                    orchestrator_tokens += spawn_tokens
                    action = "spawn"
                    decide_tokens += spawn_tokens
                else:
                    if force_planned_recurse and not pending_slots:
                        answer_obj = self._direct_terminal_slot_answer_object(
                            facts=memory.get_all(),
                            slot_state=slot_state,
                            expected_answer_type=expected_answer_type,
                        )
                        answer_tokens = 0
                    else:
                        answer_obj, answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
                            question=question,
                            facts=memory.get_all(),
                            target_profile=target_profile,
                            pending_slots=pending_slots,
                            trace=step_trace,
                            route_draft_answer=route["draft_answer"],
                        )
                        answer_obj = self._apply_answer_object_fallback(
                            answer_obj,
                            memory.get_all(),
                            route["draft_answer"],
                            expected_answer_type=expected_answer_type,
                        )
                        answer_obj = self._prefer_terminal_slot_answer_object(
                            answer_obj,
                            memory.get_all(),
                            slot_state,
                            expected_answer_type=expected_answer_type,
                        )
                    total_tokens += answer_tokens
                    orchestrator_tokens += answer_tokens
                    if not self._should_escalate_answer(answer_obj, pending_slots):
                        answer_verify_result, answer_verify_tokens = await self._verify_answer_attempt(
                            question=question,
                            step=step,
                            answer_obj=answer_obj,
                            facts=memory.get_all(),
                            target_profile=target_profile,
                            expected_answer_type=expected_answer_type,
                            step_trace=step_trace,
                            reason_tag="recurse_final",
                        )
                        total_tokens += answer_verify_tokens
                        orchestrator_tokens += answer_verify_tokens
                        verify_count += int(answer_verify_tokens > 0)
                        if answer_verify_result.get("decision") == "reject":
                            answer_rejection_count += 1
                            step_trace.append(
                                StepTrace(
                                    step=step,
                                    action="answer_rejected_escalate",
                                    tokens=decide_tokens + answer_tokens + answer_verify_tokens,
                                    cited_fact_ids=answer_obj["cited_fact_ids"],
                                    justification_confidence=answer_obj["justification_confidence"],
                                    metadata={
                                        "justification": answer_obj["justification"],
                                        "missing_slot": answer_obj["missing_slot"],
                                        "fallback_source": answer_obj.get("fallback_source", ""),
                                        "recovery_mode": "answer_verification_retry",
                                        "verify_reason": answer_verify_result.get("reason", ""),
                                        "failure_mode": answer_verify_result.get("failure_mode", ""),
                                    },
                                )
                            )
                            recovery = await self._targeted_answer_recovery(
                                question=question,
                                step=step,
                                route_draft_answer=route["draft_answer"],
                                target_profile=target_profile,
                                expected_answer_type=expected_answer_type,
                                memory=memory,
                                slot_state=slot_state,
                                step_trace=step_trace,
                                slot_name=self._final_slot_name(slot_state),
                                missing_reason=(
                                    "The candidate answer failed verification. "
                                    f"Failure mode: {answer_verify_result.get('failure_mode', 'other')}. "
                                    f"Reason: {answer_verify_result.get('reason', '')}"
                                ),
                                retrieved_doc_ids=retrieved_doc_ids,
                                retrieved_docs_total=retrieved_docs_total,
                            )
                            total_tokens += recovery["orchestrator_tokens"] + recovery["subagent_tokens"]
                            orchestrator_tokens += recovery["orchestrator_tokens"]
                            subagent_tokens += recovery["subagent_tokens"]
                            subagent_calls += recovery["subagent_calls"]
                            verify_count += recovery["verify_calls"]
                            auto_verify_calls += recovery["auto_verify_calls"]
                            duplicate_subquestion_count += recovery["duplicate_subquestion_count"]
                            retrieved_doc_ids = recovery["retrieved_doc_ids"]
                            retrieved_docs_total = recovery["retrieved_docs_total"]
                            answer_obj = recovery["answer_obj"]

                    if self._should_escalate_answer(answer_obj, pending_slots) and step < self.max_steps:
                        answer_rejection_count += 1
                        step_trace.append(
                            StepTrace(
                                step=step,
                                action="answer_rejected_escalate",
                                tokens=decide_tokens + answer_tokens,
                                cited_fact_ids=answer_obj["cited_fact_ids"],
                                justification_confidence=answer_obj["justification_confidence"],
                                metadata={
                                    "justification": answer_obj["justification"],
                                    "missing_slot": answer_obj["missing_slot"],
                                    "fallback_source": answer_obj.get("fallback_source", ""),
                                },
                            )
                        )
                        decision, spawn_tokens = await self.orchestrator.propose_spawn(
                            question=question,
                            facts=memory.get_all(),
                            trace=step_trace,
                            target_profile=target_profile,
                            pending_slots=pending_slots,
                            missing_reason=(
                                answer_obj["missing_slot"]
                                or "The current answer could not be justified from the fact pool."
                            ),
                        )
                        total_tokens += spawn_tokens
                        orchestrator_tokens += spawn_tokens
                        action = "spawn"
                        decide_tokens += spawn_tokens
                    elif (
                        self._should_escalate_answer(answer_obj, pending_slots)
                        and self._should_do_final_targeted_recovery(answer_obj, pending_slots)
                    ):
                        recovery_slot_name = self._terminal_pending_slot(slot_state)
                        recovery_pending_slots = [
                            slot
                            for slot in pending_slots
                            if str(slot.get("slot_name", "")).strip() == recovery_slot_name
                        ] or pending_slots
                        answer_rejection_count += 1
                        step_trace.append(
                            StepTrace(
                                step=step,
                                action="answer_rejected_escalate",
                                tokens=decide_tokens + answer_tokens,
                                cited_fact_ids=answer_obj["cited_fact_ids"],
                                justification_confidence=answer_obj["justification_confidence"],
                                metadata={
                                    "justification": answer_obj["justification"],
                                    "missing_slot": answer_obj["missing_slot"],
                                    "recovery_mode": "final_targeted_probe",
                                    "fallback_source": answer_obj.get("fallback_source", ""),
                                },
                            )
                        )
                        recovery_decision, recovery_tokens = await self.orchestrator.propose_spawn(
                            question=question,
                            facts=memory.get_all(),
                            trace=step_trace,
                            target_profile=target_profile,
                            pending_slots=recovery_pending_slots,
                            missing_reason=(
                                f"Resolve the terminal slot '{recovery_slot_name}' directly. "
                                "Do not answer with an already known bridge entity or upstream slot."
                            ),
                        )
                        total_tokens += recovery_tokens
                        orchestrator_tokens += recovery_tokens
                        if recovery_decision.get("action") == "spawn":
                            slot_name = recovery_decision.get("slot_name") or recovery_slot_name
                            sub_question = recovery_decision["sub_question"]
                            retrieval_query = str(recovery_decision.get("retrieval_query", "")).strip()
                            goal = recovery_decision["goal"]
                            duplicate_subquestion_count += int(
                                self._is_duplicate_subquestion(sub_question, step_trace)
                            )
                            capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                                sub_question=sub_question,
                                goal=goal,
                                prior_facts=memory.get_all(),
                                retrieval_query=retrieval_query or None,
                                slot_name=slot_name,
                                slot_hint=self._slot_hint(slot_state, slot_name),
                            )
                            total_tokens += investigate_tokens
                            subagent_tokens += investigate_tokens
                            subagent_calls += 1
                            retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                                retrieved_doc_ids,
                                retrieved_docs_total,
                                capsule,
                            )
                            fact_added = self._add_fact(
                                memory,
                                capsule,
                                step=step + 1,
                                slot_name=slot_name,
                            )
                            self._update_slot_resolution(slot_state, slot_name, capsule)
                            step_trace.append(
                                StepTrace(
                                    step=step + 1,
                                    action="spawn",
                                    sub_question=sub_question,
                                    fact_added=fact_added,
                                    tokens=recovery_tokens + investigate_tokens,
                                    slot_name=slot_name,
                                    metadata={
                                        "goal": goal,
                                        "recovery_mode": "final_targeted_probe",
                                        "retrieval_query": retrieval_query or sub_question,
                                    },
                                )
                            )
                            (
                                verify_tokens,
                                verify_delta,
                                auto_verify_delta,
                            ) = await self._maybe_verify_fact(
                                question=question,
                                step=step + 1,
                                slot_name=slot_name,
                                sub_question=sub_question,
                                capsule=capsule,
                                memory=memory,
                                slot_state=slot_state,
                                step_trace=step_trace,
                            )
                            total_tokens += verify_tokens
                            subagent_tokens += verify_tokens
                            verify_count += verify_delta
                            auto_verify_calls += auto_verify_delta
                        final_pending_slots = self._pending_slots(slot_state)
                        if force_planned_recurse and not final_pending_slots:
                            final_answer_obj = self._direct_terminal_slot_answer_object(
                                facts=memory.get_all(),
                                slot_state=slot_state,
                                expected_answer_type=expected_answer_type,
                            )
                            final_answer_tokens = 0
                        else:
                            final_answer_obj, final_answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
                                question=question,
                                facts=memory.get_all(),
                                target_profile=target_profile,
                                pending_slots=final_pending_slots,
                                trace=step_trace,
                                route_draft_answer=route["draft_answer"],
                            )
                            final_answer_obj = self._apply_answer_object_fallback(
                                final_answer_obj,
                                memory.get_all(),
                                route["draft_answer"],
                                expected_answer_type=expected_answer_type,
                            )
                            final_answer_obj = self._prefer_terminal_slot_answer_object(
                                final_answer_obj,
                                memory.get_all(),
                                slot_state,
                                expected_answer_type=expected_answer_type,
                            )
                        total_tokens += final_answer_tokens
                        orchestrator_tokens += final_answer_tokens
                        answer = final_answer_obj["answer"]
                        step_trace.append(
                            StepTrace(
                                step=step + 2,
                                action="answer",
                                tokens=final_answer_tokens,
                                cited_fact_ids=final_answer_obj["cited_fact_ids"],
                                justification_confidence=final_answer_obj["justification_confidence"],
                                metadata={
                                    "justification": final_answer_obj["justification"],
                                    "missing_slot": final_answer_obj["missing_slot"],
                                    "recovery_mode": "final_targeted_probe",
                                    "fallback_source": final_answer_obj.get("fallback_source", ""),
                                },
                            )
                        )
                        break
                    else:
                        answer = answer_obj["answer"]
                        step_trace.append(
                            StepTrace(
                                step=step,
                                action="answer",
                                tokens=decide_tokens + answer_tokens,
                                cited_fact_ids=answer_obj["cited_fact_ids"],
                                justification_confidence=answer_obj["justification_confidence"],
                                metadata={
                                    "justification": answer_obj["justification"],
                                    "missing_slot": answer_obj["missing_slot"],
                                    "fallback_source": answer_obj.get("fallback_source", ""),
                                },
                            )
                        )
                        break

            if action == "spawn":
                slot_name = decision.get("slot_name") or self._first_pending_slot(slot_state)
                sub_question = decision["sub_question"]
                retrieval_query = str(decision.get("retrieval_query", "")).strip()
                goal = decision["goal"]
                duplicate_subquestion_count += int(
                    self._is_duplicate_subquestion(sub_question, step_trace)
                )
                capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                    sub_question=sub_question,
                    goal=goal,
                    prior_facts=memory.get_all(),
                    retrieval_query=retrieval_query or None,
                    slot_name=slot_name,
                    slot_hint=self._slot_hint(slot_state, slot_name),
                )
                total_tokens += investigate_tokens
                subagent_tokens += investigate_tokens
                subagent_calls += 1
                retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                    retrieved_doc_ids,
                    retrieved_docs_total,
                    capsule,
                )
                fact_added = self._add_fact(
                    memory,
                    capsule,
                    step=step,
                    slot_name=slot_name,
                )
                self._update_slot_resolution(slot_state, slot_name, capsule)
                step_trace.append(
                    StepTrace(
                        step=step,
                        action="spawn",
                        sub_question=sub_question,
                        fact_added=fact_added,
                        tokens=decide_tokens + investigate_tokens,
                        slot_name=slot_name,
                        metadata={
                            "goal": goal,
                            "retrieval_query": retrieval_query or sub_question,
                            "echo_detected_llm_call": bool(
                                decision.get("echo_detected_llm_call", False)
                            ),
                            "echo_still_present_after_refine": bool(
                                decision.get("echo_still_present_after_refine", False)
                            ),
                            "echo_repaired_count": int(
                                decision.get("echo_repaired_count", 0)
                            ),
                        },
                    )
                )
                (
                    verify_tokens,
                    verify_delta,
                    auto_verify_delta,
                ) = await self._maybe_verify_fact(
                    question=question,
                    step=step,
                    slot_name=slot_name,
                    sub_question=sub_question,
                    capsule=capsule,
                    memory=memory,
                    slot_state=slot_state,
                    step_trace=step_trace,
                )
                total_tokens += verify_tokens
                subagent_tokens += verify_tokens
                verify_count += verify_delta
                auto_verify_calls += auto_verify_delta
                if self._should_force_budget_answer(total_tokens):
                    answer_obj, total_tokens, orchestrator_tokens = await self._force_budget_answer_object(
                        question=question,
                        facts=memory.get_all(),
                        target_profile=target_profile,
                        pending_slots=self._pending_slots(slot_state),
                        step_trace=step_trace,
                        step=step + 1,
                        total_tokens=total_tokens,
                        orchestrator_tokens=orchestrator_tokens,
                        route_draft_answer=route["draft_answer"],
                        metadata={"budget_stage": "post_spawn"},
                    )
                    answer = answer_obj["answer"]
                    break
                if force_planned_recurse and not self._pending_slots(slot_state):
                    answer_obj = self._direct_terminal_slot_answer_object(
                        facts=memory.get_all(),
                        slot_state=slot_state,
                        expected_answer_type=expected_answer_type,
                    )
                    answer = answer_obj["answer"]
                    step_trace.append(
                        StepTrace(
                            step=step + 1,
                            action="answer",
                            tokens=0,
                            cited_fact_ids=answer_obj["cited_fact_ids"],
                            justification_confidence=answer_obj["justification_confidence"],
                            metadata={
                                "justification": answer_obj["justification"],
                                "missing_slot": answer_obj["missing_slot"],
                                "fallback_source": answer_obj.get("fallback_source", ""),
                                "direct_terminal_slot_answer": True,
                            },
                        )
                    )
                    break
                continue

            if action == "refine":
                slot_name = decision.get("slot_name") or self._first_pending_slot(slot_state)
                if refine_count_by_slot.get(slot_name, 0) >= self.refine_budget_per_slot:
                    continue
                refine_count_by_slot[slot_name] = refine_count_by_slot.get(slot_name, 0) + 1
                sub_question = decision["sub_question"]
                retrieval_query = str(decision.get("retrieval_query", "")).strip()
                goal = decision["goal"]
                duplicate_subquestion_count += int(
                    self._is_duplicate_subquestion(sub_question, step_trace)
                )
                capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                    sub_question=sub_question,
                    goal=goal,
                    prior_facts=memory.get_all(),
                    retrieval_query=retrieval_query or None,
                    slot_name=slot_name,
                    slot_hint=self._slot_hint(slot_state, slot_name),
                )
                total_tokens += investigate_tokens
                subagent_tokens += investigate_tokens
                subagent_calls += 1
                retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                    retrieved_doc_ids,
                    retrieved_docs_total,
                    capsule,
                )
                fact_added = self._replace_fact(
                    memory,
                    capsule,
                    step=step,
                    slot_name=slot_name,
                )
                self._update_slot_resolution(slot_state, slot_name, capsule)
                step_trace.append(
                    StepTrace(
                        step=step,
                        action="refine",
                        sub_question=sub_question,
                        fact_added=fact_added,
                        tokens=decide_tokens + investigate_tokens,
                        slot_name=slot_name,
                        metadata={
                            "goal": goal,
                            "retrieval_query": retrieval_query or sub_question,
                            "refine_count": refine_count_by_slot[slot_name],
                            "echo_detected_llm_call": bool(
                                decision.get("echo_detected_llm_call", False)
                            ),
                            "echo_still_present_after_refine": bool(
                                decision.get("echo_still_present_after_refine", False)
                            ),
                            "echo_repaired_count": int(
                                decision.get("echo_repaired_count", 0)
                            ),
                        },
                    )
                )
                if self._should_force_budget_answer(total_tokens):
                    answer_obj, total_tokens, orchestrator_tokens = await self._force_budget_answer_object(
                        question=question,
                        facts=memory.get_all(),
                        target_profile=target_profile,
                        pending_slots=self._pending_slots(slot_state),
                        step_trace=step_trace,
                        step=step + 1,
                        total_tokens=total_tokens,
                        orchestrator_tokens=orchestrator_tokens,
                        route_draft_answer=route["draft_answer"],
                        metadata={"budget_stage": "post_refine"},
                    )
                    answer = answer_obj["answer"]
                    break
                continue

            logger.warning("Unknown action %r at step %d — forcing answer", action, step)
            answer_obj, answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
                question=question,
                facts=memory.get_all(),
                target_profile=target_profile,
                pending_slots=self._pending_slots(slot_state),
                trace=step_trace,
                route_draft_answer=route["draft_answer"],
            )
            answer_obj = self._apply_answer_object_fallback(
                answer_obj,
                memory.get_all(),
                route["draft_answer"],
                expected_answer_type=expected_answer_type,
            )
            answer_obj = self._prefer_terminal_slot_answer_object(
                answer_obj,
                memory.get_all(),
                slot_state,
                expected_answer_type=expected_answer_type,
            )
            total_tokens += answer_tokens
            orchestrator_tokens += answer_tokens
            answer = answer_obj["answer"]
            step_trace.append(
                StepTrace(
                    step=step,
                    action="answer",
                    tokens=answer_tokens,
                    cited_fact_ids=answer_obj["cited_fact_ids"],
                    justification_confidence=answer_obj["justification_confidence"],
                    metadata={
                        "justification": answer_obj["justification"],
                        "missing_slot": answer_obj["missing_slot"],
                        "fallback_source": answer_obj.get("fallback_source", ""),
                    },
                )
            )
            break
        else:
            answer_obj, answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
                question=question,
                facts=memory.get_all(),
                target_profile=target_profile,
                pending_slots=self._pending_slots(slot_state),
                trace=step_trace,
                route_draft_answer=route["draft_answer"],
            )
            answer_obj = self._apply_answer_object_fallback(
                answer_obj,
                memory.get_all(),
                route["draft_answer"],
                expected_answer_type=expected_answer_type,
            )
            answer_obj = self._prefer_terminal_slot_answer_object(
                answer_obj,
                memory.get_all(),
                slot_state,
                expected_answer_type=expected_answer_type,
            )
            total_tokens += answer_tokens
            orchestrator_tokens += answer_tokens
            answer = answer_obj["answer"]
            step_trace.append(
                StepTrace(
                    step=self.max_steps + 1,
                    action="answer",
                    tokens=answer_tokens,
                    cited_fact_ids=answer_obj["cited_fact_ids"],
                    justification_confidence=answer_obj["justification_confidence"],
                    metadata={
                        "justification": answer_obj["justification"],
                        "missing_slot": answer_obj["missing_slot"],
                        "fallback_source": answer_obj.get("fallback_source", ""),
                    },
                )
            )

        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=step_trace,
            num_subagent_calls=subagent_calls,
            num_verify_calls=verify_count,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=memory.get_all(),
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
            fact_memory_capacity=self.fact_memory_capacity,
            duplicate_subquestion_count=duplicate_subquestion_count,
            route_decision=route["action"],
            route_confidence=route["confidence"],
            route_draft_answer=route["draft_answer"],
            slot_resolution=self._slot_resolution_map(slot_state),
            auto_verify_calls=auto_verify_calls,
            answer_rejection_count=answer_rejection_count,
        )

    async def _run_adaptive(
        self,
        question: str,
        question_id: str,
        bootstrap_sub_question: str | None = None,
        bootstrap_goal: str | None = None,
        min_subagent_calls_before_answer: int = 1,
        max_steps_override: int | None = None,
    ) -> PipelineResult:
        """Default adaptive loop."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        target_profile = self._target_profile(question)
        step_trace: list[StepTrace] = []
        verify_count = 0
        total_tokens = 0
        orchestrator_tokens = 0
        subagent_tokens = 0
        subagent_calls = 0
        answer = ""
        duplicate_subquestion_count = 0
        retrieved_doc_ids: list[str] = []
        retrieved_docs_total = 0
        max_steps = max_steps_override if max_steps_override is not None else self.max_steps

        bootstrap_sub_question = (bootstrap_sub_question or question).strip() or question
        bootstrap_goal = (
            (bootstrap_goal or "").strip()
            or (
                "Answer the original question directly if the evidence supports it. "
                f"{target_profile}"
            )
        )

        bootstrap_capsule, bootstrap_tokens = await self.investigator.investigate_with_usage(
            sub_question=bootstrap_sub_question,
            goal=bootstrap_goal,
            prior_facts=[],
        )
        total_tokens += bootstrap_tokens
        subagent_tokens += bootstrap_tokens
        subagent_calls += 1
        retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
            retrieved_doc_ids,
            retrieved_docs_total,
            bootstrap_capsule,
        )

        bootstrap_fact_added = self._add_fact(memory, bootstrap_capsule, step=0)
        step_trace.append(
            StepTrace(
                step=0,
                action="spawn",
                sub_question=bootstrap_sub_question,
                claim=None,
                fact_added=bootstrap_fact_added,
                tokens=bootstrap_tokens,
            )
        )
        if (
            self._capsule_supports_answer(bootstrap_capsule, self.direct_answer_threshold)
            and self._allow_bootstrap_short_circuit(question)
        ):
            answer = bootstrap_capsule.answer
            step_trace.append(
                StepTrace(
                    step=1,
                    action="answer",
                    sub_question=None,
                    claim=None,
                    fact_added=False,
                    tokens=0,
                )
            )
            return PipelineResult(
                question_id=question_id,
                question=question,
                answer=answer,
                step_trace=step_trace,
                num_subagent_calls=subagent_calls,
                num_verify_calls=verify_count,
                total_tokens=total_tokens,
                orchestrator_tokens=orchestrator_tokens,
                subagent_tokens=subagent_tokens,
                facts_used=memory.get_all(),
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_docs_total=retrieved_docs_total,
                evidence_capsule_limit=self.investigator.evidence_capsule_limit,
                fact_memory_capacity=self.fact_memory_capacity,
                duplicate_subquestion_count=duplicate_subquestion_count,
            )

        for step in range(1, max_steps + 1):
            if self._trace_cap_reached(step_trace):
                answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                    question,
                    memory.get_all(),
                    target_profile,
                    trace=step_trace,
                )
                answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
                total_tokens += answer_tokens
                orchestrator_tokens += answer_tokens
                step_trace.append(
                    StepTrace(
                        step=step,
                        action="answer",
                        sub_question=None,
                        claim=None,
                        fact_added=False,
                        tokens=answer_tokens,
                        metadata={"trace_cap_exhausted": True},
                    )
                )
                break
            if self._should_force_budget_answer(total_tokens):
                answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                    question,
                    memory.get_all(),
                    target_profile,
                    trace=step_trace,
                )
                answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
                total_tokens += answer_tokens
                orchestrator_tokens += answer_tokens
                step_trace.append(
                    StepTrace(
                        step=step,
                        action="answer",
                        sub_question=None,
                        claim=None,
                        fact_added=False,
                        tokens=answer_tokens,
                        metadata={"budget_exhausted": True},
                    )
                )
                break
            if self.ablation_force_spawn and step < max_steps:
                decision, decide_tokens = await self.orchestrator.propose_spawn(
                    question=question,
                    facts=memory.get_all(),
                    trace=step_trace,
                    target_profile=target_profile,
                )
                action = "spawn"
            else:
                decision, decide_tokens = await self.orchestrator.decide_with_usage(
                    question=question,
                    facts=memory.get_all(),
                    trace=step_trace,
                    step=step,
                    target_profile=target_profile,
                )
                action = decision["action"]

            total_tokens += decide_tokens
            orchestrator_tokens += decide_tokens

            if self.ablation_no_verify and action == "verify":
                if step < max_steps:
                    decision, extra_tokens = await self.orchestrator.propose_spawn(
                        question=question,
                        facts=memory.get_all(),
                        trace=step_trace,
                        target_profile=target_profile,
                    )
                    total_tokens += extra_tokens
                    orchestrator_tokens += extra_tokens
                    decide_tokens += extra_tokens
                    action = "spawn"
                else:
                    decision = {"action": "answer"}
                    action = "answer"

            logger.debug("Step %d: action=%s, decision=%s", step, action, decision)

            if action == "answer":
                if subagent_calls < min_subagent_calls_before_answer and step < max_steps:
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="answer_blocked_min_depth",
                            sub_question=None,
                            claim=None,
                            fact_added=False,
                            tokens=decide_tokens,
                            metadata={
                                "subagent_calls": subagent_calls,
                                "min_subagent_calls_before_answer": min_subagent_calls_before_answer,
                            },
                        )
                    )
                    if self._trace_cap_reached(step_trace):
                        answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                            question,
                            memory.get_all(),
                            target_profile,
                            trace=step_trace,
                        )
                        answer, _, _, _ = self._apply_answer_fallback(
                            answer, memory.get_all()
                        )
                        total_tokens += answer_tokens
                        orchestrator_tokens += answer_tokens
                        step_trace.append(
                            StepTrace(
                                step=step,
                                action="answer",
                                sub_question=None,
                                claim=None,
                                fact_added=False,
                                tokens=answer_tokens,
                                metadata={
                                    "trace_cap_exhausted": True,
                                    "trace_cap_after": "answer_blocked_min_depth",
                                },
                            )
                        )
                        break
                    decision, extra_tokens = await self.orchestrator.propose_spawn(
                        question=question,
                        facts=memory.get_all(),
                        trace=step_trace,
                        target_profile=target_profile,
                        missing_reason=(
                            "This route requires recursion, so do not answer after only one retrieved fact. "
                            "Retrieve the next missing fact first."
                        ),
                    )
                    total_tokens += extra_tokens
                    orchestrator_tokens += extra_tokens
                    decide_tokens += extra_tokens
                    action = "spawn"
                    duplicate_subquestion_count += int(
                        self._is_duplicate_subquestion(decision["sub_question"], step_trace)
                    )
                    step_trace[-1].tokens = decide_tokens
                    step_trace[-1].metadata["forced_spawn_sub_question"] = decision["sub_question"]
                    step_trace[-1].metadata["forced_spawn_goal"] = decision["goal"]
                else:
                    answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                        question,
                        memory.get_all(),
                        target_profile,
                        trace=step_trace,
                    )
                    answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
                    total_tokens += answer_tokens
                    orchestrator_tokens += answer_tokens
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="answer",
                            sub_question=None,
                            claim=None,
                            fact_added=False,
                            tokens=decide_tokens + answer_tokens,
                        )
                    )
                    break

            if action == "spawn":
                sub_question = decision["sub_question"]
                goal = decision["goal"]
                duplicate_subquestion_count += int(
                    self._is_duplicate_subquestion(sub_question, step_trace)
                )

                capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                    sub_question=sub_question,
                    goal=goal,
                    prior_facts=memory.get_all(),
                )
                total_tokens += investigate_tokens
                subagent_tokens += investigate_tokens
                subagent_calls += 1
                retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                    retrieved_doc_ids,
                    retrieved_docs_total,
                    capsule,
                )

                fact_added = self._add_fact(memory, capsule, step=step)
                step_trace.append(
                    StepTrace(
                        step=step,
                        action="spawn",
                        sub_question=sub_question,
                        claim=None,
                        fact_added=fact_added,
                        tokens=decide_tokens + investigate_tokens,
                    )
                )

                if self.ablation_always_verify and capsule.fact.text:
                    verify_tokens = await self._auto_verify(step, capsule, step_trace)
                    total_tokens += verify_tokens
                    orchestrator_tokens += verify_tokens
                    verify_count += 1
                if self._trace_cap_reached(step_trace):
                    answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                        question,
                        memory.get_all(),
                        target_profile,
                        trace=step_trace,
                    )
                    answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
                    total_tokens += answer_tokens
                    orchestrator_tokens += answer_tokens
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="answer",
                            sub_question=None,
                            claim=None,
                            fact_added=False,
                            tokens=answer_tokens,
                            metadata={
                                "trace_cap_exhausted": True,
                                "trace_cap_after": step_trace[-1].action,
                            },
                        )
                    )
                    break
                continue

            if action == "verify":
                claim = decision["claim"]
                if verify_count >= self.max_verify_calls:
                    logger.debug(
                        "Verify budget exhausted (%d/%d) — forcing answer",
                        verify_count,
                        self.max_verify_calls,
                    )
                    answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                        question,
                        memory.get_all(),
                        target_profile,
                        trace=step_trace,
                    )
                    answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
                    total_tokens += answer_tokens
                    orchestrator_tokens += answer_tokens
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="answer",
                            sub_question=None,
                            claim=None,
                            fact_added=False,
                            tokens=decide_tokens + answer_tokens,
                        )
                    )
                    break

                verify_result, verify_tokens = await self.orchestrator.verify_claim_with_usage(
                    claim,
                    self._build_evidence_from_facts(memory.get_all()),
                )
                total_tokens += verify_tokens
                orchestrator_tokens += verify_tokens
                verify_count += 1
                self._log_verify(step, claim, verify_result)
                step_trace.append(
                    StepTrace(
                        step=step,
                        action="verify",
                        sub_question=None,
                        claim=claim,
                        fact_added=False,
                        tokens=decide_tokens + verify_tokens,
                    )
                )
                if self._trace_cap_reached(step_trace):
                    answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                        question,
                        memory.get_all(),
                        target_profile,
                        trace=step_trace,
                    )
                    answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
                    total_tokens += answer_tokens
                    orchestrator_tokens += answer_tokens
                    step_trace.append(
                        StepTrace(
                            step=step,
                            action="answer",
                            sub_question=None,
                            claim=None,
                            fact_added=False,
                            tokens=answer_tokens,
                            metadata={
                                "trace_cap_exhausted": True,
                                "trace_cap_after": "verify",
                            },
                        )
                    )
                    break
                continue

            logger.warning("Unknown action %r at step %d — forcing answer", action, step)
            answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                question,
                memory.get_all(),
                target_profile,
                trace=step_trace,
            )
            answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
            total_tokens += answer_tokens
            orchestrator_tokens += answer_tokens
            step_trace.append(
                StepTrace(
                    step=step,
                    action="answer",
                    sub_question=None,
                    claim=None,
                    fact_added=False,
                    tokens=decide_tokens + answer_tokens,
                )
            )
            break
        else:
            logger.info("Step budget exhausted — forcing answer generation")
            answer, answer_tokens = await self.orchestrator.generate_answer_with_usage(
                question,
                memory.get_all(),
                target_profile,
                trace=step_trace,
            )
            answer, _, _, _ = self._apply_answer_fallback(answer, memory.get_all())
            total_tokens += answer_tokens
            orchestrator_tokens += answer_tokens
            step_trace.append(
                StepTrace(
                    step=max_steps + 1,
                    action="answer",
                    sub_question=None,
                    claim=None,
                    fact_added=False,
                    tokens=answer_tokens,
                )
            )

        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=step_trace,
            num_subagent_calls=subagent_calls,
            num_verify_calls=verify_count,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            facts_used=memory.get_all(),
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
            fact_memory_capacity=self.fact_memory_capacity,
            duplicate_subquestion_count=duplicate_subquestion_count,
        )

    @staticmethod
    def _add_fact(
        memory: FactMemory,
        capsule: EvidenceCapsule,
        step: int,
        slot_name: str = "",
    ) -> bool:
        """Add capsule fact to memory if non-empty."""
        if not capsule.fact.text:
            return False
        capsule.fact.source_step = step
        capsule.fact.slot_name = slot_name
        memory.add(capsule.fact)
        return True

    @staticmethod
    def _replace_fact(
        memory: FactMemory,
        capsule: EvidenceCapsule,
        step: int,
        slot_name: str = "",
    ) -> bool:
        """Replace the slot fact during refinement, else append."""
        if not capsule.fact.text:
            return False
        capsule.fact.source_step = step
        capsule.fact.slot_name = slot_name
        memory.replace(slot_name, capsule.fact)
        return True

    @staticmethod
    def _build_evidence_from_facts(facts: list) -> str:
        """Format current facts for claim verification."""
        evidence_parts: list[str] = []
        for fact in facts:
            evidence_parts.append(fact.text)
            if fact.support_ids:
                evidence_parts.append(
                    f"  (supported by chunks: {', '.join(fact.support_ids)})"
                )
        return "\n".join(evidence_parts) if evidence_parts else "No evidence available."

    def _initialise_slot_state(
        self,
        route: dict[str, Any],
        target_profile: str,
    ) -> list[dict[str, Any]]:
        """Initialise slot state from the router output."""
        slot_state: list[dict[str, Any]] = []
        raw_slots = route.get("required_hops", []) or []
        for idx, item in enumerate(raw_slots):
            slot_name = str(item.get("slot_name", "")).strip()
            hint = str(item.get("hint", "")).strip()
            if not slot_name:
                continue
            dependency_group = item.get("dependency_group", idx)
            try:
                dependency_group = max(0, int(dependency_group))
            except (TypeError, ValueError):
                dependency_group = idx
            slot_state.append(
                {
                    "slot_name": slot_name,
                    "hint": hint,
                    "resolved": False,
                    "dependency_group": dependency_group,
                }
            )
        if not slot_state:
            slot_state.append(
                {
                    "slot_name": str(route.get("target_slot", "final_answer")).strip()
                    or "final_answer",
                    "hint": target_profile,
                    "resolved": False,
                    "dependency_group": 0,
                }
            )
        return slot_state

    @staticmethod
    def _slot_snapshot(slot_state: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a serialisable snapshot of slot state."""
        return [
            {
                "slot_name": str(slot.get("slot_name", "")),
                "hint": str(slot.get("hint", "")),
                "resolved": bool(slot.get("resolved", False)),
                "dependency_group": int(slot.get("dependency_group", 0)),
            }
            for slot in slot_state
        ]

    @staticmethod
    def _pending_slots(slot_state: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Return unresolved slot specs for prompt injection."""
        return [
            {
                "slot_name": str(slot.get("slot_name", "")),
                "hint": str(slot.get("hint", "")),
                "resolved": bool(slot.get("resolved", False)),
                "dependency_group": int(slot.get("dependency_group", 0)),
            }
            for slot in slot_state
            if not slot.get("resolved", False)
        ]

    @staticmethod
    def _slot_resolution_map(slot_state: list[dict[str, Any]]) -> dict[str, bool]:
        """Return slot resolution as a compact mapping."""
        return {
            str(slot.get("slot_name", "")): bool(slot.get("resolved", False))
            for slot in slot_state
        }

    @staticmethod
    def _first_pending_slot(slot_state: list[dict[str, Any]]) -> str:
        """Return the first unresolved slot name."""
        for slot in slot_state:
            if not slot.get("resolved", False):
                return str(slot.get("slot_name", "")).strip()
        return str(slot_state[0].get("slot_name", "final_answer")).strip() if slot_state else "final_answer"

    @staticmethod
    def _terminal_pending_slot(slot_state: list[dict[str, Any]]) -> str:
        """Return the last unresolved slot name."""
        for slot in reversed(slot_state):
            if not slot.get("resolved", False):
                return str(slot.get("slot_name", "")).strip()
        return str(slot_state[-1].get("slot_name", "final_answer")).strip() if slot_state else "final_answer"

    @staticmethod
    def _update_slot_resolution(
        slot_state: list[dict[str, Any]],
        slot_name: str,
        capsule: EvidenceCapsule,
    ) -> None:
        """Mark a slot as resolved when a grounded capsule fills it."""
        if not slot_name:
            return
        for slot in slot_state:
            if str(slot.get("slot_name", "")).strip() == slot_name:
                if (
                    capsule.fact.slot_filled
                    and capsule.fact.confidence > 0
                    and capsule.fact.text.strip()
                ):
                    slot["resolved"] = True
                break

    def _should_escalate_answer(
        self,
        answer_obj: dict[str, Any],
        pending_slots: list[dict[str, str]],
    ) -> bool:
        """Return whether an answer attempt should be rejected and escalated."""
        if pending_slots:
            return True
        if answer_obj.get("missing_slot"):
            return True
        if not answer_obj.get("cited_fact_ids"):
            return True
        if answer_obj.get("justification_confidence", 0.0) < self.answer_justification_threshold:
            return True
        if not answer_obj.get("answer", "").strip():
            return True
        return False

    @staticmethod
    def _should_do_final_targeted_recovery(
        answer_obj: dict[str, Any],
        pending_slots: list[dict[str, str]],
    ) -> bool:
        """Recover at the boundary whenever unresolved slots remain."""
        if not pending_slots:
            return False
        missing_slot = str(answer_obj.get("missing_slot", "")).strip()
        if not missing_slot:
            return True
        pending_slot_names = {
            str(slot.get("slot_name", "")).strip()
            for slot in pending_slots
            if str(slot.get("slot_name", "")).strip()
        }
        return missing_slot in pending_slot_names

    @staticmethod
    def _answer_matches_expected_type(answer: str, expected_answer_type: str) -> bool:
        """Return whether an answer shape matches the expected answer type."""
        text = str(answer or "").strip()
        if not text or Orchestrator._looks_meta_answer(text):
            return False

        expected = str(expected_answer_type or "entity").strip().lower()
        if expected in {"person", "location", "entity", "other"}:
            return True
        if expected == "yes/no":
            return text.lower() in {"yes", "no"}
        if expected == "date/year":
            lowered = text.lower()
            month_tokens = (
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            )
            return bool(
                re.search(r"\b\d{4}\b", lowered)
                or re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lowered)
                or any(token in lowered for token in month_tokens)
                or re.search(r"\b(?:ad|bc)\b", lowered)
            )
        if expected == "number":
            lowered = text.lower()
            number_words = {
                "zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
                "seventy", "eighty", "ninety", "hundred", "thousand", "million",
                "billion",
            }
            return bool(re.search(r"\d", lowered) or any(word in lowered.split() for word in number_words))
        if expected == "list":
            lowered = text.lower()
            return "," in text or ";" in text or " and " in lowered
        return True

    @classmethod
    def _best_fact_answer_with_index(
        cls,
        facts: list,
        expected_answer_type: str = "entity",
    ) -> tuple[int | None, str, float]:
        """Return the strongest grounded answer span in fact memory."""
        candidates: list[tuple[float, int, int, int, str]] = []
        for idx, fact in enumerate(facts, start=1):
            answer_span = str(getattr(fact, "answer_span", "")).strip()
            if (
                not answer_span
                or Orchestrator._looks_meta_answer(answer_span)
                or not cls._answer_matches_expected_type(answer_span, expected_answer_type)
            ):
                continue
            candidates.append(
                (fact.confidence, fact.source_step, -len(answer_span), idx, answer_span)
            )
        if not candidates:
            return None, "", 0.0
        candidates.sort(reverse=True)
        best = candidates[0]
        return best[3], best[4], best[0]

    @classmethod
    def _best_slot_fact_text_with_index(
        cls,
        facts: list,
        slot_name: str,
        expected_answer_type: str = "entity",
    ) -> tuple[int | None, str, float]:
        """Return the strongest grounded fact text for a specific slot."""
        target_slot = str(slot_name or "").strip()
        if not target_slot:
            return None, "", 0.0

        candidates: list[tuple[float, int, int, str]] = []
        for idx, fact in enumerate(facts, start=1):
            if str(getattr(fact, "slot_name", "")).strip() != target_slot:
                continue
            fact_text = str(getattr(fact, "text", "")).strip()
            if (
                not fact_text
                or Orchestrator._looks_meta_answer(fact_text)
                or not cls._answer_matches_expected_type(fact_text, expected_answer_type)
            ):
                continue
            candidates.append((fact.confidence, fact.source_step, idx, fact_text))
        if not candidates:
            return None, "", 0.0
        candidates.sort(reverse=True)
        best = candidates[0]
        return best[2], best[3], best[0]

    @classmethod
    def _best_slot_answer_with_index(
        cls,
        facts: list,
        slot_name: str,
        expected_answer_type: str = "entity",
    ) -> tuple[int | None, str, float]:
        """Return the strongest grounded answer span for a specific slot."""
        target_slot = str(slot_name or "").strip()
        if not target_slot:
            return None, "", 0.0

        candidates: list[tuple[float, int, int, int, str]] = []
        for idx, fact in enumerate(facts, start=1):
            if str(getattr(fact, "slot_name", "")).strip() != target_slot:
                continue
            answer_span = str(getattr(fact, "answer_span", "")).strip()
            if (
                not answer_span
                or Orchestrator._looks_meta_answer(answer_span)
                or not cls._answer_matches_expected_type(answer_span, expected_answer_type)
            ):
                continue
            candidates.append(
                (fact.confidence, fact.source_step, -len(answer_span), idx, answer_span)
            )
        if not candidates:
            return None, "", 0.0
        candidates.sort(reverse=True)
        best = candidates[0]
        return best[3], best[4], best[0]

    @classmethod
    def _apply_answer_fallback(
        cls,
        answer: str,
        facts: list,
        route_draft_answer: str = "",
        expected_answer_type: str = "entity",
    ) -> tuple[str, str, list[int], float]:
        """Fallback from empty/meta answers to grounded fact memory only."""
        cleaned_answer = str(answer or "").strip()
        if cleaned_answer and cls._answer_matches_expected_type(
            cleaned_answer,
            expected_answer_type,
        ):
            return cleaned_answer, "", [], 0.0

        fact_idx, fact_answer, fact_confidence = cls._best_fact_answer_with_index(
            facts,
            expected_answer_type=expected_answer_type,
        )
        if fact_answer:
            return fact_answer, "fact_memory", [fact_idx] if fact_idx else [], fact_confidence

        return "", "", [], 0.0

    @classmethod
    def _apply_answer_object_fallback(
        cls,
        answer_obj: dict[str, Any],
        facts: list,
        route_draft_answer: str = "",
        expected_answer_type: str = "entity",
    ) -> dict[str, Any]:
        """Fill empty structured answers from fact memory or router draft."""
        updated = dict(answer_obj)
        answer, fallback_source, cited_fact_ids, fallback_confidence = cls._apply_answer_fallback(
            updated.get("answer", ""),
            facts,
            route_draft_answer,
            expected_answer_type=expected_answer_type,
        )
        updated["answer"] = answer
        if fallback_source == "fact_memory" and not updated.get("cited_fact_ids"):
            updated["cited_fact_ids"] = cited_fact_ids
            updated["justification_confidence"] = max(
                float(updated.get("justification_confidence", 0.0)),
                fallback_confidence,
            )
        updated["fallback_source"] = fallback_source
        return updated

    @classmethod
    def _prefer_terminal_slot_answer_object(
        cls,
        answer_obj: dict[str, Any],
        facts: list,
        slot_state: list[dict[str, Any]],
        expected_answer_type: str = "entity",
    ) -> dict[str, Any]:
        """Prefer grounded text from the terminal slot over bridge-slot answers."""
        final_slot = cls._final_slot_name(slot_state)
        if not final_slot:
            return answer_obj

        final_fact_ids = {
            idx
            for idx, fact in enumerate(facts, start=1)
            if str(getattr(fact, "slot_name", "")).strip() == final_slot
        }
        if not final_fact_ids:
            return answer_obj

        cited_fact_ids = {
            int(item)
            for item in answer_obj.get("cited_fact_ids", [])
            if str(item).strip().isdigit()
        }
        if cited_fact_ids & final_fact_ids:
            return answer_obj

        fact_idx, fact_answer, fact_confidence = cls._best_slot_answer_with_index(
            facts,
            final_slot,
            expected_answer_type=expected_answer_type,
        )
        if not fact_idx or not fact_answer:
            fact_idx, fact_answer, fact_confidence = cls._best_slot_fact_text_with_index(
                facts,
                final_slot,
                expected_answer_type=expected_answer_type,
            )
            if not fact_idx or not fact_answer:
                return answer_obj

        updated = dict(answer_obj)
        updated["answer"] = fact_answer
        updated["cited_fact_ids"] = [fact_idx]
        updated["justification_confidence"] = max(
            float(updated.get("justification_confidence", 0.0)),
            fact_confidence,
        )
        updated["fallback_source"] = "final_slot_answer"
        return updated

    @classmethod
    def _direct_terminal_slot_answer_object(
        cls,
        *,
        facts: list,
        slot_state: list[dict[str, Any]],
        expected_answer_type: str = "entity",
    ) -> dict[str, Any]:
        """Return the terminal slot answer directly, without synthesis."""
        final_slot = cls._final_slot_name(slot_state)
        fact_idx, answer, fact_confidence = cls._best_slot_answer_with_index(
            facts,
            final_slot,
            expected_answer_type=expected_answer_type,
        )
        fallback_source = "final_slot_answer"
        if not answer:
            fact_idx, answer, fact_confidence = cls._best_slot_fact_text_with_index(
                facts,
                final_slot,
                expected_answer_type=expected_answer_type,
            )
            fallback_source = "final_slot_fact"
        if not answer:
            return cls._apply_answer_object_fallback(
                {
                    "answer": "",
                    "justification": "No grounded terminal-slot answer span available.",
                    "cited_fact_ids": [],
                    "justification_confidence": 0.0,
                    "missing_slot": final_slot,
                    "fallback_source": "",
                },
                facts,
                expected_answer_type=expected_answer_type,
            )
        return {
            "answer": answer,
            "justification": f"Direct terminal-slot answer from '{final_slot}'.",
            "cited_fact_ids": [fact_idx] if fact_idx else [],
            "justification_confidence": fact_confidence,
            "missing_slot": "",
            "fallback_source": fallback_source,
        }

    @classmethod
    def _strict_terminal_slot_answer_object(
        cls,
        *,
        facts: list,
        slot_state: list[dict[str, Any]],
        expected_answer_type: str = "entity",
    ) -> dict[str, Any]:
        """Return only the terminal slot value, with no cross-slot fallback."""
        final_slot = cls._final_slot_name(slot_state)
        fact_idx, answer, fact_confidence = cls._best_slot_answer_with_index(
            facts,
            final_slot,
            expected_answer_type=expected_answer_type,
        )
        fallback_source = "final_slot_answer"
        if not answer:
            fact_idx, answer, fact_confidence = cls._best_slot_fact_text_with_index(
                facts,
                final_slot,
                expected_answer_type=expected_answer_type,
            )
            fallback_source = "final_slot_fact"
        if not answer:
            return {
                "answer": "",
                "justification": f"No grounded answer for terminal slot '{final_slot}'.",
                "cited_fact_ids": [],
                "justification_confidence": 0.0,
                "missing_slot": final_slot,
                "fallback_source": "",
            }
        return {
            "answer": answer,
            "justification": f"Terminal slot '{final_slot}' answered directly.",
            "cited_fact_ids": [fact_idx] if fact_idx else [],
            "justification_confidence": fact_confidence,
            "missing_slot": "",
            "fallback_source": fallback_source,
        }

    @classmethod
    def _best_slot_value_with_index(
        cls,
        facts: list,
        slot_name: str,
        expected_answer_type: str = "entity",
    ) -> tuple[int | None, str, float]:
        """Return the strongest slot answer, falling back to slot fact text."""
        fact_idx, value, confidence = cls._best_slot_answer_with_index(
            facts,
            slot_name,
            expected_answer_type=expected_answer_type,
        )
        if value:
            return fact_idx, value, confidence
        return cls._best_slot_fact_text_with_index(
            facts,
            slot_name,
            expected_answer_type=expected_answer_type,
        )

    @staticmethod
    def _parallel_ready_slots(slot_state: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return unresolved slots in the earliest dependency group."""
        unresolved = [slot for slot in slot_state if not slot.get("resolved", False)]
        if not unresolved:
            return []
        min_group = min(int(slot.get("dependency_group", 0)) for slot in unresolved)
        return [
            slot for slot in unresolved if int(slot.get("dependency_group", 0)) == min_group
        ]

    @staticmethod
    def _slot_hint(slot_state: list[dict[str, Any]], slot_name: str) -> str:
        """Return the hint attached to a slot."""
        for slot in slot_state:
            if str(slot.get("slot_name", "")).strip() == slot_name:
                return str(slot.get("hint", "")).strip()
        return ""

    async def _run_parallel_slot_batch(
        self,
        *,
        question: str,
        step: int,
        route: dict[str, Any],
        target_profile: str,
        memory: FactMemory,
        slot_state: list[dict[str, Any]],
        step_trace: list[StepTrace],
    ) -> dict[str, Any]:
        """Resolve independent ready slots in parallel."""
        ready_slots = self._parallel_ready_slots(slot_state)[: self.max_parallel_hops]
        if len(ready_slots) <= 1:
            return {
                "total_tokens": 0,
                "orchestrator_tokens": 0,
                "subagent_tokens": 0,
                "subagent_calls": 0,
                "verify_count": 0,
                "auto_verify_calls": 0,
                "duplicate_subquestion_count": 0,
                "retrieved_doc_ids": [],
                "retrieved_docs_total": 0,
            }

        memory_snapshot = memory.get_all()
        batch_decisions: list[tuple[dict[str, Any], int]] = []
        orchestrator_tokens = 0

        for slot in ready_slots:
            slot_name = str(slot.get("slot_name", "")).strip()
            slot_hint = str(slot.get("hint", "")).strip() or "Resolve the slot."
            decision, decide_tokens = await self.orchestrator.propose_spawn(
                question=question,
                facts=memory_snapshot,
                trace=step_trace,
                target_profile=target_profile,
                pending_slots=[slot],
                missing_reason=(
                    f"Resolve slot '{slot_name}' ({slot_hint}) independently. "
                    "This slot can be executed in parallel with other slots in the same dependency group."
                ),
            )
            batch_decisions.append((decision, decide_tokens))
            orchestrator_tokens += decide_tokens

        investigations = await asyncio.gather(
            *[
                self.investigator.investigate_with_usage(
                    sub_question=decision["sub_question"],
                    goal=decision["goal"],
                    prior_facts=memory_snapshot,
                    retrieval_query=(
                        str(decision.get("retrieval_query", "")).strip() or None
                    ),
                    slot_name=str(decision.get("slot_name", "")).strip(),
                    slot_hint=self._slot_hint(
                        slot_state,
                        str(decision.get("slot_name", "")).strip(),
                    ),
                )
                for decision, _ in batch_decisions
            ]
        )

        result = {
            "total_tokens": orchestrator_tokens,
            "orchestrator_tokens": orchestrator_tokens,
            "subagent_tokens": 0,
            "subagent_calls": len(batch_decisions),
            "verify_count": 0,
            "auto_verify_calls": 0,
            "duplicate_subquestion_count": 0,
            "retrieved_doc_ids": [],
            "retrieved_docs_total": 0,
        }

        for slot, (decision, decide_tokens), (capsule, investigate_tokens) in zip(
            ready_slots,
            batch_decisions,
            investigations,
            strict=False,
        ):
            slot_name = str(decision.get("slot_name", "")).strip() or str(
                slot.get("slot_name", "")
            ).strip()
            sub_question = decision["sub_question"]
            goal = decision["goal"]
            retrieval_query = str(decision.get("retrieval_query", "")).strip()
            result["duplicate_subquestion_count"] += int(
                self._is_duplicate_subquestion(sub_question, step_trace)
            )
            result["total_tokens"] += investigate_tokens
            result["subagent_tokens"] += investigate_tokens
            result["retrieved_doc_ids"], result["retrieved_docs_total"] = self._merge_retrieval_stats(
                result["retrieved_doc_ids"],
                result["retrieved_docs_total"],
                capsule,
            )
            fact_added = self._add_fact(memory, capsule, step=step, slot_name=slot_name)
            self._update_slot_resolution(slot_state, slot_name, capsule)
            step_trace.append(
                StepTrace(
                    step=step,
                    action="spawn",
                    sub_question=sub_question,
                    fact_added=fact_added,
                    tokens=decide_tokens + investigate_tokens,
                    slot_name=slot_name,
                    metadata={
                        "goal": goal,
                        "retrieval_query": retrieval_query or sub_question,
                        "parallel_group": int(slot.get("dependency_group", 0)),
                        "parallel_batch_size": len(batch_decisions),
                    },
                )
            )
            verify_tokens, verify_delta, auto_verify_delta = await self._maybe_verify_fact(
                question=question,
                step=step,
                slot_name=slot_name,
                sub_question=sub_question,
                capsule=capsule,
                memory=memory,
                slot_state=slot_state,
                step_trace=step_trace,
            )
            result["total_tokens"] += verify_tokens
            result["subagent_tokens"] += verify_tokens
            result["verify_count"] += verify_delta
            result["auto_verify_calls"] += auto_verify_delta

        return result

    async def _maybe_verify_fact(
        self,
        question: str,
        step: int,
        slot_name: str,
        sub_question: str,
        capsule: EvidenceCapsule,
        memory: FactMemory,
        slot_state: list[dict[str, Any]],
        step_trace: list[StepTrace],
    ) -> tuple[int, int, int]:
        """Auto-verify brittle facts when configured triggers fire."""
        if self.ablation_no_verify or self.max_verify_calls <= 0:
            return 0, 0, 0
        if not capsule.fact.text.strip():
            return 0, 0, 0

        final_slot_name = self._final_slot_name(slot_state)
        is_final_slot = bool(slot_name) and slot_name == final_slot_name
        should_verify = False
        if self._fact_conflicts_with_memory(capsule.fact, memory.get_all()):
            should_verify = True
        if is_final_slot and capsule.fact.confidence < self.auto_verify_threshold:
            should_verify = True

        if not should_verify:
            return 0, 0, 0

        verify_goal = (
            f"Verify the fact for slot '{slot_name}' and replace it if a better grounded fact is found. "
            f"Original claim: {capsule.fact.text}"
        )
        verify_capsule, verify_probe_tokens = await self.investigator.investigate_with_usage(
            sub_question=sub_question or question,
            goal=verify_goal,
            prior_facts=memory.get_all(),
            slot_name=slot_name,
            slot_hint=self._slot_hint(slot_state, slot_name),
        )
        evidence = "\n".join(verify_capsule.support_snippets) or verify_capsule.fact.text
        verify_result, verify_tokens = await self.orchestrator.verify_claim_with_usage(
            capsule.fact.text,
            evidence or "No evidence available.",
        )
        total_tokens = verify_probe_tokens + verify_tokens

        if verify_result.get("decision") == "reject" and verify_capsule.fact.text:
            self._add_fact(memory, verify_capsule, step=step, slot_name=slot_name)
            self._update_slot_resolution(slot_state, slot_name, verify_capsule)

        step_trace.append(
            StepTrace(
                step=step,
                action="verify",
                claim=capsule.fact.text,
                tokens=total_tokens,
                slot_name=slot_name,
                metadata={
                    "decision": verify_result.get("decision", "accept"),
                    "reason": verify_result.get("reason", ""),
                    "replacement_fact": verify_capsule.fact.text,
                },
            )
        )
        return total_tokens, 1, 1

    @staticmethod
    def _fact_conflicts_with_memory(fact, facts: list) -> bool:
        """Detect simple slot-level contradictions in memory."""
        if not fact.slot_name or not fact.text.strip():
            return False
        for existing in facts:
            if existing is fact:
                continue
            if existing.slot_name and existing.slot_name == fact.slot_name:
                if existing.text.strip() and existing.text.strip().lower() != fact.text.strip().lower():
                    return True
        return False

    @staticmethod
    def _final_slot_name(slot_state: list[dict[str, Any]]) -> str:
        """Return the terminal answer slot name from the routed slot plan."""
        if not slot_state:
            return "final_answer"
        return str(slot_state[-1].get("slot_name", "final_answer")).strip() or "final_answer"

    async def _auto_verify(
        self,
        step: int,
        capsule: EvidenceCapsule,
        step_trace: list[StepTrace],
    ) -> int:
        """A6: automatically verify each spawned fact."""
        evidence = "\n".join(capsule.support_snippets) if capsule.support_snippets else capsule.fact.text
        verify_result, verify_tokens = await self.orchestrator.verify_claim_with_usage(
            capsule.fact.text,
            evidence or "No evidence available.",
        )
        self._log_verify(step, capsule.fact.text, verify_result)
        step_trace.append(
            StepTrace(
                step=step,
                action="verify",
                sub_question=None,
                claim=capsule.fact.text,
                fact_added=False,
                tokens=verify_tokens,
            )
        )
        return verify_tokens

    @staticmethod
    def _log_verify(step: int, claim: str, verify_result: dict) -> None:
        """Emit a uniform verification log."""
        verify_decision = verify_result.get("decision", "accept")
        verify_reason = verify_result.get("reason", "")
        if verify_decision == "reject":
            logger.info(
                "Claim rejected at step %d: %r — reason: %s",
                step,
                claim,
                verify_reason,
            )
        else:
            logger.debug("Claim accepted at step %d: %r", step, claim)

    async def _verify_answer_attempt(
        self,
        *,
        question: str,
        step: int,
        answer_obj: dict[str, Any],
        facts: list,
        target_profile: str,
        expected_answer_type: str,
        step_trace: list[StepTrace],
        reason_tag: str,
    ) -> tuple[dict[str, Any], int]:
        """Run answer verification on a candidate final answer."""
        if not self.enable_answer_verification:
            return {
                "decision": "accept",
                "reason": "Answer verification disabled.",
                "failure_mode": "",
            }, 0
        answer_text = str(answer_obj.get("answer", "")).strip()
        if not answer_text:
            return {
                "decision": "reject",
                "reason": "Empty answer.",
                "failure_mode": "evidence_unsupported",
            }, 0
        verify_result, verify_tokens = await self.orchestrator.verify_answer_with_usage(
            question=question,
            answer=answer_text,
            facts=facts,
            target_profile=target_profile,
            expected_answer_type=expected_answer_type,
        )
        step_trace.append(
            StepTrace(
                step=step,
                action="verify",
                claim=answer_text,
                tokens=verify_tokens,
                cited_fact_ids=list(answer_obj.get("cited_fact_ids", [])),
                justification_confidence=answer_obj.get("justification_confidence"),
                metadata={
                    "verify_kind": "answer",
                    "reason_tag": reason_tag,
                    "decision": verify_result.get("decision", "accept"),
                    "reason": verify_result.get("reason", ""),
                    "failure_mode": verify_result.get("failure_mode", ""),
                },
            )
        )
        return verify_result, verify_tokens

    @staticmethod
    def _is_duplicate_subquestion(sub_question: str, step_trace: list[StepTrace]) -> bool:
        """Return whether a proposed spawn repeats a previous spawn question."""
        normalized = sub_question.strip().lower()
        if not normalized:
            return False
        for entry in step_trace:
            if entry.action == "spawn" and entry.sub_question:
                if entry.sub_question.strip().lower() == normalized:
                    return True
        return False

    def _capsule_supports_answer(
        self,
        capsule: EvidenceCapsule,
        threshold: float,
    ) -> bool:
        """Return whether a capsule is strong enough to answer directly."""
        return bool(
            capsule.answer.strip()
            and capsule.fact.text.strip()
            and capsule.fact.support_ids
            and capsule.fact.confidence >= max(threshold, self.min_fact_confidence)
        )

    def _allow_bootstrap_short_circuit(self, question: str) -> bool:
        """Restrict bootstrap early exits to genuinely easy/direct questions."""
        if self.ablation_force_spawn:
            return False
        if not self.enable_bootstrap_short_circuit:
            return False
        return self._looks_single_hop(question)

    def _should_force_bridge_first_route(
        self,
        question: str,
        route: dict[str, Any],
    ) -> bool:
        """Override contradictory single-probe routes with recurse."""
        if str(route.get("action", "")).strip().lower() != "single_probe":
            return False
        if bool(route.get("forced_bridge_first", False)):
            return False

        required_hops = route.get("required_hops") or []
        if len(required_hops) >= 2:
            return True

        if self._looks_single_hop(question):
            return False

        sub_question = str(route.get("sub_question", "")).strip()
        retrieval_query = str(route.get("retrieval_query", "")).strip()
        looks_like_echo = (
            not sub_question
            or self.orchestrator._looks_like_question_echo(sub_question, question)
            or not retrieval_query
            or self.orchestrator._looks_like_question_echo(retrieval_query, question)
        )
        return looks_like_echo

    def _force_bridge_first_route(
        self,
        question: str,
        route: dict[str, Any],
        target_profile: str,
    ) -> dict[str, Any]:
        """Rewrite a bad single-probe route into a recurse bootstrap."""
        updated = dict(route)
        required_hops = list(updated.get("required_hops") or [])
        first_hop = required_hops[0] if required_hops else {}
        first_slot = str(first_hop.get("slot_name", "")).strip()
        first_hint = str(first_hop.get("hint", "")).strip()
        goal_parts = ["Retrieve the earliest missing bridge fact before answering."]
        if first_slot:
            goal_parts.append(f"Resolve slot: {first_slot}.")
        if first_hint:
            goal_parts.append(f"Hint: {first_hint}.")
        elif target_profile.strip():
            goal_parts.append(target_profile.strip())

        updated["action"] = "recurse"
        updated["confidence"] = min(
            float(updated.get("confidence", 0.0)),
            self.route_probe_threshold,
        )
        updated["sub_question"] = question
        updated["retrieval_query"] = question
        updated["goal"] = " ".join(goal_parts).strip()
        updated["forced_bridge_first"] = True
        updated["draft_answer"] = ""
        return updated

    def _should_force_planned_recurse_execution(
        self,
        route: dict[str, Any],
        slot_state: list[dict[str, Any]],
    ) -> bool:
        """Use the router's hop plan as the executor policy for recurse questions."""
        return bool(
            self.force_planned_recurse_execution
            and str(route.get("action", "")).strip().lower() == "recurse"
            and len(slot_state) >= 2
        )

    async def _propose_planned_spawn(
        self,
        *,
        question: str,
        target_profile: str,
        memory: FactMemory,
        slot_state: list[dict[str, Any]],
        step_trace: list[StepTrace],
    ) -> tuple[dict[str, Any], int]:
        """Propose the next recurse step using the explicit hop plan only."""
        slot_name = self._first_pending_slot(slot_state)
        focused_slots = [
            slot for slot in self._pending_slots(slot_state)
            if str(slot.get("slot_name", "")).strip() == slot_name
        ]
        slot_hint = self._slot_hint(slot_state, slot_name)
        if self.use_deterministic_plan_queries:
            sub_question, retrieval_query = self._deterministic_plan_query(
                question=question,
                slot_state=slot_state,
                memory=memory,
                slot_name=slot_name,
                slot_hint=slot_hint,
            )
            return {
                "action": "spawn",
                "sub_question": sub_question,
                "retrieval_query": retrieval_query,
                "goal": f"Resolve slot '{slot_name}' exactly.",
                "slot_name": slot_name,
                "echo_detected_llm_call": False,
                "echo_still_present_after_refine": False,
                "echo_repaired_count": 0,
            }, 0
        return await self.orchestrator.propose_slot_targeted_spawn(
            question=question,
            facts=memory.get_all(),
            trace=step_trace,
            current_slot_name=slot_name,
            current_slot_hint=slot_hint,
            target_profile=target_profile,
            pending_slots=focused_slots,
            missing_reason=(
                f"Follow the recurse plan strictly. Resolve slot '{slot_name}' next."
                + (f" Hint: {slot_hint}." if slot_hint else "")
                + " Do not answer yet. Do not switch to a later slot. "
                  "Do not ask a generic restatement of the original multi-hop question. "
                  "Do not return a nearby bridge entity when the question asks for a downstream slot value."
            ),
        )

    def _deterministic_plan_query(
        self,
        *,
        question: str,
        slot_state: list[dict[str, Any]],
        memory: FactMemory,
        slot_name: str,
        slot_hint: str,
    ) -> tuple[str, str]:
        """Build the next recurse sub-question deterministically from resolved slots."""
        replacements = self._resolved_slot_replacements(slot_state, memory.get_all())
        filled = self._fill_slot_hint(slot_hint or slot_name, replacements).strip()
        if not filled:
            filled = self._fill_slot_hint(slot_name.replace("_", " "), replacements).strip()
        if not filled:
            return question.strip(), question.strip()

        lowered = filled.lower().rstrip(" ?")
        if re.match(r"^(who|what|when|where|which|how)\b", lowered):
            sub_question = filled.rstrip(" ?") + "?"
        elif any(token in lowered for token in (" date", " year", " when ", "birth", "death")):
            sub_question = f"When {filled.rstrip(' .?')}?"
        elif any(token in lowered for token in (" location", " where ", " region", " country", " city", " state", " province", " county")):
            sub_question = f"Where is {filled.rstrip(' .?')}?"
        elif any(token in lowered for token in (" person", " spouse", " child", " sibling", " actor", " director", " author", " judge", "president", "chief")):
            sub_question = f"Who is {filled.rstrip(' .?')}?"
        else:
            sub_question = f"What is {filled.rstrip(' .?')}?"
        retrieval_query = re.sub(r"[?]+$", "", sub_question).strip()
        return sub_question, retrieval_query

    @staticmethod
    def _resolved_slot_replacements(
        slot_state: list[dict[str, Any]],
        facts: list,
    ) -> dict[str, str]:
        """Map resolved slot names to their best answer spans."""
        replacements: dict[str, str] = {}
        for slot in slot_state:
            slot_name = str(slot.get("slot_name", "")).strip()
            if not slot_name or not slot.get("resolved", False):
                continue
            answer = ""
            for fact in reversed(facts):
                if str(getattr(fact, "slot_name", "")).strip() != slot_name:
                    continue
                answer = str(getattr(fact, "answer_span", "")).strip() or str(getattr(fact, "text", "")).strip()
                if answer:
                    break
            if answer:
                replacements[slot_name] = answer
        return replacements

    @staticmethod
    def _fill_slot_hint(template: str, replacements: dict[str, str]) -> str:
        """Replace slot references with resolved answer spans."""
        text = str(template or "")
        ordered = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
        for slot_name, answer in ordered:
            slot_tokens = slot_name.replace("_", " ").strip()
            patterns = [
                rf"\bthe\s+{re.escape(slot_tokens)}\b",
                rf"\b{re.escape(slot_name)}\b",
                rf"\b{re.escape(slot_tokens)}\b",
            ]
            for pattern in patterns:
                text = re.sub(pattern, answer, text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _slot_expected_answer_type(
        self,
        slot_state: list[dict[str, Any]],
        slot_name: str,
        final_expected_answer_type: str,
    ) -> str:
        """Infer the answer type for an individual slot."""
        if slot_name == self._final_slot_name(slot_state):
            return final_expected_answer_type
        hint = self._slot_hint(slot_state, slot_name).strip()
        if not hint:
            return "entity"
        lowered = hint.lower()
        if any(token in lowered for token in (" date", " year", " when ", "birth", "death")):
            return "date/year"
        if any(token in lowered for token in (" number of", " how many", " how much", " population", " attendance", " employees", " people ")) or lowered.startswith(("number of", "population of")):
            return "number"
        if any(token in lowered for token in (" city", " country", " state", " province", " county", " region", " river", " location", " place where", " where ")):
            return "location"
        if any(token in lowered for token in (" child of", " spouse of", " sibling of", " actor", " actress", " director", " author", " founder", " president", " judge", " player", " mascot", " winner of")):
            return "person"
        if lowered.startswith(("is ", "are ", "was ", "were ", "did ", "does ", "do ")):
            return "yes/no"
        return "entity"

    def _hybrid_fill_slot_hint(
        self,
        slot_hint: str,
        slot_state: list[dict[str, Any]],
        memory: FactMemory,
    ) -> str:
        """Resolve slot-hint references with the latest grounded bridge answer."""
        replacements = self._resolved_slot_replacements(slot_state, memory.get_all())
        text = self._fill_slot_hint(slot_hint, replacements).strip()
        if not text:
            text = slot_hint.strip()
        resolved_values = [
            replacements.get(str(slot.get("slot_name", "")).strip(), "")
            for slot in slot_state
            if slot.get("resolved", False)
        ]
        resolved_values = [value for value in resolved_values if value]
        bridge_value = resolved_values[-1] if resolved_values else ""
        if bridge_value:
            text = re.sub(
                r"\b(?:the\s+)?identified\s+[a-z][a-z\s_-]*\b",
                bridge_value,
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\b(?:that|this)\s+(?:state|city|country|province|county|region|person|university|film|song|team|company)\b",
                bridge_value,
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bthe\s+(state|city|country|province|county|region|university|film|song|team|company|artist|performer|player|governor|castle|home)\b",
                bridge_value,
                text,
                count=1,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_hybrid_hint(hint: str) -> str:
        """Strip prompty prefixes from router slot hints before query generation."""
        text = re.sub(r"\s+", " ", str(hint or "")).strip()
        if not text:
            return ""
        text = re.sub(
            r"^(?:identify|find|determine|resolve|retrieve|answer)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"^expected answer type enum:\s*[^.?!]+[.?!]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"^expected answer type:\s*[^.?!]+[.?!]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"^preserve the exact answer slot asked in the original question:\s*", "", text, flags=re.IGNORECASE)
        return text.strip(" .?")

    @staticmethod
    def _fill_plan_placeholders(template: str, collected_answers: dict[int, str]) -> str:
        """Fill OPERA-style [entity from step X] placeholders from prior answers."""
        filled = str(template or "")
        pattern = re.compile(r"\[([^]]+)\s+from\s+step\s+(\d+)\]", re.IGNORECASE)
        for info_type, step_num in pattern.findall(filled):
            step_id = int(step_num)
            answer = collected_answers.get(step_id, "")
            if not answer:
                continue
            value = AdaptiveRecursivePipeline._extract_placeholder_value(answer, info_type.lower())
            filled = filled.replace(f"[{info_type} from step {step_num}]", value)
        return re.sub(r"\s+", " ", filled).strip()

    @staticmethod
    def _extract_placeholder_value(answer: str, info_type: str) -> str:
        """Extract a compact placeholder value from a prior answer."""
        text = str(answer or "").strip()
        if not text:
            return ""
        if info_type in {"date", "year", "time"}:
            match = re.search(r"\b((?:19|20)\d{2})\b", text)
            if match:
                return match.group(1)
        if info_type in {"number", "amount", "count"}:
            match = re.search(r"\b(\d[\d,\.]*)\b", text)
            if match:
                return match.group(1)
        if info_type in {"location", "place", "country", "city"}:
            match = re.search(r"\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
            if match:
                return match.group(1)
        entity_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        if entity_matches:
            return max(entity_matches, key=len)
        if len(text.split()) <= 5:
            return text
        return text.split(",")[0].split(".")[0].strip()

    @staticmethod
    def _generate_query_variants(query: str) -> list[str]:
        """Generate one deterministic retry query, OPERA-style."""
        base = str(query or "").strip()
        if not base:
            return [""]
        variants = [base]
        stripped = re.sub(r"[?.,;:]+", "", base)
        lowered = stripped.lower()
        for prefix in ("what ", "who ", "where ", "when ", "which ", "how "):
            if lowered.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
                break
        stripped = re.sub(r"\b(is|was|are|were|did|does|do|the|a|an)\b", " ", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped and stripped != base:
            variants.append(stripped)
        return variants[:2]

    @classmethod
    def _extract_plan_step_answer(
        cls,
        capsule: EvidenceCapsule,
        expected_answer_type: str,
    ) -> str:
        """Extract a compact step answer for placeholder chaining and final last-hop return."""
        answer_span = str(getattr(capsule.fact, "answer_span", "")).strip()
        if answer_span and cls._answer_matches_expected_type(answer_span, expected_answer_type):
            return answer_span
        fact_text = str(getattr(capsule.fact, "text", "")).strip()
        if fact_text and cls._answer_matches_expected_type(fact_text, expected_answer_type):
            return fact_text
        return ""

    def _hybrid_slot_query(
        self,
        *,
        question: str,
        route: dict[str, Any],
        slot_state: list[dict[str, Any]],
        memory: FactMemory,
        slot_name: str,
        expected_answer_type: str,
    ) -> tuple[str, str]:
        """Build one deterministic slot-targeted query for chain execution."""
        hint = self._clean_hybrid_hint(
            self._hybrid_fill_slot_hint(
                self._slot_hint(slot_state, slot_name) or slot_name.replace("_", " "),
                slot_state,
                memory,
            )
        )
        if hint.lower().startswith("expected answer type enum:"):
            route_query = str(route.get("retrieval_query", "")).strip()
            return question.strip(), route_query or question.strip()
        if not hint:
            return question.strip(), question.strip()

        lowered = hint.lower().strip().rstrip(" ?.")
        if re.match(r"^(who|what|when|where|which|how)\b", lowered):
            sub_question = hint.rstrip(" ?.") + "?"
        elif expected_answer_type == "date/year":
            if lowered.startswith(("date of ", "year of ")):
                subject = re.sub(r"^(date|year)\s+of\s+", "", lowered, flags=re.IGNORECASE)
                sub_question = f"When was {subject.rstrip(' ?.') }?"
            else:
                sub_question = f"When did {hint.rstrip(' ?.')}?"
        elif expected_answer_type == "number":
            if lowered.startswith(("number of ", "population of ", "attendance of ")):
                subject = re.sub(r"^(number|population|attendance)\s+of\s+", "", lowered, flags=re.IGNORECASE)
                sub_question = f"How many {subject.rstrip(' ?.')}?"
            elif lowered.startswith(("how many ", "how much ")):
                sub_question = hint.rstrip(" ?.") + "?"
            else:
                sub_question = f"How many {hint.rstrip(' ?.')}?"
        elif expected_answer_type == "location":
            if lowered.startswith(("city where ", "county where ", "country where ", "state where ", "province where ", "region where ", "place where ")):
                subject = re.sub(r"^(city|county|country|state|province|region|place)\s+where\s+", "", hint.rstrip(" ?."), flags=re.IGNORECASE)
                sub_question = f"Where did {subject}?"
            elif lowered.startswith("location of "):
                subject = re.sub(r"^location\s+of\s+", "", hint.rstrip(" ?."), flags=re.IGNORECASE)
                sub_question = f"Where is {subject}?"
            else:
                sub_question = f"Where is {hint.rstrip(' ?.')}?"
        elif expected_answer_type == "person":
            if lowered.startswith("winner of "):
                sub_question = f"Who won {hint[len('winner of '):].rstrip(' ?.')}?"
            elif lowered.startswith(("child of ", "spouse of ", "sibling of ", "father of ", "mother of ")):
                sub_question = f"Who is the {hint.rstrip(' ?.')}?"
            else:
                sub_question = f"Who is {hint.rstrip(' ?.')}?"
        elif expected_answer_type == "yes/no":
            sub_question = hint.rstrip(" ?.")
            if not re.match(r"^(is|are|was|were|did|does|do|has|have|had|can|could|will|would|should)\b", sub_question.lower()):
                sub_question = f"Is it true that {sub_question}?"
        else:
            if lowered.startswith(("name of ", "capital of ", "label of ", "league of ", "company that ", "record label of ", "mascot of ", "part of ")):
                sub_question = f"What is the {hint.rstrip(' ?.')}?"
            else:
                sub_question = f"What is {hint.rstrip(' ?.')}?"
        retrieval_query = re.sub(r"[?]+$", "", sub_question).strip()
        return sub_question, retrieval_query

    async def _targeted_answer_recovery(
        self,
        *,
        question: str,
        step: int,
        route_draft_answer: str,
        target_profile: str,
        expected_answer_type: str,
        memory: FactMemory,
        slot_state: list[dict[str, Any]],
        step_trace: list[StepTrace],
        slot_name: str,
        missing_reason: str,
        retrieved_doc_ids: list[str],
        retrieved_docs_total: int,
    ) -> dict[str, Any]:
        """Probe one exact slot, then regenerate the final answer object."""
        focused_slots = [
            slot for slot in self._pending_slots(slot_state)
            if str(slot.get("slot_name", "")).strip() == slot_name
        ] or [{
            "slot_name": slot_name,
            "hint": self._slot_hint(slot_state, slot_name),
            "resolved": False,
            "dependency_group": 0,
        }]
        decision, prompt_tokens = await self.orchestrator.propose_slot_targeted_spawn(
            question=question,
            facts=memory.get_all(),
            trace=step_trace,
            current_slot_name=slot_name,
            current_slot_hint=self._slot_hint(slot_state, slot_name),
            target_profile=target_profile,
            pending_slots=focused_slots,
            missing_reason=missing_reason,
        )
        sub_question = decision["sub_question"]
        retrieval_query = str(decision.get("retrieval_query", "")).strip()
        goal = decision["goal"]
        duplicate_subquestion_count = int(
            self._is_duplicate_subquestion(sub_question, step_trace)
        )
        capsule, investigate_tokens = await self.investigator.investigate_with_usage(
            sub_question=sub_question,
            goal=goal,
            prior_facts=memory.get_all(),
            retrieval_query=retrieval_query or None,
            slot_name=slot_name,
            slot_hint=self._slot_hint(slot_state, slot_name),
        )
        merged_ids, merged_total = self._merge_retrieval_stats(
            retrieved_doc_ids,
            retrieved_docs_total,
            capsule,
        )
        fact_added = self._add_fact(
            memory,
            capsule,
            step=step + 1,
            slot_name=slot_name,
        )
        self._update_slot_resolution(slot_state, slot_name, capsule)
        step_trace.append(
            StepTrace(
                step=step + 1,
                action="spawn",
                sub_question=sub_question,
                fact_added=fact_added,
                tokens=prompt_tokens + investigate_tokens,
                slot_name=slot_name,
                metadata={
                    "goal": goal,
                    "recovery_mode": "answer_verification_retry",
                    "retrieval_query": retrieval_query or sub_question,
                },
            )
        )
        verify_tokens, verify_delta, auto_verify_delta = await self._maybe_verify_fact(
            question=question,
            step=step + 1,
            slot_name=slot_name,
            sub_question=sub_question,
            capsule=capsule,
            memory=memory,
            slot_state=slot_state,
            step_trace=step_trace,
        )
        final_pending_slots = self._pending_slots(slot_state)
        if self.force_planned_recurse_execution and not final_pending_slots:
            answer_obj = self._direct_terminal_slot_answer_object(
                facts=memory.get_all(),
                slot_state=slot_state,
                expected_answer_type=expected_answer_type,
            )
            answer_tokens = 0
        else:
            answer_obj, answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
                question=question,
                facts=memory.get_all(),
                target_profile=target_profile,
                pending_slots=final_pending_slots,
                trace=step_trace,
                route_draft_answer=route_draft_answer,
            )
            answer_obj = self._apply_answer_object_fallback(
                answer_obj,
                memory.get_all(),
                route_draft_answer,
                expected_answer_type=expected_answer_type,
            )
            answer_obj = self._prefer_terminal_slot_answer_object(
                answer_obj,
                memory.get_all(),
                slot_state,
                expected_answer_type=expected_answer_type,
            )
        return {
            "answer_obj": answer_obj,
            "orchestrator_tokens": prompt_tokens + answer_tokens,
            "subagent_tokens": investigate_tokens + verify_tokens,
            "subagent_calls": 1,
            "verify_calls": verify_delta,
            "auto_verify_calls": auto_verify_delta,
            "duplicate_subquestion_count": duplicate_subquestion_count,
            "retrieved_doc_ids": merged_ids,
            "retrieved_docs_total": merged_total,
            "answer_tokens": answer_tokens,
        }

    @staticmethod
    def _looks_single_hop(question: str) -> bool:
        """Heuristic gate for questions that likely do not need decomposition."""
        text = question.strip().lower()
        if not text:
            return False

        multi_hop_markers = (
            " that ",
            " who ",
            " whose ",
            " where ",
            " when ",
            " which ",
            " from the ",
            " of the ",
            " by the ",
            " in the ",
            " after ",
            " before ",
            " during ",
            " founded by ",
            " educated at ",
            " headquarters ",
            " operator of ",
            " mother of ",
            " father of ",
            " stand for ",
        )
        if any(marker in text for marker in multi_hop_markers):
            return False

        bridge_prefixes = (
            "who is the",
            "what is the",
            "where is",
            "when was",
            "what year was",
            "what date was",
        )
        return any(text.startswith(prefix) for prefix in bridge_prefixes)

    @staticmethod
    def _expected_answer_type(question: str) -> str:
        """Infer a normalized expected answer type from the question text."""
        question_lower = question.strip().lower()
        if question_lower.startswith(
            ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
             "has ", "have ", "had ", "will ", "would ", "should ")
        ):
            return "yes/no"
        if question_lower.startswith(("who ", "whom ", "whose ")):
            return "person"
        if question_lower.startswith(
            (
                "where ",
                "what city",
                "which city",
                "what country",
                "which country",
                "what state",
                "which state",
                "what province",
                "which province",
                "what county",
                "which county",
                "what region",
                "which region",
                "what river",
                "which river",
                "what body of water",
            )
        ):
            return "location"
        if question_lower.startswith(("when ", "what year", "which year", "what date", "which date")):
            return "date/year"
        if question_lower.startswith(("how many", "how much")):
            return "number"
        if question_lower.startswith(("which are the", "what are the", "name the", "list the")):
            return "list"
        if question_lower.startswith("what does"):
            return "other"
        return "entity"

    @classmethod
    def _target_profile(cls, question: str) -> str:
        """Infer a lightweight answer-target hint from the question wording."""
        question_lower = question.strip().lower()
        expected_answer_type = cls._expected_answer_type(question)
        patterns = [
            ("who ", "Expected answer type: person or organization."),
            ("where ", "Expected answer type: location or institution."),
            ("when ", "Expected answer type: date or time."),
            ("how many", "Expected answer type: number or count."),
            ("how much", "Expected answer type: quantity or amount."),
            ("what year", "Expected answer type: year."),
            ("what date", "Expected answer type: date."),
            ("which year", "Expected answer type: year."),
            ("which date", "Expected answer type: date."),
            ("what city", "Expected answer type: city."),
            ("what country", "Expected answer type: country."),
            ("what state", "Expected answer type: state or province."),
            ("what province", "Expected answer type: province."),
            ("what county", "Expected answer type: county."),
            ("what river", "Expected answer type: river or body of water."),
            ("what body of water", "Expected answer type: body of water."),
            ("what does", "Expected answer type: expansion or definition."),
        ]
        for prefix, hint in patterns:
            if question_lower.startswith(prefix):
                return (
                    f"Expected answer type enum: {expected_answer_type}. {hint} "
                    f"Preserve the exact answer slot asked in the original question: {question.strip()}"
                )
        return (
            f"Expected answer type enum: {expected_answer_type}. "
            "Expected answer type: short factual span grounded in the original question. "
            f"Preserve the exact answer slot asked in the original question: {question.strip()}"
        )

    @staticmethod
    def _merge_retrieval_stats(
        current_ids: list[str],
        current_total: int,
        capsule: EvidenceCapsule,
    ) -> tuple[list[str], int]:
        """Merge retrieved-doc stats from one capsule into the run aggregate."""
        seen = set(current_ids)
        merged = list(current_ids)
        for doc_id in capsule.retrieved_doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc_id)
        return merged, current_total + int(capsule.retrieved_docs_total)

    def _should_force_budget_answer(self, total_tokens: int) -> bool:
        """Return whether the per-question token budget is exhausted."""
        return self.max_total_tokens > 0 and total_tokens >= self.max_total_tokens

    def _trace_cap_reached(self, step_trace: list[StepTrace]) -> bool:
        """Return whether there is no room left for another non-answer trace action."""
        if self.max_trace_actions <= 0:
            return False
        return len(step_trace) >= max(self.max_trace_actions - 1, 0)

    async def _force_budget_answer_object(
        self,
        question: str,
        facts: list[Any],
        target_profile: str,
        pending_slots: list[str],
        step_trace: list[StepTrace],
        step: int,
        total_tokens: int,
        orchestrator_tokens: int,
        *,
        route_draft_answer: str = "",
        trace_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Generate an immediate answer once the routed controller hits budget."""
        answer_obj, answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
            question=question,
            facts=facts,
            target_profile=target_profile,
            pending_slots=pending_slots,
            trace=step_trace,
            route_draft_answer=route_draft_answer,
        )
        answer_obj = self._apply_answer_object_fallback(
            answer_obj,
            facts,
            route_draft_answer,
        )
        total_tokens += answer_tokens
        orchestrator_tokens += answer_tokens
        trace_metadata = {
            "justification": answer_obj["justification"],
            "missing_slot": answer_obj["missing_slot"],
            "fallback_source": answer_obj.get("fallback_source", ""),
            "budget_exhausted": True,
        }
        if metadata:
            trace_metadata.update(metadata)
        step_trace.append(
            StepTrace(
                step=step,
                action="answer",
                tokens=trace_tokens + answer_tokens,
                cited_fact_ids=answer_obj["cited_fact_ids"],
                justification_confidence=answer_obj["justification_confidence"],
                metadata=trace_metadata,
            )
        )
        return answer_obj, total_tokens, orchestrator_tokens

    @staticmethod
    def _merge_retrieval_ids(
        current_ids: list[str],
        current_total: int,
        new_ids: list[str],
        new_total: int,
    ) -> tuple[list[str], int]:
        """Merge aggregated retrieval stats without an intermediate capsule object."""
        seen = set(current_ids)
        merged = list(current_ids)
        for doc_id in new_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc_id)
        return merged, current_total + int(new_total)
