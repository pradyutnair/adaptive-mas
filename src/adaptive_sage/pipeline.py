"""Adaptive Recursive SAGE pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from pathlib import Path
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
        self.bootstrap_probe_first: bool = bool(
            config.get("adaptive.bootstrap_probe_first", False)
        )
        self.bootstrap_search_top_k: int = int(
            config.get("adaptive.bootstrap_search_top_k", 4)
        )
        self.bootstrap_max_read: int = int(
            config.get("adaptive.bootstrap_max_read", 6)
        )
        self.enable_parallel_independent_hops: bool = bool(
            config.get("adaptive.enable_parallel_independent_hops", True)
        )
        self.max_parallel_hops: int = int(
            config.get("adaptive.max_parallel_hops", 2)
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

        # ------------------------------------------------------------------
        # Sufficiency controller (m1.2): single dataset-agnostic policy.
        # Effort scales with a calibrated post-probe sufficiency score s.
        # ------------------------------------------------------------------
        self.use_sufficiency_controller: bool = bool(
            config.get("adaptive.sufficiency_controller", False)
        )
        self.sufficiency_threshold: float = float(
            config.get("adaptive.sufficiency_threshold", 0.70)
        )
        self.sufficiency_max_recurse_steps: int = int(
            config.get("adaptive.sufficiency_max_recurse_steps", self.max_steps)
        )
        self.sufficiency_min_recurse_steps: int = int(
            config.get("adaptive.sufficiency_min_recurse_steps", 1)
        )

        # Sufficiency-controller ablations.
        self.ablation_sufficiency_no_probe: bool = bool(
            config.get("ablation.sufficiency_no_probe", False)
        )
        self.ablation_sufficiency_no_controller: bool = bool(
            config.get("ablation.sufficiency_no_controller", False)
        )
        self.ablation_sufficiency_random_route: bool = bool(
            config.get("ablation.sufficiency_random_route", False)
        )
        self.ablation_sufficiency_random_route_p: float = float(
            config.get("ablation.sufficiency_random_route_p", 0.5)
        )
        self.ablation_sufficiency_random_route_seed: int = int(
            config.get("ablation.sufficiency_random_route_seed", 42)
        )
        self.ablation_sufficiency_oracle_route: bool = bool(
            config.get("ablation.sufficiency_oracle_route", False)
        )
        self.ablation_sufficiency_oracle_path: str = str(
            config.get("ablation.sufficiency_oracle_path", "")
        )
        self._oracle_route_table: dict[str, bool] = {}
        if self.ablation_sufficiency_oracle_route and self.ablation_sufficiency_oracle_path:
            self._oracle_route_table = self._load_oracle_table(
                self.ablation_sufficiency_oracle_path
            )

    async def run(self, question: str, question_id: str) -> PipelineResult:
        """Execute the adaptive recursive pipeline on a question."""
        logger.info("Pipeline start: question_id=%s, max_steps=%d", question_id, self.max_steps)

        if self.max_steps == 0:
            result = await self._run_s0(question, question_id)
        elif self.use_sufficiency_controller:
            result = await self._run_sufficiency(question, question_id)
        elif self.ablation_upfront_decomposition:
            result = await self._run_upfront_decomposition(question, question_id)
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

    async def _run_s0(self, question: str, question_id: str) -> PipelineResult:
        """S0 mode: one retrieval pass, then answer."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
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

    async def _run_sufficiency(
        self, question: str, question_id: str
    ) -> PipelineResult:
        """Sufficiency-controlled adaptive run (m1.2).

        One unified policy for every question on every dataset:

        1. Run a single grounded retrieval probe.
        2. Compute sufficiency  s = s_conf * s_target * s_align  in [0, 1].
        3. If  s >= tau, answer from the probe state.
        4. Else, escalate to a recursive lane with a step budget that
           scales as ceil(MAX_STEPS * (1 - s)).

        No question-text heuristics, no per-dataset thresholds, no
        entity-class lists. The only hyperparameter is `tau`, fixed
        a priori from configuration (default 0.70).
        """
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        target_profile = self._target_profile(question)
        step_trace: list[StepTrace] = []
        total_tokens = 0
        orchestrator_tokens = 0
        subagent_tokens = 0
        subagent_calls = 0
        retrieved_doc_ids: list[str] = []
        retrieved_docs_total = 0
        answer = ""

        tau = self.sufficiency_threshold

        # ---- Route first: extract slot DAG (required_hops) before probing. ----
        # The route prompt is schema-only (no benchmark examples, no
        # interrogative rules). We use its structured output as a
        # principled scaffold so the recursive lane has explicit pending
        # slots and the probe can target the first unresolved bridge slot
        # on compositional questions instead of the raw multi-hop string.
        route, route_tokens = await self.orchestrator.route_with_usage(
            question=question,
            target_profile=target_profile,
        )
        total_tokens += route_tokens
        orchestrator_tokens += route_tokens
        required_hops: list[dict] = list(route.get("required_hops") or [])
        route_target_slot: str = str(route.get("target_slot") or "")
        is_compositional = len(required_hops) > 1

        # Ablation: skip probe entirely and always recurse with full budget.
        # Useful as the "always-MAS" upper bound on cost.
        if self.ablation_sufficiency_no_probe:
            recurse_steps = self.sufficiency_max_recurse_steps
            return await self._sufficiency_recurse(
                question=question,
                question_id=question_id,
                memory=memory,
                step_trace=step_trace,
                target_profile=target_profile,
                total_tokens=total_tokens,
                orchestrator_tokens=orchestrator_tokens,
                subagent_tokens=subagent_tokens,
                subagent_calls=subagent_calls,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_docs_total=retrieved_docs_total,
                recurse_steps=recurse_steps,
                sufficiency=0.0,
                route_label="ablation_no_probe",
                required_hops=required_hops,
                route_target_slot=route_target_slot,
            )

        # Always probe with the full question. Probing only the first-hop
        # hint on compositional questions throws away signal that often
        # short-circuits the controller (especially on hotpot-style chains
        # where one retrieval already contains the answer span). The route
        # output is used downstream as the slot scaffold for the recursive
        # lane; it does not constrain the probe's retrieval target.
        probe_sub_question = question
        probe_goal = target_profile

        probe_capsule, probe_tokens = await self.investigator.investigate_with_usage(
            sub_question=probe_sub_question,
            goal=probe_goal,
            prior_facts=[],
            search_top_k_override=self.bootstrap_search_top_k or None,
            max_read_override=self.bootstrap_max_read or None,
        )
        total_tokens += probe_tokens
        subagent_tokens += probe_tokens
        subagent_calls += 1
        retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
            retrieved_doc_ids,
            retrieved_docs_total,
            probe_capsule,
        )
        probe_fact_added = self._add_fact(memory, probe_capsule, step=0)
        step_trace.append(
            StepTrace(
                step=0,
                action="spawn",
                sub_question=probe_sub_question,
                fact_added=probe_fact_added,
                tokens=probe_tokens,
                metadata={
                    "sufficiency_probe": True,
                    "route_target_slot": route_target_slot,
                    "route_required_hops": required_hops,
                    "is_compositional": is_compositional,
                    "probe_targeted_first_hop": False,
                },
            )
        )

        # Generate a candidate answer object grounded in the probe fact.
        probe_answer_obj, probe_answer_tokens = (
            await self.orchestrator.generate_answer_object_with_usage(
                question=question,
                facts=memory.get_all(),
                target_profile=target_profile,
                pending_slots=[],
                trace=step_trace,
            )
        )
        probe_answer_obj = self._apply_answer_object_fallback(
            probe_answer_obj,
            memory.get_all(),
            "",
        )
        total_tokens += probe_answer_tokens
        orchestrator_tokens += probe_answer_tokens
        probe_answer = probe_answer_obj["answer"]
        probe_cited_fact_ids = probe_answer_obj["cited_fact_ids"]

        # ---- Compute sufficiency s = s_conf * s_target * s_align ----
        s_conf = (
            float(probe_capsule.fact.confidence)
            if probe_capsule.fact and probe_capsule.fact.text.strip()
            else 0.0
        )
        s_align = self._compute_alignment_score(probe_capsule, probe_answer)

        if self.ablation_sufficiency_random_route:
            # Replace the calibrated signal with a Bernoulli draw at the
            # configured mix rate p. Demonstrates that the learned signal
            # is load-bearing.
            rng = random.Random(
                hash((self.ablation_sufficiency_random_route_seed, question_id)) & 0xFFFFFFFF
            )
            sample = 1.0 if rng.random() < self.ablation_sufficiency_random_route_p else 0.0
            s_target = sample
            sufficiency_components = {
                "s_conf": s_conf,
                "s_align": s_align,
                "s_target": s_target,
                "source": "ablation_random_route",
                "p": self.ablation_sufficiency_random_route_p,
            }
            sufficiency = sample
            assess_tokens = 0
            assess_reason = "random_route"
        elif self.ablation_sufficiency_oracle_route:
            # Oracle: route based on a pre-computed per-question signal
            # (typically S0 correctness). Upper bound on the controller.
            oracle_easy = bool(self._oracle_route_table.get(str(question_id), False))
            sufficiency = 1.0 if oracle_easy else 0.0
            sufficiency_components = {
                "s_conf": s_conf,
                "s_align": s_align,
                "s_target": sufficiency,
                "source": "ablation_oracle_route",
            }
            assess_tokens = 0
            assess_reason = "oracle_route"
        else:
            assess_result, assess_tokens = (
                await self.orchestrator.assess_probe_sufficiency_with_usage(
                    question=question,
                    facts=memory.get_all(),
                    proposed_answer=probe_answer,
                    target_profile=target_profile,
                    trace=step_trace,
                )
            )
            total_tokens += assess_tokens
            orchestrator_tokens += assess_tokens
            s_target = float(assess_result["sufficient"])
            sufficiency = max(0.0, min(s_conf * s_target * s_align, 1.0))
            sufficiency_components = {
                "s_conf": s_conf,
                "s_align": s_align,
                "s_target": s_target,
                "source": "calibrated_probe_gate",
            }
            assess_reason = assess_result["reason"]

        step_trace.append(
            StepTrace(
                step=1,
                action="assess",
                tokens=assess_tokens,
                justification_confidence=sufficiency,
                metadata={
                    "sufficiency_probe": True,
                    "proposed_answer": probe_answer,
                    "sufficiency": sufficiency,
                    "sufficiency_components": sufficiency_components,
                    "tau": tau,
                    "reason": assess_reason,
                },
            )
        )

        # Ablation: probe runs, controller is forced to always answer.
        if self.ablation_sufficiency_no_controller:
            answer = probe_answer or self._best_fact_span(memory.get_all())
            step_trace.append(
                StepTrace(
                    step=2,
                    action="answer",
                    tokens=probe_answer_tokens,
                    cited_fact_ids=probe_cited_fact_ids,
                    justification_confidence=sufficiency,
                    metadata={
                        "sufficiency_probe": True,
                        "route": "ablation_no_controller",
                    },
                )
            )
            return self._build_sufficiency_result(
                question_id=question_id,
                question=question,
                answer=answer,
                step_trace=step_trace,
                memory=memory,
                subagent_calls=subagent_calls,
                total_tokens=total_tokens,
                orchestrator_tokens=orchestrator_tokens,
                subagent_tokens=subagent_tokens,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_docs_total=retrieved_docs_total,
                route_label="ablation_no_controller",
                sufficiency=sufficiency,
                sufficiency_components=sufficiency_components,
                route_target_slot=route_target_slot,
                required_hops=required_hops,
            )

        # Sufficient: answer from the probe.
        if sufficiency >= tau and probe_answer.strip():
            answer = probe_answer
            step_trace.append(
                StepTrace(
                    step=2,
                    action="answer",
                    tokens=probe_answer_tokens,
                    cited_fact_ids=probe_cited_fact_ids,
                    justification_confidence=sufficiency,
                    metadata={
                        "sufficiency_probe": True,
                        "route": "answer_from_probe",
                    },
                )
            )
            return self._build_sufficiency_result(
                question_id=question_id,
                question=question,
                answer=answer,
                step_trace=step_trace,
                memory=memory,
                subagent_calls=subagent_calls,
                total_tokens=total_tokens,
                orchestrator_tokens=orchestrator_tokens,
                subagent_tokens=subagent_tokens,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_docs_total=retrieved_docs_total,
                route_label="answer_from_probe",
                sufficiency=sufficiency,
                sufficiency_components=sufficiency_components,
                route_target_slot=route_target_slot,
                required_hops=required_hops,
            )

        # Insufficient: escalate to recursive lane.
        # Step budget scales linearly with (1 - s), in [min, max].
        recurse_steps = self._sufficiency_recurse_budget(sufficiency)
        # Selective probe-fact retention. The probe fact carries two kinds of
        # signal:
        #   - s_align == 1.0: the proposed answer span is present in the
        #     retrieved capsule (evidence is grounded).
        #   - s_conf  >= tau: the underlying fact distillation is itself
        #     high-confidence.
        # When both hold, sufficiency only fell below tau because the LLM
        # verifier (s_target) was uncertain about *slot match*, not about
        # *evidence quality*. That fact is genuine evidence and should
        # remain available to decide() and the final synthesizer (we
        # observed -22 contain pts on hotpot recurse when this was dropped
        # unconditionally). When either fails, the probe fact is
        # ungrounded or low-confidence noise that biases recurse toward
        # the wrong slot (we observed -10 contain pts on musique recurse
        # when this was retained unconditionally). Drop only in the latter
        # case.
        keep_probe_fact = bool(s_align >= 1.0 and s_conf >= self.sufficiency_threshold)
        if not keep_probe_fact:
            memory = FactMemory.with_strategy(
                capacity=self.fact_memory_capacity,
                strategy=self.fact_memory_strategy,
            )
        return await self._sufficiency_recurse(
            question=question,
            question_id=question_id,
            memory=memory,
            step_trace=step_trace,
            target_profile=target_profile,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            subagent_calls=subagent_calls,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            recurse_steps=recurse_steps,
            sufficiency=sufficiency,
            route_label="recurse_after_probe",
            required_hops=required_hops,
            sufficiency_components=sufficiency_components,
            route_target_slot=route_target_slot,
        )

    async def _sufficiency_recurse(
        self,
        *,
        question: str,
        question_id: str,
        memory: FactMemory,
        step_trace: list[StepTrace],
        target_profile: str,
        total_tokens: int,
        orchestrator_tokens: int,
        subagent_tokens: int,
        subagent_calls: int,
        retrieved_doc_ids: list[str],
        retrieved_docs_total: int,
        recurse_steps: int,
        sufficiency: float,
        route_label: str,
        required_hops: list[dict] | None = None,
        sufficiency_components: dict | None = None,
        route_target_slot: str = "",
    ) -> PipelineResult:
        """Recursive lane for the sufficiency controller.

        Drives the existing decide/spawn/refine orchestrator loop with a
        step budget that was derived from the probe sufficiency score.
        `required_hops` (the slot DAG returned by the route prompt) is
        passed in as `pending_slots` so decide() has structural awareness
        of the bridge slots it must resolve before answering.
        """
        recurse_steps = max(0, int(recurse_steps))
        steps_executed = 0
        for sub_step in range(recurse_steps):
            absolute_step = len(step_trace)
            decision, decide_tokens = await self.orchestrator.decide_with_usage(
                question=question,
                facts=memory.get_all(),
                trace=step_trace,
                step=sub_step,
                target_profile=target_profile,
                pending_slots=required_hops or [],
            )
            total_tokens += decide_tokens
            orchestrator_tokens += decide_tokens
            action = decision.get("action", "answer")

            if action == "answer":
                step_trace.append(
                    StepTrace(
                        step=absolute_step,
                        action="answer",
                        tokens=decide_tokens,
                        metadata={"sufficiency_recurse": True, "decision": "answer"},
                    )
                )
                break

            if action not in {"spawn", "refine"}:
                # Verify is intentionally unsupported in this path;
                # fall through to answer.
                step_trace.append(
                    StepTrace(
                        step=absolute_step,
                        action="answer",
                        tokens=decide_tokens,
                        metadata={
                            "sufficiency_recurse": True,
                            "decision": action,
                            "fallback": "non_spawn_action",
                        },
                    )
                )
                break

            sub_question = str(decision.get("sub_question", "")).strip() or question
            retrieval_query = str(decision.get("retrieval_query", "")).strip() or None
            goal = str(decision.get("goal", "")).strip() or target_profile
            slot_name = str(decision.get("slot_name", "")).strip()

            capsule, investigate_tokens = await self.investigator.investigate_with_usage(
                sub_question=sub_question,
                goal=goal,
                prior_facts=memory.get_all(),
                retrieval_query=retrieval_query,
                slot_name=slot_name,
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
            fact_added = self._add_fact(
                memory, capsule, step=absolute_step, slot_name=slot_name
            )
            step_trace.append(
                StepTrace(
                    step=absolute_step,
                    action=action,
                    sub_question=sub_question,
                    fact_added=fact_added,
                    tokens=investigate_tokens,
                    slot_name=slot_name,
                    metadata={
                        "sufficiency_recurse": True,
                        "goal": goal,
                        "retrieval_query": retrieval_query or sub_question,
                    },
                )
            )
            steps_executed += 1

        # Final answer synthesis from the accumulated fact memory.
        answer_obj, answer_tokens = await self.orchestrator.generate_answer_object_with_usage(
            question=question,
            facts=memory.get_all(),
            target_profile=target_profile,
            pending_slots=required_hops or [],
            trace=step_trace,
        )
        answer_obj = self._apply_answer_object_fallback(
            answer_obj,
            memory.get_all(),
            "",
        )
        total_tokens += answer_tokens
        orchestrator_tokens += answer_tokens
        answer = answer_obj["answer"]
        step_trace.append(
            StepTrace(
                step=len(step_trace),
                action="answer",
                tokens=answer_tokens,
                cited_fact_ids=answer_obj["cited_fact_ids"],
                justification_confidence=answer_obj["justification_confidence"],
                metadata={
                    "sufficiency_recurse_final": True,
                    "route": route_label,
                    "fallback_source": answer_obj.get("fallback_source", ""),
                },
            )
        )
        return self._build_sufficiency_result(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=step_trace,
            memory=memory,
            subagent_calls=subagent_calls,
            total_tokens=total_tokens,
            orchestrator_tokens=orchestrator_tokens,
            subagent_tokens=subagent_tokens,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_docs_total=retrieved_docs_total,
            route_label=route_label,
            sufficiency=sufficiency,
            sufficiency_components=sufficiency_components,
            route_target_slot=route_target_slot,
            required_hops=required_hops,
            recurse_steps_used=steps_executed,
        )

    def _sufficiency_recurse_budget(self, sufficiency: float) -> int:
        """Map sufficiency to a recurse step budget.

        budget = max(min, ceil(MAX_STEPS * (1 - s))), capped at MAX_STEPS.
        Lower sufficiency -> more steps; high sufficiency would have
        already short-circuited at the probe answer path. The lower
        bound is `sufficiency_min_recurse_steps` (configurable, default 1)
        so the budget is fully derived from `s` rather than a hand-set
        floor.
        """
        s = max(0.0, min(float(sufficiency), 1.0))
        scaled = math.ceil(self.sufficiency_max_recurse_steps * (1.0 - s))
        floor = max(1, self.sufficiency_min_recurse_steps)
        return max(
            floor,
            min(scaled, self.sufficiency_max_recurse_steps),
        )

    @staticmethod
    def _compute_alignment_score(
        capsule: EvidenceCapsule, proposed_answer: str
    ) -> float:
        """Indicator that the capsule actually supports the proposed answer.

        Returns 1.0 iff the capsule has a non-empty grounded fact with
        support_ids AND the proposed answer span (or its containment) is
        present in the fact text or answer span. Returns 0.0 otherwise.
        """
        if capsule is None or capsule.fact is None:
            return 0.0
        fact_text = str(capsule.fact.text or "").strip().lower()
        if not fact_text:
            return 0.0
        if not capsule.fact.support_ids:
            return 0.0
        proposed = str(proposed_answer or "").strip().lower()
        if not proposed:
            return 0.0
        capsule_span = str(capsule.fact.answer_span or "").strip().lower()
        if proposed in fact_text:
            return 1.0
        if capsule_span and (proposed in capsule_span or capsule_span in proposed):
            return 1.0
        return 0.0

    @staticmethod
    def _best_fact_span(facts: list) -> str:
        """Return the strongest grounded answer span available in memory."""
        best = ("", -1.0)
        for fact in facts:
            answer_span = str(getattr(fact, "answer_span", "")).strip()
            if not answer_span:
                continue
            confidence = float(getattr(fact, "confidence", 0.0))
            if confidence > best[1]:
                best = (answer_span, confidence)
        return best[0]

    def _build_sufficiency_result(
        self,
        *,
        question_id: str,
        question: str,
        answer: str,
        step_trace: list[StepTrace],
        memory: FactMemory,
        subagent_calls: int,
        total_tokens: int,
        orchestrator_tokens: int,
        subagent_tokens: int,
        retrieved_doc_ids: list[str],
        retrieved_docs_total: int,
        route_label: str,
        sufficiency: float,
        sufficiency_components: dict | None = None,
        route_target_slot: str = "",
        required_hops: list[dict] | None = None,
        recurse_steps_used: int = 0,
    ) -> PipelineResult:
        """Assemble a PipelineResult from the sufficiency-controlled path."""
        extras = {
            "sufficiency_score": float(sufficiency),
            "sufficiency_components": sufficiency_components or {},
            "sufficiency_tau": float(self.sufficiency_threshold),
            "route_target_slot": route_target_slot,
            "route_required_hops": required_hops or [],
            "recurse_steps_used": int(recurse_steps_used),
            "controller": "sufficiency",
        }
        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=step_trace,
            num_subagent_calls=subagent_calls,
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
            route_decision=route_label,
            route_confidence=float(sufficiency),
            route_draft_answer="",
            slot_resolution={},
            auto_verify_calls=0,
            answer_rejection_count=0,
            extras=extras,
        )

    @staticmethod
    def _load_oracle_table(path: str) -> dict[str, bool]:
        """Load a per-question oracle map from a JSONL file.

        Expected line format: {"id": "<qid>", "easy": <bool>}.
        Easy means: route to the answer-from-probe lane (S0 was correct);
        not-easy means: route to the recursive lane.
        """
        table: dict[str, bool] = {}
        oracle_path = Path(path)
        if not oracle_path.exists():
            logger.warning("Oracle route table not found: %s", path)
            return table
        with oracle_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = str(obj.get("id") or obj.get("question_id") or "").strip()
                if not qid:
                    continue
                table[qid] = bool(obj.get("easy", False))
        return table

    async def _run_upfront_decomposition(
        self, question: str, question_id: str
    ) -> PipelineResult:
        """A2: upfront decomposition, sequential execution, final answer."""
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
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

        if self.bootstrap_probe_first:
            bootstrap_goal = f"Answer this question directly. {target_profile}"
            bootstrap_capsule, bootstrap_tokens = await self.investigator.investigate_with_usage(
                sub_question=question,
                goal=bootstrap_goal,
                prior_facts=[],
                search_top_k_override=self.bootstrap_search_top_k,
                max_read_override=self.bootstrap_max_read,
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
                    sub_question=question,
                    fact_added=bootstrap_fact_added,
                    tokens=bootstrap_tokens,
                    metadata={
                        "bootstrap_probe": True,
                        "goal": bootstrap_goal,
                    },
                )
            )

            if bootstrap_fact_added:
                bootstrap_answer_obj, bootstrap_answer_tokens = (
                    await self.orchestrator.generate_answer_object_with_usage(
                        question=question,
                        facts=memory.get_all(),
                        target_profile=target_profile,
                        pending_slots=[],
                        trace=step_trace,
                    )
                )
                bootstrap_answer_obj = self._apply_answer_object_fallback(
                    bootstrap_answer_obj,
                    memory.get_all(),
                    "",
                )
                bootstrap_answer = bootstrap_answer_obj["answer"]
                bootstrap_fallback_source = bootstrap_answer_obj.get("fallback_source", "")
                bootstrap_cited_fact_ids = bootstrap_answer_obj["cited_fact_ids"]
                bootstrap_confidence = bootstrap_answer_obj["justification_confidence"]
                total_tokens += bootstrap_answer_tokens
                orchestrator_tokens += bootstrap_answer_tokens

                probe_gate, probe_gate_tokens = await self.orchestrator.assess_probe_with_usage(
                    question=question,
                    facts=memory.get_all(),
                    proposed_answer=bootstrap_answer,
                    target_profile=target_profile,
                    trace=step_trace,
                )
                total_tokens += probe_gate_tokens
                orchestrator_tokens += probe_gate_tokens
                step_trace.append(
                    StepTrace(
                        step=1,
                        action="assess",
                        tokens=probe_gate_tokens,
                        justification_confidence=probe_gate["confidence"],
                        metadata={
                            "bootstrap_probe": True,
                            "proposed_answer": bootstrap_answer,
                            "decision": probe_gate["action"],
                            "reason": probe_gate.get("reason", ""),
                        },
                    )
                )

                if probe_gate["action"] == "answer" and bootstrap_answer.strip():
                    answer = bootstrap_answer
                    step_trace.append(
                        StepTrace(
                            step=2,
                            action="answer",
                            tokens=bootstrap_answer_tokens,
                            cited_fact_ids=bootstrap_cited_fact_ids,
                            justification_confidence=bootstrap_confidence,
                            metadata={
                                "bootstrap_probe": True,
                                "fallback_source": bootstrap_fallback_source,
                                "probe_gate_reason": probe_gate.get("reason", ""),
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
                        route_decision="bootstrap_probe",
                        route_confidence=bootstrap_confidence,
                        route_draft_answer="",
                        slot_resolution={},
                        auto_verify_calls=auto_verify_calls,
                        answer_rejection_count=answer_rejection_count,
                    )
                if probe_gate["action"] == "refine":
                    answer_rejection_count += 1
                    step_trace.append(
                        StepTrace(
                            step=2,
                            action="answer_rejected_escalate",
                            tokens=bootstrap_answer_tokens,
                            cited_fact_ids=bootstrap_cited_fact_ids,
                            justification_confidence=bootstrap_confidence,
                            metadata={
                                "bootstrap_probe": True,
                                "fallback_source": bootstrap_fallback_source,
                                "probe_gate_reason": probe_gate.get("reason", ""),
                                "probe_gate_action": "refine",
                            },
                        )
                    )

                    followup_slot = str(probe_gate.get("slot_name", "")).strip()
                    followup_capsule, followup_tokens = await self.investigator.investigate_with_usage(
                        sub_question=str(probe_gate.get("sub_question", "")).strip() or question,
                        goal=str(probe_gate.get("goal", "")).strip()
                        or "Retrieve the next missing fact needed to answer the question.",
                        prior_facts=memory.get_all(),
                        retrieval_query=str(probe_gate.get("retrieval_query", "")).strip() or None,
                        slot_name=followup_slot,
                        slot_hint=target_profile,
                    )
                    total_tokens += followup_tokens
                    subagent_tokens += followup_tokens
                    subagent_calls += 1
                    retrieved_doc_ids, retrieved_docs_total = self._merge_retrieval_stats(
                        retrieved_doc_ids,
                        retrieved_docs_total,
                        followup_capsule,
                    )
                    followup_fact_added = self._add_fact(
                        memory,
                        followup_capsule,
                        step=3,
                        slot_name=followup_slot,
                    )
                    step_trace.append(
                        StepTrace(
                            step=3,
                            action="spawn",
                            sub_question=str(probe_gate.get("sub_question", "")).strip() or question,
                            fact_added=followup_fact_added,
                            tokens=followup_tokens,
                            slot_name=followup_slot,
                            metadata={
                                "bootstrap_followup": True,
                                "goal": str(probe_gate.get("goal", "")).strip(),
                                "retrieval_query": str(probe_gate.get("retrieval_query", "")).strip(),
                                "reason": probe_gate.get("reason", ""),
                            },
                        )
                    )
                    (
                        verify_tokens,
                        verify_delta,
                        auto_verify_delta,
                    ) = await self._maybe_verify_fact(
                        question=question,
                        step=3,
                        slot_name=followup_slot,
                        sub_question=str(probe_gate.get("sub_question", "")).strip() or question,
                        capsule=followup_capsule,
                        memory=memory,
                        slot_state=[],
                        step_trace=step_trace,
                    )
                    total_tokens += verify_tokens
                    subagent_tokens += verify_tokens
                    verify_count += verify_delta
                    auto_verify_calls += auto_verify_delta

                    followup_answer_obj, followup_answer_tokens = (
                        await self.orchestrator.generate_answer_object_with_usage(
                            question=question,
                            facts=memory.get_all(),
                            target_profile=target_profile,
                            pending_slots=[],
                            trace=step_trace,
                            route_draft_answer=bootstrap_answer,
                        )
                    )
                    followup_answer_obj = self._apply_answer_object_fallback(
                        followup_answer_obj,
                        memory.get_all(),
                        bootstrap_answer,
                    )
                    total_tokens += followup_answer_tokens
                    orchestrator_tokens += followup_answer_tokens

                    if not self._should_escalate_answer(followup_answer_obj, []):
                        answer = followup_answer_obj["answer"]
                        step_trace.append(
                            StepTrace(
                                step=4,
                                action="answer",
                                tokens=followup_answer_tokens,
                                cited_fact_ids=followup_answer_obj["cited_fact_ids"],
                                justification_confidence=followup_answer_obj["justification_confidence"],
                                metadata={
                                    "bootstrap_followup": True,
                                    "justification": followup_answer_obj["justification"],
                                    "missing_slot": followup_answer_obj["missing_slot"],
                                    "fallback_source": followup_answer_obj.get("fallback_source", ""),
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
                            route_decision="bootstrap_followup",
                            route_confidence=followup_answer_obj["justification_confidence"],
                            route_draft_answer=bootstrap_answer,
                            slot_resolution={},
                            auto_verify_calls=auto_verify_calls,
                            answer_rejection_count=answer_rejection_count,
                        )
                    answer_rejection_count += 1
                    step_trace.append(
                        StepTrace(
                            step=4,
                            action="answer_rejected_escalate",
                            tokens=followup_answer_tokens,
                            cited_fact_ids=followup_answer_obj["cited_fact_ids"],
                            justification_confidence=followup_answer_obj["justification_confidence"],
                            metadata={
                                "bootstrap_followup": True,
                                "justification": followup_answer_obj["justification"],
                                "missing_slot": followup_answer_obj["missing_slot"],
                                "fallback_source": followup_answer_obj.get("fallback_source", ""),
                                "probe_gate_action": "refine",
                            },
                        )
                    )
                else:
                    step_trace.append(
                        StepTrace(
                            step=2,
                            action="probe_recurse",
                            tokens=0,
                            cited_fact_ids=bootstrap_cited_fact_ids,
                            justification_confidence=bootstrap_confidence,
                            metadata={
                                "bootstrap_probe": True,
                                "fallback_source": bootstrap_fallback_source,
                                "probe_gate_reason": probe_gate.get("reason", ""),
                                "probe_gate_action": "recurse",
                            },
                        )
                    )

        route_target_profile = target_profile
        if memory.get_all():
            bootstrap_fact = memory.get_all()[-1]
            if bootstrap_fact.text.strip():
                route_target_profile = (
                    f"{target_profile}\nBootstrap grounded clue: {bootstrap_fact.text.strip()}"
                ).strip()
        route, route_tokens = await self.orchestrator.route_with_usage(
            question=question,
            target_profile=route_target_profile,
        )
        total_tokens += route_tokens
        orchestrator_tokens += route_tokens

        slot_state = self._initialise_slot_state(route, target_profile)
        route_step = len(step_trace)
        step_ceiling = route_step + self.max_steps
        step_trace.append(
            StepTrace(
                step=route_step,
                action="route",
                tokens=route_tokens,
                route_decision=route["action"],
                route_confidence=route["confidence"],
                route_draft_answer=route["draft_answer"],
                metadata={
                    "answer_type": route["answer_type"],
                    "target_slot": route["target_slot"],
                    "retrieval_query": route.get("retrieval_query", ""),
                    "pending_slots": self._slot_snapshot(slot_state),
                },
            )
        )

        next_step = route_step + 1
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
                step_trace[route_step].metadata["refined_bootstrap_sub_question"] = initial_sub_question
                step_trace[route_step].metadata["refined_bootstrap_retrieval_query"] = (
                    initial_retrieval_query or initial_sub_question
                )
                step_trace[route_step].metadata["refined_bootstrap_goal"] = initial_goal
                step_trace[route_step].tokens += refine_tokens
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
                    step_trace[route_step].metadata["slot_focused_bootstrap_sub_question"] = initial_sub_question
                    step_trace[route_step].metadata["slot_focused_bootstrap_retrieval_query"] = (
                        initial_retrieval_query or initial_sub_question
                    )
                    step_trace[route_step].metadata["slot_focused_bootstrap_goal"] = initial_goal
                    step_trace[route_step].tokens += refine_tokens

        probe_step = route_step + 1
        next_step = probe_step
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
            fact_added = self._add_fact(memory, capsule, step=probe_step, slot_name=probe_slot)
            self._update_slot_resolution(slot_state, probe_slot, capsule)
            step_trace.append(
                StepTrace(
                    step=probe_step,
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
                step=probe_step,
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
            next_step = probe_step + 1

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
                )
                total_tokens += answer_tokens
                orchestrator_tokens += answer_tokens

                if not self._should_escalate_answer(answer_obj, pending_slots):
                    answer = answer_obj["answer"]
                    step_trace.append(
                        StepTrace(
                            step=probe_step + 1,
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
                if next_step <= step_ceiling:
                    answer_rejection_count += 1
                    step_trace.append(
                        StepTrace(
                            step=probe_step + 1,
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

        for step in range(next_step, step_ceiling + 1):
            if self._should_force_budget_answer(total_tokens):
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
                            "budget_exhausted": True,
                        },
                    )
                )
                break
            pending_slots = self._pending_slots(slot_state)
            ready_parallel_slots = self._parallel_ready_slots(slot_state)
            if (
                self.enable_parallel_independent_hops
                and len(ready_parallel_slots) > 1
                and step < step_ceiling
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
            if self.ablation_force_spawn and step < step_ceiling:
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

            if action == "answer":
                if pending_slots and step < step_ceiling:
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
                    )
                    total_tokens += answer_tokens
                    orchestrator_tokens += answer_tokens

                    if self._should_escalate_answer(answer_obj, pending_slots) and step < step_ceiling:
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
                            missing_reason=answer_obj["missing_slot"]
                            or "The current answer could not be justified from the fact pool.",
                        )
                        total_tokens += spawn_tokens
                        orchestrator_tokens += spawn_tokens
                        action = "spawn"
                        decide_tokens += spawn_tokens
                    elif (
                        self._should_escalate_answer(answer_obj, pending_slots)
                        and self._should_do_final_targeted_recovery(answer_obj, pending_slots)
                    ):
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
                            pending_slots=pending_slots,
                            missing_reason=answer_obj["missing_slot"]
                            or "The current answer could not be justified from the fact pool.",
                        )
                        total_tokens += recovery_tokens
                        orchestrator_tokens += recovery_tokens
                        if recovery_decision.get("action") == "spawn":
                            slot_name = recovery_decision.get("slot_name") or self._first_pending_slot(slot_state)
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
            )
            total_tokens += answer_tokens
            orchestrator_tokens += answer_tokens
            answer = answer_obj["answer"]
            step_trace.append(
                StepTrace(
                    step=step_ceiling + 1,
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
            search_top_k_override=self.bootstrap_search_top_k,
            max_read_override=self.bootstrap_max_read,
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
                if capsule.fact.slot_filled and capsule.fact.confidence > 0:
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
        """Only recover at the boundary when exactly one missing slot is explicitly named."""
        if len(pending_slots) != 1:
            return False
        missing_slot = str(answer_obj.get("missing_slot", "")).strip()
        if not missing_slot:
            return False
        pending_slot = str(pending_slots[0].get("slot_name", "")).strip()
        return bool(pending_slot and missing_slot == pending_slot)

    @staticmethod
    def _best_fact_answer_with_index(facts: list) -> tuple[int | None, str, float]:
        """Return the strongest grounded answer span in fact memory."""
        candidates: list[tuple[float, int, int, int, str]] = []
        for idx, fact in enumerate(facts, start=1):
            answer_span = str(getattr(fact, "answer_span", "")).strip()
            if not answer_span or Orchestrator._looks_meta_answer(answer_span):
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
    ) -> tuple[str, str, list[int], float]:
        """Fallback from empty/meta answers to grounded fact memory only."""
        cleaned_answer = str(answer or "").strip()
        if cleaned_answer and not Orchestrator._looks_meta_answer(cleaned_answer):
            return cleaned_answer, "", [], 0.0

        fact_idx, fact_answer, fact_confidence = cls._best_fact_answer_with_index(facts)
        if fact_answer:
            return fact_answer, "fact_memory", [fact_idx] if fact_idx else [], fact_confidence

        return cleaned_answer, "", [], 0.0

    @classmethod
    def _apply_answer_object_fallback(
        cls,
        answer_obj: dict[str, Any],
        facts: list,
        route_draft_answer: str = "",
    ) -> dict[str, Any]:
        """Fill empty structured answers from fact memory or router draft."""
        updated = dict(answer_obj)
        answer, fallback_source, cited_fact_ids, fallback_confidence = cls._apply_answer_fallback(
            updated.get("answer", ""),
            facts,
            route_draft_answer,
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
        """Heuristic short-circuit retired in favour of the sufficiency score."""
        return False

    @staticmethod
    def _target_profile(question: str) -> str:
        """Constant, dataset-agnostic profile string.

        The previous version inferred answer-type hints from question wording,
        which leaked benchmark patterns into the controller. The sufficiency
        controller relies entirely on the post-probe sufficiency score, so the
        same string is used for every question on every dataset.
        """
        return f"Answer with the exact span the question asks for. Question: {question.strip()}"

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
