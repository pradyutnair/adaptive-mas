"""Adaptive RAG pipeline: probe → assess → decide loop.

Mirrors the sufficiency controller pattern:
1. Route: generate slot DAG (required_hops) from the question
2. Probe: send full question to investigator → capsule
3. Assess: is the probe answer sufficient? (LLM gate)
4. If sufficient → return probe answer
5. If not → decide loop: LLM generates ONE sub-question at a time
6. Synthesize: final answer from all collected facts

Key design property: investigators are isolated — raw passages never
leak back to the orchestrator. Only compact evidence capsules.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient

from .fact_memory import FactMemory
from .investigator import Investigator
from .orchestrator import Orchestrator
from .types import EvidenceCapsule, Fact, PipelineResult, StepTrace

logger = logging.getLogger(__name__)


class AdaptiveRecursivePipeline:
    """Probe-assess-decide adaptive RAG pipeline."""

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

        self.fact_memory_capacity: int = config.get("fact_memory.capacity", 4)
        self.fact_memory_strategy: str = str(config.get("fact_memory.strategy", "salience"))
        self.max_steps: int = int(config.get("orchestrator.max_steps", 4))
        self.search_top_k: int = int(config.get("adaptive.bootstrap_search_top_k", 5))
        self.max_read: int = int(config.get("adaptive.bootstrap_max_read", 6))
        self.max_total_tokens: int = int(config.get("budget.max_total_tokens", 0) or 0)
        self.sufficiency_threshold: float = float(
            config.get("adaptive.sufficiency_threshold", 0.70)
        )
        self.sufficiency_max_recurse_steps: int = int(
            config.get("adaptive.sufficiency_max_recurse_steps", self.max_steps)
        )
        self.sufficiency_min_recurse_steps: int = int(
            config.get("adaptive.sufficiency_min_recurse_steps", 1)
        )

    async def run(self, question: str, question_id: str) -> PipelineResult:
        logger.info("Pipeline start: question_id=%s", question_id)
        return await self._run_adaptive_topology(question, question_id)

    async def _run_adaptive_topology(self, question: str, question_id: str) -> PipelineResult:
        memory = FactMemory.with_strategy(
            capacity=self.fact_memory_capacity,
            strategy=self.fact_memory_strategy,
        )
        target_profile = self._target_profile(question)
        trace: list[StepTrace] = []
        total_tokens = 0
        orch_tokens = 0
        agent_tokens = 0
        ret_ids: list[str] = []
        ret_total = 0
        n_calls = 0

        route, route_tok = await self.orchestrator.route_with_usage(
            question=question,
            target_profile=target_profile,
        )
        total_tokens += route_tok
        orch_tokens += route_tok
        required_hops: list[dict] = list(route.get("required_hops") or [])
        answer_type = str(route.get("answer_type", "entity"))
        route_action = str(route.get("action", "single_probe"))

        trace.append(StepTrace(
            step=0, action="route", tokens=route_tok,
            route_decision=route_action,
            route_confidence=route.get("confidence", 0.0),
            metadata={
                "required_hops": required_hops,
                "target_slot": route.get("target_slot", ""),
                "answer_type": answer_type,
            },
        ))

        if route_action == "single_probe":
            direct_queries = [str(route.get("retrieval_query", "")).strip() or question]
            if question not in direct_queries:
                direct_queries.append(question)
            for attempt, retrieval_query in enumerate(direct_queries[:2], start=1):
                capsule, tok = await self.orchestrator.retrieve_and_distill_with_usage(
                    question=question,
                    retrieval_query=retrieval_query,
                    target_profile=target_profile,
                    answer_type=answer_type,
                    top_k=self.search_top_k,
                )
                total_tokens += tok
                orch_tokens += tok
                ret_ids, ret_total = self._merge_stats(ret_ids, ret_total, capsule)
                fact_added = self._add_fact(
                    memory, capsule, step=len(trace), slot_name="final_answer"
                )
                trace.append(StepTrace(
                    step=len(trace), action="spawn", sub_question=question,
                    fact_added=fact_added, tokens=tok, slot_name="final_answer",
                    route_decision="direct",
                    justification_confidence=capsule.fact.confidence,
                    metadata={
                        "direct_attempt": attempt,
                        "retrieval_query": retrieval_query,
                    },
                ))
                if capsule.answer and capsule.fact.slot_filled and capsule.fact.confidence >= self.sufficiency_threshold:
                    trace.append(StepTrace(
                        step=len(trace), action="answer", tokens=0,
                        justification_confidence=capsule.fact.confidence,
                        metadata={"route": "direct_semantic_probe"},
                    ))
                    return self._build_result(
                        question_id=question_id, question=question,
                        answer=capsule.answer, trace=trace, memory=memory,
                        n_calls=n_calls, total_tokens=total_tokens,
                        orch_tokens=orch_tokens, agent_tokens=agent_tokens,
                        ret_ids=ret_ids, ret_total=ret_total,
                        route_decision="direct_semantic_probe",
                        route_confidence=route.get("confidence", 0.0),
                        sufficiency=capsule.fact.confidence,
                    )

        for sub_step in range(self.max_steps):
            decision, decide_tok = await self.orchestrator.decide_with_usage(
                question=question,
                facts=memory.get_all(),
                trace=trace,
                step=sub_step,
                target_profile=target_profile,
                pending_slots=required_hops,
            )
            total_tokens += decide_tok
            orch_tokens += decide_tok
            if decision.get("action") == "answer" and not memory.get_all():
                decision = self._fallback_slot_decision(required_hops)
            decision = self._resolve_decision_placeholders(
                decision, required_hops, memory.get_all()
            )
            action = decision.get("action", "answer")

            if action == "answer":
                trace.append(StepTrace(
                    step=len(trace), action="answer", tokens=decide_tok,
                    metadata={"decide_loop": True, "decision": "answer"},
                ))
                break

            sub_question = str(decision.get("sub_question", "")).strip() or question
            retrieval_query = str(decision.get("retrieval_query", "")).strip() or sub_question
            goal = str(decision.get("goal", "")).strip() or target_profile
            slot_name = str(decision.get("slot_name", "")).strip()
            expected_type = self._expected_type_for_slot(required_hops, slot_name, answer_type)
            slot_hint = f"Expected answer type: {expected_type}. {goal}".strip()

            capsule, inv_tok = await self._investigate(
                sub_question=sub_question,
                retrieval_query=retrieval_query,
                target_profile=target_profile,
                prior_facts=[],
                slot_name=slot_name,
                slot_hint=slot_hint,
            )
            total_tokens += inv_tok
            agent_tokens += inv_tok
            n_calls += 1
            ret_ids, ret_total = self._merge_stats(ret_ids, ret_total, capsule)
            fact_added = self._add_fact(memory, capsule, step=len(trace), slot_name=slot_name)
            self._mark_resolved(required_hops, slot_name, capsule)
            trace.append(StepTrace(
                step=len(trace), action=action, sub_question=sub_question,
                fact_added=fact_added, tokens=inv_tok, slot_name=slot_name,
                justification_confidence=capsule.fact.confidence,
                metadata={
                    "decide_loop": True,
                    "goal": goal,
                    "retrieval_query": retrieval_query,
                    "expected_answer_type": expected_type,
                },
            ))

        facts = memory.get_all()
        final_answer = ""
        if facts:
            answer_obj, synth_tok = await self.orchestrator.generate_answer_object_with_usage(
                question=question,
                facts=facts,
                target_profile=f"{target_profile}\nExpected answer type: {answer_type}.",
                pending_slots=required_hops,
                trace=trace,
            )
            total_tokens += synth_tok
            orch_tokens += synth_tok
            final_answer = str(answer_obj.get("answer", "")).strip()
            trace.append(StepTrace(
                step=len(trace), action="answer", tokens=synth_tok,
                cited_fact_ids=answer_obj.get("cited_fact_ids", []),
                justification_confidence=answer_obj.get("justification_confidence", 0.0),
                metadata={"answer_source": "synthesis"},
            ))

        if not final_answer or Orchestrator._looks_meta_answer(final_answer):
            final_answer = self._best_fact_span(facts)
        if not final_answer:
            final_answer = "unknown"

        return self._build_result(
            question_id=question_id, question=question,
            answer=final_answer, trace=trace, memory=memory,
            n_calls=n_calls, total_tokens=total_tokens,
            orch_tokens=orch_tokens, agent_tokens=agent_tokens,
            ret_ids=ret_ids, ret_total=ret_total,
            route_decision="delegated_semantic_topology",
            route_confidence=route.get("confidence", 0.0),
            sufficiency=0.0,
        )

    # ------------------------------------------------------------------
    # Investigator call
    # ------------------------------------------------------------------

    @classmethod
    def _fallback_slot_decision(cls, required_hops: list[dict]) -> dict[str, Any]:
        resolved = {
            str(hop.get("slot_name", "")).strip()
            for hop in required_hops
            if hop.get("resolved")
        }
        for hop in required_hops:
            if hop.get("resolved"):
                continue
            dependencies = [str(dep).strip() for dep in (hop.get("dependencies") or [])]
            if any(dep and dep not in resolved for dep in dependencies):
                continue
            sub_question = str(hop.get("sub_question", "")).strip()
            if not sub_question:
                continue
            return {
                "action": "spawn",
                "sub_question": sub_question,
                "retrieval_query": str(hop.get("retrieval_query", "")).strip() or sub_question,
                "goal": str(hop.get("hint", "")).strip() or "Resolve the next missing slot.",
                "slot_name": str(hop.get("slot_name", "")).strip(),
            }
        return {"action": "answer"}

    @classmethod
    def _resolve_decision_placeholders(
        cls,
        decision: dict[str, Any],
        required_hops: list[dict],
        facts: list[Fact],
    ) -> dict[str, Any]:
        if decision.get("action") not in {"spawn", "refine"}:
            return decision
        values = cls._slot_values(facts)
        resolved = dict(decision)
        for key in ("sub_question", "retrieval_query", "goal"):
            resolved[key] = cls._substitute_placeholders(
                str(resolved.get(key, "")), values
            )
        unresolved_text = f"{resolved.get('sub_question', '')} {resolved.get('retrieval_query', '')}"
        if cls._has_unresolved_placeholder(unresolved_text):
            fallback = cls._fallback_slot_decision(required_hops)
            for key in ("sub_question", "retrieval_query", "goal"):
                fallback[key] = cls._substitute_placeholders(
                    str(fallback.get(key, "")), values
                )
            return fallback
        return resolved

    @staticmethod
    def _slot_values(facts: list[Fact]) -> dict[str, str]:
        values: dict[str, str] = {}
        for fact in facts:
            slot = str(getattr(fact, "slot_name", "")).strip()
            answer = str(getattr(fact, "answer_span", "")).strip()
            if slot and answer:
                values[slot] = answer
        return values

    @staticmethod
    def _substitute_placeholders(text: str, values: dict[str, str]) -> str:
        output = text
        for slot, value in values.items():
            output = output.replace("{{" + slot + "}}", value)
            output = output.replace("{" + slot + "}", value)
        return output

    @staticmethod
    def _has_unresolved_placeholder(text: str) -> bool:
        return bool(re.search(r"\{\{[^}]+\}\}|\{[^}]+\}", text))

    @staticmethod
    def _expected_type_for_slot(
        required_hops: list[dict],
        slot_name: str,
        default_type: str,
    ) -> str:
        for hop in required_hops:
            if str(hop.get("slot_name", "")).strip() == slot_name:
                return str(hop.get("expected_answer_type", "")).strip() or default_type
        return default_type

    @staticmethod
    def _mark_resolved(
        required_hops: list[dict],
        slot_name: str,
        capsule: EvidenceCapsule,
    ) -> None:
        if not slot_name or not capsule.fact.slot_filled or capsule.fact.confidence <= 0:
            return
        for hop in required_hops:
            if hop.get("slot_name") == slot_name:
                hop["resolved"] = True
                break

    async def _investigate(
        self, sub_question: str, retrieval_query: str,
        target_profile: str, prior_facts: list[Fact],
        slot_name: str, slot_hint: str,
    ) -> tuple[EvidenceCapsule, int]:
        return await self.investigator.investigate_with_usage(
            sub_question=sub_question,
            goal="Answer this sub-question with private semantic retrieval. Return the shortest supported answer span.",
            prior_facts=prior_facts,
            retrieval_query=retrieval_query or sub_question,
            slot_name=slot_name,
            slot_hint=slot_hint,
            search_top_k_override=self.search_top_k,
            max_read_override=self.max_read,
        )

    # ------------------------------------------------------------------
    # Sufficiency helpers
    # ------------------------------------------------------------------

    def _recurse_budget(self, sufficiency: float) -> int:
        raw = math.ceil(self.sufficiency_max_recurse_steps * (1.0 - sufficiency))
        return max(self.sufficiency_min_recurse_steps, min(raw, self.sufficiency_max_recurse_steps))

    @staticmethod
    def _compute_alignment(capsule: EvidenceCapsule, proposed_answer: str) -> float:
        if not proposed_answer.strip():
            return 0.0
        capsule_text = str(capsule.fact.text or "").lower()
        answer_span = str(capsule.fact.answer_span or "").lower()
        pa = proposed_answer.strip().lower()
        if pa in capsule_text or pa in answer_span:
            return 1.0
        if answer_span and answer_span in pa:
            return 1.0
        return 0.5

    # ------------------------------------------------------------------
    # Fact / memory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_fact(memory: FactMemory, capsule: EvidenceCapsule, step: int, slot_name: str = "") -> bool:
        if not capsule.fact.text:
            return False
        capsule.fact.source_step = step
        capsule.fact.slot_name = slot_name
        memory.add(capsule.fact)
        return True

    @staticmethod
    def _merge_stats(ids: list[str], total: int, capsule: EvidenceCapsule) -> tuple[list[str], int]:
        seen = set(ids)
        merged = list(ids)
        for d in capsule.retrieved_doc_ids:
            if d not in seen:
                seen.add(d)
                merged.append(d)
        return merged, total + int(capsule.retrieved_docs_total)

    @staticmethod
    def _best_fact_span(facts: list) -> str:
        best = ("", -1.0, -1)
        for f in facts:
            span = str(getattr(f, "answer_span", "")).strip()
            if span and not Orchestrator._looks_meta_answer(span):
                conf = float(getattr(f, "confidence", 0.0))
                step = int(getattr(f, "source_step", 0))
                if step > best[2] or (step == best[2] and conf > best[1]):
                    best = (span, conf, step)
        return best[0]

    @staticmethod
    def _target_profile(question: str) -> str:
        return f"Answer with the exact span the question asks for. Question: {question.strip()}"

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self, *, question_id: str, question: str, answer: str,
        trace: list[StepTrace], memory: FactMemory,
        n_calls: int, total_tokens: int, orch_tokens: int, agent_tokens: int,
        ret_ids: list[str], ret_total: int,
        route_decision: str, route_confidence: float,
        sufficiency: float = 0.0,
    ) -> PipelineResult:
        return PipelineResult(
            question_id=question_id,
            question=question,
            answer=answer,
            step_trace=trace,
            num_subagent_calls=n_calls,
            total_tokens=total_tokens,
            orchestrator_tokens=orch_tokens,
            subagent_tokens=agent_tokens,
            facts_used=memory.get_all(),
            retrieved_doc_ids=ret_ids,
            retrieved_docs_total=ret_total,
            evidence_capsule_limit=self.investigator.evidence_capsule_limit,
            fact_memory_capacity=self.fact_memory_capacity,
            route_decision=route_decision,
            route_confidence=route_confidence,
            slot_resolution={},
            extras={"controller": "probe_decide", "sufficiency": sufficiency},
        )
