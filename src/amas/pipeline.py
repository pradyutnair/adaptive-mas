"""Adaptive AMAS pipeline.

Flow:
1. DIRECT_SOLVE — orchestrator (lead agent) tries to answer with <=2 search
   calls itself. If grounded and confident, return (SAS path).
2. ESCALATE -> PLAN — orchestrator decomposes into a slot DAG with
   ``dependency_group``s.
3. PARALLEL DECIDE LOOP — at each round, spawn isolated investigators
   IN PARALLEL for every slot whose dependencies are resolved. Each
   investigator returns an EvidenceCapsule. If a slot fails, refine its
   query (via ``refine_slot``) and retry up to ``max_slot_retries`` times.
4. ANSWER (deterministic) — read the final-slot ``answer_span`` straight
   from the resolved facts. No separate synthesize LLM call.

Context-isolation invariants (AGENTS.md):
- Raw chunks NEVER appear in any prompt template.
- Chunks only enter as ``role: tool`` messages inside a single agent's
  private conversation (orchestrator's direct_solve loop, OR an investigator's
  loop). The orchestrator's plan/refine prompts NEVER see chunk text.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from .config import Config
from .investigator import Investigator
from .llm import LLMClient
from .orchestrator import Orchestrator
from .retriever import Retriever
from .types import AnswerType, EvidenceCapsule, Fact, PipelineResult, StepTrace

logger = logging.getLogger(__name__)


class AMASPipeline:
    """Adaptive Multi-Agent System pipeline."""

    def __init__(self, config: Config) -> None:
        self.config = config

        # Per-role LLM clients (mixing providers is supported).
        self.direct_llm = LLMClient.from_config(config.agent_llm("orchestrator"))
        self.plan_llm = LLMClient.from_config(
            config.agent_llm("planner") or config.agent_llm("orchestrator")
        )
        self.refine_llm = LLMClient.from_config(
            config.agent_llm("refiner") or config.agent_llm("orchestrator")
        )
        self.investigator_llm = LLMClient.from_config(config.agent_llm("investigator"))

        ret_cfg = config.raw().get("retriever", {}) or {}
        self.retriever = Retriever(
            base_url=ret_cfg.get("base_url", "http://node408:8003"),
            default_top_k=int(ret_cfg.get("top_k", 10)),
            timeout_seconds=float(ret_cfg.get("timeout_seconds", 30)),
        )

        self.orchestrator = Orchestrator(
            direct_llm=self.direct_llm,
            plan_llm=self.plan_llm,
            refine_llm=self.refine_llm,
            retriever=self.retriever,
            direct_max_searches=int(config.get("pipeline.direct_max_searches", 2)),
            direct_top_k=int(ret_cfg.get("top_k", 10)),
            max_plan_hops=int(config.get("pipeline.max_plan_hops", 6)),
        )
        self.investigator = Investigator(
            llm=self.investigator_llm,
            retriever=self.retriever,
            top_k=int(ret_cfg.get("top_k", 10)),
            min_confidence=float(config.get("pipeline.min_fact_confidence", 0.3)),
            max_searches=int(config.get("pipeline.max_searches_per_subagent", 3)),
        )

        self.max_rounds = int(config.get("pipeline.max_rounds", 4))
        self.max_slot_retries = int(config.get("pipeline.max_slot_retries", 2))
        self.sufficiency_threshold = float(
            config.get("pipeline.sufficiency_threshold", 0.6)
        )
        self.max_total_tokens = int(config.get("pipeline.max_total_tokens", 0) or 0)

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    async def run(self, question: str, question_id: str) -> PipelineResult:
        logger.info("AMAS start: qid=%s", question_id)

        trace: list[StepTrace] = []
        facts: list[Fact] = []
        ret_ids: list[str] = []
        ret_total = 0
        total_tokens = 0
        orch_tokens = 0
        agent_tokens = 0
        n_subagent_calls = 0

        # ----- Phase 1: orchestrator tries to solve directly -----
        direct_result, direct_tok, direct_ids = await self.orchestrator.direct_solve(question)
        total_tokens += direct_tok
        orch_tokens += direct_tok
        for cid in direct_ids:
            if cid not in ret_ids:
                ret_ids.append(cid)
        ret_total += len(direct_ids)
        trace.append(StepTrace(
            step=0, action="route", tokens=direct_tok,
            route_decision=direct_result["action"],
            metadata={"phase": "direct_solve",
                      "direct_searches": len(direct_ids),
                      "result": {k: v for k, v in direct_result.items()
                                 if k != "support_ids"}},
        ))

        if direct_result["action"] == "answer":
            confidence = float(direct_result.get("confidence", 0.0))
            if confidence >= self.sufficiency_threshold:
                return self._build_result(
                    question_id=question_id, question=question,
                    answer=direct_result["answer_span"],
                    trace=trace, facts=facts, n_calls=n_subagent_calls,
                    total_tokens=total_tokens, orch_tokens=orch_tokens,
                    agent_tokens=agent_tokens,
                    ret_ids=ret_ids, ret_total=ret_total,
                    route_decision="direct_solve",
                    extras={"confidence": confidence,
                            "answer_type": direct_result.get("answer_type", "entity"),
                            "support_ids": direct_result.get("support_ids", [])},
                )

        # ----- Phase 2: escalate -> plan -----
        reason = (
            direct_result.get("reason", "")
            if direct_result["action"] == "escalate"
            else "low_confidence_direct_answer"
        )
        plan_obj, plan_tok = await self.orchestrator.plan(question, reason=reason)
        total_tokens += plan_tok
        orch_tokens += plan_tok
        plan: list[dict] = list(plan_obj["plan"])
        answer_type: str = plan_obj["answer_type"]
        trace.append(StepTrace(
            step=len(trace), action="route", tokens=plan_tok,
            route_decision="decompose",
            metadata={"phase": "plan", "plan": plan,
                      "answer_type": answer_type, "escalation_reason": reason},
        ))

        if not plan:
            plan = [{
                "slot_name": "final_answer",
                "sub_question": question,
                "retrieval_query": question,
                "expected_answer_type": answer_type,
                "dependencies": [],
                "dependency_group": 0,
            }]

        # Per-slot scratch state for retries.
        attempt_state: dict[str, dict[str, Any]] = {
            hop["slot_name"]: {
                "retries": 0,
                "failed_queries": [],
                "current_sub_question": hop["sub_question"],
                "current_retrieval_query": hop["retrieval_query"],
            }
            for hop in plan
        }

        # ----- Phase 3: parallel-aware decide loop -----
        for round_idx in range(self.max_rounds):
            if self.max_total_tokens and total_tokens >= self.max_total_tokens:
                logger.info("Token budget reached (%d/%d).",
                            total_tokens, self.max_total_tokens)
                break

            # Determine ready slots: deps met, not yet resolved, retries available.
            slot_values = {
                f.slot_name: f.answer_span for f in facts
                if f.slot_filled and f.answer_span
            }
            resolved = set(slot_values.keys())
            ready_hops: list[dict] = []
            for hop in plan:
                slot = hop["slot_name"]
                if slot in resolved:
                    continue
                if attempt_state[slot]["retries"] > self.max_slot_retries:
                    continue
                deps = hop.get("dependencies", []) or []
                if not all(d in resolved for d in deps):
                    continue
                ready_hops.append(hop)

            if not ready_hops:
                break

            # Group ready hops by dependency_group: same group runs in parallel.
            groups: dict[int, list[dict]] = defaultdict(list)
            for hop in ready_hops:
                groups[int(hop.get("dependency_group", 0))].append(hop)
            current_group_id = min(groups.keys())
            current_group = groups[current_group_id]

            # Materialise current sub-question / retrieval_query for each hop,
            # substituting resolved slot values.
            tasks = []
            task_meta: list[dict[str, Any]] = []
            for hop in current_group:
                slot = hop["slot_name"]
                state = attempt_state[slot]
                sq = self.orchestrator.substitute_placeholders(
                    state["current_sub_question"], slot_values,
                )
                rq = self.orchestrator.substitute_placeholders(
                    state["current_retrieval_query"] or sq, slot_values,
                )
                expected = hop.get("expected_answer_type", answer_type)
                tasks.append(self.investigator.investigate(
                    sub_question=sq,
                    retrieval_query=rq,
                    expected_answer_type=expected,
                    slot_name=slot,
                ))
                task_meta.append({
                    "slot": slot, "sub_question": sq, "retrieval_query": rq,
                    "expected": expected, "attempt": state["retries"] + 1,
                })

            # Spawn investigators for the current group concurrently.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for meta, result in zip(task_meta, results):
                slot = meta["slot"]
                state = attempt_state[slot]
                if isinstance(result, Exception):
                    logger.warning("Investigator crashed for slot %s: %s", slot, result)
                    state["retries"] += 1
                    state["failed_queries"].append(meta["retrieval_query"])
                    trace.append(StepTrace(
                        step=len(trace), action="spawn", tokens=0,
                        sub_question=meta["sub_question"], slot_name=slot,
                        fact_added=False, justification_confidence=0.0,
                        metadata={"round": round_idx, "group": current_group_id,
                                  "attempt": meta["attempt"], "error": str(result),
                                  "retrieval_query": meta["retrieval_query"]},
                    ))
                    continue

                capsule, inv_tok = result
                total_tokens += inv_tok
                agent_tokens += inv_tok
                n_subagent_calls += 1
                ret_ids, ret_total = self._merge_ret(ret_ids, ret_total, capsule)
                added = self._accept_fact(facts, capsule, source_step=len(trace),
                                          slot_name=slot)
                trace.append(StepTrace(
                    step=len(trace), action="spawn", tokens=inv_tok,
                    sub_question=meta["sub_question"], slot_name=slot,
                    fact_added=added,
                    justification_confidence=capsule.fact.confidence,
                    metadata={"round": round_idx, "group": current_group_id,
                              "attempt": meta["attempt"],
                              "retrieval_query": meta["retrieval_query"],
                              "expected_answer_type": meta["expected"],
                              "investigator_searches": getattr(
                                  self.investigator, "last_searches_used", None,
                              )},
                ))

                if not capsule.fact.slot_filled:
                    state["retries"] += 1
                    state["failed_queries"].append(meta["retrieval_query"])

            # Refinement pass: for any slot that failed AND still has retries
            # left, ask the orchestrator to rewrite a different query.
            for hop in current_group:
                slot = hop["slot_name"]
                state = attempt_state[slot]
                if slot in {f.slot_name for f in facts if f.slot_filled}:
                    continue
                if state["retries"] > self.max_slot_retries:
                    continue
                if state["retries"] == 0:
                    continue  # never failed yet, no refinement needed
                refined, refine_tok = await self.orchestrator.refine_slot(
                    slot_name=slot,
                    expected_answer_type=hop.get("expected_answer_type", answer_type),
                    original_sub_question=hop["sub_question"],
                    failed_queries=state["failed_queries"],
                    facts=facts,
                )
                total_tokens += refine_tok
                orch_tokens += refine_tok
                trace.append(StepTrace(
                    step=len(trace), action="refine", tokens=refine_tok,
                    sub_question=refined.get("sub_question", ""), slot_name=slot,
                    metadata={"round": round_idx,
                              "retrieval_query": refined.get("retrieval_query", ""),
                              "failed_queries": list(state["failed_queries"])},
                ))
                if refined.get("retrieval_query"):
                    state["current_sub_question"] = (
                        refined.get("sub_question") or hop["sub_question"]
                    )
                    state["current_retrieval_query"] = refined["retrieval_query"]

            if self._final_slot_resolved(plan, facts):
                trace.append(StepTrace(
                    step=len(trace), action="answer", tokens=0,
                    metadata={"reason": "final_slot_resolved"},
                ))
                break

        # ----- Phase 4: deterministic answer (no synthesize call) -----
        final_answer = self._best_fact_answer(plan, facts, AnswerType.coerce(answer_type))
        if not final_answer:
            final_answer = "unknown"

        return self._build_result(
            question_id=question_id, question=question, answer=final_answer,
            trace=trace, facts=facts, n_calls=n_subagent_calls,
            total_tokens=total_tokens, orch_tokens=orch_tokens,
            agent_tokens=agent_tokens, ret_ids=ret_ids, ret_total=ret_total,
            route_decision="decompose",
            extras={"answer_type": answer_type, "plan_size": len(plan)},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_ret(
        ids: list[str], total: int, capsule: EvidenceCapsule
    ) -> tuple[list[str], int]:
        seen = set(ids)
        merged = list(ids)
        for d in capsule.retrieved_doc_ids:
            if d not in seen:
                seen.add(d)
                merged.append(d)
        return merged, total + int(capsule.retrieved_docs_total)

    @staticmethod
    def _accept_fact(
        facts: list[Fact], capsule: EvidenceCapsule, source_step: int,
        slot_name: str = "",
    ) -> bool:
        f = capsule.fact
        if not f.slot_filled or not f.answer_span:
            return False
        if slot_name:
            f.slot_name = slot_name
        f.source_step = source_step
        for i, existing in enumerate(facts):
            if existing.slot_name == f.slot_name:
                facts[i] = f
                return True
        facts.append(f)
        return True

    @staticmethod
    def _final_slot_resolved(plan: list[dict], facts: list[Fact]) -> bool:
        if not plan:
            return False
        final_slot = plan[-1]["slot_name"]
        return any(
            f.slot_filled and f.slot_name == final_slot
            for f in facts
        )

    @staticmethod
    def _best_fact_answer(
        plan: list[dict], facts: list[Fact], ans_type: AnswerType,
    ) -> str:
        if not facts:
            return ""
        if plan:
            final_slot = plan[-1]["slot_name"]
            for f in facts:
                if f.slot_name == final_slot and f.answer_span:
                    return f.answer_span
        # Fallback: highest-confidence type-matching fact.
        ranked = sorted(
            facts,
            key=lambda f: (f.confidence, f.source_step, ans_type.validate_span(f.answer_span)),
            reverse=True,
        )
        for f in ranked:
            if f.answer_span:
                return f.answer_span
        return ""

    def _build_result(
        self, *, question_id: str, question: str, answer: str,
        trace: list[StepTrace], facts: list[Fact], n_calls: int,
        total_tokens: int, orch_tokens: int, agent_tokens: int,
        ret_ids: list[str], ret_total: int, route_decision: str,
        extras: dict | None = None,
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
            facts_used=facts,
            retrieved_doc_ids=ret_ids,
            retrieved_docs_total=ret_total,
            route_decision=route_decision,
            extras=extras or {},
        )
