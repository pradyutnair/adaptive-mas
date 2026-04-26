"""Orchestrator: tool-using lead agent + planner + slot refiner.

Roles (each may use a different LLMClient):
- ``direct_solve(question)`` — tool-using loop with a ``search`` tool. Tries
  to answer easy questions in <= ``direct_max_searches`` searches. If it
  cannot ground a confident, type-correct, cited answer, it ESCALATES.
- ``plan(question, reason)`` — only called after escalation. Returns a slot
  DAG with ``dependency_group``s (parallel-aware).
- ``refine_slot(slot, failed_queries, facts)`` — called when a subagent
  failed to ground a slot. Rewrites the sub-question and retrieval query
  so the next subagent attempt is genuinely different.

There is NO synthesize step; the pipeline reads the final-slot ``answer_span``
deterministically.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .llm import LLMClient, parse_json_object, strip_thinking
from .retriever import RetrievalHit, Retriever
from .types import AnswerType, Fact

logger = logging.getLogger(__name__)


_PLACEHOLDER_BAD = {
    "target_slot", "final_answer", "answer", "slot", "result", "output", "",
}


class Orchestrator:
    """Tool-using lead agent + planner + slot refiner."""

    def __init__(
        self,
        direct_llm: LLMClient,
        plan_llm: LLMClient,
        refine_llm: LLMClient,
        retriever: Retriever,
        direct_max_searches: int = 2,
        direct_top_k: int = 10,
        max_plan_hops: int = 6,
    ) -> None:
        self.direct_llm = direct_llm
        self.plan_llm = plan_llm
        self.refine_llm = refine_llm
        self.retriever = retriever
        self.direct_max_searches = int(direct_max_searches)
        self.direct_top_k = int(direct_top_k)
        self.max_plan_hops = int(max_plan_hops)

        prompts = Path(__file__).parent / "prompts"
        self._direct_tpl = (prompts / "direct_solve.txt").read_text(encoding="utf-8")
        self._plan_tpl = (prompts / "plan.txt").read_text(encoding="utf-8")
        self._refine_tpl = (prompts / "refine.txt").read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Direct solve (tool-using)
    # ------------------------------------------------------------------

    async def direct_solve(self, question: str) -> tuple[dict[str, Any], int, list[str]]:
        """ReAct-style loop: ``search`` / ``final`` / ``escalate`` JSON actions."""
        prompt = self._direct_tpl.format(
            question=question,
            max_searches=self.direct_max_searches,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        retrieved_ids: list[str] = []
        total_tokens = 0
        searches_used = 0

        for _turn in range(self.direct_max_searches + 2):
            resp = await self.direct_llm.chat(messages=messages)
            total_tokens += resp.total_tokens
            content = strip_thinking(resp.content)
            parsed = parse_json_object(content)
            action = str(parsed.get("action", "")).strip().lower()

            if action == "search" and searches_used < self.direct_max_searches:
                messages.append({"role": "assistant", "content": content})
                hits, ids = await self._run_search(parsed)
                for cid in ids:
                    if cid not in retrieved_ids:
                        retrieved_ids.append(cid)
                messages.append({
                    "role": "user",
                    "content": json.dumps({"search_result": [
                        {"chunk_id": h.chunk_id, "score": round(h.score, 4), "text": h.text}
                        for h in hits
                    ]}),
                })
                searches_used += 1
                continue

            if action == "escalate":
                return ({"action": "escalate",
                         "reason": str(parsed.get("reason", "")).strip() or "model_escalated"},
                        total_tokens, retrieved_ids)

            if action == "final":
                return self._normalise_direct_result(parsed, retrieved_ids), total_tokens, retrieved_ids

            # Bad turn: nudge once then continue.
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "system_note": (
                        "Output exactly one JSON object with action 'search', "
                        "'final', or 'escalate'. "
                        f"You have used {searches_used}/{self.direct_max_searches} searches."
                    ),
                }),
            })

        return ({"action": "escalate", "reason": "loop_exhausted"},
                total_tokens, retrieved_ids)

    @staticmethod
    def _normalise_direct_result(parsed: dict, retrieved_ids: list[str]) -> dict[str, Any]:
        action = str(parsed.get("action", "")).strip().lower()
        if action == "escalate":
            return {
                "action": "escalate",
                "reason": str(parsed.get("reason", "")).strip() or "model_escalated",
            }
        if action != "answer":
            return {"action": "escalate", "reason": "no_action_emitted"}

        answer = str(parsed.get("answer_span", parsed.get("answer", ""))).strip()
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        support_raw = parsed.get("support_ids") or []
        rid_set = set(retrieved_ids)
        support_ids = [
            str(s).strip() for s in support_raw
            if str(s).strip() and str(s).strip() in rid_set
        ]
        if not answer or not support_ids:
            return {"action": "escalate", "reason": "ungrounded_answer"}

        ans_type = AnswerType.coerce(parsed.get("answer_type", "entity"))
        if not ans_type.validate_span(answer):
            return {"action": "escalate", "reason": "answer_type_mismatch"}

        return {
            "action": "answer",
            "answer_span": answer,
            "answer_type": ans_type.value,
            "justification": str(parsed.get("justification", "")).strip(),
            "confidence": confidence,
            "support_ids": support_ids,
        }

    # ------------------------------------------------------------------
    # Plan (decompose) — no tools
    # ------------------------------------------------------------------

    async def plan(
        self, question: str, reason: str = "",
    ) -> tuple[dict[str, Any], int]:
        """Plan a slot DAG. Retries once with a corrective hint if empty."""
        prompt = self._plan_tpl.format(
            question=question,
            reason=reason or "needs_decomposition",
            max_hops=self.max_plan_hops,
        )
        messages = [{"role": "user", "content": prompt}]
        total_tokens = 0
        plan: list[dict] = []
        answer_type = AnswerType.ENTITY
        for attempt in range(2):
            resp = await self.plan_llm.chat(messages=messages)
            total_tokens += resp.total_tokens
            cleaned = strip_thinking(resp.content)
            parsed = parse_json_object(cleaned)
            answer_type = AnswerType.coerce(parsed.get("answer_type", "entity"))
            plan = self._normalise_plan(parsed.get("plan", []), answer_type.value)
            if len(plan) >= 2:
                break
            # Empty / single-hop plan: nudge once and retry.
            messages.append({"role": "assistant", "content": cleaned[-2000:]})
            messages.append({"role": "user", "content": (
                "Your previous response did not contain a valid multi-hop plan."
                " Output ONLY the JSON object with at least 2 hops, no <think>"
                " tags, no prose. The LAST hop's expected_answer_type must equal"
                " the top-level answer_type."
            )})
        return {"answer_type": answer_type.value, "plan": plan}, total_tokens

    # ------------------------------------------------------------------
    # Slot refinement — no tools
    # ------------------------------------------------------------------

    async def refine_slot(
        self,
        slot_name: str,
        expected_answer_type: str,
        original_sub_question: str,
        failed_queries: list[str],
        facts: list[Fact],
    ) -> tuple[dict[str, Any], int]:
        prompt = self._refine_tpl.format(
            slot_name=slot_name,
            expected_answer_type=expected_answer_type,
            original_sub_question=original_sub_question,
            failed_queries=("\n".join(f"- {q}" for q in failed_queries)
                            or "(none)"),
            facts=self._format_facts(facts),
        )
        resp = await self.refine_llm.chat(messages=[{"role": "user", "content": prompt}])
        parsed = parse_json_object(strip_thinking(resp.content))
        return {
            "sub_question": str(parsed.get("sub_question", "")).strip(),
            "retrieval_query": str(parsed.get("retrieval_query", "")).strip(),
        }, resp.total_tokens

    # ------------------------------------------------------------------
    # Tool execution (used by direct_solve)
    # ------------------------------------------------------------------

    async def _run_search(self, parsed: dict) -> tuple[list[RetrievalHit], list[str]]:
        query = str(parsed.get("query", "")).strip()
        if not query:
            return [], []
        try:
            top_k = max(1, min(int(parsed.get("top_k") or self.direct_top_k), 20))
        except (TypeError, ValueError):
            top_k = self.direct_top_k
        hits = await self.retriever.retrieve(query, top_k=top_k)
        return hits, [h.chunk_id for h in hits]

    # ------------------------------------------------------------------
    # Formatting helpers (no chunk text ever)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_facts(facts: list[Fact]) -> str:
        if not facts:
            return "(none yet)"
        lines = []
        for f in facts:
            sup = ",".join(f.support_ids) if f.support_ids else "none"
            lines.append(
                f"  - slot={f.slot_name} answer={f.answer_span!r} "
                f"conf={f.confidence:.2f} support=[{sup}] | {f.text}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plan normalisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def _normalise_plan(cls, raw: Any, default_type: str) -> list[dict]:
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            slot = str(item.get("slot_name", "")).strip()
            if slot.lower() in _PLACEHOLDER_BAD:
                continue
            try:
                dep_group = int(item.get("dependency_group", 0))
            except (TypeError, ValueError):
                dep_group = 0
            out.append({
                "slot_name": slot,
                "sub_question": str(item.get("sub_question", "")).strip(),
                "retrieval_query": str(item.get("retrieval_query", "")).strip(),
                "expected_answer_type": AnswerType.coerce(
                    item.get("expected_answer_type", default_type)
                ).value,
                "dependencies": [
                    str(d).strip() for d in (item.get("dependencies") or [])
                    if str(d).strip()
                ],
                "dependency_group": dep_group,
            })
        return out

    @staticmethod
    def substitute_placeholders(text: str, slot_values: dict[str, str]) -> str:
        if not text:
            return text
        out = text
        for slot, val in slot_values.items():
            if not slot or not val:
                continue
            out = out.replace("{{" + slot + "}}", val)
            out = out.replace("{" + slot + "}", val)
        return out

    @staticmethod
    def has_unresolved_placeholder(text: str) -> bool:
        return bool(re.search(r"\{\{[^}]+\}\}", text))
