"""Orchestrator for adaptive RAG with probe-assess-decide loop.

Four LLM calls:
1. route — slot DAG (required_hops) + answer type
2. assess_probe — is the probe answer sufficient?
3. decide — given current facts + pending slots, what next? (loop)
4. generate_answer_object — synthesize final answer from capsules
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from arag.core.config import Config
from arag.core.llm import LLMClient

from .retrieval import EvidenceReader
from .types import EvidenceCapsule, Fact, StepTrace

logger = logging.getLogger(__name__)

_MAX_JSON_RETRIES = 2

_VALID_ROUTE_ACTIONS = {"single_probe", "sequential", "parallel"}
_VALID_DECIDE_ACTIONS = {"answer", "spawn", "refine"}


class Orchestrator:
    """LLM-based orchestrator: route, assess, decide, synthesize."""

    def __init__(self, config: Config, llm_client: LLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self.route_temperature: float = float(
            config.get("adaptive.route_temperature", 0.0)
        )
        self.max_steps: int = int(config.get("orchestrator.max_steps", 4))
        self.no_verify: bool = bool(config.get("ablation.no_verify", True))
        self.reader = EvidenceReader(config, llm_client)

        prompts_dir = Path(__file__).parent / "prompts"
        self._route_template = (
            prompts_dir / "orchestrator_route.txt"
        ).read_text(encoding="utf-8")
        self._probe_gate_template = (
            prompts_dir / "orchestrator_probe_gate.txt"
        ).read_text(encoding="utf-8")
        self._decide_template = (
            prompts_dir / "orchestrator_decide.txt"
        ).read_text(encoding="utf-8")
        self._answer_template = (
            prompts_dir / "orchestrator_answer.txt"
        ).read_text(encoding="utf-8")
        self._route_system = (
            prompts_dir / "orchestrator_route_system.txt"
        ).read_text(encoding="utf-8")
        self._probe_gate_system = (
            prompts_dir / "orchestrator_probe_gate_system.txt"
        ).read_text(encoding="utf-8")
        self._decide_system = (
            prompts_dir / "orchestrator_decide_system.txt"
        ).read_text(encoding="utf-8")
        self._answer_system = (
            prompts_dir / "orchestrator_answer_system.txt"
        ).read_text(encoding="utf-8")
        self._repair_prompt = (
            prompts_dir / "json_repair.txt"
        ).read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # 1. Route: slot DAG + answer type (1 LLM call)
    # ------------------------------------------------------------------

    async def route_with_usage(
        self,
        question: str,
        target_profile: str = "",
    ) -> tuple[dict[str, Any], int]:
        """Generate slot DAG (required_hops) and answer type."""
        user_content = self._route_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
        )
        messages = [
            {"role": "system", "content": self._route_system},
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(
            messages, temperature=self.route_temperature,
        )
        action = str(parsed.get("action", "")).strip().lower()
        if action == "direct_answer":
            action = "single_probe"
        if action not in _VALID_ROUTE_ACTIONS:
            action = "single_probe"

        required_hops = self._normalise_required_hops(parsed.get("required_hops", []))
        parsed_target_slot = str(parsed.get("target_slot", "")).strip()
        if self._is_placeholder_slot_name(parsed_target_slot):
            parsed_target_slot = ""
        if required_hops:
            target_slot = str(required_hops[-1].get("slot_name", "")).strip() or (
                parsed_target_slot or "final_answer"
            )
        else:
            target_slot = parsed_target_slot or "final_answer"
            required_hops = [{"slot_name": target_slot, "hint": target_profile.strip()}]

        return {
            "action": action,
            "confidence": max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0)),
            "answer_type": self._canonical_answer_type(
                parsed.get("answer_type", "entity")
            ),
            "sub_question": str(parsed.get("sub_question", "")).strip() or question,
            "retrieval_query": str(parsed.get("retrieval_query", "")).strip()
            or str(parsed.get("sub_question", "")).strip()
            or question,
            "goal": str(parsed.get("goal", "")).strip()
            or "Resolve the final answer with one grounded retrieval step.",
            "target_slot": target_slot,
            "required_hops": required_hops,
        }, tokens

    async def retrieve_and_distill_with_usage(
        self,
        *,
        question: str,
        retrieval_query: str,
        target_profile: str,
        answer_type: str,
        top_k: int | None = None,
    ) -> tuple[EvidenceCapsule, int]:
        """Direct-path retrieval owned by the orchestrator."""
        return await self.reader.retrieve_and_distill(
            sub_question=question,
            retrieval_query=retrieval_query or question,
            goal="Resolve the final answer directly with semantic retrieval.",
            slot_name="final_answer",
            slot_hint=f"{target_profile}\nExpected answer type: {answer_type}.",
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # 2. Assess probe sufficiency (1 LLM call)
    # ------------------------------------------------------------------

    async def assess_probe_with_usage(
        self,
        question: str,
        facts: list[Fact],
        proposed_answer: str,
        target_profile: str = "",
        trace: list[StepTrace] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Return a sufficiency score for the current evidence state."""
        facts_text = self._format_facts(facts)
        hop_chain = self._format_hop_chain(trace or [], facts)
        user_content = self._probe_gate_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            proposed_answer=proposed_answer.strip() or "EMPTY",
            facts=facts_text or "No facts available.",
            hop_chain=hop_chain or "No grounded hop chain available.",
        )
        messages = [
            {"role": "system", "content": self._probe_gate_system},
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(messages)
        sufficient = parsed.get("sufficient", parsed.get("confidence", 0.0))
        try:
            sufficient = float(sufficient)
        except (TypeError, ValueError):
            sufficient = 0.0
        sufficient = max(0.0, min(sufficient, 1.0))
        return {
            "sufficient": sufficient,
            "reason": str(parsed.get("reason", "")).strip(),
        }, tokens

    # ------------------------------------------------------------------
    # 3. Decide: what to do next (1 LLM call per step in loop)
    # ------------------------------------------------------------------

    async def decide_with_usage(
        self,
        question: str,
        facts: list[Fact],
        trace: list[StepTrace],
        step: int,
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Decide next action: answer, spawn, or refine."""
        facts_text = self._format_facts(facts)
        trace_summary = self._format_trace(trace)
        remaining = max(self.max_steps - step, 0)
        pending_slots_text = self._format_pending_slots(pending_slots or [])

        user_content = self._decide_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            facts=facts_text or "None yet.",
            trace_summary=trace_summary or "No steps taken yet.",
            remaining_steps=remaining,
            pending_slots=pending_slots_text or "No explicit pending slots.",
        )
        messages = [
            {"role": "system", "content": self._decide_system},
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(messages)
        action = parsed.get("action")

        if step >= self.max_steps:
            return {"action": "answer"}, tokens

        if action not in _VALID_DECIDE_ACTIONS:
            return {"action": "answer"}, tokens

        if action in {"spawn", "refine"}:
            sub_q = str(parsed.get("sub_question", "")).strip()
            goal = str(parsed.get("goal", "")).strip()
            if not sub_q or not goal:
                return {"action": "answer"}, tokens

            seen_count = 0
            for entry in trace:
                if entry.action in {"spawn", "refine"} and entry.sub_question:
                    if entry.sub_question.strip().lower() == sub_q.lower():
                        seen_count += 1
            if seen_count >= 2:
                return {"action": "answer"}, tokens

            return {
                "action": action,
                "sub_question": sub_q,
                "retrieval_query": str(parsed.get("retrieval_query", "")).strip() or sub_q,
                "goal": goal,
                "slot_name": str(parsed.get("slot_name", "")).strip(),
            }, tokens

        return {"action": "answer"}, tokens

    # ------------------------------------------------------------------
    # 4. Synthesize: final answer from capsules (1 LLM call)
    # ------------------------------------------------------------------

    async def generate_answer_object_with_usage(
        self,
        question: str,
        facts: list[Fact],
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
        trace: list[StepTrace] | None = None,
        route_draft_answer: str = "",
    ) -> tuple[dict[str, Any], int]:
        """Synthesize a structured final answer from gathered facts."""
        facts_text = self._format_facts(facts)
        hop_chain = self._format_hop_chain(trace or [], facts)
        user_content = self._answer_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            facts=facts_text or "No facts available.",
            pending_slots=self._format_pending_slots(pending_slots or [])
            or "No explicit pending slots.",
            hop_chain=hop_chain or "No grounded hop chain available.",
        )
        messages = [
            {"role": "system", "content": self._answer_system},
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(messages, max_tokens=256)
        return parsed, tokens

    # ------------------------------------------------------------------
    # LLM call infrastructure
    # ------------------------------------------------------------------

    async def _call_and_parse_with_usage(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self.llm_client.async_chat(messages, **kwargs)
        total_tokens = self._extract_total_tokens(response)
        content: str = response["message"].get("content", "")
        content = self._strip_thinking(content)
        parsed = self._parse_json(content)

        if parsed is None:
            messages_retry = list(messages) + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": self._repair_prompt},
            ]
            for _ in range(_MAX_JSON_RETRIES):
                response = await self.llm_client.async_chat(messages_retry, **kwargs)
                total_tokens += self._extract_total_tokens(response)
                content = self._strip_thinking(
                    response["message"].get("content", "")
                )
                parsed = self._parse_json(content)
                if parsed is not None:
                    break
                messages_retry.append({"role": "assistant", "content": content})
                messages_retry.append({"role": "user", "content": self._repair_prompt})

        if parsed is None:
            parsed = {}
        return parsed, total_tokens

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_facts(facts: list[Fact]) -> str:
        if not facts:
            return ""
        lines: list[str] = []
        for i, fact in enumerate(facts, start=1):
            answer_span = str(getattr(fact, "answer_span", "")).strip()
            slot_name = str(getattr(fact, "slot_name", "")).strip()
            support = ", ".join(fact.support_ids) if fact.support_ids else "none"
            slot_label = f" [slot={slot_name}]" if slot_name else ""
            answer_label = f" answer_span={answer_span}" if answer_span else ""
            lines.append(
                f"{i}. (conf={fact.confidence:.2f}, support=[{support}]{slot_label}{answer_label}) {fact.text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_pending_slots(pending_slots: list[dict[str, str]]) -> str:
        if not pending_slots:
            return ""
        lines: list[str] = []
        for slot in pending_slots:
            name = str(slot.get("slot_name", ""))
            hint = str(slot.get("hint", ""))
            resolved = "resolved" if slot.get("resolved") else "pending"
            lines.append(f"- {name}: {hint} ({resolved})")
        return "\n".join(lines)

    @staticmethod
    def _format_trace(trace: list[StepTrace]) -> str:
        if not trace:
            return ""
        lines: list[str] = []
        for entry in trace:
            slot = f" [slot={entry.slot_name}]" if entry.slot_name else ""
            if entry.action in {"spawn", "refine"}:
                if entry.fact_added:
                    conf = entry.justification_confidence or 0.0
                    meta = entry.metadata or {}
                    lines.append(
                        f"Step {entry.step}: {entry.action}{slot} — {entry.sub_question or '?'} → FOUND (conf={conf:.2f})"
                    )
                else:
                    lines.append(
                        f"Step {entry.step}: {entry.action}{slot} — {entry.sub_question or '?'} → FAILED (no evidence found, do NOT retry this query)"
                    )
            elif entry.action == "assess":
                conf = entry.justification_confidence or 0.0
                lines.append(f"Step {entry.step}: assess — conf={conf:.2f}")
            elif entry.action == "answer":
                lines.append(f"Step {entry.step}: answer")
            else:
                lines.append(f"Step {entry.step}: {entry.action}")
        return "\n".join(lines)

    @staticmethod
    def _format_hop_chain(trace: list[StepTrace], facts: list[Fact]) -> str:
        if not trace:
            return ""
        facts_by_step: dict[int, list[Fact]] = {}
        for fact in facts:
            facts_by_step.setdefault(int(fact.source_step), []).append(fact)
        lines: list[str] = []
        hop_idx = 1
        for entry in trace:
            if entry.action not in {"spawn", "refine"} or not entry.sub_question:
                continue
            step_facts = facts_by_step.get(entry.step, [])
            best_fact = ""
            if step_facts:
                if entry.slot_name:
                    matching = [
                        f for f in step_facts
                        if str(getattr(f, "slot_name", "")).strip() == str(entry.slot_name or "").strip()
                    ]
                    if matching:
                        step_facts = matching
                ranked = sorted(
                    step_facts,
                    key=lambda f: (f.confidence, bool(f.answer_span.strip())),
                    reverse=True,
                )
                best_fact = ranked[0].answer_span.strip() or ranked[0].text.strip()
            slot_label = f" [slot={entry.slot_name}]" if entry.slot_name else ""
            if best_fact:
                lines.append(f"Hop {hop_idx}{slot_label}: {entry.sub_question} -> found {best_fact}")
            else:
                lines.append(f"Hop {hop_idx}{slot_label}: {entry.sub_question} -> no grounded fact")
            hop_idx += 1
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_thinking(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned if cleaned else text.strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                result = json.loads(brace_match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _extract_total_tokens(response: dict[str, Any]) -> int:
        raw_usage = response.get("raw_response", {}).get("usage", {}) or {}
        total_tokens = raw_usage.get("total_tokens")
        if total_tokens is not None:
            return int(total_tokens)
        return int(response.get("input_tokens", 0)) + int(response.get("output_tokens", 0))

    @staticmethod
    def _canonical_answer_type(value: Any) -> str:
        cleaned = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "location": "place", "loc": "place", "country": "place", "city": "place",
            "boolean": "yes_no", "bool": "yes_no", "yes/no": "yes_no",
            "year": "date", "time": "date",
            "numeric": "number", "count": "number",
            "short factual span": "entity",
        }
        cleaned = aliases.get(cleaned, cleaned)
        return cleaned if cleaned in {"person", "place", "date", "number", "yes_no", "entity", "other"} else "entity"

    @staticmethod
    def _looks_meta_answer(text: str) -> bool:
        cleaned = str(text or "").strip().lower()
        if not cleaned:
            return True
        bad_markers = (
            "cannot be answered", "cannot be determined", "cannot determine",
            "not enough information", "insufficient information",
            "provided facts", "given facts",
            "do not provide enough", "do not contain enough",
            "unknown", "not specified", "the question", "the facts",
        )
        return any(m in cleaned for m in bad_markers)

    # ------------------------------------------------------------------
    # Required hops normalisation
    # ------------------------------------------------------------------

    def _normalise_required_hops(self, raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        hops: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("slot_name", "")).strip()
            if not name or self._is_placeholder_slot_name(name):
                continue
            hops.append({
                "slot_name": name,
                "sub_question": str(item.get("sub_question", "")).strip(),
                "retrieval_query": str(item.get("retrieval_query", "")).strip(),
                "hint": str(item.get("hint", "")).strip(),
                "expected_answer_type": self._canonical_answer_type(
                    item.get("expected_answer_type", "entity")
                ),
                "dependencies": list(item.get("dependencies") or []),
                "dependency_group": item.get("dependency_group", 0),
                "resolved": False,
            })
        return hops

    @staticmethod
    def _is_placeholder_slot_name(name: str) -> bool:
        return name.lower() in {
            "target_slot", "final_answer", "answer", "slot",
            "result", "target", "output", "",
        }
