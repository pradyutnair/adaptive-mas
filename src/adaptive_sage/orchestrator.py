"""Orchestrator for Adaptive Recursive SAGE.

The orchestrator is the central decision-maker in the adaptive pipeline.
At each step it decides whether to:

- **answer** — current facts are sufficient,
- **spawn** — delegate a missing-fact retrieval to an investigator, or
- **verify** — double-check a potentially unreliable claim.

All LLM calls use plain text generation (no tool/function calling).
Responses are parsed as JSON after stripping Qwen3-style thinking blocks.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from arag.core.config import Config
from arag.core.llm import LLMClient

from .types import Fact, StepTrace

logger = logging.getLogger(__name__)

_PLACEHOLDER_SLOT_NAMES = {"target_slot", "final_answer", "answer", "slot"}

# Maximum retries when the LLM returns malformed JSON
_MAX_JSON_RETRIES = 2

# Valid actions the orchestrator may return
_VALID_ACTIONS: set[str] = {"answer", "spawn", "verify"}
_VALID_ROUTE_ACTIONS: set[str] = {"direct_answer", "single_probe", "recurse"}

# Repair prompt sent when initial JSON parse fails
_REPAIR_PROMPT = (
    "Your previous response was not valid JSON. "
    "Please output ONLY a single JSON object with no surrounding text, "
    "no markdown code fences, and no commentary."
)

_SPAWN_ONLY_TEMPLATE = """
You must propose exactly one focused retrieval sub-question for a multi-hop QA system.

Output ONLY a JSON object:
{{"action": "spawn", "sub_question": "<focused question>", "retrieval_query": "<bridge-anchored search query>", "goal": "<what this should uncover>", "slot_name": "<which pending slot this resolves>"}}

Rules:
- Do not answer the original question.
- Do not choose verify.
- Prefer the single most useful missing fact at this point.
- The sub-question must be self-contained and specific.
- `retrieval_query` must be a concise search query, not an explanation.
- When prior facts already reveal a bridge entity, anchor `retrieval_query` on that entity plus the missing target relation/attribute.
- If no better rewrite is available, set `retrieval_query` equal to `sub_question`.
- Target one pending slot only.
- `slot_name` must copy exactly one slot name from the pending slot list.

Original question: {question}

Target profile:
{target_profile}

Facts gathered so far:
{facts}

Step history: {trace_summary}

Pending slots:
{pending_slots}

Missing reason:
{missing_reason}
"""

_DECOMPOSE_TEMPLATE = """
Decompose the multi-hop question into at most {max_subquestions} focused retrieval sub-questions.

Output ONLY a JSON object:
{{
  "sub_questions": [
    {{"sub_question": "<focused question>", "goal": "<what this should uncover>"}}
  ]
}}

Rules:
- Each sub-question must be answerable with a single retrieval step.
- Order the list so it can be executed sequentially.
- Avoid duplicates and avoid asking for the final answer directly.
- Return no more than {max_subquestions} items.

Original question: {question}

Target profile:
{target_profile}
"""


class Orchestrator:
    """LLM-based decision maker for the adaptive recursive pipeline.

    Parameters
    ----------
    config:
        Application configuration.  Must contain
        ``orchestrator.max_steps`` and ``orchestrator.max_verify_calls``.
    llm_client:
        An initialised :class:`LLMClient` for text generation.
    """

    def __init__(self, config: Config, llm_client: LLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

        self.max_steps: int = config.get("orchestrator.max_steps", 4)
        self.max_verify_calls: int = config.get("orchestrator.max_verify_calls", 1)
        self.no_verify: bool = bool(config.get("ablation.no_verify", False))
        self.route_temperature: float = float(
            config.get("adaptive.route_temperature", 0.0)
        )

        # Load prompt templates from the prompts directory
        prompts_dir = Path(__file__).parent / "prompts"
        self._route_template = (
            prompts_dir / "orchestrator_route.txt"
        ).read_text(encoding="utf-8")
        self._decide_template = (
            prompts_dir / "orchestrator_decide.txt"
        ).read_text(encoding="utf-8")
        self._verify_template = (
            prompts_dir / "orchestrator_verify.txt"
        ).read_text(encoding="utf-8")
        self._answer_template = (
            prompts_dir / "orchestrator_answer.txt"
        ).read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decide(
        self,
        question: str,
        facts: list[Fact],
        trace: list[StepTrace],
        step: int,
        target_profile: str = "",
    ) -> dict:
        """Decide the next action for the pipeline.

        Builds a prompt with the question, current facts, step-trace
        summary, and remaining step budget.  Calls the LLM, strips
        thinking tags, parses the JSON response, and validates it.

        Parameters
        ----------
        question:
            The original multi-hop question.
        facts:
            Current distilled facts from :class:`FactMemory`.
        trace:
            Step traces from previous iterations.
        step:
            Current step number (0-indexed).

        Returns
        -------
        dict
            A validated decision dict with key ``action`` in
            ``{"answer", "spawn", "verify"}``, plus optional
            ``sub_question``, ``goal``, or ``claim`` depending on the
            action.
        """
        decision, _ = await self.decide_with_usage(
            question, facts, trace, step, target_profile
        )
        return decision

    async def decide_with_usage(
        self,
        question: str,
        facts: list[Fact],
        trace: list[StepTrace],
        step: int,
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Like :meth:`decide`, but also returns token usage."""
        # Build prompt context
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
        if self.no_verify:
            user_content += (
                "\n\nAdditional rule: verify is disabled for this run. "
                "Choose only answer or spawn."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the orchestrator of a multi-hop question answering "
                    "system. Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        # Call LLM and parse JSON (with retry on parse failure)
        parsed, tokens = await self._call_and_parse_with_usage(messages)

        # --- Post-validation ---
        action: Optional[str] = parsed.get("action")

        # Force answer if step budget exhausted
        if step >= self.max_steps:
            logger.debug(
                "Step %d >= max_steps %d — forcing answer", step, self.max_steps
            )
            return {"action": "answer"}, tokens

        # Validate action field
        if action not in _VALID_ACTIONS:
            logger.warning("Invalid action %r from orchestrator, defaulting to answer", action)
            return {"action": "answer"}, tokens

        # Validate spawn fields
        if action == "spawn":
            sub_q = parsed.get("sub_question", "").strip()
            retrieval_query = str(parsed.get("retrieval_query", "")).strip()
            goal = parsed.get("goal", "").strip()
            slot_name = str(parsed.get("slot_name", "")).strip()
            if not sub_q or not goal:
                logger.warning(
                    "Spawn action missing sub_question/goal, defaulting to answer"
                )
                return {"action": "answer"}, tokens

            # Duplicate sub-question detection
            for entry in trace:
                if entry.action == "spawn" and entry.sub_question:
                    if entry.sub_question.strip().lower() == sub_q.lower():
                        logger.warning(
                            "Duplicate sub-question detected: %r — forcing answer",
                            sub_q,
                        )
                        return {"action": "answer"}, tokens

            slot_name = self._normalise_pending_slot_name(slot_name, pending_slots)
            if not retrieval_query:
                retrieval_query = sub_q

            return {
                "action": "spawn",
                "sub_question": sub_q,
                "retrieval_query": retrieval_query,
                "goal": goal,
                "slot_name": slot_name,
            }, tokens

        # Validate verify fields
        if action == "verify":
            claim = parsed.get("claim", "").strip()
            if not claim:
                logger.warning("Verify action missing claim, defaulting to answer")
                return {"action": "answer"}, tokens
            return {"action": "verify", "claim": claim}, tokens

        # action == "answer"
        return {"action": "answer"}, tokens

    async def verify_claim(self, claim: str, evidence: str) -> dict:
        """Evaluate a claim against evidence snippets.

        Parameters
        ----------
        claim:
            The claim to verify.
        evidence:
            Concatenated evidence text (typically from fact memory).

        Returns
        -------
        dict
            ``{"decision": "accept"|"reject", "reason": str}``
        """
        result, _ = await self.verify_claim_with_usage(claim, evidence)
        return result

    async def verify_claim_with_usage(self, claim: str, evidence: str) -> tuple[dict[str, Any], int]:
        """Like :meth:`verify_claim`, but also returns token usage."""
        user_content = self._verify_template.format(
            claim=claim,
            evidence=evidence or "No evidence available.",
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a claim verification assistant. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        parsed, tokens = await self._call_and_parse_with_usage(messages)

        decision = parsed.get("decision", "").strip().lower()
        reason = parsed.get("reason", "").strip()

        if decision not in {"accept", "reject"}:
            logger.warning(
                "Invalid verify decision %r, defaulting to accept", decision
            )
            decision = "accept"

        if not reason:
            reason = "No reason provided by LLM."

        return {"decision": decision, "reason": reason}, tokens

    async def route(
        self,
        question: str,
        target_profile: str = "",
    ) -> dict[str, Any]:
        """Route a question into direct-answer, single-probe, or recurse."""
        route, _ = await self.route_with_usage(question, target_profile)
        return route

    async def route_with_usage(
        self,
        question: str,
        target_profile: str = "",
    ) -> tuple[dict[str, Any], int]:
        """Like :meth:`route`, but also returns token usage."""
        user_content = self._route_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a routing controller for adaptive multi-hop QA. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            temperature=self.route_temperature,
        )
        action = str(parsed.get("action", "")).strip().lower()
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

        if action == "direct_answer" and len(required_hops) > 1:
            action = "single_probe"

        route = {
            "action": action,
            "confidence": max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0)),
            "draft_answer": str(parsed.get("draft_answer", "")).strip(),
            "sub_question": str(parsed.get("sub_question", "")).strip() or question,
            "goal": str(parsed.get("goal", "")).strip()
            or "Resolve the final answer with one grounded retrieval step.",
            "answer_type": str(parsed.get("answer_type", "")).strip() or "short factual span",
            "target_slot": target_slot,
            "required_hops": required_hops,
        }
        return route, tokens

    async def generate_answer(
        self,
        question: str,
        facts: list[Fact],
        target_profile: str = "",
    ) -> str:
        """Generate a final answer from the question and distilled facts.

        Parameters
        ----------
        question:
            The original multi-hop question.
        facts:
            All distilled facts from fact memory.

        Returns
        -------
        str
            A clean answer string (no JSON wrapper, no thinking tags).
        """
        answer, _ = await self.generate_answer_with_usage(
            question, facts, target_profile
        )
        return answer

    async def generate_answer_with_usage(
        self,
        question: str,
        facts: list[Fact],
        target_profile: str = "",
    ) -> tuple[str, int]:
        """Like :meth:`generate_answer`, but also returns token usage."""
        facts_text = self._format_facts(facts)

        user_content = self._answer_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            facts=facts_text or "No facts available.",
            pending_slots="No explicit pending slots.",
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Output ONLY the answer text, nothing else."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        response = await self.llm_client.async_chat(messages)
        content: str = response["message"].get("content", "")
        content = self._strip_thinking(content)

        # Strip any JSON wrapper the LLM might add (e.g. {"answer": "X"})
        answer = self._extract_plain_answer(content)
        return answer.strip(), self._extract_total_tokens(response)

    async def generate_answer_object_with_usage(
        self,
        question: str,
        facts: list[Fact],
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Generate a structured final answer with citations and confidence."""
        facts_text = self._format_facts(facts)
        user_content = self._answer_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            facts=facts_text or "No facts available.",
            pending_slots=self._format_pending_slots(pending_slots or [])
            or "No explicit pending slots.",
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(messages)
        cited_fact_ids = parsed.get("cited_fact_ids", [])
        if not isinstance(cited_fact_ids, list):
            cited_fact_ids = []
        normalised_missing_slot = self._normalise_pending_slot_name(
            str(parsed.get("missing_slot", "")).strip(),
            pending_slots,
        )
        return {
            "answer": str(parsed.get("answer", "")).strip(),
            "cited_fact_ids": [
                int(item) for item in cited_fact_ids
                if str(item).strip().isdigit()
            ],
            "justification_confidence": max(
                0.0,
                min(float(parsed.get("justification_confidence", 0.0)), 1.0),
            ),
            "justification": str(parsed.get("justification", "")).strip(),
            "missing_slot": normalised_missing_slot,
        }, tokens

    async def propose_spawn(
        self,
        question: str,
        facts: list[Fact],
        trace: list[StepTrace],
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
        missing_reason: str = "",
    ) -> tuple[dict[str, Any], int]:
        """Force a spawn proposal even when normal decide would answer/verify."""
        facts_text = self._format_facts(facts)
        trace_summary = self._format_trace(trace)
        user_content = _SPAWN_ONLY_TEMPLATE.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            facts=facts_text or "None yet.",
            trace_summary=trace_summary or "No steps taken yet.",
            pending_slots=self._format_pending_slots(pending_slots or [])
            or "No explicit pending slots.",
            missing_reason=missing_reason or "Retrieve the most critical missing fact.",
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an orchestrator for multi-hop QA. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(messages)
        sub_question = str(parsed.get("sub_question", "")).strip()
        retrieval_query = str(parsed.get("retrieval_query", "")).strip()
        goal = str(parsed.get("goal", "")).strip()
        slot_name = str(parsed.get("slot_name", "")).strip()
        if not sub_question:
            sub_question = question
        if not goal:
            goal = "Retrieve a missing fact needed to answer the original question."
        if not retrieval_query:
            retrieval_query = sub_question
        slot_name = self._normalise_pending_slot_name(slot_name, pending_slots)
        return {
            "action": "spawn",
            "sub_question": sub_question,
            "retrieval_query": retrieval_query,
            "goal": goal,
            "slot_name": slot_name,
        }, tokens

    async def decompose_upfront(
        self,
        question: str,
        max_subquestions: int,
        target_profile: str = "",
    ) -> tuple[list[dict[str, str]], int]:
        """Produce a sequential upfront plan of sub-questions."""
        user_content = _DECOMPOSE_TEMPLATE.format(
            question=question,
            max_subquestions=max_subquestions,
            target_profile=target_profile or "No explicit target hint available.",
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You decompose multi-hop questions into retrieval sub-questions. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(messages)
        raw_items = parsed.get("sub_questions", [])
        plan: list[dict[str, str]] = []
        seen: set[str] = set()
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                sub_question = str(item.get("sub_question", "")).strip()
                goal = str(item.get("goal", "")).strip()
                if not sub_question:
                    continue
                key = sub_question.lower()
                if key in seen:
                    continue
                seen.add(key)
                plan.append({
                    "sub_question": sub_question,
                    "goal": goal or "Retrieve a missing fact needed to answer the original question.",
                })
                if len(plan) >= max_subquestions:
                    break

        if not plan:
            plan = [{
                "sub_question": question,
                "goal": "Retrieve a missing fact needed to answer the original question.",
            }]

        return plan, tokens

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_and_parse(self, messages: list[dict]) -> dict:
        """Call the LLM, strip thinking, parse JSON — with retry on failure.

        On the first attempt, calls the LLM normally.  If JSON parsing
        fails, retries once with a repair prompt appended.
        """
        parsed, _ = await self._call_and_parse_with_usage(messages)
        return parsed

    async def _call_and_parse_with_usage(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Call the LLM, strip thinking, parse JSON, and accumulate token usage."""
        total_tokens = 0
        last_content = ""
        for attempt in range(_MAX_JSON_RETRIES + 1):
            try:
                response = await self.llm_client.async_chat(
                    messages,
                    temperature=temperature,
                )
                total_tokens += self._extract_total_tokens(response)
                last_content = response["message"].get("content", "")
                content = self._strip_thinking(last_content)
                parsed = self._parse_json(content)
                if parsed is not None:
                    return parsed, total_tokens
            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_JSON_RETRIES + 1,
                    exc,
                )

            # Append repair prompt for the retry
            if attempt < _MAX_JSON_RETRIES:
                logger.debug("Retrying with repair prompt (attempt %d)", attempt + 1)
                messages.append(
                    {"role": "assistant", "content": last_content}
                )
                messages.append({"role": "user", "content": _REPAIR_PROMPT})

        logger.error("All JSON parse attempts failed — returning fallback answer action")
        return {"action": "answer"}, total_tokens

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3-style ``<think>...</think>`` blocks."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also handle unclosed <think> blocks (thinking still in progress)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Attempt to parse a JSON object from *text*.

        Tries, in order:
        1. Direct ``json.loads`` on the whole text.
        2. Strip markdown code fences (```json ... ```) and parse.
        3. Extract the first ``{`` … ``}`` brace-delimited block and
           parse that.

        Returns ``None`` if all attempts fail.
        """
        # Try 1: direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try 2: strip markdown code fences
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
        )
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Try 3: extract brace-delimited block
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
    def _format_facts(facts: list[Fact]) -> str:
        """Format facts as a numbered string for prompt injection."""
        if not facts:
            return ""
        lines = []
        for i, fact in enumerate(facts, start=1):
            lines.append(f"{i}. {fact.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_trace(trace: list[StepTrace]) -> str:
        """Format step traces as a summary string for prompt injection."""
        if not trace:
            return ""
        parts = []
        for entry in trace:
            if entry.action == "spawn":
                parts.append(f"Step {entry.step}: spawn → {entry.sub_question}")
            elif entry.action == "verify":
                parts.append(f"Step {entry.step}: verify → {entry.claim}")
            else:
                parts.append(f"Step {entry.step}: {entry.action}")
        return "; ".join(parts)

    @staticmethod
    def _format_pending_slots(pending_slots: list[dict[str, str]]) -> str:
        """Format slot state for prompt injection."""
        if not pending_slots:
            return ""
        lines = []
        for idx, slot in enumerate(pending_slots, start=1):
            slot_name = str(slot.get("slot_name", "")).strip() or f"slot_{idx}"
            hint = str(slot.get("hint", "")).strip() or "No hint provided."
            status = "resolved" if slot.get("resolved") else "pending"
            lines.append(f"{idx}. {slot_name} [{status}] - {hint}")
        return "\n".join(lines)

    @staticmethod
    def _normalise_required_hops(raw_items: Any) -> list[dict[str, str]]:
        """Normalise required-hop slot specs returned by the router."""
        normalised: list[dict[str, str]] = []
        seen: set[str] = set()
        if not isinstance(raw_items, list):
            return normalised
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            slot_name = str(item.get("slot_name", "")).strip()
            hint = str(item.get("hint", "")).strip()
            if not slot_name or Orchestrator._is_placeholder_slot_name(slot_name):
                continue
            key = slot_name.lower()
            if key in seen:
                continue
            seen.add(key)
            normalised.append({"slot_name": slot_name, "hint": hint})
        return normalised

    @staticmethod
    def _normalise_pending_slot_name(
        slot_name: str,
        pending_slots: list[dict[str, str]] | None,
    ) -> str:
        """Return a valid pending slot name, falling back to the first pending slot."""
        cleaned = str(slot_name or "").strip()
        if Orchestrator._is_placeholder_slot_name(cleaned):
            cleaned = ""
        if not pending_slots:
            return cleaned
        valid = [
            str(item.get("slot_name", "")).strip()
            for item in pending_slots
            if str(item.get("slot_name", "")).strip()
        ]
        if cleaned in valid:
            return cleaned
        return valid[0] if valid else cleaned

    @staticmethod
    def _is_placeholder_slot_name(slot_name: str) -> bool:
        """Return whether a slot name is just a generic placeholder token."""
        cleaned = str(slot_name or "").strip().lower()
        return cleaned in _PLACEHOLDER_SLOT_NAMES

    @staticmethod
    def _extract_plain_answer(text: str) -> str:
        """Extract the plain answer text, stripping any JSON wrapper.

        If the LLM wraps the answer in JSON like ``{"answer": "X"}``,
        this extracts just ``X``.  Otherwise returns the text as-is.
        """
        # Try to parse as JSON and extract an "answer" key
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "answer" in parsed:
                return str(parsed["answer"])
        except json.JSONDecodeError:
            pass

        # Try to extract from code fence
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
        )
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, dict) and "answer" in parsed:
                    return str(parsed["answer"])
            except json.JSONDecodeError:
                pass

        # Try to find {"answer": "..."} pattern anywhere
        m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            return m.group(1).replace('\\"', '"')

        # Return as-is — it's already a plain answer
        return text

    @staticmethod
    def _extract_total_tokens(response: dict[str, Any]) -> int:
        """Extract total tokens from a chat response."""
        raw_usage = response.get("raw_response", {}).get("usage", {}) or {}
        total_tokens = raw_usage.get("total_tokens")
        if total_tokens is not None:
            return int(total_tokens)
        return int(response.get("input_tokens", 0)) + int(response.get("output_tokens", 0))
