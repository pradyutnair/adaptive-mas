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

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from arag.core.config import Config
from arag.core.llm import LLMClient, TokenBudgetExceededError

from .types import Fact, StepTrace

logger = logging.getLogger(__name__)

_PLACEHOLDER_SLOT_NAMES = {"target_slot", "final_answer", "answer", "slot"}

# Maximum retries when the LLM returns malformed JSON
_MAX_JSON_RETRIES = 2

# Valid actions the orchestrator may return
_VALID_ACTIONS: set[str] = {"answer", "spawn", "refine", "verify"}
_VALID_ROUTE_ACTIONS: set[str] = {"single_probe", "recurse"}
_VALID_EXECUTION_MODES: set[str] = {
    "direct_probe",
    "typed_plan_exec",
    "recursive_recovery",
}

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
- If the target slot depends on an entity that is still unknown, ask for that entity before asking for its attribute.
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

_SLOT_DAG_DECOMPOSE_TEMPLATE = """
Decompose the question into a typed slot DAG for adaptive multi-hop retrieval.

Output ONLY a JSON object:
{{
  "required_hops": [
    {{
      "slot_name": "<concise semantic slot name>",
      "hint": "<what this slot must resolve>",
      "expected_info_type": "<specific type tag>",
      "dependency_group": <integer>,
      "sub_question": "<focused one-hop question>",
      "retrieval_query": "<short search query>",
      "goal": "<what evidence this slot should uncover>"
    }}
  ]
}}

Rules:
- Return at most {max_subquestions} slots.
- The last slot must be the final answer slot.
- Use the same dependency_group only for slots that can be retrieved independently.
- If a slot needs the answer from an earlier slot, put it in a later dependency_group.
- Each sub_question must be answerable with one retrieval step.
- Do not collapse a relation-of-relation question into one final lookup.
- Use semantic slot names; never use placeholders such as target_slot, final_answer, answer, or slot.
- Keep retrieval_query short and search-style.
- Do not include text outside the JSON object.

Original question: {question}

Target profile:
{target_profile}
"""

_INNERMOST_TEMPLATE = """
Rewrite the question into exactly one innermost retrieval sub-question for the next search step.

Output ONLY a JSON object:
{{"sub_question": "<focused question>", "retrieval_query": "<concise search query>", "goal": "<what this should uncover>", "slot_name": "<which pending slot this resolves>"}}

Rules:
- The new sub-question must be strictly narrower than the original question.
- It must target the earliest unresolved bridge entity or attribute needed for the answer.
- If an attribute depends on an unknown entity, ask for the entity first.
- It must be answerable in a single retrieval step.
- Do not repeat the original question or a near-paraphrase of it.
- Prefer grounding on the first unresolved slot in dependency order.
- `retrieval_query` must be short and bridge-anchored.
- If no better rewrite is available, make the narrowest possible first bridge-fact question.

Original question: {question}

Candidate sub-question:
{candidate_sub_question}

Target profile:
{target_profile}

Facts gathered so far:
{facts}

Step history:
{trace_summary}

Pending slots:
{pending_slots}

Missing reason:
{missing_reason}
"""


def _normalise_expected_info_type(slot_name: str, raw_type: str) -> str:
    """Prefer slot-precise type tags over broad router labels."""
    slot_key = str(slot_name or "").strip().lower().replace(" ", "_")
    type_key = str(raw_type or "").strip().lower().replace(" ", "_")
    generic_types = {
        "",
        "other",
        "entity",
        "entities",
        "thing",
        "value",
        "span",
        "string",
        "text",
        "short_factual_span",
        "short_span",
        "location",
        "place",
        "organization",
        "org",
        "institution",
        "number",
    }
    if slot_key and type_key in generic_types:
        return slot_key
    return type_key or slot_key or "other"


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
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0

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
        self._probe_gate_template = (
            prompts_dir / "orchestrator_probe_gate.txt"
        ).read_text(encoding="utf-8")
        self._probe_state_template = (
            prompts_dir / "orchestrator_probe_state.txt"
        ).read_text(encoding="utf-8")

    def reset_usage_totals(self) -> None:
        """Reset per-run prompt/completion token counters."""
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0

    def get_usage_totals(self) -> dict[str, int]:
        """Return per-run prompt/completion token totals."""
        return {
            "prompt_tokens": self._prompt_tokens_total,
            "completion_tokens": self._completion_tokens_total,
        }

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
        remaining_total_tokens: int | None = None,
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
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
        )

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
        if action in {"spawn", "refine"}:
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
                if entry.action in {"spawn", "refine"} and entry.sub_question:
                    same_sub_question = entry.sub_question.strip().lower() == sub_q.lower()
                    same_slot = (
                        not slot_name
                        or str(entry.slot_name or "").strip() == slot_name
                    )
                    if same_sub_question and same_slot and entry.fact_added:
                        logger.warning(
                            "Duplicate grounded sub-question detected: %r — forcing answer",
                            sub_q,
                        )
                        return {"action": "answer"}, tokens

            normalised_spawn, extra_tokens = await self._normalise_spawn_decision(
                question=question,
                sub_question=sub_q,
                retrieval_query=retrieval_query,
                goal=goal,
                slot_name=slot_name,
                facts=facts,
                trace=trace,
                target_profile=target_profile,
                pending_slots=pending_slots,
                missing_reason="Retrieve the next missing fact.",
                remaining_total_tokens=(
                    None
                    if remaining_total_tokens is None
                    else max(int(remaining_total_tokens) - tokens, 0)
                ),
            )
            return {
                "action": action,
                **normalised_spawn,
            }, tokens + extra_tokens

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

    async def verify_claim_with_usage(
        self,
        claim: str,
        evidence: str,
        remaining_total_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
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

        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
        )

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
        """Route a question into single-probe or recurse."""
        route, _ = await self.route_with_usage(question, target_profile)
        return route

    async def route_with_usage(
        self,
        question: str,
        target_profile: str = "",
        remaining_total_tokens: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tokens: int | None = None,
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
            remaining_total_tokens=remaining_total_tokens,
            chat_template_kwargs=chat_template_kwargs,
            max_tokens=max_tokens,
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
            required_hops = [
                {
                    "slot_name": target_slot,
                    "hint": target_profile.strip(),
                    "expected_info_type": _normalise_expected_info_type(
                        target_slot,
                        str(parsed.get("answer_type", "")).strip(),
                    ),
                }
            ]

        expected_hop_count = parsed.get("expected_hop_count", len(required_hops))
        try:
            expected_hop_count = max(1, int(expected_hop_count))
        except (TypeError, ValueError):
            expected_hop_count = max(1, len(required_hops))

        execution_mode = str(parsed.get("execution_mode", "")).strip().lower()
        if execution_mode not in _VALID_EXECUTION_MODES:
            if action == "single_probe":
                execution_mode = "direct_probe"
            elif expected_hop_count > 1:
                execution_mode = "typed_plan_exec"
            else:
                execution_mode = "direct_probe"

        def _score(key: str, fallback: float) -> float:
            value = parsed.get(key, fallback)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = fallback
            return max(0.0, min(value, 1.0))

        compositionality_default = 0.8 if expected_hop_count > 1 else 0.2
        bridge_uncertainty_default = (
            0.75 if execution_mode == "recursive_recovery" else 0.3
        )

        route = {
            "action": action,
            "confidence": max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0)),
            "draft_answer": str(parsed.get("draft_answer", "")).strip(),
            "sub_question": str(parsed.get("sub_question", "")).strip() or question,
            "retrieval_query": str(parsed.get("retrieval_query", "")).strip()
            or str(parsed.get("sub_question", "")).strip()
            or question,
            "goal": str(parsed.get("goal", "")).strip()
            or "Resolve the final answer with one grounded retrieval step.",
            "answer_type": str(parsed.get("answer_type", "")).strip() or "short factual span",
            "target_slot": target_slot,
            "required_hops": required_hops,
            "execution_mode": execution_mode,
            "compositionality_score": _score(
                "compositionality_score",
                compositionality_default,
            ),
            "bridge_uncertainty_score": _score(
                "bridge_uncertainty_score",
                bridge_uncertainty_default,
            ),
            "expected_hop_count": expected_hop_count,
        }
        return route, tokens

    async def assess_typed_probe_state_with_usage(
        self,
        *,
        question: str,
        facts: list[Fact],
        proposed_answer: str,
        probe_question: str,
        probe_strategy: str,
        probe_slot_name: str,
        probe_slot_hint: str,
        probe_expected_info_type: str,
        probe_slot_value: str,
        target_profile: str = "",
        pending_slots: list[dict[str, Any]] | None = None,
        resolved_slots: list[str] | None = None,
        trace: list[StepTrace] | None = None,
        remaining_total_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Score both slot resolution and final-answer sufficiency after a probe."""
        facts_text = self._format_facts(facts)
        hop_chain = self._format_hop_chain(trace or [], facts)
        user_content = self._probe_state_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            probe_question=probe_question.strip() or question,
            probe_strategy=probe_strategy.strip() or "typed_probe",
            probe_slot_name=probe_slot_name.strip() or "final_answer",
            probe_slot_hint=probe_slot_hint.strip() or "No extra slot hint available.",
            probe_expected_info_type=probe_expected_info_type.strip() or "other",
            probe_slot_value=probe_slot_value.strip() or "EMPTY",
            proposed_answer=proposed_answer.strip() or "EMPTY",
            pending_slots=self._format_pending_slots(pending_slots or [])
            or "No explicit pending slots.",
            resolved_slots=", ".join(resolved_slots or []) or "None",
            facts=facts_text or "No facts available.",
            hop_chain=hop_chain or "No grounded hop chain available.",
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an adaptive QA controller. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
        )

        def _score(key: str, fallback: float = 0.0) -> float:
            value = parsed.get(key, fallback)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = fallback
            return max(0.0, min(value, 1.0))

        slot_sufficient = _score("slot_sufficient")
        answer_sufficient = _score(
            "answer_sufficient",
            slot_sufficient if proposed_answer.strip() else 0.0,
        )
        return {
            "slot_sufficient": slot_sufficient,
            "answer_sufficient": answer_sufficient,
            "slot_reason": str(parsed.get("slot_reason", "")).strip(),
            "answer_reason": str(parsed.get("answer_reason", "")).strip(),
        }, tokens

    async def assess_probe_sufficiency_with_usage(
        self,
        question: str,
        facts: list[Fact],
        proposed_answer: str,
        target_profile: str = "",
        trace: list[StepTrace] | None = None,
        remaining_total_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Return a sufficiency probability for the current evidence state.

        The prompt schema is `{sufficient: float, reason: str}` only. The
        controller (pipeline) compares the score to a global threshold; the
        LLM never selects an action label here.
        """
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
            {
                "role": "system",
                "content": (
                    "You are an adaptive QA controller. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
        )
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

    async def assess_probe_with_usage(
        self,
        question: str,
        facts: list[Fact],
        proposed_answer: str,
        target_profile: str = "",
        trace: list[StepTrace] | None = None,
        remaining_total_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Backward-compatible probe-gate wrapper used by legacy paths.

        Returns an `action` derived from the new `sufficient` score so the
        legacy routed controller still has a discrete decision to consume.
        """
        result, tokens = await self.assess_probe_sufficiency_with_usage(
            question=question,
            facts=facts,
            proposed_answer=proposed_answer,
            target_profile=target_profile,
            trace=trace,
            remaining_total_tokens=remaining_total_tokens,
        )
        sufficient = float(result["sufficient"])
        if sufficient >= 0.7 and proposed_answer.strip():
            action = "answer"
        elif sufficient >= 0.4:
            action = "refine"
        else:
            action = "recurse"
        return {
            "action": action,
            "confidence": sufficient,
            "reason": result["reason"],
        }, tokens

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
        trace: list[StepTrace] | None = None,
        route_draft_answer: str = "",
        remaining_total_tokens: int | None = None,
    ) -> tuple[str, int]:
        """Like :meth:`generate_answer`, but also returns token usage."""
        facts_text = self._format_facts(facts)
        hop_chain = self._format_hop_chain(trace or [], facts)

        user_content = self._answer_template.format(
            question=question,
            target_profile=target_profile or "No explicit target hint available.",
            facts=facts_text or "No facts available.",
            pending_slots="No explicit pending slots.",
            hop_chain=hop_chain or "No grounded hop chain available.",
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

        response = await self.llm_client.async_chat(
            messages,
            max_tokens=96,
            remaining_total_tokens=remaining_total_tokens,
        )
        self._record_usage(response)
        content: str = response["message"].get("content", "")
        content = self._strip_thinking(content)

        # Strip any JSON wrapper the LLM might add (e.g. {"answer": "X"})
        answer = self._extract_plain_answer(content).strip()
        if self._looks_meta_answer(answer):
            answer = self._best_fact_answer_span(facts)
        return answer.strip(), self._extract_total_tokens(response)

    async def generate_answer_object_with_usage(
        self,
        question: str,
        facts: list[Fact],
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
        trace: list[StepTrace] | None = None,
        route_draft_answer: str = "",
        remaining_total_tokens: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Generate a structured final answer with citations and confidence."""
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
        user_content += """

Return ONLY a single JSON object:
{
  "answer": "<short grounded answer or empty string>",
  "cited_fact_ids": [<1-based fact ids>],
  "justification_confidence": <float 0.0-1.0>,
  "justification": "<brief grounding explanation>",
  "missing_slot": "<pending slot name or empty string>"
}

Rules for `answer`:
- return the shortest final answer span only
- do not return a sentence or explanation
- if a cited fact's `answer span` resolves the final target, copy that span exactly
"""
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
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
            chat_template_kwargs=chat_template_kwargs,
            max_tokens=max_tokens,
        )
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

    async def check_hop_sufficiency_with_usage(
        self,
        question: str,
        facts: list[Fact],
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
        trace: list[StepTrace] | None = None,
        remaining_total_tokens: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tokens: int | None = 192,
    ) -> tuple[dict[str, Any], int]:
        """Lightweight answerability check for DoD early exit."""
        facts_text = self._format_facts(facts)
        hop_chain = self._format_hop_chain(trace or [], facts)
        user_content = f"""
Decide whether the grounded facts are sufficient to answer the original question now.

Output ONLY a single JSON object:
{{
  "answerable": <true|false>,
  "answer": "<short answer if answerable, else empty string>",
  "cited_fact_ids": [<1-based fact ids>],
  "confidence": <float 0.0-1.0>,
  "missing_slot": "<slot still needed or empty string>"
}}

Rules:
- Use only the grounded facts.
- Return answerable=false if any required bridge or final slot is missing.
- If answerable=true, answer must be a short final answer span copied from the facts when possible.
- Do not explain.

Question: {question}

Target profile:
{target_profile or 'No explicit target hint available.'}

Pending slots:
{self._format_pending_slots(pending_slots or []) or 'No explicit pending slots.'}

Hop chain:
{hop_chain or 'No grounded hop chain available.'}

Grounded facts:
{facts_text or 'No facts available.'}
"""
        messages = [
            {
                "role": "system",
                "content": "You are a strict grounded answerability checker. Respond with JSON only.",
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            temperature=0.0,
            remaining_total_tokens=remaining_total_tokens,
            chat_template_kwargs=chat_template_kwargs,
            max_tokens=max_tokens,
        )
        cited_fact_ids = parsed.get("cited_fact_ids", [])
        if not isinstance(cited_fact_ids, list):
            cited_fact_ids = []
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "answerable": bool(parsed.get("answerable", False)),
            "answer": str(parsed.get("answer", "")).strip(),
            "cited_fact_ids": [
                int(item) for item in cited_fact_ids if str(item).strip().isdigit()
            ],
            "justification_confidence": max(0.0, min(confidence, 1.0)),
            "justification": "hop_sufficiency_check",
            "missing_slot": self._normalise_pending_slot_name(
                str(parsed.get("missing_slot", "")).strip(),
                pending_slots,
            ),
        }, tokens

    async def propose_spawn(
        self,
        question: str,
        facts: list[Fact],
        trace: list[StepTrace],
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
        missing_reason: str = "",
        remaining_total_tokens: int | None = None,
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
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
        )
        sub_question = str(parsed.get("sub_question", "")).strip()
        retrieval_query = str(parsed.get("retrieval_query", "")).strip()
        goal = str(parsed.get("goal", "")).strip()
        slot_name = str(parsed.get("slot_name", "")).strip()
        if not sub_question:
            sub_question = question
        if not goal:
            goal = "Retrieve a missing fact needed to answer the original question."
        normalised_spawn, extra_tokens = await self._normalise_spawn_decision(
            question=question,
            sub_question=sub_question,
            retrieval_query=retrieval_query,
            goal=goal,
            slot_name=slot_name,
            facts=facts,
            trace=trace,
            target_profile=target_profile,
            pending_slots=pending_slots,
            missing_reason=missing_reason or "Retrieve the most critical missing fact.",
            remaining_total_tokens=(
                None
                if remaining_total_tokens is None
                else max(int(remaining_total_tokens) - tokens, 0)
            ),
        )
        return {
            "action": "spawn",
            **normalised_spawn,
        }, tokens + extra_tokens

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

    async def decompose_slot_dag_with_usage(
        self,
        question: str,
        max_subquestions: int,
        target_profile: str = "",
        remaining_total_tokens: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Produce a typed slot DAG fallback plan."""
        user_content = _SLOT_DAG_DECOMPOSE_TEMPLATE.format(
            question=question,
            max_subquestions=max_subquestions,
            target_profile=target_profile or "No explicit target hint available.",
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You decompose multi-hop questions into typed retrieval slots. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
            chat_template_kwargs=chat_template_kwargs,
            max_tokens=max_tokens,
        )
        hops = self._normalise_required_hops(parsed.get("required_hops", []))
        if not hops:
            raw_items = parsed.get("sub_questions", [])
            converted: list[dict[str, Any]] = []
            if isinstance(raw_items, list):
                for idx, item in enumerate(raw_items):
                    if not isinstance(item, dict):
                        continue
                    sub_question = str(item.get("sub_question", "")).strip()
                    if not sub_question:
                        continue
                    converted.append(
                        {
                            "slot_name": str(item.get("slot_name", f"slot_{idx + 1}")).strip(),
                            "hint": str(item.get("goal", sub_question)).strip(),
                            "expected_info_type": str(item.get("expected_info_type", "other")).strip(),
                            "dependency_group": idx,
                            "sub_question": sub_question,
                            "retrieval_query": str(item.get("retrieval_query", sub_question)).strip(),
                            "goal": str(item.get("goal", "")).strip(),
                        }
                    )
                    if len(converted) >= max_subquestions:
                        break
            hops = self._normalise_required_hops(converted)
        return hops[:max_subquestions], tokens

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
        remaining_total_tokens: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Call the LLM, strip thinking, parse JSON, and accumulate token usage."""
        if chat_template_kwargs is None:
            # Controller calls are schema-only. Let investigator subagents use
            # thinking, but keep routing/decision/synthesis JSON short and parseable.
            chat_template_kwargs = {"enable_thinking": False}
        total_tokens = 0
        last_content = ""
        for attempt in range(_MAX_JSON_RETRIES + 1):
            try:
                response = await self.llm_client.async_chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    chat_template_kwargs=chat_template_kwargs,
                    remaining_total_tokens=(
                        None
                        if remaining_total_tokens is None
                        else max(int(remaining_total_tokens) - total_tokens, 0)
                    ),
                )
                self._record_usage(response)
                total_tokens += self._extract_total_tokens(response)
                last_content = response["message"].get("content", "")
                content = self._strip_thinking(last_content)
                parsed = self._parse_json(content)
                if parsed is not None:
                    return parsed, total_tokens
            except TokenBudgetExceededError:
                raise
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
            answer_span = str(getattr(fact, "answer_span", "")).strip()
            confidence = f"{fact.confidence:.2f}"
            slot_name = str(getattr(fact, "slot_name", "")).strip()
            slot_prefix = f"slot: {slot_name} | " if slot_name else ""
            if answer_span:
                lines.append(
                    f"{i}. {slot_prefix}answer span: {answer_span} | confidence: {confidence} | fact: {fact.text}"
                )
            else:
                lines.append(f"{i}. {slot_prefix}confidence: {confidence} | fact: {fact.text}")
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
    def _format_pending_slots(pending_slots: list[dict[str, Any]]) -> str:
        """Format slot state for prompt injection."""
        if not pending_slots:
            return ""
        lines = []
        for idx, slot in enumerate(pending_slots, start=1):
            slot_name = str(slot.get("slot_name", "")).strip() or f"slot_{idx}"
            hint = str(slot.get("hint", "")).strip() or "No hint provided."
            status = "resolved" if slot.get("resolved") else "pending"
            dependency_group = int(slot.get("dependency_group", idx - 1))
            expected_info_type = str(slot.get("expected_info_type", "")).strip()
            retrieval_query = str(slot.get("retrieval_query", "")).strip()
            type_part = (
                f" [type {expected_info_type}]"
                if expected_info_type
                else ""
            )
            query_part = f" [query {retrieval_query}]" if retrieval_query else ""
            lines.append(
                f"{idx}. {slot_name} [{status}] [group {dependency_group}]{type_part}{query_part} - {hint}"
            )
        return "\n".join(lines)

    @staticmethod
    def _normalise_required_hops(raw_items: Any) -> list[dict[str, Any]]:
        """Normalise required-hop slot specs returned by the router."""
        normalised: list[dict[str, Any]] = []
        seen: set[str] = set()
        if not isinstance(raw_items, list):
            return normalised
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            slot_name = str(
                item.get("slot_name", item.get("target_slot", ""))
            ).strip()
            hint = str(item.get("hint", "")).strip()
            if not slot_name or Orchestrator._is_placeholder_slot_name(slot_name):
                continue
            key = slot_name.lower()
            if key in seen:
                continue
            seen.add(key)
            dependency_group = item.get("dependency_group", idx)
            try:
                dependency_group = max(0, int(dependency_group))
            except (TypeError, ValueError):
                dependency_group = idx
            normalised.append(
                {
                    "slot_name": slot_name,
                    "hint": hint,
                    "expected_info_type": _normalise_expected_info_type(
                        slot_name,
                        str(item.get("expected_info_type", "")).strip(),
                    ),
                    "dependency_group": dependency_group,
                    "sub_question": str(item.get("sub_question", "")).strip(),
                    "retrieval_query": str(item.get("retrieval_query", "")).strip(),
                    "goal": str(item.get("goal", "")).strip(),
                }
            )
        return normalised

    async def _normalise_spawn_decision(
        self,
        *,
        question: str,
        sub_question: str,
        retrieval_query: str,
        goal: str,
        slot_name: str,
        facts: list[Fact],
        trace: list[StepTrace],
        target_profile: str,
        pending_slots: list[dict[str, str]] | None,
        missing_reason: str,
        remaining_total_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Repair echoed or under-specified spawn decisions."""
        extra_tokens = 0
        cleaned_sub_question = str(sub_question or "").strip() or question
        cleaned_retrieval_query = str(retrieval_query or "").strip()
        cleaned_goal = str(goal or "").strip() or (
            "Retrieve a missing fact needed to answer the original question."
        )
        cleaned_slot_name = self._normalise_pending_slot_name(slot_name, pending_slots)
        echo_detected_llm_call = False
        echo_still_present_after_refine = False
        echo_repaired_count = 0

        if self._looks_like_question_echo(cleaned_sub_question, question):
            echo_detected_llm_call = True
            refined, refine_tokens = await self.refine_innermost_sub_question_with_usage(
                question=question,
                candidate_sub_question=cleaned_sub_question,
                facts=facts,
                trace=trace,
                target_profile=target_profile,
                pending_slots=pending_slots,
                missing_reason=missing_reason,
                remaining_total_tokens=(
                    None
                    if remaining_total_tokens is None
                    else max(int(remaining_total_tokens) - extra_tokens, 0)
                ),
            )
            extra_tokens += refine_tokens
            cleaned_sub_question = refined["sub_question"]
            cleaned_retrieval_query = refined["retrieval_query"]
            cleaned_goal = refined["goal"]
            cleaned_slot_name = (
                self._normalise_pending_slot_name(
                    refined.get("slot_name", ""),
                    pending_slots,
                )
                or cleaned_slot_name
            )
            if self._looks_like_question_echo(cleaned_sub_question, question):
                echo_still_present_after_refine = True
                echo_repaired_count = 1
            else:
                echo_repaired_count = 1

        if not cleaned_retrieval_query:
            cleaned_retrieval_query = cleaned_sub_question

        return {
            "sub_question": cleaned_sub_question,
            "retrieval_query": cleaned_retrieval_query,
            "goal": cleaned_goal,
            "slot_name": cleaned_slot_name,
            "echo_detected_llm_call": echo_detected_llm_call,
            "echo_still_present_after_refine": echo_still_present_after_refine,
            "echo_repaired_count": echo_repaired_count,
        }, extra_tokens

    async def refine_innermost_sub_question_with_usage(
        self,
        *,
        question: str,
        candidate_sub_question: str,
        facts: list[Fact],
        trace: list[StepTrace],
        target_profile: str = "",
        pending_slots: list[dict[str, str]] | None = None,
        missing_reason: str = "",
        remaining_total_tokens: int | None = None,
    ) -> tuple[dict[str, str], int]:
        """Rewrite an echoed sub-question into the innermost useful clause."""
        facts_text = self._format_facts(facts)
        trace_summary = self._format_trace(trace)
        user_content = _INNERMOST_TEMPLATE.format(
            question=question,
            candidate_sub_question=candidate_sub_question or question,
            target_profile=target_profile or "No explicit target hint available.",
            facts=facts_text or "None yet.",
            trace_summary=trace_summary or "No steps taken yet.",
            pending_slots=self._format_pending_slots(pending_slots or [])
            or "No explicit pending slots.",
            missing_reason=missing_reason or "The candidate sub-question still echoes the original question.",
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite multi-hop questions into the next single-hop retrieval question. "
                    "Always respond with a single JSON object."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        parsed, tokens = await self._call_and_parse_with_usage(
            messages,
            remaining_total_tokens=remaining_total_tokens,
        )
        sub_question = str(parsed.get("sub_question", "")).strip()
        retrieval_query = str(parsed.get("retrieval_query", "")).strip()
        goal = str(parsed.get("goal", "")).strip()
        slot_name = str(parsed.get("slot_name", "")).strip()

        if not sub_question:
            sub_question = candidate_sub_question or question
        if not retrieval_query:
            retrieval_query = sub_question
        if not goal:
            goal = "Retrieve the next missing bridge fact."

        return {
            "sub_question": sub_question,
            "retrieval_query": retrieval_query,
            "goal": goal,
            "slot_name": self._normalise_pending_slot_name(slot_name, pending_slots),
        }, tokens

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
    def _looks_like_question_echo(candidate: str, original: str) -> bool:
        """Return whether the candidate sub-question largely repeats the original."""
        candidate_norm = Orchestrator._normalise_question_text(candidate)
        original_norm = Orchestrator._normalise_question_text(original)
        if not candidate_norm or not original_norm:
            return False
        if candidate_norm == original_norm:
            return True
        if (
            candidate_norm in original_norm or original_norm in candidate_norm
        ) and min(len(candidate_norm), len(original_norm)) / max(
            len(candidate_norm), len(original_norm)
        ) >= 0.8:
            return True

        candidate_tokens = set(Orchestrator._content_tokens(candidate_norm))
        original_tokens = set(Orchestrator._content_tokens(original_norm))
        if not candidate_tokens or not original_tokens:
            return False

        overlap = len(candidate_tokens & original_tokens) / min(
            len(candidate_tokens), len(original_tokens)
        )
        ratio = difflib.SequenceMatcher(None, candidate_norm, original_norm).ratio()
        return overlap >= 0.85 or (ratio >= 0.8 and overlap >= 0.7)

    @staticmethod
    def _normalise_question_text(text: str) -> str:
        """Lowercase and remove punctuation for fuzzy echo detection."""
        cleaned = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower()))
        return cleaned.strip()

    @staticmethod
    def _content_tokens(text: str) -> list[str]:
        """Return non-trivial content tokens from a normalised question."""
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "and", "or", "but", "if", "then", "than", "that", "this",
            "these", "those", "what", "which", "who", "when", "where", "why", "how",
            "whom", "whose", "into", "about", "after", "before", "during", "through",
        }
        return [
            token
            for token in str(text).split()
            if token and token not in stop_words
        ]

    @staticmethod
    def _looks_meta_answer(text: str) -> bool:
        """Return whether the model produced commentary instead of an answer span."""
        cleaned = str(text or "").strip().lower()
        if not cleaned:
            return True
        bad_markers = (
            "cannot be answered",
            "cannot be determined",
            "cannot determine",
            "not enough information",
            "insufficient information",
            "provided facts",
            "given facts",
            "do not provide enough",
            "do not contain enough",
            "unknown",
            "not specified",
            "the question",
            "the facts",
        )
        return any(marker in cleaned for marker in bad_markers)

    @staticmethod
    def _best_fact_answer_span(facts: list[Fact]) -> str:
        """Return the strongest extracted answer span currently in memory."""
        candidates: list[tuple[float, int, int, str]] = []
        for fact in facts:
            answer_span = str(getattr(fact, "answer_span", "")).strip()
            if not answer_span or Orchestrator._looks_meta_answer(answer_span):
                continue
            candidates.append(
                (fact.confidence, fact.source_step, -len(answer_span), answer_span)
            )
        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][3]

    @staticmethod
    def _format_hop_chain(trace: list[StepTrace], facts: list[Fact]) -> str:
        """Format grounded hop traces for answer synthesis."""
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
                    matching_slot_facts = [
                        fact
                        for fact in step_facts
                        if str(getattr(fact, "slot_name", "")).strip()
                        == str(entry.slot_name or "").strip()
                    ]
                    if matching_slot_facts:
                        step_facts = matching_slot_facts
                ranked = sorted(
                    step_facts,
                    key=lambda fact: (
                        fact.confidence,
                        bool(fact.answer_span.strip()),
                        len(fact.support_ids),
                    ),
                    reverse=True,
                )
                chosen = ranked[0]
                best_fact = chosen.answer_span.strip() or chosen.text.strip()
            slot_label = f" [slot={entry.slot_name}]" if entry.slot_name else ""
            if best_fact:
                lines.append(
                    f"Hop {hop_idx}{slot_label}: {entry.sub_question} -> found {best_fact}"
                )
            else:
                lines.append(
                    f"Hop {hop_idx}{slot_label}: {entry.sub_question} -> no grounded fact"
                )
            hop_idx += 1
        return "\n".join(lines)

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

    def _record_usage(self, response: dict[str, Any]) -> None:
        """Accumulate prompt/completion usage from one LLM response."""
        self._prompt_tokens_total += int(response.get("input_tokens", 0) or 0)
        self._completion_tokens_total += int(response.get("output_tokens", 0) or 0)
