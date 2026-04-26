"""Cursor-style adaptive orchestrator.

A single tool-using agent that alternates between three actions until it can
emit a grounded final answer:

- ``search``  — retrieve and read chunks itself (chunks visible only to this
  agent's private message history).
- ``spawn``   — delegate one or more sub-questions to isolated investigator
  subagents. The orchestrator receives only :class:`EvidenceCapsule` objects
  back, never raw passages. Multiple subagents in one ``spawn`` call run IN
  PARALLEL.
- ``final``   — emit the short, grounded answer span.

Topology is fully emergent:
- Easy lookups: search -> final  (~SAS efficiency)
- 2-hop: search -> spawn -> final, or spawn -> spawn -> final
- 3-4 hop with parallel branches: spawn[a, b] -> spawn[c(a, b)] -> final
- Hybrid: search to disambiguate a bridge, then spawn with rich hints
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

from .investigator import Investigator
from .llm import LLMClient, parse_json_object, strip_thinking
from .retriever import RetrievalHit, Retriever
from .types import AnswerType, EvidenceCapsule, StepTrace

# Cheap, deterministic token estimator. Used only for budget enforcement, never
# for billing — actual usage comes from the API.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens of a chat-style message list.

    Each message has ~4 tokens of role/structure overhead in OpenAI-format
    transcripts; we add that on top of the raw content tokens.
    """
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(_TOKENIZER.encode(content))
        else:
            total += len(_TOKENIZER.encode(json.dumps(content, ensure_ascii=False)))
        total += 4
    return total + 2  # priming tokens

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Output of one ``Orchestrator.solve`` call."""

    answer: str
    answer_type: str = "entity"
    justification: str = ""
    support_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trace: list[StepTrace] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_total: int = 0
    capsules: list[EvidenceCapsule] = field(default_factory=list)
    total_tokens: int = 0
    orchestrator_tokens: int = 0
    subagent_tokens: int = 0
    chunk_tokens: int = 0
    reasoning_tokens: int = 0
    n_searches: int = 0
    n_spawn_turns: int = 0
    n_subagents: int = 0
    route: str = ""


class Orchestrator:
    """Lead agent. Picks search / spawn / final each turn until done."""

    def __init__(
        self,
        llm: LLMClient,
        investigator: Investigator,
        retriever: Retriever,
        max_turns: int = 8,
        default_top_k: int = 10,
        context_token_budget: int = 28000,
        max_response_tokens: int = 4096,
    ) -> None:
        self.llm = llm
        self.investigator = investigator
        self.retriever = retriever
        self.max_turns = int(max_turns)
        self.default_top_k = int(default_top_k)
        # Hard stop: if the next prompt would exceed this, abort the agent
        # loop and emit the best available answer. Sized with a safety margin
        # below vLLM's max_model_len.
        self.context_token_budget = int(context_token_budget)
        # Reserve room for the model's response (output tokens).
        self.max_response_tokens = int(max_response_tokens)
        self._tpl = (
            Path(__file__).parent / "prompts" / "orchestrator.txt"
        ).read_text(encoding="utf-8")

    async def solve(self, question: str) -> OrchestratorResult:
        prompt = self._tpl.format(question=question, max_turns=self.max_turns)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace: list[StepTrace] = []
        retrieved_ids: list[str] = []
        retrieved_total = 0
        capsules: list[EvidenceCapsule] = []
        total_tokens = 0
        orch_tokens = 0
        agent_tokens = 0
        chunk_tokens = 0
        n_searches = 0
        n_spawn_turns = 0
        n_subagents = 0
        last_content = ""

        budget_hit = False
        verification_count = 0
        for turn in range(self.max_turns + 1):
            # Hard budget stop: if the next prompt would exceed the context
            # budget (with response margin), break and emit best-effort final.
            est = _estimate_tokens(messages)
            if est + self.max_response_tokens > self.context_token_budget:
                logger.info(
                    "Context budget hit at turn=%d est_prompt=%d budget=%d",
                    turn, est, self.context_token_budget,
                )
                trace.append(StepTrace(
                    step=len(trace), action="answer", tokens=0,
                    metadata={"turn": turn, "reason": "context_budget_hit",
                              "estimated_prompt_tokens": est,
                              "budget": self.context_token_budget},
                ))
                budget_hit = True
                break

            resp = await self.llm.chat(messages=messages)
            total_tokens += resp.total_tokens
            orch_tokens += resp.total_tokens
            content = strip_thinking(resp.content)
            last_content = content
            parsed = parse_json_object(content)
            action = str(parsed.get("action", "")).strip().lower()

            if action == "search" and turn < self.max_turns:
                hits, ids = await self._run_search(parsed)
                for cid in ids:
                    if cid not in retrieved_ids:
                        retrieved_ids.append(cid)
                retrieved_total += len(hits)
                messages.append({"role": "assistant", "content": content})
                full_payload = json.dumps({"search_result": [
                    {"chunk_id": h.chunk_id, "score": round(h.score, 4), "text": h.text}
                    for h in hits
                ]})
                chunk_tokens += len(_TOKENIZER.encode(full_payload))
                trimmed_payload = json.dumps({"search_result": [
                    {"chunk_id": h.chunk_id, "score": round(h.score, 4),
                     "text": h.text[:300]}
                    for h in hits[:5]
                ]})
                messages.append({"role": "user", "content": trimmed_payload})
                n_searches += 1
                trace.append(StepTrace(
                    step=len(trace), action="route", tokens=resp.total_tokens,
                    route_decision="search",
                    metadata={"turn": turn, "query": parsed.get("query", ""),
                              "n_hits": len(hits)},
                ))
                continue

            if action == "spawn" and turn < self.max_turns:
                sub_specs = parsed.get("subagents") or []
                if not isinstance(sub_specs, list) or not sub_specs:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": json.dumps({
                        "system_note": "spawn must include a non-empty 'subagents' list.",
                    })})
                    continue

                tasks = []
                meta_list: list[dict[str, Any]] = []
                for spec in sub_specs:
                    if not isinstance(spec, dict):
                        continue
                    sub_q = str(spec.get("sub_question", "")).strip()
                    if not sub_q:
                        continue
                    expected = AnswerType.coerce(spec.get("expected_type", "entity")).value
                    hint = str(spec.get("hint", "")).strip()
                    slot = self._slot_name(sub_q)
                    tasks.append(self.investigator.investigate(
                        sub_question=sub_q,
                        expected_answer_type=expected,
                        hint=hint,
                        slot_name=slot,
                    ))
                    meta_list.append({"sub_question": sub_q, "expected": expected,
                                      "hint": hint, "slot": slot})

                if not tasks:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": json.dumps({
                        "system_note": "no valid subagents provided; check schema.",
                    })})
                    continue

                results = await asyncio.gather(*tasks, return_exceptions=True)
                returned_capsules: list[dict[str, Any]] = []
                for meta, result in zip(meta_list, results):
                    if isinstance(result, Exception):
                        logger.warning("Investigator crashed: %s", result)
                        returned_capsules.append({
                            "sub_question": meta["sub_question"],
                            "answer_span": "",
                            "confidence": 0.0,
                            "support_ids": [],
                            "error": str(result),
                        })
                        trace.append(StepTrace(
                            step=len(trace), action="spawn", tokens=0,
                            sub_question=meta["sub_question"],
                            slot_name=meta["slot"],
                            metadata={"turn": turn, "expected": meta["expected"],
                                      "hint": meta["hint"], "error": str(result)},
                        ))
                        continue
                    capsule, inv_tok = result
                    total_tokens += inv_tok
                    agent_tokens += inv_tok
                    chunk_tokens += capsule.chunk_tokens
                    capsules.append(capsule)
                    for cid in capsule.retrieved_doc_ids:
                        if cid not in retrieved_ids:
                            retrieved_ids.append(cid)
                    retrieved_total += capsule.retrieved_docs_total
                    returned_capsules.append({
                        "sub_question": meta["sub_question"],
                        "answer_span": capsule.answer,
                        "justification": capsule.fact.text,
                        "confidence": round(capsule.fact.confidence, 3),
                        "support_ids": capsule.fact.support_ids,
                    })
                    trace.append(StepTrace(
                        step=len(trace), action="spawn", tokens=inv_tok,
                        sub_question=meta["sub_question"],
                        slot_name=meta["slot"],
                        fact_added=capsule.fact.slot_filled,
                        justification_confidence=capsule.fact.confidence,
                        metadata={"turn": turn, "expected": meta["expected"],
                                  "hint": meta["hint"]},
                    ))
                    n_subagents += 1

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": json.dumps({"capsules": returned_capsules}),
                })
                n_spawn_turns += 1
                continue

            if action == "final":
                ans = str(parsed.get("answer_span", parsed.get("answer", ""))).strip()
                ans_type = AnswerType.coerce(parsed.get("answer_type", "entity"))

                # Guardrail: catch "answered the bridge entity" failures.
                total_actions = n_searches + n_spawn_turns
                if (verification_count < 1
                        and total_actions <= 2
                        and turn < self.max_turns
                        and ans
                        and ans.lower() not in {"unknown", "n/a", ""}):
                    verification_count += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "system_check": (
                                "MANDATORY VERIFICATION — re-read the original "
                                "question carefully before finalizing.\n"
                                f"Your draft answer is: '{ans}'\n"
                                f"Original question: '{question}'\n\n"
                                "Check TWO things:\n"
                                "1) BRIDGE vs FINAL: Is your answer the FINAL "
                                "attribute the question asks for, or a bridge "
                                "entity along the way? If bridge, `spawn` for "
                                "the remaining hop(s).\n"
                                "2) ANSWER TYPE: Does your answer match what the "
                                "question asks? (e.g. 'which university' needs a "
                                "name, 'when' needs a date, 'who' needs a person)\n\n"
                                "If both checks pass, re-emit `final` with the "
                                "same answer. Otherwise, take the corrective action."
                            ),
                        }),
                    })
                    trace.append(StepTrace(
                        step=len(trace), action="route",
                        tokens=resp.total_tokens,
                        route_decision="bridge_check",
                        metadata={"turn": turn, "draft_answer": ans},
                    ))
                    continue

                support_raw = parsed.get("support_ids") or []
                # support_ids must come from things the orchestrator has actually seen
                # (its own searches OR capsules returned by spawned investigators).
                allowed = set(retrieved_ids) | {
                    sid for c in capsules for sid in c.fact.support_ids
                }
                support_ids = [str(s).strip() for s in support_raw
                               if str(s).strip() and str(s).strip() in allowed]
                conf = self._bounded_float(parsed.get("confidence", 0.0))
                trace.append(StepTrace(
                    step=len(trace), action="answer",
                    tokens=resp.total_tokens, justification_confidence=conf,
                    metadata={"turn": turn, "answer_type": ans_type.value},
                ))
                return OrchestratorResult(
                    answer=ans,
                    answer_type=ans_type.value,
                    justification=str(parsed.get("justification", "")).strip(),
                    support_ids=support_ids,
                    confidence=conf,
                    trace=trace,
                    retrieved_ids=retrieved_ids,
                    retrieved_total=retrieved_total,
                    capsules=capsules,
                    total_tokens=total_tokens,
                    orchestrator_tokens=orch_tokens,
                    subagent_tokens=agent_tokens,
                    chunk_tokens=chunk_tokens,
                    reasoning_tokens=total_tokens - chunk_tokens,
                    n_searches=n_searches,
                    n_spawn_turns=n_spawn_turns,
                    n_subagents=n_subagents,
                    route=self._classify_route(n_searches, n_spawn_turns, n_subagents),
                )

            # Bad turn: nudge.
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": json.dumps({
                "system_note": (
                    "Output exactly one JSON object with action 'search', 'spawn', "
                    f"or 'final'. Turn budget: {self.max_turns - turn} left."
                ),
            })})

        # Loop exhausted OR budget hit: salvage best we can.
        salvage = parse_json_object(last_content)
        ans = str(salvage.get("answer_span", "")).strip()
        if not ans:
            for c in reversed(capsules):
                if c.answer:
                    ans = c.answer
                    break
        if not ans:
            ans = "unknown"
        reason = "context_budget_hit" if budget_hit else "loop_exhausted"
        return OrchestratorResult(
            answer=ans,
            answer_type=AnswerType.coerce(salvage.get("answer_type", "entity")).value,
            justification=f"({reason})",
            trace=trace,
            retrieved_ids=retrieved_ids,
            retrieved_total=retrieved_total,
            capsules=capsules,
            total_tokens=total_tokens,
            orchestrator_tokens=orch_tokens,
            subagent_tokens=agent_tokens,
            chunk_tokens=chunk_tokens,
            reasoning_tokens=total_tokens - chunk_tokens,
            n_searches=n_searches,
            n_spawn_turns=n_spawn_turns,
            n_subagents=n_subagents,
            route=self._classify_route(n_searches, n_spawn_turns, n_subagents),
        )

    # ------------------------------------------------------------------
    # Search execution
    # ------------------------------------------------------------------

    async def _run_search(self, parsed: dict) -> tuple[list[RetrievalHit], list[str]]:
        query = str(parsed.get("query", "")).strip()
        if not query:
            return [], []
        try:
            top_k = max(1, min(int(parsed.get("top_k") or self.default_top_k), 20))
        except (TypeError, ValueError):
            top_k = self.default_top_k
        hits = await self.retriever.retrieve(query, top_k=top_k)
        return hits, [h.chunk_id for h in hits]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slot_name(sub_question: str) -> str:
        # Compact identifier derived from the sub-question, for tracing.
        words = [w.lower() for w in sub_question.split() if w.isalpha()][:4]
        return "_".join(words)[:50] or "subagent"

    @staticmethod
    def _classify_route(n_searches: int, n_spawn_turns: int, n_subagents: int) -> str:
        if n_subagents == 0:
            return "search_only"          # pure SAS
        if n_searches == 0:
            return "spawn_only"            # pure delegated
        return "hybrid"                   # search + spawn

    @staticmethod
    def _bounded_float(v) -> float:
        try:
            return max(0.0, min(float(v), 1.0))
        except (TypeError, ValueError):
            return 0.0
