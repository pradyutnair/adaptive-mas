"""Single-call planner for structure-aware adaptive topology."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .llm import LLMClient, parse_json_object
from .types import AnswerType, ExecutionPlan, SubgoalNode


class Planner:
    """Produce a simple-lane decision or a dependency DAG of subgoals."""

    def __init__(self, llm: LLMClient, max_subgoals: int = 6) -> None:
        self.llm = llm
        self.max_subgoals = max(1, int(max_subgoals))
        self._template = (
            Path(__file__).parent / "prompts" / "planner.txt"
        ).read_text(encoding="utf-8")

    async def plan(
        self,
        question: str,
        fallback_answer_type: AnswerType | None = None,
    ) -> tuple[ExecutionPlan, int]:
        prompt = self._template.format(
            question=question.strip(),
            max_subgoals=self.max_subgoals,
        )
        resp = await self.llm.chat(messages=[{"role": "user", "content": prompt}])
        parsed = parse_json_object(resp.content)
        return self._parse_plan(parsed, question, fallback_answer_type), resp.total_tokens

    def _parse_plan(
        self,
        parsed: dict[str, Any],
        question: str,
        fallback_answer_type: AnswerType | None,
    ) -> ExecutionPlan:
        raw_subgoals = parsed.get("subgoals")
        if not isinstance(raw_subgoals, list):
            raw_subgoals = []

        subgoals: list[SubgoalNode] = []
        seen_ids: set[int] = set()
        for idx, item in enumerate(raw_subgoals[: self.max_subgoals], start=1):
            if not isinstance(item, dict):
                continue
            node = SubgoalNode.from_dict({**item, "id": item.get("id", idx)})
            if not node.question:
                continue
            if node.id in seen_ids:
                node.id = max(seen_ids, default=0) + 1
            seen_ids.add(node.id)
            subgoals.append(node)

        if not subgoals:
            subgoals = [
                SubgoalNode(
                    id=1,
                    question=question.strip(),
                    depends_on=[],
                    answer_type=fallback_answer_type or AnswerType.coerce(parsed.get("final_answer_type")),
                    rationale="Direct retrieval task.",
                )
            ]

        valid_ids = {node.id for node in subgoals}
        for node in subgoals:
            node.depends_on = [
                dep for dep in node.depends_on
                if dep in valid_ids and dep != node.id
            ]
            node.question = _replace_pronouns_with_placeholders(node.question, node.depends_on)

        repaired = False
        if _has_cycle(subgoals):
            repaired = True
            ordered = sorted(subgoals, key=lambda n: n.id)
            for idx, node in enumerate(ordered):
                node.depends_on = [] if idx == 0 else [ordered[idx - 1].id]
            subgoals = ordered

        complexity = str(parsed.get("complexity", "")).strip().lower()
        if complexity not in {"simple", "compositional"}:
            complexity = "simple" if len(subgoals) == 1 else "compositional"
        if len(subgoals) > 1:
            complexity = "compositional"

        final_answer_type = AnswerType.coerce(
            parsed.get(
                "final_answer_type",
                (fallback_answer_type or subgoals[-1].answer_type).value,
            )
        )
        reasoning = str(parsed.get("reasoning", "")).strip()
        if repaired:
            reasoning = (reasoning + " ").strip() + "cycle-repaired-to-chain"

        subgoals = _augment_subgoals(question, subgoals, final_answer_type)

        return ExecutionPlan(
            complexity=complexity,
            subgoals=subgoals,
            final_answer_type=final_answer_type,
            confidence=_bounded_float(parsed.get("confidence", 0.0)),
            reasoning=reasoning,
        )


def _augment_subgoals(
    question: str,
    subgoals: list[SubgoalNode],
    final_answer_type: AnswerType,
) -> list[SubgoalNode]:
    if not subgoals:
        return subgoals
    context_hint = _question_context_hint(question)
    final_relation = _relation_hint(question)

    for idx, node in enumerate(subgoals):
        extra_parts: list[str] = []
        if idx == 0 and len(subgoals) > 1:
            extra_parts.append("First resolve the bridge entity needed by the final question.")
            if context_hint:
                extra_parts.append(f"Question context: {context_hint}.")
            if final_relation:
                extra_parts.append(f"Final relation: {final_relation}.")
            if node.answer_type in {AnswerType.DATE, AnswerType.NUMBER, AnswerType.YES_NO}:
                node.answer_type = AnswerType.ENTITY
        elif node.depends_on:
            if final_relation:
                extra_parts.append(f"Use prior result to answer the final relation: {final_relation}.")
            if context_hint:
                extra_parts.append(f"Question context: {context_hint}.")

        merged = " ".join(part for part in [node.rationale.strip(), *extra_parts] if part).strip()
        node.rationale = merged

    if len(subgoals) == 1 and not subgoals[0].rationale:
        subgoals[0].rationale = "Direct retrieval task."
    return subgoals


def _replace_pronouns_with_placeholders(question: str, depends_on: list[int]) -> str:
    if len(depends_on) != 1:
        return question.strip()
    dep_id = depends_on[0]
    placeholder = f"[result_{dep_id}]"
    patterns = [
        r"this person", r"that person", r"this country", r"that country",
        r"this city", r"that city", r"this place", r"that place",
        r"this publisher", r"that publisher", r"this team", r"that team",
        r"this organization", r"that organization", r"this entity", r"that entity",
        r"this player", r"that player", r"this actor", r"that actor",
    ]
    out = question.strip()
    for pattern in patterns:
        out = re.sub(pattern, placeholder, out, flags=re.IGNORECASE)
    return out


def _question_context_hint(question: str, limit: int = 10) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", question)
    stop = {
        "what", "when", "where", "which", "who", "whom", "whose", "is", "was", "were", "are",
        "did", "does", "do", "the", "a", "an", "of", "to", "in", "on", "for", "by", "and",
        "or", "that", "this", "these", "those", "be", "been", "being", "it", "its", "their",
    }
    keep: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in stop:
            continue
        if word[0].isupper() or len(word) >= 6 or lower in {
            "birthplace", "publisher", "signed", "abolished", "compared", "capital", "county",
            "month", "year", "founded", "directed", "wrote", "originated",
        }:
            keep.append(word)
    return " ".join(keep[:limit])


def _relation_hint(question: str) -> str:
    lower = question.lower()
    patterns = [
        "signed by", "signed for", "birthplace", "publisher of", "capital of", "compared to",
        "abolished", "ended", "founded", "directed", "wrote", "originated in", "part of",
    ]
    for pattern in patterns:
        if pattern in lower:
            return pattern
    return ""


def _has_cycle(nodes: list[SubgoalNode]) -> bool:
    pending = {node.id: set(node.depends_on) for node in nodes}
    emitted: set[int] = set()
    while pending:
        ready = [node_id for node_id, deps in pending.items() if deps.issubset(emitted)]
        if not ready:
            return True
        for node_id in ready:
            emitted.add(node_id)
            pending.pop(node_id, None)
    return False


def _bounded_float(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
