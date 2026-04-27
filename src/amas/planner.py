"""Single-call planner for structure-aware adaptive topology."""

from __future__ import annotations

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
            node.depends_on = [dep for dep in node.depends_on if dep in valid_ids and dep != node.id]

        repaired = False
        pending = {node.id: set(node.depends_on) for node in subgoals}
        emitted: set[int] = set()
        while pending:
            ready = [node_id for node_id, deps in pending.items() if deps.issubset(emitted)]
            if not ready:
                repaired = True
                break
            for node_id in ready:
                emitted.add(node_id)
                pending.pop(node_id, None)
        if repaired:
            repaired = True
            subgoals = [
                SubgoalNode(
                    id=1,
                    question=question.strip(),
                    depends_on=[],
                    answer_type=fallback_answer_type or AnswerType.coerce(parsed.get("final_answer_type")),
                    rationale="Planner returned an invalid cyclic graph; fall back to one direct retrieval task.",
                )
            ]
            complexity = "simple"

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
            reasoning = (reasoning + " ").strip() + "cycle-fallback-to-direct"
        try:
            confidence = max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return ExecutionPlan(
            complexity=complexity,
            subgoals=subgoals,
            final_answer_type=final_answer_type,
            confidence=confidence,
            reasoning=reasoning,
        )
