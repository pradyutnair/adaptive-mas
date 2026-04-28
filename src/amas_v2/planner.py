"""Planner: decomposes questions into subgoal DAGs using Qwen3-14B+thinking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .llm import LLMClient, parse_json_object
from .types import AnswerType, ExecutionPlan, SubgoalNode


class Planner:
    def __init__(self, llm: LLMClient, max_subgoals: int = 4) -> None:
        self.llm = llm
        self.max_subgoals = max(1, int(max_subgoals))
        self._template = (
            Path(__file__).parent / "prompts" / "planner.txt"
        ).read_text(encoding="utf-8")

    async def plan(self, question: str) -> tuple[ExecutionPlan, int]:
        prompt = self._template.format(
            question=question.strip(),
            max_subgoals=self.max_subgoals,
        )
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        parsed = parse_json_object(resp.content)
        if not parsed:
            parsed = self._fallback_parse(resp.content, question)
        return self._build_plan(parsed, question), resp.total_tokens

    def _build_plan(self, parsed: dict[str, Any], question: str) -> ExecutionPlan:
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
            subgoals = [SubgoalNode(
                id=1, question=question.strip(), depends_on=[],
                answer_type=AnswerType.coerce(parsed.get("final_answer_type")),
                rationale="Direct retrieval.",
            )]

        valid_ids = {n.id for n in subgoals}
        for n in subgoals:
            n.depends_on = [d for d in n.depends_on if d in valid_ids and d != n.id]

        if self._has_cycle(subgoals):
            subgoals = [SubgoalNode(
                id=1, question=question.strip(), depends_on=[],
                answer_type=AnswerType.coerce(parsed.get("final_answer_type")),
                rationale="Cycle fallback.",
            )]

        complexity = str(parsed.get("complexity", "")).strip().lower()
        if complexity not in {"simple", "compositional"}:
            complexity = "simple" if len(subgoals) == 1 else "compositional"
        if len(subgoals) > 1:
            complexity = "compositional"

        final_at = AnswerType.coerce(
            parsed.get("final_answer_type", subgoals[-1].answer_type.value)
        )
        try:
            conf = max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0))
        except (TypeError, ValueError):
            conf = 0.0

        return ExecutionPlan(
            complexity=complexity,
            subgoals=subgoals,
            final_answer_type=final_at,
            confidence=conf,
            reasoning=str(parsed.get("reasoning", "")),
        )

    @staticmethod
    def _has_cycle(subgoals: list[SubgoalNode]) -> bool:
        pending = {n.id: set(n.depends_on) for n in subgoals}
        emitted: set[int] = set()
        while pending:
            ready = [nid for nid, deps in pending.items() if deps.issubset(emitted)]
            if not ready:
                return True
            for nid in ready:
                emitted.add(nid)
                pending.pop(nid, None)
        return False

    @staticmethod
    def _fallback_parse(text: str, question: str) -> dict[str, Any]:
        """Extract subgoals from natural language when JSON parsing fails."""
        from .llm import strip_thinking
        text = strip_thinking(text)
        steps = re.findall(
            r'(?:step\s*|subgoal\s*|#?\s*)(\d+)[.:)\s]+(.+?)(?=(?:step\s*|subgoal\s*|#?\s*)\d|$)',
            text, re.IGNORECASE | re.DOTALL,
        )
        if steps:
            subgoals = []
            for sid, sq in steps:
                sq_clean = sq.strip().split("\n")[0].strip(' "\'')
                if sq_clean:
                    subgoals.append({
                        "id": int(sid),
                        "question": sq_clean,
                        "depends_on": [int(sid) - 1] if int(sid) > 1 else [],
                        "answer_type": "entity",
                    })
            if subgoals:
                return {"subgoals": subgoals, "complexity": "compositional"}
        return {"subgoals": [], "complexity": "simple"}
