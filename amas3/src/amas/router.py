"""Difficulty router for AMAS.

A single-purpose LLM call that classifies the input question as `easy` or `hard`.
- `easy`: solvable by a single investigator on the original question with one or two retrievals.
- `hard`: requires explicit decomposition (planner -> DAG with rewriters) to resolve bridge entities.

The router does NOT retrieve, NOT search, NOT plan. It is a classifier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .llm import LLMClient, parse_json_object

logger = logging.getLogger(__name__)


Complexity = Literal["easy", "hard"]


@dataclass
class RouterDecision:
    """Output of the difficulty router."""

    complexity: Complexity
    confidence: float
    reasoning: str
    raw: dict


class DifficultyRouter:
    """Classify a question as `easy` (single-agent) or `hard` (planner+DAG)."""

    def __init__(self, llm: LLMClient, prompt_path: str | Path | None = None) -> None:
        self.llm = llm
        if prompt_path is None:
            prompt_path = Path(__file__).resolve().parent / "prompts" / "router.txt"
        self.prompt_template = Path(prompt_path).read_text()

    async def classify(self, question: str) -> tuple[RouterDecision, int]:
        prompt = self.prompt_template.format(question=question)
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        parsed = parse_json_object(resp.content)
        # Conservative default: when in doubt, treat as hard (planner+DAG).
        complexity_raw = str(parsed.get("complexity", "hard")).strip().lower()
        complexity: Complexity = "easy" if complexity_raw == "easy" else "hard"
        confidence = float(parsed.get("confidence", 0.0)) if parsed else 0.0
        reasoning = str(parsed.get("reasoning", ""))[:300]
        decision = RouterDecision(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            raw=parsed,
        )
        return decision, resp.total_tokens
