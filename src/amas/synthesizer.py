"""Final answer synthesis from evidence capsules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm import LLMClient, parse_json_object
from .types import AnswerType, EvidenceCapsule


class Synthesizer:
    """Convert solved subgoals into one final answer span."""

    def __init__(self, llm: LLMClient, max_answer_words: int = 8) -> None:
        self.llm = llm
        self.max_answer_words = int(max_answer_words)
        self._template = (
            Path(__file__).parent / "prompts" / "synthesize.txt"
        ).read_text(encoding="utf-8")

    async def synthesize(
        self,
        question: str,
        capsules: list[EvidenceCapsule],
        answer_type: AnswerType,
    ) -> tuple[dict[str, Any], int]:
        prompt = self._template.format(
            question=question.strip(),
            answer_type=answer_type.value,
            capsules=json.dumps(
                [capsule.to_dict() for capsule in capsules],
                ensure_ascii=False,
                indent=2,
            ),
        )
        resp = await self.llm.chat(messages=[{"role": "user", "content": prompt}])
        parsed = parse_json_object(resp.content)
        answer = str(parsed.get("answer_span", "")).strip()
        if len(answer.split()) > self.max_answer_words:
            parsed["answer_span"] = ""
            parsed["confidence"] = 0.0
        return parsed, resp.total_tokens
