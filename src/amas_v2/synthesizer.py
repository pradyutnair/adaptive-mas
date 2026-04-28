"""Synthesizer: extract precise final answer from collected facts."""

from __future__ import annotations

from pathlib import Path

from .llm import LLMClient, parse_json_object


class Synthesizer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._template = (
            Path(__file__).parent / "prompts" / "synthesize.txt"
        ).read_text(encoding="utf-8")

    async def synthesize(
        self,
        question: str,
        facts: list[dict[str, str]],
        last_evidence: str = "",
    ) -> tuple[str, int]:
        facts_text = "\n".join(
            f"- Step {f.get('step', '?')}: Q: {f.get('question', '?')} -> A: {f.get('answer', '?')} ({f.get('justification', '')})"
            for f in facts
        )
        prompt = self._template.format(
            question=question.strip(),
            facts=facts_text or "(no facts collected)",
            last_evidence=last_evidence[:500] or "(none)",
        )
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        parsed = parse_json_object(resp.content)
        answer = str(parsed.get("answer", "")).strip()
        return answer, resp.total_tokens
