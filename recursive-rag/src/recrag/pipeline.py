from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy

from .retriever import Retriever
from .tools import ToolRuntime, make_tools
from .trace import build_readable_trace, build_structured_stats


@dataclass
class PipelineConfig:
    max_iters: int = 15
    experience_library: str | None = None
    citation_gate: bool = True


def _read_experience(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def _tokens_since(lm: dspy.LM, start_idx: int) -> int:
    total = 0
    for item in getattr(lm, "history", [])[start_idx:]:
        usage = (item or {}).get("usage") or {}
        try:
            total += int(usage.get("total_tokens", 0))
        except Exception:
            pass
    return total


def _accepted_submit_answer(trajectory: dict[str, Any]) -> str:
    if not isinstance(trajectory, dict):
        return ""
    accepted = ""
    for key in sorted(trajectory):
        if not key.startswith("tool_name_") or trajectory.get(key) != "submit":
            continue
        idx = key.rsplit("_", 1)[-1]
        if trajectory.get(f"observation_{idx}") == "ACCEPTED":
            args = trajectory.get(f"tool_args_{idx}") or {}
            accepted = str(args.get("answer", "")).strip()
    return accepted


INSTRUCTIONS = (
    "You are an adaptive multi-hop QA agent. Answer the question using retrieval tools.\n\n"
    "STRATEGY:\n"
    "1. Analyze the question: is it single-hop (one fact lookup) or multi-hop (needs chaining/bridging facts)?\n"
    "2. For single-hop: one hop() call suffices.\n"
    "3. For multi-hop bridge questions (e.g. 'What is X of Y where Y requires lookup'):\n"
    "   - First hop to resolve the bridge entity (e.g. find who/what Y is).\n"
    "   - Then hop to answer the actual question using the resolved entity.\n"
    "4. For multi-hop parallel questions (e.g. comparing two independent facts):\n"
    "   - Use hop_batch to dispatch independent sub-questions concurrently.\n\n"
    "ALWAYS set expected_answer_type in hop calls. Choose from: person, place, date, number, title, organization, yes_no, entity.\n"
    "Example: hop(question='When was X born?', expected_answer_type='date')\n\n"
    "SYNTHESIS:\n"
    "- The final answer must be ONE concise span (1-6 words). NEVER list multiple values.\n"
    "- GOOD: 'Shaun Tan', '450', 'October 8, 2009', '1981'\n"
    "- BAD: 'The author is Shaun Tan', '1982, 2006, 1983'\n"
    "- When a question asks 'when/where did A, B, and C do X', it expects ONE shared answer, not separate ones.\n"
    "- If parallel hops return conflicting answers, the sub-findings are likely wrong. Do a direct hop on the\n"
    "  original question or a rephrased collective version (e.g. 'When did Japanese automakers open US plants?').\n"
    "- Pick the single best-supported finding. If findings conflict, trust higher confidence or try a new angle.\n\n"
    "NEVER repeat the same hop query. If a hop returns low confidence or unhelpful results, "
    "reformulate: ask about the bridge entity differently, change the expected_answer_type, "
    "or decompose the question further.\n\n"
    "Before finishing, call submit(answer, support_ids) with evidence_chunk_id values from hop outputs. "
    "If submit rejects, fix the answer to match cited findings."
)


class ReactRagPipeline:
    def __init__(self, root_lm: dspy.LM, sub_lm: dspy.LM, retriever: Retriever, config: PipelineConfig):
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        self.retriever = retriever
        self.config = config
        self.tool_state = ToolRuntime()
        tools = make_tools(retriever, sub_lm, self.tool_state)
        instructions = INSTRUCTIONS
        experience = _read_experience(config.experience_library)
        if experience:
            instructions += "\n\nExperience library:\n" + experience
        signature = dspy.Signature("question -> answer", instructions)
        self.react = dspy.ReAct(signature=signature, tools=tools, max_iters=config.max_iters)

    async def run(self, question: str) -> dict[str, Any]:
        self.tool_state.reset()
        root_start = len(getattr(self.root_lm, "history", []))
        sub_start = len(getattr(self.sub_lm, "history", []))
        with dspy.context(lm=self.root_lm):
            result = await asyncio.to_thread(self.react, question=question)
        trajectory = getattr(result, "trajectory", {})
        answer = _accepted_submit_answer(trajectory) or str(getattr(result, "answer", "")).strip()
        root_tokens = _tokens_since(self.root_lm, root_start)
        sub_tokens = _tokens_since(self.sub_lm, sub_start)
        findings_dicts = [f.as_dict() for f in self.tool_state.findings]
        metadata = {
            "root_tokens": root_tokens,
            "sub_tokens": sub_tokens,
            "total_tokens": root_tokens + sub_tokens,
            "hops": self.tool_state.total_hops,
            "retries": self.tool_state.total_retries,
            "tool_errors": list(self.tool_state.tool_errors),
            "findings": findings_dicts,
        }
        return {
            "question": question,
            "predicted_answer": answer,
            "answer": answer,
            "trajectory": trajectory,
            "metadata": metadata,
            "readable_trace": build_readable_trace(trajectory, findings_dicts, answer),
            "structured_stats": build_structured_stats(metadata, trajectory, answer),
        }
