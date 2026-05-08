"""8 specialized HERA agents. Each has role + tools + dual-axis prompt (operational rules + behavioral principles)."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .lm import OpenAIClient, parse_json_lenient
from .retriever import Passage, RetrieverClient, format_passages

logger = logging.getLogger(__name__)


AGENT_NAMES = [
    "QueryDecomposer",
    "QueryRewriter",
    "Retriever",
    "EvidenceSelector",
    "ContextValidator",
    "AnswerGenerator",
    "ReflectAgent",
    "ConcludeAgent",
]


# Initial role prompts. These are the seed templates; RoPE updates op_rules + behavioral_principles.
SEED_PROMPTS: dict[str, dict[str, Any]] = {
    "QueryDecomposer": {
        "role": "Decompose a multi-hop question into 1-4 atomic single-hop sub-questions.",
        "instructions": (
            "Read the user's question. Identify all bridge entities and intermediate facts that must be "
            "resolved. Output a JSON list of sub-questions in the order they need to be answered, where "
            "later sub-questions may reference answers to earlier ones via {q1}, {q2}, etc."
        ),
        "operational_rules": [
            "Limit decomposition to at most 4 sub-questions.",
            "If the question is already a single hop, return a single-element list.",
            "Use placeholder {q1}, {q2} etc. only when later subquestion truly depends on earlier one.",
        ],
        "behavioral_principles": [
            "Prefer minimum sufficient decomposition over maximum granularity.",
        ],
        "output_schema": '{"sub_questions": ["...", "..."]}',
    },
    "QueryRewriter": {
        "role": "Rewrite or expand a question into one or more search-friendly queries for retrieval.",
        "instructions": (
            "Generate 1-3 alternative phrasings of the question that maximize retrieval recall. "
            "Use synonyms, named entity variants, expand abbreviations. Output JSON."
        ),
        "operational_rules": [
            "Preserve all named entities exactly.",
            "Do not invent facts; only reformulate.",
        ],
        "behavioral_principles": [
            "Diverse phrasings beat near-duplicate paraphrases.",
        ],
        "output_schema": '{"queries": ["..."]}',
    },
    "Retriever": {
        "role": "Retrieve top-k passages from the wiki18 corpus for each given query.",
        "instructions": "Tool-only agent: invokes the BM25/dense retriever endpoint. No LLM call.",
        "operational_rules": [],
        "behavioral_principles": [],
        "output_schema": "list of Passage objects",
    },
    "EvidenceSelector": {
        "role": "Select the minimal set of passages that are sufficient to answer the question.",
        "instructions": (
            "From the candidate passages, pick those that contain claims directly relevant to the "
            "question or any sub-question. Output JSON with selected passage indices and a brief "
            "rationale."
        ),
        "operational_rules": [
            "Prefer passages that mention bridge entities resolved in earlier steps.",
            "Discard passages that only mention surface-level keyword overlap.",
        ],
        "behavioral_principles": [
            "Smaller relevant set is better than larger noisy set.",
        ],
        "output_schema": '{"selected_indices": [int, ...], "rationale": "..."}',
    },
    "ContextValidator": {
        "role": "Verify whether the assembled context is sufficient to answer the question.",
        "instructions": (
            "Given the question and selected passages, decide if there is enough evidence. "
            "If not, identify what is missing and propose a follow-up retrieval query."
        ),
        "operational_rules": [
            "If any sub-question still has no supporting passage, mark insufficient.",
        ],
        "behavioral_principles": [
            "Do not hallucinate sufficiency; require explicit textual evidence.",
        ],
        "output_schema": '{"sufficient": bool, "missing": "...", "follow_up_query": "..."}',
    },
    "AnswerGenerator": {
        "role": "Produce a concise final answer from the question and the selected passages.",
        "instructions": (
            "Use ONLY the provided passages. Output a SPAN: one entity name, one date, "
            "one short phrase, OR yes/no. NEVER write a sentence. NEVER explain. "
            "MAX 8 words. Bare span only — no period, no quotes, no leading 'The answer is'."
        ),
        "operational_rules": [
            "Output only the answer span; no preamble, no explanation, no citation, no markdown, no period.",
            "If question asks 'what year' return ONLY the 4-digit year.",
            "If question is yes/no return ONLY 'yes' or 'no'.",
            "Cap output at 8 words. Reject any answer longer than 8 words and re-extract.",
            "Answer must be EXTRACTED VERBATIM from passages or a canonical short form of an entity in passages.",
        ],
        "behavioral_principles": [
            "Be precise: extra words hurt EM; missing entities hurt F1.",
            "Span only — never a clause, never a sentence.",
        ],
        "output_schema": '{"answer": "<bare span, max 8 words>"}',
    },
    "ReflectAgent": {
        "role": "Critique a proposed answer for errors, missing reasoning, or contradictions with the context.",
        "instructions": (
            "Given the question, passages, and proposed answer, decide whether to ACCEPT or REVISE. "
            "If REVISE, provide a corrected answer."
        ),
        "operational_rules": [
            "Check: does the answer match the granularity asked (year vs full date)?",
            "Check: is the answer grounded in the passages?",
        ],
        "behavioral_principles": [
            "Bias toward ACCEPT when grounded; bias toward REVISE when ungrounded.",
        ],
        "output_schema": '{"decision": "ACCEPT|REVISE", "answer": "..."}',
    },
    "ConcludeAgent": {
        "role": "Aggregate all upstream agent outputs and emit the final answer as a bare span.",
        "instructions": (
            "Read all upstream signals (decomposition, retrieval, evidence, validation, drafts, "
            "reflection). Output a SPAN: one entity, one date, or one short phrase. NEVER write "
            "a sentence. NEVER preface with 'The answer is'. MAX 8 words."
        ),
        "operational_rules": [
            "Resolve conflicts by trusting ReflectAgent if present, else AnswerGenerator.",
            "Output only the bare span, max 8 words; no preamble, no explanation, no period.",
            "If question asks 'what year' output ONLY the 4-digit year.",
            "If yes/no question output ONLY 'yes' or 'no'.",
        ],
        "behavioral_principles": [
            "Maintain answer granularity matching the question.",
            "Span only — never a clause, never a sentence.",
        ],
        "output_schema": '{"answer": "<bare span, max 8 words>"}',
    },
}


@dataclass
class AgentPrompt:
    """Mutable role-specific prompt. RoPE updates operational_rules and behavioral_principles."""
    name: str
    role: str
    instructions: str
    operational_rules: list[str] = field(default_factory=list)
    behavioral_principles: list[str] = field(default_factory=list)
    output_schema: str = ""

    def system_prompt(self) -> str:
        op = "\n".join(f"- {r}" for r in self.operational_rules) if self.operational_rules else "(none)"
        bp = "\n".join(f"- {r}" for r in self.behavioral_principles) if self.behavioral_principles else "(none)"
        return (
            f"You are the {self.name} agent in a multi-agent QA system.\n"
            f"Role: {self.role}\n\n"
            f"Instructions:\n{self.instructions}\n\n"
            f"Operational rules:\n{op}\n\n"
            f"Behavioral principles:\n{bp}\n\n"
            f"Output schema (JSON): {self.output_schema}\n"
            f"Always respond with valid JSON matching the schema. No commentary."
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentPrompt":
        return cls(**d)


def load_seed_prompts() -> dict[str, AgentPrompt]:
    out: dict[str, AgentPrompt] = {}
    for name, spec in SEED_PROMPTS.items():
        out[name] = AgentPrompt(
            name=name,
            role=spec["role"],
            instructions=spec["instructions"],
            operational_rules=list(spec.get("operational_rules", [])),
            behavioral_principles=list(spec.get("behavioral_principles", [])),
            output_schema=spec.get("output_schema", ""),
        )
    return out


def save_prompts(prompts: dict[str, AgentPrompt], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: v.to_dict() for k, v in prompts.items()}, indent=2, ensure_ascii=False))


def load_prompts(path: str | Path) -> dict[str, AgentPrompt]:
    p = Path(path)
    if not p.exists():
        return load_seed_prompts()
    raw = json.loads(p.read_text())
    return {k: AgentPrompt.from_dict(v) for k, v in raw.items()}


# ---------------- Agent execution ----------------

@dataclass
class AgentInvocation:
    name: str
    inputs: dict[str, Any]
    output: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def _format_context_for_agent(name: str, q: str, deps: dict[str, AgentInvocation],
                               passages: list[Passage] | None = None) -> str:
    """Build the user message for an agent given upstream outputs."""
    parts = [f"Question: {q}"]
    if "QueryDecomposer" in deps:
        sub = deps["QueryDecomposer"].output.get("sub_questions", [])
        if sub:
            parts.append("Sub-questions:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sub)))
    if "QueryRewriter" in deps:
        rw = deps["QueryRewriter"].output.get("queries", [])
        if rw:
            parts.append("Rewritten queries:\n" + "\n".join(f"  - {s}" for s in rw))
    if passages is not None:
        parts.append("Passages:\n" + format_passages(passages))
    if name == "EvidenceSelector":
        # passages already attached above
        pass
    if "EvidenceSelector" in deps and name not in ("EvidenceSelector",):
        sel = deps["EvidenceSelector"].output.get("selected_indices", [])
        if sel:
            parts.append(f"Selected passage indices: {sel}")
    if "ContextValidator" in deps:
        v = deps["ContextValidator"].output
        if v:
            parts.append(f"Context validation: {json.dumps(v)}")
    if name in ("ReflectAgent", "ConcludeAgent"):
        if "AnswerGenerator" in deps:
            ans = deps["AnswerGenerator"].output.get("answer", "")
            if ans:
                parts.append(f"Proposed answer: {ans}")
        if "ReflectAgent" in deps and name == "ConcludeAgent":
            r = deps["ReflectAgent"].output
            if r:
                parts.append(f"Reflection: {json.dumps(r)}")
    return "\n\n".join(parts)


async def run_llm_agent(prompt: AgentPrompt, q: str, deps: dict[str, AgentInvocation],
                         passages: list[Passage] | None, lm: OpenAIClient) -> AgentInvocation:
    user = _format_context_for_agent(prompt.name, q, deps, passages)
    inv = AgentInvocation(name=prompt.name, inputs={"question": q, "user_msg": user})
    try:
        res = await lm.chat(prompt.system_prompt(), user, json_mode=True)
        inv.raw_text = res.text
        inv.prompt_tokens = res.prompt_tokens
        inv.completion_tokens = res.completion_tokens
        parsed = parse_json_lenient(res.text)
        if isinstance(parsed, dict):
            inv.output = parsed
        elif isinstance(parsed, list):
            inv.output = {"items": parsed}
        else:
            inv.output = {}
    except Exception as e:
        inv.error = str(e)[:300]
        logger.warning("Agent %s failed: %s", prompt.name, inv.error)
    return inv


async def run_retriever_agent(q: str, deps: dict[str, AgentInvocation],
                              retriever: RetrieverClient, topk: int) -> tuple[AgentInvocation, list[Passage]]:
    """Tool agent: takes original q + (optional) rewritten queries; merges retrieval results."""
    queries = [q]
    if "QueryRewriter" in deps:
        rw = deps["QueryRewriter"].output.get("queries", [])
        for s in rw:
            if isinstance(s, str) and s.strip() and s not in queries:
                queries.append(s.strip())
    if "QueryDecomposer" in deps:
        sub = deps["QueryDecomposer"].output.get("sub_questions", [])
        for s in sub:
            if isinstance(s, str) and "{" not in s and s not in queries:
                queries.append(s.strip())
    queries = queries[:4]
    inv = AgentInvocation(name="Retriever", inputs={"queries": queries})
    try:
        results = await retriever.retrieve_batch(queries, topk=topk)
        # Flatten + dedupe by chunk_id, keep best score.
        seen: dict[str, Passage] = {}
        for plist in results:
            for p in plist:
                if p.chunk_id not in seen or p.score > seen[p.chunk_id].score:
                    seen[p.chunk_id] = p
        merged = sorted(seen.values(), key=lambda x: -x.score)[: max(topk, 5)]
        inv.output = {"num_passages": len(merged), "queries_used": queries}
        return inv, merged
    except Exception as e:
        inv.error = str(e)[:300]
        return inv, []
