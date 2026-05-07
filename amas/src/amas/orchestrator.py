"""Orchestrator: topology sampling + execution. Qwen3-14B for orchestrator/library/RoPE prompts."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .agents import (
    AGENT_NAMES,
    AgentInvocation,
    AgentPrompt,
    SEED_PROMPTS,
    run_llm_agent,
    run_retriever_agent,
)
from .library import ExperienceLibrary, profile_question
from .lm import LMResult, OpenAIClient, VLLMClient, parse_json_lenient
from .retriever import Passage, RetrieverClient, format_passages

logger = logging.getLogger(__name__)


# Default topology used when orchestrator output is invalid (safe fallback).
FALLBACK_TOPOLOGY = {
    "query_profile": "bridge",
    "selected_agents": ["QueryDecomposer", "Retriever", "EvidenceSelector", "AnswerGenerator", "ConcludeAgent"],
    "execution_order": [
        {"step": 1, "agent": "QueryDecomposer", "depends_on": [], "mode": "sequential"},
        {"step": 2, "agent": "Retriever", "depends_on": ["QueryDecomposer"], "mode": "sequential"},
        {"step": 3, "agent": "EvidenceSelector", "depends_on": ["Retriever"], "mode": "sequential"},
        {"step": 4, "agent": "AnswerGenerator", "depends_on": ["EvidenceSelector", "Retriever"], "mode": "sequential"},
        {"step": 5, "agent": "ConcludeAgent", "depends_on": ["AnswerGenerator"], "mode": "sequential"},
    ],
}


def agent_descriptions() -> str:
    lines = []
    for name in AGENT_NAMES:
        spec = SEED_PROMPTS[name]
        lines.append(f"- {name}: {spec['role']}")
    return "\n".join(lines)


# Verbatim Appendix B — "Orchestrator: topology sampling".
ORCHESTRATOR_SYSTEM = (
    "You are an orchestrator managing a team of specialized agents for multi-hop question "
    "answering. Your goal is to design an effective multi-agent execution topology to answer "
    "the given query correctly."
)


def build_orchestrator_user(query: str, profile: str, library_text: str,
                              mutation_hint: str = "") -> str:
    base = (
        "You have access to the following agents:\n"
        f"{agent_descriptions()}\n\n"
        "You have retrieved the following relevant experiences from past executions:\n"
        f"{library_text or '(none)'}\n\n"
        "Each experience entry has the format:\n"
        "• Query Type: {query type}\n"
        "• Insight: {insight}\n"
        "• Utility score: {utility}\n\n"
        "Given the query below, design a coordination topology by:\n"
        "1. Selecting the subset of agents best suited to this query type.\n"
        "2. Specifying their execution order (sequential or parallel where appropriate).\n"
        "3. Defining dependency relationships (which agent's output feeds into which).\n\n"
        f"Query: {query}\n\n"
        "Respond in the following JSON format:\n"
        "{\n"
        '  "query_profile": "<one-sentence characterization of query type>",\n'
        '  "selected_agents": ["<agent_name>", ...],\n'
        '  "execution_order": [\n'
        '    {"step": 1, "agent": "<agent_name>", "depends_on": [],\n'
        '     "mode": "sequential|parallel"},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )
    if mutation_hint:
        # §3.5 topology mutation: still verbatim base prompt, append mutation feedback.
        base += (
            "\n\nTopology mutation feedback (prior topologies all produced F1=0):\n"
            f"{mutation_hint}\n"
            "Propose a structurally different topology: replace the failing agent or augment "
            "with additional agents."
        )
    return base


def validate_topology(topo: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and normalize the orchestrator's JSON topology. Returns None if invalid."""
    if not isinstance(topo, dict):
        return None
    selected = topo.get("selected_agents") or []
    order = topo.get("execution_order") or []
    if not isinstance(selected, list) or not isinstance(order, list):
        return None
    if not order or len(order) > 8:
        return None
    selected = [s for s in selected if s in AGENT_NAMES]
    seen_steps: dict[int, list[dict]] = {}
    cleaned: list[dict] = []
    for item in order:
        if not isinstance(item, dict):
            return None
        agent = item.get("agent")
        if agent not in AGENT_NAMES:
            return None
        if agent not in selected:
            selected.append(agent)
        try:
            step = int(item.get("step", len(cleaned) + 1))
        except Exception:
            step = len(cleaned) + 1
        deps_raw = item.get("depends_on") or []
        if not isinstance(deps_raw, list):
            deps_raw = []
        deps = [d for d in deps_raw if d in AGENT_NAMES]
        mode = item.get("mode", "sequential")
        if mode not in ("sequential", "parallel"):
            mode = "sequential"
        cleaned.append({"step": step, "agent": agent, "depends_on": deps, "mode": mode})
        seen_steps.setdefault(step, []).append(cleaned[-1])
    cleaned.sort(key=lambda x: (x["step"], x["agent"]))
    # Enforce dep ordering (deps must appear in earlier step).
    seen_agents: set[str] = set()
    last_step = -1
    for item in cleaned:
        if item["step"] < last_step:
            return None
        if item["step"] != last_step:
            # Promote pending agents from prior step into seen.
            last_step = item["step"]
        for d in item["depends_on"]:
            if d not in seen_agents:
                # dep not yet executed; relax by removing it (still valid topo)
                item["depends_on"] = [x for x in item["depends_on"] if x in seen_agents]
                break
        seen_agents.add(item["agent"])
    # Must have a terminal answering agent.
    answering = [a for a in selected if a in ("AnswerGenerator", "ConcludeAgent")]
    if not answering:
        cleaned.append({
            "step": (cleaned[-1]["step"] if cleaned else 0) + 1,
            "agent": "ConcludeAgent",
            "depends_on": [item["agent"] for item in cleaned[-2:]],
            "mode": "sequential",
        })
        selected.append("ConcludeAgent")
    return {
        "query_profile": topo.get("query_profile", "bridge"),
        "selected_agents": selected,
        "execution_order": cleaned,
    }


@dataclass
class Trajectory:
    query: str
    gold: str | list[str]
    profile: str
    topology: dict[str, Any]
    invocations: list[AgentInvocation] = field(default_factory=list)
    answer: str = ""
    f1: float = 0.0
    em: float = 0.0
    contain: float = 0.0
    acc: float = 0.0
    total_tokens: int = 0
    elapsed_s: float = 0.0
    used_insight_ids: list[str] = field(default_factory=list)
    error: str | None = None
    failed_agent: str | None = None  # for RoPE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["invocations"] = [
            {
                "name": inv.name,
                "output": inv.output,
                "raw_text_preview": (inv.raw_text or "")[:300],
                "prompt_tokens": inv.prompt_tokens,
                "completion_tokens": inv.completion_tokens,
                "error": inv.error,
            }
            for inv in self.invocations
        ]
        return d

    def short_repr(self) -> str:
        seq = " -> ".join(inv.name for inv in self.invocations)
        return f"[{seq}] ans='{self.answer[:80]}' F1={self.f1:.3f} tok={self.total_tokens}"


def normalize_answer_span(answer: str, question: str = "", max_words: int = 8) -> str:
    """Strip verbose prose so EM/F1 metrics measure the actual span.

    Rules (cheap heuristics, deterministic):
    - Strip surrounding quotes / trailing period.
    - Drop common preambles: "The answer is X", "X is the ...", "X was the ...".
    - For "what year" questions, extract first 4-digit year.
    - For yes/no questions, return first matching token.
    - Cap to `max_words`.
    """
    import re as _re
    if not answer:
        return ""
    ans = str(answer).strip().strip("\"' \t\n").rstrip(".")
    ql = (question or "").lower()

    # Yes/no shortcut
    if any(p in ql for p in ("is it true", "true or false", "yes or no")) or ql.startswith(
        ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ")
    ):
        m = _re.search(r"\b(yes|no)\b", ans, _re.I)
        if m:
            return m.group(1).lower()

    # "what year" -> first 4-digit year
    if "what year" in ql or "which year" in ql or "in what year" in ql:
        m = _re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", ans)
        if m:
            return m.group(1)

    # Drop "The answer is " / "Answer: " / leading "{Entity} is the ..."
    ans = _re.sub(r"^(the\s+)?answer\s+(is|was|would\s+be)\s+", "", ans, flags=_re.I)
    ans = _re.sub(r"^(the\s+)?answer\s*[:\-]\s*", "", ans, flags=_re.I)
    ans = _re.sub(r"^(it|this|that)\s+(is|was)\s+", "", ans, flags=_re.I)
    ans = _re.sub(r"^in\s+(the\s+year\s+)?", "", ans, flags=_re.I)

    # "X is the Y" / "X was the Y" -> X (only when "X" is capitalized noun phrase)
    m = _re.match(r"^([A-Z][\w\.\-' ]{1,60}?)\s+(is|was|are|were)\s+(the|a|an|one of)\s+", ans)
    if m:
        ans = m.group(1).strip()

    # "X, which/who ..." / "X (..." -> X
    ans = _re.split(r",\s+(which|who|that|where|when)\b", ans, maxsplit=1, flags=_re.I)[0]
    ans = _re.split(r"\s*\(", ans, maxsplit=1)[0]

    # Cap word count
    words = ans.split()
    if len(words) > max_words:
        ans = " ".join(words[:max_words])

    return ans.strip().rstrip(".,;:")


def extract_final_answer(invs: list[AgentInvocation], question: str = "") -> str:
    """Final answer = last non-empty answer from ConcludeAgent > ReflectAgent > AnswerGenerator.

    Normalized via heuristic post-process unless `AMAS_DISABLE_SPAN_NORMALIZE=1`.
    """
    import os as _os
    by_name = {inv.name: inv for inv in invs}
    raw = ""
    for name in ("ConcludeAgent", "ReflectAgent", "AnswerGenerator"):
        inv = by_name.get(name)
        if inv:
            raw = (inv.output.get("answer") or "").strip()
            if raw:
                break
    if not raw:
        for inv in reversed(invs):
            r = (inv.output.get("answer") or "").strip() if isinstance(inv.output, dict) else ""
            if r:
                raw = r
                break
    if _os.getenv("AMAS_DISABLE_SPAN_NORMALIZE", "").lower() in ("1", "true", "yes"):
        return raw
    return normalize_answer_span(raw, question=question)


class Orchestrator:
    def __init__(self, vllm: VLLMClient, openai_client: OpenAIClient,
                 retriever: RetrieverClient, library: ExperienceLibrary,
                 prompts: dict[str, AgentPrompt],
                 retriever_topk: int = 5, library_top_k: int = 5,
                 max_steps: int = 8):
        self.vllm = vllm
        self.openai = openai_client
        self.retriever = retriever
        self.library = library
        self.prompts = prompts
        self.retriever_topk = retriever_topk
        self.library_top_k = library_top_k
        self.max_steps = max_steps

    async def sample_topology(self, query: str, profile: str, *, temperature: float = 0.9,
                                mutation_hint: str = "") -> tuple[dict[str, Any], list[str], LMResult]:
        retrieved = self.library.retrieve(profile, top_k=self.library_top_k)
        ids = [e.id for e in retrieved]
        text = self.library.to_paper_format(profile=profile, top_k=self.library_top_k) if retrieved else ""
        user = build_orchestrator_user(query, profile, text, mutation_hint=mutation_hint)
        res = await self.vllm.chat(ORCHESTRATOR_SYSTEM, user, temperature=temperature, max_tokens=900)
        parsed = parse_json_lenient(res.text)
        topo = validate_topology(parsed) if isinstance(parsed, dict) else None
        if not topo:
            topo = json.loads(json.dumps(FALLBACK_TOPOLOGY))
            topo["query_profile"] = profile
        return topo, ids, res

    async def execute(self, query: str, gold: str | list[str], topology: dict[str, Any],
                       insight_ids: list[str], orch_tokens: int = 0,
                       ctx: dict[str, Any] | None = None) -> Trajectory:
        from .metric import accuracy, contain, exact_match, f1_score

        t0 = time.time()
        order = topology["execution_order"]
        # Group steps.
        steps: dict[int, list[dict]] = {}
        for item in order:
            steps.setdefault(item["step"], []).append(item)

        # Cross-turn context for LLM agents (Stage 1: ledger + belief wiring).
        # When `ctx` is None or empty, agents see no extra prefix — backwards-compatible.
        _ctx = ctx or {}
        _ledger_text = str(_ctx.get("__ledger__", "") or "")
        _belief_text = str(_ctx.get("__belief__", "") or "")

        invocations: list[AgentInvocation] = []
        by_name: dict[str, AgentInvocation] = {}
        passages: list[Passage] = []
        total_tokens = orch_tokens
        traj = Trajectory(query=query, gold=gold, profile=topology.get("query_profile", "bridge"),
                          topology=topology, used_insight_ids=insight_ids)

        for step in sorted(steps.keys()):
            items = steps[step]

            async def _run_one(item: dict) -> AgentInvocation:
                agent_name = item["agent"]
                deps_subset = {d: by_name[d] for d in item["depends_on"] if d in by_name}
                if agent_name == "Retriever":
                    inv, ps = await run_retriever_agent(query, deps_subset, self.retriever,
                                                         self.retriever_topk)
                    nonlocal_ref["passages"] = ps
                    return inv
                pr = self.prompts[agent_name]
                inv = await run_llm_agent(pr, query, deps_subset,
                                            nonlocal_ref["passages"] if agent_name in
                                            ("EvidenceSelector", "ContextValidator", "AnswerGenerator",
                                             "ReflectAgent", "ConcludeAgent") else None,
                                            self.openai,
                                            ledger_text=_ledger_text, belief_text=_belief_text)
                return inv

            # We need a mutable container for `passages` inside parallel calls.
            nonlocal_ref = {"passages": passages}
            invs = await asyncio.gather(*[_run_one(it) for it in items], return_exceptions=False)
            passages = nonlocal_ref["passages"]
            for inv in invs:
                invocations.append(inv)
                by_name[inv.name] = inv
                total_tokens += inv.prompt_tokens + inv.completion_tokens

            # Optional: if ContextValidator says insufficient and we have a follow_up_query,
            # do one extra retrieval round.
            cv = by_name.get("ContextValidator")
            if cv and cv.output.get("sufficient") is False and cv.output.get("follow_up_query"):
                fu = str(cv.output["follow_up_query"]).strip()
                if fu:
                    try:
                        more = await self.retriever.retrieve(fu, topk=self.retriever_topk)
                        seen = {p.chunk_id for p in passages}
                        for p in more:
                            if p.chunk_id not in seen:
                                passages.append(p)
                    except Exception:
                        pass

        traj.invocations = invocations
        traj.answer = extract_final_answer(invocations, question=query)
        traj.f1 = f1_score(traj.answer, gold)
        traj.em = exact_match(traj.answer, gold)
        traj.contain = contain(traj.answer, gold)
        traj.acc = max(traj.em, traj.contain)
        traj.total_tokens = total_tokens
        traj.elapsed_s = time.time() - t0
        return traj
