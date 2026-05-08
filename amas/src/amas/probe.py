"""Turn-0 probe: top-k retrieve + G self-consistency single-shot answers + ledger seed.

Drives the SAS lane (gate may exit at turn 0) and seeds Belief State for both gates.
Cheap: G=3 GPT-4o-mini calls per question + one batched retrieval.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .ledger import BeliefState, Ledger, parse_stance_from_agent
from .lm import OpenAIClient, parse_json_lenient
from .retriever import RetrieverClient, format_passages


PROBE_SYSTEM = (
    "You are a single-shot QA agent. Read the question and the retrieved passages, then output "
    "a SHORT ANSWER SPAN. One entity, one date, one short phrase, or yes/no. NEVER write a "
    "sentence. NEVER explain. Cap at 8 words. Bare span only — no quotes, no period, no preamble."
)


def build_probe_user(question: str, passages_block: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Passages:\n{passages_block}\n\n"
        "Respond ONLY with valid JSON: "
        '{"answer": "<bare span, max 8 words>", "confidence": <float 0..1>}'
    )


@dataclass
class ProbeResult:
    samples: list[dict[str, Any]]
    consensus_answer: str
    consensus_count: int
    total_tokens: int
    passage_ids: list[str]


async def run_probe(question: str, *, retriever: RetrieverClient, openai_client: OpenAIClient,
                    ledger: Ledger, belief: BeliefState,
                    topk: int = 5, group_size: int = 3, temperature: float = 0.7,
                    turn: int = 0) -> ProbeResult:
    passages = await retriever.retrieve(question, topk=topk)
    pid_list = ledger.add_passages(turn=turn, source_agent="Probe.Retriever", passages=passages)

    block = format_passages(passages, max_chars_per=600)
    user = build_probe_user(question, block)

    async def one_sample() -> tuple[str, float, int]:
        res = await openai_client.chat(PROBE_SYSTEM, user, temperature=temperature, json_mode=True)
        parsed = parse_json_lenient(res.text)
        ans = ""
        conf = 0.5
        if isinstance(parsed, dict):
            ans = str(parsed.get("answer", "")).strip()
            try:
                conf = float(parsed.get("confidence", 0.5))
            except Exception:
                conf = 0.5
        return ans, conf, res.prompt_tokens + res.completion_tokens

    coros = [one_sample() for _ in range(group_size)]
    results = await asyncio.gather(*coros, return_exceptions=False)
    total_tokens = sum(t for _, _, t in results)
    answers = [a for a, _, _ in results if a]

    # Consensus by normalized-answer majority.
    from .metric import normalize_answer
    norm_counts = Counter(normalize_answer(a) for a in answers if a)
    cons_norm, cons_count = (norm_counts.most_common(1)[0] if norm_counts else ("", 0))
    consensus_answer = ""
    if cons_count:
        for a in answers:
            from .metric import normalize_answer as _na
            if _na(a) == cons_norm:
                consensus_answer = a
                break

    samples = [{"answer": a, "confidence": c, "tokens": t} for a, c, t in results]

    # Seed belief state from samples (each sample = support).
    avg_conf = sum(c for _, c, _ in results) / max(1, len(results))
    for a, c, _ in results:
        if a:
            belief.update_from_answer(a, support=float(c), evidence_ids=pid_list[:3])

    # Ledger entry summarizing probe consensus.
    if consensus_answer:
        ledger.add(
            turn=turn, source_agent="Probe", claim=f"Probe answer: {consensus_answer}",
            passage_ids=pid_list[:3], stance="support",
            confidence=min(1.0, avg_conf * (cons_count / max(1, group_size))),
        )

    return ProbeResult(
        samples=samples,
        consensus_answer=consensus_answer,
        consensus_count=int(cons_count),
        total_tokens=int(total_tokens),
        passage_ids=pid_list,
    )
