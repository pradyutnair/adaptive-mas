"""Retrieval Probe Layer.

Issues N+1 retrievals in parallel (original Q + N candidate sub-Qs), each at
top-k=5. Computes per-probe groundedness signal. Zero LLM calls.
"""
from __future__ import annotations
import asyncio
from .retriever import Retriever
from .signals import compute_groundedness
from .types import ProbeResult


async def probe_all(
    *,
    retriever: Retriever,
    original_question: str,
    sub_questions: list[str],
) -> list[ProbeResult]:
    """Probe original Q and each sub-Q in parallel. Returns one ProbeResult per query.

    The first ProbeResult is for the original Q (label='original'); subsequent
    entries are for sub-Qs (label='sub_<i>').
    """
    queries = [original_question] + list(sub_questions)
    chunk_lists = await retriever.retrieve_batch(queries)
    out: list[ProbeResult] = []
    for i, (q, chunks) in enumerate(zip(queries, chunk_lists)):
        g, comp = compute_groundedness(q, chunks)
        label = 'original' if i == 0 else f'sub_{i-1}'
        out.append(ProbeResult(
            label=label,
            query=q,
            chunks=chunks,
            top1_score=comp['top1_score'],
            score_gap_1to5=comp['score_gap_1to5'],
            ne_coverage=comp['ne_coverage'],
            wh_target_extractable=bool(comp['wh_target_extractable']),
            groundedness=g,
        ))
    return out
