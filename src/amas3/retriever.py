"""Thin client for the wiki18 retriever (node408:8003).

Always top-k=5 per call. Re-retrieval is a separate call with a different
query (not a wider K). All baselines and HERA / SPARC-RAG / Plan*RAG also
use top-k=5; we preserve fairness.
"""
from __future__ import annotations
import asyncio
from typing import Any
import httpx
from .types import RetrievedChunk

_FIXED_TOPK = 5


class Retriever:
    def __init__(self, base_url: str = 'http://node408:8003', timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout_seconds = float(timeout_seconds)

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Single retrieval call at top-k=5."""
        results = await self.retrieve_batch([query])
        return results[0]

    async def retrieve_batch(self, queries: list[str]) -> list[list[RetrievedChunk]]:
        """Batched parallel retrieval. The retrieval server accepts multiple queries."""
        if not queries:
            return []
        url = f'{self.base_url}/retrieve'
        payload: dict[str, Any] = {'queries': queries, 'topk': _FIXED_TOPK, 'mode': 'text'}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        out = []
        for batch in data.get('results', []):
            chunks = []
            for h in batch:
                chunk_id = h.get('chunk_id', h.get('id', h.get('doc_id', '')))
                chunks.append(
                    RetrievedChunk(
                        chunk_id=str(chunk_id),
                        text=h.get('text', h.get('paragraph_text', h.get('contents', ''))),
                        score=float(h.get('score', 0.0)),
                    )
                )
            out.append(chunks)
        return out
