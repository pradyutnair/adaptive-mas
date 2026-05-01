"""Thin client for the wiki18 retriever at node408:8003."""

from __future__ import annotations

from typing import Any

import httpx

from .contracts import RetrievedChunk

_FIXED_TOPK = 5


class Retriever:
    def __init__(self, base_url: str = "http://node408:8003", timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    async def retrieve(self, query: str, k: int = _FIXED_TOPK) -> list[RetrievedChunk]:
        results = await self.retrieve_batch([query], k=k)
        return results[0] if results else []

    async def retrieve_batch(self, queries: list[str], k: int = _FIXED_TOPK) -> list[list[RetrievedChunk]]:
        if not queries:
            return []
        url = f"{self.base_url}/retrieve"
        payload: dict[str, Any] = {"queries": queries, "topk": k, "mode": "text"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        out: list[list[RetrievedChunk]] = []
        for batch in data.get("results", []):
            out.append([
                RetrievedChunk(
                    chunk_id=str(h.get("chunk_id", "")),
                    text=str(h.get("text", "")),
                    score=float(h.get("score", 0.0)),
                ) for h in batch
            ])
        return out
