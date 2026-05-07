"""Retriever client (E5 + wiki18 served on node408)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    chunk_id: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "text": self.text, "score": self.score}


class RetrieverClient:
    def __init__(self, url: str, topk: int = 5, concurrency: int = 8, timeout_s: float = 60.0):
        self.url = url
        self.topk = topk
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def retrieve(self, query: str, topk: int | None = None) -> list[Passage]:
        if not query.strip():
            return []
        body = {"queries": [query], "topk": topk or self.topk, "mode": "text"}
        async with self._sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_random_exponential(min=1, max=15),
                retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.post(self.url, json=body)
                    resp.raise_for_status()
                    data = resp.json()
        results = data.get("results", [[]])
        if not results:
            return []
        out: list[Passage] = []
        for r in results[0]:
            out.append(Passage(
                chunk_id=str(r.get("chunk_id", "")),
                text=str(r.get("text", "")),
                score=float(r.get("score", 0.0)),
            ))
        return out

    async def retrieve_batch(self, queries: list[str], topk: int | None = None) -> list[list[Passage]]:
        if not queries:
            return []
        body = {"queries": queries, "topk": topk or self.topk, "mode": "text"}
        async with self._sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_random_exponential(min=1, max=15),
                retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.post(self.url, json=body)
                    resp.raise_for_status()
                    data = resp.json()
        results = data.get("results", [])
        out: list[list[Passage]] = []
        for r in results:
            out.append([
                Passage(chunk_id=str(p.get("chunk_id", "")), text=str(p.get("text", "")),
                        score=float(p.get("score", 0.0)))
                for p in r
            ])
        return out

    async def aclose(self):
        await self._client.aclose()


def format_passages(passages: list[Passage], max_chars_per: int = 600) -> str:
    if not passages:
        return "(no passages retrieved)"
    lines = []
    for i, p in enumerate(passages, 1):
        t = p.text.replace("\n", " ").strip()
        if len(t) > max_chars_per:
            t = t[:max_chars_per] + "..."
        lines.append(f"[P{i}] {t}")
    return "\n".join(lines)
