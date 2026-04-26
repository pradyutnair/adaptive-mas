"""Retriever HTTP client for the node408 e5_Flat index.

Returns chunk-level hits. Subagents are responsible for reading and distilling;
no chunk text is ever exposed back to the orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalHit:
    """One result returned by the retriever."""

    chunk_id: str
    text: str
    score: float


class Retriever:
    """Async HTTP wrapper around the retriever REST API."""

    def __init__(
        self,
        base_url: str = "http://node408:8003",
        default_top_k: int = 10,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_top_k = int(default_top_k)
        self.timeout_seconds = float(timeout_seconds)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return await asyncio.to_thread(self._retrieve_sync, query, top_k)

    def _retrieve_sync(self, query: str, top_k: int | None) -> list[RetrievalHit]:
        payload = {
            "queries": [query],
            "topk": int(top_k or self.default_top_k),
            "mode": "text",
        }
        req = urllib.request.Request(
            f"{self.base_url}/retrieve",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw: dict[str, Any] = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("retriever request failed for %r: %s", query, exc)
            return []

        if not raw.get("success"):
            logger.warning("retriever returned failure: %s", raw)
            return []
        rows = raw.get("results") or []
        first = rows[0] if rows else []
        return [
            RetrievalHit(
                chunk_id=str(h.get("chunk_id", "")),
                text=str(h.get("text", "")),
                score=float(h.get("score", 0.0) or 0.0),
            )
            for h in first
        ]
