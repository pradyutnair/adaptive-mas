"""Retriever HTTP client for AMAS-compatible retrieval servers.

Supports both node408 and ``scripts/baseline_compat_retriever.py`` responses.
Investigators distill retrieval hits into compact evidence excerpts; raw
retrieval text is not exposed back to the orchestrator.
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
    snippets: list[str] | None = None


class Retriever:
    """Async HTTP wrapper around the retriever REST API."""

    def __init__(
        self,
        base_url: str = "http://node408:8003",
        default_top_k: int = 10,
        timeout_seconds: float = 30.0,
        request_format: str = "batch",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_top_k = int(default_top_k)
        self.timeout_seconds = float(timeout_seconds)
        self.request_format = request_format

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return await asyncio.to_thread(self._retrieve_sync, query, top_k)

    def _retrieve_sync(self, query: str, top_k: int | None) -> list[RetrievalHit]:
        k = int(top_k or self.default_top_k)
        if self.request_format == "adaptive_rag":
            payload = {"query_text": query, "max_hits_count": k}
        else:
            payload = {"queries": [query], "topk": k, "mode": "text"}
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
        first = self._extract_hits(raw)
        hits: list[RetrievalHit] = []
        for h in first:
            chunk_id = h.get("chunk_id", h.get("id", ""))
            text = h.get("text", h.get("paragraph_text", ""))
            if chunk_id == "" or text is None:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    score=float(h.get("score", 0.0) or 0.0),
                    snippets=[
                        str(s) for s in h.get("matched_sentences", [])
                        if str(s).strip()
                    ] or None,
                )
            )
        return hits

    @staticmethod
    def _extract_hits(raw: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(raw.get("retrieval"), list):
            return [h for h in raw["retrieval"] if isinstance(h, dict)]
        rows = raw.get("results") or []
        if rows and isinstance(rows[0], list):
            return [h for h in rows[0] if isinstance(h, dict)]
        if isinstance(rows, list):
            return [h for h in rows if isinstance(h, dict)]
        return []
