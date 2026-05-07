"""LLM clients: vLLM (Qwen3-14B for orchestrator/library/RoPE) + OpenAI (gpt-4o-mini for subagents)."""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import random
import time
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class LMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_s: float = 0.0


class VLLMClient:
    """Round-robin async client for multiple vLLM endpoints (chat completions)."""

    def __init__(self, endpoints: list[str], model: str, max_tokens: int = 1024,
                 temperature: float = 0.7, concurrency: int = 12, timeout_s: float = 120.0):
        self.endpoints = list(endpoints)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sem = asyncio.Semaphore(concurrency)
        self._cycle = itertools.cycle(self.endpoints)
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def _next_endpoint(self) -> str:
        async with self._lock:
            return next(self._cycle)

    async def chat(self, system: str, user: str, *, temperature: float | None = None,
                   max_tokens: int | None = None, stop: list[str] | None = None,
                   extra_body: dict | None = None) -> LMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if stop:
            body["stop"] = stop
        # Disable Qwen3 thinking by default for latency.
        if extra_body:
            body.update(extra_body)
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        async with self._sem:
            t0 = time.time()
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_random_exponential(min=1, max=20),
                retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    endpoint = await self._next_endpoint()
                    resp = await self._client.post(f"{endpoint}/chat/completions", json=body)
                    if resp.status_code >= 500:
                        raise httpx.HTTPError(f"{resp.status_code}: {resp.text[:200]}")
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    usage = data.get("usage", {}) or {}
                    return LMResult(
                        text=text,
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        model=self.model,
                        latency_s=time.time() - t0,
                    )

    async def aclose(self):
        await self._client.aclose()


class OpenAIClient:
    """OpenAI async client for subagents."""

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 768,
                 temperature: float = 0.3, concurrency: int = 12):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sem = asyncio.Semaphore(concurrency)

    async def chat(self, system: str, user: str, *, temperature: float | None = None,
                   max_tokens: int | None = None, json_mode: bool = False) -> LMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        async with self._sem:
            t0 = time.time()
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_random_exponential(min=1, max=30),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    try:
                        resp = await self._client.chat.completions.create(**kwargs)
                    except Exception as e:
                        msg = str(e).lower()
                        # Don't retry on auth / invalid request.
                        if "authentication" in msg or "api key" in msg or "invalid" in msg and "request" in msg:
                            raise
                        raise
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    return LMResult(
                        text=text,
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                        model=self.model,
                        latency_s=time.time() - t0,
                    )

    async def aclose(self):
        await self._client.close()


def parse_json_lenient(text: str) -> dict | list:
    """Best-effort JSON extraction. Strips markdown fences. Returns {} on failure."""
    if not text:
        return {}
    s = text.strip()
    # Strip markdown code fences
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    # Find first { or [
    for opener, closer in [("{", "}"), ("[", "]")]:
        if opener in s:
            start = s.find(opener)
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                c = s[i]
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start:i + 1])
                        except Exception:
                            break
            break
    try:
        return json.loads(s)
    except Exception:
        return {}
