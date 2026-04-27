"""Modular LLM client supporting OpenAI-compatible APIs and vLLM.

Design:
- One client per agent role (orchestrator, investigator, synthesizer).
- Each role can use a different model/provider/temperature/thinking setting.
- Token counts come directly from the API `usage` field, never estimated.

Examples
--------
Orchestrator on Qwen3 with thinking, investigator on GPT-4o-mini::

    orch = LLMClient(model="Qwen/Qwen3-8B", base_url="http://localhost:8001/v1",
                     enable_thinking=True, temperature=0.0)
    inv  = LLMClient(model="gpt-4o-mini", base_url="https://api.openai.com/v1",
                     api_key=os.environ["OPENAI_API_KEY"], temperature=0.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
from dataclasses import dataclass
from typing import Any

import aiohttp
import certifi

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Result of a single LLM chat call."""

    content: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCall]
    raw_message: dict[str, Any]

    @property
    def total_tokens(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)


class LLMClient:
    """OpenAI-compatible chat client (works with vLLM and OpenAI)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.timeout_seconds = timeout_seconds

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> LLMResponse:
        """Issue a single chat completion request (with optional tool use)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": int(max_tokens or self.max_tokens),
        }
        if self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        own = session is None
        if own:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            sess = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                connector=connector,
            )
        else:
            sess = session
        result = None
        try:
            for attempt in range(4):
                try:
                    async with sess.post(url, headers=headers, json=payload) as resp:
                        resp.raise_for_status()
                        result = await resp.json()
                        break
                except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                    retryable = isinstance(exc, (aiohttp.ClientConnectionError, asyncio.TimeoutError))
                    if isinstance(exc, aiohttp.ClientResponseError):
                        retryable = exc.status in {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
                    if not retryable or attempt >= 3:
                        raise
                    await asyncio.sleep(1.5 * (2 ** attempt))
        finally:
            if own:
                await sess.close()
        if result is None:
            raise RuntimeError("LLM request failed without a response")

        usage = result.get("usage", {}) or {}
        message = result["choices"][0]["message"] or {}
        content = message.get("content", "") or ""
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        return LLMResponse(
            content=content,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            tool_calls=tool_calls,
            raw_message=message,
        )

    @staticmethod
    def _parse_tool_calls(raw: Any) -> list[ToolCall]:
        if not isinstance(raw, list):
            return []
        out: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") or {}
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            out.append(ToolCall(
                id=str(item.get("id", "")),
                name=str(fn.get("name", "")),
                arguments=args,
            ))
        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LLMClient:
        """Construct from a flat config dict."""
        api_key_env = cfg.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else cfg.get("api_key")
        return cls(
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=api_key,
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 1024)),
            enable_thinking=bool(cfg.get("enable_thinking", False)),
            timeout_seconds=float(cfg.get("timeout_seconds", 600.0)),
        )


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from Qwen3 thinking output."""
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse JSON object from raw model text; tolerate prose and code fences."""
    import re

    text = strip_thinking(text)
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        try:
            v = json.loads(fence.group(1))
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            v = json.loads(brace.group())
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    return {}
