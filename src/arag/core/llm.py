"""LLM client for ARAG - unified interface for OpenAI-compatible APIs."""

import os
from typing import Any, Dict, List, Optional

import aiohttp
import requests
import tiktoken


class TokenBudgetExceededError(RuntimeError):
    """Raised when a per-question token budget cannot accommodate a call."""


class LLMClient:
    """Unified LLM client for OpenAI-compatible APIs."""
    
    # Official pricing (USD per 1M tokens): (input, cached_input, output)
    PRICING = {
        # OpenAI GPT-5 series
        "gpt-5.2-pro": (21.0, 2.1, 168.0),
        "gpt-5.2": (1.75, 0.175, 14.0),
        "gpt-5.1": (1.5, 0.15, 12.0),
        "gpt-5-mini": (0.25, 0.025, 2.0),
        "gpt-5": (1.5, 0.15, 12.0),
        # OpenAI GPT-4 series
        "gpt-4.1-mini": (0.4, 0.1, 1.6),
        "gpt-4.1-nano": (0.1, 0.025, 0.4),
        "gpt-4.1": (2.0, 0.5, 8.0),
        "gpt-4o-mini": (0.15, 0.075, 0.6),
        "gpt-4o": (2.5, 1.25, 10.0),
        "gpt-4-turbo": (10.0, 5.0, 30.0),
        # OpenAI o-series
        "o4-mini": (1.1, 0.275, 4.4),
        "o3-mini": (1.1, 0.275, 4.4),
        "o3": (10.0, 2.5, 40.0),
        "o1-mini": (1.1, 0.275, 4.4),
        "o1": (15.0, 3.75, 60.0),
        # Anthropic Claude 4 series
        "claude-4.5-opus": (5.0, 0.5, 25.0),
        "claude-4.5-sonnet": (3.0, 0.3, 15.0),
        "claude-4.5-haiku": (1.0, 0.1, 5.0),
        "claude-4-opus": (15.0, 1.5, 75.0),
        "claude-4-sonnet": (3.0, 0.3, 15.0),
        "claude-sonnet": (3.0, 0.3, 15.0),
        "claude-opus": (5.0, 0.5, 25.0),
        "claude-haiku": (1.0, 0.1, 5.0),
        # Google Gemini 3 series
        "gemini-3-pro": (2.0, 0.2, 12.0),
        "gemini-3-flash": (0.5, 0.05, 3.0),
        # Google Gemini 2.5 series
        "gemini-2.5-pro": (1.25, 0.125, 10.0),
        "gemini-2.5-flash": (0.3, 0.075, 2.5),
        # Google Gemini 2.0 series
        "gemini-2.0-flash": (0.1, 0.025, 0.4),
        "gemini-pro": (1.25, 0.125, 5.0),
        "gemini-flash": (0.075, 0.02, 0.3),
        # Default fallback
        "default": (1.0, 0.1, 5.0),
    }
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        reasoning_effort: str = None,
        chat_template_kwargs: Dict[str, Any] = None,
    ):
        self.model = model or os.getenv("ARAG_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("ARAG_API_KEY")
        self.base_url = (base_url or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1")).rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.chat_template_kwargs = chat_template_kwargs
        self.context_window = int(os.getenv("ARAG_CONTEXT_WINDOW", "16384"))
        self.max_completion_cap = int(os.getenv("ARAG_MAX_COMPLETION_TOKENS", "1024"))
        
        if not self.api_key:
            raise ValueError("API key required. Set ARAG_API_KEY environment variable or pass api_key parameter.")
        
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_tokens(str(tc.get("function", {})))
        return total

    def _resolve_max_tokens(
        self,
        messages: List[Dict[str, Any]],
        requested_max_tokens: Optional[int] = None,
        remaining_total_tokens: Optional[int] = None,
    ) -> int:
        requested = requested_max_tokens or self.max_tokens
        prompt_tokens = self.count_message_tokens(messages)
        # Keep a small safety margin because chat token counting is approximate.
        safety_margin = 128
        available = max(self.context_window - prompt_tokens - safety_margin, 1)
        if remaining_total_tokens is not None:
            remaining_total_tokens = int(remaining_total_tokens)
            remaining_for_completion = remaining_total_tokens - prompt_tokens - safety_margin
            if remaining_for_completion < 1:
                raise TokenBudgetExceededError(
                    "Token budget exhausted before LLM call "
                    f"(remaining={remaining_total_tokens}, prompt≈{prompt_tokens})"
                )
            available = min(available, remaining_for_completion)
        return max(1, min(requested, self.max_completion_cap, available))
    
    def calculate_cost(self, usage: dict) -> float:
        """Calculate API cost from usage info."""
        model_lower = self.model.lower()
        
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        prompt_details = usage.get('prompt_tokens_details', {}) or {}
        cached_tokens = prompt_details.get('cached_tokens', 0)
        input_tokens = max(prompt_tokens - cached_tokens, 0)
        
        # Find matching pricing
        for key in self.PRICING:
            if key in model_lower:
                input_price, cached_price, output_price = self.PRICING[key]
                break
        else:
            input_price, cached_price, output_price = self.PRICING["default"]
        
        usd_cost = (
            (input_tokens / 1_000_000) * input_price +
            (cached_tokens / 1_000_000) * cached_price +
            (completion_tokens / 1_000_000) * output_price
        )
        
        return round(usd_cost, 6)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
        chat_template_kwargs: Dict[str, Any] = None,
        remaining_total_tokens: int | None = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self._resolve_max_tokens(
                messages,
                max_tokens,
                remaining_total_tokens=remaining_total_tokens,
            ),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        merged_chat_template_kwargs = self.chat_template_kwargs
        if chat_template_kwargs is not None:
            merged_chat_template_kwargs = chat_template_kwargs
        if merged_chat_template_kwargs:
            payload["chat_template_kwargs"] = merged_chat_template_kwargs
        
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        
        usage = result.get("usage", {})
        
        return {
            "message": result["choices"][0]["message"],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost": self.calculate_cost(usage),
            "raw_response": result,
        }
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        **kwargs
    ) -> tuple:
        """Generate response (compatibility method for eval script)."""
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        result = self.chat(messages=messages, tools=tools, temperature=temperature)
        content = result["message"].get("content", "")
        cost = result["cost"]
        return content, cost

    async def async_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
        chat_template_kwargs: Dict[str, Any] = None,
        _session: "aiohttp.ClientSession | None" = None,
        remaining_total_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Async version of chat() using aiohttp for non-blocking HTTP."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self._resolve_max_tokens(
                messages,
                max_tokens,
                remaining_total_tokens=remaining_total_tokens,
            ),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        merged_chat_template_kwargs = self.chat_template_kwargs
        if chat_template_kwargs is not None:
            merged_chat_template_kwargs = chat_template_kwargs
        if merged_chat_template_kwargs:
            payload["chat_template_kwargs"] = merged_chat_template_kwargs

        timeout = aiohttp.ClientTimeout(total=600)
        own_session = _session is None
        session = _session or aiohttp.ClientSession(timeout=timeout)
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
        finally:
            if own_session:
                await session.close()

        usage = result.get("usage", {})
        return {
            "message": result["choices"][0]["message"],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost": self.calculate_cost(usage),
            "raw_response": result,
        }
