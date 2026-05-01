from __future__ import annotations

import os
from dataclasses import dataclass

import dspy


@dataclass
class LMConfig:
    qwen_base_urls: tuple[str, ...] = ("http://localhost:8001/v1", "http://localhost:8002/v1", "http://localhost:8003/v1")
    qwen_model: str = "Qwen/Qwen3-14B"
    qwen_think_max_tokens: int = 4096
    qwen_nothink_max_tokens: int = 512
    qwen_think_temperature: float = 0.6
    qwen_nothink_temperature: float = 0.0
    mini_model: str = "openai/gpt-4o-mini"
    mini_max_tokens: int = 1024
    mini_temperature: float = 0.0


def _qwen(*, cfg: LMConfig, replica_idx: int, thinking: bool, temperature: float | None, max_tokens: int | None) -> dspy.LM:
    base = cfg.qwen_base_urls[replica_idx % len(cfg.qwen_base_urls)]
    return dspy.LM(
        model=f"hosted_vllm/{cfg.qwen_model}",
        api_base=base,
        api_key="EMPTY",
        max_tokens=max_tokens if max_tokens is not None else (cfg.qwen_think_max_tokens if thinking else cfg.qwen_nothink_max_tokens),
        temperature=temperature if temperature is not None else (cfg.qwen_think_temperature if thinking else cfg.qwen_nothink_temperature),
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        cache=False,
    )


def make_qwen_think_lm(cfg: LMConfig | None = None, replica_idx: int = 0, temperature: float | None = None) -> dspy.LM:
    return _qwen(cfg=cfg or LMConfig(), replica_idx=replica_idx, thinking=True, temperature=temperature, max_tokens=None)


def make_qwen_nothink_lm(cfg: LMConfig | None = None, replica_idx: int = 0, max_tokens: int | None = None) -> dspy.LM:
    return _qwen(cfg=cfg or LMConfig(), replica_idx=replica_idx, thinking=False, temperature=0.0, max_tokens=max_tokens)


def make_mini_lm(cfg: LMConfig | None = None, temperature: float | None = None, max_tokens: int | None = None) -> dspy.LM:
    cfg = cfg or LMConfig()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return dspy.LM(model=cfg.mini_model, max_tokens=cfg.mini_max_tokens if max_tokens is None else max_tokens, temperature=cfg.mini_temperature if temperature is None else temperature, cache=False)


def make_lm(name: str, *, replica_idx: int = 0, temperature: float | None = None, max_tokens: int | None = None) -> dspy.LM:
    cfg = LMConfig()
    key = name.lower().strip()
    if key in {"qwen-think", "qwen14b-think", "root"}:
        return _qwen(cfg=cfg, replica_idx=replica_idx, thinking=True, temperature=temperature, max_tokens=max_tokens)
    if key in {"qwen-nothink", "qwen14b-nothink", "sub"}:
        return _qwen(cfg=cfg, replica_idx=replica_idx, thinking=False, temperature=0.0 if temperature is None else temperature, max_tokens=max_tokens)
    if key in {"mini", "gpt-4o-mini", "openai/gpt-4o-mini", "reflection"}:
        return make_mini_lm(cfg, temperature=temperature, max_tokens=max_tokens)
    if key.startswith("openai/"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        return dspy.LM(model=key, temperature=temperature or 0.0, max_tokens=max_tokens or 1024, cache=False)
    raise ValueError(f"Unknown LM name: {name}")
