"""Factory for the three DSPy LMs we use.

- qwen_think: Qwen3-8B with enable_thinking=True (planner)
- qwen_nothink: Qwen3-8B with enable_thinking=False (cheap fallback worker)
- mini: GPT-4o-mini (workers, synthesizer)
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import dspy


@dataclass
class LMConfig:
    qwen_base_urls: tuple[str, ...] = ('http://localhost:8001/v1', 'http://localhost:8002/v1', 'http://localhost:8003/v1')
    qwen_model: str = 'Qwen/Qwen3-8B'
    qwen_think_max_tokens: int = 4096
    qwen_nothink_max_tokens: int = 512
    qwen_think_temperature: float = 0.6
    qwen_nothink_temperature: float = 0.0
    mini_model: str = 'openai/gpt-4o-mini'
    mini_max_tokens: int = 1024
    mini_temperature: float = 0.0


def make_qwen_think_lm(cfg: LMConfig | None = None, replica_idx: int = 0, temperature: float | None = None) -> dspy.LM:
    cfg = cfg or LMConfig()
    base = cfg.qwen_base_urls[replica_idx % len(cfg.qwen_base_urls)]
    return dspy.LM(
        model=f'hosted_vllm/{cfg.qwen_model}',
        api_base=base,
        api_key='EMPTY',
        max_tokens=cfg.qwen_think_max_tokens,
        temperature=cfg.qwen_think_temperature if temperature is None else temperature,
        extra_body={'chat_template_kwargs': {'enable_thinking': True}},
        cache=False,
    )


def make_qwen_nothink_lm(cfg: LMConfig | None = None, replica_idx: int = 0) -> dspy.LM:
    cfg = cfg or LMConfig()
    base = cfg.qwen_base_urls[replica_idx % len(cfg.qwen_base_urls)]
    return dspy.LM(
        model=f'hosted_vllm/{cfg.qwen_model}',
        api_base=base,
        api_key='EMPTY',
        max_tokens=cfg.qwen_nothink_max_tokens,
        temperature=cfg.qwen_nothink_temperature,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
        cache=False,
    )


def make_mini_lm(cfg: LMConfig | None = None, temperature: float | None = None, max_tokens: int | None = None) -> dspy.LM:
    cfg = cfg or LMConfig()
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is not set in the environment. Source /local/yzheng/pnair/.env first.')
    return dspy.LM(
        model=cfg.mini_model,
        max_tokens=cfg.mini_max_tokens if max_tokens is None else max_tokens,
        temperature=cfg.mini_temperature if temperature is None else temperature,
        cache=False,
    )

def make_qwen14b_think_lm(cfg: LMConfig | None = None, port: int = 8003, replica_idx: int = 0) -> dspy.LM:
    """Round-robin Qwen3-14B across ports 8001/8002/8003.

    If port is explicitly passed (default 8003), use that. Otherwise pick
    by replica_idx mod 3.
    """
    cfg = cfg or LMConfig()
    ports = [8001, 8002, 8003]
    chosen_port = port if port != 8003 else ports[replica_idx % len(ports)] if replica_idx else 8003
    if replica_idx is not None and replica_idx >= 0:
        chosen_port = ports[replica_idx % len(ports)]
    return dspy.LM(
        model='hosted_vllm/Qwen/Qwen3-14B',
        api_base=f'http://localhost:{chosen_port}/v1',
        api_key='EMPTY',
        max_tokens=cfg.qwen_think_max_tokens,
        temperature=cfg.qwen_think_temperature,
        extra_body={'chat_template_kwargs': {'enable_thinking': True}},
        cache=False,
    )


def make_qwen14b_nothink_lm(
    cfg: LMConfig | None = None,
    replica_idx: int = 0,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dspy.LM:
    """Cheap fast Qwen3-14B for SAS-attempt extraction (~256 tokens)."""
    cfg = cfg or LMConfig()
    ports = [8001, 8002, 8003]
    chosen_port = ports[replica_idx % len(ports)]
    return dspy.LM(
        model='hosted_vllm/Qwen/Qwen3-14B',
        api_base=f'http://localhost:{chosen_port}/v1',
        api_key='EMPTY',
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
        cache=False,
    )


def make_qwen14b_think_small_lm(cfg: LMConfig | None = None, replica_idx: int = 0, max_tokens: int = 1024) -> dspy.LM:
    """Qwen3-14B with thinking, small token budget (for solvers/synth)."""
    cfg = cfg or LMConfig()
    ports = [8001, 8002, 8003]
    chosen_port = ports[replica_idx % len(ports)]
    return dspy.LM(
        model='hosted_vllm/Qwen/Qwen3-14B',
        api_base=f'http://localhost:{chosen_port}/v1',
        api_key='EMPTY',
        max_tokens=max_tokens,
        temperature=0.6,
        extra_body={'chat_template_kwargs': {'enable_thinking': True}},
        cache=False,
    )
