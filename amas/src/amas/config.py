from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

from dotenv import load_dotenv


def load_env(env_path: str = "/local/yzheng/pnair/.env") -> None:
    if Path(env_path).exists():
        load_dotenv(env_path, override=False)


def _normalize_endpoints(eps: Any) -> list[str]:
    """Coerce a yaml endpoint value into a list[str]. Strings are wrapped, not
    iterated character-by-character (codex-flagged hazard on Stage 2)."""
    if eps is None:
        return []
    if isinstance(eps, str):
        return [eps]
    if isinstance(eps, (list, tuple)):
        out: list[str] = []
        for e in eps:
            if not isinstance(e, str):
                raise TypeError(f"endpoint entry must be str, got {type(e).__name__}: {e!r}")
            out.append(e)
        return out
    raise TypeError(f"endpoints must be str or list[str], got {type(eps).__name__}")


def validate_probe_config(cfg: dict[str, Any]) -> None:
    """Fail fast on bad probe blocks. Called once at script startup."""
    p = cfg.get("probe") or {}
    kind = str(p.get("kind", "vllm")).lower()
    if kind not in ("vllm", "openai"):
        raise ValueError(f"probe.kind must be 'vllm' or 'openai', got {kind!r}")
    if "endpoints" in p:
        _normalize_endpoints(p["endpoints"])  # raises on bad type
    if "model" in p:
        m = p["model"]
        if not isinstance(m, str) or not m.strip():
            raise ValueError(f"probe.model must be non-empty str, got {m!r}")
    for k in ("max_tokens", "concurrency"):
        if k in p and not isinstance(p[k], int):
            raise TypeError(f"probe.{k} must be int, got {type(p[k]).__name__}")
    if "temperature" in p and not isinstance(p["temperature"], (int, float)):
        raise TypeError(f"probe.temperature must be number, got {type(p['temperature']).__name__}")


def build_probe_client(cfg: dict[str, Any], *, vllm, openai_client) -> Tuple[Any, bool]:
    """Resolve the probe LM client from `cfg['probe']`.

    Returns `(client, owned)`. `owned` is True only when a fresh VLLMClient was
    constructed here — caller must `aclose()` it on shutdown. False when the
    shared `vllm` or `openai_client` was returned as-is (caller must NOT close
    those, since they belong to the orchestrator/openai shared lifecycle).

    `kind: vllm` (default) reuses `vllm` when model + endpoints match the
    probe block; otherwise builds a fresh VLLMClient. `kind: openai` returns
    the shared OpenAIClient.
    """
    from .lm import VLLMClient  # local import to avoid early-import cycles

    probe_cfg = cfg.get("probe") or {}
    kind = str(probe_cfg.get("kind", "vllm")).lower()
    if kind == "openai":
        return openai_client, False
    if kind == "vllm":
        if "endpoints" in probe_cfg:
            eps = _normalize_endpoints(probe_cfg["endpoints"])
        else:
            eps = list(getattr(vllm, "endpoints", []))
        model = probe_cfg.get("model") or getattr(vllm, "model", "")
        if eps == list(getattr(vllm, "endpoints", [])) and model == getattr(vllm, "model", ""):
            return vllm, False
        client = VLLMClient(
            endpoints=eps, model=model,
            max_tokens=int(probe_cfg.get("max_tokens", 256)),
            temperature=float(probe_cfg.get("temperature", 0.7)),
            concurrency=int(probe_cfg.get("concurrency", 16)),
        )
        return client, True
    raise ValueError(f"unknown probe.kind: {kind!r} (expected 'vllm' or 'openai')")


@dataclass
class HERAConfig:
    # vLLM (orchestrator + library + GRPO + RoPE meta-LLM)
    vllm_endpoints: tuple[str, ...] = (
        "http://localhost:8001/v1",
        "http://localhost:8002/v1",
        "http://localhost:8003/v1",
    )
    vllm_model: str = "Qwen/Qwen3-14B"
    vllm_max_tokens: int = 1024
    vllm_temperature: float = 0.7

    # OpenAI subagents
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 768
    openai_temperature: float = 0.3
    openai_concurrency: int = 12

    # Retriever
    retriever_url: str = "http://node408:8003/retrieve"
    retriever_topk: int = 5
    retriever_concurrency: int = 8

    # GRPO group sampling
    group_size: int = 4
    rollout_temperature: float = 0.9
    eval_temperature: float = 0.0
    ood_temperature: float = 0.3

    # Experience library
    library_max_entries: int = 30
    library_top_k_retrieve: int = 5

    # RoPE
    rope_failure_buffer: int = 8
    rope_update_every: int = 30
    rope_max_op_rules_per_agent: int = 6
    rope_max_behavioral_principles_per_agent: int = 4

    # Topology
    max_topology_steps: int = 8
    topology_mutation_threshold: float = 0.0  # mutate if F1 <= this for full group

    # Paths
    project_dir: Path = field(default_factory=lambda: Path("/local/yzheng/pnair/workspace/adaptive-mas/amas"))
    exp_lib_dir: Path = field(default_factory=lambda: Path("/local/yzheng/pnair/workspace/adaptive-mas/amas/exp_lib"))
    prompts_dir: Path = field(default_factory=lambda: Path("/local/yzheng/pnair/workspace/adaptive-mas/amas/prompts"))
    results_dir: Path = field(default_factory=lambda: Path("/local/yzheng/pnair/workspace/adaptive-mas/amas/results"))
    logs_dir: Path = field(default_factory=lambda: Path("/local/yzheng/pnair/workspace/adaptive-mas/amas/logs"))

    # wandb
    wandb_project: str = "amas-eval"
    wandb_entity: str | None = None

    def resolved_endpoints(self) -> tuple[str, ...]:
        env = os.getenv("AMAS_VLLM_ENDPOINTS")
        if env:
            return tuple(s.strip() for s in env.split(",") if s.strip())
        return self.vllm_endpoints

    def resolved_retriever_url(self) -> str:
        return os.getenv("AMAS_RETRIEVER_URL", self.retriever_url)
