from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def load_env(env_path: str = "/local/yzheng/pnair/.env") -> None:
    if Path(env_path).exists():
        load_dotenv(env_path, override=False)


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
