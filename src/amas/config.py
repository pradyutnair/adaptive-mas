"""YAML config loader with per-agent LLM overrides.

Top-level structure::

    llm_defaults:
      model: Qwen/Qwen3-8B
      base_url: http://localhost:8001/v1
      temperature: 0.0
      max_tokens: 1024
      enable_thinking: false

    agents:
      planner:        # inherits from llm_defaults, overrides allowed
        enable_thinking: true
        max_tokens: 2048
      investigator:
        max_tokens: 768

    retriever:
      base_url: http://127.0.0.1:9110
      top_k: 10

    pipeline:
      max_subgoals: 6
      max_searches_per_subagent: 5
      max_evidence_hits: 6
      max_excerpt_chars: 600
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Loaded YAML config with helper accessors."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def raw(self) -> dict[str, Any]:
        return self._data

    def get(self, key_path: str, default: Any = None) -> Any:
        """Dot-path getter, e.g. ``cfg.get("retriever.top_k")``."""
        node: Any = self._data
        for part in key_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, key_path: str, value: Any) -> None:
        """Dot-path setter for runtime overrides (e.g. CLI flags)."""
        parts = key_path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def agent_llm(self, agent: str) -> dict[str, Any]:
        """Build the LLM client config for a named agent."""
        merged = dict(self._data.get("llm_defaults", {}) or {})
        merged.update(self._data.get("agents", {}).get(agent, {}) or {})
        return merged
