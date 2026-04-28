"""YAML config loader with per-agent LLM overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def raw(self) -> dict[str, Any]:
        return dict(self._data)

    def get(self, dotpath: str, default: Any = None) -> Any:
        parts = dotpath.split(".")
        node: Any = self._data
        for part in parts:
            if isinstance(node, dict):
                node = node.get(part)
            else:
                return default
            if node is None:
                return default
        return node

    def set(self, dotpath: str, value: Any) -> None:
        parts = dotpath.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def agent_llm(self, agent_name: str) -> dict[str, Any]:
        defaults = dict(self._data.get("llm_defaults", {}) or {})
        agent_cfg = (self._data.get("agents", {}) or {}).get(agent_name, {}) or {}
        defaults.update(agent_cfg)
        return defaults
