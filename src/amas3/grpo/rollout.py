"""Dataclasses for a single pipeline rollout and a TF-GRPO rollout group."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Rollout:
    """Single pipeline execution result for one question at one temperature."""
    question_id: str
    question: str
    gold_answer: str
    temperature: float
    policy_name: str = ""
    predicted_answer: str = ""
    em: float = 0.0
    f1: float = 0.0
    contain: float = 0.0
    total_tokens: int = 0
    topology: str = ""
    sampled_topology: dict[str, Any] = field(default_factory=dict)
    plan_subgoals: int = 0
    findings: list[dict] = field(default_factory=list)
    wallclock_seconds: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    dual_reward: float = 0.0
    token_efficiency: float = 0.0
    dataset: str = "default"


@dataclass
class GroupResult:
    """K rollouts for a single question."""
    question_id: str
    question: str
    gold_answer: str
    rollouts: list[Rollout] = field(default_factory=list)
    has_mixed_outcomes: bool = False
    winners: list[Rollout] = field(default_factory=list)
    losers: list[Rollout] = field(default_factory=list)
    deployment_budget: int = 0
