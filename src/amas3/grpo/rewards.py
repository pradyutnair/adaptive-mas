"""Reward shaping for TF-GRPO.

Three layers:
  1. Token-efficiency reward in [0, 1] (sigmoid around the dataset baseline).
  2. Stepped over-budget penalty, anchored to ``deployment_budget`` B when
     supplied (budget-conditioned policy) or to the dataset baseline.
  3. Dual reward r = alpha*r_task + (1-alpha)*r_eff - blank - over_budget(B).
"""
from __future__ import annotations

import math

from .metrics import compute_task_reward

TOKEN_BUDGET_BASELINES: dict[str, int] = {
    "hotpotqa": 7050,
    "2wiki": 7240,
    "musique": 7480,
    "bamboogle": 5100,
    "default": 7000,
}

# Stepped penalties applied on top of the smooth efficiency reward so pi_O sees
# a sharp signal whenever a rollout blows past the active envelope. The
# group-relative advantage stays the primary learning signal; this just
# prevents over-spending from masquerading as a winner.
OVER_BUDGET_PENALTIES: tuple[tuple[float, float], ...] = (
    (1.5, 0.10),
    (2.0, 0.10),
    (3.0, 0.10),
)


def compute_token_efficiency_reward(total_tokens: int, dataset: str = "default") -> float:
    """Sigmoid efficiency reward centered on the dataset's baseline budget.

    At ratio=1.0: 0.5; ratio=0.5: ~0.88; ratio=1.5: ~0.12.
    """
    baseline = TOKEN_BUDGET_BASELINES.get(dataset, TOKEN_BUDGET_BASELINES["default"])
    ratio = total_tokens / max(baseline, 1)
    reward = 1.0 / (1.0 + math.exp(2.0 * (ratio - 1.0)))
    return round(max(0.0, min(1.0, reward)), 4)


def compute_over_budget_penalty(
    total_tokens: int,
    dataset: str = "default",
    deployment_budget: int | None = None,
) -> float:
    """Stepped penalty for exceeding the active token envelope.

    When ``deployment_budget`` is set (budget-conditioned policy), the envelope
    is B; otherwise it falls back to the dataset's learned baseline. This
    couples the over-budget penalty to the same B that the orchestrator
    receives in its prompt, so the policy is trained on a coherent signal.
    """
    if deployment_budget is not None and deployment_budget > 0:
        envelope = int(deployment_budget)
    else:
        envelope = TOKEN_BUDGET_BASELINES.get(dataset, TOKEN_BUDGET_BASELINES["default"])
    ratio = total_tokens / max(envelope, 1)
    penalty = 0.0
    for threshold, delta in OVER_BUDGET_PENALTIES:
        if ratio > threshold:
            penalty += delta
    return penalty


def compute_dual_reward(
    em: float, f1: float, total_tokens: int,
    dataset: str = "default", alpha: float = 0.7,
    answered: bool = True,
    contain: float = 0.0,
    deployment_budget: int | None = None,
) -> float:
    """r = alpha * r_task + (1-alpha) * r_eff - blank_penalty - over_budget_penalty(B).

    alpha=0.7 makes task quality primary, efficiency secondary. Blank answers
    and budget overshoots get explicit hard penalties so the group-relative
    advantage cannot be fooled by a high-EM-but-expensive rollout. When
    ``deployment_budget`` is set, the over-budget penalty is measured against
    B rather than the dataset baseline.
    """
    r_task = compute_task_reward(em, f1, contain)
    r_eff = compute_token_efficiency_reward(total_tokens, dataset)
    r = alpha * r_task + (1 - alpha) * r_eff
    if not answered:
        r -= 0.15
    r -= compute_over_budget_penalty(total_tokens, dataset, deployment_budget=deployment_budget)
    return round(r, 4)
