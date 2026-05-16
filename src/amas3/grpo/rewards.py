"""Reward shaping for TF-GRPO.

Layers:
  1. Token-efficiency reward in [0, 1] (sigmoid). When the active
     ``deployment_budget`` B is supplied, the sigmoid is centered on B.
     Otherwise it falls back to a dataset baseline.
  2. Stepped over-budget penalty anchored to the same envelope (B or
     baseline), so the policy sees a sharp signal the moment tokens > B.
  3. Under-cap bonus: when tokens <= UNDER_CAP_RATIO * envelope
     (e.g. <=7k under B=8k), a small reward bump rewards comfortable
     headroom under the cap.
  4. Correctness-gated dual reward: token efficiency is only a meaningful
     bonus once answer quality is non-trivial.

Every cost-side term is anchored to the SAME envelope so pi_O is trained
on a coherent signal.
"""
from __future__ import annotations

import math

from .metrics import compute_task_reward

# Fallback envelopes when no deployment_budget is supplied. Kept low so any
# unsupervised run still pushes the policy toward cheap rollouts.
TOKEN_BUDGET_BASELINES: dict[str, int] = {
    "hotpotqa": 6500,
    "2wiki": 6700,
    "musique": 7000,
    "bamboogle": 5000,
    "default": 6500,
}

# Stepped penalties applied on top of the smooth efficiency reward so pi_O
# sees a sharp signal as soon as a rollout exceeds the active envelope.
# Triggered at multiples of the envelope (B or fallback baseline).
OVER_BUDGET_PENALTIES: tuple[tuple[float, float], ...] = (
    (1.00, 0.05),
    (1.25, 0.10),
    (1.50, 0.15),
)

# Under-cap bonus: rewards rollouts that come in with real headroom under
# the active envelope. Expressed as a fraction of the envelope so it scales
# naturally with B (0.875 * 8000 = 7000).
UNDER_CAP_RATIO: float = 0.875
UNDER_CAP_BONUS: float = 0.05


def reward_envelope(dataset: str, deployment_budget: int | None) -> int:
    if deployment_budget is not None and deployment_budget > 0:
        return int(deployment_budget)
    return TOKEN_BUDGET_BASELINES.get(dataset, TOKEN_BUDGET_BASELINES["default"])


def compute_token_efficiency_reward(
    total_tokens: int,
    dataset: str = "default",
    deployment_budget: int | None = None,
) -> float:
    """Sigmoid efficiency reward centered on the active envelope.

    At ratio=1.0: 0.5; ratio=0.5: ~0.88; ratio=1.5: ~0.12.
    """
    envelope = reward_envelope(dataset, deployment_budget)
    ratio = total_tokens / max(envelope, 1)
    reward = 1.0 / (1.0 + math.exp(2.0 * (ratio - 1.0)))
    return round(max(0.0, min(1.0, reward)), 4)


def compute_over_budget_penalty(
    total_tokens: int,
    dataset: str = "default",
    deployment_budget: int | None = None,
) -> float:
    """Stepped penalty for exceeding the active envelope (B or baseline)."""
    envelope = reward_envelope(dataset, deployment_budget)
    ratio = total_tokens / max(envelope, 1)
    penalty = 0.0
    for threshold, delta in OVER_BUDGET_PENALTIES:
        if ratio > threshold:
            penalty += delta
    return penalty


def compute_under_cap_bonus(
    total_tokens: int,
    dataset: str = "default",
    deployment_budget: int | None = None,
) -> float:
    """Bonus when total_tokens <= UNDER_CAP_RATIO * envelope."""
    envelope = reward_envelope(dataset, deployment_budget)
    if total_tokens <= UNDER_CAP_RATIO * envelope:
        return UNDER_CAP_BONUS
    return 0.0


def compute_dual_reward(
    em: float, f1: float, total_tokens: int,
    dataset: str = "default", alpha: float = 0.7,
    answered: bool = True,
    contain: float = 0.0,
    deployment_budget: int | None = None,
) -> float:
    """Reward quality first, then token efficiency.

    Completely wrong answers must not win because they are cheap. This is
    critical for SAS-first routing, where wrong confident shortcuts otherwise
    dominate the group reward.
    """
    r_task = compute_task_reward(em, f1, contain)
    r_eff = compute_token_efficiency_reward(total_tokens, dataset, deployment_budget)

    if r_task < 0.05:
        r = -0.30 + 0.10 * r_eff
    elif r_task < 0.30:
        r = 0.85 * r_task + 0.15 * r_eff
    else:
        r = alpha * r_task + (1 - alpha) * r_eff
        r += compute_under_cap_bonus(total_tokens, dataset, deployment_budget)

    if not answered:
        r -= 0.15
    r -= compute_over_budget_penalty(total_tokens, dataset, deployment_budget)
    return round(r, 4)
