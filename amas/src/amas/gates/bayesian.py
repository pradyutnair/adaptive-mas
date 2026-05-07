"""Route B: Bayesian belief-state entropy stop.

Stop when H(belief) < lambda * expected_token_cost(next_turn).
expected_token_cost(t+1) = rolling avg of last K turn-costs, default K=20 (history-bounded).
SAS-commit at turn 0 if H(belief_0) < lambda * cost_estimate(turn_1).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..ledger import BeliefState, Ledger
from .base import Gate, GateAction, GateDecision


@dataclass
class BayesianGate:
    lambda_: float = 0.0008  # entropy bits per token; swept on val_v3
    cost_history: deque = field(default_factory=lambda: deque(maxlen=20))
    fallback_cost: float = 5000.0  # used until history populated
    name: str = "bayesian"

    def expected_next_turn_tokens(self) -> float:
        if not self.cost_history:
            return self.fallback_cost
        return float(sum(self.cost_history) / len(self.cost_history))

    def update_history(self, turn_cost: float) -> None:
        if turn_cost > 0:
            self.cost_history.append(float(turn_cost))

    async def decide(self, *, question: str, ledger: Ledger, belief: BeliefState,
                     turn: int, ctx: dict[str, Any]) -> GateDecision:
        h = belief.entropy()
        next_cost = self.expected_next_turn_tokens()
        threshold = self.lambda_ * next_cost
        info = {
            "entropy": float(h),
            "next_cost": float(next_cost),
            "threshold": float(threshold),
            "lambda": float(self.lambda_),
        }
        ctx.setdefault("gate_calls", []).append({"turn": turn, **info})

        if not belief.candidates:
            return GateDecision(action=GateAction.CONTINUE, reason="no belief candidates",
                                info=info)
        if h <= threshold:
            action = GateAction.SAS_COMMIT if turn == 0 else GateAction.STOP
            return GateDecision(action=action, score=-h,
                                reason=f"H={h:.3f} <= λ*cost={threshold:.3f}", info=info)
        return GateDecision(action=GateAction.CONTINUE, score=-h,
                            reason=f"H={h:.3f} > λ*cost={threshold:.3f}", info=info)
