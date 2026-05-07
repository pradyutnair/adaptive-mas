"""Route B: Bayesian top-candidate confidence stop.

Original entropy-only formulation degenerates at probe G=1 (single candidate, H=0 always).
This version uses TOP CANDIDATE NET SCORE as the decision signal:

  Commit when top_score >= tau_b   (high confidence in single answer)
  Continue otherwise               (low confidence or no candidate)

A second axis (entropy) is mixed in additively when belief has ≥2 candidates:
  decision_score = top.net_score - lambda * entropy
  Commit if decision_score >= tau_b

This allows MAS lane to fire when probe is uncertain (low net_score) OR when belief
has multiple competing candidates (high entropy).

Tunable: tau_b (commit threshold), lambda (entropy penalty weight).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..ledger import BeliefState, Ledger
from .base import Gate, GateAction, GateDecision


@dataclass
class BayesianGate:
    """Score-based + entropy-penalized confidence gate.

    tau_b: commit when (top.net_score - lambda * H) >= tau_b
    lambda_: entropy penalty (bits → score units). Larger = more sensitive to disagreement.
    fallback_cost / cost_history: kept for backward-compat, used in `info` only.
    """
    tau_b: float = 1.5
    lambda_: float = 0.5  # entropy penalty weight
    cost_history: deque = field(default_factory=lambda: deque(maxlen=20))
    fallback_cost: float = 5000.0
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
        top = belief.top()
        if top is None:
            return GateDecision(action=GateAction.CONTINUE, reason="no belief candidates",
                                info={"top_score": 0.0, "entropy": 0.0,
                                      "tau_b": self.tau_b, "lambda": self.lambda_})

        h = belief.entropy()
        score = top.net_score()
        decision_score = score - self.lambda_ * h
        info = {
            "top_score": float(score),
            "entropy": float(h),
            "decision_score": float(decision_score),
            "tau_b": float(self.tau_b),
            "lambda": float(self.lambda_),
        }
        ctx.setdefault("gate_calls", []).append({"turn": turn, **info})

        if decision_score >= self.tau_b:
            action = GateAction.SAS_COMMIT if turn == 0 else GateAction.STOP
            return GateDecision(action=action, score=decision_score,
                                reason=f"score {decision_score:.3f} >= τ_b {self.tau_b:.3f}",
                                info=info)
        return GateDecision(action=GateAction.CONTINUE, score=decision_score,
                            reason=f"score {decision_score:.3f} < τ_b {self.tau_b:.3f}",
                            info=info)
