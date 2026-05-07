"""Ablation gates: oracle, random, off."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

from ..ledger import BeliefState, Ledger
from ..metric import accuracy
from .base import GateAction, GateDecision


@dataclass
class OracleGate:
    """Knows gold; commits as soon as a correct candidate exists. Upper-bound ablation."""
    name: str = "oracle"

    async def decide(self, *, question: str, ledger: Ledger, belief: BeliefState,
                     turn: int, ctx: dict[str, Any]) -> GateDecision:
        gold = ctx.get("gold")
        if gold is None or not belief.candidates:
            return GateDecision(action=GateAction.CONTINUE, reason="no gold or empty belief")
        for c in belief.candidates:
            if accuracy(c.answer, gold) > 0:
                action = GateAction.SAS_COMMIT if turn == 0 else GateAction.STOP
                return GateDecision(action=action, score=1.0,
                                     reason=f"oracle accept: {c.answer!r}",
                                     info={"answer": c.answer})
        return GateDecision(action=GateAction.CONTINUE, reason="no oracle-correct candidate")


@dataclass
class RandomGate:
    """Stops with fixed probability per turn; SAS-commits with prob/2 at turn 0. Ablation."""
    p_stop: float = 0.4
    rng_seed: int = 42
    _rng: random.Random | None = field(default=None, repr=False)
    name: str = "random"

    def __post_init__(self) -> None:
        seed = int(os.getenv("AMAS_RANDOM_GATE_SEED", str(self.rng_seed)))
        self._rng = random.Random(seed)

    async def decide(self, *, question: str, ledger: Ledger, belief: BeliefState,
                     turn: int, ctx: dict[str, Any]) -> GateDecision:
        assert self._rng is not None
        if not belief.candidates:
            return GateDecision(action=GateAction.CONTINUE, reason="empty belief")
        roll = self._rng.random()
        p = self.p_stop / 2 if turn == 0 else self.p_stop
        if roll < p:
            action = GateAction.SAS_COMMIT if turn == 0 else GateAction.STOP
            return GateDecision(action=action, score=0.0,
                                 reason=f"random p={p:.2f} roll={roll:.3f}")
        return GateDecision(action=GateAction.CONTINUE, score=0.0,
                            reason=f"random continue roll={roll:.3f}")


@dataclass
class OffGate:
    """No early exit. Always run T_max turns. Pure-MAS baseline."""
    name: str = "off"

    async def decide(self, *, question: str, ledger: Ledger, belief: BeliefState,
                     turn: int, ctx: dict[str, Any]) -> GateDecision:
        return GateDecision(action=GateAction.CONTINUE, reason="gate=off")


@dataclass
class SASOnlyGate:
    """Always commit at turn 0 from probe answer. Lower bound for SAS-natural."""
    name: str = "sas_only"

    async def decide(self, *, question: str, ledger: Ledger, belief: BeliefState,
                     turn: int, ctx: dict[str, Any]) -> GateDecision:
        if turn == 0 and belief.candidates:
            return GateDecision(action=GateAction.SAS_COMMIT, score=0.0,
                                 reason="sas_only forced commit")
        return GateDecision(action=GateAction.CONTINUE, reason="sas_only continue")
