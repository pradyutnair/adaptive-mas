"""Gate ABC + decision enum.

A gate is a per-turn function:
    decide(question, ledger, belief, turn, ctx) -> (action, info)
where action ∈ {"SAS_COMMIT", "CONTINUE", "STOP", "MUTATE"}.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..ledger import BeliefState, Ledger


class GateAction(str, Enum):
    SAS_COMMIT = "SAS_COMMIT"   # exit at turn 0 with probe answer
    CONTINUE = "CONTINUE"       # run another MAS turn
    STOP = "STOP"               # accept current top candidate
    MUTATE = "MUTATE"           # trigger topology mutation next turn


@dataclass
class GateDecision:
    action: GateAction
    score: float = 0.0
    reason: str = ""
    info: dict[str, Any] | None = None


class Gate(Protocol):
    name: str

    async def decide(self, *, question: str, ledger: Ledger, belief: BeliefState,
                     turn: int, ctx: dict[str, Any]) -> GateDecision:
        ...
