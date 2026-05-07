"""Gate registry."""
from __future__ import annotations

from typing import Any

from ..lm import OpenAIClient
from .base import Gate, GateAction, GateDecision
from .bayesian import BayesianGate
from .conformal import ConformalGate, conformal_quantile, write_calibration
from .misc import OffGate, OracleGate, RandomGate, SASOnlyGate


def make_gate(name: str, *, openai_client: OpenAIClient | None = None,
               cfg: dict[str, Any] | None = None) -> Gate:
    name = (name or "off").lower()
    cfg = cfg or {}
    if name == "conformal":
        if openai_client is None:
            raise ValueError("conformal gate requires openai_client")
        return ConformalGate.from_calibration(
            openai_client=openai_client,
            calib_path=cfg.get("calib_path", "results/route_a_calibration.json"),
            alpha=float(cfg.get("alpha", 0.05)),
        )
    if name == "bayesian":
        return BayesianGate(
            lambda_=float(cfg.get("lambda", 0.0008)),
            fallback_cost=float(cfg.get("fallback_cost", 5000.0)),
        )
    if name == "oracle":
        return OracleGate()
    if name == "random":
        return RandomGate(p_stop=float(cfg.get("p_stop", 0.4)),
                          rng_seed=int(cfg.get("rng_seed", 42)))
    if name == "off":
        return OffGate()
    if name in ("sas_only", "sasonly"):
        return SASOnlyGate()
    raise ValueError(f"unknown gate: {name}")


__all__ = [
    "Gate", "GateAction", "GateDecision",
    "ConformalGate", "BayesianGate", "OracleGate", "RandomGate", "OffGate", "SASOnlyGate",
    "conformal_quantile", "write_calibration", "make_gate",
]
