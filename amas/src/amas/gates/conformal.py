"""Route A: split-conformal foreign verifier (calibrated ContextValidator).

Per plan §3.1:
- One ContextValidator-style call per turn (~250 tok).
- Constrained YES/NO via prompt; score = log P(YES) - log P(NO) (proxy via heuristic).
- Calibration: split-conformal threshold tau_high(alpha) on val + 200q held-out.
- Defer band [tau_low, tau_high] -> one extra probe sample, then re-verify (max 1 defer).
- Coverage guarantee: SAS-error rate ≤ alpha = 0.05.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ledger import BeliefState, Ledger
from ..lm import OpenAIClient, parse_json_lenient
from ..retriever import format_passages
from .base import Gate, GateAction, GateDecision

logger = logging.getLogger(__name__)


VERIFIER_SYSTEM = (
    "You are an independent answer verifier. You have not seen the chain-of-thought of any "
    "answering agent. Given the question, the proposed answer, and a small set of supporting "
    "evidence snippets, judge whether the answer is FACTUALLY CORRECT and DIRECTLY ANSWERS the "
    "question. Respond with valid JSON only."
)


def build_verifier_user(question: str, candidate: str, ledger_block: str, evidence_block: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Proposed answer: {candidate}\n\n"
        f"Top ledger entries:\n{ledger_block}\n\n"
        f"Top retrieved passage(s):\n{evidence_block}\n\n"
        "Decide: does the proposed answer correctly answer the question, supported by the evidence?\n"
        "Respond with JSON: "
        '{"verdict": "YES|NO", "confidence": <float 0..1>, "rationale": "<one sentence>"}'
    )


@dataclass
class ConformalGate:
    """Calibrated foreign verifier."""
    openai_client: OpenAIClient
    tau_high: float
    tau_low: float
    alpha: float = 0.05
    name: str = "conformal"

    @classmethod
    def from_calibration(cls, openai_client: OpenAIClient, calib_path: str | Path,
                          alpha: float = 0.05) -> "ConformalGate":
        p = Path(calib_path)
        if p.exists():
            d = json.loads(p.read_text())
            tau_high = float(d.get("tau_high", 0.7))
            tau_low = float(d.get("tau_low", 0.3))
        else:
            # Pre-calibration default — used during P0 smoke before scripts/calibrate_routeA.py runs.
            tau_high, tau_low = 0.7, 0.3
        return cls(openai_client=openai_client, tau_high=tau_high, tau_low=tau_low, alpha=alpha)

    async def _verify_score(self, *, question: str, candidate: str,
                            ledger: Ledger, ctx: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        ledger_block = ledger.summarize_for_agent(n=8, max_chars=900)
        passages = ctx.get("passages") or []
        evidence_block = format_passages(passages[:2], max_chars_per=400) if passages else "(no passages)"
        user = build_verifier_user(question, candidate, ledger_block, evidence_block)
        res = await self.openai_client.chat(VERIFIER_SYSTEM, user, temperature=0.0,
                                            max_tokens=180, json_mode=True)
        parsed = parse_json_lenient(res.text)
        if not isinstance(parsed, dict):
            parsed = {}
        verdict = str(parsed.get("verdict", "NO")).upper()
        try:
            conf = float(parsed.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        # Convert to logit-style score: positive when YES, negative when NO.
        # log P(YES) - log P(NO) ≈ logit(conf) when verdict=YES, -logit(conf) when NO.
        eps = 1e-3
        conf = min(max(conf, eps), 1 - eps)
        s = math.log(conf / (1 - conf))
        score = s if verdict == "YES" else -s
        info = {
            "verdict": verdict, "confidence": conf, "score": score,
            "rationale": str(parsed.get("rationale", ""))[:200],
            "tokens": int(res.prompt_tokens + res.completion_tokens),
        }
        return score, info

    async def decide(self, *, question: str, ledger: Ledger, belief: BeliefState,
                     turn: int, ctx: dict[str, Any]) -> GateDecision:
        top = belief.top()
        candidate = ctx.get("candidate") or (top.answer if top else "")
        if not candidate:
            return GateDecision(action=GateAction.CONTINUE, reason="no candidate")

        score, info = await self._verify_score(
            question=question, candidate=candidate, ledger=ledger, ctx=ctx,
        )
        ctx.setdefault("gate_calls", []).append({"turn": turn, **info})

        if score >= self.tau_high:
            action = GateAction.SAS_COMMIT if turn == 0 else GateAction.STOP
            return GateDecision(action=action, score=score,
                                reason=f"score {score:.3f} >= tau_high {self.tau_high:.3f}",
                                info=info)
        if score <= -self.tau_high:
            # Strong refusal -> trigger mutation (failure mode is structural).
            return GateDecision(action=GateAction.MUTATE, score=score,
                                reason=f"score {score:.3f} <= -tau_high {self.tau_high:.3f}",
                                info=info)
        if not ctx.get("gate_deferred"):
            # Within defer band: continue, mark deferred for one extra round.
            ctx["gate_deferred"] = True
            return GateDecision(action=GateAction.CONTINUE, score=score,
                                reason=f"defer band; score {score:.3f}", info=info)
        return GateDecision(action=GateAction.CONTINUE, score=score,
                            reason="undecided after defer", info=info)


def write_calibration(path: str | Path, *, tau_high: float, tau_low: float,
                       alpha: float, n_calib: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "tau_high": float(tau_high), "tau_low": float(tau_low),
        "alpha": float(alpha), "n_calib": int(n_calib),
    }, indent=2))


def conformal_quantile(scores: list[float], alpha: float = 0.05) -> float:
    """Empirical (1-alpha) quantile with finite-sample correction (n+1 conformal)."""
    if not scores:
        return 0.7
    n = len(scores)
    s = sorted(scores)
    q = math.ceil((1 - alpha) * (n + 1)) - 1
    q = max(0, min(n - 1, q))
    return float(s[q])
