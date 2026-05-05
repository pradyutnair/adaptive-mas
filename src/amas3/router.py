"""LLM Router: decides SAS (direct) vs DAG (decompose) per question.

Replaces the heuristic groundedness threshold for routing. Uses one
lightweight LLM call after probe retrieval to make a deliberate routing
decision based on question complexity and evidence quality.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
import dspy
from .types import RetrievedChunk


class RouteQuestion(dspy.Signature):
    """Decide whether this question can be answered directly from the evidence
or needs multi-hop decomposition.

Return STRICT JSON: {"route": "direct"|"decompose", "reason": "<short>"}

Choose "direct" ONLY if:
1. The question asks for a SINGLE fact (one entity, date, number, yes/no).
2. At least one chunk explicitly contains the answer.
3. No intermediate reasoning or bridge entity resolution is needed.

Choose "decompose" if:
1. The question requires combining information from multiple sources.
2. The question contains implicit references ("the author of...", "the country where...").
3. No single chunk directly answers the question.
4. The question has comparison, temporal, or multi-entity structure.

When in doubt, choose "decompose" — it is safer to over-decompose than to
miss a multi-hop question.
"""
    question: str = dspy.InputField()
    evidence_summary: str = dspy.InputField(desc="Top retrieved chunks for the question")
    route_json: str = dspy.OutputField()


@dataclass
class RoutingDecision:
    route: str  # "direct" or "decompose"
    reason: str
    tokens: int = 0


def _parse_route(raw: str) -> dict:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def route_question(
    *,
    router_lm: dspy.LM,
    question: str,
    chunks: list[RetrievedChunk],
    excerpt_chars: int = 500,
) -> RoutingDecision:
    if not chunks:
        return RoutingDecision(route="decompose", reason="no evidence retrieved", tokens=0)

    evidence = "\n".join(
        f"[{i+1}] {c.text[:excerpt_chars]}" for i, c in enumerate(chunks[:5])
    )

    try:
        with dspy.context(lm=router_lm):
            pred = dspy.Predict(RouteQuestion)(
                question=question,
                evidence_summary=evidence,
            )
        try:
            history = router_lm.history[-1] if router_lm.history else None
            usage = (history or {}).get("usage") or {}
            tokens = int(usage.get("total_tokens", 0))
        except Exception:
            tokens = 0
    except Exception:
        return RoutingDecision(route="decompose", reason="router_error", tokens=0)

    obj = _parse_route(getattr(pred, "route_json", ""))
    route = str(obj.get("route", "decompose")).lower().strip()
    if route not in ("direct", "decompose"):
        route = "decompose"
    reason = str(obj.get("reason", ""))[:200]

    return RoutingDecision(route=route, reason=reason, tokens=tokens)
