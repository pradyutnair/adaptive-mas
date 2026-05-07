"""Evidence Ledger + Belief State.

Section 2 of plan. Single shared communication substrate consumed by:
- gates (conformal Route A reads top-N entries; bayesian Route B reads belief state)
- agents (read top-N entries when running; emit support/refute/neutral entries)
- orchestrator (carries ledger snapshots across turns inside Trajectory)

Append-only. Deterministic belief update. Display-cap N=12 entries by confidence.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .metric import normalize_answer
from .retriever import Passage


Stance = Literal["support", "refute", "neutral"]


@dataclass
class LedgerEntry:
    id: str
    turn: int
    source_agent: str
    claim: str
    passage_ids: list[str] = field(default_factory=list)
    stance: Stance = "neutral"
    target_id: str | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    answer: str
    support_score: float = 0.0
    refute_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)

    def net_score(self) -> float:
        return self.support_score - self.refute_score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f1_tok(a: str, b: str) -> float:
    ta = normalize_answer(a).split()
    tb = normalize_answer(b).split()
    if not ta or not tb:
        return 0.0
    common = Counter(ta) & Counter(tb)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(ta)
    r = n / len(tb)
    return 2 * p * r / (p + r)


@dataclass
class Ledger:
    entries: list[LedgerEntry] = field(default_factory=list)
    _next_num: int = 1

    def fresh_id(self) -> str:
        eid = f"L-{self._next_num:03d}"
        self._next_num += 1
        return eid

    def add(self, *, turn: int, source_agent: str, claim: str,
            passage_ids: list[str] | None = None, stance: Stance = "neutral",
            target_id: str | None = None, confidence: float = 0.5) -> str:
        e = LedgerEntry(
            id=self.fresh_id(), turn=turn, source_agent=source_agent,
            claim=str(claim).strip(),
            passage_ids=list(passage_ids or []), stance=stance,
            target_id=target_id, confidence=float(confidence),
        )
        self.entries.append(e)
        return e.id

    def add_passages(self, *, turn: int, source_agent: str, passages: list[Passage]) -> list[str]:
        out = []
        for p in passages:
            cid = self.add(turn=turn, source_agent=source_agent,
                           claim=p.text[:240], passage_ids=[p.chunk_id],
                           stance="neutral", confidence=float(p.score))
            out.append(cid)
        return out

    def top_n(self, n: int = 12) -> list[LedgerEntry]:
        # Sort by stance priority (support > refute > neutral) then confidence desc.
        rank = {"support": 2, "refute": 1, "neutral": 0}
        return sorted(self.entries,
                      key=lambda e: (rank.get(e.stance, 0), e.confidence),
                      reverse=True)[:n]

    def summarize_for_agent(self, *, n: int = 12, max_chars: int = 1500) -> str:
        rows = self.top_n(n)
        if not rows:
            return "(ledger empty)"
        lines = []
        for e in rows:
            tag = e.stance.upper()
            line = f"[{e.id} t{e.turn} {tag} c={e.confidence:.2f} src={e.source_agent}] {e.claim}"
            lines.append(line)
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[:max_chars] + "..."
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries], "next_num": self._next_num}


@dataclass
class BeliefState:
    candidates: list[Candidate] = field(default_factory=list)
    top_k: int = 5

    def update_from_answer(self, answer: str, *, support: float = 0.0, refute: float = 0.0,
                           evidence_ids: list[str] | None = None) -> None:
        ans = (answer or "").strip()
        if not ans:
            return
        for c in self.candidates:
            if _f1_tok(c.answer, ans) >= 0.7:
                c.support_score += support
                c.refute_score += refute
                for eid in evidence_ids or []:
                    if eid not in c.evidence_ids:
                        c.evidence_ids.append(eid)
                self._cap()
                return
        # Disjoint candidate (token-F1 < 0.3 vs all existing per plan §2.2 — we use a higher
        # similarity threshold of 0.7 above so anything below merges-vs-add boundary at 0.7).
        new_c = Candidate(
            answer=ans, support_score=max(support, 0.0), refute_score=max(refute, 0.0),
            evidence_ids=list(evidence_ids or []),
        )
        self.candidates.append(new_c)
        self._cap()

    def update_from_ledger_entry(self, entry: LedgerEntry, answer: str | None = None) -> None:
        if entry.stance == "support":
            self.update_from_answer(answer or entry.claim, support=entry.confidence,
                                    evidence_ids=[entry.id])
        elif entry.stance == "refute":
            target = answer or entry.claim
            self.update_from_answer(target, refute=entry.confidence, evidence_ids=[entry.id])

    def _cap(self) -> None:
        self.candidates.sort(key=lambda c: c.net_score(), reverse=True)
        if len(self.candidates) > self.top_k:
            self.candidates = self.candidates[: self.top_k]

    def top(self) -> Candidate | None:
        if not self.candidates:
            return None
        self._cap()
        return self.candidates[0]

    def entropy(self) -> float:
        """Shannon entropy of softmax(net_scores). Bits."""
        if not self.candidates:
            return 0.0
        scores = [c.net_score() for c in self.candidates]
        m = max(scores)
        ws = [math.exp(s - m) for s in scores]
        z = sum(ws)
        if z <= 0:
            return 0.0
        ps = [w / z for w in ws]
        h = 0.0
        for p in ps:
            if p > 0:
                h -= p * math.log2(p)
        return h

    def summarize(self, k: int = 5) -> str:
        if not self.candidates:
            return "(no candidates)"
        rows = sorted(self.candidates, key=lambda c: c.net_score(), reverse=True)[:k]
        return "\n".join(
            f"  - {c.answer!r} support={c.support_score:.2f} refute={c.refute_score:.2f}"
            for c in rows
        )

    def to_dict(self) -> dict[str, Any]:
        return {"top_k": self.top_k, "candidates": [c.to_dict() for c in self.candidates]}


def parse_stance_from_agent(agent_name: str, output: dict[str, Any]) -> Stance:
    """Map agent output to ledger stance.

    - ContextValidator: refute if sufficient=false (claim points at the missing answer)
    - EvidenceSelector: support
    - Retriever: neutral (passages only)
    - AnswerGenerator/ConcludeAgent/ReflectAgent: support (their answer is a claim)
    - ReflectAgent: refute when decision == REVISE
    """
    if agent_name == "ContextValidator":
        if output.get("sufficient") is False:
            return "refute"
        return "support"
    if agent_name == "ReflectAgent":
        if str(output.get("decision", "")).upper() == "REVISE":
            return "refute"
        return "support"
    if agent_name in ("EvidenceSelector",):
        return "support"
    if agent_name in ("AnswerGenerator", "ConcludeAgent"):
        return "support"
    return "neutral"
