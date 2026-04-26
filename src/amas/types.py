"""Core data types for AMAS pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Literal, Optional


class AnswerType(str, Enum):
    """Strict enum of allowed answer types. Mirrors the prompt schema."""

    PERSON = "person"
    PLACE = "place"
    DATE = "date"
    NUMBER = "number"
    YES_NO = "yes_no"
    ENTITY = "entity"
    OTHER = "other"

    @classmethod
    def coerce(cls, value: Any) -> "AnswerType":
        """Best-effort coercion + alias collapsing."""
        v = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "location": "place", "loc": "place", "country": "place", "city": "place",
            "boolean": "yes_no", "bool": "yes_no", "yes/no": "yes_no",
            "year": "date", "time": "date", "datetime": "date",
            "numeric": "number", "count": "number", "quantity": "number",
            "name": "person", "people": "person",
        }
        v = aliases.get(v, v)
        try:
            return cls(v)
        except ValueError:
            return cls.ENTITY

    def validate_span(self, span: str) -> bool:
        """Lightweight type-consistency check on a candidate answer span."""
        s = str(span or "").strip()
        if not s:
            return False
        if self is AnswerType.YES_NO:
            return s.lower() in {"yes", "no", "true", "false"}
        if self is AnswerType.NUMBER:
            return bool(re.search(r"\d", s)) and len(s) <= 40
        if self is AnswerType.DATE:
            return bool(re.search(r"\b(1[0-9]|20)\d{2}\b", s)) or bool(
                re.search(
                    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
                    r"january|february|march|april|june|july|august|september|"
                    r"october|november|december)\b",
                    s.lower(),
                )
            )
        if self is AnswerType.OTHER:
            # `other` is a fallback bucket. Reject generic filler answers.
            words = s.split()
            if len(words) > 12 or len(s) > 100:
                return False
            generic = {
                "yes", "no", "unknown", "nothing", "various", "multiple",
                "n/a", "none", "everyone", "everything",
            }
            return s.lower() not in generic
        # PERSON / PLACE / ENTITY: minimal heuristic — non-empty, not too long.
        return 0 < len(s) < 120


StepAction = Literal[
    "answer",
    "spawn",
    "refine",
    "verify",
    "route",
    "answer_rejected_escalate",
    "answer_blocked_pending_slots",
    "answer_blocked_min_depth",
]
"""Possible actions the orchestrator can take at each step."""


@dataclass
class Fact:
    """A distilled fact extracted by an investigator subagent."""

    text: str
    confidence: float
    confidence_self: float = 0.0
    confidence_retrieval: float = 0.0
    slot_filled: bool = False
    slot_name: str = ""
    answer_span: str = ""
    support_ids: list[str] = field(default_factory=list)
    source_step: int = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Fact:
        """Deserialize from a dictionary."""
        return cls(**data)


@dataclass
class StepTrace:
    """Trace entry for a single pipeline step."""

    step: int
    action: StepAction
    sub_question: Optional[str] = None
    claim: Optional[str] = None
    fact_added: bool = False
    tokens: int = 0
    slot_name: Optional[str] = None
    route_decision: Optional[str] = None
    route_confidence: Optional[float] = None
    route_draft_answer: Optional[str] = None
    cited_fact_ids: list[int] = field(default_factory=list)
    justification_confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StepTrace:
        """Deserialize from a dictionary."""
        return cls(**data)


@dataclass
class EvidenceCapsule:
    """Bounded evidence returned by an investigator subagent.

    Contains only distilled outputs. Raw passage text is intentionally
    NOT exposed back to the orchestrator.
    """

    answer: str
    fact: Fact
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_docs_total: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "fact": self.fact.to_dict(),
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "retrieved_docs_total": self.retrieved_docs_total,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceCapsule:
        fact_data = data["fact"]
        fact = Fact.from_dict(fact_data) if isinstance(fact_data, dict) else fact_data
        return cls(
            answer=data["answer"],
            fact=fact,
            retrieved_doc_ids=data.get("retrieved_doc_ids", []),
            retrieved_docs_total=data.get("retrieved_docs_total", 0),
        )


@dataclass
class PipelineResult:
    """Complete result from running the adaptive recursive SAGE pipeline."""

    question_id: str
    question: str
    answer: str
    step_trace: list[StepTrace] = field(default_factory=list)
    num_subagent_calls: int = 0
    num_verify_calls: int = 0
    total_tokens: int = 0
    orchestrator_tokens: int = 0
    subagent_tokens: int = 0
    facts_used: list[Fact] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_docs_total: int = 0
    evidence_capsule_limit: int = 0
    fact_memory_capacity: int = 0
    duplicate_subquestion_count: int = 0
    route_decision: str = ""
    route_confidence: float = 0.0
    route_draft_answer: str = ""
    slot_resolution: dict[str, bool] = field(default_factory=dict)
    auto_verify_calls: int = 0
    answer_rejection_count: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary with all metadata."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "step_trace": [t.to_dict() for t in self.step_trace],
            "num_subagent_calls": self.num_subagent_calls,
            "num_verify_calls": self.num_verify_calls,
            "total_tokens": self.total_tokens,
            "orchestrator_tokens": self.orchestrator_tokens,
            "subagent_tokens": self.subagent_tokens,
            "facts_used": [f.to_dict() for f in self.facts_used],
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "retrieved_docs_total": self.retrieved_docs_total,
            "evidence_capsule_limit": self.evidence_capsule_limit,
            "fact_memory_capacity": self.fact_memory_capacity,
            "duplicate_subquestion_count": self.duplicate_subquestion_count,
            "route_decision": self.route_decision,
            "route_confidence": self.route_confidence,
            "route_draft_answer": self.route_draft_answer,
            "slot_resolution": self.slot_resolution,
            "auto_verify_calls": self.auto_verify_calls,
            "answer_rejection_count": self.answer_rejection_count,
            "extras": self.extras,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineResult:
        """Deserialize from a dictionary."""
        step_trace = [
            StepTrace.from_dict(t) if isinstance(t, dict) else t
            for t in data.get("step_trace", [])
        ]
        facts_used = [
            Fact.from_dict(f) if isinstance(f, dict) else f
            for f in data.get("facts_used", [])
        ]
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            answer=data["answer"],
            step_trace=step_trace,
            num_subagent_calls=data.get("num_subagent_calls", 0),
            num_verify_calls=data.get("num_verify_calls", 0),
            total_tokens=data.get("total_tokens", 0),
            orchestrator_tokens=data.get("orchestrator_tokens", 0),
            subagent_tokens=data.get("subagent_tokens", 0),
            facts_used=facts_used,
            retrieved_doc_ids=data.get("retrieved_doc_ids", []),
            retrieved_docs_total=data.get("retrieved_docs_total", 0),
            evidence_capsule_limit=data.get("evidence_capsule_limit", 0),
            fact_memory_capacity=data.get("fact_memory_capacity", 0),
            duplicate_subquestion_count=data.get("duplicate_subquestion_count", 0),
            route_decision=data.get("route_decision", ""),
            route_confidence=data.get("route_confidence", 0.0),
            route_draft_answer=data.get("route_draft_answer", ""),
            slot_resolution=data.get("slot_resolution", {}),
            auto_verify_calls=data.get("auto_verify_calls", 0),
            answer_rejection_count=data.get("answer_rejection_count", 0),
            extras=data.get("extras", {}),
        )
