"""Core data types for AMAS v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerType(Enum):
    PERSON = "person"
    PLACE = "place"
    DATE = "date"
    NUMBER = "number"
    YES_NO = "yes_no"
    ENTITY = "entity"
    OTHER = "other"

    @classmethod
    def coerce(cls, value: Any) -> AnswerType:
        if isinstance(value, cls):
            return value
        raw = str(value or "").strip().lower()
        for member in cls:
            if member.value == raw:
                return member
        return cls.ENTITY


@dataclass
class SubgoalNode:
    id: int
    question: str
    depends_on: list[int] = field(default_factory=list)
    answer_type: AnswerType = AnswerType.ENTITY
    rationale: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> SubgoalNode:
        return cls(
            id=int(d.get("id", 0)),
            question=str(d.get("question", "")),
            depends_on=[int(x) for x in (d.get("depends_on") or [])],
            answer_type=AnswerType.coerce(d.get("answer_type")),
            rationale=str(d.get("rationale", "")),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": self.depends_on,
            "answer_type": self.answer_type.value,
            "rationale": self.rationale,
        }


@dataclass
class ExecutionPlan:
    complexity: str
    subgoals: list[SubgoalNode]
    final_answer_type: AnswerType = AnswerType.ENTITY
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "complexity": self.complexity,
            "subgoals": [s.to_dict() for s in self.subgoals],
            "final_answer_type": self.final_answer_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class Fact:
    text: str
    confidence: float = 0.0
    slot_filled: bool = False
    slot_name: str = ""
    answer_span: str = ""
    support_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "slot_filled": self.slot_filled,
            "slot_name": self.slot_name,
            "answer_span": self.answer_span,
            "support_ids": self.support_ids,
        }


@dataclass
class EvidenceCapsule:
    answer: str
    fact: Fact
    subgoal_id: int = 0
    sub_question: str = ""
    answer_type: AnswerType = AnswerType.ENTITY
    evidence_snippets: list[dict[str, str]] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_docs_total: int = 0
    failure_reason: str = ""
    search_queries: list[str] = field(default_factory=list)
    chunk_tokens: int = 0


@dataclass
class StepTrace:
    step: int = 0
    action: str = ""
    sub_question: str = ""
    fact_added: bool = False
    tokens: int = 0
    slot_name: str = ""
    route_decision: str = ""
    route_confidence: float = 0.0
    justification_confidence: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "sub_question": self.sub_question,
            "fact_added": self.fact_added,
            "tokens": self.tokens,
            "slot_name": self.slot_name,
            "route_decision": self.route_decision,
            "metadata": self.metadata,
        }


@dataclass
class PipelineResult:
    question_id: str
    question: str
    answer: str
    step_trace: list[StepTrace] = field(default_factory=list)
    num_subagent_calls: int = 0
    total_tokens: int = 0
    orchestrator_tokens: int = 0
    subagent_tokens: int = 0
    facts_used: list[Fact] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_docs_total: int = 0
    route_decision: str = ""
    route_confidence: float = 0.0
    extras: dict = field(default_factory=dict)
