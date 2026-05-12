"""Core dataclasses shared across AMAS components."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FindingStatus(str, Enum):
    OK = 'ok'
    LOW_CONFIDENCE = 'low_confidence'
    NO_EVIDENCE = 'no_evidence'
    ERROR = 'error'


class Topology(str, Enum):
    SAS = 'sas'
    LINEAR = 'linear'
    FAN_DAG = 'fan_dag'
    BRIDGE_FIRST = 'bridge_first'


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


@dataclass
class ProbeResult:
    """Outcome of one parallel retrieval probe."""
    label: str
    query: str
    chunks: list[RetrievedChunk]
    top1_score: float = 0.0
    score_gap_1to5: float = 0.0
    ne_coverage: float = 0.0
    wh_target_extractable: bool = False
    groundedness: float = 0.0


@dataclass
class SubgoalNode:
    id: int
    question: str
    depends_on: list[int] = field(default_factory=list)
    expected_answer_type: str = 'entity'
    is_final: bool = False
    rationale: str = ''


@dataclass
class Plan:
    subgoals: list[SubgoalNode] = field(default_factory=list)
    final_id: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ''
    planner_tokens: int = 0


@dataclass
class Finding:
    """Output of one Solver call. Pushed to the FindingsBus."""
    sub_question: str
    answer: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: FindingStatus = FindingStatus.OK
    hop_idx: int = 0
    node_id: int = 0
    rewrites_used: int = 0
    tokens: int = 0


@dataclass
class TopologyDecision:
    topology: Topology
    rationale: str
    confidence: float = 0.0
    sas_grounded_chunk_id: str | None = None
