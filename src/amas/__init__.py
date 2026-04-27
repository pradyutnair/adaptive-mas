"""AMAS - adaptive multi-agent QA with role-specific LLM clients."""

from .config import Config
from .fact_memory import FactMemory
from .investigator import Investigator
from .llm import LLMClient, LLMResponse, parse_json_object, strip_thinking
from .pipeline import AMASPipeline
from .planner import Planner
from .retriever import Retriever, RetrievalHit
from .types import (
    AnswerType,
    EvidenceCapsule,
    ExecutionPlan,
    Fact,
    PipelineResult,
    StepAction,
    StepTrace,
    SubgoalNode,
)

__all__ = [
    "AMASPipeline",
    "AnswerType",
    "Config",
    "EvidenceCapsule",
    "ExecutionPlan",
    "Fact",
    "FactMemory",
    "Investigator",
    "LLMClient",
    "LLMResponse",
    "PipelineResult",
    "Planner",
    "RetrievalHit",
    "Retriever",
    "StepAction",
    "StepTrace",
    "SubgoalNode",
    "parse_json_object",
    "strip_thinking",
]

__version__ = "0.2.0"
