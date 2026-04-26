"""AMAS - adaptive multi-agent QA with role-specific LLM clients."""

from .config import Config
from .fact_memory import FactMemory
from .investigator import Investigator
from .llm import LLMClient, LLMResponse, parse_json_object, strip_thinking
from .orchestrator import Orchestrator
from .pipeline import AMASPipeline
from .retriever import Retriever, RetrievalHit
from .types import AnswerType, EvidenceCapsule, Fact, PipelineResult, StepAction, StepTrace

__all__ = [
    "AMASPipeline",
    "AnswerType",
    "Config",
    "EvidenceCapsule",
    "Fact",
    "FactMemory",
    "Investigator",
    "LLMClient",
    "LLMResponse",
    "Orchestrator",
    "PipelineResult",
    "RetrievalHit",
    "Retriever",
    "StepAction",
    "StepTrace",
    "parse_json_object",
    "strip_thinking",
]

__version__ = "0.2.0"
