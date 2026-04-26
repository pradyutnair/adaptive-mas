"""Adaptive Recursive SAGE — adaptive multi-hop QA via recursive subagent spawning."""

from .types import (
    Fact,
    StepAction,
    StepTrace,
    EvidenceCapsule,
    PipelineResult,
)
from .fact_memory import FactMemory
from .investigator import Investigator
from .orchestrator import Orchestrator
from .pipeline import AdaptiveRecursivePipeline

__all__ = [
    "Fact",
    "StepAction",
    "StepTrace",
    "EvidenceCapsule",
    "PipelineResult",
    "FactMemory",
    "Investigator",
    "Orchestrator",
    "AdaptiveRecursivePipeline",
]

__version__ = "0.1.0"
