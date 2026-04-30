"""AMAS: probe-driven multi-agent collaborative search for multi-hop QA.

Roles
-----
- Planner    (Qwen3-8B + thinking, dspy.ChainOfThought): atomic decomposition.
- Probe      (no LLM, parallel retrieval): grounds the plan in the corpus.
- Topology   (no LLM, deterministic): selects {SAS, Linear, FanDAG, BridgeFirst}.
- Solver     (GPT-4o-mini, dspy module + retrieval tool): per-node extraction.
- Synth      (GPT-4o-mini, dspy.ChainOfThought): wh-target-aligned final span.

All cross-agent communication is via the FindingsBus (working memory).
Top-K is fixed at 5 on every retrieval call. Solvers may re-retrieve up to
max_retrievals_per_solver times with refined queries.
"""
from .types import Finding, FindingStatus, ProbeResult, Plan, Topology
from .retriever import Retriever
from .working_memory import FindingsBus

__all__ = [
    'Finding',
    'FindingStatus',
    'ProbeResult',
    'Plan',
    'Topology',
    'Retriever',
    'FindingsBus',
]
