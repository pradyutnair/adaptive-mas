"""Deterministic MAS execution-shape annotator.

SAS is handled before planning by the verifier-gated SAS attempt. This module
does not choose a separate SAS or bridge-first route. It only annotates the
dependency graph as linear or fan-DAG for analysis; execution itself follows
the planner's dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass
from .types import Plan, ProbeResult, Topology, TopologyDecision


@dataclass
class TopologyThresholds:
    sas_high: float = 0.7
    sub_q_low: float = 0.35


def select_topology(
    *,
    plan: Plan,
    probes: list[ProbeResult],
    thresholds: TopologyThresholds | None = None,
) -> TopologyDecision:
    th = thresholds or TopologyThresholds()
    if not probes:
        return TopologyDecision(Topology.LINEAR, rationale='no_probes')

    original = probes[0]
    sub_probes = probes[1:]

    if len(plan.subgoals) <= 1:
        return TopologyDecision(Topology.LINEAR, rationale='single_subgoal')

    has_dependencies = any(node.depends_on for node in plan.subgoals if not node.is_final or len(plan.subgoals) > 1)
    sub_g = [p.groundedness for p in sub_probes]
    low_sub = sum(1 for g in sub_g if g < th.sub_q_low)

    if low_sub > 0 and has_dependencies:
        return TopologyDecision(
            Topology.LINEAR,
            rationale=f'emergent_linear_low_groundedness_on_{low_sub}_subqs',
            confidence=1.0 - (sum(sub_g) / max(len(sub_g), 1)),
        )

    if not has_dependencies:
        return TopologyDecision(
            Topology.FAN_DAG,
            rationale='independent_subgoals_grounded',
            confidence=sum(sub_g) / max(len(sub_g), 1),
        )

    return TopologyDecision(
        Topology.LINEAR,
        rationale='sequential_dependencies',
        confidence=sum(sub_g) / max(len(sub_g), 1),
    )
