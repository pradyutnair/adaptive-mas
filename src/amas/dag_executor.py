"""Dependency-graph execution for AMAS v2."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from .investigator import Investigator
from .types import EvidenceCapsule, ExecutionPlan, Fact, StepTrace, SubgoalNode


@dataclass
class DAGExecutionResult:
    capsules: list[EvidenceCapsule] = field(default_factory=list)
    trace: list[StepTrace] = field(default_factory=list)
    subagent_tokens: int = 0
    chunk_tokens: int = 0
    n_subagents: int = 0
    n_searches: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_docs_total: int = 0
    levels: list[list[int]] = field(default_factory=list)


class DAGExecutor:
    """Run independent DAG nodes in parallel and dependent nodes in order."""

    def __init__(self, investigator: Investigator) -> None:
        self.investigator = investigator

    async def execute(self, plan: ExecutionPlan) -> DAGExecutionResult:
        levels = self._levels(plan.subgoals)
        capsules_by_id: dict[int, EvidenceCapsule] = {}
        result = DAGExecutionResult(levels=levels)
        step = 0

        for level_idx, level in enumerate(levels):
            ready_nodes = [self._node_by_id(plan.subgoals, node_id) for node_id in level]
            tasks = [
                self._run_node(node, capsules_by_id)
                for node in ready_nodes
            ]
            outputs = await asyncio.gather(*tasks)

            for node, (capsule, tokens, searches) in zip(ready_nodes, outputs):
                capsules_by_id[node.id] = capsule
                result.capsules.append(capsule)
                result.subagent_tokens += tokens
                result.chunk_tokens += capsule.chunk_tokens
                result.n_subagents += 1
                result.n_searches += searches
                for doc_id in capsule.retrieved_doc_ids:
                    if doc_id not in result.retrieved_doc_ids:
                        result.retrieved_doc_ids.append(doc_id)
                result.retrieved_docs_total += capsule.retrieved_docs_total
                step += 1
                result.trace.append(
                    StepTrace(
                        step=step,
                        action="investigate",
                        sub_question=capsule.sub_question,
                        fact_added=capsule.fact.slot_filled,
                        tokens=tokens,
                        slot_name=f"subgoal_{node.id}",
                        route_decision=f"dag_level_{level_idx}",
                        justification_confidence=capsule.fact.confidence,
                        metadata={
                            "subgoal_id": node.id,
                            "depends_on": node.depends_on,
                            "answer": capsule.answer,
                            "support_ids": capsule.fact.support_ids,
                            "chunk_tokens": capsule.chunk_tokens,
                        },
                    )
                )

        return result

    async def _run_node(
        self,
        node: SubgoalNode,
        capsules_by_id: dict[int, EvidenceCapsule],
    ) -> tuple[EvidenceCapsule, int, int]:
        unresolved = [
            dep_id for dep_id in node.depends_on
            if not self._dependency_answer(capsules_by_id.get(dep_id))
        ]
        if unresolved:
            empty = EvidenceCapsule(
                answer="",
                fact=Fact(
                    text="",
                    confidence=0.0,
                    confidence_self=0.0,
                    confidence_retrieval=0.0,
                    slot_filled=False,
                    slot_name=f"subgoal_{node.id}",
                    answer_span="",
                    support_ids=[],
                ),
                subgoal_id=node.id,
                sub_question=node.question,
                answer_type=node.answer_type,
                retrieved_doc_ids=[],
                retrieved_docs_total=0,
            )
            return empty, 0, 0

        resolved = self._resolve_question(node.question, capsules_by_id)
        dependency_hint = self._dependency_hint(node, capsules_by_id)
        hint = " ".join(part for part in [node.rationale.strip(), dependency_hint] if part).strip()
        run_node = SubgoalNode(
            id=node.id,
            question=resolved,
            depends_on=node.depends_on,
            answer_type=node.answer_type,
            rationale=node.rationale,
        )
        capsule, tokens = await self.investigator.investigate_node(
            node=run_node,
            hint=hint,
            slot_name=f"subgoal_{node.id}",
        )
        return capsule, tokens, self.investigator.last_searches_used

    @staticmethod
    def _dependency_answer(capsule: EvidenceCapsule | None) -> str:
        if not capsule:
            return ""
        return capsule.answer or capsule.fact.answer_span

    @staticmethod
    def _resolve_question(
        question: str,
        capsules_by_id: dict[int, EvidenceCapsule],
    ) -> str:
        resolved = question
        for subgoal_id, capsule in capsules_by_id.items():
            answer = DAGExecutor._dependency_answer(capsule)
            patterns = [
                rf"\[result_{subgoal_id}\]",
                rf"\[result from step {subgoal_id}\]",
                rf"\[entity from step {subgoal_id}\]",
            ]
            for pattern in patterns:
                resolved = re.sub(pattern, answer, resolved, flags=re.IGNORECASE)
        return resolved

    @staticmethod
    def _dependency_hint(
        node: SubgoalNode,
        capsules_by_id: dict[int, EvidenceCapsule],
    ) -> str:
        facts = []
        for dep_id in node.depends_on:
            capsule = capsules_by_id.get(dep_id)
            if not capsule:
                continue
            answer = DAGExecutor._dependency_answer(capsule)
            if answer:
                facts.append(f"Subgoal {dep_id} answer: {answer}. {capsule.fact.text}")
        return " ".join(facts)

    @staticmethod
    def _levels(nodes: list[SubgoalNode]) -> list[list[int]]:
        pending = {node.id: set(node.depends_on) for node in nodes}
        emitted: set[int] = set()
        levels: list[list[int]] = []

        while pending:
            ready = sorted(
                node_id for node_id, deps in pending.items()
                if deps.issubset(emitted)
            )
            if not ready:
                ready = [min(pending)]
            levels.append(ready)
            for node_id in ready:
                emitted.add(node_id)
                pending.pop(node_id, None)

        return levels

    @staticmethod
    def _node_by_id(nodes: list[SubgoalNode], node_id: int) -> SubgoalNode:
        for node in nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"missing subgoal id {node_id}")
