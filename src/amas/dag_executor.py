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
    node_statuses: dict[int, str] = field(default_factory=dict)
    terminal_ids: list[int] = field(default_factory=list)
    retry_count: int = 0
    blackboard_state: dict = field(default_factory=dict)


@dataclass
class HopState:
    id: int
    question: str
    depends_on: list[int] = field(default_factory=list)
    answer_type: str = "entity"
    rationale: str = ""
    status: str = "pending"
    resolved_question: str = ""
    answer: str = ""
    confidence: float = 0.0
    support_ids: list[str] = field(default_factory=list)
    evidence_snippets: list[dict[str, str]] = field(default_factory=list)
    failure_reason: str = ""
    attempt_count: int = 0
    max_attempts: int = 2
    query_override: str = ""
    query_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": self.depends_on,
            "answer_type": self.answer_type,
            "rationale": self.rationale,
            "status": self.status,
            "resolved_question": self.resolved_question,
            "answer": self.answer,
            "confidence": self.confidence,
            "support_ids": self.support_ids,
            "evidence_snippets": self.evidence_snippets,
            "failure_reason": self.failure_reason,
            "attempt_count": self.attempt_count,
            "query_history": self.query_history,
        }


class DAGExecutor:
    """Run independent DAG nodes in parallel and dependent nodes in order."""

    def __init__(self, investigator: Investigator, max_hop_attempts: int = 2) -> None:
        self.investigator = investigator
        self.max_hop_attempts = max(1, int(max_hop_attempts))

    async def execute(self, plan: ExecutionPlan) -> DAGExecutionResult:
        levels = self._levels(plan.subgoals)
        capsules_by_id: dict[int, EvidenceCapsule] = {}
        result = DAGExecutionResult(levels=levels, terminal_ids=self._terminal_ids(plan.subgoals))
        hop_states = {
            node.id: HopState(
                id=node.id,
                question=node.question,
                depends_on=list(node.depends_on),
                answer_type=node.answer_type.value,
                rationale=node.rationale,
                max_attempts=self.max_hop_attempts,
            )
            for node in plan.subgoals
        }
        step = 0

        while True:
            ready_nodes = [
                node for node in plan.subgoals
                if self._is_actionable(hop_states[node.id], hop_states)
            ]
            if not ready_nodes:
                self._block_unreachable_hops(hop_states)
                break

            tasks = [
                self._run_node(node, capsules_by_id, hop_states[node.id])
                for node in ready_nodes
            ]
            outputs = await asyncio.gather(*tasks)

            for node, (capsule, tokens, searches, status) in zip(ready_nodes, outputs):
                hop = hop_states[node.id]
                result.capsules.append(capsule)
                result.subagent_tokens += tokens
                result.chunk_tokens += capsule.chunk_tokens
                result.n_subagents += 1
                result.n_searches += searches
                for doc_id in capsule.retrieved_doc_ids:
                    if doc_id not in result.retrieved_doc_ids:
                        result.retrieved_doc_ids.append(doc_id)
                result.retrieved_docs_total += capsule.retrieved_docs_total

                self._record_hop_attempt(hop, capsule, status)
                rewrite_event = None
                if status == "verified":
                    capsules_by_id[node.id] = capsule
                elif hop.attempt_count < hop.max_attempts:
                    next_query, rewrite_tokens = await self.investigator.rewrite_query(
                        node=SubgoalNode(
                            id=node.id,
                            question=hop.resolved_question or node.question,
                            depends_on=node.depends_on,
                            answer_type=node.answer_type,
                            rationale=node.rationale,
                        ),
                        hint=self._dependency_hint(node, capsules_by_id, hop),
                        previous_query=(capsule.search_queries[-1] if capsule.search_queries else hop.resolved_question),
                        previous_answer=capsule.answer,
                        previous_justification=capsule.fact.text,
                        rejection_reason=hop.failure_reason,
                    )
                    result.subagent_tokens += rewrite_tokens
                    hop.query_override = next_query
                    hop.status = "retry_pending"
                    rewrite_event = {
                        "tokens": rewrite_tokens,
                        "sub_question": hop.resolved_question or node.question,
                        "metadata": {
                            "subgoal_id": node.id,
                            "attempt": hop.attempt_count,
                            "previous_query": capsule.search_queries[-1] if capsule.search_queries else "",
                            "next_query": next_query,
                            "failure_reason": hop.failure_reason,
                        },
                    }
                else:
                    hop.status = "stuck"

                result.node_statuses[node.id] = self._public_status(hop.status)
                step += 1
                result.trace.append(
                    StepTrace(
                        step=step,
                        action="investigate",
                        sub_question=capsule.sub_question,
                        fact_added=capsule.fact.slot_filled,
                        tokens=tokens,
                        slot_name=f"subgoal_{node.id}",
                        route_decision=hop.status,
                        justification_confidence=capsule.fact.confidence,
                        metadata={
                            "subgoal_id": node.id,
                            "depends_on": node.depends_on,
                            "attempt": hop.attempt_count,
                            "query": capsule.search_queries[-1] if capsule.search_queries else "",
                            "answer": capsule.answer,
                            "status": hop.status,
                            "failure_reason": hop.failure_reason,
                            "support_ids": capsule.fact.support_ids,
                            "evidence_snippets": capsule.evidence_snippets,
                            "chunk_tokens": capsule.chunk_tokens,
                        },
                    )
                )
                if rewrite_event:
                    step += 1
                    result.trace.append(
                        StepTrace(
                            step=step,
                            action="rewrite",
                            sub_question=rewrite_event["sub_question"],
                            tokens=rewrite_event["tokens"],
                            slot_name=f"subgoal_{node.id}",
                            route_decision="retry",
                            metadata=rewrite_event["metadata"],
                        )
                    )

            if not any(hop.status in {"pending", "retry_pending"} for hop in hop_states.values()):
                break

        result.node_statuses = {
            hop_id: self._public_status(hop.status)
            for hop_id, hop in hop_states.items()
        }
        result.retry_count = sum(max(0, hop.attempt_count - 1) for hop in hop_states.values())
        result.blackboard_state = {
            "retry_count": result.retry_count,
            "hops": [
                hop.to_dict()
                for hop in sorted(hop_states.values(), key=lambda item: item.id)
            ],
        }
        return result

    async def _run_node(
        self,
        node: SubgoalNode,
        capsules_by_id: dict[int, EvidenceCapsule],
        hop: HopState,
    ) -> tuple[EvidenceCapsule, int, int, str]:
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
                failure_reason="dependency unresolved",
            )
            return empty, 0, 0, "blocked"

        resolved = self._resolve_question(node.question, capsules_by_id)
        hop.status = "investigating"
        hop.attempt_count += 1
        hop.resolved_question = resolved
        dependency_hint = self._dependency_hint(node, capsules_by_id, hop)
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
            query_override=hop.query_override or None,
        )
        status = "verified" if capsule.fact.slot_filled and capsule.answer else "failed"
        return capsule, tokens, self.investigator.last_searches_used, status

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
        hop: HopState | None = None,
    ) -> str:
        facts = []
        for dep_id in node.depends_on:
            capsule = capsules_by_id.get(dep_id)
            if not capsule:
                continue
            answer = DAGExecutor._dependency_answer(capsule)
            if answer:
                facts.append(f"Subgoal {dep_id} answer: {answer}. {capsule.fact.text}")
        if hop and hop.failure_reason and hop.attempt_count > 0:
            facts.append(f"Previous attempt failed: {hop.failure_reason}.")
        return " ".join(facts)

    @staticmethod
    def _is_actionable(hop: HopState, hop_states: dict[int, HopState]) -> bool:
        if hop.status not in {"pending", "retry_pending"}:
            return False
        return all(
            hop_states.get(dep_id) and hop_states[dep_id].status == "resolved"
            for dep_id in hop.depends_on
        )

    @staticmethod
    def _block_unreachable_hops(hop_states: dict[int, HopState]) -> None:
        for hop in hop_states.values():
            if hop.status not in {"pending", "retry_pending"}:
                continue
            if any(
                dep_id in hop_states and hop_states[dep_id].status in {"stuck", "blocked"}
                for dep_id in hop.depends_on
            ):
                hop.status = "blocked"
                hop.failure_reason = "dependency unresolved"

    @staticmethod
    def _record_hop_attempt(hop: HopState, capsule: EvidenceCapsule, status: str) -> None:
        hop.answer = capsule.answer
        hop.confidence = capsule.fact.confidence
        hop.support_ids = list(capsule.fact.support_ids)
        hop.evidence_snippets = list(capsule.evidence_snippets)
        hop.failure_reason = capsule.failure_reason
        for query in capsule.search_queries:
            if query and query not in hop.query_history:
                hop.query_history.append(query)
        if status == "verified":
            hop.status = "resolved"
        elif status == "blocked":
            hop.status = "blocked"
        else:
            hop.status = "failed"

    @staticmethod
    def _public_status(status: str) -> str:
        if status == "resolved":
            return "verified"
        if status == "blocked":
            return "blocked"
        return "failed"

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
                raise ValueError("invalid execution plan: no dependency-free nodes available")
            levels.append(ready)
            for node_id in ready:
                emitted.add(node_id)
                pending.pop(node_id, None)

        return levels

    @staticmethod
    def _terminal_ids(nodes: list[SubgoalNode]) -> list[int]:
        if not nodes:
            return []
        depended_on: set[int] = set()
        for node in nodes:
            depended_on.update(node.depends_on)
        return sorted(node.id for node in nodes if node.id not in depended_on)

    @staticmethod
    def _node_by_id(nodes: list[SubgoalNode], node_id: int) -> SubgoalNode:
        for node in nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"missing subgoal id {node_id}")
