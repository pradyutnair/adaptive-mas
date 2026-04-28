"""DAG executor: runs subgoal nodes in topological order with parallel levels."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from .investigator import Investigator
from .types import EvidenceCapsule, ExecutionPlan, Fact, StepTrace, SubgoalNode


@dataclass
class DAGResult:
    capsules: list[EvidenceCapsule] = field(default_factory=list)
    trace: list[StepTrace] = field(default_factory=list)
    subagent_tokens: int = 0
    n_subagents: int = 0
    n_searches: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_docs_total: int = 0
    levels: list[list[int]] = field(default_factory=list)
    node_statuses: dict[int, str] = field(default_factory=dict)
    capsules_by_id: dict[int, EvidenceCapsule] = field(default_factory=dict)


class DAGExecutor:
    def __init__(self, investigator: Investigator, max_hop_attempts: int = 3) -> None:
        self.investigator = investigator
        self.max_hop_attempts = max(1, int(max_hop_attempts))

    async def execute(
        self,
        plan: ExecutionPlan,
        original_question: str = "",
        prior_capsules: dict[int, EvidenceCapsule] | None = None,
    ) -> DAGResult:
        levels = self._compute_levels(plan.subgoals)
        result = DAGResult(levels=levels)
        caps: dict[int, EvidenceCapsule] = dict(prior_capsules or {})
        hop_attempts: dict[int, int] = {}

        for nid, cap in caps.items():
            result.capsules.append(cap)
            result.node_statuses[nid] = "verified"

        step = 0
        for level in levels:
            tasks = []
            nodes_to_run = []
            for nid in level:
                if nid in caps:
                    continue
                node = self._node_by_id(plan.subgoals, nid)
                unresolved = [d for d in node.depends_on if d not in caps]
                if unresolved:
                    result.node_statuses[nid] = "blocked"
                    result.capsules.append(self._empty_capsule(node, "dependency unresolved"))
                    continue
                nodes_to_run.append(node)
                tasks.append(self._run_with_retry(node, caps, original_question, hop_attempts))

            if tasks:
                outputs = await asyncio.gather(*tasks)
                for node, (capsule, tokens, searches, status) in zip(nodes_to_run, outputs):
                    result.capsules.append(capsule)
                    result.subagent_tokens += tokens
                    result.n_subagents += 1
                    result.n_searches += searches
                    for did in capsule.retrieved_doc_ids:
                        if did not in result.retrieved_doc_ids:
                            result.retrieved_doc_ids.append(did)
                    result.retrieved_docs_total += capsule.retrieved_docs_total
                    result.node_statuses[node.id] = status
                    if status == "verified":
                        caps[node.id] = capsule

                    step += 1
                    result.trace.append(StepTrace(
                        step=step, action="investigate",
                        sub_question=capsule.sub_question,
                        fact_added=capsule.fact.slot_filled,
                        tokens=tokens,
                        slot_name=f"subgoal_{node.id}",
                        route_decision=status,
                        justification_confidence=capsule.fact.confidence,
                        metadata={
                            "subgoal_id": node.id,
                            "attempt": hop_attempts.get(node.id, 1),
                            "answer": capsule.answer,
                            "status": status,
                            "failure_reason": capsule.failure_reason,
                        },
                    ))

        result.capsules_by_id = caps
        return result

    async def _run_with_retry(
        self,
        node: SubgoalNode,
        caps: dict[int, EvidenceCapsule],
        original_question: str,
        hop_attempts: dict[int, int],
    ) -> tuple[EvidenceCapsule, int, int, str]:
        resolved_q = self._resolve_question(node.question, caps)
        hint = self._build_hint(node, caps, original_question)
        total_tokens = 0
        total_searches = 0
        query_override = None

        for attempt in range(self.max_hop_attempts):
            hop_attempts[node.id] = attempt + 1
            run_node = SubgoalNode(
                id=node.id, question=resolved_q,
                depends_on=node.depends_on, answer_type=node.answer_type,
                rationale=node.rationale,
            )
            capsule, tokens = await self.investigator.investigate_node(
                run_node, hint=hint, query_override=query_override,
                parent_question=original_question,
            )
            total_tokens += tokens
            total_searches += self.investigator.last_searches_used

            if capsule.fact.slot_filled and capsule.answer:
                return capsule, total_tokens, total_searches, "verified"

            if attempt < self.max_hop_attempts - 1:
                new_q, rw_tokens = await self.investigator.rewrite_query(
                    run_node, hint=hint,
                    previous_query=(capsule.search_queries[-1] if capsule.search_queries else resolved_q),
                    previous_answer=capsule.answer,
                    previous_justification=capsule.fact.text,
                )
                total_tokens += rw_tokens
                query_override = new_q

        return capsule, total_tokens, total_searches, "failed"

    @staticmethod
    def _resolve_question(question: str, caps: dict[int, EvidenceCapsule]) -> str:
        resolved = question
        for sid, cap in caps.items():
            answer = cap.answer or cap.fact.answer_span
            if not answer:
                continue
            for pat in [rf"\[result_{sid}\]", rf"\[result from step {sid}\]",
                        rf"\[entity from step {sid}\]"]:
                resolved = re.sub(pat, answer, resolved, flags=re.IGNORECASE)
        return resolved

    @staticmethod
    def _build_hint(node: SubgoalNode, caps: dict[int, EvidenceCapsule], orig_q: str) -> str:
        parts = []
        if orig_q:
            parts.append(f"Original question: {orig_q.strip()}")
        if node.rationale:
            parts.append(node.rationale.strip())
        for dep in node.depends_on:
            cap = caps.get(dep)
            if cap and (cap.answer or cap.fact.answer_span):
                parts.append(f"Subgoal {dep} answer: {cap.answer or cap.fact.answer_span}. {cap.fact.text}")
        return " ".join(parts)

    @staticmethod
    def _empty_capsule(node: SubgoalNode, reason: str) -> EvidenceCapsule:
        return EvidenceCapsule(
            answer="",
            fact=Fact(text="", confidence=0.0, slot_filled=False, slot_name=f"subgoal_{node.id}"),
            subgoal_id=node.id, sub_question=node.question, answer_type=node.answer_type,
            failure_reason=reason,
        )

    @staticmethod
    def _compute_levels(nodes: list[SubgoalNode]) -> list[list[int]]:
        pending = {n.id: set(n.depends_on) for n in nodes}
        emitted: set[int] = set()
        levels: list[list[int]] = []
        while pending:
            ready = sorted(nid for nid, deps in pending.items() if deps.issubset(emitted))
            if not ready:
                break
            levels.append(ready)
            for nid in ready:
                emitted.add(nid)
                pending.pop(nid, None)
        return levels

    @staticmethod
    def _node_by_id(nodes: list[SubgoalNode], nid: int) -> SubgoalNode:
        for n in nodes:
            if n.id == nid:
                return n
        raise ValueError(f"missing subgoal {nid}")
