"""AMAS pipeline orchestrator.

Plan -> Probe -> Route (LLM or heuristic) -> Execute -> Synth flow.
Supports SAS (direct) and DAG (decompose) lanes with adaptive topology.

Public API: AmasPipeline.run(question, qid) -> AmasResult.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
import dspy
from .planner import run_planner
from .probe import probe_all
from .retriever import Retriever
from .solver import run_solver
from .synthesizer import run_synthesizer
from .topology import TopologyThresholds, select_topology
from .multi_plan import run_multi_plan_grpo
from .bridge_resolver import run_bridge_resolver, should_resolve_bridge
from .signals import compute_groundedness
from .types import (
    EvidenceCapsule,
    Finding,
    FindingStatus,
    Plan,
    ProbeResult,
    RetrievedChunk,
    SubgoalNode,
    Topology,
    TopologyDecision,
)
from .working_memory import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class AmasResult:
    question_id: str
    question: str
    answer: str
    answer_type: str = "other"
    justification: str = ""
    support_ids: list[str] = field(default_factory=list)
    plan_subgoals: int = 0
    topology: str = ""
    topology_rationale: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    probe_groundedness: list[float] = field(default_factory=list)
    total_tokens: int = 0
    planner_tokens: int = 0
    solver_tokens: int = 0
    synth_tokens: int = 0
    rewrite_tokens: int = 0
    n_retrieval_calls: int = 0
    n_solvers_invoked: int = 0
    repair_invoked: bool = False
    wallclock_seconds: float = 0.0
    bridge_resolved: str = ""
    bridge_resolver_tokens: int = 0
    multi_plan_rewards: list = field(default_factory=list)
    multi_plan_subgoal_counts: list = field(default_factory=list)
    multi_plan_temperatures: list = field(default_factory=list)
    sas_collapse: bool = False
    sas_escalated: bool = False
    sas_attempt_tokens: int = 0
    sas_attempt_confidence: float = 0.0
    sas_attempt_grounded: bool = False
    sas_verifier_passed: bool = False
    sas_verifier_verdict: str = ""
    sas_verifier_tokens: int = 0
    # New fields for sweep analysis
    route: str = ""
    router_reason: str = ""
    router_tokens: int = 0
    verifier_calls: int = 0
    verifier_accepts: int = 0
    verifier_rejects: int = 0
    verifier_tokens: int = 0
    working_memory_capsules: list[dict] = field(default_factory=list)
    per_node_retrievals: list[dict] = field(default_factory=list)
    per_node_latency: list[dict] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)


@dataclass
class AmasPipelineConfig:
    max_retrievals_per_solver: int = 3
    repair_enabled: bool = True
    topology_thresholds: TopologyThresholds = field(default_factory=TopologyThresholds)
    excerpt_chars: int = 700
    experience_library: str = ""
    use_multi_plan: bool = False
    K_plans: int = 3
    plan_temperatures: tuple = (0.4, 0.7, 0.9)
    use_bridge_resolver: bool = False
    bridge_g_threshold: float = 0.45
    use_sas_collapse: bool = False
    tau_sas_g: float = 0.55
    tau_sas_conf: float = 0.75
    synth_recursion_rounds: int = 1
    adaptive_solver_budget: bool = False
    min_retrievals_per_solver: int = 1
    medium_retrievals_per_solver: int = 2
    max_repairs: int = 2
    # New config options
    use_verifier: bool = False
    use_llm_router: bool = False
    use_context_aware_retrieval: bool = True


class AmasPipeline:
    def __init__(
        self,
        *,
        planner_lm: dspy.LM,
        worker_lm: dspy.LM,
        synth_lm: dspy.LM,
        retriever: Retriever,
        config: AmasPipelineConfig | None = None,
        sas_lm: dspy.LM | None = None,
        router_lm: dspy.LM | None = None,
    ) -> None:
        self.planner_lm = planner_lm
        self.worker_lm = worker_lm
        self.synth_lm = synth_lm
        self.sas_lm = sas_lm or worker_lm
        self.router_lm = router_lm or worker_lm
        self.retriever = retriever
        self.config = config or AmasPipelineConfig()

    async def run(self, question: str, qid: str = "") -> AmasResult:
        t0 = time.time()
        bus = WorkingMemory()
        result = AmasResult(question_id=qid, question=question, answer="")
        result.config_snapshot = {
            "max_retrievals_per_solver": self.config.max_retrievals_per_solver,
            "use_verifier": self.config.use_verifier,
            "use_llm_router": self.config.use_llm_router,
            "use_sas_collapse": self.config.use_sas_collapse,
            "use_context_aware_retrieval": self.config.use_context_aware_retrieval,
            "repair_enabled": self.config.repair_enabled,
            "max_repairs": self.config.max_repairs,
        }
        solver_chunks_by_node: dict[int, list] = {}

        # Step 0: probe original Q
        original_chunks = await self.retriever.retrieve(question)
        result.n_retrieval_calls += 1
        g_original, comp_o = compute_groundedness(question, original_chunks)

        # Step 0.5: Route question (LLM router or heuristic + SAS attempt)
        if self.config.use_llm_router:
            from .router import route_question
            routing = route_question(
                router_lm=self.router_lm,
                question=question,
                chunks=original_chunks,
            )
            result.router_tokens = routing.tokens
            result.router_reason = routing.reason
            if routing.route == "direct":
                result.route = "sas"
                from .sas_attempt import try_sas_attempt
                sas = await asyncio.to_thread(
                    try_sas_attempt,
                    sas_lm=self.sas_lm,
                    question=question,
                    chunks=original_chunks,
                    probe_groundedness=1.0,
                    tau_g=0.0,
                    tau_conf=self.config.tau_sas_conf,
                )
                result.sas_attempt_tokens = sas.extraction_tokens
                result.sas_attempt_confidence = sas.confidence
                result.sas_attempt_grounded = sas.grounded_in_chunks
                result.sas_verifier_passed = sas.verifier_passed
                result.sas_verifier_verdict = sas.verifier_verdict
                result.sas_verifier_tokens = sas.verifier_tokens
                if sas.accepted:
                    result.sas_collapse = True
                    result.topology = "verified_sas"
                    result.topology_rationale = f"LLM router -> direct, SAS accepted (verdict={sas.verifier_verdict})"
                    result.answer = sas.answer
                    result.answer_type = sas.answer_type
                    result.justification = sas.rationale
                    result.support_ids = [c.chunk_id for c in original_chunks[:5]]
                    result.probe_groundedness = [round(g_original, 4)]
                    result.total_tokens = result.sas_attempt_tokens + result.router_tokens
                    result.wallclock_seconds = round(time.time() - t0, 3)
                    result.n_solvers_invoked = 1
                    return result
                result.sas_escalated = True
                result.route = "dag"
            else:
                result.route = "dag"
        elif self.config.use_sas_collapse:
            result.route = "sas_attempt"
            from .sas_attempt import try_sas_attempt
            sas = await asyncio.to_thread(
                try_sas_attempt,
                sas_lm=self.sas_lm,
                question=question,
                chunks=original_chunks,
                probe_groundedness=g_original,
                tau_g=self.config.tau_sas_g,
                tau_conf=self.config.tau_sas_conf,
            )
            result.sas_attempt_tokens = sas.extraction_tokens
            result.sas_attempt_confidence = sas.confidence
            result.sas_attempt_grounded = sas.grounded_in_chunks
            result.sas_verifier_passed = sas.verifier_passed
            result.sas_verifier_verdict = sas.verifier_verdict
            result.sas_verifier_tokens = sas.verifier_tokens
            if sas.accepted:
                result.sas_collapse = True
                result.route = "sas"
                result.topology = "verified_sas"
                result.topology_rationale = f"verified SAS accepted (g={g_original:.3f}, verdict={sas.verifier_verdict})"
                result.answer = sas.answer
                result.answer_type = sas.answer_type
                result.justification = sas.rationale
                result.support_ids = [c.chunk_id for c in original_chunks[:5]]
                result.probe_groundedness = [round(g_original, 4)]
                result.total_tokens = result.sas_attempt_tokens
                result.wallclock_seconds = round(time.time() - t0, 3)
                result.n_solvers_invoked = 1
                return result
            result.sas_escalated = True
            result.route = "dag"
        else:
            result.route = "dag"

        # Step 1: bridge resolution if probe-original groundedness is low
        bridge_hint = ""
        if self.config.use_bridge_resolver and should_resolve_bridge(
            probe_original_g=g_original, g_low_threshold=self.config.bridge_g_threshold
        ):
            br = await run_bridge_resolver(
                resolver_lm=self.planner_lm,
                original_question=question,
                original_probe_chunks=original_chunks,
            )
            if br.bridge_entity:
                bridge_hint = (
                    f"Hint: the bridge entity in this question "
                    f"(the phrase '{br.bridge_phrase}') resolves to "
                    f"'{br.bridge_entity}'. Use this resolution when constructing sub-questions."
                )
                result.bridge_resolved = br.bridge_entity
            result.bridge_resolver_tokens = br.tokens

        enhanced_experience = self.config.experience_library
        if bridge_hint:
            enhanced_experience = (enhanced_experience + " " + bridge_hint).strip() if enhanced_experience else bridge_hint

        # Step 2: planning
        from .types import ProbeResult as _PR
        original_probe = _PR(
            label="original", query=question, chunks=original_chunks,
            top1_score=comp_o["top1_score"], score_gap_1to5=comp_o["score_gap_1to5"],
            ne_coverage=comp_o["ne_coverage"],
            wh_target_extractable=bool(comp_o["wh_target_extractable"]),
            groundedness=g_original,
        )

        if self.config.use_multi_plan:
            mp = await run_multi_plan_grpo(
                planner_lm=self.planner_lm,
                retriever=self.retriever,
                question=question,
                experience=enhanced_experience,
                K=self.config.K_plans,
                temperatures=self.config.plan_temperatures,
            )
            plan = mp.chosen_plan
            result.planner_tokens = mp.planner_tokens
            result.plan_subgoals = len(plan.subgoals)
            result.multi_plan_rewards = [round(r, 4) for r in mp.candidate_rewards]
            result.multi_plan_subgoal_counts = list(mp.candidate_plans_subgoals)
            result.multi_plan_temperatures = list(mp.candidate_temperatures)
            chosen_sub_probes = [p for p in mp.chosen_probes if p.label != "original"]
            probes = [original_probe] + chosen_sub_probes
            mp_retrievals = sum(len(p.chunks) > 0 for p in mp.chosen_probes)
            result.n_retrieval_calls += mp_retrievals
        else:
            plan = await asyncio.to_thread(run_planner, self.planner_lm, question, enhanced_experience)
            result.planner_tokens = plan.planner_tokens
            result.plan_subgoals = len(plan.subgoals)

            sub_questions = [bus.interpolate(n.question) for n in plan.subgoals]
            sub_probes_full = await probe_all(
                retriever=self.retriever,
                original_question=question,
                sub_questions=sub_questions,
            )
            chosen_sub_probes = [p for p in sub_probes_full if p.label != "original"]
            probes = [original_probe] + chosen_sub_probes
            result.n_retrieval_calls += len(chosen_sub_probes) + 1

        result.probe_groundedness = [round(p.groundedness, 4) for p in probes]

        decision = select_topology(plan=plan, probes=probes, thresholds=self.config.topology_thresholds)
        result.topology = decision.topology.value
        result.topology_rationale = decision.rationale

        if decision.topology == Topology.SAS:
            await self._execute_sas(question, plan, probes, bus, result, solver_chunks_by_node)
        else:
            await self._execute_dag(question, plan, probes, bus, result, decision, solver_chunks_by_node)

        if self.config.repair_enabled:
            await self._maybe_repair(plan, probes, bus, result)

        final_node = self._find_final_node(plan)
        final_evidence = self._final_evidence_chunks(final_node, plan, probes, bus)

        union: list = []
        seen_ids: set = set()
        def _add(chunks):
            for c in chunks:
                cid = getattr(c, "chunk_id", None)
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    union.append(c)
        _add(final_evidence)
        for nid_key, chunks in solver_chunks_by_node.items():
            _add(chunks)
        for p in probes:
            _add(p.chunks)
        union_capped = union[:20]

        try:
            if self.config.synth_recursion_rounds > 1:
                from .synth_refine import run_synth_recursion
                synth_obj, synth_tokens = await asyncio.to_thread(
                    run_synth_recursion,
                    synth_lm=self.synth_lm,
                    original_question=question,
                    bus=bus,
                    final_evidence=union_capped,
                    experience=self.config.experience_library,
                    rounds=self.config.synth_recursion_rounds,
                )
            else:
                synth_obj, synth_tokens = await asyncio.to_thread(
                    run_synthesizer,
                    synth_lm=self.synth_lm,
                    original_question=question,
                    bus=bus,
                    final_evidence=union_capped,
                    experience=self.config.experience_library,
                )
        except Exception as e:
            logger.warning("synth call failed (%s); falling back to best finding", type(e).__name__)
            final_node = self._find_final_node(plan)
            best_finding = bus.findings_by_node.get(final_node.id) if final_node else None
            if best_finding is None:
                for f in reversed(bus.all()):
                    if f.answer:
                        best_finding = f
                        break
            if best_finding and best_finding.answer:
                synth_obj = {
                    "answer": best_finding.answer,
                    "answer_type": "other",
                    "justification": f"synth_fallback (status={best_finding.status.value})",
                    "support_ids": best_finding.evidence_ids or [],
                }
            else:
                synth_obj = {"answer": "", "answer_type": "other", "justification": "synth_failed_no_findings", "support_ids": []}
            synth_tokens = 0
        result.synth_tokens = synth_tokens
        result.answer = str(synth_obj.get("answer", "")).strip()
        result.answer_type = str(synth_obj.get("answer_type", "other"))
        result.justification = str(synth_obj.get("justification", ""))[:300]
        sup = synth_obj.get("support_ids", [])
        if isinstance(sup, list):
            result.support_ids = [str(x) for x in sup if x]

        for f in bus.all():
            result.findings.append({
                "node_id": f.node_id,
                "sub_question": f.sub_question,
                "answer": f.answer,
                "confidence": f.confidence,
                "status": f.status.value,
                "evidence_ids": f.evidence_ids,
            })

        result.working_memory_capsules = [c.to_dict() for c in bus.all_capsules()]

        result.total_tokens = (
            result.planner_tokens
            + result.solver_tokens
            + result.synth_tokens
            + result.rewrite_tokens
            + result.sas_attempt_tokens
            + result.bridge_resolver_tokens
            + result.router_tokens
            + result.verifier_tokens
        )
        result.wallclock_seconds = round(time.time() - t0, 3)
        return result

    async def _execute_sas(self, question, plan, probes, bus, result, solver_chunks_by_node):
        original = probes[0]
        node = self._find_final_node(plan)
        sr = await run_solver(
            solver_lm=self.worker_lm, rewrite_lm=self.worker_lm,
            retriever=self.retriever, sub_question=question,
            expected_answer_type=node.expected_answer_type if node else "entity",
            starting_chunks=original.chunks,
            node_id=node.id if node else 1, hop_idx=0,
            max_retrievals=self._retrieval_budget_for_node(node, plan, probes) if node else self.config.max_retrievals_per_solver,
            experience=self.config.experience_library,
            use_verifier=self.config.use_verifier,
            verifier_lm=self.worker_lm if self.config.use_verifier else None,
        )
        self._accumulate_solver_result(sr, node.id if node else 1, bus, result, solver_chunks_by_node)

    async def _execute_dag(self, question, plan, probes, bus, result, decision, solver_chunks_by_node):
        depth_levels = self._topo_levels(plan)
        for level_nodes in depth_levels:
            tasks = []
            metas = []
            for node in level_nodes:
                sub_q = bus.interpolate(node.question)
                hop_idx = self._depth_of(node, plan)
                start_chunks = self._probe_for_node(node, plan, probes, bus)

                # Context-aware child retrieval
                if self.config.use_context_aware_retrieval and node.depends_on:
                    parent_ctx = []
                    for pid in node.depends_on:
                        pf = bus.get(pid)
                        if pf and pf.answer:
                            parent_ctx.append(f"{pf.sub_question} -> {pf.answer}")
                    if parent_ctx:
                        sub_q = f"{sub_q} (given: {'; '.join(parent_ctx)})"

                metas.append((node, sub_q, hop_idx))
                tasks.append(run_solver(
                    solver_lm=self.worker_lm, rewrite_lm=self.worker_lm,
                    retriever=self.retriever, sub_question=sub_q,
                    expected_answer_type=node.expected_answer_type,
                    starting_chunks=start_chunks, node_id=node.id, hop_idx=hop_idx,
                    max_retrievals=self._retrieval_budget_for_node(node, plan, probes),
                    experience=self.config.experience_library,
                    use_verifier=self.config.use_verifier,
                    verifier_lm=self.worker_lm if self.config.use_verifier else None,
                    parent_ids=node.depends_on,
                ))
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            for (node, sub_q, hop_idx), sr in zip(metas, results_list):
                if isinstance(sr, Exception):
                    f = Finding(sub_question=sub_q, answer="", evidence_ids=[],
                                confidence=0.0, status=FindingStatus.ERROR,
                                hop_idx=hop_idx, node_id=node.id)
                    bus.append(node.id, f)
                    result.n_solvers_invoked += 1
                    continue
                self._accumulate_solver_result(sr, node.id, bus, result, solver_chunks_by_node)

    def _accumulate_solver_result(self, sr, node_id, bus, result, solver_chunks_by_node):
        bus.append(node_id, sr.finding)
        if sr.capsule:
            bus.append_capsule(sr.capsule)
            result.per_node_retrievals.append({
                "node_id": node_id,
                "retrievals_used": sr.capsule.retrievals_used,
                "retrievals_budget": sr.capsule.retrievals_budget,
            })
            result.per_node_latency.append({
                "node_id": node_id,
                "latency_seconds": sr.capsule.latency_seconds,
            })
        if sr.chunks_used and solver_chunks_by_node is not None:
            solver_chunks_by_node[node_id] = sr.chunks_used
        result.solver_tokens += sr.extraction_tokens
        result.rewrite_tokens += sr.rewrite_tokens
        result.verifier_tokens += sr.verifier_tokens
        result.verifier_calls += sr.verifier_calls
        result.verifier_accepts += sr.verifier_accepts
        result.verifier_rejects += sr.verifier_rejects
        result.n_solvers_invoked += 1
        result.n_retrieval_calls += max(0, len(sr.queries_issued) - 1)

    async def _maybe_repair(self, plan, probes, bus, result):
        repaired = 0
        max_repairs = self.config.max_repairs
        for node in plan.subgoals:
            if repaired >= max_repairs:
                break
            f = bus.findings_by_node.get(node.id)
            if f is None:
                continue
            if f.status not in (FindingStatus.NO_EVIDENCE, FindingStatus.LOW_CONFIDENCE, FindingStatus.ERROR):
                continue
            sub_q = bus.interpolate(node.question)
            sr = await run_solver(
                solver_lm=self.worker_lm, rewrite_lm=self.worker_lm,
                retriever=self.retriever, sub_question=sub_q,
                expected_answer_type=node.expected_answer_type,
                starting_chunks=None, node_id=node.id,
                hop_idx=self._depth_of(node, plan),
                max_retrievals=self._retrieval_budget_for_node(node, plan, probes),
                experience=self.config.experience_library,
                use_verifier=self.config.use_verifier,
                verifier_lm=self.worker_lm if self.config.use_verifier else None,
            )
            if sr.finding.status == FindingStatus.OK or sr.finding.confidence > f.confidence:
                bus.findings_by_node[node.id] = sr.finding
                if sr.capsule:
                    bus.capsules_by_node[node.id] = sr.capsule
                repaired += 1
                result.repair_invoked = True
            result.solver_tokens += sr.extraction_tokens
            result.rewrite_tokens += sr.rewrite_tokens
            result.verifier_tokens += sr.verifier_tokens
            result.verifier_calls += sr.verifier_calls
            result.n_solvers_invoked += 1
            result.n_retrieval_calls += max(0, len(sr.queries_issued) - 1)

    def _find_final_node(self, plan):
        for n in plan.subgoals:
            if n.is_final:
                return n
        return plan.subgoals[-1] if plan.subgoals else None

    def _depth_of(self, node, plan):
        cache = {}
        by_id = {n.id: n for n in plan.subgoals}
        def d(nid):
            if nid in cache:
                return cache[nid]
            n = by_id.get(nid)
            if not n or not n.depends_on:
                cache[nid] = 0
                return 0
            v = 1 + max(d(p) for p in n.depends_on)
            cache[nid] = v
            return v
        return d(node.id)

    def _topo_levels(self, plan):
        if not plan.subgoals:
            return []
        levels = {}
        for n in plan.subgoals:
            levels.setdefault(self._depth_of(n, plan), []).append(n)
        return [levels[k] for k in sorted(levels.keys())]

    def _probe_for_node(self, node, plan, probes, bus):
        try:
            idx = next(i for i, n in enumerate(plan.subgoals) if n.id == node.id)
        except StopIteration:
            return None
        probe_idx = idx + 1
        if probe_idx >= len(probes):
            return None
        if any(d for d in node.depends_on):
            return None
        return probes[probe_idx].chunks

    def _retrieval_budget_for_node(self, node, plan, probes):
        if not self.config.adaptive_solver_budget:
            return self.config.max_retrievals_per_solver
        try:
            idx = next(i for i, n in enumerate(plan.subgoals) if n.id == node.id)
            probe_idx = idx + 1
            g = probes[probe_idx].groundedness if probe_idx < len(probes) else 0.0
        except Exception:
            g = 0.0
        max_r = max(1, self.config.max_retrievals_per_solver)
        if g >= 0.65:
            return max(1, min(self.config.min_retrievals_per_solver, max_r))
        if g >= 0.45:
            return max(1, min(self.config.medium_retrievals_per_solver, max_r))
        return max_r

    def _final_evidence_chunks(self, node, plan, probes, bus):
        if not node:
            return probes[0].chunks if probes else []
        chunks = []
        if probes:
            try:
                idx = next(i for i, n in enumerate(plan.subgoals) if n.id == node.id)
                probe_idx = idx + 1
                if probe_idx < len(probes):
                    chunks = probes[probe_idx].chunks
            except StopIteration:
                pass
        if not chunks and probes:
            chunks = probes[0].chunks
        return chunks
