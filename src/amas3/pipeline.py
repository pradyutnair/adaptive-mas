"""AMAS pipeline executor.

Deterministic Plan -> Probe -> TopologySelect -> Execute -> Synth flow.
No emergent topology decisions (all driven by retrieval signals + plan structure).

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
from .signals import compute_groundedness
from .types import (
    Finding,
    FindingStatus,
    Plan,
    ProbeResult,
    RetrievedChunk,
    SubgoalNode,
    Topology,
    TopologyDecision,
)
from .working_memory import FindingsBus

logger = logging.getLogger(__name__)


@dataclass
class AmasResult:
    question_id: str
    question: str
    answer: str
    answer_type: str = 'other'
    justification: str = ''
    support_ids: list[str] = field(default_factory=list)
    plan_subgoals: int = 0
    topology: str = ''
    topology_rationale: str = ''
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
    multi_plan_rewards: list = field(default_factory=list)
    multi_plan_subgoal_counts: list = field(default_factory=list)
    multi_plan_temperatures: list = field(default_factory=list)
    sas_collapse: bool = False
    sas_escalated: bool = False
    sas_attempt_tokens: int = 0
    sas_attempt_confidence: float = 0.0
    sas_attempt_grounded: bool = False
    sas_verifier_passed: bool = False
    sas_verifier_verdict: str = ''
    sas_verifier_tokens: int = 0


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
    synth_recursion_rounds: int = 1
    adaptive_solver_budget: bool = False
    min_retrievals_per_solver: int = 1
    medium_retrievals_per_solver: int = 2
    max_repairs: int = 2
    use_sas_solver: bool = False
    sas_probe_min_g: float = 0.0
    sas_max_followups: int = 2
    sas_min_confidence: float = 0.65
    sas_excerpt_chars: int = 320
    sas_max_chunks: int = 5
    sas_use_verifier: bool = False
    sas_verifier_min_confidence: float = 0.6
    synth_slim: bool = False
    synth_excerpt_chars: int = 220
    synth_max_excerpts: int = 6
    role_prompts: dict[str, str] = field(default_factory=dict)
    max_plan_subgoals: int = 4
    # Legacy escape hatch. The GRPO orchestrator no longer emits a strict SAS
    # route; its sas_first strategy always honors verifier-driven escalation.
    sas_strict_single_pass: bool = False
    # Soft runtime token budget B passed to the executor for logging and used
    # downstream in reward shaping (sigmoid centered on B). The executor does
    # NOT hard-truncate rollouts on B; over-budget penalties are applied via
    # the reward signal so the GRPO group still produces a learning signal.
    deployment_budget: int = 0


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
    ) -> None:
        self.planner_lm = planner_lm
        self.worker_lm = worker_lm
        self.synth_lm = synth_lm
        self.sas_lm = sas_lm or worker_lm
        self.retriever = retriever
        self.config = config or AmasPipelineConfig()

    def _role_experience(self, role: str, extra: str = "") -> str:
        parts = []
        if self.config.experience_library:
            parts.append(self.config.experience_library)
        role_prompt = (self.config.role_prompts or {}).get(role, "")
        if role_prompt:
            parts.append(f"Role-specific evolved prompt for {role}:\n{role_prompt}")
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)

    async def run(self, question: str, qid: str = '') -> AmasResult:
        t0 = time.time()
        bus = FindingsBus()
        result = AmasResult(question_id=qid, question=question, answer='')
        solver_chunks_by_node: dict[int, list] = {}

        # Step 0: probe original Q first (for bridge gating + reused as probes[0])
        original_chunks = await self.retriever.retrieve(question)
        result.n_retrieval_calls += 1
        g_original, comp_o = compute_groundedness(question, original_chunks)

        # Step 0.25: SAS solver (one cheap LLM call; optional followups or escalate)
        if self.config.use_sas_solver and g_original >= self.config.sas_probe_min_g:
            from .amas_orchestrator import run_amas_orchestrator
            sas = await run_amas_orchestrator(
                sas_lm=self.sas_lm if self.sas_lm is not None else self.worker_lm,
                retriever=self.retriever,
                question=question,
                probe_chunks=original_chunks,
                max_followups=self.config.sas_max_followups,
                min_answer_confidence=self.config.sas_min_confidence,
                chunk_excerpt_chars=self.config.sas_excerpt_chars,
                max_chunks_per_step=self.config.sas_max_chunks,
                experience=self._role_experience("sas_solver"),
            )
            result.sas_attempt_tokens = sas.tokens
            result.sas_attempt_confidence = sas.confidence
            result.sas_attempt_grounded = bool(sas.support_ids)
            result.sas_verifier_verdict = sas.action
            result.n_retrieval_calls += max(0, sas.retrieval_calls - 1)
            if sas.action == 'answer' and sas.answer:
                accept = True
                if self.config.sas_use_verifier:
                    from .amas_orchestrator import run_amas_verifier
                    vr = await run_amas_verifier(
                        verifier_lm=self.synth_lm,
                        question=question,
                        answer=sas.answer,
                        justification=sas.justification,
                        chunks=sas.chunks_used,
                        support_ids=sas.support_ids,
                        excerpt_chars=self.config.sas_excerpt_chars,
                        max_chunks=self.config.sas_max_chunks,
                    )
                    result.sas_verifier_tokens = vr.tokens
                    result.sas_verifier_verdict = f"{vr.decision}|{vr.failure_reason[:80]}"
                    result.sas_verifier_passed = vr.decision == 'accept'
                    accept = vr.decision == 'accept' and vr.confidence >= self.config.sas_verifier_min_confidence
                if accept:
                    result.sas_collapse = True
                    result.topology = 'sas_solver_answer'
                    result.topology_rationale = f"sas_solver answered (conf={sas.confidence:.2f}, retrievals={sas.retrieval_calls})"
                    result.answer = sas.answer
                    result.answer_type = sas.answer_type
                    result.justification = sas.justification[:300]
                    result.support_ids = sas.support_ids or [c.chunk_id for c in sas.chunks_used[:5]]
                    result.probe_groundedness = [round(g_original, 4)]
                    result.total_tokens = result.sas_attempt_tokens + result.sas_verifier_tokens
                    result.wallclock_seconds = round(time.time() - t0, 3)
                    result.n_solvers_invoked = 1
                    return result
            # Legacy strict SAS lane. The current GRPO topology layer does not
            # use this path; sas_first escalates naturally when SAS is weak.
            if self.config.sas_strict_single_pass:
                fallback_answer = sas.answer or ''
                fallback_support = list(sas.support_ids or [])
                fallback_just = sas.justification or ''
                fallback_type = getattr(sas, 'answer_type', '') or 'other'
                if not fallback_answer:
                    for h in reversed(sas.search_history or []):
                        cand = (h or {}).get('answer') or ''
                        if cand:
                            fallback_answer = cand
                            fallback_support = [str(x) for x in (h.get('support_ids') or [])]
                            fallback_just = f"sas_low_conf (conf={h.get('confidence', 0.0):.2f})"
                            break
                result.sas_collapse = True
                result.topology = 'sas_strict_singlepass'
                result.topology_rationale = (
                    f"sas-strict: action={sas.action} "
                    f"conf={sas.confidence:.2f} retrievals={sas.retrieval_calls}"
                )
                result.answer = fallback_answer
                result.answer_type = fallback_type
                result.justification = (fallback_just or '')[:300]
                result.support_ids = fallback_support or [c.chunk_id for c in sas.chunks_used[:5]]
                result.probe_groundedness = [round(g_original, 4)]
                result.total_tokens = result.sas_attempt_tokens + result.sas_verifier_tokens
                result.wallclock_seconds = round(time.time() - t0, 3)
                result.n_solvers_invoked = 1
                return result
            result.sas_escalated = True

        enhanced_experience = self._role_experience("planner")

        # Step 2: planning (single or multi-plan GRPO)
        from .types import ProbeResult
        original_probe = ProbeResult(
            label='original', query=question, chunks=original_chunks,
            top1_score=comp_o['top1_score'], score_gap_1to5=comp_o['score_gap_1to5'],
            ne_coverage=comp_o['ne_coverage'],
            wh_target_extractable=bool(comp_o['wh_target_extractable']),
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
            chosen_sub_probes = [p for p in mp.chosen_probes if p.label != 'original']
            probes = [original_probe] + chosen_sub_probes
            # Each candidate plan generated len(sub_qs)+1 retrievals (original + sub).
            # We pre-fetched the original once. Multi-plan does K probes-of-original (wasted)
            # plus K probes per sub-question per candidate.
            mp_retrievals = sum(len(p.chunks) > 0 for p in mp.chosen_probes)
            result.n_retrieval_calls += mp_retrievals
        else:
            plan = await asyncio.to_thread(
                run_planner,
                self.planner_lm,
                question,
                enhanced_experience,
                self.config.max_plan_subgoals,
            )
            result.planner_tokens = plan.planner_tokens
            result.plan_subgoals = len(plan.subgoals)

            sub_questions = [bus.interpolate(n.question) for n in plan.subgoals]
            sub_probes_full = await probe_all(
                retriever=self.retriever,
                original_question=question,
                sub_questions=sub_questions,
            )
            chosen_sub_probes = [p for p in sub_probes_full if p.label != 'original']
            probes = [original_probe] + chosen_sub_probes
            result.n_retrieval_calls += len(chosen_sub_probes) + 1  # sub-Q probes + the original re-probe inside probe_all

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

        # Build union of evidence: probe-original + all sub-probe chunks + all solver chunks_used
        union: list = []
        seen_ids = set()
        def _add(chunks):
            for c in chunks:
                cid = getattr(c, 'chunk_id', None)
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    union.append(c)
        _add(final_evidence)
        for node_id, chunks in solver_chunks_by_node.items():
            _add(chunks)
        for p in probes:
            _add(p.chunks)
        # Cap to 20 chunks max to keep synth context tight
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
                    experience=self._role_experience("synth"),
                    rounds=self.config.synth_recursion_rounds,
                )
            else:
                synth_evidence = union_capped
                if self.config.synth_slim:
                    # Drop full chunks; rely on findings (answer + justification + tiny support excerpts)
                    synth_evidence = []
                    # Re-tag chunks to short excerpts to keep token-cost minimal even if synth peeks
                    short = []
                    seen = set()
                    for f in bus.all():
                        for eid in (f.evidence_ids or [])[:2]:
                            if eid in seen:
                                continue
                            seen.add(eid)
                            for c in union_capped:
                                if c.chunk_id == eid:
                                    short.append(type(c)(chunk_id=c.chunk_id, text=c.text[:self.config.synth_excerpt_chars], score=getattr(c, 'score', 0.0)))
                                    break
                        if len(short) >= self.config.synth_max_excerpts:
                            break
                    synth_evidence = short[:self.config.synth_max_excerpts]
                synth_obj, synth_tokens = await asyncio.to_thread(
                    run_synthesizer,
                    synth_lm=self.synth_lm,
                    original_question=question,
                    bus=bus,
                    final_evidence=synth_evidence,
                    experience=self._role_experience("synth"),
                )
        except Exception as e:
            logger.warning('synth call failed (%s); falling back to best non-bridge finding', type(e).__name__)
            # Pick the highest-confidence finding from the LAST topological level as fallback
            final_node = self._find_final_node(plan)
            best_finding = bus.findings_by_node.get(final_node.id) if final_node else None
            if best_finding is None:
                # any finding with answer
                for f in reversed(bus.all()):
                    if f.answer:
                        best_finding = f
                        break
            if best_finding and best_finding.answer:
                synth_obj = {
                    'answer': best_finding.answer,
                    'answer_type': 'other',
                    'justification': f'synth_fallback (status={best_finding.status.value})',
                    'support_ids': best_finding.evidence_ids or [],
                }
            else:
                synth_obj = {'answer': '', 'answer_type': 'other', 'justification': 'synth_failed_no_findings', 'support_ids': []}
            synth_tokens = 0
        result.synth_tokens = synth_tokens
        result.answer = str(synth_obj.get('answer', '')).strip()
        result.answer_type = str(synth_obj.get('answer_type', 'other'))
        result.justification = str(synth_obj.get('justification', ''))[:300]
        sup = synth_obj.get('support_ids', [])
        if isinstance(sup, list):
            result.support_ids = [str(x) for x in sup if x]

        for f in bus.all():
            result.findings.append({
                'node_id': f.node_id,
                'sub_question': f.sub_question,
                'answer': f.answer,
                'confidence': f.confidence,
                'status': f.status.value,
                'evidence_ids': f.evidence_ids,
            })

        result.total_tokens = (
            result.planner_tokens
            + result.solver_tokens
            + result.synth_tokens
            + result.rewrite_tokens
            + result.sas_attempt_tokens
        )
        result.wallclock_seconds = round(time.time() - t0, 3)
        return result

    async def _execute_sas(
        self,
        question: str,
        plan: Plan,
        probes: list[ProbeResult],
        bus: FindingsBus,
        result: AmasResult,
        solver_chunks_by_node: dict | None = None,
    ) -> None:
        original = probes[0]
        node = self._find_final_node(plan)
        sr = await run_solver(
            solver_lm=self.worker_lm,
            rewrite_lm=self.worker_lm,
            retriever=self.retriever,
            sub_question=question,
            expected_answer_type=node.expected_answer_type if node else 'entity',
            starting_chunks=original.chunks,
            node_id=node.id if node else 1,
            hop_idx=0,
            max_retrievals=self._retrieval_budget_for_node(node, plan, probes) if node else self.config.max_retrievals_per_solver,
            experience=self._role_experience("solver"),
        )
        result.solver_tokens += sr.extraction_tokens
        result.rewrite_tokens += sr.rewrite_tokens
        result.n_solvers_invoked += 1
        result.n_retrieval_calls += max(0, len(sr.queries_issued) - 1)
        bus.append(node.id if node else 1, sr.finding)
        if sr.chunks_used and solver_chunks_by_node is not None:
            solver_chunks_by_node[node.id if node else 1] = sr.chunks_used

    async def _execute_dag(
        self,
        question: str,
        plan: Plan,
        probes: list[ProbeResult],
        bus: FindingsBus,
        result: AmasResult,
        decision: TopologyDecision,
        solver_chunks_by_node: dict | None = None,
    ) -> None:
        depth_levels = self._topo_levels(plan)
        for level_nodes in depth_levels:
            tasks = []
            metas = []
            for node in level_nodes:
                sub_q = bus.interpolate(node.question)
                hop_idx = self._depth_of(node, plan)
                start_chunks = self._probe_for_node(node, plan, probes, bus)
                metas.append((node, sub_q, hop_idx))
                tasks.append(run_solver(
                    solver_lm=self.worker_lm,
                    rewrite_lm=self.worker_lm,
                    retriever=self.retriever,
                    sub_question=sub_q,
                    expected_answer_type=node.expected_answer_type,
                    starting_chunks=start_chunks,
                    node_id=node.id,
                    hop_idx=hop_idx,
                    max_retrievals=self._retrieval_budget_for_node(node, plan, probes),
                    experience=self._role_experience("solver"),
                ))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (node, sub_q, hop_idx), sr in zip(metas, results):
                if isinstance(sr, Exception):
                    f = Finding(sub_question=sub_q, answer='', evidence_ids=[],
                                confidence=0.0, status=FindingStatus.ERROR,
                                hop_idx=hop_idx, node_id=node.id)
                    bus.append(node.id, f)
                    result.n_solvers_invoked += 1
                    continue
                bus.append(node.id, sr.finding)
                if sr.chunks_used and solver_chunks_by_node is not None:
                    solver_chunks_by_node[node.id] = sr.chunks_used
                result.solver_tokens += sr.extraction_tokens
                result.rewrite_tokens += sr.rewrite_tokens
                result.n_solvers_invoked += 1
                result.n_retrieval_calls += max(0, len(sr.queries_issued) - 1)

    async def _maybe_repair(
        self,
        plan: Plan,
        probes: list[ProbeResult],
        bus: FindingsBus,
        result: AmasResult,
    ) -> None:
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
                solver_lm=self.worker_lm,
                rewrite_lm=self.worker_lm,
                retriever=self.retriever,
                sub_question=sub_q,
                expected_answer_type=node.expected_answer_type,
                starting_chunks=None,
                node_id=node.id,
                hop_idx=self._depth_of(node, plan),
                max_retrievals=self._retrieval_budget_for_node(node, plan, probes),
                experience=self._role_experience("solver"),
            )
            if sr.finding.status == FindingStatus.OK or sr.finding.confidence > f.confidence:
                bus.findings_by_node[node.id] = sr.finding
                repaired += 1
                result.repair_invoked = True
            result.solver_tokens += sr.extraction_tokens
            result.rewrite_tokens += sr.rewrite_tokens
            result.n_solvers_invoked += 1
            result.n_retrieval_calls += max(0, len(sr.queries_issued) - 1)

    def _find_final_node(self, plan: Plan) -> SubgoalNode | None:
        for n in plan.subgoals:
            if n.is_final:
                return n
        return plan.subgoals[-1] if plan.subgoals else None

    def _depth_of(self, node: SubgoalNode, plan: Plan) -> int:
        cache: dict[int, int] = {}
        by_id = {n.id: n for n in plan.subgoals}
        def d(nid: int) -> int:
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

    def _topo_levels(self, plan: Plan) -> list[list[SubgoalNode]]:
        if not plan.subgoals:
            return []
        by_id = {n.id: n for n in plan.subgoals}
        levels: dict[int, list[SubgoalNode]] = {}
        for n in plan.subgoals:
            levels.setdefault(self._depth_of(n, plan), []).append(n)
        return [levels[k] for k in sorted(levels.keys())]

    def _probe_for_node(
        self,
        node: SubgoalNode,
        plan: Plan,
        probes: list[ProbeResult],
        bus: FindingsBus,
    ) -> list[RetrievedChunk] | None:
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

    def _retrieval_budget_for_node(
        self,
        node: SubgoalNode,
        plan: Plan,
        probes: list[ProbeResult],
    ) -> int:
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

    def _final_evidence_chunks(
        self,
        node: SubgoalNode | None,
        plan: Plan,
        probes: list[ProbeResult],
        bus: FindingsBus,
    ) -> list[RetrievedChunk]:
        if not node:
            return probes[0].chunks if probes else []
        f = bus.findings_by_node.get(node.id) if node else None
        chunks: list[RetrievedChunk] = []
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
