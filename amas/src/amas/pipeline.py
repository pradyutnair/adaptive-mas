"""AMAS pipeline: probe → gate → MAS turns → gate → STOP / MUTATE / CONTINUE.

Multi-turn extension of HERA orchestration. Trajectory now spans turns 1..t_stop with a
shared Evidence Ledger + Belief State.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .agents import AGENT_NAMES, AgentInvocation, AgentPrompt
from .gates import Gate, GateAction
from .ledger import BeliefState, Ledger, parse_stance_from_agent
from .library import ExperienceLibrary, profile_question
from .lm import OpenAIClient, VLLMClient
from .metric import accuracy, contain, exact_match, f1_score
from .orchestrator import Orchestrator, Trajectory, extract_final_answer, normalize_answer_span
from .probe import run_probe
from .retriever import RetrieverClient

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    turn: int
    topology: dict[str, Any]
    invocations: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    f1: float = 0.0
    em: float = 0.0
    contain: float = 0.0
    acc: float = 0.0
    tokens: int = 0
    elapsed_s: float = 0.0
    gate_action: str = ""
    gate_score: float = 0.0
    gate_reason: str = ""
    gate_info: dict[str, Any] | None = None
    ledger_size: int = 0
    belief_entropy: float = 0.0
    belief_top: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AmasResult:
    qid: str
    question: str
    gold: Any
    profile: str
    gate: str
    final_answer: str
    em: float = 0.0
    f1: float = 0.0
    contain: float = 0.0
    acc: float = 0.0
    total_tokens: int = 0
    probe_tokens: int = 0
    gate_tokens: int = 0
    mas_tokens: int = 0
    router_tokens: int = 0
    n_turns: int = 0
    sas_committed: bool = False
    # Stage 4: SAS/MAS lane selected by the router after the probe runs.
    router_lane: str = "AUTO"
    router_reason: str = ""
    escalated_from_sas: bool = False
    elapsed_s: float = 0.0
    turns: list[TurnRecord] = field(default_factory=list)
    used_insight_ids: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [t for t in d["turns"]]
        return d


def _ingest_invocations_into_ledger(invs: list[AgentInvocation], *, turn: int,
                                    ledger: Ledger, belief: BeliefState,
                                    last_passage_ids: list[str]) -> tuple[str, list[str]]:
    """Append per-agent claims to ledger; update belief with answers.

    Returns (current_top_answer, new_passage_ids_added_this_turn).
    """
    new_pids: list[str] = []
    last_answer = ""
    for inv in invs:
        out = inv.output or {}
        if not isinstance(out, dict):
            continue
        if inv.name == "Retriever":
            # Retriever invocation logs how many passages — actual passages are added
            # to ledger by Orchestrator separately via ctx, so skip here.
            continue
        # AnswerGenerator / ConcludeAgent / ReflectAgent: claim = answer span
        ans = str(out.get("answer", "") or "").strip()
        if ans and inv.name in ("AnswerGenerator", "ConcludeAgent", "ReflectAgent"):
            stance = parse_stance_from_agent(inv.name, out)
            cid = ledger.add(turn=turn, source_agent=inv.name, claim=ans,
                              passage_ids=last_passage_ids[:3],
                              stance=stance, confidence=0.6)
            # Belief update: support stance lifts; refute stance lowers.
            if stance == "support":
                belief.update_from_answer(ans, support=0.6, evidence_ids=[cid])
            elif stance == "refute":
                belief.update_from_answer(ans, refute=0.6, evidence_ids=[cid])
            last_answer = ans
            continue
        # ContextValidator: writes refute when sufficient=False with missing.
        if inv.name == "ContextValidator":
            sufficient = bool(out.get("sufficient", False))
            missing = str(out.get("missing", "") or "")[:200]
            if not sufficient and missing:
                ledger.add(turn=turn, source_agent=inv.name,
                            claim=f"Context insufficient: {missing}",
                            stance="refute", confidence=0.5)
            continue
        # EvidenceSelector / QueryDecomposer / QueryRewriter: rationale or sub-questions
        if inv.name == "EvidenceSelector":
            rat = str(out.get("rationale", "") or "")[:240]
            if rat:
                ledger.add(turn=turn, source_agent=inv.name, claim=rat,
                            stance="support", confidence=0.4)
            continue
        if inv.name == "QueryDecomposer":
            sub = out.get("sub_questions") or []
            if isinstance(sub, list) and sub:
                ledger.add(turn=turn, source_agent=inv.name,
                            claim=f"Decomposition: {sub}",
                            stance="neutral", confidence=0.3)
            continue
    return last_answer, new_pids


async def run_amas(question: str, gold: Any, *,
                   qid: str = "",
                   profile: str | None = None,
                   gate: Gate,
                   orchestrator: Orchestrator,
                   retriever: RetrieverClient,
                   openai_client: OpenAIClient,
                   probe_client: Any | None = None,
                   t_max: int = 3,
                   probe_group_size: int = 3,
                   probe_topk: int = 5,
                   rollout_temperature: float = 0.0,
                   ) -> AmasResult:
    """Run the AMAS pipeline.

    `probe_client` is the chat client used for the turn-0 probe. If unset, the
    probe falls back to `openai_client` (legacy behaviour). For the cross-family
    Qwen3-14B probe, callers should pass a VLLMClient as `probe_client`.
    """
    t0 = time.time()
    ledger = Ledger()
    belief = BeliefState(top_k=5)
    qprofile = profile or profile_question(question, qid=qid)
    probe_lm = probe_client if probe_client is not None else openai_client

    res = AmasResult(qid=qid, question=question, gold=gold, profile=qprofile,
                     gate=getattr(gate, "name", str(gate)),
                     final_answer="")

    # ---- Turn 0: probe ----
    try:
        probe = await run_probe(question, retriever=retriever, lm_client=probe_lm,
                                ledger=ledger, belief=belief,
                                topk=probe_topk, group_size=probe_group_size,
                                temperature=0.7, turn=0)
        res.probe_tokens = probe.total_tokens
    except Exception as e:
        res.error = f"probe failed: {str(e)[:200]}"
        probe = None

    # ---- Stage 4: router lane decision (SAS / MAS / AUTO) ----
    router_lane = "AUTO"
    if probe and probe.consensus_answer:
        try:
            agreement = float(probe.consensus_count) / float(max(1, len(probe.samples)))
            router_lane, rinfo = await orchestrator.route_lane(
                question, qprofile,
                probe_answer=probe.consensus_answer,
                probe_agreement=agreement,
            )
            res.router_lane = router_lane
            res.router_reason = str(rinfo.get("reason", ""))
            res.router_tokens = int(rinfo.get("tokens", 0))
        except Exception as e:
            logger.warning("router_lane failed: %s", e)
            router_lane = "AUTO"
            res.router_lane = "AUTO"
            res.router_reason = f"router error: {str(e)[:80]}"

    # ---- Gate at turn 0 ----
    ctx: dict[str, Any] = {"gold": gold,
                            "passages": [],
                            "candidate": probe.consensus_answer if probe else "",
                            "router_lane": router_lane,
                            }
    try:
        gate_dec = await gate.decide(question=question, ledger=ledger, belief=belief,
                                     turn=0, ctx=ctx)
    except Exception as e:
        logger.warning("Gate failed at t=0: %s", e)
        gate_dec = None

    rec0 = TurnRecord(
        turn=0,
        topology={"selected_agents": ["Probe"], "execution_order": [{"step": 1, "agent": "Probe"}]},
        invocations=[{"name": "Probe", "answer": (probe.consensus_answer if probe else "")}],
        answer=(probe.consensus_answer if probe else ""),
        tokens=res.probe_tokens,
        gate_action=(gate_dec.action.value if gate_dec else "CONTINUE"),
        gate_score=(gate_dec.score if gate_dec else 0.0),
        gate_reason=(gate_dec.reason if gate_dec else ""),
        gate_info=(gate_dec.info if gate_dec else None),
        ledger_size=len(ledger.entries),
        belief_entropy=belief.entropy(),
        belief_top=(belief.top().to_dict() if belief.top() else None),
    )
    res.turns.append(rec0)

    # Stage 4 lane policy: lane="MAS" forces multi-agent path even if the gate
    # would have SAS-committed (router judged this is multi-hop). lane="SAS" or
    # lane="AUTO" preserves the gate's SAS_COMMIT path. Escalation: lane="SAS"
    # with a non-commit gate decision goes to MAS turn 1 with __rejected_probe__
    # in agent context so MAS knows what the probe said and why it was rejected.
    rejected_probe_msg = ""
    # Escalation triggers only when the gate is an actual verifier and produced a
    # non-commit decision. OffGate just disables the gate path, so a CONTINUE from
    # it is not a real rejection and should not flag the probe as wrong.
    gate_name = getattr(gate, "name", "")
    if (
        router_lane == "SAS"
        and gate_dec
        and gate_dec.action != GateAction.SAS_COMMIT
        and gate_name != "off"
    ):
        res.escalated_from_sas = True
        rejected_probe_msg = (
            f"Probe answer was {probe.consensus_answer!r} but the verifier rejected "
            f"it (action={gate_dec.action.value}, score={gate_dec.score:.2f}). "
            f"Reason: {gate_dec.reason or '(none)'}. "
            "Do not reuse this answer unless independently supported by fresh evidence "
            "you find in the retrieval passages cited below."
        )

    if (
        gate_dec and gate_dec.action == GateAction.SAS_COMMIT
        and probe and probe.consensus_answer
        and router_lane != "MAS"
    ):
        res.sas_committed = True
        res.final_answer = normalize_answer_span(probe.consensus_answer, question=question)
        res.n_turns = 0
        _finalize(res, gold, t0)
        return res

    # ---- MAS turns 1..t_max ----
    mutation_hint = ""
    # Stage 4: lane filter for library retrieval inside sample_topology.
    # router_lane="MAS" or SAS-escalation -> request MAS-lane (and "any") insights.
    # router_lane="AUTO"/"SAS"-no-escalation -> legacy lane-agnostic retrieval.
    if router_lane == "MAS" or res.escalated_from_sas:
        lane_for_lib = "MAS"
    else:
        lane_for_lib = None
    for turn_idx in range(1, t_max + 1):
        try:
            topo, ids, sample_res = await orchestrator.sample_topology(
                question, qprofile, temperature=rollout_temperature,
                mutation_hint=mutation_hint, lane=lane_for_lib)
            # Stage 1: pass ledger + belief summaries into agent contexts. MAS LLM
            # agents read whatever evidence the pipeline has accumulated up to this
            # turn (probe ingestion + earlier MAS turns). Topology sampling does
            # NOT see ledger/belief at this stage — that is a deliberate scope cut.
            agent_ctx = {
                "__ledger__": ledger.summarize_for_agent(n=8),
                "__belief__": belief.summarize(k=5),
            }
            if rejected_probe_msg:
                agent_ctx["__rejected_probe__"] = rejected_probe_msg
            traj: Trajectory = await orchestrator.execute(
                question, gold, topo, ids,
                orch_tokens=sample_res.prompt_tokens + sample_res.completion_tokens,
                ctx=agent_ctx,
            )
        except Exception as e:
            logger.warning("MAS turn %d failed: %s", turn_idx, e)
            res.error = f"mas turn {turn_idx} failed: {str(e)[:200]}"
            break

        res.mas_tokens += traj.total_tokens
        res.used_insight_ids.extend(ids)

        # Pull passages from invocations (Retriever's). They are in the orchestrator's local
        # `passages` var which we don't have direct access to here; re-extract by inspecting
        # Retriever invocation output for the queries used (we can't reconstruct passage texts
        # without re-running, so we just stamp count).
        retr_inv = next((inv for inv in traj.invocations if inv.name == "Retriever"), None)
        last_pids: list[str] = []  # passage ids from this turn unknown at pipeline level

        # Ingest invocations into ledger + belief
        last_ans, _ = _ingest_invocations_into_ledger(
            traj.invocations, turn=turn_idx, ledger=ledger, belief=belief,
            last_passage_ids=last_pids,
        )
        # The trajectory's final answer is the most authoritative for this turn.
        if traj.answer:
            belief.update_from_answer(traj.answer, support=0.7,
                                      evidence_ids=[e.id for e in ledger.entries[-2:]])

        # Update Bayesian gate's cost history if applicable
        if hasattr(gate, "update_history"):
            try:
                gate.update_history(traj.total_tokens)
            except Exception:
                pass

        # Gate at end of turn
        ctx["candidate"] = traj.answer
        try:
            gate_dec = await gate.decide(question=question, ledger=ledger, belief=belief,
                                         turn=turn_idx, ctx=ctx)
        except Exception as e:
            logger.warning("Gate failed at t=%d: %s", turn_idx, e)
            gate_dec = None

        rec = TurnRecord(
            turn=turn_idx,
            topology=topo,
            invocations=[{"name": inv.name, "tokens": inv.prompt_tokens + inv.completion_tokens,
                           "answer": (inv.output.get("answer") if isinstance(inv.output, dict) else None)}
                          for inv in traj.invocations],
            answer=traj.answer,
            f1=traj.f1, em=traj.em, contain=traj.contain, acc=traj.acc,
            tokens=traj.total_tokens,
            elapsed_s=traj.elapsed_s,
            gate_action=(gate_dec.action.value if gate_dec else "CONTINUE"),
            gate_score=(gate_dec.score if gate_dec else 0.0),
            gate_reason=(gate_dec.reason if gate_dec else ""),
            gate_info=(gate_dec.info if gate_dec else None),
            ledger_size=len(ledger.entries),
            belief_entropy=belief.entropy(),
            belief_top=(belief.top().to_dict() if belief.top() else None),
        )
        res.turns.append(rec)

        if gate_dec and gate_dec.action == GateAction.STOP:
            # belief.top() is the ensemble of probe + MAS by support scoring; keep it.
            top = belief.top()
            res.final_answer = normalize_answer_span(
                (top.answer if top else traj.answer) or traj.answer,
                question=question,
            )
            res.n_turns = turn_idx
            _finalize(res, gold, t0)
            return res

        if gate_dec and gate_dec.action == GateAction.MUTATE:
            # Force topology mutation hint for next turn (orchestrator will resample at temp 0.95)
            mutation_hint = (
                f"  - prior topology: {[s['agent'] for s in topo['execution_order']]}\n"
                f"  - prior answer: {traj.answer!r} (gate refused)\n"
                f"  - try a structurally different topology"
            )
            # Use trajectory.failed_agent if known
            continue

        # CONTINUE: drop mutation hint
        mutation_hint = ""

    # ---- T_max exhausted: pick top belief candidate (ensemble) ----
    top = belief.top()
    if top:
        res.final_answer = normalize_answer_span(top.answer, question=question)
    else:
        res.final_answer = res.turns[-1].answer if res.turns else ""
    res.n_turns = len([t for t in res.turns if t.turn > 0])
    _finalize(res, gold, t0)
    return res


def _finalize(res: AmasResult, gold: Any, t0: float) -> None:
    res.total_tokens = res.probe_tokens + res.mas_tokens + res.gate_tokens
    res.em = exact_match(res.final_answer, gold)
    res.f1 = f1_score(res.final_answer, gold)
    res.contain = contain(res.final_answer, gold)
    res.acc = accuracy(res.final_answer, gold)
    res.elapsed_s = time.time() - t0
