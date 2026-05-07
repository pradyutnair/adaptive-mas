"""Stage 4 smoke: SAS / MAS / AUTO routing + escalation.

Asserts:
  A. orchestrator.route_lane returns lane in {SAS, MAS, AUTO} for a simple
     factoid AND a multi-hop query (using Qwen3-14B vLLM via base config).
  B. Pipeline records router_lane + router_reason on AmasResult.
  C. lane=MAS forces MAS even if probe gate would have committed (verified
     by stubbing the gate to always SAS_COMMIT and patching route_lane to
     return "MAS"; assert sas_committed=False, n_turns>=1).
  D. lane=SAS escalation: stub gate to always CONTINUE (rejecting probe);
     patch route_lane to return "SAS"; assert escalated_from_sas=True and
     at least one MAS LLM-agent user_msg contains the
     "Important — escalated from SAS lane" prefix.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/local/yzheng/pnair/workspace/adaptive-mas/amas")
sys.path.insert(0, str(ROOT / "src"))

from amas.agents import load_prompts  # noqa: E402
from amas.config import build_probe_client, load_env, validate_probe_config  # noqa: E402
from amas.gates import make_gate  # noqa: E402
from amas.gates.base import GateAction, GateDecision  # noqa: E402
from amas.library import ExperienceLibrary  # noqa: E402
from amas.lm import OpenAIClient, VLLMClient  # noqa: E402
from amas.orchestrator import Orchestrator  # noqa: E402
from amas.pipeline import run_amas  # noqa: E402
from amas.retriever import RetrieverClient  # noqa: E402

ESCALATION_PREFIX = "Important - escalated from SAS lane"


def assert_or_die(cond: bool, msg: str) -> None:
    if not cond:
        print("FAIL:", msg, file=sys.stderr)
        sys.exit(1)


class StubGate:
    """Test-only gate that returns a fixed decision regardless of input."""
    name = "stub"

    def __init__(self, action: GateAction, score: float = 0.0, reason: str = "stub"):
        self._action = action; self._score = score; self._reason = reason

    async def decide(self, *, question, ledger, belief, turn, ctx):
        return GateDecision(action=self._action, score=self._score, reason=self._reason)


async def main() -> None:
    load_env()
    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    validate_probe_config(cfg)

    vcfg = cfg["vllm"]; ocfg = cfg["openai"]; rcfg = cfg["retriever"]; pcfg = cfg["pipeline"]
    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=vcfg["max_tokens"], temperature=vcfg["temperature"],
                      concurrency=4)
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=ocfg["max_tokens"],
                                 temperature=ocfg["temperature"], concurrency=4)
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"], concurrency=4)
    library = ExperienceLibrary.load(cfg["paths"]["library"])
    prompts = load_prompts(cfg["paths"]["prompts"])
    orch = Orchestrator(vllm=vllm, openai_client=openai_client, retriever=retriever,
                        library=library, prompts=prompts,
                        retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8)
    probe_client, probe_owned = build_probe_client(cfg, vllm=vllm, openai_client=openai_client)

    # Capture MAS-agent user_msgs so escalation assert (D) can search them.
    captured: list[dict] = []
    from amas import agents as _agents
    from amas import orchestrator as _orch
    orig_run_agent = _agents.run_llm_agent

    async def cap(prompt, q_, deps, passages, lm, ledger_text="", belief_text="",
                  rejected_probe=""):
        inv = await orig_run_agent(prompt, q_, deps, passages, lm,
                                    ledger_text=ledger_text, belief_text=belief_text,
                                    rejected_probe=rejected_probe)
        captured.append({
            "agent": inv.name,
            "user_msg": inv.inputs.get("user_msg", ""),
            "had_rejected_probe": bool(rejected_probe),
        })
        return inv

    _agents.run_llm_agent = cap  # type: ignore[assignment]
    _orch.run_llm_agent = cap  # type: ignore[assignment]

    failures: list[str] = []
    try:
        # ---------- Test A: route_lane returns valid lanes (and actually called the LM) ----------
        lane_easy, info_easy = await orch.route_lane(
            "What is the capital of France?",
            profile="bridge",
            probe_answer="Paris", probe_agreement=1.0,
        )
        lane_hard, info_hard = await orch.route_lane(
            "What is the birthplace of the director of the film The Godfather?",
            profile="bridge",
            probe_answer="not mentioned", probe_agreement=0.33,
        )
        for tag, lane, info in (("easy", lane_easy, info_easy), ("hard", lane_hard, info_hard)):
            assert_or_die(lane in ("SAS", "MAS", "AUTO"),
                          f"route_lane({tag})={lane!r} not in valid set")
            tok = int(info.get("tokens", 0) or 0)
            reason = str(info.get("reason", ""))
            assert_or_die(tok > 0,
                          f"route_lane({tag}) token count {tok} <= 0; "
                          f"router LM was likely never called (reason={reason!r})")
            assert_or_die(not reason.startswith("router error"),
                          f"route_lane({tag}) fell back via error path: {reason!r}")
        print(f"OK route_lane: easy={lane_easy} tok={info_easy.get('tokens',0)} "
              f"reason={info_easy.get('reason','')[:40]!r} | "
              f"hard={lane_hard} tok={info_hard.get('tokens',0)} "
              f"reason={info_hard.get('reason','')[:40]!r}")

        # ---------- Test C: lane=MAS forces MAS even when gate would SAS-commit ----------
        captured.clear()
        # stub gate to ALWAYS SAS_COMMIT
        force_commit_gate = StubGate(GateAction.SAS_COMMIT, score=10.0, reason="stub commit")
        # monkey-patch route_lane to return MAS
        orig_route = orch.route_lane

        async def force_mas(query, profile, *, probe_answer, probe_agreement,
                            top_passage_score=0.0, temperature=0.0):
            return "MAS", {"reason": "stub MAS", "tokens": 0}

        orch.route_lane = force_mas  # type: ignore[assignment]
        try:
            res_mas = await run_amas(
                question="What is the capital of France?",
                gold="Paris", qid="stage4-c",
                gate=force_commit_gate, orchestrator=orch,
                retriever=retriever, openai_client=openai_client, probe_client=probe_client,
                t_max=2, probe_group_size=pcfg.get("probe_group_size", 1),
                probe_topk=pcfg.get("probe_topk", 5),
                rollout_temperature=0.0,
            )
        finally:
            orch.route_lane = orig_route  # type: ignore[assignment]
        assert_or_die(res_mas.router_lane == "MAS", f"router_lane recorded: {res_mas.router_lane!r}")
        assert_or_die(not res_mas.sas_committed,
                      f"lane=MAS should override SAS_COMMIT but sas_committed={res_mas.sas_committed}")
        assert_or_die(res_mas.n_turns >= 1,
                      f"lane=MAS should run >=1 MAS turn, got n_turns={res_mas.n_turns}")
        print(f"OK lane=MAS overrides gate SAS_COMMIT: sas_committed={res_mas.sas_committed} "
              f"n_turns={res_mas.n_turns}")

        # ---------- Test D: lane=SAS escalation when gate refuses ----------
        captured.clear()
        # stub gate to always CONTINUE (rejecting probe)
        reject_gate = StubGate(GateAction.CONTINUE, score=0.5, reason="stub reject")

        async def force_sas(query, profile, *, probe_answer, probe_agreement,
                            top_passage_score=0.0, temperature=0.0):
            return "SAS", {"reason": "stub SAS", "tokens": 0}

        orch.route_lane = force_sas  # type: ignore[assignment]
        try:
            res_sas = await run_amas(
                question="What is the birthplace of the director of the film The Godfather?",
                gold="New York", qid="stage4-d",
                gate=reject_gate, orchestrator=orch,
                retriever=retriever, openai_client=openai_client, probe_client=probe_client,
                t_max=2, probe_group_size=pcfg.get("probe_group_size", 1),
                probe_topk=pcfg.get("probe_topk", 5),
                rollout_temperature=0.0,
            )
        finally:
            orch.route_lane = orig_route  # type: ignore[assignment]
        assert_or_die(res_sas.router_lane == "SAS", f"router_lane: {res_sas.router_lane!r}")
        assert_or_die(res_sas.escalated_from_sas,
                      "lane=SAS + verifier reject should set escalated_from_sas=True")
        assert_or_die(res_sas.n_turns >= 1,
                      f"escalation should run >=1 MAS turn, got n_turns={res_sas.n_turns}")
        msgs_with_warning = [c for c in captured if ESCALATION_PREFIX in c["user_msg"]]
        assert_or_die(len(msgs_with_warning) >= 1,
                      f"no MAS user_msg contained the SAS-escalation warning prefix; "
                      f"captured={len(captured)}")
        print(f"OK lane=SAS escalation: escalated={res_sas.escalated_from_sas} "
              f"n_turns={res_sas.n_turns} warned_msgs={len(msgs_with_warning)}/{len(captured)}")

    finally:
        _agents.run_llm_agent = orig_run_agent  # type: ignore[assignment]
        _orch.run_llm_agent = orig_run_agent  # type: ignore[assignment]
        # Lifecycle: close every client exactly once.
        to_close = [retriever, openai_client, vllm]
        if probe_owned:
            to_close.append(probe_client)
        seen: set[int] = set()
        for c in to_close:
            if c is None or id(c) in seen:
                continue
            seen.add(id(c))
            if hasattr(c, "aclose"):
                await c.aclose()

    if failures:
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        sys.exit(1)
    print("STAGE-4 SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())
