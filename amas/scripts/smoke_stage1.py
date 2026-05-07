"""Stage 1 smoke: verify ledger + belief wiring into MAS LLM agent prompts.

Asserts (per codex review):
  A. Turn 0 (probe path) does not place ledger/belief headers in any LLM-agent
     user_msg. Probe runs OpenAIClient.chat directly (not through run_llm_agent),
     so this is trivially satisfied — we still print captures to confirm.
  B. At least one MAS turn at t>=1 contains the 'Cross-turn evidence ledger:'
     header in some run_llm_agent user_msg, because the probe ingests into
     ledger before any MAS turn runs.
  C. The check inspects AgentInvocation.inputs['user_msg'] captured by
     monkey-patching agents.run_llm_agent — not stdout.

Forces gate=off so MAS executes both turns, t_max=2.
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
from amas.config import load_env  # noqa: E402
from amas.gates import make_gate  # noqa: E402
from amas.library import ExperienceLibrary  # noqa: E402
from amas.lm import OpenAIClient, VLLMClient  # noqa: E402
from amas.orchestrator import Orchestrator  # noqa: E402
from amas.pipeline import run_amas  # noqa: E402
from amas.retriever import RetrieverClient  # noqa: E402

LEDGER_HDR = "Cross-turn evidence ledger:"
BELIEF_HDR = "Current top candidates:"


async def main() -> None:
    load_env()
    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text())

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
    gate = make_gate("off", openai_client=openai_client, cfg=cfg.get("gate", {}))

    smoke_path = Path("/local/yzheng/pnair/workspace/reproduction/hera/data/train_3_smoke.jsonl")
    rows = [json.loads(l) for l in smoke_path.read_text().splitlines() if l.strip()]

    # Monkey-patch run_llm_agent to capture every (agent_name, user_msg).
    # IMPORTANT: orchestrator.py imports run_llm_agent by name at module load, so we
    # must patch BOTH the agents module binding AND the orchestrator module binding.
    captured: list[dict] = []
    from amas import agents as _agents
    from amas import orchestrator as _orch
    orig = _agents.run_llm_agent

    async def cap(prompt, q_, deps, passages, lm, ledger_text="", belief_text=""):
        inv = await orig(prompt, q_, deps, passages, lm,
                         ledger_text=ledger_text, belief_text=belief_text)
        captured.append({
            "agent": inv.name,
            "user_msg": inv.inputs.get("user_msg", ""),
            "had_ledger_kwarg": bool(ledger_text and ledger_text.strip()
                                     and ledger_text.strip() != "(ledger empty)"),
            "had_belief_kwarg": bool(belief_text and belief_text.strip()
                                     and belief_text.strip() != "(no candidates)"),
        })
        return inv

    _agents.run_llm_agent = cap  # type: ignore[assignment]
    _orch.run_llm_agent = cap  # type: ignore[assignment]

    n_with_header = 0
    n_with_kwarg = 0
    n_total = 0
    seen_any_ledger = False
    seen_any_belief = False
    failures: list[str] = []
    try:
        for r in rows[:3]:
            captured.clear()
            res = await run_amas(
                question=r["question"], gold=r.get("answer", ""),
                qid=r.get("id", ""), gate=gate,
                orchestrator=orch, retriever=retriever, openai_client=openai_client,
                t_max=2, probe_group_size=pcfg.get("probe_group_size", 1),
                probe_topk=pcfg.get("probe_topk", 5),
                rollout_temperature=0.0,
            )
            n_msgs_this_q = len(captured)
            with_hdr_this_q = sum(1 for c in captured if LEDGER_HDR in c["user_msg"])
            with_kwarg_this_q = sum(1 for c in captured if c["had_ledger_kwarg"])
            n_total += n_msgs_this_q
            n_with_header += with_hdr_this_q
            n_with_kwarg += with_kwarg_this_q
            if any(LEDGER_HDR in c["user_msg"] for c in captured):
                seen_any_ledger = True
            if any(BELIEF_HDR in c["user_msg"] for c in captured):
                seen_any_belief = True
            print(f"qid={r.get('id')} sas={res.sas_committed} "
                  f"n_turns={res.n_turns} captures={n_msgs_this_q} "
                  f"with_ledger_kwarg={with_kwarg_this_q} with_ledger_hdr={with_hdr_this_q}")
            if captured:
                snippet = captured[0]["user_msg"][:240].replace("\n", " | ")
                print(f"  first[{captured[0]['agent']}]: {snippet!r}")

    finally:
        _agents.run_llm_agent = orig  # type: ignore[assignment]
        _orch.run_llm_agent = orig  # type: ignore[assignment]

    print()
    print(f"TOTAL captures: {n_total}  with_ledger_kwarg={n_with_kwarg}  "
          f"with_ledger_hdr={n_with_header}  any_ledger={seen_any_ledger}  "
          f"any_belief={seen_any_belief}")

    # Asserts
    if n_total == 0:
        failures.append("ASSERT: no MAS LLM-agent invocations captured (pipeline may have errored)")
    if not seen_any_ledger:
        failures.append("ASSERT B FAILED: no MAS user_msg contained 'Cross-turn evidence ledger:'")
    if n_with_kwarg > 0 and n_with_header == 0:
        failures.append("ASSERT INTERNAL: ledger kwarg passed but header missing in user_msg "
                        "— prepend logic broken")

    if failures:
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        sys.exit(1)
    print("STAGE-1 SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())
