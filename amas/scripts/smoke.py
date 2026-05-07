"""P0 smoke test: 3 questions × 2 gates (off + conformal) end-to-end.

Validates: vLLM endpoint reachability, retriever, openai, full pipeline w/ ledger + belief,
gate decision path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from amas.agents import load_prompts
from amas.config import load_env
from amas.gates import make_gate
from amas.library import ExperienceLibrary
from amas.lm import OpenAIClient, VLLMClient
from amas.orchestrator import Orchestrator
from amas.pipeline import run_amas
from amas.retriever import RetrieverClient


SMOKE_QUESTIONS = [
    {"id": "smoke-1", "question": "What is the capital of France?", "answer": "Paris"},
    {"id": "smoke-2", "question": "Who wrote The Great Gatsby?",
     "answer": "F. Scott Fitzgerald"},
    {"id": "smoke-3",
     "question": "What is the birthplace of the director of the film The Godfather?",
     "answer": "New York"},
]


async def run_one_gate(gate_name: str, cfg: dict) -> None:
    print(f"\n========== GATE: {gate_name} ==========")
    vcfg = cfg["vllm"]
    ocfg = cfg["openai"]
    rcfg = cfg["retriever"]
    pcfg = cfg["pipeline"]

    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=vcfg["max_tokens"], temperature=vcfg["temperature"],
                      concurrency=4)
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=ocfg["max_tokens"],
                                 temperature=ocfg["temperature"], concurrency=4)
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"], concurrency=4)
    library = ExperienceLibrary()
    prompts = load_prompts("__seed__")

    orch = Orchestrator(vllm=vllm, openai_client=openai_client, retriever=retriever,
                        library=library, prompts=prompts,
                        retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8)
    gate = make_gate(gate_name, openai_client=openai_client, cfg=cfg.get("gate", {}))

    for q in SMOKE_QUESTIONS:
        try:
            res = await run_amas(
                question=q["question"], gold=q["answer"], qid=q["id"],
                gate=gate, orchestrator=orch, retriever=retriever,
                openai_client=openai_client,
                t_max=pcfg["t_max"], probe_group_size=pcfg["probe_group_size"],
                probe_topk=pcfg["probe_topk"],
                rollout_temperature=pcfg.get("rollout_temperature", 0.0),
            )
            print(f"[{q['id']}] {q['question']}")
            print(f"  pred='{res.final_answer}' gold='{q['answer']}'")
            print(f"  EM={res.em} F1={res.f1:.2f} Acc={res.acc} "
                  f"tokens={res.total_tokens} turns={res.n_turns} sas={res.sas_committed}")
            print(f"  gate trace: " + " | ".join(
                f"t{t.turn}:{t.gate_action}({t.gate_reason[:40]})" for t in res.turns))
        except Exception as e:
            print(f"[{q['id']}] FAILED: {e}")

    await retriever.aclose()
    await vllm.aclose()
    await openai_client.aclose()


async def main() -> None:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env()
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    for gate_name in ("off", "bayesian", "conformal"):
        await run_one_gate(gate_name, cfg)


if __name__ == "__main__":
    asyncio.run(main())
