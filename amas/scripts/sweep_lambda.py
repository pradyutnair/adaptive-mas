"""Sweep Bayesian gate lambda on val 200q. Find Pareto knee.

Lambda controls H(belief) < lambda * E[next_turn_cost] threshold.
- Small lambda → SAS-commit when entropy very low → conservative (more MAS turns).
- Large lambda → SAS-commit easily → more probe-only.

Output: per-lambda EM/Acc/tokens table; pick smallest lambda that maintains EM ≥ AMAS-off.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from amas.agents import load_prompts
from amas.config import load_env
from amas.gates.bayesian import BayesianGate
from amas.library import ExperienceLibrary
from amas.lm import OpenAIClient, VLLMClient
from amas.orchestrator import Orchestrator
from amas.pipeline import run_amas
from amas.retriever import RetrieverClient


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING)
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Init clients once
    vcfg = cfg["vllm"]; ocfg = cfg["openai"]; rcfg = cfg["retriever"]; pcfg = cfg["pipeline"]
    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=vcfg["max_tokens"], temperature=vcfg["temperature"],
                      concurrency=vcfg["concurrency"])
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=ocfg["max_tokens"],
                                 temperature=ocfg["temperature"], concurrency=ocfg["concurrency"])
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"], concurrency=rcfg["concurrency"])

    paths = cfg["paths"]
    library = ExperienceLibrary.load(paths["library"]) if Path(paths["library"]).exists() else ExperienceLibrary()
    prompts = load_prompts(paths["prompts"]) if Path(paths["prompts"]).exists() else load_prompts("__seed__")
    orchestrator = Orchestrator(vllm=vllm, openai_client=openai_client, retriever=retriever,
                                library=library, prompts=prompts,
                                retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8)

    # Load 200q val matched seed=42 from same source
    import random
    raw = json.loads(Path(args.questions).read_text())
    rng = random.Random(args.seed)
    if args.n < len(raw):
        questions = rng.sample(raw, args.n)
    else:
        questions = raw

    sweep = [float(x) for x in args.lambdas.split(",")]
    summary_rows = []

    for lam in sweep:
        gate = BayesianGate(lambda_=lam, fallback_cost=5000.0)
        out_file = out_dir / f"lambda_{lam:.5f}.jsonl"
        fh = open(out_file, "w")

        sem = asyncio.Semaphore(args.concurrency)
        async def one(q):
            async with sem:
                return await run_amas(
                    question=q.get("question",""), gold=q.get("answer",""),
                    qid=str(q.get("id","")),
                    gate=gate, orchestrator=orchestrator, retriever=retriever,
                    openai_client=openai_client,
                    t_max=pcfg["t_max"], probe_group_size=pcfg["probe_group_size"],
                    probe_topk=pcfg["probe_topk"],
                    rollout_temperature=pcfg.get("rollout_temperature", 0.0),
                )
        results = await asyncio.gather(*[one(q) for q in questions])
        for r in results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        fh.close()
        n = len(results)
        em = sum(r.em for r in results) / n
        f1 = sum(r.f1 for r in results) / n
        acc = sum(r.acc for r in results) / n
        tok = sum(r.total_tokens for r in results) / n
        sas = sum(r.sas_committed for r in results) / n
        row = {"lambda": lam, "n": n, "em": em, "f1": f1, "acc": acc,
               "avg_tokens": tok, "sas_rate": sas}
        summary_rows.append(row)
        print(f"lambda={lam:.5f}  em={em:.3f} f1={f1:.3f} acc={acc:.3f} tok={tok:.0f} sas={sas:.2%}")

    (out_dir / "lambda_sweep.json").write_text(json.dumps(summary_rows, indent=2))
    # wandb summary log
    try:
        import wandb
        run = wandb.init(project=cfg["wandb"]["project_eval"], entity=cfg["wandb"].get("entity"), name="lambda_sweep", config={"lambdas": sweep, "n": args.n}, reinit=True)
        for r in summary_rows:
            wandb.log({"lambda/lambda": r["lambda"], "lambda/em": r["em"], "lambda/f1": r["f1"], "lambda/acc": r["acc"], "lambda/avg_tokens": r["avg_tokens"], "lambda/sas_rate": r["sas_rate"]})
        wandb.finish()
    except Exception as e:
        logging.warning("wandb log failed: %s", e)

    await retriever.aclose(); await vllm.aclose(); await openai_client.aclose()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--lambdas", default="0.00005,0.0001,0.0005,0.001,0.002,0.005",
                    help="comma-separated lambda values")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
