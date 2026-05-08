"""Sweep conformal gate alpha on val 200q.

For each alpha ∈ {0.01, 0.05, 0.1, 0.2}:
  - calibrate Route A on routeA_calib (200q held-out from train pool)
  - evaluate on val 200q (different)
  - measure SAS-error rate (gate accepted but answer wrong) — should ≤ alpha
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
from amas.gates.conformal import ConformalGate, conformal_quantile, write_calibration
from amas.library import ExperienceLibrary
from amas.lm import OpenAIClient, VLLMClient
from amas.orchestrator import Orchestrator
from amas.pipeline import run_amas
from amas.retriever import RetrieverClient


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING)
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

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

    # Load calibration scores. If a calib raw file exists, use it; else require precomputed.
    calib_raw_path = Path(args.calib_raw)
    if not calib_raw_path.exists():
        print(f"calib raw not found at {calib_raw_path}; run calibrate_routeA first")
        return
    calib_rows = [json.loads(l) for l in calib_raw_path.read_text().splitlines() if l.strip()]
    correct_scores = [r["score"] for r in calib_rows if r.get("is_correct")]

    # Load val 200q
    import random
    raw = json.loads(Path(args.questions).read_text())
    rng = random.Random(args.seed)
    questions = rng.sample(raw, args.n) if args.n < len(raw) else raw

    alphas = [float(x) for x in args.alphas.split(",")]
    summary_rows = []
    for alpha in alphas:
        tau = conformal_quantile(correct_scores, alpha=alpha)
        gate = ConformalGate(openai_client=openai_client, tau_high=tau, tau_low=tau-0.4, alpha=alpha)
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

        n = len(results)
        em = sum(r.em for r in results) / n
        acc = sum(r.acc for r in results) / n
        tok = sum(r.total_tokens for r in results) / n
        sas = sum(r.sas_committed for r in results) / n
        # SAS-error: SAS-committed AND answer wrong
        sas_err = sum(1 for r in results if r.sas_committed and r.acc == 0) / max(1, sum(1 for r in results if r.sas_committed))
        row = {"alpha": alpha, "tau": tau, "n": n, "em": em, "acc": acc,
               "avg_tokens": tok, "sas_rate": sas, "sas_error_rate": sas_err}
        summary_rows.append(row)
        print(f"alpha={alpha:.3f} tau={tau:.3f}  em={em:.3f} acc={acc:.3f} tok={tok:.0f} "
              f"sas={sas:.2%} sas_err={sas_err:.2%}")

    (out_dir / "alpha_sweep.json").write_text(json.dumps(summary_rows, indent=2))
    # wandb summary log
    try:
        import wandb
        run = wandb.init(project=cfg["wandb"]["project_eval"], entity=cfg["wandb"].get("entity"), name="alpha_sweep", config={"alphas": alphas, "n": args.n}, reinit=True)
        for r in summary_rows:
            wandb.log({"alpha/alpha": r["alpha"], "alpha/tau": r["tau"], "alpha/em": r["em"], "alpha/acc": r["acc"], "alpha/avg_tokens": r["avg_tokens"], "alpha/sas_rate": r["sas_rate"], "alpha/sas_error_rate": r["sas_error_rate"]})
        wandb.finish()
    except Exception as e:
        import logging; logging.warning("wandb log failed: %s", e)

    await retriever.aclose(); await vllm.aclose(); await openai_client.aclose()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--calib-raw", default="results/route_a_calibration.raw.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--alphas", default="0.01,0.05,0.1,0.2")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
