"""Sweep Bayesian gate tau_b on val 200q. Find Pareto knee.

tau_b: commit when (top.net_score - lambda * entropy) >= tau_b.
Higher tau_b -> stricter commit -> more MAS lane usage.
"""
from __future__ import annotations
import argparse, asyncio, json, logging, sys
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


async def main(args):
    logging.basicConfig(level=logging.WARNING)
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vcfg = cfg["vllm"]; ocfg = cfg["openai"]; rcfg = cfg["retriever"]; pcfg = cfg["pipeline"]
    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=vcfg["max_tokens"], temperature=vcfg["temperature"], concurrency=vcfg["concurrency"])
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=ocfg["max_tokens"],
                                  temperature=ocfg["temperature"], concurrency=ocfg["concurrency"])
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"], concurrency=rcfg["concurrency"])
    paths = cfg["paths"]
    library = ExperienceLibrary.load(paths["library"]) if Path(paths["library"]).exists() else ExperienceLibrary()
    prompts = load_prompts(paths["prompts"]) if Path(paths["prompts"]).exists() else load_prompts("__seed__")
    orchestrator = Orchestrator(vllm=vllm, openai_client=openai_client, retriever=retriever,
                                 library=library, prompts=prompts,
                                 retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8)
    import random
    raw = json.loads(Path(args.questions).read_text())
    rng = random.Random(args.seed)
    questions = rng.sample(raw, args.n) if args.n < len(raw) else raw

    sweep = [float(x) for x in args.taus.split(",")]
    summary_rows = []
    for tau_b in sweep:
        gate = BayesianGate(tau_b=tau_b, lambda_=0.5)
        sem = asyncio.Semaphore(args.concurrency)
        async def one(q):
            async with sem:
                return await run_amas(question=q.get("question",""), gold=q.get("answer",""),
                                       qid=str(q.get("id","")), gate=gate, orchestrator=orchestrator,
                                       retriever=retriever, openai_client=openai_client,
                                       t_max=pcfg["t_max"], probe_group_size=pcfg["probe_group_size"],
                                       probe_topk=pcfg["probe_topk"],
                                       rollout_temperature=pcfg.get("rollout_temperature", 0.0))
        results = await asyncio.gather(*[one(q) for q in questions])
        n = len(results)
        em = sum(r.em for r in results)/n; f1 = sum(r.f1 for r in results)/n
        acc = sum(r.acc for r in results)/n; tok = sum(r.total_tokens for r in results)/n
        sas = sum(r.sas_committed for r in results)/n
        row = {"tau_b": tau_b, "n": n, "em": em, "f1": f1, "acc": acc, "avg_tokens": tok, "sas_rate": sas}
        summary_rows.append(row)
        print("tau_b=%.2f em=%.3f f1=%.3f acc=%.3f tok=%.0f sas=%.2f%%" %
              (tau_b, em, f1, acc, tok, sas*100))
    (out_dir / "tau_b_sweep.json").write_text(json.dumps(summary_rows, indent=2))
    try:
        import wandb
        wandb.init(project=cfg["wandb"]["project_eval"], name="tau_b_sweep",
                   config={"taus": sweep, "n": args.n}, reinit=True)
        for r in summary_rows:
            wandb.log({"tau_b/tau_b": r["tau_b"], "tau_b/em": r["em"], "tau_b/f1": r["f1"],
                       "tau_b/acc": r["acc"], "tau_b/avg_tokens": r["avg_tokens"],
                       "tau_b/sas_rate": r["sas_rate"]})
        wandb.finish()
    except Exception as e:
        logging.warning("wandb log failed: %s", e)
    await retriever.aclose(); await vllm.aclose(); await openai_client.aclose()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--taus", default="0.5,1.0,1.5,2.0,2.5,3.0")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
