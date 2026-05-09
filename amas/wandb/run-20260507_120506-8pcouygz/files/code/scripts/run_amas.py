"""Per-question runner. Loads config, builds clients, gate, orchestrator; runs AMAS pipeline.

Usage:
  python scripts/run_amas.py \
      --questions /local/yzheng/pnair/data/musique/questions_1000_seedfull_combined.json \
      --out-dir results/run01_amas/musique \
      --gate conformal --t-max 3 --n 1000 --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm.asyncio import tqdm_asyncio

from amas.agents import load_prompts
from amas.config import HERAConfig, load_env
from amas.gates import make_gate
from amas.library import ExperienceLibrary
from amas.lm import OpenAIClient, VLLMClient
from amas.orchestrator import Orchestrator
from amas.pipeline import AmasResult, run_amas
from amas.retriever import RetrieverClient


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def load_questions(path: str | Path, n: int | None = None,
                    seed: int = 42) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Unsupported questions format: {type(raw)}")
    if n is not None and n < len(raw):
        rng = random.Random(seed)
        raw = rng.sample(raw, n)
    return raw


async def main(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    load_env()
    cfg = load_config(args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Clients ----
    vcfg = cfg["vllm"]
    vllm = VLLMClient(
        endpoints=vcfg["endpoints"],
        model=vcfg["model"],
        max_tokens=vcfg["max_tokens"],
        temperature=vcfg["temperature"],
        concurrency=vcfg["concurrency"],
    )
    ocfg = cfg["openai"]
    openai_client = OpenAIClient(
        model=ocfg["model"],
        max_tokens=ocfg["max_tokens"],
        temperature=ocfg["temperature"],
        concurrency=ocfg["concurrency"],
    )
    rcfg = cfg["retriever"]
    retriever = RetrieverClient(
        url=rcfg["url"], topk=rcfg["topk"], concurrency=rcfg["concurrency"],
    )

    # ---- Library + prompts ----
    paths = cfg["paths"]
    lib_path = Path(paths["library"])
    prompts_path = Path(paths["prompts"])
    library = ExperienceLibrary.load(lib_path) if lib_path.exists() else ExperienceLibrary()
    prompts = load_prompts(prompts_path) if prompts_path.exists() else load_prompts("__seed__")

    orchestrator = Orchestrator(
        vllm=vllm, openai_client=openai_client, retriever=retriever,
        library=library, prompts=prompts,
        retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8,
    )

    # ---- Gate ----
    gate_cfg = dict(cfg.get("gate", {}))
    if args.gate:
        gate_cfg["name"] = args.gate
    gate = make_gate(
        gate_cfg.get("name", "conformal"),
        openai_client=openai_client,
        cfg=gate_cfg,
    )

    # ---- W&B ----
    use_wandb = bool(cfg.get("wandb", {}).get("enabled", True)) and not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg["wandb"]["project_eval"],
                entity=cfg["wandb"].get("entity"),
                name=args.run_name or f"{Path(args.questions).stem}_{gate_cfg.get('name','off')}",
                config={
                    "gate": gate_cfg,
                    "pipeline": cfg["pipeline"],
                    "openai_model": ocfg["model"],
                    "vllm_model": vcfg["model"],
                    "retriever_topk": rcfg["topk"],
                    "n": args.n,
                    "questions": args.questions,
                },
                reinit=True,
            )
        except Exception as e:
            logging.warning("wandb init failed: %s", e)
            wandb_run = None

    # ---- Questions ----
    questions = load_questions(args.questions, n=args.n, seed=args.seed)
    pcfg = cfg["pipeline"]

    sem = asyncio.Semaphore(args.concurrency)

    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"
    fh = open(pred_path, "w")

    async def one(q: dict[str, Any], idx: int) -> AmasResult:
        async with sem:
            try:
                res = await run_amas(
                    question=q.get("question", ""),
                    gold=q.get("answer", ""),
                    qid=str(q.get("id", "")),
                    gate=gate,
                    orchestrator=orchestrator,
                    retriever=retriever,
                    openai_client=openai_client,
                    t_max=pcfg["t_max"],
                    probe_group_size=pcfg["probe_group_size"],
                    probe_topk=pcfg["probe_topk"],
                    rollout_temperature=pcfg.get("rollout_temperature", 0.0),
                )
            except Exception as e:
                logging.exception("question %s failed", q.get("id"))
                res = AmasResult(qid=str(q.get("id", "")), question=q.get("question", ""),
                                 gold=q.get("answer", ""), profile="bridge",
                                 gate=gate.name, final_answer="", error=str(e)[:300])
            d = res.to_dict()
            d["question"] = q.get("question", "")
            d["gold"] = q.get("answer", "")
            d["question_type"] = q.get("question_type", "")
            d["source"] = q.get("source", "")
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            return res

    t0 = time.time()
    coros = [one(q, i) for i, q in enumerate(questions)]
    results: list[AmasResult] = await tqdm_asyncio.gather(*coros, desc=f"[{Path(args.questions).stem}]")
    fh.close()
    elapsed = time.time() - t0

    n = len(results)
    em = sum(r.em for r in results) / max(1, n)
    f1 = sum(r.f1 for r in results) / max(1, n)
    cont = sum(r.contain for r in results) / max(1, n)
    acc = sum(r.acc for r in results) / max(1, n)
    avg_tok = sum(r.total_tokens for r in results) / max(1, n)
    avg_turns = sum(r.n_turns for r in results) / max(1, n)
    sas_rate = sum(int(r.sas_committed) for r in results) / max(1, n)

    summary = {
        "n": n, "em": em, "f1": f1, "contain": cont, "acc": acc,
        "avg_tokens": avg_tok, "avg_turns": avg_turns, "sas_rate": sas_rate,
        "elapsed_s": elapsed,
        "gate": gate.name, "config": str(args.config),
        "questions": str(args.questions),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if wandb_run is not None:
        try:
            import wandb
            wandb.log({
                "eval/em": em, "eval/f1": f1, "eval/contain": cont, "eval/acc": acc,
                "eval/avg_tokens": avg_tok, "eval/avg_turns": avg_turns,
                "eval/sas_rate": sas_rate, "eval/n": n, "eval/elapsed_s": elapsed,
            })
            artifact = wandb.Artifact(name=f"predictions_{Path(args.questions).stem}",
                                      type="predictions")
            artifact.add_file(str(pred_path))
            artifact.add_file(str(summary_path))
            wandb.log_artifact(artifact)
            wandb.finish()
        except Exception as e:
            logging.warning("wandb log failed: %s", e)

    await retriever.aclose()
    await vllm.aclose()
    await openai_client.aclose()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gate", default=None,
                     help="Override gate name (conformal|bayesian|oracle|random|off|sas_only)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
