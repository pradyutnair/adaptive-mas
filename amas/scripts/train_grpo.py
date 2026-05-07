"""TF-GRPO training: HERA-faithful, multi-turn trajectory schema.

Per query in stratified train set:
  1. Profile question (uses cached annotations if present).
  2. G=4 rollouts via Orchestrator.sample_topology + execute (no gate; pure HERA-style training).
  3. rank by (F1 desc, tokens asc).
  4. mixed-outcome -> SA extraction (verbatim Appendix B); ADD/MERGE/PRUNE/KEEP.
  5. all-fail -> topology mutation (Algorithm 6).
  6. Library/prompts checkpointed every batch.
  7. wandb logged: per-step F1/tokens, library size, insights added, RoPE deltas.

Run after P0 smoke + P1 ledger validation. Multi-turn extension is implicit in pipeline.run_amas
(used at inference); training itself runs single-turn HERA-style (paper Algorithm 2 verbatim).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from amas.agents import load_prompts, save_prompts
from amas.config import load_env
from amas.data import load_qa_jsonl
from amas.grpo import GroupResult, grpo_step
from amas.library import ExperienceLibrary, profile_question
from amas.lm import OpenAIClient, VLLMClient
from amas.orchestrator import Orchestrator
from amas.retriever import RetrieverClient
from amas.rope import FailureBuffer, add_traj_to_buffer, rope_update_agent


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_fh = open(log_path, "w")

    vcfg = cfg["vllm"]
    ocfg = cfg["openai"]
    rcfg = cfg["retriever"]
    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=vcfg["max_tokens"], temperature=vcfg["temperature"],
                      concurrency=vcfg["concurrency"])
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=ocfg["max_tokens"],
                                 temperature=ocfg["temperature"], concurrency=ocfg["concurrency"])
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"],
                                concurrency=rcfg["concurrency"])

    library = ExperienceLibrary.load(args.init_library) if args.init_library and Path(args.init_library).exists() else ExperienceLibrary(max_entries=args.library_max); library.max_entries = args.library_max
    prompts = load_prompts(args.init_prompts) if args.init_prompts and Path(args.init_prompts).exists() else load_prompts("__seed__")
    orch = Orchestrator(vllm=vllm, openai_client=openai_client, retriever=retriever,
                        library=library, prompts=prompts,
                        retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8)

    buffer = FailureBuffer(capacity=args.rope_buffer)

    # wandb
    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                project=cfg["wandb"]["project_grpo"],
                entity=cfg["wandb"].get("entity"),
                name=args.run_name or f"grpo_{Path(args.train_path).stem}",
                config={
                    "group_size": args.group_size,
                    "rollout_temperature": args.rollout_temperature,
                    "library_max": args.library_max,
                    "rope_buffer": args.rope_buffer,
                    "rope_per_query": args.rope_per_query,
                    "rope_min_buffer": args.rope_min_buffer,
                    "train_path": args.train_path,
                    "epochs": args.epochs,
                    "openai_model": ocfg["model"],
                    "vllm_model": vcfg["model"],
                },
                reinit=True,
            )
        except Exception as e:
            logging.warning("wandb init failed: %s", e)

    examples = load_qa_jsonl(args.train_path)
    if args.train_size:
        examples = examples[: args.train_size]
    logging.info("Loaded %d training examples", len(examples))

    step = 0
    f1_window = deque(maxlen=20)
    tok_window = deque(maxlen=20)

    for epoch in range(args.epochs):
        for ex in examples:
            step += 1
            t0 = time.time()
            orch._current_qid = ex.id  # type: ignore
            try:
                gr: GroupResult = await grpo_step(
                    orch, vllm, library,
                    query=ex.question, gold=ex.answer,
                    group_size=args.group_size,
                    temperature=args.rollout_temperature,
                    enable_mutation=True,
                )
            except Exception as e:
                logging.exception("grpo_step failed for %s: %s", ex.id, e)
                continue
            elapsed = time.time() - t0

            # Add failed trajectories to RoPE buffer
            for t in gr.trajectories:
                if t.failed_agent:
                    add_traj_to_buffer(buffer, t)

            # Per-query RoPE
            rope_updates = 0
            if args.rope_per_query and gr.failed_agents:
                for fa in set(gr.failed_agents):
                    if buffer.size(fa) >= args.rope_min_buffer:
                        try:
                            new_p = await rope_update_agent(
                                orch, vllm, prompts, buffer, fa,
                                num_variants=args.rope_variants,
                                max_failures=args.rope_max_failures,
                            )
                            if new_p:
                                prompts[fa] = new_p
                                rope_updates += 1
                        except Exception as e:
                            logging.warning("RoPE update for %s failed: %s", fa, e)

            best = gr.trajectories[0] if gr.trajectories else None
            avg_f1 = sum(t.f1 for t in gr.trajectories) / max(1, len(gr.trajectories))
            avg_tok = sum(t.total_tokens for t in gr.trajectories) / max(1, len(gr.trajectories))
            f1_window.append(avg_f1)
            tok_window.append(avg_tok)

            row = {
                "step": step, "qid": ex.id, "profile": gr.profile,
                "best_f1": best.f1 if best else 0.0,
                "best_em": best.em if best else 0.0,
                "best_tokens": best.total_tokens if best else 0,
                "best_answer": best.answer if best else "",
                "avg_f1": avg_f1, "avg_tokens": avg_tok,
                "n_insights": len(gr.insights),
                "n_failed_agents": len(gr.failed_agents),
                "rope_updates": rope_updates,
                "library_size": len(library.entries),
                "elapsed_s": elapsed,
            }
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()

            if run is not None:
                try:
                    import wandb
                    wandb.log({
                        "train/step": step,
                        "train/best_f1": row["best_f1"],
                        "train/avg_f1": row["avg_f1"],
                        "train/avg_tokens": row["avg_tokens"],
                        "train/best_tokens": row["best_tokens"],
                        "train/library_size": row["library_size"],
                        "train/n_insights_added": row["n_insights"],
                        "train/rope_updates": row["rope_updates"],
                        "train/f1_ma20": sum(f1_window) / max(1, len(f1_window)),
                        "train/tokens_ma20": sum(tok_window) / max(1, len(tok_window)),
                        "train/elapsed_s": row["elapsed_s"],
                    }, step=step)
                except Exception:
                    pass

            # Checkpoint every batch.
            if step % args.batch_size == 0:
                library.save(out_dir / "library.json")
                save_prompts(prompts, out_dir / "prompts.json")
                logging.info("step %d: best F1=%.3f tokens=%d lib=%d",
                             step, row["best_f1"], row["best_tokens"], row["library_size"])

    log_fh.close()
    library.save(out_dir / "library.json")
    save_prompts(prompts, out_dir / "prompts.json")
    if run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    await retriever.aclose()
    await vllm.aclose()
    await openai_client.aclose()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rollout-temperature", type=float, default=0.9)
    ap.add_argument("--library-max", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--rope-per-query", action="store_true")
    ap.add_argument("--rope-min-buffer", type=int, default=2)
    ap.add_argument("--rope-buffer", type=int, default=8)
    ap.add_argument("--rope-variants", type=int, default=3)
    ap.add_argument("--rope-max-failures", type=int, default=4)
    ap.add_argument("--init-library", default=None, help="warm-start from existing library.json")
    ap.add_argument("--init-prompts", default=None, help="warm-start from existing prompts.json")
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
