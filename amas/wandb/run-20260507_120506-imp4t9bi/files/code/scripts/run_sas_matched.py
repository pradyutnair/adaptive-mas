"""SAS-matched baseline (Tran & Kiela 2026 §D.1.2 SAS-L scaffold).

Single-agent Qwen3-14B with thinking enabled at AMAS mean-token budget.
Used to test the SAS-vs-MAS critique: if SAS matches AMAS at same budget, multi-agent
contribution evaporates. Plan §6 calls this baseline mandatory.

Pipeline per question:
  1. Retrieve top-k passages.
  2. Single Qwen3-14B+thinking call with SAS-L user prefix scaffold (paper §D.1.2).
  3. Extract answer span via the same span-normaliser AMAS uses.

Usage:
  python scripts/run_sas_matched.py \
      --questions /local/yzheng/pnair/data/musique/questions_1000_seedfull_combined.json \
      --out-dir results/sas_matched/musique --n 200 --concurrency 16 \
      --thinking-budget 4096
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from amas.config import load_env
from amas.lm import VLLMClient, parse_json_lenient
from amas.metric import accuracy, contain, exact_match, f1_score
from amas.orchestrator import normalize_answer_span
from amas.retriever import RetrieverClient, format_passages


SAS_L_SYSTEM = (
    "You are a helpful assistant. Think step by step, then answer. "
    "Be as succinct as possible. Do NOT repeat the question. Return ONLY the final answer requested."
)

SAS_L_USER_TEMPLATE = (
    "I want you to analyze the following question from multiple perspectives before answering.\n\n"
    "1. Identify ambiguities.\n"
    "2. Formulate at least two plausible interpretations.\n"
    "3. Evaluate the interpretations and choose the most likely one.\n"
    "4. Answer based on the most likely interpretation.\n\n"
    "Passages:\n{passages}\n\n"
    "The question is: {question}\n\n"
    "Final answer (one entity, one date, one short phrase, or yes/no — max 8 words, "
    "bare span, no preamble):"
)


def load_questions(path: str | Path, n: int | None, seed: int = 42) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"unsupported: {type(raw)}")
    if n is not None and n < len(raw):
        rng = random.Random(seed)
        raw = rng.sample(raw, n)
    return raw


async def run_one(q: dict, *, retriever: RetrieverClient, vllm: VLLMClient,
                  topk: int, max_tokens: int, thinking: bool) -> dict:
    t0 = time.time()
    qtext = q.get("question", "")
    gold = q.get("answer", "")

    try:
        passages = await retriever.retrieve(qtext, topk=topk)
    except Exception as e:
        logging.warning("retrieval failed: %s", e)
        passages = []

    pblock = format_passages(passages, max_chars_per=600)
    user = SAS_L_USER_TEMPLATE.format(passages=pblock, question=qtext)

    extra = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    res = await vllm.chat(SAS_L_SYSTEM, user, temperature=0.0, max_tokens=max_tokens,
                          extra_body=extra)
    raw_pred = (res.text or "").strip()
    # Strip <think>...</think> blocks (Qwen3 thinking mode wraps reasoning in this).
    import re
    raw_pred = re.sub(r"<think>.*?</think>", "", raw_pred, flags=re.DOTALL).strip()
    # Strip "Final answer:" prefix
    raw_pred = re.sub(r"^.*?(final\s+answer:?)\s*", "", raw_pred,
                      flags=re.IGNORECASE | re.DOTALL).strip()
    pred = normalize_answer_span(raw_pred, question=qtext, max_words=10)

    em = exact_match(pred, gold)
    f1 = f1_score(pred, gold)
    cont = contain(pred, gold)
    acc = accuracy(pred, gold)
    return {
        "qid": str(q.get("id", "")),
        "question": qtext, "gold": gold,
        "pred": pred, "raw_pred": raw_pred[:300],
        "em": em, "f1": f1, "contain": cont, "acc": acc,
        "tokens": int(res.prompt_tokens + res.completion_tokens),
        "elapsed_s": time.time() - t0,
        "question_type": q.get("question_type", ""),
    }


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vcfg = cfg["vllm"]
    rcfg = cfg["retriever"]
    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=args.max_tokens, temperature=0.0,
                      concurrency=vcfg["concurrency"])
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"],
                                concurrency=rcfg["concurrency"])

    questions = load_questions(args.questions, args.n, args.seed)

    # wandb
    use_wandb = not args.no_wandb
    run = None
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project=cfg["wandb"]["project_eval"],
                entity=cfg["wandb"].get("entity"),
                name=args.run_name or f"sas_matched_{Path(args.questions).stem}",
                config={
                    "max_tokens": args.max_tokens,
                    "thinking": args.thinking,
                    "topk": rcfg["topk"],
                    "n": args.n,
                    "model": vcfg["model"],
                },
                reinit=True,
            )
        except Exception as e:
            logging.warning("wandb init failed: %s", e)

    sem = asyncio.Semaphore(args.concurrency)

    async def one(q):
        async with sem:
            try:
                return await run_one(q, retriever=retriever, vllm=vllm,
                                     topk=rcfg["topk"], max_tokens=args.max_tokens,
                                     thinking=args.thinking)
            except Exception as e:
                logging.exception("question %s failed", q.get("id"))
                return {"qid": str(q.get("id", "")), "question": q.get("question", ""),
                        "gold": q.get("answer", ""), "pred": "", "em": 0.0, "f1": 0.0,
                        "contain": 0.0, "acc": 0.0, "tokens": 0, "error": str(e)[:200]}

    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"
    fh = open(pred_path, "w")

    rows = []
    coros = [one(q) for q in questions]
    t0 = time.time()
    for fut in tqdm_asyncio.as_completed(coros, desc=Path(args.questions).stem):
        r = await fut
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        rows.append(r)
    fh.close()
    elapsed = time.time() - t0
    n = len(rows)
    em = sum(r["em"] for r in rows) / max(1, n)
    f1 = sum(r["f1"] for r in rows) / max(1, n)
    cont = sum(r["contain"] for r in rows) / max(1, n)
    acc = sum(r["acc"] for r in rows) / max(1, n)
    avg_tok = sum(r["tokens"] for r in rows) / max(1, n)

    summary = {
        "method": "sas_matched", "n": n,
        "em": em, "f1": f1, "contain": cont, "acc": acc,
        "avg_tokens": avg_tok, "elapsed_s": elapsed,
        "thinking": args.thinking, "max_tokens": args.max_tokens,
        "questions": str(args.questions),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if run is not None:
        try:
            import wandb
            wandb.log({"eval/em": em, "eval/f1": f1, "eval/acc": acc,
                       "eval/avg_tokens": avg_tok, "eval/n": n,
                       "eval/elapsed_s": elapsed})
            wandb.finish()
        except Exception:
            pass

    await retriever.aclose()
    await vllm.aclose()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096,
                     help="thinking budget cap; matched to AMAS mean")
    ap.add_argument("--thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
