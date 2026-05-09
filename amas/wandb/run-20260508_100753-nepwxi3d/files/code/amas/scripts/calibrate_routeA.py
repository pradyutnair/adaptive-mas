"""Split-conformal calibration of Route A foreign verifier.

Runs the verifier on a held-out calibration set. For each calibration item:
  - Build (question, candidate=probe-or-MAS-answer, top ledger entries).
  - Score = log P(YES) - log P(NO) (via verifier confidence).
  - Compute SAS-correct (probe answer accuracy) and SAS-error labels.
Then takes empirical (1-alpha) quantile of correct-item scores -> tau_high.
Writes results/route_a_calibration.json.

Usage:
  python scripts/calibrate_routeA.py \
      --questions data/routeA_calib_200.jsonl \
      --out results/route_a_calibration.json --alpha 0.05 --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from amas.config import load_env
from amas.gates.conformal import (
    VERIFIER_SYSTEM,
    build_verifier_user,
    conformal_quantile,
    write_calibration,
)
from amas.config import build_probe_client, load_env, validate_probe_config
from amas.ledger import BeliefState, Ledger
from amas.lm import OpenAIClient, VLLMClient, parse_json_lenient
from amas.metric import accuracy
from amas.probe import run_probe
from amas.retriever import RetrieverClient, format_passages


async def calibrate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING)
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate_probe_config(cfg)

    rcfg = cfg["retriever"]
    ocfg = cfg["openai"]
    vcfg = cfg["vllm"]
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"], concurrency=rcfg["concurrency"])
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=300,
                                 temperature=0.0, concurrency=ocfg["concurrency"])
    # Build a vLLM client only if probe config will use it. We avoid double-creating
    # endpoints when probe is configured `kind: openai`.
    probe_kind = str((cfg.get("probe") or {}).get("kind", "vllm")).lower()
    vllm = None
    if probe_kind == "vllm":
        vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                          max_tokens=vcfg["max_tokens"], temperature=vcfg["temperature"],
                          concurrency=vcfg.get("concurrency", 12))
    probe_client, probe_owned = build_probe_client(cfg, vllm=vllm, openai_client=openai_client)

    qpath = Path(args.questions)
    if qpath.suffix == ".jsonl":
        items = [json.loads(l) for l in qpath.read_text().splitlines() if l.strip()]
    else:
        items = json.loads(qpath.read_text())
    if args.n:
        items = items[: args.n]

    sem = asyncio.Semaphore(args.concurrency)

    async def one(q: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            ledger = Ledger()
            belief = BeliefState(top_k=5)
            probe = await run_probe(
                q["question"], retriever=retriever, lm_client=probe_client,
                ledger=ledger, belief=belief,
                topk=rcfg["topk"], group_size=cfg["pipeline"]["probe_group_size"],
                temperature=0.7, turn=0,
            )
            candidate = probe.consensus_answer
            if not candidate:
                return {"id": q.get("id", ""), "score": None, "is_correct": False}
            ledger_block = ledger.summarize_for_agent(n=8, max_chars=900)
            user = build_verifier_user(q["question"], candidate, ledger_block, "(no passages snippet)")
            res = await openai_client.chat(VERIFIER_SYSTEM, user, temperature=0.0,
                                           max_tokens=180, json_mode=True)
            parsed = parse_json_lenient(res.text) or {}
            verdict = str(parsed.get("verdict", "NO")).upper()
            try:
                conf = float(parsed.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            import math
            eps = 1e-3
            conf = min(max(conf, eps), 1 - eps)
            s = math.log(conf / (1 - conf))
            score = s if verdict == "YES" else -s
            is_correct = bool(accuracy(candidate, q.get("answer", "")) > 0)
            return {"id": q.get("id", ""), "answer": candidate, "gold": q.get("answer", ""),
                    "score": float(score), "is_correct": is_correct, "verdict": verdict, "conf": conf}

    rows = await tqdm_asyncio.gather(*[one(q) for q in items], desc="calibrate")
    rows = [r for r in rows if r.get("score") is not None]
    correct_scores = [r["score"] for r in rows if r["is_correct"]]
    incorrect_scores = [r["score"] for r in rows if not r["is_correct"]]

    tau_high = conformal_quantile(correct_scores, alpha=args.alpha) if correct_scores else 0.7
    # Defer band lower bound: 25th percentile of incorrect-item scores.
    if incorrect_scores:
        incorrect_scores_sorted = sorted(incorrect_scores)
        idx = max(0, int(0.25 * len(incorrect_scores_sorted)))
        tau_low = float(incorrect_scores_sorted[idx])
    else:
        tau_low = 0.3

    # wandb summary log
    try:
        import wandb
        run = wandb.init(project=cfg["wandb"]["project_eval"], entity=cfg["wandb"].get("entity"), name="routeA_calibration_alpha%.3f" % args.alpha, config={"alpha": args.alpha, "n": args.n, "questions": args.questions}, reinit=True)
        wandb.log({"calib/n_total": len(rows), "calib/n_correct": len(correct_scores), "calib/n_incorrect": len(incorrect_scores), "calib/tau_high": tau_high, "calib/tau_low": tau_low})
        wandb.finish()
    except Exception as e:
        logging.warning("wandb log failed: %s", e)

    out_path = Path(args.out)
    write_calibration(out_path, tau_high=tau_high, tau_low=tau_low,
                       alpha=args.alpha, n_calib=len(rows))
    print(json.dumps({
        "n_total": len(rows), "n_correct": len(correct_scores),
        "n_incorrect": len(incorrect_scores),
        "tau_high": tau_high, "tau_low": tau_low, "alpha": args.alpha,
        "out": str(out_path),
    }, indent=2))

    raw_path = out_path.with_suffix(".raw.jsonl")
    raw_path.write_text("\n".join(json.dumps(r) for r in rows))

    # Close every client exactly once (handles shared/owned/None combinations).
    to_close = [retriever, openai_client]
    if vllm is not None:
        to_close.append(vllm)
    if probe_owned:
        to_close.append(probe_client)
    seen: set[int] = set()
    for c in to_close:
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        if hasattr(c, "aclose"):
            await c.aclose()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default="results/route_a_calibration.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(calibrate(parse_args()))
