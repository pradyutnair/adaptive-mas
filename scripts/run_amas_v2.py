"""AMAS v2 runner: process questions through AMASv2Pipeline.

Usage::

    .venv/bin/python scripts/run_amas_v2.py \
      --config configs/amas_v2.yaml \
      --questions data/musique/opera408_50.json \
      --output-dir results/amas_v2_pilot50 \
      --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from amas_v2.config import Config
from amas_v2.pipeline import AMASv2Pipeline
from amas_v2.types import PipelineResult

logger = logging.getLogger("amas_v2.runner")


def _completed_ids(out_dir: Path) -> set[str]:
    f = out_dir / "predictions.jsonl"
    if not f.exists():
        return set()
    done = set()
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(str(json.loads(line).get("id", "")))
        except json.JSONDecodeError:
            pass
    return done


def _to_row(result: PipelineResult, gold: str, wallclock: float) -> dict:
    return {
        "id": result.question_id,
        "question": result.question,
        "answer": result.answer,
        "gold_answer": gold,
        "metadata": {
            "step_trace": [t.to_dict() for t in result.step_trace],
            "num_subagent_calls": result.num_subagent_calls,
            "total_tokens": result.total_tokens,
            "orchestrator_tokens": result.orchestrator_tokens,
            "subagent_tokens": result.subagent_tokens,
            "facts_used": [f.to_dict() for f in result.facts_used],
            "retrieved_doc_ids": result.retrieved_doc_ids,
            "retrieved_docs_total": result.retrieved_docs_total,
            "route_decision": result.route_decision,
            "wallclock_seconds": round(wallclock, 3),
            "extras": result.extras,
        },
    }


async def _process_one(
    q: dict, pipeline: AMASv2Pipeline, sem: asyncio.Semaphore,
    out_file: Path, lock: asyncio.Lock,
) -> dict:
    qid = str(q.get("id", ""))
    qtext = str(q.get("question", ""))
    gold = str(q.get("answer", ""))
    async with sem:
        t0 = time.time()
        try:
            result = await pipeline.run(question=qtext, question_id=qid)
            elapsed = time.time() - t0
            row = _to_row(result, gold, elapsed)
        except Exception as exc:
            elapsed = time.time() - t0
            logger.exception("FAILED %s: %s", qid, exc)
            row = {
                "id": qid, "question": qtext, "answer": "", "gold_answer": gold,
                "metadata": {
                    "error": str(exc),
                    "wallclock_seconds": round(elapsed, 3),
                    "total_tokens": 0, "orchestrator_tokens": 0,
                    "subagent_tokens": 0, "num_subagent_calls": 0,
                    "step_trace": [], "facts_used": [],
                    "retrieved_doc_ids": [], "retrieved_docs_total": 0,
                    "route_decision": "error", "extras": {},
                },
            }
        async with lock:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        status = "OK" if row["answer"] else "BLANK"
        logger.info("[%s] %s ans=%s gold=%s tok=%d",
                     status, qid, row["answer"][:60], gold[:40],
                     row["metadata"].get("total_tokens", 0))
        return row


async def _run(args) -> None:
    config = Config.from_yaml(args.config)
    if args.server_url:
        config.set("agents.planner.base_url", args.server_url)
    if args.retriever_url:
        config.set("retriever.base_url", args.retriever_url)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"
    (out_dir / "config_used.yaml").write_text(
        json.dumps(config.raw(), indent=2, ensure_ascii=False), encoding="utf-8",
    )

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    done = _completed_ids(out_dir)
    todo = [q for q in questions if str(q.get("id", "")) not in done]
    logger.info("Loaded %d questions; %d done; %d to run.", len(questions), len(done), len(todo))
    if not todo:
        print("Nothing to do.")
        return

    pipeline = AMASv2Pipeline(config)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    t0 = time.time()
    tasks = [_process_one(q, pipeline, sem, out_file, lock) for q in todo]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - t0
    succeeded = sum(1 for r in results if isinstance(r, dict) and r.get("answer", ""))
    total = len(results)
    print(f"Done: {total} processed in {elapsed:.1f}s (answered={succeeded}/{total})")


def main() -> None:
    p = argparse.ArgumentParser(description="AMAS v2 runner")
    p.add_argument("--config", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--server-url", default=None)
    p.add_argument("--retriever-url", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
