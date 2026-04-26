"""AMAS runner: process a questions JSON file through AMASPipeline.

Usage::

    python scripts/run_amas.py \\
      --config configs/amas.yaml \\
      --questions data/musique/questions_smoke50_seed42.json \\
      --output-dir results/amas_smoke5 \\
      --server-url http://localhost:8001/v1 \\
      --retriever-url http://node408:8003 \\
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

from amas.config import Config  # noqa: E402
from amas.pipeline import AMASPipeline  # noqa: E402
from amas.types import PipelineResult  # noqa: E402

logger = logging.getLogger("amas.runner")


def _completed_ids(out_dir: Path) -> set[str]:
    f = out_dir / "predictions.jsonl"
    if not f.exists():
        return set()
    done = set()
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(str(json.loads(line).get("id", "")))
        except json.JSONDecodeError:
            pass
    return done


def _to_output_row(result: PipelineResult, gold: str, wallclock: float) -> dict:
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
    q: dict, pipeline: AMASPipeline, sem: asyncio.Semaphore,
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
            row = _to_output_row(result, gold, elapsed)
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
        return row


async def _run(
    config_path: str, questions_path: str, output_dir: str,
    server_url: str | None, retriever_url: str | None, concurrency: int,
) -> None:
    config = Config.from_yaml(config_path)
    if server_url:
        config.set("llm_defaults.base_url", server_url)
    if retriever_url:
        config.set("retriever.base_url", retriever_url)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"
    (out_dir / "config_used.yaml").write_text(
        json.dumps(config.raw(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    questions: list[dict] = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    done = _completed_ids(out_dir)
    todo = [q for q in questions if str(q.get("id", "")) not in done]
    logger.info("Loaded %d questions; %d done; %d to run.",
                len(questions), len(done), len(todo))
    if not todo:
        return

    pipeline = AMASPipeline(config)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    t0 = time.time()
    tasks = [_process_one(q, pipeline, sem, out_file, lock) for q in todo]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - t0
    succeeded = sum(1 for r in results if isinstance(r, dict) and r.get("answer", "") != "")
    print(f"Done: {len(results)} processed in {elapsed:.1f}s "
          f"(answered={succeeded}/{len(results)})")


def main() -> None:
    p = argparse.ArgumentParser(description="AMAS runner")
    p.add_argument("--config", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--server-url", default=None,
                   help="Override llm_defaults.base_url (e.g. http://localhost:8001/v1)")
    p.add_argument("--retriever-url", default=None,
                   help="Override retriever.base_url (e.g. http://node408:8003)")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(_run(
        config_path=args.config, questions_path=args.questions,
        output_dir=args.output_dir, server_url=args.server_url,
        retriever_url=args.retriever_url, concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
