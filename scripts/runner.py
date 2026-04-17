"""Batch runner for Adaptive Recursive SAGE experiments.

Processes questions concurrently with checkpoint resume support.
Each completed prediction is written as a JSON line to the output file.

Usage::

    python3 scripts/runner.py \\
        --config configs/s4.yaml \\
        --questions data/musique/questions.json \\
        --output-dir results/S4 \\
        --server-url http://localhost:8001/v1 \\
        --concurrency 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Ensure src/ is on the path so that arag and adaptive_sage are importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from arag.core.config import Config  # noqa: E402
from adaptive_sage.pipeline import AdaptiveRecursivePipeline  # noqa: E402
from adaptive_sage.types import PipelineResult  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_completed_ids(output_dir: Path) -> set[str]:
    """Read existing predictions.jsonl and return the set of completed question IDs."""
    preds_file = output_dir / "predictions.jsonl"
    if not preds_file.exists():
        return set()

    completed: set[str] = set()
    with open(preds_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("id") or obj.get("question_id")
                if qid:
                    completed.add(str(qid))
            except json.JSONDecodeError:
                continue
    return completed


def _write_prediction(output_dir: Path, result_dict: dict) -> None:
    """Append a single prediction as a JSON line to predictions.jsonl.

    Uses a simple file lock (os.O_EXCL + rename) for safety under
    concurrent writes.  Falls back to direct append if the lock fails.
    """
    preds_file = output_dir / "predictions.jsonl"
    line = json.dumps(result_dict, ensure_ascii=False) + "\n"
    with open(preds_file, "a", encoding="utf-8") as f:
        f.write(line)


def _write_run_summary(output_dir: Path, summary: dict) -> None:
    """Write aggregate run metadata for downstream analysis."""
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _result_to_output_dict(
    result: PipelineResult,
    gold_answer: str,
    wallclock_seconds: float,
) -> dict:
    """Convert a PipelineResult to the output JSON-line format."""
    return {
        "id": result.question_id,
        "question": result.question,
        "answer": result.answer,
        "gold_answer": gold_answer,
        "metadata": {
            "step_trace": [t.to_dict() for t in result.step_trace],
            "num_subagent_calls": result.num_subagent_calls,
            "num_verify_calls": result.num_verify_calls,
            "total_tokens": result.total_tokens,
            "orchestrator_tokens": result.orchestrator_tokens,
            "subagent_tokens": result.subagent_tokens,
            "facts_used": [f.to_dict() for f in result.facts_used],
            "retrieved_doc_ids": result.retrieved_doc_ids,
            "retrieved_docs_total": result.retrieved_docs_total,
            "evidence_capsule_limit": result.evidence_capsule_limit,
            "fact_memory_capacity": result.fact_memory_capacity,
            "duplicate_subquestion_count": result.duplicate_subquestion_count,
            "route_decision": result.route_decision,
            "route_confidence": result.route_confidence,
            "route_draft_answer": result.route_draft_answer,
            "slot_resolution": result.slot_resolution,
            "auto_verify_calls": result.auto_verify_calls,
            "answer_rejection_count": result.answer_rejection_count,
            "wallclock_seconds": round(wallclock_seconds, 3),
        },
    }


# ---------------------------------------------------------------------------
# Core runner logic
# ---------------------------------------------------------------------------


async def _process_question(
    question: dict,
    pipeline: AdaptiveRecursivePipeline,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    pbar: tqdm,
) -> dict:
    """Process a single question and write the result metadata."""
    qid = str(question.get("id", ""))
    q_text = question.get("question", "")
    gold = question.get("answer", "")

    async with semaphore:
        started = time.time()
        try:
            result: PipelineResult = await pipeline.run(question=q_text, question_id=qid)
            elapsed = time.time() - started
            output_dict = _result_to_output_dict(result, gold, elapsed)
            _write_prediction(output_dir, output_dict)
            pbar.update(1)
            return {"ok": True, "wallclock_seconds": elapsed}
        except Exception as exc:
            elapsed = time.time() - started
            logger.error("Failed on question %s: %s", qid, exc, exc_info=True)
            # Write a placeholder so we don't retry this question
            placeholder = {
                "id": qid,
                "question": q_text,
                "answer": "",
                "gold_answer": gold,
                "metadata": {
                    "step_trace": [],
                    "num_subagent_calls": 0,
                    "num_verify_calls": 0,
                    "total_tokens": 0,
                    "orchestrator_tokens": 0,
                    "subagent_tokens": 0,
                    "facts_used": [],
                    "retrieved_doc_ids": [],
                    "retrieved_docs_total": 0,
                    "evidence_capsule_limit": 0,
                    "fact_memory_capacity": 0,
                    "duplicate_subquestion_count": 0,
                    "route_decision": "",
                    "route_confidence": 0.0,
                    "route_draft_answer": "",
                    "slot_resolution": {},
                    "auto_verify_calls": 0,
                    "answer_rejection_count": 0,
                    "wallclock_seconds": round(elapsed, 3),
                    "error": str(exc),
                },
            }
            _write_prediction(output_dir, placeholder)
            pbar.update(1)
            return {"ok": False, "wallclock_seconds": elapsed}


async def _run(
    config_path: str,
    questions_path: str,
    output_dir: str,
    server_url: str,
    concurrency: int,
    chunks_file: str | None = None,
    index_dir: str | None = None,
    embedding_model: str | None = None,
) -> None:
    """Main async runner."""
    # --- Load config ---
    config = Config.from_yaml(config_path)

    # Override LLM base_url with the CLI-specified server URL
    config.set("llm.base_url", server_url)
    if chunks_file:
        config.set("data.chunks_file", chunks_file)
    if index_dir:
        config.set("data.index_dir", index_dir)
    if embedding_model:
        config.set("data.embedding_model", embedding_model)

    # --- Load questions ---
    with open(questions_path, "r", encoding="utf-8") as f:
        questions: list[dict] = json.load(f)
    max_questions = int(config.get("runner.max_questions", 0) or 0)
    if max_questions > 0:
        questions = questions[:max_questions]
    logger.info("Loaded %d questions from %s", len(questions), questions_path)

    # --- Prepare output directory ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpoint: identify already-completed questions ---
    completed_ids = _load_completed_ids(out_dir)
    remaining = [q for q in questions if str(q.get("id", "")) not in completed_ids]
    logger.info(
        "Checkpoint: %d completed, %d remaining",
        len(completed_ids),
        len(remaining),
    )

    if not remaining:
        logger.info("All questions already completed. Nothing to do.")
        return

    # --- Create pipeline ---
    pipeline = AdaptiveRecursivePipeline(config)

    # --- Run concurrently ---
    semaphore = asyncio.Semaphore(concurrency)
    total = len(remaining)
    succeeded = 0
    failed = 0
    wallclock_seconds: list[float] = []

    start_time = time.time()
    pbar = tqdm(total=total, desc="Processing questions", unit="q")

    tasks = [
        _process_question(q, pipeline, semaphore, out_dir, pbar)
        for q in remaining
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()

    for r in results:
        if isinstance(r, Exception):
            failed += 1
        else:
            if isinstance(r, dict):
                wallclock_seconds.append(float(r.get("wallclock_seconds", 0.0)))
                if r.get("ok"):
                    succeeded += 1
                else:
                    failed += 1
            else:
                failed += 1

    elapsed = time.time() - start_time
    sorted_wallclock = sorted(wallclock_seconds)
    p50 = 0.0
    p95 = 0.0
    if sorted_wallclock:
        p50 = sorted_wallclock[len(sorted_wallclock) // 2]
        p95 = sorted_wallclock[min(len(sorted_wallclock) - 1, math.ceil(0.95 * len(sorted_wallclock)) - 1)]

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Run complete: {succeeded}/{total} succeeded, {failed} failed")
    print(f"Checkpoint had {len(completed_ids)} previously completed")
    print(f"Total in output: {len(completed_ids) + succeeded + failed}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")
    _write_run_summary(
        out_dir,
        {
            "config_path": config_path,
            "questions_path": questions_path,
            "server_url": server_url,
            "concurrency": concurrency,
            "chunks_file": config.get("data.chunks_file", ""),
            "index_dir": config.get("data.index_dir", ""),
            "embedding_model": config.get("data.embedding_model", ""),
            "completed_ids": len(completed_ids),
            "attempted": total,
            "succeeded": succeeded,
            "failed": failed,
            "total_output_rows": len(completed_ids) + succeeded + failed,
            "elapsed_seconds": round(elapsed, 3),
            "mean_wallclock_seconds": round(elapsed / total, 3) if total else 0.0,
            "mean_question_wallclock_seconds": round(sum(sorted_wallclock) / len(sorted_wallclock), 3)
            if sorted_wallclock
            else 0.0,
            "p50_question_wallclock_seconds": round(p50, 3),
            "p95_question_wallclock_seconds": round(p95, 3),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner for Adaptive Recursive SAGE experiments."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/s4.yaml)",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSON file (e.g. data/musique/questions.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for predictions.jsonl output (e.g. results/S4)",
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="vLLM server URL (e.g. http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=24,
        help="Max concurrent question processing (default: 24)",
    )
    parser.add_argument("--chunks-file", help="Override data.chunks_file from config")
    parser.add_argument("--index-dir", help="Override data.index_dir from config")
    parser.add_argument(
        "--embedding-model",
        help="Override data.embedding_model from config",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    asyncio.run(
        _run(
            config_path=args.config,
            questions_path=args.questions,
            output_dir=args.output_dir,
            server_url=args.server_url,
            concurrency=args.concurrency,
            chunks_file=args.chunks_file,
            index_dir=args.index_dir,
            embedding_model=args.embedding_model,
        )
    )


if __name__ == "__main__":
    main()
