from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.lm import make_lm
from recrag.pipeline import PipelineConfig, ReactRagPipeline
from recrag.retriever import Retriever


def load_questions(path: str, n: int | None) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if n is not None:
        rows = rows[:n]
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    toks = [int(r.get("metadata", {}).get("total_tokens", 0)) for r in rows]
    hops = [int(r.get("metadata", {}).get("hops", 0)) for r in rows]
    retries = [int(r.get("metadata", {}).get("retries", 0)) for r in rows]
    return {
        "rows": len(rows),
        "mean_tokens": sum(toks) / len(toks) if toks else 0.0,
        "mean_hops": sum(hops) / len(hops) if hops else 0.0,
        "mean_retries": sum(retries) / len(retries) if retries else 0.0,
        "answered": sum(1 for r in rows if str(r.get("answer", "")).strip()),
    }


async def run_one(q: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any]:
    root_lm = make_lm(args.root_lm, replica_idx=idx)
    sub_lm = make_lm(args.sub_lm, replica_idx=idx, max_tokens=args.sub_max_tokens)
    pipeline = ReactRagPipeline(
        root_lm,
        sub_lm,
        Retriever(args.retriever_url),
        PipelineConfig(max_iters=args.max_iters, experience_library=args.experience_library, citation_gate=not args.no_citation_gate),
    )
    pred = await pipeline.run(str(q.get("question", "")))
    return {
        "id": str(q.get("id", idx)),
        "question": str(q.get("question", "")),
        "answer": pred.get("answer", ""),
        "predicted_answer": pred.get("answer", ""),
        "metadata": pred.get("metadata", {}),
        "trajectory": pred.get("trajectory", {}),
    }


async def run_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {pred_path}; pass --overwrite")
    questions = load_questions(args.questions, args.n)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict[str, Any] | None] = [None] * len(questions)

    async def guarded(i: int, q: dict[str, Any]) -> None:
        async with sem:
            rows[i] = await run_one(q, args, i)
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
            print(json.dumps({"done": i + 1, "id": rows[i]["id"], "answer": rows[i]["answer"], "tokens": rows[i]["metadata"].get("total_tokens", 0)}, ensure_ascii=False), flush=True)

    if pred_path.exists():
        pred_path.unlink()
    await asyncio.gather(*[guarded(i, q) for i, q in enumerate(questions)])
    final_rows = [r for r in rows if r is not None]
    summary = summarize(final_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "RUN_NOTE.md").write_text("# Run Note\n\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root-lm", default="qwen14b-think")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--questions", required=True)
    p.add_argument("--n", type=int)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--experience-library")
    p.add_argument("--gepa-program")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--no-citation-gate", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    asyncio.run(run_all(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
