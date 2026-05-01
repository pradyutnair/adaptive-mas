from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

import dspy

from recrag.contracts import normalize_answer
from recrag.grpo.library import ExperienceLibrary
from recrag.grpo.signatures import ExtractGroupOps, OptimizeBatch, SummarizeRollout
from recrag.lm import make_lm
from recrag.pipeline import PipelineConfig, ReactRagPipeline
from recrag.trace import build_readable_trace, build_structured_stats
from recrag.retriever import Retriever


def norm_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def _loads_ops(text: str) -> list[dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


async def compile_grpo(args: argparse.Namespace) -> ExperienceLibrary:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))[: args.n_train]
    random.Random(args.seed).shuffle(questions)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "compile_trace.jsonl"
    lib = ExperienceLibrary.load(args.seed_library) if args.seed_library else ExperienceLibrary()
    reflection_lm = make_lm(args.reflection_lm, temperature=0.0)
    with trace_path.open("w", encoding="utf-8") as trace_f:
        for epoch in range(args.epochs):
            for start in range(0, len(questions), args.batch_size):
                batch = questions[start : start + args.batch_size]
                proposal_texts: list[str] = []
                for q in batch:
                    summaries: list[str] = []
                    for g in range(args.group_size):
                        root_lm = make_lm(args.root_lm, replica_idx=g, temperature=args.temperature)
                        sub_lm = make_lm(args.sub_lm, replica_idx=g, temperature=0.0)
                        pipeline = ReactRagPipeline(root_lm, sub_lm, Retriever(args.retriever_url), PipelineConfig(max_iters=args.max_iters))
                        pred = await pipeline.run(str(q.get("question", "")))
                        gold_ans = str(q.get("answer", ""))
                        score = norm_em(pred.get("answer", ""), gold_ans)
                        readable = build_readable_trace(
                            pred.get("trajectory", {}),
                            pred.get("metadata", {}).get("findings", []),
                            pred.get("answer", ""),
                            gold=gold_ans,
                        )
                        stats = build_structured_stats(
                            pred.get("metadata", {}),
                            pred.get("trajectory", {}),
                            pred.get("answer", ""),
                            gold=gold_ans,
                        )
                        with dspy.context(lm=reflection_lm):
                            summ = dspy.Predict(SummarizeRollout)(
                                question=str(q.get("question", "")),
                                gold_answer=gold_ans,
                                trajectory=readable,
                                score=score,
                                stats=json.dumps(stats, ensure_ascii=False),
                            ).summary
                        summaries.append(str(summ))
                        trace_f.write(json.dumps({"epoch": epoch, "id": q.get("id"), "group": g, "score": score, "readable_trace": readable, "stats": stats}, ensure_ascii=False) + "\n")
                        trace_f.flush()
                    with dspy.context(lm=reflection_lm):
                        ops_raw = dspy.Predict(ExtractGroupOps)(summaries="\n\n".join(summaries), current_library=lib.to_text()).ops_json
                    lib.apply_ops(_loads_ops(str(ops_raw)))
                    proposal_texts.append(lib.to_text())
                with dspy.context(lm=reflection_lm):
                    merged = dspy.Predict(OptimizeBatch)(batch_proposals="\n\n".join(proposal_texts), current_library=lib.to_text()).merged_library
                lib.merge_text(str(merged))
                lib.save_text(args.out_txt)
                lib.save_json(args.out_json)
    return lib


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/musique/opera_matched/questions_50.json")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--root-lm", default="qwen14b-think")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--reflection-lm", default="gpt-4o-mini")
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--seed-library")
    p.add_argument("--out-dir", default="results/grpo_logs")
    p.add_argument("--out-txt", default="compiled/grpo_E.txt")
    p.add_argument("--out-json", default="compiled/grpo_E.json")
    return p


def main() -> None:
    asyncio.run(compile_grpo(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
