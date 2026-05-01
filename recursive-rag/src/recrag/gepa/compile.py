from __future__ import annotations

import argparse
import json
from pathlib import Path

import dspy

from recrag.gepa.metric import metric
from recrag.lm import make_lm
from recrag.pipeline import PipelineConfig, ReactRagPipeline
from recrag.program import ReactProgram
from recrag.retriever import Retriever


def compile_gepa(args: argparse.Namespace) -> None:
    train = json.loads(Path(args.questions).read_text(encoding="utf-8"))[: args.n_train]
    root_lm = make_lm(args.root_lm)
    sub_lm = make_lm(args.sub_lm)
    pipeline = ReactRagPipeline(root_lm, sub_lm, Retriever(args.retriever_url), PipelineConfig(max_iters=args.max_iters, experience_library=args.experience_library))
    program = ReactProgram(pipeline)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(dspy, "GEPA"):
        out.write_text(json.dumps({"status": "skipped", "reason": "dspy.GEPA unavailable"}, indent=2), encoding="utf-8")
        return
    examples = [dspy.Example(question=q.get("question", ""), answer=q.get("answer", "")).with_inputs("question") for q in train]
    optimizer = dspy.GEPA(metric=metric, reflection_lm=make_lm(args.reflection_lm), auto=args.auto)
    compiled = optimizer.compile(program, trainset=examples)
    try:
        compiled.save(str(out))
    except Exception:
        out.write_text(json.dumps({"status": "compiled", "note": "compiled object has no save()"}, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/musique/opera_matched/questions_50.json")
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--root-lm", default="qwen14b-think")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--reflection-lm", default="gpt-4o-mini")
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--experience-library", default="compiled/grpo_E.txt")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    p.add_argument("--out", default="compiled/gepa_program.json")
    return p


def main() -> None:
    compile_gepa(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
