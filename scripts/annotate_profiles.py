#!/usr/bin/env python3
"""Annotate question profiles with GPT-4o-mini.

Mirrors HERA's `scripts/annotate_profiles.py` but uses our paper's 5-class
vocabulary: ``bridge``, ``intersection``, ``temporal``, ``causal``, ``any``.

Reads train and test JSON files, calls GPT-4o-mini once per question, writes
one JSONL row per question with the canonical reasoning_type. Annotated
files are consumed by ``characterize_query_profile`` (priority over the
keyword heuristic) and by ``scripts/build_train_set.py``.

Files are cached so re-runs skip already-annotated questions.

Usage:
  uv run python scripts/annotate_profiles.py --target train
  uv run python scripts/annotate_profiles.py --target test
  uv run python scripts/annotate_profiles.py --target all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import dspy

REASONING_TYPES = ("bridge", "intersection", "temporal", "causal", "any")

TRAIN_FILES = {
    "hotpotqa": REPO_ROOT / "data" / "cache_train" / "hotpotqa_train_cache_seed42_150.json",
    "2wikimultihop": REPO_ROOT / "data" / "cache_train" / "2wikimultihop_train_cache_seed42_150.json",
    "musique": REPO_ROOT / "data" / "cache_train" / "musique_train_cache_seed42_150.json",
}

TEST_FILES = {
    "hotpotqa": REPO_ROOT / "data" / "hotpotqa" / "questions_1000_seed42.json",
    "2wikimultihop": REPO_ROOT / "data" / "2wikimultihop" / "questions_1000_seed42.json",
    "musique": REPO_ROOT / "data" / "musique" / "questions_1000_seedfull_combined.json",
    "bamboogle": REPO_ROOT / "data" / "bamboogle" / "questions_125.json",
}

OUT_DIR = REPO_ROOT / "data" / "annotations"

SYSTEM = (
    "You annotate multi-hop QA queries with a single reasoning_type from a closed "
    "vocabulary. Always respond with valid JSON; do not include any prose."
)

USER_TEMPLATE = """Classify this question into exactly one reasoning_type:

- bridge: sequential dependency through an intermediate entity (e.g., "Which university did the author of The Old Man and the Sea attend?")
- intersection: requires combining or comparing answers across two or more constraints (e.g., "Who was born earlier, Marie Curie or Albert Einstein?", "Which scientists won a Nobel Prize and served as a university president?")
- temporal: reasoning over dates, time ordering, or temporal containment (e.g., "What year did the Berlin Wall fall?", "Who was president when the Manhattan Project began?")
- causal: explaining cause-effect chains across events (e.g., "Why did the 2008 financial crisis lead to increased banking regulation?")
- any: factoid lookups or questions that do not fit the above (e.g., "What is the capital of France?")

Question: {question}

Respond with strict JSON:
{{"reasoning_type": "<bridge|intersection|temporal|causal|any>"}}
"""

log = logging.getLogger("annotate")


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(d.get("id", "")).strip()
            if qid:
                out[qid] = d
    return out


def parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


async def annotate_one(lm: dspy.LM, qid: str, question: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        prompt = USER_TEMPLATE.format(question=question)
        try:
            response = await asyncio.to_thread(lm, prompt)
            raw = response[0] if isinstance(response, list) else str(response)
            obj = parse_json_object(raw)
            rt = str(obj.get("reasoning_type", "")).strip().lower()
        except Exception as exc:
            log.warning("annotate failed id=%s: %s", qid, str(exc)[:120])
            rt = ""
        if rt not in REASONING_TYPES:
            rt = "any"
        return {"id": qid, "question": question, "reasoning_type": rt}


async def annotate_file(
    lm: dspy.LM,
    examples: list[dict],
    out_path: Path,
    concurrency: int,
    progress_every: int = 50,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_existing(out_path)
    todo = [e for e in examples if str(e.get("id", "")).strip() not in cache]
    log.info("Annotating %d new examples (cached %d) -> %s", len(todo), len(cache), out_path)
    if not todo:
        return

    sem = asyncio.Semaphore(concurrency)
    tasks = [annotate_one(lm, str(ex["id"]), str(ex["question"]), sem) for ex in todo]
    with open(out_path, "a", encoding="utf-8") as f:
        n = 0
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % progress_every == 0:
                log.info("  %d / %d annotated", n, len(todo))


async def run_target(target: str, model: str, concurrency: int, train_limit: int) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set; required for GPT-4o-mini annotation.")
    lm = dspy.LM(
        model=f"openai/{model}",
        max_tokens=64,
        temperature=0.0,
        api_key=api_key,
    )

    if target in ("train", "all"):
        for dataset, path in TRAIN_FILES.items():
            if not path.exists():
                log.warning("missing train file: %s", path)
                continue
            data = load_json(path)
            if train_limit > 0:
                data = data[:train_limit]
            await annotate_file(lm, data, OUT_DIR / f"annot_train_{dataset}.jsonl", concurrency)
    if target in ("test", "all"):
        for dataset, path in TEST_FILES.items():
            if not path.exists():
                log.warning("missing test file: %s", path)
                continue
            data = load_json(path)
            await annotate_file(lm, data, OUT_DIR / f"annot_test_{dataset}.jsonl", concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-4o-mini reasoning-type annotation.")
    parser.add_argument("--target", choices=("train", "test", "all"), default="all")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--train-limit", type=int, default=0,
                        help="Annotate only the first N train examples per dataset; 0 = all.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(run_target(args.target, args.model, args.concurrency, args.train_limit))
    log.info("Done. Annotations at %s", OUT_DIR)


if __name__ == "__main__":
    main()
