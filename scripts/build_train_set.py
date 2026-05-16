#!/usr/bin/env python3
"""Build a stratified mixed-pool training set for HERA / TF-GRPO.

Loads:
  - data/cache_train/<dataset>_train_cache_seed42_150.json  (raw questions)
  - data/annotations/annot_train_<dataset>.jsonl            (GPT-4o-mini labels)

Stratifies by (dataset, reasoning_type) so the mixed pool has balanced
coverage of bridge / intersection / temporal / causal / any across all three
training datasets. Falls back to the heuristic profiler when annotations are
missing, so this script works even before annotate_profiles.py has been run.

Output: data/train_stratified.jsonl (or --out), one row per question:
  {"id": ..., "question": ..., "answer": ..., "dataset": ..., "reasoning_type": ...}

This is the file ``scripts/run_hera_train.py`` reads when ``--stratified-pool``
is passed.

Usage:
  uv run python scripts/build_train_set.py --per-dataset 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from amas3.grpo import PROFILE_CLASSES, characterize_query_profile  # noqa: E402

TRAIN_FILES = {
    "hotpotqa": REPO_ROOT / "data" / "cache_train" / "hotpotqa_train_cache_seed42_150.json",
    "2wikimultihop": REPO_ROOT / "data" / "cache_train" / "2wikimultihop_train_cache_seed42_150.json",
    "musique": REPO_ROOT / "data" / "cache_train" / "musique_train_cache_seed42_150.json",
}
ANNOT_DIR = REPO_ROOT / "data" / "annotations"
DEFAULT_OUT = REPO_ROOT / "data" / "train_stratified.jsonl"

log = logging.getLogger("build_train_set")


def load_json_list(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def load_annotations(dataset: str) -> dict[str, str]:
    path = ANNOT_DIR / f"annot_train_{dataset}.jsonl"
    out: dict[str, str] = {}
    if not path.exists():
        log.info("no GPT-4o annotations for %s (looked for %s); will use heuristic", dataset, path)
        return out
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
            rt = str(d.get("reasoning_type", "")).strip().lower()
            if qid and rt in PROFILE_CLASSES:
                out[qid] = rt
    log.info("loaded %d annotations for %s", len(out), dataset)
    return out


def reasoning_type_for(example: dict, annotations: dict[str, str], dataset: str) -> str:
    qid = str(example.get("id", "")).strip()
    if qid and qid in annotations:
        return annotations[qid]
    return characterize_query_profile(example.get("question", ""), dataset, qid=qid or None)


def stratified_sample(
    examples: list[dict],
    annotations: dict[str, str],
    dataset: str,
    per_dataset: int,
    rng: random.Random,
) -> list[dict]:
    """HERA paper §4: bucket by reasoning_type, balance coverage.

    Splits the per-dataset cap evenly across reasoning types that have at
    least one example, then top-up from larger buckets until the cap is met.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        rt = reasoning_type_for(ex, annotations, dataset)
        if rt not in PROFILE_CLASSES:
            rt = "any"
        ex_tagged = dict(ex)
        ex_tagged["dataset"] = dataset
        ex_tagged["reasoning_type"] = rt
        buckets[rt].append(ex_tagged)

    classes_present = [c for c in PROFILE_CLASSES if buckets.get(c)]
    log.info(
        "  %s: %d examples in %d buckets %s",
        dataset, len(examples), len(classes_present),
        {c: len(buckets[c]) for c in classes_present},
    )
    per_bucket = max(1, per_dataset // max(1, len(classes_present)))
    chosen: list[dict] = []
    for cls in classes_present:
        rng.shuffle(buckets[cls])
        chosen.extend(buckets[cls][:per_bucket])
    # Top up from the largest remaining buckets if we are short of the cap.
    if len(chosen) < per_dataset:
        remaining: list[dict] = []
        chosen_ids = {str(e.get("id", "")) for e in chosen}
        for cls in classes_present:
            for ex in buckets[cls][per_bucket:]:
                if str(ex.get("id", "")) not in chosen_ids:
                    remaining.append(ex)
        rng.shuffle(remaining)
        chosen.extend(remaining[: per_dataset - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:per_dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified TF-GRPO training set.")
    parser.add_argument("--per-dataset", type=int, default=50,
                        help="Questions per dataset after stratification (50x3=150 total).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--datasets", default="hotpotqa,2wikimultihop,musique")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    rng = random.Random(args.seed)
    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = set(selected_datasets) - set(TRAIN_FILES)
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")

    all_chosen: list[dict] = []
    summary: dict[str, dict[str, int]] = {}
    for dataset in selected_datasets:
        path = TRAIN_FILES[dataset]
        if not path.exists():
            raise SystemExit(f"missing train file: {path}")
        examples = load_json_list(path)
        annotations = load_annotations(dataset)
        chosen = stratified_sample(examples, annotations, dataset, args.per_dataset, rng)
        all_chosen.extend(chosen)
        counts: dict[str, int] = {}
        for ex in chosen:
            counts[ex["reasoning_type"]] = counts.get(ex["reasoning_type"], 0) + 1
        summary[dataset] = counts

    rng.shuffle(all_chosen)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_chosen:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("Wrote %d stratified examples to %s", len(all_chosen), out_path)
    log.info("Per-dataset reasoning_type counts: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
