#!/usr/bin/env python3
"""Create deterministic sampled question sets and shard them."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic sampled shards.")
    parser.add_argument("--questions", required=True, help="Source questions JSON")
    parser.add_argument("--output-prefix", required=True, help="Output file prefix")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size")
    parser.add_argument("--num-shards", type=int, default=3, help="Number of shards")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before sampling even if sample size >= dataset size",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rng = random.Random(args.seed)
    selected = list(questions)
    if args.shuffle or args.sample_size < len(selected):
        rng.shuffle(selected)
    selected = selected[: min(args.sample_size, len(selected))]

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    n = len(selected)
    shards = max(1, args.num_shards)
    base = n // shards
    remainder = n % shards
    start = 0
    for idx in range(shards):
        size = base + (1 if idx < remainder else 0)
        shard_rows = selected[start : start + size]
        start += size
        with open(
            prefix.parent / f"{prefix.name}_shard{idx}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(shard_rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
