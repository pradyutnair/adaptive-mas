"""Plot a seeded-1000 Pareto chart for selected MuSiQue methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze import _aggregate_rows, _gold_by_id, _load_questions  # noqa: E402


METHOD_SPECS = {
    "S0": ["results/S0_1000_seeded_shard0/predictions.jsonl", "results/S0_1000_seeded_shard1/predictions.jsonl", "results/S0_1000_seeded_shard2/predictions.jsonl"],
    "S0_no_think": ["results/s0_no_think_full1000_c24_combined.jsonl"],
    "A1": ["results/A1_1000_seeded_shard0/predictions.jsonl", "results/A1_1000_seeded_shard1/predictions.jsonl", "results/A1_1000_seeded_shard2/predictions.jsonl"],
    "iter16": ["results/M1_1_iter16_1000_shard0/predictions.jsonl", "results/M1_1_iter16_1000_shard1/predictions.jsonl", "results/M1_1_iter16_1000_shard2/predictions.jsonl"],
    "iter27_no_think": ["results/iter27_full1000_c24_combined.jsonl"],
    "iter27_think": ["results/iter27_think_full1000_c24_combined.jsonl"],
}


def _load_predictions(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot contain vs mean tokens.")
    parser.add_argument(
        "--questions",
        default="data/musique/questions_1000_seedfull_combined.json",
    )
    parser.add_argument(
        "--output",
        default="results/pareto_musique_1000.png",
    )
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    gold_lookup = _gold_by_id(questions)

    plotted: list[tuple[str, float, float]] = []
    for method, paths in METHOD_SPECS.items():
        predictions = _load_predictions(paths)
        if not predictions:
            continue
        row = _aggregate_rows(method, predictions, gold_lookup)
        plotted.append((method, float(row["mean_tokens"]), float(row["contain"])))

    if not plotted:
        raise FileNotFoundError("No prediction files found for Pareto plot.")

    plt.figure(figsize=(8, 5))
    for method, mean_tokens, contain in plotted:
        plt.scatter(mean_tokens, contain, s=70)
        plt.annotate(method, (mean_tokens, contain), xytext=(6, 4), textcoords="offset points")
    plt.xlabel("Mean Tokens")
    plt.ylabel("Contain")
    plt.title("MuSiQue Seeded-1000 Pareto")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main()
