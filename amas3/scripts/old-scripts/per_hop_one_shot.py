"""One-shot per-hop breakdown for selected seeded-1000 methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze import _compute_per_hop_breakdown, _gold_by_id, _hop_by_id, _load_questions  # noqa: E402


METHOD_SPECS = {
    "S0": ["results/S0_1000_seeded_shard0/predictions.jsonl", "results/S0_1000_seeded_shard1/predictions.jsonl", "results/S0_1000_seeded_shard2/predictions.jsonl"],
    "S0_no_think": ["results/s0_no_think_full1000_c24_combined.jsonl"],
    "A1": ["results/A1_1000_seeded_shard0/predictions.jsonl", "results/A1_1000_seeded_shard1/predictions.jsonl", "results/A1_1000_seeded_shard2/predictions.jsonl"],
    "iter16": ["results/M1_1_iter16_1000_shard0/predictions.jsonl", "results/M1_1_iter16_1000_shard1/predictions.jsonl", "results/M1_1_iter16_1000_shard2/predictions.jsonl"],
    "iter27_no_think": ["results/iter27_full1000_c24_combined.jsonl"],
    "iter27_think": ["results/iter27_think_full1000_c24_combined.jsonl"],
}


def _materialize_variant_dir(base: Path, name: str, source_paths: list[str]) -> None:
    variant_dir = base / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    with open(variant_dir / "predictions.jsonl", "w", encoding="utf-8") as out:
        for path_str in source_paths:
            path = Path(path_str)
            if not path.exists():
                continue
            out.write(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Selected per-hop breakdown.")
    parser.add_argument(
        "--questions",
        default="data/musique/questions_1000_seedfull_combined.json",
    )
    parser.add_argument(
        "--temp-results-dir",
        default="results/_selected_methods_tmp",
    )
    parser.add_argument(
        "--output",
        default="results/per_hop_selected_methods.md",
    )
    args = parser.parse_args()

    temp_results_dir = Path(args.temp_results_dir)
    temp_results_dir.mkdir(parents=True, exist_ok=True)
    variants: list[str] = []
    for method, paths in METHOD_SPECS.items():
        _materialize_variant_dir(temp_results_dir, method, paths)
        variants.append(method)

    questions = _load_questions(args.questions)
    gold_lookup = _gold_by_id(questions)
    hop_lookup = _hop_by_id(questions)
    rows = _compute_per_hop_breakdown(temp_results_dir, variants, gold_lookup, hop_lookup)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("| Method | Hop | Count | EM | F1 | Contain | Mean Tokens |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['variant']} | {row['hop']} | {row['count']} | "
                f"{row['norm_em']:.4f} | {row['token_f1']:.4f} | {row['contain']:.4f} | "
                f"{row['mean_tokens']:.1f} |\n"
            )
    print(output)


if __name__ == "__main__":
    main()
