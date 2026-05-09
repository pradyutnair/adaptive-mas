#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="node409:/local/yzheng/pnair/workspace/05-mas/results"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)/paper_results/latest"

mkdir -p "$LOCAL_ROOT"

copy_one() {
  local remote_name="$1"
  local local_dir="$2"
  local local_name="$3"
  mkdir -p "$LOCAL_ROOT/$local_dir"
  scp "$REMOTE_ROOT/$remote_name" "$LOCAL_ROOT/$local_dir/$local_name"
}

# MuSiQue
copy_one "s0_matched_musique_seeded1000_fair_v4_combined.jsonl" "musique/s0_matched" "predictions.jsonl"
copy_one "s0_matched_musique_seeded1000_fair_v4_eval.json" "musique/s0_matched" "eval.json"
copy_one "s0_matched_musique_seeded1000_fair_v4_run_summary.json" "musique/s0_matched" "run_summary.json"

copy_one "a1_matched_musique_seeded1000_fair_v4_combined.jsonl" "musique/a1_matched" "predictions.jsonl"
copy_one "a1_matched_musique_seeded1000_fair_v4_eval.json" "musique/a1_matched" "eval.json"
copy_one "a1_matched_musique_seeded1000_fair_v4_run_summary.json" "musique/a1_matched" "run_summary.json"

copy_one "iter30_think_full1000_combined.jsonl" "musique/iter30_think" "predictions.jsonl"
copy_one "iter30_think_full1000_eval.json" "musique/iter30_think" "eval.json"

copy_one "iter27_think_full1000_c24_combined.jsonl" "musique/iter27_think" "predictions.jsonl"
copy_one "iter27_think_full1000_c24_eval.json" "musique/iter27_think" "eval.json"

copy_one "iter27_full1000_c24_combined.jsonl" "musique/iter27_no_think" "predictions.jsonl"
copy_one "iter27_full1000_c24_eval.json" "musique/iter27_no_think" "eval.json"

copy_one "s0_no_think_full1000_c24_combined.jsonl" "musique/s0_no_think" "predictions.jsonl"
copy_one "s0_no_think_full1000_c24_eval.json" "musique/s0_no_think" "eval.json"

# HotpotQA
copy_one "s0_matched_hotpotqa_seed42_fair_v4_combined.jsonl" "hotpotqa/s0_matched" "predictions.jsonl"
copy_one "s0_matched_hotpotqa_seed42_fair_v4_eval.json" "hotpotqa/s0_matched" "eval.json"
copy_one "s0_matched_hotpotqa_seed42_fair_v4_run_summary.json" "hotpotqa/s0_matched" "run_summary.json"

copy_one "a1_matched_hotpotqa_seed42_fair_v4_combined.jsonl" "hotpotqa/a1_matched" "predictions.jsonl"
copy_one "a1_matched_hotpotqa_seed42_fair_v4_eval.json" "hotpotqa/a1_matched" "eval.json"
copy_one "a1_matched_hotpotqa_seed42_fair_v4_run_summary.json" "hotpotqa/a1_matched" "run_summary.json"

copy_one "iter30_think_hotpotqa_seed42_fair_v4_combined.jsonl" "hotpotqa/iter30_think" "predictions.jsonl"
copy_one "iter30_think_hotpotqa_seed42_fair_v4_eval.json" "hotpotqa/iter30_think" "eval.json"
copy_one "iter30_think_hotpotqa_seed42_fair_v4_run_summary.json" "hotpotqa/iter30_think" "run_summary.json"

# 2WikiMultiHopQA
copy_one "s0_matched_2wikimultihop_seed42_fair_v4_combined.jsonl" "2wikimultihop/s0_matched" "predictions.jsonl"
copy_one "s0_matched_2wikimultihop_seed42_fair_v4_eval.json" "2wikimultihop/s0_matched" "eval.json"
copy_one "s0_matched_2wikimultihop_seed42_fair_v4_run_summary.json" "2wikimultihop/s0_matched" "run_summary.json"

copy_one "a1_matched_2wikimultihop_seed42_fair_v4_combined.jsonl" "2wikimultihop/a1_matched" "predictions.jsonl"
copy_one "a1_matched_2wikimultihop_seed42_fair_v4_eval.json" "2wikimultihop/a1_matched" "eval.json"
copy_one "a1_matched_2wikimultihop_seed42_fair_v4_run_summary.json" "2wikimultihop/a1_matched" "run_summary.json"

copy_one "iter30_think_2wikimultihop_seed42_fair_v4_combined.jsonl" "2wikimultihop/iter30_think" "predictions.jsonl"
copy_one "iter30_think_2wikimultihop_seed42_fair_v4_eval.json" "2wikimultihop/iter30_think" "eval.json"
copy_one "iter30_think_2wikimultihop_seed42_fair_v4_run_summary.json" "2wikimultihop/iter30_think" "run_summary.json"

echo "Synced latest completed results into $LOCAL_ROOT"
