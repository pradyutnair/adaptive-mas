#!/bin/bash
# Run sufficiency on 1000q for musique + hotpotqa + 2wikimultihop, single GPU,
# sequential. Baselines (s0_matched, iter30_think) live in paper_results/latest/.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVER=${SERVER:-http://localhost:8001/v1}
CONC=${CONC:-24}
TS=$(date +%Y%m%d_%H%M%S)
ROOT=results/sufficiency_1000q_${TS}
mkdir -p "$ROOT"
echo "$ROOT" > results/.sufficiency_1000q_latest

run() {
  local ds=$1 qfile=$2 chunks=$3 idx=$4
  local out="$ROOT/${ds}/sufficiency"
  mkdir -p "$out"
  echo "[$(date +%T)] >>> $ds (1000q) -> $out"
  python3 scripts/runner.py \
    --config configs/m1_2.sufficiency.yaml \
    --questions "$qfile" \
    --output-dir "$out" \
    --server-url "$SERVER" \
    --concurrency "$CONC" \
    --chunks-file "$chunks" \
    --index-dir "$idx" \
    --embedding-model intfloat/e5-base-v2 \
    > "$out/run.log" 2>&1
  echo "[$(date +%T)] <<< $ds done ($(wc -l < "$out/predictions.jsonl") preds)"
}

run musique       data/musique/questions_1000_seedfull_combined.json data/musique/chunks.json       data/musique/index_e5_base_v2
run hotpotqa      data/hotpotqa/questions_1000_seed42.json           data/hotpotqa/chunks.json      data/hotpotqa/index
run 2wikimultihop data/2wikimultihop/questions_1000_seed42.json      data/2wikimultihop/chunks.json data/2wikimultihop/index_e5_base_v2

echo "ALL DONE -> $ROOT"
