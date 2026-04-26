#!/bin/bash
# Smoke50: run ONLY the new sufficiency controller on 50q (seed42)
# for musique + hotpotqa. Baselines (s0_matched, iter30_think) come from
# the existing 1000q runs in paper_results/latest/, sliced to the same 50 IDs.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVER=http://localhost:8001/v1
EMBEDDING_MODEL=${EMBEDDING_MODEL:-/local/yzheng/pnair/.cache/huggingface/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd}
CONFIG=${CONFIG:-configs/m1_2.sufficiency.yaml}
CONC=${CONC:-16}
TS=$(date +%Y%m%d_%H%M%S)
ROOT=results/smoke50_${TS}
mkdir -p "$ROOT"
echo "$ROOT" > results/.smoke50_latest

run() {
  local ds=$1
  local out="$ROOT/${ds}/sufficiency"
  local q="data/${ds}/questions_smoke50_seed42.json"
  local chunks="data/${ds}/chunks.json"
  local idx="data/${ds}/index_e5_base_v2"
  mkdir -p "$out"
  echo "[$(date +%T)] >>> sufficiency on $ds -> $out"
  python3 scripts/runner.py \
    --config "$CONFIG" \
    --questions "$q" \
    --output-dir "$out" \
    --server-url "$SERVER" \
    --concurrency "$CONC" \
    --chunks-file "$chunks" \
    --index-dir "$idx" \
    --embedding-model "$EMBEDDING_MODEL" \
    2>&1 | tee "$out/run.log" | tail -1
  echo "[$(date +%T)] <<< sufficiency on $ds done"
}

for ds in musique hotpotqa; do
  run "$ds"
done

echo "ALL DONE -> $ROOT"
