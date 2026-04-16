#!/bin/bash
set -euo pipefail

cd /local/yzheng/pnair/workspace/05-mas

set +u
source /etc/profile.d/lmod.sh
set -u
module load cuda12.6/toolkit/12.6
source /local/yzheng/pnair/workspace/05-mas/.venv/bin/activate

export HF_HOME=/local/yzheng/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/local/yzheng/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface/hub
export SENTENCE_TRANSFORMERS_HOME=/local/yzheng/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ARAG_EMBEDDING_MODEL=/local/yzheng/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd
export PYTHONPATH=/local/yzheng/pnair/workspace/05-mas/src
export ARAG_CONTEXT_WINDOW=16384
export ARAG_MAX_COMPLETION_TOKENS=1024

WATCH_PIDS=(219988 219989 219990)
CONCURRENCY=24

wait_for_pid_exit() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
}

wait_for_iter16_finish() {
  echo "Waiting for iter16 PIDs to exit: ${WATCH_PIDS[*]}"
  for pid in "${WATCH_PIDS[@]}"; do
    wait_for_pid_exit "$pid"
  done
  echo "iter16 full1000 run finished."
}

run_shard() {
  local config_path="$1"
  local questions_path="$2"
  local output_dir="$3"
  local port="$4"
  CUDA_VISIBLE_DEVICES= python3 scripts/runner.py \
    --config "$config_path" \
    --questions "$questions_path" \
    --output-dir "$output_dir" \
    --server-url "http://localhost:${port}/v1" \
    --concurrency "$CONCURRENCY"
}

run_batch() {
  local variant="$1"
  local config_path="$2"
  echo "Starting ${variant} full1000 batch..."
  run_shard \
    "$config_path" \
    "data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard0.json" \
    "results/${variant}_1000_seeded_shard0" \
    "8001" &
  local pid0=$!
  run_shard \
    "$config_path" \
    "data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard1.json" \
    "results/${variant}_1000_seeded_shard1" \
    "8002" &
  local pid1=$!
  run_shard \
    "$config_path" \
    "data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard2.json" \
    "results/${variant}_1000_seeded_shard2" \
    "8003" &
  local pid2=$!
  wait "$pid0" "$pid1" "$pid2"
  echo "${variant} full1000 batch finished."
}

wait_for_iter16_finish
run_batch "S0" "configs/s0.yaml"
run_batch "A1" "configs/a1.yaml"
