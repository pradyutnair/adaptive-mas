#!/bin/bash
set -euo pipefail

cd /local/yzheng/pnair/workspace/05-mas
export SLURM_NODELIST=node409
source /local/yzheng/pnair/workspace/05-mas/.venv/bin/activate
export HF_HOME=/local/yzheng/.cache/huggingface
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/local/yzheng/.cache/huggingface
export PYTHONPATH=/local/yzheng/pnair/workspace/05-mas/src

RUN_SUFFIX="${1:-fair_v3}"
CONCURRENCY="${CONCURRENCY:-24}"
RETRY_LIMIT="${RETRY_LIMIT:-3}"
export CONCURRENCY RETRY_LIMIT

mkdir -p logs

cleanup_old_jobs() {
  pkill -f "scripts/runner.py --config configs/s0_matched.yaml" || true
  pkill -f "scripts/runner.py --config configs/a1_matched.yaml" || true
  pkill -f "scripts/runner.py --config configs/m1_1.iter30_think.yaml" || true
  pkill -f "run_paper_matrix_node409.sh" || true
  pkill -f "queue_emnlp_followups.sh" || true
}

run_dataset() {
  local DATASET="$1"
  local PREFIX="$2"
  local TAG="$3"
  echo "=== DATASET ${DATASET} ${TAG} ==="
  bash scripts/run_paper_matrix_node409.sh "$DATASET" "$PREFIX" "$TAG"
}

cleanup_old_jobs
for PORT in 8001 8002 8003; do
  bash scripts/stop_vllm.sh "$PORT" >/dev/null 2>&1 || true
done

run_dataset musique data/musique/questions_1000_seedfull "musique_seeded1000_${RUN_SUFFIX}"
run_dataset hotpotqa data/hotpotqa/questions_1000_seed42 "hotpotqa_seed42_${RUN_SUFFIX}"
run_dataset 2wikimultihop data/2wikimultihop/questions_1000_seed42 "2wikimultihop_seed42_${RUN_SUFFIX}"
