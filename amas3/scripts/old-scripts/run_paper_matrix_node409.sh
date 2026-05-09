#!/bin/bash
set -euo pipefail

cd /local/yzheng/pnair/workspace/05-mas
source /local/yzheng/pnair/workspace/05-mas/.venv/bin/activate
export HF_HOME=/local/yzheng/.cache/huggingface
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/local/yzheng/.cache/huggingface
export PYTHONPATH=/local/yzheng/pnair/workspace/05-mas/src

DATASET="${1:?dataset key required: musique|hotpotqa|2wikimultihop}"
QUESTION_PREFIX="${2:?question prefix required, e.g. data/hotpotqa/questions_1000_seed42}"
RUN_TAG="${3:-${DATASET}_paper}"
CONCURRENCY="${CONCURRENCY:-24}"

case "$DATASET" in
  musique)
    CHUNKS_FILE="data/musique/chunks.json"
    INDEX_DIR="data/musique/index_e5_base_v2"
    COMBINED_QUESTIONS="data/musique/questions_1000_seedfull_combined.json"
    ;;
  hotpotqa)
    CHUNKS_FILE="data/hotpotqa/chunks.json"
    INDEX_DIR="data/hotpotqa/index"
    COMBINED_QUESTIONS="${QUESTION_PREFIX}.json"
    ;;
  2wikimultihop)
    CHUNKS_FILE="data/2wikimultihop/chunks.json"
    INDEX_DIR="data/2wikimultihop/index_e5_base_v2"
    COMBINED_QUESTIONS="${QUESTION_PREFIX}.json"
    ;;
  *)
    echo "Unknown dataset: $DATASET" >&2
    exit 1
    ;;
esac

EMBEDDING_MODEL="intfloat/e5-base-v2"
RETRY_LIMIT="${RETRY_LIMIT:-3}"

ensure_servers() {
  for spec in "0 8001" "1 8002" "2 8003"; do
    set -- $spec
    local GPU="$1"
    local PORT="$2"
    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      bash scripts/stop_vllm.sh "$PORT" >/dev/null 2>&1 || true
      bash scripts/start_vllm.sh "$GPU" "$PORT"
    fi
  done
}

clear_variant_outputs() {
  local NAME="$1"
  rm -rf \
    "results/${NAME}_${RUN_TAG}_shard0" \
    "results/${NAME}_${RUN_TAG}_shard1" \
    "results/${NAME}_${RUN_TAG}_shard2"
  rm -f \
    "results/${NAME}_${RUN_TAG}_combined.jsonl" \
    "results/${NAME}_${RUN_TAG}_eval.json" \
    "results/${NAME}_${RUN_TAG}_run_summary.json"
  rm -f "logs/${NAME}_${RUN_TAG}_shard0.log" "logs/${NAME}_${RUN_TAG}_shard1.log" "logs/${NAME}_${RUN_TAG}_shard2.log"
}

stage_has_errors() {
  local NAME="$1"
  python scripts/check_stage_health.py \
    --questions "$COMBINED_QUESTIONS" \
    --inputs \
      "results/${NAME}_${RUN_TAG}_shard0/predictions.jsonl" \
      "results/${NAME}_${RUN_TAG}_shard1/predictions.jsonl" \
      "results/${NAME}_${RUN_TAG}_shard2/predictions.jsonl"
}

run_wave() {
  local CONFIG="$1"
  local NAME="$2"
  local ATTEMPT=1
  while [ "$ATTEMPT" -le "$RETRY_LIMIT" ]; do
    echo "[${DATASET}] Running ${NAME}, attempt ${ATTEMPT}/${RETRY_LIMIT}"
    clear_variant_outputs "$NAME"
    ensure_servers
    for IDX in 0 1 2; do
      local PORT=$((8001 + IDX))
      local QUESTIONS="${QUESTION_PREFIX}_shard${IDX}.json"
      local OUTDIR="results/${NAME}_${RUN_TAG}_shard${IDX}"
      nohup env CUDA_VISIBLE_DEVICES= python scripts/runner.py \
        --config "$CONFIG" \
        --questions "$QUESTIONS" \
        --output-dir "$OUTDIR" \
        --server-url "http://localhost:${PORT}/v1" \
        --concurrency "$CONCURRENCY" \
        --chunks-file "$CHUNKS_FILE" \
        --index-dir "$INDEX_DIR" \
        --embedding-model "$EMBEDDING_MODEL" \
        > "logs/${NAME}_${RUN_TAG}_shard${IDX}.log" 2>&1 &
    done
    wait
    if stage_has_errors "$NAME"; then
      echo "[${DATASET}] ${NAME} completed cleanly"
      break
    fi
    echo "[${DATASET}] ${NAME} failed health checks; restarting servers"
    for PORT in 8001 8002 8003; do
      bash scripts/stop_vllm.sh "$PORT" >/dev/null 2>&1 || true
    done
    ATTEMPT=$((ATTEMPT + 1))
  done
  if [ "$ATTEMPT" -gt "$RETRY_LIMIT" ]; then
    echo "[${DATASET}] ${NAME} failed after ${RETRY_LIMIT} attempts" >&2
    exit 1
  fi
  python scripts/combine_shards_and_eval.py \
    --questions "$COMBINED_QUESTIONS" \
    --output-prefix "results/${NAME}_${RUN_TAG}" \
    --inputs \
      "results/${NAME}_${RUN_TAG}_shard0/predictions.jsonl" \
      "results/${NAME}_${RUN_TAG}_shard1/predictions.jsonl" \
      "results/${NAME}_${RUN_TAG}_shard2/predictions.jsonl" \
    --run-summaries \
      "results/${NAME}_${RUN_TAG}_shard0/run_summary.json" \
      "results/${NAME}_${RUN_TAG}_shard1/run_summary.json" \
      "results/${NAME}_${RUN_TAG}_shard2/run_summary.json"
}

mkdir -p logs
run_wave "configs/s0_matched.yaml" "s0_matched"
run_wave "configs/a1_matched.yaml" "a1_matched"
run_wave "configs/m1_1.iter30_think.yaml" "iter30_think"
