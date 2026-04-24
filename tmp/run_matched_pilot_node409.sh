#!/bin/bash
set -euo pipefail

MODE="${1:?mode required: adaptive|opera}"
NAME="${2:?name required}"
SPEC="${3:?config path for adaptive, model name for opera}"
QUESTION_PREFIX="${4:?question prefix required, e.g. data/musique/questions_pilot200_seed42}"
RUN_TAG="${5:-pilot}"

cd /local/yzheng/pnair/workspace/adaptive-mas
source .venv/bin/activate
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

export HF_HOME=/local/yzheng/.cache/huggingface
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/local/yzheng/.cache/huggingface
export PYTHONPATH=/local/yzheng/pnair/workspace/adaptive-mas/src

CONCURRENCY="${CONCURRENCY:-16}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat/e5-base-v2}"
CHUNKS_FILE="${CHUNKS_FILE:-data/musique/chunks.json}"
INDEX_DIR="${INDEX_DIR:-data/musique/index_e5_base_v2}"
SUFFICIENCY_BASELINE="${SUFFICIENCY_BASELINE:-results/sufficiency_1000q_20260418_215804/musique/sufficiency/predictions.jsonl}"
ITER55_BASELINE="${ITER55_BASELINE:-results/audit/iter55_predictions.jsonl}"
OPERA_BASELINE="${OPERA_BASELINE:-/local/yzheng/pnair/workspace/results/05-mas-results/external_baselines/opera_full/musique/opera_musique_1000_combined.jsonl}"
OPERA_REPO="${OPERA_REPO:-/local/yzheng/pnair/workspace/baseline_repos/OPERA}"
OPERA_RETRIEVER_URL="${OPERA_RETRIEVER_URL:-http://127.0.0.1:9102}"
export ARAG_CONTEXT_WINDOW="${ARAG_CONTEXT_WINDOW:-12288}"
export ARAG_MAX_COMPLETION_TOKENS="${ARAG_MAX_COMPLETION_TOKENS:-768}"

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

mkdir -p logs
rm -rf \
  "results/${NAME}_${RUN_TAG}_shard0" \
  "results/${NAME}_${RUN_TAG}_shard1" \
  "results/${NAME}_${RUN_TAG}_shard2"
rm -f \
  "results/${NAME}_${RUN_TAG}_combined.jsonl" \
  "results/${NAME}_${RUN_TAG}_matched_compare.json" \
  "logs/${NAME}_${RUN_TAG}_shard0.log" \
  "logs/${NAME}_${RUN_TAG}_shard1.log" \
  "logs/${NAME}_${RUN_TAG}_shard2.log"

ensure_servers

for IDX in 0 1 2; do
  PORT=$((8001 + IDX))
  QUESTIONS="${QUESTION_PREFIX}_shard${IDX}.json"
  OUTDIR="results/${NAME}_${RUN_TAG}_shard${IDX}"
  mkdir -p "$OUTDIR"
  if [ "$MODE" = "adaptive" ]; then
    nohup env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" scripts/runner.py \
      --config "$SPEC" \
      --questions "$QUESTIONS" \
      --output-dir "$OUTDIR" \
      --server-url "http://localhost:${PORT}/v1" \
      --concurrency "$CONCURRENCY" \
      --chunks-file "$CHUNKS_FILE" \
      --index-dir "$INDEX_DIR" \
      --embedding-model "$EMBEDDING_MODEL" \
      > "logs/${NAME}_${RUN_TAG}_shard${IDX}.log" 2>&1 &
  elif [ "$MODE" = "opera" ]; then
    nohup env CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" "${OPERA_REPO}/run_opera_05mas.py" \
      --questions-file "$QUESTIONS" \
      --output-file "$OUTDIR/predictions.jsonl" \
      --base-url "http://localhost:${PORT}/v1" \
      --model-name "$SPEC" \
      --retriever-url "$OPERA_RETRIEVER_URL" \
      --thinking \
      > "logs/${NAME}_${RUN_TAG}_shard${IDX}.log" 2>&1 &
  else
    echo "Unknown mode: $MODE" >&2
    exit 1
  fi
done
wait

cat \
  "results/${NAME}_${RUN_TAG}_shard0/predictions.jsonl" \
  "results/${NAME}_${RUN_TAG}_shard1/predictions.jsonl" \
  "results/${NAME}_${RUN_TAG}_shard2/predictions.jsonl" \
  > "results/${NAME}_${RUN_TAG}_combined.jsonl"

COMPARE_ARGS=(
  --questions "${QUESTION_PREFIX}.json"
  --run "sufficiency=${SUFFICIENCY_BASELINE}"
  --run "iter55=${ITER55_BASELINE}"
  --run "${NAME}=results/${NAME}_${RUN_TAG}_combined.jsonl"
  --output "results/${NAME}_${RUN_TAG}_matched_compare.json"
)

if [ -f "$OPERA_BASELINE" ] && [ "$NAME" != "opera_pilot200_matched" ]; then
  COMPARE_ARGS+=(--run "opera=${OPERA_BASELINE}")
fi

.venv/bin/python scripts/compare_subset_runs.py "${COMPARE_ARGS[@]}"
