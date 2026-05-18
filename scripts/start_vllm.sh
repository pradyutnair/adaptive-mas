#!/bin/bash
# Launch a Qwen3-8B vLLM server with big context + tool-calling enabled.
# Usage: bash scripts/start_vllm.sh <gpu_id> <port>
set -euo pipefail
GPU_ID=${1:?"Usage: $0 <gpu_id> <port>"}
PORT=${2:?"Usage: $0 <gpu_id> <port>"}

ROOT=/local/yzheng/pnair/workspace/adaptive-mas
LOG_DIR="$ROOT/logs"
PID_FILE="$LOG_DIR/vllm-${PORT}.pid"
LOG_FILE="$LOG_DIR/vllm-${PORT}.log"

mkdir -p "$LOG_DIR"

VENV_PY=/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python

export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6
export CUDA_PATH=/cm/shared/apps/cuda12.6/toolkit/12.6
export PATH="/cm/local/apps/cuda-driver/libs/current/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/cm/local/apps/cuda-driver/libs/current/lib64:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/local/yzheng/.cache/huggingface
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface
export VLLM_CACHE_ROOT=/local/yzheng/.cache/vllm
export VLLM_NO_USAGE_STATS=1
export FLASHINFER_WORKSPACE_BASE=/local/yzheng

mkdir -p "$VLLM_CACHE_ROOT"

if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "Server already healthy on port ${PORT}"
  exit 0
fi

nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "$VENV_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen/Qwen3-14B \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 96 \
  --max-num-batched-tokens 24576 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"
echo "vLLM PID $SERVER_PID on GPU $GPU_ID port $PORT (--max-model-len 32768)"

for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server ready on port ${PORT}"
    exit 0
  fi
  sleep 2
done

echo "Server failed to start within 240s; tailing log"
tail -n 60 "$LOG_FILE" || true
exit 1
