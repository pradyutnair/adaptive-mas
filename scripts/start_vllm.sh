#!/bin/bash
set -euo pipefail
# Usage: ./start_vllm.sh <gpu_id> <port>
GPU_ID=${1:?"Usage: $0 <gpu_id> <port>"}
PORT=${2:?"Usage: $0 <gpu_id> <port>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="$ROOT/logs"
PID_FILE="$LOG_DIR/vllm-${PORT}.pid"
LOG_FILE="$LOG_DIR/vllm-${PORT}.log"

mkdir -p "$LOG_DIR"
source "$ROOT/.venv/bin/activate" 2>/dev/null || source /var/scratch/yzheng/pnair/venvs/msc-thesis/bin/activate
PYTHON_BIN="${VLLM_PYTHON_BIN:-$ROOT/.venv/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi
export HF_HOME=/local/yzheng/.cache/huggingface
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/local/yzheng/.cache/huggingface
export XDG_CACHE_HOME=/local/yzheng/.cache
export VLLM_CACHE_ROOT=/local/yzheng/.cache/vllm
export FLASHINFER_WORKSPACE_BASE=/local/yzheng
export VLLM_NO_USAGE_STATS=1
export CUDA_DRIVER_ROOT=/cm/local/apps/cuda-driver/libs/current
export PATH="$CUDA_DRIVER_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_DRIVER_ROOT/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$VLLM_CACHE_ROOT"
cd "$LOG_DIR"

if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "Server already healthy on port ${PORT}"
  exit 0
fi

MODEL_NAME="${VLLM_MODEL:-Qwen/Qwen3-8B}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-16384}"

nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --enable-prefix-caching \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"
echo "vLLM server PID: $SERVER_PID on GPU $GPU_ID port $PORT"
for i in $(seq 1 90); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server ready on port ${PORT}"
    exit 0
  fi
  sleep 2
done
echo "Server failed to start within 180s; tailing log"
tail -n 40 "$LOG_FILE" || true
exit 1
