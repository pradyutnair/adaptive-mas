#!/bin/bash
# Usage: ./start_vllm.sh <gpu_id> <port>
# Starts vLLM server for Qwen3-8B on specified GPU and port
GPU_ID=${1:?"Usage: $0 <gpu_id> <port>"}
PORT=${2:?"Usage: $0 <gpu_id> <port>"}
source /etc/profile.d/lmod.sh 2>/dev/null
module load cuda12.6/toolkit/12.6 2>/dev/null
source /local/yzheng/pnair/workspace/05-mas/.venv/bin/activate 2>/dev/null || source /var/scratch/yzheng/pnair/venvs/msc-thesis/bin/activate
export HF_HOME=/local/yzheng/.cache/huggingface
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface/hub
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --port $PORT --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser hermes &
SERVER_PID=$!
echo "vLLM server PID: $SERVER_PID on GPU $GPU_ID port $PORT"
for i in $(seq 1 60); do
  curl -sf http://localhost:$PORT/v1/models > /dev/null 2>&1 && echo "Server ready" && exit 0
  sleep 2
done
echo "Server failed to start within 120s"
exit 1
