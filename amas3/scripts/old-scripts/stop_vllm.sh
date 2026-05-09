#!/bin/bash
set -euo pipefail
PORT=${1:?"Usage: $0 <port>"}
ROOT=/local/yzheng/pnair/workspace/05-mas
PID_FILE="$ROOT/logs/vllm-${PORT}.pid"
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE" || true)
  if [ -n "${PID:-}" ]; then
    kill "$PID" 2>/dev/null || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi
lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
echo "Stopped server on port $PORT"
