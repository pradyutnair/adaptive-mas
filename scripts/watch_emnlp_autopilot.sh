#!/bin/bash
set -euo pipefail

cd /local/yzheng/pnair/workspace/05-mas
RUN_SUFFIX="${1:-fair_v4}"
POLL_SECS="${POLL_SECS:-180}"
LOG_FILE="logs/watch_emnlp_autopilot_${RUN_SUFFIX}.log"

mkdir -p logs

expected_done() {
  for dataset in musique_seeded1000 hotpotqa_seed42 2wikimultihop_seed42; do
    for variant in s0_matched a1_matched iter30_think; do
      [ -f "results/${variant}_${dataset}_${RUN_SUFFIX}_eval.json" ] || return 1
    done
  done
  return 0
}

launch_autopilot() {
  nohup bash scripts/emnlp_autopilot_node409.sh "$RUN_SUFFIX" \
    > "logs/emnlp_autopilot_${RUN_SUFFIX}.log" 2>&1 < /dev/null &
  echo "$(date '+%F %T') relaunched autopilot pid=$!" >> "$LOG_FILE"
}

while true; do
  if expected_done; then
    echo "$(date '+%F %T') all expected eval files present; watcher exiting" >> "$LOG_FILE"
    exit 0
  fi

  if ! pgrep -f "scripts/emnlp_autopilot_node409.sh ${RUN_SUFFIX}" >/dev/null; then
    launch_autopilot
  fi

  {
    echo "[$(date '+%F %T')] snapshot"
    ps -ef | grep -E "emnlp_autopilot_node409|run_paper_matrix_node409|scripts/runner.py|vllm.entrypoints.openai.api_server" | grep -v grep || true
    tail -n 20 "logs/emnlp_autopilot_${RUN_SUFFIX}.log" 2>/dev/null || true
    echo "---"
  } >> "$LOG_FILE"

  sleep "$POLL_SECS"
done
