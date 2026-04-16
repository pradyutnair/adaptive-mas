#!/bin/bash
set -euo pipefail

cd /local/yzheng/pnair/workspace/05-mas
source /etc/profile.d/lmod.sh && module load cuda12.6/toolkit/12.6
source /local/yzheng/pnair/workspace/05-mas/.venv/bin/activate 2>/dev/null || source /var/scratch/yzheng/pnair/venvs/msc-thesis/bin/activate
export HF_HOME=/local/yzheng/.cache/huggingface
export PYTHONPATH=/local/yzheng/pnair/workspace/05-mas/src

QUESTIONS=data/musique/questions.json
CONCURRENCY=24

run_variant() {
  local config_path="$1"
  local variant_name="$2"
  local port="$3"
  python3 scripts/runner.py \
    --config "$config_path" \
    --questions "$QUESTIONS" \
    --output-dir "results/$variant_name" \
    --server-url "http://localhost:${port}/v1" \
    --concurrency "$CONCURRENCY"
}

run_wave() {
  (
    for item in "$1" "$2" "$3"; do
      [ -z "$item" ] && continue
      IFS=':' read -r config_path variant_name port <<<"$item"
      run_variant "$config_path" "$variant_name" "$port"
    done
  ) &
  local pid1=$!

  (
    for item in "$4" "$5" "$6"; do
      [ -z "$item" ] && continue
      IFS=':' read -r config_path variant_name port <<<"$item"
      run_variant "$config_path" "$variant_name" "$port"
    done
  ) &
  local pid2=$!

  (
    for item in "$7" "$8" "$9"; do
      [ -z "$item" ] && continue
      IFS=':' read -r config_path variant_name port <<<"$item"
      run_variant "$config_path" "$variant_name" "$port"
    done
  ) &
  local pid3=$!

  wait "$pid1" "$pid2" "$pid3"
}

echo 'Running pilot wave...'
run_wave \
  "configs/p0.yaml:P0:8001" \
  "" \
  "" \
  "configs/p1.yaml:P1:8002" \
  "" \
  "" \
  "" \
  "" \
  ""

echo 'Running scaling wave...'
run_wave \
  "configs/s0.yaml:S0:8001" \
  "configs/s3.yaml:S3:8001" \
  "" \
  "configs/s1.yaml:S1:8002" \
  "configs/s4.yaml:S4:8002" \
  "" \
  "configs/s2.yaml:S2:8003" \
  "configs/m1.yaml:M1:8003" \
  ""

echo 'Running ablation wave...'
run_wave \
  "configs/a1.yaml:A1:8001" \
  "configs/a4.yaml:A4:8001" \
  "configs/a7_small.yaml:A7S:8001" \
  "configs/a2.yaml:A2:8002" \
  "configs/a5.yaml:A5:8002" \
  "configs/a8.yaml:A8:8002" \
  "configs/a3.yaml:A3:8003" \
  "configs/a6.yaml:A6:8003" \
  "configs/a7_large.yaml:A7L:8003"

echo 'Running ablation tradeoff wave...'
run_wave \
  "configs/a7.yaml:A7:8001" \
  "" \
  "" \
  "configs/a8_4.yaml:A8L:8001" \
  "" \
  "" \
  "" \
  "" \
  ""

echo 'Running eval...'
for dir in results/*; do
  if [ -d "$dir" ] && [ -f "$dir/predictions.jsonl" ]; then
    variant=$(basename "$dir")
    python3 scripts/eval_offline.py \
      --predictions "$dir/predictions.jsonl" \
      --questions "$QUESTIONS" \
      --output "$dir/predictions_eval_summary.json"
  fi
done

if [ -f "results/S0/predictions.jsonl" ] && [ -f "results/M1/predictions.jsonl" ]; then
  echo 'Running oracle routing...'
  python3 scripts/oracle_routing.py \
    --config configs/d1.yaml \
    --results-dir results \
    --questions "$QUESTIONS"
fi

echo 'Running analysis...'
python3 scripts/analyze.py --results-dir results --questions "$QUESTIONS"
echo 'DONE'
