#!/bin/bash
set -euo pipefail

cd /local/yzheng/pnair/workspace/05-mas
RUN_SUFFIX="${1:-fair_v4}"

export HF_HOME=/local/yzheng/.cache/huggingface
export TRANSFORMERS_CACHE=/local/yzheng/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/local/yzheng/.cache/huggingface
export PYTHONPATH=/local/yzheng/pnair/workspace/05-mas/src
export CONCURRENCY="${CONCURRENCY:-24}"
export RETRY_LIMIT="${RETRY_LIMIT:-3}"

mkdir -p logs

echo "[continue] waiting for MuSiQue a1_matched runners"
while pgrep -f "python scripts/runner.py --config configs/a1_matched.yaml --questions data/musique/questions_1000_seedfull_shard" >/dev/null; do
  sleep 60
done

python3 scripts/check_stage_health.py \
  --questions data/musique/questions_1000_seedfull_combined.json \
  --inputs \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard0/predictions.jsonl \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard1/predictions.jsonl \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard2/predictions.jsonl

python3 scripts/combine_shards_and_eval.py \
  --questions data/musique/questions_1000_seedfull_combined.json \
  --output-prefix results/a1_matched_musique_seeded1000_${RUN_SUFFIX} \
  --inputs \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard0/predictions.jsonl \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard1/predictions.jsonl \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard2/predictions.jsonl \
  --run-summaries \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard0/run_summary.json \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard1/run_summary.json \
    results/a1_matched_musique_seeded1000_${RUN_SUFFIX}_shard2/run_summary.json

echo "[continue] MuSiQue a1_matched finalized; starting HotpotQA"
bash scripts/run_paper_matrix_node409.sh hotpotqa data/hotpotqa/questions_1000_seed42 hotpotqa_seed42_${RUN_SUFFIX}

echo "[continue] HotpotQA done; starting 2Wiki"
bash scripts/run_paper_matrix_node409.sh 2wikimultihop data/2wikimultihop/questions_1000_seed42 2wikimultihop_seed42_${RUN_SUFFIX}

echo "[continue] all remaining datasets complete"
