#!/bin/bash
# Sequentially run AMAS variant 2 on MuSiQue, HotpotQA, 2Wiki, Bamboogle.
set -e
cd /local/yzheng/pnair/workspace/adaptive-mas
set -a && . /local/yzheng/pnair/.env && set +a
export DSPY_CACHEDIR=/local/yzheng/pnair/.dspy_cache
export PYTHONPATH=src

declare -A TASKS=(
  ['musique_1000']='data/musique/questions.json'
  ['hotpotqa_1000']='data/hotpotqa/questions_1000_seed42.json'
  ['2wiki_1000']='data/2wikimultihop/questions_1000_seed42.json'
  ['bamboogle_125']='data/bamboogle/questions_125.json'
)

for name in musique_1000 hotpotqa_1000 2wiki_1000 bamboogle_125; do
  qfile=${TASKS[$name]}
  outdir=results/amas_v3_qwen3_8b_think_4omini_${name}
  mkdir -p $outdir
  echo "=== $(date) Starting $name ==="
  .venv/bin/python scripts/run_amas.py \
      --questions $qfile \
      --output-dir $outdir \
      --retriever-url http://node408:8003 \
      --max-retrievals 3 \
      --worker mini \
      --planner-replica 0 \
      --concurrency 6 > $outdir/run.log 2>&1
  .venv/bin/python scripts/eval_offline.py \
      --predictions $outdir/predictions.jsonl \
      --questions $qfile \
      --output $outdir/eval.json
  echo "=== $(date) $name eval ==="
  cat $outdir/eval.json
done
echo "=== $(date) ALL DONE ==="
