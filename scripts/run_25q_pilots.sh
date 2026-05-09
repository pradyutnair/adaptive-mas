#!/usr/bin/env bash
set -euo pipefail
cd /local/yzheng/pnair/workspace/adaptive-mas
source /local/yzheng/pnair/.env

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTBASE="results/base_isa_25q_${TIMESTAMP}"
mkdir -p "$OUTBASE"

COMMON="--retriever-url http://node408:8003 \
  --worker qwen14b_nothink --synth-mode qwen14b_nothink \
  --planner-model qwen3-14b --planner-mode think \
  --solver-budget 1024 --synth-budget 2048 \
  --use-isa --isa-max-rounds 3 --isa-accept-threshold 0.7 --isa-g-threshold 0.65 \
  --use-bridge-resolver --repair --max-retrievals 3 \
  --max-plan-subgoals 6 --synth-max-chunks 15 --synth-excerpt-chars 700 \
  --concurrency 24 --limit 25"

echo "=== Starting 25q pilots: $OUTBASE ==="

.venv/bin/python scripts/run_amas.py --questions data/pilot/musique_100_seed409.json --output-dir "$OUTBASE/musique" $COMMON &
.venv/bin/python scripts/run_amas.py --questions data/pilot/hotpotqa_100_seed409.json --output-dir "$OUTBASE/hotpotqa" $COMMON &
.venv/bin/python scripts/run_amas.py --questions data/pilot/2wikimultihop_100_seed409.json --output-dir "$OUTBASE/2wiki" $COMMON &
.venv/bin/python scripts/run_amas.py --questions data/bamboogle/questions_125.json --output-dir "$OUTBASE/bamboogle" $COMMON &
wait

echo "=== Running evaluations ==="
for ds in musique hotpotqa 2wiki bamboogle; do
  qfile=""
  case $ds in
    musique) qfile="data/pilot/musique_100_seed409.json" ;;
    hotpotqa) qfile="data/pilot/hotpotqa_100_seed409.json" ;;
    2wiki) qfile="data/pilot/2wikimultihop_100_seed409.json" ;;
    bamboogle) qfile="data/bamboogle/questions_125.json" ;;
  esac
  .venv/bin/python scripts/eval_offline.py --predictions "$OUTBASE/$ds/predictions.jsonl" --questions "$qfile" --output "$OUTBASE/$ds/eval.json"
  echo "$ds: $(cat "$OUTBASE/$ds/eval.json")"
done

echo ""
echo "=== ISA stats ==="
.venv/bin/python -c "
import json, os, glob
outbase = '$OUTBASE'
for ds in ['musique', 'hotpotqa', '2wiki', 'bamboogle']:
    pred_file = f'{outbase}/{ds}/predictions.jsonl'
    if not os.path.exists(pred_file): continue
    rows = [json.loads(l) for l in open(pred_file)]
    isa_acc = sum(1 for r in rows if r['metadata'].get('isa_accepted'))
    isa_esc = sum(1 for r in rows if r['metadata'].get('isa_escalated'))
    total = len(rows)
    print(f'{ds}: {total}q, ISA accepted={isa_acc} ({100*isa_acc/max(total,1):.0f}%), escalated to MAS={isa_esc} ({100*isa_esc/max(total,1):.0f}%)')
"

echo ""
echo "Results: $OUTBASE"
echo "$OUTBASE" > /tmp/base_isa_outbase.txt
