#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=/local/yzheng/pnair/.cache/huggingface/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd
Q=data/musique/questions_smoke50_hard_3hop2_4hop.json
TS=$(date +%Y%m%d_%H%M%S)
ROOT=results/hard_subset_probe_vs_adaptive_${TS}
mkdir -p "$ROOT"
echo "$ROOT" > results/.hard_subset_latest
run_one() {
  local name=$1 config=$2 port=$3
  local out="$ROOT/$name"
  mkdir -p "$out"
  PATH="$PWD/.venv/bin:$PATH" python3 scripts/runner.py \
    --config "$config" \
    --questions "$Q" \
    --output-dir "$out" \
    --server-url "http://localhost:${port}/v1" \
    --concurrency 24 \
    --chunks-file data/musique/chunks.json \
    --index-dir data/musique/index_e5_base_v2 \
    --embedding-model "$MODEL" \
    > "$out/run.log" 2>&1
}
run_one probe_only configs/_runtime/m1_2.hard_probe_only.yaml 8001 &
pid_a=$!
run_one current_adaptive configs/_runtime/m1_2.hard_current_adaptive.yaml 8002 &
pid_b=$!
wait $pid_a $pid_b
.venv/bin/python - <<'PY'
import json,pathlib,re,glob
root=pathlib.Path('results/.hard_subset_latest').read_text().strip()
def norm(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())
def contain(a,b):
    a=norm(a); b=norm(b); return bool(a and b and (a in b or b in a))
for name in ['probe_only','current_adaptive']:
    p=pathlib.Path(root)/name/'predictions.jsonl'
    rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    toks=[r.get('metadata',{}).get('total_tokens',0) for r in rows]
    print(name, 'n', len(rows), 'contain', sum(contain(r.get('answer',''),r.get('gold_answer','')) for r in rows)/len(rows), 'avg_tokens', sum(toks)/len(toks), 'empty', sum(not r.get('answer') for r in rows))
PY
