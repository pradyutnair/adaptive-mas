#!/bin/bash
# Run all ablations + canonical sufficiency on musique 200q sequentially on
# one GPU. Each variant writes its own subdir under results/abl_musique200_<ts>/.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVER=${SERVER:-http://localhost:8001/v1}
CONC=${CONC:-24}
TS=$(date +%Y%m%d_%H%M%S)
ROOT=results/abl_musique200_${TS}
QFILE=data/musique/questions_200_seedfull_first.json
CHUNKS=data/musique/chunks.json
INDEX=data/musique/index_e5_base_v2

mkdir -p "$ROOT"
echo "$ROOT" > results/.abl_musique200_latest

# Variant_name -> config_path
declare -a VARIANTS=(
  "sufficiency:configs/m1_2.sufficiency.yaml"
  "abl_tau_050:configs/m1_2.abl_tau_050.yaml"
  "abl_tau_060:configs/m1_2.abl_tau_060.yaml"
  "abl_tau_080:configs/m1_2.abl_tau_080.yaml"
  "abl_tau_090:configs/m1_2.abl_tau_090.yaml"
  "abl_no_probe:configs/m1_2.abl_no_probe.yaml"
  "abl_no_controller:configs/m1_2.abl_no_controller.yaml"
  "abl_random_route:configs/m1_2.abl_random_route.yaml"
  "abl_oracle_route:configs/_runtime/m1_2.abl_oracle_route_musique200.yaml"
)

for entry in "${VARIANTS[@]}"; do
  name="${entry%%:*}"
  cfg="${entry##*:}"
  out="$ROOT/${name}"
  mkdir -p "$out"
  echo "[$(date +%T)] >>> ${name} (200q musique) -> ${out}"
  python3 scripts/runner.py \
    --config "$cfg" \
    --questions "$QFILE" \
    --output-dir "$out" \
    --server-url "$SERVER" \
    --concurrency "$CONC" \
    --chunks-file "$CHUNKS" \
    --index-dir "$INDEX" \
    --embedding-model intfloat/e5-base-v2 \
    > "$out/run.log" 2>&1
  npred=$(wc -l < "$out/predictions.jsonl" 2>/dev/null || echo 0)
  echo "[$(date +%T)] <<< ${name} done (${npred} preds)"
done

echo "ALL DONE -> $ROOT"
