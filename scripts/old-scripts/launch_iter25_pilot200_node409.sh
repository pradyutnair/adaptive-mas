#!/usr/bin/env bash
set -euo pipefail
cd /local/yzheng/pnair/workspace/05-mas
source .venv/bin/activate
export PYTHONPATH=/local/yzheng/pnair/workspace/05-mas/src
export HF_HOME=/local/yzheng/.cache/huggingface
for i in 0 1 2; do
  rm -rf "/local/yzheng/pnair/workspace/05-mas/results/M1_1_iter25_pilot200_shard${i}"
  rm -f "/local/yzheng/pnair/workspace/05-mas/logs/run-M1_1_iter25_pilot200_shard${i}.log"
done
nohup env CUDA_VISIBLE_DEVICES= python scripts/runner.py --config /local/yzheng/pnair/workspace/05-mas/configs/m1_1.iter25.yaml --questions /local/yzheng/pnair/workspace/05-mas/data/musique/questions_pilot200_seed42_shard0.json --output-dir /local/yzheng/pnair/workspace/05-mas/results/M1_1_iter25_pilot200_shard0 --server-url http://localhost:8001/v1 --concurrency 24 > /local/yzheng/pnair/workspace/05-mas/logs/run-M1_1_iter25_pilot200_shard0.log 2>&1 &
echo $!
nohup env CUDA_VISIBLE_DEVICES= python scripts/runner.py --config /local/yzheng/pnair/workspace/05-mas/configs/m1_1.iter25.yaml --questions /local/yzheng/pnair/workspace/05-mas/data/musique/questions_pilot200_seed42_shard1.json --output-dir /local/yzheng/pnair/workspace/05-mas/results/M1_1_iter25_pilot200_shard1 --server-url http://localhost:8002/v1 --concurrency 24 > /local/yzheng/pnair/workspace/05-mas/logs/run-M1_1_iter25_pilot200_shard1.log 2>&1 &
echo $!
nohup env CUDA_VISIBLE_DEVICES= python scripts/runner.py --config /local/yzheng/pnair/workspace/05-mas/configs/m1_1.iter25.yaml --questions /local/yzheng/pnair/workspace/05-mas/data/musique/questions_pilot200_seed42_shard2.json --output-dir /local/yzheng/pnair/workspace/05-mas/results/M1_1_iter25_pilot200_shard2 --server-url http://localhost:8003/v1 --concurrency 24 > /local/yzheng/pnair/workspace/05-mas/logs/run-M1_1_iter25_pilot200_shard2.log 2>&1 &
echo $!
