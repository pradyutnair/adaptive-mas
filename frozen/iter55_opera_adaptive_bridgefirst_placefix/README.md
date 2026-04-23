# Iter55 Frozen Snapshot

This directory freezes the `iter55_opera_adaptive_bridgefirst_placefix` MuSiQue audit run and the current local code snapshot around that path.

Result source:

- predictions copied from `node409:/local/yzheng/pnair/workspace/adaptive-mas/results/audit/iter55_predictions.jsonl`
- logs copied from `node409:/local/yzheng/pnair/workspace/adaptive-mas/logs/iter55_*`

Offline MuSiQue 1000 eval:

- `EM 0.241`
- `F1 0.3474`
- `contain 0.403`
- `answered 1000/1000`

Notes:

- there was no packaged `eval.json` or `run_summary.json` next to the remote audit file
- this snapshot keeps the raw predictions, local config/code, and the remote logs

Frozen artifacts:

- `config/iter55_opera_adaptive_bridgefirst_placefix.yaml`
- `code/scripts/run_adaptive_opera_hybrid.py`
- `code/src/adaptive_sage/pipeline.py`
- `code/src/adaptive_sage/orchestrator.py`
- `code/src/adaptive_sage/investigator.py`
- `code/src/adaptive_sage/types.py`
- `code/src/adaptive_sage/prompts/orchestrator_route.txt`
- `code/src/adaptive_sage/prompts/orchestrator_answer.txt`
- `code/src/adaptive_sage/prompts/investigator_distill.txt`
- `code/src/adaptive_sage/prompts/investigator_distill_strict.txt`
- `code/src/arag/tools/read_chunk.py`
- `results/musique_predictions.jsonl`
- `results/musique_eval.json`
- `logs/iter55_musique1000_adaptive_bridgefirst_placefix_20260422_shard0.log`
- `logs/iter55_musique1000_adaptive_bridgefirst_placefix_20260422_shard1.log`
- `logs/iter55_musique1000_adaptive_bridgefirst_placefix_20260422_shard2.log`
- `logs/iter55_resume_shard0.log`
- `logs/iter55_resume_shard1.log`
- `logs/iter55_resume_shard2.log`
