# M1.2 Sufficiency Frozen Snapshot

This directory freezes the `m1_2_sufficiency` 1000-question result bundle and the current local code snapshot around the sufficiency controller path.

Result source:

- artifacts copied from `node409:/local/yzheng/pnair/workspace/adaptive-mas/paper_results/sufficiency_1000q_20260418_215804`

Matched-set summary from `results/compare_1000q.json`:

- MuSiQue:
  `contain 0.366`, `F1 0.4097`, `EM 0.287`, `mean_tokens 50.0k`
- HotpotQA:
  `contain 0.673`, `F1 0.6877`, `EM 0.529`, `mean_tokens 19.1k`
- 2WikiMultihop:
  `contain 0.695`, `F1 0.6544`, `EM 0.542`, `mean_tokens 32.6k`

Offline per-dataset evals from the frozen predictions:

- MuSiQue:
  `contain 0.384`, `F1 0.4202`, `EM 0.300`, `answered 965/1000`
- HotpotQA:
  `contain 0.689`, `F1 0.6981`, `EM 0.542`, `answered 989/1000`
- 2WikiMultihop:
  `contain 0.726`, `F1 0.6710`, `EM 0.565`, `answered 976/1000`

Notes:

- `compare_1000q.json` is the matched common-set summary used in your paper analysis
- `results/*/eval.json` are standalone offline evals over the frozen prediction files

Frozen artifacts:

- `config/m1_2.sufficiency.yaml`
- `code/src/adaptive_sage/pipeline.py`
- `code/src/adaptive_sage/orchestrator.py`
- `code/src/adaptive_sage/investigator.py`
- `code/src/adaptive_sage/types.py`
- `code/src/adaptive_sage/prompts/orchestrator_route.txt`
- `code/src/adaptive_sage/prompts/orchestrator_answer.txt`
- `code/src/adaptive_sage/prompts/investigator_distill.txt`
- `code/src/adaptive_sage/prompts/investigator_distill_strict.txt`
- `code/src/arag/tools/read_chunk.py`
- `results/compare_1000q.json`
- `results/musique/predictions.jsonl`
- `results/musique/eval.json`
- `results/musique/run_summary.json`
- `results/hotpotqa/predictions.jsonl`
- `results/hotpotqa/eval.json`
- `results/hotpotqa/run_summary.json`
- `results/2wikimultihop/predictions.jsonl`
- `results/2wikimultihop/eval.json`
- `results/2wikimultihop/run_summary.json`
