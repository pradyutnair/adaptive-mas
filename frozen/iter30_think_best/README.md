# Iteration 30 Think Frozen Snapshot

This directory freezes the best `iter30_think` controller snapshot and its main evaluation artifacts.

Frozen decision:

- Best branch: `iter30_think`
- Status: frozen as best adaptive-thinking snapshot
- Freeze commit on `main`: `f1057ad`

Primary MuSiQue 1000 result:

- `EM 0.282`
- `F1 0.4086`
- `contain 0.393`
- `answered 954/1000`
- `mean_tokens 59.4k`

Fair matrix results:

- HotpotQA 1000:
  - `EM 0.493`
  - `F1 0.6544`
  - `contain 0.653`
  - `answered 990/1000`
- 2Wiki 1000:
  - `EM 0.480`
  - `F1 0.5985`
  - `contain 0.635`
  - `answered 969/1000`

Frozen artifacts:

- `config/m1_1.iter30_think.yaml`
- `pipeline.iter30.py`
- `results/iter30_think_full1000_combined.jsonl`
- `results/iter30_think_full1000_eval.json`
- `results/iter30_think_hotpotqa_seed42_fair_v4_combined.jsonl`
- `results/iter30_think_hotpotqa_seed42_fair_v4_eval.json`
- `results/iter30_think_hotpotqa_seed42_fair_v4_run_summary.json`
- `results/iter30_think_2wikimultihop_seed42_fair_v4_combined.jsonl`
- `results/iter30_think_2wikimultihop_seed42_fair_v4_eval.json`
- `results/iter30_think_2wikimultihop_seed42_fair_v4_run_summary.json`
