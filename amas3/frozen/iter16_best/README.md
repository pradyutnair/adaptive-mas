# Iteration 16 Frozen Snapshot

This directory freezes the best `M1.1` pilot branch from the overnight controller iteration cycle.

Frozen decision:

- Best branch: `iteration 16`
- Status: frozen as best
- Reason: no single remaining clean controller fix justified another patch cycle

Pilot summary:

- `EM 0.285`
- `contain 0.320`
- `avg_subagents 2.81`
- `avg_tokens 62.6k`
- `S0-easy EM 0.805`
- `S0-hard EM 0.151`

Easy-failure forensic summary (`S0-correct / iter16-wrong`, `n=8`):

- `1` grounded-answer preservation failure
- `2` surface-form mismatches
- `5` genuine retrieval / semantic-selection misses

Frozen artifacts:

- `config/m1_1.iter16.yaml`
- `pipeline.iter16.py`
- `results/M1_1_shard0_iter16seed_20260415_064101.predictions.jsonl`
- `results/M1_1_shard1_iter16seed_20260415_064101.predictions.jsonl`
- `results/M1_1_shard2_iter16seed_20260415_064101.predictions.jsonl`
