Latest merged result snapshot pulled from `node409`.

Included:
- only latest completed merged runs
- `predictions.jsonl`
- `eval.json`
- `run_summary.json` when available

Datasets:
- `musique`
- `hotpotqa`
- `2wikimultihop`

Methods:
- `s0_matched`
- `a1_matched`
- `iter30_think`
- `iter27_think` (MuSiQue only)
- `iter27_no_think` (MuSiQue only)
- `s0_no_think` (MuSiQue only)

Excluded:
- incomplete runs
- per-shard outputs when a merged combined file exists
- external baselines still in progress
