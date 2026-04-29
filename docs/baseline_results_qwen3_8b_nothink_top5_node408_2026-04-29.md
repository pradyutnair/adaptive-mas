# Qwen3-8B Baselines, No Thinking, Top-k 5, Node408

Date: 2026-04-29

Config:

- `/local/yzheng/pnair/baseline/configs/full_baselines_qwen3_8b_nothink_top5_node408.json`
- Model: `Qwen/Qwen3-8B`
- Thinking: disabled
- Retriever: `http://node408:8003/retrieve`
- Retrieval top-k: 5
- Questions: 1000 each from `/local/yzheng/pnair/data`
- Combined output root: `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408`

All combined prediction files have 1000 rows. Blank answers: 3 total, all in OPERA-CoT:

- `opera_cot_2wiki`: `d5a1a526088311ebbd6dac1f6bf848b6`, `6837abca0bb011ebab90acde48001122`
- `opera_cot_hotpotqa`: `5a8b770f5542995d1e6f13a7`

No `<think>` traces were found in any combined prediction file.

| Method | Dataset | Contain | EM | F1 | Mean tokens | Mean wall sec | Predictions | Eval |
|---|---|---:|---:|---:|---:|---:|---|---|
| naive_rag | hotpotqa | 0.377 | 0.301 | 0.4174 | 886.7 | 1.557 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/naive_rag_hotpotqa/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/naive_rag_hotpotqa/eval.json` |
| naive_rag | 2wiki | 0.295 | 0.253 | 0.3031 | 924.9 | 1.657 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/naive_rag_2wiki/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/naive_rag_2wiki/eval.json` |
| naive_rag | musique | 0.079 | 0.038 | 0.1073 | 881.0 | 1.801 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/naive_rag_musique/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/naive_rag_musique/eval.json` |
| ircot | hotpotqa | 0.448 | 0.347 | 0.4638 | 6046.4 | 4.578 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ircot_hotpotqa/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ircot_hotpotqa/eval.json` |
| ircot | 2wiki | 0.474 | 0.243 | 0.3317 | 7106.4 | 5.135 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ircot_2wiki/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ircot_2wiki/eval.json` |
| ircot | musique | 0.127 | 0.081 | 0.1564 | 7409.2 | 5.397 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ircot_musique/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ircot_musique/eval.json` |
| opera_cot | hotpotqa | 0.327 | 0.098 | 0.2142 | 2227.0 | 16.445 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/opera_cot_hotpotqa/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/opera_cot_hotpotqa/eval.json` |
| opera_cot | 2wiki | 0.304 | 0.046 | 0.1515 | 2258.2 | 15.981 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/opera_cot_2wiki/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/opera_cot_2wiki/eval.json` |
| opera_cot | musique | 0.058 | 0.015 | 0.0634 | 2206.4 | 16.531 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/opera_cot_musique/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/opera_cot_musique/eval.json` |
| ma_rag | hotpotqa | 0.420 | 0.270 | 0.3865 | 8067.2 | 31.331 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ma_rag_hotpotqa/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ma_rag_hotpotqa/eval.json` |
| ma_rag | 2wiki | 0.412 | 0.265 | 0.3458 | 9195.3 | 28.858 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ma_rag_2wiki/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ma_rag_2wiki/eval.json` |
| ma_rag | musique | 0.177 | 0.106 | 0.1982 | 9028.6 | 31.961 | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ma_rag_musique/predictions.jsonl` | `/local/yzheng/pnair/baseline/results/full/combined_nothink_top5_node408/ma_rag_musique/eval.json` |

