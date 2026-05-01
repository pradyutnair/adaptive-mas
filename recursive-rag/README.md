# Recursive RAG

DSPy ReAct root agent with autonomous retrieval sub-agents.

## Architecture

- Root LM: planner, verifier, synthesizer via `dspy.ReAct`.
- Tools:
  - `hop(question)`: one autonomous retrieval/extraction sub-agent.
  - `hop_batch([q1, q2, ...])`: parallel hop dispatch.
  - `submit(answer, support_ids)`: programmatic citation gate.
- Sub-LM loop per hop:
  - retrieve(query, k=5)
  - extract answer span
  - if confidence >= 0.7, return
  - else propose query rewrite and retry
  - up to 3 retrieval attempts, return best finding

No ensembling, pooling, majority voting, or best-of-N selection is used.

## Run

```bash
../adaptive-mas/.venv/bin/python scripts/run.py \
  --questions data/musique/opera_matched/questions_50.json \
  --out-dir results/react_raw_50 \
  --root-lm qwen14b-think \
  --sub-lm qwen14b-nothink \
  --retriever-url http://node408:8003 \
  --n 50
```
