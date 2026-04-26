# AGENTS.md - Adaptive Multi-Agent RAG Thesis Repo

## Mission

This repository supports Pradyut Nair's MSc thesis on adaptive multi-agent retrieval-augmented generation for complex question answering. The immediate research goal is:

- Match single-agent efficiency while approaching multi-agent performance.
- On MuSiQue, achieve `contain >= 0.40` with mean tokens `< 20k`, ideally `10k-15k`.
- Main metric is contain. Preserve high EM/F1 as well. Always use scripts/eval_offline.py for evaluation metrics. Do not use your own custom metrics.
- Use the exact same question IDs as the OPERA baseline for all OPERA comparisons.
- Build a principled adaptive multi-agent system: easy questions should fall back to single-agent behavior; harder MuSiQue-style questions should decompose, allocate effort, and call focused subagents.

This is thesis-critical work with roughly one month remaining. Prioritize reproducible progress over speculative rewrites.

## Hard Constraints

- No ensembling, pooling, majority voting, best-of-N, or answer selection across multiple independent generations.
- No hacks that use gold answers, OPERA predictions, or baseline outputs as features.
- No 1000q run unless the exact 50q pilot clears the intended quality/efficiency bar or the user explicitly asks for a diagnostic full run.
- Always count tokens from actual API/vLLM usage fields, not estimates.
- Always compare against the exact same question IDs as OPERA.
- Use node408 retriever for target runs: `curl -X POST http://node408:8003/retrieve`.
- Do not confuse local fallback retrieval with target retrieval. Local fallback results are diagnostic only.
- Save configs, predictions, intermediate metadata, and eval summaries for every experiment.
- Keep code clean enough for EMNLP-style methods: explainable routing, grounded evidence, reproducible configs.

## Compute and Paths

Primary active machine:

- SSH host: `node409`
- Active repo: `/local/yzheng/pnair/workspace/adaptive-mas`
- Python: `/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python`
- Do not use system `python3`; it is old and can break scripts.

Retriever:

- Required target retriever: `http://node408:8003/retrieve`
- Health check:

```bash
curl -sS -m 10 -X POST http://node408:8003/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["health check"],"topk":1,"mode":"text"}'
```

Generator:

- vLLM server: `http://localhost:8001/v1`
- Model: `Qwen/Qwen3-8B`
- Other local ports `8002` and `8003` are also vLLM servers; do not mistake localhost `8003` for the node408 retriever.

Important files:

- OPERA baseline: `results/external_baselines/opera_full/musique/opera_musique_1000_combined.jsonl`
- Exact OPERA-matched 50q: `data/musique/opera_matched/questions_50.json`
- Exact OPERA-matched 1000q: `data/musique/opera_matched/questions_1000.json`
- Main runner: `scripts/runner.py`
- Offline evaluator: `scripts/eval_offline.py`
- Current adaptive implementation: `src/adaptive_sage/`
- Core LLM client: `src/arag/core/llm.py`
- Important configs:
  - `configs/m4.single_retry_fallback.yaml`
  - `configs/m5.cursor_style.yaml`
  - `configs/m5.cursor_style_budget18k.yaml`

## Required Evaluation Protocol

1. Use OPERA baseline only for ID matching and metric comparison.
2. Never use OPERA answers/predictions inside our method.
3. Run exact 50q pilot first:

```bash
.venv/bin/python scripts/<runner>.py \
  --questions data/musique/opera_matched/questions_50.json \
  --output-dir results/<descriptive_run_name> \
  --server-url http://localhost:8001/v1 \
  --retriever-url http://node408:8003
```

4. Evaluate:

```bash
.venv/bin/python scripts/eval_offline.py \
  --predictions results/<run>/predictions.jsonl \
  --questions data/musique/opera_matched/questions_50.json \
  --output results/<run>/eval.json
```

5. Compute actual mean token usage from prediction metadata:

```bash
.venv/bin/python - <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
toks = [r.get("metadata", {}).get("total_tokens", r.get("total_tokens", 0)) for r in rows]
print("rows", len(rows))
print("mean_total_tokens_actual", sum(toks) / len(toks) if toks else 0)
PY results/<run>/predictions.jsonl
```

6. Only after a passing 50q pilot, run the exact 1000q file:

```bash
.venv/bin/python scripts/<runner>.py \
  --questions data/musique/opera_matched/questions_1000.json \
  --output-dir results/<descriptive_1000q_run> \
  --server-url http://localhost:8001/v1 \
  --retriever-url http://node408:8003
```

## Current Known State

Exact OPERA-matched files have already been created:

- `data/musique/opera_matched/questions_50.json`
- `data/musique/opera_matched/questions_1000.json`

OPERA reference metrics on these IDs:

- First 50 OPERA: `contain=0.40`, mean tokens about `20127`
- Full 1000 OPERA: `contain=0.361`, mean tokens about `20346`

Best valid node408 pilot so far:

- Run: `results/opera_lite_opera50_planthink_20260426_002803`
- This was not OPERA code despite the confusing name. It was a decomposition prototype using node408 retrieval.
- Result: `contain=0.28`, `norm_em=0.26`, `token_f1=0.3033`
- Mean tokens: about `4156`
- Lesson: planning/decomposition helps, but answer/evidence sufficiency is still weak.

Other observed results:

- Direct single-agent top20 node408: `contain=0.08`, mean tokens about `3759`
- Adaptive v2 sufficiency-gated node408: `contain=0.10`, mean tokens about `4708`
- Decomposition no-thinking prototype: `contain=0.02`, mean tokens about `1579`
- Decomposition with reader thinking on 10q: worse quality and too slow
- Old M5 exact-50: `contain=0.34`, mean tokens about `31273`; not acceptable because it used local index and exceeded token budget.
- Capped M5 attempts currently fail with blank answers due Qwen thinking/JSON parsing issues.

## Implementation Lessons

- Qwen3 with `enable_thinking: true` often spends many tokens in `<think>` and may return no parseable JSON if capped too tightly.
- Qwen3 with thinking disabled is fast but often stops at bridge answers or accepts weak evidence.
- Thinking only for decomposition has been the best tradeoff so far.
- Reader/synthesizer prompts must preserve exact spans, especially full dates like `June 1982`, acronym expansions like `Sea, Air, and Land`, and role/character answers like `Vito Corleone`.
- The main failure mode is not only retrieval recall; it is evidence sufficiency and final-target alignment.
- Avoid broad single-shot retrieval: MuSiQue usually needs concrete bridge-conditioned follow-up queries.
- Do not spend full-run compute until 50q has convincing evidence of success.

## Recommended Next Direction

Build a clean adaptive method, not another ad hoc runner:

1. Route each question:
   - single probe for simple entity/date/count questions where direct retrieval has high evidence sufficiency
   - decomposed plan for bridge/compositional questions
2. Use node408 retrieval for every target result.
3. For decomposed questions:
   - use limited thinking for planning only
   - use no-thinking evidence readers
   - add strict evidence sufficiency checks before accepting a subgoal
   - retry retrieval only when evidence is explicitly insufficient
4. Improve final-target alignment:
   - final answer must answer the original question, not a bridge
   - preserve exact spans from evidence
   - require cited evidence for the final answer
5. Add a small diagnostic report per run:
   - route distribution
   - mean tokens by route
   - contain/EM/F1 by route
   - failed examples with plan, queries, accepted facts, final evidence

## Naming Discipline

Do not name our methods after baselines. Previous `opera_lite` filenames caused confusion. Prefer names like:

- `adaptive_decompose_node408`
- `adaptive_sufficiency_node408`
- `single_probe_fallback_node408`
- `route_then_decompose_node408`

## Baseline Comparison Rule

OPERA is an external baseline only. Use:

- its question IDs,
- its reported predictions for offline comparison,
- its trajectory only for qualitative diagnosis.

Do not use:

- OPERA predictions as candidates,
- OPERA answers as supervision,
- OPERA trajectory as runtime input,
- OPERA-derived per-question routing labels.

## Working Style for Future Agents

- Start by checking current processes on node409.
- Verify node408 retriever health before any target run.
- Read the latest run outputs before launching new jobs.
- Keep updates concise.
- If a pilot fails, inspect examples before changing prompts.
- Prefer small 10q or 50q pilots with exact IDs over long runs.
- Leave a clear run note in the result directory when a run is important.
- Do not overwrite previous predictions.
- Stop bad runs early when early rows show blank answers, parse failures, or obvious route collapse.
# AGENTS.md - Adaptive Multi-Agent RAG Thesis Repo

## Mission

This repository supports Pradyut Nair's MSc thesis on adaptive multi-agent retrieval-augmented generation for complex question answering. The immediate research goal is:

- Match single-agent efficiency while approaching multi-agent performance.
- On MuSiQue, achieve `contain >= 0.40` with mean tokens `< 20k`, ideally `10k-15k`.
- Main metric is contain. Preserve high EM/F1 as well. Always use scripts/eval_offline.py for evaluation metrics. Do not use your own custom metrics.
- Use the exact same question IDs as the OPERA baseline for all OPERA comparisons.
- Build a principled adaptive multi-agent system: easy questions should fall back to single-agent behavior; harder MuSiQue-style questions should decompose, allocate effort, and call focused subagents.

This is thesis-critical work with roughly one month remaining. Prioritize reproducible progress over speculative rewrites.

## Hard Constraints

- No ensembling, pooling, majority voting, best-of-N, or answer selection across multiple independent generations.
- No hacks that use gold answers, OPERA predictions, or baseline outputs as features.
- No 1000q run unless the exact 50q pilot clears the intended quality/efficiency bar or the user explicitly asks for a diagnostic full run.
- Always count tokens from actual API/vLLM usage fields, not estimates.
- Always compare against the exact same question IDs as OPERA.
- Use node408 retriever for target runs: `curl -X POST http://node408:8003/retrieve`.
- Do not confuse local fallback retrieval with target retrieval. Local fallback results are diagnostic only.
- Save configs, predictions, intermediate metadata, and eval summaries for every experiment.
- Keep code clean enough for EMNLP-style methods: explainable routing, grounded evidence, reproducible configs.

## Compute and Paths

Primary active machine:

- SSH host: `node409`
- Active repo: `/local/yzheng/pnair/workspace/adaptive-mas`
- Python: `/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python`
- Do not use system `python3`; it is old and can break scripts.

Retriever:

- Required target retriever: `http://node408:8003/retrieve`
- Health check:

```bash
curl -sS -m 10 -X POST http://node408:8003/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["health check"],"topk":1,"mode":"text"}'
```

Generator:

- vLLM server: `http://localhost:8001/v1`
- Model: `Qwen/Qwen3-8B`
- Other local ports `8002` and `8003` are also vLLM servers; do not mistake localhost `8003` for the node408 retriever.

Important files:

- OPERA baseline: `results/external_baselines/opera_full/musique/opera_musique_1000_combined.jsonl`
- Exact OPERA-matched 50q: `data/musique/opera_matched/questions_50.json`
- Exact OPERA-matched 1000q: `data/musique/opera_matched/questions_1000.json`
- Main runner: `scripts/runner.py`
- Offline evaluator: `scripts/eval_offline.py`
- Current adaptive implementation: `src/adaptive_sage/`
- Core LLM client: `src/arag/core/llm.py`
- Important configs:
  - `configs/m4.single_retry_fallback.yaml`
  - `configs/m5.cursor_style.yaml`
  - `configs/m5.cursor_style_budget18k.yaml`

## Required Evaluation Protocol

1. Use OPERA baseline only for ID matching and metric comparison.
2. Never use OPERA answers/predictions inside our method.
3. Run exact 50q pilot first:

```bash
.venv/bin/python scripts/<runner>.py \
  --questions data/musique/opera_matched/questions_50.json \
  --output-dir results/<descriptive_run_name> \
  --server-url http://localhost:8001/v1 \
  --retriever-url http://node408:8003
```

4. Evaluate:

```bash
.venv/bin/python scripts/eval_offline.py \
  --predictions results/<run>/predictions.jsonl \
  --questions data/musique/opera_matched/questions_50.json \
  --output results/<run>/eval.json
```

5. Compute actual mean token usage from prediction metadata:

```bash
.venv/bin/python - <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
toks = [r.get("metadata", {}).get("total_tokens", r.get("total_tokens", 0)) for r in rows]
print("rows", len(rows))
print("mean_total_tokens_actual", sum(toks) / len(toks) if toks else 0)
PY results/<run>/predictions.jsonl
```

6. Only after a passing 50q pilot, run the exact 1000q file:

```bash
.venv/bin/python scripts/<runner>.py \
  --questions data/musique/opera_matched/questions_1000.json \
  --output-dir results/<descriptive_1000q_run> \
  --server-url http://localhost:8001/v1 \
  --retriever-url http://node408:8003
```

## Current Known State

Exact OPERA-matched files have already been created:

- `data/musique/opera_matched/questions_50.json`
- `data/musique/opera_matched/questions_1000.json`

OPERA reference metrics on these IDs:

- First 50 OPERA: `contain=0.40`, mean tokens about `20127`
- Full 1000 OPERA: `contain=0.361`, mean tokens about `20346`

Best valid node408 pilot so far:

- Run: `results/opera_lite_opera50_planthink_20260426_002803`
- This was not OPERA code despite the confusing name. It was a decomposition prototype using node408 retrieval.
- Result: `contain=0.28`, `norm_em=0.26`, `token_f1=0.3033`
- Mean tokens: about `4156`
- Lesson: planning/decomposition helps, but answer/evidence sufficiency is still weak.

Other observed results:

- Direct single-agent top20 node408: `contain=0.08`, mean tokens about `3759`
- Adaptive v2 sufficiency-gated node408: `contain=0.10`, mean tokens about `4708`
- Decomposition no-thinking prototype: `contain=0.02`, mean tokens about `1579`
- Decomposition with reader thinking on 10q: worse quality and too slow
- Old M5 exact-50: `contain=0.34`, mean tokens about `31273`; not acceptable because it used local index and exceeded token budget.
- Capped M5 attempts currently fail with blank answers due Qwen thinking/JSON parsing issues.

## Implementation Lessons

- Qwen3 with `enable_thinking: true` often spends many tokens in `<think>` and may return no parseable JSON if capped too tightly.
- Qwen3 with thinking disabled is fast but often stops at bridge answers or accepts weak evidence.
- Thinking only for decomposition has been the best tradeoff so far.
- Reader/synthesizer prompts must preserve exact spans, especially full dates like `June 1982`, acronym expansions like `Sea, Air, and Land`, and role/character answers like `Vito Corleone`.
- The main failure mode is not only retrieval recall; it is evidence sufficiency and final-target alignment.
- Avoid broad single-shot retrieval: MuSiQue usually needs concrete bridge-conditioned follow-up queries.
- Do not spend full-run compute until 50q has convincing evidence of success.

## Recommended Next Direction

Build a clean adaptive method, not another ad hoc runner:

1. Route each question:
   - single probe for simple entity/date/count questions where direct retrieval has high evidence sufficiency
   - decomposed plan for bridge/compositional questions
2. Use node408 retrieval for every target result.
3. For decomposed questions:
   - use limited thinking for planning only
   - use no-thinking evidence readers
   - add strict evidence sufficiency checks before accepting a subgoal
   - retry retrieval only when evidence is explicitly insufficient
4. Improve final-target alignment:
   - final answer must answer the original question, not a bridge
   - preserve exact spans from evidence
   - require cited evidence for the final answer
5. Add a small diagnostic report per run:
   - route distribution
   - mean tokens by route
   - contain/EM/F1 by route
   - failed examples with plan, queries, accepted facts, final evidence

## Naming Discipline

Do not name our methods after baselines. Previous `opera_lite` filenames caused confusion. Prefer names like:

- `adaptive_decompose_node408`
- `adaptive_sufficiency_node408`
- `single_probe_fallback_node408`
- `route_then_decompose_node408`

## Baseline Comparison Rule

OPERA is an external baseline only. Use:

- its question IDs,
- its reported predictions for offline comparison,
- its trajectory only for qualitative diagnosis.

Do not use:

- OPERA predictions as candidates,
- OPERA answers as supervision,
- OPERA trajectory as runtime input,
- OPERA-derived per-question routing labels.

## Working Style for Future Agents

- Start by checking current processes on node409.
- Verify node408 retriever health before any target run.
- Read the latest run outputs before launching new jobs.
- Keep updates concise.
- If a pilot fails, inspect examples before changing prompts.
- Prefer small 25q or 50q pilots with exact IDs over long runs.
- Leave a clear run note in the result directory when a run is important.
- Do not overwrite previous predictions.
- Stop bad runs early when early rows show blank answers, parse failures, or obvious route collapse.
