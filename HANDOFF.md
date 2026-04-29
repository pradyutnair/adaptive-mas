# AMAS v2 Agent Handoff - Complete Context

## THE GOAL

Achieve **EM >= 0.30 on 1000 MuSiQue questions** using the `node408:8003` retriever (full 21M Wikipedia index). The user does NOT care about the contain metric -- only normalized exact match (EM). This is for Pradyut Nair's MSc thesis on Adaptive Multi-Agent Systems for RAG.

## WHAT EXISTS RIGHT NOW

Branch `amas-v2` in `/local/yzheng/pnair/workspace/adaptive-mas` has a working system:

- **Code**: `src/amas_v2/` -- complete pipeline (planner, investigator, DAG executor, synthesizer, pipeline orchestrator)
- **Runner**: `scripts/run_amas_v2.py`
- **Config**: `configs/amas_v2.yaml`
- **Prompts**: `src/amas_v2/prompts/` (planner.txt, analyze.txt, rewrite.txt, synthesize.txt, strategist_review.txt)

### Current Best Results

| Run | N | EM | Contain | Blank | Mean Tokens |
|-----|---|----|---------|-------|-------------|
| v2.0 50q pilot | 50 | **0.32** | 0.48 | 4/50 (8%) | 8,313 |
| v2.0 1000q | 1000 | **0.186** | 0.329 | 72/1000 (7.2%) | 8,458 |
| v2.3 50q pilot | 50 | **0.34** | 0.56 | 11/50 (22%) | 6,168 |

The 50q pilots look promising (0.32-0.34 EM) but performance drops at 1000q scale (0.186). OPERA baseline gets EM=0.212 on the same 1000q (but using an easier ARAG corpus on node9110, not the full 21M Wikipedia we use).

## MODEL CONFIGURATION

- **Planner (orchestrator)**: Qwen3-14B with thinking enabled, running on `localhost:8003/v1` (vLLM)
- **Investigator (subagents)**: gpt-4o-mini via OpenAI API
- **Synthesizer**: gpt-4o-mini via OpenAI API
- **Retriever**: `http://node408:8003/retrieve` -- full 21M Wikipedia, request format: `{"queries": ["query"], "topk": 7, "mode": "text"}`

To run anything, you MUST first: `source /local/yzheng/pnair/.env && export OPENAI_API_KEY`

Also available but NOT currently used:
- Qwen3-8B on `localhost:8001/v1`
- Another model on `localhost:8002/v1`

## ARCHITECTURE

```
Question → Planner (Qwen3-14B+thinking) → Execution Plan (DAG of subgoals)
  ↓
  If simple (1 subgoal): Direct investigator call → answer
  If compositional (2+ subgoals): DAG Executor
    ↓
    For each level of the DAG (topologically sorted):
      Run investigator nodes in parallel
      Each node: retrieve → analyze evidence → answer or retry
      Retry loop (up to 3 attempts):
        Rewrite query using failure reason + retrieved docs → re-retrieve → re-analyze
      Placeholder resolution: [result_1] → actual answer from prior hop
    ↓
    If final hop succeeded: Synthesizer extracts final answer from all collected facts
    If final hop failed: Strategist Review (Qwen3-14B) decides to accept/revise/add hop
    If still no answer: Fallback direct retrieval on original question
```

## DIAGNOSED FAILURE MODES (from v2.0 1000q analysis)

### 1. "Bridge OK, final wrong" -- 621/814 failures (76%)
The planner decomposes correctly, the bridge entity is found, but the final hop gives the wrong answer. Root cause: the investigator marks answers as "verified" even when they're wrong (wrong bridge propagates). The retriever often returns docs about the right topic but missing the specific fact needed.

### 2. Blanks -- 72/1000 (7.2%)
No answer produced. The fallback (direct retrieval on original question) isn't effective enough for complex multi-hop questions.

### 3. Near-misses -- 71/1000 (7.1%)
Answer is right but too verbose or too short. Examples:
- "Hubbard County, Minnesota" vs gold "Hubbard County"
- "February 2, 1848" vs gold "1848"
- "five games per year" vs gold "five"
The synthesizer adds qualifiers or the investigator returns too much context.

### 4. Under-decomposition -- 188/1000
Planner used fewer hops than the question actually needs (e.g., 2 subgoals for a 3-hop question). This was partially fixed in v2.3 with improved planner prompt.

### 5. Prompt example leakage -- 18/1000 (FIXED)
The synthesizer prompt contained "mid-June" as an example, and the model outputted "mid-June" as a default answer for 18 questions. This has been fixed by removing all concrete answer examples from prompts.

### 6. No adaptive routing
All 50q questions routed as "compositional" (MuSiQue is inherently multi-hop). The system never collapses to single-agent. This is actually correct for MuSiQue but means we can't demonstrate the adaptive efficiency claim on this dataset.

### 7. No parallel hop execution in practice
Only 1/50 plans had parallel levels. Nearly all plans are strictly sequential chains. This means we're not leveraging parallelism -- a key thesis differentiator.

## WHAT OPERA DOES (the baseline that works)

OPERA's architecture is dead simple and it's what we should learn from:

**Plan Agent prompt** (question decomposition):
```
You are a strategic planning agent. Given a complex multi-hop question, decompose it into a sequence of simpler sub-goals with dependency modeling.
[question]
Return JSON array of {subgoal_id, subgoal, dependencies}
Requirements:
- Use placeholder mechanism: [entity from step X] for dependencies
- Each subgoal should be answerable with a small set of documents
- Maintain logical flow and clear dependencies
```

**Analysis-Answer Agent prompt** (read evidence):
```
You are an analysis and answering agent. Given a sub-question and retrieved documents, determine if you can answer the question and provide analysis.
[sub_question + docs]
Return JSON: {status: "yes"/"no", answer: "...", analysis: "..."}
```

**Rewrite Agent prompt** (query reformulation -- sees failure reason):
```
You are an expert query rewriter for information retrieval.
[original question + failure reason]
Return JSON: {rewritten_query, strategy, keywords}
```

**Key OPERA design decisions:**
- **No synthesizer** -- final answer = last hop's answer directly
- **Docs truncated to 300 chars** per document, max 5 docs
- **max_retries = 2** per sub-question
- **Simple prompts** -- no concrete examples, no disambiguation rules
- **Sequential execution** -- each hop waits for the previous one
- **If a hop fails, dependent hops are skipped**
- **temperature = 0.1** for all LLM calls

## CRITICAL BUGS AND GUARDRAILS

1. **NEVER put concrete answer examples in prompts.** The model latches onto them. We lost 18 EM points from "mid-June" leaking from a synthesizer prompt example. Use abstract formatting rules only.

2. **The synthesizer can hurt more than it helps.** It only improved 1 answer out of 1000 but degraded some others and introduced the mid-June leak. Consider removing it entirely and just using the last hop's answer. BUT if you remove it, you need the `_extract_final_answer` logic to be smart -- don't return intermediate bridge entities when the final hop fails.

3. **Evidence truncation matters.** 300 chars (OPERA's setting) is too aggressive for our setup. 800 chars was the original. 500 chars worked OK in v2.3. The sweet spot is probably 400-600 chars.

4. **The rewrite agent needs to see retrieved docs.** We just added this (passing `previous_evidence` to the rewrite template). OPERA's rewrite agent also sees a docs preview. Without seeing what WAS retrieved, the rewriter can't formulate a meaningfully different query.

5. **`source /local/yzheng/pnair/.env && export OPENAI_API_KEY`** must be run before any execution. The OpenAI API key is not in the environment by default.

6. **localhost:8003 is Qwen3-14B (vLLM), NOT the retriever.** The retriever is `node408:8003`. This is a common confusion point.

## FILE PATHS

```
/local/yzheng/pnair/workspace/adaptive-mas/          # repo root
├── src/amas_v2/                                       # current system
│   ├── pipeline.py                                    # main orchestrator
│   ├── planner.py                                     # question decomposition
│   ├── investigator.py                                # evidence retrieval + reading
│   ├── dag_executor.py                                # DAG execution with retries
│   ├── synthesizer.py                                 # final answer extraction
│   ├── llm.py                                         # LLM client (OpenAI + vLLM)
│   ├── retriever.py                                   # HTTP retriever client
│   ├── config.py                                      # YAML config loader
│   ├── types.py                                       # data types
│   └── prompts/                                       # prompt templates
│       ├── planner.txt
│       ├── analyze.txt
│       ├── rewrite.txt
│       ├── synthesize.txt
│       └── strategist_review.txt
├── src/amas/                                          # older v1 system (reference)
├── scripts/
│   ├── run_amas_v2.py                                 # v2 runner
│   ├── eval_offline.py                                # official evaluator
│   ├── analyze_failures.py                            # failure analysis
│   └── analyze_final_hop.py                           # final hop failure analysis
├── configs/
│   └── amas_v2.yaml                                   # current config
├── data/musique/
│   ├── opera408_50.json                               # 50q pilot questions
│   ├── questions_1000_seedfull_combined.json           # 1000q full set
│   └── opera_matched/
│       ├── questions_50.json                          # OPERA-matched 50q
│       └── questions_1000.json                        # OPERA-matched 1000q
├── results/
│   ├── amas_v2_smoke10/                               # v2.0 10q smoke (EM=0.20)
│   ├── amas_v2_pilot50/                               # v2.0 50q pilot (EM=0.32)
│   ├── amas_v2_1000q/                                 # v2.0 1000q (EM=0.186)
│   ├── amas_v2_3_pilot50/                             # v2.3 50q pilot (EM=0.34)
│   └── external_baselines/opera_full/musique/         # OPERA predictions
└── /local/yzheng/pnair/workspace/baseline_repos/OPERA # OPERA source code
```

## HOW TO RUN

```bash
# Health checks
curl -sS -m 5 http://localhost:8003/v1/models  # Qwen3-14B
curl -sS -m 5 -X POST http://node408:8003/retrieve -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1,"mode":"text"}'

# Run 50q pilot
cd /local/yzheng/pnair/workspace/adaptive-mas
source /local/yzheng/pnair/.env && export OPENAI_API_KEY
.venv/bin/python scripts/run_amas_v2.py \
  --config configs/amas_v2.yaml \
  --questions data/musique/opera408_50.json \
  --output-dir results/<run_name> \
  --concurrency 8

# Evaluate
.venv/bin/python scripts/eval_offline.py \
  --predictions results/<run_name>/predictions.jsonl \
  --questions data/musique/opera408_50.json \
  --output results/<run_name>/eval.json

# 1000q (only after 50q passes)
.venv/bin/python scripts/run_amas_v2.py \
  --config configs/amas_v2.yaml \
  --questions data/musique/questions_1000_seedfull_combined.json \
  --output-dir results/<run_name_1000q> \
  --concurrency 12
```

## MY OPINIONS ON HOW TO HIT 0.30 EM

### The core problem is NOT the architecture -- it's answer quality per hop

The planner works well. The DAG structure works. The pipeline orchestration works. The problem is that at each hop, the investigator produces the wrong answer ~40% of the time. When you chain 2-3 hops, error compounds: 0.6^2 = 0.36 accuracy for 2-hop, 0.6^3 = 0.216 for 3-hop. This matches our observed EM by hop count (2-hop: 22.7%, 3-hop: 11.3%, 4-hop: 9.6%).

### Concrete improvements that would move the needle

**1. Smarter retrieval, not more retrieval**
The investigator generates 3 query variants and retrieves top-7 for each. But the variants are often just keyword rearrangements. OPERA uses a dedicated rewrite agent that sees what was already retrieved and targets what's missing. We just added this capability but haven't tested it at scale. The rewrite agent should be the primary mechanism for improving per-hop accuracy.

**2. Answer verification before accepting**
Currently the investigator accepts any answer with status="sufficient" and non-empty answer_span. There's no cross-checking. A simple improvement: after getting an answer, do a verification retrieval with "[answer] [key terms from question]" to confirm the answer appears in independent evidence. This would catch many wrong bridge entities.

**3. Better decomposition depth**
188/1000 questions were under-decomposed (planner used fewer hops than needed). The v2.3 planner prompt has better depth rules but hasn't been tested at scale. Each missing hop means the investigator has to answer a question that's too complex for a single retrieval.

**4. Fix the near-misses with post-processing**
46 answers were too verbose (gold="1848", pred="February 2, 1848"). A simple answer post-processor that strips dates to years when the question asks "when was X" or strips location qualifiers ("County, State" -> "County") would recover ~20-30 EM points. This is low-hanging fruit.

**5. The synthesizer should be optional and conservative**
When the last hop has a good answer, use it directly. Only invoke the synthesizer when the last hop failed and you need to construct an answer from partial facts. The synthesizer introduces token cost and a risk of answer degradation.

**6. Adaptive routing for efficiency (thesis claim)**
MuSiQue is all multi-hop, so every question routes to DAG execution. To demonstrate the adaptive efficiency claim, you need to either:
- Test on a mixed dataset (some simple, some complex questions)
- Show that the system correctly identifies question complexity and allocates proportional effort (2-hop gets 2 agents, 4-hop gets 4)
- Compare token usage: our system should use ~7k tokens for 2-hop vs OPERA's ~20k for everything

**7. Parallel execution is under-utilized**
Nearly all plans are sequential chains. To get actual parallelism, the planner needs to identify independent sub-questions. For example: "What is the population of the city where X was born, and the area of the country where Y happened?" should decompose into two parallel branches. This is a thesis differentiator vs Plan*RAG and OPERA but requires planner prompt engineering.

### What NOT to do

- Don't add more LLM calls (verification agents, meta-reasoning, etc.) -- each call adds tokens and latency
- Don't put concrete answer examples in any prompt -- they WILL leak
- Don't trust 10q smoke tests -- too noisy, use 50q minimum
- Don't change the architecture radically -- the current structure works, the problem is per-hop accuracy
- Don't use the system `python3` -- use `.venv/bin/python`
- Don't confuse localhost:8003 (Qwen3-14B vLLM) with node408:8003 (retriever)

### The realistic path to 0.30 EM

Starting from v2.3 (EM=0.186 estimated, 0.34 on 50q):
1. Fix near-misses with answer post-processing: +20-30 EM → ~0.206-0.216
2. Fix blank answers with better fallback: +15-20 EM → ~0.221-0.236
3. Improved decomposition (already in v2.3 prompt): +10-15 EM → ~0.231-0.251
4. Better rewrite agent (now sees docs): +10-20 EM → ~0.241-0.271
5. Answer verification on bridge entities: +15-30 EM → ~0.256-0.301

The sum of these improvements could plausibly reach 0.30 EM. The biggest single lever is improving per-hop answer quality through better retrieval and verification.
