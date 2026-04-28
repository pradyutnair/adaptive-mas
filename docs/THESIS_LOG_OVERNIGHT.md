# Overnight log (autonomous work, deterministic-amas branch)

User went to sleep at ~22:10 with directive: improve SAAT, ensure single-agent collapse fires on easy questions, good decomposition, good rewriting, good disambiguation, paper-worthy method.

Locked architecture: SAAT (deterministic-amas) = planner (gpt-4o) -> DAG executor (parallel/sequential hops) with per-hop rewriter + direct_recovery fallback. Investigator gpt-4o-mini. Retriever node408 full Wiki18, top_k=5.

## Iter 0 (baseline before overnight): saat_4oplan_v2_4omini_sub_node408_top5_max3_opera40850_20260427_220419

50q OPERA-matched MuSiQue:
- contain 0.40 / EM 0.38 / F1 0.418 / mean 6,520 tokens / median 5,335 / max 13,380 / blanks 3
- routes: compositional 33, direct_recovery 17, simple 0
- rewrite events: 60

## Iter 1 (running): planner prompt v1

Hypothesis: planner over-classifies compositional (0/50 used simple), and mis-formulates final-hop questions on named-qualified-place answers (Mississippi case) and bridge interpretation (Lady Godiva birthplace = Mercia not Coventry).

Change: rewrote planner prompt with:
- explicit simple-route example (1-hop lookup with named subject)
- final-hop wording rules (do not echo a value already in question; preserve qualifiers like "delta", "kingdom")
- bridge interpretation example (Lady Godiva -> Mercia kingdom)
- disambiguation rules (era / role / location cues)

Run: saat_4oplan_v3_planprompt_4omini_sub_node408_top5_max3_opera40850_*

Pending eval.

## Iter 1 result (planner prompt v1)

50q OPERA-matched MuSiQue:
- contain 0.36 / EM 0.34 / F1 0.411 / mean 6,868 / blanks 4
- routes: compositional 34, direct_recovery 14, simple 2 (planner did fire simple route, but only 2/50)
- rewrite events: 57

**Regression vs Iter 0**: contain -0.04, EM -0.04, blanks +1. The longer planner prompt biased compositional decompositions worse (probably the named-place-qualifier rule and bridge-interpretation example.) Reverted to v0 planner prompt.

## Iter 2 (running): rewriter prompt v1

Hypothesis: rewriter is the heaviest rescue mechanism (60 events on 50q v0). Make it more decisive about query DIVERSIFICATION, granularity, and bridge-entity inclusion.

Change: rewriter prompt now contains explicit failure-diagnosis branches: paraphrase->term-dense, wrong-granularity->add granularity term, bridge-instead-of-attribute->bridge-name+relation, repeated-empty->add disambiguating anchor, named-place-qualifier->include qualifier.

Run: saat_4oplan_v4_rewriteprompt_4omini_sub_node408_top5_max3_opera40850_*

Pending eval.

## PAUSED at user request (resume tomorrow)

State at pause:
- Branch: deterministic-amas
- Prompts: BOTH REVERTED to v0 (known-good baseline). planner.txt and rewrite.txt match git HEAD.
- Locked config (use this tomorrow): `configs/_runtime/saat_4oplan_4omini_sub_node408_top5_max3_v2.yaml`
  - planner: gpt-4o
  - investigator: gpt-4o-mini
  - retriever: node408 top_k=5
  - max_searches_per_subagent: 3, max_hop_attempts: 3, final_recovery_attempts: 4

## Verified locked baseline (tomorrow's reference)

50q OPERA-matched MuSiQue at `results/saat/saat_4oplan_v2_4omini_sub_node408_top5_max3_opera40850_20260427_220419/`:
- contain 0.40, EM 0.38, F1 0.418
- mean 6,520 tokens, median 5,335, max 13,380
- blanks 3 (terminal-hop stuck), answered 47/50
- routes: compositional 33, direct_recovery 17, simple 0
- rewrite events: 60

## Tomorrow's plan (in order)

1. Verify state. Confirm planner.txt and rewrite.txt match HEAD on deterministic-amas. Confirm v2 config still exists. Re-pull the v2 50q baseline numbers to align expectations.
2. Iter 2 retry (rewriter prompt) more conservatively. Only change ONE clause at a time. Test 50q. The previous rewriter rewrite was incomplete (killed at 37/50), so no clean number for it.
3. Eliminate the 3 blanks. Two approaches to test:
   - bump max_hop_attempts from 3 to 4 (one more rewrite chance per stuck hop).
   - add a "desperate-mode final" path: when DAG has stuck terminal node AND direct_recovery fails, run investigator on the original question with all resolved hop answers as hints; whatever it returns, emit. Better than blank.
4. Try fixing single-agent collapse activation. Currently 0/50 use the simple route; loosening planner triggers regressed. Alternative: add a pre-planner direct probe in pipeline.run() that runs investigator on the original question first; if it returns a high-confidence non-empty answer, finalize. Adds ~1 LLM call per question but might convert some compositional+recovery questions to cheap simple. Test 50q for both contain and cost.
5. Once 50q clears: contain >= 0.42, mean <= 8k, blanks <= 2, then scale to 1000q MuSiQue + 50q HotpotQA + 50q 2Wiki.
6. Update docs/THESIS.md with the SAAT architecture as main row and 1000q numbers.

## Iteration log this evening

- Iter 0 (locked baseline v2): contain 0.40, mean 6.5k, blanks 3 [LOCKED]
- Iter 1 (planner prompt v1, longer with examples): contain 0.36, regressed, REVERTED
- Iter 2 (rewriter prompt v1, term-dense diagnosis branches): killed at 37/50 due to user pause, no clean number, REVERTED to be safe

## Hard rules to keep tomorrow

- Stay on deterministic-amas branch.
- Always use node408 retriever.
- gpt-4o planner + gpt-4o-mini investigator (locked).
- Never run 1000q without 50q clearing the bar.
- Save every run dir + config_used.yaml.
- Make at most ONE prompt change per iteration so the regression vs improvement signal is clean.

## Iter 3 result (REVERTED): pre-planner direct probe v1

Patch added a `_pre_probe()` method to `pipeline.py` (95 lines) that runs the investigator on the original question before the planner. Threshold: `pre_probe_confidence=0.78`.

50q OPERA-matched MuSiQue at `results/saat/saat_4oplan_v5_preprobe_4omini_sub_node408_top5_max3_opera40850_*`:
- contain 0.20 (-0.20 vs v2 baseline) -- DISASTER
- EM 0.18 (-0.20)
- mean 5,217 tokens (-1.3k cheaper, but quality crater)
- routes: simple=17, compositional=22, direct_recovery=11
- 17/17 probe attempts accepted (100% accept rate -- the threshold gate didn't gate anything)

Per-route on this run:
- simple (probe accepted, n=17): contain 0.176
- compositional (DAG ran, n=22): contain 0.227
- direct_recovery (n=11): contain 0.182

**Diagnosis of the failure mode:**

Inspection of the 17 probe-accepted questions: all 17 had `confidence=1.00`. The investigator emits maximum confidence too easily. The probe accepted bridge-entity answers as final answers on multi-hop questions:

- "Who married the actor from Terminator?" -> probe said `Linda Hamilton` (an actor in Terminator; not the spouse of an actor). Gold: `Maria Shriver` (Schwarzenegger's spouse).
- "How many times did the plague occur in the city where the painter of The Battle..." -> probe said `during the plague epidemic` (a phrase from a chunk, not the count). Gold: `22`.
- "Where was the person who wrote about the rioting being a dividing factor..." -> probe said `University of Cambridge`. Gold: `University of Glasgow`.

The probe was right on 3/17: "Erik Hort birthplace -> Rockland County", "SEAL acronym -> Sea, Air, and Land", "Turn Me On writer by Norah Jones -> John D. Loudermilk". These are GENUINE 1-2 hop questions where retrieval surfaces the answer in one search.

**Lesson:** confidence-threshold-only gating does not work. The investigator's confidence is poorly calibrated; it returns 1.0 even on bridge-entity-confounded questions. Need a SECOND signal:

Options for a future smarter probe gate:
A. Question-shape filter: only allow probe if the question lacks "X of Y" / nested-of / "the X who ..." / "the place where X did Y" structures.
B. Validator LLM call: after probe returns an answer, run a separate small LLM call asking "Does this answer match the EXACT target asked by the question, or is it a bridge entity?" Reject if the validator says bridge.
C. Disable probe on MuSiQue (always plan), enable on HotpotQA / 2Wiki where direct lookups are more common.

Reverted: `git checkout HEAD -- src/amas/pipeline.py`.

## State at end of overnight session

- v2 locked baseline holds: contain 0.40, EM 0.38, mean 6.5k tokens, blanks 3.
- All prompts at v0 (HEAD).
- pipeline.py at v0 (no pre-probe).
- Branch: deterministic-amas, working tree clean except for new configs and result dirs.

## Tomorrow's plan (revised)

1. (Optional, before any probe attempt) Build a question-shape classifier OR a validator. Pick option C above as the safest: run the probe ONLY when the question has no nested-of / no "the X who" / no temporal-conditional structure. This is a regex/heuristic gate, not a learned classifier. Test on MuSiQue 50q first; if it doesn't crater contain, run cross-dataset to check whether HotpotQA / 2Wiki get useful simple-route activation.

2. Consider abandoning pre-probe entirely. The post-DAG `direct_recovery` is already a single-agent fallback; the current architecture has TWO topology paths in active use on MuSiQue (compositional 33, direct_recovery 17). The pre-emptive single-agent collapse is structurally desirable but quantitatively might cost more than it gives on bridge-heavy datasets.

3. (Independent of the probe question) Run the locked v2 config on:
   - 1000q MuSiQue OPERA-matched (~22 min wall-clock).
   - 50q HotpotQA OPERA-matched.
   - 50q 2Wiki OPERA-matched.

4. Update docs/THESIS.md with the locked architecture, locked numbers, and topology-distribution story.
