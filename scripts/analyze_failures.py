import json, re, string
from collections import Counter

def norm(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split()).strip()

preds = [json.loads(l) for l in open('results/amas_v2_1000q/predictions.jsonl')]
qs = json.load(open('data/musique/questions_1000_seedfull_combined.json'))
gold = {str(q['id']): q['answer'] for q in qs}

blank = near_miss = synth_degraded = bridge_fail = all_fail = wrong_final = 0
synth_examples = []
near_examples = []

for p in preds:
    g = gold.get(str(p['id']), '')
    np_, ng = norm(p['answer']), norm(g)
    if np_ == ng: continue
    meta = p.get('metadata', {})
    if not p['answer'].strip():
        blank += 1; continue
    if ng in np_ or np_ in ng:
        near_miss += 1
        near_examples.append(f'  pred="{p["answer"]}" gold="{g}"')
        continue
    for t in meta.get('step_trace', []):
        if t.get('action') == 'synthesize':
            raw = t.get('metadata', {}).get('raw_answer', '')
            synth = t.get('metadata', {}).get('synth_answer', '')
            if norm(raw) == ng and norm(synth) != ng:
                synth_degraded += 1
                synth_examples.append(f'  raw="{raw}" synth="{synth}" gold="{g}"')
                break
    statuses = meta.get('extras', {}).get('node_statuses', {})
    if statuses:
        str_statuses = {str(k): v for k, v in statuses.items()}
        first_failed = str_statuses.get('1') in ('failed', 'blocked')
        all_bad = all(v in ('failed', 'blocked') for v in str_statuses.values())
        if all_bad: all_fail += 1
        elif first_failed: bridge_fail += 1
        else: wrong_final += 1

print(f"Correct: 186/1000")
print(f"Blank: {blank}")
print(f"Near-miss (contain not EM): {near_miss}")
print(f"Synth degraded correct->wrong: {synth_degraded}")
print(f"Bridge entity failed: {bridge_fail}")
print(f"All hops failed: {all_fail}")
print(f"Bridge OK, final wrong: {wrong_final}")

print(f"\nSynth degradation ({synth_degraded}):")
for e in synth_examples[:10]: print(e)
print(f"\nNear-miss ({near_miss}):")
for e in near_examples[:20]: print(e)

# By hop count from question ID pattern
hop_em = Counter()
hop_n = Counter()
for p in preds:
    qid = p['id']
    if '4hop' in qid: h = 4
    elif '3hop' in qid: h = 3
    elif '2hop' in qid: h = 2
    else: h = 0
    hop_n[h] += 1
    g = gold.get(str(p['id']), '')
    if norm(p['answer']) == norm(g): hop_em[h] += 1

print(f"\nEM by actual question difficulty:")
for h in sorted(hop_n):
    print(f"  {h}-hop: {hop_em[h]}/{hop_n[h]} = {hop_em[h]/hop_n[h]:.3f}")

# Plan hop vs actual hop mismatch
over_decomp = under_decomp = 0
for p in preds:
    qid = p['id']
    if '4hop' in qid: actual = 4
    elif '3hop' in qid: actual = 3
    elif '2hop' in qid: actual = 2
    else: continue
    plan = p.get('metadata', {}).get('extras', {}).get('plan', {})
    planned = len(plan.get('subgoals', []))
    if planned > actual: over_decomp += 1
    elif planned < actual: under_decomp += 1

print(f"\nDecomposition mismatch:")
print(f"  Over-decomposed (planned > actual): {over_decomp}")
print(f"  Under-decomposed (planned < actual): {under_decomp}")

out = json.dumps({"blank": blank, "near_miss": near_miss, "synth_degraded": synth_degraded,
                   "bridge_fail": bridge_fail, "all_fail": all_fail, "wrong_final": wrong_final})
open('/tmp/failure_analysis.json', 'w').write(out)
