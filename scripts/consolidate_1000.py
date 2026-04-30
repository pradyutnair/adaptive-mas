#!/usr/bin/env python3
"""Consolidate the 3 prediction files into a final 1000-question prediction set.

For each question ID in the 1000q gold set, pick the best prediction:
- Prefer non-empty answer
- If multiple non-empty, prefer the most recent (re-run > remainder > partial)
- If all empty, leave empty
"""
import json
ROOT = '/local/yzheng/pnair/workspace/adaptive-mas'

all_qs = json.load(open(f'{ROOT}/data/musique/questions.json'))
gold_ids = [q['id'] for q in all_qs]

# Load in priority order: latest first
sources = [
    ('rerun', f'{ROOT}/results/amas_pro_synthunion_qwen3_14b_4omini_musique_1000q_empties_rerun/predictions.jsonl'),
    ('remainder', f'{ROOT}/results/amas_pro_synthunion_qwen3_14b_4omini_musique_1000q_remainder596/predictions.jsonl'),
    ('partial', f'{ROOT}/results/amas_pro_synthunion_qwen3_14b_4omini_musique_1000q/predictions.jsonl'),
]

best = {}
for label, path in sources:
    try:
        for line in open(path):
            if not line.strip(): continue
            r = json.loads(line)
            qid = r['id']
            ans = (r.get('answer') or '').strip()
            # only update if this is the first OR current best is empty AND this has answer
            if qid not in best:
                best[qid] = (label, r)
            elif not (best[qid][1].get('answer') or '').strip() and ans:
                best[qid] = (label, r)
    except FileNotFoundError:
        print(f'(missing: {path})')

# Build the final prediction file in gold order
out_rows = []
src_counter = {}
for qid in gold_ids:
    if qid in best:
        label, r = best[qid]
        src_counter[label] = src_counter.get(label, 0) + 1
        out_rows.append(r)
    else:
        # Question wasn't in any prediction file — emit empty stub
        gold_q = next((q for q in all_qs if q['id'] == qid), {})
        out_rows.append({
            'id': qid,
            'question': gold_q.get('question', ''),
            'gold_answer': gold_q.get('answer', ''),
            'answer': '',
            'prediction': '',
            'metadata': {'error': 'no_prediction_in_any_file'},
        })
        src_counter['missing'] = src_counter.get('missing', 0) + 1

out = f'{ROOT}/results/amas_pro_synthunion_qwen3_14b_4omini_musique_1000q/predictions_final_1000.jsonl'
with open(out, 'w') as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

answered = sum(1 for r in out_rows if (r.get('answer') or '').strip())
print(f'final 1000 predictions: {len(out_rows)}, answered: {answered}/{len(out_rows)}')
print(f'sources: {src_counter}')
print(f'wrote: {out}')
