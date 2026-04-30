#!/usr/bin/env python3
"""Download MuSiQue training split (answerable) via HF datasets and save
as JSON list compatible with our runner schema.

Output: data/musique/musique_train_500.json (sampled stratified by hop).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

random.seed(42)

try:
    from datasets import load_dataset
except ImportError:
    print('Need: pip install datasets', file=sys.stderr)
    sys.exit(1)

ds = load_dataset('dgslibisey/MuSiQue', split='train', cache_dir='/local/yzheng/pnair/.hf_cache')

print('total train rows:', len(ds))
print('schema sample:', {k: type(v).__name__ for k, v in ds[0].items()})

buckets = defaultdict(list)
for row in ds:
    qid = row.get('id', '')
    if qid.startswith('2hop'):
        hop = 2
    elif qid.startswith('3hop'):
        hop = 3
    elif qid.startswith('4hop'):
        hop = 4
    else:
        continue
    buckets[hop].append({'id': qid, 'question': row['question'], 'answer': row['answer']})

for h in sorted(buckets):
    random.shuffle(buckets[h])
    print(f'  {h}-hop: {len(buckets[h])}')

target = {2: 260, 3: 160, 4: 80}
selected = []
for h, n in target.items():
    selected.extend(buckets[h][:n])

random.shuffle(selected)
out = Path('data/musique/musique_train_500.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(selected, indent=2))
print(f'wrote {len(selected)} examples to {out}')
