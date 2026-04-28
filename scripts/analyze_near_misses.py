import json, re, string

def norm(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split()).strip()

preds = [json.loads(l) for l in open('results/amas_v2_1000q/predictions.jsonl')]
qs = json.load(open('data/musique/questions_1000_seedfull_combined.json'))
gold = {str(q['id']): q['answer'] for q in qs}

pred_in_gold = 0
gold_in_pred = 0
both = 0

for p in preds:
    g = gold.get(str(p['id']), '')
    np_, ng = norm(p['answer']), norm(g)
    if np_ == ng: continue
    if not p['answer'].strip(): continue
    pig = np_ in ng
    gip = ng in np_
    if pig and gip: both += 1
    elif pig: pred_in_gold += 1
    elif gip: gold_in_pred += 1

print(f"pred subset of gold (too short): {pred_in_gold}")
print(f"gold subset of pred (too verbose): {gold_in_pred}")
print(f"both directions: {both}")

# Show too-verbose examples
print("\nToo verbose (gold in pred):")
for p in preds:
    g = gold.get(str(p['id']), '')
    np_, ng = norm(p['answer']), norm(g)
    if np_ == ng: continue
    if not p['answer'].strip(): continue
    if ng in np_ and np_ != ng:
        print(f'  pred="{p["answer"]}" gold="{g}"')
