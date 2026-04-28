import json, re, string

def norm(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split()).strip()

preds = [json.loads(l) for l in open('results/amas_v2_1000q/predictions.jsonl')]
qs = json.load(open('data/musique/questions_1000_seedfull_combined.json'))
gold = {str(q['id']): q['answer'] for q in qs}

# Focus on 2-hop questions where bridge was verified but final answer wrong
examples = []
for p in preds:
    g = gold.get(str(p['id']), '')
    if norm(p['answer']) == norm(g): continue
    if not p['answer'].strip(): continue
    meta = p.get('metadata', {})
    statuses = meta.get('extras', {}).get('node_statuses', {})
    plan = meta.get('extras', {}).get('plan', {})
    subgoals = plan.get('subgoals', [])
    if len(subgoals) != 2: continue
    s1 = str(statuses.get('1', statuses.get(1, '')))
    s2 = str(statuses.get('2', statuses.get(2, '')))
    if s1 != 'verified': continue
    # Bridge verified, look at what happened
    trace = meta.get('step_trace', [])
    hop1_ans = ""
    hop2_ans = ""
    for t in trace:
        md = t.get('metadata', {})
        if md.get('subgoal_id') == 1 and md.get('status') == 'verified':
            hop1_ans = md.get('answer', '')
        if md.get('subgoal_id') == 2:
            hop2_ans = md.get('answer', '')
    examples.append({
        'qid': p['id'],
        'question': p['question'][:80],
        'hop1_q': subgoals[0].get('question', '')[:60],
        'hop1_ans': hop1_ans,
        'hop2_q': subgoals[1].get('question', '')[:60],
        'hop2_status': s2,
        'pred': p['answer'],
        'gold': g,
    })

print(f"2-hop questions with bridge verified but final wrong: {len(examples)}")
print()
for e in examples[:20]:
    print(f"Q: {e['question']}")
    print(f"  Hop1: {e['hop1_q']} -> {e['hop1_ans']}")
    print(f"  Hop2: {e['hop2_q']} [{e['hop2_status']}]")
    print(f"  pred={e['pred']}  gold={e['gold']}")
    print()
