import json, re, string
from collections import Counter

def norm(s):
    s = (s or '').lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())
def em(p, g): return float(norm(p) == norm(g))
def f1(p, g):
    pt, gt = norm(p).split(), norm(g).split()
    if not pt or not gt: return 0.0
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0: return 0.0
    pr, rc = n/len(pt), n/len(gt)
    return 2*pr*rc/(pr+rc)
def contain(p, g): return float(norm(g) in norm(p))

base = {}
for l in open('frozen/amas-final/results/balanced/hotpot/predictions.jsonl'):
    r = json.loads(l); base[r['id']] = r
import sys
pilot_path = sys.argv[1] if len(sys.argv) > 1 else 'compiled/pilot50_balanced/predictions.jsonl'
pilot = {}
for l in open(pilot_path):
    r = json.loads(l); pilot[r['id']] = r

ids = list(pilot.keys())
print(f'pilot={len(pilot)} matched={sum(1 for q in ids if q in base)}/{len(ids)}')

def score(r):
    a = r.get('answer') or r.get('prediction') or ''
    g = r['gold_answer']
    return em(a,g), f1(a,g), contain(a,g), int(r.get('metadata',{}).get('total_tokens',0))

be_sum=bf_sum=bc_sum=bt_sum=0
pe_sum=pf_sum=pc_sum=pt_sum=0
changed=[]
for qid in ids:
    b = base[qid]; p = pilot[qid]
    be, bf, bc, bt = score(b); pe, pf, pc, pt = score(p)
    be_sum+=be; bf_sum+=bf; bc_sum+=bc; bt_sum+=bt
    pe_sum+=pe; pf_sum+=pf; pc_sum+=pc; pt_sum+=pt
    if be != pe:
        changed.append((qid, be, pe, (b.get('answer') or b.get('prediction') or '')[:40], (p.get('answer') or p.get('prediction') or '')[:40], p['gold_answer'][:40]))
n = len(ids)
print()
print(f'{"":10s} {"EM":>6s} {"F1":>6s} {"CT":>6s} {"avg_tok":>9s}')
print(f'{"baseline":10s} {be_sum/n:6.3f} {bf_sum/n:6.3f} {bc_sum/n:6.3f} {bt_sum/n:9.0f}')
print(f'{"pilot":10s} {pe_sum/n:6.3f} {pf_sum/n:6.3f} {pc_sum/n:6.3f} {pt_sum/n:9.0f}')

regressions=[r for r in changed if r[1]==1 and r[2]==0]
gains=[r for r in changed if r[1]==0 and r[2]==1]
print(f'\nEM-changed: {len(changed)}/{n}  regressions={len(regressions)}  gains={len(gains)}')
for qid, be, pe, ba, pa, g in regressions[:10]:
    print(f'  REGRESS {qid} base={ba!r} pilot={pa!r} gold={g!r}')
for qid, be, pe, ba, pa, g in gains[:8]:
    print(f'  GAIN    {qid} base={ba!r} pilot={pa!r} gold={g!r}')
