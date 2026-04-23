#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

def load_jsonl(path):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def pct(vals,p):
    vals=sorted(vals)
    if len(vals)==1:
        return float(vals[0])
    k=(len(vals)-1)*(p/100.0)
    f=int(k)
    c=min(f+1,len(vals)-1)
    if f==c:
        return float(vals[f])
    d=k-f
    return float(vals[f]*(1-d)+vals[c]*d)

def summarize(rows, label):
    totals=[]; orch=[]; sub=[]; wall=[]
    for r in rows:
        md=r.get('metadata',{}) or {}
        totals.append(float(md.get('total_tokens',0) or 0))
        orch.append(float(md.get('orchestrator_tokens',0) or 0))
        sub.append(float(md.get('subagent_tokens',0) or 0))
        wall.append(float(md.get('wallclock_seconds',0) or 0))
    return {
        'label': label,
        'n': len(rows),
        'mean_total_tokens': round(sum(totals)/len(totals),2),
        'median_total_tokens': round(statistics.median(totals),2),
        'p90_total_tokens': round(pct(totals,90),2),
        'mean_orchestrator_tokens': round(sum(orch)/len(orch),2),
        'mean_subagent_tokens': round(sum(sub)/len(sub),2),
        'mean_wallclock_seconds': round(sum(wall)/len(wall),2),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    specs={
        'sufficiency_v6_musique':'paper_results/sufficiency_1000q_20260418_215804/musique/sufficiency/predictions.jsonl',
        'sufficiency_v6_hotpotqa':'paper_results/sufficiency_1000q_20260418_215804/hotpotqa/sufficiency/predictions.jsonl',
        'sufficiency_v6_2wikimultihop':'paper_results/sufficiency_1000q_20260418_215804/2wikimultihop/sufficiency/predictions.jsonl',
        'iter55_musique':'results/audit/iter55_predictions.jsonl',
    }
    out={}
    for key,src in specs.items():
        out[key]=summarize(load_jsonl(src), key)
    Path(args.output).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main()
