#!/usr/bin/env python3
"""Scaling-law analyzer for RQ3 (inference-time scaling laws).

Computes per-question recursion-depth from existing prediction metadata and
groups by depth to show:
  - depth distribution (% of questions at each depth)
  - mean tokens per depth
  - mean wallclock per depth
  - norm_em per depth

Recursion-depth derivation (training-free, retrieval-grounded):
  depth = 0  iff sas_collapse=True
           1  else if no recursion fired (no solver-refine, no synth-recursion>1)
           2  else if any solver-refine fired (low-conf hop got refined)
           3  else if synth-recursion ran (>=2 rounds of synth)
           4  else if bridge-resolved (extra pre-step) — combined with 1-3

For now, depth captures monotonic effort: 0 (single-agent) -> 1 (MAS plain)
-> 2 (MAS+solver-refine) -> 3 (MAS+synth-recurse) -> 4 (+bridge pre-resolve).
"""
import argparse
import json
import re
import string
from pathlib import Path

_ARTICLES = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_PUNCT = set(string.punctuation)


def normalize(s):
    s = (s or '').lower()
    s = _ARTICLES.sub(' ', s)
    s = ''.join(c if c not in _PUNCT else ' ' for c in s)
    return ' '.join(s.split()).strip()


def f1_token(pred, gold):
    p = set(normalize(pred).split())
    g = set(normalize(gold).split())
    if not p or not g:
        return 0.0
    common = p & g
    if not common:
        return 0.0
    pr = len(common) / len(p)
    rc = len(common) / len(g)
    return 2 * pr * rc / (pr + rc)


def derive_depth(row):
    m = row.get('metadata') or {}
    if m.get('sas_collapse'):
        return 0
    has_bridge = bool(m.get('bridge_resolved') or m.get('bridge_resolver_tokens', 0))
    findings = m.get('findings') or []
    # heuristic: solver-refine fires when extraction_tokens unusually high relative to chunks
    # we don't track refine count directly, so infer from solver_tokens > 4 * n_solvers heuristic
    n_solvers = m.get('n_solvers_invoked', 0) or 1
    solver_tokens = m.get('solver_tokens', 0) or 0
    avg_per_solver = solver_tokens / max(n_solvers, 1)
    # solver-refine doubles the per-solver token cost; threshold ~3000
    has_solver_refine = avg_per_solver >= 3000
    has_synth_recurse = (m.get('synth_tokens', 0) or 0) >= 3000  # synth-recurse doubles synth cost
    base = 1
    if has_solver_refine:
        base = 2
    if has_synth_recurse:
        base = max(base, 3)
    if has_bridge:
        base = max(base, 4)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.predictions) if l.strip()]
    by_depth = {}
    for r in rows:
        d = derive_depth(r)
        by_depth.setdefault(d, []).append(r)

    print(f'\n=== RQ3 Scaling-Law (recursion depth) on {len(rows)} predictions ===\n')
    print(f'{"depth":<6}{"count":<8}{"share":<10}{"mean_tok":<12}{"mean_sec":<10}{"em":<8}{"f1":<8}{"contain":<10}')
    print('-' * 72)
    overall_em = 0
    overall_f1 = 0
    overall_contain = 0
    overall_tokens = 0
    overall_secs = 0
    for d in sorted(by_depth.keys()):
        bucket = by_depth[d]
        n = len(bucket)
        em = 0
        f1 = 0
        contain = 0
        tok = 0
        sec = 0
        for r in bucket:
            pred = r.get('answer', '') or r.get('prediction', '')
            gold = r.get('gold_answer', '')
            np_, ng = normalize(pred), normalize(gold)
            if np_ and ng and np_ == ng:
                em += 1
            if np_ and ng and (ng in np_ or np_ in ng):
                contain += 1
            f1 += f1_token(pred, gold)
            tok += (r.get('metadata') or {}).get('total_tokens', 0)
            sec += (r.get('metadata') or {}).get('wallclock_seconds', 0)
        share = n / max(len(rows), 1)
        print(f'{d:<6}{n:<8}{share:<10.1%}{tok / max(n, 1):<12.0f}{sec / max(n, 1):<10.1f}{em / max(n, 1):<8.3f}{f1 / max(n, 1):<8.3f}{contain / max(n, 1):<10.3f}')
        overall_em += em
        overall_f1 += f1
        overall_contain += contain
        overall_tokens += tok
        overall_secs += sec

    n = len(rows)
    print('-' * 72)
    print(f'{"all":<6}{n:<8}{1.0:<10.0%}{overall_tokens / max(n, 1):<12.0f}{overall_secs / max(n, 1):<10.1f}{overall_em / max(n, 1):<8.3f}{overall_f1 / max(n, 1):<8.3f}{overall_contain / max(n, 1):<10.3f}')

    # SAS-collapse analysis (if any)
    sas_rows = [r for r in rows if (r.get('metadata') or {}).get('sas_collapse')]
    if sas_rows:
        sas_em = sum(1 for r in sas_rows if normalize(r.get('answer', '')) == normalize(r.get('gold_answer', '')) and r.get('answer'))
        print(f'\nSAS-collapse precision: {sas_em}/{len(sas_rows)} = {sas_em / len(sas_rows):.1%}  (false-positive rate {1 - sas_em / len(sas_rows):.1%})')

    # Topology distribution
    from collections import Counter
    topo_counter = Counter((r.get('metadata') or {}).get('topology', '') for r in rows)
    print('\nTopology distribution:')
    for t, c in topo_counter.most_common():
        print(f'  {t!r}: {c} ({c / max(len(rows), 1):.1%})')

    if args.output:
        out = {
            'by_depth': {
                str(d): {
                    'n': len(bucket),
                    'share': len(bucket) / max(len(rows), 1),
                    'mean_tokens': sum(((r.get('metadata') or {}).get('total_tokens', 0)) for r in bucket) / max(len(bucket), 1),
                    'mean_secs': sum(((r.get('metadata') or {}).get('wallclock_seconds', 0)) for r in bucket) / max(len(bucket), 1),
                    'norm_em': sum(1 for r in bucket if normalize(r.get('answer', '')) == normalize(r.get('gold_answer', '')) and r.get('answer')) / max(len(bucket), 1),
                    'token_f1': sum(f1_token(r.get('answer', ''), r.get('gold_answer', '')) for r in bucket) / max(len(bucket), 1),
                    'contain': sum(1 for r in bucket if r.get('answer') and normalize(r.get('gold_answer', '')) and (normalize(r.get('gold_answer', '')) in normalize(r.get('answer', '')) or normalize(r.get('answer', '')) in normalize(r.get('gold_answer', '')))) / max(len(bucket), 1),
                }
                for d, bucket in by_depth.items()
            },
            'overall': {
                'n': len(rows),
                'mean_tokens': overall_tokens / max(len(rows), 1),
                'mean_secs': overall_secs / max(len(rows), 1),
                'norm_em': overall_em / max(len(rows), 1),
                'token_f1': overall_f1 / max(len(rows), 1),
                'contain': overall_contain / max(len(rows), 1),
            },
            'topology': {t: c for t, c in topo_counter.most_common()},
            'sas_precision': {
                'count': len(sas_rows),
                'em': sum(1 for r in sas_rows if normalize(r.get('answer', '')) == normalize(r.get('gold_answer', '')) and r.get('answer')),
            } if sas_rows else None,
        }
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f'\nwrote: {args.output}')


if __name__ == '__main__':
    main()
