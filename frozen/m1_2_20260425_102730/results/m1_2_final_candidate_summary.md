# m1.2 isolated-retrieval summary

| run | n | contain | avg tokens | empty |
|---|---:|---:|---:|---:|
| isolated_nonempty_1000 | 1000 | 0.323 | 23439.6 | 2 |
| fullbudget_max3_1000 | 1000 | 0.387 | 29563.4 | 0 |

## fullbudget_max3 per-call

| calls | n | contain | avg tokens |
|---:|---:|---:|---:|
| 1 | 492 | 0.563 | 14417.5 |
| 2 | 138 | 0.116 | 34853.6 |
| 3 | 228 | 0.268 | 43988.0 |
| 4 | 142 | 0.232 | 53738.9 |

## fullbudget_max3 per-hop

| hop | n | contain | avg tokens |
|---|---:|---:|---:|
| 2hop | 518 | 0.500 | 23567.7 |
| 3hop | 316 | 0.316 | 34570.0 |
| 4hop | 166 | 0.169 | 38742.3 |

## diagnosis

- Candidate: `fullbudget_max3`, 1000q MuSiQue contain 0.387 at 29.56k tokens.
- Main gain: fixes the destructive one-step recurse budget bug; improves isolated retrieval baseline from 0.323 to 0.387.
- Open issue: calls=2 slice remains weak (n=138, contain 0.116), causing the remaining miss to 0.40.
- Rejected variants: floor2, top4, nopriorsnips all failed to beat max3 on 50q.
