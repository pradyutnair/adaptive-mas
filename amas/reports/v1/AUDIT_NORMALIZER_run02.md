# Phase 0 Normalizer Audit

AMAS stored summaries (current headline, post in-pipeline normalize_answer_span).
HERA-repro raw preds rescored under three regimes:
- R1 raw    : skip normalize_answer_span. score raw vs gold via metric.normalize_answer.
- R2 cap8   : normalize_answer_span(max_words=8). current report uses this for HERA-renormalized.
- R3 cap64  : normalize_answer_span(max_words=64). cap effectively off; tests how much the 8-word cap penalises HERA verbose prose.

Delta = AMAS-X minus HERA in given regime, percentage points.

## musique (n=1000)

| method | EM | F1 | contain | Acc | tok |
|---|---:|---:|---:|---:|---:|
| HERA R1 raw | 0.126 | 0.215 | 0.157 | 0.157 | 6684 |
| HERA R2 cap8 | 0.126 | 0.215 | 0.157 | 0.157 | 6684 |
| HERA R3 cap64 | 0.126 | 0.215 | 0.157 | 0.157 | 6684 |
| AMAS-off | 0.112 | 0.209 | 0.145 | 0.145 | 9651 |
| AMAS-conformal | 0.104 | 0.193 | 0.133 | 0.133 | 5979 |
| AMAS-bayesian | 0.073 | 0.156 | 0.098 | 0.098 | 870 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | -1.4 | -0.5 | -1.2 | -1.2 |
| AMAS-off | R2 cap8 | -1.4 | -0.5 | -1.2 | -1.2 |
| AMAS-off | R3 cap64 | -1.4 | -0.5 | -1.2 | -1.2 |
| AMAS-conformal | R1 raw | -2.2 | -2.1 | -2.4 | -2.4 |
| AMAS-conformal | R2 cap8 | -2.2 | -2.1 | -2.4 | -2.4 |
| AMAS-conformal | R3 cap64 | -2.2 | -2.1 | -2.4 | -2.4 |
| AMAS-bayesian | R1 raw | -5.3 | -5.9 | -5.9 | -5.9 |
| AMAS-bayesian | R2 cap8 | -5.3 | -5.9 | -5.9 | -5.9 |
| AMAS-bayesian | R3 cap64 | -5.3 | -5.9 | -5.9 | -5.9 |

## hotpotqa (n=1000)

| method | EM | F1 | contain | Acc | tok |
|---|---:|---:|---:|---:|---:|
| HERA R1 raw | 0.374 | 0.491 | 0.436 | 0.436 | 6347 |
| HERA R2 cap8 | 0.375 | 0.491 | 0.436 | 0.436 | 6347 |
| HERA R3 cap64 | 0.375 | 0.491 | 0.436 | 0.436 | 6347 |
| AMAS-off | 0.372 | 0.513 | 0.461 | 0.461 | 9141 |
| AMAS-conformal | 0.355 | 0.491 | 0.447 | 0.447 | 3775 |
| AMAS-bayesian | 0.316 | 0.455 | 0.417 | 0.417 | 852 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | -0.2 | +2.2 | +2.5 | +2.5 |
| AMAS-off | R2 cap8 | -0.3 | +2.2 | +2.5 | +2.5 |
| AMAS-off | R3 cap64 | -0.3 | +2.2 | +2.5 | +2.5 |
| AMAS-conformal | R1 raw | -1.9 | +0.0 | +1.1 | +1.1 |
| AMAS-conformal | R2 cap8 | -2.0 | +0.1 | +1.1 | +1.1 |
| AMAS-conformal | R3 cap64 | -2.0 | +0.1 | +1.1 | +1.1 |
| AMAS-bayesian | R1 raw | -5.8 | -3.5 | -1.9 | -1.9 |
| AMAS-bayesian | R2 cap8 | -5.9 | -3.5 | -1.9 | -1.9 |
| AMAS-bayesian | R3 cap64 | -5.9 | -3.5 | -1.9 | -1.9 |

## 2wikimultihop (n=1000)

| method | EM | F1 | contain | Acc | tok |
|---|---:|---:|---:|---:|---:|
| HERA R1 raw | 0.239 | 0.307 | 0.301 | 0.301 | 6694 |
| HERA R2 cap8 | 0.239 | 0.307 | 0.301 | 0.301 | 6694 |
| HERA R3 cap64 | 0.239 | 0.307 | 0.301 | 0.301 | 6694 |
| AMAS-off | 0.293 | 0.409 | 0.410 | 0.410 | 9975 |
| AMAS-conformal | 0.261 | 0.378 | 0.374 | 0.374 | 6393 |
| AMAS-bayesian | 0.239 | 0.345 | 0.329 | 0.329 | 906 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | +5.4 | +10.3 | +10.9 | +10.9 |
| AMAS-off | R2 cap8 | +5.4 | +10.3 | +10.9 | +10.9 |
| AMAS-off | R3 cap64 | +5.4 | +10.3 | +10.9 | +10.9 |
| AMAS-conformal | R1 raw | +2.2 | +7.1 | +7.3 | +7.3 |
| AMAS-conformal | R2 cap8 | +2.2 | +7.1 | +7.3 | +7.3 |
| AMAS-conformal | R3 cap64 | +2.2 | +7.1 | +7.3 | +7.3 |
| AMAS-bayesian | R1 raw | +0.0 | +3.8 | +2.8 | +2.8 |
| AMAS-bayesian | R2 cap8 | +0.0 | +3.8 | +2.8 | +2.8 |
| AMAS-bayesian | R3 cap64 | +0.0 | +3.8 | +2.8 | +2.8 |

## bamboogle (n=125)

| method | EM | F1 | contain | Acc | tok |
|---|---:|---:|---:|---:|---:|
| HERA R1 raw | 0.368 | 0.519 | 0.408 | 0.408 | 6244 |
| HERA R2 cap8 | 0.368 | 0.519 | 0.408 | 0.408 | 6244 |
| HERA R3 cap64 | 0.368 | 0.519 | 0.408 | 0.408 | 6244 |
| AMAS-off | 0.392 | 0.515 | 0.400 | 0.400 | 8975 |
| AMAS-conformal | 0.336 | 0.454 | 0.352 | 0.352 | 4827 |
| AMAS-bayesian | 0.272 | 0.371 | 0.280 | 0.280 | 819 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | +2.4 | -0.4 | -0.8 | -0.8 |
| AMAS-off | R2 cap8 | +2.4 | -0.4 | -0.8 | -0.8 |
| AMAS-off | R3 cap64 | +2.4 | -0.4 | -0.8 | -0.8 |
| AMAS-conformal | R1 raw | -3.2 | -6.4 | -5.6 | -5.6 |
| AMAS-conformal | R2 cap8 | -3.2 | -6.4 | -5.6 | -5.6 |
| AMAS-conformal | R3 cap64 | -3.2 | -6.4 | -5.6 | -5.6 |
| AMAS-bayesian | R1 raw | -9.6 | -14.8 | -12.8 | -12.8 |
| AMAS-bayesian | R2 cap8 | -9.6 | -14.8 | -12.8 | -12.8 |
| AMAS-bayesian | R3 cap64 | -9.6 | -14.8 | -12.8 | -12.8 |
