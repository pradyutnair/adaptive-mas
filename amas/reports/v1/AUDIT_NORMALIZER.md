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
| HERA R1 raw | 0.097 | 0.181 | 0.194 | 0.194 | 8937 |
| HERA R2 cap8 | 0.099 | 0.173 | 0.129 | 0.129 | 8937 |
| HERA R3 cap64 | 0.099 | 0.181 | 0.166 | 0.166 | 8937 |
| AMAS-off | 0.112 | 0.209 | 0.145 | 0.145 | 9651 |
| AMAS-conformal | 0.104 | 0.193 | 0.133 | 0.133 | 5979 |
| AMAS-bayesian | 0.073 | 0.156 | 0.098 | 0.098 | 870 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | +1.5 | +2.8 | -4.9 | -4.9 |
| AMAS-off | R2 cap8 | +1.3 | +3.7 | +1.6 | +1.6 |
| AMAS-off | R3 cap64 | +1.3 | +2.8 | -2.1 | -2.1 |
| AMAS-conformal | R1 raw | +0.7 | +1.3 | -6.1 | -6.1 |
| AMAS-conformal | R2 cap8 | +0.5 | +2.1 | +0.4 | +0.4 |
| AMAS-conformal | R3 cap64 | +0.5 | +1.2 | -3.3 | -3.3 |
| AMAS-bayesian | R1 raw | -2.4 | -2.5 | -9.6 | -9.6 |
| AMAS-bayesian | R2 cap8 | -2.6 | -1.7 | -3.1 | -3.1 |
| AMAS-bayesian | R3 cap64 | -2.6 | -2.5 | -6.8 | -6.8 |

## hotpotqa (n=1000)

| method | EM | F1 | contain | Acc | tok |
|---|---:|---:|---:|---:|---:|
| HERA R1 raw | 0.264 | 0.388 | 0.524 | 0.524 | 8411 |
| HERA R2 cap8 | 0.310 | 0.436 | 0.427 | 0.427 | 8411 |
| HERA R3 cap64 | 0.310 | 0.432 | 0.500 | 0.500 | 8411 |
| AMAS-off | 0.372 | 0.513 | 0.461 | 0.461 | 9141 |
| AMAS-conformal | 0.355 | 0.491 | 0.447 | 0.447 | 3775 |
| AMAS-bayesian | 0.316 | 0.455 | 0.417 | 0.417 | 852 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | +10.8 | +12.5 | -6.3 | -6.3 |
| AMAS-off | R2 cap8 | +6.2 | +7.7 | +3.4 | +3.4 |
| AMAS-off | R3 cap64 | +6.2 | +8.1 | -3.9 | -3.9 |
| AMAS-conformal | R1 raw | +9.1 | +10.3 | -7.7 | -7.7 |
| AMAS-conformal | R2 cap8 | +4.5 | +5.6 | +2.0 | +2.0 |
| AMAS-conformal | R3 cap64 | +4.5 | +5.9 | -5.3 | -5.3 |
| AMAS-bayesian | R1 raw | +5.2 | +6.7 | -10.7 | -10.7 |
| AMAS-bayesian | R2 cap8 | +0.6 | +2.0 | -1.0 | -1.0 |
| AMAS-bayesian | R3 cap64 | +0.6 | +2.3 | -8.3 | -8.3 |

## 2wikimultihop (n=1000)

| method | EM | F1 | contain | Acc | tok |
|---|---:|---:|---:|---:|---:|
| HERA R1 raw | 0.108 | 0.227 | 0.577 | 0.577 | 9172 |
| HERA R2 cap8 | 0.159 | 0.317 | 0.386 | 0.386 | 9172 |
| HERA R3 cap64 | 0.159 | 0.289 | 0.532 | 0.532 | 9172 |
| AMAS-off | 0.293 | 0.409 | 0.410 | 0.410 | 9975 |
| AMAS-conformal | 0.261 | 0.378 | 0.374 | 0.374 | 6393 |
| AMAS-bayesian | 0.239 | 0.345 | 0.329 | 0.329 | 906 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | +18.5 | +18.2 | -16.7 | -16.7 |
| AMAS-off | R2 cap8 | +13.4 | +9.2 | +2.4 | +2.4 |
| AMAS-off | R3 cap64 | +13.4 | +12.1 | -12.2 | -12.2 |
| AMAS-conformal | R1 raw | +15.3 | +15.0 | -20.3 | -20.3 |
| AMAS-conformal | R2 cap8 | +10.2 | +6.0 | -1.2 | -1.2 |
| AMAS-conformal | R3 cap64 | +10.2 | +8.9 | -15.8 | -15.8 |
| AMAS-bayesian | R1 raw | +13.1 | +11.7 | -24.8 | -24.8 |
| AMAS-bayesian | R2 cap8 | +8.0 | +2.8 | -5.7 | -5.7 |
| AMAS-bayesian | R3 cap64 | +8.0 | +5.6 | -20.3 | -20.3 |

## bamboogle (n=125)

| method | EM | F1 | contain | Acc | tok |
|---|---:|---:|---:|---:|---:|
| HERA R1 raw | 0.312 | 0.443 | 0.432 | 0.432 | 8488 |
| HERA R2 cap8 | 0.344 | 0.448 | 0.360 | 0.360 | 8488 |
| HERA R3 cap64 | 0.344 | 0.471 | 0.416 | 0.416 | 8488 |
| AMAS-off | 0.392 | 0.515 | 0.400 | 0.400 | 8975 |
| AMAS-conformal | 0.336 | 0.454 | 0.352 | 0.352 | 4827 |
| AMAS-bayesian | 0.272 | 0.371 | 0.280 | 0.280 | 819 |

Delta vs HERA (pp):

| method | regime | dEM | dF1 | dcontain | dAcc |
|---|---|---:|---:|---:|---:|
| AMAS-off | R1 raw | +8.0 | +7.2 | -3.2 | -3.2 |
| AMAS-off | R2 cap8 | +4.8 | +6.7 | +4.0 | +4.0 |
| AMAS-off | R3 cap64 | +4.8 | +4.4 | -1.6 | -1.6 |
| AMAS-conformal | R1 raw | +2.4 | +1.1 | -8.0 | -8.0 |
| AMAS-conformal | R2 cap8 | -0.8 | +0.7 | -0.8 | -0.8 |
| AMAS-conformal | R3 cap64 | -0.8 | -1.7 | -6.4 | -6.4 |
| AMAS-bayesian | R1 raw | -4.0 | -7.2 | -15.2 | -15.2 |
| AMAS-bayesian | R2 cap8 | -7.2 | -7.6 | -8.0 | -8.0 |
| AMAS-bayesian | R3 cap64 | -7.2 | -10.0 | -13.6 | -13.6 |
