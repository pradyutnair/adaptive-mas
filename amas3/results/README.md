# results/

Layout:

```
results/
├── README.md                 # this file
├── amas_pro/                 # MAIN CONTRIBUTION — AMAS-PRO method
│   ├── README.md
│   ├── homogeneous_qwen3_14b/    # all-Qwen3-14B (no closed-source LM)
│   └── heterogeneous_qwen3_14b_4omini/   # Qwen3-14B + GPT-4o-mini solver
├── external_baselines/       # published baselines re-run for fair comparison
├── amas_legacy/              # prior-iteration runs kept for ablation reference
│   ├── v1_v2_v3_qwen3_8b/    # earlier AMAS variants
│   ├── older_amas_runs/      # older mix5 / mix10 / opera10 runs
│   ├── older_results/        # archived earlier work
│   ├── saat/                 # SAAT structure-aware adaptive topology runs
│   ├── sufficiency_results/  # sufficiency-probe variant
│   ├── baselines/            # other baseline re-runs (CoT-only, etc.)
│   ├── diagnostics/          # one-off diagnostic outputs
│   └── gepa_compile_logs/    # DSPy GEPA compile state from earlier run
└── _runtime/                 # vLLM/retriever server logs (operational)
```

The headline result lives at:

`amas_pro/homogeneous_qwen3_14b/1000q_2026-05-01/eval_final.json` — MuSiQue 1000q at norm_em 0.226, F1 0.307, all-Qwen3-14B, training-free.

See `amas_pro/README.md` for full method description, scaling-law analysis, and reproduction commands.
