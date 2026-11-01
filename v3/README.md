# v3: experiments on the designer's core

Each folder is one experiment from the v3 design round (`docs/v3design/`). The authority is `docs/ARCHITECTURE.md`, and results that settle an open item are recorded there.

| folder | what | report |
|---|---|---|
| `x345/` | X3: pure-neuron hippocampus (chain-fed, Hebbian, no records); X4: merging siblings' learned weights; X5: tick cost on CPU and A800 | `x345/REPORT.md` |
| `x4b/` | the designer's 40/40/20 merge rule and other operators; does a merged child relearn its parents' lessons faster than a fresh birth? | `x4b/REPORT.md` |
| `x1c/` | the designer's core (ARCHITECTURE.md) on a continuous tick-by-tick dialogue stream, 12 arms × 5 seeds; answers 0 in every arm | `x1c/REPORT.md` |
| `x7/` | HARVEST items 13 (convergent sensory thresholds), 26 (babbling learns sound → muscles), 28a/b (syllable competition, pace): 13, 26, 28a fail; 28b partial | `x7/REPORT.md` |
| `x1d/` | same question → same path? 63 birth regimes, then learning with the corrected held-unit rule: no reproducible path, answers 0; motor release works when forced; patch-mean gain control bounds the PFC (batch 2: graded praise, growth/decay-only, oracle path) | `x1d/REPORT.md`, `x1d/REPORT_batch2.md` |
