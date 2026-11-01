# v3: experiments on the designer's core

Each folder is one experiment from the v3 design round (`docs/v3design/`). The authority is `docs/ARCHITECTURE.md`, and results that settle an open item are recorded there.

| folder | what | report |
|---|---|---|
| `x345/` | X3: pure-neuron hippocampus (chain-fed, Hebbian, no records); X4: merging siblings' learned weights; X5: tick cost on CPU and A800 | `x345/REPORT.md` |
| `x4b/` | the designer's 40/40/20 merge rule and other operators; does a merged child relearn its parents' lessons faster than a fresh birth? | `x4b/REPORT.md` |
| `x1c/` | the designer's core (ARCHITECTURE.md) on a continuous tick-by-tick dialogue stream, 12 arms × 5 seeds; answers 0 in every arm | `x1c/REPORT.md` |
| `x7/` | HARVEST items 13 (convergent sensory thresholds), 26 (babbling learns sound → muscles), 28a/b (syllable competition, pace): 13, 26, 28a fail; 28b partial | `x7/REPORT.md` |
| `x1d/` | same question → same path? 63 birth regimes, then learning with the corrected held-unit rule: no reproducible path, answers 0; motor release works when forced; patch-mean gain control bounds the PFC (batch 2: graded praise, growth/decay-only, oracle path) | `x1d/REPORT.md`, `x1d/REPORT_batch2.md` |
| `x6/` | HARVEST items 17–22 on the X3 hippocampus: pause bridge via inner voice partial; strength ratio instead of a switch fails; cued recall partial; sizing settled (k fixed, C ∝ s, decay ∝ 1/s); one-shot fact stored but not retrieved | `x6/REPORT.md`, `x6/tables/` |
| `x9/` | HARVEST item 14, the concept area offline: a two-layer form gives held-out class decode 0.93, within > between, cross-modal overlap; class code still drifts 0.11 | `x9/REPORT.md`, `x9/res/test_agg.txt` |
| `x1f/` | hand-wired generation 0 (designer's systematic method): speech basics and wired thinking work (wired pairs 0.96–0.98 live), but new pairs are not learned on any route, even with perfect paths; the failing link is a specific learned prediction | `x1f/REPORT.md`, `x1f/REGISTRY.md` |
| `x1e/` | learning the PFC top layer's internal links (L33), quiet wait, overhear-first, ×4 scale: no question-specific path, answers 0; likely cause: associative capacity (too dense) | `x1e/REPORT.md`, `x1e/res/` |
| `x8/` | the two correction mechanisms in the X4 harness: concave decrease keeps memories for life but slows corrections; upset reversal is what makes corrections work; adults learn fast via the fast store; retention is lost to interference (frame-level codes) | `x8/REPORT.md`, `x8/results/` |
