# Generation-0 registry of hand-added connections (X1f, 2026-11-01)

Every set below is added after birth (birth listening + 200 exchanges with all learning rates 0), is never changed by
learning, and is exempt from the one global decay (birth links, ARCH 2.8). Code: `wire.py`, applied in `core6.py`.
Offspring would inherit these sets with the birth wiring (ARCH 2.12). Test pairs: only never-wired pairs are used to
test learning.

## Main arm G0 (speech basics only; ARCH 2.11 step 1, 2.7)

| set | area | from -> to | how computed | why |
|---|---|---|---|---|
| A1 | auditory -> motor | L1 chain copy of each auditory content feature (lit when the sound is heard *or* inner-heard) -> the 24 muscle neurons | non-negative least squares (ridge 0.3) from every syllable's 10-feature content code to its 4-muscle set; 598 links; replaces X1c's stand-in `Mmap` and is used by both routes that used it (PFC return = inner hearing, and the direct auditory -> motor link) | "thinking a syllable's sound produces it by instinct" (designer 11-01). The stand-in sums the muscle sets of every syllable sharing a feature: 4th vs 5th muscle only 1.5x apart, forced whole answers garbled 77-95 %. With A1: forced whole answers said exactly 0.91-1.00 (5 seeds). X1c's body has no order / force grades inside a syllable, so A1 is a set map, not a sequence map. |
| A2 | motor | the muscle neurons about to fire this tick -> two groups of blocking neurons (same-tick feed-forward) | block if fewer than 3 or more than 5 muscles would fire | release-by-sharpness: only a syllable-shaped command wins; a mixed inner hearing ("unsure") gives silence, not garble (ARCH 2.7 "没把握时结果是不说"). Mixed forced inner hearing: said 0.14, garble 0 (without A2: said 0.82-1.00, garble 0.82-1.00). |
| knobs | motor | existing parameters | c_mb 1.2 -> 0 (no muscle -> blocker latch); g_dir 0.5 -> 0.25 (direct echo link) | the latch kept speech going after the inner hearing stopped; the strong echo link made the heard own syllable compete with the next inner-heard one (garble at syllable transitions) |

Not hand-wired in G0: any thinking, any Q -> A link. PFC parameters changed (not connections): f2 0.18, f3 0.08,
patch-mean on the whole top-layer drive (pm_ff 1) - see PLAN.md.

## Diagnostic arm G0T (A1 + A2 + hand-wired thinking B and responding C, base pairs only)

Base set = half of the 22 seen pairs (11), chosen per seed by `rng(seed + 4000).permutation`; the other 11 seen pairs
and all 10 held-out pairs are never wired.

| set | area | from -> to | how computed | why |
|---|---|---|---|---|
| B0 | PFC top layer (L3) | a working-memory (WM) part = 4-5 of the 12 X3 patches (214 units in seed 0, chosen spread out) | its random L2->L3 links (6,413) and random birth L3->L3 links into it are removed; WM units fire at a fixed threshold 1.0 on their raw drive (no patch-mean) | the random L3 codes are voice-dependent and the question-end code is frame-only (cross same frame 0.28 vs same 0.32, X1d), so a held thought needs its own clean units |
| B1 | L1 -> WM | heard (forward-chain) content copies of a syllable -> WM assembly | +a/10 per copy (a = 5.5 for a filler hold, 0.8 for a step) | the WM's sound input; voice-free and 95 % reliable (L2/L3 codes are not) |
| B2 | WM <-> WM, self | inside each 5-unit assembly | 3.0 total from the 4 co-members | the assembly holds itself |
| B3 | WM <-> WM, inhibition | every pair of WM units not wired together | -3.5 / 5 per link | "rarely co-active neurons inhibit each other" (ARCH 2.4): one held assembly at a time, the newest wins |
| B4 | WM -> WM, path | H_X (filler hold) -> Q1(f,X); Qk -> Q(k+1); Qn -> D(f,X) ('answered', holds the topic, predicts nothing) | 0.6 total from the predecessor + 0.8 from the heard last question syllable / answer syllable k (each alone below threshold) | a question-specific thought that holds through the pause and then moves one step per heard (own or adult) answer syllable |
| C1 | WM -> L1 (top -> L1) | Qk -> the 10 L1 content copies of answer syllable a_k | 1.0 per copy (split over the 5 members), all copies (not masked by M31) | inner hearing of the answer; said by A1 through the existing inner-hearing route. Plan early (Q1 lights at question end while the brake holds), fire late (release when the partner stops) |

Assemblies per seed: 8 filler holds max + 11 base pairs x (answer length + 1) ~ 33; non-overlapping.
