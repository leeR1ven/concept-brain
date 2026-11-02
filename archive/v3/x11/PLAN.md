# X11 plan: a large dialogue corpus in the model's world, and a scaling test by overhearing

Written 2026-11-02. Authority: `docs/ARCHITECTURE.md`. The brain is X10's code, unchanged (copies in this folder; snapshot and checksums in `x10_snapshot/`). Everything new is world, data and scoring. Labels: **DOC** = ARCH rule, **CHOICE** = a detail ARCH does not fix (registered below), **MEASURED**.

Designer's goal (11-02): "你试试下载大量对话类数据集，转换成我们能用的数据集，然后根据不同的训练量，看看会不会越来越好".

## A. Data

| item | value |
|---|---|
| source | LCCC-base (thu-coai, MIT), `silver/lccc` on Hugging Face; the server reaches it through hf-mirror.com (huggingface.co is blocked there) |
| raw | train 6,820,506 dialogues (370 MB gz, sha256 2162e0ed…), valid 20,000, test 10,000 |
| local copy | `<local corpus dir>/lccc_raw/` (raw), `<local corpus dir>/lccc_syl/` (train shards 0–1, valid, test, stats), `<local corpus dir>/lccc_sample/` (the uniform 130k sample); never in a repo |
| server | `~/ckpt/x11_data/raw/` and `~/ckpt/x11_data/lccc_base/` (shards) |
| converter | `data/convert.py` (pypinyin 0.55 wheel unpacked next to it, no install) |

**Conversion choices [D-register].**
- D1 syllable = toneless base + tone. The base inventory is every toneless reading in pypinyin's character dictionary: 426 bases. Toned id = base × 5 + tone − 1 (tone 5 = neutral), stored as uint16.
- D2 surface tones: pypinyin `tone_sandhi=True` (3-3 → 2-3, 一 / 不), as heard.
- D3 cleaning: spaces removed (LCCC is char-tokenised); digits read in Chinese (≤ 4 digits without a leading 0 as a cardinal number, e.g. 31 → 三十一; longer runs and years before 年 digit by digit); punctuation and symbols dropped, no pause inserted; a dialogue with any Latin letter, an unreadable character, or an utterance left empty is dropped.
- D4 erhua: 儿 after a syllable stays a separate syllable `er5`.
- D5 shard (npz, 100k dialogues): `syl` uint16, `turn_off`, `dlg_off`, `src` (source line). Speakers alternate by turn (A, B, A, …).
- D6 the training stream for the scaling test (`data/train.npz`, 110,000 dialogues, 2.86 M syllables, 26.0 syllables and 2.88 turns per dialogue): a uniform random sample (seed 2026) of 130k lines of the train split (`<local corpus dir>/lccc_sample/`), converted, keeping dialogues whose turns all have ≤ 30 syllables (92 %), seeded shuffle. Birth set: 80 dialogues of the valid split. Held-out set: 300 dialogues of the test split (27.1 syllables, 2.96 turns per dialogue); the first 200 are scored. (The file's first shards are multi-turn heavy, 3.8 turns per dialogue, so they were not used.)
- Full conversion (server, 31 min): 6,600,513 of 6,820,506 train dialogues kept (latin 214,558, empty 5,423, unreadable 12); 19.3 M turns; 202 M syllables; 411 of 426 bases and 1,357 toned types occur; 67 shards, 637 MB, in `~/ckpt/x11_data/lccc_base/` (shards 0–1, valid and test also in `<local corpus dir>/lccc_syl/`).

## B. The world

**Voice [V1–V5]** (`voice.py`, `phon.py`). Each syllable is rendered over its 3 ticks from its parts, so similar syllables give similar spectra (ARCH §二.2):
- V1 parts (`phon.parse`): initial, medial (i / u / ü), nucleus, coda (i / u / n / ŋ); apical and retroflex -i, syllabic m / n / ng.
- V2 tick 0 = initial + start of the medial or nucleus (frication / burst noise by place, aspiration as a whispered vowel, voicing amount, F2 locus pull; nasals, l, r as voiced murmurs); tick 1 = nucleus (formants interpolated from the medial); tick 2 = coda (glide formants or nasal murmur) or the nucleus fading.
- V3 vowels: F1–F5 as Lorentzian peaks on a glottal tilt (targets from standard Mandarin formant tables).
- V4 tone = F0 contour over the 3 ticks in semitones from the speaker's base (T1 3.5 3.5 3.5; T2 0 1.5 4; T3 −1.5 −3.5 −2; T4 4.5 1.5 −2.5; T5 0.5 0 −0.5 and 4 dB weaker).
- V5 speakers: 4 adults (C1 F 215 Hz, C2 M 120 Hz, D1 F 240 Hz, D2 M 105 Hz; formant scale 0.94–1.22; tilt; breathiness) and the child (300 Hz, scale 1.35). Per turn: F0 offset ±1 semitone, level ±2 dB; F0 declines 0.08 semitone per syllable (≤ 2); 0.3 semitone jitter per syllable.

**Cochlea [W1′–W3′]** (`voice.Cochlea`, same interface as X10's).
- W1′ channels: 36 pitch channels (place code of F0, 70–530 Hz, 1 semitone each; the periodicity reading) + 96 spectral channels (ERB-spaced 100 Hz–8 kHz). Spectral energy = dB mapped over a 30 dB range to 0–1.2; per channel jitter × U(0.85, 1) and |N(0, 0.04)| noise as X10 W1.
- W2′ own voice: bone conduction = low-passed (1 kHz corner) and × 0.6 (X10 W2's 0.6).
- W3′ thermometer: L = 4·s levels per channel, lineage-fixed thresholds U(0.12, 1.0), each level an on/off pair: **exactly half on, every tick** (asserted). A0 = 132 × 8 = 1,056 bits at s = 1 (X10: 1,080).
- M4′ brake lines (birth wiring): up lines of the spectral channels ≥ 1.5 kHz (188 at s = 1); the own voice reaches them weakly. MEASURED: adult speech lights median 39 (p10 17), silence ≤ 1; `n_brake_per_s` = 20 so that adult speech gives a brake drive ≈ 2, as in X10.

**Body [B1–B4]** (`body.py`): 24 muscles — lips (close, round, spread, labiodental), tongue tip (stop, groove, retroflex, lateral), blade (palatal), body (front, back, high, low, mid, velar closure, velar frication), velum, jaw, larynx (voice, spread glottis, closure hold, pitch high, pitch low), breath.
- B1 a syllable = 3 muscle patterns (tick 0: initial + medial / nucleus; tick 1: nucleus; tick 2: coda or nucleus); tone by pitch muscles (T1 HHH, T2 L·H, T3 LL·, T4 H·L, T5 none and no breath).
- B2 MEASURED: all 2,130 toned syllables have distinct 3-tick patterns; 4.7–6.8 muscles per tick.
- B3 similar sounds share muscles (b / p differ only in voice vs spread glottis; an / ang only in the coda's tip vs velar closure).
- B4 mouth: a tick's muscles that match a syllable tick exactly make that tick's sound. Not used in the scaling test (no speech wiring; the child is silent and any muscle activity is only counted).

**Overhearing world [D7]** (`world11.py`): a pause of 12–24 ticks, then two different adult voices (of 4) alternate turns; 3 ticks per syllable; 4–9 ticks between turns. MEASURED ≈ 104 ticks per dialogue (≈ 26 syllables). Every dialogue's plan comes from its own generator (life seed, dialogue index), so the world resumes exactly at any tick.

**Input check (MEASURED, `check_voice.py`, nearest centroid on the 3-tick A0 pattern, 1,212 toned types of the valid split, 4 voices).**

| | toned | toneless | tone |
|---|---|---|---|
| same voices, new tokens | 0.84 | 0.93 | 0.90 |
| new (unheard) voice | 0.35 | 0.62 | 0.51 |

Errors are phonetically near: 77 % keep the final, 75 % the initial (aspiration pairs zh/ch, b/p, g/k, j/q, d/t; m/n; -n/-ng; i/in). Levels per channel: 3 → toneless 0.88, 4 → 0.89, 5 → 0.91 (2-token centroids); 4 chosen.

## C. Brain (X10 code unchanged)

X10 base B + P1 (coordinator 11-02): P1 patch inhibition, reciprocal loop g_loop 1.0, w44 0, learned gains g44 = g_hh = g_h1 = 0.1 (R2), η 0.02, η_H 0.3, A1/A2 chain band upper cut 0.6, medium PFC (A1 by the band, deeper units ≥ 0.9 r_l), s = 1. Sizes follow from the same whole-brain rules: sound area = channels × 2L × 2 units × 4 layers; PFC L1 = one copy per chained unit (+ force + stop); L2 = L3 = L4 = L1; hippocampus 5,000·s; motor 24 muscles × m (m = s). No speech wiring (G1P / MC off). Birth: 80 dialogues (≈ 8.4k ticks, the length of X10's birth period), learning off; thresholds, bands and online calibration as X10.

## D. Scaling test (overhearing only)

- One life per seed (0, 1) overhears the training stream continuously, learning on; checkpoints at 1k, 3k, 10k, 30k, 100k dialogues (plus the newborn at 0). Jobs ≤ 90 min chain through `~/ckpt/x11_s1_seed<k>/` (whole Brain object + world state; deleted at the end).
- **Main measure (frozen copy, fixed held-out set: 200 test-split dialogues, fixed voices / pauses / noise).** For each syllable occurrence after the first of a dialogue, onset tick t0:
  - PFC prediction = sound copies lit internally in L1 (x_int, from L4(t−1)) on ticks t0+1 … t0+3. By the pipeline (L1 → L2 → L3 → L4 → L1) none of this syllable's sound can have reached L4 by then: a genuine prediction.
  - Ranked by cosine against every toned syllable's code (≈ 1.3k candidates); top-1 / top-5, toned and toneless; empty predictions count as misses. Code = expected rises of each sound copy in the same window when the syllable is heard (4 voices × 2 contexts, newborn sound area, no returns; the sound area never learns, so the codes hold for life).
  - Hippocampus: sound copies lit internally in H1 (h_int) on t0+1 … t0+2 (its prospective window), same ranking.
  - Per-tick J(x_int, copies that rose) on speech ticks; readout ceiling = the heard rises ranked the same way.
- **Latency (derivation, X10 code).**
  - Sound: A1(t) = f(A0(t)); A_l(t) = f(A_{l−1}(t−1)), so layer l carries A0 from l − 1 ticks earlier.
  - Chains: L1ext(t) = sources(t − 1), one tick.
  - PFC: L2(t) = f(L1(t − 1)), L3(t) = f(L2(t − 1)), L4(t) = f(L3(t − 1)) (+ L4(t − 1) laterals): L1 → L4 is 3 ticks. x_int(t) = f(L4(t − 1)): one more tick.
  - So x_int(t) is driven by L1ext(t − 4) = A1(t − 5) = A0(t − 5) at the newest (deeper copies are older still).
  - A syllable lasts 3 ticks. For the target with onset t0 (previous syllable on t0−3 … t0−1, the one before on t0−6 … t0−4), x_int on t0+1 … t0+3 sees A0 up to t0−4 … t0−2: mostly the syllable before last, plus the first two ticks of the previous one. Its own sound reaches L4 only at t0+4 (A0(t0) → L1ext(t0+1) → L4(t0+4)), so x_int(t0+5) is the first tick that could use it.
  - The late window t0+2 … t0+4 sees all three ticks of the previous syllable through A1.
  - Hence both the lag-1 references (bigram, trigram on the previous syllables) and the lag-2 references (bigram from the syllable before last; trigram from the two before the previous) are reported at every checkpoint.
- **Reference:** unigram; lag-1 bigram and trigram (back-off); lag-2 bigram and trigram. All are trained on exactly the dialogues heard so far and scored on the same occurrences: directly (top-k) and through the codes (predicted code = Σ p(s) code_s, ranked like the brain).
- **Response (closest to "hears and responds"):** the first syllable of a reply turn (after the inter-turn pause). Brain (PFC both windows, hippocampus) vs unigram, the unigram of reply-initial syllables, and the bigrams and trigrams across the turn boundary (the count models see a dialogue as one stream).
- **Frozen code for the whole curve** (coordinator 11-02): no code or parameter change between checkpoints. A change starts a new curve.
- **Health:** per 10-dialogue window (learning on): L2/L1, L3/L2, L4/L3 in [0.5, 2], Hs in [0.5, 2] × kh, no PFC / Hs layer over 50 %; L1, L1int, L4, A4, Hs trends; counts in the eval.
- s = 4 only if the s = 1 curve improves.

## E. Conformance (what is new against ARCH)

| § | rule | X11 |
|---|---|---|
| 2.1 | input layer = thermometer on/off pairs, always exactly half on; silence is a reading | W3′, asserted every tick |
| 2.1 | own vs other voice similar but different, no labels | same spectra and pitch code; own voice low-passed and weaker (W2′); voices differ only physically (F0, formant scale) — no voice channels, no labels |
| 2.2 | similar sounds → similar units | V1–V5 build each sound from shared parts; MEASURED errors are phonetic neighbours |
| 2.7 | only muscles; brake reads only the input layer, graded by loudness | B1–B4; M4′ |
| 2.11 | overhear many dialogues between others (step 2); others' lines only, no echo training | D7 |
| 2.12 | scale the whole brain together | one s; A0 levels and every area × s as X10 |
| — | no symbolic input | the brain gets only A0 (X10 assert); labels only in the scorer |

## F. Results, curve 1 (frozen code, md5 in `results/CODE_MD5_curve1.txt`; stopped after job 2 by the coordinator, 11-02)

Held-out next-syllable prediction (200 test dialogues, 5,008 occurrences, 1,262 candidates; seed 0 / seed 1). Tables: `agg_scale.py results/curve1/j2/*.json`.

| dialogues | PFC top-1 | PFC top-5 | per-tick J | hippocampus | unigram t1/t5 | bigram lag 1 | bigram lag 2 | trigram lag 2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.001 / 0.000 | 0.008 / 0.006 | 0.016 / 0.013 | empty | – | – | – | – |
| 1,000 | 0.002 / 0.001 | 0.007 / 0.007 | 0.021 / 0.019 | empty | 0.031 / 0.144 | 0.137 / 0.277 | 0.041 / 0.126 | 0.037 / 0.118 |
| 3,000 | 0.003 / 0.001 | 0.009 / 0.007 | 0.021 / 0.018 | empty | 0.031 / 0.144 | 0.150 / 0.306 | 0.047 / 0.142 | 0.045 / 0.127 |

- Chance 0.0008 / 0.004; readout ceiling (heard rises) 0.34 top-1; the interpolated count model read through the codes 0.10.
- Response (first syllable of a reply turn): PFC top-5 0.003–0.008; reply-initial unigram 0.093 / 0.25–0.27.
- Health: every 10-dialogue window passes; L4 / L3 ≈ 1.15; Hs 66–80.
- Speed: 20–23 ms per tick per process with two seeds in one GPU job ≈ 1,600 dialogues per seed per hour.
- Count-model reference to 100k (local, `ngram_curve.py`): trigram lag 1 0.222 / 0.396, bigram lag 2 0.057 / 0.168, trigram lag 2 0.076 / 0.179 at 100k.

**Diagnostics (frozen; `diag_scale.py`, `diag_l4.py`, `results/diag1/`).**
- The learned L4 → L1 drive alone never reaches θ1 (max 0.42–0.45 θ1, p90 0.63); all of x_int (74–83 per tick) is the birth loop. The graded learned drive ranks the target at top-5 0.011–0.014 (chance 0.004) and the previous syllable at 0.012–0.023.
- Saturation: 98 % of the L4 → L1 candidate synapses and 96 % of the Hs → H1 synapses have weight > 0 after 1k dialogues; Hs ↔ Hs 81 %. H1's internal drive is ≈ 0.01 θ_h1, so the hippocampus never recalls.
- Specificity through the PFC (Jaccard same − different syllable, peak over latency, newborn / 3.4k): L1ext +0.15 / +0.14; L2 +0.08 / +0.08; L3 +0.06 / +0.05; L4 +0.047 / +0.044 (L4 ≈ 15–25 % on; same 0.24 vs different 0.21). Learning did not change it.
- Reading: post events are frequent and the eligible pre populations dense, so the one rule potentiates nearly every pair; the decay is too slow to prune at this event rate; √(out·in) then makes the drive flat and small (ARCH §四.20, X1e). Curve 2 waits for the X10 builder's fixes (PFC count preservation for life; working hippocampal recall) and the coordinator's audit.
