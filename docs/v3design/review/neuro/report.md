# Neuroscience review of ARCH_v3_core

**Reviewer:** adversarial reviewer, background in systems and computational neuroscience. Review only: no training runs and no server jobs. The repo and other agents' folders were not touched.

**Toy checks.** All are local numpy, one process at a time, under 70 MB each, run after `free -m`. Scripts, outputs and JSON are in `docs/v3design/review/neuro/`:
- `ca3_capacity.py/.out`, `ca3_palimp.py/.out`, `ca3_sparse.py/.out`: the pure-neuron CA3;
- `anchor_replay_toy.py/.out/.json`: the drift_toy world plus replay readings, distractors and independent jitter;
- `p5b_arith.py/.out`: the P5b reward arithmetic;
- `meta_toy.py/.out/.json`: designer idea #22, metaplasticity;
- `wide_thought_toy.py/.out/.json`: designer idea #23, a wider thought.

**Labels:** PROVED (code, count or arithmetic), SIMULATED (a run, file named), CITED (paper or report), ASSUMED (judgement).

---

## 0. Bottom line

The skeleton holds up from the brain side. Log, stimulus-specific time cells anchored on the latest event exist in lPFC and the hippocampus. The efference "self" source, the encode/replay direction of μ and the BG release-with-urgency are all real circuit motifs. The anchor also survived a test the synthesis did not run (SIMULATED): independent per-event jitter ×[0.5, 2] on both the format wait and the give-up delay still gives 0.96 at 3,000 untutored dialogues.

But the design is not buildable as written:
- **The hippocampal replay mode contradicts the held context it was meant to protect.** Read literally, §1.3 says μ gates only "live afferent pathways". Replay is then learned in the live LATEST × age context, and the anchored core collapses from 0.99 to 0.00 (SIMULATED).
- **The pure-neuron CA3 at the stated size holds about 1,000 episodes.** A bench life writes about 10⁴. Random-kick "spontaneous uniform" replay never reaches a stored episode (SIMULATED).
- **P5b ("being responded to") is a co-occurrence reward.** It pays more for an extra turn timed just before the partner's predictable onset (PROVED arithmetic).

**Verdict: fix, then build.** Every fix below is generic and changes a definition, not the mechanism count.

---

## 1. What survives neuro scrutiny (short)

| claim | judgement |
|---|---|
| Latest event × age as stimulus-specific time cells (log-compressed, scalar width) | Real: Tiganj et al. 2018; Cruzado et al. 2020; Cao, Bladon & Howard 2022; MacDonald et al. 2011; Umbach et al. 2020 (CITED). The 2-octave bins match scalar timing. The AND conjunction item × bin on a dendritic branch is plausible (Poirazi & Mel 2003; Losonczy et al. 2008, CITED). |
| Anchor robust to timing jitter | SIMULATED (`anchor_replay_toy.out`): independent per-event jitter on the format wait and the give-up gives 0.96 ± 0.02 at ×[0.5, 2] and 0.99 at ×[0.75, 1.33], after 3,000 untutored dialogues. The agent answers at τ ≈ 4–5, before any give-up bin. |
| μ's direction (high: afferent gain and encoding; low: recurrence and replay) | Consistent with Hasselmo 2006. Cholinergic activation suppresses SWRs (Vandecasteele et al. 2014, CITED). |
| δ timing | Assume about 67 ms per tick (a 3-tick syllable ≈ 200 ms, ASSUMED). Then τ_e = 20 ticks ≈ 1.3 s, inside the 0.3–2 s dopamine window for spine plasticity (Yagishita et al. 2014, CITED). |
| Self source via the efference copy; forward model explains own voice away | Corollary discharge (Eliades & Wang 2003, 2008; Crapse & Sommer 2008, CITED). Sound. |
| M4 disinhibition plus an accumulator | Gurney, Prescott & Redgrave 2001; Bogacz & Gurney 2007; Cisek et al. 2009 (CITED). Sound in form. |
| Recruitment of the least-used, best-matching units | Matches memory allocation to excitable neurons (CREB; Han et al. 2007; Yiu et al. 2014, CITED). |

---

## 2. Problems, ranked

### #1 FATAL as written: replay (μ low) collides with the held anchored context

**Evidence.**
- **The three statements do not fit together.** §1.3: μ is "(i) the gain of live **afferent** pathways into MIX and the hippocampus; (ii) when μ < 0.05 (about 12 silent ticks) the hippocampus reactivates". M3: "its context drives MIX, while μ gates the live pathways down". §1.1: "EVERY unit has one branch on LATEST × TIME". LATEST and TIME are internal, not afferent. So under the literal text, every MIX unit stays anchored on the live LATEST (the held question) during replay, and the replayed successor is taught in the context "held Q × current age".
- **SIMULATED** (`anchor_replay_toy.out`, ST context, format 600 → untutored 3,000, p_sil 0.3, 5 seeds):

  | replay | after formats | 300 untutored | 3,000 untutored |
  |---|---|---|---|
  | none | 0.98 | 0.99 | 0.99 |
  | stored (CA1 supplies the stored anchored context) | 0.99 | 1.00 | 1.00 |
  | live (anchor stays on the live LATEST) | 0.99 | **0.26** | **0.00** |

  Route R-c of §6.5 ("closed by construction") is closed only under the stored reading.
- **The 12-tick switch sits inside the answer window** (PROVED from the gene defaults):
  - τ_μ = 4, so μ < 0.05 at about 12 ticks.
  - Formats come at τ_f ≈ 10. Development jitter ×1.33 gives up to 13; sealed ×2 gives up to 20. W4's polite caregivers wait longer.
  - Told answers and give-ups come at 45.
  - So many waits for an answer run in replay mode. In the brain, waiting for a turn is an attentive, high-arousal state (theta, not SWR). Awake SWRs occur in disengaged immobility (Jadhav et al. 2012; Karlsson & Frank 2009, CITED).
- **Unspecified side effects.**
  - Replay "drives SYL as the target". If that produces SYL output, it triggers M1 onset transients, which load LATEST (so the held Q is lost) and raise μ.
  - CA1 is an M4 source, so the agent could speak replayed successors. This was v2.x's "says partner lines".
  - μ also rises with any sensory change. With saccades or a moving fovea in world2, μ may never fall, so there would be no replay at all.

**Smallest fix** (one definition):
- μ gates every **non-CA1** input of MIX and DG (LATEST, TIME, THOUGHT, EYE, MEAN). CA1 then supplies the stored context with its age.
- μ also gates M4's release channel: no speaking in replay mode, which matches immobility during SWRs.
- Replayed targets enter SYL only on its teaching (apical) input. They cause no somatic output, so no event transient.
- Define μ = max(sensory presence, GEN's predicted presence in the next ~3τ_μ ticks, phasic |δ|). Basal-forebrain cholinergic cells burst to reinforcement (Hangya et al. 2015, CITED).
- With this, an expected answer keeps the system in encode mode, and the toy gives 1.00.

### #2 SERIOUS: the pure-neuron hippocampus (S6) at the stated size cannot do M3's job; "uniform spontaneous replay" is not an attractor-net behaviour

**Evidence** (SIMULATED; N 10⁴, fixed fan-in 1,000, k 100, one-shot Willshaw, 50 % cue, 3 k-WTA iterations).

`ca3_capacity.out`, recall after storing P episodes:

| P | 250 | 1,000 | 2,500 | 5,000 |
|---|---|---|---|---|
| recall overlap | 0.98 | 0.86 | 0.25 | 0.02 |

The failure is catastrophic, with no forgetting.

`ca3_palimp.out` adds LTD at 0.09 for a steady fill of 0.1. Recall by episode age:

| age | 10 | 100 | 300 | 600 | 1,000 |
|---|---|---|---|---|---|
| recall overlap | 0.87 | 0.78 | 0.51 | 0.16 | 0.02 |

Other results from the same runs:
- **Weak cues:** a 20 % cue at age 100 recalls only 0.45. θ_rec = 0.2 would accept that as a chimeric "recall".
- **Replay:** a random kick reaches a stored episode in **0 of 30** trials at every load. A cue of 30 % of the last episode reaches it in 30 of 30.

Load against need:
- A bench life writes about 4,071 heard lines (CITED develop) plus own lines (which L6 adds), 1,600 object events and person events: about 10⁴ writes (ASSUMED).
- That is a 10–30× gap against the record store, which the drift, naming and NEXT/COMPLETION routes rely on.

**Scaling (PROVED).** The fill after P one-shot writes is 1 − exp(−P f²), whatever N is. Clean recall needs a fill of about 0.1 with C·f·cue ≥ ~5, so capacity ≈ 0.1/f². Under §4 (fixed fractions and fixed fan-in), CA3 capacity therefore stays at about 1,000 for every s, while writes grow with L ∝ s. "Hippocampal half-life as a fraction of capacity × s" cannot hold.

**Fix point checked** (SIMULATED, `ca3_sparse.out`): N 10⁴, k 32 (f 0.0032), C 3,100.

| episodes stored | 5,000 | 10,000 | 15,000 |
|---|---|---|---|
| recall overlap | 0.98 | 0.85 | 0.62 |

This uses 31 M one-bit synapses: 3× the design's CA3, about 93 MB at 3 B per synapse.

**Smallest fix:**
- CA3 uses a fixed active count (k ≈ 32) and fan-in ∝ s, so M3 synapses grow ∝ s². This is a declared cost: at s = 4 a batch of 64 no longer fits in 80 GB.
- Write strength ∝ μ at offset (salience), so predictable events are written weakly and fall off the palimpsest first.
- Replay is **cued**, from THOUGHT/DG or the last episode, and follows successor links. Drop "uniformly".
- S6's check must also measure replay fidelity and recall at the palimpsest horizon. Run this module test now (local, minutes), before L6, not at L12.

### #3 SERIOUS: P5b is non-contingent and can be harvested by timing

**Evidence** (PROVED, `p5b_arith.out`):
- Setup: untutored timing. The next probe comes at max(Q_end + 45, stop + 13). The W4 partner is polite. γ 0.95, τ_e 20.
- Staying quiet after an early answer returns 0.037.
- One extra syllable at Q_end + 29 returns 0.101 (2.7×) with a graded trace, and 0.194 against 0 with a thresholded trace.
- GEN's age-specific "someone will start" prediction makes the best time learnable.
- So in untutored life after maturity, δ favours extra turns, the opposite of §6.4's "learned NoGo (unrewarded … second bouts)".
- Infants detect contingency: P(response | act) − P(response | no act) (Watson 1972; Gergely & Watson 1999; Goldstein, King & West 2003, CITED). They do not reward co-occurrence.
- Also, own voice must not count as a "person-like onset", or every utterance rewards itself.

**Smallest fix:**
- At each release, set the efference trace's amplitude to 1 − p̂, where p̂ is GEN's predicted probability of a partner onset within τ_e. Predictable responses then pay about 0.
- Route P4's voice input through the forward-model residual (corollary discharge).
- Caveat (ASSUMED): this also shrinks the 5b reward for predictable greeting replies. The greeting then rests on the reply coming earlier than predicted, plus P5c.

### #4 SERIOUS: M4 is unspecified where it matters

**Evidence:**
- **Release formula mismatch (PROVED).**
  - §1.2 and §6 use I += (G + u)·e.
  - `checks.py` uses I += G·e + u ("G=1.0, e=0.0 → release after 7 ticks").
  - The second version releases every ~7 ticks even when nothing is predicted.
- **Credit assignment.** A single global δ trains five Go_s/NoGo_s gates with no source-specific eligibility, so every gate gets the same update. PBWM works only with per-stripe traces (O'Reilly & Frank 2006, CITED).
- **ε = 0.05 never anneals.** That conflicts with designer decision #20 (0 errors after maturity, in every tenth of life) and the 0.85 accuracy bar.
- **R-e arithmetic (PROVED).** Σδ = −V₀ holds per episode. Over the critic's unlearning the total is −V₀/η_c ≈ −3.3 V₀, so the stated NoGo bound of 0.3 is about 3× too small. The protection actually comes from u.

**Smallest fix:**
- Pick (G + u)·e and correct `checks.py`.
- Give gate s an eligibility ∝ c_{s,j*}, its contribution to the released program.
- Set ε ∝ the release channel's own η (it falls with n_use), so exploration fades with maturity. Thura & Cisek 2017 show urgency in the BG output is context-modulated (CITED).

### #5 SERIOUS: M2 as written already blocks corrections, and the lineage n_use cap does nothing

**Evidence:**
- **SIMULATED** (`meta_toy.out`, rule R1 = M2's η·max(1/n_use, 0.1)): 0 of 20 changed answers were corrected after 8 salient re-teachings.
- **PROVED:** η_eff = 0.035 per teaching, so p_new = 1 − 0.965ⁿ against p_old = 0.965ⁿ. The answer flips only at n ≈ 20.
- **§2.4(b):** n_use is reset to min(n_use, 10). With a floor of 0.1, max(1/10, 0.1) = 0.1 is the floor itself. Inherited synapses stay at minimum plasticity, against "plasticity is high again in childhood". This holds if the prediction-form floor is the 0.1 of G12, which the text leaves ambiguous.

**Fix:** see §3 (the two-component synapse). At minimum, make the cap well above 1/floor, or tie the floor to age.

### #6 SERIOUS: LATEST loading is ungated, and event segmentation is world-fitted

**Evidence:**
- M1 says that at every onset the new event wins (λ_c), from any pathway (other, self, EYE/P4).
- SIMULATED with distractor onsets at r = 0.02 per tick, which is crowd-like: other voices are on 16–52 % of the day in world2 (CITED hearing). Untutored accuracy at 3,000 falls to 0.96, and to 0.84 when distractors also fall in the probe window. At r = 0.005 it is 0.97.
- PFC delay activity survives distractors, while sensory (IT) buffers are reset by each stimulus (Miller, Erickson & Desimone 1996). BG gates what enters working memory (O'Reilly & Frank 2006; McNab & Klingberg 2008, CITED).
- "Event = line, ended by an offset transient" works only because bench lines have no internal pauses. In real speech, pauses within a turn are often longer than gaps between turns (Heldner & Edlund 2010, CITED). The offset detector's time constant is a hidden world constant.

**Smallest fix:**
- A new event loads LATEST only through a competition: its drive × attention gain (P3 onset, P4b addressed) against the held event's. The partner's attended next line still always loads, which avoids ARCH_v2's #10 harm.
- Event boundaries = GEN's END prediction plus the offset transient: prediction-error segmentation (Zacks et al. 2007; Baldassano et al. 2017, CITED). That reuses parts the design already learns.

### #7 SERIOUS: world-fitted structure hidden in P4

**Evidence:** P4 detectors have fixed weights on world codes: "the person-code block", "approval contour", "smile", and "my name". An innate name detector is impossible: infants learn their name at about 4.5 months (Mandel, Jusczyk & Pisoni 1995, CITED). Fixed weights onto arbitrary world bit-codes is world knowledge wired into a prior. CONSPEC-like detectors work only because they are matched to real sensory physics (Johnson & Morton 1991, CITED).

**Smallest fix:**
- Define each P4 detector on sensor physics: a pitch contour on the voice channel, a configural face pattern on the retina.
- Make the world render those cues lawfully (W7/W8).
- Delete "my name"; it is learned through M2 plus contingent P5b.

### #8 MINOR–SERIOUS: three M1 parts have no robust binary-neuron form

- **λ_c = 0.3 graded holding.** Binary recurrent assemblies are bistable. A partial assembly either completes or dies, and graded holding needs fine-tuned line attractors (Seung 1996, CITED). Fix: carry recency in the age conjunction, or use activity-silent short-term facilitation (Mongillo, Barak & Tsodyks 2008).
- **"≤ k_th items" with a fixed-k k-WTA** forces spurious units on when fewer items are held. Fix: an inhibition ceiling (threshold-based, "at most k").
- **π from "best-match margins"** needs localist category readouts in a distributed code. Fix: compute π from the pool's own gap between the k-th and (k+1)-th drive, which the inhibitory pool has access to.

### #9 MINOR

- BTSP is a CA1 finding (Bittner et al. 2017). One-shot CA3 needs mossy-fibre "detonation" (Henze, Wittner & Buzsáki 2002; Vyleta, Borges-Merjane & Jonas 2016). Fix the citation.
- Real replay is prioritised, not uniform (Mattar & Daw 2018; Singer & Frank 2009; Ambrose, Pfeiffer & Foster 2016).
- The compute estimate (§7.5) prices silent ticks at 0.2 ms. But 85 % of ticks are silent, and those are where replay runs: CA3 iterations plus cortical M2 updates. That could roughly double the cost (ASSUMED).

---

## 3. Designer idea #22: strong, often-used weights resist change

**Brain evidence (CITED):**
- **Metaplasticity** (Abraham & Bear 1996; Abraham 2008) is mainly a BCM sliding threshold. It is homeostatic: after strong activity LTP gets harder and LTD *easier*. That is the opposite of "harder to lower".
- **Soft-bound STDP** (Bi & Poo 1998; van Rossum, Bi & Turrigiano 2000): LTP ∝ (w_max − w) and LTD ∝ w. Strong synapses are harder to raise but *easier* to lower. Only half agrees. The char-neuron network already uses concave growth (1 − w/w_max)^pow.
- **Spine stability** (Yang, Pan & Gan 2009; Xu et al. 2009; Matsuzaki et al. 2004; Hayashi-Takagi et al. 2015): lifelong retention is structural and *additive*. New spines form with learning; a few are kept for life; erasing them erases the skill. New learning uses new spines rather than rewriting old ones.
- **Tagging and capture** (Frey & Morris 1997; Redondo & Morris 2011): early LTP decays within hours unless captured. Capture is triggered by strong induction or by dopamine/noradrenaline novelty signals (Lisman, Grace & Duzel 2011; Takeuchi et al. 2016; behavioural tagging, Moncada & Viola 2007). It works while awake.
- **Reconsolidation:** a reactivated memory plus a prediction error becomes labile and can be updated (Nader, Schafe & LeDoux 2000; Sevenster, Beckers & Kindt 2013).
- **Cascade models** (Fusi, Drew & Abbott 2005; Benna & Fusi 2016): forgetting becomes power-law and lifetime ∝ N, but unrehearsed memories still fade.

**Toy** (SIMULATED, `meta_toy.out`; 5 seeds; GEN-like delta rule):
- Childhood: 32 old answers taught.
- Adult life: 20,000 events, with about 12 non-salient wrong lessons per old item, in the same context.
- At event 10,000, four answers change and are re-taught (salient) 8 times.
- Columns: old answers still right at 20k / changed answers right at 12k / changed answers still right at 20k / overheard new items right at 20k.

| rule | noisy drift | systematic drift |
|---|---|---|
| R0 plain | 0.00 / 1.0 / 0.15 / 1.0 | 0.00 / 1.0 / 0.00 / 1.0 |
| R1 = ARCH_v3 M2 | 0.91 / **0.0** / 0.0 / 1.0 | 0.73 / **0.0** / 0.0 / 1.0 |
| R2 = designer, rigidity on the signed weight | 0.86 / **0.0** / 0.0 / 1.0 | 0.70 / **0.0** / 0.0 / 1.0 |
| R4 tag + capture, gated by salience (τ_early 2,000 events ≈ 3 h) | 0.68 / 1.0 / 0.85 / 1.0 | 0.49 / 1.0 / 0.75 / 1.0 |
| R4 with τ_early 300 | 0.76 / 1.0 / 0.90 / 0.61 | 0.74 / 1.0 / 0.90 / 0.62 |

The untuned Benna–Fusi chain (R3) forgot unrehearsed items (old answers 0.0).

**What the toy shows:**
- **R2 behaves like R1, which M2 already contains.** It delays forgetting but blocks correction.
- **Why rigidity fails here.** With a winner-take-all readout, forgetting comes from a *competitor's* weak weight growing. Rigidity of the strong weight does not stop that.
- **No rule gets both.** With wrong lessons and corrections arriving in the same context, no local rule reached ≥ 0.95 retention *and* fast correction. A synapse can tell them apart only by salience or repetition. It does not remove v2.2's cause, which is wrong lessons in the right context. The anchor is the cure; metaplasticity is a second line.

**Circuit form I recommend** instead of rigidity by |w|. Apply it per excitatory synapse; the design already splits signed weights into E and E→I paths (S4).
- **Two components:** w = w_early + w_late.
- **w_early** is M2 at full rate and decays over about 10⁵ ticks (hours).
- **Tag** = a recent |Δw_early|.
- **Capture:** a δ burst inside the tag window moves κ·w_early into w_late, in either sign. This also gives reconsolidation when a memory is reactivated with a large prediction error.
- **w_late** changes only by capture.
- **Corrections** are expressed at once by w_early and the I path. They are carried one-shot by the hippocampus and consolidated by cued awake replay (awake replay is coupled to VTA activity, Gomperts, Kloosterman & Wilson 2015). This is complementary learning systems without sleep (McClelland, McNaughton & O'Reilly 1995).
- **Cost:** one more state variable per plastic synapse (+50–100 % weight memory). The slow part reads δ, which breaks "at most one broadcast" for that part; declare it. One gene (τ_early) and one constant (κ).

**"Kept for life without sleep"** is plausible for salient, consolidated knowledge. Corrections stay possible through the always-plastic early part, δ-gated capture, reconsolidation, and hippocampal one-shot memory plus replay.

**Prediction for X2:** the designer's rule will postpone v2.2's collapse, not remove it. X2 must report three things: teachings needed to switch a changed answer, adult new-item n₅₀, and re-learning of inherited timing.

---

## 4. Designer idea #23: a wider, distraction-proof thought and useful spontaneous recall

**Brain evidence (CITED):**
- **Two kinds of store.** IT-like buffers are reset by every stimulus; PFC is distractor-resistant (Miller, Erickson & Desimone 1996; Sakai, Rowe & Passingham 2002; Jacob & Nieder 2014). BG input gating protects it (O'Reilly & Frank 2006; McNab & Klingberg 2008). In bump attractors, weak distractors are resisted and strong ones overwrite (Compte et al. 2000; Brunel & Wang 2001).
- **Spontaneous thought:** the default-mode network (Raichle et al. 2001; Buckner, Andrews-Hanna & Schacter 2008; Christoff et al. 2016) is coupled to SWRs (Kaplan et al. 2016; Higgins et al. 2021). Its content leans toward current concerns (Klinger 2013).
- **Replay is prioritised:** by reward (Singer & Frank 2009; Ambrose et al. 2016), by gain × need (Mattar & Daw 2018), and toward weakly learned items (Schapiro et al. 2018).
- **Encoding is tagged for later use:** by dopamine (Shohamy & Adcock 2010; McNamara et al. 2014) and by SWRs during encoding (Yang et al. 2024).

**Mapping onto ARCH_v3.**
- LATEST is the IT-like reset buffer. It must stay unconditional for attended input: ARCH_v2 #10's surprise gate fell from 0.69 to 0.15 because lines must enter the learning context.
- THOUGHT is the protected PFC.
- But today λ_c = 0.3 demotes all held content at every onset. The effective depth is about 2–3 events, whatever k_th is, so **a wider k_th alone does nothing**.

**Realisable without a new mechanism:**
1. **THOUGHT entry by competition.** The outgoing LATEST enters only if its drive × (recency or attention gain) beats the weakest held item. Held items leave only by competition and fatigue. This is the char-network's single global threshold, which is M1's k-WTA with persistence and co-holding links. The binary-form caveat of #8 applies.
2. **Utility-weighted replay.** Write strength ∝ μ, with μ widened to include |δ| bursts as in fix #1. Recall is ∝ strength^ρ (already in M3). Replay is cued from THOUGHT (current concerns), and its priority is multiplied by the replayed successor's current prediction error, which limits itself (Mattar & Daw's gain). That guards against rich-get-richer rumination.
3. **Keep a write floor for non-salient events.** Overheard answers (W2) are useful but carry no reward to the agent.

**Cost** (SIMULATED, `wide_thought_toy.out`): widening the older context fed into MIX from 1 event to 6:

| | 1 event | 6 events |
|---|---|---|
| right after 150 formats | 0.88 | 0.74 |
| right at 3,000 untutored | 0.99 | 0.90 |
| at 3,000 with distractors | 0.96 | 0.84 |

There is no collapse. The toy overstates the cost: under P2's 6 branch sources only about 31 % of MIX units have a THOUGHT branch, which caps the dilution. So widen k_th moderately.

---

## 5. Other checks

**§6 chains with the stated numbers:**
- 6.1 and 6.2: the arithmetic is right. But formats with τ_f > 12 and every told answer arrive in replay mode (#1), and π needs a neural definition (#8).
- 6.3 greeting: works. The P5b fix lowers the reward for predictable replies (ASSUMED still enough).
- 6.4 extra turn: the learning signal has the wrong sign under P5b (#3, PROVED).
- 6.5 R-c: holds only under the stored-context replay reading (#1, SIMULATED).

**Hidden world fitting:**
- P4 on world IDs and an innate "my name" (#7);
- event = line set by offset transients (#6);
- the μ switch at 12 silent ticks placed between the world's 10-tick formats and 45-tick gaps (#1).

The answer/give-up delay ratio of 4.5 against the 2-octave bins passed the jitter test (SIMULATED).

**Migration and first experiments:**
- **X1:** add arms for the replay reading (stored vs live), distractor rate, independent jitter, and untutored extra-turn harvest under P5b.
- **New local X0 (minutes), before L6:** a CA3 module test at the design point and the fix point, with cued replay, the palimpsest horizon and θ_rec cue strength. The S6 swap is the biggest pure-neuron risk and sits too late at L12. L6 adds more writes (own lines).
- **X2:** carry the R2 rule with correction cost, as above.

**"Ends as pure neurons within realistic compute":**
- Yes for M1 (with #8), M2, M4, B1, B2, P1–P3 and P5.
- P4 needs the redefinition in #7.
- M3 is the bottleneck: about 3× the CA3 synapses at s = 1, and ∝ s² with scale (PROVED and SIMULATED). The s = 4 batch of 64 would no longer fit in the A800's 80 GB.

---

## 6. Verdict: fix, then build

The 3 fixes that matter most:
1. **Define μ fully** (#1). It gates all non-CA1 inputs of MIX and DG (including LATEST and TIME) plus M4's release. Replayed targets go to SYL's teaching input only. μ includes predicted input and |δ| bursts. In the toy this takes replay from 0.00 to 1.00.
2. **Make M3 buildable in neurons** (#2). Cued, salience-weighted replay; a sparse CA3 with a fixed active count (k ≈ 32, C ≈ 3,100 holds 10k episodes at 0.85) and fan-in ∝ s; a replay-fidelity check in S6; the module test before L6.
3. **Make P5b contingent** (#3): trace amplitude = 1 − GEN's predicted partner onset, own voice excluded. Then add source-specific M4 eligibility and an ε that anneals (#4).

Next in line: #5 (correction-capable two-component plasticity, which is also the right form for idea #22) and #6 (competitive LATEST and THOUGHT entry, which is also the right form for idea #23).

---

## summary_zh

1. 大方向在脑科学上站得住："最近一件事×过了多久"的时间细胞、自我发声的传出副本、基底节放行、乙酰胆碱"记录/回放"的方向都有实据。我还测了大人等答和放弃的时间各自随机打乱（0.5–2 倍），正确率仍有 0.96。
2. **致命问题（照原文）：回放和"手里拿着的问题"冲突。** 原文只让 μ 压低外部输入，最近一件事的锚点还连着当前的问题，回放内容就被学成"这个问题的答案"。小实验从 0.99 掉到 0。修法：μ 压住除海马以外的全部输入（包括"最近一件事"和计时），回放时不许说话；回放只送教学信号，不算新事件；μ 还要算上"预计马上有人说话"，等答案时不进入回放。
3. **海马换成真神经元后装不下：** 按原尺寸约存 1,000 段经历，一生要写约 1 万段。随机自发回放从来回不到存过的记忆，必须由当前内容提示。修法：编码更稀疏，同时活跃的单元数固定，连接数随规模增加（约 3 倍突触即可存 1 万段）；回放改为"有提示、按重要性"；在 L6 之前先单独测海马。
4. **"被回应"奖励只看前后是否接上，不看是否因我而起：** 会学会抢在对方开口前多说一句，长大后必然多话。改成"对方的回应超出预期才算"，自己的声音不算。
5. 其他问题：基底节闸门的功劳分配没写清，探索率永不下降；现有学习规则已让"改正一个熟答案"要约 20 次；任何声音都能挤掉"最近一件事"（人群里降到 0.84）；先天社交识别器写死了世界编码和"自己的名字"。
6. **您的 #22（常用强连接难改）：** 生物上更像"早期突触变化若不被巩固会消退，碰到有价值的时刻才转成长期的"，不是"越强越僵"。小实验里"越强越僵"能延缓遗忘，但 20 个改动的答案一个都改不过来。推荐"快、慢两部分突触 + 奖励时刻巩固 + 海马先记改正"，不用睡觉也能一辈子记住，同时还能改错。它只能推迟 v2.2 的崩溃，治本仍靠"最近一件事×时间"。
7. **您的 #23（更宽、不被打断的思考，独处时回忆有用的事）：** 不用加新机制就能实现。"最近一件事"照常更新，保证学习；"思考区"不再被每个新声音压低，新内容要靠竞争进入，就是您文字网络里的全局阈值；回放按重要性、由当前思考提示。代价是思考区越宽学得越慢（小实验 0.99→0.90，实际模型里会小一些），所以宽度适度即可。
8. **结论：修正后再建。** 最要紧的三件：定义清楚 μ 和回放、让海马能用真神经元实现、让"被回应"奖励看是否因我而起。
