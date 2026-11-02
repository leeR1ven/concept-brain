# X10 code-to-clause map

Every row of PLAN.md §1 (conformance table) and every register entry of §3 / §4, with the file:function that implements it. Files are in `arch/v2/x10/`. "—" means not built, with the reason.

## A. Conformance table (PLAN §1)

| § | clause | implemented in |
|---|---|---|
| 2.1 | binary neurons | every state is a `torch.bool` tensor: `learning.py:Pop`, `sound.py:SoundArea.step`, `pfc.py:PFC.step`, `hippocampus.py:Hippocampus.step`, `motor.py:Motor.step` |
| 2.1 | inhibitory = ordinary neurons with negative outputs | blocking and stop neurons: `motor.py:Motor.step` (`inh` term); patch inhibitory neurons: `pfc.py:PFC.inhib`, `hippocampus.py:Hippocampus.inhib1`; Hs same-tick mean-drive inhibition: `hippocampus.py:Hippocampus.step` |
| 2.1 | distributed codes, no item neurons | no syllable id in any brain module. Ids appear only in `world_adapter.py`, `wiring.py:measure_codes` (birth measurement for G1) and `probes.py` (scoring) |
| 2.1 | thermometer pairs, always exactly half on | `world_adapter.py:Cochlea.code` (asserts half on every tick) |
| 2.1 | deeper sensory layers sparser; threshold set from the unit's own received signal, rising with depth; sensory only | `sound.py:SoundArea.calibrate` (per-unit (1 − r_l) quantile of its own birth drive; r_l = 0.20 / 0.12 / 0.08 / 0.05). PFC thresholds are fixed per layer: `pfc.py:PFC.calibrate` |
| 2.1 | signal strength = number lit | loudness → thermometer levels: `world_adapter.py:Cochlea.energy`, `Cochlea.code`. Split/delete off |
| 2.1 | own vs other voice | `world_adapter.py:Cochlea.energy` (`own_level` 0.6 on content; own timbre channels) |
| 2.2 | 3-D local sparse net, a few nearby inputs | `sound.py:SoundArea.__init__` (K = 6 inputs within ±dy channels, ±dz levels; radius widens with depth) |
| 2.2 | near-input layers fixed | `sound.py` has no learning; `ByPost` tables never change (`learning.py:ByPost`) |
| 2.2 | similar sounds → similar units | measured: `sizing/sound_stats2.py`; checked in (a2): `probes.py:newborn_codes` |
| 2.2 | vision | — not needed (no picture in the world) |
| 2.2 | concept area | — staged arm C+ (§四.14); `common.py:config(concept=False)`, not yet built |
| 2.3 | 1:1 chains from every layer to the PFC and the hippocampus | `brain.py:Chains.__init__`, `Chains.gather`; the same chain vector feeds `pfc.py:PFC.step(x_ext)` and `hippocampus.py:Hippocampus.step(x_ext)` (`brain.py:Brain.step`) |
| 2.3 | each forward chain has one return to the same S | `brain.py:Chains.scatter` (`fwd_src == ret_dst`); return drive: `sound.py:SoundArea.step` (`ret * ret_w`), `motor.py:Motor.step` (`ret_force`, `ret_stop`) |
| 2.3 | no chain for always-on or never-on units (designer 11-02: a tunable cut, only near the input) | `brain.py:Chains.__init__` (band on birth rates, applied to sound layers below `band_depth_sound`; deeper units all chained); per-area knobs `band`, `band_sound`, `band_motor`, `band_depth_sound` in `common.py:config`; report `brain.py:Brain.make_chain_report`. A0 and tonic blocking neurons are never sources |
| 2.3 | chain count follows split/delete | — not needed (§二.9 off) |
| 2.4 | L1, two middle layers, last layer | `pfc.py:PFC.__init__` (`ff[0..2]`), `PFC.step` |
| 2.4 | middle layers fixed | `ByPost` tables in `pfc.py:PFC.__init__`; no learning on them |
| 2.4 | last(t) → L1(t+1) Hebbian, local | `pfc.py:PFC.__init__` (`L41`, radius `r_learn`), `PFC.learn`, `PFC.l1_post` |
| 2.4 | loop holds a thought; internal lines count fully | birth loop `pfc.py:PFC.__init__` (`loop`, `geom.py:reciprocal_loop` / `random_local_loop`); `PFC.step` (`x = x_ext \| x_int`, then L2 reads `L1.on` with no halving) |
| 2.4 | count preservation | `pfc.py:PFC.calibrate` (fixed θ_l and patch-neuron gain g_l per layer, set once at birth so each layer's count follows the layer below tick by tick); checked by `unit_checks.py` #8 and `probes.py:activity` / `a1_pass` |
| 2.4 | PFC inhibition order (P0 → P1) | `pfc.py:PFC.inhib` (`inhib='P0'` or `'P1'`); grid in `run_a.py` |
| 2.4 | motor inhibitory neurons kept | `motor.py:Motor` |
| 2.4 | two-layer weights by out and in | `learning.py:ByPre.drive` (non-birth branch), totals kept by `ByPre._apply` and `ByPre.recount` |
| 2.4 | thinking-range size | follows from `pfc.py:PFC.calibrate` (count preservation); `rho` in `common.py:config` for arm ρ4 |
| 2.4 | rarely co-active inhibit, related excite | net effect of `pfc.py:PFC.inhib` (patch neuron) and the learned `L44` (`PFC.step`, `PFC.learn`); diagnostic d7: `probes.py:lateral_sign` |
| 2.4 | continuous, no resets; excitability knob | no reset anywhere. `probes.py:Frozen` restores neuron states after probes only. Knobs `g_loop`, `w44` in `common.py:config` |
| 2.4 | speech caused by thought | `brain.py:Brain.step`: the only routes to muscles are G1 (from sound A1) and motor returns, in `motor.py:Motor.step` |
| 2.5 | output = internally lit L1 copies → 1:1 returns | `pfc.py:PFC.step` returns `x_int`; `brain.py:Brain.step` → `Chains.scatter` → `SoundArea.step` / `Motor.step` next tick |
| 2.5 | return strength tunable; return weaker than next-item link | `common.py:config(g_ret)`; `sound.py:SoundArea.calibrate` (`ret_w`); d10 in `probes.py:table_stats` |
| 2.5 | prediction biases hearing | `sound.py:SoundArea.step` (return drive adds to the feed-forward drive) |
| 2.6 | own population; chains in; fast Hebbian; fastest decay | `hippocampus.py:Hippocampus.__init__`, `step`, `learn`; γ_H in `common.py:gammas` |
| 2.6 | recall via return chains, never PFC directly | `hippocampus.py:Hippocampus.step` returns `h_int` → `brain.py:Brain.step` (`R = x_int \| h_int`) → `Chains.scatter`; no Hs → PFC table exists |
| 2.6 | stores everything; no records | one rule on tables (`learning.py:ByPre.learn`); no per-memory structures |
| 2.6 | no birth Hs ↔ Hs recurrence | `hippocampus.py:Hippocampus.__init__` (`HH` starts at 0, acquired only) |
| 2.7 | muscles only; tonic blocking; force grades | `motor.py:Motor.__init__`, `Motor.step` (muscle on iff ≥ ⌈s/2⌉ of its force neurons) |
| 2.7 | competition, no hard gate | `motor.py:Motor.step` (ladder blocking recruited by pooled drive P; no count rule) |
| 2.7 | blocking without chains; stop with chains | `brain.py:Chains.__init__` (force and stop sources only) |
| 2.7 | brake reads only the input layer, graded by loudness | `world_adapter.py:Cochlea.__init__` (`brake_lines`: other voices' up lines; own timbre left out = birth wiring); `motor.py:Motor.step` (`brake`) |
| 2.7 | sound → muscle hand-wired in gen 0, decaying | `wiring.py:wire_g1`, `motor.py:Motor.set_g1` (acquired `ByPre`, γ_motor) |
| 2.7 | babbling, within-syllable order, motor association area | — not needed (overhearing; X1c body) |
| 2.7 | stop after speaking learned; PFC stop | stop copies are PFC inputs (`brain.py:Chains`); learned only via `PFC.learn` |
| 2.8 | held-form rule; co-rising; fall does nothing | `learning.py:ByPre.learn` with the eligibility from `learning.py:Pop.elig_prev` / `Pop.rose` |
| 2.8 | always-on shallow units grow no links | they are not chain sources (`brain.py:Chains`), so they reach no acquired table; G1 sources are chain units only (`wiring.py:wire_g1`) |
| 2.8 | concave growth, ≥ 1 % plasticity; concave weakening and decay | `learning.py:ByPre.plast`, `ByPre.learn`, `ByPre.decay` |
| 2.8 | one decay; birth exempt; PFC slowest, hippocampus fastest | `common.py:gammas`; `learning.py:ByPre.decay` (`birth=True` tables return early); `brain.py:Brain.step` (every `decay_every` ticks) |
| 2.8 | time = order; no timers or self-hold | no such term in any step function |
| 2.8 | reward | — not needed; `brain.py:Brain.step` asserts `r == 0.0` |
| 2.9 / 2.10 | split/delete; amygdala | — not needed |
| 2.11 | one continuous life; overhearing; systematic steps 1–3 | `life.py:Life` (birth, then the same `OverhearWorld` continues); `run_b.py`, `run_c.py` |
| 2.12 | gen-0 wiring registered, decaying, tests on unwired content | `wiring.py` docstring and `wire_g1`; registry saved by `run_b.py` / `run_c.py`; QA probes on all pairs: `probes.py:qa_probe` |
| 2.12 | whole-brain scaling | `common.py:config(s)`; sizes in `world_adapter.py:Cochlea`, `sound.py`, `motor.py`, `brain.py:Chains` (PFC = chain count), `hippocampus.py` (`nh_per_s`, `chh_per_s`, `ch1_per_s`) |
| 2.12 | pure neurons; registered equivalents | equivalents listed in §B (C1 return, patch means, ELL tables) |
| 2.12 | sealed questions | held-out pairs never spoken (`world6.OverhearWorld`); scored in `probes.py:qa_probe` (`held`) |
| — | no symbolic input to the brain | `brain.py:Brain.step` asserts (bool A0 of the right shape, `r == 0.0`); `unit_checks.py` #5 |

## B. Register entries (PLAN §3, §4)

| id | implemented in |
|---|---|
| W1 graded channels | `world_adapter.py:Cochlea.energy` |
| W2 own voice 0.6 | `world_adapter.py:Cochlea.energy` (`own_level`) |
| W3 thermometer 3s levels | `world_adapter.py:Cochlea.__init__`, `Cochlea.code` |
| W4 world unchanged | `world.py`, `world6.py` (byte-identical copies of x1g); `world_adapter.py:Env` replaces only `_render` |
| S1 four layers | `common.py:config(n_layers=4)`, `sound.py:SoundArea.__init__` |
| S2 K = 6 local inputs, comparable weights, radius widening with depth (revised 11-02) | `sound.py:SoundArea.__init__` (`sound_fan`, `sound_dy`, `sound_dz`) |
| S3 2 units per position | `common.py:config(units_per_pos=2)` |
| S4 per-unit threshold from own birth drive (revised 11-02) | `sound.py:SoundArea.calibrate` |
| S5 chain band and PFC-size presets (small / medium default / large; deeper layers cut at `band_rel` × their own target rate) | `common.py:config`, `common.py:PFC_SIZE`; resolved in `brain.py:Brain.__init__`; `brain.py:Chains.__init__`, `Brain.make_chain_report`, `Brain.motor_birth_rates` |
| S2′ no unit wired to need an impossible coincidence | `sound.py:SoundArea.__init__` (continuous weights, own-drive thresholds); dead or always-on units are reported, not removed |
| S6 no sound learning | `sound.py` (`ByPost` only) |
| C1 return on internal activation; g_ret 1.05 × own threshold | `pfc.py:PFC.step` (`x_int`), `hippocampus.py:Hippocampus.step` (`h_int`); `sound.py:SoundArea.calibrate` (`ret_w = g_ret · max(thr, w_min)`), `motor.py:Motor.step` |
| C2 A0 has no chain | `brain.py:Chains.__init__` (sound layers A1–A4 only) |
| C3 one-tick relays | `brain.py:Brain.step` (gather from t − 1 states; returns applied next tick) |
| P1 two middle layers | `pfc.py:PFC.__init__` |
| P2 sizes, positions | `pfc.py:PFC.__init__` (`n = rho · n1`), `brain.py:Chains.__init__` (copy positions) |
| P3 feed-forward 24 in r 0.1; fixed calibrated thresholds | `geom.py:local_fanin`, `pfc.py:PFC.calibrate` |
| P4 P0 / P1 patch inhibition (gain g_l per layer) | `geom.py:patches`, `pfc.py:PFC.inhib`, `PFC.calibrate` |
| P5 net signed effective lateral weight | `pfc.py:PFC.step` (`lat`, `L44`, then `inhib`) |
| P6 reciprocal or random birth loop | `geom.py:reciprocal_loop`, `geom.py:random_local_loop`, `pfc.py:PFC.__init__` |
| P7 L4 → L1 learning, post = chain-input onset | `pfc.py:PFC.l1_post`, `PFC.learn`; same definition in `hippocampus.py:Hippocampus.learn` |
| P8 L4 ↔ L4 birth + acquired | `pfc.py:PFC.__init__`, `PFC.learn` |
| P9 two-layer weights with floors | `learning.py:ByPre.drive`, `ByPre._apply`, `ByPre.recount` |
| P10 internal activation, full count | `pfc.py:PFC.step` |
| P11 thinking range by count preservation | `pfc.py:PFC.calibrate` |
| P12 gains | `common.py:config` (`g31`, `g44`, `g_loop`, `w44`, `g_pm`); grid in `run_a.py` |
| R1 rule | `learning.py:ByPre.learn` |
| R2 rates | `common.py:config` (`eta`, `eta_m`, `w_ref`, `floor`; `eta_h`, `eta_h_m`) |
| R3 decay, PFC slowest | `common.py:gammas`, `learning.py:ByPre.decay` |
| R4 acquired ≥ 0 | `learning.py:ByPre._apply` (`clamp_min(0)`) |
| R5 no reward | `brain.py:Brain.step` (assert) |
| M1 force grades | `motor.py:Motor.__init__` (`thr_f`), `Motor.step` |
| M2 blocking: tonic + ladder | `motor.py:Motor.step`; ladder set in `wiring.py:wire_g1` |
| M3 stop | `motor.py:Motor.step` |
| M4 brake (own timbre excluded = birth wiring) | `world_adapter.py:Cochlea.__init__` (`brake_lines`), `motor.py:Motor.step` |
| M5 G1 from sound A1 (equal latency with the brake) | `wiring.py:wire_g1` (sources `chains.sound_idx[0]`), `motor.py:Motor.g1_drive` |
| M6 release scale | `wiring.py:wire_g1` (gain, ladder) |
| H1 hippocampal copies | `brain.py:Brain.step` (the same `x_ext` feeds `Hippocampus.step`) |
| H2 Hs: fixed recode, fixed threshold, same-tick inhibition | `hippocampus.py:Hippocampus.__init__`, `calibrate`, `step` |
| H3 no birth recurrence; learned HH, H1t | `hippocampus.py:Hippocampus.__init__`, `learn` |
| H4 no store/recall switch | `hippocampus.py:Hippocampus.step` (learned links always act) |
| H5 nothing added for alike-starting episodes | — (nothing to implement) |
| G1 gen-0 sound → muscle | `wiring.py:measure_codes`, `select_codes`, `nnls`, `g1_quality`, `wire_g1` (A-matrix variant chosen by the map's own quality); `motor.py:Motor.set_g1` |
| DPC (diagnostic) | — only if (c) fails; not built |
| DIAGNOSTIC forced inner hearing | `brain.py:Brain.step(force_int)` → `pfc.py:PFC.step(force_int)`; used only by `probes.py:forced_say`, `speech_basics` |
| birth period | `life.py:Life.__init__` (child silent, learning off) |

## C. Harnesses and checks

| stage | file |
|---|---|
| (a) a1 / a2 / a3 over the grids (`--grid loop`: inhibition × loop form × g_loop × w44; `--grid size`: PFC size × threshold fractions + the old band-everywhere) | `run_a.py`, `probes.py:activity`, `a1_pass`, `question_codes`, `decode`, `newborn_codes` |
| (b) b1–b7 | `run_b.py`, `probes.py:speech_basics`, `forced_say` |
| (c) checkpoints | `run_c.py`, `probes.py:qa_probe` (primary pass, d3, d4), `newborn_codes` (d1, d2), `table_stats` (d6, d10), `return_share` (d8), `lateral_sign` (d7), d9 inline. d5 (G1 sharpness) is not coded: `motor.last` holds the drive for it |
| unit checks | `unit_checks.py` |
| timing pilot | `timing.py` |

## Notes

1. **Not built yet.** These need a decision or are staged:
   - the concept area (C+);
   - the DPC diagnostic arm.
2. **Chained s ≥ 8 lives.** These need a checkpoint that survives between runner jobs. The runner deletes each job folder, so this needs the coordinator's runner support.

## D. Implementation-only pieces (not part of the model)

- `learning.py:ByPre` `dense` path: tables of ≤ `dense_limit` entries are scanned whole each tick, with no host sync. Larger tables gather the active rows only. Both give the same rule and drive.
- `common.py:Prof`: synchronised per-section timing (`timing.py`).
