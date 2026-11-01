(Saved by the coordinator from the agent's final hand-back.)

# X3 / X4 / X5 report: experiments for the designer's pure-neuron core

All work is in this folder (`v3/x345/`). There was one GPU job and no SSH.

**Labels:** PROVED = code or arithmetic; SIMULATED = a run (file named); CITED = a source; ASSUMED = judgement.

## 0. Bottom line

| | result | pass? |
|---|---|---|
| **X3** hippocampus, cue = 50 % of the stored hippocampal pattern | recall 1.00 at every age bin below 10 k ticks; 0.86 at 10–20 k; 0.17 at 20–50 k | **PASS** within a horizon of ≈10 k ticks (≈0.5 bench day, ≈50 heard lines) |
| X3, same cue, best settings (η ×4, fan-out 3,100, or s = 2) | horizon ≈20 k ticks | PASS |
| X3, end to end: cue enters through the chain copies at 50 % of the input units | ≤ 0.2 at every age | **FAIL**: the sparse recode also separates the cue from its own stored code |
| **X4** merge | disjoint lessons: plain mean 0.42, sign-elected 0.73 (parents 0.83); conflicts give no answer 0.93 (mean) / 0.78 (sign); alignment changes nothing | **FAIL** |
| X4, siblings differing by 3 % of AUD units (stable IDs) | the non-reference sibling's lessons are lost (0.09–0.22); alignment does not repair it | FAIL |
| **X5** tick cost | local numpy, s = 1: 7.3 ms per speech tick. A800 torch, s = 1: 10.7 / 13.6 / 15.8 ms per batched tick at B = 1 / 16 / 64, i.e. 0.25 ms per agent-tick at B = 64. s = 4: 11.3 / 15.1 ms at B = 1 / 16; B = 64 does not fit under the 18 GB cap | measured |

## 1. Sources checked (working rule) and the binding updates applied

**Read:**
- DESIGNER_SPEC (rows 14–24);
- r2b §3, §5, §6;
- IDEAS #19 and #22–#24;
- **`docs/ARCHITECTURE.md`**, followed as authoritative after the coordinator's message: §二.1 threshold not quota, gain control; §二.3 chains only for feature units; §二.4 two-layer weights; §二.6 the hippocampus; §二.8 the rule, including reverse weakening; §三, the list of wrong old rules;
- the neuro report and toys (`ca3_*.py`, `merge_toy.py`) and the ml `merge_toy.py` / `ca3_capacity.py`. I found no separate neuro addendum file;
- designer engine code: chain repo `plastic2.py` (rule, decay every T, prune), `brain.py` (`link()` 1:1, `remove()` keeps IDs), `structure.py` (split: 40–60 % usage band, 2 children × 2 inputs, appended IDs);
- ana `plastic.py` (η_seq 0.02 > η_sim 0.008, two-layer budget, retain_by_depth), ana `dentate.py` / `hippocampus.py` (time ring, now withdrawn), ana ARCHITECTURE / NEXT_STEPS §6 (chain → hippocampus → Hebbian = short-term memory);
- `designer_msgs_clean.txt` messages 41, 44, 48, 50, 110, 119, 128, 137, 537, 540.

**Problems solved from the sources:**
- **Common units swamp recall.** With plain weights recall was 0.0; with two-layer propagation p = w/√(S_out·S_in) it worked (ARCH §二.4).
- **Activity ran away along a line** (code sizes 5 → 128) through the birth recurrence. Fixed by gain control, "总激活越多，越难再往上推" (ARCH §二.1), implemented as inhibition equal to the mean recurrent drive. Fresh recall went from 0.8 to 1.0.
- **Fixed quota.** Replaced by a smoothed quantile tracker (count varies) with a 2× inhibitory ceiling (ARCH §二.1 and §三). Top-k survives only in the experimenter's decoder.
- **Moving recall forward.** Refractoriness and "later inhibits earlier" are forbidden (§三). I used η_seq > η_co (ana's ratio) plus reverse-order weakening (§二.8).
- **Sustained units.** Kept the strict "only just-changed units" rule (plastic2; msgs 48/50).
- **Coordinator's binding answers, all applied:**
  - recall goes out through the hippocampus's own return chains (learned Hs → AUD/VOICE) and is measured at the source AUD neurons (decoded char, plus assembly fidelity);
  - copies exist only for feature units (never-on AUD/VOICE/EYE units get none; fan-ins are scaled to the feature share);
  - split/delete with stable IDs is tested in both X3 and X4.

## 2. X3: the designer's hippocampus

**Setup (SIMULATED; `x3_hippo.py`, `x3_main.py`, `x3_results.json`):**
- **Stream:** the recorded `stream.json` (7 MB, not committed: regenerate with `python3 -I record_stream.py`), 120 days, tick by tick, 3 ticks per char.
- **Copies:** 1:1 chain copies of AUD (2,048 units, 24-unit char assemblies from a fixed physics seed), VOICE (8-unit sets) and EYE (1,024 world bits).
- **Probes:** novel 6-char lines of familiar chars, put into silent daytime gaps. Kinds:
  - unique;
  - repeated 3× back to back;
  - repeated 4× spread over 3 k ticks;
  - similar pairs sharing a 3-char prefix: same scene and voice, same scene with another voice, or a different scene.
- **Hippocampus:**
  - its own population Hs: N 10,000, set point k 64, fixed random fan-in from the copies;
  - its own random birth recurrence (designer #5), which carries order as context with no time cells;
  - plastic Hs ↔ Hs links: fan-out 2,000 potential partners;
  - the rule: falling → rising +η·g, rising ↔ rising +η_co·g, reverse −η·g, with g = 1/(1+w);
  - one decay, γ = 0.99 every 200 ticks, links below 0.01 deleted;
  - no records, no per-memory cells.
- **Test:** cue = a fraction of the stored Hs pattern of moment 3. Free recall runs 3 steps; score = chars 4–6 correct at AUD. A second, end-to-end test feeds the first 3 moments through the copies.
- **Memory:** about 350–500 MB per run.

**Recall against age** (base; unique probes; mean, with n per bin):

| age (ticks) | 0–1 k | 1–2 k | 2–5 k | 5–10 k | 10–20 k | 20–50 k | ≥ 50 k |
|---|---|---|---|---|---|---|---|
| 50 % stored cue | 1.00 (5) | 1.00 (12) | 1.00 (13) | 1.00 (34) | 0.86 (62) | 0.17 | 0.00 |
| 20 % stored cue | 0.93 | 0.86 | 0.87 | 0.77 | 0.51 | 0.05 | 0 |
| full input through copies | 0.93 | 0.94 | 0.90 | 0.81 | 0.54 | 0.07 | 0 |
| 50 % input through copies | 0.20 | 0.17 | 0.13 | 0.10 | 0.04 | 0 | 0 |
| fidelity at AUD neurons (50 % cue) | 0.94 | 0.95 | 0.94 | 0.89 | 0.75 | 0.20 | 0.02 |

**Horizon** = the last age bin with recall ≥ 0.9 at 50 % cue; in brackets, recall at 5–10 k / 10–20 k.

| knob | values → horizon |
|---|---|
| γ per 200 ticks | 0.95 → 2 k (0.73 / 0.12); 0.98 → 10 k (0.96 / 0.70); **0.99 → 10 k** (1.00 / 0.86); 0.995 → 10 k (0.96 / 0.89); 0.999 → **none**: fresh recall only 0.3–0.47 |
| η (×0.25 / ×1 / ×4) | 10 k (0.92 / 0.80) / 10 k / **20 k** (1.00 / 0.95) |
| active count k (32 / 64 / 128) | 10 k for all; 0.94 / 1.00 / 0.92 at 5–10 k; 0.79 / 0.86 / 0.73 at 10–20 k |
| connectivity C (1,000 / 2,000 / 3,100) | 10 k (0.77 at 10–20 k) / 10 k / **20 k** (0.94) |
| size s (N_H 2.5 k / 5 k / 10 k / 20 k; k ∝ s; C fixed) | 1 k / 5 k / 10 k / 20 k. In limit, at C 1,500: s = 1 gives 0.85 and s = 2 gives 0.91 at 10–20 k |

What the sweeps show (all SIMULATED):
- **The horizon is not monotonic in γ.** Forgetting too slowly (0.999) saturates the network, so even fresh episodes fail. The palimpsest needs the decay.
- Before links are pruned, recall is limited by fresher overlapping links. Decay alone would allow ≈100 k ticks (PROVED: 2·0.99ⁿ < 0.01 at n ≈ 527 decays).
- With a fixed fan-out and a fixed active fraction, the horizon grew about ∝ s here. That contradicts the reviewers' flat-capacity prediction for random patterns; I did not isolate the mechanism.

**Repeated against unique** (at 10–20 k):
- unique 0.86; 3× back to back 0.96; 4× spread 0.91;
- all fail beyond 20 k;
- so concave growth re-strengthens the same links without piling up, as the designer intends.

**Random kicks against cued recall:**
- base, 2,000 kicks: 13 % silent; 22 % produce a heard world bigram; 11 % a probe bigram; the rest are chimeras. Cued recall is 1.00;
- copy arm: 74 % of kicks run into familiar world sequences ("returns to a topic").

**Separation of similar episodes** (age < 20 k, 50 % cue, shared 3-char prefix; right continuation / the twin's continuation):

| arm | same scene + voice | other voice | other scene |
|---|---|---|---|
| base (recode + birth recurrence) | 0.95 / 0.00 | 1.00 / 0.00 | 0.98 / 0.00 |
| recode, no recurrence | 0.46 / 0.14 | 0.75 / 0.07 | 0.62 / 0.15 |
| no recode (copies only) | 0.39 / 0.39 | 0.36 / 0.36 | 0.31 / 0.31 (a coin flip) |

Without the recode the hippocampus stores only pooled char transitions: fresh recall is 0.67 and it cannot separate.

**Split/delete tolerance** (paired runs, same probes; 3 % or 5 % of feature copies deleted with IDs kept and the same number of children added at t = 100 k; `x3_pert.json`):
- stored-cue recall unchanged: 1.00 / 1.00 / 1.00 at every age;
- fidelity at AUD neurons unchanged (0.99–1.00 below 10 k);
- end-to-end input cue loses 0.09–0.18 (for example 0.80 → 0.64 / 0.67 at 5–10 k).

**Assumptions and shortcuts (ASSUMED):**
- Hs is recomputed only when its input changes (event-driven); one recall step = one transition, so replay is 3× compressed.
- Learned links do not act during encoding. Letting them act cut fresh recall to 0.08.
- The copy → Hs links are fixed. Trained ones collapsed (rich get richer).
- EYE is not read out.

**Open:** a working end-to-end degraded cue through the copies needs a learned input path that does not collapse.

## 3. X4: merge (SIMULATED; `x4_merge.py`, `x4_merge.json`)

**Setup:**
- One lineage birth wiring and a fixed physics seed.
- AUD 1,024 units (16-unit assemblies); PFC 2,048 units (smoothed quantile threshold, birth recurrence with gain control).
- Lessons "q1 q2 → a" learned in the PFC → AUD prediction links with the designer's rule; cortex decay 0.999 every 200 ticks; 300 filler lines.
- Release only if the best char beats the inhibitory default and is sharp (≥ 1.25 × the second).
- 3 seeds.

**Results:**

| scenario | operator | disjoint lessons | conflict: no answer |
|---|---|---|---|
| fixed near-input links; code Jaccard between siblings 0.84 | plain mean | 0.42 | 0.93 |
| | sign-elected (TIES, trim 10 %) | 0.73 (parents 0.83) | 0.78 |
| | either, with alignment | the same: alignment is 86 % identity and changes nothing | |
| pool of 4 siblings | mean | 0.13 | 0.78 |
| | sign-elected | 0.50 | 0.96 |
| siblings with different 3 % AUD split/delete (stable IDs) | sign-elected | reference sibling 0.76–0.78; the other 0.09–0.22, with or without alignment | |
| trained PFC input links (ARCH §三), two-layer normalised | any | parents themselves 0.22 | |

**Findings:**
- **Fail.** Distributed codes share PFC units across lessons, so the merge adds the other sibling's competing links.
- 3 % different input units change the sparse PFC codes by about 60 % (Jaccard 0.84 → 0.42). That is not a permutation, so activation matching cannot repair it.
- Trained input links destabilise learning before any merge.

**Open:** merge before structures diverge, a merge operator that keeps the union of both structures, or less brittle codes (ASSUMED).

## 4. X5: tick benchmark (SIMULATED; `x5_bench.py`, `x5_numpy.json`, `x5_areas.json`, `x5_torch.json`, `x5_torch_run.log`)

**r2b areas at s = 1 (sizes ASSUMED):** ear thermometer pairs, two local layers, AUD, voice, eye, chain copies, a 16 k PFC, the 10 k X3 hippocampus, motor. That is 55 k units and 32 M synapses (130 M at s = 4). The model uses CSR by pre, the quantile threshold, Hebbian learning on changed units and decay every 200 ticks.

| | speech tick | silent tick | per agent-tick |
|---|---|---|---|
| local numpy, whole core, s = 1 (RSS 484 MB) | 7.3 ms | 6.3 ms | 7.3 ms. Per area, 8.0 ms in total, of which the hippocampus's drive is 3.9 ms |
| local numpy, s = 4 (per area; fan-out capped at 500 and scaled linearly) | ≈38 ms | ≈35 ms | |
| local numpy, s = 4 with the hippocampal fan-out also ×4 | ≈97 ms | | |
| A800, s = 1, B = 1 / 16 / 64 | 10.7 / 13.6 / 15.8 ms | 6.4 / 12.1 / 14.1 ms | 10.7 / 0.85 / **0.25** ms; peak 9.2 GB at B = 64 |
| A800, s = 4, B = 1 / 16 | 11.3 / 15.1 ms | | 11.3 / 0.94 ms |
| A800, s = 4 at B = 64, or s = 1 with the hippocampal fan-out ×4 at B = 64 | out of memory: ≈31 GB of weights > the 17.6 GB cap | | |

**Implications (PROVED arithmetic on the measured ticks):**
- s = 1, one 10⁶-tick life on one local core: ≈7,000 s. r2b §4 assumed 1,000–2,000 s, so it is about 4× optimistic; the 20 M-synapse hippocampus dominates.
- A800 at B = 64: ≈250 s per agent-life, i.e. 64 lives in ≈4.4 h, or 3 chained 90-min jobs. That is 16× better than gpuport's 259 ms per batched tick.
- s = 4 with L ∝ s: ≈3,800 s per agent-life at B = 16, i.e. ≈17 h for 16 lives.
- The GPU is kernel-launch bound: a floor of about 10 ms per tick. The world simulation is not included.

## 5. Rule exceptions (honest)
- The `s_2` X3 run peaked at **642 MB**, over the 600 MB limit. I reran it in limit as `s_2_C1500` (517 MB); its numbers are in the size row above.
- Three early debug runs (`dbg5.py`, before chunking) reached 1.0–1.14 GB. I fixed this before running the suite.
- Debug scripts (`dbg*.py`, `grid.py`) are left in the folder.

## summary_zh
1. **X3 海馆（您的形式）通过，但有条件。** 设定：自己的一群二值神经元，靠链式副本输入、赫布连接、凹形增长、统一降权；不存记录，也没有时间神经元。
   - 给出存储模式的一半作提示，1 万拍以内回忆是 1.00，1–2 万拍降到 0.86，之后忘掉（约半天、50 句话）。
   - 学得快一点（η×4）、连接多一点、或者规模大一倍，能记约 2 万拍。
   - 降权太慢（0.999）反而全记不住：遗忘是必需的。
   - 重复几次只略好一点，不会越存越多，符合您说的凹形增长。
2. **海马内部的稀疏重编码加自身的出生随机回路是必需的。**
   - 有了它，开头相同的两段话分得开（0.95–1.00，对方的接续一次都没出现）。
   - 只用链式副本时，两段话各占一半，等于掷硬币。
   - 随机刺激大多回不到真的经历（只有 11–22% 回到听过的片段）。
3. **问题：** 如果提示从感觉区副本进来、只给一半输入单元，回忆只有 0.2 以下。稀疏重编码把提示本身也"分开"了，需要一条学出来、又不会塌缩的输入通路。
4. **分裂/删除：** 3–5% 的副本单元增删后，海马内部回忆和送回听觉区的保真度不变；从感觉区进来的提示下降 0.1–0.2。
5. **X4 融合没通过。**
   - 不同课程的平均融合只保留 0.42；按符号选的融合保留 0.73（父母 0.83）；矛盾的课融合后大多不回答（0.93）。
   - 激活对齐没有作用。
   - 两兄弟的输入层若有 3% 单元不同，前额叶编码就变了约六成，另一方学的几乎全丢，对齐也救不回来。
6. **X5 计算量：** 本地 numpy 每拍约 7 毫秒，一生约 2 小时，比 r2b 估计的慢约 4 倍，主要花在海马。A800 上 64 个模型一起跑，每个模型每拍 0.25 毫秒，一生约 4 分钟；规模 4 倍时 64 个装不下显存上限。
7. **违规说明：** 有一次实验内存到了 642MB（超过 600MB），已在限额内重跑；早期三次调试曾到 1GB，已修正。
