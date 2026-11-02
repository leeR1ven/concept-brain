(Saved by the coordinator from the agent's final hand-back.)

**X4b report: the designer's merge rule (40/40/20 + slight noise) in the X4 harness, and whether a merged child relearns faster**

Labels: PROVED = code or arithmetic. SIMULATED = a run (files listed at the end). ASSUMED = my judgement.

## 0. Verdict

- **Does 40/40/20 keep more than mean or sign at birth?** No. It keeps the same as the plain mean and much less than sign. Disjoint lessons right on day 0: cross 0.43, mean 0.42, sign 0.73. In the pool of 4: 0.12, 0.13 and 0.50.
  - PROVED: per connection, the expected child is 0.4A + 0.4B + 0.2(A+B)/2 = (A+B)/2, which is the mean. For a pool of n it is Σ(0.8/n)·D_k + 0.2·mean = mean.
  - Each answer's drive sums about 650 connections, so the random picks average out (SIMULATED).
- **Main result: is the 40/40/20 + noise child a head start?** No. It learns both parents' lessons clearly **slower** than a fresh birth (paired CIs exclude 0):

  | scenario | savings vs fresh |
  |---|---|
  | fixed_in | −29 % |
  | 3 % split/delete | −34 % |
  | pool of 4 | −55 % |

  - On the lessons it does not know at birth, the child needs about twice the repetitions (6.5 vs 3.3).
  - It is still much faster than a parent learning the other parent's lessons. A parent takes about 9 repetitions where fresh takes 2.3, and knows only 0.28–0.43 of the new lessons after 10 repetitions.
- **Biggest savings:** sign merge + the same slight noise, with the inherited learned part shrunk to 10 % at birth (one strong pass of the existing global decay; no new mechanism).

  | scenario | savings | share of taught lessons right at birth ("innate") |
  |---|---|---|
  | fixed_in | +42 % | 0.75 |
  | 3 % split/delete | +28 % | 0.53 |
  | pool of 4 | +30 % | 0.48 |

  - All CIs exclude 0.
  - It learns the remaining lessons at fresh speed and reaches the same final level as fresh.
  - Second best is sign ×0.3: +39 / +19 / +27 %. The designer's cross with the same ×0.1 shrink gives +27 / +21 / +9 %.
- **Limit (SIMULATED):** in absolute terms the best saving is about 1 repetition. Fresh reaches its plateau in about 2 repetitions in this harness (η = 0.3). All the saving comes from the innate share; the unknown remainder is not learned faster. ASSUMED: in a world where fresh learning is slower, the absolute saving grows with the innate share.

## 1. Setup

**Harness:** `x4_merge.py`, copied unchanged and imported. Same lineage, agents, rule, lessons (20 shared, 30 disjoint per parent, 15 conflict), raising (10 repetitions + 300 filler lines), readout and 3 seeds.
- Check (SIMULATED): the mean and sign rows reproduce X4 exactly — 0.42 / 0.73, pool 0.13 / 0.50, conflict-none 0.93 / 0.78.

**What the operators act on:** the learned deltas D = w − birth, in the reference sibling's index space, mapped by stable ID as in X4.
- PROVED: with a shared birth wiring this is identical to merging total weights, because every parent's total is birth + D_k and the rule keeps that ≥ 0.
- In the split/delete case, the deltas are what each parent learned. They go onto the reference's birth wiring, as in X4.

**Operators:**
- 'cross': per connection, mean with probability 0.2, otherwise one parent drawn uniformly. That is 40/40/20 for two parents and 0.2 each + 0.2 mean for four. Measured proportions: 0.400 / 0.399 / 0.200, and 0.198–0.201 for the pool.
- 'pick': the same with no mean component.
- **Noise:** each connection's total weight × (1 + σ·N(0,1)), clipped at 0. Main σ = 0.1; σ = 0.03 and 0.3 were also run. 'noise' touches only the inherited PFC→AUD links. 'noise_all' also perturbs the AUD→PFC and PFC→PFC birth links.

**The child (ASSUMED reading of the 11-01 addendum plus ARCH §二.8):**
- The merged links become the child's birth links: no decay, learned part starts at 0.
- Variant 'cross/learned': the inherited part keeps decaying. Result −26 % vs −29 %: no real difference, because 0.999 per 200 ticks removes only about 10 % in 10 rounds.
- 'shrink ×λ': the inherited learned part is multiplied by λ before birth. PROVED: ×0.1 equals about 2,300 passes of the 0.999 decay, roughly half a life of disuse.

**Life after birth:**
- 10 rounds. Each round: every non-conflicting lesson once (shared + all disjoint) plus len/2 fillers, shuffled, with the same order for every agent. Tests after each round, with no learning.
- "Reps to correct" = first round answered right; 0 = innate; never within 10 is counted as 11.
- Savings = 1 − reps_child / reps_fresh.
- Paired difference = per lesson against fresh, with a bootstrap 95 % CI over lessons pooled across the 3 seeds. Seed-level variance is not separated.
- Fresh = the lineage birth with the reference's structure.
- Parents 0 and 1 hear the same curriculum; their new lessons are the other parents' lessons.

## 2. Life after birth (main result; SIMULATED)

Column notes:
- **innate** = right at day 0, on all taught lessons / on the disjoint lessons only.
- **Δreps [CI]** = paired difference against fresh; negative means the child is faster.
- **not known at birth** = mean reps for the child vs fresh on the lessons the child did not know at day 0.

### fixed_in (2 siblings)

| child | innate | right after 1 / 2 / 10 reps | reps | savings | Δreps [CI] | not known at birth |
|---|---|---|---|---|---|---|
| fresh | 0 / 0 | 0.67 / 0.83 / 0.83 | 2.25 | – | – | – |
| mean | 0.55 / 0.42 | 0.60 / 0.65 / 0.78 | 3.12 | −39 % | +0.88 [+0.41, +1.34] | 6.9 vs 3.3 |
| sign | 0.77 / 0.73 | 0.81 / 0.80 / 0.81 | 1.69 | +25 % | −0.56 [−0.91, −0.21] | 7.4 vs 5.0 |
| cross 40/40/20 | 0.55 / 0.43 | 0.60 / 0.67 / 0.79 | 2.93 | −30 % | +0.68 [+0.24, +1.13] | 6.5 vs 3.3 |
| **cross+noise(0.1)** | 0.55 / 0.43 | 0.60 / 0.66 / 0.79 | 2.91 | **−29 %** | +0.66 [+0.21, +1.10] | 6.5 vs 3.3 |
| cross+noise_all(0.1) | 0.51 / 0.38 | 0.58 / 0.60 / 0.77 | 3.27 | −45 % | +1.02 [+0.54, +1.48] | 6.6 vs 3.1 |
| cross/learned | 0.55 / 0.43 | 0.60 / 0.67 / 0.80 | 2.84 | −26 % | +0.59 [+0.17, +1.02] | 6.3 vs 3.3 |
| cross+noise ×0.3 | 0.53 / 0.39 | 0.71 / 0.81 / 0.82 | 1.92 | +15 % | −0.33 [−0.60, −0.04] | 4.0 vs 3.2 |
| cross+noise ×0.1 | 0.48 / 0.33 | 0.74 / 0.82 / 0.83 | 1.64 | +27 % | −0.61 [−0.79, −0.43] | 3.1 vs 3.0 |
| sign+noise | 0.77 / 0.73 | 0.81 / 0.80 / 0.82 | 1.70 | +25 % | −0.55 [−0.91, −0.19] | 7.4 vs 5.0 |
| sign+noise ×0.3 | 0.76 / 0.72 | 0.82 / 0.84 / 0.83 | 1.36 | +39 % | −0.89 [−1.15, −0.63] | 5.7 vs 4.8 |
| **sign+noise ×0.1** | 0.75 / 0.70 | 0.75 / 0.83 / 0.83 | 1.31 | **+42 %** | −0.94 [−1.18, −0.71] | 5.2 vs 4.6 |

Parents 0 / 1 learning the other's lessons:
- New lessons right after 0 / 3 / 10 reps: 0.00 / 0.06 / 0.40 and 0.02 / 0.10 / 0.41.
- Reps 9.2 / 8.9 against fresh's 2.3 on the same lessons: Δ +7.3 / +6.2.

### 3 % split/delete (stable IDs)

| child | innate | right after 1 / 2 / 10 reps | reps | savings | Δreps [CI] | not known at birth |
|---|---|---|---|---|---|---|
| fresh | 0 / 0 | 0.65 / 0.78 / 0.83 | 2.45 | – | – | – |
| mean | 0.45 / 0.30 | 0.51 / 0.57 / 0.76 | 3.37 | −38 % | +0.93 [+0.45, +1.39] | 6.1 vs 3.1 |
| sign | 0.57 / 0.48 | 0.62 / 0.67 / 0.79 | 2.68 | −10 % | +0.24 [−0.16, +0.64] | 6.2 vs 3.7 |
| cross | 0.45 / 0.30 | 0.52 / 0.59 / 0.76 | 3.27 | −34 % | +0.83 [+0.37, +1.30] | 5.9 vs 3.1 |
| **cross+noise(0.1)** | 0.45 / 0.30 | 0.52 / 0.60 / 0.76 | 3.27 | **−34 %** | +0.82 [+0.36, +1.29] | 5.9 vs 3.1 |
| cross+noise_all(0.1) | 0.39 / 0.26 | 0.49 / 0.55 / 0.69 | 3.79 | −55 % | +1.35 [+0.85, +1.83] | 6.2 vs 3.0 |
| cross+noise ×0.3 | 0.43 / 0.28 | 0.66 / 0.77 / 0.82 | 2.13 | +13 % | −0.31 [−0.62, −0.03] | 3.7 vs 3.1 |
| cross+noise ×0.1 | 0.40 / 0.25 | 0.71 / 0.78 / 0.83 | 1.92 | +21 % | −0.53 [−0.80, −0.28] | 3.2 vs 3.0 |
| sign+noise | 0.56 / 0.48 | 0.62 / 0.67 / 0.79 | 2.67 | −9 % | +0.23 [−0.17, +0.64] | 6.1 vs 3.6 |
| sign+noise ×0.3 | 0.55 / 0.47 | 0.73 / 0.75 / 0.82 | 1.98 | +19 % | −0.47 [−0.75, −0.18] | 4.4 vs 3.6 |
| **sign+noise ×0.1** | 0.53 / 0.45 | 0.73 / 0.79 / 0.83 | 1.75 | **+28 %** | −0.70 [−0.93, −0.47] | 3.8 vs 3.5 |

Parents learning the other's lessons:
- New lessons right after 10 reps: 0.43.
- Reps 8.6–9.0: Δ +5.7 to +6.9.

### Pool of 4

Fresh plateaus at 0.63 here because 140 lessons exceed the harness's capacity, so lessons never learned (counted as 11) weigh heavily in every mean.

| child | innate | right after 1 / 2 / 10 reps | reps | savings | Δreps [CI] | not known at birth |
|---|---|---|---|---|---|---|
| fresh | 0 / 0 | 0.45 / 0.58 / 0.63 | 3.77 | – | – | – |
| mean | 0.25 / 0.13 | 0.30 / 0.35 / 0.53 | 5.97 | −58 % | +2.20 [+1.77, +2.65] | 8.0 vs 4.2 |
| sign | 0.56 / 0.50 | 0.59 / 0.63 / 0.64 | 3.14 | +17 % | −0.63 [−1.01, −0.23] | 7.1 vs 5.4 |
| pool-cross (0.2 each + 0.2 mean) | 0.24 / 0.12 | 0.31 / 0.36 / 0.54 | 5.82 | −54 % | +2.05 [+1.63, +2.49] | 7.7 vs 4.2 |
| **pool-cross+noise(0.1)** | 0.24 / 0.12 | 0.31 / 0.36 / 0.54 | 5.83 | **−55 %** | +2.06 [+1.64, +2.50] | 7.7 vs 4.2 |
| pool-cross+noise_all(0.1) | 0.22 / 0.11 | 0.28 / 0.31 / 0.51 | 6.13 | −63 % | +2.36 [+1.91, +2.81] | 7.9 vs 4.2 |
| pool-cross+noise ×0.3 | 0.23 / 0.11 | 0.42 / 0.52 / 0.63 | 3.93 | −4 % | +0.16 [−0.12, +0.44] | 5.1 vs 4.2 |
| pool-cross+noise ×0.1 | 0.19 / 0.06 | 0.49 / 0.58 / 0.63 | 3.43 | +9 % | −0.34 [−0.55, −0.15] | 4.2 vs 4.1 |
| sign+noise | 0.57 / 0.51 | 0.59 / 0.64 / 0.64 | 3.14 | +17 % | −0.63 [−1.02, −0.24] | 7.2 vs 5.5 |
| sign+noise ×0.3 | 0.54 / 0.49 | 0.61 / 0.64 / 0.64 | 2.74 | +27 % | −1.03 [−1.35, −0.71] | 6.0 vs 5.3 |
| **sign+noise ×0.1** | 0.48 / 0.42 | 0.57 / 0.61 / 0.62 | 2.62 | **+30 %** | −1.15 [−1.40, −0.90] | 5.1 vs 5.0 |

Parents 0 / 1 learning the other three's lessons:
- New lessons right after 10 reps: 0.34 / 0.28.
- Reps 9.1–9.4: Δ +5.4 / +5.5.

## 3. Which lessons are innate (SIMULATED)

**Shared lessons** (both parents knew them): 0.88–0.97 right at birth under every operator.

**Disjoint lessons:** it depends on how confident the owning parent was (its margin = right answer's drive / best other's drive).

| scenario | parent margin in the top half | bottom half | parent itself did not answer |
|---|---|---|---|
| fixed_in, cross | 0.63 | 0.22 | 0.00 |
| fixed_in, sign | 0.97 | 0.49 | 0.07 |
| pool of 4, cross | 0.20 | 0.04 | – |
| pool of 4, sign | 0.74 | 0.27 | – |

- In the split/delete case, the reference parent's lessons are innate at 0.47 (cross) / 0.76 (sign). The other parent's are innate at only 0.13 / 0.20.
- The child inherits the parents' strongest knowledge; weak knowledge is lost.
- Shrinking costs a little innate share: sign ×0.1 gives 0.70 / 0.45 / 0.42 vs 0.73 / 0.48 / 0.50 for plain sign. In return, it removes the slowdown on the rest.

## 4. Day 0 (secondary; SIMULATED)

Columns: shared lessons right | disjoint lessons right | disjoint answered wrongly | conflict → no answer.

**fixed_in** (parents' own lessons: 0.83)

| operator | shared right | disjoint right | disjoint wrong | conflict none |
|---|---|---|---|---|
| mean | 0.92 | 0.42 | 0.04 | 0.93 |
| sign | 0.90 | 0.73 | 0.01 | 0.78 |
| cross | 0.92 | 0.43 | 0.05 | 0.93 |
| pick | 0.92 | 0.40 | 0.05 | 0.93 |
| cross+noise σ 0.03 / 0.1 / 0.3 | 0.92 | 0.43 / 0.43 / 0.45 | 0.04–0.05 | 0.93 |
| cross+noise_all σ 0.03 / 0.1 / 0.3 | 0.92 / 0.93 / 0.78 | 0.41 / 0.41 / 0.17 | 0.03–0.07 | 0.91–0.96 |
| sign+noise ×1 / ×0.3 / ×0.1 | 0.90 / 0.90 / 0.88 | 0.73 / 0.72 / 0.70 | 0.01 | 0.78 / 0.80 / 0.84 |

PFC code overlap (Jaccard) with the un-noised child under noise_all: 0.82 / 0.62 / 0.36.

**3 % split/delete** (per parent: reference / other)

| operator | shared right | disjoint right | disjoint wrong | conflict none |
|---|---|---|---|---|
| mean | 0.88 | 0.30 (0.47 / 0.13) | 0.09 | 0.71 |
| sign | 0.83 | 0.48 (0.76 / 0.20) | 0.03 | 0.27 |
| cross, and cross+noise(0.1) | 0.88 | 0.30 (0.47 / 0.13) | 0.12 | 0.69 |
| pick | 0.88 | 0.28 | 0.09 | 0.67 |
| cross+noise_all σ 0.1 / 0.3 | 0.83 / 0.70 | 0.24 / 0.16 | 0.11 / 0.09 | 0.73 / 0.93 |
| sign+noise ×0.1 | 0.78 | 0.45 | 0.02 | 0.33 |

**Pool of 4** (parents' own lessons: 0.85)

| operator | shared right | disjoint right | disjoint wrong | conflict none |
|---|---|---|---|---|
| mean | 0.95 | 0.13 | **0.34** | 0.78 |
| sign | 0.88 | 0.50 | 0.01 | 0.96 |
| cross, and cross+noise(0.1) | 0.97 | 0.12 | **0.35 / 0.34** | 0.73 |
| pick | 0.95 | 0.12 | 0.34 | 0.84 |
| cross+noise_all σ 0.1 / 0.3 | 0.92 / 0.75 | 0.10 / 0.03 | 0.35 / 0.37 | 0.71 / 0.80 |
| sign+noise ×0.1 | 0.83 | 0.42 | 0.01 | 1.00 |

In a pool, mean and cross make the child answer about a third of the disjoint lessons **wrongly** at birth; sign does so about 1 % of the time. Childhood errors are allowed (ARCH §二.11), but this is another point for sign.

## 5. Why merged children and adults relearn slowly (SIMULATED, `diag.py`, seed 0)

- The PFC units of a new question already carry strong links to old answers: out-strength 175 in a parent vs 10 in a fresh birth. AUD in-strength is 291 vs 20.
- PROVED: under two-layer weights each new increment is divided by √(S_out·S_in), so the same Δw counts about 16 times less in the adult.
- Parent, new lessons: the right answer's drive goes 0.79 → 1.45 → 1.67 θ over two repetitions. The old competitor stays at about 2.3 θ, so 0 → 3 % right.
- Fresh: 0.6 → 3.25 θ after one repetition, 77 % right.
- The merged child inherits this load. Shrinking the inherited part lowers it, which is why ×0.1 restores fresh learning speed.
- ASSUMED: this is a general "adults learn new things about 4× slower" effect in the current harness. It bears on ARCH §二.11 (keeps learning all life) and §四 item 7 (two correction mechanisms, new things land on fast connections first), not only on merging.

## 6. Notes for ARCH §四 item 12 (ASSUMED; for the coordinator to decide)

- **Where 40/40/20 belongs.** In IDEAS #19 the 40/40/20 rule was stated for "genes", the innate-table numbers. There it is fine: a few numbers, no distributed codes. For learned links, sign election keeps far more.
- **Recommended merge for learned links.** Sign election, slight noise, and a strong shrink at birth (λ ≈ 0.1–0.3). This uses the existing global decay; λ could itself be a gene.
- **Where the noise goes.** Noise only on the inherited learned links, not on the shared birth wiring.
  - σ = 0.1 on all links moves the PFC codes by about 40 % and costs 10–20 points of savings.
  - Noise drawn per child would also break the next generation's mergeability, which X4 showed needs identical birth wiring.
  - σ ≤ 0.3 on the learned links changes nothing measurable.

## 7. Rules and files

**Box rules:**
- Ran `free -m` first.
- One python process at a time: pilot → diag → main (17.5 min) → follow-up (8.2 min). All waiting was done in bash.
- Peak 381 MB (main) and 360 MB (follow-up).
- No server jobs. Repo and x345 untouched.
- The follow-up re-ran fresh with the same seeds; it reproduced the main run exactly (assert in `x4b_main_table.py`).

**Files** in this folder (`v3/x4b/`):

- **Code:**
  - `x4_merge.py`: the unchanged X4 harness, imported from `../x345/`.
  - `x4b.py`: operators and life.
  - `x4b_extra.py`: sign with shrink.
  - `diag.py`: the mechanism diagnostic.
  - `x4b_table.py`, `x4b_main_table.py`: the table scripts.
- **Results:**
  - `x4b.json`, `x4b.out`, `x4b_table.out`.
  - `x4b_extra.json`, `x4b_extra.out`, `x4b_main_table.out`.
  - `pilot.json`, `time_x4b.txt`, `time_extra.txt`.

## summary_zh
1. 设计者的"40% 随父、40% 随母、20% 取平均"用在学到的连接上，平均起来就等于简单平均：只有一方会的课，出生时保住 0.43（简单平均 0.42，按正负号选 0.73）；四个一起融合时只有 0.12，还有三分之一会答错。
2. 主要结果：按这个规则（再加一点随机）生出的后代，重学父母的课比一个全新出生的模型还慢 29%–55%；出生时不会的那些课，它要多花大约一倍的次数。
3. 原因：后代继承了大人连接已经很满的前额叶单元，新学的连接被"摊薄"了。父母自己去学对方的课更慢，10 次后只学会三到四成。
4. 最省时间的做法：按正负号选来融合，加一点随机，出生时把继承来的部分整体降到原来的 10%（相当于用现有的统一降权狠狠降一次）。这样三种情况分别省 42%、28%、30%，出生时就会约一半到四分之三的课，其余的课和新生儿学得一样快，最后水平也一样。
5. 省下的绝对时间约 1 次重复，因为在这个测试里新生儿 2 次左右就学到顶了。好处几乎全来自"天生就会"的那部分，剩下的课并不会学得更快。
6. 天生会的，是父母学得最牢的课；父母自己就不太会的，后代也几乎不会。
7. 随机只能加在继承的连接上：加到输入和回路的出生连线上，前额叶编码会变约四成，效果变差，下一代也没法再融合。
8. 建议：40/40/20 留给基因（本能参数），学到的连接改用"按正负号选 + 稍加随机 + 出生时整体降权"。另外，大人学新东西比新生儿慢约 4 倍，这一点关系到"一直学习"，值得单独研究。