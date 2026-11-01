(Saved by the coordinator from the agent's final hand-back. In the repo: the 29 MB of per-run results are not committed; `res/test_agg.txt` holds the full tables. `stream.json` comes from `../x345/record_stream.py`.)

# X9 report: item 14, the abstract concept area (offline module test)

**Verdict: PARTIAL.** Built from existing parts, the concept area gives held-out fillers class-like codes (37/40 correct), clear within-class sharing, cross-modal overlap and no collapse. The class-carrying code still drifts 0.114 over the last third, and the pass limit is 0.06.

Labels: PROVED = argument or count from the data; SIMULATED = my runs; ASSUMED = my judgement or shortcut. All test numbers are on seeds 0–9, mean ± SD; the form and parameters were chosen on dev seeds 100–101 only.

## 1. Verdict table

| item | verdict | key numbers (T6 = the refinement, 10 seeds) | paired T6 − L20 (same two layers, no context) |
|---|---|---|---|
| 14 concept area | **PARTIAL** | held-out class decode **37/40 = 0.925** (L20: 20/40) | held decode +0.425 ± 0.237 (up on 9/10) |
| | | within vs between **0.166 ± 0.020 vs 0.068 ± 0.009**, ratio 2.46, within higher on 10/10 | within − between +0.100 ± 0.021 (10/10); held margin +0.090 ± 0.026 (10/10) |
| | | cross-modal matched vs mismatched **0.062 vs 0.040** (10/10); 12-way top-1 0.62 ± 0.11 (chance 0.083) | top-1 −0.164 ± 0.121: context makes codes shared within a class |
| | | collapse index 0.040 (class layer), 0.033 (whole area); within-class overlap 0.17 (< 0.7) | collapse +0.009 |
| | | **drift, class code 0.114 ± 0.012 (FAIL, limit ≤ 0.06)**; whole-area word code 0.049 ± 0.005 (pass) | class-code drift +0.059 ± 0.014 (worse on 10/10) |
| | | leak control LEAN (held fillers removed from every class frame): 18/40 = 0.45, so no leak (limit ≤ 0.7) | |

The other arms, all SIMULATED (pooled held decode, within/between, class-code drift):
- **B0** (single layer, no context): 15/40, 0.025/0.031, drift 0.013. Best cross-modal: top-1 0.85.
- **T3** (β2 = 3): 29/40, 2.28×, drift 0.094. A weaker context lowers drift a little but loses class.
- **LETTER** (rule applied to the letter, 5 seeds): 19/20, 2.62×, drift 0.161. Whole-area drift 0.132, cross-modal top-1 0.49.
- **BENCH** (pure bench stream, 5 seeds): 10/20, 1.01×. Chance, as PROVED below.
- **Single-layer forms** (5 seeds each):
  - Tb: 12/20, 1.17×.
  - Tl: 16/20, but 1.04× and margin 0.02, so the codes are not class codes.
- **Dev seed 101, post-hoc, one seed** (β2, class-code drift / held):
  - β2 = 2: 0.060 / 3 of 4.
  - β2 = 3: 0.067 / 3 of 4.
  - β2 = 6: 0.093 / 4 of 4.
  - Other knobs (β2 = 6): a bigger code (10 %) 0.075; slower input-layer decay 0.092; 2.5× class dialogues 0.099, class stronger; half the dialogues: drift 0.117, held 2 of 4.
  - I found no setting that passes both drift and class.

## 2. What was built

There was no usable harness, so I wrote `concept.py`. It reuses X3's parts:
- the char assemblies (2,048 units, 24 per char, physics seed 7, X3's char order);
- the EYE bits;
- the `Thr` count control (smoothed quantile with a 2× ceiling; instant quantile at test);
- the two-layer weight form w/√(out·in);
- concave growth and the one global decay.

**Data** (PROVED that it is needed): in the bench Life stream the only food-only frames are `X好不好吃` and `X好吃`. In both, X opens the line, so a food has no class-specific left context. Holding out taste would leave a held food with no class information at all. So, as in v2/meaning's probe, I inserted 1,200 class exchanges into silent daytime gaps:
- 你要X吗 (all objects), 你吃X吗 (foods), 你玩X吗 (toys), X好不好吃;
- the eye shows X during the exchange, using the bench's own eye codes;
- 2 foods and 2 of 皮球/汽车/鸭子 are never heard in the taste frame (the bench's own taste lines for them are removed too).

**Final form, L2.** The area got two layers, which ARCH §四.14 leaves open ("要不要分几层").
- **C1** (2,000 units, about 6 % on): birth fan-in of 40 AUD + 6 EYE comparable inputs, with feed-forward learning by the rule.
- **C2** (2,000 units, about 6 % on): no sensory input.
  - Context: the previous moment's C1 activity through born random links K (fixed).
  - Current item: C1 → C2 links F, born silent, learned by the rule.
  - Drive: d = 6·K-push + F-push.
- Words that follow the same context are learned onto the same randomly indexed C2 units. This is the neural version of v2's index lines, with no per-item lines and no batch PMI.
- Composition check (seed 0, SIMULATED; chance per set 0.06):
  - foods' class codes hold 0.12–0.51 of the 吃-indexed units; toys hold 0.17–0.33 of the 玩-indexed units;
  - animals also hold 0.28–0.36 of the 的-indexed units (我的X叫);
  - held fillers carry their class's units: 牛奶 0.51 吃, 汽车 0.27 玩.

**Why this form** (Amendment A in PLAN.md, written before any test seed):
- **Single layer fails** (SIMULATED, dev): the char's own birth code reinforces itself and leaves at most 5 % of the state to context. When the context is made strong enough to enter, drift rises to 0.15–0.34 and identity is lost.
- **The context must act in the same update as the arriving word** (PROVED from the rule). A prediction that lights units before the word arrives only grows predicted → word links, and the rule then weakens word → predicted links. This rules out a PFC prediction arriving a tick early.
- **Choice of route:** I used the previous moment's activity, not a PFC return route. The designer's order is PFC first, and a 1:1 return chain that arrives in the same update gives the same arithmetic.

**Rule deviation (declared).** The "just off" case of §二.8 is not applied to the feed-forward chain inputs (η_ff_off = 0). Reason (PROVED): 吃's input links to food units would grow 6× faster than any one food's, so 吃 would light the food units first, and the rule would then weaken the arriving food's links to them. The LETTER arm shows the cost: stronger class code, but drift 0.13–0.16 and cross-modal top-1 0.49.

**Other changes, all declared:**
- Two-layer weights are computed per source area (harvest h4 #10), because the eye had swamped the area (AUD share of the state was 4 %). Gains: eye 0.07, context 6.
- Class metrics use the deep layer's first-char code, with one item per first char (小 counted once). PROVED reason: a word's second char always has the same left context, its first char.
- Deep-layer decay γ 0.99995 per 200 ticks, the slowest in the area, as §二.8 asks for deeper layers.

**Shortcuts (ASSUMED):**
- event-driven updates (X3's);
- "local" fan-in = random sampling, because the assemblies have no topography;
- VOICE is not fed into the area.

**Compute:** about 5 min and 240 MB per life on the A800 CPU queue; 70 test runs and about 30 dev runs; the runner worked throughout.

**What it means for the full model:**
- The two-layer area can supply class meaning.
- The world must teach at least two class-specific frames; the bench Life does not.
- The open problem is stability: a form in which the deep layer's own code holds it in place without losing class.

## 3. Files

In the agent's working folder (the repo copy holds the code, plans, specs and `res/test_agg.txt`):
- `PLAN.md`: plan and Amendment A.
- `concept.py`: the harness, all arms.
- `batch.py`, `mkjob.sh`, `agg.py`, `show.py`, `show2.py`: job runner and table tools.
- `diag3.py`–`diag6.py`: diagnostics; `diag6.py` is the composition check.
- `stream.json`: the bench stream (copy).
- Specs: `dev1.txt`, `dev2a.txt`, `dev2b.txt`, `dev3.txt`, `testA.txt`, `testB.txt`.
- `res/test/*.json|log`: 70 test runs, with code snapshots at 2/3 of life and at the end.
- `res/test_agg.txt`: the full tables and paired differences.
- `res/dev1`, `res/dev2`, `res/dev3`, `res/local`, `res/local_dev101.txt`: dev runs.
- `concept_v1.py`: the single-layer version.

## summary_zh
1. 第14条 抽象概念区：部分行。
2. 用现有零件搭成两层：第一层接听觉、视觉的链式输入；第二层不接感觉，只接"上一刻第一层的活动"（出生随机连接）和"这一刻第一层"（学出来的连接），同一条赫布规则。
3. 没教过"X好不好吃"的东西，能按类别认对：37/40 = 0.93（不加上下文时 0.50，就是瞎猜）。
4. 同类编码比异类像：0.166 对 0.068，10 个种子全都如此；没有坍缩（不同字的重叠只有 0.04）。
5. 听到名字和看到东西会点亮相似的神经元：0.062 对 0.040，10/10；12 选 1 认对 0.62（瞎猜 0.08）。
6. 不行的地方：类别编码在一生最后三分之一里还变了 11%（要求 ≤6%），整个区的字编码只变 4.9%（合格）。上下文越强，类别越清楚，但也变得越多，没找到两全的参数。
7. 只用一层不行：字自己的编码每听一次就加强一次，上下文挤不进去。
8. 现在的基准世界本身教不出"食物"这一类（食物只出现在"X好不好吃"的开头）；世界里要有第二个只给食物用的句式，比如"你吃X吗"。
9. 规则的一处细化：前一个字"刚灭"时，不加强感觉输入到概念区的连接。否则"吃"会先点亮食物的神经元，反而把"苹果"连过去的线减弱（照字面规则跑，编码每段时间变 13–16%）。