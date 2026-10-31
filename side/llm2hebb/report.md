(Saved by the coordinator from the agent's final reply; the harness blocked the agent's own write.)

# llm2hebb: converting a back-propagation LLM into a Hebbian-style model that keeps learning (updated full report)

**Coverage.** This report covers R1–R3 from the first report plus the new tests 1, 1b, 2 and 3. Everything is standalone; no ARCH_v2 code was used.

The harness blocks a `report.md`, so this message is the report. Base folder: this directory (`side/llm2hebb/` in the repo).

Labels: **PROVED** = argument or arithmetic, **SIMULATED** = my GPU run (file given), **CITED**, **ASSUMED**.

## Setup

**Infrastructure.**
- All new jobs (g4–g7) ran in `queue_a800gpu`, one at a time; the V100 is retired.
- Models came from ModelScope's copies of the public repositories, because hf-mirror.com stalled.
- Packages are in `~/v2lab/venv_llm/site` (installed with pip `--target`).
- The A800 disk now holds about 6.3 GB of my models, out of 30 GB.

**Models.**
- Qwen2.5-0.5B-Instruct: used for R2, R3 and tests 1, 1b and 2.
- RWKV-7 World 0.1B / 0.4B / 1.5B: used for R1 and test 3. I wrote my own RWKV-7 implementation; it matches the official package to |Δlogit| ≤ 4e-5.

**Facts.**
- Invented two-token names, e.g. `王子涵的小狗叫栗蔻`. Recall means both tokens are exactly right.
- Test 1, 1b and 2 use the same 10k facts as R3.

**Evaluation sets.**
- Three phrasings of each question:
  - rewrite: `X的Y叫`
  - paraphrase: `大家都知道，X的Y叫`
  - QA: `问：X的Y叫什么名字？\n答：X的Y叫`
- Leakage is measured on unwritten subjects in two sets: other people, and the same person with another relation.
- General ability is measured by corpus perplexity and by 9 world-knowledge probes. Nine probes is a coarse measure.

## 1. Literature map (CITED; unchanged)

**Exact local equivalents of back-propagation.**
- The delta rule is exactly back-propagation for the last layer.
- Predictive coding converges to back-propagation (Whittington & Bogacz 2017; Millidge et al. 2022, 2023). It equals back-propagation *exactly* under the fixed-prediction assumption (Song et al. 2020).
  - Costs: the transposed Jacobians of the next layer, and at least L relaxation steps per update.
  - Its inference landscape becomes ill-conditioned in deep, trained networks (Innocenti, Achour & Buckley, NeurIPS 2025, μPC).

**Approximations that cannot inherit a transformer as it is.**
- Equilibrium propagation needs an energy-based network with symmetric weights (Scellier & Bengio 2017).
- Feedback alignment and DFA work only because the forward weights *learn* to align with the feedback (Lillicrap et al. 2016; Nøkland 2016; Launay et al. 2020).
- Target propagation needs trained inverses (Lee et al. 2015; Bartunov et al. 2018).

**Already Hebbian: fast weights.**
- Linear attention, DeltaNet and RWKV-7 update their state with S += v kᵀ or the delta rule at inference (Schlag, Irie & Schmidhuber 2021; Yang et al. 2024, 2025; Peng et al. 2025).
- The state is fixed in size, and its recall lags softmax attention (Arora et al. 2024).

**One-shot writes into the slow weights.**
- ROME/MEMIT write facts with a rank-one, delta-rule-form update (Meng et al. 2022, 2023).
- Sequential edits collapse after about 10²–10³ edits (Gupta et al. 2024).

**Complementary learning systems.**
- The brain pairs a fast sparse hippocampus with slow interleaved cortical consolidation (McClelland, McNaughton & O'Reilly 1995; Kumaran et al. 2016).
- LLM analogues: kNN-LM (Khandelwal et al. 2020), Memorizing Transformers (Wu et al. 2022), Larimar (Das et al. 2024).
- Fine-tuning on new facts raises hallucination (Gekhman et al. 2024), and retrieval beats fine-tuning for injecting knowledge (Ovadia et al. 2024).
- DG-style coding (random expansion + k-WTA) separates patterns (FlyHash: Dasgupta et al. 2017); sparse autoencoders follow Gao et al. 2024.

## 2. R1: RWKV-7 fast weights in one never-reset stream (SIMULATED; `results/g2`, `results/g6`)

**PROVED.** The state update S ← S·diag(w) − (Sk̂)(a⊙k̂)ᵀ + v kᵀ is a local delta rule run at inference. Here the Hebbian equivalence is exact.

Recall of the cloze form against the distance in tokens between the fact and the question:

| model | condition | <64 | 64–256 | 256–1k | 1k–4k | >4k | QA <64 |
|---|---|---|---|---|---|---|---|
| 0.1B | reset every 4k (in distribution) | 0.97 | 0.81 | 0.34 | 0.02 | 0 | 0.72 |
| 0.4B | reset every 4k | 0.92 | 0.64 | 0.40 | 0.08 | 0 | 0.61 |
| **1.5B** | reset every 4k | 0.99 | 0.78 | 0.38 | 0.11 | 0 | 0.96 |
| 0.4B / **1.5B** | never reset | 0.29 / 0.30 | 0.07 / 0.13 | 0.005 / 0.005 | 0 | 0 | 0.14 / 0.26 |

Held-out cross-entropy read from the never-reset state, by tokens already streamed:

| model | fresh | 4k | 16k | 64k | 256k | 1M |
|---|---|---|---|---|---|---|
| 0.1B | 4.04 | 3.94 | 3.99 | 4.49 | 5.28 | 6.71 |
| 0.4B | 3.88 | 3.71 | 3.75 | 4.07 | 5.85 | 9.86 |
| **1.5B** | 3.69 | 3.49 | 3.54 | 4.17 | **10.6** | – |

- **Test 3 (1.5B).** The 1.5B model has a far better QA at short delay (0.96), but the same horizon (cloze 0.38 at 256–1k tokens, 0 beyond 4k).
- It drifts out of distribution even faster without resets: the state norm goes from 42 to 308 at 256k tokens. Inside the stream, CE at 64–128k is 5.35 never-reset vs 3.49 with resets.
- **Reading.** Scale does not turn the fast-weight state into a long-term memory. It is a one-shot working memory of about 10²–10³ tokens, and it needs resets.

## 3. R2: local rules on the slow weights (Qwen 0.5B; SIMULATED, `results/g1–g3`)

**Cosine between each rule's update and the true gradient:**
- predictive coding with the fixed-prediction assumption: **1.000** (exactly back-propagation);
- standard predictive coding: diverges at every step size from 1 down to 0.03 (cos ≈ 0, update norm 10²–10¹⁵ × the gradient);
- layer-local delta rule: 0.20 (1.00 at the top block, < 0.1 below the middle);
- DFA: 0.00.

**Online stream** (30 facts × 3 presentations, then 60 chat lines):
- Only Adam back-propagation learned facts: 47 % after 3 presentations, 13 % after the delay; knowledge 9/9 → 3/9.
- The local rules (layer-local delta, DFA, back-propagation on W_down only, SGD) learned ≤ 7 % at any learning rate, and still lost knowledge.
- Being local does not remove forgetting.

## 4. R3: one-shot Hebbian rank-one writes into W_down (SIMULATED; `results/g1–g3`)
- Best write: layer 18, output-embedding target, whitened keys.
  - newest-100 accuracy: 0.40 / 0.04 / 0.03 at n = 10 / 100 / 10k;
  - first-100 retained: 0 from n = 300 on;
  - perplexity: 836 → 1302 at 10k writes;
  - leakage to other subjects: about 1.0.
- Sequential fine-tuning is worse: perplexity reaches 61k after 100 facts and 4.8 M after 1k.

## 5. Test 1: an entity-keyed DG episodic store next to frozen Qwen (SIMULATED; `r23/store.py`, `store2.py`; `results/g4`, `results/g7`)

**v1: contextual keys** (the residual at the person's and the subject's last token, layers 4–16, DG or dense).

It failed. Keys of the same subject in two phrasings were less alike than keys of different subjects in one phrasing:
- strict thresholds: paraphrase and QA ≈ 0.00–0.03, leakage 0 at 100 facts but 0.27 (DG) / 0.73 (dense) at 10k;
- loose thresholds (layer 4): paraphrase 0.90 and QA 0.89, but leakage 1.00.

**v2: token-level entity keys.**
- Every subject token (about 5 tokens) is DG-coded from the input embeddings: N 16,384, k = 32, so about 7 keys per fact.
- A fact is recognised when all its subject-token keys find a match.
- It fires where a layer-18 site key matches (threshold ts). Text gets a one-token prefix, so no content token sits at position 0.

Store v2, layer 0, ts = 0.1:

| n | rewrite | para | QA | first 100 | all | leak: other people | leak: same person, other relation | false hits, corpus | fires on an entity mentioned without a question | fires on non-answer positions of the QA prompt |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1.00 | 1.00 | 0.60 | 1.00 | – | 0 | 0 | 0 | 0 | 0.065 |
| 100 | 1.00 | 1.00 | 0.83 | 1.00 | – | 0 | 0 | 0 | 0 | 0.071 |
| 1,000 | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 | 0 | 0 | 0 | 0 | 0.072 |
| 3,000 | 1.00 | 1.00 | 0.85 | 1.00 | – | 0 | 0 | 0 | 0 | 0.072 |
| **10,000** | **1.00** | **0.99** | **0.76** | **1.00** | **0.998** | **0.00** | **0.003** | **0.000** | **0.00** | 0.064 |

- **The site gate trades QA against false fires inside the question.**
  - ts 0.2: QA 0.35, 0.048 false fires; ts 0.3: QA 0.10, 0.032.
  - With the gate off (ts 0): QA 1.00, but the store fires at every position after any mention of the entity (1.00) — useless.
- **Contextual entity layers fail on paraphrase.** Layer 2: paraphrase 0.37; layers 4–8: paraphrase 0.00.
- **Cost:** 1.6 ms per write; the network never changes, so perplexity and knowledge do not change.
- **Result: all four targets are met at 10k facts** (para ≥ 0.8, QA ≥ 0.6, leakage ≤ 0.05, false hits ≈ 0).
- **Caveats.**
  - Inside the QA prompt the store fires wrongly at about 6 % of the other positions, mostly the first `叫`. Corpus text and plain mentions produce no false hits.
  - The match is essentially lexical identity of the entity. It works because every phrasing repeats the entity string; aliases and pronouns would not match (ASSUMED, not tested).

## 6. Test 1b: the designer's hypothesis "dense vectors are the problem; feature neurons fix it" (SIMULATED; `r23/sparse.py`, `results/g5`)

**Setup.**
- A new layer W_new is added in parallel with layer 18's MLP output: residual += W_new·s(k).
- s(k) is the sparse code of the dense MLP key: a random projection to N units + k-WTA, a dense projection (no k-WTA), or a top-k sparse autoencoder trained here.
  - The autoencoder: 16,384 features, k = 64, trained on 190k tokens; it explains 0.80 of the held-out variance, and 99 % of its features are alive.
- Facts are written one-shot with the delta rule, towards the name's output-embedding direction.
- The sites are the name-predicting positions; the last row also writes at the subject site.

Key overlap at the predicting site (cosine; Jaccard in brackets), and the write results:

| code | active fraction | overlap: different facts, same relation | other relation | same fact, own paraphrase | newest at n = 10 / 100 / 1k / 10k | first 100 at n = 300 | para / QA at n = 10 | leakage at 100 / 10k | perplexity at 10k | knowledge at 10k |
|---|---|---|---|---|---|---|---|---|---|---|
| dense projection | 100 % | 0.94 | 0.65 | 0.87 | 0.20 / 0.01 / 0.01 / 0.02 | 0.01 | 0.10 / 0.10 | 1.00 / 1.00 | 1,462 | 3/9 |
| random projection 64k | 10 % | 0.84 (0.65) | 0.49 | 0.72 | 0.30 / 0.04 / 0.04 / 0.04 | 0.01 | 0.10 / 0.10 | 1.00 / 1.00 | 1,199 | 5/9 |
| random projection 64k | 2 % | 0.75 (0.56) | 0.33 | 0.57 | 0.80 / 0.11 / 0.16 / 0.13 | 0.01 | 0.20 / 0.10 | 0.99 / 0.84 | 1,002 | 4/9 |
| random projection 64k | 0.5 % | 0.68 (0.49) | 0.23 | 0.48 | 0.90 / 0.16 / 0.21 / 0.20 | 0.02 | 0.20 / 0.10 | 0.91 / 0.87 | 925 | 8/9 |
| random projection 64k | 0.1 % | 0.63 (0.45) | 0.17 | 0.40 | 1.00 / 0.21 / 0.26 / 0.28 | 0.03 | 0.10 / 0.10 | 0.94 / 0.87 | 907 | 8/9 |
| random projection 16k | 1 % | 0.71 (0.51) | 0.27 | 0.52 | 0.90 / 0.14 / 0.17 / 0.20 | 0.01 | 0.20 / 0.10 | 0.97 / 0.85 | 1,164 | 3/9 |
| sparse autoencoder 16k | 0.4 % | 0.94 (0.65) | 0.72 | 0.87 | 0.20 / 0.01 / 0.01 / 0.02 | 0.01 | 0.10 / 0.10 | 1.00 / 1.00 | 997 | 3/9 |
| random projection 0.5 % + subject site | 0.5 % | (as 0.5 %) | | | 0.90 / 0.14 / 0.21 / 0.22 | 0.01 | 0.20 / 0.10 | 0.95 / 0.87 | 1,008 | 6/9 |

R3's dense write directly into W_down scored newest 0.40 / 0.04 / 0.05 / 0.03, with perplexity 1,302 at 10k.

**How many facts fit before collapse** (largest n with newest-facts accuracy ≥ 0.5): dense under 10; 2 % and 0.5 % codes about 10; 0.1 % code about 30.

**Verdict: the hypothesis holds only in part.**
- **What sparse codes fix.** Writes interfere far less with *ordinary text*: perplexity after 10k writes is 907 vs 1,462, knowledge 8/9 vs 3/9. The first few one-shot facts are learned better (n = 10: 1.00 vs 0.20).
- **What they do not fix.**
  - Facts still overwrite each other: newest-100 ≤ 0.28 and first-100 ≈ 0 at every sparsity.
  - Paraphrase ≤ 0.2, QA ≤ 0.1, and leakage to other subjects 0.84–1.0.
- **The mechanism is overlap.** In the dense code, two *different* facts of the same relation (0.94) are more alike than one fact's own two phrasings (0.87). Random sparse coding shrinks every overlap, but it keeps that order at every sparsity (0.1 %: 0.63 vs 0.40), because it preserves similarity (LSH). The learned autoencoder keeps the dense similarities almost unchanged (0.94 / 0.87).
- **Entity-specific keying is also needed.** When the key is the entity's own identity (test 1 v2), the same 10k facts reach para 0.99, QA 0.76 and leakage 0.00. What the key is *about* matters more than whether it is sparse.
- Contextual codes of the entity (layers 2–8) fail as well, even when sparse.

## 7. Test 2: replay consolidation of the store into the weights (SIMULATED; `r23/consolidate.py`, `results/g4`)

**Protocol.**
- Facts arrive in 20 "days" of 100 (2,000 facts).
- After each day, replay trains all decoder layers with back-propagation: Adam, 150 steps per day.
- Each step's minibatch holds 8 new facts, 8 older facts recalled from the store (rehearsal), and 16 ordinary-text sequences (interleaving).
- The tests use the **weights alone**, with the store removed.

Results at n = 2,000:

| condition | all facts | first 100 | today's para / QA | perplexity (836 at start) | knowledge | neighbour subjects that now get an invented name |
|---|---|---|---|---|---|---|
| **replay, lr 1e-5: new + rehearsal + ordinary text** | **0.976** | **1.00** | 0.85 / 0.54 | **477** | **6/9** | 1.00 |
| same, lr 3e-5 | 0.943 | 0.98 | 0.79 / 0.67 | 709 | 1/9 | 1.00 |
| no ordinary text | 0.972 | 0.98 | 0.84 / 0.72 | 8,801 | 3/9 | 1.00 |
| no rehearsal | 0.082 | 0.00 | 0.38 / 0.14 | 450 | 0/9 | 1.00 |
| new facts only | 0.090 | 0.00 | 0.72 / 0.63 | 930,638 | 0/9 | 1.00 |
| plain sequential fine-tuning (R3, all layers) at n = 1,000 | 0.005 | 0.00 | 0.03 / 0.04 | 4.8 M | 0/9 | 1.00 |

The replay run over time (lr 1e-5):

| facts so far | all facts | knowledge | perplexity |
|---|---|---|---|
| 100 | 1.00 | 9/9 | 354 |
| 500 | 0.994 | 7/9 | 315 |
| 1,000 | 0.983 | 6/9 | 331 |
| 2,000 | 0.976 | 6/9 | 477 |

**What each part of replay does.**
- Without rehearsal, the previous day's facts drop to 0.11 within one day.
- Without ordinary text, general ability collapses (perplexity × 10).
- With both, 98 % of the facts survive in the weights alone.
- Perplexity falls because training on the chat corpus is partly domain adaptation.
- Cost: 0.45 s of training per fact (300 ms per step), vs 1.6 ms for a store write.

**Side effect.** After consolidation the network outputs an invented name for *any* unseen `X的Y叫` (leakage 1.00). The store has leakage 0. This matches the finding that fine-tuning on new facts raises hallucination (Gekhman et al. 2024). The store must therefore stay the authority on which facts exist, or the replay needs negative examples ("unknown").

## 8. Direct answer to the designer (updated)
- **Exact conversion.**
  - What the network computes: yes — keep the weights.
  - The last layer's learning rule: yes — the delta rule is back-propagation there.
  - Predictive coding with the fixed-prediction assumption: yes, measured cos 1.000. But that is back-propagation sent through local messages, and it forgets in exactly the same way.
- **Cheaper local rules** are about orthogonal to the gradient (cos 0–0.2) or unstable. They do not learn, yet they still damage old knowledge.
- **The fast-weight state is already Hebbian, exactly.** It is working memory of about 10²–10³ tokens at every size (0.1–1.5B), and it degrades without resets.
- **Hebbian writes into the slow weights interfere.** Sparse feature codes reduce the damage to ordinary text, but not the facts overwriting each other: different facts' keys stay more alike than one fact's own paraphrases.
- **The new hypothesis (dense vs feature neurons).** Sparsity is necessary but not sufficient. The decisive factor is entity-specific keying.
- **What works: a complementary-learning-systems design.**
  1. An inherited frozen network.
  2. An entity-keyed sparse episodic store. It learns in one shot with zero forgetting: 10k facts, para 0.99, QA 0.76, leakage 0.
  3. Interleaved replay, which consolidates 98 % of 2k facts into the weights while keeping perplexity and losing 3 of 9 knowledge probes.
  4. Plain sequential fine-tuning, by contrast, collapses within 100 facts.

## 9. Recommendation
- **Use** a frozen (inherited) network + RWKV-style fast weights for the current context (with resets) + the entity-keyed DG store as the long-term one-shot memory + slow interleaved replay into the weights.
- **Keep the store as the gate on what exists**: answer from the store and suppress names for unknown entities. Consolidation causes hallucination.
- **Next tests:**
  - aliases and pronouns as entity keys (a learned entity code, or a coreference step);
  - a sharper answer-site gate, e.g. a site key per relation;
  - "unknown" negatives during replay, to cut the 1.00 leakage after consolidation;
  - consolidating with fewer replays or LoRA, to lower the knowledge loss.

## 10. Files
All paths are relative to this directory:
- **Code:** `r1/rwkv7.py`, `r1/r1_stream.py`, `r23/r2.py`, `r23/r3.py`, `r23/store.py`, `r23/store2.py`, `r23/sparse.py`, `r23/consolidate.py`
- **Results:**
  - `results/g4/store.jsonl`, `results/g4/consolidate.jsonl`
  - `results/g5/sparse.jsonl`
  - `results/g6/store2.jsonl`, `results/g6/r1.jsonl` (1.5B)
  - `results/g7/store2b.jsonl`
  - earlier runs in `results/g1–g3`
- **Job scripts:** `jobs/llm2hebb_g4` … `jobs/llm2hebb_g7`

## summary_zh
1. 把反向传播大模型"等价"转成赫布式模型：前向计算和输出层可以精确等价；用局部消息实现的预测编码与反向传播完全一致（余弦 1.000），但遗忘也一样。更便宜、更局部的规则与梯度几乎无关，学不会新事实，却照样损伤旧知识。
2. RWKV-7 的状态本身就是精确的赫布快权重，但从 0.1B 到 1.5B 都只能记住几百到一千个词元；不重置运行就会漂移，模型变差（1.5B 在 26 万词元时交叉熵从 3.7 升到 10.6）。
3. 设计者的新假设"稠密向量是问题，特征神经元能解决"只对了一部分。稀疏特征编码明显减少了对普通文本的伤害（困惑度 1462→907，常识保留 8/9），但事实之间仍然互相覆盖，泄漏 0.84–1.0。原因是不同事实的键比同一事实两种说法的键更相似（稠密 0.94 对 0.87），稀疏化保留了这个顺序；训练出来的稀疏自编码器也不例外。
4. 真正起作用的是"按实体本身做键"：用实体词元的稀疏码（齿状回式）做索引的情节记忆，一万条事实一次学会、零遗忘，换说法 0.99、问答 0.76、泄漏 0、普通文本误触发 0。代价是只认同一实体字符串，别名还没测。
5. 每天把记忆交错回放进权重（新事实 + 旧事实复习 + 普通文本）后，只靠权重就保住了 2000 条中的 98%，困惑度没坏，常识丢了 3/9。没有复习就忘掉旧事实；没有普通文本模型就崩溃；逐条微调 100 条内就崩溃。副作用是巩固后模型会给没见过的人编名字，所以要靠记忆来判断"知道与否"。
6. 建议：冻结继承的网络 + 快权重工作记忆（定期重置）+ 按实体做键的稀疏海马式记忆 + 交错回放的慢速巩固。
