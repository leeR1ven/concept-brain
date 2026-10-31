(Saved by the coordinator from the agent's final reply; the harness blocked the agent's own write.)

# llm2hebb: can a back-propagation LLM be converted into a Hebbian-style model that keeps learning? (IDEAS #18, standalone)

The harness blocked writing `report.md` ("subagents should return findings as text"), so this message is the report. Base folder: this directory (`side/llm2hebb/` in the repo).

Labels: **PROVED** = argument or arithmetic, **SIMULATED** = my GPU run (file and numbers given), **CITED**, **ASSUMED**. I followed both scope changes: E1/E2/E3 were dropped, and the synth/bench copies were deleted. Only R1–R3 were run.

## 0. Setup and infrastructure
- **Servers and queues.** The V100 box ran only the setup checks. All experiments ran in the A800 GPU queue (`queue_a800gpu`), one job at a time.
- **Where the models came from.** hf-mirror.com was reachable for the API but stalled on large files (one 7 MB file took 27 min, then the host went unreachable). huggingface.co was blocked. So the public weights came from **ModelScope's copies of the same public repositories** (`Qwen/Qwen2.5-0.5B-Instruct`, `Blink_DL/rwkv-7-world`). Both servers now hold them in `~/v2lab/models/` (about 2.3 GB each).
- **Packages.** The server has no `python3-venv`, so packages went to `~/v2lab/venv_llm/site` with `pip --target` through the server's aliyun mirror: transformers 4.46.3, tokenizers, safetensors, rwkv 0.8.32. They are used via `PYTHONPATH`.
- **Models:**
  - Qwen2.5-0.5B-Instruct (softmax transformer, 24 blocks) for R2 and R3.
  - RWKV-7 World 0.4B and 0.1B (delta-rule recurrent state, multilingual) for R1.
- **My RWKV-7 implementation.** I wrote a batched plain-torch + Triton version of RWKV-7 (`r1/rwkv7.py`). On the V100 it matches the official `rwkv` package to max |Δlogit| 4e-5. On the A800, with TF32 projections, it matches to 0.01 with identical argmax (SIMULATED, `results/env/`).
- **Facts.** Every fact is an invented two-token name (`我的小狗叫豆米。`, `王子涵的小狗叫栗蔻`). Recall means both tokens are exactly right.

## 1. Literature map (CITED; about one page)

| family | local rule | relation to the BP gradient | conditions | meaning for a pretrained LLM |
|---|---|---|---|---|
| delta rule | dW = η(y−p)xᵀ | **exact** for one output layer | none | any read-out can learn locally, exactly like BP |
| predictive coding / inference learning (Whittington & Bogacz 2017; Song et al. 2020; Millidge, Tschantz & Buckley 2022; Millidge et al. 2023; Song et al. 2024) | per-layer prediction errors; activities relax; dW_l = J_Wᵀ ε_l | approaches BP at equilibrium with a small nudge; **exactly BP** with the fixed-prediction assumption (FPA) and step 1 | needs the next layer's transposed Jacobian (weight transport) and ≥ L relaxation steps; the inference landscape becomes ill-conditioned with depth and with training (Innocenti, Achour & Buckley, NeurIPS 2025, μPC) | the weights carry over exactly. But the update follows the gradient, so it inherits BP's interference. "Prospective configuration" is reported to reduce interference, in small networks only |
| equilibrium propagation (Scellier & Bengio 2017; Laborieux et al. 2021; Laborieux & Zenke 2022) | contrastive Hebbian over a free and a nudged phase | gradient as β → 0 | needs an energy function and symmetric weights | a transformer is not an energy-based model; it would have to be re-trained as one (Hoover et al. 2023 Energy Transformer; attention as one Hopfield step, Ramsauer et al. 2021) |
| feedback alignment / DFA (Lillicrap et al. 2016; Nøkland 2016; Launay et al. 2020) | error sent back through fixed random matrices | not the gradient; forward weights align with the feedback *during training* | works from scratch | a pretrained net is not aligned with any random feedback, so its updates start about orthogonal to the gradient |
| target propagation (Lee et al. 2015; Meulemans et al. 2020; Ernoult et al. 2022) | each layer regresses to targets made by learned inverses | Gauss-Newton-like | needs trained inverses; scales poorly (Bartunov et al. 2018) | the inverses would have to be trained first |
| fast weights (Ba et al. 2016; Schlag, Irie & Schmidhuber 2021; DeltaNet / Gated DeltaNet, Yang et al. 2024, 2025; RWKV-7, Peng et al. 2025; TTT, Sun et al. 2024; Titans, Behrouz et al. 2025) | S += v kᵀ, or the delta rule S += β(v − Sk)kᵀ, at every token at inference | the inference itself *is* a Hebbian/delta-rule memory | fixed-size state that decays; recall lags softmax attention (Arora et al. 2024) | the part of an LLM that already learns locally and continually. Softmax models can be linearised by a short fine-tune (T2R, Kasai et al. 2021; SUPRA, Mercat et al. 2024; LoLCATs, Zhang et al. 2024) |
| model editing (Geva et al. 2021; ROME/MEMIT, Meng et al. 2022, 2023) | rank-one key→value write of delta-rule form | the write is local; ROME finds its target v* by BP | sequential edits degrade, then collapse: ROME after about 100–1,000 edits, MEMIT after about 1,400 (Gupta, Rao & Anumanchipalli 2024) | one-shot slow-weight writes are possible but limited in capacity |
| episodic memory / complementary learning systems (McCloskey & Cohen 1989; McClelland et al. 1995; Kumaran et al. 2016; kNN-LM, Khandelwal et al. 2020; Memorizing Transformers, Wu et al. 2022; Larimar, Das et al. 2024) | non-parametric key→value store | — | — | one-shot learning without interference. Fine-tuning learns new facts slowly and raises hallucination (Gekhman et al. 2024); retrieval beats fine-tuning for injecting knowledge (Ovadia et al. 2024) |

## 2. R1: the attention/recurrent state as Hebbian memory (RWKV-7, one unbroken stream)

**PROVED (from code, checked numerically).** RWKV-7's state update is S ← S·diag(w) − (S k̂)(a⊙k̂)ᵀ + v kᵀ. That is a local, error-correcting delta rule, applied at inference with no back-propagation. Here the Hebbian equivalence is exact.

The state has a fixed size:
- 0.4B model: 24 × 16 × 64² = 1.57 M numbers (6.3 MB);
- by comparison, Qwen-0.5B's KV cache grows by 12 KB per token (bf16), i.e. 12 GB per 1 M tokens.

**Protocol (SIMULATED, `r1/r1_stream.py`, results in `results/g2/r1*.json*`):**
- 8 streams × 262,144 tokens of corpus chat, with 40 facts per stream.
- At about 270 checkpoints the state is forked, every fact told so far is asked in two forms, and the fork is discarded (cloze `我的小狗叫`; QA `问：…叫什么名字？答：你的小狗叫`). 92,080 queries per condition.
- Control: the same streams with the state reset every 4,096 tokens (the training context length).

Recall vs delay (fraction correct; cloze / QA):

| model, condition | <64 tokens | 64–256 | 256–1k | 1k–4k | >4k |
|---|---|---|---|---|---|
| 0.4B, reset every 4k (in distribution) | 0.92 / 0.61 | 0.64 / 0.22 | 0.40 / 0.11 | 0.08 / 0.02 | 0.00 / 0.00 |
| 0.4B, never reset (all facts) | 0.29 / 0.14 | 0.07 / 0.00 | 0.005 / 0 | 0 / 0 | 0 / 0 |
| 0.4B, never reset, facts after 32k tokens | 0.25 / 0.11 | 0.04 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 0.4B, never reset, fact in the first 8k (n = 8) | 1.00 / 0.75 | 0.75 / 0 | 0 / 0 | 0 / 0 | – |
| 0.1B, reset | 0.97 / 0.72 | 0.81 / 0.17 | 0.34 / 0.02 | 0.02 / 0 | 0 / 0 |
| 0.1B, never reset | 0.34 / 0.14 | 0.10 / 0.02 | 0.01 / 0 | 0 / 0 | 0 / 0 |

- Recall across a reset boundary is 0.000 (n = 44,072), as expected.
- With delay under 4k tokens, recall falls with the number of later facts: 0.44 with none, 0.18 with 1–2 (cloze, 0.4B, reset). This is confounded with delay.

The never-reset state degrades. Cross-entropy on held-out text, read from the forked state (0.4B, mean of 2 streams):

| state after | fresh | 4k | 16k | 64k | 256k | 1M |
|---|---|---|---|---|---|---|
| CE | 3.93 | 3.71 | 3.75 | 4.07 | 5.85 | 9.86 |
| state norm | — | 34 | 33 | 34 | 42 | 229 |

- The 0.1B model shows the same pattern: 4.08 → 3.94 → 3.99 → 4.49 → 5.28 → 6.71.
- CE inside the stream (0.4B) at 128k–256k tokens: 5.07 never-reset vs 3.72 with resets.
- Throughput: 14.5k tokens/s (0.4B) and 35k tokens/s (0.1B), 8 streams.

**Reading.**
- The pretrained Hebbian fast-weight state is a one-shot **working memory**. Its horizon is about 10²–10³ tokens of unrelated text.
- It is not a long-term store, and it cannot run as one unbroken life. Beyond about 16–32k tokens (4–8× its training length) the state drifts out of distribution, general ability decays, and so does recall.

## 3. R2: local rules on the slow weights (Qwen2.5-0.5B)

**Cosine between each local update and the true gradient**, per layer, mean over 16 lines (SIMULATED, `results/g1/r2_cos.jsonl`, `results/g2/r2_cos2.jsonl`):

| rule | cos (all layers) | layers 0/6/12/18/23 | update size / gradient size | s per update |
|---|---|---|---|---|
| PC with FPA (T = 24, step 1) | **1.000** | 0.999/1/1/1/1 | 1.00 | 1.9 |
| standard PC, γ 0.5/0.2/0.1/0.03, β 0.01–1, T 6–192 | −0.01 … 0.04 | ≈ 0 | 10² … 10¹⁵ (relaxation diverges) | 0.65–12.9 |
| layer-local delta rule on W_down (logit-lens error, no inter-layer messages) | 0.20 | 0.03/0.04/0.08/0.32/**1.00** | — | 0.10 |
| DFA (random feedback) on W_down | −0.005 | ≈ 0 everywhere | — | 0.07 |

- PC-FPA *is* BP, computed by local messages, so its learning trajectory is identical to BP-SGD (PROVED, and checked to cos 1.000).
- Standard PC diverged at every step size I could afford (γ ≥ 0.03, up to 192 steps ≈ 100× the cost of BP). This matches μPC's report that the PC inference landscape is ill-conditioned for deep, trained networks.

**Online stream:** 30 facts × 3 presentations, each followed by a chat line, then 60 more chat lines; one update per line (SIMULATED, `results/g2/`, `results/g3/r2_online.jsonl`). Columns:
- cloze after 1 / 3 presentations / after the delay;
- corpus perplexity p0 → p3 → after the delay;
- world-knowledge cloze (9 items the base model gets right; p0 = 1.0) at p3 and after the delay.

| method, lr | cloze 1 / 3 / delay | perplexity 836 → p3 → delay | knowledge p3 / delay | ms per update |
|---|---|---|---|---|
| BP-Adam 1e-5 | 0.07 / **0.47** / 0.13 | 1117 → 703 | 0.33 / 0.33 | 93 |
| BP-Adam 3e-6 | 0 / 0.20 / 0.07 | 595 → 507 | 0.78 / 0.78 | 100 |
| BP-Adam 3e-5 | 0.03 / 0.13 / 0 | 4425 → 2245 | 0 / 0 | 97 |
| BP-SGD 1e-4 / 1e-3 | 0.03 / 0.07 at p3 | 851 / 1161 | 0.33 / 0 | 74–86 |
| BP-SGD ≥ 0.01 (and PC-FPA, which is identical) | 0 | 4.7k–159k (destroyed) | 0 | 74–86 |
| BP on W_down only, 1e-3 / 1e-2 | 0 / 0.03 | 1173 / 5494 | 0 | 50–70 |
| layer-local delta 1e-3 (1e-2 destroys) | 0 / 0.03 / 0 | 2141 → 3847 | 0.22 | 64–76 |
| DFA 1e-5 (1e-4 destroys) | 0 / 0 / 0 | 1027 → 1794 | 0.44 / 0.22 | 41 |

**Reading.**
- No local rule learned the facts without damaging old knowledge.
- The gradient-exact local rule (PC-FPA) has exactly BP's learning and forgetting.
- The truly local rules (layer-local delta, DFA) learn about nothing, yet still erase world knowledge.
- Only BP-Adam learned facts: 47 % after 3 presentations. It lost most of them within 60 chat lines (13 %) and lost two-thirds of the knowledge probes.
- So being local does not remove catastrophic forgetting. The cause is dense, shared weights trained on a non-stationary stream.
- Caveats: one seed per setting, and the lr grid is coarse.

## 4. R3: one-shot Hebbian rank-one writes into an MLP layer (no BP at write time)

The rule: W_down ← W_down + (v* − W k) k̃ᵀ / (kᵀ k̃), with k̃ = k (pure delta), C⁻¹k (ROME-style whitening), or the top-64 entries of k (sparse key).
- **Key:** the MLP input at the predicting position ("last") or at the subject's last token ("subj").
- **Target v\*:** either the model's own in-context representation (layer-l residual with the fact in context minus without it), or the name's output embedding scaled to s·|h|.

**Single writes** (40 dev facts, each written alone into the original model; `results/g1/r3_dev.jsonl`):
- In-context targets do almost nothing. Rewrite efficacy is 0 at layers 3–15 and 0.40 at layer 18 with gain 2.
- Subject-site keys give 0 at every layer: without BP to find v*, the subject token does not carry the name.
- Output-embedding target, s = 2, layers 15/18: efficacy 1.00, paraphrase 0.90–1.00, QA 0.58–1.00. But specificity collapses: other subjects' prompts keep their prediction in only 0.00–0.30 of cases, and 0.63–1.00 of them now produce a written name.
- s = 0.5 at layer 18: efficacy 0.68–0.75, specificity 0.77–0.85, leakage 0.10–0.14.
- Whitening (C⁻¹) always helps specificity.

**Sequential writes** (`results/g2`, `results/g3`). Columns: accuracy on the newest 100 facts after n writes; the first 100 facts; perplexity; knowledge (9 items); cost per fact.

| run | n = 10 | 30 | 100 | 1k | 10k | first 100 at 1k | perplexity (836) | knowledge at the end | ms per fact |
|---|---|---|---|---|---|---|---|---|---|
| Hebbian, layer 18, s 1, whitened | 0.40 | 0.10 | 0.04 | 0.05 | 0.03 | 0.00 | 956 at 1k, 1302 at 10k | 1/9 | 0.6 + 1.5 forward |
| Hebbian, s 1, pure delta | 0.10 | 0.07 | 0.01 | 0.01 | 0.02 | 0.00 | 1988 / 2527 | 3/9 | 0.3 |
| Hebbian, s 1, sparse key (top 64) | 0.10 | 0.07 | 0.01 | 0.02 | 0.02 | 0.00 | 3934 / **232,691** (collapse) | 0 | 0.5 |
| Hebbian, s 0.5, whitened (to 3k) | 0.20 | 0.07 | 0.04 | 0.01 | – | 0.00 | 859 / 880 at 3k | 7/9 | 0.45 |
| Hebbian, in-context target, gain 2, whitened (to 3k) | 0.10 | 0.03 | 0.01 | 0.01 | – | 0.00 | 882 / 904 | 5/9 | 0.5 |
| sequential FT, layer 18 only (Adam 5e-4, ≤ 10 steps) | 0.30 | 0.13 | 0.05 | 0.03 | – | 0.00 | 9,199 at 30, **44,690** at 1k | 0 | 190–350 |
| sequential FT, all layers (Adam 1e-5) | 0.40 | 0.10 | 0.02 | 0.04 | – | 0.00 | 61k at 100, **4.8 M** at 1k | 0 | 250–420 |
| **episodic key→value store** (same keys, network frozen, nearest neighbour, θ 0.98) | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | **1.00** | unchanged (frozen); false hits on ordinary text 0.0 % | 9/9 | 1.5 |

The episodic store has a weakness: paraphrase is 0.6 at n = 10 (θ 0.8) but only 0.06 at 10k, and QA is ≤ 0.1. Leakage to other subjects grows with n (θ 0.98: 0.07 at n = 10, 0.92 at 10k).

**Where it collapses.**
- One layer's Hebbian writes stop holding new facts after about 30 writes. The earliest facts are overwritten within 100–300 writes.
- General ability decays slowly with whitening (+14 % perplexity at 1k writes, +56 % at 10k), and catastrophically with sparse keys.
- Back-prop fine-tuning destroys general ability within 30–100 facts.

**Why (PROVED).** The write makes W′k = v* hold exactly, and shifts the output for any other key k′ by (v* − Wk)(k̃·k′)/(k̃·k). All the "X的Y叫" keys are nearly parallel, so writes overwrite each other almost at once. And a 4,864-dimensional key space cannot hold more than 4,864 independent exact associations.

## 5. Direct answer to the designer

**Can a back-prop LLM be converted, equivalently, into a Hebbian-style model?**
- **What it computes: yes, exactly.** Keep the weights. A predictive-coding network with zero errors has the same forward pass (PROVED).
- **How it learns: exact only in two places.**
  - The output layer: the delta rule *is* back-propagation there.
  - Predictive coding with the fixed-prediction assumption reproduces back-propagation exactly (cos 1.000 per layer, SIMULATED). But that is back-propagation sent through local messages: it needs the transposed Jacobians and L sequential steps.
- **Everything cheaper or more local is far from the gradient** on a pretrained 24-block model:
  - the layer-local delta rule: cos 0.20, about 0.05 below the middle layers;
  - DFA: cos 0.00;
  - standard predictive coding: its relaxation diverged.
- **"Recomputing" the weights as if Hebbian learning had grown them is impossible.** Different rules grow different weights, and there is no map back. It is also unnecessary: the weights can simply be inherited.
- **One part is already exactly Hebbian:** the state of delta-rule LLMs (RWKV-7, DeltaNet).

**After conversion, does it keep learning?** No, not from the conversion itself.
- A local rule that matches the gradient forgets exactly as back-propagation does. A local rule that does not match it learns nothing and still damages old knowledge (R2).
- The Hebbian fast-weight state learns in one shot, but it forgets after hundreds of tokens, and it degrades after about 16–32k tokens without a reset (R1).
- One-shot Hebbian writes into the slow weights work once or a few times, then overwrite each other because the keys overlap (R3).
- The limits are interference from shared codes, finite capacity and decay — not the non-locality of back-propagation.
- The one thing that learned 10,000 facts in one shot with **zero forgetting and zero damage** was a separate episodic key→value store on a frozen network. It failed on paraphrase and specificity because its keys were not specific to the subject.

## 6. Recommendation
- **Do not convert the slow weights to local learning** in order to get continual learning: it is either back-propagation in disguise or it does not learn.
- **Most promising combination: inherit, then add the brain's complementary systems.**
  - A frozen (inherited) pretrained network.
  - **R1** fast weights as working memory, with periodic reset or renormalisation (or a model trained for unbounded streams).
  - An **episodic store** for one-shot long-term facts, with **pattern-separated keys**: subject- or entity-specific, a sparse random expansion + k-WTA, several keys per fact.
  - Slow consolidation into the weights by **interleaved replay**, or by small whitened rank-one batches spread across layers with capacity management.
- So: R1 + episodic store + slow consolidation. R3 alone or R2 alone is not viable.
- **Next tests:**
  1. An entity-keyed, DG-style sparse store on the same 10k facts. Target: paraphrase ≥ 0.8 and leakage ≤ 0.05.
  2. Replay consolidation of that store into the weights, interleaved with old text, measuring forgetting.
  3. RWKV-7 1.5B (it is on ModelScope) or a state-normalised variant, to see whether the fast-weight horizon grows.

## 7. Files
All paths are relative to this directory:
- **R1 code:** `r1/rwkv7.py`, `r1/r1_stream.py`, `r1/check_rwkv.py`
- **R2/R3 code:** `r23/common.py`, `r23/facts.py`, `r23/r2.py`, `r23/r3.py` (modes `dev`, `seq`, `ft`, `knn`), `r23/pick.py`
- **Results:** `results/g1/`, `results/g2/`, `results/g3/`, `results/env/` (`jsonl` files + `run.log`)
- **Tables:** `tables.py` prints the markdown tables from the results.
- **Literature map:** `lit_map.md`
- **Job scripts:** `jobs/llm2hebb_*/job.sh`

## summary_zh
1. 能"等价转换"的只有两处。一是前向计算：权重直接继承，结果完全相同。二是学习规则里的输出层：增量规则就是反向传播。用局部消息实现的预测编码（固定预测假设）可以精确等于反向传播（实测与真梯度余弦 1.000），但它本质上就是反向传播，同样会遗忘。真正局部、又便宜的规则在预训练的深层模型上与梯度几乎无关：逐层局部增量规则余弦约 0.2，随机反馈约 0；标准预测编码的推断过程直接发散。
2. 局部本身并不消除灾难性遗忘。在线学习新事实时，只有反向传播 Adam 学会了（三次后 47%），但 60 句闲聊后只剩 13%，常识也丢了三分之二。各种局部规则基本学不会，却同样损伤旧知识。
3. RWKV-7 的状态本身就是赫布（增量规则）快权重：新事实一次就记住，但只能保持几百个词元；不重置运行超过约 1.6–3.2 万词元后，状态漂移，模型变差（到 100 万词元时交叉熵从 3.9 升到 9.9）。
4. 往 MLP 里做一次性赫布秩一写入：单条可以写进去，但几十条后就互相覆盖，因为键高度相似。反向传播逐条微调会在 30–100 条内把模型搞坏。
5. 唯一在一万条事实上做到一次学会、零遗忘、不伤旧知识的，是一个冻结网络旁边的独立情节记忆（键→值存储）。不过它的键不够区分主语，换个说法问就答不上。
6. 建议：不要指望"把权重改成赫布学习"来实现持续学习。应当继承预训练网络，配上快权重工作记忆、模式分离的海马式情节记忆，以及交错回放的慢速巩固。下一步测试：按实体做键的稀疏记忆，以及回放巩固。
