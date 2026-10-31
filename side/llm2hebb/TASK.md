# Task (REVISED, standalone): convert a back-propagation-trained LLM into a Hebbian-style model that keeps learning

## The designer's idea (2026-10-31). It is independent of our brain architecture.
"我的意思是你单独测试把用反向传播训练出来的语言模型转换成赫布连接式的模型，但是和我们训练的当前模型没有任何关系。这是一个独立的想法……我想的是大语言模型等价转换成赫布学习式的模型后，就能持续学习了。"

In English:
- Take a language model trained by back-propagation.
- Convert it, as equivalently as possible, into a Hebbian-style model: one whose learning uses local, Hebbian / delta-rule
  updates instead of back-propagation.
- Test whether the converted model can then keep learning continually.

**This has nothing to do with our ARCH_v2 brain model.** Do not use or modify anything in `arch/v2/synth`, `bench` or
the other agents' folders. Ignore the earlier version of this task, including E1, E3 and the "meaning layer" idea.

## Three conversion routes to test. Report each honestly, including where "equivalent" is exact, approximate or false.

**R1. Fast weights: the attention or recurrent state as Hebbian memory.**
- Linear attention is a Hebbian fast-weight memory: the state S accumulates key·valueᵀ outer products (CITED Schlag,
  Irie & Schmidhuber 2021).
- Some modern pretrained LLMs update their recurrent state with the **delta rule**, which is a local error-correcting
  Hebbian rule: DeltaNet / Gated DeltaNet, and RWKV-7.
- A pretrained softmax transformer can also be converted to linear attention by short linearisation (T2R, Kasai et al.
  2021; LoLCATs, Zhang et al. 2024).
- **Test.** Take a small pretrained delta-rule / RWKV-7 model, ideally multilingual with Chinese, 0.1–0.5B. Run it as
  one unbroken stream that is **never reset**, so its Hebbian state carries across "sessions".
  - Feed it new facts (invented names) interleaved with long stretches of unrelated text.
  - Measure recall of each fact against delay and the amount of intervening text.
  - Measure whether its general ability (perplexity on held-out text) degrades over a very long unreset stream.
  - Compare with the same model reset every session. That is today's LLM usage: no continual learning.

**R2. Local learning rules on the slow weights.**
- Keep the pretrained architecture and weights, and replace back-propagation with a local rule. Two options:
  - predictive coding / inference learning, which approximates the back-propagation gradient with only local
    computations at equilibrium (CITED Whittington & Bogacz 2017; Millidge, Tschantz & Buckley 2022);
  - a purely local delta rule per layer, using layer-local targets.
- **Test** on a small model, a ≤ 0.5B transformer or a smaller GPT-2-class model if it is too slow:
  - the cosine between the local update and the true gradient, per layer;
  - online learning of new facts from a stream, one presentation at a time;
  - forgetting of old knowledge.
- Compare with ordinary back-propagation fine-tuning on the same stream.
- Key question: does "local" by itself remove catastrophic forgetting, or does a local rule that approximates the
  gradient forget just as much?

**R3. One-shot Hebbian writes of facts into the MLP layers.**
- An MLP layer acts as a key → value memory. ROME / MEMIT write a fact with a rank-one, delta-rule update
  W += (v − Wk) kᵀ / (kᵀk) (CITED Meng et al. 2022, 2023). That is local in the sense of pre (k) × error (v − Wk).
- **Test** a Hebbian-only variant, with no back-propagation through the network at write time: k is the layer input for
  the subject, and v comes from the model's own representation of the target. Measure:
  - **sequential continual learning:** write facts one after another (100 → 1,000 → 10,000) and track the accuracy of
    new facts, of earlier facts, and of general ability (perplexity);
  - where it collapses.
- Compare with sequential fine-tuning.

## Output
Write `report.md` with:
- **A direct answer to the designer.** Can a back-propagation LLM be equivalently converted into a Hebbian-style model?
  In what sense is that exact, approximate or impossible? After conversion, does it keep learning, and what limits it
  (capacity, interference, forgetting)?
- Tables for R1–R3.
- **A recommendation.** Which route, or which combination (for example R1 + R3, or a sparse/episodic store added to
  avoid interference), is the most promising for an LLM that learns continually, and what to try next.
- A short literature map, about one page.

## Rules
- Work only in `arch/v2/llm2hebb/`. No git. Nothing in the other agents' folders.
- **Server jobs.** Use the coordinator's runner only; read `arch/v2/gpu/README.md` and follow it exactly. You never
  SSH.
  - Prefer the V100 queue `queue/`: its GPU is idle. You may use `queue_a800/` when a slot is free.
  - At most 1 of your jobs per queue at a time. Name jobs `llm2hebb_<label>`.
- **Allowed for you only:**
  - jobs may download public pretrained models, tokenizers and small public text datasets from Hugging Face or a mirror
    (for example hf-mirror.com) into the shared cache `~/v2lab/models/` (download once, reuse);
  - jobs may `pip install` needed packages (for example `transformers`, `safetensors`, `tokenizers`, `datasets`) into a
    virtualenv at `~/v2lab/venv_llm/`.
  - No other network use.
  - If a server has no internet access, say so and adapt: small models, or uploads under 100 MB.
- **Local box:** 4 cores and 3 GB RAM, shared with other agents. Only tiny checks, one process under 500 MB.
- Labels: PROVED / SIMULATED / CITED / ASSUMED.

**Return the full report inline in your final reply**, because the harness may block the write. Write in English and
end with a short plain Chinese summary (summary_zh).
