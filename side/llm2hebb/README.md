# llm2hebb: can a back-propagation LLM become a Hebbian-style model that keeps learning? (IDEAS #18)

**A standalone experiment.** It has nothing to do with the brain architecture in `v2/`. Full report: [report.md](report.md).
Literature map: [lit_map.md](lit_map.md). The task as given: [TASK.md](TASK.md).

Models: Qwen2.5-0.5B-Instruct (softmax transformer) and RWKV-7 World 0.1B / 0.4B (delta-rule recurrent state), public
weights. Facts are invented names ("我的小狗叫豆米。"). All runs were on one A800 GPU, one seed per setting.

## Answer in four lines
1. **The forward computation converts exactly:** keep the weights. **The learning converts exactly only where it is
   already back-propagation:** the output layer's delta rule, and predictive coding with the fixed-prediction assumption
   (cosine with the true gradient 1.000 in every layer). Cheaper truly local rules are far from the gradient: layer-local
   delta 0.20, random feedback 0.00. Standard predictive coding diverged.
2. **Local learning does not remove forgetting.** The gradient-exact local rule forgets exactly like back-propagation.
   The truly local rules learn almost nothing, yet still damage old knowledge. Only back-propagation (Adam) learned the
   facts online (47 % after 3 presentations), and it kept only 13 % after 60 chat lines.
3. **RWKV-7's state already is a Hebbian (delta-rule) memory.** It learns a fact in one shot but keeps it only for
   about 10²–10³ tokens. Never reset, the state drifts after about 16–32k tokens: cross-entropy 3.9 → 9.9 at 1 M tokens.
4. **One-shot rank-one Hebbian writes into an MLP layer** hold for a few facts, then overwrite each other after about
   30, because the keys of similar sentences are nearly parallel. Back-propagation fine-tuning one fact at a time breaks
   the model within 30–100 facts. **A separate episodic key → value store** on a frozen network learned 10,000 facts in
   one shot with no forgetting and no damage. It failed on paraphrases, though, because its keys were not specific to
   the subject.

**Recommendation:** do not convert the slow weights to local learning to get continual learning. Inherit the pretrained
network, and add fast-weight working memory, a pattern-separated episodic store and slow replay consolidation. Next
tests: an entity-keyed sparse store, then replay consolidation into the weights.

## Files
- `r1/`: RWKV-7 in plain torch + Triton (`rwkv7.py`, checked against the official package), the never-reset stream
  test (`r1_stream.py`).
- `r23/`: Qwen experiments. `r2.py` covers gradient cosines and the online stream; `r3.py` covers rank-one writes,
  sequential fine-tuning and the episodic store.
- `results/`: `g1`–`g3` are the three GPU batches, as `jsonl` plus `run.log`; `env/` holds the setup checks.
  `r1_*_rec.json.gz` are the per-query R1 records.
- `jobs/`: the job scripts as they ran on the server.
- `tables.py results/` prints the tables.
