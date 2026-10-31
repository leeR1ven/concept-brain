# llm2hebb: can a back-propagation LLM become a Hebbian-style model that keeps learning? (IDEAS #18)

**A standalone experiment.** It has nothing to do with the brain architecture in `v2/`. Full report: [report.md](report.md).
Literature map: [lit_map.md](lit_map.md). The task as given: [TASK.md](TASK.md).

Models: Qwen2.5-0.5B-Instruct (softmax transformer) and RWKV-7 World 0.1B / 0.4B / 1.5B (delta-rule recurrent state),
public weights. Facts are invented names ("我的小狗叫豆米。"). All runs were on one A800 GPU, one seed per setting.

## Answer
1. **The forward computation converts exactly:** keep the weights. **The learning converts exactly only where it is
   already back-propagation:** the output layer's delta rule, and predictive coding with the fixed-prediction assumption
   (cosine with the true gradient 1.000 in every layer). Cheaper truly local rules are far from the gradient: layer-local
   delta 0.20, random feedback 0.00. Standard predictive coding diverged.
2. **Local learning does not remove forgetting.** The gradient-exact local rule forgets exactly like back-propagation.
   The truly local rules learn almost nothing, yet still damage old knowledge.
3. **RWKV-7's state already is a Hebbian (delta-rule) memory**, at every size from 0.1B to 1.5B. It learns a fact in
   one shot but keeps it for only about 10²–10³ tokens. Never reset, it drifts: the 1.5B model's cross-entropy goes
   from 3.7 to 10.6 at 256k tokens.
4. **One-shot rank-one Hebbian writes into an MLP layer** overwrite each other after about 10–30 facts. Back-propagation
   fine-tuning one fact at a time breaks the model within 100 facts.
5. **The designer's hypothesis "dense vectors are the problem; feature neurons fix it" holds only in part (test 1b).**
   - A sparse feature code (random expansion + k-WTA) cuts the damage to ordinary text: perplexity after 10k writes
     907 vs 1,462 dense, knowledge 8/9 vs 3/9.
   - It does not stop the facts overwriting each other (newest-100 ≤ 0.28, leakage 0.84–1.0). The reason: two
     *different* facts of the same kind have more alike keys than one fact's own two phrasings (dense 0.94 vs 0.87).
     Sparse coding keeps that order. A trained sparse autoencoder does too.
6. **What the key stands for decides it (test 1).** An episodic store keyed by sparse (DG-style) codes of the entity's
   own tokens, beside a frozen Qwen, learned 10,000 facts in one shot. Paraphrase 0.99, QA 0.76, leakage to other
   people 0.00, false hits on ordinary text 0, and the network is untouched. Limits: it matches only the same entity
   string (aliases and pronouns are untested), and it misfires at about 6 % of the other positions inside a QA prompt.
7. **Replay consolidation (test 2).** Each "day" the store's facts were replayed into the weights, mixed with older
   facts and ordinary text. With the store removed, the weights alone kept 98 % of 2,000 facts; perplexity stayed fine,
   and 3 of the 9 knowledge probes were lost. Without the older facts, earlier days were forgotten. Without ordinary
   text, the model collapsed.
   - Caveat: this replay trains by back-propagation (Adam), not by a local rule.
   - Side effect: afterwards the model invents a name for *any* unknown person (leakage 1.00), so the store must stay
     the judge of what is known.

**Recommendation.** Converting the slow weights to local learning does not give continual learning. What worked is the
brain's complementary-learning layout:
- an inherited frozen network;
- fast-weight working memory, with resets;
- an entity-keyed sparse, hippocampus-like store for one-shot facts;
- slow interleaved replay into the weights.

Next tests: aliases and pronouns as keys, a sharper answer gate, "unknown" negatives during replay, and cheaper
consolidation.

## Files
- `r1/`: RWKV-7 in plain torch + Triton (`rwkv7.py`, checked against the official package), the never-reset stream
  test (`r1_stream.py`).
- `r23/`: Qwen experiments. `r2.py` covers gradient cosines and the online stream; `r3.py` covers rank-one writes,
  sequential fine-tuning and the first episodic store. `store.py` / `store2.py` hold the entity-keyed store
  (test 1), `sparse.py` the sparse-feature writes (test 1b), and `consolidate.py` the replay (test 2).
- `results/`: `g1`–`g7` are the GPU batches, as `jsonl` plus `run.log`; `env/` holds the setup checks.
  `r1_*_rec.json.gz` are the per-query R1 records.
- `jobs/`: the job scripts as they ran on the server.
- `tables.py results/` prints the tables.
