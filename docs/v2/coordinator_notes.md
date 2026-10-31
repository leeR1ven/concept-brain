# Coordinator notes for the ARCH_v2 synthesis
- (2026-10-31) Designer asked about local connectivity (neurons connect only to nearby neurons of the next layer).
  Their net3d already is local, but its feed-forward weights are fixed (neighbour weights <= 0.01), so the ear copies
  its input lines. Synthesis item: let the LOCAL feed-forward weights of the sensory 3D nets learn with a competitive
  Hebbian rule (Fukushima 1980 neocognitron; Linsker 1986; Krotov & Hopfield 2019), so the upper layers become
  combination detectors (syllable parts → syllables → chunks) feeding LEX/SYL. Check: unit selectivity, layer-0
  vs layer-3 decodability of syllables/words, no loss of the body's F detector (F reads layer 0 / input energy).
- Designer's ideas to keep: PFC = recurrent next-input predictor (loop through the PFC, not input only);
  hold by loop (with gate, self-hold, content); sensory/motor = I/O of the PFC core but must learn features.
