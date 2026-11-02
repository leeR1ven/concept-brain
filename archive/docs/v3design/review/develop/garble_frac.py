"""share of random 4-muscle babbles that the real mouth rule decodes as garble '~' (full mouth / lexicon mouth)."""
import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'v2', 'bench'))
from physics import Voice, corpus_chars
from world import LEXICON_CHARS
v = Voice(corpus_chars(), seed=0, physics='lawful'); rng = np.random.default_rng(3)
for vocab in ('full', 'lexicon'):
    if vocab == 'lexicon':
        v.set_mouth_vocab(LEXICON_CHARS)
    n = 5000; g = 0; lexhit = 0
    for _ in range(n):
        c = v.decode(tuple(sorted(int(m) for m in rng.choice(32, 4, replace=False))))
        g += c == '~'; lexhit += c in LEXICON_CHARS
    print(f'{vocab:8s} random babble: garble {g/n:.3f}, decodes to a lexicon char {lexhit/n:.3f}')
