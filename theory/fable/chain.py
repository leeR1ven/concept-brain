"""Can a thought feed the next thought (docs/THEORY_fable.md section 4.2)? Forward computation only, newborn.
A syllable B's copies (recorded by codes.py) are lit as a thought for 3 ticks in silence (as the readout would do
after hearing A), with the real return to the sense areas on. Measured over the following ticks:
  echo    copies lit from outside by the return (the inner voice coming back through the ear's areas);
  L4      last-layer units lit, and their overlap with B's code when B was really heard (ticks 4..6);
for a thought of m copies (the budget), of 180 copies (one tick of real hearing) and of all B's copies.
usage (from a directory holding core/ and data/):  python3 chain.py codes.npz [n=20] [key=value]"""
import json
import sys

import numpy as np
import torch

from core.config import config
from core.corpus_life import CorpusLife
from core.net import DEV

PRE = 12


def unpack(a, n):
    return np.unpackbits(a, axis=-1)[..., :n].astype(bool)


def main():
    z = np.load(sys.argv[1])
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[3:] if '=' in a)}
    n_syl = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    b, p = life.brain, life.brain.pfc
    n1 = int(z['n1'])
    ext = unpack(z['ext'], n1)[:, 0]                     # [S, T, n] train hearing
    l4 = unpack(z['l4'], n1)[:, 0]
    syl = z['syl']
    cnt = ext[:, 2:6].any(1).sum(0)                      # how many syllables light each copy
    rng = np.random.default_rng(0)
    budget = int(round(float(z['budget'])))
    rows = []
    for i in rng.choice(len(syl), n_syl, replace=False):
        copies = np.nonzero(ext[i, 2:6].any(0))[0]
        spec = copies[np.argsort(cnt[copies])]            # most specific first
        code = l4[i, 5:8].any(0)
        for name, sel in (('budget', spec[:budget]), ('one tick', spec[:180]), ('all', copies)):
            force = torch.zeros(n1, dtype=torch.bool, device=DEV)
            force[torch.as_tensor(sel, device=DEV)] = True
            rec = []

            def r(t):
                rec.append((p.L1ext.on.clone(), p.L4.on.clone(), p.x_int.clone()))
                p.force_int = force if PRE <= t < PRE + 2 else None     # lit as thought on ticks PRE, PRE+1, PRE+2
            p.force_int = None
            life.play([None] * (PRE + 12), rec=r)
            p.force_int = None
            E = torch.stack([q[0] for q in rec]).numpy()[PRE - 1:]
            A4 = torch.stack([q[1] for q in rec]).numpy()[PRE - 1:]
            ovl = [(A4[t] & code).sum() / max(1, np.sqrt(A4[t].sum() * code.sum())) for t in range(len(A4))]
            rows.append(dict(syl=int(syl[i]), thought=name, n_thought=int(len(sel)),
                             echo=[int(x) for x in E.sum(1)], L4=[int(x) for x in A4.sum(1)],
                             L4_overlap_with_heard=[round(float(x), 2) for x in ovl]))
    agg = {}
    for name in ('budget', 'one tick', 'all'):
        rs = [r for r in rows if r['thought'] == name]
        agg[name] = dict(n_thought=round(float(np.mean([r['n_thought'] for r in rs])), 1),
                         echo_per_tick=[round(float(x), 1) for x in np.mean([r['echo'] for r in rs], 0)],
                         L4_per_tick=[round(float(x), 1) for x in np.mean([r['L4'] for r in rs], 0)],
                         L4_overlap=[round(float(x), 2) for x in np.mean([r['L4_overlap_with_heard'] for r in rs], 0)],
                         chance_overlap=round(float(code.mean()), 3))
    print('CHAIN', json.dumps(dict(ticks='from the first thought tick (index 0) on', **agg)), flush=True)


if __name__ == '__main__':
    main()
