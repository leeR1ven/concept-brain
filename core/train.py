"""One long life: the model overhears the dialogue world with learning on, and is watched as it goes.
Every `every` ticks: activity per layer and whether the PFC last layer / hippocampus are stuck (share of units
also on 3 ticks earlier). Every `probe` dialogues (frozen, state restored): after each seen question, the rank of
the answer's first syllable (of 34) by the share of its sound copies lit inside the PFC and the hippocampus during
the 12 ticks after the question (a record of what it "thinks of", not a goal).
Usage: python -u -m core.train [dialogues] [key=value ...]"""
import json
import sys
import time

import numpy as np
import torch

from .config import config
from .life import Life, log
from .net import DEV
from .world import world as W
from .world.adapter import C2, SID, line_script


def answer_ranks(life):
    b, w, codes = life.brain, life.env.w, life.codes
    out = {'pfc': [], 'hip': []}
    for (f, x) in w.pairs:
        q, a = W.question(f, x), W.answer(f, x, w.props)
        rec = {'pfc': [], 'hip': []}
        life.play([None] * 4 + line_script(q, C2) + [None] * 12, mute=True,
                  rec=lambda t: (rec['pfc'].append(b.pfc.x_int.clone()), rec['hip'].append(b.hip.h_int.clone())))
        e, t = 4 + 3 * len(q), SID[a[0]]
        for k in out:
            H = torch.stack(rec[k])[e:].any(0)
            s = torch.stack([H[c].float().mean() if c.numel() else torch.zeros((), device=DEV) for c in codes])
            out[k].append(int((s > s[t]).sum()) + int((s == s[t]).sum()))     # ties count against
    return {k: round(float(np.mean(v)), 1) for k, v in out.items()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and '=' not in sys.argv[1] else 5000
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[1:] if '=' in a)}
    every, probe, seed = over.pop('every', 2000), over.pop('probe', 250), over.pop('seed', 0)
    life = Life(config(**over), seed)
    b, w = life.brain, life.env.w
    names = b.count_names() + ['L4 stuck %', 'Hs stuck %']
    print('PROBE', 0, json.dumps(answer_ranks(life)), flush=True)
    acc, k, t, start, nxt = torch.zeros(len(names), device=DEV), 0, 0, w.n_ex, probe
    h4, hs = [], []
    b.learning = True
    t0 = time.time()
    while w.n_ex - start < n:
        life.tick()
        h4 = (h4 + [b.pfc.L4.on.clone()])[-4:]
        hs = (hs + [b.hip.Hs.on.clone()])[-4:]
        st = [(h[-1] & h[0]).sum() / h[-1].sum().clamp_min(1) * 100 if len(h) == 4 else torch.zeros((), device=DEV)
              for h in (h4, hs)]
        acc += torch.cat([b.counts().float(), torch.stack(st).float()])
        k += 1
        t += 1
        if t % every == 0:
            print('COUNT', w.n_ex - start, f'{(time.time() - t0) / t * 1000:.0f} ms/tick',
                  json.dumps({nm: round(v) for nm, v in zip(names, (acc / k).tolist())}), flush=True)
            acc.zero_()
            k = 0
        if w.n_ex - start >= nxt:
            b.learning = False
            print('PROBE', nxt, json.dumps(answer_ranks(life)), flush=True)
            b.learning = True
            nxt += probe
    # consolidation check (ARCH 三 海马体: long-term memory is what the cortex learned from repeated recall): clear
    # the hippocampus's learned links and probe again; whatever the PFC still thinks of is in the cortex now
    b.learning = False
    before = answer_ranks(life)
    for tb in b.hip.plastic():
        tb.w.zero_()
    b.hip.usual.zero_()
    print('HIPPOCAMPUS CLEARED', json.dumps(dict(before=before, after=answer_ranks(life))), flush=True)
    log('done')


if __name__ == '__main__':
    main()
