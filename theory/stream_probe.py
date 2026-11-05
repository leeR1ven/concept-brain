"""Does the readout learn the world's regularities from a continuous stream? No labels, no syllable classes.
The fixed newborn network hears the dialogues continuously (its neurons run, nothing in it learns). Beside it, three
readouts last layer (t-1) -> copies newly lit from outside (t) learn online, tick by tick:
  count        W += eta a x^T                  (plain Hebbian counting; each copy's mean push taken out when read)
  delta weak   W += eta a (x - W^T a)^T        (prediction error), with weak decay
  delta strong the same with strong decay      (decay / learning rate plays the role of a ridge term)
Test, repeated as it listens: never-heard lines in their own order and with their syllables shuffled (same sounds).
On each tick from the 2nd syllable on, the m strongest predicted copies, m = how many copies really came on; share of
them that did. own minus shuffled = order learned from the stream. Chance level: the same with a frequency predictor.
usage: python -u -m stream_probe <dialogues> [key=value for the newborn]  (from a folder with core/ and data/)"""
import json
import sys
import time

import numpy as np
import torch

from core.config import config
from core.corpus_life import CorpusLife
from core.exam import first_lines

torch.set_num_threads(4)
DECAY = {'delta_weak': 1e-5, 'delta_strong': 3e-4}          # per tick
CHECK = (100, 300, 1000, 3000)


def main():
    total = int(sys.argv[1])
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, 's': 1, **over}), 0, 'data')
    b, p = life.brain, life.brain.pfc
    b.learning = False
    n = p.L4.on.shape[0]
    W = {k: torch.zeros(n, n, dtype=torch.float32) for k in ('count', *DECAY)}
    mean_d = {k: torch.zeros(n) for k in W}
    freq = torch.zeros(n)
    rng = np.random.default_rng(3)
    lines = first_lines(life.heldout, range(life.heldout.n), 120, lo=4, hi=10)

    def shuf(ln):
        for _ in range(20):
            s = [ln[i] for i in rng.permutation(len(ln))]
            if s != ln:
                return s
        return ln
    pairs = [(ln, shuf(ln)) for ln in lines]

    def prec(ln):
        """per predictor, the mean share of top-m predicted copies that came on, from the 2nd syllable on"""
        rec = []
        life.play([None] * 15 + [(x, k, 'C1', 0.0, 0.0) for x in ln for k in range(3)] + [None] * 3,
                  rec=lambda t: rec.append((p.L4.prev.clone(), p.L1ext.rose().clone())))
        acc = {k: [] for k in list(W) + ['freq']}
        for t in range(15 + 4, len(rec)):
            a, x = rec[t]
            m = int(x.sum())
            if m == 0 or not a.any():
                continue
            idx = torch.nonzero(a).squeeze(1)
            for k, w in W.items():
                e = w[idx].sum(0) - mean_d[k]
                acc[k].append(float(x[torch.topk(e, m).indices].float().mean()))
            acc['freq'].append(float(x[torch.topk(freq, m).indices].float().mean()))
        return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}

    def report(done, secs):
        R = [(prec(a), prec(c)) for a, c in pairs]
        out = {}
        for k in list(W) + ['freq']:
            d = np.array([r[0][k] - r[1][k] for r in R])
            out[k] = dict(own=round(float(np.mean([r[0][k] for r in R])), 4),
                          shuffled=round(float(np.mean([r[1][k] for r in R])), 4),
                          gap=round(float(d.mean()), 4), se=round(float(d.std() / np.sqrt(len(d))), 4))
        print('STREAM', json.dumps(dict(dialogues=done, over=over, res=out, secs=round(secs))), flush=True)

    t0, ticks = time.time(), 0
    prev = torch.zeros(n, dtype=torch.bool)
    n0 = life.env.n_ex
    check = [c for c in CHECK if c <= total]
    while True:
        life.tick()
        x = p.L1ext.rose()
        idx = torch.nonzero(prev).squeeze(1)
        if len(idx):
            ticks += 1
            xf = x.float()
            freq += (xf - freq) * 1e-4
            W['count'][idx] += 0.02 * xf
            for k, d in DECAY.items():
                w = W[k]
                y = w[idx].sum(0)
                w[idx] += (0.5 / len(idx)) * (xf - y)
                mean_d[k] += (y - mean_d[k]) * 1e-3
                if ticks % 100 == 0 and d:
                    w.mul_(1.0 - d * 100)
            y = W['count'][idx].sum(0)
            mean_d['count'] += (y - mean_d['count']) * 1e-3
        prev = p.L4.on.clone()
        done = life.env.n_ex - n0
        if check and done >= check[0]:
            report(done, time.time() - t0)
            check.pop(0)
            if not check:
                break


if __name__ == '__main__':
    main()
