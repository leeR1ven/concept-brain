"""The hippocampus' own code: how consistent is it, as a function of its fan-in (docs/THEORY_fable.md section 5).
Its units take `fan` random copies with weights U(0.5, 1.5); the kh most driven light. Inputs: the recorded first-layer
copies of the same syllable heard twice (ear noise), in another voice, and of different syllables (tick 3, the peak).
Overlap of the lit sets = shared / kh; chance = kh / nh. numpy only.
usage: python3 hippo.py codes.npz [nh=20000]"""
import json
import sys

import numpy as np

from readout import load


def main():
    z = load(sys.argv[1])
    nh = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    X = [z['ext'][h][:, 4].astype(np.float32) for h in range(3)]      # tick 3
    S, n1 = X[0].shape
    use = np.arange(0, S, 4)                                            # 150 syllables
    rng = np.random.default_rng(0)
    rep = {}
    for fan in (24, 48, 96, 192, 384, 768):
        idx = rng.integers(0, n1, (nh, fan))
        w = rng.uniform(0.5, 1.5, (nh, fan)).astype(np.float32)
        D = [np.zeros((len(use), nh), np.float32) for _ in range(3)]
        for a in range(0, nh, 1000):
            R = np.zeros((min(1000, nh - a), n1), np.float32)
            for j in range(R.shape[0]):
                np.add.at(R[j], idx[a + j], w[a + j])
            for h in range(3):
                D[h][:, a:a + R.shape[0]] = X[h][use] @ R.T
        for kh in (50, 200, 800):
            tops = [np.argpartition(-d, kh - 1, axis=1)[:, :kh] for d in D]
            sets = [[set(r) for r in t] for t in tops]
            same = np.mean([len(sets[0][i] & sets[1][i]) / kh for i in range(len(use))])
            voice = np.mean([len(sets[0][i] & sets[2][i]) / kh for i in range(len(use))])
            other = np.mean([len(sets[0][i] & sets[0][(i + 7) % len(use)]) / kh for i in range(len(use))])
            mu = float(np.mean([(X[0][use] @ (np.bincount(idx[u], minlength=n1) > 0)).mean()
                                for u in range(0, nh, 2000)]))
            rep[f'fan {fan} kh {kh}'] = dict(same=round(float(same), 3), voice=round(float(voice), 3),
                                              other=round(float(other), 3), chance=round(kh / nh, 4),
                                              active_inputs_per_unit=round(mu, 2))
    inp = [z['ext'][h][:, 4] for h in range(3)]
    j = lambda a, b: float(((a & b).sum(1) / np.maximum((a | b).sum(1), 1)).mean())
    rep['input copies (Jaccard)'] = dict(same=round(j(inp[0], inp[1]), 3), voice=round(j(inp[0], inp[2]), 3),
                                         other=round(j(inp[0], np.roll(inp[0], 7, axis=0)), 3))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
