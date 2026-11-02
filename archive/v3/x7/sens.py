"""X7 Part A (HARVEST #13): convergent thresholds in the designer's 3-D local random sensory net.

Net = local3d `threednet/model.py` layout (own position + random neighbours dx,dy in {-1,0,1}, dz in {-2..2}),
threshold = local3d `threshold_activation.py` form: fire iff sum(w x) > theta_l * sum(w)  (no bias, no top-k).
Sound = project `world/voice.py` (8 bands, 10 levels, 6 slices per syllable); ear sheet = 24 slices (4 syllables).
Arms: dom1 (1 dominant input + 3 weak), eq4, eq7 (comparable inputs).
"""
import json
import sys
import time

import numpy as np

import os as _os, sys as _sys   # repo: world.py from ../x1c, voice.py from ../../world
_H = _os.path.dirname(_os.path.abspath(__file__))
for _d in (_os.path.join(_H, '..', 'x1c'), _os.path.join(_H, '..', '..', 'world'), _H):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
from voice import Voice

F, TW, LV = 8, 24, 10
SLOT = 6
ZS = (40, 60, 120, 240)
TARGET = (0.30, 0.15, 0.08, 0.04)
import os
if os.environ.get('X7_TARGET'):
    TARGET = tuple(float(x) for x in os.environ['X7_TARGET'].split(','))
NSYL = 40


def coords(dim):
    x, y, z = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), np.arange(dim[2]), indexing="ij")
    return np.stack([x, y, z], -1).reshape(-1, 3)


def build(arm, seed):
    rng = np.random.default_rng(seed)
    fan, dom = {'dom1': (4, True), 'eq4': (4, False), 'eq7': (7, False)}[arm]
    dims = [(F, TW, 2 * LV)] + [(F, TW, z) for z in ZS]
    layers = []
    for l in range(1, len(dims)):
        prev, out = dims[l - 1], dims[l]
        oc = coords(out)
        n = len(oc)
        selfc = oc.copy()
        selfc[:, 2] = np.round(oc[:, 2] * (prev[2] - 1) / max(out[2] - 1, 1)).astype(int)
        idx = np.zeros((n, fan), np.int64)
        w = np.zeros((n, fan), np.float32)
        idx[:, 0] = np.ravel_multi_index(selfc.T, prev)
        w[:, 0] = 1.0 if dom else rng.random(n) * 0.5 + 0.5
        for j in range(1, fan):
            nb = selfc.copy()
            nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
            nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
            nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
            idx[:, j] = np.ravel_multi_index(nb.T, prev)
            w[:, j] = (rng.random(n) * 0.5 + 0.5) * (0.01 if dom else 1.0)
        layers.append(dict(idx=idx, w=w, base=w.sum(1), pos=oc[:, :2]))
    return layers


def encode(grid):
    """grid (B, F, TW) levels 0..LV -> (B, F*TW*2LV) bool, exactly half on."""
    t = np.arange(LV)
    up = grid[..., None] > t
    return np.stack([up, ~up], -1).reshape(grid.shape[0], -1)


def ratio(h, L, chunk=16):
    out = np.empty((h.shape[0], len(L['base'])), np.float32)
    for i in range(0, h.shape[0], chunk):
        hh = h[i:i + chunk].astype(np.float32)
        out[i:i + chunk] = (hh[:, L['idx']] * L['w'][None]).sum(-1) / L['base'][None]
    return out


def forward(h0, layers, thetas):
    hs = [h0]
    h = h0
    for L, th in zip(layers, thetas):
        h = ratio(h, L) > th
        hs.append(h)
    return hs


def calibrate(h0, layers):
    thetas = []
    h = h0
    for L, tg in zip(layers, TARGET):
        r = ratio(h, L)
        th = float(np.quantile(r, 1 - tg))
        thetas.append(th)
        h = r > th
    return thetas


# ---------------------------------------------------------------- stimuli
class Sounds:
    def __init__(self, seed):
        self.v = Voice(range(NSYL), F=F, T=SLOT, levels=LV, seed=seed)
        self.rng = np.random.default_rng(seed + 99)

    def syl(self, c, who='other', renoise=False):
        g = self.v.hear(c, who)
        if renoise:
            base = self.v.sound[c]
            floor = base <= 1                                   # noise-floor cells (peaks are >= 5)
            g = g.copy()
            new = self.rng.integers(0, 2, size=g.shape)
            dev = self.v.voice[who]
            g[floor] = np.clip(new + dev, 0, LV)[floor]
        return g

    def window(self, items):
        """items: list of (slot, grid) -> (F, TW)"""
        w = np.zeros((F, TW), int)
        for s, g in items:
            w[:, s * SLOT:(s + 1) * SLOT] = np.maximum(w[:, s * SLOT:(s + 1) * SLOT], g)
        return w


def jacc(a, b):
    i = (a & b).sum(-1)
    u = (a | b).sum(-1)
    return np.where(u > 0, i / np.maximum(u, 1), 1.0)


def rankdata(x):
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), float)
    xs = x[o]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def run(arm, seed):
    t0 = time.time()
    S = Sounds(seed)
    rng = np.random.default_rng(seed + 7)
    layers = build(arm, seed)
    # calibration: 40 one-syllable windows, random slot
    cal = np.stack([S.window([(int(rng.integers(4)), S.syl(c))]) for c in range(NSYL)])
    thetas = calibrate(encode(cal), layers)
    # test classes
    classes = {}
    classes['silence'] = np.zeros((1, F, TW), int)
    classes['syl1'] = np.stack([S.window([(int(rng.integers(4)), S.syl(int(rng.integers(NSYL))))]) for _ in range(40)])
    classes['syl2'] = np.stack([S.window([(s, S.syl(int(rng.integers(NSYL)))) for s in rng.choice(4, 2, replace=False)])
                                for _ in range(40)])
    sents = [[int(c) for c in rng.integers(NSYL, size=4)] for _ in range(40)]
    classes['sentence'] = np.stack([S.window([(s, S.syl(c)) for s, c in enumerate(sn)]) for sn in sents])
    sents2 = [[int(c) for c in rng.integers(NSYL, size=4)] for _ in range(40)]
    classes['two_spk'] = np.stack([np.maximum(S.window([(s, S.syl(c)) for s, c in enumerate(a)]),
                                              S.window([(s, S.syl(c, 'self')) for s, c in enumerate(b)]))
                                   for a, b in zip(sents, sents2)])
    acts = {k: forward(encode(g), layers, thetas) for k, g in classes.items()}
    for k, hs in acts.items():
        assert np.all(hs[0].sum(1) == F * TW * LV), 'input layer must be exactly half on'
    nl = len(layers) + 1
    sizes = [acts['silence'][l].shape[1] for l in range(nl)]
    res = dict(arm=arm, seed=seed, thetas=thetas, sizes=sizes)
    res['count'] = {k: [float(hs[l].sum(1).mean()) for l in range(nl)] for k, hs in acts.items()}
    res['count_sd'] = {k: [float(hs[l].sum(1).std()) for l in range(nl)] for k, hs in acts.items()}
    # chain-bearing units: usage <= 0.6 over the classes (equal weight per class)
    usage = [np.mean([hs[l].mean(0) for hs in acts.values()], 0) for l in range(nl)]
    chain = [u <= 0.6 for u in usage]
    res['n_chain'] = [int(c.sum()) for c in chain]
    res['count_chain'] = {k: [float((hs[l] & chain[l][None]).sum(1).mean()) for l in range(nl)] for k, hs in acts.items()}
    # similarity set: 40 syllables at slot 1 x {other, self, renoise}
    sim = []
    for var in ('other', 'self', 'renoise'):
        for c in range(NSYL):
            g = S.syl(c, 'self' if var == 'self' else 'other', renoise=(var == 'renoise'))
            sim.append(S.window([(1, g)]))
    sim = np.stack(sim)
    hs = forward(encode(sim), layers, thetas)
    n = len(sim)
    iu = np.triu_indices(n, 1)
    din = np.abs(sim[:, None] - sim[None]).sum((2, 3))[iu].astype(float)
    res['rho'], res['J_sim_voice'], res['J_sim_noise'], res['J_diff'], res['J_sent_diff'], res['J_sent_voice'] = \
        [], [], [], [], [], []
    oth = np.arange(NSYL)
    pa, pb = np.triu_indices(NSYL, 1)
    for l in range(nl):
        h = hs[l]
        hf = h.astype(np.float32)
        inter = hf @ hf.T
        cnt = hf.sum(1)
        ham = (cnt[:, None] + cnt[None] - 2 * inter)[iu]
        res['rho'].append(spearman(din, ham.astype(float)))
        res['J_sim_voice'].append(float(jacc(h[oth], h[oth + NSYL]).mean()))
        res['J_sim_noise'].append(float(jacc(h[oth], h[oth + 2 * NSYL]).mean()))
        res['J_diff'].append(float(jacc(h[pa], h[pb]).mean()))
        hsn = acts['sentence'][l]
        res['J_sent_diff'].append(float(jacc(hsn[pa], hsn[pb]).mean()))
        cm = chain[l][None]
        hc = h & cm
        hcf = hc.astype(np.float32)
        ic = hcf @ hcf.T
        cc = hcf.sum(1)
        res.setdefault('rho_chain', []).append(spearman(din, (cc[:, None] + cc[None] - 2 * ic)[iu].astype(float)))
        res.setdefault('Jc_sim_voice', []).append(float(jacc(hc[oth], hc[oth + NSYL]).mean()))
        res.setdefault('Jc_diff', []).append(float(jacc(hc[pa], hc[pb]).mean()))
        hsc = hsn & cm
        res.setdefault('Jc_sent_diff', []).append(float(jacc(hsc[pa], hsc[pb]).mean()))
    # sentence similar pair: same sentence other vs own voice
    sv = np.stack([S.window([(s, S.syl(c, 'self')) for s, c in enumerate(sn)]) for sn in sents])
    hsv = forward(encode(sv), layers, thetas)
    res['J_sent_voice'] = [float(jacc(acts['sentence'][l], hsv[l]).mean()) for l in range(nl)]
    res['Jc_sent_voice'] = [float(jacc(acts['sentence'][l] & chain[l][None], hsv[l] & chain[l][None]).mean())
                            for l in range(nl)]
    # recruitment by extra L1 units (scattered vs coherent patch), background = similarity set 'other'
    bg = [h[:NSYL] for h in hs]
    k1 = float(bg[1].sum(1).mean())
    rec = {}
    pos1 = layers[0]['pos']
    for frac in (0.01, 0.03):
        k = max(1, int(round(frac * k1)))
        for mode in ('scatter', 'patch'):
            ch = np.zeros((NSYL, nl - 2))
            for i in range(NSYL):
                h1 = bg[1][i].copy()
                off = np.nonzero(~h1)[0]
                if mode == 'scatter':
                    add = rng.choice(off, k, replace=False)
                else:
                    cx, cy = int(rng.integers(1, F - 1)), int(rng.integers(1, TW - 1))
                    inpatch = off[(np.abs(pos1[off, 0] - cx) <= 1) & (np.abs(pos1[off, 1] - cy) <= 1)]
                    add = rng.choice(inpatch, min(k, len(inpatch)), replace=False)
                h1[add] = True
                h = h1[None]
                for l in range(1, len(layers)):
                    h = ratio(h, layers[l]) > thetas[l]
                    ch[i, l - 1] = (h[0] ^ bg[l + 1][i]).sum() / len(add)
            rec[f'{mode}_{frac}'] = ch.mean(0).tolist()
    res['recruit_per_unit'] = rec
    res['k1'] = k1
    res['secs'] = time.time() - t0
    return res


if __name__ == '__main__':
    arms = sys.argv[1].split(',')
    seeds = [int(s) for s in sys.argv[2].split(',')]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    allr = []
    for seed in seeds:
        for arm in arms:
            r = run(arm, seed)
            allr.append(r)
            print(json.dumps({k: r[k] for k in ('arm', 'seed', 'thetas', 'secs')}), flush=True)
            print(' count', {k: [round(x) for x in v] for k, v in r['count'].items()}, flush=True)
            print(' chain', {k: [round(x) for x in v] for k, v in r['count_chain'].items()}, flush=True)
            print(' rho', [round(x, 3) for x in r['rho']], 'Jdiff', [round(x, 3) for x in r['J_diff']],
                  'Jsim', [round(x, 3) for x in r['J_sim_voice']], flush=True)
            print(' Jsent', [round(x, 3) for x in r['J_sent_diff']], 'chain: rho', [round(x, 3) for x in r['rho_chain']],
                  'Jdiff', [round(x, 3) for x in r['Jc_diff']], 'Jsim', [round(x, 3) for x in r['Jc_sim_voice']],
                  'Jsent', [round(x, 3) for x in r['Jc_sent_diff']], flush=True)
            print(' recruit', {k: [round(x, 2) for x in v] for k, v in r['recruit_per_unit'].items()}, flush=True)
    if out:
        with open(out, 'w') as f:
            json.dump(allr, f, indent=1)
