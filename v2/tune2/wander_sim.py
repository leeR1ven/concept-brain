# -*- coding: utf-8 -*-
"""Offline replay of the wandering thought (R11) on a life's saved co-holding links T_W, heard counts and the thought at
every wandering onset (life_t2.py npz).  Each silence starts from the real thought and runs as many wandering steps as
the real silence had; the fatigue trace carries over between silences (as in the agent).  ASSUMED approximation: the
final T_W is used for every silence (in the life it grows over time).
Dynamics per step (core_synth.tick_thought) + optional hub control `hub` (h):
    f += (a - f)/fat_tau ; a *= 1 - fat*f ; s = assoc * (a @ W) / colsum(W)^h ; a = max(a, s) ; a /= max(a) ; k-WTA
  h = 0: ARCH_v2.  h > 0: each item unit's total incoming link weight divides its input (postsynaptic normalisation:
  a hub that every item links to gains no advantage from being linked by every item).
usage: python3 wander_sim.py out/life/L_base_s5_wander.npz [grid]"""
import sys, json, itertools
import numpy as np

FRAME = set('这是什么在哪里你要吗我好不吃颜色有几个刚才叫看宝的对呀谢客气晚安再见拜早上')
sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/bench_copy')
from world import OBJECTS, NOVEL, NAMES, COLOURS, COUNTS, SLOTS   # noqa: E402
CONTENT = set(''.join(OBJECTS + NOVEL + NAMES + COLOURS + COUNTS + SLOTS)) - FRAME


def load(path):
    d = np.load(path, allow_pickle=False)
    cmap = {int(k): v for k, v in json.loads(str(d['cmap'])).items()}
    sw = d['snap_w'].astype(int)
    steps = np.diff(np.append(sw, int(d['n_wander'])))
    return dict(W=d['T_W'], n=d['item_n'][:d['T_W'].shape[0]].astype(float), snaps=d['snaps'], steps=steps, cmap=cmap)


def simulate(D, assoc=0.5, fat=0.1, fat_tau=20.0, k=12, theta=0.5, hub=0.0, c0=0.0):
    W = D['W'].astype(np.float64)
    V = W.shape[0]
    cs = W.sum(0)
    Wn = W / np.maximum(cs + c0, 1e-9)[None, :] ** hub if hub > 0 else W
    f = np.zeros(V)
    held_cnt = np.zeros(V)
    dom = []
    ends = []
    for a0, n in zip(D['snaps'], D['steps']):
        a = a0.astype(np.float64).copy()
        if a.max() <= 0 or n <= 0:
            continue
        for _ in range(int(n)):
            f += (a - f) / fat_tau
            a *= (1.0 - fat * f)
            s = assoc * (a @ Wn)
            np.maximum(a, s, out=a)
            m = a.max()
            if m > 0:
                a /= m
            h = np.flatnonzero(a >= theta)
            if len(h) > k:
                o = h[np.argsort(-a[h], kind='stable')]
                a[o[k:]] = np.minimum(a[o[k:]], theta * 0.99)
                h = o[:k]
            held_cnt[h] += 1
            dom.append(int(np.argmax(a)))
        ends.append((int(n), tuple(sorted(np.flatnonzero(a >= theta).tolist()))))
    return held_cnt, dom, ends


def metrics(D, held_cnt, dom, ends):
    n = D['n']
    V = len(n)
    p = n / n.sum()
    top = np.argsort(-n)[:20]
    is_c = np.zeros(V, bool)
    for j, chs in D['cmap'].items():
        if j < V and any(ch in CONTENT for ch in chs):
            is_c[j] = True
    q = held_cnt / max(held_cnt.sum(), 1)
    nz = q > 0
    dc = np.bincount(dom, minlength=V).astype(float)
    dn = dc / max(dc.sum(), 1)
    dnz = dn > 0
    name = lambda j: ''.join(D['cmap'].get(int(j), ['?']))
    fav = np.argsort(-q)[:8]
    t5 = np.argsort(-q)[:5]
    long_ends = [e for n_, e in ends if n_ >= 36]               # silences >= 2,000 ticks (36+ steps)
    t5s = set(int(j) for j in t5)
    endc = [np.mean([is_c[i] for i in e]) for e in long_ends if e]
    endf = [any(i in t5s for i in e) for e in long_ends]
    r = np.zeros(V)
    for e in long_ends:
        for i in e:
            r[i] += 1.0 / max(len(e), 1)
    rn = r / max(r.sum(), 1e-9)
    rz = rn > 0
    e5 = np.argsort(-rn)[:5]
    e5c = [j for j in np.argsort(-(rn * is_c))[:5] if rn[j] > 0]
    favc = [j for j in np.argsort(-(q * is_c))[:5] if q[j] > 0]
    return dict(held_top20=round(float(q[top].sum()), 3), heard_top20=round(float(p[top].sum()), 3),
                excess_top20=round(float(q[top].sum() - p[top].sum()), 3),
                KL=round(float((q[nz] * np.log2(q[nz] / np.maximum(p[nz], 1e-9))).sum()), 3),
                H_held=round(float(-(q[nz] * np.log2(q[nz])).sum()), 2), H_heard=round(float(-(p[p > 0] * np.log2(p[p > 0])).sum()), 2),
                eff_topics=round(float(2 ** -(q[nz] * np.log2(q[nz])).sum()), 1), distinct_dom=int(dnz.sum()),
                top1_dom=round(float(dn.max()), 3), content_held=round(float(q[is_c].sum()), 3),
                content_heard=round(float(p[is_c].sum()), 3), content_excess=round(float(q[is_c].sum() - p[is_c].sum()), 3),
                ends_distinct=round(len(set(e for _, e in ends)) / max(len(ends), 1), 3), n_silences=len(ends),
                top5_held=round(float(q[t5].sum()), 3), top5_lift=round(float(q[t5].sum() / max(p[t5].sum(), 1e-9)), 2),
                top5_content=int(is_c[t5].sum()), long_n=len(long_ends),
                long_end_content=round(float(np.mean(endc)), 3) if endc else None,
                long_end_top5=round(float(np.mean(endf)), 3) if endf else None,
                long_end_distinct=round(len(set(long_ends)) / max(len(long_ends), 1), 3),
                end_top5=[name(j) for j in e5], end_top5_share=round(float(rn[e5].sum()), 3),
                end_top5_heard=round(float(p[e5].sum()), 3), end_H=round(float(-(rn[rz] * np.log2(rn[rz])).sum()), 2),
                end_top5_content=[name(j) for j in e5c], end_c5_share=round(float(rn[e5c].sum()), 3) if e5c else 0.0,
                end_c5_heard=round(float(p[e5c].sum()), 3) if e5c else 0.0,
                fav=[name(j) for j in fav], fav_content=[name(j) for j in favc])


def main():
    paths = [a for a in sys.argv[1:] if a.endswith('.npz')]
    grid = dict(assoc=[0.5], fat=[0.1], fat_tau=[20.0], k=[12], hub=[0.0], c0=[0.0])
    for a in sys.argv[1:]:
        if '=' in a:
            k, v = a.split('=')
            grid[k] = [float(x) for x in v.split(',')]
    for path in paths:
        D = load(path)
        print(f"# {path}: silences {len(D['snaps'])}, steps {int(D['steps'].sum())}", flush=True)
        for vals in itertools.product(*grid.values()):
            kw = dict(zip(grid.keys(), vals))
            kw['k'] = int(kw['k'])
            m = metrics(D, *simulate(D, **kw))
            print(json.dumps(dict(kw, **m), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
