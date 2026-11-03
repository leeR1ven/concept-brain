"""The shared concept area (ARCH 三 感觉区 · 抽象概念区), and growing it into a brain that is already living.

ConceptArea: one population for every sensory area. It reads the feature units of all of them (the chain sources of
the sound and touch areas; vision later), lights a fixed share of its units each tick (the most driven above each
unit's usual level), and learns feature -> concept links when both are on together (同时亮的互相加强). An object
or word keeps its concept units on while its features change (another place in the view, another voice or speed),
so the new features get linked to the same concept: invariance is learned. Like every area it has a 1:1 copy in the
PFC and in the hippocampus, and a return from each copy.

add_concept(life): grows the area into a living brain (ARCH 三 链式神经元: when neurons are added, their chain
copies are added too). Every existing neuron and learned link is kept; only new ones are added beside them."""
import math

import numpy as np
import torch

from .brain import reciprocal_loop
from .life import log
from .net import DEV, Fixed, Links, Pop, local_net, own_thresholds, radius_lists, random_lists, topk_on


class ConceptArea:
    def __init__(self, n_src, c, rng):
        s = c['s']
        self.c, self.n_src = c, n_src
        self.n, self.k = c['nc'] * s, c['kc'] * s
        K = c['rec_fan']
        self.birth = Fixed(rng.integers(0, n_src, (self.n, K)), rng.uniform(0.5, 1.5, (self.n, K)))
        self.learned = Links(random_lists(n_src, self.n, c['n_cl'], rng), 0.0, self.n, c, gain=c['g_c'],
                             gamma=c['gamma_pfc'], name='feature -> concept')
        self.pop, self.src = Pop(self.n), Pop(n_src)
        self.usual = torch.zeros(self.n, device=DEV)
        self.win = torch.ones((), device=DEV)           # the usual drive of a lit unit (scale of a return)
        self.adapt = False

    def step(self, src, ret):
        d = self.birth.drive(src) + self.learned.drive(src)
        e = d - self.usual
        if self.adapt:
            self.usual += e / self.c['tau_usual']
        if ret is not None:
            e = e + self.c['g_ret'] * self.win * ret       # a return lights its own concept unit
        on = topk_on(e, self.k)
        if self.adapt and on.any():
            self.win += (e[on].mean() - self.win) / self.c['tau_usual']
        self.src.set(src)
        self.pop.set(on)

    def learn(self):
        """同时亮的互相加强: feature A -> concept B grows while both are on (A on now or just before)"""
        L, fw = self.learned, self.c['fall_win']
        a = torch.nonzero(self.src.on | self.src.held(fw)).squeeze(1)
        if a.numel() == 0:
            return
        f = torch.cat([self.pop.on, torch.zeros(1, dtype=torch.bool, device=DEV)])
        L.apply(a, self.c['eta'] * L.plast(L.w[a]) * f[L.idx[a]].float())


def _extend_pop(p, n_new):
    k = n_new - p.n
    p.on = torch.cat([p.on, torch.zeros(k, dtype=torch.bool, device=DEV)])
    p.prev = torch.cat([p.prev, torch.zeros(k, dtype=torch.bool, device=DEV)])
    p.age = torch.cat([p.age, torch.full((k,), 99, dtype=torch.int16, device=DEV)])
    p.age_prev = torch.cat([p.age_prev, torch.full((k,), 99, dtype=torch.int16, device=DEV)])
    p.n = n_new


def _extend_links(L, idx_new, n_post_new, keep_rows):
    """new candidate lists idx_new [rows, width]; the first keep_rows rows keep their old weights (their old
    targets come first in the new lists, in the same order)"""
    w = torch.zeros(idx_new.shape, device=DEV)
    old_len = (L.idx < L.n_post).sum(1)
    j = torch.arange(L.idx.shape[1], device=DEV)[None, :]
    m = j < old_len[:, None]
    assert torch.equal(torch.where(m, L.idx, 0), torch.where(m, idx_new[:keep_rows, :L.idx.shape[1]], 0)), \
        'old targets must come first in the new lists'
    w[:keep_rows, :L.w.shape[1]] = torch.where(m, L.w, torch.zeros_like(L.w))
    L.idx, L.w, L.n_post, L.n_pre = idx_new, w, n_post_new, idx_new.shape[0]


def add_concept(life, cal_ticks=1000):
    b, c = life.brain, life.c
    rng = np.random.default_rng(b.seed + 900)
    ch, pfc, hip = b.chains, b.pfc, b.hip
    n_old = ch.n
    _, _, n_src, _ = ch.seg('force')[0]                   # sound and touch copies come first: the feature units
    cc = ConceptArea(n_src, c, rng)
    n_new = n_old + cc.n
    # chains: one more block of copies at the end (old indices unchanged)
    W = int(math.ceil(math.sqrt(cc.n)))
    u = np.arange(cc.n)
    g = np.stack([np.full_like(u, 4), u // W, u % W], 1)
    ch.segs.append(('concept', 0, n_old, n_new))
    ch.grid = np.concatenate([ch.grid, g])
    p = np.clip(np.stack([g[:, 1] / max(1, W - 1), g[:, 2] / max(1, W - 1)], 1) +
                rng.normal(0, 0.005, (cc.n, 2)), 0, 1)
    ch.pos = torch.cat([ch.pos, torch.as_tensor(p, dtype=torch.float32, device=DEV)])
    ch.n = n_new
    # PFC: new units at the new positions, made by the same generator; old rows unchanged
    for ly in pfc.ff:
        idx, w = local_net(ch.grid, g, c['K'], c['dy'][0], c['dz'][0], rng)
        ly.idx = torch.cat([ly.idx, torch.as_tensor(idx, device=DEV)])
        ly.w = torch.cat([ly.w, torch.as_tensor(w, device=DEV)])
    li = reciprocal_loop(pfc.ff, n_new, c['loop_fan'], rows=slice(n_old, n_new))
    lw = torch.as_tensor(rng.uniform(0.5, 1.5, tuple(li.shape)), dtype=torch.float32, device=DEV)
    pfc.loop.idx = torch.cat([torch.where(pfc.loop.idx >= n_old, n_new, pfc.loop.idx), li])
    pfc.loop.w = torch.cat([pfc.loop.w, c['g_loop'] * lw / lw.sum(1, keepdim=True)])
    pfc.loop.n_post, pfc.loop.n_pre = n_new, n_new
    _extend_links(pfc.L41, radius_lists(ch.pos, ch.pos, c['r_learn']), n_new, n_old)
    _extend_links(pfc.L44, radius_lists(ch.pos, ch.pos, c['r_learn'], True), n_new, n_old)
    for pp in pfc.pops():
        _extend_pop(pp, n_new)
    pfc.n = n_new
    pfc.usual = torch.cat([pfc.usual, torch.zeros(cc.n, device=DEV)])
    pfc.x_int = torch.cat([pfc.x_int, torch.zeros(cc.n, dtype=torch.bool, device=DEV)])
    inf = torch.full((cc.n,), float('inf'), device=DEV)
    pfc.thr2, pfc.thr3 = torch.cat([pfc.thr2, inf]), torch.cat([pfc.thr3, inf])   # set after a short stream
    # hippocampus: copies of the new units; each Hs unit also gets a few inputs from them and recall targets in them
    for pp in (hip.H1, hip.H1ext):
        _extend_pop(pp, n_new)
    hip.h_int = torch.cat([hip.h_int, torch.zeros(cc.n, dtype=torch.bool, device=DEV)])
    hip.usual = torch.cat([hip.usual, torch.zeros(cc.n, device=DEV)])
    kin = max(1, c['rec_fan'] // 3)
    hip.rec.idx = torch.cat([hip.rec.idx, torch.as_tensor(rng.integers(n_old, n_new, (hip.nh, kin)), device=DEV)], 1)
    hip.rec.w = torch.cat([hip.rec.w, torch.as_tensor(rng.uniform(0.5, 1.5, (hip.nh, kin)), dtype=torch.float32,
                                                      device=DEV)], 1)
    kout = max(1, c['n_h1'] * cc.n // n_old)
    new_t = torch.as_tensor(n_old + random_lists(hip.nh, cc.n, kout, rng), device=DEV)
    hip.H1t.idx = torch.cat([torch.where(hip.H1t.idx >= n_old, n_new, hip.H1t.idx), new_t], 1)
    hip.H1t.w = torch.cat([hip.H1t.w, torch.zeros(new_t.shape, device=DEV)], 1)
    hip.H1t.n_post = n_new
    b.concept, b.n1 = cc, n_new
    ch.ret_concept = torch.zeros(cc.n, device=DEV)
    log(f'concept area added: {cc.n} units ({cc.k} on per tick) reading {n_src} feature units; PFC {n_old} -> {n_new}')
    # the new PFC units' own thresholds, from the drive they receive over a short stretch of life (ARCH 二.4)
    L1s, L2s = [], []
    for _ in range(cal_ticks):
        life.tick()
        L1s.append(pfc.L1.on.clone())
        L2s.append(pfc.L2.on.clone())
    X1, X2 = torch.stack(L1s), torch.stack(L2s)
    rho = c.get("pfc_mid_rate") or b.birth_density          # the middle layers' own rate, as at birth
    new2 = Fixed(pfc.ff[0].idx[n_old:], pfc.ff[0].w[n_old:])
    D2 = new2.drive_batch(X1)
    pfc.thr2[n_old:] = own_thresholds(D2, rho)
    X2[:, n_old:] = D2 > pfc.thr2[n_old:]
    new3 = Fixed(pfc.ff[1].idx[n_old:], pfc.ff[1].w[n_old:])
    pfc.thr3[n_old:] = own_thresholds(new3.drive_batch(X2), rho)
    log(f'new PFC units calibrated over {cal_ticks} ticks')
    return cc
