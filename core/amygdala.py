"""The amygdala (ARCH 三 杏仁核, innate value), grown into a living brain (designer 11-12: "把模型做成真正完整的模型，包含
所有的区域").

- Its own population (n units). A "pleased" group (n_pleased) is wired at birth to the skin's pat receptor: when the
  adults are pleased they pat the child (the world's pleasure arrives as a sensation, not as a number; ARCH 四:
  好坏只来自先天价值). A pat lights the pleased group by its innate wiring alone.
- Learned links from the copies the PFC hears (the senses through the concept area, force and stop copies) to its
  units: the one rule, the one decay (ARCH: 同时亮就加强，再由降权留下各次都亮的共同部分). They light its units in
  anticipation; a return from the PFC (a thought of being pleased) lights them too.
- Its units have copies in PFC layer 1 and in the hippocampus like any area, so the PFC learns what comes before
  being pleased (Fable a35 #1).
- The reward the brain uses (Brain.reward, ARCH 四): the share of the pleased group lit by the pat itself, minus what
  the PFC's graded table expected of those copies this tick (an expected pat changes nothing). Learned or recalled
  activity never counts as reward (Fable a35: it would learn to be pleased by what is frequent).
- No innate output to the blockers ("pleased -> speak more" is the loop F7 measured); an upset group is left for later
  (the n - n_pleased units have no innate input yet)."""
import math

import numpy as np
import torch

from .brain import reciprocal_loop
from .life import log
from .net import DEV, Fixed, Links, Pop, local_net, own_thresholds, radius_lists, random_lists, topk_on
from .grow import _extend_links, _extend_pop


def pat_bits(r, n_pat):
    """the skin's pat receptor: a thermometer of the pat's strength (0..1), n_pat levels"""
    return torch.arange(n_pat, device=DEV) < int(round(float(max(0.0, min(1.0, r))) * n_pat))


class Amygdala:
    def __init__(self, n, n_pleased, n_pat, n1, rows_heard, c, rng):
        self.n, self.np_, self.n_pat, self.c = n, n_pleased, n_pat, c
        K = max(4, n_pat // 2)
        self.innate = Fixed(rng.integers(0, n_pat, (n_pleased, K)), rng.uniform(0.5, 1.5, (n_pleased, K)))
        self.theta_innate = 0.5 * self.innate.w.sum(1)          # a firm pat lights them; a light one about half
        width = int(c.get('amyg_fan', 8))
        idx = random_lists(n1, n, width, rng)
        idx[~np.asarray(rows_heard)] = n                        # only copies the PFC hears are candidates
        self.learned = Links(idx, 0.0, n, c, gamma=c['gamma_pfc'], name='copies -> amygdala')
        self.pop = Pop(n)
        self.usual = torch.zeros(n, device=DEV)
        self.k = max(1, int(round(c.get('amyg_share', 0.05) * n)))
        self.adapt = False
        self.r_innate = 0.0

    def step(self, copies, pat, ret=None):
        """copies: bool [n1] the copies the PFC hears now; pat: the pat's strength; ret: the PFC's returns to it"""
        on = torch.zeros(self.n, dtype=torch.bool, device=DEV)
        inn = self.innate.drive(pat_bits(pat, self.n_pat)) >= self.theta_innate
        on[:self.np_] = inn
        self.r_innate = float(inn.float().mean())
        e = self.learned.drive(copies) - self.usual              # above each unit's own usual level (ARCH 二.4)
        if self.adapt:
            self.usual += e / self.c['tau_usual']
        on |= (e > self.c['theta1']) & topk_on(e, self.k)
        if ret is not None:
            on |= ret > 0
        self.pop.set(on)
        return on

    def learn(self, pre):
        """pre: Pop - the PFC's external copies (what was heard); the one rule (A on or just on when B rises)"""
        prev = self.pop.prev if getattr(self.pop, 'prev', None) is not None else torch.zeros_like(self.pop.on)
        self.learned.learn(pre, self.pop.rose(), prev, self.c['eta'])


def add_amygdala(life, n=400, n_pleased=200, n_pat=16, cal_ticks=500):
    """grows the amygdala into the living brain (as grow.add_concept does the concept area): its copies are added at
    the end of PFC layer 1 and of the hippocampus's copies; every existing neuron and learned link is kept"""
    b, c = life.brain, life.c
    rng = np.random.default_rng(b.seed + 1300)
    ch, pfc, hip = b.chains, b.pfc, b.hip
    n_old = ch.n
    n_new = n_old + n
    heard_old = getattr(b, 'pfc_in', None)
    rows_heard = np.zeros(n_new, bool)
    rows_heard[:n_old] = True if heard_old is None else heard_old.cpu().numpy()
    am = Amygdala(n, n_pleased, n_pat, n_new, rows_heard, c, rng)
    # chains: one more block of copies at the end (old indices unchanged)
    W = int(math.ceil(math.sqrt(n)))
    u = np.arange(n)
    g = np.stack([np.full_like(u, 6), u // W, u % W], 1)
    ch.segs.append(('amyg', 0, n_old, n_new))
    ch.grid = np.concatenate([ch.grid, g])
    p = np.clip(np.stack([g[:, 1] / max(1, W - 1), g[:, 2] / max(1, W - 1)], 1) + rng.normal(0, 0.005, (n, 2)), 0, 1)
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
    for tb in getattr(pfc, 'P', {}).values():
        _extend_links(tb, radius_lists(ch.pos, ch.pos, c['r_learn']), n_new, n_old)
    G1P = getattr(pfc, 'G1P', None)
    if G1P is not None:                                    # its sink index moves to the new end
        G1P.idx = torch.where(G1P.idx >= n_old, n_new, G1P.idx)
        G1P.n_post = n_new
    if getattr(pfc, 'G1C', None) is not None:
        pfc.G1C.n = n_new
    if heard_old is not None:                              # the PFC hears the amygdala's copies
        b.pfc_in = pfc.pfc_in = torch.cat([heard_old, torch.ones(n, dtype=torch.bool, device=DEV)])
    GT = getattr(pfc, 'GT', None)
    if GT is not None:                                     # the graded table: new rows and columns, old values kept
        m = len(GT.sel)
        for nm in ('E', 'I'):
            M = torch.zeros((m + n, m + n), device=DEV)
            M[:m, :m] = getattr(GT, nm)
            setattr(GT, nm, M)
        GT.sel = torch.cat([GT.sel, torch.arange(n_old, n_new, device=DEV)])
        GT.n = n_new
        for nm in ('gt_p', 'gt_used'):
            setattr(pfc, nm, torch.cat([getattr(pfc, nm), torch.zeros(n, device=DEV)]))
        pfc.x_int_prev = torch.cat([pfc.x_int_prev, torch.zeros(n, dtype=torch.bool, device=DEV)])
    for nm in ('pred_mask', 'motor_mask'):
        if getattr(pfc, nm, None) is not None:
            setattr(pfc, nm, torch.cat([getattr(pfc, nm), torch.zeros(n, dtype=torch.bool, device=DEV)]))
    for pp in pfc.pops():
        _extend_pop(pp, n_new)
    pfc.n = n_new
    pfc.usual = torch.cat([pfc.usual, torch.zeros(n, device=DEV)])
    if getattr(pfc, 'usual4', None) is not None:
        pfc.usual4 = torch.cat([pfc.usual4, torch.zeros(n, device=DEV)])
    pfc.x_int = torch.cat([pfc.x_int, torch.zeros(n, dtype=torch.bool, device=DEV)])
    inf = torch.full((n,), float('inf'), device=DEV)
    pfc.thr2, pfc.thr3 = torch.cat([pfc.thr2, inf]), torch.cat([pfc.thr3, inf])
    # hippocampus: copies of the new units; each Hs unit gets a few inputs from them and recall targets in them
    for pp in (hip.H1, hip.H1ext):
        _extend_pop(pp, n_new)
    hip.h_int = torch.cat([hip.h_int, torch.zeros(n, dtype=torch.bool, device=DEV)])
    hip.usual = torch.cat([hip.usual, torch.zeros(n, device=DEV)])
    kin = max(1, int(c.get('hip_fan', 150)) // 30)
    hip.rec.idx = torch.cat([hip.rec.idx, torch.as_tensor(rng.integers(n_old, n_new, (hip.nh, kin)), device=DEV)], 1)
    hip.rec.w = torch.cat([hip.rec.w, torch.as_tensor(rng.uniform(0.5, 1.5, (hip.nh, kin)), dtype=torch.float32,
                                                      device=DEV)], 1)
    kout = max(1, c['n_h1'] * n // n_old)
    new_t = torch.as_tensor(n_old + random_lists(hip.nh, n, kout, rng), device=DEV)
    hip.H1t.idx = torch.cat([torch.where(hip.H1t.idx >= n_old, n_new, hip.H1t.idx), new_t], 1)
    hip.H1t.w = torch.cat([hip.H1t.w, torch.zeros(new_t.shape, device=DEV)], 1)
    hip.H1t.n_post = n_new
    b.amyg, b.n1 = am, n_new
    b.amyg_pleased = torch.arange(n_old, n_old + n_pleased, device=DEV)   # the pleased group's copies
    ch.ret_amyg = torch.zeros(n, device=DEV)
    life.registry = getattr(life, 'registry', []) + [('amygdala: pat -> pleased group (innate)', n_pleased, n_pat)]
    log(f'amygdala added: {n} units ({n_pleased} pleased, innate from a {n_pat}-level pat); PFC {n_old} -> {n_new}')
    L1s, L2s = [], []
    for _ in range(cal_ticks):                             # the new PFC units' own thresholds (ARCH 二.4)
        life.tick()
        L1s.append(pfc.L1.on.clone())
        L2s.append(pfc.L2.on.clone())
    X1, X2 = torch.stack(L1s), torch.stack(L2s)
    rho = c.get('pfc_mid_rate') or b.birth_density
    new2 = Fixed(pfc.ff[0].idx[n_old:], pfc.ff[0].w[n_old:])
    D2 = new2.drive_batch(X1)
    pfc.thr2[n_old:] = own_thresholds(D2, rho)
    X2[:, n_old:] = D2 > pfc.thr2[n_old:]
    new3 = Fixed(pfc.ff[1].idx[n_old:], pfc.ff[1].w[n_old:])
    pfc.thr3[n_old:] = own_thresholds(new3.drive_batch(X2), rho)
    log(f'new PFC units calibrated over {cal_ticks} ticks')
    return am
