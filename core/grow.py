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
                             gamma=c.get('gamma_concept') or c['gamma_pfc'], name='feature -> concept')
        self.pop, self.src = Pop(self.n), Pop(n_src)
        self.usual = torch.zeros(self.n, device=DEV)
        self.win = torch.ones((), device=DEV)           # the usual drive of a lit unit (scale of a return)
        self.adapt = False

    def setup_b(self, sight_bits, rng):
        """config concept_rule 'b' (bench 11-10, fable-lab stay_bench / coord-lab stay_bench_b): every feature linked
        to every unit by learned links, excitatory E >= 0 and through the feature's inhibitory partner I >= 0 (no link
        changes sign), changed by the one graded check. With sight_bits: a second sense (ARCH 三: the concept area
        takes every sense) - birth links from the sight input bits; what is seen lights its own units."""
        self.E = torch.zeros((self.n_src, self.n), device=DEV)
        self.I = torch.zeros((self.n_src, self.n), device=DEV)
        self.mu = torch.zeros(self.n, device=DEV)          # each unit's usual learned push (taken out, ARCH 二.4)
        self._chk = None
        self.vbirth = None
        if sight_bits:
            K = self.c['rec_fan']
            self.vbirth = Fixed(rng.integers(0, sight_bits, (self.n, K)), rng.uniform(0.5, 1.5, (self.n, K)))
            self.v_rest = None                             # each unit's sight drive when nothing is seen (set by add_concept)

    def step_b(self, src, ret, sight, seen=None):
        """A unit is lit when it is among the strongest k of its bottom-up push from the sound (birth drive above its
        usual level, in units of a usual winner's drive, + its learned push above its usual level), or a return lights
        it, or what is seen lights it. The check (in learn): lit vs its bottom-up push from the sound, clipped to
        0..1; a unit that is not lit and was pushed less than a winner takes no part. So a unit its own sound input
        lights changes nothing; a unit lit by a return or by sight while the sound pushed it too little gains links
        from the sound features lit now (designer 11-10: "原本没有连接的…产生了赫布连接"); a unit pushed like a winner
        that did not light loses."""
        c = self.c
        e = self.birth.drive(src) - self.usual
        if self.adapt:
            self.usual += e / c['tau_usual']
            ob = topk_on(e, self.k)
            if ob.any():
                self.win += (e[ob].mean() - self.win) / c['tau_usual']
        a = torch.nonzero(src).squeeze(1)
        q = (self.E[a].sum(0) - self.I[a].sum(0)) if a.numel() else torch.zeros(self.n, device=DEV)
        if self.adapt:
            self.mu += (q - self.mu) / c['tau_usual']
        tot = e / self.win.clamp_min(1e-6) + c['g_c'] * (q - self.mu)
        lit = topk_on(tot, self.k)
        heard = lit                                      # lit from the sound alone
        if sight is not None and self.vbirth is not None and (self.v_rest is not None or seen is not None):
            vd = self.vbirth.drive(sight)                # seen (sight_as_sense): the eye's input differs from nothing
            if seen is None and c.get('sight_ref', 'usual') == 'rest':  # old: above the drive when nothing is seen.
                lit = lit | topk_on(vd - self.v_rest - 1e-4, self.k)   # Units whose few sight links sit on bits off
            elif seen if seen is not None else bool(((vd - self.v_rest).abs() > 1e-6).any()):   # at rest win for
                # every sign (sets overlap 13 %, chance 1 %)
                # each unit's own usual drive while something is seen (ARCH 二.4): sets as distinct as chance (1.2 %)
                self.v_n = getattr(self, 'v_n', 0) + 1
                r = max(1.0 / self.v_n, 1.0 / c['tau_usual'])
                vu = getattr(self, 'v_usual', None)
                self.v_usual = vd.clone() if vu is None else vu + r * (vd - vu)
                sv = topk_on(vd - self.v_usual, self.k)
                if seen is not None and sv.any():        # sight_as_sense (Fable a40 A2): a sign only partly on the
                    m = float((vd - self.v_usual)[sv].mean())   # fovea (the board drifting) lights a weaker, other set:
                    vw = getattr(self, 'v_win', None)    # it teaches only when its drive above usual is at least
                    self.v_win = m if vw is None else vw + r * (m - vw)   # sight_guard x the usual for a seen sign
                    if m < c.get('sight_guard', 0.7) * self.v_win:
                        sv = torch.zeros_like(sv)
                lit = lit | sv
        came = lit                                       # what the check compares: lit from below (sound, sight)
        self.below = lit                                 # what is sent on as heard (Fable a37: not the echoes)
        if ret is not None:
            lit = lit | (ret > 0)
            if c.get('concept_ret_teach', True):         # False (test switch): a unit lit only by a thought's return
                came = lit                               # takes no part as "lit" (an echo, as in the PFC's own check)
        pred = tot.clamp(0.0, 1.0)
        pred = torch.where(came | (tot >= 1.0), pred, torch.zeros_like(pred))
        self._chk = (a, came.float() - pred)
        self.src.set(src)
        if not c.get('sight_to_code', True):             # test switch: the seen units teach but are not sent on
            lit = heard | (ret > 0) if ret is not None else heard
            self.below = heard
        self.pop.set(lit)

    def step(self, src, ret, sight=None, seen=None):
        if getattr(self, 'E', None) is not None:
            return self.step_b(src, ret, sight, seen)
        d = self.birth.drive(src) + self.learned.drive(src)
        e = d - self.usual
        if self.adapt:
            self.usual += e / self.c['tau_usual']
        if ret is not None and self.c.get('concept_ret_adds'):
            on = topk_on(e, self.k) | (ret > 0)            # a return lights its own unit beside what is heard (as in
        else:                                              # the sensory areas), instead of taking a heard unit's place
            if ret is not None:
                e = e + self.c['g_ret'] * self.win * ret   # a return lights its own concept unit
            on = topk_on(e, self.k)
        if self.adapt and on.any():
            self.win += (e[on].mean() - self.win) / self.c['tau_usual']
        self.src.set(src)
        self.pop.set(on)

    def learn(self):
        """同时亮的互相加强: feature A -> concept B grows while both are on (A on now or just before)"""
        if getattr(self, 'E', None) is not None:         # rule b: the graded check, shared by the features lit now
            if self._chk is not None and self._chk[0].numel():
                a, dd = self._chk
                d = (0.5 * self.c.get('concept_step', 0.15) / a.numel()) * dd[None, :]
                self.E[a] = (self.E[a] + d).clamp_min(0.0)
                self.I[a] = (self.I[a] - d).clamp_min(0.0)
            return
        L, fw, eta = self.learned, self.c['fall_win'], self.c.get('eta_concept', 0.0)
        if not eta:                                      # birth links only (config eta_concept, 11-10)
            return
        a = torch.nonzero(self.src.on | self.src.held(fw)).squeeze(1)
        if a.numel() == 0:
            return
        f = torch.cat([self.pop.on, torch.zeros(1, dtype=torch.bool, device=DEV)])
        L.apply(a, eta * L.plast(L.w[a]) * f[L.idx[a]].float())


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
    lay = c.get('concept_from')                           # which layers of the sensory areas it reads (None: all)
    src_idx = None
    if lay is not None:                                   # entries: a number = that layer of the SOUND area; or
        want = {('sound', int(x)) if isinstance(x, (int, float)) else (str(x[0]), int(x[1])) for x in lay}   # (area, layer)
        assert all(k in ('sound', 'touch', 'sight') for k, _ in want), f'concept_from: unknown area in {lay}'
        src_idx = torch.cat([torch.arange(a, bb, device=DEV) for kind, l, a, bb in ch.segs if (kind, l) in want])
        n_src = len(src_idx)
    cc = ConceptArea(n_src, c, rng)
    cc.src_idx = src_idx
    if c.get('concept_rule') == 'b':
        nb = 0 if c.get('sight_area') else int(c.get('sight_bits', 0))   # a visual area: sight comes as its features
        cc.setup_b(nb, rng)
        if nb:                                            # each unit's sight drive when nothing is seen
            cc.v_rest = cc.vbirth.drive(torch.as_tensor(life.sight_of(None), device=DEV))
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
    for tb in getattr(pfc, 'P', {}).values():             # the other layers' next-tick tables (predict_from)
        _extend_links(tb, radius_lists(ch.pos, ch.pos, c['r_learn']), n_new, n_old)
    if c.get('pfc_from_concept'):                         # sensory copies reach the PFC only via the concept area
        m = torch.zeros(n_new, dtype=torch.bool, device=DEV)
        for kind, l, a, bb in ch.segs:
            if kind in ('concept', 'force', 'stop'):
                m[a:bb] = True
        b.pfc_in = pfc.pfc_in = m
    if c.get('pfc_graded'):                               # the graded next-tick table among the copies the PFC hears
        from .net import GradedTable
        # the echo of a thought must come back exactly one tick later for the check to skip it (PFC.learn)
        assert c.get('concept_sync'), 'pfc_graded needs concept_sync (the echo of a thought would come 2 ticks later)'
        gm = getattr(pfc, 'pfc_in', None)
        if gm is None:
            gm = torch.zeros(n_new, dtype=torch.bool, device=DEV)
            gm[n_old:n_new] = True
        pfc.GT = GradedTable(torch.nonzero(gm).squeeze(1), n_new, c)
        pfc.gt_p = torch.zeros(n_new, device=DEV)
        pfc.gt_used = pfc.gt_p
        pfc.x_int_prev = torch.zeros(n_new, dtype=torch.bool, device=DEV)
    if c.get('predict_from_concept'):                     # the first layer's next-tick links start at concept copies
        pfc.pred_mask = torch.zeros(n_new, dtype=torch.bool, device=DEV)
        pfc.pred_mask[n_old:n_new] = True
    for pp in pfc.pops():
        _extend_pop(pp, n_new)
    pfc.n = n_new
    pfc.usual = torch.cat([pfc.usual, torch.zeros(cc.n, device=DEV)])
    if getattr(pfc, "usual4", None) is not None:
        pfc.usual4 = torch.cat([pfc.usual4, torch.zeros(cc.n, device=DEV)])
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
    if getattr(b, 'pfc_in', None) is not None and c.get('hip_from_heard', True):
        # with pfc_from_concept the hippocampus hears only the concept / force / stop copies: its birth fan-in is drawn
        # from them (hip_fan each). Drawn over all copies a unit had about 0.07 lit inputs and its recall was generic
        # (few-hearings test 11-11). Its learned links start from zero with the new fan-in.
        src = torch.nonzero(b.pfc_in).squeeze(1).cpu().numpy()
        fan = int(c.get('hip_fan', 150))
        hip.rec = Fixed(src[rng.integers(0, len(src), (hip.nh, fan))], rng.uniform(0.5, 1.5, (hip.nh, fan)))
        for tb in hip.plastic():
            tb.w.zero_()
        hip.usual.zero_()
        hip.usual_s = None
    b.concept, b.n1 = cc, n_new
    ch.ret_concept = torch.zeros(cc.n, device=DEV)
    log(f'concept area added: {cc.n} units ({cc.k} on per tick) reading {n_src} feature units; PFC {n_old} -> {n_new}')
    # the new PFC units' own thresholds, from the drive they receive over a short stretch of life (ARCH 二.4)
    L1s, L2s, Ss = [], [], []
    for _ in range(cal_ticks):
        life.tick()
        L1s.append(pfc.L1.on.clone())
        L2s.append(pfc.L2.on.clone())
        Ss.append(cc.src.on.clone())
    if getattr(cc, 'E', None) is not None:
        # rule b compares "lit" with the push in units of a usual winner's drive: each unit's usual birth drive and
        # that scale are set from this stretch, so that the check does not start from a wrong level
        D = cc.birth.drive_batch(torch.stack(Ss))
        cc.usual = D.mean(0)
        cc.win = torch.topk(D - cc.usual[None, :], cc.k, dim=1).values.mean()
        del D
    X1, X2 = torch.stack(L1s), torch.stack(L2s)
    rho = c.get("pfc_mid_rate") or b.birth_density          # the middle layers' own rate, as at birth
    new2 = Fixed(pfc.ff[0].idx[n_old:], pfc.ff[0].w[n_old:])
    D2 = new2.drive_batch(X1)
    pfc.thr2[n_old:] = own_thresholds(D2, rho)
    X2[:, n_old:] = D2 > pfc.thr2[n_old:]
    new3 = Fixed(pfc.ff[1].idx[n_old:], pfc.ff[1].w[n_old:])
    pfc.thr3[n_old:] = own_thresholds(new3.drive_batch(X2), rho)
    log(f'new PFC units calibrated over {cal_ticks} ticks')
    if c.get('speak'):                                    # generation 0: the speech link and the motor pools (11-11)
        from .speak import wire_speech
        wire_speech(life)
    return cc


def sight_as_sense(life):
    """config sight_area: the visual area's feature copies reach the concept area as a second sense, as the sight bits
    did in build 9 (v9, ARCH 三: the concept area takes every sense): what is seen lights its own units (its own
    strongest k, by birth links from the visual copies, above each unit's usual drive while something is seen), and
    the graded check teaches the sound features to light them; the seen units teach but are not sent on
    (sight_to_code False). Mixed into the sound features (concept_from ('sight', 0)) the copies only joined the sound's
    one top-k and taught nothing: life_A's concept code stayed at full10's level (a linear reader names the syllable
    from it 65 % from 3000 to 11000 dialogues; v9 83 % at 1000, 95 % at 12000). Grows into a living brain: the
    sound rows of the learned E / I links are kept, the visual rows dropped; birth links on visual copies get 0."""
    b, c = life.brain, life.c
    cc, ch = b.concept, b.chains
    assert getattr(cc, 'E', None) is not None and cc.src_idx is not None, 'sight_as_sense needs rule b and concept_from'
    rng = np.random.default_rng(b.seed + 901)
    vis = torch.zeros(ch.n, dtype=torch.bool, device=DEV)
    for kind, l, a, bb in ch.segs:
        if kind == 'sight':
            vis[a:bb] = True
    is_v = vis[cc.src_idx]
    keep, vrow = torch.nonzero(~is_v).squeeze(1), torch.nonzero(is_v).squeeze(1)
    assert vrow.numel(), 'sight_as_sense: the concept area reads no visual copies'
    cc.sight_x, cc.src_idx = cc.src_idx[vrow], cc.src_idx[keep]
    cc.E, cc.I = cc.E[keep].contiguous(), cc.I[keep].contiguous()
    pos = torch.full((cc.n_src,), -1, dtype=torch.int64, device=DEV)
    pos[keep] = torch.arange(keep.numel(), device=DEV)
    bi = pos[cc.birth.idx]
    cc.birth = Fixed(bi.clamp_min(0), torch.where(bi >= 0, cc.birth.w, torch.zeros_like(cc.birth.w)))
    cc.n_src = keep.numel()
    cc.src = Pop(cc.n_src)
    K = c['rec_fan']
    cc.vbirth = Fixed(rng.integers(0, vrow.numel(), (cc.n, K)), rng.uniform(0.5, 1.5, (cc.n, K)))
    cc.v_rest, cc.v_n, cc.v_usual = None, 0, None
    from .vision import pairs
    b.sight_blank = torch.as_tensor(pairs(life.sight_of(None)) if c.get('sight_area') else life.sight_of(None),
                                    device=DEV)
    c['sight_to_code'] = cc.c['sight_to_code'] = False
    log(f'sight as a second sense of the concept area: {vrow.numel()} visual copies light their own {cc.k} units; '
        f'the sound reads {cc.n_src} feature copies')


def gt_drop_motor(life):
    """The graded next-copy table keeps only the copies that are heard or felt (concept, amygdala ...), not the
    force / stop copies (11-12 night, full10 frozen: the table had learned the child's own babble, force copies
    following force copies; as intents skip the thought budget, 57 of them lit every tick, the same whatever was
    heard, and speech said them). What to say then comes from what is thought, through the speech link (ARCH 五:
    想到回答，就会自动说出来). Learned values among the kept copies are kept."""
    b = life.brain
    pfc = b.pfc
    GT = getattr(pfc, 'GT', None)
    if GT is None:
        return 0
    motor = torch.zeros(b.n1, dtype=torch.bool, device=DEV)
    for kind, l, a, bb in b.chains.segs:
        if kind in ('force', 'stop'):
            motor[a:bb] = True
    keep = ~motor[GT.sel]
    k = torch.nonzero(keep).squeeze(1)
    GT.E, GT.I = GT.E[k][:, k].contiguous(), GT.I[k][:, k].contiguous()
    n_drop = int((~keep).sum())
    GT.sel = GT.sel[k]
    life.registry = getattr(life, 'registry', []) + [('graded table without motor copies', int(len(k)), n_drop)]
    log(f'graded table: {n_drop} motor copies dropped, {len(k)} kept')
    return n_drop
