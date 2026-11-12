"""The brain (ARCHITECTURE.md 三): sense areas -> 1:1 chains -> PFC and hippocampus -> 1:1 returns; motor area.
Brain.step(a0, r) is the only entry point: the input-layer bits and the innate value signal r."""
import collections
import math

import numpy as np
import torch

from .net import DEV, Fixed, Links, Pop, local_net, own_thresholds, radius_lists, random_lists, topk_on


# ============================================================================================ sense areas
class SenseArea:
    """ARCH 三 感觉区: an input layer of thermometer pairs on a (channel, level position) grid, then local layers made
    by the one generator (二.4), fewer lit deeper (二.5). Layer 1 reads the input of this tick, layer l the layer
    below one tick earlier. A return lights its unit (g_ret x the unit's own threshold)."""

    def __init__(self, NC, Z, rates, c, rng):
        self.NC, self.Z, self.D, self.c = NC, Z, c['upp'], c
        pre = np.array([(0, y, z) for y in range(NC) for z in range(Z)])
        self.grid, self.layers = [], []
        for l in range(len(rates)):
            post = np.array([(0, y, z) for y in range(NC) for z in range(Z) for _ in range(self.D)])
            dz = c['dz'][l] * (c['s'] if l > 0 else 1)
            self.layers.append(Fixed(*local_net(pre, post, c['K'], c['dy'][l], dz, rng)))
            self.grid.append(post)
            pre = post
        self.rates = rates
        self.n = [len(g) for g in self.grid]
        self.state = [torch.zeros(n, dtype=torch.bool, device=DEV) for n in self.n]

    def calibrate(self, A):
        """A: bool [T, n_in], the birth period. Every unit's own threshold (二.4). Returns the layer states [T, n_l]."""
        states, x = [], A
        self.thr, self.ret_w = [], []
        for ly, r in zip(self.layers, self.rates):
            D = ly.drive_batch(x)
            thr = own_thresholds(D, r)
            self.thr.append(thr)
            self.ret_w.append(self.c['g_ret'] * torch.maximum(thr, ly.w.min(1).values))
            x = D > thr
            states.append(x)
        return states

    def step(self, a_in, ret):
        prev = [a_in] + self.state[:-1]
        self.state = [ly.drive(p) + r * rw > th
                      for ly, p, r, rw, th in zip(self.layers, prev, ret, self.ret_w, self.thr)]
        return self.state


# ============================================================================================ chains
class Chains:
    """ARCH 三 链式神经元: every feature unit of every layer has one copy in PFC L1 and one in the hippocampus, and a
    return to itself only. Feature units = birth firing rate inside the band. Force and stop neurons are all chained;
    blockers never. Copies keep their source's grid (block per area, the source's (y, z)); the sheet position (both
    grid coordinates scaled to [0, 1]) is where learned links look for near partners."""

    def __init__(self, sound, s_rates, touch, t_rates, motor, band, sight=None, v_rates=()):
        lo, hi = band
        self.sound_idx = [torch.nonzero((r >= lo) & (r <= hi)).squeeze(1) for r in s_rates]
        self.touch_idx = [torch.nonzero((r >= lo) & (r <= hi)).squeeze(1) for r in t_rates]
        self.sight_idx = [torch.nonzero((r >= lo) & (r <= hi)).squeeze(1) for r in v_rates]   # config sight_area
        segs, grid, pos = [], [], []
        a = 0

        def add(kind, l, n, g, Y, Z):
            nonlocal a
            segs.append((kind, l, a, a + n))
            grid.append(g)
            pos.append(np.stack([g[:, 1] / max(1, Y - 1), g[:, 2] / max(1, Z - 1)], 1))
            a += n
        for l, ix in enumerate(self.sound_idx):
            add('sound', l, len(ix), sound.grid[l][ix.cpu().numpy()], sound.NC, sound.Z)
        for l, ix in enumerate(self.touch_idx):
            g = touch.grid[l][ix.cpu().numpy()].copy()
            g[:, 0] = 1
            add('touch', l, len(ix), g, touch.NC, touch.Z)
        for l, ix in enumerate(self.sight_idx):
            g = sight.grid[l][ix.cpu().numpy()].copy()
            g[:, 0] = 5
            add('sight', l, len(ix), g, sight.NC, sight.Z)
        u = np.arange(motor.nf)
        add('force', 0, motor.nf, np.stack([np.full_like(u, 2), u // motor.m, u % motor.m], 1), motor.nmus, motor.m)
        u = np.arange(motor.ns)
        add('stop', 0, motor.ns, np.stack([np.full_like(u, 3), u, np.zeros_like(u)], 1), motor.ns, 2)
        self.segs, self.n = segs, a
        self.grid = np.concatenate(grid)
        jit = np.random.default_rng(7).normal(0, 0.005, (a, 2))
        self.pos = torch.as_tensor(np.clip(np.concatenate(pos) + jit, 0, 1), dtype=torch.float32, device=DEV)

    def seg(self, kind):
        return [s for s in self.segs if s[0] == kind]

    def gather(self, sound, touch, force, stop, concept=None, sight=None):
        return torch.cat([sound[l][ix] for l, ix in enumerate(self.sound_idx)] +
                         [touch[l][ix] for l, ix in enumerate(self.touch_idx)] +
                         ([] if sight is None else [sight[l][ix] for l, ix in enumerate(self.sight_idx)]) +
                         [force, stop] + ([] if concept is None else [concept]))

    def scatter(self, R, sound_n, touch_n):
        """returns R (bool [n]) to its own sources"""
        rs = [torch.zeros(n, device=DEV) for n in sound_n]
        rt = [torch.zeros(n, device=DEV) for n in touch_n]
        if getattr(self, 'sight_n', None):
            self.ret_sight = [torch.zeros(n, device=DEV) for n in self.sight_n]
        for kind, l, a, b in self.segs:
            if kind == 'sound':
                rs[l][self.sound_idx[l]] = R[a:b].float()
            elif kind == 'touch':
                rt[l][self.touch_idx[l]] = R[a:b].float()
            elif kind == 'sight':                         # config sight_area: kept here (the call returns four)
                self.ret_sight[l][self.sight_idx[l]] = R[a:b].float()
            elif kind == 'force':
                rf = R[a:b].float()
            elif kind == 'stop':
                rp = R[a:b]
            elif kind == 'amyg':                          # the amygdala (core/amygdala.py): a thought of it
                self.ret_amyg = R[a:b].float()
            else:
                self.ret_concept = R[a:b].float()
        return rs, rt, rf, rp


# ============================================================================================ PFC
def reciprocal_loop(ff, n1, fan, rows=None):
    """ARCH 三 前额叶 birth loop: each last-layer unit returns to the `fan` first-layer copies with the largest path
    weight into it (back through the middle layers, however many); rows: only these last-layer units"""
    out = []
    iL, wL = (ff[-1].idx, ff[-1].w) if rows is None else (ff[-1].idx[rows], ff[-1].w[rows])
    for a in range(0, iL.shape[0], 256):
        idx, p = iL[a:a + 256], wL[a:a + 256]
        for f in reversed(ff[:-1]):
            p = p[..., None] * f.w[idx]
            idx = f.idx[idx]
        dense = torch.zeros((idx.shape[0], n1), device=DEV)
        dense.scatter_add_(1, idx.reshape(idx.shape[0], -1), p.reshape(idx.shape[0], -1))
        v, i = torch.topk(dense, fan, dim=1)
        out.append(torch.where(v != 0, i, torch.full_like(i, n1)))  # no path at all: no link (a shallow PFC has few)
    return torch.cat(out)


class PFC:
    """ARCH 三 前额叶. L1 = the copies. L2, L3, L4 = one unit per L1 position, made by the one generator (二.4),
    fixed, each unit's own threshold at L1's birth density. L4 (the output) lights exactly as many units as L1 (the
    input) (六). L4(t) -> L1(t+1): the birth loop (holds a thought) and learned links to near copies (what comes
    next); L4 <-> L4 learned links (related units excite each other), kept weak (六)."""

    def __init__(self, ch, c, rng):
        n = ch.n
        self.c, self.n = c, n
        # layers: L1 (the copies), up to two middle layers (L2, L3), the last layer (L4). The layers only put time
        # between input and output, one tick each (designer 11-09: fewer may do; pfc_depth 2 = L1 -> L4)
        self.depth = int(c.get('pfc_depth', 4))
        self.ff = [Fixed(*local_net(ch.grid, ch.grid, c['K'], c['dy'][0], c['dz'][0], rng))
                   for _ in range(self.depth - 1)]
        li = reciprocal_loop(self.ff, n, c['loop_fan'])
        lw = torch.as_tensor(rng.uniform(0.5, 1.5, tuple(li.shape)), dtype=torch.float32, device=DEV) * (li < n)
        self.loop = Links(li, c['g_loop'] * lw / lw.sum(1, keepdim=True).clamp_min(1e-9), n, c, learned=False, name='loop')
        self.L41 = Links(radius_lists(ch.pos, ch.pos, c['r_learn']), 0.0, n, c, gain=c['g31'], gamma=c['gamma_pfc'],
                         name='L4->L1')
        self.L44 = Links(radius_lists(ch.pos, ch.pos, c['r_learn'], True), 0.0, n, c, gain=c['g44'],
                         gamma=c['gamma_pfc'], name='L4<->L4')
        self.G1P = None                                    # generation-0 speech link (gen0.py)
        # the other layers also learn what comes next in the first layer (11-10, opt-in): each layer is a tick later
        # than the one before, so the first layer's own copies say what was heard just now, the last layer what was
        # heard four or more syllables ago (combined); the push into a copy is the sum, checked by the copy as always
        self.P = {nm: Links(radius_lists(ch.pos, ch.pos, c['r_learn']), 0.0, n, c, gain=c['g31'], gamma=c['gamma_pfc'],
                            name=f'{nm}->L1') for nm in c.get('predict_from', ('L4',)) if nm != 'L4'}
        self.RR = None                                     # the range layer's own links (setup_range), off by default
        self.L1, self.L1ext, self.L2, self.L3, self.L4 = (Pop(n) for _ in range(5))
        self.x_int = torch.zeros(n, dtype=torch.bool, device=DEV)
        self.usual = torch.zeros(n, device=DEV)            # each copy's usual internal drive (its own threshold level)
        self.adapt = False                                 # the usual level follows the drive while the brain lives
        self.force_int = None                              # probes only: copies lit as if thought

    def plastic(self):
        return [self.L41, self.L44] + list(getattr(self, 'P', {}).values()) + \
            ([self.G1P] if self.G1P is not None else []) + ([self.GT] if getattr(self, 'GT', None) is not None else [])

    def calibrate(self, X):
        """X: bool [T, n] the birth chain input (loop and learning off)"""
        rho = float(X.float().mean())
        self.ext_usual = float(X.float().sum(1).mean())     # the usual external count (the thought budget's base)
        mid = self.c.get('pfc_mid_rate') or rho              # middle layers' own rate (ARCH 二.5: free)
        S, thr = X, []
        for f in self.ff[:-1]:                             # the middle layers
            D = f.drive_batch(S)
            thr.append(own_thresholds(D, mid))
            S = D > thr[-1]
            del D
        self.thr2, self.thr3 = (thr + [None, None])[:2]
        D4 = self.ff[-1].drive_batch(S)
        # how much the input push into the last layer differs between its units (each unit's own mean taken out):
        # the most the last layer's own links may add to one unit, so that what is heard keeps setting its state
        # (designer 11-04: links capped at a value; the value is the input's own spread, not tuned)
        self.L44.cap = float((D4 - D4.mean(0)).std(1).mean())
        # the checked links share each copy's change among the lit last-layer units linked to it, so one link is small;
        # their gain is set so a link has the size of the others (above the prune level): 1 / the usual number of lit
        # last-layer units linked to a copy (in-degree x the last layer's share lit, which equals the input's)
        indeg = torch.bincount(self.L41.idx.reshape(-1), minlength=self.n + 1)[:self.n].float()
        self.L41.gain = self.c['g31'] / max(1.0, float(indeg.mean()) * rho)
        for tb in getattr(self, 'P', {}).values():         # the same size of link for the other layers' tables
            tb.gain = self.L41.gain
        return rho

    def setup_range(self, X, rng):
        """The range (designer ideas 15, 23 and 11-02: excitability = context length; how fast it fades sets how much
        of what came before is held). The last middle layer lights `pfc_range` x its usual count. Units rising together
        (a word entering) grow links among themselves at once (the one rule), so they keep each other lit; the links fade
        every tick (the one decay, at its own fast rate), so a word leaves the range after a while. X: bool [T, n] what
        the copies usually receive. Tested 11-10 (docs/LOGIC.md 8): the context it gives grows from 5-6 to 15-21 ticks,
        but as it takes the place of a combining layer, the logic language is learned worse; so it stays off."""
        c, n = self.c, self.n
        mids = self.depth - 2
        S = X
        for f, thr in list(zip(self.ff, (self.thr2, self.thr3)))[:mids - 1]:
            S = f.drive_batch(S) > thr
        D = self.ff[mids - 1].drive_batch(S)
        if c.get('range_rate'):                            # the range layer's own sparseness (a word lights few units
            thr = own_thresholds(D, c['range_rate'])       # of it, so several fit; designer 11-10: like the
            if mids == 2:                                  # hippocampus, sparser)
                self.thr3 = thr
            else:
                self.thr2 = thr
        thr = (self.thr2, self.thr3)[mids - 1]
        usual = float((D > thr).float().sum(1).mean())
        self.k_range = c['pfc_range'] * usual
        spread = float((D - D.mean(0)).std(1).mean())
        del D
        w = c.get('range_rw', 300)
        self.RR = Links(random_lists(n, n, w, rng, True), 0.0, n, c, gain=1.0, gamma=c.get('range_fade', 0.8),
                        name='range')
        # a word's units entering together get about w x (its share lit) partners, each link ~ the step after one rise;
        # together they push like `range_rr` x the spread of the input push
        partners = w * usual / n
        self.RR.gain = c.get('range_rr', 2.0) * spread / max(1e-6, c.get('range_eta', 0.3) * partners)

    def pred_src(self, nm, on):
        """the units of layer nm whose next-tick links take part: in the first layer, with predict_mask set (the
        concept copies, config predict_from_concept), only those (11-10: copies of local sound features are shared by
        many syllables; as predictors they drown the concept copies, which each stand for few)"""
        m = getattr(self, 'pred_mask', None)
        return on & m if (nm == 'L1' and m is not None) else on

    def push(self):
        """the PFC's push into the first-layer copies for the coming tick (each layer as it is now)"""
        a4 = self.L4.on
        d1 = self.loop.drive(a4) + self.L41.drive(a4)
        for nm, tb in getattr(self, 'P', {}).items():      # the other layers' push (each as it was a tick ago)
            d1 = d1 + tb.drive(self.pred_src(nm, getattr(self, nm).on))
        if self.G1P is not None:
            d1 = d1 + self.G1P.drive(a4)
        G1C = getattr(self, 'G1C', None)
        if G1C is not None:                                # config speak: thought concept copies -> force copies
            if getattr(G1C, 'tendency', False) and getattr(self, 'e1', None) is not None:
                d1 = d1 + G1C.drive(self.e1)               # core/speak2.py: the tendency (the push above usual)
            else:
                d1 = d1 + G1C.drive(self.x_int)
        GT = getattr(self, 'GT', None)
        if GT is not None:       # the graded table (config pfc_graded): its push is a share 0..1; a copy that always
            self.gt_p = GT.drive(self.L1.on)       # comes after what is lit now is pushed past theta + margin
            d1 = d1 + self.c.get('graded_gain', self.c['theta1'] + self.c['check_margin']) * self.gt_p
        return d1

    def step(self, x_ext):
        a4 = self.L4.on
        d1 = self.push()
        if getattr(self, 'GT', None) is not None:
            self.gt_used = self.gt_p                       # what the graded table pushed for this tick (for its check)
            self.x_int_prev = self.x_int                   # what was thought a tick ago (its echo may come back now)
        # a copy's own threshold follows the drive it usually receives (ARCH 二.4): only drive above its own usual level
        # counts. Excitability = context length (三, 六): at most `context` x the external input count are lit inside.
        e1 = d1 - self.usual
        if self.adapt:                                     # its usual level is the drive it gets while not lit from
            rest = ~x_ext.bool()                           # outside, so a copy lit most of the time can still be
            self.usual += rest * (d1 - self.usual) / self.c['tau_usual']   # pushed firmly and its links stop growing
        n_ext = float(x_ext.sum())
        if getattr(self, 'ext_usual', None) is None:
            self.ext_usual = 0.026 * x_ext.shape[0]
        if self.adapt:
            self.ext_usual += (n_ext - self.ext_usual) / self.c['tau_usual']
        # how many copies may be lit from inside: context x the usual external count (not this tick's), so what is
        # recalled in a quiet moment can still light up (designer 11-04: A heard, then quiet, B is thought of)
        self.budget = self.c['context'] * self.ext_usual
        # a copy already lit from outside gains nothing from being thought: the thought budget goes to the others
        cand = e1.masked_fill(x_ext.bool(), -1e9)
        if getattr(self, 'pfc_in', None) is not None:      # copies that never come from outside are not thought
            cand = cand.masked_fill(~self.pfc_in, -1e9)
        x_int = (e1 > self.c['theta1']) & topk_on(cand, self.budget)
        mm = getattr(self, 'motor_mask', None)
        if mm is not None:                                 # config speak: an intent to move (a force copy) is not a
            x_int = x_int | ((e1 > self.c['theta1']) & mm & ~x_ext.bool())   # thought: no budget; whether it moves is a
        self.e1 = e1                                       # competition in the motor area (ARCH 三 说不说是一场竞争)                                       # kept for probes (graded internal drive)
        if self.force_int is not None:
            x_int = x_int | self.force_int
        # each layer reads the one before as it was a tick ago (one tick per layer)
        mids = [self.L2, self.L3][:getattr(self, 'depth', 4) - 2]
        src, new = self.L1.on, []
        RR = getattr(self, 'RR', None)
        for i, (f, thr, P) in enumerate(zip(self.ff, (self.thr2, self.thr3), mids)):
            if RR is not None and i == len(mids) - 1:    # the range: its input above its own threshold + its own links
                e = f.drive(src) - thr + RR.drive(P.on)
                new.append((e > 0) & topk_on(e, self.k_range))
            else:
                new.append(f.drive(src) > thr)
            src = P.on
        if RR is not None:
            RR.decay()                                     # the range's links fade every tick
        # the last layer: each unit's own usual level too (二.4); the most driven above it, as many as the input
        d4 = self.ff[len(mids)].drive(src) + self.L44.drive(a4)
        if getattr(self, 'usual4', None) is None:
            self.usual4 = torch.zeros_like(d4)
        e4 = d4 - self.usual4
        if self.adapt:
            self.usual4 += e4 / self.c['tau_usual']
        k4 = self.L1.on.sum()                              # as many as the input layer lights
        if self.c.get('l4_usual'):                       # ... or as its usual external count, if more: in a quiet
            k4 = max(float(k4), self.ext_usual)            # moment the last layer still thinks with its usual size
        n4 = topk_on(e4, k4)                               # (like the thought budget, ARCH 六; designer rule 11-04)
        self.L1ext.set(x_ext)
        self.L1.set(x_ext | x_int)
        for P, nw in zip(mids, new):
            P.set(nw)
        self.L4.set(n4)
        self.x_int = x_int
        return x_int

    def learn(self):
        fw, eta = self.c['fall_win'], self.c['eta']
        # last layer -> first layer: each copy checks what the last layer did to it (ARCH 四, designer 11-07)
        P = getattr(self, 'P', {})
        step = self.c['check_step'] / (1 + len(P))         # one copy's change per tick, shared by the tables
        self.L41.learn_check(self.L4.prev, self.L1ext.on, self.e1, self.c['theta1'], self.c['check_margin'], step)
        for nm, tb in P.items():
            tb.learn_check(self.pred_src(nm, getattr(self, nm).prev), self.L1ext.on, self.e1, self.c['theta1'],
                           self.c['check_margin'], step)
        GT = getattr(self, 'GT', None)
        if GT is not None:
            # each copy checks the graded table's push against what came. A copy the PFC itself thought a tick ago
            # comes back through its return as if from outside (an echo): it takes no part, else every thought would
            # confirm itself (THEORY_fable 2.4). What the hippocampus recalls does count: that is how the cortex learns it.
            pre, came, used = self.L1.prev, self.L1ext.on, self.gt_used
            G1C = getattr(self, 'G1C', None)
            if G1C is not None and self.c.get('mirror'):
                # config mirror (designer 11-11: the motor area learns regular speech from the dialogues): a syllable
                # heard lights, silently, the intent copies of saying it, for this check only (never sent to the
                # muscles): the table learns "after the movements of A come the movements of B" from what it hears
                xc = self.L1ext.on[G1C.units].float()
                ov = (G1C.code @ xc) / G1C.size
                i = int(ov.argmax())
                mir = torch.zeros_like(came)
                if float(ov[i]) >= self.c.get('mirror_min', 0.5):
                    mir[G1C.f0:G1C.f1] = G1C.copies[i]
                mp = getattr(self, 'mirror_prev', None)
                if mp is not None and bool(mp.any()):
                    add = mp & ~pre
                    used = used + GT.drive(add)
                    pre = pre | mp
                came = came | mir
                self.mirror_prev = mir
            GT.learn(pre, came, used, self.c.get('graded_step', 0.15), skip=self.x_int_prev)
        if self.G1P is not None:                           # the hand-wired generation-0 speech link keeps the timing
            onset = self.L1ext.rose() & ~self.L1.prev      # rule, so thinking a sound without saying it does not
            self.G1P.learn(self.L4, onset, self.L1ext.held(fw), eta)   # unlearn it (ARCH 六: kept at practised strength)
        self.L44.learn(self.L4, self.L4.rose(), self.L4.held(fw), eta)
        if getattr(self, 'RR', None) is not None:          # a word entering the range: its units link at once
            P = (self.L2, self.L3)[self.depth - 3]
            self.RR.learn(P, P.rose(), P.held(fw), self.c.get('range_eta', 0.3))

    def pops(self):
        return [self.L1, self.L1ext, self.L2, self.L3, self.L4]


# ============================================================================================ hippocampus
class Hippocampus:
    """ARCH 三 海马体: copies H1 -> its own population Hs (nh x s units: random birth input, no birth recurrence, a
    fixed count kh per tick, so it gets sparser as the brain grows; partners per unit stay fixed) -> learned Hs <-> Hs and Hs -> H1 links that grow and decay fast. Recall lights H1 copies; their
    returns light the source units, which reach the PFC through the chains."""

    def __init__(self, n1, c, rng):
        s = c['s']
        nh = c['nh'] * s
        self.c, self.nh = c, nh
        self.rec = Fixed(rng.integers(0, n1, (nh, c['rec_fan'])), rng.uniform(0.5, 1.5, (nh, c['rec_fan'])))
        self.HH = Links(random_lists(nh, nh, c['n_hh'], rng, True), 0.0, nh, c, gain=c['g_hh'],
                        gamma=c['gamma_hip'], name='Hs<->Hs')     # internal links: the cortex's rule
        self.H1t = Links(random_lists(nh, n1, c['n_h1'], rng), 0.0, n1, c,
                         gamma=c['gamma_hip'], name='Hs->H1', plain=True)
        self.H1, self.H1ext, self.Hs = Pop(n1), Pop(n1), Pop(nh)
        self.h_int = torch.zeros(n1, dtype=torch.bool, device=DEV)
        self.usual = torch.zeros(n1, device=DEV)
        self.adapt = False

    def plastic(self):
        return [self.HH, self.H1t]

    def step(self, x_ext, budget=None):
        ah = self.Hs.on
        e = self.H1t.drive(ah) - self.usual                       # above each copy's own usual recall drive (二.4)
        if self.adapt:
            self.usual += e / self.c['tau_usual']
        h_int = topk_on(e, self.c['context'] * x_ext.sum() if budget is None else budget)   # the most strongly recalled
        ds = self.rec.drive(self.H1.on) + self.HH.drive(ah)      # its own units: above their own usual level (二.4)
        if getattr(self, 'usual_s', None) is None:
            self.usual_s = torch.zeros_like(ds)
        es = ds - self.usual_s
        if self.adapt:
            self.usual_s += es / self.c['tau_usual']
        hs = topk_on(es, self.c['kh'])
        self.H1ext.set(x_ext)
        self.H1.set(x_ext | h_int)
        self.Hs.set(hs)
        self.h_int = h_int
        return h_int

    def learn(self):
        fw, eta = self.c['fall_win'], self.c['eta_h']
        self.HH.learn(self.Hs, self.Hs.rose(), self.Hs.held(fw), eta)
        self.H1t.learn(self.Hs, self.H1ext.rose() & ~self.H1.prev, self.H1ext.held(fw), eta)

    def pops(self):
        return [self.H1, self.H1ext, self.Hs]


# ============================================================================================ motor
class Motor:
    """ARCH 三 运动区. m force neurons per muscle (graded thresholds; more lit = more force); blockers: half tonic
    (default: don't move), half recruited by the summed command (a diffuse command meets more blocking); stop neurons
    (lit by the brake from the input layer and by their returns). A force neuron's only excitation is its return:
        fires iff  g_ret_force x (threshold + tonic blocking) [if its copy is lit] + pools - blocking - stops > threshold.
    Collapse pools (generation 0, gen0.py): per syllable an excitatory and an inhibitory group fed by the lit returns.
    A muscle moves when at least half of its force neurons fire."""

    def __init__(self, c, brake_lines):
        s = c['s']
        m = c['m'] * s
        self.c, self.m = c, m
        self.nmus = int(c.get('n_mus', 24))               # channels: 24 muscles, or 72 (24 x onset / nucleus / offset
        # all in one tick, config speak with syl_ticks 1: the mouth gives a whole syllable per tick as the ear hears one)
        self.nf, self.nb, self.ns = self.nmus * m, 8 * m, 8 * m
        k = torch.arange(self.nf, device=DEV) % m
        self.muscle_of = torch.arange(self.nf, device=DEV) // m
        self.thr_f = c['theta_m'] * (1.0 + k.float() / m)
        self.n_tonic = 4 * m
        self.ret_drive = c['g_ret_force'] * (self.thr_f + c['b_block'] * self.n_tonic / (8.0 * m))
        sharp = float(self.ret_drive.view(self.nmus, m)[:int(c.get('mus_per_syl', 4))].sum())   # one syllable's channels
        self.ladder = sharp * (1.0 + (torch.arange(4 * m, device=DEV) + 1.0) / (4 * m))
        g = np.random.default_rng(c['lineage_seed'] + 900)
        self.thr_s = torch.as_tensor(c['theta_stop'] * g.uniform(0.8, 1.2, self.ns), dtype=torch.float32,
                                     device=DEV)
        self.brake_lines = torch.as_tensor(brake_lines, device=DEV)
        self.n_brake = c['n_brake'] * s
        self.force, self.stop, self.lit = Pop(self.nf), Pop(self.ns), Pop(self.nf)
        self.pools = None

    def brake(self, a0):
        """the innate brake: the input layer's loud-voice lines -> stop neurons (only makes movement stop)"""
        return (a0[self.brake_lines].float().sum() / self.n_brake) >= self.thr_s

    def plastic(self):
        return [] if self.pools is None else self.pools['tables']

    def step(self, a0, ret_force, ret_stop):
        c, m = self.c, self.m
        self.braked = self.brake(a0)                       # the stop neurons the innate brake lights now
        stop = self.braked | ret_stop
        lit = ret_force > 0
        st, cp = getattr(self, 'settle', None), getattr(self, 'cpools', None)
        d = lit.float() * (self.ret_drive if st is None and cp is None else self.ret_prep)
        P = d.sum()
        nblock = self.n_tonic + (P >= self.ladder).sum()
        inh = c['b_block'] * nblock / (8.0 * m) + c['c_stop'] * stop.float().sum() / (8.0 * m)
        if cp is not None:                                 # core/speak3.py: pools choose in concept space
            e1 = getattr(self, 'e1', None)
            w = -1 if e1 is None else cp.settle(e1, float(stop.float().mean()), getattr(self, 'heard_now', None))
            if w >= 0:
                own = cp.P[w] > 0
                d = torch.where(own, d + self.complete, d - self.complete)
        elif st is not None:                               # core/speak2.py: only a winning syllable pool moves muscles
            w = st.settle(lit)
            if w >= 0:
                own = st.P[w] > 0
                d = torch.where(own, d + self.complete, d - self.complete)
        elif self.pools is not None:
            pe, pi = self.pool_fire(lit)
            d = d + self.pools['exc'].drive(pe) - self.pools['inh'].drive(pi)
            self.pools['pe'].set(pe)
            self.pools['pi'].set(pi)
        force = d - inh > self.thr_f
        self.lit.set(lit)
        self.force.set(force)
        self.stop.set(stop)
        on = torch.zeros(self.nmus, device=DEV).index_add_(0, self.muscle_of, force.float())
        return on >= math.ceil(m / 2)

    def pool_fire(self, lit):
        """a pool neuron fires when all its sampled inputs are lit (same tick)"""
        p = self.pools
        return [tb.drive(lit) >= 0.875 * tb.drive(torch.ones_like(lit)).clamp_min(1e-9)
                for tb in (p['in_e'], p['in_i'])]

    def learn(self):
        if self.pools is None:
            return
        p, fw, eta = self.pools, self.c['fall_win'], self.c['eta']
        for tb, pool in ((p['in_e'], p['pe']), (p['in_i'], p['pi'])):
            tb.learn(self.lit, pool.rose(), pool.held(fw), eta)
        for tb, pool in ((p['exc'], p['pe']), (p['inh'], p['pi'])):
            tb.learn(pool, self.force.rose(), self.force.held(fw), eta)

    def pops(self):
        out = [self.force, self.stop, self.lit]
        return out + ([self.pools['pe'], self.pools['pi']] if self.pools is not None else [])


# ============================================================================================ the brain
class Brain:
    def __init__(self, c, seed, coch, skin, birth_A0, birth_V=None):
        """birth_A0: bool [T, n0], the birth period heard by the silent newborn (learning off)"""
        self.c, self.seed = c, seed
        self.n_ear = coch.n                                # the input bits: the ear's, then (config sight_bits) the
        self.n0 = coch.n + int(c.get('sight_bits', 0)) * (2 if c.get('sight_area') else 1)   # bits of what is seen
        self.sight_prev = None
        lin = np.random.default_rng(c['lineage_seed'])          # family wiring: the same for every seed
        own = np.random.default_rng(seed + 500)
        self.sound = SenseArea(coch.NC, coch.Z, c['sound_rates'], c, lin)
        self.skin = skin
        self.touch = SenseArea(skin.NC, skin.Z, c['touch_rates'], c, lin)
        self.motor = Motor(c, coch.brake_lines)
        A0 = torch.as_tensor(birth_A0, device=DEV)
        s_states = self.sound.calibrate(A0)
        t_states = self.touch.calibrate(skin.birth_stream(A0.shape[0], seed))
        self.sight, v_states = None, None
        if c.get('sight_area'):                            # a visual area, made like the others (ARCH 二.4, 三 感觉区):
            side = int(round(int(c['sight_bits']) ** 0.5))  # the eye: one thermometer pair per pixel of the sign
            self.sight = SenseArea(side, 2 * side, c['sight_rates'], c, lin)
            v_states = self.sight.calibrate(torch.as_tensor(birth_V, device=DEV))
        # a sound unit's rate is counted while a voice is heard (the innate brake hears one); a unit lit for much of
        # any voice has little feature (ARCH 三 链式神经元)
        voice = torch.stack([self.motor.brake(a).any() for a in A0])
        s_rates = [x[l:][voice[:len(voice) - l]].float().mean(0) for l, x in enumerate(s_states)]
        self.chains = Chains(self.sound, s_rates,
                             self.touch, [x.float().mean(0) for x in t_states], self.motor, c['band'],
                             self.sight, [] if v_states is None else [x.float().mean(0) for x in v_states])
        if self.sight is not None:
            self.chains.sight_n = self.sight.n
            self.ret_sight = [torch.zeros(n, device=DEV) for n in self.sight.n]
            self.chains.ret_sight = self.ret_sight
        del t_states
        self.n1 = self.chains.n
        self.pfc = PFC(self.chains, c, own)
        self.hip = Hippocampus(self.n1, c, own)
        # the birth chain stream (time-parallel): sound states one tick late, touch at rest, stop neurons from the
        # brake, force neurons silent
        T = min(A0.shape[0] - 1, 3000)
        rest = self.rest_touch()
        stop = torch.stack([self.motor.brake(a) for a in A0[-T - 1:-1]])
        X = torch.cat([st[-T - 1:-1, ix] for st, ix in zip(s_states, self.chains.sound_idx)] +
                      [rest[l][ix][None].expand(T, -1) for l, ix in enumerate(self.chains.touch_idx)] +
                      ([] if v_states is None else
                       [st[-T - 1:-1, ix] for st, ix in zip(v_states, self.chains.sight_idx)]) +
                      [torch.zeros((T, self.motor.nf), dtype=torch.bool, device=DEV), stop], 1)
        del v_states
        del s_states
        self.birth_density = self.pfc.calibrate(X)
        del X
        self.ret_sound = [torch.zeros(n, device=DEV) for n in self.sound.n]
        self.ret_touch = [torch.zeros(n, device=DEV) for n in self.touch.n]
        self.ret_force = torch.zeros(self.motor.nf, device=DEV)
        self.ret_stop = torch.zeros(self.motor.ns, dtype=torch.bool, device=DEV)
        self.touch.state = rest
        self.learning = False
        self.t = 0
        self.rlog = collections.deque()                    # reward tool: recent applied changes

    def rest_touch(self, ticks=4):
        """the touch layers when nothing moves"""
        zero = self.skin.code(torch.zeros(24, device=DEV))
        no_ret = [torch.zeros(n, device=DEV) for n in self.touch.n]
        for _ in range(ticks):
            st = self.touch.step(zero, no_ret)
        return [x.clone() for x in st]

    def concept_sent(self):
        """the concept units sent on as heard: those lit from below (sound, sight). A unit lit only by a thought's
        return is that thought itself (its copy is already lit inside); sent on as heard it came back the next tick as
        if heard (11-12 night: lit thoughts at t matched the external concept copies at t+1 exactly), so the PFC,
        the hippocampus and the amygdala could not tell heard from thought and the PFC's checks confirmed its own
        earlier thoughts (Fable a37 #3). config concept_send_echo: the old behaviour."""
        cc = self.concept
        if self.c.get('concept_send_echo', False) or getattr(cc, 'below', None) is None:
            return cc.pop.on
        return cc.below

    def plastic(self):
        cc, am = getattr(self, 'concept', None), getattr(self, 'amyg', None)
        return self.pfc.plastic() + self.hip.plastic() + self.motor.plastic() + ([] if cc is None else [cc.learned]) + \
            ([] if am is None else [am.learned])

    def pops(self):
        cc, am = getattr(self, 'concept', None), getattr(self, 'amyg', None)
        return self.pfc.pops() + self.hip.pops() + self.motor.pops() + ([] if cc is None else [cc.pop, cc.src]) + \
            ([] if am is None else [am.pop])

    def step(self, a0, r=0.0):
        n0 = getattr(self, 'n_ear', self.n0) + int(self.c.get('sight_bits', 0)) * (2 if self.c.get('sight_area') else 1)
        em = getattr(self, 'eye_motion', None)
        n0 += 0 if em is None else self.n_periph
        assert a0.dtype == torch.bool and tuple(a0.shape) == (n0,), 'the brain receives only the input bits'
        if em is not None:                                 # the eye's periphery (core/vision.py): local motion neurons
            per, a0 = a0[-self.n_periph:], a0[:-self.n_periph]   # -> the eye muscles (innate)
            self.eye_muscles = em.step(per[0::2])
        sight = None
        if a0.shape[0] > getattr(self, 'n_ear', self.n0):  # what is seen now reaches the concept area a tick later,
            sight, a0 = a0[self.n_ear:], a0[:self.n_ear]   # together with the sound features of this tick
        sight_was, self.sight_prev = getattr(self, 'sight_prev', None), sight
        ch = self.chains
        cc = getattr(self, 'concept', None)
        stop_on = self.motor.stop.on
        if not self.c.get('chain_brake_stop', True) and getattr(self.motor, 'braked', None) is not None:
            # a stop neuron lit by the innate brake alone says nothing about what is heard (lit all the time others
            # speak): like other always-lit neurons it sends no chain signal (ARCH 三 链式神经元; designer 11-10: test it)
            stop_on = stop_on & ~self.motor.braked
        sa = getattr(self, 'sight', None)
        x_ext = ch.gather(self.sound.state, self.touch.state, self.motor.force.on, stop_on,
                          None if cc is None else self.concept_sent(), None if sa is None else sa.state)   # t-1
        seen = None
        if sa is not None:                                 # the visual area reads what is seen now (read by the
            if cc is not None and getattr(cc, 'sight_x', None) is not None:   # concept area through its copies a
                bl = self.sight_blank                      # tick later, with the sound features of the same tick;
                seen = sight_was is not None and bool((sight_was != bl).any())   # grow.sight_as_sense: as a sense)
            sa.step(sight, ch.ret_sight)
            sight_was = None
        if cc is not None:                                 # the shared concept area reads every sensory feature
            cc.adapt = self.learning
            src = x_ext[:cc.n_src] if getattr(cc, 'src_idx', None) is None else x_ext[cc.src_idx]
            if getattr(cc, 'E', None) is not None:
                sx = getattr(cc, 'sight_x', None)
                cc.step(src, getattr(ch, 'ret_concept', None), sight_was if sx is None else x_ext[sx], seen)
            else:
                cc.step(src, getattr(ch, 'ret_concept', None))
            if self.c.get('concept_sync'):               # its copies reach the PFC with the features it read (11-10)
                x_ext = torch.cat([x_ext[:-cc.n], self.concept_sent()])
        am = getattr(self, 'amyg', None)
        if am is not None:                                 # the amygdala (ARCH 三 杏仁核): the pat now (r is the pat's
            am.adapt = self.learning                       # strength: the world's pleasure as a sensation) and the
            pin = getattr(self, 'pfc_in', None)            # copies the PFC hears; its copies join them this tick
            heard = x_ext if pin is None else x_ext & pin[:x_ext.shape[0]]
            am.step(torch.cat([heard, torch.zeros(am.n, dtype=torch.bool, device=DEV)]), r,
                    getattr(ch, 'ret_amyg', None))
            x_ext = torch.cat([x_ext, am.pop.on])
        self.sound.step(a0, self.ret_sound)
        cpf = getattr(getattr(self.motor, 'cpools', None), 'fresh', False)
        self.motor.e1 = getattr(self.motor, 'e1_fresh', None) if cpf else getattr(self.pfc, 'e1', None)
        # concept-space pools: the PFC's expectation (pfc.e1 was made at the start of the last tick, for that tick's
        # arrivals; with fresh, its newest push, made at the end of the last tick, for what comes next)
        self.motor.heard_now = None if cc is None else getattr(cc, 'below', None)   # and what arrives from below now
        muscles = self.motor.step(a0, self.ret_force, self.ret_stop)
        cnt = torch.zeros(self.motor.nmus, device=DEV).index_add_(0, self.motor.muscle_of, self.motor.force.on.float())
        if self.motor.nmus != 24:                          # 72 channels: a muscle feels the strongest of its three parts
            cnt = cnt.view(-1, 24).max(0).values
        self.touch.step(self.skin.code(self.skin.force(cnt)), self.ret_touch)    # the body's force, felt now
        self.pfc.adapt = self.hip.adapt = self.learning
        if getattr(self, 'pfc_in', None) is not None:      # the PFC and hippocampus hear the senses only through the
            x_ext = x_ext & self.pfc_in                    # concept area (designer 11-10); force/stop copies direct
        R = self.pfc.step(x_ext)
        h_int = self.hip.step(x_ext, self.pfc.budget)
        if self.c.get('hip_recall', True):                 # False: the hippocampus stores but its recall lights nothing
            R = R | h_int                                  # (a registered test switch, to see the PFC alone first)
        self.ret_sound, self.ret_touch, self.ret_force, self.ret_stop = ch.scatter(R, self.sound.n, self.touch.n)
        if self.learning:
            tabs = self.plastic()
            for tb in tabs:
                tb.log = []
            self.pfc.learn()
            self.hip.learn()
            self.motor.learn()
            if cc is not None:
                cc.learn()
            if am is not None:
                am.learn(self.pfc.L1ext)
            self.rlog.append((self.t, [(tb, rows, d) for tb in tabs for rows, d in tb.log]))
            while self.rlog and self.t - self.rlog[0][0] >= self.c['r_win']:
                self.rlog.popleft()
            for tb in tabs:
                tb.log = None
            if (self.t + 1) % self.c['decay_every'] == 0:
                for tb in tabs:
                    tb.decay()
        if am is not None:
            # the reward: the pleased group lit by the pat itself, minus what the PFC expected of those copies this
            # tick (Fable a35: an expected pat changes nothing; learned or recalled activity never counts)
            exp = 0.0
            if getattr(self.pfc, 'GT', None) is not None and am.r_innate > 0:
                exp = float(self.pfc.gt_used[self.amyg_pleased].clamp(0.0, 1.0).mean())
            rb = max(0.0, am.r_innate - exp)
            st = self.__dict__.setdefault('r_stats', dict(pats=0, innate=0.0, expected=0.0, reward=0.0))
            if r > 0:
                st['pats'] += 1
                st['innate'] += am.r_innate
                st['expected'] += exp
                st['reward'] += rb
            if rb > 0:
                self.reward(rb)
        elif r != 0.0:
            self.reward(r)
        if getattr(getattr(self.motor, 'cpools', None), 'fresh', False):   # the newest push, for the motor area next
            self.motor.e1_fresh = self.pfc.push() - self.pfc.usual         # tick (core/speak3.py)
        self.t += 1
        return muscles

    def reward(self, r):
        """ARCH 四: the changes of the last r_win ticks (fading with age) are added again x r_pleased when pleased;
        when upset they are reversed (x -(1 + r_upset)) and the speech state now (L4) is linked to the stop copies."""
        c = self.c
        gain = c['r_pleased'] * r if r > 0 else -(1.0 + c['r_upset']) * (-r)
        act = None
        if c.get('reward_scope', 'action') == 'action':     # only the last layer -> action copies (ARCH 四)
            act = torch.zeros(self.n1 + 1, dtype=torch.bool, device=DEV)
            for kind, l, a, bb in self.chains.segs:
                if kind in ('force', 'stop'):
                    act[a:bb] = True
        for t0, entries in self.rlog:
            f = gain * (1.0 - (self.t - t0) / c['r_win'])
            for tb, rows, d in entries:
                if act is None:
                    tb.apply(rows, f * d, log=False)
                elif tb is self.pfc.L41:
                    tb.apply(rows, f * d * act[tb.idx[rows].clamp(max=self.n1)].float(), log=False)
        self.rlog.clear()
        if r < 0:
            L = self.pfc.L41
            _, _, a, b = self.chains.seg('stop')[0]
            rows = torch.nonzero(self.pfc.L4.on).squeeze(1)
            if rows.numel():
                ix = L.idx[rows]
                L.apply(rows, c['eta'] * c['r_stop'] * (-r) * L.plast(L.w[rows]) * ((ix >= a) & (ix < b)).float(),
                        log=False)

    # ------------------------------------------------------------------------------ probes: neuron states only
    def snapshot(self):
        sa = getattr(self, 'sight', None)
        return dict(sound=[x.clone() for x in self.sound.state], touch=[x.clone() for x in self.touch.state],
                    sight=None if sa is None else ([x.clone() for x in sa.state],
                                                   [x.clone() for x in self.chains.ret_sight]),
                    rs=[x.clone() for x in self.ret_sound], rt=[x.clone() for x in self.ret_touch],
                    rf=self.ret_force.clone(), rp=self.ret_stop.clone(), pops=[p.state() for p in self.pops()],
                    rc=getattr(self.chains, 'ret_concept', None), ra=getattr(self.chains, 'ret_amyg', None))

    def restore(self, sn):
        if sn.get('sight') is not None:
            self.sight.state = [x.clone() for x in sn['sight'][0]]
            self.chains.ret_sight = [x.clone() for x in sn['sight'][1]]
        self.sound.state = [x.clone() for x in sn['sound']]
        self.touch.state = [x.clone() for x in sn['touch']]
        self.ret_sound = [x.clone() for x in sn['rs']]
        self.ret_touch = [x.clone() for x in sn['rt']]
        self.ret_force, self.ret_stop = sn['rf'].clone(), sn['rp'].clone()
        for p, st in zip(self.pops(), sn['pops']):
            p.load(st)
        if sn.get('rc') is not None:
            self.chains.ret_concept = sn['rc'].clone()
        if sn.get('ra') is not None:
            self.chains.ret_amyg = sn['ra'].clone()

    def counts(self):
        p, h = self.pfc, self.hip
        v = [x.sum() for x in self.sound.state] + [p.L1ext.on.sum(), p.x_int.sum(), p.L1.on.sum(), p.L2.on.sum(),
                                                   p.L3.on.sum(), p.L4.on.sum(), h.h_int.sum(), h.Hs.on.sum(),
                                                   self.motor.force.on.sum(), self.motor.stop.on.sum()]
        if getattr(self, 'concept', None) is not None:
            v.append(self.concept.pop.on.sum())
        return torch.stack(v)

    def count_names(self):
        return [f'A{l + 1}' for l in range(len(self.sound.n))] + \
            ['L1ext', 'L1int', 'L1', 'L2', 'L3', 'L4', 'H1int', 'Hs', 'force', 'stop'] + \
            (['concept'] if getattr(self, 'concept', None) is not None else [])
