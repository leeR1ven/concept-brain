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

    def __init__(self, sound, s_rates, touch, t_rates, motor, band):
        lo, hi = band
        self.sound_idx = [torch.nonzero((r >= lo) & (r <= hi)).squeeze(1) for r in s_rates]
        self.touch_idx = [torch.nonzero((r >= lo) & (r <= hi)).squeeze(1) for r in t_rates]
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
        u = np.arange(motor.nf)
        add('force', 0, motor.nf, np.stack([np.full_like(u, 2), u // motor.m, u % motor.m], 1), 24, motor.m)
        u = np.arange(motor.ns)
        add('stop', 0, motor.ns, np.stack([np.full_like(u, 3), u, np.zeros_like(u)], 1), motor.ns, 2)
        self.segs, self.n = segs, a
        self.grid = np.concatenate(grid)
        jit = np.random.default_rng(7).normal(0, 0.005, (a, 2))
        self.pos = torch.as_tensor(np.clip(np.concatenate(pos) + jit, 0, 1), dtype=torch.float32, device=DEV)

    def seg(self, kind):
        return [s for s in self.segs if s[0] == kind]

    def gather(self, sound, touch, force, stop):
        return torch.cat([sound[l][ix] for l, ix in enumerate(self.sound_idx)] +
                         [touch[l][ix] for l, ix in enumerate(self.touch_idx)] + [force, stop])

    def scatter(self, R, sound_n, touch_n):
        """returns R (bool [n]) to its own sources"""
        rs = [torch.zeros(n, device=DEV) for n in sound_n]
        rt = [torch.zeros(n, device=DEV) for n in touch_n]
        for kind, l, a, b in self.segs:
            if kind == 'sound':
                rs[l][self.sound_idx[l]] = R[a:b].float()
            elif kind == 'touch':
                rt[l][self.touch_idx[l]] = R[a:b].float()
            elif kind == 'force':
                rf = R[a:b].float()
            else:
                rp = R[a:b]
        return rs, rt, rf, rp


# ============================================================================================ PFC
def reciprocal_loop(ff, n1, fan):
    """ARCH 三 前额叶 birth loop: each last-layer unit returns to the `fan` first-layer copies with the largest path
    weight into it (through L3 and L2)"""
    out = []
    i3, w3 = ff[2].idx, ff[2].w
    for a in range(0, i3.shape[0], 256):
        i2 = ff[1].idx[i3[a:a + 256]]
        p2 = w3[a:a + 256, :, None] * ff[1].w[i3[a:a + 256]]
        i1 = ff[0].idx[i2]
        p1 = p2[..., None] * ff[0].w[i2]
        dense = torch.zeros((i2.shape[0], n1), device=DEV)
        dense.scatter_add_(1, i1.reshape(i2.shape[0], -1), p1.reshape(i2.shape[0], -1))
        out.append(torch.topk(dense, fan, dim=1).indices)
    return torch.cat(out)


class PFC:
    """ARCH 三 前额叶. L1 = the copies. L2, L3, L4 = one unit per L1 position, made by the one generator (二.4),
    fixed, each unit's own threshold at L1's birth density. L4 (the output) lights exactly as many units as L1 (the
    input) (六). L4(t) -> L1(t+1): the birth loop (holds a thought) and learned links to near copies (what comes
    next); L4 <-> L4 learned links (related units excite each other), kept weak (六)."""

    def __init__(self, ch, c, rng):
        n = ch.n
        self.c, self.n = c, n
        self.ff = [Fixed(*local_net(ch.grid, ch.grid, c['K'], c['dy'][0], c['dz'][0], rng)) for _ in range(3)]
        li = reciprocal_loop(self.ff, n, c['loop_fan'])
        lw = torch.as_tensor(rng.uniform(0.5, 1.5, tuple(li.shape)), dtype=torch.float32, device=DEV)
        self.loop = Links(li, c['g_loop'] * lw / lw.sum(1, keepdim=True), n, c, learned=False, name='loop')
        self.L41 = Links(radius_lists(ch.pos, ch.pos, c['r_learn']), 0.0, n, c, gain=c['g31'], gamma=c['gamma_pfc'],
                         name='L4->L1')
        self.L44 = Links(radius_lists(ch.pos, ch.pos, c['r_learn'], True), 0.0, n, c, gain=c['g44'],
                         gamma=c['gamma_pfc'], name='L4<->L4')
        self.G1P = None                                    # generation-0 speech link (gen0.py)
        self.L1, self.L1ext, self.L2, self.L3, self.L4 = (Pop(n) for _ in range(5))
        self.x_int = torch.zeros(n, dtype=torch.bool, device=DEV)
        self.force_int = None                              # probes only: copies lit as if thought

    def plastic(self):
        return [self.L41, self.L44] + ([self.G1P] if self.G1P is not None else [])

    def calibrate(self, X):
        """X: bool [T, n] the birth chain input (loop and learning off)"""
        rho = float(X.float().mean())
        D2 = self.ff[0].drive_batch(X)
        self.thr2 = own_thresholds(D2, rho)
        S2 = D2 > self.thr2
        del D2
        self.thr3 = own_thresholds(self.ff[1].drive_batch(S2), rho)
        return rho

    def step(self, x_ext):
        a4 = self.L4.on
        d1 = self.loop.drive(a4) + self.L41.drive(a4)
        if self.G1P is not None:
            d1 = d1 + self.G1P.drive(a4)
        # excitability = context length (ARCH 三 前额叶, 六): at most `context` x the external input count of copies
        # are lit from inside, the most driven of those over threshold
        x_int = (d1 > self.c['theta1']) & topk_on(d1, self.c['context'] * x_ext.sum())
        if self.force_int is not None:
            x_int = x_int | self.force_int
        n2 = self.ff[0].drive(self.L1.on) > self.thr2
        n3 = self.ff[1].drive(self.L2.on) > self.thr3
        n4 = topk_on(self.ff[2].drive(self.L3.on) + self.L44.drive(a4), self.L1.on.sum())   # output = input count
        self.L1ext.set(x_ext)
        self.L1.set(x_ext | x_int)
        self.L2.set(n2)
        self.L3.set(n3)
        self.L4.set(n4)
        self.x_int = x_int
        return x_int

    def learn(self):
        fw, eta = self.c['fall_win'], self.c['eta']
        onset = self.L1ext.rose() & ~self.L1.prev           # a copy driven on from outside (not already thought)
        recent = self.L1ext.held(fw)
        for tb in [self.L41] + ([self.G1P] if self.G1P is not None else []):
            tb.learn(self.L4, onset, recent, eta)
        self.L44.learn(self.L4, self.L4.rose(), self.L4.held(fw), eta)

    def pops(self):
        return [self.L1, self.L1ext, self.L2, self.L3, self.L4]


# ============================================================================================ hippocampus
class Hippocampus:
    """ARCH 三 海马体: copies H1 -> its own population Hs (random birth input, no birth recurrence, a fixed count kh
    per tick) -> learned Hs <-> Hs and Hs -> H1 links that grow and decay fast. Recall lights H1 copies; their
    returns light the source units, which reach the PFC through the chains."""

    def __init__(self, n1, c, rng):
        s = c['s']
        nh = c['nh'] * s
        self.c, self.nh = c, nh
        self.rec = Fixed(rng.integers(0, n1, (nh, c['rec_fan'])), rng.uniform(0.5, 1.5, (nh, c['rec_fan'])))
        self.HH = Links(random_lists(nh, nh, c['n_hh'] * s, rng, True), 0.0, nh, c, gain=c['g_hh'],
                        gamma=c['gamma_hip'], name='Hs<->Hs')     # internal links: the cortex's rule
        self.H1t = Links(random_lists(nh, n1, c['n_h1'] * s, rng), 0.0, n1, c,
                         gamma=c['gamma_hip'], name='Hs->H1', plain=True)
        self.H1, self.H1ext, self.Hs = Pop(n1), Pop(n1), Pop(nh)
        self.h_int = torch.zeros(n1, dtype=torch.bool, device=DEV)

    def plastic(self):
        return [self.HH, self.H1t]

    def step(self, x_ext):
        ah = self.Hs.on
        d = self.H1t.drive(ah)
        h_int = topk_on(d, self.c['context'] * x_ext.sum())      # a fixed share: the most strongly recalled
        hs = topk_on(self.rec.drive(self.H1.on) + self.HH.drive(ah), self.c['kh'])
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
        self.nf, self.nb, self.ns = 24 * m, 8 * m, 8 * m
        k = torch.arange(self.nf, device=DEV) % m
        self.muscle_of = torch.arange(self.nf, device=DEV) // m
        self.thr_f = c['theta_m'] * (1.0 + k.float() / m)
        self.n_tonic = 4 * m
        self.ret_drive = c['g_ret_force'] * (self.thr_f + c['b_block'] * self.n_tonic / (8.0 * m))
        sharp = float(self.ret_drive.view(24, m)[:4].sum())               # one syllable: 4 muscles, every grade
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
        stop = self.brake(a0) | ret_stop
        lit = ret_force > 0
        d = lit.float() * self.ret_drive
        P = d.sum()
        nblock = self.n_tonic + (P >= self.ladder).sum()
        inh = c['b_block'] * nblock / (8.0 * m) + c['c_stop'] * stop.float().sum() / (8.0 * m)
        if self.pools is not None:
            pe, pi = self.pool_fire(lit)
            d = d + self.pools['exc'].drive(pe) - self.pools['inh'].drive(pi)
            self.pools['pe'].set(pe)
            self.pools['pi'].set(pi)
        force = d - inh > self.thr_f
        self.lit.set(lit)
        self.force.set(force)
        self.stop.set(stop)
        on = torch.zeros(24, device=DEV).index_add_(0, self.muscle_of, force.float())
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
    def __init__(self, c, seed, coch, skin, birth_A0):
        """birth_A0: bool [T, n0], the birth period heard by the silent newborn (learning off)"""
        self.c, self.seed, self.n0 = c, seed, coch.n
        lin = np.random.default_rng(c['lineage_seed'])          # family wiring: the same for every seed
        own = np.random.default_rng(seed + 500)
        self.sound = SenseArea(coch.NC, coch.Z, c['sound_rates'], c, lin)
        self.skin = skin
        self.touch = SenseArea(skin.NC, skin.Z, c['touch_rates'], c, lin)
        self.motor = Motor(c, coch.brake_lines)
        A0 = torch.as_tensor(birth_A0, device=DEV)
        s_states = self.sound.calibrate(A0)
        t_states = self.touch.calibrate(skin.birth_stream(A0.shape[0], seed))
        self.chains = Chains(self.sound, [x.float().mean(0) for x in s_states],
                             self.touch, [x.float().mean(0) for x in t_states], self.motor, c['band'])
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
                      [torch.zeros((T, self.motor.nf), dtype=torch.bool, device=DEV), stop], 1)
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

    def plastic(self):
        return self.pfc.plastic() + self.hip.plastic() + self.motor.plastic()

    def pops(self):
        return self.pfc.pops() + self.hip.pops() + self.motor.pops()

    def step(self, a0, r=0.0):
        assert a0.dtype == torch.bool and tuple(a0.shape) == (self.n0,), 'the brain receives only the input bits'
        ch = self.chains
        x_ext = ch.gather(self.sound.state, self.touch.state, self.motor.force.on, self.motor.stop.on)   # t-1
        self.sound.step(a0, self.ret_sound)
        muscles = self.motor.step(a0, self.ret_force, self.ret_stop)
        cnt = torch.zeros(24, device=DEV).index_add_(0, self.motor.muscle_of, self.motor.force.on.float())
        self.touch.step(self.skin.code(self.skin.force(cnt)), self.ret_touch)    # the body's force, felt now
        R = self.pfc.step(x_ext) | self.hip.step(x_ext)
        self.ret_sound, self.ret_touch, self.ret_force, self.ret_stop = ch.scatter(R, self.sound.n, self.touch.n)
        if self.learning:
            tabs = self.plastic()
            for tb in tabs:
                tb.log = []
            self.pfc.learn()
            self.hip.learn()
            self.motor.learn()
            self.rlog.append((self.t, [(tb, rows, d) for tb in tabs for rows, d in tb.log]))
            while self.rlog and self.t - self.rlog[0][0] >= self.c['r_win']:
                self.rlog.popleft()
            for tb in tabs:
                tb.log = None
            if (self.t + 1) % self.c['decay_every'] == 0:
                for tb in tabs:
                    tb.decay()
        if r != 0.0:
            self.reward(r)
        self.t += 1
        return muscles

    def reward(self, r):
        """ARCH 四: the changes of the last r_win ticks (fading with age) are added again x r_pleased when pleased;
        when upset they are reversed (x -(1 + r_upset)) and the speech state now (L4) is linked to the stop copies."""
        c = self.c
        gain = c['r_pleased'] * r if r > 0 else -(1.0 + c['r_upset']) * (-r)
        for t0, entries in self.rlog:
            f = gain * (1.0 - (self.t - t0) / c['r_win'])
            for tb, rows, d in entries:
                tb.apply(rows, f * d, log=False)
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
        return dict(sound=[x.clone() for x in self.sound.state], touch=[x.clone() for x in self.touch.state],
                    rs=[x.clone() for x in self.ret_sound], rt=[x.clone() for x in self.ret_touch],
                    rf=self.ret_force.clone(), rp=self.ret_stop.clone(), pops=[p.state() for p in self.pops()])

    def restore(self, sn):
        self.sound.state = [x.clone() for x in sn['sound']]
        self.touch.state = [x.clone() for x in sn['touch']]
        self.ret_sound = [x.clone() for x in sn['rs']]
        self.ret_touch = [x.clone() for x in sn['rt']]
        self.ret_force, self.ret_stop = sn['rf'].clone(), sn['rp'].clone()
        for p, st in zip(self.pops(), sn['pops']):
            p.load(st)

    def counts(self):
        p, h = self.pfc, self.hip
        v = [x.sum() for x in self.sound.state] + [p.L1ext.on.sum(), p.x_int.sum(), p.L1.on.sum(), p.L2.on.sum(),
                                                   p.L3.on.sum(), p.L4.on.sum(), h.h_int.sum(), h.Hs.on.sum(),
                                                   self.motor.force.on.sum(), self.motor.stop.on.sum()]
        return torch.stack(v)

    def count_names(self):
        return [f'A{l + 1}' for l in range(len(self.sound.n))] + \
            ['L1ext', 'L1int', 'L1', 'L2', 'L3', 'L4', 'H1int', 'Hs', 'force', 'stop']
