"""The model generator (designer 11-13: "你要看透这个架构的本质，当你理解透本质后，就能很简单的用生成器做出来这个模型了").

The whole brain is made at birth from docs/ARCHITECTURE.md, with no switches and no exceptions:
- every area is made the same way (net.local_net: local, random, fixed birth weights; each neuron's own threshold
  from what it receives), fewer lit deeper; neurons are on or off;
- senses: hearing, touch (the muscles' force), sight; the concept area takes every sense: what is heard and felt is
  its input, what is seen lights its own units and teaches the others (grow.ConceptArea rule b);
- the PFC hears through the concept area (designer 11-10: it sits behind the association areas) plus the motor and
  amygdala neurons: its first layer is the copies of the concept units, the force and stop neurons and the amygdala;
  each copy has a return; a copy lit from inside is a thought - the only output - and every thought, an intention
  included, competes in the one thought budget;
- what comes next is learned only by the last layer -> first layer links, checked by each copy (ARCH 四; no separate
  error-correcting table, ARCH 七); last-layer <-> last-layer links and the hippocampus use the one temporal
  Hebbian rule; every learned link takes the one global decay;
- the hippocampus stores everything and its recall lights the copies' sources through the returns;
- the motor area: force neurons (more = stronger), tonic and recruited blockers, stop neurons; a return only
  prepares a force neuron; the generation-0 settling among the wired syllables' muscle neurons gives one whole
  syllable or nothing (ARCH 三 运动区落定; speak2.SettlePools);
- generation 0 (ARCH 五 1, registered): in the PFC, "sound <-> the muscles that make it": each syllable's sign lights
  a fixed set of concept units (the birth links of sight), and the copies of that set are wired to the copies of the
  syllable's force neurons at practised strength (the sound is taught to light the same units);
- the amygdala: a pat lights the pleased group (innate); r = pat-lit share - what the PFC expected.
Probes and lives use the same interface as brain.Brain (step, snapshot / restore, pfc, hip, motor, concept, chains)."""
import collections
import math

import numpy as np
import torch

from .amygdala import Amygdala
from .brain import PFC, Hippocampus, Motor, SenseArea
from .grow import ConceptArea
from .net import DEV, Fixed, Links, Pop, topk_on
from .vision import GRID, EyeMotion, pairs


class Copies:
    """the PFC's first layer: one copy per concept unit, force neuron, stop neuron and amygdala unit; blocks keep
    their source's grid (block number, y, z); the sheet position is where learned links look for near partners"""

    def __init__(self, parts):
        segs, grid, pos, a = [], [], [], 0
        for blk, (kind, n, Y) in enumerate(parts):
            u = np.arange(n)
            g = np.stack([np.full_like(u, blk), u // Y, u % Y], 1)
            segs.append((kind, 0, a, a + n))
            grid.append(g)
            pos.append(np.stack([g[:, 1] / max(1, (n - 1) // Y), g[:, 2] / max(1, Y - 1)], 1))
            a += n
        self.segs, self.n = segs, a
        self.grid = np.concatenate(grid)
        jit = np.random.default_rng(7).normal(0, 0.005, (a, 2))
        self.pos = torch.as_tensor(np.clip(np.concatenate(pos) + jit, 0, 1), dtype=torch.float32, device=DEV)

    def seg(self, kind):
        return [s for s in self.segs if s[0] == kind]

    def scatter(self, R):
        return {kind: R[a:b] for kind, l, a, b in self.segs}


class Model:
    def __init__(self, c, seed, coch, skin, pat72, sids, sign_of, birth_A0, birth_V, birth_T):
        c['sight_to_code'] = False                         # seen units teach the sound; what is sent on is heard
        """birth_A0: bool [T, n_ear] what the newborn hears; birth_V: bool [T, 2 x sight_bits] the fovea's pairs;
        birth_T: bool [T, n_touch_in] the skin at rest; pat72: bool [n_syl, 72] each syllable's muscle channels;
        sids: the wired syllables; sign_of(sid): the sign's pixels (bool [sight_bits])"""
        self.c, self.seed = c, seed
        lin = np.random.default_rng(c['lineage_seed'])          # family wiring
        own = np.random.default_rng(seed + 500)
        self.n_ear = coch.n
        self.skin = skin
        side = int(round(int(c['sight_bits']) ** 0.5))
        self.sound = SenseArea(coch.NC, coch.Z, c['sound_rates'], c, lin)
        self.touch = SenseArea(skin.NC, skin.Z, c['touch_rates'], c, lin)
        self.sight = SenseArea(side, 2 * side, c['sight_rates'], c, lin)
        self.motor = Motor(c, coch.brake_lines)
        A0, V0, T0 = (torch.as_tensor(x, device=DEV) for x in (birth_A0, birth_V, birth_T))
        S, Tt, Vs = self.sound.calibrate(A0), self.touch.calibrate(T0), self.sight.calibrate(V0)
        lo, hi = c['band']
        voice = torch.stack([self.motor.brake(a).any() for a in A0])
        rate = lambda X: X.float().mean(0)
        self.f_sound = torch.nonzero((lambda r: (r >= lo) & (r <= hi))(rate(S[0][voice]))).squeeze(1)
        self.f_touch = torch.nonzero((lambda r: (r >= lo) & (r <= hi))(rate(Tt[0]))).squeeze(1)
        self.f_sight = torch.nonzero((lambda r: (r >= lo) & (r <= hi))(rate(Vs[0]))).squeeze(1)
        # the concept area: heard and felt features are its input; seen features its second sense (rule b)
        n_src = len(self.f_sound) + len(self.f_touch)
        cc = ConceptArea(n_src, c, own)
        cc.setup_b(0, own)
        K = c['rec_fan']
        cc.vbirth = Fixed(own.integers(0, len(self.f_sight), (cc.n, K)), own.uniform(0.5, 1.5, (cc.n, K)))
        cc.v_rest, cc.v_n, cc.v_usual, cc.sight_x = None, 0, None, True
        src0 = torch.cat([S[0][:, self.f_sound], Tt[0][:, self.f_touch]], 1)
        D = cc.birth.drive_batch(src0)
        cc.usual = D.mean(0)
        cc.win = torch.topk(D - cc.usual[None, :], cc.k, dim=1).values.mean()
        Cb = torch.zeros((D.shape[0], cc.n), dtype=torch.bool, device=DEV)
        Cb.scatter_(1, torch.topk(D - cc.usual[None, :], cc.k, dim=1).indices, True)
        del D
        self.concept = cc
        self.blank = torch.as_tensor(pairs(sign_of(None)), device=DEV)   # what the fovea shows when nothing is there
        # the amygdala's own neurons
        n_am, n_pl, n_pat = 400, 200, 16
        m = self.motor
        self.chains = Copies([('concept', cc.n, int(math.ceil(math.sqrt(cc.n)))), ('force', m.nf, m.m),
                              ('stop', m.ns, 2), ('amyg', n_am, 20)])
        self.n1 = self.chains.n
        self.seg = {k: (a, b) for k, l, a, b in self.chains.segs}
        self.amyg = Amygdala(n_am, n_pl, n_pat, self.n1, np.ones(self.n1, bool), c, own)
        a_, _ = self.seg['amyg']
        self.amyg_pleased = torch.arange(a_, a_ + n_pl, device=DEV)
        self.pfc = PFC(self.chains, c, own)
        self.hip = Hippocampus(self.n1, c, own)
        # the PFC's birth stream: concept units as born, the brake's stop neurons, nothing else
        T = Cb.shape[0]
        stop = torch.stack([self.motor.brake(a) for a in A0])
        X = torch.zeros((T, self.n1), dtype=torch.bool, device=DEV)
        ca, cb = self.seg['concept']
        X[:, ca:cb] = Cb
        s0, s1 = self.seg['stop']
        X[:, s0:s1] = stop
        self.birth_density = self.pfc.calibrate(X)
        del X, Cb
        # the motor area's settling among the wired syllables (generation 0)
        from .speak2 import SettlePools
        P = torch.as_tensor(pat72[sids], dtype=torch.bool, device=DEV)
        m.settle, m.settle_sids = SettlePools(P), list(sids)   # pools read which muscles are prepared
        m.settle.channel_of, m.settle.Pf = m.muscle_of, P[:, m.muscle_of].float()
        m.ret_prep = 0.6 * (m.thr_f + c['b_block'] * m.n_tonic / (8.0 * m.m))
        m.complete = m.thr_f.max() + c['b_block'] * m.n_tonic / (8.0 * m.m) + 0.5
        # generation 0 in the PFC: each sign's concept units -> its syllable's force copies (ARCH 五 1)
        self.speech_link(sids, sign_of, pat72)
        self.eye_motion, self.n_periph = EyeMotion(c), 2 * GRID * GRID
        self.eye_muscles = torch.zeros(4, dtype=torch.bool, device=DEV)
        self.ret = {k: torch.zeros(b - a, device=DEV) for k, (a, b) in self.seg.items()}
        self.sight_prev = None
        self.touch.state = [x[-1].clone() for x in Tt]
        self.learning, self.t = False, 0
        self.rlog = collections.deque()
        self.registry = [('generation 0: sign concept units -> syllable force copies', len(sids), int(self.g0_units)),
                         ('motor settling pools (generation 0)', len(sids), 1),
                         ('amygdala: pat -> pleased group (innate)', n_pl, n_pat)]

    def speech_link(self, sids, sign_of, pat72):
        cc, m = self.concept, self.motor
        ca, _ = self.seg['concept']
        f0, f1 = self.seg['force']
        sa = self.sight

        def feats(px):                                     # the sight area's first layer for a still picture
            return (sa.layers[0].drive(px) > sa.thr[0])[self.f_sight]
        VD = torch.stack([cc.vbirth.drive(feats(torch.as_tensor(pairs(sign_of(s)), device=DEV))) for s in sids])
        cc.v_usual, cc.v_n = VD.mean(0), len(sids)          # each unit's usual drive while something is seen
        rows, cols = [], []
        self.sign_units = {}
        for i, s in enumerate(sids):                       # the units a sign lights by sight alone (its own top-k)
            u = topk_on(VD[i] - cc.v_usual, cc.k)
            self.sign_units[int(s)] = u
            fn = torch.as_tensor(np.nonzero(pat72[s])[0], device=DEV) * m.m   # one copy per muscle (its lowest grade)
            uu = torch.nonzero(u).squeeze(1)
            rows.append(uu.repeat_interleave(len(fn)))
            cols.append(fn.repeat(len(uu)))
        r, cl = torch.cat(rows), torch.cat(cols)
        pair = torch.unique(r * (f1 - f0 + 1) + cl)            # each unit -> muscle link once (a unit in several
        r, cl = pair // (f1 - f0 + 1), pair % (f1 - f0 + 1)    # signs' sets must not push a shared muscle twice)
        n_pre = cc.n
        per = torch.bincount(r, minlength=n_pre)
        width = int(per.max())
        idx = torch.full((n_pre, width), f1 - f0, dtype=torch.int64, device=DEV)
        order = torch.argsort(r)
        r, cl = r[order], cl[order]
        start = torch.cumsum(per, 0) - per
        pos = torch.arange(len(r), device=DEV) - start[r]
        idx[r, pos] = cl
        self.g0_units = int((per > 0).sum())
        # practised strength (ARCH 六): a syllable's whole set pushes each of its muscles' copies to about 2 theta
        w = 2.0 * self.c['theta1'] / max(1.0, float(cc.k))
        self.g0 = Links(idx, w, f1 - f0, self.c, gamma=1.0, name='speech gen 0')
        self.g0.w *= (idx < f1 - f0)

    # ------------------------------------------------------------------------------------------------------------
    def step(self, a0, r=0.0):
        c, cc, p, m, am = self.c, self.concept, self.pfc, self.motor, self.amyg
        per, a0 = a0[-self.n_periph:], a0[:-self.n_periph]   # the eye's periphery -> motion neurons -> eye muscles
        self.eye_muscles = self.eye_motion.step(per[0::2])
        sight, a0 = a0[self.n_ear:], a0[:self.n_ear]
        seen = self.sight_prev is not None and bool((self.sight_prev != self.blank).any())
        self.sight_prev = sight
        # the concept area reads the features of a tick ago (heard, felt; seen as its second sense)
        src = torch.cat([self.sound.state[0][self.f_sound], self.touch.state[0][self.f_touch]])
        cc.adapt = self.learning
        cc.step(src, self.ret['concept'], self.sight.state[0][self.f_sight], seen)
        self.sight.step(sight, [torch.zeros(n, device=DEV) for n in self.sight.n])
        stop_on = m.stop.on
        x_ext = torch.cat([cc.pop.on, m.force.on, stop_on, am.pop.on])
        am.adapt = self.learning
        am.step(x_ext, r, self.ret['amyg'])
        a_, b_ = self.seg['amyg']
        x_ext[a_:b_] = am.pop.on
        self.sound.step(a0, [torch.zeros(n, device=DEV) for n in self.sound.n])
        muscles = m.step(a0, self.ret['force'], self.ret['stop'] > 0)
        cnt = torch.zeros(m.nmus, device=DEV).index_add_(0, m.muscle_of, m.force.on.float()).view(-1, 24).max(0).values
        self.touch.step(self.skin.code(self.skin.force(cnt)), [torch.zeros(n, device=DEV) for n in self.touch.n])
        # generation 0: thought concept copies push their syllables' force copies (with the PFC's other pushes)
        ca, cb = self.seg['concept']
        f0, f1 = self.seg['force']
        g0 = torch.zeros(self.n1, device=DEV)
        g0[f0:f1] = self.g0.drive(p.x_int[ca:cb])
        p.extra = g0
        p.adapt = self.hip.adapt = self.learning
        R = p.step(x_ext)
        R = R | self.hip.step(x_ext, p.budget)             # recall lights the copies' sources too
        self.ret = {k: v.float() for k, v in self.chains.scatter(R).items()}
        if self.learning:
            tabs = self.plastic()
            for tb in tabs:
                tb.log = []
            p.learn()
            self.hip.learn()
            cc.learn()
            am.learn(p.L1ext)
            self.rlog.append((self.t, [(tb, rows, d) for tb in tabs for rows, d in tb.log]))
            while self.rlog and self.t - self.rlog[0][0] >= c['r_win']:
                self.rlog.popleft()
            for tb in tabs:
                tb.log = None
            if (self.t + 1) % c['decay_every'] == 0:
                for tb in tabs:
                    tb.decay()
                cc.decay()
        if am.r_innate > 0:                                # the reward: pat-lit share minus what the PFC expected
            exp = float(p.e1[self.amyg_pleased].clamp(0.0, 1.0).mean()) if getattr(p, 'e1', None) is not None else 0.0
            rb = max(0.0, am.r_innate - exp)
            if rb > 0 and self.learning:
                self.reward(rb)
        self.t += 1
        return muscles

    @property
    def ret_force(self):
        return self.ret['force']

    @property
    def ret_stop(self):
        return self.ret['stop'] > 0

    def plastic(self):
        return self.pfc.plastic() + self.hip.plastic() + [self.amyg.learned]

    def reward(self, r):
        """ARCH 四: the changes of the last r_win ticks to last layer -> action copies (force, stop) are added again"""
        c = self.c
        act = torch.zeros(self.n1 + 1, dtype=torch.bool, device=DEV)
        for k in ('force', 'stop'):
            a, b = self.seg[k]
            act[a:b] = True
        for t0, entries in self.rlog:
            f = c['r_pleased'] * r * (1.0 - (self.t - t0) / c['r_win'])
            for tb, rows, d in entries:
                if tb is self.pfc.L41:
                    tb.apply(rows, f * d * act[tb.idx[rows].clamp(max=self.n1)].float(), log=False)
        self.rlog.clear()

    def pops(self):
        return self.pfc.pops() + self.hip.pops() + self.motor.pops() + [self.concept.pop, self.concept.src,
                                                                          self.amyg.pop]

    def snapshot(self):
        return dict(sound=[x.clone() for x in self.sound.state], touch=[x.clone() for x in self.touch.state],
                    sight=[x.clone() for x in self.sight.state], ret={k: v.clone() for k, v in self.ret.items()},
                    pops=[q.state() for q in self.pops()], x_int=self.pfc.x_int.clone(), sp=self.sight_prev)

    def restore(self, sn):
        self.sound.state = [x.clone() for x in sn['sound']]
        self.touch.state = [x.clone() for x in sn['touch']]
        self.sight.state = [x.clone() for x in sn['sight']]
        self.ret = {k: v.clone() for k, v in sn['ret'].items()}
        for q, st in zip(self.pops(), sn['pops']):
            q.load(st)
        self.pfc.x_int = sn['x_int'].clone()
        self.sight_prev = sn['sp']

    def count_names(self):
        return ['L1ext', 'L1int', 'L4', 'Hs', 'concept', 'force', 'stop', 'amyg']

    def counts(self):
        p = self.pfc
        return torch.stack([p.L1ext.on.sum(), p.x_int.sum(), p.L4.on.sum(), self.hip.Hs.on.sum(),
                            self.concept.pop.on.sum(), self.motor.force.on.sum(), self.motor.stop.on.sum(),
                            self.amyg.pop.on.sum()])
