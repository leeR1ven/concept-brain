"""Motor area [M1-M4, M6 revised 11-02]. Force neurons (s per muscle, graded thresholds), blocking neurons (half
tonic, half recruited by the pooled muscle drive on a ladder; no chains), stop neurons (brake + returns; chains).
No count gate. Designer 11-02 (ARCH 2.7, 2.11.1/3): the sound area does NOT connect to the motor area; sound and
action are linked only through the PFC. The only excitatory input of a force neuron is its own 1:1 return chain
(from its PFC L1 copy when that copy is lit internally):
    force neuron fires iff  return - b * blocking_on/(8s) - c * stops_on/(8s) > its threshold,
    return drive = g_ret_force x (its threshold + the tonic blocking), so g_ret_force > 1 can beat tonic blocking.
The body counts a muscle as on when at least half of its force neurons fire.
The brake reads only the input layer (others' voices) and only drives stop neurons."""
import math

import numpy as np
import torch

from common import DEV
from learning import ByPre, Pop


class Motor:
    def __init__(self, c, coch, gam):
        self.c = c
        s = c['s']
        m = c['force_per_muscle'] or s                                             # arm M (11-02): m force neurons
        self.s, self.m = s, m                                                      # per muscle (default m = s)
        self.nf, self.nb, self.ns = 24 * m, 8 * m, 8 * m
        k = torch.arange(self.nf, device=DEV) % m
        self.muscle_of = torch.arange(self.nf, device=DEV) // m
        self.thr_f = c['theta_m'] * (1.0 + k.float() / m)                         # [M1] graded force thresholds
        self.n_tonic = 4 * m                                                       # [M2]
        tonic_inh = c['b_block'] * self.n_tonic / (8.0 * m)
        self.ret_drive = c['g_ret_force'] * (self.thr_f + tonic_inh)              # [C1, M6] return into a force neuron
        p_sharp = float(self.ret_drive.view(24, m)[:4].sum())                      # one syllable: 4 muscles x m grades
        j = torch.arange(4 * m, device=DEV, dtype=torch.float32)
        self.ladder = p_sharp * (1.0 + (j + 1) / (4 * m))                          # recruits 1x..2x a sharp command
        g = np.random.default_rng(c['lineage_seed'] + 900)
        self.thr_s = torch.as_tensor(c['theta_stop'] * g.uniform(0.8, 1.2, self.ns), dtype=torch.float32, device=DEV)
        self.brake_lines = torch.as_tensor(coch.brake_lines, device=DEV)           # [M4] birth reflex
        self.n_brake = c['n_brake_per_s'] * s
        self.gam = gam
        self.force, self.stop = Pop(self.nf), Pop(self.ns)
        self.block = torch.zeros(self.nb, dtype=torch.bool, device=DEV)
        self.last = {}
        self.MC = None                       # gen-0 motor collapse wiring MC (pattern pools), set by set_mc
        self.inject = None                   # DIAGNOSTIC 0/1 drive into force neurons (collapse test only)

    def set_mc(self, need):
        """[MC, 11-02] gen-0 motor synergy wiring: per syllable a conjunction group of excitatory pool neurons
        (-> the syllable's own force neurons) and one of inhibitory pool neurons (-> the force neurons of the muscles
        outside the syllable). Each pool neuron samples one force neuron of each of the syllable's 4 muscles and fires
        when all its sampled inputs fire (same tick; fast interneurons). Not program neurons: they run no sequence.
        All tables are acquired (gamma_motor, the one rule); inhibitory links stay inhibitory (subtracted, >= 0).
        need: bool [n_syllables, 24]."""
        c, m = self.c, self.m
        need = np.asarray(need, bool)
        ns, nI = need.shape[0], c['mc_nI']
        g = np.random.default_rng(c['lineage_seed'] + 950)                         # lineage-fixed (inherited)
        mus_of = self.muscle_of.cpu().numpy()

        def samples():
            S = np.zeros((ns * nI, 4), np.int64)
            for s in range(ns):
                mu = np.nonzero(need[s])[0]
                for n in range(nI):
                    S[s * nI + n] = mu * m + g.integers(0, m, 4)
            return S

        def pre_table(S, name):                                                    # force -> pool (sampled inputs)
            lists = [[] for _ in range(self.nf)]
            for p, row in enumerate(S):
                for f in row:
                    lists[f].append(p)
            w = max(1, max(len(x) for x in lists))
            idx = np.full((self.nf, w), S.shape[0], np.int32)
            for f, x in enumerate(lists):
                idx[f, :len(x)] = x
            return ByPre(idx, (idx < S.shape[0]).astype(np.float32), self.nf, S.shape[0], name, birth=False,
                         eta=c['eta'], eta_m=c['eta_m'], gain=1.0, gamma=self.gam['motor'], c=c)

        def out_table(rows, name):                                                 # pool -> force
            w = max(len(r) for r in rows)
            idx = np.full((len(rows), w), self.nf, np.int32)
            for p, r in enumerate(rows):
                idx[p, :len(r)] = r
            return ByPre(idx, (idx < self.nf).astype(np.float32), len(rows), self.nf, name, birth=False,
                         eta=c['eta'], eta_m=c['eta_m'], gain=1.0, gamma=self.gam['motor'], c=c)
        syl_of = np.repeat(np.arange(ns), nI)
        own = [np.nonzero(need[s][mus_of])[0] for s in syl_of]
        riv = [np.nonzero(~need[s][mus_of])[0] for s in syl_of]
        self.mc_in_e = pre_table(samples(), 'motor MC force -> exc pool (gen-0)')
        self.mc_in_i = pre_table(samples(), 'motor MC force -> inh pool (gen-0)')
        self.mc_exc = out_table(own, 'motor MC exc pool -> own force (gen-0)')
        self.mc_inh = out_table(riv, 'motor MC inh pool -> rival force (gen-0, inhibitory)')
        self.pool_e, self.pool_i = Pop(ns * nI), Pop(ns * nI)
        self.ret_pop = Pop(self.nf)                                                # lit returns (pool input when mc_src='returns')
        self.mc_k = c['mc_k'] / 4.0 - 0.125                                        # all 4 of 4 sampled inputs
        rel = self.thr_f + c['b_block'] * self.n_tonic / (8.0 * m)
        relmed = float(rel.median())
        ge, gi = [], []
        for s in range(ns):                                                        # one fully active pool
            act = torch.zeros(ns * nI, dtype=torch.bool, device=DEV)
            act[s * nI:(s + 1) * nI] = True
            o = torch.as_tensor(need[s][mus_of], device=DEV)
            ge.append(self.mc_exc.drive(act)[o].median())
            gi.append(self.mc_inh.drive(act)[~o].median())
        self.mc_exc.gain = float(c['mc_e'] * relmed / torch.stack(ge).median())
        self.mc_inh.gain = float(c['mc_i'] * relmed / torch.stack(gi).median())
        self.MC = True
        return dict(set='MC pattern pools', syllables=ns, per_group=nI, links=int(sum(len(r) for r in own) +
                    sum(len(r) for r in riv) + 8 * ns * nI), gain_exc=self.mc_exc.gain, gain_inh=self.mc_inh.gain,
                    e=c['mc_e'], i=c['mc_i'])

    def _pools(self, cand):
        """same-tick conjunction: a pool neuron fires when its sampled force neurons fire (drive >= share of full)"""
        out = []
        for tb in (self.mc_in_e, self.mc_in_i):
            full = tb.drive(torch.ones(self.nf, dtype=torch.bool, device=DEV))
            out.append(tb.drive(cand) >= self.mc_k * full.clamp_min(1e-9))
        return out

    def tables(self):
        if self.MC is None:
            return []                                                              # no table from outside the motor area
        return [self.mc_in_e, self.mc_in_i, self.mc_exc, self.mc_inh]

    def step(self, a0, ret_force, ret_stop, scaffold=None):
        """scaffold: bool [nf] or None. Generation-0 'forced regular speaking' (designer 11-02): the teacher moves
        the mouth, i.e. these force neurons fire regardless of the blockers on the forced ticks. A program outside the
        brain, logged by the harness; None in every other stage."""
        c, m = self.c, self.m
        brake = a0[self.brake_lines].float().sum() / self.n_brake
        stop = (brake >= self.thr_s) | ret_stop
        lit = ret_force.float()
        if self.inject is not None:
            lit = (lit + self.inject).clamp_max(1.0)
        d = lit * self.ret_drive
        P = d.sum()
        lad = (P >= self.ladder).sum()
        nblock = self.n_tonic + lad
        inh = c['b_block'] * nblock / (8.0 * m) + c['c_stop'] * stop.float().sum() / (8.0 * m)
        if self.MC is not None:                                                     # [MC] same-tick pattern pools
            src = (lit > 0) if c['mc_src'] == 'returns' else (d - inh > self.thr_f)   # afferent or force output
            pe, pi = self._pools(src)
            self.ret_pop.set(lit > 0)
            d = d + self.mc_exc.drive(pe) - self.mc_inh.drive(pi)
            self.pool_e.set(pe)
            self.pool_i.set(pi)
        force = d - inh > self.thr_f
        if scaffold is not None:
            force = force | scaffold
        block = torch.zeros(self.nb, dtype=torch.bool, device=DEV)
        block[:self.n_tonic] = True
        block[self.n_tonic:] = P >= self.ladder
        self.block = block
        self.force.set(force)
        self.stop.set(stop)
        self.last = dict(P=P, d=d, inh=inh)
        on = torch.zeros(24, device=DEV).index_add_(0, self.muscle_of, force.float())
        return on >= math.ceil(m / 2)

    def learn(self):
        if self.MC is None:
            return
        fw = self.c['fall_win']
        fr = self.force.rose()
        pre = self.ret_pop if self.c['mc_src'] == 'returns' else self.force          # what the pools read
        for tb, pool in ((self.mc_in_e, self.pool_e), (self.mc_in_i, self.pool_i)):   # input -> pool
            pr = pool.rose()
            tb.learn(pre.elig_prev(fw), pre.rose(), pr, pr, pool.age_prev <= fw)
        for tb, pool in ((self.mc_exc, self.pool_e), (self.mc_inh, self.pool_i)):      # pool -> force
            tb.learn(pool.elig_prev(fw), pool.rose(), fr, fr, self.force.age_prev <= fw)

    def pops(self):
        return [self.force, self.stop] + ([self.pool_e, self.pool_i, self.ret_pop] if self.MC is not None else [])
