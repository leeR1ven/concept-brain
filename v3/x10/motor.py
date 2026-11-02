"""Motor area [M1-M6]. Force neurons (s per muscle, graded thresholds), blocking neurons (half tonic, half recruited
by the pooled muscle drive on a ladder; no chains), stop neurons (brake + returns; chains). No count gate.
A force neuron fires iff  G1 drive + motor return - b * blocking_on/(8s) - c * stops_on/(8s) > its threshold.
The body counts a muscle as on when at least half of its force neurons fire.
Timing: stop(t) and force(t) both read this tick's input (brake from A0(t), G1 from sound layer A1(t) = f(A0(t))),
so an echo of heard speech and the brake reach the muscles with the same latency [M5]."""
import math

import numpy as np
import torch

from common import DEV
from learning import ByPre, Pop


class Motor:
    def __init__(self, c, coch, gam):
        self.c = c
        s = c['s']
        self.s = s
        self.nf, self.nb, self.ns = 24 * s, 8 * s, 8 * s
        k = torch.arange(self.nf, device=DEV) % s
        self.muscle_of = torch.arange(self.nf, device=DEV) // s
        self.thr_f = c['theta_m'] * (1.0 + k.float() / s)                         # [M1] graded force thresholds
        self.n_tonic = 4 * s                                                       # [M2]
        self.ladder = None                                                         # set by wiring (M6)
        g = np.random.default_rng(c['lineage_seed'] + 900)
        self.thr_s = torch.as_tensor(c['theta_stop'] * g.uniform(0.8, 1.2, self.ns), dtype=torch.float32, device=DEV)
        self.brake_lines = torch.as_tensor(coch.brake_lines, device=DEV)           # [M4] birth reflex
        self.n_brake = c['n_brake_per_s'] * s
        self.gam = gam
        self.G1 = None                                                             # [G1] set by wiring.py
        self.g1_src = None                                                         # layer-A1 unit indices of G1's pre
        self.force, self.stop = Pop(self.nf), Pop(self.ns)
        self.g1pre = None
        self.block = torch.zeros(self.nb, dtype=torch.bool, device=DEV)
        self.last = {}

    def set_g1(self, src, w):
        """src: int64 indices into sound layer A1 (the pre units); w: float [len(src), nf] non-negative."""
        n = len(src)
        idx = torch.arange(self.nf, device=DEV, dtype=torch.int32).repeat(n, 1)
        self.G1 = ByPre(idx, w, n, self.nf, 'motor G1 sound A1 -> force (gen-0, acquired)', birth=False,
                        eta=self.c['eta'], eta_m=self.c['eta_m'], gain=self.c['g_G1'], gamma=self.gam['motor'],
                        c=self.c)
        self.g1_src = src
        self.g1pre = Pop(n)

    def tables(self):
        return [self.G1] if self.G1 is not None else []

    def g1_drive(self, s1):
        if self.G1 is None:
            return torch.zeros(self.nf, device=DEV)
        return self.G1.drive(s1[self.g1_src])

    def step(self, a0, s1, ret_force, ret_stop):
        c, s = self.c, self.s
        brake = a0[self.brake_lines].float().sum() / self.n_brake
        stop = (brake >= self.thr_s) | ret_stop
        d = self.g1_drive(s1) + ret_force * (c['g_ret'] * self.thr_f)
        P = d.sum()
        lad = (P >= self.ladder).sum() if self.ladder is not None else torch.zeros((), device=DEV)
        nblock = self.n_tonic + lad
        inh = c['b_block'] * nblock / (8.0 * s) + c['c_stop'] * stop.float().sum() / (8.0 * s)
        force = d - inh > self.thr_f
        block = torch.zeros(self.nb, dtype=torch.bool, device=DEV)
        block[:self.n_tonic] = True
        if self.ladder is not None:
            block[self.n_tonic:] = P >= self.ladder
        self.block = block
        self.force.set(force)
        self.stop.set(stop)
        if self.g1pre is not None:
            self.g1pre.set(s1[self.g1_src])
        self.last = dict(P=P, d=d)
        on = torch.zeros(24, device=DEV).index_add_(0, self.muscle_of, force.float())
        return on >= math.ceil(s / 2)

    def learn(self):
        if self.G1 is None:
            return
        fw = self.c['fall_win']
        el = self.g1pre.elig_prev(fw)
        ro = self.g1pre.rose()
        r = self.force.rose()
        self.G1.learn(el, ro, r, r, self.force.age_prev <= fw)

    def pops(self):
        out = [self.force, self.stop]
        if self.g1pre is not None:
            out.append(self.g1pre)
        return out
