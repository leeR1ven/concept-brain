"""Sound area [S1-S6, revised 11-02 by the designer: the layers below A0 are mostly feature units].
A0 = thermometer pairs from the cochlea (exactly half on, no chains).
A1..A4: each unit takes K random local inputs (comparable weights U(0.5, 1.5)) from the layer below, within a
radius that widens with depth, so deeper units cover more channels [S2].
Threshold [S4]: set from the unit's OWN received signal during the birth period (ARCH 2.1 "每个神经元的门槛按它自己
收到的信号来定"): the (1 - r_l) quantile of its own birth drive, with r_l falling with depth (the fraction rises with
depth). Fire iff drive > threshold; if that would never fire, fire at >= the quantile when that rate stays inside the
chain band (e.g. a silence-class unit). Birth wiring is lineage-fixed and never learns [S6].
Timing: A1(t) = f(A0(t)); Al(t) = f(A(l-1)(t-1)). Returns add g_ret x (own threshold) to a chain unit's drive [C1]."""
import numpy as np
import torch

from common import DEV
from learning import ByPost


class SoundArea:
    def __init__(self, coch, c):
        self.c = c
        Y, Z, D, s = coch.NC, coch.Z, c['units_per_pos'], c['s']
        self.Y, self.Z, self.D = Y, Z, D
        self.n0 = coch.n
        g = np.random.default_rng(c['lineage_seed'] + 100)
        K = c['sound_fan']
        DY = c['sound_dy']
        DZ = [dz * (s if l > 0 else 1) for l, dz in enumerate(c['sound_dz'])]
        self.layers, self.pos, self.wmin = [], [], []
        prevD, n_prev = 1, self.n0
        for l in range(c['n_layers']):
            n = Y * Z * D
            p = np.arange(n) // D
            yy, zz = p // Z, p % Z
            idx = np.zeros((n, K), np.int64)
            for j in range(K):
                y2 = np.clip(yy + g.integers(-DY[l], DY[l] + 1, n), 0, Y - 1)
                z2 = np.clip(zz + g.integers(-DZ[l], DZ[l] + 1, n), 0, Z - 1)
                idx[:, j] = (y2 * Z + z2) * prevD + g.integers(0, prevD, n)
            w = g.uniform(0.5, 1.5, (n, K)).astype(np.float32)
            self.layers.append(ByPost(idx, w, n_prev, n, f'sound A{l + 1}'))
            self.wmin.append(torch.as_tensor(w.min(1), device=DEV))
            self.pos.append(torch.as_tensor(np.stack([yy / max(1, Y - 1), zz / max(1, Z - 1)], 1),
                                            dtype=torch.float32, device=DEV))
            prevD, n_prev = D, n
        self.n = [ly.n_post for ly in self.layers]
        self.thr = [None] * c['n_layers']
        self.ret_w = [None] * c['n_layers']
        self.state = [torch.zeros(n, dtype=torch.bool, device=DEV) for n in self.n]

    # ------------------------------------------------------------------ birth calibration [S4]
    def calibrate(self, A0):
        """A0: bool [T, n0] birth stream (learning off, no returns). Sets every unit's own threshold from its own
        birth drive. Returns the time-parallel layer states [T, n_l] (bool, on DEV) for the chain band [S5]."""
        lo_b, hi_b = self.c['band']
        states = []
        x = A0.to(DEV)
        for l, ly in enumerate(self.layers):
            T = x.shape[0]
            dr = torch.empty((T, ly.n_post), device=DEV)
            for a in range(0, T, 512):
                dr[a:a + 512] = ly.drive_batch(x[a:a + 512].float())
            k = max(1, min(T, int(np.ceil((1.0 - self.c['sound_rates'][l]) * T))))
            q = torch.empty(ly.n_post, device=DEV)
            for a in range(0, ly.n_post, 2048):
                q[a:a + 2048] = torch.kthvalue(dr[:, a:a + 2048], k, dim=0).values
            gt = (dr > q).float().mean(0)
            ge = (dr >= q).float().mean(0)
            use_ge = (gt < lo_b) & (ge <= hi_b)
            thr = torch.where(use_ge, q - 1e-4, q)
            self.thr[l] = thr
            self.ret_w[l] = self.c['g_ret'] * torch.maximum(thr, self.wmin[l])
            s_l = dr > thr
            states.append(s_l)
            x = s_l
            del dr
        return states

    # ------------------------------------------------------------------ one tick
    def step(self, a0, ret):
        """a0: bool [n0] this tick's input; ret: list of float [n_l] return inputs (0/1). Synchronous update."""
        prev = [a0] + self.state[:-1]
        new = []
        for l, ly in enumerate(self.layers):
            d = ly.drive(prev[l].float()) + ret[l] * self.ret_w[l]
            new.append(d > self.thr[l])
        self.state = new
        return new
