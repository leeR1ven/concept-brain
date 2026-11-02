"""Hippocampus [H1-H5]: copies H1 (one per chain source, own forward and return chains) -> own population Hs
(fixed random recode, X3) -> learned Hs <-> Hs and Hs -> H1 (fast rule, fastest decay). No birth Hs <-> Hs
recurrence, no quota, no store/recall switch. Hs activity: fixed threshold + same-tick inhibition equal to the
mean drive (X3's gain control). Recall lights H1 copies internally -> their returns light the source units."""
import torch

import geom
from common import DEV
from learning import ByPost, ByPre, Pop, topk_pos


class Hippocampus:
    def __init__(self, c, pos1, gam, seed):
        self.c = c
        s = c['s']
        n1, nh = len(pos1), c['nh_per_s'] * s
        self.n1, self.nh = n1, nh
        g = torch.Generator(device=DEV)
        g.manual_seed(seed + 700)
        ridx = torch.randint(0, n1, (nh, c['rec_fan']), generator=g, device=DEV)      # [H2] global random recode
        rw = 0.5 + torch.rand(ridx.shape, generator=g, device=DEV)
        self.rec = ByPost(ridx, rw / rw.sum(1, keepdim=True), n1, nh, 'hip recode H1->Hs')
        self.HH = ByPre(geom.random_lists(nh, nh, c['chh_per_s'] * s, seed + 710, exclude_self=True), 0.0, nh, nh,
                        'hip Hs<->Hs acquired', birth=False, eta=c['eta_h'], eta_m=c['eta_h_m'], gain=c['g_hh'],
                        gamma=gam['hip'], c=c)
        self.H1t = ByPre(geom.random_lists(nh, n1, c['ch1_per_s'] * s, seed + 720), 0.0, nh, n1,
                         'hip Hs->H1 acquired', birth=False, eta=c['eta_h'], eta_m=c['eta_h_m'], gain=c['g_h1'],
                         gamma=gam['hip'], c=c)
        self.patch = geom.patches(pos1, c['patch'], seed + 730)
        self.theta_hs = None
        self.H1, self.H1ext, self.Hs = Pop(n1), Pop(n1), Pop(nh)
        self.h_int = torch.zeros(n1, dtype=torch.bool, device=DEV)

    def tables(self):
        return [self.rec, self.HH, self.H1t]

    def plastic(self):
        return [self.HH, self.H1t]

    def inhib1(self, d):
        if self.c['inhib'] == 'P0':
            return d
        pid, n_p = self.patch
        s = torch.zeros(n_p, device=DEV).index_add_(0, pid, d)
        cnt = torch.zeros(n_p, device=DEV).index_add_(0, pid, torch.ones_like(d))
        return d - self.c['g_pm'] * (s / cnt.clamp_min(1))[pid]

    def calibrate(self, X):
        """X: bool [T, n1] birth chain stream. Fixed Hs threshold giving about kh active on average."""
        T = X.shape[0]
        D = torch.empty((T, self.nh), device=DEV)
        for a in range(0, T, 64):
            d = self.rec.drive_batch(X[a:a + 64].float())
            D[a:a + 64] = d - self.c['g_ih'] * d.mean(1, keepdim=True)
        lo, hi = float(D.min()), float(D.max())
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if (D > mid).float().sum(1).mean().item() > self.c['kh']:
                lo = mid
            else:
                hi = mid
        self.theta_hs = hi
        cnt = (D > hi).float().sum(1)
        return dict(theta=hi, mean=cnt.mean().item(), p10=cnt.quantile(0.1).item(), p90=cnt.quantile(0.9).item())

    def step(self, x_ext):
        ah = self.Hs.on                                                             # Hs(t-1), as a mask
        raw = self.H1t.drive(ah)
        sub = self.inhib1(raw)
        h_int = sub > self.c['theta_h1']
        self.dh1 = (raw, sub)                                                       # kept for diagnostics only
        x = x_ext | h_int
        d = self.rec.drive(self.H1.on.float()) + self.HH.drive(ah)
        e = d - self.c['g_ih'] * d.mean()
        hs = topk_pos(e, self.c['kh']) if self.c['hs_count'] == 'k' else e > self.theta_hs   # [fix B] k fixed (X6)
        self.H1ext.set(x_ext)
        self.H1.set(x)
        self.Hs.set(hs)
        self.h_int = h_int
        return h_int

    def learn(self):
        fw = self.c['fall_win']
        el = self.Hs.elig_prev(fw)
        ro = self.Hs.rose()
        r = self.Hs.rose()
        self.HH.learn(el, ro, r, r, self.Hs.age_prev <= fw)
        if self.c['post'] == 'chain':
            ev = self.H1ext.rose() & ~self.H1.prev
            rc = self.H1ext.age_prev <= fw
        else:
            ev = self.H1.rose()
            rc = self.H1.age_prev <= fw
        self.H1t.learn(el, ro, ev, ev, rc)

    def pops(self):
        return [self.H1, self.H1ext, self.Hs]
