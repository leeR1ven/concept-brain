"""The one learning rule, the one decay, two-layer weights, and the sparse tables they act on [R1-R4, P9].

Tables are never dense N x N:
  ByPost  fixed fan-in table (birth, never learns): idx[n_post, F] -> drive = sum_k w * x[idx]
  ByPre   fixed fan-out table, padded with a sink index n_post: idx[n_pre, C] (int32)
          birth=True : fixed weights, raw drive
          birth=False: acquired weights (>= 0), learned by the rule, decayed, propagated with two-layer weights
"""
import torch

from common import DEV


class ByPost:
    def __init__(self, idx, w, n_pre, n_post, name):
        self.idx = torch.as_tensor(idx, dtype=torch.int64, device=DEV)
        self.w = torch.as_tensor(w, dtype=torch.float32, device=DEV)
        self.n_pre, self.n_post, self.name = n_pre, n_post, name
        assert self.idx.shape == (n_post, self.idx.shape[1])

    def drive(self, x):
        """x: float [n_pre] (0/1)"""
        return (x[self.idx] * self.w).sum(1)

    def drive_batch(self, X):
        """X: float [T, n_pre] -> [T, n_post] (time-parallel, birth calibration only)"""
        return (X[:, self.idx] * self.w).sum(-1)

    def shape(self):
        return tuple(self.idx.shape)


DENSE_LIMIT = 1.5e8          # tables up to this many entries are scanned whole each tick (no host sync)


class ByPre:
    def __init__(self, idx, w, n_pre, n_post, name, birth, eta=0.0, eta_m=0.0, gain=1.0, gamma=1.0, c=None):
        self.idx = torch.as_tensor(idx, dtype=torch.int32, device=DEV)        # sink = n_post marks padding
        self.w = torch.as_tensor(w, dtype=torch.float32, device=DEV)
        if self.w.dim() == 0:
            self.w = torch.full(self.idx.shape, float(self.w), device=DEV)
        assert self.idx.shape == self.w.shape and self.idx.shape[0] == n_pre
        self.n_pre, self.n_post, self.name, self.birth = n_pre, n_post, name, birth
        self.w = self.w * (self.idx < n_post)
        self.eta, self.eta_m, self.gain, self.gamma = eta, eta_m, gain, gamma
        self.c = c
        limit = c['dense_limit'] if c is not None else DENSE_LIMIT
        self.dense = self.idx.numel() <= limit
        self.idx_l = self.idx.reshape(-1).long() if self.dense else None
        if not birth:
            self.recount()

    def shape(self):
        return tuple(self.idx.shape)

    # ---------------------------------------------------------------- two-layer weights [P9]
    def recount(self):
        """exact out / in totals (called at build and after every decay pass)"""
        self.out = self.w.sum(1)
        self.inn = torch.zeros(self.n_post + 1, device=DEV)
        step = max(1, (1 << 24) // max(1, self.idx.shape[1]))
        for a in range(0, self.n_pre, step):
            self.inn.index_add_(0, self.idx[a:a + step].reshape(-1).long(), self.w[a:a + step].reshape(-1))

    def drive(self, act):
        """act: bool mask [n_pre] of the pre units that are on (or int64 indices). Returns the drive into every
        post unit. Small tables are scanned whole (no host sync); large ones gather the active rows only."""
        out = torch.zeros(self.n_post + 1, device=DEV)
        if self.dense:
            if act.dtype != torch.bool:
                m = torch.zeros(self.n_pre, dtype=torch.bool, device=DEV)
                m[act] = True
                act = m
            xf = act.float()[:, None]
            if self.birth:
                v = self.w * xf
            else:
                wr = self.c['w_ref']
                v = self.gain * self.w * xf / torch.sqrt(self.out.clamp_min(wr)[:, None] *
                                                         self.inn[self.idx_l].view(self.idx.shape).clamp_min(wr))
            out.index_add_(0, self.idx_l, v.reshape(-1))
            return out[:-1]
        if act.dtype == torch.bool:
            act = torch.nonzero(act).squeeze(1)
        if act.numel() == 0:
            return out[:-1]
        ix = self.idx[act].long()
        w = self.w[act]
        if self.birth:
            v = w
        else:
            wr = self.c['w_ref']
            v = self.gain * w / torch.sqrt(self.out[act].clamp_min(wr)[:, None] * self.inn[ix].clamp_min(wr))
        out.index_add_(0, ix.reshape(-1), v.reshape(-1))
        return out[:-1]

    # ---------------------------------------------------------------- the rule [R1, R2, R4]
    def plast(self, w):
        return (1.0 / (1.0 + w / self.c['w_ref'])).clamp_min(self.c['floor'])

    def _apply(self, rows, dw):
        """add dw to rows (unique), keep acquired weights >= 0, and update the two-layer totals exactly"""
        old = self.w[rows]
        new = (old + dw).clamp_min(0.0)
        d = new - old
        self.w[rows] = new
        self.out[rows] += d.sum(1)
        self.inn.index_add_(0, self.idx[rows].reshape(-1).long(), d.reshape(-1))

    def learn(self, pre_elig, pre_rose, post_event, post_rose, post_recent):
        """Held-form temporal Hebbian rule (ARCH 2.8), one tick:
           strengthen A->B when B's event happens now and A was on, held, or fell <= fall_win ticks before;
           co-rising A and B strengthen;
           weaken A->B when A rises now while B was on (or fell <= fall_win ticks before) and B did not rise now.
           A fall changes nothing. Growth, weakening and decay are all slowed by the concave factor.
        pre_elig, pre_rose: bool masks [n_pre]; post_*: bool [n_post] (a False sink is appended here)."""
        if self.birth:
            return
        f = torch.zeros(1, dtype=torch.bool, device=DEV)
        ev = torch.cat([post_event, f])
        ro = torch.cat([post_rose, f])
        rc = torch.cat([post_recent, f])
        if self.dense:                                        # whole table, no host sync
            sh = self.idx.shape
            evx, rox, rcx = ev[self.idx_l].view(sh), ro[self.idx_l].view(sh), rc[self.idx_l].view(sh)
            pw = self.plast(self.w)
            el, rs = pre_elig[:, None], pre_rose[:, None]
            dw = self.eta * pw * ((el & evx) | (rs & rox)).float() - self.eta_m * pw * (rs & rcx & ~rox).float()
            new = (self.w + dw).clamp_min(0.0)
            d = new - self.w
            self.w = new
            self.out += d.sum(1)
            self.inn.index_add_(0, self.idx_l, d.reshape(-1))
            return
        pre_elig = torch.nonzero(pre_elig).squeeze(1)
        pre_rose = torch.nonzero(pre_rose).squeeze(1)
        if pre_elig.numel():
            m = ev[self.idx[pre_elig].long()]
            dw = self.eta * self.plast(self.w[pre_elig]) * m
            self._apply(pre_elig, dw)
        if pre_rose.numel():
            ix = self.idx[pre_rose].long()
            co, wk = ro[ix], rc[ix] & ~ro[ix]
            pw = self.plast(self.w[pre_rose])
            dw = self.eta * pw * co - self.eta_m * pw * wk
            self._apply(pre_rose, dw)

    # ---------------------------------------------------------------- the one decay [R3]
    def decay(self):
        if self.birth:
            return
        step = max(1, (1 << 24) // max(1, self.idx.shape[1]))          # row chunks: bounded temporaries
        for a in range(0, self.n_pre, step):
            w = self.w[a:a + step]
            w -= (1.0 - self.gamma) * w * self.plast(w)
            w[w < self.c['prune']] = 0.0
        self.recount()


class Pop:
    """state of one population for the rule: on(t), on(t-1), ticks since last on (0 = on now)."""

    def __init__(self, n):
        self.n = n
        self.on = torch.zeros(n, dtype=torch.bool, device=DEV)
        self.prev = torch.zeros(n, dtype=torch.bool, device=DEV)
        self.age = torch.full((n,), 99, dtype=torch.int16, device=DEV)
        self.age_prev = torch.full((n,), 99, dtype=torch.int16, device=DEV)

    def set(self, new):
        self.prev = self.on
        self.age_prev = self.age
        self.on = new
        self.age = torch.where(new, torch.zeros_like(self.age), (self.age + 1).clamp_max(99))

    def rose(self):
        return self.on & ~self.prev

    def elig_prev(self, win):
        """on at t-1, or fell within win ticks before t-1"""
        return self.age_prev <= win

    def state(self):
        return [self.on.clone(), self.prev.clone(), self.age.clone(), self.age_prev.clone()]

    def load(self, st):
        self.on, self.prev, self.age, self.age_prev = [x.clone() for x in st]
