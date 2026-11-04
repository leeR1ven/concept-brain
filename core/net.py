"""Neurons, the one local-net generator, learned links, the one learning rule and the one decay.
ARCHITECTURE.md 二 (neurons and nets) and 四 (learning)."""
from collections import defaultdict

import numpy as np
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Pop:
    """One population: on now, on one tick ago, and ticks since it was last on (0 = on now)."""

    def __init__(self, n):
        self.n = n
        self.on = torch.zeros(n, dtype=torch.bool, device=DEV)
        self.prev = torch.zeros(n, dtype=torch.bool, device=DEV)
        self.age = torch.full((n,), 99, dtype=torch.int16, device=DEV)
        self.age_prev = self.age.clone()

    def set(self, new):
        self.prev, self.age_prev = self.on, self.age
        self.on = new
        self.age = torch.where(new, torch.zeros_like(self.age), (self.age + 1).clamp_max(99))

    def rose(self):
        return self.on & ~self.prev

    def held(self, win):
        """on at t-1, or went off at most `win` ticks before t-1 (ARCH 四: just lit, held on, or just off)"""
        return self.age_prev <= win

    def state(self):
        return [x.clone() for x in (self.on, self.prev, self.age, self.age_prev)]

    def load(self, st):
        self.on, self.prev, self.age, self.age_prev = [x.clone() for x in st]


# ------------------------------------------------------------------------------------------- birth wiring
def local_net(pre, post, K, dy, dz, rng):
    """ARCH 二.4, the one generator for every area: each post unit takes K random pre units near it (same block,
    |dy|, |dz| windows on the grid), with comparable fixed weights U(0.5, 1.5).
    pre, post: int arrays [n, 3] of (block, y, z) grid positions. Returns idx [n_post, K], w [n_post, K]."""
    cell = defaultdict(list)
    for i, p in enumerate(map(tuple, pre)):
        cell[p].append(i)
    cell = {k: np.asarray(v) for k, v in cell.items()}
    by_block = defaultdict(list)
    for i, b in enumerate(pre[:, 0]):
        by_block[int(b)].append(i)
    idx = np.empty((len(post), K), np.int64)
    for j, (b, y, z) in enumerate(post):
        for k in range(K):
            for _ in range(64):
                c = cell.get((b, y + int(rng.integers(-dy, dy + 1)), z + int(rng.integers(-dz, dz + 1))))
                if c is not None:
                    idx[j, k] = c[rng.integers(len(c))]
                    break
            else:                                           # nothing in the window: the nearest unit of the block
                cand = np.asarray(by_block[int(b)])
                d = np.abs(pre[cand, 1] - y) / (dy + 0.5) + np.abs(pre[cand, 2] - z) / (dz + 0.5)
                idx[j, k] = cand[np.argmin(d + 1e-3 * rng.random(len(cand)))]
    return idx, rng.uniform(0.5, 1.5, idx.shape).astype(np.float32)


def own_thresholds(D, rate):
    """ARCH 二.4: a unit's threshold comes from its own received signal: over the birth ticks D [T, n] it fires on
    about `rate` of them. When the drive is tied at that level, the unit fires at > or at >= the level, whichever
    comes closer to `rate`."""
    T = D.shape[0]
    k = max(1, min(T, int(np.ceil((1.0 - rate) * T))))
    q = torch.cat([torch.kthvalue(D[:, a:a + 4096], k, dim=0).values for a in range(0, D.shape[1], 4096)])
    gt = (D > q).float().mean(0)
    ge = (D >= q).float().mean(0)
    return torch.where((ge - rate).abs() < (gt - rate).abs(), q - 1e-4, q)


class Fixed:
    """Fixed fan-in links (birth wiring, never learn): idx [n_post, K], w [n_post, K]."""

    def __init__(self, idx, w):
        self.idx = torch.as_tensor(idx, dtype=torch.int64, device=DEV)
        self.w = torch.as_tensor(w, dtype=torch.float32, device=DEV)

    def drive(self, x):
        return (x.float()[self.idx] * self.w).sum(1)

    def drive_batch(self, X):
        return torch.cat([(X[a:a + 64].float()[:, self.idx] * self.w).sum(-1) for a in range(0, X.shape[0], 64)])


def topk_on(d, k):
    """the k most driven units with positive drive (ARCH 六: fixed count per tick)"""
    out = torch.zeros(d.shape[0], dtype=torch.bool, device=DEV)
    k = int(k)
    if k > 0:
        v, i = torch.topk(d, min(k, d.shape[0]))
        out[i[v > 0]] = True
    return out


# ------------------------------------------------------------------------------------------- learned links
class Links:
    """Links from each pre unit to a list of possible post units (padded with the sink n_post).
    learned=False: fixed birth weights. learned=True: grown by the one rule and decayed by the one decay.
    The drive into a post unit is gain x the sum of the weights from the pre units that are on.
    plain=True (the hippocampus, ARCH 三 海马体): its own growth rule. While A is on, A->B grows by the full step
    eta when B rises (in order, no concave slowing), so one experience leaves a strong link."""

    def __init__(self, idx, w, n_post, c, learned=True, gain=1.0, gamma=1.0, name='', plain=False):
        self.idx = torch.as_tensor(idx, dtype=torch.int64, device=DEV)
        self.w = torch.as_tensor(w, dtype=torch.float32, device=DEV).expand(self.idx.shape).clone()
        self.w *= (self.idx < n_post)
        self.n_pre, self.n_post = self.idx.shape[0], n_post
        self.c, self.learned, self.gain, self.gamma, self.name, self.plain = c, learned, gain, gamma, name, plain
        self.log = None                                     # reward tool: list of this tick's applied changes

    def drive(self, act):
        """act: bool [n_pre] -> drive into every post unit"""
        out = torch.zeros(self.n_post + 1, device=DEV)
        a = torch.nonzero(act).squeeze(1)
        if a.numel():
            out.index_add_(0, self.idx[a].reshape(-1), (self.gain * self.w[a]).reshape(-1))
        return out[:-1]

    def plast(self, w):
        """concave: the bigger a link, the harder it changes; about 1 % plasticity is always kept (ARCH 四)"""
        if self.plain:
            return torch.ones_like(w)
        return ((1.0 + w / self.c['w_ref']) ** -self.c['pow']).clamp_min(self.c['floor'])

    def apply(self, rows, dw, log=True):
        old = self.w[rows]
        new = (old + dw).clamp_min(0.0)
        d = new - old
        self.w[rows] = new
        if log and self.log is not None:
            self.log.append((rows, d))

    def learn(self, pre, b_rose, b_on, eta):
        """ARCH 四, the one rule. pre: Pop; b_rose: bool [n_post] post units that rose now; b_on: bool [n_post] post
        units on at t-1 or just off. When B rises while A is on (just lit, held, or just off), A->B grows; A and B
        rising together grow; when A rises again while B is still on and B did not rise, A->B weakens (at half the
        rate). B going off changes nothing."""
        if not self.learned:
            return
        fw = self.c['fall_win']
        f = torch.zeros(1, dtype=torch.bool, device=DEV)
        b_rose = torch.cat([b_rose, f])
        b_on = torch.cat([b_on, f])
        a = torch.nonzero(pre.held(fw) | pre.rose()).squeeze(1)
        if a.numel() == 0:
            return
        ix = self.idx[a]
        if self.plain:                                      # hippocampus: what followed grows by the full step
            self.apply(a, eta * b_rose[ix].float())
            return
        a_rose = pre.rose()[a][:, None]
        grow = b_rose[ix]
        weaken = a_rose & b_on[ix] & ~b_rose[ix]
        p = self.plast(self.w[a])
        self.apply(a, eta * p * (grow.float() - weaken.float()))

    def decay(self):
        """ARCH 四, the one decay: learned links are multiplied by gamma (slowed by the same concave factor); links
        that fall below `prune` are removed. A table with a cap (the PFC last layer's own links): all links into one
        unit together push it at most `cap`; above that they are scaled down together."""
        if not self.learned:
            return
        self.w -= (1.0 - self.gamma) * self.w * self.plast(self.w)
        self.w[self.w < self.c['prune']] = 0.0
        cap = getattr(self, 'cap', None)
        if cap:                                        # all links into a unit together push it at most `cap`
            tot = torch.zeros(self.n_post + 1, device=DEV).index_add_(0, self.idx.reshape(-1), self.w.reshape(-1))
            self.w *= (cap / (self.gain * tot).clamp_min(1e-12)).clamp(max=1.0)[self.idx]


def radius_lists(pos_pre, pos_post, r, exclude_self=False):
    """for every pre unit, all post units within radius r on the sheet (padded with the sink len(pos_post))"""
    lists = []
    for a in range(0, len(pos_pre), 1024):
        m = torch.cdist(pos_pre[a:a + 1024], pos_post) < r
        if exclude_self:
            ar = torch.arange(a, min(a + 1024, len(pos_pre)), device=DEV)
            m[ar - a, ar] = False
        lists += [torch.nonzero(row).squeeze(1) for row in m]
    w = max(1, max(len(x) for x in lists))
    idx = torch.full((len(lists), w), len(pos_post), dtype=torch.int64, device=DEV)
    for i, x in enumerate(lists):
        idx[i, :len(x)] = x
    return idx


def random_lists(n_pre, n_post, width, rng, exclude_self=False):
    """`width` random possible partners per pre unit (no topography)"""
    idx = np.empty((n_pre, width), np.int64)
    for i in range(n_pre):
        pool = rng.choice(n_post - (1 if exclude_self else 0), width, replace=False)
        if exclude_self:
            pool[pool >= i] += 1
        idx[i] = pool
    return idx
