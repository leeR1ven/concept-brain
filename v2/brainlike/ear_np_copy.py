# numpy replica of brain/net3d.py Net3D (fixed-fraction top-k per layer) + SenseArea muting + Stream timing.
import json, sys, numpy as np
from voice_copy import Voice, Stream

SOUND, PAUSE = 3, 4


class Net3D:
    def __init__(self, H, W, levels=10, zs=(40, 60, 120, 240), frac=(0.4, 0.3, 0.2, 0.1), fan_in=4, alpha=0.01,
                 bias_scale=0.5, spare=0.25, seed=0):
        self.H, self.W, self.levels, self.frac = H, W, levels, frac
        rng = np.random.default_rng(seed)
        z0 = levels * 2
        self.dims = [(H, W, z0)] + [(H, W, z) for z in zs]
        self.idx, self.w, self.b, self.cap, self.start, alive = [], [], [], [], [], []
        n_total = 0
        for l in range(1, len(self.dims)):
            prev, out = self.dims[l - 1], self.dims[l]
            x, y, z = np.meshgrid(np.arange(out[0]), np.arange(out[1]), np.arange(out[2]), indexing="ij")
            oc = np.stack([x, y, z], -1).reshape(-1, 3); n = len(oc)
            cap = n + int(round(n * spare))
            selfc = oc.copy(); selfc[:, 2] = np.round(oc[:, 2] * (prev[2] - 1) / max(out[2] - 1, 1)).astype(int)
            idx = np.zeros((cap, fan_in), np.int64); w = np.zeros((cap, fan_in), np.float32); b = np.zeros(cap, np.float32)
            idx[:n, 0] = np.ravel_multi_index(selfc.T, prev); w[:n, 0] = 1.0
            for j in range(1, fan_in):
                nb = selfc.copy()
                nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
                nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
                nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
                idx[:n, j] = np.ravel_multi_index(nb.T, prev)
                w[:n, j] = (rng.random(n) * 0.5 + 0.5) * alpha
            b[:n] = rng.random(n) * bias_scale
            a = np.zeros(cap, bool); a[:n] = True
            self.idx.append(idx); self.w.append(w); self.b.append(b); alive.append(a)
            self.cap.append(cap); self.start.append(n_total); n_total += cap
        self.n = n_total
        self.layer = np.concatenate([np.full(c, l) for l, c in enumerate(self.cap)])
        self.alive = np.concatenate(alive)

    def encode(self, img):
        img = img.reshape(self.H, self.W, -1)
        up = img[..., None] > np.arange(self.levels)
        return np.stack([up, ~up], -1).reshape(-1)

    def respond(self, lines, extra=None):
        h = lines.astype(np.float32)
        out = np.zeros(self.n, bool)
        for l, (idx, w, b, s0, cap) in enumerate(zip(self.idx, self.w, self.b, self.start, self.cap)):
            s = (h[idx] * w).sum(1) + b
            al = self.alive[s0:s0 + cap]
            k = int(round(self.frac[l] * al.sum()))
            if extra is not None:
                s = s + extra[s0:s0 + cap]
            s = np.where(al, s, -np.inf)
            on = np.zeros(cap, bool)
            on[np.argpartition(-s, k)[:k]] = True
            out[s0:s0 + cap] = on
            h = on.astype(np.float32)
        return out


class World:
    """Ear front end exactly as talk.py: Voice(chars, seed), Stream(step=T//SOUND); ear cortex = layers >= 1."""

    def __init__(self, chars, seed=0):
        self.v = Voice(chars, seed=seed)
        self.net = Net3D(self.v.F, self.v.T, seed=seed)
        self.stream = Stream(self.v, step=self.v.T // SOUND)
        self.cortex = (self.net.layer >= 1) & self.net.alive
        self.x = np.ones(self.net.n)            # output reserve (U 0.5, tau 6), as Region.deplete
        self.s = np.zeros(self.net.n, bool); self.prev = self.s.copy()

    def hear(self, other=None, self_=None, new_other=False):
        win = self.stream.tick(other, self_, new_other)
        return self.net.encode(win)

    def step(self, lines):
        """one tick of the ear: returns (s_prev, x_prev) used by downstream currents this tick, then updates."""
        s_prev, x_prev = self.s, self.x.copy()
        self.prev, self.s = self.s, self.net.respond(lines) & self.net.alive
        # deplete for neurons that sent current this tick (on at t-1), then recover
        self.x = np.where(self.prev, self.x * 0.5, self.x); self.x = self.x + (1 - self.x) / 6.0
        return s_prev, x_prev

    def warmup_mute(self, n=60, seed=1, thr=0.6):
        rng = np.random.default_rng(seed); chars = self.v.chars; use = np.zeros(self.net.n); k = 0
        for _ in range(n):
            ch = chars[rng.integers(len(chars))]
            for j in range(SOUND):
                self.step(self.hear(other=ch, new_other=(j == 0))); use += self.s; k += 1
            for _ in range(PAUSE):
                self.step(self.hear()); use += self.s; k += 1
        u = use / k
        self.muted = self.cortex & (u > thr)
        self.feat = self.cortex & ~self.muted        # sends chains / feeds F and T
        return int(self.muted.sum())


def load_pairs():
    pool = [json.loads(l) for l in open("/root/work/corpus/dialogs.jsonl", encoding="utf-8")]
    pool = [c for c in pool if len(c) >= 2]
    chars = sorted({ch for c in pool for u in c for ch in u})
    return pool, chars
