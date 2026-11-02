# Fixed ear front end (brainstem / cochlear-nucleus analogue): exact replica of Net3D layer 0 of brain/net3d.py
# (same constructor + seed as ear0.py / archv1.py), FEAT0 = layer-0 units with warm-up usage <= 0.6, window energy.
# Nothing learned reaches this layer.  Used by the toy world and by the agent (the agent only sees the (F,T) window).
import numpy as np
from ear_np_copy import Net3D

SOUND = 3


class Ear:
    def __init__(self, F=8, T=6, seed=0):
        net = Net3D(F, T, seed=seed)
        self.idx, self.w, self.b = net.idx[0], net.w[0], net.b[0]
        cap = net.cap[0]
        self.alive = net.alive[:cap].copy()
        self.k = int(round(net.frac[0] * self.alive.sum()))
        self.n = cap
        self.encode = net.encode
        self.feat = None                         # FEAT0 indices (set by warmup)

    def layer0(self, win):
        h = self.encode(win).astype(np.float32)
        s = (h[self.idx] * self.w).sum(1) + self.b
        s = np.where(self.alive, s, -np.inf)
        on = np.zeros(self.n, bool)
        on[np.argpartition(-s, self.k)[:self.k]] = True
        return on & self.alive

    def warmup(self, voice, chars, n=60, seed=1, pause=4):
        """FEAT0 exactly as ARCH_v1: 60 random chars (other voice), each followed by `pause` silent ticks."""
        from voice_copy import Stream
        st = Stream(voice, step=voice.T // SOUND)
        rng = np.random.default_rng(seed)
        use = np.zeros(self.n); k = 0
        for _ in range(n):
            ch = chars[rng.integers(len(chars))]
            for j in range(SOUND):
                use += self.layer0(st.tick(ch, None, j == 0)); k += 1
            for _ in range(pause):
                use += self.layer0(st.tick(None, None)); k += 1
        self.feat = np.nonzero(self.alive & (use / k <= 0.6))[0]
        return len(self.feat)

    def feat0(self, win):
        return self.layer0(win)[self.feat]
