"""X10 sizing v2 (designer 11-02): sound area of mostly feature units.
Each unit: K random local inputs (comparable weights) from the layer below, radius growing with depth;
threshold set from the unit's OWN birth drive: the (1 - r_l) quantile of its drive, r_l falling with depth
('>' at the quantile; if that never fires, '>=' when that stays inside the band). Reports band membership,
per-unit rates, silence vs speech chain counts, and code overlaps."""
import sys, time
import numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '../../x1g'))
import world as W

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
S = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FORM = sys.argv[3] if len(sys.argv) > 3 else 'K6'
t0 = time.time()
ph = W.Physics(); rng = np.random.default_rng(5)
SRC = []
def graded(self, srcs):
    SRC.append(list(srcs)); return W.render(self.ph, srcs, self.rng)
W.World._render = graded
wd = W.World(ph, seed=0, care='real', distract=True)
for t in range(T):
    wd.step(np.zeros(ph.n_mus, bool))
NC = ph.n_a
E = np.zeros((T, NC), np.float32); lab = np.full(T, -1); voi = np.full(T, -1)
for t, srcs in enumerate(SRC):
    e = np.zeros(NC, np.float32)
    for kind, payload, voice, loud in srcs:
        if kind == 's': c = ph.content[payload]
        elif kind == 'm':
            c = ph.content[payload].copy(); on = np.nonzero(c)[0]; c[rng.choice(on, 3, replace=False)] = False
        else: c = ph.garble_content(payload)
        g = 0.6 if voice == W.OWN else 1.0
        e[:ph.n_c] += loud * g * c * rng.uniform(0.85, 1.0, ph.n_c)
        e[ph.n_c + voice * ph.n_vu: ph.n_c + (voice + 1) * ph.n_vu] += loud * 0.9
        if kind == 's' and loud >= 1.0 and voice != W.OWN:
            lab[t] = payload; voi[t] = voice
    e += np.abs(rng.normal(0, 0.04, NC)); E[t] = np.minimum(e, 1.5)
L = 3 * S
th = np.sort(np.random.default_rng(11).uniform(0.12, 1.0, (NC, L)), 1)
up = E[:, :, None] >= th[None]
A0 = np.stack([up, ~up], 3).reshape(T, -1)
Y, Z, D = NC, 2 * L, 2
if FORM == 'K6':   K, DY, DZ = [6, 6, 6, 6], [1, 2, 4, 8], [2, 3 * S, 4 * S, 6 * S]
elif FORM == 'K4': K, DY, DZ = [4, 4, 4, 4], [1, 2, 4, 8], [2, 3 * S, 4 * S, 6 * S]
elif FORM == 'W6': K, DY, DZ = [6, 6, 6, 6], [4, 8, 16, 32], [2 * S, 3 * S, 4 * S, 6 * S]
elif FORM == 'K8': K, DY, DZ = [6, 8, 8, 8], [1, 2, 4, 8], [2, 3 * S, 4 * S, 6 * S]
R = [0.20, 0.12, 0.08, 0.05]
BAND = (0.002, 0.90)
prev = A0.astype(np.float32); prevD = 1
layers = []
for l in range(4):
    n = Y * Z * D; p = np.arange(n) // D; yy, zz = p // Z, p % Z
    g = np.random.default_rng(100 + l)
    idx = np.zeros((n, K[l]), int)
    for j in range(K[l]):
        y2 = np.clip(yy + g.integers(-DY[l], DY[l] + 1, n), 0, Y - 1)
        z2 = np.clip(zz + g.integers(-DZ[l], DZ[l] + 1, n), 0, Z - 1)
        idx[:, j] = (y2 * Z + z2) * prevD + g.integers(0, prevD, n)
    w = g.uniform(0.5, 1.5, (n, K[l])).astype(np.float32)
    d = np.einsum('tnk,nk->tn', prev[:, idx], w)
    q = np.quantile(d, 1 - R[l], axis=0)
    gt = (d > q).mean(0); ge = (d >= q).mean(0)
    use_ge = (gt < BAND[0]) & (ge <= BAND[1])
    act = np.where(use_ge[None], d >= q[None], d > q[None])
    layers.append(act); prev = act.astype(np.float32); prevD = D
sil = np.array([len(s) == 0 for s in SRC]); spk = lab >= 0
print(f'form {FORM} T={T} s={S} A0={A0.shape[1]} layer={Y*Z*D} sil_frac={sil.mean():.2f}')
tot = 0; cs = np.zeros(T)
for l, act in enumerate(layers):
    r = act.mean(0); ch = (r >= BAND[0]) & (r <= BAND[1])
    sp = ((r >= 0.01) & (r <= 0.30)).mean()
    cnt = act[:, ch].sum(1); cs += cnt; tot += ch.sum()
    print(f'  A{l+1}: in band {ch.mean():.2f} ({ch.sum()}), dead {(r < BAND[0]).mean():.2f}, always-on {(r > BAND[1]).mean():.2f}, '
          f'rate 1-30% {sp:.2f}, median rate of chain units {np.median(r[ch]):.3f}, layer frac {act.mean():.3f}; '
          f'chain units on: silence {cnt[sil].mean():.0f} speech {cnt[spk].mean():.0f}')
print(f'  total chain units {tot}; on: silence {cs[sil].mean():.0f}+-{cs[sil].std():.0f} speech {cs[spk].mean():.0f}+-{cs[spk].std():.0f}; '
      f'PFC L1 frac silence {cs[sil].mean()/tot:.3f} speech {cs[spk].mean()/tot:.3f}')
def jac(a, b): return (a & b).sum() / max(1, (a | b).sum())
for l in (1, 3):
    act = layers[l]; r = act.mean(0); ch = (r >= BAND[0]) & (r <= BAND[1])
    codes = {}
    for t in np.nonzero(spk)[0]: codes.setdefault((lab[t], voi[t]), []).append(act[t, ch])
    m = {k: np.mean(v, 0) >= 0.5 for k, v in codes.items() if len(v) >= 6}
    keys = list(m); same, diff = [], []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]: (same if a[0] == b[0] else diff).append(jac(m[a], m[b]))
    silc = act[sil][:, ch].mean(0) >= 0.5
    print(f'  A{l+1} codes: J same syl/other voice {np.mean(same):.2f}  J diff syl {np.mean(diff):.2f}  J(syl, silence) {np.mean([jac(m[k], silc) for k in keys]):.2f}  code size {np.mean([m[k].sum() for k in keys]):.0f}  silence code {silc.sum()}')
print(f'{time.time()-t0:.1f}s')
