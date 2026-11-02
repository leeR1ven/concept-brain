"""X10 sizing: sound area statistics at s=1 (no learning, child silent).
graded channel energy -> thermometer pairs (always half on) -> 4 local3d-style dom1 layers with per-neuron
threshold theta_l * sum(w_in) (calibrated once to target sparsity) -> chain band by firing rate.
Reports active chain-bearing counts in silence vs speech, per layer, and code overlaps."""
import sys, time
import numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '../../x1g'))
import world as W

T = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
S = int(sys.argv[2]) if len(sys.argv) > 2 else 1
t0 = time.time()
ph = W.Physics()
rng = np.random.default_rng(5)
SRC = []
def graded(self, srcs):
    SRC.append(list(srcs))
    return W.render(self.ph, srcs, self.rng)
W.World._render = graded
wd = W.World(ph, seed=0, care='real', distract=True)
for t in range(T):
    wd.step(np.zeros(ph.n_mus, bool))
NC = ph.n_a                                   # 180 channels (150 content + 30 voice)
E = np.zeros((T, NC), np.float32)
lab = np.full(T, -1); voi = np.full(T, -1)
for t, srcs in enumerate(SRC):
    e = np.zeros(NC, np.float32)
    for kind, payload, voice, loud in srcs:
        if kind == 's':
            c = ph.content[payload].copy()
        elif kind == 'm':
            c = ph.content[payload].copy(); on = np.nonzero(c)[0]; c[rng.choice(on, 3, replace=False)] = False
        else:
            c = ph.garble_content(payload)
        g = 0.6 if voice == W.OWN else 1.0       # own voice: lower air-borne content level (registered choice)
        e[:ph.n_c] += loud * g * c * rng.uniform(0.85, 1.0, ph.n_c)
        e[ph.n_c + voice * ph.n_vu: ph.n_c + (voice + 1) * ph.n_vu] += loud * 0.9
        if kind == 's' and loud >= 1.0 and voice != W.OWN:
            lab[t] = payload; voi[t] = voice
    e += np.abs(rng.normal(0, 0.04, NC))
    E[t] = np.minimum(e, 1.5)
L = 3 * S
th = np.sort(np.random.default_rng(11).uniform(0.12, 1.0, (NC, L)), 1)   # lineage-fixed level thresholds
up = E[:, :, None] >= th[None]                                         # T x NC x L
A0 = np.stack([up, ~up], 3).reshape(T, NC, 2 * L)                      # sheet y = channel, z = (level, polarity)
A0f = A0.reshape(T, -1)
assert np.all(A0f.sum(1) == NC * L)
Y, Z = NC, 2 * L
D = 2                                                                  # units per sheet position in A1..A4
TARGET = [0.25, 0.15, 0.10, 0.06]
fan, alpha = 4, 0.05
prev = A0f.astype(np.float32); prevD = 1
layers = []
for l in range(4):
    n = Y * Z * D
    pos = np.arange(n) // D
    yy, zz = pos // Z, pos % Z
    idx = np.zeros((n, fan), int); w = np.zeros((n, fan), np.float32)
    lrng = np.random.default_rng(100 + l)
    idx[:, 0] = (yy * Z + zz) * prevD + lrng.integers(0, prevD, n); w[:, 0] = 1.0
    for j in range(1, fan):
        y2 = np.clip(yy + lrng.integers(-1, 2, n), 0, Y - 1); z2 = np.clip(zz + lrng.integers(-2, 3, n), 0, Z - 1)
        idx[:, j] = (y2 * Z + z2) * prevD + lrng.integers(0, prevD, n); w[:, j] = alpha * lrng.uniform(0.5, 1, n)
    drive = np.einsum('tnf,nf->tn', prev[:, idx], w) / w.sum(1)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if (drive > mid).mean() > TARGET[l]: lo = mid
        else: hi = mid
    act = drive > hi
    layers.append((hi, act))
    prev = act.astype(np.float32); prevD = D
sil = np.array([len(s) == 0 for s in SRC]); spk = lab >= 0
print(f'T={T} s={S} A0={A0f.shape[1]} layer size={Y*Z*D} sil_frac={sil.mean():.2f} adult_speech_frac={spk.mean():.2f}')
for band in [(0.002, 0.6), (0.002, 0.9), (0.002, 0.95), (0.002, 0.99)]:
    tot_n = 0; ts = np.zeros(T)
    rows = []
    for l, (thl, act) in enumerate(layers):
        r = act.mean(0); ch = (r >= band[0]) & (r <= band[1])
        a = act[:, ch]; cnt = a.sum(1); ts += cnt; tot_n += ch.sum()
        rows.append(f'L{l+1}: theta {thl:.3f} frac {act.mean():.3f} chains {ch.sum()} ({ch.mean():.2f}) on: sil {cnt[sil].mean():.0f} speech {cnt[spk].mean():.0f} (mean rate of chain units {r[ch].mean():.3f})')
    print(f'band {band}: total chains {tot_n}; chain units on: silence {ts[sil].mean():.0f}+-{ts[sil].std():.0f}, adult speech {ts[spk].mean():.0f}+-{ts[spk].std():.0f}, ratio {ts[spk].mean()/max(1,ts[sil].mean()):.2f}; PFC L1 frac sil {ts[sil].mean()/tot_n:.3f} speech {ts[spk].mean()/tot_n:.3f}')
    for rr in rows: print('   ', rr)
# code overlap at L4 chain units (band 0.002-0.95): same syllable other voice vs different syllables (adult voices)
for HI in (0.6, 0.9):
 for l in (1, 3):
     act = layers[l][1]; r = act.mean(0); ch = (r >= 0.002) & (r <= HI)
     codes = {}
     for t in np.nonzero(spk)[0]:
         codes.setdefault((lab[t], voi[t]), []).append(act[t, ch])
     m = {k: np.mean(v, 0) >= 0.5 for k, v in codes.items() if len(v) >= 6}
     def jac(a, b): return (a & b).sum() / max(1, (a | b).sum())
     keys = list(m); same, diff = [], []
     for i, a in enumerate(keys):
         for b in keys[i + 1:]:
             (same if a[0] == b[0] else diff).append(jac(m[a], m[b]))
     silc = act[sil][:, ch].mean(0) >= 0.5
     sv = [jac(m[k], silc) for k in keys]
     print(f'band<= {HI} L{l+1} chain codes: J same syl/other voice {np.mean(same):.2f}  J diff syl {np.mean(diff):.2f}  J(syl, silence) {np.mean(sv):.2f}  code size {np.mean([m[k].sum() for k in keys]):.0f}  silence code {silc.sum()}')
print(f'{time.time()-t0:.1f}s')
