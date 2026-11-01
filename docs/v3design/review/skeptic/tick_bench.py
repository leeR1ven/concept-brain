"""Skeptic micro-benchmark (local numpy, 1 process, < 400 MB): what does one tick of the s = 1 pure-neuron core of
ARCH_v3_core §7.5 cost, using the doc's own sizes (synth/checks.py) and the cheapest event-driven numpy form
(CSR fan-out lists + bincount scatter-add, argpartition k-WTA)?  Times three kinds of tick:
  silent : ear stack (EAR0 -> SYL, VOICE) on noise, per-unit set-points, TIME, GEN read + per-tick prediction learning,
           critic + 6 M4 channels with per-synapse eligibility traces (tau_e 20, delta every tick), mu, accumulator.
  speech : silent + MIX recompute (48k units, 3 branches x fan-in 3, AND + k-WTA 1 %), THOUGHT, DG.
  replay : one reactivation = CA3 attractor completion (10k units, fan-in 1000, 5 iterations, k-WTA 1 %), CA1 -> MIX
           recompute, GEN learning (what one of G20's 8 reactivations per quiet period needs at least).
Run: python3 -I tick_bench.py   (local core; the A800 runner's cores are ~1.8x faster per the gpu-server notes)
"""
import numpy as np, time
rng = np.random.default_rng(0)
f32 = np.float32


def csr(n_pre, n_post, fan_in):
    """fan-in lists -> fan-out CSR (pre -> post) for event-driven scatter-add"""
    pre = rng.integers(0, n_pre, size=n_post * fan_in).astype(np.int32)
    post = np.repeat(np.arange(n_post, dtype=np.int32), fan_in)
    o = np.argsort(pre, kind='stable'); pre, post = pre[o], post[o]
    ptr = np.searchsorted(pre, np.arange(n_pre + 1)).astype(np.int64)
    w = rng.random(n_post * fan_in).astype(f32)
    return ptr, post, w


def drive(C, active, n_post):
    ptr, post, w = C
    if len(active) == 0:
        return np.zeros(n_post, f32), np.zeros(0, np.int64)
    lens = ptr[active + 1] - ptr[active]; starts = ptr[active]; offs = np.cumsum(lens) - lens
    idx = np.arange(lens.sum()) - np.repeat(offs, lens) + np.repeat(starts, lens)
    return np.bincount(post[idx], weights=w[idx], minlength=n_post).astype(f32), idx


def kwta(d, k):
    return np.argpartition(-d, k)[:k]


# ---- sizes from synth/checks.py (s = 1)
N_EAR, N_SYL, N_VOI, N_LAT, N_TH, N_MIX, N_GEN, N_DG, N_CA3, N_CA1 = 1920, 2048, 256, 16384, 4096, 48000, 2049, 6000, 10000, 10000
EAR_in = 48 * 6
C_ear = csr(EAR_in, N_EAR, 4); C_syl = csr(N_EAR, N_SYL, 64); C_voi = csr(N_EAR, N_VOI, 64)
C_th = csr(N_LAT, N_TH, 64)
C_mix = [csr(N_LAT + 36, N_MIX, 3), csr(N_TH + N_LAT, N_MIX, 3), csr(N_TH + 1024 + 512, N_MIX, 3)]   # anchor + 2 random
C_gen = csr(N_MIX, N_GEN, 1500)          # = fan-out 64 from MIX
W_val = np.zeros((N_MIX, 7), f32); E_val = np.zeros((N_MIX, 7), f32)    # critic + 6 M4 channels (5 sources + release)
C_dg = csr(N_LAT + N_TH + 1024 + 512 + 36, N_DG, 12)
C_ca3 = csr(N_CA3, N_CA3, 1000); C_dg3 = csr(N_DG, N_CA3, 50); C_ca1 = csr(N_CA3, N_CA1, 64)
C_ca1mix = csr(N_CA1, N_MIX, 3)
setpt = np.zeros(N_EAR + N_SYL + N_VOI + 1024, f32)
print('memory of the weights built here ~%.0f MB' % (sum(c[2].nbytes + c[1].nbytes for c in
      [C_ear, C_syl, C_voi, C_th, *C_mix, C_gen, C_dg, C_ca3, C_dg3, C_ca1, C_ca1mix]) / 1e6), flush=True)

lat_act = rng.choice(N_LAT, 200, replace=False); th_act = rng.choice(N_TH, 192, replace=False)
mix_act = rng.choice(N_MIX, 480, replace=False)
trace_rows = rng.choice(N_MIX, 480 * 12, replace=False)       # MIX units active in the last ~3 tau_e ticks
eta, decay = f32(0.3), f32(np.exp(-1 / 20))


def silent_tick():
    x = (rng.random(EAR_in) < 0.3).nonzero()[0]                      # noisy ear window
    d, _ = drive(C_ear, x, N_EAR); ear = kwta(d, 768)
    ds, _ = drive(C_syl, ear, N_SYL); dv, _ = drive(C_voi, ear, N_VOI)
    q = np.concatenate([d, ds, dv, np.zeros(1024, f32)])
    setpt[:] += f32(0.01) * (f32(0.25) - (q < setpt))                 # sign-rule quantile tracker, every unit
    syl = kwta(ds - setpt[N_EAR:N_EAR + N_SYL], 16)
    g, gidx = drive(C_gen, mix_act, N_GEN)                            # GEN prediction (per tick, O1 'when')
    p = np.clip(g, 0, 1); t = np.zeros(N_GEN, f32); t[syl] = 1
    C_gen[2][gidx] += f32(0.35 / 480) * (t - p)[C_gen[1][gidx]]       # prediction-form update on lit synapses
    E_val[trace_rows] *= decay; E_val[mix_act] = 1                   # eligibility traces on value synapses
    v = W_val[mix_act].sum(0); delta = f32(np.tanh(0.01 - v[0] * 0.05) * 0.01)
    W_val[trace_rows] += eta * delta * E_val[trace_rows]              # delta every tick (TD)
    return v


def speech_tick():
    silent_tick()
    lat = np.concatenate([lat_act, N_LAT + rng.choice(36, 4, replace=False)])
    b0, _ = drive(C_mix[0], lat, N_MIX)
    b1, _ = drive(C_mix[1], np.concatenate([th_act, N_TH + lat_act]), N_MIX)
    b2, _ = drive(C_mix[2], np.concatenate([th_act, N_TH + rng.choice(1536, 60, replace=False)]), N_MIX)
    elig = (b0 >= 1) & (b1 >= 1) & (b2 >= 1); s = np.where(elig, b0 + b1 + b2, -1).astype(f32)
    m = kwta(s, 480)
    dth, _ = drive(C_th, lat_act, N_TH); kwta(dth, 192)
    ddg, _ = drive(C_dg, np.concatenate([lat_act, N_LAT + th_act]), N_DG); kwta(ddg, 120)
    return m


def replay():
    seed_units = rng.choice(N_DG, 120, replace=False)
    d, _ = drive(C_dg3, seed_units, N_CA3); a = kwta(d, 100)
    for _ in range(5):                                                 # attractor completion
        d, _ = drive(C_ca3, a, N_CA3); a = kwta(d, 100)
    c1, _ = drive(C_ca1, a, N_CA1); c1a = kwta(c1, 100)
    b, _ = drive(C_ca1mix, c1a, N_MIX)
    b0, _ = drive(C_mix[0], lat_act, N_MIX)
    m = kwta(b + b0, 480)
    g, gidx = drive(C_gen, m, N_GEN); p = np.clip(g, 0, 1); t = np.zeros(N_GEN, f32); t[:16] = 1
    C_gen[2][gidx] += f32(0.35 / 480) * (t - p)[C_gen[1][gidx]]
    return m


for name, f, n in (('silent', silent_tick, 400), ('speech', speech_tick, 150), ('replay', replay, 60)):
    f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    dt = (time.perf_counter() - t0) / n
    print(f'{name:7s} {dt * 1e3:7.2f} ms per call (local core)', flush=True)

# ---- breakdown of the silent tick
import collections
T = collections.Counter(); N = 300
for _ in range(N):
    t0 = time.perf_counter()
    x = (rng.random(EAR_in) < 0.3).nonzero()[0]
    d, _ = drive(C_ear, x, N_EAR); ear = kwta(d, 768)
    ds, _ = drive(C_syl, ear, N_SYL); dv, _ = drive(C_voi, ear, N_VOI); t1 = time.perf_counter(); T['ear stack'] += t1 - t0
    q = np.concatenate([d, ds, dv, np.zeros(1024, f32)]); setpt[:] += f32(0.01) * (f32(0.25) - (q < setpt))
    syl = kwta(ds - setpt[N_EAR:N_EAR + N_SYL], 16); t2 = time.perf_counter(); T['set-points + SYL k-WTA'] += t2 - t1
    g, gidx = drive(C_gen, mix_act, N_GEN); t3 = time.perf_counter(); T['GEN read'] += t3 - t2
    p = np.clip(g, 0, 1); tt = np.zeros(N_GEN, f32); tt[syl] = 1
    C_gen[2][gidx] += f32(0.35 / 480) * (tt - p)[C_gen[1][gidx]]; t4 = time.perf_counter(); T['GEN per-tick learning'] += t4 - t3
    E_val[trace_rows] *= decay; E_val[mix_act] = 1
    v = W_val[mix_act].sum(0); W_val[trace_rows] += eta * f32(0.001) * E_val[trace_rows]; t5 = time.perf_counter(); T['value traces + TD'] += t5 - t4
for k, v in T.items():
    print(f'   silent part {k:28s} {v / N * 1e3:6.2f} ms')
