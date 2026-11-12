"""The check-up of a grown brain against what is known of real cortex (GENERATOR: no training until it passes).
No learning: the brain is grown, born (own thresholds), then given silence, sounds and signs.
Each line: the measure, our value, the range reported for cortex, PASS / FAIL.
usage: GEN_ARGS="key=value ..." python -m core.gen.validate"""
import json

import numpy as np
import torch

from . import t2_concept as T

DEV = T.DEV
torch.manual_seed(1)
results = []


def check(name, value, lo, hi, ref):
    ok = lo <= value <= hi
    results.append(ok)
    print(f'{"PASS" if ok else "FAIL"}  {name}: {value:.3f}  (cortex {lo}-{hi}; {ref})', flush=True)


N, n_in, idx, area, inh, pos = T.N, T.n_in, T.idx, T.area, T.inh, T.pos
valid = idx >= 0
pre = idx.clamp_min(0)
post = torch.arange(N, device=DEV)[:, None].expand_as(idx)
# ---------------------------------------------------------------- structure
L = (pos[post[valid]] - pos[pre[valid]]).norm(dim=1)
check('inhibitory share of neurons', inh[n_in:].float().mean().item(), 0.15, 0.30, 'Markram 2004')
check('links longer than 3x the median length (long-range share)', (L > 3 * L.median()).float().mean().item(),
      0.01, 0.20, 'exponential distance rule, Markov 2014')
# exponential decay: log of the length histogram falls about linearly (r^2 of a line fit)
# the exponential distance rule is about the CHANCE that two neurons at distance r are linked (links / pairs), not the
# raw length histogram (in 3D the number of neighbours grows as r^2)
g_ = torch.Generator(device='cpu').manual_seed(5)
ci = torch.nonzero(area > 0).squeeze(1)
a_ = ci[torch.randint(len(ci), (400000,), generator=g_).to(DEV)]
b_ = ci[torch.randint(len(ci), (400000,), generator=g_).to(DEV)]
Dp = (pos[a_] - pos[b_]).norm(dim=1)
edges = torch.linspace(0.5, 8.0, 16, device=DEV)
lk = torch.histc(L, 15, 0.5, 8.0)
pr = torch.histc(Dp, 15, 0.5, 8.0) * (float(len(ci)) ** 2 / 400000)
pc = (lk / pr.clamp_min(1)).cpu().numpy()
xc = (0.5 * (edges[1:] + edges[:-1])).cpu().numpy()
m = pc > 0
r2 = np.corrcoef(xc[m], np.log(pc[m]))[0, 1] ** 2 if m.sum() > 2 else 0.0
slope = np.polyfit(xc[m], np.log(pc[m]), 1)[0] if m.sum() > 2 else 0.0
check('link chance falls exponentially with distance (r^2 of log chance vs distance)', float(r2), 0.8, 1.0,
      f'exponential distance rule, Ercsey-Ravasz 2013; slope {slope:.2f} per unit')
# area-to-area links both ways
na = int(area.max()) + 1
C = torch.zeros(na, na, device=DEV)
C.index_put_((area[pre[valid]], area[post[valid]]), torch.ones(int(valid.sum()), device=DEV), accumulate=True)
pairs = [(1, 2), (3, 2)]
recip = [min(float(C[a, b]), float(C[b, a])) / max(1.0, max(float(C[a, b]), float(C[b, a]))) for a, b in pairs]
check('between areas, feedback / feedforward link count (sensory <-> concept)', float(np.mean(recip)), 0.1, 1.0,
      'most inter-areal pathways are reciprocal, Felleman & Van Essen 1991; Markov 2014')
# local reciprocity among excitatory neurons vs chance
E = torch.nonzero((~inh) & (torch.arange(N, device=DEV) >= n_in)).squeeze(1)
S = set()
vv = valid.cpu().numpy()
ix = idx.cpu().numpy()
for j in E.cpu().numpy()[:6000]:
    for k in ix[j][vv[j]]:
        S.add((int(k), int(j)))
bi = sum((j, i) in S for (i, j) in S)
n_e = len(E)
p = len(S) / max(1, n_e * n_e)
ratio = (bi / max(1, len(S))) / max(1e-12, p)
check('reciprocal pairs vs chance (excitatory)', ratio, 2.0, 1e9, 'overrepresented about 4x, Song 2005')
# ---------------------------------------------------------------- activity
T.birth()


def run(xin_fn, ticks, B=16, sub=False):
    act = torch.zeros(B, N, dtype=torch.bool, device=DEV)
    T.reset(B)
    T.SUB = [] if sub else None
    rec, eI, iI = [], [], []
    for t in range(ticks):
        xin = xin_fn(t, B)
        bas, apx = T.drives(act)
        a = torch.cat([act.float(), torch.zeros(B, 1, device=DEV)], 1)
        ixx = torch.where(idx >= 0, idx, N)
        con = a[:, ixx] * T.w[None]
        inh_p = torch.cat([inh, torch.zeros(1, dtype=torch.bool, device=DEV)])
        exc = (con * (~inh_p[ixx]).float()[None]).sum(-1)
        inn = (con * inh_p[ixx].float()[None]).sum(-1)
        act = T.step(xin, act)
        rec.append(act.clone())
        eI.append(exc)
        iI.append(inn)
    S = torch.stack(T.SUB) if sub else None
    T.SUB = None
    return torch.stack(rec), torch.stack(eI), torch.stack(iI), S


R, EX, IN, SS = run(lambda t, B: T.organ_in([(T.ear([]), T.eye(None))] * B), 200, sub=bool(T.P['spiking']))
cortex = (area > 0)
if SS is not None:                                               # spiking: measure on the sub-steps (dt ms)
    r_c = SS[:, :, cortex].float()
    hz = r_c.mean().item() * 1000.0 / T.P['dt']
    check('spontaneous firing rate in silence (Hz)', hz, 0.1, 10.0, 'sparse, low rates, Barth & Poulet 2012')
    b5 = r_c[:r_c.shape[0] // 2 * 2].reshape(-1, 2, *r_c.shape[1:]).sum(1)   # 2-step bins
    pop = b5.mean(2)
    chi = (pop.var(0).mean() / b5.var(0).mean().clamp_min(1e-12) / b5.shape[2]).sqrt().item() * np.sqrt(b5.shape[2])
    chi = float((pop.var(0).mean() / b5.var(0).mean().clamp_min(1e-12)).sqrt().item())
    rr = SS[:, 0, cortex].cpu().numpy()
else:
    r_c = R[:, :, cortex].float()                                # [T, B, n]
    check('spontaneous rate (share lit per tick, silence)', r_c.mean().item(), 0.005, 0.15, 'sparse, Barth & Poulet 2012')
    pop = r_c.mean(2)                                            # [T, B]
    chi = (pop.var(0).mean() / r_c.var(0).mean().clamp_min(1e-12)).sqrt().item()
    rr = R[:, 0, cortex].cpu().numpy()
check('synchrony chi (silence; 0 = asynchronous, 1 = all together)', chi, 0.0, 0.3,
      'asynchronous irregular state, Brunel 2000')
# irregularity: coefficient of variation of the inter-event intervals of active neurons
cv = []
for j in np.nonzero(rr.sum(0) >= 5)[0][:2000]:
    t_ = np.nonzero(rr[:, j])[0]
    d = np.diff(t_)
    if len(d) >= 4 and d.mean() > 0:
        cv.append(d.std() / d.mean())
check('irregularity CV of intervals (silence)', float(np.mean(cv)) if cv else 0.0, 0.5, 1.5,
      'Poisson-like, Softky & Koch 1993')
# sound input: E/I balance and sparseness
dl = T.stream(range(15000, 15040), 9100, False)


def sound_fn(t, B):
    k = [(t + 37 * b) % len(dl) for b in range(B)]
    return T.organ_in([(dl[i][0], dl[i][1]) for i in k])


R2, EX2, IN2, _ = run(sound_fn, 200)
ex = EX2[:, :, cortex & ~inh].mean(2).flatten()
iv = IN2[:, :, cortex & ~inh].mean(2).flatten()
check('E/I balance: correlation of excitatory and inhibitory input over time (sounds)',
      torch.corrcoef(torch.stack([ex, iv]))[0, 1].item(), 0.5, 1.0, 'Okun & Lampl 2008')
print('EVOKED share firing per tick (speech):', json.dumps({f'{nm} {"I" if ii else "E"}': round(R2[:, :, (area == k) & (inh == ii)].float().mean().item(), 3)
                                                            for nm, k in (('auditory', 1), ('concept', 2), ('visual', 3)) for ii in (False, True)}),
      '| silence:', json.dumps({f'{nm} {"I" if ii else "E"}': round(R[:, :, (area == k) & (inh == ii)].float().mean().item(), 3)
                                for nm, k in (('auditory', 1), ('concept', 2), ('visual', 3)) for ii in (False, True)}), flush=True)
lit = R2[:, :, cortex].float()
check('share firing in a tick (sounds)', lit.mean().item(), 0.01, 0.3, 'sparse coding, Olshausen & Field 2004')
heard = ((area == 1) | (area == 2))[cortex]                      # speech alone: the visual area has nothing to see
silent_frac = (lit.mean((0, 1))[heard] == 0).float().mean().item()
check('auditory and concept neurons never firing in 200 ticks of speech (dead share)', silent_frac, 0.0, 0.3,
      'few silent neurons')
# latency by depth: after a syllable from silence, first tick each area responds above its silence level
rows = []
for sid in [int(x) for x in np.argsort(-np.bincount(T.train.syl[:200000].astype(np.int64)))[:16]]:
    rows.append(sid)


def onset_fn(t, B):
    xs = []
    for b in range(B):
        e = T.ear([(rows[b % len(rows)], -1, 'C1', 0.0, 0.0)]) if 4 <= t < 4 + T.P['hold'] else T.ear([])
        xs.append((e, T.eye(None)))
    return T.organ_in(xs)


R3, _, _, S3 = run(onset_fn, 20, sub=bool(T.P['spiking']))
lat = {}
for nm, k in (('auditory shallow', None), ('auditory deep', None), ('concept', 2)):
    if nm.startswith('auditory'):
        dsel = (area == 1) & ((T.depth < 0.5) if 'shallow' in nm else (T.depth >= 0.5))
    else:
        dsel = area == 2
    if S3 is not None:                                           # sub-steps; onset at tick 4
        trace = S3[:, :, dsel].float().mean((1, 2))
        on0 = 4 * T.P['steps']
        base = trace[:on0].mean()
        over = torch.nonzero(trace[on0:] > base * 1.5 + 1e-3).squeeze(1)
        lat[nm] = float(over[0]) * T.P['dt'] if over.numel() else 999.0
    else:
        trace = R3[:, :, dsel].float().mean((1, 2))
        base = trace[:4].mean()
        over = torch.nonzero(trace[4:] > base * 1.5 + 1e-3).squeeze(1)
        lat[nm] = int(over[0]) if over.numel() else 99
print('LATENCY (after onset; ms when spiking, else ticks)', json.dumps(lat), flush=True)
check('latency rises along the pathway (auditory shallow < deep <= concept)',
      float(lat['auditory shallow'] < lat['auditory deep'] <= lat['concept']), 1, 1, 'Schmolesky 1998 (vision)')
# mixed selectivity: concept neurons that respond to sound alone and to a sign alone
def sign_fn(t, B):
    xs = []
    for b in range(B):
        sg = T.eye(rows[b % len(rows)]) if 4 <= t < 4 + T.P['hold'] else T.eye(None)
        xs.append((T.ear([]), sg))
    return T.organ_in(xs)


R4, _, _, _ = run(sign_fn, 20)
cs = (area == 2)
b3, b4 = R3[:4, :, cs].float().mean((0, 1)), R4[:4, :, cs].float().mean((0, 1))
snd = R3[5:12, :, cs].float().mean((0, 1)) > 2 * b3 + 0.01          # a sparse neuron answers relative to its own rest
see = R4[5:12, :, cs].float().mean((0, 1)) > 2 * b4 + 0.01
both = (snd & see).float().sum().item() / max(1.0, (snd | see).float().sum().item())
check('concept neurons answering both a sound and a sign (of those answering either)', both, 0.1, 1.0,
      'multimodal convergence, Patterson 2007; mixed selectivity, Rigotti 2013')
print(f'CHECKUP {sum(results)} of {len(results)} pass', flush=True)


ids = T.identity_by_stage()
print('IDENTITY by stage (nearest class mean, 60 syllables, chance ~0.017)', json.dumps(ids), flush=True)
check('concept area tells syllables apart (identity, sound only)', ids['concept'], 0.2, 1.0,
      'phoneme identity in auditory cortex, Mesgarani 2014')
print(f'CHECKUP {sum(results)} of {len(results)} pass (with information)', flush=True)
