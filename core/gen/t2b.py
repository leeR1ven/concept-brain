"""Generator step 2, binary (designer 11-11 essence; 11-13: binary neurons, not spiking): one network grown by the one
wiring formula (core/gen/grow.py). The ear is at one end, the eye at the other; the two branches grow toward each
other and meet in the middle. Every neuron is on or off; each neuron's threshold follows its own rate; competition by
the grown inhibitory neurons. Neurons where the branches meet also have many learnable dendrites from both sides
(drawn by the same formula with a long reach); each branch learns to predict whether its own neuron fires
(change = step x (fired - its push), shared by the active inputs) - the one rule; the sign seen is an independent
source, so it teaches the sound and the sound teaches nothing wrong. Real dialogues through the cochlea, one tick
per syllable; the syllable's sign shown on a share p_see of its hearings. Probe: sound alone (eye blank) - does a
syllable light the same middle neurons in every voice?
usage: python -m core.gen.t2b [key=value ...]"""
import json
import os
import sys
import time

import numpy as np
import torch

from ..config import config
from ..world import phon
from ..world.corpus import Dialogues, script_dialogue
from ..world.voice import ADULTS, Cochlea, Voice
from .grow import DEV, grow

P = dict(data=os.environ.get('CONCEPT_DATA', 'concept-data'), seed=0, L=16.0, Lm=6.0, W=24.0, H=12.0, dens=0.6,
         lam_e=1.5, lam_i=1.0, k_side=3.0, K_e=12, K_i=16, p_inh=0.2, rate_in=0.2, rate_out=0.06, rate_m=0.08,
         g_inh=0.5, birth=4000, adapt=0.002, K_l=400, lam_l=8.0, step=0.15, B=16, n_learn=3000, probe_every=500,
         n_cls=60, p_see=1.0, side=16, birth_see=0.3, patch=2.0, proj_depth=99.0, p_birth=0.0, g_proj=1.0, sub=1, win=3, p_long=0.1, lam_long=12.0, apical=1)
for a in sys.argv[1:]:
    k, v = a.split('=', 1)
    P[k] = json.loads(v) if v[:1] in '-0123456789[{' else v
torch.manual_seed(P['seed'])
c = config(levels_per_s=4, syl_ticks=1, gap_turn=[2, 4], gap_dlg=[6, 12])
coch = Cochlea(Voice(phon.load_inventory()), c, P['seed'])
train = Dialogues.load(os.path.join(P['data'], 'train.npz'))
side, L, Lm, W, H = P['side'], P['L'], P['Lm'], P['W'], P['H']
n_ear, n_eye = coch.n, side * side * 2
X = 2 * L + Lm                                            # ear at x = 0, eye at x = X
ear = torch.stack([torch.full((n_ear,), -0.5), (torch.arange(n_ear) // coch.Z).float() * W / coch.NC,
                   (torch.arange(n_ear) % coch.Z).float() * H / coch.Z], 1)
eye = torch.stack([torch.full((n_eye,), X + 0.5), (torch.arange(n_eye) // (2 * side)).float() * W / side,
                   (torch.arange(n_eye) % (2 * side)).float() * H / (2 * side)], 1)
n = int(X * W * H * P['dens'])
pos = torch.rand(n, 3) * torch.tensor([X, W, H])
mid = lambda x: (x >= L) & (x < L + Lm)                   # noqa: E731  where the branches meet


def fields(Q):
    m = len(Q)
    gg = torch.Generator(device='cpu').manual_seed(P['seed'] + 1)
    inh = (torch.rand(m, generator=gg) < P['p_inh']).to(Q.device)
    x = Q[:, 0]
    toward = torch.where(x < L + Lm / 2, 1.0, -1.0)      # each branch grows toward the middle
    d = torch.stack([toward, torch.zeros_like(x), torch.zeros_like(x)], 1)
    d = torch.where(inh[:, None], torch.randn(m, 3, device=Q.device), d)
    depth = torch.minimum(x, X - x).clamp(0, L) / L       # 0 at an organ, 1 at the middle
    rate = P['rate_in'] * (P['rate_out'] / P['rate_in']) ** depth
    rate = torch.where(mid(x), torch.full_like(rate, P['rate_m']), rate)
    f = lambda e, i: torch.where(inh, torch.full((m,), float(i), device=Q.device), torch.full((m,), float(e), device=Q.device))  # noqa: E731
    # axon lengths: most short, a few long (GENERATOR 二.2) - the long ones are the projections between regions
    gl = torch.Generator(device='cpu').manual_seed(P['seed'] + 2)
    long_ = (torch.rand(m, generator=gl) < P['p_long']).to(Q.device) & ~inh
    lam = torch.where(long_, torch.full((m,), float(P['lam_long']), device=Q.device), f(P['lam_e'], P['lam_i']))
    return dict(dir=d, lam=lam, ks=f(P['k_side'], 0.0),
                kr=torch.where(mid(x) | inh, 0.0, P['k_side']), K=f(P['K_e'], P['K_i']), inh=inh, rate=rate,
                g_inh=P['g_inh'])


t0 = time.time()
net = grow(torch.cat([ear, eye]), pos, fields, seed=P['seed'])
n_in = n_ear + n_eye
xs = net.pos[:, 0]
# every excitatory neuron (not the organs) also has learnable dendrites: K_l candidates drawn by the same wiring
# formula with the reach lam_l (birth links stay fixed; learned links start at 0) - the one rule everywhere
M = torch.nonzero(~net.inh & (torch.arange(net.n, device=DEV) >= n_in)).squeeze(1)
nM = len(M)
MID = torch.nonzero(mid(xs) & ~net.inh).squeeze(1)      # the excitatory neurons where the branches meet (measured)


def fields_l(Q):
    f = fields(Q)
    f['K'] = torch.where(f['inh'], 0.0, float(P['K_l']))
    f['lam'] = f['lam'] * P['lam_l'] / P['lam_e']          # the same short / long mix, a little wider
    return f


lnet = grow(torch.cat([ear, eye]), pos, fields_l, seed=P['seed'] + 11)
LI = lnet.idx[M]
VALID = (LI >= 0)
LI = LI.clamp_min(0)
VALID &= ~net.inh[LI]                                    # learned links grow from excitatory neurons
side_of = torch.where(net.pos[LI, 0] < L + Lm / 2, 0, 1)  # the dendritic branch: 0 = from the ear's side, 1 = eye's
BW = torch.zeros(LI.shape, device=DEV)
LW = torch.zeros(LI.shape, device=DEV)                    # learned weights (signed: the negative part stands for an
# the front (apical) dendrite: for neurons that receive from the other side, birth weights on those inputs (as every
# birth link) - what arrives from the front lights a neuron by itself when it is among the strongest such
front_ok = ((side_of == 1) & VALID).sum(1) > 0
front_ok &= mid(xs[M])                                   # where the two senses meet
BW_ap = torch.where((side_of == 1) & VALID & front_ok[:, None], 0.5 + torch.rand(LI.shape, device=DEV),
                    torch.zeros(LI.shape, device=DEV))
usual_ap = torch.zeros(len(M), device=DEV)
kap = max(1, int(round(P['rate_m'] * int(front_ok.sum()))))
print('GROWN', json.dumps(dict(neurons=net.n, organs=n_in, learners=nM, learnable_per_neuron=P['K_l'],
                               valid=round(float(VALID.float().mean()), 3),
                               secs=round(time.time() - t0, 1))), flush=True)   # inhibitory partner, registered)


def sign_of(sid):
    r = np.random.default_rng([c['lineage_seed'], 5150, 0 if sid is None else int(sid) + 1])
    v = r.random(side * side) < 0.5
    return np.stack([v, ~v], 1).reshape(-1)


blank = torch.as_tensor(sign_of(None), device=DEV)
SIGN = {}


def eye_bits(sids):
    out = []
    for s in sids:
        if s is None:
            out.append(blank)
        else:
            if s not in SIGN:
                SIGN[s] = torch.as_tensor(sign_of(s), device=DEV)
            out.append(SIGN[s])
    return torch.stack(out)


# ARCH 六 / 二.4 (designer): in every small patch a fixed share lights - the neurons pushed most above their own usual
# drive. Patches: slabs of width `patch` along x (the depth), so each depth keeps its own rate.
slab = torch.div(xs, P['patch'], rounding_mode='floor').long()
slab[:n_in] = -1
slabs = [torch.nonzero((slab == k) & (torch.arange(net.n, device=DEV) >= n_in)).squeeze(1) for k in
         range(int(slab.max()) + 1)]
slabs = [ix for ix in slabs if ix.numel()]
kk = [max(1, int(round(float(net.rate[ix].mean()) * ix.numel()))) for ix in slabs]
usual = torch.zeros(net.n, device=DEV)


AGE = {}                                                  # ticks since each neuron was last on, per stream batch


def step(x_in, act, learn=False, adapt=0.0):
    """one tick: every neuron reads the others a tick ago; the middle neurons add their learned dendrites"""
    age = AGE.get(act.shape[0])
    if age is None or age.shape != act.shape:
        age = torch.full(act.shape, 99, dtype=torch.int16, device=DEV)
    age = torch.where(act, torch.zeros_like(age), (age + 1).clamp_max(99))
    AGE[act.shape[0]] = age
    d = net.drive(act)
    a = act.float()
    pushes = (a[:, LI] * LW[None]).view(a.shape[0], nM, -1)
    push = torch.stack([(pushes * (side_of == s)[None]).sum(-1) for s in (0, 1)], -1)   # [B, nM, 2]
    d[:, M] += push.sum(-1) + (a[:, LI] * BW[None]).sum(-1)
    e = d - usual[None]
    new = torch.zeros_like(act)
    for ix, k in zip(slabs, kk):
        top = e[:, ix].topk(k, dim=1).indices
        new[:, ix] = torch.zeros_like(new[:, ix]).scatter_(1, top, True)
    if P['apical']:                                       # GENERATOR 二.5 (designer 11-11): what arrives from the front
        ap = (a[:, LI] * ((LW + BW_ap) * (side_of == 1))[None]).sum(-1)   # (the other sense) can light its neuron by
        ea = ap - usual_ap[None]                          # itself ('顶端让它亮了、基底却没预测到，基底就加强')
        if adapt:
            usual_ap.add_(adapt * (ap.mean(0) - usual_ap))
        topa = torch.zeros((a.shape[0], nM), dtype=torch.bool, device=DEV)
        topa.scatter_(1, ea.topk(kap, dim=1).indices, True)
        topa &= (ea > 0) & front_ok[None]
        new[:, M] |= topa
    new[:, :n_in] = x_in
    if adapt:                                             # each neuron's usual drive (its own threshold level)
        usual.add_(adapt * (d.mean(0) - usual))
    if learn:                                             # each branch predicts its own neuron (the one rule): the
        fired = new[:, M].float()                         # inputs that lit just before or just went off share it
        recent = (age <= P['win']).float()                # (ARCH 四 'A 刚亮或刚灭'): what arrives a little sooner or
        for s in (0, 1):                                  # later is fine
            on = recent[:, LI] * ((side_of == s) & VALID)[None].float()   # [B, nM, K]
            k = on.sum(-1).clamp_min(1.0)
            err = fired - push[..., s].clamp(0.0, 1.0)
            LW.add_((P['step'] * (err / k)[..., None] * on).mean(0))
    return new


# birth: changing sounds and blank eyes; every neuron finds its own threshold
rng = np.random.default_rng(P['seed'])


def tick(x_in, act, learn=False, adapt=0.0):
    """one syllable's time: `sub` network steps with the input held (a link takes a few ms, a syllable ~200 ms);
    learning and the usual drive on the last step"""
    for k in range(P['sub']):
        last = k == P['sub'] - 1
        act = step(x_in, act, learn=learn and last, adapt=adapt if last else 0.0)
    return act


def stream(d0, B):
    """B dialogues side by side, one tick per syllable: ([ear bits], [sid or None])"""
    plans = [script_dialogue(train, (d0 + b) % train.n, np.random.default_rng([P['seed'], d0 + b]), (2, 4), 4, 1, 1)
             for b in range(B)]
    T = max(len(p_) for p_ in plans)
    for t in range(T):
        srcs = [p_[t] if t < len(p_) else None for p_ in plans]
        A = torch.as_tensor(np.stack([coch.a0([] if s is None else [s[:5]]) for s in srcs]), device=DEV)
        yield A, [None if s is None else int(s[0]) for s in srcs]


act = torch.zeros(P['B'], net.n, dtype=torch.bool, device=DEV)
t = 0
bi = 0
birth_rng = np.random.default_rng(P['seed'] + 3)
while t < P['birth']:                                    # the newborn hears dialogue and now and then sees a sign
    for A, sids in stream(10 ** 6 + bi * P['B'], P['B']):
        seen = [s if (s is not None and birth_rng.random() < P['birth_see']) else None for s in sids]
        act = tick(torch.cat([A, eye_bits(seen)], 1), act, adapt=P['adapt'])
        t += 1
    bi += 1
cnt = np.bincount(train.syl[:200000].astype(np.int64))
cls = [int(x) for x in np.argsort(-cnt)[:P['n_cls']]]


def probe():
    """each class syllable alone in each adult voice from silence, eye blank; the middle neurons' code over the ticks
    the sound needs to arrive (union over a window); cross-voice agreement vs other syllables"""
    codes = {}
    for v in ADULTS:
        a = torch.zeros(len(cls), net.n, dtype=torch.bool, device=DEV)
        per = []
        seq = [None] * 6 + ['S'] + [None] * 12
        for k, x in enumerate(seq):
            A = torch.as_tensor(np.stack([coch.a0([] if x is None else [(s, -1, v, 0.0, 0.0)]) for s in cls]),
                                device=DEV)
            a = tick(torch.cat([A, eye_bits([None] * len(cls))], 1), a)
            if k >= 6:
                per.append(a[:, MID].clone())
        codes[v] = per
    best = None
    V = list(codes)
    for k in range(len(codes[V[0]])):                     # tick by tick after the sound: where is it clearest?
        same, other = [], []
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                J = jac(codes[V[i]][k].float(), codes[V[j]][k].float())
                same.append(J.diag().mean().item())
                other.append(((J.sum() - J.diag().sum()) / (len(cls) ** 2 - len(cls))).item())
        r = (np.mean(same), np.mean(other), k + 1)
        if best is None or r[0] - r[1] > best[0] - best[1]:
            best = r
    return dict(cross_voice_same=round(float(best[0]), 3), different_syllables=round(float(best[1]), 3),
                tick_after_sound=best[2], middle_lit=round(float(codes[V[0]][0].float().sum(1).mean()), 1))


def jac(a, b):
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None, :] - inter).clamp_min(1)


def diag(n_ticks=60):
    """while hearing dialogue (with signs shown): the share of neurons lit by depth along each branch and in the middle;
    and whether a sign alone lights the same middle neurons each time (the teacher's own clarity)"""
    a = torch.zeros(P['B'], net.n, dtype=torch.bool, device=DEV)
    rows = []
    for k, (A, sids) in enumerate(stream(5 * 10 ** 6, P['B'])):
        a = tick(torch.cat([A, eye_bits(sids)], 1), a)
        rows.append(a.float().mean(0))
        if k >= n_ticks:
            break
    R = torch.stack(rows[5:]).mean(0)
    ex = ~net.inh
    ex[:n_in] = False
    out = {}
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16)):
        out[f'ear side {lo}-{hi}'] = round(float(R[ex & (xs >= lo) & (xs < hi)].mean()), 3)
        out[f'eye side {lo}-{hi}'] = round(float(R[ex & (X - xs >= lo) & (X - xs < hi)].mean()), 3)
    out['middle'] = round(float(R[MID].mean()), 3)
    # the teacher: each class sign alone (silence), twice, middle code after it arrives
    cs = []
    for rep in range(2):
        a = torch.zeros(len(cls), net.n, dtype=torch.bool, device=DEV)
        per = []
        for k in range(20):
            A = torch.as_tensor(np.stack([coch.a0([]) for _ in cls]), device=DEV)
            a = tick(torch.cat([A, eye_bits(cls if k == 4 else [None] * len(cls))], 1), a)
            if k >= 4:
                per.append(a[:, MID].clone())
        cs.append(per)
    best = None
    for k in range(len(cs[0])):
        J = jac(cs[0][k].float(), cs[1][k].float())
        r = (float(J.diag().mean()), float((J.sum() - J.diag().sum()) / (J.numel() - len(J))), k + 1)
        if best is None or r[0] - r[1] > best[0] - best[1]:
            best = r
    out['sign alone (one tick): same sign twice / different signs / tick'] = [round(best[0], 3), round(best[1], 3), best[2]]
    return out


def info():
    """how much 'which syllable' is still there by depth along the ear's branch (sound alone, 4 voices): a linear
    reader trained on 3 voices, tested on the 4th (the ear itself first)"""
    out = {}
    states = {}
    for v in ADULTS:
        a = torch.zeros(len(cls), net.n, dtype=torch.bool, device=DEV)
        per = []
        for k, x in enumerate([None] * 6 + ['S'] + [None] * 10):
            A = torch.as_tensor(np.stack([coch.a0([] if x is None else [(s, -1, v, 0.0, 0.0)]) for s in cls]), device=DEV)
            a = tick(torch.cat([A, eye_bits([None] * len(cls))], 1), a)
            per.append(a.clone())
        states[v] = per
    ex = ~net.inh
    groups = {'ear': torch.arange(n_ear, device=DEV)}
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16)):
        groups[f'{lo}-{hi}'] = torch.nonzero(ex & (xs >= lo) & (xs < hi) & (torch.arange(net.n, device=DEV) >= n_in)).squeeze(1)
    groups['middle'] = MID
    V = list(states)
    for name, ix in groups.items():
        best = 0.0
        for k in range(6, 17):
            Xs = [states[v][k][:, ix].float() for v in V]
            acc = []
            for test in range(4):
                Xtr = torch.cat([Xs[i] for i in range(4) if i != test])
                Ytr = torch.eye(len(cls), device=DEV).repeat(3, 1)
                Kt = Xtr @ Xtr.T
                Aw = torch.linalg.solve(Kt + 1.0 * torch.eye(len(Kt), device=DEV), Ytr)
                pred = (Xs[test] @ Xtr.T) @ Aw
                acc.append(float((pred.argmax(1) == torch.arange(len(cls), device=DEV)).float().mean()))
            best = max(best, float(np.mean(acc)))
        out[name] = round(best, 3)
    return out


print('BIRTH done', round(time.time() - t0), 's; probe', json.dumps(probe()), flush=True)
print('INFO (linear reader, unseen voice; chance', round(1 / P['n_cls'], 3), ')', json.dumps(info()), flush=True)
print('DIAG', json.dumps(diag()), flush=True)
heard, d0 = 0, 0
see_rng = np.random.default_rng(P['seed'] + 7)
while heard < P['n_learn']:
    for A, sids in stream(d0, P['B']):
        shown = [s if (s is not None and see_rng.random() < P['p_see']) else None for s in sids]
        act = tick(torch.cat([A, eye_bits(shown)], 1), act, learn=True, adapt=P['adapt'])
    d0 += P['B']
    heard += P['B']
    if heard % P['probe_every'] < P['B']:
        # how big the learned links are next to the birth drive into the middle, while hearing (signs shown)
        a = torch.zeros(P['B'], net.n, dtype=torch.bool, device=DEV)
        bd, lp = [], []
        for k, (A, sids) in enumerate(stream(7 * 10 ** 6 + heard, P['B'])):
            a = tick(torch.cat([A, eye_bits(sids)], 1), a)
            af = a.float()
            bd.append(net.drive(a)[:, MID].abs().mean().item())
            lp.append((af[:, LI] * LW[None]).sum(-1).abs().mean().item())
            if k > 30:
                break
        print('LEARN', heard, 'dialogues', json.dumps(probe()), 'birth drive into the middle', round(float(np.mean(bd)), 3),
              'learned push', round(float(np.mean(lp)), 3), '|LW| max', round(float(LW.abs().max()), 3),
              f'{time.time() - t0:.0f} s', flush=True)
        print('INFO', heard, json.dumps(info()), flush=True)
