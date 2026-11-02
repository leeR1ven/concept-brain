"""Unit checks requested by the coordinator (11-02). Each prints PASS/FAIL with numbers; exit code 1 if any fails.
usage: python3 unit_checks.py [--s 1] [--out checks.json]"""
import argparse
import sys

import numpy as np
import torch

import world as W
from brain import Brain
from common import DEV, config, gammas, log, save_json
from learning import ByPre, Pop
from life import Life
from world_adapter import Cochlea, Env
import probes

ap = argparse.ArgumentParser()
ap.add_argument('--s', type=int, default=1)
ap.add_argument('--out', default='checks.json')
a = ap.parse_args()
R = {}


def check(name, ok, **info):
    R[name] = dict(ok=bool(ok), **info)
    log(('PASS ' if ok else 'FAIL ') + name, info)


c = config(a.s)

# 1. the input layer is exactly half on every tick (asserted inside Cochlea.code; counted here over 3,000 ticks)
ph = W.Physics()
coch = Cochlea(ph, c, 0)
env = Env(W.World(ph, seed=0, care='real', distract=True), coch)       # the stream used by sizing/sound_stats2.py
A0, sil, spk = [], [], []
for t in range(5000):
    x = env.step(np.zeros(24, bool))
    A0.append(x)
    sil.append(len(env.srcs) == 0)
    spk.append(any(k == 's' and loud >= 1.0 and v != W.OWN for k, _, v, loud in env.srcs))
A0 = np.array(A0)
half = A0.sum(1)
check('1 A0 exactly half on every tick', np.all(half * 2 == A0.shape[1]), ticks=len(A0), n0=A0.shape[1],
      min_on=int(half.min()), max_on=int(half.max()))

# 2. the chain band reproduces the PLAN 2.4 counts (sizing v2, K6, s=1, same kind of stream: World real + TV).
#    The sizing applied the band at every depth, so this check uses band_depth_sound = 4; the default (band on A1
#    only, designer 11-02) is reported in 2c.
br = Brain(config(a.s, band_depth_sound=4), 0, coch, A0, np.array(spk))
sil, spk = np.array(sil), np.array(spk)
ch = br.chains
n_layer = [len(ix) for ix in ch.sound_idx]
st = br.sound.calibrate(torch.as_tensor(A0, device=DEV))
lit = sum(s[:, ix].float().sum(1).cpu().numpy() for s, ix in zip(st, ch.sound_idx))
ref = dict(per_layer=[1670, 2126, 2160, 2160], total=8116, silence=363, speech=1346) if a.s == 1 else \
    dict(per_layer=[6700, 8481, 8639, 8640], total=32460, silence=1292, speech=5202)
tot = sum(n_layer)
ok2 = abs(tot / ref['total'] - 1) <= 0.10 and abs(lit[sil].mean() / ref['silence'] - 1) <= 0.35 and \
    abs(lit[spk].mean() / ref['speech'] - 1) <= 0.35
band_frac = [n / m for n, m in zip(n_layer, br.sound.n)]
check('2 chain band reproduces the 2.4 counts (+-10 % units, +-35 % lit)', ok2, chains_per_layer=n_layer, ref=ref,
      band_frac=band_frac, lit_silence=float(lit[sil].mean()), lit_speech=float(lit[spk].mean()),
      layer_frac=[float(s.float().mean()) for s in st])
check('2b A2-A4 mostly feature units (>= 80 % in band)', min(band_frac[1:]) >= 0.8, band_frac=band_frac)
del st, br
br = Brain(c, 0, coch, A0, np.array(spk))
ch = br.chains
check('2c default chains (band on A1 only; deeper units all chained): report', True, n1=br.n1,
      report=br.chain_report)

# 3. every forward chain has exactly one return to the same S
fw, rt = ch.fwd_src, ch.ret_dst
same = bool(torch.equal(fw, rt))
uniq = len({tuple(r) for r in fw.cpu().tolist()}) == fw.shape[0]
rng = np.random.default_rng(0)
ok3 = same and uniq
bad = 0
for j in rng.choice(br.n1, 300, replace=False):
    Rv = torch.zeros(br.n1, dtype=torch.bool, device=DEV)
    Rv[int(j)] = True
    rs, rf, rst = ch.scatter(Rv, br.sound.n, br.motor.nf, br.motor.ns)
    lit_n = sum(int(x.sum()) for x in rs) + int(rf.sum()) + int(rst.sum())
    area, layer, unit = fw[int(j)].tolist()
    hit = (rs[layer][unit] > 0) if area == 0 else (rf[unit] > 0 if area == 1 else rst[unit])
    bad += int(lit_n != 1 or not bool(hit))
    # forward: lighting exactly that source lights exactly copy j
    S = [torch.zeros(n, dtype=torch.bool, device=DEV) for n in br.sound.n]
    fo = torch.zeros(br.motor.nf, dtype=torch.bool, device=DEV)
    so = torch.zeros(br.motor.ns, dtype=torch.bool, device=DEV)
    (S[layer] if area == 0 else fo if area == 1 else so)[unit] = True
    X = ch.gather(S, fo, so)
    bad += int(X.sum() != 1 or not bool(X[int(j)]))
assert ok3 and bad == 0, 'chain pairs are not 1:1'
check('3 every forward chain has exactly one return to the same S (assert)', ok3 and bad == 0, n=br.n1,
      sampled=300, mismatches=bad, a0_has_chain=False, blocking_have_chain=False)

# 4. no dense N x N anywhere
viol = []
for tb in br.tables():
    n_pre, n_post = tb.n_pre, tb.n_post
    width = tb.shape()[1]
    if n_post > 1000 and (width > 0.25 * n_post):
        viol.append((tb.name, tb.shape(), n_post))


def scan(obj, path, depth=0):
    if depth > 3:
        return
    for k, v in vars(obj).items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and min(v.shape) >= 1000 and v.shape[0] == v.shape[1]:
            viol.append((path + '.' + k, tuple(v.shape)))
        elif hasattr(v, '__dict__') and not isinstance(v, (torch.Tensor, type)) and depth < 3:
            scan(v, path + '.' + k, depth + 1)
        elif isinstance(v, list):
            for i, e in enumerate(v):
                if isinstance(e, torch.Tensor) and e.dim() == 2 and min(e.shape) >= 1000 and e.shape[0] == e.shape[1]:
                    viol.append((f'{path}.{k}[{i}]', tuple(e.shape)))
                elif hasattr(e, '__dict__'):
                    scan(e, f'{path}.{k}[{i}]', depth + 1)


scan(br, 'brain')
check('4 no dense N x N anywhere', not viol, violations=viol,
      tables=[(tb.name, tb.shape(), tb.n_post) for tb in br.tables()])

# 5. the brain's step receives only the A0 bit vector and r = 0.0 (asserts)
a0 = torch.as_tensor(A0[0], device=DEV)
fails = 0
for bad_call in (lambda: br.step(a0.float(), 0.0), lambda: br.step(a0[:-1], 0.0), lambda: br.step(a0, 1.0),
                 lambda: br.step(a0, 0)):
    try:
        bad_call()
    except AssertionError:
        fails += 1
br.step(a0, 0.0)
import inspect
sig = list(inspect.signature(Brain.step).parameters)
check('5 step receives only A0 and r (asserts fire on anything else)', fails == 4 and sig == ['self', 'a0', 'r'],
      rejected=fails, signature=sig)

# 6. decay order: hippocampus fastest, PFC slowest
g = gammas(c)
tg = {tb.name: tb.gamma for tb in br.plastic()}
ok6 = g['hip'] < g['motor'] < g['pfc'] and all(
    (v == g['hip']) if n.startswith('hip') else (v == g['pfc']) if n.startswith('PFC') else (v == g['motor'])
    for n, v in tg.items())
check('6 decay order hippocampus < motor/G1 < PFC (per 200 ticks)', ok6, gammas=g, tables=tg)

# 7. G1P (gen-0 speech wiring inside the PFC) is in the decaying, learning part
life = Life(config(a.s, birth_dialogues=60), 0)
G1 = life.brain.pfc.G1P
w0 = G1.w.clone()
in_plastic = any(tb is G1 for tb in life.brain.plastic())
G1.decay()
pl = (1.0 / (1.0 + w0 / c['w_ref'])).clamp_min(c['floor'])
exp = w0 - (1 - G1.gamma) * w0 * pl
exp[exp < c['prune']] = 0
ok7 = in_plastic and not G1.birth and torch.allclose(G1.w, exp, atol=1e-7) and float(w0.sum()) > 0 \
    and G1.gamma == gammas(c)['pfc'] and G1.eta == c['eta']
check('7 G1P is in the decaying, learning part (gamma_pfc, the rule, concave decay)', ok7, gamma=G1.gamma,
      links=int((w0 > 0).sum()), registry=life.registry['G1P'])

# 9. no table maps sound-area units to force neurons (designer 11-02: sound -> action only through the PFC)
b9 = life.brain
targets_force = [tb.name for tb in b9.tables() if tb.n_post == b9.motor.nf and not tb.name.startswith('motor MC')]
import inspect as _insp
motor_args = list(_insp.signature(type(b9.motor).step).parameters)
ok9 = not targets_force and all(t.name.startswith('motor MC') and t.n_pre == b9.motor.nf for t in b9.motor.tables()) \
    and not hasattr(b9.motor, 'G1') \
    and motor_args == ['self', 'a0', 'ret_force', 'ret_stop', 'scaffold']
check('9 no table maps sound-area units to force neurons; force neurons get only their 1:1 returns (the scaffold '
      'argument is the teacher program of the learned-speech arm, off by default)', ok9,
      tables_targeting_force_neurons=targets_force, motor_step_args=motor_args,
      g1p_targets='PFC L1 force copies', tables=[(tb.name, tb.n_pre, tb.n_post) for tb in b9.tables()])

# 8. count preservation at birth: the calibrated feed-forward PFC (birth loop and lateral off, so L1 = chain
#    input) passes on about as many units as it receives, online, in silence and in speech. With the default loop
#    the same numbers are reported (they depend on the (a) choice of loop gain).
ff = Life(config(a.s, birth_dialogues=60, g_loop=0.0, w44=0.0), 0, wire=False)
cnt = ff.brain.calib['pfc_counts']
act = probes.activity(ff, 40)
ratios = {k: v for k, v in act.items() if '/' in k}
ok8 = all(abs(x / cnt[0] - 1) <= 0.05 for x in cnt[1:]) and all(0.67 <= v <= 1.5 for v in ratios.values())
check('8 count preservation at birth (calibration within 5 %; online L2/L1, L3/L2, L4/L3 in [0.67, 1.5], '
      'silence and speech; loop off)', ok8, calib_counts=cnt, calib_theta=ff.brain.calib['pfc_theta'],
      calib_g=ff.brain.calib['pfc_g'], online=ratios,
      counts={k: (round(act[k]['silence']), round(act[k]['speech'])) for k in act if '/' not in k})
del ff
life.brain.learning = False
act = probes.activity(life, 40)
check('8b with the default loop (g_loop 1.0, w44 0.5): report only', True,
      online={k: v for k, v in act.items() if '/' in k},
      counts={k: (round(act[k]['silence']), round(act[k]['speech'])) for k in act if '/' not in k})

# extra: the learning rule on a toy table (strengthen, co-rise, weaken, fall does nothing)
cc = dict(c)
idx = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEV)
tb = ByPre(idx, torch.zeros(1, 3, device=DEV), 1, 3, 'toy', birth=False, eta=0.1, eta_m=0.05, gamma=1.0, c=cc)
pre_el = torch.tensor([True], device=DEV)
none = torch.tensor([False], device=DEV)
tb.learn(pre_el, none, torch.tensor([True, False, False], device=DEV), torch.tensor([True, False, False], device=DEV),
         torch.tensor([False, False, False], device=DEV))
w1 = tb.w.clone()
tb.learn(none, pre_el, torch.zeros(3, dtype=torch.bool, device=DEV), torch.tensor([False, True, False], device=DEV),
         torch.tensor([True, False, False], device=DEV))
w2 = tb.w.clone()
ok_r = w1[0, 0] > 0 and w1[0, 1] == 0 and w2[0, 1] > 0 and w2[0, 0] < w1[0, 0] and w2[0, 2] == 0
check('extra: rule signs (strengthen / co-rise / weaken / fall nothing)', ok_r, w_after_1=w1.tolist(),
      w_after_2=w2.tolist())

# extra: a return alone lights its sound unit (inner hearing) [C1]
b = life.brain
with probes.Frozen(life):
    play_ok = 0
    for trial in range(20):
        jj = int(torch.randint(0, sum(len(ix) for ix in b.chains.sound_idx), (1,)))
        area, layer, unit = b.chains.fwd_src[jj].tolist()
        b.force_int = torch.zeros(b.n1, dtype=torch.bool, device=DEV)
        b.force_int[jj] = True
        b.step(torch.as_tensor(A0[0], device=DEV), 0.0)
        b.force_int = None
        b.step(torch.as_tensor(A0[0], device=DEV), 0.0)
        play_ok += int(bool(b.sound.state[layer][unit]))
check('extra: a return alone lights its own sound unit (g_ret 1.05)', play_ok == 20, lit=play_ok, of=20)


# 10. MC pattern pools (gen-0 motor synergy, 11-02): Dale (excitatory group -> own syllable's force neurons only,
#     inhibitory group -> other muscles' force neurons only, weights >= 0, subtracted), one sampled force neuron per
#     muscle of the syllable, every MC table acquired and decaying at gamma_motor, nothing from outside the motor
#     area reaches force neurons; a clean 0/1 syllable input is said clean and stops when the input stops.
from motor import Motor
cm = config(a.s, mc=True, force_per_muscle=8, g_ret_force=1.6, b_block=1.2)
mot = Motor(cm, life.coch, gammas(cm))
reg = mot.set_mc(life.ph.need_mat)
need = torch.as_tensor(life.ph.need_mat, device=DEV)
nI, mm = cm['mc_nI'], mot.m
syl_of = torch.arange(need.shape[0] * nI, device=DEV) // nI
ok_e = ok_i = ok_s = True
for p in range(0, need.shape[0] * nI, 7):
    s_ = int(syl_of[p])
    own = need[s_][mot.muscle_of]
    te = mot.mc_exc.idx[p][mot.mc_exc.idx[p] < mot.nf].long()
    ti = mot.mc_inh.idx[p][mot.mc_inh.idx[p] < mot.nf].long()
    ok_e &= bool(own[te].all()) and te.numel() == int(own.sum())
    ok_i &= bool((~own[ti]).all()) and ti.numel() == int((~own).sum())
    for tb in (mot.mc_in_e, mot.mc_in_i):
        pre = torch.nonzero((tb.idx == p).any(1)).squeeze(1)
        ok_s &= sorted(mot.muscle_of[pre].tolist()) == sorted(torch.nonzero(need[s_]).squeeze(1).tolist())
w_ok = all(bool((t.w >= 0).all()) and not t.birth and t.gamma == gammas(cm)['motor'] for t in mot.tables())
said = []
a0_sil = torch.as_tensor(A0[int(np.nonzero(sil)[0][0])], device=DEV)      # a silent tick (no brake)
for x in range(0, need.shape[0], 3):
    mot.force.set(torch.zeros(mot.nf, dtype=torch.bool, device=DEV))
    seq = []
    for t in range(6):
        mot.inject = need[x][mot.muscle_of].float() if t < 3 else None
        mus = mot.step(a0_sil, torch.zeros(mot.nf, dtype=torch.bool, device=DEV),
                       torch.zeros(mot.ns, dtype=torch.bool, device=DEV))
        seq.append(life.ph.decode(mus.cpu().numpy()))
    mot.inject = None
    said.append(seq[:3] == [('s', x)] * 3 and seq[4:] == [None, None])
check('10. MC pools: Dale split, sampling, acquired at gamma_motor; clean input said and stops', ok_e and ok_i and ok_s
      and w_ok and all(said), exc_targets_own=ok_e, inh_targets_rivals=ok_i, samples_one_per_muscle=ok_s,
      acquired_decaying=w_ok, clean_said_and_stops=f'{sum(said)}/{len(said)}', registry=reg)

save_json(a.out, R)
sys.exit(0 if all(v['ok'] for v in R.values()) else 1)
