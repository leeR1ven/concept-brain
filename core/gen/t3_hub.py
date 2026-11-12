"""Generator v2 (GENERATOR.md 11-12; Fable a16, a17): brain-constrained sheets after Garagnani, Tomasello and
Pulvermueller (2008-2018) with a Diehl & Cook (2015) first stage.

Sheets (2-D, excitatory cells; inhibition is a graded population, a registered shortcut):
  A  auditory first stage: each cell samples ~200 ear fibres across the tonotopic axis (Gaussian, sigma 1/3 of it);
  V  visual first stage: each cell samples ~100 eye cells across the image;
  H  hub: where A's and V's long axons both land (plus a skip projection from the ear); local links inside it.
Sulci (designer 11-12): local links only inside a sheet; between sheets only long axons. Feedback H->A, H->V is
born weak and learned.
Neurons: leaky integrate-and-fire, 5 ms steps, white noise, fast adaptation (per spike, tau 10 steps), a slow
adaptive threshold (Diehl & Cook), sheet-wide inhibition in proportion to the share firing; local inhibition in H.
ONE rule on every excitatory link (Artola-Broecher-Singer, as in Garagnani 2009 / Tomasello 2018):
  pre active and post spiked                         -> stronger   ("came")
  pre active, no post spike, excitatory drive >= th- -> weaker     ("pushed, did not come")
  pre silent and post spiked                         -> weaker     (what varies while the cell fires goes:
                                                                    the designer's way of forming abstraction)
Forward groups keep a fixed sum per cell and source; local and feedback groups are capped.
Run: birth calibration -> phase a (A and V learn; H runs, frozen) -> gate (A, V identity) -> phase b (H and feedback
learn; A and V at a tenth). Probes never learn or move thresholds.
usage: python -m core.gen.t3_hub [key=value ...]   (sheets=A: the dry run, ear and A only)"""
import json
import os
import sys
import time

import numpy as np
import torch

from ..config import config
from ..world import phon
from ..world.corpus import Dialogues, script_dialogue
from ..world.voice import Cochlea, Voice
from ..world.voice import ADULTS
from .concept_eval import evaluate, random_projection
from .grow import DEV
VIDX = {a: i for i, a in enumerate(ADULTS)}

P = dict(data=os.environ.get('CONCEPT_DATA', 'concept-data'), seed=1, B=16, hold=3, steps=10, p_see=0.5, side=16, sheets='AVH',
         # organs (as in t2_concept)
         tau_ad=3.0, k_ad=0.6, th_lo=0.05, th_hi=1.2, slope=0.08, p_sp=0.005, p_max=0.4, g_eye=0.8, norm=1,
         sigma_n=0.1, c_norm=0.5,
         # sizes and candidates
         nA=3000, nV=800, nH=2500, K_EA=200, sig_EA=0.333, K_YV=100, sig_YV=0.333, K_AH=700, K_VH=400, K_EH=0,
         K_HH=40, sig_HH=0.1, K_HA=30, K_HV=20,
         # drive shares and caps
         d_mean=0.5, sh_AH=0.6, sh_VH=0.4, sh_EH=0.0, cap_HH=0.2, cap_fb=0.1, wmax_x=5.0, w_fb0=0.02,
         # neurons
         tau_m=4.0, noise=0.05, a_fast=0.3, tau_fast=10.0, a_slow=0.03, tau_slow=2e4, k_G=1.0, tau_G=2.0,
         target=0.03, k_L=0.3, r_L=0.1, tau_L=2.0,
         # the rule
         tau_pre=4.0, pre_on=0.4, tau_D=4.0, th_minus=0.85, lr_AV=0.001, lr_H=0.01, lr_slow=0.0005, b_slow=0.1,
         k_F=0.8, wmax_th=0.1, cap_HH_th=1.0, cap_fb_th=0.5, slow_floor=0.5,
         # run
         birth_ticks=600, n_a=1000, n_b=2000, probe_every=500, id_nd=256, gate_sil=0.70, gate_mid=0.70,
         gate_V=0.95, gate=1, gate_plus=0.5, ckpt='', learn_first=1, th_plus=0.0, th_plus_q=0.9, n_b1=1000, n_held=10,
         plastic_HH=0, plastic_fb=0, bound_AH=1, plastic_VH=0, slow_H=0, wmax_k=1.2, cap_row_x=3.0,
         sat_events=30.0, c2_frac=0.1, rule_AH='b', eta_b=0.1, push_min=0.5, core_min=15, wmax_kb=3.0, eval_nd=768, eval_ntr=6,
         held_voice=-1, tgt_q=0.03, fbV0=0)
for a in sys.argv[1:] if __name__ == '__main__' else os.environ.get('GEN_ARGS', '').split():
    k, v = a.split('=', 1)
    try:
        P[k] = json.loads(v)
    except json.JSONDecodeError:
        P[k] = v
torch.manual_seed(P['seed'])
g = torch.Generator(device='cpu').manual_seed(P['seed'])
t0 = time.time()

# ---- the world and the organs (unchanged from t2_concept)
c = config(levels_per_s=4, syl_ticks=1, gap_turn=[2, 4], gap_dlg=[6, 12])
coch = Cochlea(Voice(phon.load_inventory()), c, P['seed'])
train = Dialogues.load(os.path.join(P['data'], 'train.npz'))
NE, ZE = coch.NC, coch.Z
n_ear, side = coch.n, P['side']
n_eye = side * side * 2
ORG = None


def ear(srcs):
    return coch.energy(srcs).astype(np.float32)


def eye(sid):
    if sid is None:
        return np.full(side * side, 0.5, np.float32)
    return (np.random.default_rng([4321, 5150, int(sid) + 1]).random(side * side) < 0.5).astype(np.float32)


def organ_in(items):
    """per stream (ear energy, eye image) -> firing probability per step of the ear fibres and the eye cells"""
    global ORG
    e = torch.as_tensor(np.stack([a for a, _ in items]), device=DEV)
    if ORG is None or ORG.shape != e.shape:
        ORG = e.clone()
    ORG += (e - ORG) / P['tau_ad']
    eff = e - P['k_ad'] * ORG
    th = torch.linspace(P['th_lo'], P['th_hi'], ZE, device=DEV)
    r = torch.sigmoid((eff[:, :, None] - th) / P['slope'])
    if P['norm']:
        r = P['c_norm'] * r / (P['sigma_n'] + r.mean((1, 2), keepdim=True))
    pe = P['p_sp'] + (P['p_max'] - P['p_sp']) * r.clamp(0, 1)
    img = torch.as_tensor(np.stack([b for _, b in items]), device=DEV).view(-1, 1, side, side)
    sur = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(img, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
    on, off = (img - sur).clamp_min(0), (sur - img).clamp_min(0)
    rv = P['g_eye'] * torch.stack([on, off], -1).reshape(img.shape[0], -1)
    if P['norm']:
        rv = P['c_norm'] * rv / (P['sigma_n'] + rv.mean(1, keepdim=True))
    pv = (P['p_sp'] + (P['p_max'] - P['p_sp']) * rv).clamp(0, 1)
    return pe.reshape(e.shape[0], -1), pv


HELD = set()                                              # syllables never shown a sign while learning (control)


def stream(ds, tag, see, mode='normal', learning=False):
    """per tick: (ear energy, eye image, sound label, sign label). see: share of syllables whose sign is shown.
    mode 'sight': the ear hears silence; 'conflict': each syllable with the sign of another, drawn at random"""
    rg = np.random.default_rng([P['seed'], tag, int(ds[0]) if len(ds) else 0])
    srcs = [script_dialogue(train, d, np.random.default_rng([P['seed'], tag, d]), (2, 4), 8, 1, P['hold']) for d in ds]
    pool = [int(xx[0]) for src in srcs for xx in src if xx is not None]
    out, shown, sgn, last = [], False, -1, -1
    silent = ear([])
    for src in srcs:
        for xx in src:
            lab = -1 if xx is None else int(xx[0])
            if xx is None:
                shown, sgn = False, -1
            elif lab != last:                           # decided once per syllable (held several ticks)
                shown = rg.random() < see and not (learning and (lab in HELD or VIDX.get(xx[2], -1) == P['held_voice']))
                sgn = lab
                if mode == 'conflict' and len(set(pool)) > 1:
                    while sgn == lab:
                        sgn = pool[int(rg.integers(len(pool)))]
            last = lab
            e = silent if (mode == 'sight' or xx is None) else ear([xx[:5]])
            out.append((e, eye(sgn) if (shown and sgn >= 0) else eye(None), lab, sgn if shown else -1,
                        -1 if xx is None else VIDX.get(xx[2], -1), -1 if xx is None else int(xx[5])))
    return out


# ---- sheets and links
SH = [s for s in 'AVH' if s in P['sheets']]
n_of = dict(A=P['nA'], V=P['nV'], H=P['nH'])
pos = {s: torch.rand(n_of[s], 2, generator=g) for s in SH}     # each sheet a unit square (cells' places)
ear_ch = (torch.arange(n_ear) // ZE).float() / NE                 # an ear fibre's place on the tonotopic axis
eye_xy = torch.stack([(torch.arange(n_eye) // 2 % side).float(), (torch.arange(n_eye) // 2 // side).float()], 1) / side


def pick(logp, K):
    """K distinct candidates per row, drawn with probability exp(logp) (Gumbel top-k)"""
    gum = -torch.log(-torch.log(torch.rand(logp.shape, generator=g).clamp(1e-12, 1 - 1e-12)))
    return (logp + gum).topk(K, dim=1).indices


class Group:
    """the links from one source to one sheet: idx [n_post, K], w [n_post, K]; kind 'fwd' (fixed sum per cell),
    'cap' (sum and each weight capped)"""
    def __init__(self, name, pre, post, idx, kind, w0=0.1, cap=None, lr='AV'):
        self.name, self.pre, self.post, self.kind, self.lr = name, pre, post, kind, lr
        self.idx = idx.to(DEV)
        self.w = (torch.rand(idx.shape, generator=g) * w0).to(DEV)
        self.S = None                                   # fixed sum (fwd), set at birth
        self.cap = cap
        self.wmax = None
        self.I = None                                   # inhibitory partners (rule b groups only)
        self.wmean0 = float(self.w.mean())
        self.ev = torch.zeros(3, device=DEV)            # counts of the three plasticity events

    def scale_to(self, S):
        self.S = S
        self.w *= S / self.w.sum(1, keepdim=True).clamp_min(1e-12)
        self.wmean0 = float(self.w.mean())


G = []
if 'A' in SH:
    c0 = pos['A'][:, :1]                                # a cell's preferred place on the tonotopic axis
    G.append(Group('ear>A', 'ear', 'A', pick(-(ear_ch[None] - c0) ** 2 / (2 * P['sig_EA'] ** 2), P['K_EA']), 'fwd'))
if 'V' in SH:
    d2 = ((eye_xy[None] - pos['V'][:, None]) ** 2).sum(-1)
    G.append(Group('eye>V', 'eye', 'V', pick(-d2 / (2 * P['sig_YV'] ** 2), P['K_YV']), 'fwd'))
if 'H' in SH:
    nH = P['nH']
    for nm, pre, K in (('A>H', 'A', P['K_AH']), ('V>H', 'V', P['K_VH']), ('ear>H', 'ear', P['K_EH'])):
        if (pre == 'ear' and K > 0) or pre in SH:
            npre = n_ear if pre == 'ear' else n_of[pre]
            G.append(Group(nm, pre, 'H', pick(torch.zeros(nH, npre), K), 'fwd', lr='H'))
    d2 = ((pos['H'][:, None] - pos['H'][None]) ** 2).sum(-1)
    d2.fill_diagonal_(float('inf'))
    G.append(Group('H>H', 'H', 'H', pick(-d2 / (2 * P['sig_HH'] ** 2), P['K_HH']), 'cap', lr='slow'))
    for nm, post, K in (('H>A', 'A', P['K_HA']), ('H>V', 'V', P['K_HV'])):
        if post in SH:
            G.append(Group(nm, 'H', post, pick(torch.zeros(n_of[post], nH), K), 'cap', w0=P['w_fb0'], lr='slow'))
    NBR = (d2 <= P['r_L'] ** 2).float().to(DEV)          # local inhibition neighbourhood in H
    NBR /= NBR.sum(1, keepdim=True).clamp_min(1.0)
GB = {gr.name: gr for gr in G}
if P['fbV0'] and 'H>V' in GB:                            # rule c: no path from H's own activity into its teacher
    GB['H>V'].w.zero_()                                  # (Fable a25 #3)

# ---- state
B = P['B']
st = {}


def reset():
    global ORG
    ORG = None
    for s in SH:
        n = n_of[s]
        st[s] = dict(V=torch.zeros(B, n, device=DEV), D=torch.zeros(B, n, device=DEV),
                     Af=torch.zeros(B, n, device=DEV), spk=torch.zeros(B, n, device=DEV),
                     prev=torch.zeros(B, n, device=DEV), tr=torch.zeros(B, n, device=DEV),
                     Gi=torch.zeros(B, 1, device=DEV), Li=torch.zeros(B, n, device=DEV))
    st['ear'] = dict(spk=torch.zeros(B, n_ear, device=DEV), tr=torch.zeros(B, n_ear, device=DEV))
    st['eye'] = dict(spk=torch.zeros(B, n_eye, device=DEV), tr=torch.zeros(B, n_eye, device=DEV))


TH0 = {s: 1.0 for s in SH}                                # each sheet's base threshold (calibrated at birth)
THS = {s: torch.ones(n_of[s], device=DEV) for s in SH}   # each cell's slow adaptive threshold
PLASTIC = {}                                              # group name -> learning-rate factor (0 = frozen)
RATIO_REC = None                                          # birth diagnostic: H's drive / need at its spikes
SLOW_ON = False                                           # the slow threshold moves (learning phases only)
a_m, a_D, a_f = np.exp(-1 / P['tau_m']), np.exp(-1 / P['tau_D']), np.exp(-1 / P['tau_fast'])
a_G, a_L, a_pre = np.exp(-1 / P['tau_G']), np.exp(-1 / P['tau_L']), np.exp(-1 / P['tau_pre'])
a_s = np.exp(-1 / P['tau_slow'])
tgt_sub = P['target'] / P['steps']                        # target share of cells spiking per step


def sub_step(pe, pv, learn):
    """one 5 ms step for every sheet; plasticity on the groups in PLASTIC when learn"""
    st['ear']['spk'] = (torch.rand_like(pe) < pe).float()
    st['eye']['spk'] = (torch.rand_like(pv) < pv).float()
    for src in ('ear', 'eye'):
        st[src]['tr'] = st[src]['tr'] * a_pre + st[src]['spk']
    new = {}
    for s in SH:
        z = st[s]
        exc = torch.zeros_like(z['V'])
        for gr in G:
            if gr.post == s:
                if gr.I is not None:                    # rule b: an excitatory and an inhibitory partner per link
                    cA = (st[gr.pre]['spk'][:, gr.idx] * (gr.w - gr.I)[None]).sum(-1)
                    TK['DA'] = TK['DA'] + cA            # the student drive (sound only), summed over the tick
                    exc = exc + cA
                else:
                    cV = (st[gr.pre]['spk'][:, gr.idx] * gr.w[None]).sum(-1)
                    if s == 'H' and gr.pre == 'V':
                        TK['DV'] = TK['DV'] + cV        # the other branch's drive (rule c's teacher), over the tick
                    exc = exc + cV
        exc = exc - P['k_F'] * exc.mean(1, keepdim=True)   # feedforward inhibition in the same step (every forward
        z['D'] = z['D'] * a_D + (1 - a_D) * exc       # pathway brings its own; Fable a18 #5); D: drive, not reset
        inh = P['k_G'] * TH0[s] * z['Gi']
        if s == 'H':
            inh = inh + P['k_L'] * TH0[s] * z['Li']
        noise = P['noise'] * TH0[s] * torch.randn_like(z['V'])
        z['V'] = z['V'] * a_m + (1 - a_m) * (exc - inh) + noise
        thr = THS[s][None] + z['Af']
        z['need'] = thr + inh                           # what firing needs now (for the rule's "pushed" band)
        # theta+ reference: what firing needs at the target activity (inhibition at its set level), fixed like their
        # theta+ on V; the moment's inhibition rises with every extra input and hid the sign's extra drive (11-12)
        z['ref'] = thr + (P['k_G'] + (P['k_L'] if s == 'H' else 0.0)) * TH0[s]
        sp = (z['V'] > thr).float()
        if RATIO_REC is not None and s == 'H':          # birth diagnostic for theta+ (Fable a19)
            seen = (pv.mean(1) > 2 * P['p_sp'])[:, None].expand_as(sp)
            r_ = z['D'] / z['ref'].clamp_min(1e-9)
            RATIO_REC['shown'].append(r_[(sp > 0) & seen])
            RATIO_REC['sound'].append(r_[(sp > 0) & ~seen])
        z['V'] = torch.where(sp > 0, torch.zeros_like(z['V']), z['V'])
        z['Af'] = z['Af'] * a_f + P['a_fast'] * TH0[s] * sp
        new[s] = sp
    for s in SH:                                       # all sheets update together (their inputs were last step's)
        z = st[s]
        sp = new[s]
        z['prev'], z['spk'] = z['spk'], sp
        z['tr'] = z['tr'] * a_pre + sp
        z['Gi'] = z['Gi'] * a_G + (1 - a_G) * sp.mean(1, keepdim=True) / tgt_sub
        if s == 'H':
            z['Li'] = z['Li'] * a_L + (1 - a_L) * (sp @ NBR.T) / tgt_sub
        if SLOW_ON and any(PLASTIC.get(gr.name, 0) > 0 for gr in G if gr.post == s):   # (only where links learn)
            # slow adaptive threshold (Diehl & Cook), two-sided around the
            THS[s].add_(P['a_slow'] * TH0[s] * (sp.mean(0) - tgt_sub))   # target, averaged over the streams
            THS[s].clamp_(min=P['slow_floor'] * TH0[s])  # (Fable a18 #2: summed over streams it silenced the sheet)
    if learn:
        for gr in G:
            f = PLASTIC.get(gr.name, 0.0)
            if f > 0 and gr.I is None:                   # rule-b groups learn once per tick (learn_tick)
                plast(gr, f)


def plast(gr, f):
    """the ABS rule on one group, summed over the streams; then the group's sum rule"""
    zpost = st[gr.post]
    act = (st[gr.pre]['tr'][:, gr.idx] > P['pre_on'])                 # [B, n, K] pre active
    spiked = ((zpost['spk'] + zpost['prev']) > 0)[:, :, None]        # post spiked in the last 2 steps
    thm = P['th_minus'] * zpost['need']                       # near miss: within th_minus of what firing needs
    push = (zpost['D'] >= thm)[:, :, None]                   # (Fable a18 #1)
    rec = spiked
    if P['th_plus'] > 0:                                     # learning needs STRONG depolarisation (the ABS rule's
        strong = (zpost['D'] >= P['th_plus'] * zpost['ref'])[:, :, None]    # theta+; Fable a19): a cell barely
        rec = spiked & strong                                # fired by one sense does not learn; convergence does
    c1 = act & rec
    c2 = act & ~spiked & push                                # "pushed, did not come": no spike at all (a weak spike
    c3 = ~act & rec                                          # is neither; Fable a20 bug)
    na = act.float().sum(-1, keepdim=True)
    wf = na / (act.shape[-1] - na).clamp_min(1.0)           # case 3 weighs as much in all as case 1 (Fable a19)
    if gr.kind == 'bound':                                   # bounded weights (their model): about sat_events
        s1 = gr.wmax / P['sat_events'] * f                   # strengthenings fill a link; weakening near misses is a
        dw = (s1 * c1.float() - s1 * P['c2_frac'] * c2.float() - s1 * c3.float() * wf).sum(0)   # tenth (a20)
    else:
        lr = {'AV': P['lr_AV'], 'H': P['lr_H'], 'slow': P['lr_slow']}[gr.lr] * f
        dw = lr * gr.wmean0 * (c1.float() - c2.float() - c3.float() * wf).sum(0)
    gr.ev += torch.stack([c1.float().sum(), c2.float().sum(), c3.float().sum()])
    gr.w.add_(dw).clamp_(min=0.0)
    if gr.kind == 'fwd':
        gr.w.mul_(gr.S / gr.w.sum(1, keepdim=True).clamp_min(1e-12))
    elif gr.kind == 'bound':                                 # each weight in [0, w_max]; a loose row cap (down only)
        gr.w.clamp_(max=gr.wmax)
        s_ = gr.w.sum(1, keepdim=True)
        gr.w.mul_(torch.where(s_ > gr.cap, gr.cap / s_.clamp_min(1e-12), torch.ones_like(s_)))
    else:
        gr.w.clamp_(max=P['wmax_th'] * TH0[gr.post])
        cap = gr.cap if gr.cap is not None else 1e9
        s_ = gr.w.sum(1, keepdim=True)
        gr.w.mul_(torch.where(s_ > cap, cap / s_.clamp_min(1e-12), torch.ones_like(s_)))


def run(n_d, d0, tag, learn, see=True, mode='normal', record=False):
    """B dialogues in parallel streams; record: per tick spike counts per sheet and the ear, and the labels"""
    per = max(1, n_d // B)
    S = [stream(range(d0 + b * per, d0 + (b + 1) * per), tag, P['p_see'] if see is True else see, mode, learn)
         for b in range(B)]
    T = min(len(s) for s in S)
    reset()
    rec = {k: [] for k in SH + ['ear', 'eye']}
    labs, sgns, vcs, dls = [], [], [], []
    for t in range(T):
        pe, pv = organ_in([(S[b][t][0], S[b][t][1]) for b in range(B)])
        cnt = {k: 0.0 for k in rec}
        TK['DA'] = 0.0
        TK['DV'] = 0.0
        TK['nH'] = 0.0
        TK['nA'] = 0.0
        for _ in range(P['steps']):
            sub_step(pe, pv, learn)
            if 'H' in SH:
                TK['nH'] = TK['nH'] + st['H']['spk']
                TK['nA'] = TK['nA'] + st['A']['spk']
            if record:
                for k in rec:
                    cnt[k] = cnt[k] + st[k]['spk']
        # a syllable's first tick: the V and A drives reach H a step or two late, so its tick mean is partial; it is
        # left out of the targets' statistics and of rule c's update (Fable a25 #1)
        nfirst = torch.tensor([t > 0 and S[b][t][2] == S[b][t - 1][2] for b in range(B)], device=DEV)
        if DV_REC is not None:                          # birth: the V-branch drive on shown ticks (rule c targets)
            shw = torch.tensor([S[b][t][3] >= 0 for b in range(B)], device=DEV) & nfirst
            if bool(shw.any()):
                DV_REC['dv'].append(v_centred()[shw].half())
                DV_REC['sign'].extend([S[b][t][3] for b in range(B) if bool(shw[b])])
        if learn and 'A>H' in GB and GB['A>H'].I is not None and PLASTIC.get('A>H', 0) > 0:
            learn_tick(GB['A>H'], torch.tensor([S[b][t][2] >= 0 for b in range(B)], device=DEV),
                       torch.tensor([S[b][t][3] >= 0 for b in range(B)], device=DEV) & nfirst)
        if record:
            for k in rec:
                rec[k].append(cnt[k].half())
            labs.append([S[b][t][2] for b in range(B)])
            sgns.append([S[b][t][3] for b in range(B)])
            vcs.append([S[b][t][4] for b in range(B)])
            dls.append([S[b][t][5] for b in range(B)])
    if record == 'meta':
        return ({k: torch.stack(v, 1) for k, v in rec.items()}, np.array(labs).T, np.array(sgns).T,
                np.array(vcs).T, np.array(dls).T)
    if record:
        return {k: torch.stack(v, 1) for k, v in rec.items()}, np.array(labs).T, np.array(sgns).T
    return T


TK = {}                                                   # per-tick accumulators (student drive, spike counts)
DV_REC = None                                             # birth: V-branch drive per shown tick, and the sign
THV = None                                                # rule c: each H cell's fixed V-branch threshold (birth)


def v_centred():
    """the V-branch drive on each H cell, tick mean, centred like the student drive (feedforward inhibition)"""
    dv = TK['DV'] / P['steps'] if torch.is_tensor(TK['DV']) else torch.zeros(B, P['nH'], device=DEV)
    return dv - P['k_F'] * dv.mean(1, keepdim=True)
LB = dict(fired=[], quiet=[])                             # learning diagnostics: D_A / ref of fired and quiet H cells
X2U = None                                                # each H cell's usual sum of squared input rates in sound


def learn_tick(gr, sound, shown=None):
    """rule b on A>H once per 50 ms tick (Fable a21): came = the H cell fired in the tick (on a shown tick that is the
    sign's set); pred = clip(D_A / ref): what sound alone would do; fired -> 1 - pred; quiet and pushed (D_A >= half
    ref) -> -pred; else nothing. dw_j = eta * ref * dd * x_j / sum x^2 (x_j = pre spikes per step in the tick), so one
    update moves D_A / ref by eta * dd. Positive goes to E, negative to the inhibitory partner I; both in [0, w_max]."""
    global X2U
    ref = (THS['H'] + (P['k_G'] + P['k_L']) * TH0['H'])[None]          # [1, nH]
    DA = TK['DA'] / P['steps']                                          # [B, nH] tick mean of the student drive
    DA = DA - P['k_F'] * DA.mean(1, keepdim=True)                       # as the soma sees it (feedforward inhibition)
    x = TK['nA'][:, gr.idx] / P['steps']                                 # [B, nH, K] pre spikes per step
    x2 = (x * x).sum(-1)                                                 # [B, nH]
    if bool(sound.any()):
        m_ = x2[sound].mean(0)
        X2U = m_.clone() if X2U is None else X2U + (m_ - X2U) / 1000.0
    fl = X2U[None] if X2U is not None else x2
    pred = (DA / ref).clamp(0.0, 1.0)
    if P['rule_AH'] == 'c':                       # rule c (Fable a24; de Sa 1994 in dendritic form): the target is the
        fired = v_centred() > THV[None]           # OTHER branch's drive above its fixed threshold, never the cell's own
        on = (sound & shown)[:, None]             # spike; no update while the other branch is silent (unseen ticks)
        dd = torch.where(on, fired.float() - pred, torch.zeros_like(pred))
        fired = fired & on
    else:
        fired = TK['nH'] > 0
        dd = torch.where(fired, 1.0 - pred, torch.where(DA >= P['push_min'] * ref, -pred, torch.zeros_like(pred)))
    d = (P['eta_b'] * ref * dd / torch.maximum(x2, fl).clamp_min(1e-9))[:, :, None] * x
    d = d.sum(0)                                                         # each stream's hearing counts
    W = (gr.w - gr.I + d).clamp(-gr.wmax, gr.wmax)                    # the signed link moves once by d (Fable a22 #1:
    gr.w.copy_(W.clamp(min=0))                                          # the split update moved it twice), then it is
    gr.I.copy_((-W).clamp(min=0))                                       # carried by E (positive) or I (negative)
    gr.ev += torch.stack([fired.float().sum(), (dd < 0).float().sum(), torch.zeros((), device=DEV)])
    if len(LB['fired']) < 2000:
        r_ = DA / ref
        quiet = ~fired if P['rule_AH'] != 'c' else (~fired & (sound & shown)[:, None])   # rule c: non-targets, shown
        if bool(quiet.any()):
            LB['fired'].append(r_[fired].mean().item() if bool(fired.any()) else float('nan'))
            LB['quiet'].append(r_[quiet].mean().item())


# ---- birth: forward sums and base thresholds calibrated so each sheet fires near its target share
def birth():
    global SLOW_ON
    SLOW_ON = False
    PLASTIC.clear()
    # mean firing per step of every source over speech with signs (organs first)
    R, L0, _ = run(64, 16000, 7000, False, True, record=True)
    snd = torch.as_tensor(L0 >= 0, device=DEV)         # calibrate on ticks with sound (Fable a18 #6)
    rate = {k: (R[k][snd].float().mean() / P['steps']).item() for k in R}
    for gr in G:                                         # organ-fed forward groups: mean drive per step = d_mean
        if gr.kind == 'fwd' and gr.pre in ('ear', 'eye') and gr.post in ('A', 'V'):   # (drive = sum x input rate)
            gr.scale_to(P['d_mean'] / max(rate[gr.pre], 1e-6))
    for rnd in range(2):                                 # A/V thresholds, then H (its inputs need A and V firing)
        for _ in range(8):
            R, L0, _ = run(48, 16100 + rnd * 100, 7001, False, True, record=True)
            snd = torch.as_tensor(L0 >= 0, device=DEV)
            for s in SH:
                share = (R[s][snd] > 0).float().mean().item()
                TH0[s] *= float(np.exp(np.clip(0.8 * np.log(max(share, 1e-4) / P['target']), -1.0, 1.0)))
                THS[s].fill_(TH0[s])
        rate = {k: (R[k][snd].float().mean() / P['steps']).item() for k in R}
        if rnd == 0 and 'H' in SH:
            sh = dict(A=P['sh_AH'], V=P['sh_VH'], ear=P['sh_EH'])
            for gr in G:
                if gr.post == 'H' and gr.kind == 'fwd':
                    gr.scale_to(P['d_mean'] * sh[gr.pre] / max(rate[gr.pre], 1e-6))
    for gr in G:                                         # caps in threshold units (Fable a18 #4): a row of local or
        if gr.name == 'H>H':                             # feedback links can give at most this much at once
            gr.cap = P['cap_HH_th'] * TH0['H']
        if gr.name in ('H>A', 'H>V'):
            gr.cap = P['cap_fb_th'] * TH0[gr.post]
    if 'H' in SH and P['th_plus'] == 0.0:               # theta+: the 90th percentile of D / need at H spikes on
        global RATIO_REC                                 # sound-only ticks; a sign-and-sound spike should pass it
        RATIO_REC = dict(shown=[], sound=[])
        run(48, 16400, 7002, False, True)
        rs, rw = torch.cat(RATIO_REC['sound']), torch.cat(RATIO_REC['shown'])
        RATIO_REC = None
        if rs.numel() >= 0.1 * max(rw.numel(), 1):      # sound alone fires H often: theta+ above its usual spikes
            P['th_plus'] = float(torch.quantile(rs.float()[:200000], P['th_plus_q']))
            P['th_plus_rule'] = 'sound-only q'
        else:                                            # sound alone hardly fires H (weak born sound drive): the
            P['th_plus'] = float(torch.quantile(rw.float()[:200000], 0.5))   # strong half of sign-driven spikes learn
            P['th_plus_rule'] = 'shown median'
        P['th_plus_pass'] = round(float((rw >= P['th_plus']).float().mean()), 3) if rw.numel() else -1
        print('THETA+', json.dumps(dict(th_plus=round(P['th_plus'], 3), rule=P['th_plus_rule'],
                                        shown_spikes_above=P['th_plus_pass'],
                                        sound_median=round(float(rs.float().median()), 3) if rs.numel() else -1,
                                        shown_median=round(float(rw.float().median()), 3) if rw.numel() else -1,
                                        n_sound=int(rs.numel()), n_shown=int(rw.numel()))), flush=True)
    if 'A>H' in GB and P['bound_AH']:                    # w_max: the usual number of active A inputs, all at w_max,
        gr = GB['A>H']                                    # bring a cell to wmax_k x the theta+ reference (Fable a20)
        n_act = active_inputs()['A>H']
        frac = n_act / gr.idx.shape[1]
        spk_act = rate['A'] / max(frac, 1e-6)             # spikes per step of an input while it is active
        ref_H = float((THS['H'] + (P['k_G'] + P['k_L']) * TH0['H']).mean())
        gr.wmax = (P['wmax_kb'] if P['rule_AH'] in ('b', 'c') else P['wmax_k']) * ref_H / max(n_act * spk_act, 1e-6)
        gr.kind = 'bound'
        gr.w.clamp_(max=gr.wmax)
        if P['rule_AH'] in ('b', 'c'):
            gr.I = torch.zeros_like(gr.w)
        gr.cap = P['cap_row_x'] * float(gr.w.sum(1).mean())
        print('A>H BOUNDED', json.dumps(dict(w_max=round(gr.wmax, 4), born_mean=round(float(gr.w.mean()), 4),
                                             w_max_over_born_mean=round(gr.wmax / float(gr.w.mean()), 2),
                                             w_max_over_TH0=round(gr.wmax / TH0['H'], 2), active=n_act,
                                             spikes_per_step_active=round(spk_act, 3), row_cap=round(gr.cap, 2))),
              flush=True)
    if 'H' in SH and P['rule_AH'] == 'c':
        v_targets()
    share = {s: round((R[s][snd] > 0).float().mean().item(), 4) for s in SH}
    print('BIRTH', json.dumps(dict(theta0={s: round(TH0[s], 3) for s in SH}, share_per_tick=share,
                                   rates={k: round(v, 5) for k, v in rate.items()},
                                   sums={gr.name: round(float(gr.w.sum(1).mean()), 3) for gr in G},
                                   active_inputs=active_inputs())), f'({time.time() - t0:.0f} s)', flush=True)


def v_targets():
    """rule c (Fable a24): each H cell's V-branch threshold, set once at birth so the cell is a target for tgt_q of
    the signs (per-sign mean drive, quantile over signs), then fixed. Check: target set per shown tick, its
    reliability per sign (the cells in >= 80% of a sign's ticks), and each cell's share of signs."""
    global DV_REC, THV
    DV_REC = dict(dv=[], sign=[])
    run(96, 16600, 7003, False, 1.0)
    dv, sg = torch.cat(DV_REC['dv']).float(), np.array(DV_REC['sign'])
    DV_REC = None
    signs = np.unique(sg)
    M = torch.stack([dv[torch.as_tensor(sg == c_, device=DEV)].mean(0) for c_ in signs])   # [n_signs, nH]
    well = torch.as_tensor([int((sg == c_).sum()) >= 8 for c_ in signs], device=DEV)
    THV = torch.quantile(M[well], 1.0 - P['tgt_q'], dim=0)   # rare signs' noisy means kept out (Fable a25 #2)
    T = dv > THV[None]
    core, tick_in_core = [], []
    for c_ in signs:
        m = torch.as_tensor(sg == c_, device=DEV)
        if int(m.sum()) >= 8:
            fr = T[m].float().mean(0)
            cs = fr >= 0.8
            core.append(int(cs.sum()))
            if bool(cs.any()):
                tick_in_core.append(float((T[m][:, cs]).float().mean()))
    per_cell = (M[well] > THV[None]).float().mean(0)
    print('V TARGET', json.dumps(dict(signs=int(len(signs)), well_sampled=int(well.sum()), ticks=int(len(sg)),
                                      set_per_tick=round(float(T.float().sum(1).mean()), 1),
                                      set_by_sign_mean=round(float((M[well] > THV[None]).float().sum(1).mean()), 1),
                                      core80_median=float(np.median(core)) if core else 0.0,
                                      core_cells_on_per_tick=round(float(np.mean(tick_in_core)), 3) if tick_in_core else 0,
                                      cell_share_of_signs=[round(float(per_cell.mean()), 4),
                                                           round(float(per_cell.max()), 4)],
                                      cells_never_target=round(float((per_cell == 0).float().mean()), 3))), flush=True)
    P['v_core80'] = float(np.median(core)) if core else 0.0


def active_inputs():
    """Fable a17 warning 1: per group, the mean number of candidates active (trace > pre_on) at a post cell"""
    out = {}
    for gr in G:
        a = (st[gr.pre]['tr'][:, gr.idx] > P['pre_on']).float().sum(-1).mean().item()
        out[gr.name] = round(a, 1)
    return out


# ---- probes (no plasticity, no threshold change)
def ncm(X, Y, keep=None, split=None):
    """nearest class mean identity, 60 commonest classes, half the hearings make the means"""
    Y = torch.as_tensor(Y, device=DEV)
    top = torch.argsort(torch.bincount(Y.clamp_min(0)), descending=True)[:60]
    k = torch.isin(Y, top) & (Y >= 0)
    if keep is not None:
        k = k & torch.as_tensor(keep, device=DEV)
    half = torch.rand(len(Y), generator=torch.Generator(device='cpu').manual_seed(3)).to(DEV) < 0.5
    cls = [c_ for c_ in top.tolist() if bool((half & k & (Y == c_)).any())]
    M = torch.stack([X[half & k & (Y == c_)].mean(0) for c_ in cls])
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    te = (~half) & k & torch.isin(Y, torch.as_tensor(cls, device=DEV))
    if not bool(te.any()):
        return -1.0
    Xn = X[te] / X[te].norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos_ = {c_: i for i, c_ in enumerate(cls)}
    ti = torch.as_tensor([pos_[int(c_)] for c_ in Y[te].tolist()], device=DEV)
    alive = X[te].norm(dim=1) > 0
    ok = ((Xn @ Mn.T).argmax(1) == ti) & alive
    if split is not None:                                # accuracy of the held-out classes and of the rest
        yt = Y[te].tolist()
        hm = torch.as_tensor([int(y_) in split for y_ in yt], device=DEV)
        return [round(ok[hm].float().mean().item(), 3) if bool(hm.any()) else -1,
                round(ok[~hm].float().mean().item(), 3) if bool((~hm).any()) else -1]
    return round(ok.float().mean().item(), 3)


def hearings(R, L, Sg, lag=0):
    """per hearing: spike counts over its held ticks (from lag), the sound label, the sign label, after silence"""
    hold = P['hold']
    X = {k: [] for k in R}
    Y, Z, F = [], [], []
    for b in range(L.shape[0]):
        for t in range(L.shape[1] - hold - lag):
            y = L[b, t]
            if y >= 0 and (t == 0 or L[b, t - 1] != y):
                for k in R:
                    X[k].append(R[k][b, t + lag:t + lag + hold].float().sum(0))
                Y.append(int(y))
                Z.append(int(Sg[b, t]))
                F.append(t == 0 or L[b, t - 1] < 0)
    return {k: torch.stack(v) for k, v in X.items()}, np.array(Y), np.array(Z), np.array(F)


def probe(tag):
    global SLOW_ON
    out = {}
    slow_was, SLOW_ON = SLOW_ON, False                  # probes never move thresholds (Fable a17 warning 6)
    R, L, Sg = run(P['id_nd'], 15000, 9200, False, False, record=True)
    for lag in (0, 1):
        X, Y, Z, F = hearings(R, L, Sg, lag)
        for k in X:
            v = ncm(X[k], Y)
            if v > out.get(k, -1):
                out[k] = v
        if 'A' in X:
            out[f'A after silence lag{lag}'] = ncm(X['A'], Y, F)
            out[f'A mid-turn lag{lag}'] = ncm(X['A'], Y, ~F)
    out['share'] = {s: round((R[s] > 0).float().mean().item(), 4) for s in SH}
    if 'H' in SH or 'V' in SH:                          # sight only: V and H read the sign
        Rs, Ls, Ss = run(P['id_nd'], 15000, 9200, False, 1.0, mode='sight', record=True)
        Xs, Ys, _, _ = hearings(Rs, Ls, Ss, 0)
        for k in [s for s in ('V', 'H') if s in Xs]:
            out[k + ' sight'] = ncm(Xs[k], Ys)
        if 'H' in Xs:
            sets = (Xs['H'] > 0).float()
            same, other = jaccard(sets, Ys)
            out['H sight set overlap same/other'] = [same, other]
            # row 5: does sound alone fire that syllable's sight set (cells firing in >= half its showings)?
            Xh, Yh, _, _ = hearings(R, L, Sg, 0)
            sset = {}
            for c_ in np.unique(Ys):
                m = torch.as_tensor(Ys == c_, device=DEV)
                if int(m.sum()) >= 2:
                    sset[int(c_)] = sets[m].mean(0) >= 0.5
            ov = []
            for i in range(len(Yh)):
                s_ = sset.get(int(Yh[i]))
                if s_ is not None and int(s_.sum()) > 0:
                    ov.append(((Xh['H'][i] > 0) & s_).float().sum().item() / float(s_.sum()))
            out['H sound fires sight set'] = round(float(np.mean(ov)), 3) if ov else -1
            # the teacher's core sets (cells firing in >= half the showings of a sign; Fable a21) and whether sound
            # fires its own syllable's core more than other syllables' cores
            keys = [k_ for k_ in sset if int(sset[k_].sum()) > 0]
            nshow = {int(c_): int((Ys == c_).sum()) for c_ in np.unique(Ys)}
            well = [int(sset[k_].sum()) for k_ in sset if nshow.get(k_, 0) >= 8]   # (Fable a22 #2: a class shown twice
            out['H core set size median'] = float(np.median(well)) if well else 0.0  # makes a union, not a core)
            if len(keys) > 1:
                Sm = torch.stack([sset[k_] for k_ in keys]).float()            # [n_cls, nH]
                kpos = {k_: i for i, k_ in enumerate(keys)}
                sm_, ot_ = [], []
                for i in range(len(Yh)):
                    j = kpos.get(int(Yh[i]))
                    if j is None:
                        continue
                    o_ = (Sm @ (Xh['H'][i] > 0).float()) / Sm.sum(1)
                    sm_.append(o_[j].item())
                    ot_.append(((o_.sum() - o_[j]) / (len(keys) - 1)).item())
                out['H sound fires core same/other'] = [round(float(np.mean(sm_)), 3), round(float(np.mean(ot_)), 3)] if sm_ else -1
            # cross-modal (the function that matters): class means from SIGHT-only H codes, sound-only H codes read
            # against them. Binding = hearing a syllable gives the code its sign gives.
            out['H cross-modal sound->sign'] = cross_ncm(Xs['H'], Ys, Xh['H'], Yh)
            yy = np.array([y_ for y_ in Yh if y_ in set(Ys.tolist())])
            out['cross-modal baseline (commonest class)'] = round(float(np.bincount(yy).max() / max(len(yy), 1)), 3) if len(yy) else -1
            out['H alive under sound'] = round(float((Xh['H'].norm(dim=1) > 0).float().mean()), 3)
            if HELD:                                     # held-out syllables (never shown a sign while learning)
                out['H sound held/other'] = ncm(Xh['H'], Yh, split=HELD)
                out['H cross-modal held/other'] = cross_ncm(Xs['H'], Ys, Xh['H'], Yh, split=HELD)
        if 'H' in SH:                                    # conflict: sound with a mismatched sign
            Rc, Lc, Sc = run(P['id_nd'], 15000, 9200, False, 1.0, mode='conflict', record=True)
            Xc, Yc, Zc, _ = hearings(Rc, Lc, Sc, 0)
            out['H conflict by sound/by sign'] = [ncm(Xc['H'], Yc), ncm(Xc['H'], Zc)]
    if 'H>A' in GB:                                      # row 8: A without feedback
        w_ = GB['H>A'].w.clone()
        GB['H>A'].w.zero_()
        Rf, Lf, Sf = run(P['id_nd'], 15000, 9200, False, False, record=True)
        GB['H>A'].w.copy_(w_)
        Xf, Yf, _, _ = hearings(Rf, Lf, Sf, 0)
        out['A without feedback'] = ncm(Xf['A'], Yf)
    if P['eval_nd'] and 'H' in SH:                      # the designer's definition (concept_eval.py; Fable a23)
        Re, Le, Se, Ve, De = run(P['eval_nd'], 15000, 9300, False, False, record='meta')
        hold = P['hold']
        Xe = {k: [] for k in ('ear', 'A', 'H')}
        ye, ve, de = [], [], []
        for b in range(Le.shape[0]):
            for t in range(Le.shape[1] - hold):
                y_ = Le[b, t]
                if y_ >= 0 and (t == 0 or Le[b, t - 1] != y_):
                    for k in Xe:
                        Xe[k].append(Re[k][b, t:t + hold].float().sum(0))
                    ye.append(int(y_))
                    ve.append(int(Ve[b, t]))
                    de.append(int(De[b, t]))
        Xe = {k: torch.stack(v_) for k, v_ in Xe.items()}
        kH = float((Xe['H'] > 0).float().sum(1).mean())
        Xe['RP'] = random_projection(Xe['A'], P['nH'], kH, seed=P['seed'])   # the "no concept" control
        ye, ve, de = np.array(ye), np.array(ve), np.array(de)
        ev = [evaluate(Xe, ye, ve, de, n_tr=P['eval_ntr'], seed=P['seed'], half=h) for h in (0, 1)]
        out['concept eval'] = ev[0]
        out['concept eval (halves swapped)'] = ev[1]
    SLOW_ON = slow_was
    print('PROBE', tag, json.dumps(out), f'({time.time() - t0:.0f} s)', flush=True)
    return out


def cross_ncm(Xa, Ya, Xb, Yb, split=None):
    """means from one set of hearings (e.g. sight), tested on another (e.g. sound), 60 commonest shared classes;
    each unit's mean over the test set is removed from the test codes (the shared part is not about the syllable)"""
    Ya_t, Yb_t = torch.as_tensor(Ya, device=DEV), torch.as_tensor(Yb, device=DEV)
    common = [int(c_) for c_ in torch.argsort(torch.bincount(Yb_t.clamp_min(0)), descending=True)[:60].tolist()
              if bool((Ya_t == c_).any())]
    if len(common) < 2:
        return -1.0
    Ma = torch.stack([Xa[Ya_t == c_].mean(0) for c_ in common])
    Ma = Ma - Ma.mean(0, keepdim=True)
    Mn = Ma / Ma.norm(dim=1, keepdim=True).clamp_min(1e-9)
    k = torch.isin(Yb_t, torch.as_tensor(common, device=DEV))
    Xt = Xb[k] - Xb[k].mean(0, keepdim=True)
    Xn = Xt / Xt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos_ = {c_: i for i, c_ in enumerate(common)}
    ti = torch.as_tensor([pos_[int(c_)] for c_ in Yb_t[k].tolist()], device=DEV)
    alive = Xb[k].norm(dim=1) > 0                       # a silent response names nothing (11-12: silent H made the
    ok = ((Xn @ Mn.T).argmax(1) == ti) & alive          # argmax constant, scoring the commonest class's frequency)
    if split is not None:
        hm = torch.as_tensor([int(y_) in split for y_ in Yb_t[k].tolist()], device=DEV)
        return [round(ok[hm].float().mean().item(), 3) if bool(hm.any()) else -1,
                round(ok[~hm].float().mean().item(), 3) if bool((~hm).any()) else -1]
    return round(ok.float().mean().item(), 3)


def jaccard(sets, Y):
    """mean Jaccard overlap of the firing sets between hearings of the same class and of different classes"""
    n = min(len(Y), 600)
    S_ = sets[:n]
    inter = S_ @ S_.T
    uni = S_.sum(1)[:, None] + S_.sum(1)[None] - inter
    j = (inter / uni.clamp_min(1)).cpu().numpy()
    Yn = Y[:n]
    same = (Yn[:, None] == Yn[None]) & ~np.eye(n, dtype=bool)
    diff = Yn[:, None] != Yn[None]
    return (round(float(j[same].mean()), 3) if same.any() else -1), round(float(j[diff].mean()), 3)


def learn_report(tag, ticks):
    """row 7: the three events per tick and the share of weights at a bound, per group"""
    out = {}
    for gr in G:
        ev = (gr.ev / max(ticks, 1)).tolist()
        top_ = gr.wmax if gr.kind == 'bound' else (P['wmax_th'] * TH0[gr.post] if gr.kind == 'cap' else float('inf'))
        atb = ((gr.w <= 1e-9) | (gr.w >= top_ * 0.999)).float().mean().item()
        out[gr.name] = dict(up=round(ev[0], 1), down_pushed=round(ev[1], 1), down_silent=round(ev[2], 1),
                            at_bound=round(atb, 3))
        gr.ev.zero_()
    if LB['fired']:
        key = 'rule c D_A/ref target/non-target (shown)' if P['rule_AH'] == 'c' else 'rule b D_A/ref fired/quiet'
        out[key] = [round(float(np.nanmean(LB['fired'])), 3), round(float(np.nanmean(LB['quiet'])), 3)]
        LB['fired'].clear()
        LB['quiet'].clear()
    print('LEARNING', tag, json.dumps(out), flush=True)


def learn_phase(n, d0, tag):
    done = 0
    while done < n:
        k = min(P['probe_every'], n - done)
        T = run(k, d0 + done, tag, True, True, record=False)
        done += k
        learn_report(f'{tag} {done}', T * B)
        yield done


def main():
    global SLOW_ON
    if P['n_held'] and 'H' in SH:                        # 10 of the commonest 60 probe syllables never get a sign
        labs = [x[2] for b in range(B) for x in stream(range(15000 + b * (P['id_nd'] // B),
                                                              15000 + (b + 1) * (P['id_nd'] // B)), 9200, False)]
        cnt = np.bincount(np.array([l_ for l_ in labs if l_ >= 0]))
        top = [int(c_) for c_ in np.argsort(-cnt)[:60]]
        rh = np.random.default_rng(P['seed'] + 31)
        HELD.update(int(c_) for c_ in rh.choice(top, size=P['n_held'], replace=False))
        print('HELD OUT', sorted(HELD), flush=True)
    birth()
    if P['learn_first']:
        probe('birth')
    # phase a: the first stages learn; H runs, frozen
    PLASTIC.clear()
    pr = probe('birth') if not P['learn_first'] else None
    if P['learn_first']:                                 # learn_first=0 (11-12): the first stages keep their birth
        for gr in G:                                     # state (they already hold the organ's information; their
            if gr.post in ('A', 'V') and gr.pre in ('ear', 'eye'):   # unsupervised learning lowered it)
                PLASTIC[gr.name] = 1.0
        SLOW_ON = True
        for done in learn_phase(P['n_a'], 0, 7600):
            pr = probe(f'a {done}')
    SLOW_ON = True
    ok_A = 'A' not in SH or (pr.get('A after silence lag0', 0) >= P['gate_sil'] and pr.get('A mid-turn lag0', 0) >= P['gate_mid'])
    ok_V = 'V' not in SH or pr.get('V sight', 0) >= P['gate_V']
    ok_plus = P.get('th_plus_pass', 1.0) >= P['gate_plus'] or P['rule_AH'] in ('b', 'c')   # (theta+ unused)
    if P['rule_AH'] == 'b':                              # the teacher must be thick enough (Fable a21)
        ok_plus = ok_plus and pr.get('H core set size median', 0) >= P['core_min']
    if P['rule_AH'] == 'c':                              # rule c: the V-branch target sets are thick and steady
        ok_plus = P.get('v_core80', 0) >= P['core_min']
    print('GATE', json.dumps(dict(A=ok_A, V=ok_V, theta_plus=ok_plus)), flush=True)
    if 'H' not in SH or (P['gate'] and not (ok_A and ok_V and ok_plus)):
        print('STOP: phase b not started (dry run, or the first stages did not pass)', flush=True)
        return
    # phase b: H and feedback learn; A and V at a tenth
    for gr in G:
        if gr.name in ('A>H', 'ear>H') or (gr.name == 'V>H' and P['plastic_VH']) or \
                (gr.name == 'H>H' and P['plastic_HH']) or (gr.name in ('H>A', 'H>V') and P['plastic_fb']):
            PLASTIC[gr.name] = 1.0
        elif gr.name in PLASTIC:
            PLASTIC[gr.name] = P['b_slow']
    if not P['slow_H']:                                  # the slow threshold retuned H under sign-always input until
        SLOW_ON = False                                  # sound alone could not fire it (Fable a20): off
    see0 = P['p_see']
    P['p_see'] = 1.0                                     # part 1: the sign with every syllable (their protocol)
    for done in learn_phase(P['n_b1'], 4000, 7700):
        probe(f'b1 {done}')
    P['p_see'] = see0                                    # part 2: the sign half the time (does binding survive?)
    for done in learn_phase(P['n_b'] - P['n_b1'], 4000 + P['n_b1'], 7701):
        probe(f'b2 {done}')


if __name__ == '__main__':
    print('GROWN', json.dumps(dict(sheets=SH, cells=n_of, groups={gr.name: list(gr.idx.shape) for gr in G},
                                   ear=n_ear, eye=n_eye)), f'({time.time() - t0:.0f} s)', flush=True)
    main()
