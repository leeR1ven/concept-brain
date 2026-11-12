"""Generator test 2 (GENERATOR 四.2): grow an auditory area, a visual area and a concept area between them from the
field rules; real dialogues through the cochlea; the sign of a syllable shown to the eye on part of its hearings.
Concept neurons face the ear: sound arrives from behind (basal, learned), the sign from the front (apical, the
teacher). Learned basal links change by the one graded rule (GENERATOR 二.5): basal push predicts firing.
Probes by sound alone (eye blank): is a syllable's concept code the same each hearing, and does it get more so?
usage: python -m core.gen.t2_concept [key=value ...]"""
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
from .grow import DEV, unit

P = dict(data=os.environ.get('CONCEPT_DATA', 'concept-data'), seed=0, dens=1.0, Lx=10.0, Wy=48.0, Hz=8.0, Lc=6.0,
         lam_e=1.5, lam_proj=4.0, lam_i=1.0, k_side=3.0, K_e=12, K_i=16, K_c=24, K_l=150, p_inh=0.2, p_apical=0.3,
         rate_in=0.2, rate_out=0.06, rate_c=0.10, g_inh=0.5, birth=3000, adapt=0.05, B=16, hold=3, p_see=0.5,
         proj_depth=0.6, birth_see=0.1, lam_loc=1.0, p_long=0.3, k_dend=0.5, local_iso=1, p_fb=0.33,
         spiking=1, dt=5.0, tau_m=20.0, t_ref=5.0, steps=10, p_on=0.3, p_off=0.01,
         organ='real', tau_ad=3.0, k_ad=0.6, th_lo=0.05, th_hi=1.2, slope=0.08, p_sp=0.005, p_max=0.4, g_eye=0.8,
         eta_inh=0.0, norm=1, sigma_n=0.1, c_norm=0.5, U_dep=0.3, tau_rec=300.0, b_ad=0.3, tau_ad_n=100.0,
         eta_us=0.002, k_loc=0.0, learn_all=1, eta_stdp=0.002, a_minus=1.05, w_max=1.0, lam_loc_c=1.0,
         g_ap=3.0, tau_rate=1000.0, tau_rate_birth=20.0, x_tar=0.05, row_sum=4.0,
         soma_inh=1, ap_gate=0.3, k_dend_l=3.0, rate_band=3.0, tick_rule=1, ap_centre=1, eta_b=0.15, vb_aud=1,
         cand_aud=1, K_l_draw=100, floor_life=1, ffi=0, g_inh_c=1.0, K_first=0, first_depth=1.0, dens_s=1.0,
         pre_stdp=0, freeze_sens=1, fb_mod=0, p_apical_s=0.3, ap_gate_s=0.5, tau_relax=5000.0, fb_cut=0, fb_src=0,
         g_ap_s=1.0, mu_use=1, g_drv=1.0, g_mod=1.0, na_floor=0, g_cc=1.0, apic_h=0, dep_s=1.0, ad_s=1.0, id_nd=64, phi_cal=0, dbg=1, g_vc=1.0, life_fix=0, scale_eps=0.002, rate_ix=1.0, recal=0, teach_plateau=0, g_vcs=0.0,
         pl_frac=0.02, pl_mode=1, pl_only=0, pl_tau=3.0, pl_m=2.0, eta_push=0, cand_ear=0, K_vt=0,
         n_learn=2000, probe_every=500, n_probe=150, step=0.15, g_l=1.0, side=16)
for a in (sys.argv[1:] if __name__ == '__main__' else os.environ.get('GEN_ARGS', '').split()):
    k, v = a.split('=', 1)
    try:
        P[k] = json.loads(v)
    except json.JSONDecodeError:
        P[k] = v
torch.manual_seed(P['seed'])
g = torch.Generator(device='cpu').manual_seed(P['seed'])
c = config(levels_per_s=4, syl_ticks=1, gap_turn=[2, 4], gap_dlg=[6, 12])
bases = phon.load_inventory()
coch = Cochlea(Voice(bases), c, P['seed'])
train = Dialogues.load(os.path.join(P['data'], 'train.npz'))
NE, ZE = coch.NC, coch.Z
n_ear, side = coch.n, P['side']
n_eye = side * side * 2
Lx, Wy, Hz, Lc = P['Lx'], P['Wy'], P['Hz'], P['Lc']
xc0, xc1 = Lx + 1.0, Lx + 1.0 + Lc                    # concept slab
xv0 = xc1 + 1.0                                         # visual slab (faces -x)
# organ surfaces: the ear at x = -0.5, the eye at x = xv0 + Lx + 0.5
ear = torch.stack([torch.full((n_ear,), -0.5), (torch.arange(n_ear) // ZE).float() * Wy / NE,
                   (torch.arange(n_ear) % ZE).float() * Hz / ZE], 1)
eye = torch.stack([torch.full((n_eye,), xv0 + Lx + 0.5), (torch.arange(n_eye) // (2 * side)).float() * Wy / side,
                   (torch.arange(n_eye) % (2 * side)).float() * Hz / (2 * side)], 1)


def slab(x0, x1, dens=None):
    m = int((x1 - x0) * Wy * Hz * (P['dens'] if dens is None else dens))
    return torch.rand(m, 3, generator=g) * torch.tensor([x1 - x0, Wy, Hz]) + torch.tensor([x0, 0.0, 0.0])


# the sensory sheets are denser than their organ (real A1 / V1 have far more neurons than nerve fibres)
pa, pc, pv = slab(0.0, Lx, P['dens'] * P['dens_s']), slab(xc0, xc1), slab(xv0, xv0 + Lx, P['dens'] * P['dens_s'])
pos = torch.cat([ear, eye, pa, pc, pv]).to(DEV)
n_in = n_ear + n_eye
N = len(pos)
area = torch.zeros(N, dtype=torch.long, device=DEV)    # 0 organ, 1 auditory, 2 concept, 3 visual
a0, a1 = n_in, n_in + len(pa)
c0, c1 = a1, a1 + len(pc)
v0, v1 = c1, N
area[a0:a1], area[c0:c1], area[v0:v1] = 1, 2, 3
inh = (torch.rand(N, generator=g) < P['p_inh']).to(DEV)
inh[:n_in] = False
# ---- parameter fields (the only place where the areas differ)
x = pos[:, 0]
depth = torch.where(area == 3, (xv0 + Lx - x) / Lx, x / Lx).clamp(0, 1)       # 0 at the organ, 1 deepest
fwd = torch.where((area == 3) | ((area == 0) & (torch.arange(N, device=DEV) >= n_ear)), -1.0, 1.0)
direction = torch.zeros(N, 3, device=DEV)
direction[:, 0] = fwd
direction = torch.where(inh[:, None], torch.randn(N, 3, device=DEV), direction)
u = unit(direction + 0.15 * torch.randn(N, 3, device=DEV))
lam = torch.where(inh, torch.full((N,), P['lam_i'], device=DEV),
                  P['lam_e'] + (P['lam_proj'] - P['lam_e']) * ((depth >= P['proj_depth']) | (area == 2)).float())
lam[:n_in] = P['lam_e']
ks = torch.where(inh, 0.0, P['k_side'])
kr = ks.clone()
K = torch.where(inh, float(P['K_i']), torch.where(area == 2, float(P['K_c']), float(P['K_e']))).long()
K[:n_in] = 0
if P['K_first']:                                        # the first stage (layer 4-like, next to an organ) pools many
    near = (area == 1) & ~inh & (pos[:, 0] < P['first_depth'])   # organ fibres (Fable 11-11 Q4; real L4 cells take
    near |= (area == 3) & ~inh & (pos[:, 0] > xv0 + Lx - P['first_depth'])   # ~100 thalamic inputs)
    K = torch.where(near, torch.full_like(K, int(P['K_first'])), K)
rate = torch.where(area == 2, P['rate_c'], P['rate_in'] * (P['rate_out'] / P['rate_in']) ** depth)
rate = torch.where(inh, rate * P['rate_ix'], rate)     # fast-spiking interneurons fire several times more than pyramids
ap = torch.where((area == 2) & ~inh, P['p_apical'], 0.0)          # concept neurons also have an apical lobe (front)
if P['fb_mod']:                                        # sensory pyramidal cells have one too: feedback from the front
    ap = torch.where(((area == 1) | (area == 3)) & ~inh, P['p_apical_s'], ap)   # lands there (Larkum 2013)
# real neurons (11-11 check-up): the local axon and the basal dendrites spread around the cell, so near neurons connect
# both ways (Song 2005); only the long axon of a projection neuron has a direction (Felleman & Van Essen 1991)
proj = (torch.rand(N, generator=g) < P['p_long']).to(DEV) & ~inh & (depth >= P['proj_depth'])
proj[:n_in] = False                                    # the organ's bits reach only the first stage (as the thalamus does)
# a share p_fb of the projection neurons send their long axon back (feedback: Felleman & Van Essen 1991; in the concept
# area, which has no further area here, half go back to each sense)
fb = (torch.rand(N, generator=g) < P['p_fb']).to(DEV) & proj
fb = torch.where(area == 2, (torch.rand(N, generator=g) < 1.0).to(DEV) & proj, fb)
ax_sign = torch.where(fb, -1.0, 1.0)
ax_sign = torch.where((area == 2) & fb, torch.where(torch.rand(N, generator=g).to(DEV) < 0.5, -1.0, 1.0), ax_sign)


def draw(rows, Kn, lobe_back=True, lobe_front=None, cut=4.0, chunk=256, kd_l=None):
    """for neurons `rows`, draw Kn inputs each by the wiring formula; lobes: basal (behind, the receive side) and,
    with weight lobe_front[j], apical (in front). Returns idx [len(rows), Kmax], and per input whether it is apical."""
    Kmax = int(Kn.max())
    idx = torch.full((len(rows), Kmax), -1, dtype=torch.long, device=DEV)
    apic = torch.zeros((len(rows), Kmax), dtype=torch.bool, device=DEV)
    for s in range(0, len(rows), chunk):
        rr = rows[s:s + chunk]
        d = (pos[None, rr, :] - pos[:, None, :]).transpose(0, 1)          # [c, N, 3]
        r = d.norm(dim=-1).clamp_min(1e-9)
        cs = (d * u[None]).sum(-1) / r
        cr = (d * u[rr][:, None, :]).sum(-1) / r
        if P['local_iso']:                              # local spread (no side) + the projection neurons' long axon
            cs_ax = cs * ax_sign[None, :]               # a feedback neuron's long axon goes back
            far = torch.where(proj[None, :], -r / P['lam_proj'] + ks[None, :] * (cs_ax - 1), torch.full_like(r, -float('inf')))
            loc_reach = torch.where(area == 2, P['lam_loc_c'], P['lam_loc'])   # the concept area's local axons
            loc = -r / torch.where(inh, lam, loc_reach)[None, :]
            if P['k_loc']:                              # the local axon leans forward a little (excitatory neurons)
                loc = loc + torch.where(inh, 0.0, P['k_loc'])[None, :] * (cs - 1)
            send = torch.logaddexp(loc, far)
            kd = torch.where(inh[rr], 0.0, P['k_dend'] if kd_l is None else kd_l)[:, None]
        else:
            send = -r / lam[None, :] + ks[None, :] * (cs - 1)
            kd = kr[rr][:, None]
        back = kd * (cr - 1)
        front = kd * (-cr - 1)
        wf = (lobe_front[rr][:, None] if lobe_front is not None else torch.zeros(len(rr), 1, device=DEV))
        lw = send + torch.logaddexp(back, torch.log(wf.clamp_min(1e-12)) + front)
        lw[r > cut * lam[None, :]] = -float('inf')
        lw[torch.arange(len(rr)), rr] = -float('inf')
        lw[:, n_in:][:, :0] = 0
        gum = -torch.log(-torch.log(torch.rand(lw.shape, generator=g).to(DEV).clamp_min(1e-12)))
        top = (lw + gum).topk(Kmax, dim=1)
        ok = torch.isfinite(lw.gather(1, top.indices)) & (torch.arange(Kmax, device=DEV)[None] < Kn[rr][:, None])
        idx[s:s + len(rr)] = torch.where(ok, top.indices, torch.full_like(top.indices, -1))
        frontal = (cr.gather(1, top.indices) < 0) & proj[top.indices] & (r.gather(1, top.indices) > 2 * P['lam_loc'])
        apic[s:s + len(rr)] = ok & frontal & (wf > 0)
    return idx, apic


t0 = time.time()
rows = torch.arange(n_in, N, device=DEV)
idx_b, apic_b = draw(rows, K, lobe_front=ap)
idx = torch.full((N, idx_b.shape[1]), -1, dtype=torch.long, device=DEV)
apical = torch.zeros_like(idx, dtype=torch.bool)
idx[n_in:], apical[n_in:] = idx_b, apic_b
w = torch.where(idx >= 0, 0.5 + torch.rand(idx.shape, generator=g).to(DEV), torch.zeros(idx.shape, device=DEV))
w = w * (float(P.get('K_ref') or P['K_e']) / K.clamp_min(1).float())[:, None].clamp(max=1.0)   # many inputs: each one
# weaker (K_ref: the same total spread over more inputs, so the membrane sums many small inputs as in real cortex)
# drivers and modulators (Sherman & Guillery 1998; Fable 11-12): the terminals of a long FORWARD axon are few but strong,
# those of a long FEEDBACK axon weak, so a higher area hears its senses and does not overwrite them
_src = idx.clamp_min(0)
_long = (idx >= 0) & proj[_src] & ((pos[_src] - pos[:, None]).norm(dim=-1) > 2 * P['lam_loc'])
# up or down is read from one field, h = the distance from the nearest organ (no area labels)
_h = torch.minimum(pos[:, 0] + 0.5, (xv0 + Lx + 0.5) - pos[:, 0])
_up = _h[:, None] > _h[_src] + 0.5                      # the target is farther from the organs than the source
_dn = _h[:, None] < _h[_src] - 0.5
_cross = area[:, None] != area[_src]                    # the link leaves its sheet (the sheets are separate slabs;
_up, _dn = _up & _cross, _dn & _cross                   # Fable 11-12: h peaks inside the concept slab, so links within it
w = torch.where(_long & _up, w * P['g_drv'], w)         # must not count as drivers or modulators)
w = torch.where(_long & _dn, w * P['g_mod'], w)
if P['apic_h']:                                         # apical = a long DOWN-going terminal (a return from farther in),
    apical = _long & _dn & (~inh)[:, None]              # read from h, not from the side it arrives on (Fable 11-12): a
    # second sense arrives up-going, as a basal driver like sound, and teaches by firing the cell (the other sense)
if P['g_cc'] != 1.0:                                    # diagnostic: the concept area's own excitatory links
    w = torch.where((area[:, None] == 2) & (area[_src] == 2) & ~inh[_src] & (idx >= 0), w * P['g_cc'], w)
_fb = ((area == 1) | (area == 3))[:, None] & (area[idx.clamp_min(0)] == 2) & (idx >= 0)   # concept -> sensory links
if P['fb_cut']:                                         # diagnostic (11-12): no feedback from the concept area at all
    w = torch.where(_fb, torch.zeros_like(w), w)
if P['fb_src']:                                         # feedback from the concept area lands on the sensory cells'
    apical = apical | (_fb & ~inh[idx.clamp_min(0)])    # apical end and only amplifies (Larkum 2013), forward unchanged
if P['g_inh_c'] != 1.0:                                 # stronger competition among concept cells (Fable 11-11)
    w = torch.where((area[:, None] == 2) & inh[idx.clamp_min(0)] & (idx >= 0), w * P['g_inh_c'], w)
sign = torch.where(inh, -P['g_inh'], 1.0).float()
# learned basal candidates of the concept neurons (excitatory ones): potential synapses by the same formula
crow = torch.nonzero((area == 2) & ~inh).squeeze(1)
# every excitatory cortical neuron learns by the one rule (designer 11-11; GENERATOR 二.5); learn_all=0: concept only
lrow = torch.nonzero((area > 0) & ~inh).squeeze(1) if P['learn_all'] else crow
lidx, lap = draw(lrow, torch.full((N,), P['K_l_draw'] if P['cand_aud'] else P['K_l'], device=DEV).long(),
                 lobe_front=None, kd_l=P['k_dend_l'])
if P['cand_aud']:                                       # concept units learn from the auditory side only (Fable 11-11:
    bad = (area[lrow] == 2)[:, None] & ((area[lidx.clamp_min(0)] != 1) | inh[lidx.clamp_min(0)])   # (excitatory only)
    lidx = torch.where(bad, torch.full_like(lidx, -1), lidx)           # "inside links" that merged, F3)
order = torch.argsort((lidx < 0).int(), dim=1, stable=True)          # valid candidates first, drawn order kept
lidx = lidx.gather(1, order)[:, :P['K_l']]
if P['cand_ear']:                                       # diagnostic (Fable 11-12): concept cells learn from the ear's own
    _ce = (area[lrow] == 2)                             # fibres (a skip-level projection), bypassing the auditory sheet:
    _ge = torch.Generator(device='cpu').manual_seed(P['seed'] + 5)   # is the student input the cap?
    _r = torch.argsort(torch.rand(int(_ce.sum()), n_ear, generator=_ge), dim=1)[:, :lidx.shape[1]].to(DEV)
    lidx[_ce] = _r
from_aud = (area[idx.clamp_min(0)] == 1) & (idx >= 0)
from_vis = (area[idx.clamp_min(0)] == 3) & (idx >= 0)
E = torch.zeros(lidx.shape, device=DEV)
I = torch.zeros(lidx.shape, device=DEV)
mu = torch.zeros(len(lrow), device=DEV)
NAU = torch.zeros(len(lrow), device=DEV)
RT = None                                                # each learning cell's share of ticks with a spike (set in life)
VU = torch.zeros(len(lrow), device=DEV)                  # each concept cell's usual sign drive while something is seen
VU2 = torch.zeros(len(lrow), device=DEV)                 # ... of the integrated (plateau) drive, and its spread
VV = torch.zeros(len(lrow), device=DEV)
VLP = None
EYEP = NSH = None
RECAL = False
CL = torch.nonzero(area[lrow] == 2).squeeze(1)          # the concept rows among the learning rows
_vE = torch.nonzero((area == 3) & ~inh).squeeze(1)      # the visual pyramids
_gv = torch.Generator(device='cpu').manual_seed(P['seed'] + 9)
TIDX = _vE[torch.randint(0, len(_vE), (len(CL), max(1, int(P['K_vt']))), generator=_gv).to(DEV)]
TW = (0.5 + torch.rand(TIDX.shape, generator=_gv)).to(DEV)
BORN_SIG = [int(idx.sum().item()), int(lidx.sum().item()), int((idx >= 0).sum().item()), int((lidx >= 0).sum().item()),
            float(w.sum().item())]                      # the wiring and born weights as grown (checkpoint guard)
PLREC = []
CUR_LAB = None
CE = torch.nonzero((area == 2) & ~inh).squeeze(1)        # the concept pyramids
PHI_NB, PHI_MAX = 60, 3.0                                # the measured transfer: push bins up to 3 thresholds
PHN = torch.zeros(PHI_NB, device=DEV)
PHD = torch.zeros(PHI_NB, device=DEV)
DBG = []                 # each learning cell's usual count of input spikes in a sound tick
theta = torch.zeros(N, device=DEV)
src_area = area[idx.clamp_min(0)]
print('GROWN', json.dumps(dict(neurons=N, inputs=n_in, auditory=len(pa), concept=len(pc), visual=len(pv),
                               s=round(time.time() - t0, 1),
                               concept_inputs_from={nm: round(((src_area[c0:c1] == k) & (idx[c0:c1] >= 0)).float().sum(1).mean().item(), 1)
                                                    for nm, k in (('ear/eye', 0), ('auditory', 1), ('concept', 2), ('visual', 3))},
                               concept_apical_share=round((apical[c0:c1] & (idx[c0:c1] >= 0)).float().sum().item() /
                                                          max(1.0, (idx[c0:c1] >= 0).float().sum().item()), 3),
                               learning_neurons=len(lrow))), flush=True)


def drives(act):
    """act bool or float [B, N] (float: spikes x what each synapse still has to give) -> basal, apical drive [B, N]"""
    a = torch.cat([act.float() * sign, torch.zeros(act.shape[0], 1, device=DEV)], 1)
    ix = torch.where(idx >= 0, idx, N)
    contrib = a[:, ix] * w[None]
    ap_ = (contrib * apical[None].float()).sum(-1)
    return contrib.sum(-1) - ap_, ap_


def drives3(act):
    """-> excitatory basal, inhibition (at the soma: basket cells, Fable 11-11), excitatory apical drive [B, N]"""
    a = torch.cat([act.float() * sign, torch.zeros(act.shape[0], 1, device=DEV)], 1)
    ix = torch.where(idx >= 0, idx, N)
    contrib = a[:, ix] * w[None]
    neg = (contrib < 0).float()
    inhib = -(contrib * neg).sum(-1)
    pos = contrib * (1 - neg)
    apx = (pos * apical[None].float()).sum(-1)
    aud = (pos * (from_aud & ~apical)[None].float()).sum(-1)
    vis = (pos * from_vis[None].float()).sum(-1)
    return pos.sum(-1) - apx, inhib, apx, aud, vis


def learned(act):
    a = act.float()
    ix = lidx.clamp_min(0)
    on = a[:, ix] * (lidx >= 0).float()[None]                           # [B, nC, K_l]
    W = E - I
    if P['ffi']:                                         # feed-forward inhibition (Fable 11-11): interneurons with the
        nv = (lidx >= 0).float().sum(1, keepdim=True).clamp_min(1.0)    # same afferents cancel the common part, so
        cm = (area[lrow] == 2)[:, None].float()                        # only the push above the mean weight counts
        W = W - cm * (W * (lidx >= 0).float()).sum(1, keepdim=True) / nv   # (concept rows only; Fable 11-12)
    return (on * W[None]).sum(-1), on


V = REF = None                                          # spiking: membrane potentials and refractory steps left
SUB = None                                              # set to a list to record every sub-step's spikes (check-up)


XR = AD = None                                          # each neuron's synaptic resources (depression), adaptation


VB = TR = VBL = None                                    # basal potential (reset / not reset), every neuron's spike trace
APU = torch.zeros(N, device=DEV)                        # each neuron's usual apical drive
RATE = torch.zeros(N, device=DEV)                       # each neuron's slow running rate (per sub-step)
THB = None                                              # thresholds at the end of birth (where relaxation returns to)
GAP = torch.where(area == 2, P['g_ap'], P['g_ap_s']).float()  # apical gain: concept teacher / sensory feedback
_sens = (area == 1) | (area == 3)                        # depression and adaptation may be weaker in the sensory sheets
UDV = torch.where(_sens, P['U_dep'] * P['dep_s'], P['U_dep']).float()   # (a field value; Fable 11-12: they carry the
BADV = torch.where(_sens, P['b_ad'] * P['ad_s'], P['b_ad']).float()     # previous syllable into the next one)
FROZEN = False                                          # the senses frozen (thresholds, inhibitory plasticity, STDP)
TAU_R = 20.0                                            # its time constant in ticks (birth: fast, life: slow)


def reset(B):
    global V, REF, ORG, XR, AD, VB, TR, VBL
    ORG = None
    VB = torch.zeros(B, len(lrow), device=DEV)
    VBL = torch.zeros(B, len(lrow), device=DEV)
    TR = torch.zeros(B, N, device=DEV)
    XR = torch.ones(B, N, device=DEV)
    AD = torch.zeros(B, N, device=DEV)
    V = torch.zeros(B, N, device=DEV)
    REF = torch.zeros(B, N, dtype=torch.int16, device=DEV)
    step.last = None


def step(xin, act, learn=False, adapt=0.0):
    """one world tick. spiking=1 (designer 11-11: a spiking network): P['steps'] sub-steps of dt ms each; a neuron
    integrates its input with a leak (time constant tau_m), fires when its potential passes its own threshold, resets
    and rests t_ref ms (its physiology, not a device to move thoughts: ARCH 七). The organ's bits fire at p_on per
    sub-step when on, p_off when off. Returns, per neuron, whether it fired in this tick (as the binary model did)."""
    if not P['spiking']:
        return step_binary(xin, act, learn, adapt)
    global V, REF
    if V is None or V.shape[0] != xin.shape[0]:
        reset(xin.shape[0])
    alpha = float(np.exp(-P['dt'] / P['tau_m']))
    n_ref = int(round(P['t_ref'] / P['dt']))
    fired = torch.zeros(xin.shape[0], N, dtype=torch.bool, device=DEV)
    spk = getattr(step, 'last', None)
    if spk is None or spk.shape != fired.shape:
        spk = torch.zeros_like(fired)
    p_in = xin if xin.dtype.is_floating_point else torch.where(xin, P['p_on'], P['p_off'])
    seen = (xin[:, n_ear:].float().mean(1) > 2 * P['p_sp']) if xin.dtype.is_floating_point else \
        torch.ones(xin.shape[0], dtype=torch.bool, device=DEV)
    vb_sum = torch.zeros(xin.shape[0], len(lrow), device=DEV)
    c_post = torch.zeros(xin.shape[0], len(lrow), device=DEV)
    c_pre = torch.zeros(xin.shape[0], N, device=DEV)
    q_sum = torch.zeros(xin.shape[0], len(lrow), device=DEV)
    vis_sum = torch.zeros(xin.shape[0], len(lrow), device=DEV)
    dec_r = float(np.exp(-P['dt'] / P['tau_rec']))
    dec_a = float(np.exp(-P['dt'] / P['tau_ad_n']))
    for _ in range(P['steps']):
        # short-term depression (Tsodyks & Markram 1997): an excitatory neuron's spike gives what its synapses still
        # hold; each spike uses a share U_dep, and it comes back over tau_rec. Inhibitory synapses do not depress here.
        eff = spk.float() * torch.where(inh[None], torch.ones_like(XR), XR)
        if P['soma_inh']:
            bas, inhib, apx, aud, vis = drives3(eff)
            if P['teach_plateau'] and P['K_vt']:         # the plateau dendrite has its own many inputs from the other
                vis_sum[:, CL] += (spk.float()[:, TIDX] * TW[None]).sum(-1)   # sense (a real tuft sums hundreds; 8 inputs'
            elif P['teach_plateau']:                     # spike counts were noise: same-sign set overlap 0.04, 11-12)
                vis_sum += vis[:, lrow]                  # (BTSP; Bittner 2017): it marks which cells learn and drives
            if P['teach_plateau']:
                bas[:, CE] -= (1.0 - P['g_vcs']) * vis[:, CE]   # their soma only by g_vcs (interneurons keep it)
        else:
            bas, apx = drives(eff)
            inhib, aud = torch.zeros_like(bas), bas
        q, on = learned(spk)
        bas[:, lrow] = bas[:, lrow] + P['g_l'] * (q - mu[None])
        if P['ap_centre']:                               # the teacher counts above each unit's own usual level (F5)
            if (learn or adapt) and bool(seen.any()):    # its usual level while something is seen (Fable 11-11)
                APU.add_((apx[seen].mean(0) - APU) / (1000.0 * P['steps']))
            apx = (apx - APU[None]).clamp_min(0)
        if P['fb_mod'] or P['fb_src']:                   # in the sensory sheets feedback only amplifies a cell its own
            sens_r = (area[lrow] != 2)                   # input already drives (Larkum 1999 BAC firing)
            gate_s = torch.ones_like(apx)
            gate_s[:, lrow[sens_r]] = (VBL[:, sens_r] > P['ap_gate_s'] * theta[lrow[sens_r]][None]).float()
            apx = apx * gate_s
        if P['ap_gate']:                                 # apical input amplifies a cell its basal input already drives
            gate = torch.ones_like(apx)                  # (Larkum 1999 BAC firing; Fable 11-11)
            gate[:, lrow] = (VB > P['ap_gate'] * theta[lrow][None]).float()
            apx = apx * gate
        if P['fb_src']:                                  # the sensory apical gain is its own (the concept one is a teacher)
            tot = bas + GAP[None] * apx - inhib
        else:
            tot = bas + P['g_ap'] * apx - inhib           # inhibition acts at the soma, outside the basal prediction
        V.mul_(alpha).add_(tot - AD)
        can = REF <= 0
        new = (V > theta[None]) & can
        new[:, :n_in] = torch.rand(p_in.shape, device=DEV) < p_in
        V[new] = 0.0
        XR.sub_(UDV[None] * XR * new.float()).add_((1.0 - XR) * (1.0 - dec_r))
        # spike-frequency adaptation (Benda & Herz 2003): each spike makes the neuron harder to fire for a while
        AD.mul_(dec_a).add_(BADV[None] * theta[None].clamp_min(0) * (1.0 - alpha) * new.float())
        REF.sub_(1).clamp_(min=0)
        REF[new] = n_ref
        LIFE = (TAU_R != P['tau_rate_birth']) and FROZEN   # the concept area's learning (senses frozen)
        if adapt and P['eta_inh'] and not (P['life_fix'] and LIFE):
            # inhibitory plasticity (Vogels 2011): a link from an inhibitory neuron that fired grows when its target
            # fires above its own rate and shrinks when below, so each neuron gets just enough inhibition
            tgt_all = (rate / P['steps'])
            err = (new.float().mean(0) - tgt_all) / tgt_all.clamp_min(1e-6)            # [N] relative rate error
            pre_inh = torch.cat([(spk.float().mean(0) * inh.float()), torch.zeros(1, device=DEV)])
            pi = pre_inh[torch.where(idx >= 0, idx, N)]                                 # [N, K]
            if FROZEN:                                    # frozen senses: only concept cells keep tuning their inhibition
                err = err * (area == 2).float()
            w.add_(P['eta_inh'] * pi * err.clamp(-1, 3)[:, None]).clamp_(min=0.0, max=5.0)
        if adapt and not (P['life_fix'] and LIFE):      # each neuron keeps its own rate (per tick: rate / steps):
            # its threshold moves in proportion to itself (intrinsic plasticity, Desai 1999), so a silent neuron with
            # weak input still comes down to its rate
            tgt = rate[n_in:] / P['steps']
            RATE[n_in:] += (new[:, n_in:].float().mean(0) - RATE[n_in:]) / (TAU_R * P['steps'])
            # only outside a wide band (rate_band x either way): rates may differ a lot between neurons (11-11)
            hi, lo = RATE[n_in:] > tgt * P['rate_band'], RATE[n_in:] < tgt / P['rate_band']
            if not P['floor_life'] and TAU_R != P['tau_rate_birth']:
                lo = torch.zeros_like(lo)                 # in life a quiet cell stays quiet (Fable 11-11): no forced spikes
            err = torch.where(hi, RATE[n_in:] / (tgt * P['rate_band']) - 1, torch.where(lo, RATE[n_in:] * P['rate_band'] / tgt - 1, torch.zeros_like(tgt)))
            upd = 1.0 + (adapt * err).clamp(-0.2, 0.2)
            if FROZEN:                                    # the senses are frozen: their thresholds stay (Fable 11-12)
                upd = torch.where(area[n_in:] == 2, upd, torch.ones_like(upd))
            theta[n_in:] *= upd
            theta[n_in:].clamp_(min=1e-3)                 # a negative threshold would only run away (11-12 bug)
            if THB is not None and not P['floor_life'] and TAU_R != P['tau_rate_birth']:
                # without a floor, a threshold that rose relaxes slowly back toward its birth value (Fable 11-12)
                rel = (THB[n_in:] - theta[n_in:]) / (P['tau_relax'] * P['steps'])
                theta[n_in:] += torch.where(hi, torch.zeros_like(rel), rel) * (1.0 if not FROZEN else (area[n_in:] == 2).float())
        # the basal potential alone (what the neuron's own inputs from behind would make of it) and the spike traces
        # the basal (student) potential: input from the auditory side only plus the learned push (Fable 11-11: born
        # concept-local excitation goes to the soma, else the prediction already contains the teacher); a low-pass
        # that is not reset by spikes (else a predicted spike reads as half predicted)
        vin = torch.where((area[lrow] == 2)[None], aud[:, lrow] + P['g_l'] * (q - mu[None]), bas[:, lrow]) \
            if P['vb_aud'] else bas[:, lrow]
        VB.mul_(alpha).add_(vin)
        VBL.mul_(alpha).add_(vin)
        if learn and not (P['tick_rule'] and P['eta_stdp'] == 0.0):    # (nothing to do here once the senses froze)
            # dendritic prediction of somatic spiking (Urbanczik & Senn 2014): the basal potential predicts a spike;
            # a spike it did not predict (the apical teacher made it) strengthens the links from inputs just active,
            # a predicted spike that did not come weakens them. Excitatory part E and inhibitory partner part I >= 0.
            if not P['tick_rule']:
                mu.add_((q.mean(0) - mu) / (1000.0 * P['steps']))
            th = theta[lrow].clamp_min(1e-6)
            pred = ((VB > th[None]) & can[:, lrow]).float()   # the soma's own transfer of the basal potential alone
            dd = new[:, lrow].float() - pred                                   # [B, nL]
            valid_l = (lidx >= 0).float()[None]
            pre_tr = TR[:, lidx.clamp_min(0)] * valid_l                       # [B, nL, K_l]
            d = (0.5 * P['eta_us'] * dd[:, :, None] * pre_tr).mean(0)
            # sensory areas (no teacher): spike-timing Hebbian (ARCH 四 时序赫布; Bi & Poo 1998; Diehl & Cook 2015):
            # pre just before post strengthens, post just before pre weakens; soft bounds toward 0 and w_max
            pre_sp = new[:, lidx.clamp_min(0)].float() * valid_l
            post_sp = new[:, lrow].float()
            post_tr = TR[:, lrow]
            st = ((pre_tr - P['x_tar']) * post_sp[:, :, None] * (P['w_max'] - E)[None]
                  - P['a_minus'] * post_tr[:, :, None] * pre_sp * E[None]).mean(0) * P['eta_stdp']
            sens = (area[lrow] != 2)[:, None]
            d = d * (0.0 if P['tick_rule'] else 1.0)
            E_new = torch.where(sens, (E + st).clamp(0.0, P['w_max']), (E + d).clamp_min(0.0))
            # synaptic scaling (Turrigiano 2008; Diehl & Cook 2015): a sensory neuron's learned links keep a fixed sum
            rs = E_new.sum(1, keepdim=True)
            E_new = torch.where(sens & (rs > 0), E_new * (P['row_sum'] / rs.clamp_min(1e-9)), E_new)
            E.copy_(E_new)
            I.copy_(torch.where(sens, I, (I - d).clamp_min(0.0)))
        if P['tick_rule']:
            vb_sum += VBL
            c_post += new[:, lrow].float()
            c_pre += new.float()
            q_sum += q
        VB[new[:, lrow]] = 0.0
        TR.mul_(alpha).add_(new.float())
        if SUB is not None:
            SUB.append(new.clone())
        fired |= new
        spk = new
    if learn and P['tick_rule']:
        # rule b over the tick (50 ms; F4, Fable 11-11): came = the unit fired in the tick; pushed = its mean basal
        # potential over the tick in units of its threshold, clipped to 0..1; a unit that did not fire and was pushed
        # less than its threshold takes no part; the change is shared by the candidate inputs that fired in the tick.
        # Its usual learned push is removed only while there is sound (Fable 11-11: silence diluted it).
        crow_l = (area[lrow] == 2)
        th = theta[lrow].clamp_min(1e-6)
        push = (vb_sum / P['steps']) / th[None]
        came = (c_post > 0).float()
        if P['teach_plateau'] and bool(seen.any()):
            # the plateau set: per stream, the concept cells whose sign drive is most above their own usual level while
            # something is seen (a fixed share, pl_frac); for them the tick counts as "came" whatever the soma did
            z = vis_sum / P['steps']
            VU.add_(crow_l.float() * (z[seen].mean(0) - VU) / 1000.0)
            dev = torch.where(crow_l[None], z - VU[None], torch.full_like(z, -1e9))
            k_pl = max(1, int(P['pl_frac'] * int(crow_l.sum())))
            top = dev.topk(k_pl, dim=1).indices
            pl = torch.zeros_like(came).scatter_(1, top, 1.0) * seen[:, None].float()
            if P['pl_mode'] == 2:
                # (Fable 11-12) a real plateau integrates the sign drive over a showing (~150 ms) on the dendrite, and
                # each cell has its own threshold: its usual seen level plus pl_m of its own spread (local, no ties)
                global VLP
                if VLP is None or VLP.shape != z.shape:
                    VLP = torch.zeros_like(z)
                # integrate within one showing: a new sign on the eye (its pattern changes) starts it again (Fable 11-12:
                # a decaying trace carried the previous sign into the next syllable)
                global EYEP, NSH
                eye_now = xin[:, n_ear:n_in]
                if EYEP is None or EYEP.shape != eye_now.shape:
                    EYEP, NSH = eye_now.clone(), torch.zeros(eye_now.shape[0], device=DEV)
                onset = (eye_now != EYEP).any(1) | ~seen
                NSH = torch.where(onset, torch.zeros_like(NSH), NSH)
                VLP = torch.where(onset[:, None], torch.zeros_like(VLP), VLP)
                VLP = torch.where(seen[:, None], (VLP * NSH[:, None] + z) / (NSH[:, None] + 1.0), torch.zeros_like(VLP))
                NSH = torch.where(seen, NSH + 1.0, torch.zeros_like(NSH))
                EYEP = eye_now.clone()
                vs = VLP[seen]
                VV.add_(crow_l.float() * (((vs - VU2[None]) ** 2).mean(0) - VV) / 1000.0)
                VU2.add_(crow_l.float() * (vs.mean(0) - VU2) / 1000.0)
                pl = ((VLP > VU2[None] + P['pl_m'] * VV.clamp_min(1e-12).sqrt()[None]) & crow_l[None]
                      & seen[:, None]).float()
            if P['pl_only']:                             # on a seen tick the teacher alone says who came (the sign's own
                came = torch.where(seen[:, None] & crow_l[None], pl, came)   # inhibition silences the rest)
            else:
                came = torch.maximum(came, pl)
            if P['dbg'] and len(PLREC) < 400 and CUR_LAB is not None and not RECAL:
                for b in torch.nonzero(seen).squeeze(1).tolist():
                    if CUR_LAB[b] >= 0:
                        PLREC.append((pl[b, crow_l].bool().cpu(), CUR_LAB[b]))
        sound = (xin[:, :n_ear].float().mean(1) > 2 * P['p_sp']) if xin.dtype.is_floating_point else torch.ones(xin.shape[0], dtype=torch.bool, device=DEV)
        if P['phi_cal']:
            # the soma's real transfer (11-12): a spiking cell fires on chance coincidences, so push / threshold is not its
            # chance of firing. Measure it where only the sound acts (sound, no sign): the chance a concept cell fires
            # in a tick, by its push. Then "came - predicted" is zero on average without a teacher, and only the spikes
            # the other sense adds teach (rule b's meaning).
            pb = (push * (PHI_NB / PHI_MAX)).long().clamp(0, PHI_NB - 1)
            m = (sound & ~seen)[:, None] & crow_l[None]
            if bool(m.any()):
                PHN.mul_(0.999).index_add_(0, pb[m], came[m])
                PHD.mul_(0.999).index_add_(0, pb[m], torch.ones_like(came[m]))
            mid = ((torch.arange(PHI_NB, device=DEV) + 0.5) * (PHI_MAX / PHI_NB)).clamp(max=1.0)
            phi = (PHN + 20.0 * mid) / (PHD + 20.0)
            pred = phi[pb]
        else:
            pred = push.clamp(0.0, 1.0)
            pred = torch.where((came > 0) | (push >= 1.0), pred, torch.zeros_like(pred))
        dd = (came - pred) * crow_l[None].float()
        on = c_pre[:, lidx.clamp_min(0)] * (lidx >= 0).float()[None]             # spike counts in the tick
        na = on.sum(-1).clamp_min(1.0)
        if P['na_floor']:                                # a nearly empty input (silence, a syllable's edge) moves almost
            na = torch.maximum(na, NAU[None])            # nothing: share by at least the usual count (Fable 11-12)
        d = (0.5 * P['eta_b'] * (dd / na)[:, :, None] * on).sum(0)   # each stream's hearing counts in full
        if P['eta_push']:                                # the step in units of the push (Fable 11-12): a weight change
            d = d * (theta[lrow] * (1.0 - alpha) * P['steps'])[:, None]   # moves the potential / threshold as offline
        E.add_(d).clamp_(min=0.0)
        I.sub_(d).clamp_(min=0.0)
        if P['life_fix'] and RT is not None:
            # in life thresholds stay; a cell made too active by what it LEARNED scales down its learned links (synaptic
            # scaling, Turrigiano 2008; Fable 11-12): only the cause is penalised, not the teacher's born drive. The rate
            # is counted in the target's unit: ticks with at least one spike.
            RT.add_(((c_post > 0).float().mean(0) - RT) / P['tau_rate'])
            over = crow_l & (RT > (rate[lrow] * P['rate_band']))
            if bool(over.any()):
                f = torch.where(over, 1.0 - P['scale_eps'], 1.0)[:, None]
                E.mul_(f)
                I.mul_(f)
        if P['dbg']:                                     # concept rows: how learning goes (11-12)
            cm = crow_l[None].float().expand_as(dd)
            DBG.append(torch.stack([came[:, crow_l].mean(), pred[:, crow_l].mean(), (dd.clamp(min=0) * cm).sum() / cm.sum(),
                                    (dd.clamp(max=0) * cm).sum() / cm.sum(), push[:, crow_l].mean(),
                                    theta[lrow][crow_l].mean()]))
        if bool(sound.any()):
            NAU.add_((on.sum(-1)[sound].mean(0) - NAU) / 1000.0)
        if bool(sound.any()) and P['mu_use']:          # mu_use=0: with ffi the usual push is cancelled by the input
            mu.add_(crow_l.float() * ((q_sum[sound] / P['steps']).mean(0) - mu) / 1000.0)
    step.last = spk
    return fired


def step_binary(xin, act, learn=False, adapt=0.0):
    bas, apx = drives(act)
    q, on = learned(act)
    qd = q - mu[None]
    bas[:, lrow] = bas[:, lrow] + P['g_l'] * qd
    tot = bas + apx
    new = tot > theta[None]
    new[:, :n_in] = xin
    if adapt:
        r = new[:, n_in:].float().mean(0)
        theta[n_in:] += adapt * (r - rate[n_in:]) * tot[:, n_in:].abs().mean(0).clamp_min(1e-3)
    if learn:
        mu.add_((q.mean(0) - mu) / 1000.0)
        th = theta[lrow].clamp_min(1e-6)
        pred = (bas[:, lrow] / th[None]).clamp(0, 1)
        lit = new[:, lrow].float()
        dd = torch.where((lit > 0) | (bas[:, lrow] >= th[None]), lit - pred, torch.zeros_like(pred))   # [B, nC]
        na = on.sum(-1).clamp_min(1.0)
        d = (0.5 * P['step'] * (dd / na)[:, :, None] * on).mean(0)
        E.copy_((E + d).clamp_min(0.0))
        I.copy_((I - d).clamp_min(0.0))
    return new


def sign_of(sid):
    r = np.random.default_rng([4321, 5150, int(sid) + 1])
    v = r.random(side * side) < 0.5
    return np.stack([v, ~v], 1).reshape(-1)


BLANK = np.stack([np.random.default_rng([4321, 5150, 0]).random(side * side) < 0.5] * 2, 1).reshape(-1)
BLANK[1::2] = ~BLANK[0::2]


def ear(srcs):
    """what reaches the ear this tick: organ 'real' (designer 11-11: like the real organs) = the cochlea's energy per
    frequency channel (the auditory nerve is modelled in organ_in); else the thermometer bits"""
    return coch.energy(srcs).astype(np.float32) if P['organ'] == 'real' else coch.a0(srcs)


def eye(sid):
    """what is seen: the sign of a syllable (a fixed random pattern of bright and dark pixels), None = nothing (an even
    grey field); organ 'bits': the old thermometer pairs"""
    if P['organ'] != 'real':
        return BLANK if sid is None else sign_of(sid)
    if sid is None:
        return np.full(side * side, 0.5, np.float32)
    return (np.random.default_rng([4321, 5150, int(sid) + 1]).random(side * side) < 0.5).astype(np.float32)


ORG = None                                              # the ear's adaptation state (per stream, per channel)


def organ_in(items):
    """items: per stream (ear, eye) -> the organ neurons' firing probability per sub-step [B, n_in].
    Ear (auditory nerve; Liberman 1978, Sachs & Abbas 1974): per frequency channel ZE fibres with thresholds from th_lo
    to th_hi; each fires at p_sp spontaneously and up to p_max with sound above its threshold; every channel adapts to
    its recent level (tau_ad ticks, share k_ad), so a steady loudness fades and changes stand out.
    Eye (retina; Kuffler 1953): per pixel an ON and an OFF ganglion cell with a centre-surround field: ON fires when
    the centre is brighter than its surround, OFF when darker; an even field gives only the spontaneous rate."""
    if P['organ'] != 'real':
        return torch.as_tensor(np.stack([np.concatenate([a, b]) for a, b in items]), device=DEV)
    global ORG
    e = torch.as_tensor(np.stack([a for a, _ in items]), device=DEV)
    if ORG is None or ORG.shape != e.shape:
        ORG = e.clone()
    ORG += (e - ORG) / P['tau_ad']
    eff = e - P['k_ad'] * ORG
    th = torch.linspace(P['th_lo'], P['th_hi'], ZE, device=DEV)
    r = torch.sigmoid((eff[:, :, None] - th) / P['slope'])
    if P['norm']:                                       # divisive normalization (Carandini & Heeger 2012): each fibre
        r = P['c_norm'] * r / (P['sigma_n'] + r.mean((1, 2), keepdim=True))   # over the whole ear's activity:
    pe = P['p_sp'] + (P['p_max'] - P['p_sp']) * r.clamp(0, 1)                 # loud sounds do not flood the cortex
    img = torch.as_tensor(np.stack([b for _, b in items]), device=DEV).view(-1, 1, side, side)
    sur = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(img, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
    on, off = (img - sur).clamp_min(0), (sur - img).clamp_min(0)
    rv = P['g_eye'] * torch.stack([on, off], -1).reshape(img.shape[0], -1)
    if P['norm']:
        rv = P['c_norm'] * rv / (P['sigma_n'] + rv.mean(1, keepdim=True))
    pv = (P['p_sp'] + (P['p_max'] - P['p_sp']) * rv).clamp(0, 1)
    return torch.cat([pe.reshape(e.shape[0], -1), pv], 1)


def stream(ds, tag, see):
    """per tick: ear bits, eye bits, label (sid or -1). see: the share of syllables whose sign is shown"""
    see = P['p_see'] if see is True else float(see)
    rg = np.random.default_rng([P['seed'], tag, int(ds[0]) if len(ds) else 0])
    out = []
    shown = False
    for d in ds:
        src = script_dialogue(train, d, np.random.default_rng([P['seed'], tag, d]), (2, 4), 8, 1, P['hold'])
        for xx in src:
            ebits = ear([] if xx is None else [xx[:5]])
            if xx is None:
                shown = False
            elif not out or out[-1][2] != int(xx[0]):    # decided once per syllable (it is held several ticks)
                shown = rg.random() < see
            out.append((ebits, eye(xx[0]) if shown else eye(None), -1 if xx is None else int(xx[0])))
    return out


def run_streams(n_d, d0, tag, learn, see, record=False, adapt=None):
    """B dialogues in parallel streams; returns the concept states and labels when record"""
    per = n_d // P['B']
    S = [stream(range(d0 + b * per, d0 + (b + 1) * per), tag, see) for b in range(P['B'])]
    T = min(len(s) for s in S)
    act = torch.zeros(P['B'], N, dtype=torch.bool, device=DEV)
    rec, lab = [], []
    for t in range(T):
        xin = organ_in([(S[b][t][0], S[b][t][1]) for b in range(P['B'])])
        global CUR_LAB
        CUR_LAB = [S[b][t][2] for b in range(P['B'])]
        act = step(xin, act, learn=learn, adapt=(0.005 if learn else 0.0) if adapt is None else adapt)
        if record:
            rec.append(act[:, crow].clone())
            lab.append([S[b][t][2] for b in range(P['B'])])
    return (torch.stack(rec, 1) if record else None), (np.array(lab).T if record else None)


def probe(tag):
    """sound alone: per lag, consistency of each syllable's concept code (cosine to its own mean vs another's) and
    identity (nearest class mean, half / half)"""
    X, L = run_streams(P['n_probe'], 15000, 9000, False, False, record=True)      # [B, T, nC], [B, T]
    best = None
    for lag in range(1, 12):
        Xs = X[:, lag:].reshape(-1, X.shape[-1]).float()
        Ls = torch.as_tensor(L[:, :-lag].reshape(-1), device=DEV)
        m = Ls >= 0
        Xs, Ls = Xs[m], Ls[m]
        cnt = torch.bincount(Ls)
        top = torch.argsort(cnt, descending=True)[:100]
        keep = torch.isin(Ls, top)
        Xs, Ls = Xs[keep], Ls[keep]
        half = torch.rand(len(Ls), device=DEV) < 0.5
        cls = {int(s): Xs[half & (Ls == s)].mean(0) for s in top.tolist() if bool((half & (Ls == s)).any())}
        ks_ = list(cls)
        M = torch.stack([cls[k] for k in ks_])
        Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
        Te, Tl = Xs[~half], Ls[~half]
        Tn = Te / Te.norm(dim=1, keepdim=True).clamp_min(1e-9)
        pos_ = {k: i for i, k in enumerate(ks_)}
        ok = torch.tensor([int(l) in pos_ for l in Tl.tolist()], device=DEV)
        Tn, Tl = Tn[ok], Tl[ok]
        ti = torch.tensor([pos_[int(l)] for l in Tl.tolist()], device=DEV)
        sc = Tn @ Mn.T
        own = sc.gather(1, ti[:, None]).mean().item()
        other = sc.mean().item()
        ident = (sc.argmax(1) == ti).float().mean().item()
        if best is None or ident > best['identity']:
            best = dict(lag=lag, own=round(own, 3), other=round(other, 3), identity=round(ident, 3),
                        lit=round(X.float().mean().item(), 3))
    print('PROBE', tag, json.dumps(best), f'({time.time() - t0:.0f} s)', flush=True)


# ---- birth: random sounds and signs, each neuron finds its own threshold
def birth():
    theta[n_in:] = drives(torch.rand(8, N, device=DEV) < 0.1)[0].mean(0)[n_in:] + 1e-3
    theta[n_in:] = theta[n_in:].clamp_min(0.1 * theta[n_in:].abs().mean().item())   # a threshold is positive (11-12)
    if P['spiking']:
        theta[n_in:] *= 1.0 / (1.0 - float(np.exp(-P['dt'] / P['tau_m'])))
        reset(P['B'])
    act = torch.zeros(P['B'], N, dtype=torch.bool, device=DEV)
    births = stream(range(16000, 16000 + 60), 7000, P['birth_see'])
    global TAU_R
    TAU_R = P['tau_rate_birth']
    rgb = np.random.default_rng(P['seed'] + 77)
    for t in range(P['birth']):
        if t % P['hold'] == 0:
            k = rgb.integers(len(births), size=P['B'])
        xin = organ_in([(births[i][0], births[i][1]) for i in k])
        act = step(xin, act, adapt=P['adapt'])
    TAU_R = P['tau_rate']
    global THB
    THB = theta.clone()
    print('BIRTH done', json.dumps({nm: round(act[:, area == k_].float().mean().item(), 3) for nm, k_ in
                                     (('auditory', 1), ('concept', 2), ('visual', 3))}), f'({time.time() - t0:.0f} s)',
          flush=True)


# ---------------------------------------------------------------- information: can each stage tell the syllables apart?
def identity_by_stage(n_d=None, B=16, mode='sound'):
    """sound only. Spike counts per neuron over each syllable's held ticks (+ lag ticks), per stage: the ear's fibres,
    auditory shallow / deep, concept. Nearest class mean (half the hearings make the means, half are tested), the 60
    most common syllables; chance about 1/60. Real auditory cortex carries phoneme identity (Mesgarani 2014)."""
    global SUB
    n_d = n_d or int(P['id_nd'])                        # more dialogues: steadier class means (64 underrates sparse codes)
    per = n_d // B
    S = [stream(range(15000 + b * per, 15000 + (b + 1) * per), 9200, False if mode == 'sound' else 1.0)
         for b in range(B)]
    if mode == 'sight':                                 # the signs alone: the ear hears silence
        silent = ear([])
        S = [[(silent, x[1], x[2]) for x in st] for st in S]
    SL = None
    if mode == 'conflict':                              # each syllable heard with the sign of another, drawn at random
        rgc = np.random.default_rng(P['seed'] + 99)     # (Fable 11-12): does the area follow the sound or the sign?
        pool = [x[2] for st in S for x in st if x[2] >= 0]
        SL, S2 = [], []
        for st in S:
            sl, s2, cur = [], [], -1
            for i, x in enumerate(st):
                if x[2] < 0:
                    cur = -1
                elif i == 0 or st[i - 1][2] != x[2]:
                    cur = x[2]
                    while cur == x[2]:
                        cur = pool[rgc.integers(len(pool))]
                sl.append(cur)
                s2.append((x[0], eye(cur) if cur >= 0 else eye(None), x[2]))
            SL.append(sl)
            S2.append(s2)
        S = S2
    Tn = min(len(s) for s in S)
    act = torch.zeros(B, N, dtype=torch.bool, device=DEV)
    reset(B)
    stages = {'ear fibres': torch.arange(N, device=DEV) < n_ear,
              'eye cells': (torch.arange(N, device=DEV) >= n_ear) & (torch.arange(N, device=DEV) < n_in),
              'visual area': (area == 3) & ~inh,
              'auditory shallow': (area == 1) & (depth < 0.5) & ~inh,
              'auditory deep': (area == 1) & (depth >= 0.5) & ~inh,
              'concept': (area == 2) & ~inh}
    counts, labels = [], []
    for t in range(Tn):
        SUB = []
        act = step(organ_in([(S[b][t][0], S[b][t][1]) for b in range(B)]), act)
        counts.append(torch.stack(SUB).float().sum(0))                     # [B, N] spikes in this tick
        labels.append([S[b][t][2] for b in range(B)])
    SUB = None
    C = torch.stack(counts, 1)                                               # [B, T, N]
    Lb = np.array(labels)                                                    # [T, B]
    hold = P['hold']
    out = {}
    runs = [(nm, sel, None) for nm, sel in stages.items()]
    if SL is not None:                                  # the same hearings named by the sign that was shown
        Ls = np.array([[SL[b][t] for b in range(B)] for t in range(Tn)])
        runs = [(nm, sel, None) for nm, sel in stages.items() if nm in ('auditory deep', 'concept')] + \
               [(nm + ' by sign', sel, Ls) for nm, sel in stages.items() if nm in ('auditory deep', 'concept')]
    for nm, sel, Lname in runs:
        best = 0.0
        for lag in (0, 1, 2, 3, 4, 6):
            X, Y = [], []
            for b in range(B):
                t = 0
                while t < Tn - hold - lag:
                    y = Lb[t, b]
                    if y >= 0 and (t == 0 or Lb[t - 1, b] != y):                 # a syllable's first held tick
                        X.append(C[b, t + lag:t + lag + hold][:, sel].sum(0))
                        Y.append(int(y) if Lname is None else int(Lname[t, b]))
                    t += 1
            X = torch.stack(X)
            Y = torch.as_tensor(Y, device=DEV)
            top = torch.argsort(torch.bincount(Y), descending=True)[:60]
            k = torch.isin(Y, top)
            X, Y = X[k], Y[k]
            half = torch.rand(len(Y), generator=torch.Generator(device='cpu').manual_seed(3)).to(DEV) < 0.5
            cls = [c for c in top.tolist() if bool((half & (Y == c)).any())]
            M = torch.stack([X[half & (Y == c)].mean(0) for c in cls])
            Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
            te = (~half) & torch.isin(Y, torch.as_tensor(cls, device=DEV))
            Xn = X[te] / X[te].norm(dim=1, keepdim=True).clamp_min(1e-9)
            pos_ = {c: i for i, c in enumerate(cls)}
            ti = torch.as_tensor([pos_[int(c)] for c in Y[te].tolist()], device=DEV)
            alive = X[te].norm(dim=1) > 0                       # a silent stage cannot name anything (Fable 11-11)
            acc = (((Xn @ Mn.T).argmax(1) == ti) & alive).float().mean().item()
            best = max(best, acc)
        out[nm] = round(best, 3)
        out[nm + ' silent'] = round((X.sum(1) == 0).float().mean().item(), 2)
    return out


def growth_sig():
    """what a checkpoint must agree on: the wiring (idx, lidx) and the born weights, as grown from these arguments"""
    return BORN_SIG


def save_state(path):
    keep = {k: v for k, v in P.items() if k not in ('ckpt', 'n_learn', 'probe_every', 'eta_b', 'recal', 'id_nd')}
    torch.save(dict(theta=theta, w=w, E=E, I=I, mu=mu, NAU=NAU, APU=APU, RATE=RATE, THB=THB, args=keep,
                    sig=growth_sig()), path)


def load_state(path):
    global THB, TAU_R, FROZEN
    st = torch.load(path, map_location=DEV)
    if P['cand_ear']:                                    # only the concept rows' candidates differ: they learned nothing
        assert float(st['E'][(area[lrow] == 2)].abs().sum()) == 0.0, 'concept rows learned in this checkpoint'
        st['sig'] = st['sig'][:1] + [growth_sig()[1]] + st['sig'][2:3] + [growth_sig()[3]] + st['sig'][4:]
    if 'sig' not in st or st['sig'][:4] != growth_sig()[:4]:
        raise SystemExit('CHECKPOINT MISMATCH: the wiring differs from ' + str(path))
    for k in ('g_inh', 'rate_ix', 'U_dep', 'b_ad', 'dep_s', 'ad_s', 'tau_m', 'steps', 'rate_in', 'rate_out', 'organ'):
        if st['args'].get(k) != P.get(k):            # run-time settings the frozen senses were settled under
            raise SystemExit(f'CHECKPOINT MISMATCH: {k} was {st["args"].get(k)}, now {P.get(k)} ' + str(path))
    if abs(st['sig'][4] - growth_sig()[4]) > 1e-3 * abs(growth_sig()[4]):
        raise SystemExit('CHECKPOINT MISMATCH: the born weights differ (g_drv/g_mod/g_inh_c/K_ref?) ' + str(path))
    for k, v in (('theta', theta), ('w', w), ('E', E), ('I', I), ('mu', mu), ('NAU', NAU), ('APU', APU),
                 ('RATE', RATE)):
        v.copy_(st[k])
    THB = st['THB']
    TAU_R = P['tau_rate']
    if P['freeze_sens']:
        P['eta_stdp'] = 0.0
        FROZEN = True


def main():
    if P.get('ckpt') and os.path.exists(str(P['ckpt'])):   # birth and the senses' learning done before (same growth)
        load_state(P['ckpt'])
        print('LOADED', P['ckpt'], f'({time.time() - t0:.0f} s)', flush=True)
    else:
        grow_senses()
        if P.get('ckpt'):
            save_state(P['ckpt'])
    if P['g_vc'] != 1.0:                                 # diagnostic (11-12): a stronger second sense (the teacher) into
        vc = (area[:, None] == 2) & (area[idx.clamp_min(0)] == 3) & (idx >= 0)   # the concept area, after the senses
        w.copy_(torch.where(vc, w * P['g_vc'], w))
    main_concept()


def grow_senses():
    birth()
    if P['pre_stdp']:                                   # the senses learn first (STDP only), then are frozen while the
        eb = P['eta_b']                                 # concept area learns (Fable 11-11: a fixed student input)
        P['eta_b'] = 0.0
        run_streams(int(P['pre_stdp']), 4000, 7600, True, True)
        P['eta_b'] = eb
        if P['freeze_sens']:
            global FROZEN
            P['eta_stdp'] = 0.0
            FROZEN = True
        print('SENSES LEARNED', int(P['pre_stdp']), f'({time.time() - t0:.0f} s)', flush=True)


def recalibrate(n):
    """the concept area's thresholds and inhibition settle (fast, as at birth) to what the learned senses now send;
    the senses stay frozen and nothing is learned (11-12: a checkpoint made with thresholds fixed after birth)"""
    global TAU_R
    lf, eb = P['life_fix'], P['eta_b']
    P['life_fix'], P['eta_b'] = 0, 0.0
    TAU_R = P['tau_rate_birth']
    global RECAL
    RECAL = True
    run_streams(int(n), 10000, 7650, True, True, adapt=P["adapt"])   # dialogues 10000.. are unused elsewhere
    RECAL = False
    TAU_R = P['tau_rate']
    P['life_fix'], P['eta_b'] = lf, eb
    PHN.zero_()
    PHD.zero_()
    print('RECALIBRATED', int(n), json.dumps(dict(theta_concept=round(theta[(area == 2) & ~inh].mean().item(), 3))),
          f'({time.time() - t0:.0f} s)', flush=True)


def main_concept():
    if P['recal']:
        recalibrate(P['recal'])
    global RT
    RT = rate[lrow].clone()                              # start each cell at its target share of ticks
    print('IDENTITY 0', json.dumps(identity_by_stage()), flush=True)
    print('SIGHT IDENTITY 0', json.dumps(identity_by_stage(mode='sight')), flush=True)
    print('CONFLICT IDENTITY 0', json.dumps(identity_by_stage(mode='conflict')), flush=True)
    done = 0
    while done < P['n_learn']:
        run_streams(P['probe_every'], done, 7700, True, True)
        done += P['probe_every']
        if PLREC:
            M_ = torch.stack([r[0] for r in PLREC]).float()
            L_ = np.array([r[1] for r in PLREC])
            inter = M_ @ M_.T
            uni = M_.sum(1)[:, None] + M_.sum(1)[None] - inter
            jac = (inter / uni.clamp_min(1)).numpy()
            same = (L_[:, None] == L_[None]) & ~np.eye(len(L_), dtype=bool)
            diff = L_[:, None] != L_[None]
            print('PLATEAU SETS', done, json.dumps(dict(n=len(L_), size=round(M_.sum(1).mean().item(), 1),
                  same_sign=round(float(jac[same].mean()) if same.any() else -1, 3),
                  other_sign=round(float(jac[diff].mean()), 3))), flush=True)
            PLREC.clear()
        if DBG:
            dbm = torch.stack(DBG).mean(0).tolist()
            DBG.clear()
            print('CONCEPT LEARNING', done, json.dumps(dict(zip(('came', 'pred', 'dd_pos', 'dd_neg', 'push', 'theta'),
                                                               [round(v, 4) for v in dbm]))), flush=True)
            if P['phi_cal']:
                print('PHI', json.dumps([round(v, 3) for v in ((PHN + 1e-9) / (PHD + 1e-9)).tolist()[:30]]), flush=True)
        print('LEARNED', done, json.dumps(dict(E_pos=round((E > 0).float().mean().item(), 3),
                                              I_pos=round((I > 0).float().mean().item(), 3),
                                              mean_w=round((E - I).mean().item(), 4))), flush=True)
        print('IDENTITY', done, json.dumps(identity_by_stage()), f'({time.time() - t0:.0f} s)', flush=True)
        print('SIGHT IDENTITY', done, json.dumps(identity_by_stage(mode='sight')), flush=True)
        print('CONFLICT IDENTITY', done, json.dumps(identity_by_stage(mode='conflict')), flush=True)


if __name__ == '__main__':
    main()
