"""The minimal brain generator (GENERATOR.md 11-12 night; Fable a27).

Designer (11-12): "不需要完全按旧模型重新搭建模型，使用生成器生成极简模型就行，旧模型太乱了。"; "概念区本质上应该其实就是其他
各区输出层的信号当作概念区的输入层吧…由多点慢慢汇聚到一点。"; "本质上应该也没有前额叶，前额叶就是最上面几层，相当于概念区最后那几层，
但是有循环神经网络而已。"

ONE KIND OF LAYER. Binary units. Per input branch, the birth drive above each unit's own usual level (ARCH 二.4) in
units of a usual winner's drive, plus the learned push above its usual level; the k most driven fire (a fixed count:
nothing to calibrate, never silent, never all on).
ONE WIRING TABLE (the tree):
    ear -> A1 (fixed)  ->  W (taught by what is seen)  ->  C (taught by the top's expectation)
    the top ("PFC" = the last layers): W, R1 (sparse: what was just heard) and R2 (recurrent: what came before);
    its learned top-down links expect the next tick's W code (the surface). That expectation is the return; C is
    taught with the code its fixed wiring gives to the confidently expected W pattern (Fable a29).
ONE RULE (rule b, branch form): a learning branch moves toward what ANOTHER source lit in the same tick, and only on
ticks where that teacher acts:  dw_ij = step_ij * (came_j - pred_j) * x_i / (#active inputs)
    pred_j = clip(the branch's push, 0, 1), counted where came_j or the push >= 1; split into excitatory E and
    inhibitory I links; step_ij = step * 3 / (1 + 2 |w_ij| / w_ref): a new link moves at 3x, a mature one at about 1x
    (his "前期让模型权重改变的强一点", by link, not by age: ARCH 四's concave rule); a taught unit is strengthened up to
    `margin` x a usual winner's push. Teachers: W - the units what is seen lights (chosen against each unit's own usual
    sight drive); C - see above; the top's links (graded: every target corrects) - the W code that came bottom-up.
TIMING (the fault found 11-12 in the old line: returns came a tick late and carried the failed guesses about the
previous syllable): the expectation E(t) comes from the top's state at t-1 BEFORE the sweep of t; every layer learns
at the end of t against E(t), the sight of t and the codes of t: the same slot. One tick = one syllable.
B dialogues run in parallel; a unit's update is divided by the number of streams where its error was non-zero.
Probes never learn and never move usual levels."""
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
from .concept_eval import evaluate, random_projection

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
P = dict(data=os.environ.get('CONCEPT_DATA', 'concept-data'), seed=1, B=16, n_d=6000, probe_every=1000, eval_nd=768, eval_ntr=6,
         # layers: size, share firing, birth fan-in (Fable a27: about 5-12 active inputs per unit at birth)
         nA1=4000, kA1=0.04, KA1=24, dyA1=2.0, nW=4000, kW=0.02, KW=150, KWs=24, nC=4000, kC=0.02, KC=250,
         nR1=8000, kR1=0.005, KR1=400, nR2=4000, kR2=0.10, KR2=250, KR2r=60, g_rec=0.7, share_R1=0.8, min_elit=20,
         split_hi_R1=10.0, top_target='W',
         # 11-12 night (Fable a32; designer: "模型可能接收着新的信息，慢慢的就忘了旧的信息了" and "人总是在重复接受重复的信息"):
         # a budget on each R1 unit's learned excitatory weight (a unit cannot own many syllables); splits divide the
         # parent's links instead of copying them; the top's rows slow down as their input has been used (less
         # forgetting: toward the whole history); replay = old dialogues heard again between new ones (hippocampus)
         cap_R1=6.0, split_mode='divide', top_n0=40.0, top_fmin=0.04, replay=0, repeat=1,
         # the rule
         step=0.15, margin=2.0, step_top=0.05, tau_usual=1000.0, g_c=1.0, e_conf=0.3, e_margin=0.2, cal_ticks=2000,
         # neurons split and vanish (the designer's early rule, 11-12: "总是激活的神经元会分裂成多个神经元…而总是不激活的
         # 神经元有会慢慢消失"): rates over sound ticks; a never-lit unit's place becomes the child of an always-lit one
         split=1, rate_tau=3000.0, split_hi=4.0, dead_lo=0.05, split_every=500, split_max=0.01, split_start=2000,
         # the world
         n_sight=256, p_see=0.5, n_held=10, held_voice=3, gap_turn=[2, 4],
         # the hippocampus (ARCH 海马体; designer: it remembers fast because its weights change fast; context is the
         # PFC's; Fable a33): own units, random copies of W and R2 (no R1: the same information as W, worse), no birth
         # recurrence, fixed number lit; recall links H(t-1) -> W(t), excitatory only, one per stream (each stream is
         # one life), stored in one hearing (step_H: the first hearing brings the push to ~1.2, margin 2), faded with a
         # half-life of h_half ticks (the one global decay, every h_every ticks). rep_fresh: each repeat in new voices.
         H=0, nH=8000, kH=0.005, KHW=250, KHR2=50, step_H=1.2, h_half=6000.0, h_every=50, rep_fresh=1, probe_tag=7700,
         probes=1, save='')
for a in sys.argv[1:] if __name__ == '__main__' else os.environ.get('GEN_ARGS', '').split():
    k_, v_ = a.split('=', 1)
    try:
        P[k_] = json.loads(v_)
    except json.JSONDecodeError:
        P[k_] = v_
torch.manual_seed(P['seed'])
RNG = np.random.default_rng(P['seed'])
T0 = time.time()
B = P['B']
VIDX = {a: i for i, a in enumerate(ADULTS)}


def say(*a):
    print(*a, f'({time.time() - T0:.0f} s)', flush=True)


# ---- the world (as the old line: one tick per syllable, the ear's thermometer bits, a fixed pattern per sign)
WC = config(levels_per_s=4, syl_ticks=1, gap_turn=P['gap_turn'], gap_dlg=[6, 12])
COCH = Cochlea(Voice(phon.load_inventory()), WC, P['seed'])
TRAIN = Dialogues.load(os.path.join(P['data'], 'train.npz'))
N_EAR = COCH.n
SILENT = COCH.a0([])
_SG = {}


def sight(sid):
    """the sign of a syllable: a fixed random pattern, half on (the world, not the brain)"""
    if sid not in _SG:
        _SG[sid] = np.random.default_rng([P['seed'], 5150, int(sid)]).random(P['n_sight']) < 0.5
    return _SG[sid]


HELD = set()                                              # syllables never shown a sign while learning


def stream(ds, tag, see, learning):
    """per tick: (ear bits, sign or -1, syllable or -1, voice or -1, dialogue, (turn, place) or None)"""
    rg = np.random.default_rng([P['seed'], tag, int(ds[0])])
    out = []
    for d in ds:
        src = script_dialogue(TRAIN, d, np.random.default_rng([P['seed'], tag, d]), tuple(P['gap_turn']), 8, 1, 1)
        for x in src:
            if x is None:
                out.append((SILENT, -1, -1, -1, d, None))
                continue
            sid, vo = int(x[0]), VIDX[x[2]]
            shown = rg.random() < see and not (learning and (sid in HELD or vo == P['held_voice']))
            out.append((COCH.a0([x[:5]]), sid if shown else -1, sid, vo, d, (int(x[6]), int(x[7]))))
    return out


def batch_in(S, t):
    ear = torch.as_tensor(np.stack([S[b][t][0] for b in range(B)]), device=DEV).float()
    sg = [S[b][t][1] for b in range(B)]
    eye = torch.as_tensor(np.stack([sight(s) if s >= 0 else np.zeros(P['n_sight'], bool) for s in sg]),
                          device=DEV).float()
    return ear, eye, torch.as_tensor([s >= 0 for s in sg], device=DEV)


def ear_on(ear):
    """the streams hearing a syllable (the ear differs from silence)"""
    return (ear != SIL_T[None]).any(1)


SIL_T = torch.as_tensor(SILENT, device=DEV).float()


def topk_b(d, k):
    """per row, the k most driven units with positive drive"""
    out = torch.zeros_like(d, dtype=torch.bool)
    v, i = torch.topk(d, int(k), dim=1)
    out.scatter_(1, i, v > 0)
    return out


def rule_b(E, I, x, came, push, act, step, w_ref, graded=False, blocks=None, concave=True, row_f=None):
    """the one rule on a learned branch (E, I: [n_in, n]); x [B, n_in] its inputs; came [B, n] what another source lit;
    push [B, n] the branch's push in winner units; act [B] the streams where the teacher acts this tick.
    The clause form (W, C; the old line's): pred counts where came or push >= 1. graded (the top, Fable a28 #1): every
    target corrects toward what came, so the push converges to the chance of coming given the context (the clause
    form only ever rises until 1: "has ever followed", not "is likely to follow"). blocks: input ranges normalised
    each by its own active count and sharing the step equally (a28 #4: 400 recurrent units drowned 80 W units).
    Streams: each link's update is divided by the number of streams in which its input was active and its unit had
    an error (a28 #2)."""
    pred = push.clamp(0.0, 1.0)
    if not graded:
        # a unit the teacher lit is strengthened until its push reaches `margin` x a usual winner's (ARCH 六's margin;
        # Fable a30: stopping at 1 left taught units tied with birth-noise winners, so each hearing's noise chose the
        # code: class means good, per-hearing cores ~0.5 cells); an untaught unit is weakened only above 1
        pred = torch.where(came, (push / P['margin']).clamp(0.0, 1.0),
                           torch.where(push >= 1.0, pred, torch.zeros_like(pred)))
    dd = (came.float() - pred) * act[:, None].float()
    if blocks is None:
        xn = x / x.sum(1, keepdim=True).clamp_min(1.0)
    else:
        xn = torch.cat([sh * x[:, a:b] / x[:, a:b].sum(1, keepdim=True).clamp_min(1.0) for a, b, sh in blocks], 1)
    cnt = (x > 0).float().T @ (dd != 0).float()                   # streams per link with input and error
    d = 0.5 * step * (xn.T @ dd) / cnt.clamp_min(1.0)
    if concave:
        d = d * 3.0 / (1.0 + 2.0 * (E - I).abs() / w_ref)         # new links fast, mature links slow (concave)
    if row_f is not None:
        d = d * row_f[:, None]
    E.add_(d).clamp_(min=0.0)
    I.sub_(d).clamp_(min=0.0)
    return float((dd > 0).float().sum()), float((dd < 0).float().sum())


# ---- the one kind of layer
class Layer:
    """binary units; a fixed birth branch per source (random links, fan-in K, weights 0.5..1.5; idx may be given for
    local wiring) and, optionally, one learned branch from a source (dense E and I links, zero at birth)"""

    def __init__(self, name, n, k, srcs, learn=None, n_learn=0):
        self.name, self.n, self.k = name, n, max(1, int(round(k * n)))
        self.birth, self.n_src = {}, {s: v[0] for s, v in srcs.items()}
        for s, (n_s, K, idx) in srcs.items():
            if idx is None:
                idx = RNG.integers(0, n_s, (n, K))
            self.birth[s] = (torch.as_tensor(idx, device=DEV),
                             torch.as_tensor(RNG.uniform(0.5, 1.5, (n, K)), dtype=torch.float32, device=DEV))
        self.learn = learn
        if learn is not None:
            self.E = torch.zeros(n_learn, n, device=DEV)
            self.I = torch.zeros(n_learn, n, device=DEV)
            self.mu = torch.zeros(n, device=DEV)
        self.usual = {s: torch.zeros(n, device=DEV) for s in srcs}
        self.win = {s: torch.ones((), device=DEV) for s in srcs}
        self.on = torch.zeros(B, n, dtype=torch.bool, device=DEV)
        self.push = None
        self.rate = torch.full((n,), self.k / n, device=DEV)    # each unit's share of sound ticks it fires in

    def branch(self, s, x, adapt, rows=None):
        """a source's birth drive above each unit's own usual level, in units of a usual winner's drive. rows: the
        streams whose drive moves the usual level (default all)"""
        idx, w = self.birth[s]
        e = (x[:, idx] * w).sum(-1) - self.usual[s]
        if adapt:
            m = torch.ones(B, dtype=torch.bool, device=DEV) if rows is None else rows
            if bool(m.any()):
                self.usual[s] += e[m].mean(0) / P['tau_usual']
                top = topk_b(e[m], self.k)
                if bool(top.any()):
                    self.win[s] += (e[m][top].mean() - self.win[s]) / P['tau_usual']
        return e / self.win[s].clamp_min(1e-6)

    def raw(self, s, x):
        """a source's raw birth drive (its usual level is the mean of this)"""
        idx, w = self.birth[s]
        return (x[:, idx] * w).sum(-1)

    def learned(self, x, adapt, rows=None):
        """the learned branch's push above its own usual level, in winner units (the usual level moves on rows)"""
        q = x @ (self.E - self.I)
        if adapt and (rows is None or bool(rows.any())):
            qq = q if rows is None else q[rows]
            self.mu += (qq - self.mu).mean(0) / P['tau_usual']
        return P['g_c'] * (q - self.mu)


def local_ear(n, K, dy):
    """A1's birth inputs: around a centre channel on the tonotopic axis (Gaussian spread dy channels), any level"""
    NC, Z = COCH.NC, COCH.Z
    c0 = RNG.integers(0, NC, n)
    ch = np.clip(np.rint(c0[:, None] + RNG.normal(0.0, dy, (n, K))), 0, NC - 1).astype(np.int64)
    return ch * Z + RNG.integers(0, Z, (n, K))


# ---- the tree
nA1, nW, nC, nR1, nR2 = P['nA1'], P['nW'], P['nC'], P['nR1'], P['nR2']
A1 = Layer('A1', nA1, P['kA1'], {'ear': (N_EAR, P['KA1'], local_ear(nA1, P['KA1'], P['dyA1']))})
W = Layer('W', nW, P['kW'], {'A1': (nA1, P['KW'], None), 'eye': (P['n_sight'], P['KWs'], None)}, learn='A1', n_learn=nA1)
C = Layer('C', nC, P['kC'], {'W': (nW, P['KC'], None)}, learn='W', n_learn=nW)
# the top (Fable a29, a31): R1 sparse and LEARNED - competitive Hebbian on W taught by its own winners, so its units
# settle on each syllable's always-lit core cells (his "概念就是总是激活的神经元") and the top reads concepts, not the
# feature mixture; R2 recurrent (what came before). The top's learned links [R1, R2](t-1) -> W(t) expect the surface.
R1 = Layer('R1', nR1, P['kR1'], {'W': (nW, P['KR1'], None)}, learn='W', n_learn=nW)
R2 = Layer('R2', nR2, P['kR2'], {'W': (nW, P['KR2'], None), 'R2': (nR2, P['KR2r'], None)})
# the top's links: [R1, R2](t-1) -> W(t), or with top_target R1 -> R1(t): the concepts' own recurrent association
# (his "前额叶再有一个循环神经网络记住连续的事"; the W code is mostly occasionally-lit features, so expecting it blurred)
N_PRE, N_TGT = nR1 + nR2, (nW if P['top_target'] == 'W' else nR1)
BLOCKS = [(0, nR1, P['share_R1']), (nR1, nR1 + nR2, 1.0 - P['share_R1'])]
USE_H = bool(P['H'])
if USE_H:
    nH = P['nH']
    H = Layer('H', nH, P['kH'], {'W': (nW, P['KHW'], None), 'R2': (nR2, P['KHR2'], None)})
    HT = torch.zeros(B, nH, nW, device=DEV)                       # recall links per stream, excitatory only
TE = torch.zeros(N_PRE, N_TGT, device=DEV)
TI = torch.zeros(N_PRE, N_TGT, device=DEV)
WREF = dict(W=1.0 / max(1, A1.k), C=1.0 / max(1, W.k), R1=1.0 / max(1, W.k))   # winner level / active inputs
ST = dict(pre=torch.zeros(B, N_PRE, device=DEV), usualE=torch.zeros(N_TGT, device=DEV), Ex=None, dlg=0,
          snd_prev=torch.zeros(B, dtype=torch.bool, device=DEV), nrow=torch.zeros(N_PRE, device=DEV),
          usualH=torch.zeros(nW, device=DEV), wmean=torch.full((nW,), P['kW'], device=DEV), ExT=None, ExH=None,
          hidx=None, hval=None, rep=None, hT=0)
EV = {}
BI = torch.arange(B, device=DEV)[:, None]
KM = {}
KEY_LAYERS = lambda: dict(A1=A1, W=W, R1=R1, R2=R2, **({'H': H} if USE_H else {}))


def reset():
    for ly in (A1, W, C, R1, R2):
        ly.on = torch.zeros(B, ly.n, dtype=torch.bool, device=DEV)
    ST['pre'] = torch.zeros(B, N_PRE, device=DEV)
    ST['snd_prev'] = torch.zeros(B, dtype=torch.bool, device=DEV)
    ST['hidx'], ST['hval'] = None, None
    if USE_H:
        H.on = torch.zeros(B, nH, dtype=torch.bool, device=DEV)


def ev(k, v):
    EV[k] = EV.get(k, 0.0) + v


def tick(ear, eye, seen, adapt, learn):
    """one syllable: the top's expectation for this tick, the bottom-up sweep, the top, then learning (same slot)"""
    pre = ST['pre']                                               # [R1, R2] as they were at t-1
    ExT = pre @ (TE - TI)                                         # the top's expectation of W at t (a share 0..1)
    ExH = torch.zeros_like(ExT)
    if USE_H and ST['hidx'] is not None:                          # the hippocampus's recall from H(t-1): its rows
        rows = HT[BI, ST['hidx']]                                 # [B, kH, nW]
        ExH = (rows * ST['hval'][..., None].float()).sum(1)
    Ex = ExT + ExH                                                # recall arrives as a return on W, beside the top's
    snd = ear_on(ear)                                             # usual levels move on streams hearing a syllable
    if adapt and bool(snd.any()):
        ST['usualE'] += (ExT - ST['usualE'])[snd].mean(0) / P['tau_usual']
        ST['usualH'] += (ExH - ST['usualH'])[snd].mean(0) / P['tau_usual']
    eu = (ExT - ST['usualE']) + (ExH - ST['usualH'])
    # the bottom-up sweep
    A1.push = A1.branch('ear', ear, adapt, rows=snd)
    A1.on = topk_b(A1.push, A1.k)
    a1 = A1.on.float()
    W.push = W.branch('A1', a1, adapt, rows=snd) + W.learned(a1, adapt, rows=snd)
    W.on = topk_b(W.push, W.k)                                    # the code sent on: from the sound
    vlit = topk_b(W.branch('eye', eye, adapt, rows=seen), W.k) & seen[:, None]   # what is seen lights its own units
    w = W.on.float()
    C.push = C.branch('W', w, adapt, rows=snd) + C.learned(w, adapt, rows=snd)
    C.on = topk_b(C.push, C.k)
    # the top: what was just heard (R1) and what came before (R2, recurrent)
    R1.push = R1.branch('W', w, adapt, rows=snd) + R1.learned(w, adapt, rows=snd)
    R1.on = topk_b(R1.push, R1.k)
    R2.push = R2.branch('W', w, adapt, rows=snd) + P['g_rec'] * R2.branch('R2', pre[:, nR1:], adapt, rows=snd)
    R2.on = topk_b(R2.push, R2.k)
    if USE_H:                                                     # the hippocampus: copies of W and R2 (the PFC's
        H.push = H.branch('W', w, adapt, rows=snd) + H.branch('R2', R2.on.float(), adapt, rows=snd)   # context)
        hv_, hi_ = torch.topk(H.push, H.k, dim=1)
        hval_ = hv_ > 0
        H.on = torch.zeros_like(H.push, dtype=torch.bool).scatter_(1, hi_, hval_)
    if learn and ST['rep'] is not None and bool((snd & ST['snd_prev']).any()):
        # online (designer 11-12: the PFC predicts the rough trend, the exact next word is recall): the share of the
        # actual W code inside the top W.k of each expectation, on sound after sound, by hearing; 'usual' = the
        # average W pattern (knowing nothing about what comes next)
        m = snd & ST['snd_prev']
        for nm, X in (('top', ExT), ('H', ExH), ('sum', Ex), ('usual', ST['wmean'][None].expand_as(ExT))):
            hit = W.on.gather(1, torch.topk(X, W.k, dim=1)[1]).float().sum(1) / W.k
            ev(f'on r{ST["rep"]} {nm}', float(hit[m].sum()))
        ev(f'on r{ST["rep"]} n', float(m.sum()))
    if learn:
        # W: taught by what is seen, on the ticks something is seen
        u, dn = rule_b(W.E, W.I, a1, vlit, W.push, seen, P['step'], WREF['W'])
        ev('W up', u)
        ev('W down', dn)
        # R1: competitive Hebbian - its own winners teach it, on sound ticks (Fable a31: the core cells come on every
        # hearing and are reinforced every time; the varying ones only sometimes, and are cut when pushed and losing)
        u, dn = rule_b(R1.E, R1.I, w, R1.on, R1.push, snd, P['step'], WREF['R1'])
        ev('R1 up', u)
        ev('R1 down', dn)
        if P['cap_R1'] > 0:                                       # each unit's learned excitatory budget (Diehl &
            cs = R1.E.sum(0)                                      # Cook's fixed sum): learning a second syllable
            R1.E.mul_((P['cap_R1'] / cs.clamp_min(1e-9)).clamp(max=1.0)[None, :])   # takes weight from the first
        # the surface the top expects confidently (push >= e_conf and >= e_margin above its own usual), at most k
        # (only when the top expects W; with top_target R1, C is not taught)
        if P['top_target'] != 'W':
            Ex_w, eu_w = torch.zeros(B, nW, device=DEV), torch.zeros(B, nW, device=DEV)
        else:
            Ex_w, eu_w = Ex, eu
        conf = (Ex_w >= P['e_conf']) & (eu_w >= P['e_margin'])
        elitW = topk_b(torch.where(conf, eu_w, torch.zeros_like(eu_w)), W.k)
        nE = elitW.float().sum(1)
        # only an expectation made in speech about speech teaches (11-12 smoke 4: most confident expectations were
        # "the silence goes on" at the first syllable of a turn: 2.6 % right, chance level)
        act = (nE >= P['min_elit']) & snd & ST['snd_prev']
        # C's teacher (Fable a29 B): the code C's FIXED wiring gives to that expected surface pattern, against each
        # unit's usual drive scaled to the pattern's size. C's learned links never shape their own target (the loop
        # through the targets made C generic in the 11-12 smoke). For shown syllables W is already voice-blind, so
        # the target equals C's own birth code; the effect belongs to the held syllables and the held voice.
        ce = C.raw('W', elitW.float()) - C.usual['W'][None] * (nE / W.k)[:, None]
        ctgt = topk_b(ce, C.k) & act[:, None]
        u, dn = rule_b(C.E, C.I, w, ctgt, C.push, act, P['step'], WREF['C'])
        ev('C up', u)
        ev('C down', dn)
        ev('C taught ticks (sound)', float((act & snd).float().sum()))
        ev('sound ticks', float(snd.float().sum()))
        ev('W E-lit units (sound)', float(elitW[snd].float().sum()))
        ev('W E-lit in the actual W code (sound)', float((elitW & W.on)[snd].float().sum()))
        if bool(snd.any()):
            for ly in (W, C, R1):
                ly.rate += (ly.on[snd].float().mean(0) - ly.rate) / P['rate_tau']
            ST['wmean'] += (W.on[snd].float().mean(0) - ST['wmean']) / P['rate_tau']
        if USE_H and ST['hidx'] is not None:
            # the recall links: the clause form of rule b on its own push (separate from the top's error, so the top
            # keeps learning the statistics): what came is strengthened until the push reaches the margin; what did not
            # come is weakened only where it was recalled (push >= 1). Every tick, sound or not ("什么都存"). Each of
            # the stream's active rows moves by step_H / (active rows): one hearing brings the push to ~step_H.
            hv = ST['hval']
            pr = torch.where(W.on, (ExH / P['margin']).clamp(0.0, 1.0),
                             torch.where(ExH >= 1.0, ExH.clamp(0.0, 1.0), torch.zeros_like(ExH)))
            upd = P['step_H'] * (W.on.float() - pr) / hv.float().sum(1, keepdim=True).clamp_min(1.0)
            HT[BI, ST['hidx']] = (HT[BI, ST['hidx']] + upd[:, None, :] * hv[..., None].float()).clamp_(min=0.0)
            ev('H recall up', float((W.on.float() - pr > 0).float().sum()))
            ev('H recall down', float((W.on.float() - pr < 0).float().sum()))
            ST['hT'] += 1
            if ST['hT'] % P['h_every'] == 0:                       # the one global decay, fast for the hippocampus
                HT.mul_(0.5 ** (P['h_every'] / P['h_half']))
        # the top's links: its expectation against the W code that came bottom-up (every tick with input)
        has = pre.sum(1) > 0
        rf = None
        if P['top_fmin'] < 1.0:                                   # a row slows as its input has been used: the
            rf = (P['top_n0'] / (P['top_n0'] + ST['nrow'])).clamp(min=P['top_fmin'])   # estimate keeps old data
            ST['nrow'] += (pre[has] > 0).float().sum(0)
        u, dn = rule_b(TE, TI, pre, W.on if P['top_target'] == 'W' else R1.on, ExT, has, P['step_top'], None, graded=True,
                       blocks=BLOCKS, concave=False, row_f=rf)
        ev('top up', u)
        ev('top down', dn)
        ev('ticks', float(B))
    ST['pre'] = torch.cat([R1.on.float(), R2.on.float()], 1)
    ST['snd_prev'] = snd
    ST['Ex'], ST['ExT'], ST['ExH'] = Ex, ExT, ExH
    if USE_H:
        ST['hidx'], ST['hval'] = hi_, hval_


def split_layer(L, tcol, prow=None, hi=None):
    """the designer's rule: units firing far more than their share split, units that never fire vanish. Done as one
    move so the count stays fixed: a never-lit unit's place becomes the child of an always-lit one. The child takes
    the parent's input wiring with +-30 % noise; the learned links are shared (each keeps half, with noise), so the pair
    together pushes as the parent did and each must re-earn its inputs; a child of W gets a new sight wiring (a new
    teacher), so the two are pulled to different things. tcol: the unit's column in the top's targets; prow: its row
    in the top's predictors (W only)."""
    share = L.k / L.n
    busy = torch.nonzero(L.rate > (hi or P['split_hi']) * share).squeeze(1)
    if len(busy) == 0:
        return 0
    # the places: units that never fire; if there are too few, the least used below half their share (11-12: C had
    # generic units in 13 of 27 cores but no dead unit, so nothing split; the two halves of his rule are separate)
    order = torch.argsort(L.rate)
    dead = order[L.rate[order] < 0.5 * share][:len(busy)]
    m = min(len(busy), len(dead), max(1, int(P['split_max'] * L.n)))
    if m == 0:
        return 0
    busy = busy[torch.argsort(-L.rate[busy])][:m]
    dead = dead[:m]
    nz = lambda shape: torch.empty(shape, device=DEV).uniform_(0.7, 1.3)
    div = P['split_mode'] == 'divide'
    for src_name, (idx, w) in L.birth.items():
        if div and src_name != 'eye':                               # divide (Fable a32): the child's birth wiring is
            idx[dead] = torch.randint(0, L.n_src[src_name], (m, idx.shape[1]), device=DEV)   # redrawn, so competition
            w[dead] = torch.empty((m, idx.shape[1]), device=DEV).uniform_(0.5, 1.5)          # pulls the two apart
            L.usual[src_name][dead] = L.usual[src_name].mean()
        elif src_name == 'eye':                                       # a new teacher wiring for the child, with its
            idx[dead] = torch.randint(0, P['n_sight'], (m, idx.shape[1]), device=DEV)   # own reference (the bits are
            w[dead] = torch.empty((m, idx.shape[1]), device=DEV).uniform_(0.5, 1.5)     # half on; Fable a30 #2)
            L.usual[src_name][dead] = 0.5 * w[dead].sum(1)
        else:
            idx[dead] = idx[busy]
            w[dead] = w[busy] * nz((m, idx.shape[1]))
            L.usual[src_name][dead] = L.usual[src_name][busy]
    if L.learn is not None:
        mask = (torch.rand((L.E.shape[0], m), device=DEV) < 0.5).float() if div else None
        for M in (L.E, L.I):
            if div:                                                 # each learned link goes to one of the two
                par = M[:, busy]
                M[:, dead] = par * mask
                M[:, busy] = par * (1.0 - mask)
            else:
                half = 0.5 * M[:, busy]
                M[:, dead] = half * nz(half.shape)
                M[:, busy] = half
        L.mu[busy] *= 0.5
        L.mu[dead] = L.mu[busy]
    if L is W:                                                      # the child keeps the parent's learned outgoing
        for M in (C.E, C.I, R1.E, R1.I):                            # links, half each (a30 #3; the fixed birth
            M[busy, :] *= 0.5                                       # wiring downstream stays as drawn)
            M[dead, :] = M[busy, :]
    for M in (TE, TI):
        if tcol is not None:                                        # the top expects the child as it did the parent
            M[:, tcol[dead]] = M[:, tcol[busy]]
        if prow is not None:                                        # and predicts half from each
            M[prow[busy], :] *= 0.5
            M[prow[dead], :] = M[prow[busy], :]
    if tcol is not None:
        ST['usualE'][tcol[dead]] = ST['usualE'][tcol[busy]]
    if L is W:                                                      # the recall of the child as of the parent
        ST['usualH'][dead] = ST['usualH'][busy]
        ST['wmean'][dead] = ST['wmean'][busy]
        if USE_H:
            HT[:, :, dead] = HT[:, :, busy]
    L.rate[busy] *= 0.5
    L.rate[dead] = L.rate[busy]
    return m


def run(ds_per_stream, d0, tag, learn, see, record=False, adapt=None, lists=None, keymem=None):
    """keymem 'store': keep each stream's H code per (dialogue, turn, place); 'compare': the share of it that comes
    again at the same place on this hearing (Fable a33: the key must recur, or nothing can be recalled)"""
    if lists is None:
        lists = [range(d0 + b * ds_per_stream, d0 + (b + 1) * ds_per_stream) for b in range(B)]
    S = [stream(lists[b], tag, see, learn) for b in range(B)]
    T = min(len(s) for s in S)
    reset()
    rec = dict(A1=[], W=[], C=[], R1=[], Ex=[], ExT=[], ExH=[], H=[], lab=[], voice=[], dlg=[])
    if keymem == 'store':
        KM['idx'], KM['at'] = {nm: [] for nm in KEY_LAYERS()}, [dict() for _ in range(B)]
    last_d = [None] * B
    for t in range(T):
        ear, eye, seen = batch_in(S, t)
        tick(ear, eye, seen, learn if adapt is None else adapt, learn)
        if keymem == 'store':                                     # every layer's code, by place in the dialogue
            for nm, ly in KEY_LAYERS().items():
                v_, i_ = torch.topk(ly.on.float(), ly.k, dim=1)
                KM['idx'][nm].append(torch.where(v_ > 0, i_, torch.full_like(i_, -1)))
            for b in range(B):
                if S[b][t][5] is not None:
                    KM['at'][b][(S[b][t][4],) + S[b][t][5]] = t
        elif keymem == 'compare' and KM.get('idx'):
            t0 = [KM['at'][b].get((S[b][t][4],) + S[b][t][5], -1) if S[b][t][5] is not None else -1 for b in range(B)]
            ok = [b for b in range(B) if t0[b] >= 0]
            if ok:
                for nm, ly in KEY_LAYERS().items():                 # the share of hearing 1's code that is lit again
                    old = torch.stack([KM['idx'][nm][t0[b]][b] for b in ok])
                    same = (ly.on[ok].gather(1, old.clamp(min=0)) & (old >= 0)).float().sum(1)
                    ev(f'again {nm} r{ST["rep"]}', float((same / (old >= 0).float().sum(1).clamp_min(1.0)).sum()))
                ev(f'again n r{ST["rep"]}', float(len(ok)))
        if learn and P['split'] and ST['dlg'] >= P['split_start'] and t % P['split_every'] == P['split_every'] - 1:
            tw = P['top_target'] == 'W'
            ev('W split', float(split_layer(W, torch.arange(nW, device=DEV) if tw else None)))
            ev('C split', float(split_layer(C, None)))
            ar1 = torch.arange(nR1, device=DEV)
            ev('R1 split', float(split_layer(R1, None if tw else ar1, ar1, P['split_hi_R1'])))
        if learn:
            for b in range(B):
                if S[b][t][4] != last_d[b]:
                    last_d[b] = S[b][t][4]
                    ST['dlg'] += 1
        if record:
            rec['A1'].append(A1.on.cpu())
            rec['W'].append(W.on.cpu())
            rec['C'].append(C.on.cpu())
            rec['R1'].append(R1.on.cpu())
            rec['Ex'].append(ST['Ex'].half().cpu())
            rec['ExT'].append(ST['ExT'].half().cpu())
            rec['ExH'].append(ST['ExH'].half().cpu())
            rec['H'].append(H.on.cpu() if USE_H else torch.zeros(B, 1, dtype=torch.bool))
            rec['lab'].append([S[b][t][2] for b in range(B)])
            rec['voice'].append([S[b][t][3] for b in range(B)])
            rec['dlg'].append([S[b][t][4] for b in range(B)])
    if record:
        return {k: (torch.stack(v, 1) if torch.is_tensor(v[0]) else np.array(v).T) for k, v in rec.items()}
    return T


# ---- measures
BIGRAM = None


def bigram(n_d=8000):
    """next-syllable counts within turns of the first n_d training dialogues (the yardstick for the top)"""
    cnt = {}
    for d in range(n_d):
        for tr in TRAIN.turns(d):
            for a, b in zip(tr[:-1], tr[1:]):
                cnt.setdefault(int(a), {}).setdefault(int(b), 0)
                cnt[int(a)][int(b)] += 1
    return cnt


def trend(R_):
    """designer 11-12: "模型前额叶预测的应该是大致趋势而不是精确的下一个字吧？" - the share of the actual W code (sound
    after sound) inside the top W.k of each expectation: partial matches count (W's features are shared by similar
    syllables). 'usual' = the average W pattern, i.e. knowing nothing about what comes next."""
    lab = R_['lab']
    m = torch.as_tensor((lab[:, 1:] >= 0) & (lab[:, :-1] >= 0))
    Wn = R_['W'][:, 1:][m].float()
    out = {}
    for nm, X in (('top', R_['ExT'][:, 1:][m].float()), ('recall', R_['ExH'][:, 1:][m].float()),
                  ('sum', R_['Ex'][:, 1:][m].float()), ('usual', ST['wmean'].cpu()[None].expand(len(Wn), -1))):
        out[nm] = round(float((Wn.gather(1, torch.topk(X, W.k, dim=1)[1]).sum(1) / W.k).mean()), 3)
    return out


def next_only(R_, which='Ex'):
    """next-syllable top-1 / top-10 from an expectation (Ex = top + recall, ExT the top, ExH the recall) on a
    recording (class means from the same recording)"""
    lab = R_['lab']
    TG = R_['W' if P['top_target'] == 'W' else 'R1'].float()
    Ex = R_[which].float()
    items = [(b, t) for b in range(B) for t in range(lab.shape[1] - 1) if lab[b, t] >= 0 and lab[b, t + 1] >= 0]
    nxt = np.array([lab[b, t + 1] for b, t in items])
    cnt = np.bincount(nxt)
    cands = [int(c_) for c_ in np.argsort(-cnt)[:300] if cnt[c_] > 0]
    pos = {c_: i for i, c_ in enumerate(cands)}
    M = torch.stack([TG[[b for (b, t), l_ in zip(items, nxt) if l_ == c_],
                        [t + 1 for (b, t), l_ in zip(items, nxt) if l_ == c_]].mean(0) for c_ in cands])
    M = M - M.mean(0, keepdim=True)
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    usualE = dict(Ex=ST['usualE'] + ST['usualH'], ExT=ST['usualE'], ExH=ST['usualH'])[which].cpu()
    rk = []
    for (b, t), nx in zip(items, nxt):
        if int(nx) not in pos:
            continue
        e = Ex[b, t + 1] - usualE
        sc = Mn @ (e / e.norm().clamp_min(1e-9))
        rk.append(int((sc > sc[pos[int(nx)]]).sum() + (sc == sc[pos[int(nx)]]).sum() - 1))
    rk = np.array(rk)
    return dict(top1=round(float(np.mean(rk == 0)), 3), top10=round(float(np.mean(rk < 10)), 3), n=len(rk))


def probe(tag):
    global BIGRAM
    R_ = run(P['eval_nd'] // B, 15000, 9300, False, False, record=True)
    lab, vo, dl = R_['lab'], R_['voice'], R_['dlg']
    keys = ('A1', 'W', 'C', 'R1')
    X = {k: [] for k in keys}
    y, v, d = [], [], []
    for b in range(B):
        for t in range(lab.shape[1]):
            if lab[b, t] >= 0:
                for k in keys:
                    X[k].append(R_[k][b, t])
                y.append(int(lab[b, t]))
                v.append(int(vo[b, t]))
                d.append(int(dl[b, t]))
    X = {k: torch.stack(val).float().to(DEV) for k, val in X.items()}
    y, v, d = np.array(y), np.array(v), np.array(d)
    X['RP'] = random_projection(X['A1'], nC, float(X['C'].sum(1).mean()), seed=P['seed'])
    out = {'lit per hearing': {k: round(float(val.sum(1).mean()), 1) for k, val in X.items()},
           'never lit share': {k: round(float((val.sum(0) == 0).float().mean()), 3) for k, val in X.items()}}
    held = np.isin(y, list(HELD))
    for part, m in (('shown', ~held), ('held', held)):
        if m.sum() > 50:
            mt = torch.as_tensor(m, device=DEV)
            out[part] = evaluate({k: val[mt] for k, val in X.items()}, y[m], v[m], d[m], n_tr=P['eval_ntr'],
                                 seed=P['seed'])
    # next syllable: the top's expectation for the tick after a syllable, against the class means of what came
    if BIGRAM is None:
        BIGRAM = bigram()
    # forgetting (designer 11-12): the same readout on training dialogues heard first and heard last
    # each stream is given the dialogues that stream heard (its own hippocampus), in voices set by probe_tag
    per = P['probe_every'] // B
    parts = (('Ex', 'sum'), ('ExT', 'top'), ('ExH', 'recall'))
    for nm, d_ in (('first', 0), ('last', ST.get('d0_last'))):
        if d_ is None:
            continue
        R2_ = run(0, 0, P['probe_tag'], False, False, record=True,
                  lists=[range(d_ + b * per, d_ + b * per + min(24, per)) for b in range(B)])
        out[f'next syllable, training dialogues heard {nm}'] = {p_: next_only(R2_, k_) for k_, p_ in parts}
        out[f'trend, training dialogues heard {nm}'] = trend(R2_)
        del R2_
    out['next syllable, never heard, by part'] = {p_: next_only(R_, k_) for k_, p_ in parts}
    out['trend, never heard'] = trend(R_)
    if USE_H:                                                     # is the key context-specific? the H codes of one
        Hc = R_['H']                                              # syllable at different places, against W's
        rr, ov = np.random.default_rng(7), dict(H=[], W=[])
        occ = {}
        for b in range(B):
            for t in range(lab.shape[1]):
                if lab[b, t] >= 0:
                    occ.setdefault(int(lab[b, t]), []).append((b, t))
        for c_, L_ in occ.items():
            if len(L_) < 2:
                continue
            for _ in range(min(10, len(L_))):
                i1, i2 = rr.choice(len(L_), 2, replace=False)
                (b1, t1), (b2, t2) = L_[i1], L_[i2]
                if dl[b1, t1] == dl[b2, t2]:
                    continue
                ov['H'].append(float((Hc[b1, t1] & Hc[b2, t2]).sum()) / H.k)
                ov['W'].append(float((R_['W'][b1, t1] & R_['W'][b2, t2]).sum()) / W.k)
        out['code overlap, same syllable, other dialogue'] = {k_: round(float(np.mean(v_)), 3) for k_, v_ in ov.items()}
    TG = R_['W' if P['top_target'] == 'W' else 'R1'].float()   # what the top expects
    Ex = R_['Ex'].float()
    items = [(b, t) for b in range(B) for t in range(lab.shape[1] - 1) if lab[b, t] >= 0 and lab[b, t + 1] >= 0]
    cnt = np.bincount(np.array([lab[b, t + 1] for b, t in items]))
    cands = [int(c_) for c_ in np.argsort(-cnt)[:300] if cnt[c_] > 0]
    pos = {c_: i for i, c_ in enumerate(cands)}
    lab_next = np.array([lab[b, t + 1] for b, t in items])
    M = torch.stack([TG[[b for (b, t), l_ in zip(items, lab_next) if l_ == c_],
                        [t + 1 for (b, t), l_ in zip(items, lab_next) if l_ == c_]].mean(0) for c_ in cands])
    M = M - M.mean(0, keepdim=True)                               # what all candidates share says nothing
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    usualE = ST['usualE'].cpu()
    r_pfc, r_big, r_frq = [], [], []
    for b, t in items:
        nx = int(lab[b, t + 1])
        if nx not in pos:
            continue
        e = Ex[b, t + 1] - usualE                                 # above each copy's own usual expectation
        sc = Mn @ (e / e.norm().clamp_min(1e-9))
        r_pfc.append(int((sc > sc[pos[nx]]).sum() + (sc == sc[pos[nx]]).sum() - 1))   # ties count against
        bc = BIGRAM.get(int(lab[b, t]), {})
        sb = np.array([bc.get(c_, 0) for c_ in cands], float) + 1e-6 * np.arange(len(cands))[::-1]
        r_big.append(int((sb > sb[pos[nx]]).sum()))
        r_frq.append(pos[nx])
    st = lambda r: dict(top1=round(float(np.mean(np.array(r) == 0)), 3),
                        top10=round(float(np.mean(np.array(r) < 10)), 3), mean_rank=round(float(np.mean(r)), 1))
    # the linear bound of the current code (Fable a29): a ridge map from R1 at t to W at t+1, fitted on the even
    # dialogues, read the same way on the odd ones. A weak top with a good bound = the rule; a low bound = the code.
    its = [(b, t) for b, t in items if int(lab[b, t + 1]) in pos]
    tr_ = [i for i, (b, t) in enumerate(its) if dl[b, t] % 2 == 0]
    te_ = [i for i, (b, t) in enumerate(its) if dl[b, t] % 2 == 1]
    Yr = torch.stack([TG[b, t + 1] for b, t in its]).to(DEV)
    bound = {}
    for nm, lam in (('W', 10.0), ('R1', 10.0)):
        Xr = torch.stack([R_[nm][b, t] for b, t in its]).float().to(DEV)
        Xa = Xr[tr_]
        A = Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=DEV)
        Wm = torch.linalg.solve(A, Xa.T @ Yr[tr_])
        Yh = Xr[te_] @ Wm
        Yh = Yh - (Xa @ Wm).mean(0, keepdim=True)
        sc = (Yh / Yh.norm(dim=1, keepdim=True).clamp_min(1e-9)) @ Mn.to(DEV).T
        tru = torch.as_tensor([pos[int(lab[its[i][0], its[i][1] + 1])] for i in te_], device=DEV)
        st_ = sc.gather(1, tru[:, None])
        rk = ((sc > st_).sum(1) + (sc == st_).sum(1) - 1).cpu().numpy()
        bound[f'from {nm}'] = st(rk)
        del A, Wm, Xr
    # the yardstick of this readout: the same ridge from the one-hot current syllable (what counting knows)
    oh = torch.zeros(len(its), len(cands) + 1, device=DEV)
    for i, (b, t) in enumerate(its):
        oh[i, pos.get(int(lab[b, t]), len(cands))] = 1.0
    A = oh[tr_].T @ oh[tr_] + 1.0 * torch.eye(oh.shape[1], device=DEV)
    Wm = torch.linalg.solve(A, oh[tr_].T @ Yr[tr_])
    Yh = oh[te_] @ Wm - (oh[tr_] @ Wm).mean(0, keepdim=True)
    sc = (Yh / Yh.norm(dim=1, keepdim=True).clamp_min(1e-9)) @ Mn.to(DEV).T
    tru = torch.as_tensor([pos[int(lab[its[i][0], its[i][1] + 1])] for i in te_], device=DEV)
    st_ = sc.gather(1, tru[:, None])
    bound['from the one-hot syllable'] = st(((sc > st_).sum(1) + (sc == st_).sum(1) - 1).cpu().numpy())
    out['next syllable'] = dict(top=st(r_pfc), bigram=st(r_big), frequency=st(r_frq), ridge_on_probe_half=bound,
                                n=len(r_pfc), candidates=len(cands))
    say('PROBE', tag, json.dumps(out))
    return out


def main():
    cnt = np.bincount(TRAIN.syl[:200000].astype(np.int64))
    top = [int(c_) for c_ in np.argsort(-cnt)[:60]]
    HELD.update(int(c_) for c_ in np.random.default_rng(P['seed'] + 31).choice(top, P['n_held'], replace=False))
    say('GROWN', json.dumps(dict(layers={ly.name: [ly.n, ly.k] for ly in (A1, W, C, R1, R2) + ((H,) if USE_H else ())}, ear=N_EAR,
                                 held=sorted(HELD), held_voice=P['held_voice'], P=P)))
    # calibration (Fable a28 #3): usual levels and winner scales settle with the rule off, before anything is judged
    cal = max(1, int(P['cal_ticks'] / 80))
    T = run(cal, 14000, 7600, False, P['p_see'], adapt=True)
    say('CALIBRATED', json.dumps(dict(ticks=T, win={ly.name: {s_: round(float(w_), 3) for s_, w_ in ly.win.items()}
                                                     for ly in (A1, W, C, R1, R2) + ((H,) if USE_H else ())})))
    if P['probes']:
        probe('birth')
    per = P['probe_every'] // B
    d0, done = 0, 0
    while done < P['n_d']:
        EV.clear()
        for r in range(int(P['repeat'])):                         # each block heard `repeat` times in a row
            ST['rep'] = r                                         # (designer: "尽量多听几遍就能稳定预测"); with
            T = run(per, d0, 7700 + (r if P['rep_fresh'] else 0), True, P['p_see'],   # rep_fresh in new voices
                    keymem='store' if r == 0 else 'compare')      # (Fable a33: real repetition is the same words)
        ST['rep'] = None
        KM.clear()
        ST['d0_last'] = d0
        if P['replay'] > 0 and d0 > 0:                            # old dialogues heard again between new ones
            rr = np.random.default_rng([P['seed'], 4411, d0])     # (the hippocampus's replay, as re-hearing)
            old = rr.integers(0, d0, (B, int(per * P['replay'])))
            run(0, 0, 7700, True, P['p_see'], lists=[list(map(int, o)) for o in old])
        d0 += per * B
        done += per * B
        n_t = max(EV.get('ticks', 1.0), 1.0)
        online = {}
        for r in range(int(P['repeat'])):                         # the share of the actual W code expected, by hearing
            n_ = EV.pop(f'on r{r} n', 0.0)
            online[f'hearing {r + 1}'] = {nm: round(EV.pop(f'on r{r} {nm}', 0.0) / max(n_, 1.0), 3)
                                          for nm in ('top', 'H', 'sum', 'usual')}
            kn = EV.pop(f'again n r{r}', 0.0)
            if kn > 0:                                            # share of hearing 1's code lit again, per layer
                online[f'hearing {r + 1}']['code again'] = {nm: round(EV.pop(f'again {nm} r{r}', 0.0) / kn, 3)
                                                            for nm in KEY_LAYERS()}
        say('ONLINE', done, json.dumps(online))
        say('LEARNED', done, json.dumps(dict(ticks=T, **{k: round(v_ if 'split' in k else v_ / n_t, 3)
                                                         for k, v_ in EV.items() if k != 'ticks'},
                                             W_EI=[float(W.E.mean()), float(W.I.mean())],
                                             C_EI=[float(C.E.mean()), float(C.I.mean())],
                                             top_EI=[float(TE.mean()), float(TI.mean())],
                                             busy_dead={ly.name: [round(float((ly.rate > P['split_hi'] * ly.k / ly.n).float().mean()), 4),
                                                                  round(float((ly.rate < P['dead_lo'] * ly.k / ly.n).float().mean()), 4)]
                                                        for ly in (W, C, R1)})))
        if P['probes']:
            probe(f'{done}')
        if P['save']:                                             # the trained brain, so later checks need no
            save(P['save'])                                       # retraining (designer 11-12)


def save(path):
    """every layer (birth wiring, learned links, usual levels), the top's links and state; the hippocampus's recall
    links only with save_hip (2 GB)"""
    lys = [A1, W, C, R1, R2] + ([H] if USE_H else [])
    st = {k: v for k, v in ST.items() if k not in ('Ex', 'ExT', 'ExH', 'hidx', 'hval')}
    torch.save(dict(P=P, layers={ly.name: {k: v for k, v in ly.__dict__.items() if k not in ('on', 'push')}
                                 for ly in lys},
                    TE=TE, TI=TI, ST=st, HELD=sorted(HELD), HT=HT if (USE_H and P.get('save_hip')) else None), path)
    say('SAVED', path)


def load(path):
    """the state saved by save() into this process's layers (built with the same P)"""
    d = torch.load(path, weights_only=False, map_location=DEV)
    for ly in [A1, W, C, R1, R2] + ([H] if USE_H else []):
        for k, v in d['layers'][ly.name].items():
            setattr(ly, k, v)
    TE.copy_(d['TE'])
    TI.copy_(d['TI'])
    ST.update(d['ST'])
    HELD.update(d['HELD'])
    if USE_H and d.get('HT') is not None:
        HT.copy_(d['HT'])
    return d


if __name__ == '__main__':
    main()
