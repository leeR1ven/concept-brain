# -*- coding: utf-8 -*-
"""ARCH_v1 integrated numpy toy (synthesizer).  One tick = all currents from t-1 states -> all parts fire ->
learning -> reserve / set point -> decay every 100 EXPERIENCED ticks.  Rule = brain/net.py Path.learn/decay exactly
(forward x reserve, concave, back term floor w0, per-target budget, gamma).

Parts (all kept parts of the real model are simulated, not stubbed):
  world   : voice_copy.py = world/voice.py + one flag (own-voice re-articulation onset)
  ear     : exact Net3D replica (8x6, zs 40/60/120/240, frac .4/.3/.2/.1, top-k), reserve U .5 tau 6, warm-up muting,
            LEARNED ear->ear (k 96, w0 U(0,.01), eta .03, g .999, budget 1)          [kept]
  mctx    : exact Net3D replica (8x4, levels 3, zs 20/40/80, frac .4/.3/.2) of the muscle-force map at t-1,
            learned internal (k 96 ...), learned ear->mctx (k32, .005, .01, B1) and mctx->ear (k32, .005, .01, B.5)
  hip     : mirrors of ear cortex + mctx cortex + motor M/D; chain 3.0 (no reserve); top 5 %; internal learned
            k 96, w0 U(0,.01), eta .5, gamma .7, budget 2 (plastic ALWAYS, also in tests)                  [kept]
  returns : hip -> relay (column span 3, need 3) -> ear cortex / mctx cortex +0.5                            [kept]
  F       : other-is-talking = 0.01 * energy of the ear INPUT window at t-1 (pre-cortical) - 10*CD >= 0.1
  F2, GO  : GO(t) = F(t-2) & !F(t-1) & !SPK(t-1) & !CD(t-1)  (offset pulse; no learned input)
  CD      : 4-unit delay line driven by GO | SPK  (own-voice veto into F and GO)
  T       : PFC working memory: top-K of [fixed random FEAT0(layer-0) input x reserve + 3.0 self-hold - set point + eps]
  P       : output gate: P = T(t-1) & GO(t-1)   (one tick)
  ring    : R1(t) = GO(t-1); boundary(t) = (R1(t-1) & !F(t-1)) | (SPK(t-1) & phase(t-1)==2)
  Y       : syllable programs (local recruitment), WTA at boundaries, Y = 1-tick pulse (learning target)
  Hp      : held program [B, B+2] -> D/I/M slaved (same tick), force ramp 1-2-3
  Hc      : late one-hot copy of the program, on [B+2, B+3] (learning source / context for the next boundary)
  END     : pulse at a learned stop; breath counter <= 12; floor theta
  learned content paths (gated: a target learns only when its release had external drive):
            P->Y  (eta .1/K_T, budget .5 per program = ~5 events), Hc->Y (eta .01, budget 2), Hc->END (eta .01, none)
  SLOT    : optional extra content currents into Y at boundaries (content specialist), capped
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path.insert(0, HERE)
from voice_v1copy import Voice, Stream  # noqa: E402

SOUND = 3
CORPUS = __import__('physics').corpus_path()


def load_corpus():
    pool = [json.loads(l) for l in open(CORPUS, encoding='utf-8')]
    pool = [c for c in pool if len(c) >= 2]
    chars = sorted({ch for c in pool for u in c for ch in u})
    return pool, chars


def pairs_of(dialogs):
    """(other, reply) pairs of a list of dialogues; a trailing unanswered line is dropped (protocol change)."""
    return [(c[i], c[i + 1]) for c in dialogs for i in range(0, len(c) - 1, 2)]


# ============================================================================== exact Net3D replica (numpy)
class Net3D:
    def __init__(self, H, W, levels=10, zs=(40, 60, 120, 240), frac=(.4, .3, .2, .1), alpha=.01, bias=.5,
                 spare=.25, seed=0):
        rng = np.random.default_rng(seed)
        self.H, self.W, self.levels, self.frac, self.zs = H, W, levels, frac, zs
        z0 = levels * 2
        self.dims = [(H, W, z0)] + [(H, W, z) for z in zs]
        self.idx, self.w, self.b, self.cap, self.start, alive = [], [], [], [], [], []
        tot = 0
        for l in range(1, len(self.dims)):
            prev, out = self.dims[l - 1], self.dims[l]
            x, y, z = np.meshgrid(np.arange(out[0]), np.arange(out[1]), np.arange(out[2]), indexing='ij')
            oc = np.stack([x, y, z], -1).reshape(-1, 3)
            n = len(oc); cap = n + int(round(n * spare))
            selfc = oc.copy(); selfc[:, 2] = np.round(oc[:, 2] * (prev[2] - 1) / max(out[2] - 1, 1)).astype(int)
            idx = np.zeros((cap, 4), np.int64); w = np.zeros((cap, 4), np.float32); b = np.zeros(cap, np.float32)
            idx[:n, 0] = np.ravel_multi_index(selfc.T, prev); w[:n, 0] = 1.0
            for j in range(1, 4):
                nb = selfc.copy()
                nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
                nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
                nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
                idx[:n, j] = np.ravel_multi_index(nb.T, prev)
                w[:n, j] = (rng.random(n) * .5 + .5) * alpha
            b[:n] = rng.random(n) * bias
            a = np.zeros(cap, bool); a[:n] = True
            self.idx.append(idx); self.w.append(w); self.b.append(b); alive.append(a)
            self.cap.append(cap); self.start.append(tot); tot += cap
        self.n = tot
        self.layer = np.concatenate([np.full(c, l) for l, c in enumerate(self.cap)])
        self.alive = np.concatenate(alive)
        self.kk = [int(round(self.frac[l] * self.alive[s0:s0 + c].sum())) for l, (s0, c) in
                   enumerate(zip(self.start, self.cap))]

    def encode(self, img):
        img = img.reshape(self.H, self.W, -1)
        up = img[..., None] > np.arange(self.levels)
        return np.stack([up, ~up], -1).reshape(-1)

    def respond(self, lines, extra=None):
        h = lines.astype(np.float32)
        out = np.zeros(self.n, bool)
        for l, (idx, w, b, s0, cap) in enumerate(zip(self.idx, self.w, self.b, self.start, self.cap)):
            s = (h[idx] * w).sum(1) + b
            al = self.alive[s0:s0 + cap]
            k = self.kk[l]
            if extra is not None:
                s = s + extra[s0:s0 + cap]
            s = np.where(al, s, -np.inf)
            on = np.zeros(cap, bool)
            on[np.argpartition(-s, k)[:k]] = True
            out[s0:s0 + cap] = on
            h = on.astype(np.float32)
        return out


class column_counter:
    """relay rule of brain.py neighbours(): each cortex slot counts active mirrors among itself and the slots of the
    same (x,y) column in its layer with |dz| <= span (born slots); spare slots count only themselves.
    (A class, not a closure, so that a trained model can be pickled.)"""

    def __init__(self, net: Net3D, span=3):
        self.net, self.span = net, span

    def __call__(self, A):
        net, span = self.net, self.span
        out = np.zeros(net.n, np.int32)
        for l in range(1, len(net.cap)):
            s0, cap = net.start[l], net.cap[l]
            Z = net.zs[l]; n = net.H * net.W * Z
            a = A[s0:s0 + n].reshape(net.H * net.W, Z).astype(np.int32)
            cs = np.concatenate([np.zeros((net.H * net.W, 1), np.int32), np.cumsum(a, axis=1)], axis=1)
            z = np.arange(Z); lo = np.maximum(z - span, 0); hi = np.minimum(z + span, Z - 1) + 1
            out[s0:s0 + n] = (cs[:, hi] - cs[:, lo]).ravel()
            out[s0 + n:s0 + cap] = A[s0 + n:s0 + cap]
        return out


# ============================================================================== learned paths (exact Path replica)
class SparsePath:
    """k random targets per source (drawn from dst_ids), w0 ~ U(0, w_init); stores w (as net.Path does);
    exact learn (forward eta*x/(1+(w-w0)/w_ref) on rising targets; back w <- max(w-eta, w0)) and decay
    (learned part x gamma, then per-target budget).  Cut edges point to a sentinel target n_dst."""

    def __init__(self, n_src, n_dst, src_ids, dst_ids, k, w_init, eta, gamma, budget, rng, w_ref=.2, back=True):
        self.to = np.full((n_src, k), n_dst, np.int64)
        self.w0 = np.zeros((n_src, k), np.float32)
        self.to[src_ids] = np.asarray(dst_ids)[rng.integers(0, len(dst_ids), (len(src_ids), k))]
        self.w0[src_ids] = (rng.random((len(src_ids), k)) * w_init).astype(np.float32)
        self.w = self.w0.copy()
        self.n_dst, self.eta, self.gamma, self.budget, self.w_ref, self.back = n_dst, eta, gamma, budget, w_ref, back
        self._ext = np.zeros(n_dst + 1, bool)

    @property
    def L(self):
        return self.w - self.w0

    def cut_src(self, mask):
        self.w0[mask] = 0; self.w[mask] = 0; self.to[mask] = self.n_dst

    def cut_dst(self, mask):
        m = np.r_[mask, False]
        hit = m[self.to]
        self.w0[hit] = 0; self.w[hit] = 0; self.to[hit] = self.n_dst

    def current(self, s, x=None):
        a = np.flatnonzero(s)
        if not len(a):
            return np.zeros(self.n_dst, np.float32)
        w = self.w[a]
        if x is not None:
            w = w * x[a, None]
        return np.bincount(self.to[a].ravel(), weights=w.ravel(), minlength=self.n_dst + 1)[:self.n_dst]

    def learn(self, s_prev, s_now, d_prev, d_now, x=None):
        """s_* / d_*: source / target states (already masked to neurons whose paths are not cut)."""
        ext = self._ext
        rising = d_now & ~d_prev
        rows = np.flatnonzero(s_prev)
        if len(rows) and rising.any():
            ext[:-1] = rising
            rr, kk = np.nonzero(ext[self.to[rows]])
            if len(rr):
                src = rows[rr]; w = self.w[src, kk]
                inc = self.eta / (1 + (w - self.w0[src, kk]) / self.w_ref)
                if x is not None:
                    inc = inc * x[src]
                self.w[src, kk] = w + inc
        if self.back and d_prev.any():
            rows = np.flatnonzero(s_now)
            if len(rows):
                ext[:-1] = d_prev
                rr, kk = np.nonzero(ext[self.to[rows]])
                if len(rr):
                    src = rows[rr]
                    self.w[src, kk] = np.maximum(self.w[src, kk] - self.eta, self.w0[src, kk])

    def decay(self):
        L = self.w - self.w0
        if self.gamma < 1:
            L *= self.gamma
        if self.budget > 0:
            tot = np.bincount(self.to.ravel(), weights=L.ravel(), minlength=self.n_dst + 1)
            sc = np.minimum(1.0, self.budget / np.maximum(tot, 1e-12)).astype(np.float32)
            sc[-1] = 1.0
            L *= sc[self.to]
        self.w = self.w0 + L


class DensePath:
    """dense src x dst learned table, w0 = 0 (decision paths into programs / END).  Exact rule; optional per-target
    third factor: a target learns (forward at its rise, back on the next tick) only if it had external drive."""

    def __init__(self, n_src, n_dst, eta, gamma, budget, w_ref=.2):
        self.L = np.zeros((n_src, n_dst), np.float32)
        self.eta, self.gamma, self.budget, self.w_ref = eta, gamma, budget, w_ref
        self.n_back = 0; self.n_fwd = 0

    def current(self, src_ids):
        if not len(src_ids):
            return np.zeros(self.L.shape[1], np.float32)
        return self.L[src_ids].sum(0)

    def learn(self, src_prev, src_now, rise_ids, held_prev_ids):
        """src_prev / src_now: active source ids at t-1 / t; rise_ids: gated targets rising at t;
        held_prev_ids: gated targets on at t-1 (back term at t for sources on at t)."""
        if len(rise_ids) and len(src_prev):
            ix = np.ix_(src_prev, rise_ids)
            sub = self.L[ix]
            self.L[ix] = sub + self.eta / (1 + sub / self.w_ref)
            self.n_fwd += sub.size
        if len(held_prev_ids) and len(src_now):
            ix = np.ix_(src_now, held_prev_ids)
            sub = self.L[ix]
            self.n_back += int((sub > 0).sum())
            self.L[ix] = np.maximum(sub - self.eta, 0)

    def decay(self):
        if self.gamma < 1:
            self.L *= self.gamma
        if self.budget > 0:
            tot = self.L.sum(0)
            self.L *= np.minimum(1.0, self.budget / np.maximum(tot, 1e-12)).astype(np.float32)[None, :]


# ============================================================================== configuration
@dataclass
class Cfg:
    seed: int = 0
    # turn circuit
    f_w: float = 0.01            # F: weight per "up" input line (= loudness unit) of the ear input window at t-1
    f_thr: float = 0.1           # F threshold (any sound: silent window = 0, 2 slices of speech >= 22 units)
    cd_len: int = 4              # corollary-discharge delay line length
    # T / P (PFC working memory and its output gate)
    NT: int = 4000
    KT: int = 200
    t_fanin: int = 30
    t_hold: float = 3.0
    t_sp_beta: float = 0.3
    t_sp_tau: float = 1000.0
    # programs
    NP: int = 2048
    cue_w: float = 10.0          # teacher cue per matching muscle (exact program 40, any other <= 30)
    end_push: float = 100.0
    theta: float = 0.02          # floor: best program score below it -> silence / stop
    rho: float = 1.0             # 'max' rule: stop iff s_END >= rho * s_best
    end_rule: str = 'max'        # 'max' | 'sum': stop iff s_END >= (1-Ub)^n * sum_c max(learned_c, 0) + cue drive
    Ub: float = 0.1              # breath discount of the 'sum' rule
    rep: float = 0.0             # within-reply penalty on programs already released in this turn (use-depression)
    breath: int = 12
    eta_py_event: float = 0.1    # P->Y: total increment per event (eta = this / K_T)
    b_py: float = 0.5            # P->Y per-program budget (~5 events)
    eta_cy: float = 0.01
    b_cy: float = 2.0
    eta_ce: float = 0.01
    wref_py: float = 0.2
    wref_cy: float = 0.2
    wref_ce: float = 0.2
    eta_py2_event: float = 0.0   # optional SLOW first-syllable path from the same gate P (0 = off); CONTENT.md §4
    b_py2: float = 1.5
    wref_py2: float = 0.6
    g_py2: float = 1.0           # no forgetting (budget-bounded, interference only)
    g_dec: float = 0.999         # decision paths gamma per 100 experienced ticks
    gate: bool = True            # third factor: learn only externally driven releases
    # kept realistic parts
    ear_learn: bool = True
    hip_on: bool = True
    returns: bool = True
    ret_hip: float = 0.5
    mctx_on: bool = True
    decay_T: int = 100
    exp_clock: bool = True       # decay clock counts experienced ticks only (sound in the window or speaking)
    slot: object = None          # content slot plug-in: object with .score(brain, com, first) at boundaries,
    #                              .learn(brain, Y_new, gate_now) and .tick(brain, ev) every tick (or None)


# ============================================================================== the brain
class Brain:
    M_N, LV = 32, 3

    def __init__(self, voice: Voice, cfg: Cfg):
        self.v, self.c = voice, cfg
        c = cfg
        rng = np.random.default_rng(c.seed)
        self.rng = rng
        # ---------------- ear
        self.ear = E = Net3D(voice.F, voice.T, seed=c.seed)
        ne = E.n
        self.cand_e = np.flatnonzero(E.layer >= 1)
        self.e_cx = (E.layer >= 1) & E.alive
        self.e_muted = np.zeros(ne, bool)
        self.e = np.zeros(ne, bool); self.ex = np.ones(ne, np.float32)
        self.L0 = np.flatnonzero(E.layer == 0)          # first 3D layer (never split, no extra current)
        self.e_in = SparsePath(ne, ne, self.cand_e, self.cand_e, 96, 0.01, 0.03, 0.999, 1.0, rng)
        # ---------------- motor cortex (proprioception)
        self.mH, self.mW = 8, 4
        self.mc = Mn = Net3D(self.mH, self.mW, levels=3, zs=(20, 40, 80), frac=(.4, .3, .2), seed=c.seed + 5)
        nc = Mn.n
        self.cand_c = np.flatnonzero(Mn.layer >= 1)
        self.m_cx = (Mn.layer >= 1) & Mn.alive
        self.m_muted = np.zeros(nc, bool)
        self.m = np.zeros(nc, bool); self.mx = np.ones(nc, np.float32)
        self.m_in = SparsePath(nc, nc, self.cand_c, self.cand_c, 96, 0.01, 0.03, 0.999, 1.0, rng)
        self.e2m = SparsePath(ne, nc, self.cand_e, self.cand_c, 32, 0.005, 0.01, 0.999, 1.0, rng)
        self.m2e = SparsePath(nc, ne, self.cand_c, self.cand_e, 32, 0.005, 0.01, 0.999, 0.5, rng)
        # ---------------- motor neurons (for chains): M 96 (muscle x level), D 32 ; I has no chain (tonic)
        self.motor = np.zeros(self.M_N * self.LV + self.M_N, bool)       # [M (96) | D (32)]
        # ---------------- hippocampus
        self.nh = nh = ne + nc + len(self.motor)
        hid = np.r_[self.cand_e, ne + self.cand_c, ne + nc + np.arange(len(self.motor))]
        self.hip_in = SparsePath(nh, nh, hid, hid, 96, 0.01, 0.5, 0.7, 2.0, rng)
        self.h = np.zeros(nh, bool)
        self.h_alive = np.zeros(nh, bool)
        self.cnt_e = column_counter(E); self.cnt_m = column_counter(Mn)
        self.relay_e = np.zeros(ne, bool); self.relay_m = np.zeros(nc, bool)
        # ---------------- T / P
        self.FEAT0 = None
        self.T = np.zeros(c.NT, bool); self.u = np.full(c.NT, c.KT / c.NT); self.P = np.zeros(c.NT, bool)
        self.t_tie = (rng.random(c.NT) * 0.01).astype(np.float32)
        # ---------------- turn units
        self.F = False; self.F2 = False; self.GO = False; self.R1 = False; self.R2 = False
        self.CD = np.zeros(c.cd_len, bool)
        self.E_prev = 0.0
        # ---------------- programs
        self.prog = np.zeros((c.NP, self.M_N), bool)
        self.committed = np.zeros(c.NP, bool)
        self.n_prog = 0
        self.PY = DensePath(c.NT, c.NP, c.eta_py_event / c.KT, c.g_dec, c.b_py, w_ref=c.wref_py)
        self.PY2 = DensePath(c.NT, c.NP, c.eta_py2_event / c.KT, c.g_py2, c.b_py2, w_ref=c.wref_py2) \
            if c.eta_py2_event > 0 else None
        self.CY = DensePath(c.NP, c.NP, c.eta_cy, c.g_dec, c.b_cy, w_ref=c.wref_cy)
        self.CE = DensePath(c.NP, 1, c.eta_ce, c.g_dec, 0.0, w_ref=c.wref_ce)
        self.Hp = -1; self.phase = 0; self.SPK = False; self.nsyl = 0
        self.said_turn = np.zeros(c.NP, np.float32)
        self.Yh = [-1, -1, -1, -1]      # Y pulse ids at t-1, t-2, t-3, t-4
        self.Y = -1; self.END = False
        self.Yext_prev = False; self.ENDext_prev = False
        self.force = np.zeros(self.M_N, int)
        self.learning = True
        self.t = 0; self.t_exp = 0
        self.slot_cap = 7.0

    # ------------------------------------------------------------------ helpers
    @property
    def e_live(self):
        return self.e_cx & ~self.e_muted

    @property
    def m_live(self):
        return self.m_cx & ~self.m_muted

    def Hc_at(self, lag):
        """late one-hot program copy Hc(t) = {Y(t-2), Y(t-3)} (delay line inside the program).  Called at the start
        of step t (Yh = [Y(t-1), Y(t-2), Y(t-3), Y(t-4)]): lag 0 -> Hc(t), lag 1 -> Hc(t-1)."""
        return np.array(sorted({y for y in self.Yh[1 + lag:3 + lag] if y >= 0}), dtype=np.int64)

    def sync_hip_alive(self):
        ne, nc = self.ear.n, self.mc.n
        self.h_alive[:ne] = self.e_live
        self.h_alive[ne:ne + nc] = self.m_live
        self.h_alive[ne + nc:] = True
        self.kh = int(round(0.05 * self.h_alive.sum()))

    def mute(self, ear_ids_mask, mc_ids_mask):
        """featureless neurons leave the cortex: no chain, no learned in/out paths, no relay."""
        self.e_muted |= ear_ids_mask; self.m_muted |= mc_ids_mask
        for p, sm, dm in ((self.e_in, self.e_muted, self.e_muted), (self.m_in, self.m_muted, self.m_muted),
                          (self.e2m, self.e_muted, self.m_muted), (self.m2e, self.m_muted, self.e_muted)):
            p.cut_src(sm); p.cut_dst(dm)
        self.sync_hip_alive()

    def muscle_lines(self):
        img = np.zeros(self.mH * self.mW, int)
        img[:self.M_N] = self.force
        return self.mc.encode(img.reshape(self.mH, self.mW))

    # ------------------------------------------------------------------ one tick
    def step(self, lines, E_now, cue=None, endpush=False):
        """lines: ear input lines at t (computed from the window at t); E_now: window energy at t (read by F at t+1);
        cue: teacher muscle set (frozenset of 4) present at this tick, or None; endpush: teacher END push.
        Returns event dict."""
        c = self.c
        ne, nc = self.ear.n, self.mc.n
        ev = {}
        # ====================== currents from t-1 states
        e_live = self.e_live; m_live = self.m_live
        e_src = self.e & e_live                               # non-muted cortex neurons that send
        m_src = self.m & m_live
        ein = np.zeros(ne, np.float32)
        if c.ear_learn:
            ein += self.e_in.current(e_src, self.ex)
        if c.mctx_on:
            ein += self.m2e.current(m_src, self.mx)
        if c.returns and c.hip_on:
            ein += c.ret_hip * self.relay_e
        ein = np.where(e_live, ein, 0).astype(np.float32)   # SenseArea.fire: extra current only into (non-muted) cortex
        if c.mctx_on:
            mcur = self.m_in.current(m_src, self.mx) + self.e2m.current(e_src, self.ex)
            if c.returns and c.hip_on:
                mcur += c.ret_hip * self.relay_m
            mcur = np.where(m_live, mcur, 0).astype(np.float32)
        if c.hip_on:
            hcur = np.zeros(self.nh, np.float32)
            hcur[:ne] = 3.0 * e_src
            hcur[ne:ne + nc] = 3.0 * m_src
            hcur[ne + nc:] = 3.0 * self.motor
            hcur += self.hip_in.current(self.h)
            if c.returns:
                relay_e_new = (self.cnt_e(self.h[:ne]) >= 3) & e_live
                relay_m_new = (self.cnt_m(self.h[ne:ne + nc]) >= 3) & m_live
        ncd = int(self.CD.sum())
        F_new = (c.f_w * self.E_prev - 10.0 * ncd) >= c.f_thr
        F2_new = self.F
        GO_new = (1.0 * self.F2 - 2.0 * self.F - 2.0 * self.SPK - 2.0 * ncd) >= 1.0
        CD_new = np.r_[self.GO or self.SPK, self.CD[:-1]]
        # T (PFC working memory) from ear layer 0 (FEAT0) x reserve at t-1
        a = (self.e[self.FEAT0] * self.ex[self.FEAT0]).astype(np.float32)
        drive = a[self.T_in].sum(1)
        tsc = drive + c.t_hold * self.T - c.t_sp_beta * np.maximum(self.u / (c.KT / c.NT) - 1.0, 0) + self.t_tie
        T_new = np.zeros(c.NT, bool); T_new[np.argpartition(-tsc, c.KT)[:c.KT]] = True
        P_new = self.T & self.GO                              # output gate: one tick after GO
        R1_new = self.GO
        R2_new = self.GO or (self.SPK and self.phase == 1)     # pre-boundary tick: content read-out clock
        bnd = (self.R1 and not self.F) or (self.SPK and self.phase == 2)
        P_ids_prev = np.flatnonzero(self.P)
        Hc_prev = self.Hc_at(1)                               # copy state at t-1
        # ====================== all parts fire (state t)
        e_new = self.ear.respond(lines, ein) & self.ear.alive
        if c.mctx_on:
            m_new = self.mc.respond(self.muscle_lines(), mcur) & self.mc.alive
        else:
            m_new = self.m
        if c.hip_on:
            sh = np.where(self.h_alive, hcur, -np.inf)
            h_new = np.zeros(self.nh, bool); h_new[np.argpartition(-sh, self.kh)[:self.kh]] = True
        # ---- motor: boundary selection (same-tick teacher cue), hold, D/I/M slaved
        Y_new, END_new, Yext, ENDext = -1, False, False, False
        first = bnd and not self.SPK
        if bnd:
            ev['bnd'] = True
            if first:
                self.nsyl = 0
            com = np.flatnonzero(self.committed)
            sc = self.PY.current(P_ids_prev)[com] + self.CY.current(Hc_prev)[com] if len(com) else np.zeros(0)
            if self.PY2 is not None and len(com):
                sc = sc + self.PY2.current(P_ids_prev)[com]
            if c.slot is not None and len(com):
                add = np.minimum(c.slot.score(self, com, first), self.slot_cap)
                sc = sc + add
            s_end = float(self.CE.current(Hc_prev)[0]) + (c.end_push if endpush else 0.0)
            cue_ov = None
            if cue is not None:
                cv = np.zeros(self.M_N, bool); cv[list(cue)] = True
                ov = (self.prog[com] & cv).sum(1) if len(com) else np.zeros(0, int)
                if not (ov == 4).any():                       # local recruitment: no program reached full cue drive
                    if self.n_prog < c.NP:
                        y = self.n_prog; self.n_prog += 1
                        self.prog[y] = cv; self.committed[y] = True
                        com = np.r_[com, y]
                        sc = np.r_[sc, 0.0 + (min(c.slot.score(self, np.array([y]), first)[0], self.slot_cap)
                                              if c.slot is not None else 0.0)]
                        ov = np.r_[ov, 4]
                        ev['recruit'] = y
                    else:
                        ev['pool_full'] = True
                sc = sc + c.cue_w * ov
                cue_ov = ov
            sc_learn = sc if cue is None else sc - c.cue_w * cue_ov
            if c.rep > 0 and not first and len(com):
                sc = sc - c.rep * self.said_turn[com]
            best = int(np.argmax(sc)) if len(com) else -1
            s_best = float(sc[best]) if best >= 0 else 0.0
            ev['s_best'] = s_best; ev['s_end'] = s_end
            ev['s_best_learned'] = float(sc_learn[best]) if best >= 0 else 0.0
            if best >= 0 and len(com) > 1:
                ev['margin'] = s_best - float(np.partition(sc, -2)[-2])
            if c.end_rule == 'sum':
                comp = (1 - c.Ub) ** self.nsyl * float(np.maximum(sc_learn, 0).sum()) + \
                    (c.cue_w * float(cue_ov.max()) if cue is not None and len(cue_ov) else 0.0)
            else:
                comp = c.rho * s_best
            if not first and (endpush or (s_end > 0 and s_end >= comp)):     # teacher END = clamp (forced stop)
                ev['stop'] = 'END'; END_new = True; ENDext = bool(endpush)
            elif not first and self.nsyl >= c.breath:
                ev['stop'] = 'breath'
            elif best < 0 or s_best < c.theta:
                ev['stop'] = 'floor' if not first else 'silent'
            else:
                Y_new = int(com[best]); Yext = cue is not None and bool(cue_ov[best] == 4)
                self.Hp = Y_new; self.phase = 0; self.nsyl += 1
                if first:
                    self.said_turn[:] = 0
                self.said_turn[Y_new] = 1
                ev['rel'] = Y_new; ev['first'] = first
            if 'stop' in ev:
                self.Hp = -1; self.phase = 0
        elif self.SPK:
            self.phase += 1
        SPK_new = self.Hp >= 0
        D = np.zeros(self.M_N, bool)
        if SPK_new:
            D = self.prog[self.Hp].copy()
        # I_m = [1.2 - 2 D_m >= 1] = not D ; M_{m,l} = D_m & l <= phase   (same tick)
        Mlv = D[:, None] & (np.arange(self.LV)[None, :] <= self.phase)
        force = Mlv.sum(1)
        # ====================== learning (prev = t-1, now = t)
        if self.learning:
            if c.ear_learn:
                self.e_in.learn(e_src, e_new & e_live, e_src, e_new & e_live, self.ex)
            if c.mctx_on:
                self.m_in.learn(m_src, m_new & m_live, m_src, m_new & m_live, self.mx)
                self.e2m.learn(e_src, e_new & e_live, m_src, m_new & m_live, self.ex)
                self.m2e.learn(m_src, m_new & m_live, e_src, e_new & e_live, self.mx)
            if c.hip_on:
                self.hip_in.learn(self.h, h_new, self.h, h_new)
            # decision paths: targets are 1-tick pulses; third factor = the target's own external drive
            g_now = (not c.gate) or Yext
            g_prev = (not c.gate) or self.Yext_prev
            rise = np.array([Y_new], np.int64) if (Y_new >= 0 and g_now) else np.zeros(0, np.int64)
            held = np.array([self.Y], np.int64) if (self.Y >= 0 and g_prev) else np.zeros(0, np.int64)
            Hc_now = self.Hc_at(0)                            # copy state at t
            self.PY.learn(P_ids_prev, np.flatnonzero(P_new), rise, held)
            if self.PY2 is not None:
                self.PY2.learn(P_ids_prev, np.flatnonzero(P_new), rise, held)
            self.CY.learn(Hc_prev, Hc_now, rise, held)
            ge_now = (not c.gate) or ENDext
            ge_prev = (not c.gate) or self.ENDext_prev
            self.CE.learn(Hc_prev, Hc_now, np.array([0]) if (END_new and ge_now) else np.zeros(0, np.int64),
                          np.array([0]) if (self.END and ge_prev) else np.zeros(0, np.int64))
            self.u += (T_new - self.u) / c.t_sp_tau
            if c.slot is not None:
                c.slot.learn(self, Y_new, g_now, self.Y, g_prev)
        # ====================== commit
        self.e_prev_state = self.e
        old_e, old_m = self.e, self.m
        self.e = e_new
        self.ex = np.where(old_e, self.ex * 0.5, self.ex); self.ex += (1 - self.ex) / 6.0
        if c.mctx_on:
            self.m = m_new
            self.mx = np.where(old_m, self.mx * 0.5, self.mx); self.mx += (1 - self.mx) / 6.0
        if c.hip_on:
            self.h = h_new
            if c.returns:
                self.relay_e, self.relay_m = relay_e_new, relay_m_new
        self.F, self.F2, self.GO, self.R1, self.CD = bool(F_new), bool(F2_new), bool(GO_new), bool(R1_new), CD_new
        self.R2 = bool(R2_new)
        self.T, self.P = T_new, P_new
        self.Yh = [Y_new, self.Yh[0], self.Yh[1], self.Yh[2]]
        self.Y, self.END = Y_new, END_new
        self.Yext_prev, self.ENDext_prev = Yext, ENDext
        self.SPK = SPK_new
        self.motor[:] = False
        self.motor[:self.M_N * self.LV] = Mlv.ravel()
        self.motor[self.M_N * self.LV:] = D
        self.force = force
        self.E_prev = E_now
        self.t += 1
        experienced = (E_now > 0) or SPK_new
        if self.learning:
            if (not c.exp_clock) or experienced:
                self.t_exp += 1
                if self.t_exp % c.decay_T == 0:
                    self.decay()
        ev.update(F=self.F, GO=self.GO, CD=int(self.CD.sum()), P=int(P_new.sum()), SPK=SPK_new, phase=self.phase,
                  force=force, E=E_now, R2=self.R2)
        if c.slot is not None:
            c.slot.tick(self, ev)
        return ev

    def decay(self):
        c = self.c
        for p in ([self.e_in] if c.ear_learn else []) + ([self.m_in, self.e2m, self.m2e] if c.mctx_on else []) + \
                ([self.hip_in] if c.hip_on else []) + [self.PY, self.CY, self.CE] + ([self.PY2] if self.PY2 is not None else []):
            p.decay()

    # ------------------------------------------------------------------ setup
    def finish_warmup(self, usage_e, usage_m, n):
        ue = usage_e / n; um = usage_m / n
        self.mute(self.e_cx & (ue > 0.6), self.m_cx & (um > 0.6))
        L0 = self.L0
        self.FEAT0 = L0[self.ear.alive[L0] & (ue[L0] <= 0.6)]
        rng = np.random.default_rng(self.c.seed + 77)
        self.T_in = rng.integers(0, len(self.FEAT0), (self.c.NT, self.c.t_fanin))
        return int((self.e_cx & (ue > 0.6)).sum()), int((self.m_cx & (um > 0.6)).sum()), len(self.FEAT0)


# ============================================================================== body + harness
class Talker:
    def __init__(self, chars, cfg: Cfg):
        self.v = Voice(chars, seed=cfg.seed)
        self.b = Brain(self.v, cfg)
        self.stream = Stream(self.v, step=self.v.T // SOUND)
        self.saying = None; self.onset = False
        self.prev_force = np.zeros(32, int)
        self.need = {c: frozenset(self.v.need[c]) for c in chars}
        self.set2c = {s: c for c, s in self.need.items()}
        self.log = []
        self.keep_log = False

    def tick(self, other=None, new_other=False, cue=None, endpush=False):
        heard = self.stream.tick(other, self.saying, new_other, self.onset)
        E_now = float(heard.sum())
        lines = self.b.ear.encode(heard)
        ev = self.b.step(lines, E_now, cue=cue, endpush=endpush)
        f = ev['force']
        said = self.v.mouth({f"m{i}": int(x) for i, x in enumerate(f)})
        act, pact = f > 0, self.prev_force > 0
        if said is None:
            onset = False
        elif not pact.any() or not np.array_equal(act, pact):
            onset = True                                   # a new muscle set starts
        else:
            onset = bool(f[act].max() < self.prev_force[act].max())   # same set re-articulated (force restarts)
        self.prev_force = f.copy()
        self.saying, self.onset = said, bool(onset)
        ev['said'] = said; ev['onset'] = bool(onset); ev['other'] = other is not None
        if self.keep_log:
            self.log.append(ev)
        return ev

    # ---------------------------------------------------------------- protocol pieces
    def listen(self, line):
        evs = []
        for ch in line:
            for k in range(SOUND):
                evs.append(self.tick(other=ch, new_other=(k == 0)))
        return evs

    def silence(self, n):
        return [self.tick() for _ in range(n)]

    def teach(self, reply, max_ticks=80):
        """teacher: cue the next syllable every tick until the model's own boundary takes it; after the last syllable,
        push END until the model's boundary takes it.  The teacher never pushes GO."""
        evs = []; i = 0
        for _ in range(max_ticks):
            cue = self.need[reply[i]] if i < len(reply) else None
            ev = self.tick(cue=cue, endpush=(i >= len(reply)))
            evs.append(ev)
            if 'rel' in ev:
                i += 1
            if 'stop' in ev and i >= len(reply):
                break
        return evs

    def free(self, n_after_stop=12, max_ticks=60):
        """no teacher: run until the model's turn has ended (+ n_after_stop ticks) or max_ticks."""
        evs = []; stop_at = None
        for k in range(max_ticks):
            ev = self.tick(); evs.append(ev)
            if 'stop' in ev and stop_at is None:
                stop_at = k
            if stop_at is not None and k >= stop_at + n_after_stop:
                break
        return evs

    def warmup(self, n=60, seed=1, pause=4):
        b = self.b
        b.learning = False
        rng = np.random.default_rng(seed)
        ue = np.zeros(b.ear.n); um = np.zeros(b.mc.n); k = 0
        # FEAT0 unknown before warm-up: T runs on a dummy input during warm-up
        b.FEAT0 = b.L0[:1]; b.T_in = np.zeros((b.c.NT, b.c.t_fanin), int)
        b.sync_hip_alive()
        for _ in range(n):
            ch = self.v.chars[rng.integers(len(self.v.chars))]
            for e in self.listen(ch) + self.silence(pause):
                ue += b.e; um += b.m; k += 1
        out = b.finish_warmup(ue, um, k)
        self.silence(10)
        b.learning = True
        return out


def said_text(evs):
    """the syllables actually produced: one char per mouth onset (re-articulated repeats count)."""
    return ''.join(e['said'] for e in evs if e.get('onset') and e['said'] is not None)


def clone(talker):
    return copy.deepcopy(talker)
