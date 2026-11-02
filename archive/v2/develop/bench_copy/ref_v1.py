# -*- coding: utf-8 -*-
"""The v1-equivalent reference agent ("the bar"): ARCH_v1 minus the parts that reach no decision.

  body.py (F/GO/CD, beat, programs + WTA, END, D/I/M, counter)   + frontend.Ear0 / FEAT0 (exact layer 0)
  + T (fixed expansion of FEAT0, self-hold) and its gate P = T & GO
  + learned decision paths with the exact rule of brain/net.py (DensePath copied from archv1.py):
      P->Y fast (first syllable), optional P->Y slow, Hc->Y (next syllable), Hc->END (stop),
      gated by external drive (a target learns only on releases the teacher drove)
  + the echo (copy) route in the content SLOT (echo_slot_copy.EchoSlot = arch/synth/echo_slot.py): auditory mirror Z of
    each program (learned from the own voice), latch of heard chars, replay pulse at R2, learned diagonal E->Y.
Left out (PROVED in the reviews to reach no decision): ear cortex + its recurrence, motor cortex, hippocampus, relays.
With cfg='mem' (ARCH_v1's adopted default, res_mem_s0) or 'base' (its base-route control) the decisions are those of
archv1.Brain with the same Cfg (selftest.py checks tick-identical mouth output against archv1_copy.py).

    agent = RefV1(world, cfg='mem', seed=0)     # world: for the birth warm-up windows (innate calibration only)
    World('B', tutor='force').run(agent, 2000)
"""
from __future__ import annotations

import numpy as np

from body import Body
from echo_slot_copy import EchoSlot
from frontend import Ear0, WorkingMemoryT, feat0, feat0_v1, v1_tie

CFGS = {
    # ARCH_v1's base-route control (res_base_s*): no slot, no slow path, 'max' stop rule, learnability scales
    'base': dict(cue_w=10.0, theta=0.02, end_rule='max', rho=1.0, Ub=0.1, rep=0.0, eta_py_event=0.1, b_py=0.5,
                 wref_py=0.2, eta_py2_event=0.0, b_py2=1.5, wref_py2=0.6, g_py2=1.0, eta_cy=0.01, b_cy=2.0,
                 wref_cy=0.2, eta_ce=0.01, wref_ce=0.2, g_dec=0.999, slot=None),
    # ARCH_v1's adopted default (res_mem_s0): echo slot + fast/slow memory, summed-drive stop, used flag, cue 12
    'mem': dict(cue_w=12.0, theta=0.002, end_rule='sum', rho=1.0, Ub=0.1, rep=0.05, eta_py_event=0.1, b_py=0.5,
                wref_py=0.2, eta_py2_event=0.06, b_py2=1.5, wref_py2=0.6, g_py2=1.0, eta_cy=0.001, b_cy=0.2,
                wref_cy=0.02, eta_ce=0.001, wref_ce=0.02, g_dec=0.999,
                slot=dict(eta_e=0.003, B_e=0.15, w_ref=0.02, mult=1.0)),
}


class DensePath:
    """archv1.DensePath (copied): dense src x dst table, w0 = 0, forward eta/(1+L/w_ref) at the target's rise for
    sources on at t-1, back L <- max(L - eta, 0) for sources on at t when the target was on at t-1; decay = gamma then
    per-target budget."""

    def __init__(self, n_src, n_dst, eta, gamma, budget, w_ref=.2):
        self.L = np.zeros((n_src, n_dst), np.float32)
        self.eta, self.gamma, self.budget, self.w_ref = eta, gamma, budget, w_ref
        self.n_back = 0; self.n_fwd = 0

    def current(self, src_ids):
        if not len(src_ids):
            return np.zeros(self.L.shape[1], np.float32)
        return self.L[src_ids].sum(0)

    def learn(self, src_prev, src_now, rise_ids, held_prev_ids):
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

    def decay(self, n_used=None):
        """n_used: only targets [0, n_used) can be non-zero (programs are committed in order) -> same result, faster."""
        L = self.L if n_used is None else self.L[:, :n_used]
        if self.gamma < 1:
            L *= self.gamma
        if self.budget > 0:
            tot = L.sum(0)
            L *= np.minimum(1.0, self.budget / np.maximum(tot, 1e-12)).astype(np.float32)[None, :]


class _C:                                       # the attribute names EchoSlot reads from its brain (b.c.decay_T)
    decay_T = 100


class RefV1:
    """bench agent.  learning=True always (the decision paths stay closed without a teacher: gate=True)."""

    def __init__(self, world=None, cfg='mem', seed=0, NT=4000, KT=200, NP=2048, gate=True, exact_v1=True,
                 warm_windows=None, **over):
        c = dict(CFGS[cfg] if isinstance(cfg, str) else cfg); c.update(over)
        self.cfg = c
        self.gate = gate
        self.body = Body(n_prog=NP, cue_w=c['cue_w'], theta=c['theta'], rho=c['rho'], end_rule=c['end_rule'],
                         Ub=c['Ub'], rep=c['rep'], recruit='cue', seed=seed)
        self.ear = Ear0(seed)
        wins = warm_windows if warm_windows is not None else world.warmup_windows()
        self.FEAT0 = feat0_v1(self.ear, wins) if exact_v1 else feat0(self.ear, wins)
        self.Tm = WorkingMemoryT(len(self.FEAT0), NT, KT, seed=seed, tie=v1_tie(seed, NT) if exact_v1 else None)
        # T after archv1's warm-up = top-K of the tie-breaker (uniform dummy drive, hold); ear state = after the
        # warm-up chars + 10 silent ticks (as Talker.warmup)
        self.Tm.T[np.argpartition(-self.Tm.tie.astype(np.float64), KT)[:KT]] = True
        self.ear.reset()
        silent = np.zeros_like(np.asarray(wins[0]))
        for wdw in list(wins) + [silent] * 10:
            self.ear.step(wdw)
        self.P = np.zeros(NT, bool)
        self.KT = KT
        self.PY = DensePath(NT, NP, c['eta_py_event'] / KT, c['g_dec'], c['b_py'], w_ref=c['wref_py'])
        self.PY2 = DensePath(NT, NP, c['eta_py2_event'] / KT, c['g_py2'], c['b_py2'], w_ref=c['wref_py2']) \
            if c['eta_py2_event'] > 0 else None
        self.CY = DensePath(NP, NP, c['eta_cy'], c['g_dec'], c['b_cy'], w_ref=c['wref_cy'])
        self.CE = DensePath(NP, 1, c['eta_ce'], c['g_dec'], 0.0, w_ref=c['wref_ce'])
        self.slot = EchoSlot(NP, **c['slot']) if c.get('slot') else None
        self.slot_cap = 7.0
        self.learning = True
        self.t = 0; self.t_exp = 0
        self.c = _C()
        self.tick_event = None
        self.n_back_decision = 0

    # ---------------- attributes the echo slot reads (archv1.Brain names)
    @property
    def e(self):
        return self.ear.s

    @property
    def committed(self):
        return self.body.committed

    @property
    def Y(self):
        return self.body.Y

    @property
    def F(self):
        return self.body.F

    @property
    def F2(self):
        return self.body.F2

    @property
    def R2(self):
        return self.body.R2

    # ---------------- boundary scores (archv1 Brain.step)
    def _scores(self, body, com, first):
        P_prev = self._P_prev_ids
        sc = self.PY.current(P_prev)[com] + self.CY.current(body._hc_at(1))[com]
        if self.PY2 is not None:
            sc = sc + self.PY2.current(P_prev)[com]
        if self.slot is not None:
            sc = sc + np.minimum(self.slot.score(self, com, first), self.slot_cap)
        s_end = float(self.CE.current(body._hc_at(1))[0])
        return sc, s_end

    def step(self, obs):
        win = obs['ear']
        E_now = float(win.sum())
        cue = obs.get('cue')
        cue_set, endpush = None, False
        if cue is not None:
            nz = np.flatnonzero(np.asarray(cue) > 0)
            if len(nz):
                cue_set = frozenset(int(m) for m in nz)
            else:
                endpush = True
        b = self.body
        f = self.FEAT0
        T_new = self.Tm.step(self.ear.s[f], self.ear.x[f])          # T from FEAT0 x reserve at t-1
        P_new = self.Tm.T & b.GO                                     # P = T(t-1) & GO(t-1)
        self._P_prev_ids = np.flatnonzero(self.P)
        self.ear.step(win)                                           # layer 0 at t
        force = b.step(E_now, scorer=self._scores, cue=cue_set, endpush=endpush)
        if self.learning:
            g_now = (not self.gate) or b.Yext
            g_prev = (not self.gate) or b.Yext_prev
            rise = np.array([b.Y], np.int64) if (b.Y >= 0 and g_now) else np.zeros(0, np.int64)
            held = np.array([b.Y_prev], np.int64) if (b.Y_prev >= 0 and g_prev) else np.zeros(0, np.int64)
            Pn = np.flatnonzero(P_new)
            self.PY.learn(self._P_prev_ids, Pn, rise, held)
            if self.PY2 is not None:
                self.PY2.learn(self._P_prev_ids, Pn, rise, held)
            self.CY.learn(b.Hc_prev, b.Hc, rise, held)
            ge_now = (not self.gate) or b.ENDext
            ge_prev = (not self.gate) or b.ENDext_prev
            self.CE.learn(b.Hc_prev, b.Hc, np.array([0]) if (b.END and ge_now) else np.zeros(0, np.int64),
                          np.array([0]) if (b.END_prev and ge_prev) else np.zeros(0, np.int64))
            if self.slot is not None:
                self.slot.learn(self, b.Y, g_now, b.Y_prev, g_prev)
        self.Tm.learning = self.learning
        self.Tm.commit(T_new)
        self.P = P_new
        self.t += 1
        experienced = (E_now > 0) or b.SPK
        if self.learning and experienced:
            self.t_exp += 1
            if self.t_exp % 100 == 0:
                n = b.n_prog
                for p in (self.PY, self.CY) + ((self.PY2,) if self.PY2 is not None else ()):
                    p.decay(n)
                self.CE.decay()
        if self.slot is not None:
            self.slot.tick(self, dict(CD=int(b.CD.sum())))
        self.tick_event = b.stop if b.stop in ('END', 'floor', 'breath') else None
        return force

    @property
    def backterm_hits(self):
        return self.PY.n_back + self.CY.n_back + self.CE.n_back + (self.PY2.n_back if self.PY2 is not None else 0)
