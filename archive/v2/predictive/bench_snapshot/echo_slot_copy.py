# bench copy of arch/synth/echo_slot.py; only change: a view instead of a copy of Wz[:, com] (marked "bench")
# -*- coding: utf-8 -*-
"""Content-slot plug-in for ARCH_v1: the echo (copy) route, on ARCH_v1's timeline (B1 = L+6, R2 = B-1).

  Z_y   auditory mirror of program y: learned FEAT0 -> Z (exact rule: forward eta/(1+w/w_ref), back -eta floor 0,
        decay gamma per 100 experienced ticks, per-target budget B_z).  Teaching signal: the program's own delay line
        clamps Z_y for one tick at B+4 after the model itself said y at B..B+2 (own voice full in the window at B+3).
        Z learning only while the corollary discharge is on (own voice possible in the ear) = the content
        specialist's z_gate='own'.
  det   every tick: drive = FEAT0(t-1) . W_z ; winner-take-all over committed programs; detected if drive >= theta*B_z.
  latch while F is on (the other is talking), a detection sets latch_y; the latch is cleared when a new line of the
        other starts (F rises) and entry y is cleared from B+1 once y has been said (inhibition of return).
  E     replay pulse E_y = latch_y AND R2 (the pre-boundary tick B-1) -> one-tick source (timing law satisfied).
  E->Y  learned DIAGONAL weight w_y (E_y -> Y_y), exact rule, cap B_e, gated by external drive like all decision paths.
  score at boundary B: mult * w_y * E_y(B-1)  (capped by the core at slot_cap = 7).
"""
import collections

import numpy as np


class EchoSlot:
    def __init__(self, NP, eta_z=0.005, B_z=2.0, gamma_z=0.999, theta=0.75, eta_e=0.05, B_e=2.0, gamma_e=0.999,
                 mult=1.0, w_ref=0.2, z_self_train=True):
        self.NP = NP
        self.eta_z, self.B_z, self.gamma_z, self.theta = eta_z, B_z, gamma_z, theta
        self.eta_e, self.B_e, self.gamma_e, self.mult, self.w_ref = eta_e, B_e, gamma_e, mult, w_ref
        self.Wz = None
        self.f_prev = None
        self.z_prev = np.zeros(NP, bool)
        self.latch = np.zeros(NP, bool)
        self.E = np.zeros(NP, bool); self.E_prev = np.zeros(NP, bool)
        self.we = np.zeros(NP, np.float32)
        self.yhist = collections.deque([-1] * 5, maxlen=5)
        self.stats = collections.Counter()
        self.z_self_train = z_self_train

    # --------------------------------------------------------------- boundary read-out
    def score(self, b, com, first):
        """called inside step t (before the tick hook): self.E is E(t-1), the pulse at the pre-boundary tick."""
        return self.mult * self.we[com] * self.E[com]

    # --------------------------------------------------------------- decision learning (called inside Brain.step)
    def learn(self, b, Y_new, g_now, Y_prev, g_prev):
        """inside step t: forward (E_y on at t-1 = self.E, Y_y rises at t, gated).  The back term (E_y on at t while
        Y_y was on at t-1) needs E(t) and is applied in tick(); it can never fire because R2 is never on at B+1."""
        if Y_new >= 0 and g_now and self.E[Y_new]:
            w = self.we[Y_new]
            self.we[Y_new] = w + self.eta_e / (1 + w / self.w_ref)
            self.stats['E_fwd'] += 1
        self.pending = (Y_new, g_now)

    # --------------------------------------------------------------- every tick (after the brain committed tick t)
    def tick(self, b, ev):
        if b.FEAT0 is None or len(b.FEAT0) < 2:
            return
        f_now = b.e[b.FEAT0]
        if self.Wz is None:
            self.Wz = np.zeros((len(b.FEAT0), self.NP), np.float32)
            self.f_prev = np.zeros(len(b.FEAT0), bool)
        self.yhist.appendleft(b.Y)                                    # yhist[k] = Y(t-k)
        # ---- Z teaching: program said at B = t-4 -> Z clamp at t
        z_now = np.zeros(self.NP, bool)
        y4 = self.yhist[4]
        if y4 >= 0:
            z_now[y4] = True
        # ---- detection from FEAT0(t-1) (current into Z at t)
        com = np.flatnonzero(b.committed)
        det = -1
        if len(com) and self.f_prev.any():
            n = len(com)
            Wc = self.Wz[:, :n] if com[-1] == n - 1 else self.Wz[:, com]     # bench: view when contiguous (speed)
            drive = self.f_prev.astype(np.float32) @ Wc
            k = int(np.argmax(drive))
            if drive[k] >= self.theta * self.B_z:
                det = int(com[k])
                if self.z_self_train and y4 < 0:
                    z_now[det] = True                                  # detection itself fires Z (self-training)
        # ---- latch
        if b.F and not b.F2:                                          # a new line of the other starts
            self.latch[:] = False
        if det >= 0 and b.F:
            self.latch[det] = True; self.stats['latched'] += 1
        if b.Y >= 0:                                                  # said -> inhibition of return from B+1
            self.clear_next = b.Y
        elif getattr(self, 'clear_next', -1) >= 0:
            self.latch[self.clear_next] = False; self.clear_next = -1
        # ---- Z learning (exact rule), gated by the corollary discharge: only while the own voice can be in the ear
        if b.learning and ev['CD'] > 0:
            rise = np.flatnonzero(z_now & ~self.z_prev)
            src = np.flatnonzero(self.f_prev)
            if len(rise) and len(src):
                ix = np.ix_(src, rise); w = self.Wz[ix]
                self.Wz[ix] = w + self.eta_z / (1 + w / self.w_ref)
            held = np.flatnonzero(self.z_prev)
            src2 = np.flatnonzero(f_now)
            if len(held) and len(src2):
                ix = np.ix_(src2, held); self.Wz[ix] = np.maximum(self.Wz[ix] - self.eta_z, 0)
        # ---- replay pulse E(t) = latch & R2(t); back term of E->Y for the release at t-1
        self.E_prev = self.E
        self.E = self.latch & bool(b.R2)
        yp, gp = getattr(self, 'pending_prev', (-1, False))
        if b.learning and yp >= 0 and gp and self.E[yp]:
            self.we[yp] = max(self.we[yp] - self.eta_e, 0.0); self.stats['E_back'] += 1
        self.pending_prev = getattr(self, 'pending', (-1, False)); self.pending = (-1, False)
        self.z_prev = z_now
        self.f_prev = f_now
        # decay rides on the brain's experience clock
        if b.learning and b.t_exp > getattr(self, 't_exp_seen', 0) and b.t_exp % b.c.decay_T == 0:
            self.t_exp_seen = b.t_exp
            self.decay()

    def decay(self):
        self.Wz *= self.gamma_z
        tot = self.Wz.sum(0)
        self.Wz *= np.minimum(1.0, self.B_z / np.maximum(tot, 1e-12)).astype(np.float32)[None, :]
        self.we *= self.gamma_e
        np.minimum(self.we, self.B_e, out=self.we)
