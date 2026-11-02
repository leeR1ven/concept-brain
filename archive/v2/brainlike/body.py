# ARCH_v1's body as an innate reflex scaffold (brainstem / BG / thalamic GO), copied from arch/synth/archv1.py
# Brain.step (turn units, beat, program WTA, END / breath counter, D/I/M), with a "mind" interface:
#   at each boundary the mind supplies scores per program and for END; the body does F/GO/CD, the 3-tick beat,
#   the WTA, D/I/M and the counter, and exposes its signals.
# Programs = muscle sets (4 of 32).  Created WITHOUT a teacher by the mind (babbling: body.add_program(set)).
# Inputs per tick: E_now = window energy at t (F reads it at t+1, exactly as ARCH_v1).
import numpy as np

M_N, LV = 32, 3


class Body:
    def __init__(self, NP=512, breath=12, theta=0.05):
        self.NP, self.breath, self.theta = NP, breath, theta
        self.prog = np.zeros((NP, M_N), bool)
        self.n_prog = 0
        self.F = self.F2 = self.GO = self.R1 = self.R2 = False
        self.CD = np.zeros(4, bool)
        self.SPK = False
        self.Hp = -1; self.phase = 0; self.nsyl = 0
        self.Y = -1; self.END = False
        self.E_prev = 0.0
        self.force = np.zeros(M_N, int)
        self.t = 0
        self.go_req = False                  # spontaneous vocal play: the mind may request an idle-time GO
        self.trial = None                    # babble trial set (not a program): held for one beat, then dropped

    def add_program(self, mset):
        """commit a muscle set as a program (babbling keeps it); returns its id (existing id if already present)."""
        v = np.zeros(M_N, bool); v[list(mset)] = True
        if self.n_prog:
            hit = np.nonzero((self.prog[:self.n_prog] == v).all(1))[0]
            if len(hit):
                return int(hit[0])
        if self.n_prog >= self.NP:
            return -1
        y = self.n_prog; self.prog[y] = v; self.n_prog += 1
        return y

    def step(self, E_now, mind):
        """one tick.  mind.boundary(body, first) -> (prog_scores (n_prog,), end_score) is called on boundary ticks.
        Returns the event dict (also kept in self.ev)."""
        ev = {}
        ncd = int(self.CD.sum())
        F_new = (0.01 * self.E_prev - 10.0 * ncd) >= 0.1
        F2_new = self.F
        GO_new = (1.0 * self.F2 - 2.0 * self.F - 2.0 * self.SPK - 2.0 * ncd) >= 1.0
        idle = (not self.F) and (not self.SPK) and ncd == 0 and (not self.GO) and (not self.R1) and self.Hp == -1
        if self.go_req and idle:
            GO_new = True
        self.go_req = False
        CD_new = np.r_[self.GO or self.SPK, self.CD[:-1]]
        R1_new = self.GO
        R2_new = self.GO or (self.SPK and self.phase == 1)
        bnd = (self.R1 and not self.F) or (self.SPK and self.phase == 2)
        first = bnd and not self.SPK
        Y_new, END_new = -1, False
        if bnd and self.Hp == -2:                           # end of a babble trial beat: stop, no selection
            ev['stop'] = 'trial'; self.Hp = -1; self.phase = 0; ev['trial_end'] = self.trial; self.trial = None
        elif bnd:
            ev['bnd'] = True; ev['first'] = first
            if first:
                self.nsyl = 0
            sc, s_end = mind.boundary(self, first)
            if isinstance(sc, str) and sc == 'trial':        # the mind babbles a trial set for one beat
                self.Hp = -2; self.phase = 0; ev['trial'] = self.trial; sc = np.zeros(0); s_end = 0.0
                self.SPK = True
            sc = np.asarray(sc, float)[:self.n_prog] if self.n_prog else np.zeros(0)
            best = int(np.argmax(sc)) if len(sc) else -1
            s_best = float(sc[best]) if best >= 0 else -np.inf
            ev['s_best'] = s_best; ev['s_end'] = float(s_end)
            if self.Hp == -2:
                pass
            elif not first and s_end >= max(s_best, self.theta):
                ev['stop'] = 'END'; END_new = True
            elif not first and self.nsyl >= self.breath:
                ev['stop'] = 'breath'
            elif best < 0 or s_best < self.theta:
                ev['stop'] = 'floor' if not first else 'silent'
            else:
                Y_new = best; self.Hp = best; self.phase = 0; self.nsyl += 1
                ev['rel'] = best
            if 'stop' in ev and self.Hp != -2:
                self.Hp = -1; self.phase = 0
        elif self.SPK:
            self.phase += 1
        SPK_new = self.Hp >= 0 or self.Hp == -2
        if self.Hp >= 0:
            D = self.prog[self.Hp].copy()
        elif self.Hp == -2:
            D = np.zeros(M_N, bool); D[list(self.trial)] = True
        else:
            D = np.zeros(M_N, bool)
        force = np.where(D, self.phase + 1, 0)            # M_{m,l} = D_m & l <= phase  -> force 1,2,3
        self.F, self.F2, self.GO, self.R1, self.CD = bool(F_new), bool(F2_new), bool(GO_new), bool(R1_new), CD_new
        self.R2 = bool(R2_new)
        self.Y, self.END = Y_new, END_new
        self.SPK = SPK_new
        self.force = force
        self.E_prev = E_now
        self.t += 1
        ev.update(F=self.F, GO=self.GO, CD=int(self.CD.sum()), SPK=SPK_new, phase=self.phase, force=force)
        self.ev = ev
        return ev
