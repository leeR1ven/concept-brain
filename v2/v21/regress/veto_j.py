# -*- coding: utf-8 -*-
"""Full staged J protocol (synth/run_j.py, unchanged) with the role-aware listening veto of drift_veto.py built into
the agent class (deepcopy-safe).  VETO=1|0 from the environment; W = 12 ticks.  Output goes to this folder.
usage: VETO=1 python3 veto_j.py tag=veto seed=0 nS=600 w_eye_layout=retino rebind=1 ct=1 init_min_tau=10 hip_rho=2 ctx=0 out=..."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..', '..', 'synth')
sys.path.insert(0, SYN); sys.path.insert(0, os.path.join(SYN, 'bench_copy'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
import rules                       # noqa
import run_j                       # noqa
from agent_synth import SynthAgent  # noqa
VETO, W = int(os.environ.get('VETO', '1')), 12
_orig_learn = rules.Delta.learn
def _learn(self, A, target, total, J=None, **kw):
    st = getattr(self, '_vt', None)
    if st is None or not st['rec_on'] or self._vtc._replaying or len(A) == 0:
        return _orig_learn(self, A, target, total, J=J, **kw)
    Jj = np.arange(self.W.shape[1]) if J is None else np.asarray(J)
    Ai = np.asarray(A)
    before = self.W[np.ix_(Ai, Jj)].copy()
    out = _orig_learn(self, A, target, total, J=J, **kw)
    st['buf'].append((self, Ai, Jj, self.W[np.ix_(Ai, Jj)] - before))
    return out
rules.Delta.learn = _learn

class VetoAgent(SynthAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._vt = dict(rec_on=False, buf=[], pending=None, pv=None, prev_mouth=False, vetoed=0, kept=0, cut=0)
        for d in (self.core.gen, self.core.gate):
            d._vt = self._vt; d._vtc = self.core
        self.stats['veto'] = 0

    def _revert(self, p):
        S, core = self._vt, self.core
        for obj, Ai, Jj, d in p['buf']:
            obj.W[np.ix_(Ai, Jj)] -= d
        pv, rec = p['pv'], p['rec']
        if pv is not None and rec is not None and pv.get('ep', -1) >= 0 and pv.get('next') == rec.get('ep'):
            pv['next'] = None; core.next_of[pv['ep']] = -1; S['cut'] += 1
        S['vetoed'] += 1; self.stats['veto'] = S['vetoed']

    def _decide(self, p):
        """VETO=2: an own reply that only repeats the heard line (imitation, symmetric routines like 你好 -> 你好) does
        not mark the line as a question; any other own reply does."""
        S = self._vt
        own = p.get('own')
        items = list(own.get('items', [])) if own is not None else []
        heard = list((p['rec'] or {}).get('items', []))
        if len(items) and all(i in heard for i in items):
            S['rep'] = S.get('rep', 0) + 1; S['kept'] += 1
            return
        self._revert(p)

    def step(self, obs):
        S, core = self._vt, self.core
        was, pv_before = self.in_line, core.last_rec
        m = SynthAgent.step(self, obs)
        if not (VETO and self.learning and core.c.learn_listen):
            S['rec_on'] = False; S['pending'] = None; S['buf'] = []
            return m
        now = self.in_line
        if not was and now:
            p = S['pending']
            if p is not None:
                if p.get('answered'):
                    self._decide(p)
                else:
                    S['kept'] += 1
            S['pending'] = None; S['buf'] = []; S['rec_on'] = True; S['pv'] = pv_before
        if was and not now:
            S['pending'] = dict(buf=S['buf'], t=self.t, pv=S['pv'], rec=core.last_rec); S['buf'] = []; S['rec_on'] = False
            if VETO == 3:            # feedback on MY line (it contains all my last utterance's items): never a question
                own = S.get('own_last'); heard = set((core.last_rec or {}).get('items', []))
                if own and self.t - S.get('own_last_t', -10 ** 9) <= 60 and set(own) <= heard:
                    S['pending']['feedback'] = True
        mouth = bool(np.any(np.asarray(m) > 0))
        if S['prev_mouth'] and not mouth and self.own_st is not None:
            S['own_last'] = list(self.own_st.get('items', [])); S['own_last_t'] = self.t
        p = S['pending']
        if p is not None and p.get('feedback'):
            if mouth and not S['prev_mouth']:
                S['fb'] = S.get('fb', 0) + 1; S['kept'] += 1; S['pending'] = None
        elif p is not None:
            if not p.get('answered'):
                if mouth and not S['prev_mouth']:
                    if self.t - p['t'] <= W:
                        if VETO in (1, 3):
                            self._revert(p); S['pending'] = None
                        else:
                            p['answered'] = True; p['own'] = self.own_st; p['quiet'] = 0
                    else:
                        S['kept'] += 1; S['pending'] = None
            else:
                p['quiet'] = 0 if mouth else p['quiet'] + 1
                if p['quiet'] >= 2:
                    self._decide(p); S['pending'] = None
        S['prev_mouth'] = mouth
        return m

run_j.SynthAgent = VetoAgent
if __name__ == '__main__':
    run_j.main()
