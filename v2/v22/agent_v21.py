# -*- coding: utf-8 -*-
"""ARCH_v2.1 agent = ARCH_v2's SynthAgent on the RoleCore (core_v21.py).  With every new key at its default this is
ARCH_v2 (SynthAgent) bit for bit (test_equiv_v21.py).

Agent-side pieces of role-aware listening:
  * efference: the tick the own mouth starts an utterance -> core.ral_mouth_onset(); every tick core.ral_tick()
    (role timeout); core.storing = the agent's learning flag (frozen copies store nothing).
  * voice categories (voice=1; perception only, a hook for world2): per heard syllable, the raw ear window minus the
    running raw template of its syllable category (speaker-normalisation residual), sign-quantised, matched to voice
    templates with vigilance; a line's voice = the majority over its syllables.  Logged per line in self.voice_log.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from agent_think import ThinkAgent                       # noqa: E402
from agent_synth import SynthAgent, split_kwargs         # noqa: E402
from core_think import TCfg                              # noqa: E402
from core_v21 import RoleCore, RCfg                      # noqa: E402

V21_AGENT_KEYS = {'voice': int, 'val_veto': float, 'dip': int, 'soc_feats': int, 'react_T': int}


class VoiceCats:
    def __init__(self, cap=8, theta=0.7, rate=0.05, n_min=10):
        self.tmpl, self.cnt = {}, {}
        self.V, self.vn = [], []
        self.cap, self.theta, self.rate, self.n_min = cap, theta, rate, n_min

    def observe(self, cat, win, learn=True):
        w = np.asarray(win, np.float32).ravel()
        t = self.tmpl.get(cat)
        n = self.cnt.get(cat, 0)
        if t is None:
            if learn:
                self.tmpl[cat] = w.copy(); self.cnt[cat] = 1
            return -1
        res = w - t
        if learn:
            self.cnt[cat] = n + 1
            self.tmpl[cat] = t + (w - t) * max(1.0 / (n + 1), self.rate)
        if n < self.n_min:
            return -1
        s = np.sign(np.round(res)).astype(np.float32)
        nz = s != 0
        if nz.sum() < 3:
            return 0                               # the usual voice of this syllable (no residual)
        best, bj = 0.0, -1
        for j, v in enumerate(self.V):
            m = float((np.sign(np.round(v)) == s)[nz].mean())
            if m > best:
                best, bj = m, j
        if best >= self.theta:
            if learn:
                self.V[bj] += (s - self.V[bj]) * self.rate; self.vn[bj] += 1
            return bj + 1
        if learn and len(self.V) < self.cap:
            self.V.append(s.copy()); self.vn.append(1)
            return len(self.V)
        return -1


class AgentV21(SynthAgent):
    def __init__(self, world, seed=0, quiet_ticks=12, wander=0, rebind=0, wander_after=200, turn_feats='R',
                 init_min_tau=0, voice=0, val_veto=0.0, dip=0, soc_feats=0, react_T=50, **kw):
        ag, cfgkw = split_kwargs(kw)
        ral = {k: cfgkw.pop(k) for k in list(cfgkw) if k in RCfg.RAL}
        tc = {k: v for k, v in cfgkw.items() if hasattr(TCfg(), k) and k not in ('retino', 'seed', 'V', 'n_eye')}
        ThinkAgent.__init__(self, world, seed=seed, **ag, **tc)
        layout = getattr(world, 'p', {}).get('eye_layout', 'random')
        if layout == 'retino':
            cfgkw.setdefault('retino', True); cfgkw.setdefault('seg', 240); cfgkw.setdefault('rel', True)
        cfg = RCfg(V=self.V, n_eye=len(world.eye), seed=seed, **{'retino': False, **cfgkw}, **ral)
        self.core = self._make_core(cfg)
        self.quiet_ticks = quiet_ticks
        self.quiet = 0
        self.wander = wander
        self.wander_after = wander_after
        self.fav_log = []
        self.rebind = rebind
        self.turn_feats = turn_feats
        self.init_min_tau = init_min_tau
        self.stats['rebound'] = 0
        self.voice = VoiceCats() if voice else None
        self.voice_log = []                         # (t_end, majority voice, n syllables) per heard line
        self._line_voices = []
        self._win = None
        # step 3 (agent side): NoGo veto on self-initiated release by the learned value, dips, social features
        self.val_veto, self.dip, self.soc_feats, self.react_T = val_veto, dip, soc_feats, react_T
        self._dip_t = None            # end of the own utterance still waiting for a reaction
        self._t_other = -10 ** 9      # last onset of another voice
        self._t_arr = self._t_left = -10 ** 9
        self._eye_prev = None
        self.tstats['blocked_val'] = 0

    def _make_core(self, cfg):
        return RoleCore(cfg)

    def _detect(self, f):
        j = SynthAgent._detect(self, f)
        if self.voice is not None and j is not None and self._win is not None:
            self._line_voices.append(self.voice.observe(j, self._win, learn=self.learning))
        return j

    def step(self, obs):
        core = self.core
        core.storing = self.learning
        core.now = self.t
        was_mouth = getattr(self, 'mouth_ago', 99) == 0
        was_line = self.in_line
        had_scene = self.base is not None
        self._th_kind = self.th['kind'] if self.th is not None else None   # the thought that may speak this tick
        self._win = obs['ear']
        force = SynthAgent.step(self, obs)
        if self.dip:
            self._dip_check(obs.get('reward', 0.0))
        if self.voice is not None and was_line and not self.in_line:
            vs = [v for v in self._line_voices if v >= 0]
            maj = max(set(vs), key=vs.count) if vs else -1
            if len(self.voice_log) < 200000:
                self.voice_log.append((self.t, maj, len(self._line_voices)))
            if core.last_rec is not None:
                core.last_rec['voice'] = maj
            self._line_voices = []
        if core.c.ral:
            core.now = self.t
            if core.c.ral_q and ((self.base is not None) != had_scene or getattr(self, 'base_changed', False)):
                core.ral_scene_change()             # the scene went away / changed: no turn outcome for the last line
            if self.mouth_ago == 0 and not was_mouth and self.learning:
                core.ral_mouth_onset()
            core.ral_tick()
        return force

    # ================================================================== step 3: the signed value loop (agent side)
    def _feats(self, th):
        f = SynthAgent._feats(self, th)
        if self.soc_feats:
            t = self.t                           # innate coarse social detectors (perceivable cues only)
            f = f + [('heard', int(t - self._t_other < 300)),       # someone spoke recently (a person is around)
                     ('arr', int(t - self._t_arr < 60)),            # something new appeared in silence
                     ('left', int(t - self._t_left < 60))]          # something disappeared in silence
        return f

    def _gain(self, th):
        if self.val_veto > 0:                    # value vetoes (below); it no longer scales the eagerness
            th['V'] = self.V_go.p(th['feats'])
            return float(self.g0[th['kind']])
        return SynthAgent._gain(self, th)

    def _intend(self):
        th = self.th
        SynthAgent._intend(self)
        if self.val_veto > 0 and self.REL and th is not None and self.th is th and th['kind'] == 'scene' \
                and not th['released'] and th.get('V', 1.0) < self.val_veto:
            self.REL = False                     # NoGo: what I would start here is not expected to be welcome
            self.tstats['blocked_val'] += 1

    def _other_onset(self):
        self._t_other = self.t
        self._dip_t = None                       # a reaction came
        return SynthAgent._other_onset(self)

    def _parse_eye(self, eye):
        out = SynthAgent._parse_eye(self, eye)
        if self.soc_feats:
            e = np.asarray(eye).astype(bool)
            p = self._eye_prev
            if p is not None and not self.in_line:
                if (e & ~p).sum() > 2:
                    self._t_arr = self.t
                if (p & ~e).sum() > 2:
                    self._t_left = self.t
            self._eye_prev = e
        return out

    def _dip_check(self, r):
        """step 3: a self-initiated utterance that gets no reaction at all within react_T ticks -> a dip (delta = -V)."""
        core = self.core
        if r:
            self._dip_t = None
        if core.actor_t == self.t - 1 and core.actor_tr and self._dip_arm():
            self._dip_t = self.t        # own_finish ran this step (self.t is already + 1)
            self._dip_keys = list(self.elig[0]) if self.elig is not None else None
        if self._dip_t is not None and self.t - self._dip_t > self.react_T:
            self._dip_t = None
            if self.learning:
                self._dip_fire()

    def _dip_arm(self):
        """which own utterances can be followed by a dip.  ARCH_v2.1 as reported (v21final, life_s3b): every own
        utterance ('all').  'scene': self-initiated utterances only (v21's rejected v21fin variant; the code in the v21
        folder was left in this state, see the v2.2 report)."""
        scope = getattr(self, 'dip_scope', 'all')
        return scope != 'scene' or getattr(self, '_th_kind', None) == 'scene'

    def _dip_fire(self):
        self.core.reward(0.0)
