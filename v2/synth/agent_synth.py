# -*- coding: utf-8 -*-
"""ARCH_v2 synthesized agent = think's agent (speech caused by thought: intention + BG gate + 6 reflexes) on tune's
core (develop + awake recall + 10k hippocampus) with the synthesis additions of core_synth.py.

policy='v1', think_k=3, ior=0 and every synthesis key at its default reproduce tune's DevAgent bit for bit
(test_equiv_synth.py).  The awake recall of tune (quiet moment = 12 silent ticks, not speaking, not inside a heard
line) is kept unchanged.

Extra agent keys:
  quiet_ticks   quiet moment length for awake recall (tune: 12)
  turn_feats    'R' (think), 'S' (line surprise instead of R), 'RS' (both): features of the learned turn brake / eagerness
  init_min_tau  self-initiated (scene) thoughts may release only after this many ticks (reflex 7; 0 = think)
  rebind        1: a heard category without a program is re-matched against the babbling memory at each hearing
  wander        > 0: in silence the thought spreads along learned item associations (favourite thoughts) every
                `wander` ticks (needs cfg ct_assoc > 0)
  retino        set automatically when the world's eye_layout is 'retino' (innate retinotopy: slot blocks of 240)
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from agent_think import ThinkAgent           # noqa: E402
from core_think import TCfg                  # noqa: E402
from core_synth import SynthCore, SCfg       # noqa: E402

THINK_KEYS = set(inspect.signature(ThinkAgent.__init__).parameters) - {'self', 'world', 'cfgkw'}
SYNTH_AGENT_KEYS = {'quiet_ticks': int, 'wander': int, 'rebind': int, 'wander_after': int, 'turn_feats': str,
                    'init_min_tau': int}


def split_kwargs(kw):
    ag, cfg = {}, {}
    for k, v in kw.items():
        if k in THINK_KEYS or k in SYNTH_AGENT_KEYS:
            ag[k] = v
        else:
            cfg[k] = v
    return ag, cfg


class SynthAgent(ThinkAgent):
    def __init__(self, world, seed=0, quiet_ticks=12, wander=0, rebind=0, wander_after=200, turn_feats='R',
                 init_min_tau=0, **kw):
        ag, cfgkw = split_kwargs(kw)
        tc = {k: v for k, v in cfgkw.items() if hasattr(TCfg(), k) and k not in ('retino', 'seed', 'V', 'n_eye')}
        super().__init__(world, seed=seed, **ag, **tc)
        layout = getattr(world, 'p', {}).get('eye_layout', 'random')
        if layout == 'retino':
            cfgkw.setdefault('retino', True); cfgkw.setdefault('seg', 240); cfgkw.setdefault('rel', True)
        V = self.V
        cfg = SCfg(V=V, n_eye=len(world.eye), seed=seed, **{'retino': False, **cfgkw})
        self.core = SynthCore(cfg)
        self.quiet_ticks = quiet_ticks
        self.quiet = 0
        self.wander = wander
        self.wander_after = wander_after
        self.fav_log = []                  # (t, silence, thought items) at the first heard onset after a long silence
        self.rebind = rebind
        self.turn_feats = turn_feats
        self.init_min_tau = init_min_tau
        self.stats['rebound'] = 0

    def _feats(self, th):
        """features of a thought for the turn learners (think's) - optionally the heard line's surprise S (was the line
        predicted by the readout: a recast/confirmation usually is, a new question is not) instead of / besides think's
        R = 'I spoke since the previous heard line', which in a partner-without-feedback world marks every new
        question exactly like a recast."""
        f = ThinkAgent._feats(self, th)
        mode = self.turn_feats
        if mode in ('S', 'RS') and th['kind'] == 'reply':
            sp = self.core.last_line_surprise
            f.append(('S', -1 if sp is None else int(sp >= 0.5)))
        if mode == 'S':
            f = [k for k in f if k[0] != 'R']
        return f

    def _intend(self):
        """think's intention + one innate reflex for SELF-INITIATED speech: a scene thought (no prompt) may release
        only after it has been open for init_min_tau ticks of silence (a scene thought closes as soon as another
        speaker starts, so a partner who speaks within that time can never be talked over)."""
        th = self.th
        ThinkAgent._intend(self)
        if self.REL and th is not None and self.th is th and th['kind'] == 'scene' and not th['released'] and \
                self.t - th['t_open'] < self.init_min_tau:
            self.REL = False
            self.tstats['blocked_init_min'] = self.tstats.get('blocked_init_min', 0) + 1

    def _parse_eye(self, eye):
        """think's parser; with an innate retinotopic eye the attended slot is WHERE the pointing is (the block that
        holds the extra units), not an arbitrary learned label."""
        base, att = ThinkAgent._parse_eye(self, eye)
        c = self.core.c if hasattr(self, 'core') else None
        if att is not None and c is not None and c.retino and base is not None:
            e = np.asarray(eye).astype(bool)
            extra = e & ~base
            seg = c.seg
            cnt = [int(extra[s * seg:(s + 1) * seg].sum()) for s in range(3)]
            att = int(np.argmax(cnt)) if max(cnt) >= 4 else att
        return base, att

    def _detect(self, f):
        """perception (develop's) + new-word programs: a heard category that has no program yet is matched again
        against the babbling memory each time it is heard (imitation attempt), so a syllable whose first template did
        not match becomes speakable when a clean instance is heard."""
        j = ThinkAgent._detect(self, f)
        if self.rebind and j is not None and j not in self.cat2prog and len(self.bab_set) and self.learning:
            self._bind_from_memory(j, f)
            if j in self.cat2prog:
                self.stats['rebound'] += 1
        return j

    def step(self, obs):
        self.core.now = self.t
        force = ThinkAgent.step(self, obs)
        core = self.core
        E = float(np.asarray(obs['ear']).sum())
        if E == 0 and not self.in_line and not self._own_voice():
            self.quiet += 1
        else:
            self.quiet = 0
        if self.quiet == self.quiet_ticks and self.learning and core.c.recall_n > 0:
            core.stats['quiet'] += 1
            core.awake_recall()                              # quiet moment: rehearse stored episodes (no sleep)
        if self.wander and self.quiet > self.wander_after and self.quiet % self.wander == 0:
            core.now = self.t
            core.tick_thought()
        if self.quiet >= 2000:
            self._long_quiet = self.quiet
        elif self.quiet == 0 and getattr(self, '_long_quiet', 0) and self.in_line:
            if len(self.fav_log) < 5000:
                self.fav_log.append((self.t, self._long_quiet, core.thought_items()))
            self._long_quiet = 0
        return force
