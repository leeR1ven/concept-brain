# -*- coding: utf-8 -*-
"""ThinkAgent: the develop agent whose speech is caused by its THOUGHT, not by a timer.

Per tick (after perception, as develop):
  * a thought opens at the end of another speaker's line ('reply') or at a new scene / pointing onset in silence
    ('scene'); it closes when the own utterance ends, when another speaker starts, or after T_MAX ticks.
  * while the thought is open and nobody (incl. the own voice) is sounding: one thinking step (core_think).
  * speech intention (BG accumulator):   I(t) = I(t-1) + G * e(t) + u
        e(t) = readout confidence after the step (margin of the best first item)
        G    = clip(g0[kind] + beta * (V_go(ctx) - rbar), 0, g_max)      learned eagerness (partner's reaction)
        u    = innate urgency per tick (reply thoughts only)               -> a guess by ~1/u ticks at the latest
  * release (REL -> body GO):  I >= 1  and  p_oth(ctx, tau) < nogo     (learned: "the other will start soon")
    and the innate brake in the body: F (other talking), own voice, CD, GO/R1 in flight.
  * one thought -> at most one utterance (it closes when the utterance ends): no monologue by construction.
Learned turn-taking (local, PR form, binary features of the held thought: kind, prompt items, first/last item,
'I spoke since the last line'):
  P_oth  target y = 1 if another speaker's line starts within the next L ticks, else 0 (listening; M = 1; the own
         speech does not censor it: what the partner does while/after the agent speaks is part of the regularity).  Features are crossed with tau (ticks since the thought opened; time cells).
  V_go   target y = r = the partner's reaction (obs reward on the first tick of the next line after the own
         utterance) for the context of the released thought (eligibility held until that reaction).
Policies: 'free' (the intention), 'v1' (the body's fixed GO timer; thought completed at the boundary = develop),
          'budget:k' (release as soon as k steps are done; the thought is frozen at k steps; F brake kept),
          'asap' (release at once, keep thinking until the first boundary).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from agent import DevAgent, BENCH            # noqa: E402
from body_think import ThinkBody             # noqa: E402
from core_think import ThinkCore, TCfg       # noqa: E402
from rules import Templates                  # noqa: E402
from frontend import Ear0, feat0             # noqa: E402


class SparsePR:
    """p = clip(b + sum_i w[key_i], 0, 1) over binary keys;  learn: w[key_i] += eta * M * (y - p) / n  (PR / NLMS)."""

    def __init__(self, eta=0.2, init=0.0):
        self.w = {}
        self.eta, self.init = eta, init
        self.n_updates = 0

    def p(self, keys, bias=None):
        b = self.init if bias is None else bias
        return float(np.clip(b + sum(self.w.get(k, 0.0) for k in keys), 0.0, 1.0))

    def learn(self, keys, y, p=None, M=1.0, bias=None):
        if not keys:
            return
        p = self.p(keys, bias) if p is None else p
        d = self.eta * M * (y - p) / len(keys)
        if d == 0:
            return
        for k in keys:
            self.w[k] = self.w.get(k, 0.0) + d
        self.n_updates += 1


class ThinkAgent(DevAgent):
    T_MAX = 40

    def __init__(self, world, seed=0, V=256, theta_z=0.75, z_rate=0.1, learn_cats=True, policy='free',
                 theta_I=1.0, g0_reply=1.0, g0_scene=0.3, u_reply=0.05, u_scene=0.0, beta=2.0, g_max=3.0,
                 nogo=0.25, nogo_late=0.05, L=10, eta_oth=0.2, eta_go=0.2, learn_turn=True, tau_w=7, urg_gate=True,
                 close_on_scene=True, oth_kind=False, brake_now=False, life_eye=False, eye_feats=False,
                 no_react=False, late_release=True, u_R1=1.0, own_tail=False, **cfgkw):
        self.seed = seed
        self.ear = Ear0(seed=world.seed)
        self.F0 = feat0(Ear0(seed=world.seed), world.warmup_windows())
        self.ear.set_feat0(self.F0)
        self.NF = len(self.F0)
        self.Z = Templates(self.NF, cap=V, theta=theta_z, rate=z_rate)
        self.V = V
        self.theta_z = theta_z
        self.learn_cats = learn_cats
        cfg = TCfg(V=V, n_eye=len(world.eye), retino=False, seed=seed, **cfgkw)
        self.core = ThinkCore(cfg)
        self.policy = policy
        self.body = ThinkBody(n_prog=V + 64, theta=cfg.floor, end_rule='max', rho=1.0, recruit='babble',
                              keep='manual', seed=seed, go_mode='v1' if policy == 'v1' else 'gate', brake_now=brake_now)
        self.life_eye = life_eye
        self.eye_feats = eye_feats      # the visible eye bits are features of the turn learners (who is there, what)
        self.no_react = no_react        # no reaction within 50 ticks of the own utterance -> V_go target 0
        self.late_release = late_release  # False: a reply thought may release only inside the transition window
        self.u_R1 = u_R1                # urgency factor when the line is a reaction to the own utterance (R = 1)
        self.own_tail = own_tail        # own voice = mouth active or <= 2 ticks ago (the ear window), NOT the CD timer
        self.mouth_ago = 99
        self.cat2prog, self.prog2cat = {}, {}
        self.bab_pat = np.zeros((0, self.NF), bool)
        self.bab_set = []
        self.imitate = True
        self.theta_bind = 0.85
        self.E_prev = 0.0
        self.in_line = False
        self.clk = 0
        self.st = None
        self.base = None
        self.att_codes = []
        self.scene, self.att = None, None
        self.own_st = None
        self.learning = True
        self.stats = dict(items=0, new_cats=0, lines=0, own=0, babbles=0, bound=0)
        self.det_log = []
        self.t = 0
        self.mute = False
        # ---- intention parameters (innate) and learned turn-taking
        self.theta_I, self.g0 = theta_I, {'reply': g0_reply, 'scene': g0_scene}
        self.u = {'reply': u_reply, 'scene': u_scene}
        self.beta, self.g_max, self.nogo, self.L = beta, g_max, nogo, L
        self.P_oth = SparsePR(eta=eta_oth)
        self.V_go = SparsePR(eta=eta_go, init=0.5)          # neutral prior = rbar
        self.rbar, self.rbar_n = 0.5, 0
        self.learn_turn = learn_turn
        self.tau_w, self.urg_gate, self.close_on_scene = tau_w, urg_gate, close_on_scene
        self.nogo_late = nogo_late
        self.oth_kind = oth_kind        # turn-taking features crossed with the thought kind (no scene/reply cross-talk)
        self.th = None                 # the open thought
        self.pend = []                 # pending P_oth predictions: (t_made, keys, p)
        self.elig = None               # (keys, t_stop) of the last released thought, until the partner's reaction
        self.spoke_since_line = False
        self.R_line = False
        self.scene_react = None
        self.prev_scene_on = False
        self.prev_att = None
        self.REL = False
        self.th_log = []               # one record per thought (kind, t_open, tau_rel, k_rel, e_rel, outcome, ...)
        self.tstats = dict(released=0, blocked_nogo=0, closed_other=0, closed_tmax=0, scene_opened=0,
                           scene_spoke=0, reply_spoke=0)

    # ---------------------------------------------------------------- features of a thought (held PFC line code)
    def _feats(self, th):
        P = th['st']['P']
        f = [('kind', th['kind']), ('R', int(self.R_line))]
        f += [('p', int(i)) for i in sorted(set(P))]
        if P:
            f += [('p1', int(P[0])), ('pl', int(P[-1]))]
        if th['st']['att_tutor'] is not None:
            f.append(('pt', 1))
        if getattr(self, 'eye_feats', False) and th['st']['eye'] is not None:
            f += [('e', int(i)) for i in np.flatnonzero(th['st']['eye'])]
        return f

    def _oth_keys(self, th, tau):
        tc = min(tau, 30)
        if getattr(self, 'oth_kind', False):
            return [(th['kind'], 'tau', tc)] + [(th['kind'],) + k + (tc,) for k in th['feats']]
        return [('tau', tc)] + [(k, tc) for k in th['feats']]

    def _gain(self, th):
        v = self.V_go.p(th['feats'])
        th['V'] = v
        return float(np.clip(self.g0[th['kind']] + self.beta * (v - self.rbar), 0.0, self.g_max))

    # ---------------------------------------------------------------- thoughts
    def _open(self, kind):
        if self.mute:
            return
        th = self.core.think_new(kind, self.scene, self.att, t=self.t)
        th.update(I=0.0, released=False, t_rel=None, k_rel=None, e_rel=None, tau_rel=None, outcome=None,
                  spoke=False, first_bnd=False)
        th['feats'] = self._feats(th)
        if getattr(self, 'oth_kind', False):
            th['feats'] = [k if k[0] == 'kind' else (kind,) + k for k in th['feats']]
        th['G'] = self._gain(th)
        self.th = th
        if kind == 'scene':
            self.tstats['scene_opened'] += 1

    def _close(self, outcome):
        th = self.th
        if th is None:
            return
        th['outcome'] = outcome
        self.th_log.append(dict(kind=th['kind'], t_open=th['t_open'], t_close=self.t, tau_rel=th['tau_rel'],
                                k_rel=th['k_rel'], e_rel=th['e_rel'], k=th['k'], outcome=outcome, spoke=th['spoke'],
                                G=th['G'], e0=th['e'][0] if th['e'] else None, e_max=max(th['e']) if th['e'] else None,
                                adopted=th['st'].get('ep'), p_rel=th.get('p_rel'), t_rel=th['t_rel'],
                                R=self.R_line, feats=th['feats'][:6]))
        if len(self.th_log) > 200000:
            self.th_log = self.th_log[-100000:]
        self.th = None
        self.REL = False

    def _other_onset(self):
        """another speaker starts: finalize turn predictions, the partner's reaction, close the thought."""
        if self.learning and self.learn_turn:
            for (t0, keys, p, b) in self.pend:
                if self.t - t0 <= self.L:
                    self.P_oth.learn(keys, 1.0, p, bias=b)
        self.pend = []
        if self.th is not None:
            self.tstats['closed_other'] += 1
            self._close('other')
        self.R_line = self.spoke_since_line                  # this line follows an own utterance (a reaction?)
        self.spoke_since_line = False

    def _reaction(self, r):
        """first tick of the next line after the own utterance: the partner's contingent reaction (reward r)."""
        if self.elig is None:
            return
        keys, t_stop = self.elig
        self.elig = None
        if self.t - t_stop > 50:
            return
        if self.learning and self.learn_turn:
            self.V_go.learn(keys, float(r > 0))
            self.rbar_n += 1
            self.rbar += (float(r > 0) - self.rbar) / min(self.rbar_n, 200)

    # ---------------------------------------------------------------- scorer for the body
    def _scorer(self, body, com, first):
        if self.mute:
            return np.full(len(com), -1.0), 0.0
        core = self.core
        if first or self.own_st is None:
            th = self.th
            if th is None:                                    # (v1 GO after a line without an open thought)
                th = core.think_new('reply', self.scene, self.att, t=self.t)
            if self.policy == 'v1':
                core.think_complete(th)
            core.actor_tr = []
            st = core.st_from(th)
            core.own = st
            self.own_st = st
            if self.th is not None:
                self.th['first_bnd'] = True
        prog, s_end = core.own_scores(self.own_st)
        sc = np.array([prog[self.prog2cat[y]] if y in self.prog2cat else -1.0 for y in com])
        return sc, s_end

    # ---------------------------------------------------------------- the intention (one tick)
    def _intend(self):
        th_rel = lambda x: x is not None and x['released']
        th = self.th
        b = self.body
        busy = b.SPK or b.Hp != -1 or ((b.GO or b.R1 or bool(b.CD.any())) and not th_rel(th))
        if th is None or busy or self.in_line:
            return
        tau = self.t - th['t_open']
        if tau > self.T_MAX:
            self.tstats['closed_tmax'] += 1
            self._close('tmax')
            return
        pol = self.policy if th['kind'] == 'reply' else 'free'    # forced policies act on replies only
        if th['released']:                                   # GO refused by the brake: keep the release asserted
            if pol in ('free', 'asap') and tau > 0:
                self.core.think_step(th)
            self.REL = True
            return
        if pol.startswith('budget:'):
            kb = int(pol.split(':')[1])
            if th['k'] < kb and tau > 0:
                self.core.think_step(th)
            if th['k'] >= kb or th['done']:
                self.REL = True
            return
        if pol == 'asap':
            if tau > 0:
                self.core.think_step(th)
            self.REL = True
            return
        if pol == 'v1':
            if not th['done'] and tau > 0:
                self.core.think_step(th)
            return
        # free: think, accumulate, check the learned brake (the opening tick is step 0, done in think_new)
        e = self.core.think_step(th) if (not th['done'] and tau > 0) else th['e'][-1]
        u = self.u[th['kind']]
        if self.urg_gate:                                   # urgency only where speaking has paid (learned)
            u *= min(1.0, th['V'] / 0.5)
        if th['kind'] == 'reply' and self.R_line:          # a reaction to my own utterance: no reply is owed
            u *= getattr(self, 'u_R1', 1.0)
        th['I'] += th['G'] * e + u
        keys = self._oth_keys(th, tau)
        prior = 0.0 if (th['kind'] == 'reply' and tau <= self.tau_w) else 1.0   # innate transition window
        p_o = self.P_oth.p(keys, prior)
        if self.learning and self.learn_turn:
            self.pend.append((self.t, keys, p_o, prior))
        th['p_o'] = p_o; th['prior'] = prior
        late = th['kind'] == 'reply' and tau > self.tau_w
        if late and not getattr(self, 'late_release', True):
            return                                           # hard transition window: no late reply
        if th['I'] >= self.theta_I:
            if p_o < (self.nogo if prior == 0.0 else self.nogo_late):
                self.REL = True
            else:
                self.tstats['blocked_nogo'] += 1

    # ---------------------------------------------------------------- one tick
    def step(self, obs):
        win = obs['ear']
        E = float(np.asarray(win).sum())
        self._E_now = E
        s = self.ear.step(win)
        f = s[self.F0]
        had_scene = self.base is not None
        self.scene, self.att = self._parse_eye(obs['eye'])
        core = self.core
        r = obs.get('reward', 0.0)
        if getattr(self, 'no_react', False) and self.elig is not None and self.t - self.elig[1] > 50:
            if self.learning and self.learn_turn:           # the expected reaction did not come: target 0
                self.V_go.learn(self.elig[0], 0.0)
                self.rbar_n += 1
                self.rbar += (0.0 - self.rbar) / min(self.rbar_n, 200)
            self.elig = None
        # finalize old turn predictions (window passed without another speaker)
        if self.pend and self.t - self.pend[0][0] > self.L:
            keep = []
            for (t0, keys, p, b) in self.pend:
                if self.t - t0 > self.L:
                    if self.learning and self.learn_turn:
                        self.P_oth.learn(keys, 0.0, p, bias=b)
                else:
                    keep.append((t0, keys, p, b))
            self.pend = keep
        if not self._own_voice():
            if E > 0 and self.E_prev == 0 and not self.in_line:
                self._reaction(r)
                self._other_onset()
                sr = getattr(self, 'scene_react', None)
                if sr is not None and self.t - sr[0] <= 10:      # a contingent reaction to the own spontaneous speech:
                    core.prev = dict(items=[], eye=sr[1], att=sr[2], code=core.hip.dg(core._cue([], sr[1])),
                                     next=None, ep=-1, own=False, scene_thought=True)   # heard in the scene context
                self.scene_react = None
                self.in_line = True; self.clk = 0
                core.heard_onset(t_now=self.t)
                self.st = core.begin_line(self.scene, self.att)
            elif self.in_line and E > 0:
                self.clk += 1
            if self.in_line and E > 0 and self.clk % 3 == 2:
                c = self._detect(f)
                if c is not None:
                    self.stats['items'] += 1
                    if self.st['att_tutor'] is None and self.att is not None:
                        self.st['att_tutor'] = self.att
                    core.line_item(self.st, c, learn=self.learning)
            if self.in_line and E == 0:
                self.in_line = False
                core.line_end(self.st, learn=self.learning, store=self.learning)
                self.stats['lines'] += 1
                self.st = None
                self._open('reply')
        elif self.in_line:
            self.in_line = False
            core.line_end(self.st, learn=self.learning, store=self.learning)
            self.st = None
        # the scene disappears or changes: the old thought is dropped (context switch)
        if self.close_on_scene and self.th is not None and ((had_scene and self.base is None) or self.base_changed) \
                and not self.th['released']:
            self._close('context')
        # scene / pointing onset in silence -> a scene thought (self-initiated speech)
        scene_on = self.base is not None
        new_scene = scene_on and (not had_scene or self.base_changed)
        new_point = self.att is not None and self.prev_att is None
        if (new_scene or new_point) and self.th is None and not self.in_line and not self._own_voice() and E == 0:
            self._open('scene')
        self.prev_att = self.att
        self.REL = False
        self._intend()
        self.body.REL = self.REL
        force = self.body.step(E, scorer=self._scorer, reward=r)
        b = self.body
        if b.GO and self.th is not None and not self.th['released']:
            th = self.th
            th['released'] = True; th['t_rel'] = self.t; th['k_rel'] = th['k']
            th['e_rel'] = th['e'][-1] if th['e'] else None; th['tau_rel'] = self.t - th['t_open']
            th['p_rel'] = (th.get('p_o'), th.get('prior'), round(th['I'], 3), round(th['G'], 3))
            self.tstats['released'] += 1
        if b.Y >= 0 and self.own_st is not None:
            c = self.prog2cat.get(b.Y)
            if c is not None:
                core.own_advance(self.own_st, c); self.stats['own'] += 1
            if self.th is not None:
                self.th['spoke'] = True
        if b.stop is not None and b.stop != 'babble':
            spoke = self.own_st is not None and len(self.own_st.get('items', [])) > 0
            if self.own_st is not None:
                core.own_stop(self.own_st, b.stop == 'END')
                core.own_finish(self.own_st, t_end=self.t)
            if spoke:
                self.spoke_since_line = True
                th = self.th
                if th is not None:
                    self.elig = (list(th['feats']), self.t)
                    if th['kind'] == 'scene' and th['st']['eye'] is not None:
                        self.scene_react = (self.t, th['st']['eye'], th['st']['att'])
                    self.tstats['scene_spoke' if th['kind'] == 'scene' else 'reply_spoke'] += 1
            self.own_st = None
            if self.th is not None and (self.policy != 'v1' or self.th['first_bnd']):
                self._close('spoke' if spoke else 'silent')
        if r and self.learning:
            core.reward(r)
        self.mouth_ago = 0 if force.any() else getattr(self, 'mouth_ago', 99) + 1
        self.E_prev = E
        self.t += 1
        return force.astype(np.float32)

    def _own_voice(self):
        if getattr(self, 'own_tail', False):     # CD also runs after a GO whose first boundary was refused: not a voice
            return self.body.SPK or getattr(self, 'mouth_ago', 99) <= 2
        return self.body.SPK or bool(self.body.CD.any())

    # ---------------------------------------------------------------- eye (develop's, + a scene-change flag)
    def _parse_eye(self, eye):
        e = np.asarray(eye).astype(bool)
        self.base_changed = False
        if not e.any():
            if self.base is not None and getattr(self.core.c, "dialogue_reset", False):
                self.core.prev = None; self.core.prompt_saved = None
            self.base = None
            self._prev_eye = e
            return None, None
        if self.base is None:
            self.base = e.copy()
        extra = e & ~self.base
        missing = self.base & ~e
        if missing.sum() > 2:
            self.base = e.copy(); extra[:] = False
            self.base_changed = True
        elif getattr(self, 'life_eye', False) and getattr(self, '_E_now', 1.0) == 0 and not self.in_line \
                and not self._own_voice():
            pe = getattr(self, '_prev_eye', None)
            new_bits = extra & ~pe if pe is not None else extra
            if new_bits.sum() > 2:                            # something NEW appeared in silence (person, object);
                self.base = self.base | new_bits             # bits that persist from a line (pointing) stay extra
                extra = extra & ~new_bits
                self.base_changed = True
        self._prev_eye = e
        att = None
        if extra.sum() >= 4:
            best, bj = 0.0, -1
            for j, c in enumerate(self.att_codes):
                jac = (c & extra).sum() / max((c | extra).sum(), 1)
                if jac > best:
                    best, bj = jac, j
            if best >= 0.6:
                att = bj
            elif len(self.att_codes) < 3:
                self.att_codes.append(extra.copy()); att = len(self.att_codes) - 1
        return self.base, att
