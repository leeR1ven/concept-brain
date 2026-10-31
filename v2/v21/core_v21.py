# -*- coding: utf-8 -*-
"""ARCH_v2.1 core = ARCH_v2's SynthCore + ROLE-AWARE LISTENING (RAL, step 1).  Every switch is OFF by default;
ral=0 is ARCH_v2 bit for bit (same calls, same order, no extra random draws).

RAL in one sentence: a heard line's learning AS AN ANSWER (the readout updates of GEN + gates made in the context of
the held thought, and its hippocampal NEXT link) is held on an eligibility trace until the line's role is known.

  * eligibility trace e_ij of a readout synapse = the R2 step it took while the line was heard (applied at once, so
    the readout behaves as if it had learned; a cancel subtracts e_ij; a commit drops the trace).  Pre, post, own trace.
  * one broadcast signal resolves it, within ral_W ticks of the line's end:
      ANSWERED    the agent's own mouth starts (efference copy), or         -> the line was a question / initiation:
                  (ral_other=1) another line starts within ral_W_other         CANCEL (and cut its NEXT link)
                  ticks (someone answered it at once: adjacency pair)
      CONTINGENT  the line reacts to the agent's own last utterance:        -> feedback or a recast: COMMIT, even if
                  it carries a reward (one at its onset), or (ral_cont 'rw')   the agent then speaks again (echoing a
                  content overlaps the efference copy (Jaccard >= ral_theta),  recast does not make it a question)
                  within ral_T ticks of the utterance
      neither     COMMIT (another speaker's reply, a statement, a line nobody answered); with ral_strict=1 a line heard
                  right AFTER an own utterance that is not contingent is CANCELLED (the partner moved on: "moving on"
                  is neither the answer to my prompt nor a punishment of what I said)
  * own utterances enter the thought under the SELF role: T_own (items, time) = the efference copy contingency reads.
    With ral_hip_own=1 they also enter the hippocampus as self-tagged episode cells, so the episodic sequence follows
    the real dialogue (Q -> my A -> L); self cells are never recalled by a heard cue and never replayed as a
    listening target (ral_replay_own = 0).  The line I answered (Q) gets L as its NEXT only when L is committed as
    its answer; otherwise Q's NEXT stays my own line (hip_own) or nothing (the link is cut).
  * ral_init > 0: a cancelled line (someone else's question / greeting) is re-learned in the prompt-free scene
    context (the context of a self-initiated thought) at gain ral_init: "what others start with in this scene".
Stats in core.ral_stats.  The symbolic harness (regress/gen_probe_v21.py) and the agent (agent_v21.py) drive it
through heard_onset / line_item / line_end (as before) + ral_mouth_onset() + ral_tick() + core.now."""
from __future__ import annotations

import numpy as np

from core_synth import SynthCore, SCfg
from rules import Delta, OneShotChain


class RCfg(SCfg):
    RAL = dict(ral=0, ral_W=12, ral_cont='r', ral_theta=0.3, ral_T=60, ral_strict=0, ral_hip_own=0,
               ral_replay_own=0.0, ral_cut=1, ral_init=0.0, ral_other=0, ral_W_other=6, ral_q=0, ral_q_eta=0.1, ral_q_thr=0.5, ral_q_prior=0.5,
               say_rec=0, say_loc=0, say_gain=1.0, say_loc_thr=0.8, say_noprompt=1, scene_held=0,
               val_act=0, eta_act=0.05, act_T=0, rec_fix=0)

    def __init__(self, **kw):
        for k, v in self.RAL.items():
            setattr(self, k, v)
        base = {k: v for k, v in kw.items() if k not in self.RAL}
        super().__init__(**base)
        for k, v in kw.items():
            if k in self.RAL:
                setattr(self, k, v)


class RecDelta(Delta):
    """Delta whose steps can be recorded: the eligibility trace of the synapses that just learned."""
    rec = None

    def learn(self, A, target, total, J=None, g=1.0, eta=None, grp=None):
        if self.rec is None or len(A) == 0 or g == 0:
            return Delta.learn(self, A, target, total, J=J, g=g, eta=eta, grp=grp)
        Jj = np.arange(self.W.shape[1]) if J is None else np.asarray(J)
        Ai = np.asarray(A)
        before = self.W[np.ix_(Ai, Jj)].copy()
        out = Delta.learn(self, A, target, total, J=J, g=g, eta=eta, grp=grp)
        d = self.W[np.ix_(Ai, Jj)] - before
        if np.any(d):
            self.rec.append((self, Ai, Jj, d))
        return out


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class RoleCore(SynthCore):
    def __init__(self, cfg):
        super().__init__(cfg)
        c = self.c
        if c.ral:                                   # same shapes and parameters; only the recording hook differs
            g, q = self.gen, self.gate
            self.gen = RecDelta(g.n_out, cap=g.W.shape[0], eta=g.eta, mature=g.mature, m_min=g.m_min, nu=g.nu)
            self.gate = RecDelta(q.n_out, cap=q.W.shape[0], eta=q.eta, mature=q.mature, m_min=q.m_min, nu=q.nu)
        self.gateR = None                           # step 2: the "say what I recognise / where it is" gate (2 outputs)
        if c.say_rec or c.say_loc:
            self.gateR = RecDelta(2, cap=self.gate.W.shape[0], eta=c.eta_g)
        self._last_keys = None
        self.act = Delta(c.V + 1, cap=self.gen.W.shape[0], eta=c.eta_act) if c.val_act else None   # step 3
        self.actor_t = -10 ** 9
        self.val_stats = dict(rewards=0, dips=0, expired=0)
        self.say_stats = dict(drives=0, rec_cands=0, loc_cands=0, learn=0)
        self.own_ep = np.zeros(c.hip_cap, bool)     # self-tagged episode cells
        self.T_own = None                           # SELF role of the thought: the own last utterance (efference copy)
        self.ral_cur = None                         # the heard line being learned (its trace)
        self.ral_pend = None                        # a finished heard line whose role is not yet known
        self.ral_q = None                           # the heard line that the own last utterance answered
        self.ral_rew_t = -10 ** 9                   # tick of the last reward
        self.storing = True                         # the agent's learning flag (frozen copies store nothing)
        self.ral_stats = dict(lines=0, after_own=0, contingent=0, cont_reward=0, cont_overlap=0, answered=0,
                              cancel=0, commit=0, unresolved=0, strict_cancel=0, cut=0, own_stored=0, init_relearn=0,
                              cont_answered=0, cont_answered_items=0, other_answered=0)
        self._ral_watch = None                      # a contingent line: would the pure veto have cancelled it?
        self.q_w = {}                               # learned question detector: item -> weight (ral_q)
        self.ral_stats.update(q_kept=0, q_pos=0, q_neg=0)

    # ------------------------------------------------------------------ the trace of a heard line
    def _ral_on(self):
        return bool(self.c.ral) and not self._replaying and self.c.learn_listen

    def heard_onset(self, t_now=None, contingent=None):
        super().heard_onset(t_now=t_now, contingent=contingent)
        if self._ral_on() and self.storing:
            self._ral_open()

    def _ral_open(self):
        p = self.ral_pend
        if p is not None:                           # a new heard line starts: the previous one was not answered by me;
            quick = self.c.ral_other and self.now - p['t_end'] <= self.c.ral_W_other
            if quick:                               # ... but if someone answers it at once, it was a question too
                self.ral_stats['other_answered'] += 1
                p['other'] = True
            self._ral_resolve(answered=bool(quick))
        buf = []
        self.gen.rec = buf; self.gate.rec = buf
        if self.gateR is not None:
            self.gateR.rec = buf
        to = self.T_own
        after_own = to is not None and to['fresh'] and self.now - to['t'] <= self.c.ral_T
        self.ral_cur = dict(buf=buf, after_own=after_own, q=self.ral_q if after_own else None,
                            own=list(to['items']) if after_own else None, own_t=to['t'] if after_own else None,
                            pv=self.last_rec, t_on=self.now)
        if to is not None:
            to['fresh'] = False                     # the efference copy is compared with the first line after it

    def _line_end_ral(self, st, learn=True, store=True):
        cur = self.ral_cur
        super().line_end(st, learn=learn, store=store)
        if self.c.ral:
            self.gen.rec = None; self.gate.rec = None
            if self.gateR is not None:
                self.gateR.rec = None
            self.ral_cur = None
            if cur is not None and learn and self._ral_on():
                self._ral_close(cur, st)

    def _ral_close(self, cur, st):
        c, S = self.c, self.ral_stats
        S['lines'] += 1
        items = list(st['items'])
        rec = self.last_rec
        cont = False
        if cur['after_own']:
            S['after_own'] += 1
            if 'r' in c.ral_cont and self.ral_rew_t >= max(cur['own_t'], cur['t_on'] - 2):   # the line carries a reward
                cont = True; S['cont_reward'] += 1
            elif 'w' in c.ral_cont and jaccard(items, cur['own']) >= c.ral_theta:
                cont = True; S['cont_overlap'] += 1
        if cont:
            S['contingent'] += 1
        link = None                                  # the NEXT link this line's end made: previous record -> line
        if rec is not None and rec.get('ep', -1) >= 0:
            pv = cur['pv']
            if pv is not None and pv.get('ep', -1) >= 0 and pv.get('next') == rec['ep']:
                link = (pv, rec['ep'])
        cur.update(t_end=self.now, cont=cont, rec=rec, link=link, items=items, eye=st.get('eye'), att=st.get('att'))
        self.ral_pend = cur
        if cont:
            self._ral_commit(cur)
            self.ral_pend = None
            self._ral_watch = (self.now, len(items))

    # ------------------------------------------------------------------ learned question detector (ral_q)
    def q_p(self, items):
        it = set(i for i in items if 0 <= i < self.c.V)
        return float(np.clip(self.c.ral_q_prior + sum(self.q_w.get(i, 0.0) for i in it), 0.0, 1.0))

    def _q_learn(self, items, y):
        """R8/PR form: w_i += eta * (y - p) / n over the line's items (pre = item heard, post = turn outcome)."""
        it = set(i for i in items if 0 <= i < self.c.V)
        if not it:
            return
        d = self.c.ral_q_eta * (y - self.q_p(it)) / len(it)
        for i in it:
            self.q_w[i] = self.q_w.get(i, 0.0) + d
        self.ral_stats['q_pos' if y else 'q_neg'] += 1

    def _q_reward(self):
        """a reward for my utterance (<= ral_T ticks after it): the heard line I answered was a question (target 1)
        and what I said was an answer (target 0).  Once per own utterance.  Pre = the line's items held in the
        thought (heard line) or the efference copy (own line); post = the reward broadcast."""
        to = self.T_own
        if to is None or to.get('q_done') or self.now - to['t'] > self.c.ral_T:
            return
        to['q_done'] = True
        q = self.ral_q
        if q is not None and q.get('items'):
            self._q_learn(q['items'], 1.0)
        self._q_learn(to['items'], 0.0)

    def ral_scene_change(self):
        pass

    # ------------------------------------------------------------------ resolution
    def _link_answer(self, p):
        """a committed line heard after my own utterance becomes the NEXT of the line I answered"""
        q, rec = p.get('q'), p.get('rec')
        if q is None or rec is None or rec.get('ep', -1) < 0 or q.get('ep', -1) < 0:
            return
        if self.hip.content[q['ep']] is q and self.hip.content[rec['ep']] is rec:
            q['next'] = rec['ep']; self.next_of[q['ep']] = rec['ep']; self.next_born[q['ep']] = self.hip.born[rec['ep']]

    def _ral_commit(self, p):
        self.ral_stats['commit'] += 1
        if p['after_own']:
            self._link_answer(p)

    def _ral_cancel(self, p):
        S = self.ral_stats
        for obj, Ai, Jj, d in p['buf']:
            obj.W[np.ix_(Ai, Jj)] -= d
        S['cancel'] += 1
        if self.c.ral_cut and p.get('link') is not None:
            pv, ep = p['link']
            if pv.get('next') == ep:
                pv['next'] = None; self.next_of[pv['ep']] = -1; S['cut'] += 1
        if self.c.ral_init > 0 and p.get('items'):
            self._ral_init_learn(p)

    def _ral_resolve(self, answered):
        p = self.ral_pend
        self.ral_pend = None
        if p is None:
            return
        c, S = self.c, self.ral_stats
        if answered and c.ral_q and not p.get('other') and self.q_p(p['items']) < c.ral_q_thr:
            S['q_kept'] += 1                        # I spoke after it, but it does not look like a question:
            S['unresolved'] += 1                    # feedback / a statement -> keep
            self._ral_commit(p)
        elif answered:
            S['answered'] += 1
            self._ral_cancel(p)
        elif c.ral_strict and p['after_own']:
            S['strict_cancel'] += 1
            self._ral_cancel(p)
        else:
            S['unresolved'] += 1
            self._ral_commit(p)

    def ral_mouth_onset(self):
        """the own mouth starts (efference): a pending heard line that ended <= ral_W ticks ago was answered by me."""
        w = self._ral_watch
        if w is not None:
            if self.now - w[0] <= self.c.ral_W:
                self.ral_stats['cont_answered'] += 1; self.ral_stats['cont_answered_items'] += w[1]
            self._ral_watch = None
        p = self.ral_pend
        if p is not None:
            self._ral_resolve(answered=self.now - p['t_end'] <= self.c.ral_W)

    def ral_tick(self):
        p = self.ral_pend
        if p is not None and self.now - p['t_end'] > self.c.ral_W:
            self._ral_resolve(answered=False)

    # ------------------------------------------------------------------ the initiate role key (optional)
    def _ral_init_learn(self, p):
        """a line someone else started (a question / greeting) is learned in the prompt-free scene context."""
        c = self.c
        eye, att = p.get('eye'), p.get('att')
        if eye is None:
            return
        vr = sorted(int(i) for i in self._recognize(eye, att)) if att is not None else []
        sc = dict(items=[], eye=eye, att_tutor=att, used={'P': set(), 'N': set(), 'C': set()}, att_match=None,
                  kind='scene', no_vr=False, P=[], src={'P': OneShotChain([])}, fam=0, att=att, m=None, VR=vr)
        saved = self.prev
        self.prev = None
        try:
            for it in list(p['items']) + [self.END]:
                cands = self._learn_step(sc, it, g=c.ral_init)
                if it != self.END:
                    self._advance(sc, it, cands)
        finally:
            self.prev = saved
        self.ral_stats['init_relearn'] += 1

    # ------------------------------------------------------------------ reward = the partner's reaction
    def reward(self, r):
        if r:
            self.ral_rew_t = self.now
            if self.c.ral and self.c.ral_q and r > 0 and self.storing:
                self._q_reward()
        c = self.c
        if not c.val_act and not c.act_T:
            return super().reward(r)
        # step 3: signed value loop.  The actor's eligibility (the own choices) expires act_T ticks after the own
        # utterance; the actor writes to its own slow table ACT (added to the own program scores only, never to the
        # prediction of what others say), not inside GEN.  r = 0 (a dip) is a legal call: delta = -V.
        if not self.actor_tr:
            return 0.0
        if c.act_T and self.now - self.actor_t > c.act_T:
            self.actor_tr = []; self.val_stats['expired'] += 1
            return 0.0
        A0 = self.actor_tr[0][0]
        delta = self.critic.update(A0, r)
        self.val_stats['rewards' if r else 'dips'] += 1
        if c.reward_actor and delta != 0:
            tab = self.act if self.act is not None else self.gen
            eta = c.eta_act if self.act is not None else c.eta_actor
            for A, y in self.actor_tr:
                if len(A):
                    tab.W[A, y] += eta * delta / len(A)
        self.actor_tr = []
        return delta

    def own_scores(self, st):
        prog, s_end = super().own_scores(st)
        if self.act is not None and '_A' in st and len(st['_A']):
            d = self.act.drive(st['_A'])
            prog = prog + d[:self.c.V]
            s_end = s_end + float(d[self.END])
        return prog, s_end

    # ------------------------------------------------------------------ own speech: the SELF role of the thought
    def own_finish(self, st, t_end=None):
        out = super().own_finish(st, t_end=t_end)
        c = self.c
        self.actor_t = self.now
        if not c.ral or not st.get('items'):
            return out
        items = [i for i in st['items'] if 0 <= i < c.V]
        if not items:
            return out
        lr = self.last_rec
        if lr is not None and not lr.get('own'):
            self.ral_q = lr                          # the heard line this utterance answered
        self.T_own = dict(items=list(items), t=self.now, fresh=True)
        if c.ral_hip_own and c.hippo and self.storing and not self._replaying:
            code = self.hip.dg(self._cue(items, st.get('eye')))
            rec = dict(items=list(items), eye=st.get('eye'), att=st.get('att'), code=code, next=None, ep=-1, own=True)
            ep = self.hip.store(code, rec)
            self.next_of[ep] = -1
            self.own_ep[ep] = True
            pv = self.last_rec
            if pv is not None and pv.get('ep', -1) >= 0 and self.hip.content[pv['ep']] is pv:
                pv['next'] = ep; self.next_of[pv['ep']] = ep; self.next_born[pv['ep']] = self.hip.born[ep]
            rec['ep'] = ep
            self.last_rec = rec                     # the next heard line follows MY line in the episodic sequence
            self.ral_stats['own_stored'] += 1
        return out

    def end_line(self, st, store=True):
        super().end_line(st, store=store)
        rec = self.last_rec
        if rec is not None and rec.get('ep', -1) >= 0:
            self.own_ep[rec['ep']] = False          # a heard cell may overwrite a self-tagged slot

    # ------------------------------------------------------------------ self-tagged cells: not recalled by heard cues
    def _recall_next(self, th):
        if not (self.c.ral and self.c.ral_hip_own) or th['code'] is None or self.hip.n == 0 or not self.c.hippo:
            return super()._recall_next(th)
        ov = self.hip.overlaps(th['code'])
        sig = np.maximum(self.hip.sig[:self.hip.n], 0)
        rho = self.c.hip_rho
        sc = ov * (np.sqrt(sig) if rho == 0.5 else sig ** rho)
        sc[self.own_ep[:self.hip.n]] = -1
        for v in th['visited']:
            if 0 <= v < len(sc):
                sc[v] = -1
        e = int(np.argmax(sc))
        if sc[e] < 0 or ov[e] < self.hip.theta or self.hip.sig[e] <= 0.05:
            return -1, float(ov[e]) if len(ov) else 0.0
        return e, float(ov[e])

    def awake_recall(self):
        """ARCH_v2's awake recall; with self cells: a self cell is never the replayed cue, and an episode whose
        successor is a self cell is replayed only with ral_replay_own > 0 (at eta_recall * ral_replay_own)."""
        c = self.c
        if not (c.ral and c.ral_hip_own):
            return super().awake_recall()
        if not (c.hippo and c.recall_n > 0 and c.learn_listen) or self.hip.n == 0:
            return 0
        saved = (self.T_a.copy(), self.T_dom, self.T_t, self.last_rec, self.T_f.copy())
        n = self.hip.n
        nx = self.next_of[:n]
        ok = nx >= 0
        ok[ok] = self.hip.born[nx[ok]] == self.next_born[:n][ok]
        ok &= self.hip.sig[:n] > 0.05
        ok &= ~self.own_ep[:n]
        if c.ral_replay_own <= 0:
            ok &= ~self.own_ep[np.where(nx >= 0, nx, 0)]
        idx = np.flatnonzero(ok)
        k = 0
        if len(idx):
            w = np.maximum(self.hip.sig[idx], 1e-6).astype(np.float64) ** c.rec_pow
            k = min(c.recall_n, len(idx))
            pick = self.rng.choice(idx, size=k, replace=False, p=w / w.sum())
            saved_prev, saved_ps = self.prev, self.prompt_saved
            for e in pick:
                nxt = self.hip.content[self.hip.content[int(e)]['next']]
                eta = c.eta_recall * (c.ral_replay_own if nxt.get('own') else 1.0)
                self._replay(int(e), True, eta)
                if c.rec_boost > 0:
                    self.hip.sig[e] += c.rec_boost * (1.0 - self.hip.sig[e])
            self.prev, self.prompt_saved = saved_prev, saved_ps
            self.stats['awake_recalls'] += k
        self.T_a, self.T_dom, self.T_t, self.last_rec, self.T_f = saved
        return k

    # ================================================================== step 2: comprehension feeds production
    def _keys(self, st, cands):
        keys = super()._keys(st, cands)
        self._last_keys = keys
        return keys

    def _grow(self):
        super()._grow()
        if self.gateR is not None:
            self.gateR.grow(self.mix.n)
        if self.act is not None:
            self.act.grow(self.mix.n)

    def _say_sets(self, st):
        """candidates of the copy route: words recognised at the attended slot (VR) and the attended slot's location
        word (inverse of word -> slot), minus what this line already holds (and, by default, the prompt's items)."""
        c = self.c
        att, eye = st.get('att'), st.get('eye')
        if att is None or eye is None:
            return [], []
        used = set(st['items'])
        if c.say_noprompt:
            used |= set(st.get('P') or [])
        rec = []
        if c.say_rec and not st.get('no_vr'):
            if 'VR' not in st:
                st['VR'] = [int(i) for i in self._recognize(eye, att)]
            rec = [int(i) for i in st['VR'] if int(i) not in used]
        loc = []
        if c.say_loc:
            ok = np.flatnonzero((self.attD.n[:c.V] >= c.attd_n) & (self.attD.W[:c.V, att] >= c.say_loc_thr))
            loc = [int(i) for i in ok if int(i) not in used]
        return rec, loc

    def _say_route(self, st, total, create):
        rec, loc = self._say_sets(st)
        st['_R'] = None
        if not rec and not loc:
            return
        keys = self._last_keys or []
        kf = [k for k in keys if not any(isinstance(x, str) and x[:2] in ('CP', 'CN', 'CC') for x in k[:4])]
        AgR = self.mix.units(kf, create=create)
        if create:
            self._grow()
        AgR = AgR[AgR < self.gateR.W.shape[0]] if len(AgR) else AgR
        gr = self.gateR.drive(AgR)
        p = np.clip(gr, 0, 1)
        g = self.c.say_gain
        for i in rec:
            total[i] += g * p[0]
        for i in loc:
            total[i] += g * p[1]
        st['_R'] = (AgR, gr, rec, loc)
        self.say_stats['drives'] += 1
        self.say_stats['rec_cands'] += len(rec); self.say_stats['loc_cands'] += len(loc)

    def _drives(self, st):
        A, d, gd, total, cands = super()._drives(st)
        if self.gateR is not None:
            self._say_route(st, total, create=True)
        return A, d, gd, total, cands

    def _total_ro(self, st):
        total, A = super()._total_ro(st)
        if self.gateR is not None:
            self._say_route(st, total, create=False)
        return total, A

    def _learn_step(self, st, target, g=1.0, eta=None):
        cands = super()._learn_step(st, target, g=g, eta=eta)
        R = st.get('_R') if self.gateR is not None else None
        if R is not None and len(R[0]):
            AgR, gr, rec, loc = R
            tg, J = [], []
            if rec:
                J.append(0); tg.append(1.0 if target in rec else 0.0)
            if loc:
                J.append(1); tg.append(1.0 if target in loc else 0.0)
            c = self.c
            self.gateR.learn(AgR, np.array(tg, np.float32), gr[J], J=J, g=g,
                             eta=None if eta is None else eta * c.eta_g / c.eta)
            self.say_stats['learn'] += 1
        return cands

    def think_new(self, kind, eye, att, t=0):
        """scene_held=1: a self-initiated (scene) thought reads the held thought (its items become the context P),
        so what the agent keeps thinking about can reach speech."""
        th = super().think_new(kind, eye, att, t=t)
        if self.c.scene_held and kind == 'scene' and self.c.ct:
            v = self._ct_view()
            if v is not None and v.get('bag'):
                th['st']['P'] = list(v['items'])
                th['st']['Pbag'] = list(v['bag'])
                th['st']['src'] = {'P': OneShotChain(th['st']['P'])}
        return th

    # ================================================================== fix of a stale visual-recognition cache
    def line_end(self, st, learn=True, store=True):
        n0 = self.eye_n
        out = RoleCore._line_end_ral(self, st, learn=learn, store=store)
        if self.eye_n != n0:
            self.eye_ver = getattr(self, 'eye_ver', 0) + 1      # eye_rate (the base-rate filter) changed
        return out

    def _recognize(self, eye, s):
        """rec_fix=1: SynthCore caches recognition per (slot, eye) and invalidates it only when the primes learn, but
        the result also depends on eye_rate (rate_max filter) and on eye_n crossing 50 (coordinator / gpuport).  The
        fixed cache is also invalidated whenever eye_rate changes."""
        c = self.c
        if not c.rec_fix or not c.rel:
            return super()._recognize(eye, s)
        key = (s, eye.tobytes())
        ver = (self.prime_version, getattr(self, 'eye_ver', 0))
        hit = self._rec_cache.get(key)
        if hit is not None and hit[0] == ver:
            return hit[1]
        V = c.V
        idx = np.nonzero(self.prime.n[:V] >= c.match_n)[0]
        out = np.zeros(0, np.int64)
        if len(idx):
            conf = self._conf(idx)
            e = self._rel(eye, s)
            n = conf.sum(1); good = n >= c.conf_min
            if good.any():
                m = (conf[good] & e[None, :]).sum(1) / n[good]
                out = idx[good][m >= c.rec_thr]
        if len(self._rec_cache) > 5000:
            self._rec_cache = {}
        self._rec_cache[key] = (ver, out)
        return out
