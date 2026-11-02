# -*- coding: utf-8 -*-
"""ARCH_v2 synthesis: the cognitive core = tune's core (develop + awake recall) + think's staged thoughts, plus the
synthesis additions.  Every addition is OFF by default; with all defaults this core is ThinkCore (== develop/tune's
Core in think_k=3, ior=0 mode, verified bit-for-bit by the equivalence test).

Additions (cfg keys):
  CT  continuous thought (PFC-1 without a register):  ct=True
      The thought holds item assemblies with activation a_i in [0, 1] and the order of the line that entered last
      (its successor links).  No clear, no load:
        * another speaker's line ends:  a <- ct_lam * a   (competition from the new input: old content fades)
                                        a_i <- max(a_i, g_i) for the line's items, g_i = ct_gain * w_i
                                        w_i = 1, or (ct_surp > 0) max(ct_surp, surprise_i): predicted items enter weakly
                                        at most ct_k items stay >= ct_theta (k-WTA: the weakest are pushed down)
        * the own line ends:            a_i <- max(a_i, ct_own) for its items (efference-predicted: default 0)
        * every tick (silence too):     a <- a * exp(-dt / ct_tau)  (adaptation; ct_tau = 0: no fading)
      The readout's context P = the items with a_i >= ct_theta (ordered by the dominant line); the copy chain = the
      dominant line's held items in order; recall cue = the dominant line's stored key.  Hippocampal NEXT links still
      follow the sequence of heard lines (last_rec).  With ct_lam < ct_theta, ct_tau = 0, ct_surp = 0, ct_own = 0 the
      thought equals develop's previous-line register exactly (equivalence test), so every behavioural difference
      comes from the continuous parameters.
  REC recency in recall:  hip_rho (score = overlap * sigma^hip_rho; develop = 0.5)
  RET retinotopic attention (bench eye_layout='retino'): seg=240, rel=True: primes, recognition, match and the focus
      keys F live in the slot-relative frame (one map per slot position, the same weights for every slot).
  NAME naming target ("say the name of what I attend to"): name_ctx > 0: a heard line that contains a word recognised
      at the attended slot is also learned from the attended object alone (prompt-free scene context, the context of
      a self-initiated scene thought), at rate name_ctx.
"""
from __future__ import annotations

import numpy as np

from core import BOS, NONE
from core_think import ThinkCore, TCfg
from rules import OneShotChain, Outstar, pair_keys


class SCfg(TCfg):
    EXTRA = dict(ct=False, ct_lam=0.3, ct_theta=0.5, ct_tau=0.0, ct_k=12, ct_own=0.0, ct_surp=0.0, ct_gain=1.0,
                 ct_assoc=0.0, ct_assoc_eta=0.02, ct_fat=0.1, ct_fat_tau=20.0,
                 hip_rho=0.5, seg=0, rel=False, name_ctx=0.0, name_vr_only=True)

    def __init__(self, **kw):
        for k, v in self.EXTRA.items():
            setattr(self, k, v)
        base = {k: v for k, v in kw.items() if k not in self.EXTRA}
        super().__init__(**base)
        for k, v in kw.items():
            if k in self.EXTRA:
                setattr(self, k, v)


class SynthCore(ThinkCore):
    def __init__(self, cfg):
        super().__init__(cfg)
        c = self.c
        V = c.V
        self.now = 0
        # ---- continuous thought
        self.T_a = np.zeros(V, np.float32)
        self.T_dom = None              # record of the line that entered last with full strength (order + key)
        self.T_t = 0
        self.last_rec = None           # last heard line record (hippocampal sequence, independent of the thought)
        self._replaying = False
        self.ct_stats = dict(updates=0, faded_empty=0, weak_entries=0, residue_items=0, views=0)
        self.T_W = None
        self.T_f = np.zeros(V, np.float32)        # fatigue trace of held items (wandering)
        if c.ct and c.ct_assoc > 0:
            self.T_W = np.zeros((V, V), np.float32)
        # ---- retinotopic, slot-relative frame
        if c.seg <= 0:
            c.seg = c.n_eye // 3
        if c.rel:
            assert c.retino, 'rel needs retino'
            self.prime = Outstar(V + 1, c.seg, m_min=c.eta_prime)
            self.eye_rate = np.zeros(c.seg, np.float32)
        self.name_stats = dict(lines=0, items=0)
        self.last_line_surprise = None

    # ================================================================ retina helpers
    def _seg(self, s):
        seg = self.c.seg
        return slice(s * seg, (s + 1) * seg)

    def _rel(self, eye, s):
        return eye[self._seg(s)]

    def _match(self, Pi, eye):
        c = self.c
        if not c.rel:
            return super()._match(Pi, eye)
        out = np.zeros(3, np.float32)
        if len(Pi) == 0:
            return out
        Wp = self._conf(Pi)                                  # (|Pi|, seg): slot-relative expectations
        n = Wp.sum(1)
        good = n >= c.conf_min
        if not good.any():
            return out
        Wg, ng = Wp[good], n[good]
        for s in range(3):
            e = self._rel(eye, s)
            out[s] = float(((Wg & e[None, :]).sum(1) / ng).max())
        return out

    def _recognize(self, eye, s):
        c = self.c
        if not c.rel:
            return super()._recognize(eye, s)
        key = (s, eye.tobytes())
        hit = self._rec_cache.get(key)
        if hit is not None and hit[0] == self.prime_version:
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
        self._rec_cache[key] = (self.prime_version, out)
        return out

    def _slot_bits(self, eye, s):
        if not self.c.rel:
            return super()._slot_bits(eye, s)
        e = self._rel(eye, s)
        return np.nonzero(e)[0]                              # slot-relative bit ids (the same code at every slot)

    # ================================================================ keys (ThinkCore's, + bag, + relative focus)
    def _keys(self, st, cands):
        c = self.c
        n = len(st['items'])
        L = st['items'][-1] if st['items'] else BOS
        P = list(dict.fromkeys(st.get('Pbag') if st.get('Pbag') is not None else st['P']))
        g = {'P': P, 'L': [L], 'N': [min(n, 6)], 'FAM': [st['fam']]}
        if st['P']:
            g['P1'] = [st['P'][0]]; g['PL'] = [st['P'][-1]]
        elif c.nop_keys and st.get('kind') == 'scene':
            g['NOP'] = [1]
        for k in ('P', 'N', 'C'):
            cand = cands.get(k, (None, -1))[0]
            g['C' + k] = [cand if cand is not None else NONE]
            if c.gate_freq:
                n_c = self.item_n[cand] if (cand is not None and 0 <= cand < c.V) else -1
                g['F' + k] = [-1 if cand is None else (0 if n_c < 20 else 1 if n_c < 300 else 2)]
        att = st['att']
        if c.focus and att is not None and st['eye'] is not None:
            if c.rel:
                g['F'] = [('r', int(b)) for b in self._slot_bits(st['eye'], att)]
            else:
                narrow = (not c.retino) and c.rf_focus and self.rf[att].sum() >= 8
                g['F'] = [(att, int(b), 'r') if narrow else (att, int(b)) for b in self._slot_bits(st['eye'], att)]
            g['A'] = [att]
        else:
            g['A'] = [NONE]
        if c.vis_rec and att is not None and st['eye'] is not None and not st.get('no_vr'):
            if 'VR' not in st:
                st['VR'] = [int(i) for i in self._recognize(st['eye'], att)]
            g['VR'] = st['VR']
        if st.get('m') is not None:
            g['M'] = [int(st['m'] >= 0.5) + int(st['m'] >= 0.8)]
        role = 'own' if (self.prev is not None and self.prev.get('own')) else 'oth'
        pairs = [('P', 'M'), ('L', 'M'), ('P', 'L'), ('P', 'N'), ('L', 'N'), ('P1', 'N'), ('PL', 'N'),
                 ('P', 'CP'), ('L', 'CP'), ('N', 'CP'), ('P', 'CN'), ('L', 'CN'), ('FAM', 'CN'),
                 ('P', 'CC'), ('L', 'CC'), ('FAM', 'CC'), ('P', 'FAM'), ('N', 'FAM'),
                 ('P', 'A'), ('L', 'A'), ('N', 'A'), ('P', 'F'), ('L', 'F'), ('P', 'VR'), ('L', 'VR'), ('N', 'VR'),
                 ('P', 'FP'), ('L', 'FP'), ('N', 'FP'), ('FAM', 'FN'), ('P', 'FN'), ('FAM', 'FC'), ('P', 'FC'), ('L', 'FC')]
        if 'NOP' in g:
            pairs = pairs + [('NOP', 'VR'), ('NOP', 'F'), ('NOP', 'A'), ('NOP', 'L'), ('NOP', 'CC'), ('NOP', 'FC')]
        keys = pair_keys(g, pairs)
        if c.role_units:
            keys = [(role,) + k if len(k) == 4 else (k[0] + '|' + role, k[1]) for k in keys]
        return keys

    # ================================================================ recency in recall
    def _recall_next(self, th):
        if th['code'] is None or self.hip.n == 0 or not self.c.hippo:
            return -1, 0.0
        ov = self.hip.overlaps(th['code'])
        sig = np.maximum(self.hip.sig[:self.hip.n], 0)
        rho = self.c.hip_rho
        sc = ov * (np.sqrt(sig) if rho == 0.5 else sig ** rho)
        for v in th['visited']:
            if 0 <= v < len(sc):
                sc[v] = -1
        e = int(np.argmax(sc))
        if sc[e] < 0 or ov[e] < self.hip.theta or self.hip.sig[e] <= 0.05:
            return -1, float(ov[e]) if len(ov) else 0.0
        return e, float(ov[e])

    # ================================================================ continuous thought
    def _ct_decay(self):
        c = self.c
        if c.ct_tau > 0 and self.now > self.T_t:
            self.T_a *= np.float32(np.exp(-(self.now - self.T_t) / c.ct_tau))
        self.T_t = self.now

    def _ct_view(self):
        """the thought as the readout's context record (items in the dominant line's order + the held bag)."""
        c = self.c
        self._ct_decay()
        held = self.T_a >= c.ct_theta
        if self.T_W is not None and c.ct_assoc > 0 and held.any():
            pass                                              # association spreading acts in silence (tick_thought)
        if not held.any() or self.T_dom is None:
            if self.T_dom is not None:
                self.ct_stats['faded_empty'] += 1
            return None
        dom = self.T_dom
        items = [i for i in dom['items'] if 0 <= i < c.V and held[i]]
        inl = set(items)
        extra = [int(i) for i in np.flatnonzero(held) if int(i) not in inl]
        extra.sort(key=lambda i: -float(self.T_a[i]))
        bag = list(dict.fromkeys(items)) + extra               # line order first (bit-identical to develop's P)
        self.ct_stats['views'] += 1
        self.ct_stats['residue_items'] += len(extra)
        if not bag:
            return None
        v = dict(items=items if items else bag, bag=bag, eye=dom['eye'], att=dom['att'],
                 code=dom['code'], next=dom.get('next'), ep=dom.get('ep', -1), own=False, ct=True)
        return v

    def _ct_input(self, rec, surprise=None):
        """another speaker's completed line flows into the thought (no clear, no load)."""
        c = self.c
        self._ct_decay()
        items = [i for i in rec['items'] if 0 <= i < c.V]
        if not items:
            return
        g = np.full(len(items), c.ct_gain, np.float32)
        if c.ct_surp > 0 and surprise is not None:
            g = c.ct_gain * np.maximum(np.asarray(surprise, np.float32)[:len(items)], c.ct_surp)
        strength = float(np.clip(np.mean(g), 0.0, 1.0))
        self.T_a *= np.float32(1.0 - (1.0 - c.ct_lam) * strength)   # competition: held content fades in proportion
        for i, gi in zip(items, g):
            if gi > self.T_a[i]:
                self.T_a[i] = gi
        if float(np.mean(g)) >= c.ct_theta:
            self.T_dom = rec
        else:
            self.ct_stats['weak_entries'] += 1
        held = np.flatnonzero(self.T_a >= c.ct_theta)
        if len(held) > c.ct_k:                                # k-WTA: only ct_k items stay above threshold
            order = held[np.argsort(-self.T_a[held], kind='stable')]
            self.T_a[order[c.ct_k:]] = np.minimum(self.T_a[order[c.ct_k:]], c.ct_theta * 0.99)
        if self.T_W is not None:                              # co-held items associate (slow, normalised)
            h = self.T_a >= c.ct_theta
            hi = np.flatnonzero(h)
            if len(hi) > 1:
                self.T_W[np.ix_(hi, hi)] += c.ct_assoc_eta
                np.fill_diagonal(self.T_W, 0)
                s = self.T_W[hi].sum(1, keepdims=True)
                self.T_W[hi] /= np.maximum(s, 1.0)
        self.ct_stats['updates'] += 1

    def tick_thought(self):
        """long silence: the thought keeps running (never idle-empty).  Held items fatigue, items associated with
        them rise (learned co-holding links T_W: favourite thoughts), the strongest item is renormalised to 1
        (homeostasis) and at most ct_k items stay above threshold.  Called by the agent every `wander` quiet ticks
        once the silence is longer than `wander_after` (beyond any answer window)."""
        c = self.c
        if not c.ct or self.T_W is None or c.ct_assoc <= 0:
            return
        self._ct_decay()
        a = self.T_a
        if a.max() <= 0:
            return
        self.T_f += (a - self.T_f) / c.ct_fat_tau
        a *= (1.0 - c.ct_fat * self.T_f)
        spread = c.ct_assoc * (a @ self.T_W)
        np.maximum(a, spread, out=a)
        m = float(a.max())
        if m > 0:
            a /= m
        held = np.flatnonzero(a >= c.ct_theta)
        if len(held) > c.ct_k:
            order = held[np.argsort(-a[held], kind='stable')]
            a[order[c.ct_k:]] = np.minimum(a[order[c.ct_k:]], c.ct_theta * 0.99)
        self.ct_stats['wander'] = self.ct_stats.get('wander', 0) + 1

    def thought_items(self, thr=None):
        thr = self.c.ct_theta if thr is None else thr
        return [int(i) for i in np.flatnonzero(self.T_a >= thr)]

    def begin_line(self, eye, att):
        c = self.c
        if c.ct and not self._replaying and not (self.prev is not None and self.prev.get('scene_thought')):
            self.prev = self._ct_view()
        st = super().begin_line(eye, att)
        if c.ct and self.prev is not None and 'bag' in self.prev:
            st['Pbag'] = self.prev['bag']
        return st

    def think_new(self, kind, eye, att, t=0):
        c = self.c
        if c.ct and kind == 'reply' and not self._replaying and \
                not (self.prev is not None and self.prev.get('scene_thought')):
            self.prev = self._ct_view()
        th = super().think_new(kind, eye, att, t=t)
        prev = self.prev if kind == 'reply' else None
        if self.c.ct and prev is not None and 'bag' in prev:
            th['st']['Pbag'] = prev['bag']
        return th

    def st_from(self, th):
        st = super().st_from(th)
        if 'Pbag' in th['st']:
            st['Pbag'] = th['st']['Pbag']
        return st

    def own_finish(self, st, t_end=None):
        c = self.c
        if c.ct:
            if c.ct_own > 0 and st.get('items'):
                self._ct_decay()
                for i in st['items']:
                    if 0 <= i < c.V and c.ct_own > self.T_a[i]:
                        self.T_a[i] = c.ct_own
            return
        return super().own_finish(st, t_end=t_end)

    # ---------------------------------------------------------------- line end: Core.line_end + relative primes
    def line_end(self, st, learn=True, store=True):
        c = self.c
        if learn and c.learn_listen and self.prev is not None and st['items']:
            self._learn_step(st, self.END)
            if c.ct_surp > 0:
                st.setdefault('surp', [])
        items, eye = st['items'], st['eye']
        if learn and eye is not None and eye.any():
            self.eye_n += 1
            ev = eye.astype(np.float32)
            if c.rel:
                ev = np.mean([ev[self._seg(s)] for s in range(3)], 0)
            self.eye_rate += (ev - self.eye_rate) * max(1.0 / self.eye_n, 0.01)
        if learn and eye is not None and items:
            Pi = self._items(items)
            a_ext, a_m = st['att_tutor'], st['att_match']
            if a_ext is None and a_m is None and c.match and c.self_ground:
                ms = self._match(Pi, eye)
                srt = np.sort(ms)
                if srt[-1] >= c.match_min and srt[-1] - srt[-2] >= c.match_gap:
                    a_m = int(np.argmax(ms))
            a = a_ext if a_ext is not None else a_m
            if c.att_learn and a is not None:
                self.attD.learn(Pi, np.eye(3, dtype=np.float32)[a])
            a_p = a_ext if a_ext is not None else (a_m if c.prime_internal else None)
            if c.match and a_p is not None:
                src = self._psrc(a_p, Pi)
                if c.rel:
                    self.prime.learn(src, self._rel(eye, a_p).astype(np.float32))
                    self.prime_version += 1
                elif c.retino:
                    sg = self._seg(a_p); J = np.arange(c.n_eye)[sg]
                    self.prime.learn(src, eye[sg].astype(np.float32), J=J)
                    self.prime_version += 1
                else:
                    self.prime.learn(src, eye.astype(np.float32))
                    self.prime_version += 1
                    self.rf_age += 1
                    if self.rf_age % 50 == 0:
                        self._update_rf()
        if learn and c.name_ctx > 0 and items and eye is not None and st.get('att') is not None and \
                self.prev is not None and c.learn_listen:
            self._name_learn(st)
        sp = st.get('surp')
        self.last_line_surprise = float(np.mean(sp)) if sp else None
        self.end_line(st, store=store)

    def _name_learn(self, st):
        """naming target: the heard line names the attended object -> learn it from the attended object alone."""
        c = self.c
        att, eye = st['att'], st['eye']
        vr = set(int(i) for i in self._recognize(eye, att))
        items = st['items']
        if not vr or not any(i in vr for i in items):
            return
        sc = dict(items=[], eye=eye, att_tutor=st.get('att_tutor'), used={'P': set(), 'N': set(), 'C': set()},
                  att_match=st.get('att_match'), kind='scene', no_vr=False, P=[], src={'P': OneShotChain([])},
                  fam=0, att=att, m=None, VR=sorted(vr))
        for it in list(items) + [self.END]:
            cands = self._learn_step(sc, it, g=c.name_ctx)
            if it != self.END:
                self._advance(sc, it, cands)
        self.name_stats['lines'] += 1
        self.name_stats['items'] += len(items)

    # ---------------------------------------------------------------- end_line with the thought
    def _learn_step(self, st, target, g=1.0, eta=None):
        """Core._learn_step verbatim + the heard item's surprise 1 - p(target) (a by-product of the prediction)."""
        c = self.c
        A, d, gd, total, cands = self._drives(st)
        if 0 <= target < c.V and st.get('kind') != 'scene':
            st.setdefault('surp', []).append(float(1.0 - np.clip(total[target], 0, 1)))
        if c.use_gen and 0 <= target <= c.V:
            tvec = np.zeros(c.V + 1, np.float32); tvec[target] = 1.0
            J = np.nonzero((tvec > 0) | (np.clip(total, 0, 1) > 1e-4))[0]
            self.gen.learn(A, tvec[J], total[J], J=J, g=g, eta=eta, grp=self._g(A))
        tg, J = [], []
        for j, k in enumerate(('P', 'N', 'C')):
            cand = cands.get(k, (None, -1))[0]
            if cand is not None:
                J.append(j); tg.append(1.0 if cand == target else 0.0)
        if J:
            Ag = st['_Ag']
            self.gate.learn(Ag, np.array(tg, np.float32), gd[J], J=J, g=g,
                            eta=None if eta is None else eta * c.eta_g / c.eta, grp=self._g(Ag))
        self.stats['learn_events'] += 1
        self.n_learn += 1
        return cands

    def line_item(self, st, it, learn=True):
        """perception of a heard item; when the readout does not learn (frozen copy, no context) the surprise is
        computed read-only so that the turn learners see the same feature in use as in training."""
        c = self.c
        if not (learn and c.learn_listen and self.prev is not None) and 0 <= it < c.V and self.prev is not None:
            total, _ = self._total_ro(st)
            st.setdefault('surp', []).append(float(1.0 - np.clip(total[it], 0, 1)))
        super().line_item(st, it, learn)

    def end_line(self, st, store=True):
        c = self.c
        if not c.ct:
            return super().end_line(st, store=store)
        items = st['items']
        if c.ctx:
            x = np.zeros(c.V + c.n_eye, bool)
            x[self._items(items)] = True
            if st['eye'] is not None:
                x[c.V:] = st['eye']
            self.ctx.step(x)
        code = self.hip.dg(self._cue(items, st['eye']))
        rec = dict(items=list(items), eye=st['eye'], att=st['att'], code=code, next=None, ep=-1)
        if store and c.hippo and items:
            ep = self.hip.store(code, rec)
            self.next_of[ep] = -1
            self.hip.tick(1)
            pv = self.last_rec
            if pv is not None and pv.get('ep', -1) >= 0 and self.hip.content[pv['ep']] is pv:
                pv['next'] = ep
                self.next_of[pv['ep']] = ep; self.next_born[pv['ep']] = self.hip.born[ep]
            rec['ep'] = ep
        self.last_rec = rec
        surp = st.get('surp') if c.ct_surp > 0 else None
        if surp is not None and len(surp) < len([i for i in items if 0 <= i < c.V]):
            surp = None
        self._ct_input(rec, surprise=surp)
        self.prev = self._ct_view()
        self.stats['lines'] += 1

    # ---------------------------------------------------------------- awake recall: the thought is untouched
    def _replay(self, e, learn, eta):
        self._replaying = True
        try:
            return super()._replay(e, learn, eta)
        finally:
            self._replaying = False

    def awake_recall(self):
        if not self.c.ct:
            return super().awake_recall()
        saved = (self.T_a.copy(), self.T_dom, self.T_t, self.last_rec, self.T_f.copy())
        k = super().awake_recall()
        self.T_a, self.T_dom, self.T_t, self.last_rec, self.T_f = saved
        return k
