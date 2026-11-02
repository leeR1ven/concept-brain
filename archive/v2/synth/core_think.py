# -*- coding: utf-8 -*-
"""ThinkCore = the develop core + THOUGHTS: the context of a reply is built by a sequence of internal steps, one per
tick, and the readout's confidence is available after every step (it drives the speech intention, agent_think.py).

A thought opens when another speaker's line ends ('reply' thought about that line = develop's prompt context) or when a
new scene / pointing appears in silence ('scene' thought: no prompt; self-initiated speech).  Steps of a reply thought:
  k=0  the held prompt (PFC-1) + the tutor's pointing (develop: P, CP, A, F)                      [no recall, no VR]
  k=1  hippocampal recall #1 with develop's cue (the prompt's stored DG key)  -> NEXT / COMPLETION candidates, FAM
  k=2  comparator: per-slot prime vs scene -> internal attention (if nobody points), match level M
  k=3  visual recognition (VR) at the attended slot (analysis by synthesis)
  k>=4 serial memory search: the next-best episode for the same key (inhibition of return on visited episodes); its
       candidates are adopted when they make the readout more decisive (margin + adopt_margin)
develop's begin_line == steps 0..3 done in zero time.  A scene thought: k=0 attention = pointing, else an innate scan
(slot k mod 3); recall with the scene as the cue; VR; prompt-free keys ('NOP' pairs).
Listening uses the COMPLETED thought (all steps; the reply of the other comes much later than the thought needs, except
the 3-tick model reply), so the readouts learn the context the agent has when it has finished thinking.

Readout confidence of a context st: e = clip(s1 - max(s2, 0), 0, 1) over the first-item program scores (drive of all
routes, read-only: no MIX unit is created for thinking)."""
from __future__ import annotations

import numpy as np

from core import Core, Cfg, BOS, NONE
from rules import OneShotChain, pair_keys


class TCfg(Cfg):
    def __init__(self, **kw):
        extra = dict(think_k=8, ior=True, adopt_margin=0.05, nop_keys=True, scene_recall=True)
        for k, v in extra.items():
            setattr(self, k, v)
        base = {k: v for k, v in kw.items() if k not in extra}
        super().__init__(**base)
        for k, v in kw.items():
            if k in extra:
                setattr(self, k, v)


class ThinkCore(Core):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.think_stats = dict(opened=0, steps=0, adopted=0, ior_recalls=0)

    # ------------------------------------------------------------------ keys (develop's + no-VR flag + prompt-free keys)
    def _keys(self, st, cands):
        c = self.c
        n = len(st['items'])
        L = st['items'][-1] if st['items'] else BOS
        P = list(dict.fromkeys(st['P']))
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

    # ------------------------------------------------------------------ read-only drive (thinking creates no units)
    def _total_ro(self, st):
        c = self.c
        cands = self._cands(st)
        keys = self._keys(st, cands)
        A = self.mix.units(keys, create=False)
        A = A[A < self.gen.W.shape[0]] if len(A) else A
        if c.gate_noid:
            Ag = self.mix.units([k for k in keys if not any(isinstance(x, str) and x[:2] in ('CP', 'CN', 'CC')
                                                              for x in k[:4])], create=False)
            Ag = Ag[Ag < self.gate.W.shape[0]] if len(Ag) else Ag
        else:
            Ag = A
        d = self.gen.drive(A) if c.use_gen else np.zeros(c.V + 1, np.float32)
        pg = np.clip(self.gate.drive(Ag), 0, 1)
        total = d.copy()
        on = {'P': c.copy_prompt, 'N': c.copy_next, 'C': c.copy_comp}
        for j, k in enumerate(('P', 'N', 'C')):
            cand = cands.get(k, (None, -1))[0]
            if cand is not None and on[k] and 0 <= cand < c.V:
                total[cand] += c.gain * pg[j]
        return total, A

    def confidence(self, st):
        total, A = self._total_ro(st)
        prog = total[:self.c.V]
        j = int(np.argmax(prog))
        s1 = float(prog[j])
        s2 = float(np.partition(prog, -2)[-2]) if len(prog) > 1 else 0.0
        return float(np.clip(s1 - max(s2, 0.0), 0.0, 1.0)), s1, j

    # ------------------------------------------------------------------ thoughts
    def think_new(self, kind, eye, att, t=0):
        """kind 'reply': about self.prev (the line just heard); 'scene': no prompt (self-initiated)."""
        prev = self.prev if kind == 'reply' else None
        st = dict(items=[], eye=eye, att_tutor=att, used={'P': set(), 'N': set(), 'C': set()}, att_match=None,
                  kind=kind, no_vr=True)
        st['P'] = prev['items'] if prev else []
        st['src'] = {'P': OneShotChain(st['P'])}
        st['fam'] = 0
        st['att'] = att
        st['m'] = None
        th = dict(kind=kind, st=st, k=0, visited=set(), prev=prev, t_open=t, e=[], s1=[], y1=[], done=False,
                  code=None)
        if prev is not None and prev.get('ep', -1) >= 0:
            th['visited'].add(prev['ep'])
        if kind == 'reply' and prev is not None:
            th['code'] = prev['code']
        elif kind == 'scene' and eye is not None:
            th['code'] = self.hip.dg(self._cue([], eye))
        self.think_stats['opened'] += 1
        e, s1, y1 = self.confidence(st)
        th['e'].append(e); th['s1'].append(s1); th['y1'].append(y1)
        return th

    def _set_recall(self, st, e, ov):
        rec = self.hip.content[e]
        st['fam'] = 1 + int(ov >= 0.45) + int(ov >= 0.7)
        nxt = rec.get('next')
        c = self.c
        st['src'] = {'P': st['src']['P']}
        if c.copy_next and nxt is not None and self.hip.content[nxt] is not None:
            st['src']['N'] = OneShotChain(self.hip.content[nxt]['items'])
        if c.copy_comp:
            pb = set(st['P'])
            st['src']['C'] = OneShotChain([i for i in rec['items'] if i not in pb])
        st['ep'] = e

    def _recall_next(self, th):
        """the best not-yet-visited episode for the thought's key (WTA with inhibition of return)."""
        if th['code'] is None or self.hip.n == 0 or not self.c.hippo:
            return -1, 0.0
        ov = self.hip.overlaps(th['code'])
        sc = ov * np.sqrt(np.maximum(self.hip.sig[:self.hip.n], 0))
        for v in th['visited']:
            if 0 <= v < len(sc):
                sc[v] = -1
        e = int(np.argmax(sc))
        if sc[e] < 0 or ov[e] < self.hip.theta or self.hip.sig[e] <= 0.05:
            return -1, float(ov[e]) if len(ov) else 0.0
        return e, float(ov[e])

    def think_step(self, th):
        """one internal step (one tick).  Returns the readout confidence after it."""
        if th['done']:
            return th['e'][-1]
        c = self.c
        st = th['st']
        k = th['k'] + 1
        th['k'] = k
        self.think_stats['steps'] += 1
        scene = th['kind'] == 'scene'
        if k == 1:                                           # recall #1
            if (not scene or c.scene_recall) and c.hippo:
                e, ov = self._recall_next(th)
                if e >= 0:
                    th['visited'].add(e)
                    self.stats['recall'] += 1
                    self._set_recall(st, e, ov)
        elif k == 2:                                         # comparator -> attention
            eye = st['eye']
            if scene:
                if st['att_tutor'] is None and eye is not None:
                    st['att'] = th.get('scan', 0) % 3        # innate scan: look at a slot
            elif eye is not None and st['P']:
                Pi = self._items(st['P'])
                ms = self._match(Pi, eye) if c.match else np.zeros(3, np.float32)
                srt = np.sort(ms)
                if srt[-1] >= 0.5 and srt[-1] - srt[-2] >= 0.3:
                    st['att_match'] = int(np.argmax(ms))
                if st['att_tutor'] is None and c.att_learn:
                    d = self._attd(Pi) + c.kappa * ms
                    j = int(np.argmax(d))
                    if d[j] > 0.5:
                        st['att'] = j
                if c.match and st['att'] is not None:
                    st['m'] = float(ms[st['att']])
        elif k == 3:                                         # visual recognition at the attended slot
            st['no_vr'] = False
            st.pop('VR', None)
        else:                                                # serial memory search (IOR)
            if scene and st['att_tutor'] is None and st['eye'] is not None and k <= 5:
                th['scan'] = th.get('scan', 0) + 1           # scene thought: look at the next slot
                old = (st['att'], st.get('VR'))
                st['att'] = th['scan'] % 3; st.pop('VR', None)
                e_new = self.confidence(st)[0]
                if e_new <= th['e'][-1]:
                    st['att'] = old[0]
                    if old[1] is not None:
                        st['VR'] = old[1]
                    else:
                        st.pop('VR', None)
            elif c.ior and c.hippo:
                e, ov = self._recall_next(th)
                if e >= 0:
                    th['visited'].add(e)
                    self.think_stats['ior_recalls'] += 1
                    alt = dict(st); alt['used'] = {kk: set(v) for kk, v in st['used'].items()}
                    self._set_recall(alt, e, ov)
                    e_new = self.confidence(alt)[0]
                    if e_new > th['e'][-1] + c.adopt_margin:
                        st.update({kk: alt[kk] for kk in ('src', 'fam', 'ep')})
                        self.think_stats['adopted'] += 1
                else:
                    th['done'] = True
        if k >= c.think_k:
            th['done'] = True
        e, s1, y1 = self.confidence(st)
        th['e'].append(e); th['s1'].append(s1); th['y1'].append(y1)
        return e

    def think_complete(self, th, upto=None):
        upto = self.c.think_k if upto is None else upto
        while not th['done'] and th['k'] < upto:
            self.think_step(th)
        return th

    def st_from(self, th):
        """a fresh line context carrying the thought's results (for speaking or for listening)."""
        s = th['st']
        st = dict(items=[], eye=s['eye'], att_tutor=s['att_tutor'], used={'P': set(), 'N': set(), 'C': set()},
                  att_match=s['att_match'], kind=s['kind'], no_vr=s.get('no_vr', False), P=s['P'],
                  src=dict(s['src']), fam=s['fam'], att=s['att'], m=s['m'])
        if 'VR' in s:
            st['VR'] = s['VR']
        return st

    # ------------------------------------------------------------------ listening: the completed thought is the context
    def begin_line(self, eye, att):
        kind = 'scene' if (self.prev is not None and self.prev.get('scene_thought')) else 'reply'
        th = self.think_new(kind, eye, att)
        self.think_complete(th)
        st = self.st_from(th)
        st['att_tutor'] = att
        return st
