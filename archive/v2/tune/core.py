# -*- coding: utf-8 -*-
"""ARCH_v2-develop cognitive core (event level: detected items in, program scores out).

Areas (prototype sizes; GPU sizes in the report):
  BUF_prev / BUF_cur  ordered utterance memory (PFC-1): items + one-shot chain (R1); rotate at each line offset.
  MIX                 recruited conjunctive units (PFC-2, mixed selectivity, R3) over the step context.
  GEN                 error-driven readout MIX -> programs + END (cortex, R2).
  GATE                BG output gates (R2): COPY_prompt, COPY_next (recalled answer), COPY_comp (recall completion).
  ATT                 attention: tutor's pointing (innate gaze following) or internal = direct word->slot readout
                      (R2) + kappa * MATCH (per-slot prime of the heard words vs the scene).
  PRIME               (slot, word) -> expected eye units (R2), taught under joint attention.
  HIP                 episodic index (R4): DG k-WTA of (line bag, first, last, scene, CTX) + one-shot cells.
  CTX                 slow dialogue context (PFC-4): leaky k-WTA layer updated at line offsets (temporal context).
  CRITIC              value of the reply context; RPE = reward - V gates the actor update of own choices (R6).
Every route adds a current to the same program units; one WTA per beat (the kept body); END competes (max rule).
"""
from __future__ import annotations

import numpy as np

from rules import Delta, MixDict, Hippo, Critic, OneShotChain, Outstar, pair_keys

BOS = -1
NONE = -2


class Ctx:
    """slow context layer: leaky potentials + k-WTA (no synchronous period-2: persistence lambda > 0.5)."""

    def __init__(self, n_in, n=800, k=24, lam=0.75, fan=6, seed=0, beta=2.0, tau=200.0):
        r = np.random.default_rng(seed)
        self.idx = r.integers(0, n_in, size=(n, fan))
        self.n, self.k, self.lam = n, k, lam
        self.u = np.zeros(n, np.float32)
        self.s = np.zeros(n, bool)
        self.tie = r.random(n).astype(np.float32) * 1e-3
        self.use = np.full(n, k / n, np.float32)     # homeostatic set point: each unit's own usage raises its threshold
        self.beta, self.tau = beta, tau
        self.in_rate = np.full(n_in, 0.05, np.float32)

    def step(self, x):
        xf = x.astype(np.float32)
        if self.beta == 0:                                 # the old layer (no set point, no novelty weighting)
            drive = xf[self.idx].sum(1)
            self.u = self.lam * self.u + (1 - self.lam) * drive
            s = np.zeros(self.n, bool); s[np.argpartition(-(self.u + self.tie), self.k)[:self.k]] = True
            self.s = s
            return s
        self.in_rate += (xf - self.in_rate) / self.tau
        xw = xf / np.maximum(self.in_rate, 0.01)          # frequent inputs count less (novelty weighting)
        drive = xw[self.idx].sum(1)
        self.u = self.lam * self.u + (1 - self.lam) * drive
        sc = self.u - self.beta * np.maximum(self.use / (self.k / self.n) - 1, 0) * (self.u.mean() + 1e-6) + self.tie
        s = np.zeros(self.n, bool); s[np.argpartition(-sc, self.k)[:self.k]] = True
        self.use += (s - self.use) / self.tau
        self.s = s
        return s


class Cfg:
    def __init__(self, **kw):
        self.V = 400                 # item / program capacity
        self.n_eye = 400
        self.eta = 0.35              # GEN learning rate (normalised step)
        self.eta_g = 0.5             # gates
        self.eta_sleep = 0.2
        self.mature = 0.0            # per-source maturation (0 = off)
        self.gain = 1.0              # copy route gain into the candidate program
        self.floor = 0.25            # best score below the floor -> silence / stop
        self.max_len = 8
        # modules (developmental switches / ablations)
        self.use_gen = True
        self.copy_prompt = True
        self.copy_next = True
        self.copy_comp = True
        self.hippo = True
        self.ctx = True
        self.att_learn = True
        self.match = True            # per-slot prime vs scene: attention + MATCH signal
        self.kappa = 2.0
        self.focus = True
        self.reward_actor = True
        self.sleep_replay = True
        self.learn_listen = True
        self.retino = True           # innate retinotopy: eye bits grouped by slot (3 equal segments); False = flat eye
        self.branch_norm = False
        self.rf_focus = True         # flat eye: attention narrows to the slot's learned receptive field
        self.rf_thr = 0.6
        self.eta_prime = 0.02
        self.eta_attd = 0.02
        self.match_thr = 0.6
        self.rec_thr = 0.85
        self.vis_rec = True
        self.self_ground = True
        self.rate_max = 0.4          # expectations count only on eye units that are not usually on
        self.conf_min = 6            # a word's confident expectation must cover >= 6 eye units      # a line's own located word teaches the line's other words (attention, primes)          # recognised words at the attended slot as a context group
        self.match_n = 6
        self.attd_n = 2
        self.prime_internal = False  # primes learn only under external joint attention (internal teaching = a loop)
        self.hip_gamma = 0.9995      # per stored line
        self.hip_cap = 3000
        self.hip_ndg = 3000
        self.hip_kdg = 60
        self.mix_cap = 600000
        self.ctx_n = 800
        self.ctx_k = 24
        self.gen_mmin = 0.0          # floor of GEN plasticity under maturation (mature > 0)
        self.sleep_mode = 'strength' # 'strength' (sample by trace strength, downscale all) | 'priority'
        self.sleep_passes = 3
        self.sleep_k = 60
        self.hip_theta = 0.2
        self.hip_bigram = False      # order-sensitive episodic key (hashed ordered pairs of the line)
        self.own_rotate = False
        self.dialogue_reset = False  # a new dialogue (scene appears after a gap) clears the previous-line context
        self.role_units = False
        self.gate_noid = False
        self.gate_freq = False        # ...but the gates see the candidate's familiarity class (function vs content words)        # BG gates see the context, not which item the candidate is (generalises to new words)
        self.hip_novelty = 1.0      # a line matching a stored episode this well refreshes it instead of a new cell       # context units carry the turn role (previous line mine / the partner's)       # the own reply is a line of the dialogue (the next heard line follows it)...
        self.contingency = 10        # ...unless that line starts within this many ticks: a contingent reaction (recast)
        self.w_scene = 1.0
        self.w_ctx = 1.0
        self.ctx_lam = 0.9           # slow dialogue context: persistence per line
        self.ctx_homeo = True        # context units: homeostatic set point + novelty-weighted input (no generic state)
        # ---------------- tune: new mechanisms / exposed constants (defaults reproduce develop exactly)
        self.mat_norm = 0.0          # nu: learning-time normalisation |A|^(1-nu) * (sum_i m_i)^nu (1 = plastic units share the step)
        self.mature_g = 0.0          # gate maturation constant (0 = off)
        self.gate_mmin = 0.0
        self.eta_critic = 0.2
        self.eta_actor = 0.3
        self.wd_gen = 0.0            # weight decay of GEN per 1000 learning events, scaled by each unit's plasticity m_i
        self.wd_gate = 0.0
        self.wd_every = 200
        self.recall_n = 0            # awake recall: episodes replayed per quiet moment (0 = off)
        self.eta_recall = 0.2        # learning rate of a recalled episode (as if re-heard)
        self.rec_pow = 1.0           # sampling weight sigma^rec_pow (1 = strength/recency, 0 = uniform over stored episodes)
        self.rec_boost = 0.0         # a recalled trace is strengthened: sigma += rec_boost * (1 - sigma)
        self.split_use = 0           # split: a unit used >= split_use times ... (0 = off)
        self.split_err = 0.5         # ... that is active in >= split_n events with error >= split_err
        self.split_n = 5             # spawns fresh children keyed (parent, prompt first word, prompt last word)
        self.recruit_p = 1.0         # MIX: a new conjunction gets a unit with this probability per co-occurrence
        self.prune_age = 0           # MIX: delete units unused for this many lookups (0 = off) ...
        self.prune_every = 20000     # ... checked every this many learning events
        self.ctx_beta = 2.0          # CTX set-point strength (with ctx_homeo)
        self.ctx_tau = 200.0         # CTX usage / input-rate time constant (lines)
        self.match_min = 0.5         # comparator: grounded if best slot match >= match_min ...
        self.match_gap = 0.3         # ... and leads the second by >= match_gap
        self.att_thr = 0.5           # internal attention: attend slot j if its drive > att_thr
        self.seed = 0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)


class Core:
    def __init__(self, cfg: Cfg):
        self.c = c = cfg
        V = c.V
        self.END = V
        self.mix = MixDict(cap=c.mix_cap, recruit_p=c.recruit_p, seed=c.seed)
        self.gen = Delta(V + 1, cap=50000, eta=c.eta, mature=c.mature, m_min=c.gen_mmin, nu=c.mat_norm)
        self.gate = Delta(3, cap=50000, eta=c.eta_g, mature=c.mature_g, m_min=c.gate_mmin, nu=c.mat_norm)
        self.contra = np.zeros(50000, np.int32)          # split: contradiction events per unit
        self.splitf = np.zeros(50000, bool)              # split: unit has split (children take new learning)
        self.is_child = np.zeros(50000, bool)
        self.n_learn = 0
        self.next_of = np.full(c.hip_cap, -1, np.int64)  # awake recall: episode -> its successor cell (+ birth check)
        self.next_born = np.full(c.hip_cap, -1, np.int64)
        self.attD = Outstar(V + 1, 3, m_min=c.eta_attd)                      # word -> slot (左边 -> left)                          # direct word -> slot (左边 -> left)
        self.prime = Outstar((V if c.retino else 3 * V) + 1, c.n_eye, m_min=c.eta_prime)   # (slot, word) -> eye   # (slot, word) -> eye units
        self.actor_tr = []           # eligibility: (A, chosen output) of the own reply
        self.critic = Critic(cap=50000, eta=c.eta_critic)
        self.ctx = Ctx(V + c.n_eye, n=c.ctx_n, k=c.ctx_k, lam=c.ctx_lam, seed=c.seed + 5,
                       beta=c.ctx_beta if c.ctx_homeo else 0.0, tau=c.ctx_tau)
        groups = {'bag': V, 'first': V, 'last': V, 'scene': c.n_eye, 'ctx': self.ctx.n}
        fan = [10, 4, 4, 10, 10]
        wts = {'bag': 1.0, 'first': 1.0, 'last': 1.0, 'scene': c.w_scene, 'ctx': c.w_ctx}
        if c.hip_bigram:
            groups['bigram'] = 2 * V; fan.append(6); wts['bigram'] = 1.0
        self.hip = Hippo(groups, n_dg=c.hip_ndg, k_dg=c.hip_kdg, fan=tuple(fan), weights=wts,
                         cap=c.hip_cap, seed=c.seed + 7, gamma=c.hip_gamma, theta=c.hip_theta)
        self.prev = None             # the previous line in the dialogue (heard, or the own reply)
        self.prompt_saved = None     # the prompt the own reply answered (restored for a contingent reaction)
        self.own_end_t = None
        self.rng = np.random.default_rng(c.seed)
        self.stats = {'recall': 0, 'lines': 0, 'learn_events': 0, 'splits': 0, 'awake_recalls': 0, 'pruned': 0,
                      'quiet': 0}
        self.item_n = np.zeros(V + 1, np.int64)          # how often each item has been heard (its own usage count)
        self.rf = [np.zeros(c.n_eye, bool) for _ in range(3)]
        self._rec_cache = {}
        self.prime_version = 0
        self.eye_rate = np.zeros(c.n_eye, np.float32)      # each eye unit's own mean activity (base rate)
        self.eye_n = 0
        self.rf_age = 0

    # ------------------------------------------------------------------ helpers
    def _items(self, items):
        return np.array(sorted(set(i for i in items if 0 <= i < self.c.V)), np.int64)

    def _bag(self, items):
        v = np.zeros(self.c.V, bool)
        v[self._items(items)] = True
        return v

    def _cue(self, items, eye):
        c = self.c
        f = np.zeros(c.V, bool); l = np.zeros(c.V, bool)
        if items:
            if 0 <= items[0] < c.V:
                f[items[0]] = True
            if 0 <= items[-1] < c.V:
                l[items[-1]] = True
        out = {'bag': self._bag(items), 'first': f, 'last': l,
               'scene': eye if eye is not None else np.zeros(c.n_eye, bool),
               'ctx': self.ctx.s if c.ctx else np.zeros(self.ctx.n, bool)}
        if c.hip_bigram:
            bg = np.zeros(2 * c.V, bool)
            for a, b in zip(items[:-1], items[1:]):
                if 0 <= a < c.V and 0 <= b < c.V:
                    bg[(a * 7919 + b * 31) % (2 * c.V)] = True
            out['bigram'] = bg
        return out

    def _seg(self, s):
        seg = self.c.n_eye // 3
        return slice(s * seg, (s + 1) * seg)

    def _psrc(self, s, Pi):
        return Pi if self.c.retino else s * self.c.V + Pi

    def _conf(self, src):
        c = self.c
        cf = self.prime.conf(src, c.match_thr, c.match_n)
        if self.eye_n >= 50:
            cf &= (self.eye_rate < c.rate_max)[None, :]
        return cf

    def _match(self, Pi, eye):
        """per slot: max over the heard words of the fraction of that word's confident visual expectation present
        (outstar P(unit | word, slot) >= match_thr after >= match_n teaching events); top-down prime vs scene."""
        out = np.zeros(3, np.float32)
        if len(Pi) == 0:
            return out
        c = self.c
        for s in range(3):
            Wp = self._conf(self._psrc(s, Pi))
            if c.retino:
                Wp = Wp[:, self._seg(s)]; e = eye[self._seg(s)]
            else:
                e = eye
            n = Wp.sum(1)
            good = n >= c.conf_min
            if good.any():
                out[s] = float(((Wp[good] & e[None, :]).sum(1) / n[good]).max())
        return out

    def _recognize(self, eye, s):
        """visual recognition by analysis-by-synthesis: the words whose confident expected appearance at slot s
        (outstar prime) is present in the scene (>= rec_thr of it).  Slot-invariant code of what is attended."""
        c = self.c
        key = (s, eye.tobytes())
        hit = self._rec_cache.get(key)
        if hit is not None and hit[0] == self.prime_version:
            return hit[1]
        V = c.V
        src = np.arange(V) if c.retino else s * V + np.arange(V)
        idx = np.nonzero(self.prime.n[src] >= c.match_n)[0]
        out = np.zeros(0, np.int64)
        if len(idx):
            conf = self._conf(src[idx])
            e = eye
            if c.retino:
                conf = conf[:, self._seg(s)]; e = eye[self._seg(s)]
            n = conf.sum(1); good = n >= c.conf_min
            if good.any():
                m = (conf[good] & e[None, :]).sum(1) / n[good]
                out = idx[good][m >= c.rec_thr]
        if len(self._rec_cache) > 5000:
            self._rec_cache = {}
        self._rec_cache[key] = (self.prime_version, out)
        return out

    def _attd(self, Pi):
        """word -> slot: max over the heard words of P(slot | word) (words with enough evidence)"""
        if len(Pi) == 0:
            return np.zeros(3, np.float32)
        ok = self.attD.n[Pi] >= self.c.attd_n
        if not ok.any():
            return np.zeros(3, np.float32)
        return self.attD.W[Pi[ok]].max(0)

    def _update_rf(self):
        """slot s's receptive field = eye units that some word's (slot s) prime predicts confidently, minus units
        predicted at every slot (position-invariant parts of a code)"""
        c = self.c
        V = c.V
        W = self.prime.W
        ok = self.prime.n[:3 * V] >= c.match_n
        raw = [(W[s * V:(s + 1) * V][ok[s * V:(s + 1) * V]] >= c.rf_thr).any(0) for s in range(3)]
        common = raw[0] & raw[1] & raw[2]
        self.rf = [r & ~common for r in raw]

    def _slot_bits(self, eye, s):
        on = np.nonzero(eye)[0]
        if not self.c.retino:
            if self.c.rf_focus and self.rf[s].sum() >= 8:
                return on[self.rf[s][on]]
            return on
        seg = self.c.n_eye // 3
        return on[(on >= s * seg) & (on < (s + 1) * seg)]

    # ------------------------------------------------------------------ line context (heard or own)
    def begin_line(self, eye, att):
        """a line starts: its context = the previous line (prompt) + episodic recall + attention."""
        c = self.c
        st = dict(items=[], eye=eye, att_tutor=att, used={'P': set(), 'N': set(), 'C': set()}, att_match=None)
        prev = self.prev
        st['P'] = prev['items'] if prev else []
        st['src'] = {'P': OneShotChain(st['P'])}
        st['fam'] = 0
        if c.hippo and prev is not None and self.hip.n > 0:
            e, ov = self.hip.recall(prev['code'], exclude=prev.get('ep', -1))
            if e >= 0:
                self.stats['recall'] += 1
                rec = self.hip.content[e]
                st['fam'] = 1 + int(ov >= 0.45) + int(ov >= 0.7)
                nxt = rec.get('next')
                if c.copy_next and nxt is not None and self.hip.content[nxt] is not None:
                    st['src']['N'] = OneShotChain(self.hip.content[nxt]['items'])
                if c.copy_comp:
                    pb = set(st['P'])
                    st['src']['C'] = OneShotChain([i for i in rec['items'] if i not in pb])
        st['att'] = att
        st['m'] = None
        if eye is not None and st['P']:
            Pi = self._items(st['P'])
            ms = self._match(Pi, eye) if c.match else np.zeros(3, np.float32)
            srt = np.sort(ms)
            if srt[-1] >= c.match_min and srt[-1] - srt[-2] >= c.match_gap:
                st['att_match'] = int(np.argmax(ms))          # grounded: the comparator found the named thing
            if att is None and c.att_learn:
                d = self._attd(Pi) + c.kappa * ms
                j = int(np.argmax(d))
                if d[j] > c.att_thr:
                    st['att'] = j
            if c.match and st['att'] is not None:
                st['m'] = float(ms[st['att']])
        return st

    def _cands(self, st):
        last = st['items'][-1] if st['items'] else None
        return {k: ch.candidate(last, st['used'][k]) for k, ch in st['src'].items()}

    def _keys(self, st, cands):
        c = self.c
        n = len(st['items'])
        L = st['items'][-1] if st['items'] else BOS
        P = list(dict.fromkeys(st['P']))
        g = {'P': P, 'L': [L], 'N': [min(n, 6)], 'FAM': [st['fam']]}
        if st['P']:
            g['P1'] = [st['P'][0]]; g['PL'] = [st['P'][-1]]
        for k in ('P', 'N', 'C'):
            cand = cands.get(k, (None, -1))[0]
            g['C' + k] = [cand if cand is not None else NONE]
            if c.gate_freq:                               # candidate familiarity class (no identity): novel / rare / common
                n_c = self.item_n[cand] if (cand is not None and 0 <= cand < c.V) else -1
                g['F' + k] = [-1 if cand is None else (0 if n_c < 20 else 1 if n_c < 300 else 2)]
        att = st['att']
        if c.focus and att is not None and st['eye'] is not None:
            narrow = (not c.retino) and c.rf_focus and self.rf[att].sum() >= 8
            g['F'] = [(att, int(b), 'r') if narrow else (att, int(b)) for b in self._slot_bits(st['eye'], att)]
            g['A'] = [att]
        else:
            g['A'] = [NONE]
        if c.vis_rec and att is not None and st['eye'] is not None:
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
        keys = pair_keys(g, pairs)
        if c.role_units:                                  # turn role gates every context unit (no cross-role interference)
            keys = [(role,) + k if len(k) == 4 else (k[0] + '|' + role, k[1]) for k in keys]
        return keys

    def _g(self, A):
        return self.mix.grp[A] if self.c.branch_norm else None

    def _grow(self):
        n = self.mix.n
        for r in (self.gen, self.gate):
            r.grow(n)
        self.critic.grow(n)
        if n > len(self.contra):
            m = max(n, int(1.25 * len(self.contra)) + 1)
            for nm in ('contra', 'splitf', 'is_child'):
                a = getattr(self, nm); b = np.zeros(m, a.dtype); b[:len(a)] = a; setattr(self, nm, b)

    def _drives(self, st):
        c = self.c
        cands = self._cands(st)
        keys = self._keys(st, cands)
        A = self.mix.units(keys)
        if c.split_use > 0 and len(A):
            self._grow()
            sp = A[self.splitf[A]]
            if len(sp):                                   # split units: fresh children for this prompt frame take new learning
                P1 = st['P'][0] if st['P'] else NONE
                PL = st['P'][-1] if st['P'] else NONE
                ck = [(self.mix.key_of[int(u)], '^', P1, PL) for u in sp if int(u) in self.mix.key_of]
                n0 = self.mix.n
                Ac = self.mix.units(ck, tick=False)
                if self.mix.n > n0:
                    self._grow()
                    self.is_child[Ac] = True
                if len(Ac):
                    A = np.union1d(A, Ac)
        if c.gate_noid:                                   # gates read context only, never the candidate's identity
            Ag = self.mix.units([k for k in keys if not any(isinstance(x, str) and x[:2] in ('CP', 'CN', 'CC') for x in k[:4])])
        else:
            Ag = A
        st['_Ag'] = Ag
        self._grow()
        d = self.gen.drive(A) if c.use_gen else np.zeros(c.V + 1, np.float32)
        gd = self.gate.drive(Ag)
        pg = np.clip(gd, 0, 1)
        total = d.copy()
        on = {'P': c.copy_prompt, 'N': c.copy_next, 'C': c.copy_comp}
        for j, k in enumerate(('P', 'N', 'C')):
            cand = cands.get(k, (None, -1))[0]
            if cand is not None and on[k] and 0 <= cand < c.V:
                total[cand] += c.gain * pg[j]
        return A, d, gd, total, cands

    # ------------------------------------------------------------------ learning while hearing (predict what comes next)
    def _learn_step(self, st, target, g=1.0, eta=None):
        """target: item id or END.  Prediction = summed drive of all routes (dendritic prediction)."""
        c = self.c
        A, d, gd, total, cands = self._drives(st)
        if c.use_gen and 0 <= target <= c.V:
            tvec = np.zeros(c.V + 1, np.float32); tvec[target] = 1.0
            J = np.nonzero((tvec > 0) | (np.clip(total, 0, 1) > 1e-4))[0]
            self.gen.learn(A, tvec[J], total[J], J=J, g=g, eta=eta, grp=self._g(A))
            if c.split_use > 0 and self.gen.last_err >= c.split_err and len(A):
                Ai = A[(self.gen.use[A] >= c.split_use) & ~self.splitf[A] & ~self.is_child[A]]
                if len(Ai):
                    self.contra[Ai] += 1
                    new = Ai[self.contra[Ai] >= c.split_n]
                    if len(new):
                        self.splitf[new] = True; self.stats['splits'] += len(new)
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
        if (c.wd_gen > 0 or c.wd_gate > 0) and self.n_learn % c.wd_every == 0:
            self._decay()
        if c.prune_age > 0 and self.n_learn % c.prune_every == 0:
            self._prune()
        return cands

    def _decay(self):
        """global weight decay (the one clean-up), per path; a matured unit's synapses decay as slowly as they learn"""
        c = self.c
        n = self.mix.n
        idx = np.arange(n)
        for r, lam in ((self.gen, c.wd_gen), (self.gate, c.wd_gate)):
            if lam > 0 and n:
                f = (1.0 - lam / 1000.0 * r.plasticity(idx)) ** c.wd_every
                r.W[:n] *= f[:, None].astype(r.W.dtype)

    def _prune(self):
        """delete MIX units unused for prune_age lookups (their synapses are removed)"""
        before = len(self.mix.free)
        k = self.mix.prune(self.c.prune_age, readouts=(self.gen, self.gate, self.critic))
        freed = np.array(self.mix.free[before:], np.int64)
        if len(freed):
            self.contra[freed] = 0; self.splitf[freed] = False; self.is_child[freed] = False
        self.stats['pruned'] += k

    def _advance(self, st, item, cands):
        st['items'].append(item)
        for k, (cand, pos) in cands.items():
            if cand is not None and cand == item:
                st['used'][k].add(pos)

    def line_item(self, st, it, learn=True):
        c = self.c
        if 0 <= it < c.V:
            self.item_n[it] += 1
        if learn and c.learn_listen and self.prev is not None:
            cands = self._learn_step(st, it)
        else:
            cands = self._cands(st)
        self._advance(st, it, cands)

    def line_end(self, st, learn=True, store=True):
        c = self.c
        if learn and c.learn_listen and self.prev is not None and st['items']:
            self._learn_step(st, self.END)
        items, eye = st['items'], st['eye']
        if learn and eye is not None and eye.any():
            self.eye_n += 1
            self.eye_rate += (eye.astype(np.float32) - self.eye_rate) * max(1.0 / self.eye_n, 0.01)
        if learn and eye is not None and items:
            Pi = self._items(items)
            a_ext, a_m = st['att_tutor'], st['att_match']
            if a_ext is None and a_m is None and c.match and c.self_ground:
                ms = self._match(Pi, eye)                      # the line's own words: "苹果在左边" grounds 左边
                srt = np.sort(ms)
                if srt[-1] >= c.match_min and srt[-1] - srt[-2] >= c.match_gap:
                    a_m = int(np.argmax(ms))
            a = a_ext if a_ext is not None else a_m          # teachers: pointing or the comparator, never attD itself
            if c.att_learn and a is not None:
                self.attD.learn(Pi, np.eye(3, dtype=np.float32)[a])
            a_p = a_ext if a_ext is not None else (a_m if c.prime_internal else None)
            if c.match and a_p is not None:
                src = self._psrc(a_p, Pi)
                if c.retino:
                    sg = self._seg(a_p); J = np.arange(c.n_eye)[sg]
                    self.prime.learn(src, eye[sg].astype(np.float32), J=J)
                    self.prime_version += 1
                else:
                    self.prime.learn(src, eye.astype(np.float32))
                    self.prime_version += 1
                    self.rf_age += 1
                    if self.rf_age % 50 == 0:
                        self._update_rf()
        self.end_line(st, store=store)

    def heard_onset(self, t_now=None, contingent=None):
        """a line by another speaker starts: a contingent reaction to the own reply answers the own reply's prompt"""
        if self.prompt_saved is not None:
            if contingent is None:
                contingent = t_now is not None and self.own_end_t is not None and t_now - self.own_end_t <= self.c.contingency
            if contingent:
                self.prev = self.prompt_saved
            self.prompt_saved = None

    def own_finish(self, st, t_end=None):
        """the own reply ended: it becomes the previous line (not stored as an episode, never a teaching target)"""
        if not self.c.own_rotate or not st.get('items'):
            return
        self.prompt_saved = self.prev
        code = self.hip.dg(self._cue(st['items'], st['eye']))
        self.prev = dict(items=list(st['items']), eye=st['eye'], att=st['att'], code=code, next=None, ep=-1, own=True)
        self.own_end_t = t_end

    def hear_line(self, items, eye=None, att=None, learn=True, store=True, contingent=None):
        """a line by another speaker: predict it item by item from the previous line (learning by listening)."""
        self.heard_onset(contingent=contingent if contingent is not None else False)
        st = self.begin_line(eye, att)
        for it in items:
            self.line_item(st, it, learn)
        self.line_end(st, learn, store)

    def end_line(self, st, store=True):
        c = self.c
        items = st['items']
        if c.ctx:
            x = np.zeros(c.V + c.n_eye, bool)
            x[self._items(items)] = True
            if st['eye'] is not None:
                x[c.V:] = st['eye']
            self.ctx.step(x)
        code = self.hip.dg(self._cue(items, st['eye']))
        rec = dict(items=list(items), eye=st['eye'], att=st['att'], code=code, next=None, ep=-1)
        reuse = -1
        if store and c.hippo and items and c.hip_novelty < 1 and self.hip.n:
            ov = self.hip.overlaps(code)
            e = int(np.argmax(ov))
            old = self.hip.content[e]
            if ov[e] >= c.hip_novelty and old is not None and old['items'] == list(items):
                reuse = e
        if reuse >= 0:                                    # familiar: refresh the old trace (no duplicate)
            self.hip.sig[reuse] = 1.0
            old = self.hip.content[reuse]
            pv = self.prev
            if pv is not None and pv.get('ep', -1) >= 0 and self.hip.content[pv['ep']] is pv:
                pv['next'] = reuse
                self.next_of[pv['ep']] = reuse; self.next_born[pv['ep']] = self.hip.born[reuse]
            self.prev = old
            self.stats['lines'] += 1
            return
        if store and c.hippo and items:
            ep = self.hip.store(code, rec)
            self.next_of[ep] = -1
            self.hip.tick(1)
            pv = self.prev
            if pv is not None and pv.get('ep', -1) >= 0 and self.hip.content[pv['ep']] is pv:
                pv['next'] = ep
                self.next_of[pv['ep']] = ep; self.next_born[pv['ep']] = self.hip.born[ep]
            rec['ep'] = ep
        self.prev = rec
        self.stats['lines'] += 1

    # ------------------------------------------------------------------ own speech (the body asks at each boundary)
    def own_begin(self, eye, att):
        self.actor_tr = []
        st = self.begin_line(eye, att)
        self.own = st
        return st

    def own_scores(self, st):
        A, d, gd, total, cands = self._drives(st)
        st['_A'], st['_cands'] = A, cands
        return total[:self.c.V], float(total[self.END])

    def own_advance(self, st, item):
        self.actor_tr.append((st['_A'], item))
        self._advance(st, item, st['_cands'])

    def own_stop(self, st, by_end):
        if by_end and '_A' in st:
            self.actor_tr.append((st['_A'], self.END))

    def speak(self, eye=None, att=None, explore=0.0):
        """symbolic harness: the same boundary loop the body runs (max rule, floor, breath)."""
        c = self.c
        st = self.own_begin(eye, att)
        out = []
        for n in range(c.max_len + 1):
            prog, s_end = self.own_scores(st)
            prog = prog.copy()
            if explore > 0:
                prog += self.rng.random(c.V) * explore
            y = int(np.argmax(prog)); s_best = float(prog[y])
            if n > 0 and s_end > 0 and s_end >= s_best:
                self.own_stop(st, True); break
            if s_best < c.floor or n == c.max_len:
                break
            self.own_advance(st, y)
            out.append(y)
        return out

    def reward(self, r):
        """partner's contingent reaction: RPE = r - V(first-step context); actor update on the own choices."""
        c = self.c
        if not self.actor_tr:
            return 0.0
        A0 = self.actor_tr[0][0]
        delta = self.critic.update(A0, r)
        if c.reward_actor and delta != 0:
            for A, y in self.actor_tr:
                if len(A):
                    self.gen.W[A, y] += c.eta_actor * delta / len(A)
        self.actor_tr = []
        return delta

    # ------------------------------------------------------------------ sleep: replay episodes into the slow readouts
    def _replay(self, e, learn, eta):
        """reinstate episode e as the prompt and present its NEXT line; returns the mean prediction error"""
        c = self.c
        rec = self.hip.content[e]
        nxt = self.hip.content[rec['next']]
        self.prev = rec
        st = self.begin_line(nxt['eye'], nxt['att'])
        err = []
        for it in list(nxt['items']) + [self.END]:
            if learn:
                A, d, gd, total, cands = self._drives(st)
                p = float(np.clip(total[it], 0, 1)) if 0 <= it <= c.V else 0.0
                cands = self._learn_step(st, it, eta=eta)
            else:
                A, d, gd, total, cands = self._drives(st)
                p = float(np.clip(total[it], 0, 1)) if 0 <= it <= c.V else 0.0
            err.append(1.0 - p)
            if it != self.END:
                self._advance(st, it, cands)
        return float(np.mean(err))

    def awake_recall(self):
        """a quiet moment (nothing heard, not speaking): recall recall_n stored episodes (sampled by strength^rec_pow)
        and learn their successor line as if re-heard (rule R2 at eta_recall); the recalled trace is strengthened.
        The ongoing context (previous line, CTX) is untouched afterwards.  No sleep, no downscaling."""
        c = self.c
        if not (c.hippo and c.recall_n > 0 and c.learn_listen) or self.hip.n == 0:
            return 0
        n = self.hip.n
        nx = self.next_of[:n]
        ok = nx >= 0
        ok[ok] = self.hip.born[nx[ok]] == self.next_born[:n][ok]
        ok &= self.hip.sig[:n] > 0.05
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return 0
        w = np.maximum(self.hip.sig[idx], 1e-6).astype(np.float64) ** c.rec_pow
        k = min(c.recall_n, len(idx))
        pick = self.rng.choice(idx, size=k, replace=False, p=w / w.sum())
        saved_prev, saved_ps = self.prev, self.prompt_saved
        for e in pick:
            self._replay(int(e), True, c.eta_recall)
            if c.rec_boost > 0:
                self.hip.sig[e] += c.rec_boost * (1.0 - self.hip.sig[e])
        self.prev, self.prompt_saved = saved_prev, saved_ps
        self.stats['awake_recalls'] += k
        return k

    def sleep(self, n=200, downscale=0.9, decay=0.0):
        c = self.c
        if not (c.hippo and c.sleep_replay) or self.hip.n == 0:
            return 0
        saved_prev, saved_ctx = self.prev, (self.ctx.u.copy(), self.ctx.s.copy(), self.ctx.use.copy(), self.ctx.in_rate.copy())
        ok = lambda e: (self.hip.content[e] is not None and self.hip.content[e].get('next') is not None
                        and self.hip.content[self.hip.content[e]['next']] is not None)
        k = 0
        if c.sleep_mode == 'priority':
            # candidates: strong recent traces; priority = cortical prediction error x strength (unconsolidated first)
            cand = [e for e in self.hip.sample(min(400, self.hip.n), self.rng) if ok(e)]
            if cand:
                errs = np.array([self._replay(e, False, 0.0) for e in cand])
                pr = errs * self.hip.sig[cand]
                top = [cand[i] for i in np.argsort(-pr)[:c.sleep_k] if errs[i] > 0.2]
                for e in top:
                    for _ in range(c.sleep_passes):
                        self._replay(e, True, c.eta_sleep)
                    k += 1
                # downscale only traces the cortex now predicts (consolidated); keep the others
                for i, e in enumerate(cand):
                    if self._replay(e, False, 0.0) < 0.2:
                        self.hip.sig[e] *= downscale
        else:
            for e in self.hip.sample(n, self.rng):
                if not ok(e):
                    continue
                self._replay(e, True, c.eta_sleep)
                k += 1
            self.hip.sig[:self.hip.n] *= downscale            # hippocampal traces fade as cortex takes over
        self.prev = saved_prev
        self.ctx.u, self.ctx.s, self.ctx.use, self.ctx.in_rate = saved_ctx
        if decay > 0:
            self.gen.scale(1 - decay)                      # global downscaling (the one clean-up mechanism)
        return k
