# -*- coding: utf-8 -*-
"""tune2 core = ARCH_v2's SynthCore + (1) wiring locality of the learned PFC -> input links (GEN rows),
(2) context instrumentation of the continuous thought, (3) wandering diversity records.
Every new key is OFF by default: with the defaults this core is SynthCore exactly (same RNG streams, same code paths).

Wiring (cfg key `wiring`):
  'full'   every MIX unit may link to every item unit (ARCH_v2).
  'local'  a MIX unit links only to item units within radius `loc_r` of its anchor on the input sheet.
  'sw'     small world: 'local' + `n_lr` random long-range targets per unit (drawn once, at the unit's birth).
  'rand'   control: only `n_lr` random targets (no locality; same kind of sparsity).
  END (the 'line ends' unit) is reachable by every unit in every setting.
Input sheet (topography): an item unit (perceptual syllable category) sits where its sound is on the ear's 3-D layer 0:
  pos_j = mean (band, slice, depth) coordinate of the layer-0 units active in the category's first exemplar, each axis
  scaled to [0, 1].  Similar sounds -> near (shared active units).  This is the only innate layout the agent has
  (the designer's 3-D sensory net); it is set once when the category is recruited (neurons do not move).
Anchor of a MIX unit = the sheet position of the item it reads (priority L > CP > CN > CC > VR > P1 > PL > P: the
  current input first); units that read no item (count, familiarity, slot, eye bits, ...) sit at the position of a
  random existing item (somewhere on the sheet).
Learning: R2 with the step normalised per postsynaptic unit by its active inputs: dW_ij = eta*g*(t_j - p_j)/|A_j|,
  A_j = active units wired to j (a local, postsynaptic quantity).  With full wiring |A_j| = |A| = ARCH_v2's rule; the
  'full' path is left untouched (bit-identical).  The reward actor's update is masked the same way.
Context keys (cfg key `ctx_keys`):
  'bag'  ARCH_v2: every held item joins one bag P (older lines confound the new one).
  'age'  P = the newest (dominant) line's held items only (= ARCH_v2 when one line is held); every older held item
         becomes a separate feature ('O', (age, item)), age = how many lines back it was heard (9 = brought by
         wandering), crossed only with L, N and the newest line's last item PL.  So held lines are told apart and the
         readout can use (or ignore) older content without changing the keys of the newest line.
"""
from __future__ import annotations

import numpy as np

from core import BOS, NONE
from core_synth import SynthCore, SCfg

ITEM_GROUPS = ('L', 'CP', 'CN', 'CC', 'VR', 'P1', 'PL', 'P')
_PRI = {g: i for i, g in enumerate(ITEM_GROUPS)}


class T2Cfg(SCfg):
    EXTRA2 = dict(wiring='full', loc_r=0.12, n_lr=0, ctx_track=True, wander_track=True, ctx_keys='bag',
                  ct_hub=0.0)

    def __init__(self, **kw):
        for k, v in self.EXTRA2.items():
            setattr(self, k, v)
        base = {k: v for k, v in kw.items() if k not in self.EXTRA2}
        super().__init__(**base)
        for k, v in kw.items():
            if k in self.EXTRA2:
                setattr(self, k, v)


class T2Core(SynthCore):
    def __init__(self, cfg):
        super().__init__(cfg)
        c = self.c
        V = c.V
        self.item_pos = np.full((V + 1, 3), np.nan, np.float32)        # input-sheet positions (END: none)
        self.w_rng = np.random.default_rng(c.seed + 4242)               # own stream: does not touch the others
        self.anchor = np.full((0, 3), np.nan, np.float32)
        self.lr = np.zeros((0, max(c.n_lr, 1)), np.int16)
        self.allowed = np.zeros((0, V + 1), bool)                          # unit -> output wiring (rows grow)
        self.n_anch = 0
        self.wstats = dict(masked_events=0, allowed_frac_sum=0.0, units_no_item=0, units_item=0)
        # context instrumentation: item sets of the last other-speaker lines (newest first)
        self.recent_lines = []
        self.ctx_stats = dict(reply_views=0, held_sum=0, back=[0] * 8, back_any=[0] * 8)
        self.wander_log = []           # (dominant item, held items) at every wandering step (life)

    # ================================================================ wiring
    def set_item_pos(self, j, pos):
        if 0 <= j < self.c.V and np.isnan(self.item_pos[j, 0]):
            self.item_pos[j] = pos
            if self.c.wiring in ('local', 'sw') and self.n_anch:          # a new input unit: wire the units near it
                n = self.n_anch
                d2 = ((self.anchor[:n] - pos[None, :]) ** 2).sum(-1)
                self.allowed[:n, j] |= np.nan_to_num(d2, nan=1e9) <= self.c.loc_r ** 2

    def _ensure_anchors(self):
        n = self.mix.n
        if n <= self.n_anch:
            return
        V = self.c.V
        if n > len(self.anchor):
            m = max(n, int(1.25 * len(self.anchor)) + 64)
            a = np.full((m, 3), np.nan, np.float32); a[:len(self.anchor)] = self.anchor; self.anchor = a
            lr = np.zeros((m, self.lr.shape[1]), np.int16); lr[:len(self.lr)] = self.lr; self.lr = lr
            al = np.zeros((m, V + 1), bool); al[:len(self.allowed)] = self.allowed; self.allowed = al
        c = self.c
        known = np.flatnonzero(~np.isnan(self.item_pos[:V, 0]))
        for u in range(self.n_anch, n):
            k = self.mix.key_of.get(u)
            best, bi = None, 99
            if k is not None:
                comps = [(k[0], k[1])] if len(k) == 2 else [(k[0], k[1]), (k[2], k[3])]
                for g, v in comps:
                    if g in _PRI and isinstance(v, (int, np.integer)) and 0 <= v < V and _PRI[g] < bi \
                            and not np.isnan(self.item_pos[v, 0]):
                        best, bi = int(v), _PRI[g]
            if best is not None:
                self.anchor[u] = self.item_pos[best]; self.wstats['units_item'] += 1
            else:
                if len(known):
                    self.anchor[u] = self.item_pos[known[self.w_rng.integers(len(known))]]
                self.wstats['units_no_item'] += 1
            if c.n_lr > 0:
                self.lr[u] = self.w_rng.integers(0, V, size=self.lr.shape[1])
            row = self.allowed[u]
            row[self.END] = True
            if c.wiring in ('local', 'sw') and not np.isnan(self.anchor[u, 0]):
                d2 = ((self.item_pos[:V] - self.anchor[u][None, :]) ** 2).sum(-1)
                row[:V] = np.nan_to_num(d2, nan=1e9) <= c.loc_r ** 2
            if c.wiring in ('sw', 'rand') and c.n_lr > 0:
                row[self.lr[u]] = True
        self.n_anch = n

    def _mask(self, A, J):
        """(|A|, |J|) bool: unit a is wired to output j (rows filled at the unit's birth, columns at an item's)."""
        self._ensure_anchors()
        return self.allowed[np.ix_(A, np.asarray(J))]

    def _gen_learn(self, A, target, total, J, g=1.0, eta=None):
        """R2 on GEN with the wiring mask and per-target normalisation (wiring != 'full')."""
        gen = self.gen
        if len(A) == 0 or g == 0:
            return
        eta = gen.eta if eta is None else eta
        p = np.clip(total, 0.0, 1.0)
        err = (target - p).astype(np.float32)
        J = np.asarray(J); keep = np.abs(err) > 1e-4; J, err = J[keep], err[keep]
        if len(J) == 0:
            gen.last_err = 0.0
            return
        gen.last_err = float(np.abs(err).max())
        M = self._mask(A, J)
        cnt = M.sum(0)
        ok = cnt > 0
        if not ok.any():
            return
        M, J, err, cnt = M[:, ok], J[ok], err[ok], cnt[ok]
        step = ((eta * g) * err / cnt)[None, :] * M
        gen.W[np.ix_(A, J)] += step.astype(gen.W.dtype)
        gen.use[A] += 1.0
        self.wstats['masked_events'] += 1
        self.wstats['allowed_frac_sum'] += float(M.mean())

    def _learn_step(self, st, target, g=1.0, eta=None):
        c = self.c
        if c.wiring == 'full':
            return super()._learn_step(st, target, g=g, eta=eta)
        A, d, gd, total, cands = self._drives(st)
        if 0 <= target < c.V and st.get('kind') != 'scene':
            st.setdefault('surp', []).append(float(1.0 - np.clip(total[target], 0, 1)))
        if c.use_gen and 0 <= target <= c.V:
            tvec = np.zeros(c.V + 1, np.float32); tvec[target] = 1.0
            J = np.nonzero((tvec > 0) | (np.clip(total, 0, 1) > 1e-4))[0]
            self._gen_learn(A, tvec[J], total[J], J, g=g, eta=eta)
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

    def reward(self, r):
        c = self.c
        if c.wiring == 'full':
            return super().reward(r)
        if not self.actor_tr:
            return 0.0
        A0 = self.actor_tr[0][0]
        delta = self.critic.update(A0, r)
        if c.reward_actor and delta != 0:
            for A, y in self.actor_tr:
                if len(A):
                    m = self._mask(A, np.array([y]))[:, 0]
                    if m.any():
                        self.gen.W[A[m], y] += c.eta_actor * delta / int(m.sum())
        self.actor_tr = []
        return delta

    def wiring_summary(self):
        n = self.n_anch
        if n == 0:
            return {}
        V = self.c.V
        known = ~np.isnan(self.item_pos[:V, 0])
        al = self.allowed[:n, :V][:, known]
        return dict(units=int(n), items=int(known.sum()), fanout_items_mean=round(float(al.sum(1).mean()), 2),
                    fanout_share=round(float(al.mean()), 4))

    def gen_nnz(self):
        n = self.mix.n
        return int(np.count_nonzero(self.gen.W[:n])), int(n)

    # ================================================================ age-separated context keys
    def _ct_view(self):
        v = super()._ct_view()
        c = self.c
        if v is None or c.ctx_keys != 'age':
            return v
        items = [i for i in v['items']]
        dom = set(items) if v['items'] is not v['bag'] else set()
        if not dom:                                         # the newest line has nothing held: keep ARCH_v2's view
            return v
        extra = [i for i in v['bag'] if i not in dom]
        if not extra:
            return v
        old = []
        for i in extra:
            age = 9
            for k, L in enumerate(self.recent_lines[1:], 1):
                if i in L:
                    age = k
                    break
            old.append((age, int(i)))
        v = dict(v)
        v['bag'] = list(dict.fromkeys(items))
        v['old'] = old
        return v

    def st_from(self, th):
        st = super().st_from(th)
        if th['st'].get('Pold'):
            st['Pold'] = th['st']['Pold']
        return st

    def _keys(self, st, cands):
        keys = super()._keys(st, cands)
        old = st.get('Pold')
        if old and self.c.ctx_keys == 'age':
            L = st['items'][-1] if st['items'] else BOS
            n = min(len(st['items']), 6)
            PL = st['P'][-1] if st['P'] else NONE
            for v in old:
                keys += [('O', v), ('O', v, 'L', L), ('O', v, 'N', n), ('O', v, 'PL', PL)]
        return keys

    # ================================================================ context instrumentation
    def _ct_input(self, rec, surprise=None):
        super()._ct_input(rec, surprise=surprise)
        if self.c.ctx_track:
            items = [i for i in rec['items'] if 0 <= i < self.c.V]
            if items:
                self.recent_lines.insert(0, set(items))
                del self.recent_lines[8:]

    def think_new(self, kind, eye, att, t=0):
        th = super().think_new(kind, eye, att, t=t)
        c = self.c
        prev = self.prev if kind == 'reply' else None
        if c.ctx_keys == 'age' and prev is not None and prev.get('old') and not self._replaying:
            th['st']['Pold'] = prev['old']
        if c.ct and c.ctx_track and kind == 'reply' and not self._replaying and self.recent_lines:
            held = set(int(i) for i in np.flatnonzero(self.T_a >= c.ct_theta))
            cs = self.ctx_stats
            cs['reply_views'] += 1
            cs['held_sum'] += len(held)
            newer = set()
            for k, L in enumerate(self.recent_lines):
                own = L - newer                                  # items only this line (k lines back) brought
                if own and (own & held):
                    cs['back'][k] += 1
                if L & held:
                    cs['back_any'][k] += 1
                newer |= L
        return th

    # ================================================================ wandering records (favourites / diversity)
    def tick_thought(self):
        c = self.c
        if c.ct_hub > 0 and c.ct and self.T_W is not None and c.ct_assoc > 0:
            self._tick_thought_hub()
        else:
            super().tick_thought()
        if c.wander_track and self.T_W is not None and c.ct_assoc > 0 and self.T_a.max() > 0:
            held = np.flatnonzero(self.T_a >= c.ct_theta)
            if len(self.wander_log) < 400000:
                self.wander_log.append((int(np.argmax(self.T_a)), tuple(int(i) for i in held)))

    def _tick_thought_hub(self):
        """SynthCore.tick_thought with postsynaptic normalisation of the spread: item j's input from the held items is
        divided by (its total incoming co-holding weight)^ct_hub, so a hub linked from every item gains no advantage."""
        c = self.c
        self._ct_decay()
        a = self.T_a
        if a.max() <= 0:
            return
        self.T_f += (a - self.T_f) / c.ct_fat_tau
        a *= (1.0 - c.ct_fat * self.T_f)
        cs = self.T_W.sum(0)
        spread = c.ct_assoc * (a @ self.T_W) / np.maximum(cs, 1e-9) ** c.ct_hub
        np.maximum(a, spread.astype(np.float32), out=a)
        m = float(a.max())
        if m > 0:
            a /= m
        held = np.flatnonzero(a >= c.ct_theta)
        if len(held) > c.ct_k:
            order = held[np.argsort(-a[held], kind='stable')]
            a[order[c.ct_k:]] = np.minimum(a[order[c.ct_k:]], c.ct_theta * 0.99)
        self.ct_stats['wander'] = self.ct_stats.get('wander', 0) + 1
