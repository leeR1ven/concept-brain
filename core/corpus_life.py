"""One life in the big dialogue world: real Chinese dialogues (LCCC, converted to toned syllables) spoken by four
synthetic adult voices; the child overhears (ARCH 五 训练顺序 2). The brain is the same core brain; only the world
differs. Data files (not in the repo): <data>/birth.npz, train.npz, heldout.npz (core/world/corpus.py format)."""
import os

import numpy as np
import torch

from .brain import Brain
from .life import log
from .net import DEV
from .world import phon
from .world.adapter import Skin
from .world.body import Body
from .world.corpus import DialogueWorld, Dialogues, script_dialogue
from .world.voice import ADULTS, Cochlea, Voice


class CorpusLife:
    """(next_ranks: the next-syllable test, see there)"""
    def __init__(self, c, seed, data):
        self.c, self.seed = c, seed
        bases = phon.load_inventory()
        self.voice = Voice(bases)
        self.coch = Cochlea(self.voice, c, seed)
        self.body = Body(bases)
        self.skin = Skin(c['m'] * c['s'], self.body.need_mat)
        if c.get('speak'):                                 # the mouth gives a whole syllable per tick (core/speak.py)
            from .speak import Mouth, patterns72
            assert c.get('syl_ticks', 3) == 1, 'speak: one tick per syllable only'
            c['n_mus'] = 72
            c['mus_per_syl'] = int(np.median(patterns72(self.body).sum(1)))
            self.mouth = Mouth(self.body, c.get('mouth_tol', 1.0))
        self.said = []                                     # (tick, syllable or 'garble') the child said
        self.train = Dialogues.load(os.path.join(data, 'train.npz'))
        self.heldout = Dialogues.load(os.path.join(data, 'heldout.npz'))
        birth = DialogueWorld(Dialogues.load(os.path.join(data, 'birth.npz')), self.coch, seed)
        A0 = []
        while birth.n_ex < c['birth_dialogues']:
            A0.append(birth.step())
        log(f'birth: {len(A0)} ticks, {birth.n_ex} dialogues')
        V = None
        if c.get('sight_area'):                            # what the newborn sees: nothing, or now and then a sign
            g = np.random.default_rng(seed + 4343)
            cnt = np.bincount(self.train.syl[:200000].astype(np.int64))
            common = np.argsort(-cnt)[:600]
            V = np.array([self.eye(int(common[g.integers(len(common))]) if g.random() < 0.3 else None)
                          for _ in range(len(A0))])
        self.brain = Brain(c, seed, self.coch, self.skin, np.array(A0), V)
        del A0
        b = self.brain
        log(f'brain: sound {b.sound.n}, touch {b.touch.n}, PFC {b.n1} x 4, hippocampus {b.hip.nh}, '
            f'motor {b.motor.nf} force; birth density {b.birth_density:.3f}')
        self.env = DialogueWorld(self.train, self.coch, seed)
        self.muscles = np.zeros(int(c.get('n_mus', 24)), bool)
        self.own = None                                    # the syllable the child said, heard by it the next tick

    def sight_of(self, sid):
        """config sight_bits: the input bits of what is seen with a syllable (its written sign: one fixed random
        pattern per syllable, half on; None = nothing seen: also a fixed pattern, ARCH 二.3). World, not brain."""
        nb = int(self.c.get('sight_bits', 0))
        r = np.random.default_rng([self.c['lineage_seed'], 5150, 0 if sid is None else int(sid) + 1])
        return r.random(nb) < 0.5

    def eye(self, sid=None):
        """the input bits of the eye: the sight bits as they are (stand-in), or with a visual area one thermometer pair per
        pixel of the sign (row by row: pixel on, pixel off)"""
        v = self.sight_of(sid)
        if not self.c.get('sight_area'):
            return v
        return np.stack([v, ~v], 1).reshape(-1)

    def with_sight(self, a0, sid=None, count=False):
        """the ear's bits, then the sight bits (if the world has sight); with an eye (core/vision.py) the fovea is
        what the retina sees where the gaze is, then the periphery's coarse cells"""
        if not self.c.get('sight_bits'):
            return a0
        sc = getattr(self, 'scene', None)
        if sc is None:
            return np.concatenate([a0, self.eye(sid)])
        from .vision import pairs
        fov, coarse = sc.retina(None if sid is None else self.sight_of(sid), self.sight_of(None), count)
        return np.concatenate([a0, pairs(fov), pairs(coarse)])

    def tick(self):
        own = getattr(self, 'own', None)
        a0 = self.env.step(self.muscles, [] if own is None else [(own, -1, 'own', 0.0, 0.0)])
        if self.c.get('sight_bits'):                       # the sign is seen with a share p_see of the syllables heard
            if not hasattr(self, '_see_rng'):
                self._see_rng = np.random.default_rng(self.seed + 4242)
            cur = self.env.cur
            sid = None if cur is None else cur[0]
            if sid is None and own is not None and self.c.get('see_own'):   # the child's own clear syllable, heard
                sid = own                                  # now, is seen like the adults' (Fable a40 B3: so the own
            see = sid is not None and self._see_rng.random() < self.c.get('p_see', 0.5)   # voice gets the taught code)
            if getattr(self, 'scene', None) is not None:
                self.scene.world_step(cur)
            a0 = self.with_sight(a0, sid if see else None, count=True)
        r, self.r_next = getattr(self, 'r_next', 0.0), 0.0
        self.muscles = self.brain.step(torch.as_tensor(a0, device=DEV), r).cpu().numpy()
        if getattr(self, 'scene', None) is not None:     # the eye muscles move the gaze
            self.scene.move(self.brain.eye_muscles.cpu().numpy())
        lk = getattr(self.brain.pfc, 'G1C', None)
        if self.brain.learning and hasattr(lk, 'n_learn'):  # core/speak2.py: the own voice teaches the speech link:
            q = self.__dict__.setdefault('_said_q', [])   # the concept units it lights from below (2-3 ticks after
            for t0, mv in q:                              # the mouth moved) -> the force neurons that moved
                if self.env.t - t0 in (2, 3):
                    lk.learn(self.brain.concept.below, mv)
            q[:] = [(t0, mv) for t0, mv in q if self.env.t - t0 < 3]
        self.own = None
        if getattr(self, 'mouth', None) is not None:
            x = self.mouth.say(self.muscles)
            if x is not None:
                self.said.append((self.env.t, x))
                self.own = x if x != 'garble' else None
                if self.own is not None and hasattr(getattr(self.brain.pfc, 'G1C', None), 'n_learn'):
                    self.__dict__.setdefault('_said_q', []).append((self.env.t, self.brain.motor.force.on.clone()))
                if self.c.get('reward') and self.own is not None:
                    self.r_next = self.pleased(self.own)

    def pleased(self, x):
        """config reward (ARCH 五: the world keeps teaching; adults are pleased when the child says something close,
        and that is the reward; no repeat-after-me): the syllable the child said is among the next r_ahead syllables
        the adults are about to say: r_match; the same syllable in another tone: r_near; otherwise nothing. The brain
        gets it the next tick (brain.reward: the changes that led to it are strengthened)."""
        c, env = self.c, self.env
        # the next r_ahead syllables the adults will say, silence skipped (Fable a36: the child's voice comes about two
        # ticks after its thought, so 'the very next tick' could almost never be met; 3 syllables, exact or other tone)
        ahead = [y for y in env.plan[env.pos:] if y is not None][:int(c.get('r_ahead', 3))]
        def value(x):
            if any(int(y[0]) == x for y in ahead):
                return float(c.get('r_match', 1.0))
            if any(int(y[0]) // 5 == x // 5 for y in ahead):
                return float(c.get('r_near', 0.5))
            return 0.0
        r = value(x)
        if r:
            self.rewarded = getattr(self, 'rewarded', 0) + 1
        # the yardstick (Fable a35): what a babbler saying the commonest syllable at the same moments would get
        if getattr(self, '_common_sid', None) is None:
            self._common_sid = int(np.argmax(np.bincount(self.train.syl[:200000].astype(np.int64))))
        self.utter = getattr(self, 'utter', 0) + 1
        self.babbler = getattr(self, 'babbler', 0) + (value(self._common_sid) > 0)
        return r

    def play(self, srcs, rec=None):
        """probe: one source (sid, tick, voice, f0_offset, level) or None per tick; learning off, neuron states
        restored afterwards; the child's mouth is muted"""
        b = self.brain
        was, b.learning = b.learning, False
        snap = b.snapshot()
        for t, x in enumerate(srcs):
            # probes are by sound alone: nothing is seen
            b.step(torch.as_tensor(self.with_sight(self.coch.a0([] if x is None else [x])), device=DEV), 0.0)
            if rec is not None:
                rec(t)
        b.restore(snap)
        b.learning = was

    def syl(self, x, v):
        """the per-tick sources of one syllable: its 3 ticks, or the whole syllable in one tick (syl_ticks 1)"""
        if self.c.get('syl_ticks', 3) == 1:
            return [(int(x), -1, v, 0.0, 0.0)] * self.c.get('syl_hold', 1)
        return [(int(x), k, v, 0.0, 0.0) for k in range(3)]

    def code(self, sid, p_rel=0.5, layers=None):
        """the sound copies a syllable lights (sound area alone, from silence, the four adult voices)"""
        snd, ch = self.brain.sound, self.brain.chains
        n = sum(len(ix) for ix in ch.sound_idx)
        P = torch.zeros(n, device=DEV)
        no_ret = [torch.zeros(k, device=DEV) for k in snd.n]
        saved = [x.clone() for x in snd.state]
        for v in ADULTS:
            snd.state = [torch.zeros_like(x) for x in saved]
            H = []
            ks = self.syl(sid, v)
            for x in [None] * 4 + ks + [None] * 6:
                st = snd.step(torch.as_tensor(self.coch.a0([] if x is None else [x]), device=DEV), no_ret)
                H.append(torch.cat([st[l][ix] for l, ix in enumerate(ch.sound_idx)]).float())
            H = torch.stack(H)
            a = 0
            for l, ix in enumerate(ch.sound_idx):
                if layers is None or l in layers:
                    P[a:a + len(ix)] += H[4 + l: 4 + len(ks) + l, a:a + len(ix)].mean(0) / len(ADULTS)
                a += len(ix)
        snd.state = saved
        _, _, a0, _ = ch.seg('sound')[0]
        return a0 + torch.nonzero(P >= p_rel).squeeze(1)

    def reply_ranks(self, n=60):
        """held-out dialogues: after a turn ends (silence before the reply), the rank of the reply's first syllable
        among the first syllables of all probed replies, by the share of its sound copies lit inside the PFC and the
        hippocampus during the 8 silent ticks. A record of what the model thinks of, not a goal."""
        b, D = self.brain, self.heldout
        items = []
        for d in range(D.n):
            tr = D.turns(d)
            for j in range(len(tr) - 1):
                if len(tr[j]) and len(tr[j + 1]):
                    items.append((tr[j], int(tr[j + 1][0])))
            if len(items) >= n:
                break
        cands = sorted({s for _, s in items})
        if not hasattr(self, '_codes') or set(self._codes) != set(cands):
            self._codes = {s: self.code(s) for s in cands}
        out = {'pfc': [], 'hip': []}
        for prompt, s in items:
            src = [None] * 4 + [y for x in prompt for y in self.syl(x, ADULTS[0])] + [None] * 8
            rec = {'pfc': [], 'hip': []}
            self.play(src, rec=lambda t: (rec['pfc'].append(b.pfc.x_int.clone()), rec['hip'].append(b.hip.h_int.clone())))
            e = len(src) - 8
            for k in out:
                H = torch.stack(rec[k])[e:].any(0)
                sc = {x: float(H[cd].float().mean()) if cd.numel() else 0.0 for x, cd in self._codes.items()}
                out[k].append(sum(v > sc[s] for v in sc.values()) + sum(v == sc[s] for v in sc.values()))
        return {**{k: round(float(np.mean(v)), 1) for k, v in out.items()}, 'chance': (len(cands) + 1) / 2}

    def pair_test(self, n_pairs=20, min_count=20, after=3):
        """The designer's direct test (A -> B: is B lit?), 11-10. Pairs: syllables A whose most common follower B
        follows it most reliably in the training dialogues heard so far (count >= min_count). A alone, in each adult
        voice, from silence; during the `after` silent ticks that follow, the share of B's concept code (the concept
        copies lit in >= half of B's hearings, from next_ranks' codes) that the PFC lights from inside (thought), vs
        the same share for every other candidate syllable. Returns the mean share for B and for the others, how often B
        is the most lit candidate, and the share of the thought budget that goes to concept copies."""
        import collections
        b, p = self.brain, self.brain.pfc
        segs = {(kind, l): (a, bb) for kind, l, a, bb in b.chains.segs}
        if ('concept', 0) not in segs:
            return {}
        ca, cb = segs[('concept', 0)]
        if getattr(self, '_ncodes', None) is None:
            self.next_ranks(n_dlg=1)
        S = self._nS[' concept']                                 # mean concept copies per candidate
        k = max(1, int(round(float((S.sum(1)).mean()))))         # its code: the copies most often lit when it is heard
        code = torch.zeros_like(S, dtype=torch.bool).scatter_(1, torch.topk(S, k, dim=1).indices, True) & (S > 0)
        pos = {x: i for i, x in enumerate(self._ncand)}
        seq = self.train.syl[:self.train.turn_off[self.train.dlg_off[min(self.env.n_ex, self.train.n)]]].tolist()
        c1 = collections.Counter(seq)
        c2 = collections.Counter(zip(seq[:-1], seq[1:]))
        best = {}
        for (a, bb), k in c2.items():
            if a in pos and bb in pos and a != bb and k >= min_count and code[pos[bb]].any():
                if k / c1[a] > best.get(a, (0, 0))[1]:
                    best[a] = (bb, k / c1[a])
        pairs = sorted(best.items(), key=lambda kv: -kv[1][1])[:n_pairs]
        shareB, shareO, top, conc = [], [], [], []
        n_code = code.sum(1).clamp_min(1).float()
        for a, (bb, pr) in pairs:
            for v in ADULTS:
                rec = []
                src = [None] * 6 + self.syl(a, v) + [None] * after
                self.play(src, rec=lambda t: rec.append(p.x_int.clone()))
                lit = torch.stack(rec[-after:]).any(0)
                sh = (lit[ca:cb][None, :] & code).sum(1).float() / n_code
                sh[~self._nhave] = -1
                ib = pos[bb]
                shareB.append(float(sh[ib]))
                others = torch.cat([sh[:ib], sh[ib + 1:]])
                shareO.append(float(others[others >= 0].mean()))
                top.append(bool(sh[ib] >= sh.max()) and float(sh[ib]) > 0)
                conc.append(float(lit[ca:cb].sum()) / max(1.0, float(lit.sum())))
        return {'pairs': len(pairs), 'P(B|A) mean': round(float(np.mean([pr for _, (_, pr) in pairs])), 2),
                'B share': round(float(np.mean(shareB)), 3), 'others share': round(float(np.mean(shareO)), 3),
                'B most lit': round(float(np.mean(top)), 2), 'thought in concept copies': round(float(np.mean(conc)), 2)}

    def speak_test(self, n_pairs=20, min_count=20, after=8):
        """config speak (11-11): the designer's test by output. Pairs as pair_test (A whose most common follower B is
        most reliable in what was heard); A alone in each adult voice, from silence, then `after` quiet ticks with the
        mouth on (its own voice heard), learning off, states restored. Returns how often it said anything, how often it
        said B, said A (an echo), and the syllables it said most."""
        import collections
        if getattr(self, 'mouth', None) is None:
            return {}
        b = self.brain
        seq = self.train.syl[:self.train.turn_off[self.train.dlg_off[min(self.env.n_ex, self.train.n)]]].tolist()
        c1 = collections.Counter(seq)
        c2 = collections.Counter(zip(seq[:-1], seq[1:]))
        wired = set(getattr(self, 'speak_sids', []))
        best = {}
        for (a, bb), k in c2.items():
            if a != bb and k >= min_count and bb in wired and k / c1[a] > best.get(a, (0, 0))[1]:
                best[a] = (bb, k / c1[a])
        pairs = sorted(best.items(), key=lambda kv: -kv[1][1])[:n_pairs]
        said_any, said_b, said_a, garble, words = 0, 0, 0, 0, collections.Counter()
        n = 0
        for a, (bb, _) in pairs:
            for v in ADULTS:
                was, b.learning = b.learning, False
                snap = b.snapshot()
                own, out = None, []
                for x in [None] * 6 + self.syl(a, v) + [None] * after:
                    a0 = self.coch.a0(([] if x is None else [x]) + ([] if own is None else [(own, -1, 'own', 0.0, 0.0)]))
                    mus = b.step(torch.as_tensor(self.with_sight(a0), device=DEV), 0.0).cpu().numpy()
                    y = self.mouth.say(mus)
                    out.append(y)
                    own = y if (y is not None and y != 'garble') else None
                b.restore(snap)
                b.learning = was
                tail = [y for y in out[7:] if y is not None]
                n += 1
                said_any += bool(tail)
                said_b += bb in tail
                said_a += a in tail
                garble += sum(y == 'garble' for y in tail)
                words.update(y for y in tail if y != 'garble')
        from .world import phon
        bases = phon.load_inventory()
        return {'trials': n, 'said anything': round(said_any / max(1, n), 2), 'said B': round(said_b / max(1, n), 2),
                'said A (echo)': round(said_a / max(1, n), 2), 'garbles per trial': round(garble / max(1, n), 2),
                'most said': [phon.toned_name(bases, int(w)) for w, _ in words.most_common(8)]}

    def said_since(self, t0):
        """what the child said in life since tick t0: syllables, garbles, the most said"""
        import collections
        from .world import phon
        xs = [x for t, x in self.said if t >= t0]
        ws = collections.Counter(x for x in xs if x != 'garble')
        bases = phon.load_inventory()
        return {'syllables': len(xs) - sum(x == 'garble' for x in xs), 'garbles': sum(x == 'garble' for x in xs),
                'rewarded so far': getattr(self, 'rewarded', 0),
                'judged utterances': getattr(self, 'utter', 0), 'a babbler of the commonest would be rewarded':
                    getattr(self, 'babbler', 0), 'amygdala (pats, innate, expected, reward)':
                    getattr(self.brain, 'r_stats', None),
                'eye': None if getattr(self, 'scene', None) is None else self.scene.stats,
                'distinct': len(ws), 'most said': [(phon.toned_name(bases, int(w)), k) for w, k in ws.most_common(8)]}

    def __getstate__(self):
        """the probes' codes are not part of the life: they are not saved with it (they are made anew when needed)"""
        drop = ('_ncand', '_ncands', '_nS', '_ncodes', '_nhave', '_codes')
        return {k: v for k, v in self.__dict__.items() if k not in drop}

    def next_ranks(self, n_dlg=40, n_cands=300, n_code=300):
        """The next-syllable test (11-10): held-out dialogues heard as in life (learning off, states restored). At the
        first tick of each syllable, the PFC's push into the first-layer copies for the coming tick is what it
        expects; the syllable heard now is what comes. Two sets of copies: the sound area's first layer ('', dense:
        many syllables share them) and the concept area (' concept', sparse), if there is one. Each candidate's code =
        the copies that come when it is heard, averaged over n_code training dialogues heard frozen (in context, every
        voice; the sensory and concept areas do not learn, so the codes stay valid). The 300 most common syllables
        ranked by the cosine of the push with their codes: the true one's rank (1 = best, chance about n_cands / 2).
        'ceiling': the copies that really come, read the same way. Yardsticks: counts over the training dialogues heard
        so far (ngram_ranks), and the same counts as a count-weighted mix of the codes ('as copies': how well a push in
        these copies can do with perfect counts; a mix of dense codes blurs toward the common ones)."""
        import collections
        from .world.corpus import script_dialogue
        b, D, p, c = self.brain, self.heldout, self.brain.pfc, self.c
        segs = {(kind, l): (a, bb) for kind, l, a, bb in b.chains.segs}
        spaces = {'': segs[('sound', 0)]}
        if ('concept', 0) in segs:
            spaces[' concept'] = segs[('concept', 0)]
        gt, sk, hold = tuple(c.get('gap_turn', (4, 10))), c.get('syl_ticks', 3), c.get('syl_hold', 1)

        def first(src, t):
            x = src[t]
            if x is None or (sk == 3 and x[1] != 0):
                return False
            return not (t > 0 and src[t - 1] is not None and src[t - 1][5:] == x[5:])

        # ticks from a syllable's first tick in the ear to its copies in the PFC's first layer: the sound area's first
        # layer 1; the concept area 1 with concept_sync, else 2 (it reads the sound copies of the tick before). The
        # push that expects those copies is made a tick before they come. (Valid for syl_hold 1.)
        lag = {k: (2 if (k == ' concept' and not c.get('concept_sync')) else 1) for k in spaces}
        # the codes are made anew at every call: a living brain's codes move (its units' usual levels, what its
        # thoughts return to the senses), and codes kept from an earlier probe (or an earlier n_code) are not its codes
        cnt = np.bincount(self.train.syl.astype(np.int64))
        self._ncand = [int(x) for x in np.argsort(-cnt)[:n_cands]]
        pos = {x: i for i, x in enumerate(self._ncand)}
        S = {k: torch.zeros(n_cands, bb - a, device=DEV) for k, (a, bb) in spaces.items()}
        N = {k: torch.zeros(n_cands, device=DEV) for k in spaces}
        for d in range(min(n_code, self.train.n)):
            src = script_dialogue(self.train, d, np.random.default_rng([self.seed, 7700, d]), gt, 8, sk, hold)
            got = []
            self.play([None if x is None else x[:5] for x in src], rec=lambda t: got.append(p.L1ext.on.clone()))
            for t in range(len(src)):
                if first(src, t) and src[t][0] in pos:
                    for k, (a, bb) in spaces.items():
                        if t + lag[k] < len(got):
                            S[k][pos[src[t][0]]] += got[t + lag[k]][a:bb].float()
                            N[k][pos[src[t][0]]] += 1
        self._nS = {k: v / N[k].clamp_min(1)[:, None] for k, v in S.items()}
        self._ncodes = {k: v / v.norm(dim=1, keepdim=True).clamp_min(1e-9) for k, v in self._nS.items()}
        self._nhave = torch.stack([v > 0 for v in N.values()]).all(0)
        ranks, items = collections.defaultdict(list), []

        def rank(v, i, k):
            sc = (v / v.norm().clamp_min(1e-9)) @ self._ncodes[k].T
            sc[~self._nhave] = -1e9
            return float((sc > sc[i]).sum() + 0.5 * ((sc == sc[i]).sum() - 1) + 1)

        mlag = max(lag.values())
        for d in range(min(n_dlg, D.n)):
            src = script_dialogue(D, d, np.random.default_rng([self.seed, 8800, d]), gt, 8, sk, hold)
            ctx, got, pu = [], [], []

            def rec(t):
                d1 = p.push()                              # made now, for the first layer of the coming tick
                got.append({k: p.L1ext.on[a:bb].clone() for k, (a, bb) in spaces.items()})
                pu.append({k: d1[a:bb].clone() for k, (a, bb) in spaces.items()})
            self.play([None if x is None else x[:5] for x in src], rec=rec)
            for t in range(len(src)):
                if not first(src, t):
                    continue
                x = src[t]
                if x[0] in pos and bool(self._nhave[pos[x[0]]]) and t + mlag < len(src):
                    i = pos[x[0]]
                    for k in spaces:
                        ranks['brain' + k].append(rank(pu[t + lag[k] - 1][k], i, k))
                        ranks['ceiling' + k].append(rank(got[t + lag[k]][k].float(), i, k))
                    items.append((list(ctx), x[0], self._ncand))
                ctx.append(x[0])
        out = {k: np.array(v) for k, v in ranks.items()}
        out.update({k: np.array(v) for k, v in ngram_ranks(self.train, items, self.env.n_ex).items()})
        seq = self.train.syl[:self.train.turn_off[self.train.dlg_off[min(self.env.n_ex, self.train.n)]]].tolist()
        c1 = collections.Counter(seq)
        c2 = collections.Counter(zip(seq[:-1], seq[1:]))
        f1 = torch.tensor([c1[y] for y in self._ncand], device=DEV, dtype=torch.float32)
        for k in spaces:
            for name in ('freq as copies', 'prev 1 as copies'):
                rs = []
                for ctx, true, _ in items:
                    w = f1 if name == 'freq as copies' or not ctx else \
                        torch.tensor([c2[(ctx[-1], y)] for y in self._ncand], device=DEV, dtype=torch.float32)
                    if float(w.sum()) == 0:
                        w = f1
                    rs.append(rank(w @ self._nS[k], pos[true], k))
                out[name + k] = np.array(rs)
        return {k: {'mean rank': round(float(v.mean()), 1), 'top1': round(float((v <= 1).mean()), 3),
                    'top10': round(float((v <= 10).mean()), 3)} for k, v in out.items()} | \
            {'items': len(items), 'chance': (n_cands + 1) / 2}


def ngram_ranks(train, items, upto):
    """yardsticks for next_ranks: the rank of the true next syllable by its count after the same 0, 1 or 2 syllables
    in the first `upto` training dialogues (ties: average rank); candidates = those of next_ranks"""
    import collections
    seq = train.syl[:train.turn_off[train.dlg_off[min(upto, train.n)]]]
    c1 = collections.Counter(seq.tolist())
    c2 = collections.Counter(zip(seq[:-1].tolist(), seq[1:].tolist()))
    c3 = collections.Counter(zip(seq[:-2].tolist(), seq[1:-1].tolist(), seq[2:].tolist()))
    out = {'freq': [], 'prev 1': [], 'prev 2': []}
    for ctx, true, cands in items:
        for name, f in (('freq', lambda x: c1[x]), ('prev 1', lambda x: c2[(ctx[-1], x)] if len(ctx) >= 1 else c1[x]),
                        ('prev 2', lambda x: c3[(ctx[-2], ctx[-1], x)] if len(ctx) >= 2 else 0)):
            sc = np.array([f(x) for x in cands], float)
            st = sc[cands.index(true)]
            out[name].append(float((sc > st).sum() + 0.5 * ((sc == st).sum() - 1) + 1))
    return out
