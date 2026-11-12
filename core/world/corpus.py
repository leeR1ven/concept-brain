"""X11 overhearing world (CPU, numpy) [D1-D6]: converted LCCC dialogues spoken by adult voices, tick by tick.
Every dialogue: a pause, then two different adult voices (of the 4) alternate turns (A, B, A, ...); 3 ticks per
syllable; a pause between turns; per turn an F0 offset and a level change, F0 declines slowly within a turn. The
child overhears (ARCH 2.11 step 2); its mouth is read by the body each tick (child silent in the scaling test: any
muscle activity is only counted). The brain receives only A0; labels (syllable, onset, dialogue) stay here for the
scorer. Resumable at dialogue boundaries (state() / restore())."""
import numpy as np

from .voice import ADULTS


class Dialogues:
    """a list of dialogues from converted shards: syl uint16, turn_off, dlg_off (data/convert.py format)"""

    def __init__(self, syl, turn_off, dlg_off):
        self.syl, self.turn_off, self.dlg_off = syl, turn_off, dlg_off
        self.n = len(dlg_off) - 1

    def turns(self, d):
        a, b = self.dlg_off[d], self.dlg_off[d + 1]
        return [self.syl[self.turn_off[t]:self.turn_off[t + 1]] for t in range(a, b)]

    @staticmethod
    def load(path):
        z = np.load(path)
        return Dialogues(z['syl'], z['turn_off'], z['dlg_off'])

    @staticmethod
    def select(shard_paths, n, max_turn=30, seed=0, skip=0):
        """dialogues skip .. skip + n - 1 of a seeded shuffle (within each shard, shards in order) of the dialogues
        whose turns all have <= max_turn syllables; returns a compact Dialogues"""
        rng = np.random.default_rng(seed)
        picked = []
        for p in shard_paths:
            z = np.load(p)
            s, to, do = z['syl'], z['turn_off'], z['dlg_off']
            tlen = np.diff(to)
            mx = np.maximum.reduceat(tlen, do[:-1])
            keep = np.nonzero(mx <= max_turn)[0]
            for d in keep[rng.permutation(len(keep))]:
                picked.append([s[to[t]:to[t + 1]] for t in range(do[d], do[d + 1])])
                if len(picked) >= n + skip:
                    break
            if len(picked) >= n + skip:
                break
        picked = picked[skip:skip + n]
        tl = [len(t) for dl in picked for t in dl]
        nt = [len(dl) for dl in picked]
        return Dialogues(np.concatenate([t for dl in picked for t in dl]).astype(np.uint16),
                         np.concatenate([[0], np.cumsum(tl)]).astype(np.int64),
                         np.concatenate([[0], np.cumsum(nt)]).astype(np.int64))

    def save(self, path):
        np.savez(path, syl=self.syl, turn_off=self.turn_off, dlg_off=self.dlg_off)


class DialogueWorld:
    """resumable anywhere: every dialogue's plan (pause before it, voices, offsets, pauses between turns) comes from
    its own generator seeded by (life seed, dialogue index); the cochlea's noise generator is saved with the state"""

    def __init__(self, dlgs, coch, seed, start=0, gap_dlg=None, gap_turn=None):
        self.D, self.coch, self.seed = dlgs, coch, seed
        self.gap_dlg = tuple(gap_dlg or coch.c.get('gap_dlg', (12, 25)))     # pauses between dialogues / turns
        self.gap_turn = tuple(gap_turn or coch.c.get('gap_turn', (4, 10)))
        self.i, self.pos = start, 0    # dialogue being spoken, tick within its plan
        self.n_ex = 0                  # dialogues finished
        self.t = 0
        self.cur = None                # (sid, tick, voice, f0_off, level, dialogue, turn, position) or None (pause)
        self.child_ticks = 0           # ticks on which the child's muscles moved (should stay 0 here)
        self.plan = self._plan(self.i)

    def _plan(self, i):
        d = i % self.D.n
        rng = np.random.default_rng([self.seed, 7000, i])
        pre = int(rng.integers(*self.gap_dlg))
        return script_dialogue(self.D, d, rng, self.gap_turn, pre, self.coch.c.get('syl_ticks', 3),
                               self.coch.c.get('syl_hold', 1))

    def step(self, muscles=None, extra=()):
        """extra: sources heard on top of the dialogue this tick (the child's own voice, config speak)"""
        if muscles is not None and muscles.any():
            self.child_ticks += 1
        x = self.plan[self.pos]
        self.pos += 1
        if self.pos == len(self.plan):
            self.n_ex += 1
            self.i += 1
            self.pos = 0
            self.plan = self._plan(self.i)
        self.cur = x
        self.t += 1
        return self.coch.a0(([] if x is None else [x[:5]]) + list(extra))

    def adult_speaking(self):
        return self.cur is not None

    def state(self):
        return dict(i=self.i, pos=self.pos, n_ex=self.n_ex, t=self.t, coch_rng=self.coch.rng.bit_generator.state,
                    child_ticks=self.child_ticks)

    def restore(self, st):
        self.i, self.pos, self.n_ex, self.t, self.child_ticks = st['i'], st['pos'], st['n_ex'], st['t'], st['child_ticks']
        self.coch.rng.bit_generator.state = st['coch_rng']
        self.plan = self._plan(self.i)


def script_dialogue(dlgs, d, rng, gap_turn=(4, 10), pre=12, syl_ticks=3, hold=1):
    """the rendering plan of dialogue d: list per tick of None (pause) or (sid, tick, voice, f0_off, level, d, turn,
    pos). Two different adult voices alternate turns; per turn an F0 offset (+-1 semitone) and a level change (+-2 dB);
    F0 declines 0.08 semitone per syllable (at most 2) plus 0.3 semitone jitter per syllable. syl_ticks 1: the whole
    syllable at once, held for `hold` ticks (each stage of the brain is a tick later than the one before; held, the
    stages hear the same syllable at its last tick, as a real syllable outlasts the steps between areas)."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng([int(rng), 9100, d])
    ab = rng.choice(len(ADULTS), 2, replace=False)
    out = [None] * pre
    turns = dlgs.turns(d)
    for j, tr in enumerate(turns):
        v = ADULTS[ab[j % 2]]
        f0, lev = rng.uniform(-1.0, 1.0), rng.uniform(-2.0, 2.0)
        for p, sid in enumerate(tr):
            off = f0 - min(2.0, 0.08 * p) + rng.normal(0.0, 0.3)
            out += ([(int(sid), k, v, off, lev, d, j, p) for k in range(3)] if syl_ticks == 3
                    else [(int(sid), -1, v, off, lev, d, j, p)] * hold)   # syl_ticks 1: the whole syllable at once
        if j < len(turns) - 1:
            out += [None] * int(rng.integers(*gap_turn))
    return out


def common_order(dlgs, max_syl=40, ref=None):
    """dialogue indices, the most ordinary first: by the mean log-frequency of their syllables (frequencies over the
    whole set, or over `ref`, e.g. everyday talk); dialogues longer than max_syl syllables go last"""
    src = dlgs.syl if ref is None else ref.syl
    f = np.bincount(src.astype(np.int64), minlength=max(int(src.max()), int(dlgs.syl.max())) + 1).astype(np.float64)
    lf = np.log(f + 1.0)[dlgs.syl.astype(np.int64)]
    a, b = dlgs.turn_off[dlgs.dlg_off[:-1]], dlgs.turn_off[dlgs.dlg_off[1:]]
    cs = np.concatenate([[0.0], np.cumsum(lf)])
    n = np.maximum(b - a, 1)
    score = (cs[b] - cs[a]) / n - 100.0 * (n > max_syl)
    return np.argsort(-score, kind='stable')


class CurriculumWorld(DialogueWorld):
    """the same world, but which dialogue comes next is drawn from a pool of the most ordinary dialogues; the pool
    starts small (each dialogue heard many times) and doubles every `every` dialogues (more and more variety). Within
    the pool the more ordinary ones come up more often (rank = pool x u^skew, u uniform): everyday talk is heard again
    and again, rarer talk now and then."""

    def __init__(self, dlgs, coch, seed, order, pool=200, every=5000, start=0, **kw):
        self.order, self.pool0, self.every = order, pool, every
        super().__init__(dlgs, coch, seed, start=start, **kw)

    def pool(self, i):
        return int(min(len(self.order), self.pool0 * 2 ** ((i - self.i0) // self.every)))

    def _plan(self, i):
        if not hasattr(self, 'i0'):
            self.i0 = i
        r = np.random.default_rng([self.seed, 7300, i])
        d = int(self.order[min(self.pool(i) - 1, int(self.pool(i) * r.random() ** getattr(self, 'skew', 2.0)))])
        rng = np.random.default_rng([self.seed, 7000, i])
        pre = int(rng.integers(*self.gap_dlg))
        return script_dialogue(self.D, d, rng, self.gap_turn, pre, self.coch.c.get('syl_ticks', 3),
                               self.coch.c.get('syl_hold', 1))


def concat(a, b):
    """dialogues of a, then items of b, as one Dialogues"""
    return Dialogues(np.concatenate([a.syl, b.syl]),
                     np.concatenate([a.turn_off, b.turn_off[1:] + a.turn_off[-1]]),
                     np.concatenate([a.dlg_off, b.dlg_off[1:] + a.dlg_off[-1]]))


def mixed_order(a, b, every=3):
    """a's items ordered by commonness, with one of b's (also by commonness) after every `every` of a's"""
    oa, ob = common_order(a), common_order(b, ref=a)        # text: ordinary by the measure of everyday talk
    blen = b.turn_off[b.dlg_off[1:]] - b.turn_off[b.dlg_off[:-1]]
    ob = ob[blen[ob] >= 8] + a.n                      # text items shorter than 8 syllables (list numbers etc.) left out
    out, j = [], 0
    for i, d in enumerate(oa):
        out.append(d)
        if (i + 1) % every == 0 and j < len(ob):
            out.append(ob[j])
            j += 1
    return np.array(out + list(ob[j:]), np.int64)
