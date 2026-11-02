"""X11 overhearing world (CPU, numpy) [D1-D6]: converted LCCC dialogues spoken by adult voices, tick by tick.
Every dialogue: a pause, then two different adult voices (of the 4) alternate turns (A, B, A, ...); 3 ticks per
syllable; a pause between turns; per turn an F0 offset and a level change, F0 declines slowly within a turn. The
child overhears (ARCH 2.11 step 2); its mouth is read by the body each tick (child silent in the scaling test: any
muscle activity is only counted). The brain receives only A0; labels (syllable, onset, dialogue) stay here for the
scorer. Resumable at dialogue boundaries (state() / restore())."""
import numpy as np

from voice import ADULTS


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

    def __init__(self, dlgs, coch, seed, start=0, gap_dlg=(12, 25), gap_turn=(4, 10)):
        self.D, self.coch, self.seed = dlgs, coch, seed
        self.gap_dlg, self.gap_turn = gap_dlg, gap_turn
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
        return script_dialogue(self.D, d, rng, self.gap_turn, pre)

    def step(self, muscles=None):
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
        return self.coch.a0([] if x is None else [x[:5]])

    def adult_speaking(self):
        return self.cur is not None

    def state(self):
        return dict(i=self.i, pos=self.pos, n_ex=self.n_ex, t=self.t, coch_rng=self.coch.rng.bit_generator.state,
                    child_ticks=self.child_ticks)

    def restore(self, st):
        self.i, self.pos, self.n_ex, self.t, self.child_ticks = st['i'], st['pos'], st['n_ex'], st['t'], st['child_ticks']
        self.coch.rng.bit_generator.state = st['coch_rng']
        self.plan = self._plan(self.i)


def script_dialogue(dlgs, d, rng, gap_turn=(4, 10), pre=12):
    """the rendering plan of dialogue d: list per tick of None (pause) or (sid, tick, voice, f0_off, level, d, turn,
    pos). Two different adult voices alternate turns; per turn an F0 offset (+-1 semitone) and a level change (+-2 dB);
    F0 declines 0.08 semitone per syllable (at most 2) plus 0.3 semitone jitter per syllable."""
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
            out += [(int(sid), k, v, off, lev, d, j, p) for k in range(3)]
        if j < len(turns) - 1:
            out += [None] * int(rng.integers(*gap_turn))
    return out
