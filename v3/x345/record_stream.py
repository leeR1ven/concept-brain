"""Record the symbolic event stream of the v2.2 bench Life world (v2/bench/life.py, read only) for X3.

A silent agent lives in the Life world; the world's content comes from its own rng, so the content is the real Life's.
Sound is skipped (a fast silent _tick): only the symbolic events are logged.
  line event : (t, 'line', who, text, eye indices at that moment)
  eye event  : (t, 'eye', '', '', new eye indices)  -- objects appear / move / leave, a person arrives / leaves, dark
Run: python3 -I record_stream.py [n_events]   -> stream.json
"""
import json
import os
import sys

sys.dont_write_bytecode = True
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'v2', 'bench')   # repo: v2/bench
sys.path.insert(0, BENCH)
from life import Life  # noqa: E402
from physics import SOUND  # noqa: E402


class Silent:
    privileged = False

    def step(self, obs):
        return None


class Rec(Life):
    def __init__(self, *a, **k):
        self.ev = []
        super().__init__(*a, **k)

    def _tick(self, other=None, other2=None, cue=None, reward=0.0):      # silent agent, no sound computed
        partner = other is not None or other2 is not None
        if self.trk:
            for tr in self.trk:
                tr.tick(self.t, None, False, partner, None)
        self.t += 1
        return None, False

    def _say(self, who, text, reward_first=0.0):
        self.ev.append((int(self.t), 'line', who, text, [int(i) for i in self.eye.nonzero()[0]]))
        super()._say(who, text, reward_first)                 # ticks through the fast silent _tick

    def _set_scene(self, scene, attend=None):
        old = getattr(self, 'eye', None)
        super()._set_scene(scene, attend)
        if hasattr(self, 'ev') and old is not None and not (old.shape == self.eye.shape and (old == self.eye).all()):
            self.ev.append((int(self.t), 'eye', '', '', [int(i) for i in self.eye.nonzero()[0]]))


if __name__ == '__main__':
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    L = Rec(seed=0, tutor='model')
    ag = Silent()
    while sum(1 for e in L.ev if e[1] == 'line') < n_target:
        L.run(ag, ticks=25000)
    ev = L.ev
    n_line = sum(1 for e in ev if e[1] == 'line')
    print(f'{len(ev)} events ({n_line} lines, {len(ev) - n_line} eye changes), {L.t} ticks, {L.day_i} days')
    from collections import Counter
    c = Counter(e[3] for e in ev if e[1] == 'line')
    print('distinct lines', len(c), 'top', c.most_common(12))
    print('lines heard once', sum(1 for v in c.values() if v == 1))
    json.dump(dict(events=ev, ticks=L.t, days=L.day_i), open(os.path.join(os.path.dirname(__file__), 'stream.json'), 'w'),
              ensure_ascii=False)
