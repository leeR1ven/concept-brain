"""X1f worlds (X1c world.py imported unchanged).

OverhearWorld (designer 11-01, ARCH sec. 2.11 "systematic training", step 2): the child overhears many dialogues
between adults.  Four adult voices (C1, C2, D1, D2 at full loudness); every dialogue: one adult asks a seen pair's
question, another answers after a short pause; sometimes the asker confirms ("对" + answer); narration lines; empty-room
periods; timing jitter.  Nobody asks the child; the child's own mouth is still decoded and heard (it is counted as
"talk during overhearing").  Held-out pairs are never spoken.
"""
import numpy as np
from world import World, question, answer, narration, C1, C2, D1, D2

ADULTS = (C1, C2, D1, D2)


class OverhearWorld(World):
    def __init__(self, phys, seed, gap=(2, 7), p_confirm=0.3, p_narr=0.3, empty_every=60, empty_len=300, **kw):
        super().__init__(phys, seed, p_narr=p_narr, empty_every=empty_every, empty_len=empty_len, **kw)
        self.rng = np.random.default_rng(seed + 7000)
        self.gap = gap
        self.p_confirm = p_confirm
        self.phase = 'overhear'
        self.proc = self._caregiver()

    def _caregiver(self):
        rng = self.rng
        while True:
            if self.empty_every and self.n_ex > 0 and self.n_ex % self.empty_every == 0 and not getattr(self, '_emptied', False):
                self._emptied = True
                self.empty = True
                yield from self._wait(self.empty_len)
                self.empty = False
            else:
                self._emptied = False
            yield from self._wait(int(rng.integers(6, 17)))
            ab = rng.choice(len(ADULTS), 2, replace=False)
            a, b = ADULTS[ab[0]], ADULTS[ab[1]]
            if rng.random() < self.p_narr:
                x = int(rng.integers(8))
                yield from self._say(narration(x, self.props, rng), a if rng.random() < 0.5 else b)
                yield from self._wait(int(rng.integers(4, 10)))
            f, x = self.pairs[int(rng.integers(len(self.pairs)))]
            self.n_ex += 1
            A = answer(f, x, self.props)
            ex = dict(n=self.n_ex, f=f, x=x, A=A, phase='overhear', tut=False, utts=[], resp=None, state='over',
                      overheard=True, teach_new=0, talkover=0)
            self.ex = ex
            self.exchanges.append(ex)
            yield from self._say(question(f, x), a)
            ex['t_q'] = self.t
            yield from self._wait(int(rng.integers(self.gap[0], self.gap[1])))
            yield from self._say(A, b)
            if rng.random() < self.p_confirm:
                yield from self._wait(int(rng.integers(2, 4)))
                yield from self._say(['对'] + A, a)
            ex['state'] = 'gap'


def overhear_mech(world, a, b):
    """the child should stay quiet while it overhears: utterances per dialogue, garbled ones, while an adult speaks."""
    exs = [e for e in world.exchanges if a < e['n'] <= b]
    n = max(1, len(exs))
    utts = [u for u in world.utts if a < u['ex'] <= b]
    return dict(n_dialogues=len(exs), child_utts_per_dlg=round(len(utts) / n, 4),
                garbled_per_dlg=round(sum(1 for u in utts if u['garble'] > 0) / n, 4),
                during_adult_speech_per_dlg=round(sum(1 for u in utts if u['cg_speaking']) / n, 4),
                in_q_gap_per_dlg=round(sum(1 for e in exs for u in utts if u['ex'] == e['n'] and e.get('t_q') is not None
                                           and e['t_q'] - 1 <= u['t0'] <= e['t_q'] + 8) / n, 4))
