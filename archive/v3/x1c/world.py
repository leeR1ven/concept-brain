"""X1c continuous world: syllables arrive tick by tick; caregivers speak, pause and react; the child's mouth
output is decoded by body physics and heard back one tick later.  No symbolic Q/A/age events reach the brain:
the brain only receives a binary auditory vector per tick plus the innate value detectors (pleased/upset/hunger relief).

World statistics follow docs/v3design/review/develop/caregiver_toy.py (format self-answer, re-ask, move-on,
politeness, care continuing, give-up jitter) -- re-implemented, the file is not imported.
"""
import numpy as np

FILL = ['苹', '桃', '蕉', '饭', '球', '车', '书', '鞋']          # first 4 are food
FRAME_SYL = ['甜', '么', '给', '吗', '啥', '色', '几', '号']
ANS_SYL = ['香', '不', '能', '吃', '要', '红', '绿', '黄', '的', '大', '小']
OK_SYL = ['对']
TV_SYL = ['天', '气', '新', '闻', '广', '告']
SYL = FILL + FRAME_SYL + ANS_SYL + OK_SYL + TV_SYL
SID = {s: i for i, s in enumerate(SYL)}
NSYL = len(SYL)
COLS = ['红', '绿', '黄']
SIZES = ['大', '小']
FRAMES = ['taste', 'offer', 'colour', 'size']
NVOICE = 5            # 0 own, 1 C1, 2 C2, 3 D1, 4 D2 (distractors)
OWN, C1, C2, D1, D2 = range(5)


class Physics:
    """Lineage-fixed sound and body physics (same for every seed)."""

    def __init__(self, seed=1234, n_c=150, k_syl=10, n_vu=6, n_mus=24, k_mus=4):
        rng = np.random.default_rng(seed)
        self.n_c, self.k_syl, self.n_vu, self.n_mus = n_c, k_syl, n_vu, n_mus
        self.n_a = n_c + NVOICE * n_vu
        self.content = np.zeros((NSYL, n_c), bool)
        for s in range(NSYL):
            self.content[s, rng.choice(n_c, k_syl, replace=False)] = True
        self.voice = np.zeros((NVOICE, self.n_a), bool)
        for v in range(NVOICE):
            self.voice[v, n_c + v * n_vu: n_c + (v + 1) * n_vu] = True
        need = []
        while len(need) < NSYL:                       # 4-of-24 muscle sets, pairwise overlap <= 2
            m = frozenset(rng.choice(n_mus, k_mus, replace=False).tolist())
            if all(len(m & o) <= 2 for o in need):
                need.append(m)
        self.need = need
        self.need_mat = np.zeros((NSYL, n_mus), bool)
        for s, m in enumerate(need):
            self.need_mat[s, list(m)] = True
        self.garble_rng_seed = seed + 7

    def decode(self, muscles):
        """mouth rule: all 4 muscles = clear syllable, 3 of 4 (and <= 5 active) = mumbled, else garble."""
        if not muscles.any():
            return None
        hit = self.need_mat[:, muscles].sum(1)
        n = int(muscles.sum())
        s = int(np.argmax(hit))
        if hit[s] == 4 and n == 4:
            return ('s', s)
        if hit[s] >= 3 and n <= 5:
            return ('m', s)
        return ('g', tuple(np.nonzero(muscles)[0].tolist()))

    def garble_content(self, key):
        r = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        v = np.zeros(self.n_c, bool)
        v[r.choice(self.n_c, 8, replace=False)] = True
        return v


def render(ph, srcs, rng):
    v = np.zeros(ph.n_a, bool)
    for kind, payload, voice, loud in srcs:
        if kind == 's':
            c = ph.content[payload].copy()
        elif kind == 'm':
            c = ph.content[payload].copy()
            on = np.nonzero(c)[0]
            c[rng.choice(on, 3, replace=False)] = False
        else:
            c = ph.garble_content(payload)
        full = np.zeros(ph.n_a, bool)
        full[:ph.n_c] = c
        full |= ph.voice[voice]
        keep = rng.random(ph.n_a) < loud * 0.95
        v |= full & keep
    v |= rng.random(ph.n_a) < 0.004
    return v



def make_task(rng):
    col = {}
    while True:
        c = rng.integers(3, size=8)
        if all((c == k).sum() >= 2 for k in range(3)):
            break
    while True:
        z = rng.integers(2, size=8)
        if all((z == k).sum() >= 3 for k in range(2)):
            break
    props = {x: dict(food=x < 4, col=COLS[c[x]], size=SIZES[z[x]]) for x in range(8)}
    # held-out pairs (~30 %): every filler keeps >= 2 taught frames; a held-out taste/offer pair keeps the other one
    while True:
        held = set()
        cand = [(f, x) for f in range(4) for x in range(8)]
        for i in rng.permutation(len(cand)):
            if len(held) >= 10:
                break
            f, x = cand[i]
            trial = held | {(f, x)}
            ok = sum((g, x) not in trial for g in range(4)) >= 2
            if f in (0, 1) and ((1 - f), x) in trial:
                ok = False
            ok = ok and sum((f, y) not in trial for y in range(8)) >= 5
            if ok:
                held = trial
        if len(held) == 10:
            break
    return props, held


def question(f, x):
    X = FILL[x]
    return [[X, '甜', '么'], ['给', X, '吗'], [X, '啥', '色'], [X, '几', '号']][f]


def answer(f, x, props):
    p = props[x]
    if f == 0:
        return ['香'] if p['food'] else ['不', '能', '吃']
    if f == 1:
        return ['要'] if p['food'] else ['不', '要']
    if f == 2:
        return [p['col'], '的']
    return [p['size']]


def narration(x, props, rng):
    p = props[x]
    return [FILL[x], p['col'], '的'] if rng.random() < 0.6 else [FILL[x], p['size']]


class World:
    def __init__(self, phys, seed, care='format', distract=False, p_narr=0.3, empty_every=60, empty_len=300,
                 p_over=None, wait=12):
        self.ph = phys
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1000)
        self.props, self.held = make_task(np.random.default_rng(seed + 2000))
        self.pairs = [(f, x) for f in range(4) for x in range(8) if (f, x) not in self.held]
        self.care = care                      # 'format' | 'real'
        self.distract = distract
        self.p_narr = p_narr
        self.wait = wait
        self.p_over = (0.15 if care == 'real' else 0.0) if p_over is None else p_over
        self.empty_every, self.empty_len = empty_every, empty_len
        self.phase = 'teach'                  # 'teach' | 'free' (untutored); tutoring flag per exchange
        self.p_tut_free = 0.0                 # care continuing during free use
        self.t = 0
        self.hunger = 0.0
        self.r_pending = 0.0
        self.mouth_now = None
        self.cg_now = None                    # (syl, voice) the caregiver is producing this tick
        self.cg_line_active = False
        # utterance tracker
        self.sil = 99
        self.utt = None
        self.utts = []                        # finished utterances
        self.last_mouth_key = None
        self.exchanges = []
        self.over_ex = []
        self.ex = None
        self.empty = False
        self.n_ex = 0
        self.tv_q = []
        self.forced_pair = None
        self.proc = self._caregiver()
        self.sound = np.zeros(phys.n_a, bool)

    # ------------------------------------------------------------------ rendering
    def _render(self, srcs):
        return render(self.ph, srcs, self.rng)

    # ------------------------------------------------------------------ child utterance tracking
    def _track(self, mouth):
        if mouth is None:
            self.sil += 1
            if self.utt is not None and self.sil >= 3:
                self.utts.append(self.utt)
                if self.ex is not None:
                    self.ex['utts'].append(self.utt)
                self.utt = None
            self.last_mouth_key = None
            return
        key = (mouth[0] if mouth[0] != 'm' else 's', mouth[1])
        if self.utt is None:
            self.utt = dict(t0=self.t, syl=[], garble=0, mumble=0, cg_speaking=self.cg_line_active,
                            empty=self.empty, state=self.ex['state'] if self.ex else 'none',
                            ex=self.n_ex, phase=self.phase, yield_fail=False, ticks=0)
        self.utt['ticks'] += 1
        if key != self.last_mouth_key:
            if mouth[0] == 'g':
                self.utt['garble'] += 1
                self.utt['syl'].append('~')
            else:
                if mouth[0] == 'm':
                    self.utt['mumble'] += 1
                self.utt['syl'].append(SYL[mouth[1]])
        self.last_mouth_key = key
        self.sil = 0

    # ------------------------------------------------------------------ caregiver
    def _say(self, line, voice):
        """yield 3 ticks per syllable; marks talk-over and failure to yield."""
        self.cg_line_active = True
        child_was_speaking = self.utt is not None and self.sil == 0
        over = 0
        for i, s in enumerate(line):
            for k in range(3):
                self.cg_now = (SID[s], voice)
                yield
                if self.sil == 0 and self.utt is not None:
                    over += 1
                    if child_was_speaking and over >= 4:
                        self.utt['yield_fail'] = True
        self.cg_now = None
        self.cg_line_active = False
        self._last_over = over

    def _wait(self, n):
        for _ in range(n):
            yield

    def _tutoring(self):
        if self.phase == 'teach':
            return True
        return self.rng.random() < self.p_tut_free

    def _caregiver(self):
        rng = self.rng
        while True:
            # empty room
            if self.empty_every and self.n_ex > 0 and self.n_ex % self.empty_every == 0 and not getattr(self, '_emptied', False):
                self._emptied = True
                self.empty = True
                yield from self._wait(self.empty_len)
                self.empty = False
            else:
                self._emptied = False
            tut = self._tutoring()
            real = self.care == 'real'
            voice = C1 if (not real or rng.random() < 0.5) else C2
            jit = (lambda d: max(2, int(round(d * (1 + 0.5 * (2 * rng.random() - 1)))))) if real else (lambda d: d)
            yield from self._wait(jit(int(rng.integers(8, 17))))
            if tut and rng.random() < self.p_narr:
                x = int(rng.integers(8))
                yield from self._say(narration(x, self.props, rng), voice)
                yield from self._wait(jit(int(rng.integers(5, 10))))
            if tut and rng.random() < self.p_over:             # overheard: one adult asks the other
                f, x = self.pairs[int(rng.integers(len(self.pairs)))]
                a, b = (C1, C2) if rng.random() < 0.5 else (C2, C1)
                self.ex = dict(n=self.n_ex, f=f, x=x, A=[], phase=self.phase, tut=tut, utts=[], resp=None, state='over',
                               self_answer=False, teach_new=0, overheard=True)
                self.over_ex.append(self.ex)
                yield from self._say(question(f, x), a)
                yield from self._wait(int(rng.integers(3, 6)))
                yield from self._say(answer(f, x, self.props), b)
                yield from self._wait(jit(int(rng.integers(5, 10))))
                self.ex['state'] = 'gap'
                self.n_over = getattr(self, 'n_over', 0) + 1
            if self.forced_pair is not None:
                f, x = self.forced_pair
            else:
                f, x = self.pairs[int(rng.integers(len(self.pairs)))]
            Q, A = question(f, x), answer(f, x, self.props)
            self.n_ex += 1
            ex = dict(n=self.n_ex, f=f, x=x, A=A, phase=self.phase, tut=tut, care=self.care, utts=[], resp=None,
                      self_answer=False, reask=False, moveon=False, t_q=None, lat=None, state='q', talkover=0,
                      teach_new=0)
            self.ex = ex
            self.exchanges.append(ex)
            yield from self._say(Q, voice)
            ex['talkover'] += int(self._last_over > 0)
            if tut and self._last_over > 0 and rng.random() < 0.5:
                self.r_pending -= 1.0                                 # upset: interrupted
            ex['t_q'] = self.t
            W = jit(self.wait)
            resp = yield from self._await_reply(ex, W)
            if resp is None and tut:
                r = rng.random()
                if real and r < 0.15:
                    ex['moveon'] = True
                elif real and r < 0.45:
                    ex['reask'] = True
                    yield from self._wait(2)
                    ex['state'] = 'q'
                    yield from self._say(Q, voice)
                    ex['t_q2'] = self.t
                    resp = yield from self._await_reply(ex, jit(self.wait))
                    if resp is None:
                        ex['state'] = 'react'
                        yield from self._say(A, voice)
                        ex['self_answer'] = True
                        ex['teach_new'] += 1
                else:
                    ex['state'] = 'react'
                    yield from self._say(A, voice)
                    ex['self_answer'] = True
                    ex['teach_new'] += 1
            if resp is not None and tut:
                yield from self._react(ex, resp, A, voice, f, x)
            ex['state'] = 'gap'
            if not tut:
                yield from self._wait(jit(int(rng.integers(3, 8))))

    def _await_reply(self, ex, W):
        ex['state'] = 'wait'
        n0 = len(self.utts)
        for k in range(W):
            if self.utt is not None and self.utt['t0'] >= ex['t_q'] - 1 and not self.utt['cg_speaking']:
                break
            yield
        else:
            return None
        u = self.utt
        ex['lat'] = u['t0'] - ex['t_q']
        ex['state'] = 'child'
        for k in range(25):                    # wait until the child has been silent 3 ticks
            if self.utt is None:
                break
            yield
        ex['resp'] = u
        return u

    def _react(self, ex, u, A, voice, f, x):
        rng = self.rng
        ex['state'] = 'react'
        yield from self._wait(int(rng.integers(2, 4)))
        syl = u['syl']
        correct = syl == A
        first_ok = len(syl) > 0 and syl[0] == A[0]
        if correct or first_ok:
            if rng.random() < (0.8 if correct else 0.5):
                self.r_pending += 1.0                                  # pleased
            if f == 1 and self.props[x]['food'] and syl[:1] == ['要']:
                self.r_pending += self.hunger                          # hunger relief: she gives the food
                self.hunger = 0.0
            yield from self._say(['对'] + A, voice)
        else:
            if rng.random() < 0.25:
                self.r_pending -= 1.0                                  # upset
            yield from self._say(question(f, x), voice)                # recast: she says the question and answer again
            yield from self._wait(2)
            yield from self._say(A, voice)
        ex['teach_new'] += 1

    # ------------------------------------------------------------------ distractors
    def _tv(self):
        if not self.distract:
            return None
        if not self.tv_q and self.rng.random() < 1 / 25:
            v = D1 if self.rng.random() < 0.5 else D2
            line = [int(self.rng.integers(NSYL)) for _ in range(int(self.rng.integers(2, 5)))]
            self.tv_q = [(s, v) for s in line for _ in range(3)]
        if self.tv_q:
            return self.tv_q.pop(0)
        return None

    # ------------------------------------------------------------------ one tick
    def step(self, muscles):
        mouth = self.ph.decode(muscles)
        self.mouth_now = mouth
        self._track(mouth)
        self.cg_now = None
        next(self.proc)
        self.hunger = min(1.0, self.hunger + 1 / 1500)
        srcs = []
        if mouth is not None:
            srcs.append((mouth[0], mouth[1], OWN, 1.0))
        if self.cg_now is not None:
            srcs.append(('s', self.cg_now[0], self.cg_now[1], 1.0))
        tv = self._tv()
        if tv is not None:
            srcs.append(('s', tv[0], tv[1], 0.4))
        self.sound = self._render(srcs)
        r = self.r_pending
        self.r_pending = 0.0
        self.t += 1
        return self.sound, r
