# -*- coding: utf-8 -*-
"""Ear-alone recordings in world2 (measurement harness; world2 itself is unchanged, only instrumented by subclassing).

World: World2(n_agents=N, seed, noise).  Bodies 0..K-1 are silent LISTENERS (their ear windows are recorded); bodies
K..N-1 are CHATTER bodies (reference speakers, privileged): after >= GAP ticks in which no source was audible to them
(ground truth, so the crowd behaves the same at every noise level) they start, with probability p per tick, an
utterance of 2-4 random lexicon chars (force ramp 1-2-3 per char, like world2's Scripted).
Recorded per listener and tick (daytime and night):
  fresh[t]  = the 2 newest slices of its ear window (8, 2) int8   (the window = last 3 fresh pairs)
  gt[t]     = audible sources: (kind 'c' caregiver / 'a' agent, source id, line id (start tick), char, phase 0..2, att)
Ground truth is for scoring only; the front end under test reads `fresh` (= obs['ear']) only.
usage: python3 record.py N K p noise days seed [day:sigma,...]  -> rec/w2_N{N}_p{p}_s{noise}_seed{seed}[_sched].pkl"""
import os, sys, time, pickle, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bench') if os.path.isdir(os.path.join(HERE, 'bench')) else os.path.join(HERE, '..', 'bench'))
from world2 import World2                              # noqa: E402
from world2.agents import Scripted                     # noqa: E402
from world2.core import N_MOUTH, INAUD                 # noqa: E402

GAP = 10


class GTWorld(World2):
    def __init__(self, *a, listeners=(), **kw):
        super().__init__(*a, **kw)
        self.L = list(listeners)
        self.gt = {L: [] for L in self.L}
        self.cg_text = {}                    # (pid, t0) -> text
        self.aud_all = None                  # (N,) bool: any source audible this tick (for chatter gap rule)

    def _say_start(self, cg):
        op = cg.op
        spk = op[2].get('speaker', cg)
        before = self._say.get(spk.pid)
        super()._say_start(cg)
        e = self._say.get(spk.pid)
        if e is not None and e is not before:
            self.cg_text[(spk.pid, e[1])] = op[1]

    def _sound(self, t):
        N = self.N
        rec = {L: [] for L in self.L}
        aud = np.zeros(N, bool)
        posN = self.pos[:N]
        for pid, e in self._say.items():
            k = t - e[1]
            if k < 0 or k >= len(e[0]):
                continue
            ci, ph = divmod(k, 3)
            text = self.cg_text.get((pid, e[1]), '')
            ch = text[ci] if ci < len(text) else '?'
            av = self.ATT[e[2]][posN]
            aud |= av < INAUD
            for L in self.L:
                if av[L] < INAUD:
                    rec[L].append(('c', int(pid), int(e[1]), ch, ph, int(av[L])))
        for j, (tok, ph, ch) in self._asrc.items():
            b = self._open.get(j)
            t0 = int(b['t0']) if b is not None else -1
            av = self.ATT[int(self.pos[j])][posN].copy()
            av[j] = INAUD
            aud |= av < INAUD
            for L in self.L:
                if av[L] < INAUD:
                    rec[L].append(('a', int(j), t0, ch, ph // 2, int(av[L])))
        self._gt_cur = {L: tuple(rec[L]) for L in self.L}
        self._gt_t = t
        self.aud_all = aud
        super()._sound(t)

    def gt_tick(self):
        """called by the bodies every tick (after _sound): the sources of this tick (none if no sound tick ran)"""
        cur = self._gt_cur if getattr(self, '_gt_t', -1) == self.t else None
        for L in self.L:
            self.gt[L].append(cur[L] if cur is not None else ())
        if cur is None:
            self.aud_all = None


class Crowd(Scripted):
    """listeners 0..K-1 silent; chatter bodies K..N-1."""

    def __init__(self, n, K, p, seed=0):
        super().__init__(n)
        self.K, self.p = K, p
        self.rng = np.random.default_rng(seed)
        self.quiet = np.zeros(n, np.int64)
        self.fresh = {L: [] for L in range(K)}
        self.utter = 0

    def bench_attach(self, world):
        super().bench_attach(world)
        self.chars = sorted(self.need)
        from world2 import lex
        lexc = set(''.join(lex.all_lexicon_texts()))
        self.chars = [c for c in self.chars if c in lexc]

    def bench_prompt(self, j, info):
        pass

    def step(self, obs):
        ear = obs['ear']
        self.w.gt_tick()
        for L in range(self.K):
            self.fresh[L].append(np.array(ear[L, :, 4:6], np.int8))
        aud = self.w.aud_all if self.w.aud_all is not None else np.zeros(self.N, bool)
        self.quiet = np.where(aud, 0, self.quiet + 1)
        if self.p > 0:
            cand = np.flatnonzero((self.quiet >= GAP) & (self.rng.random(self.N) < self.p))
            for j in cand:
                if j < self.K or j in self.active:
                    continue
                Ln = int(self.rng.integers(2, 5))
                txt = ''.join(self.chars[int(i)] for i in self.rng.integers(len(self.chars), size=Ln))
                self.say(j, txt, delay=0)
                self.utter += 1
                self.quiet[j] = 0
        return super().step(obs)


def main():
    N, K, p, noise, days, seed = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), \
        float(sys.argv[5]), int(sys.argv[6])
    t0 = time.time()
    w = GTWorld(n_agents=N, seed=seed, noise=noise, listeners=range(K))
    cr = Crowd(N, K, p, seed=seed + 1)
    ticks = int(days * 30000)
    sched = dict((int(x.split(':')[0]), float(x.split(':')[1])) for x in (sys.argv[7].split(',') if len(sys.argv) > 7 else []) if x)
    if sched:                                       # noise schedule by day (a world2 setting changed during the life)
        for d in range(int(days)):
            if d in sched:
                w.noise = sched[d]
            w.run(cr, ticks=30000)
    else:
        w.run(cr, ticks=ticks)
    out = dict(N=N, K=K, p=p, noise=noise, days=days, seed=seed, ticks=ticks, sched=sched,
               fresh={L: np.stack(cr.fresh[L]) for L in range(K)}, gt=w.gt, cg_text=w.cg_text,
               warm=w.warmup_windows(), utter=cr.utter, n_say=sum(1 for r in w.log if r.get('type') == 'say'),
               day_ticks=w.p['day_ticks'], night_ticks=w.p['night_ticks'], secs=round(time.time() - t0, 1))
    os.makedirs(os.path.join(HERE, 'rec'), exist_ok=True)
    fn = os.path.join(HERE, 'rec', f'w2_N{N}_p{p}_s{noise}_seed{seed}' + ('_sched' if sched else '') + '.pkl')
    pickle.dump(out, open(fn, 'wb'))
    print(f'{fn} N {N} K {K} p {p} noise {noise} ticks {ticks} utter {cr.utter} cg lines {out["n_say"]} {out["secs"]}s',
          flush=True)


if __name__ == '__main__':
    main()
