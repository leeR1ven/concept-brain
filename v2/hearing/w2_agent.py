# -*- coding: utf-8 -*-
"""The full ARCH_v2.1 agent (AgentH; v2.1 final life config) living in world2 as mouth-only bodies, with noise and
crowds (noise_agent.py-style, extended).  Bodies 0..K-1 are full agents; bodies K..N-1 are CHATTER reference bodies
(privileged: they start 2-4 random lexicon chars with probability p per tick after >= 10 ticks in which no source was
audible to them, so the crowd is the same at every noise level).  Ground truth (scoring only): the caregiver and agent
lines that reached each full agent, by attenuation.
Counts per full agent: heard lines, items, categories, quiet moments, awake recalls, hippocampal episodes, releases,
utterances, brake-on share, releases refused by the brake; world2 headline (talk over partner etc.).
usage: python3 w2_agent.py N K p noise ticks seed [agent keys, e.g. h_gate=1 h_conf=1 ...] [noise_at=t:sigma,...]"""
import os, sys, time, json, collections
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v21')); sys.path.insert(1, os.path.join(HERE, '..', 'synth')); sys.path.insert(0, HERE)
from agent_h import AgentH, typed_h                       # noqa: E402
from run_v21 import TUNE_BEST, ARCH_V2, typed            # noqa: E402
from life_v21 import THINK_LIFE                           # noqa: E402
sys.path.insert(0, os.path.join(HERE, 'bench') if os.path.isdir(os.path.join(HERE, 'bench')) else os.path.join(HERE, '..', 'bench'))
from world2 import PerAgent, MouthOnly, evaluate, headline  # noqa: E402
from world2.core import N_ACT, N_MOUTH                     # noqa: E402
from world2 import lex                                     # noqa: E402
from record import GTWorld                                 # noqa: E402
from ear_eval import true_lines                            # noqa: E402

V21FINAL = dict(ral=1, ral_other=1, say_rec=1, say_loc=1, scene_held=1, val_veto=0.2, dip=1, act_T=60, soc_feats=1,
                rec_fix=1)
LIFE = dict(ct_assoc=0.5, wander=50, wander_after=200)


class Shim:
    def __init__(self, w2, n_eye):
        self.seed = w2.seed
        self.voice = w2.voice
        self.eye = np.zeros(n_eye, np.uint8)
        self.p = {'eye_layout': 'random'}
        self._w2 = w2

    def warmup_windows(self):
        return self._w2.warmup_windows()


class ChatterBody:
    def __init__(self, j, w, p, rng, chars):
        self.j, self.w, self.p, self.rng, self.chars = j, w, p, rng, chars
        self.quiet = 0
        self.plan = collections.deque()
        self.out = np.zeros(N_ACT, np.float32)
        self.need = {c: sorted(s) for c, s in w.voice.need.items()}
        self.utter = 0

    def step(self, o):
        aud = self.w.aud_all
        self.quiet = 0 if (aud is not None and aud[self.j]) else self.quiet + 1
        if not self.plan and self.p > 0 and self.quiet >= 10 and self.rng.random() < self.p:
            for i in self.rng.integers(len(self.chars), size=int(self.rng.integers(2, 5))):
                for lv in (1.0, 2.0, 3.0):
                    self.plan.append((self.chars[int(i)], lv))
            self.utter += 1
        self.out[:] = 0
        if self.plan:
            ch, lv = self.plan.popleft()
            self.out[self.need[ch]] = lv
            self.quiet = 0
        return self.out


class PerAgentGT(PerAgent):
    def __init__(self, agents, w, full):
        super().__init__(agents)
        self.w, self.full = w, full
        self.F_on = np.zeros(len(full)); self.day = 0

    def step(self, obs):
        self.w.gt_tick()
        out = super().step(obs)
        if (self.w.t % 30000) < 24000:
            self.day += 1
            for i, a in enumerate(self.full):
                self.F_on[i] += a.agent.body.F
        return out


def main():
    N, K, p, noise, ticks, seed = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), \
        int(float(sys.argv[5])), int(sys.argv[6])
    A = dict(a.split('=', 1) for a in sys.argv[7:] if '=' in a)
    sched = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in A.pop('noise_at', '').split(',') if x]
    tag = A.pop('tag', 'x')
    kw = dict(TUNE_BEST); kw.update(ARCH_V2); kw.update(LIFE); kw.update(THINK_LIFE); kw.update(V21FINAL)
    for k, v in A.items():
        kw[k] = typed_h(k, v, typed)
    t0 = time.time()
    w = GTWorld(n_agents=N, seed=seed, noise=noise, listeners=range(K))
    sh = Shim(w, 1600)
    lexc = sorted(set(''.join(lex.all_lexicon_texts())) & set(w.voice.need))
    rng = np.random.default_rng(seed + 1)
    full, bodies = [], []
    for j in range(N):
        if j < K:
            a = AgentH(sh, seed=seed * 100 + j, **kw)
            a.babble(w.voice, 35960)
            m = MouthOnly(a); full.append(m); bodies.append(m)
        else:
            bodies.append(ChatterBody(j, w, p, rng, lexc))
    pa = PerAgentGT(bodies, w, full)
    t_built = time.time() - t0
    done, chunks = 0, []
    marks = sorted(set([t for t, _ in sched] + [ticks]))
    for m in marks:
        if m > done:
            w.run(pa, ticks=m - done); done = m
        for t, s in sched:
            if t == m:
                w.noise = s
        ch = dict(t=done, noise=w.noise, lines=[x.agent.stats['lines'] for x in full], cats=[x.agent.Z.n for x in full],
                  recalls=[x.agent.core.stats['awake_recalls'] for x in full],
                  spoke=[x.agent.tstats['reply_spoke'] + x.agent.tstats['scene_spoke'] for x in full],
                  floor=[None if x.agent.gate is None else round(x.agent.gate.mu, 2) for x in full])
        chunks.append(ch)
        print(f'[{tag}] {ch} {time.time() - t0:.0f}s', flush=True)
    res = evaluate(w.log)
    TLs = [true_lines(w.gt[L]) for L in range(K)]
    near = [sum(1 for v in TL.values() if v['att0'] == 0) for TL in TLs]
    cg_near = [sum(1 for k, v in TL.items() if v['att0'] == 0 and k[0] == 'c') for TL in TLs]
    ag = [x.agent for x in full]
    out = dict(tag=tag, N=N, K=K, p=p, noise=noise, sched=sched, ticks=ticks, seed=seed, secs=round(time.time() - t0),
               built_secs=round(t_built), keys={k: v for k, v in A.items()},
               true_lines=[len(TL) for TL in TLs], true_near=near, true_cg_near=cg_near,
               lines=[a.stats['lines'] for a in ag], items=[a.stats['items'] for a in ag], cats=[a.Z.n for a in ag],
               quiet=[a.core.stats['quiet'] for a in ag], awake_recalls=[a.core.stats['awake_recalls'] for a in ag],
               bound=[len(a.cat2prog) for a in ag], hip=[a.core.hip.n for a in ag], released=[a.tstats.get('released', 0) for a in ag],
               spoke=[a.tstats['reply_spoke'] + a.tstats['scene_spoke'] for a in ag],
               reply_spoke=[a.tstats['reply_spoke'] for a in ag],
               go_blocked=[a.body.go_blocked for a in ag], now_blocked=[a.body.now_blocked for a in ag],
               F_on=[round(float(x), 3) for x in pa.F_on / max(1, pa.day)],
               hstats=[getattr(a, 'hstats', None) for a in ag],
               gate=[None if a.gate is None else dict(mu=round(a.gate.mu, 2), nu=round(a.gate.nu, 2), **a.gate.stats)
                     for a in ag],
               chatter_utter=sum(b.utter for b in bodies if isinstance(b, ChatterBody)),
               chunks=chunks, headline=headline(res),
               mech=[{k: int(w.mon[k][j]) for k in ('overlap_partner', 'interruptions_partner', 'overlap_cg',
                                                    'interruptions', 'garble', 'mouth_ticks', 'bouts') if k in w.mon}
                     for j in range(K)],
               per_agent=res.get('per_agent', [])[:K],
               overlap=res.get('overlap'), initiative={k: res.get('initiative', {}).get(k) for k in ('n', 'appropriate')})
    print(json.dumps(out, ensure_ascii=False, default=str), flush=True)
    os.makedirs(os.path.join(HERE, 'results', 'w2'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'w2', f'{tag}.json'), 'a') as f:
        f.write(json.dumps(out, ensure_ascii=False, default=str) + '\n')


if __name__ == '__main__':
    main()
