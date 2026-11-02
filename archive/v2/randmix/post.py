# -*- coding: utf-8 -*-
"""Extra measurements on a trained agent (out/agents/<tag>_s<seed>.pkl, written by run_rm.py after BCDD).
Bench and metrics unchanged: every probe is a bench World on the training seed (same physics, held-out sets and eye
codes), scored as the bench scores a free reply (reply in acceptable).
  1. held-out per task: B / C / D probe worlds with held_frac 0.5, 150 dialogues each (frozen as run_j's probes: B, C
     learning off; D with episode storage on, readout off).  Next to every held-out task: the trivial references on the
     SAME records: echo (reply = the prompt) and the best constant answer per task (the one string that is acceptable
     most often on that task's held-out records: an upper bound for any constant reply).
  2. fault tolerance: run_j's final probe worlds (held_frac 0.3, 80 dialogues) clean, and with one syllable of every
     prompt replaced (a uniformly drawn position) by (r) a random other lexicon syllable or (s) the most
     similar-sounding lexicon syllable (largest cosine of the heard 8x6 sound grid).  Same schedule (the corruption
     uses its own rng and keeps the prompt's length, so all timing is identical).
  3. drift: C probe before, after 300 and after 900 untutored C dialogues (learning on), as run_j / review2 drift_own.
usage: python3 post.py tag=<tag> seed=<s> [parts=hfd] [nH=150] [nF=80] [n1=300] [n2=600]"""
import os, sys, json, time, copy, pickle, resource
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np                                                   # noqa: E402
import rcore                                                         # noqa: E402,F401  (pickle classes)
from agent import BENCH                                              # noqa: E402
sys.path.insert(0, BENCH)
import world as world_mod                                            # noqa: E402
from world import World                                              # noqa: E402

A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
tag, seed = A['tag'], int(A['seed'])
parts = A.get('parts', 'hfd')
nH, nF, n1, n2 = int(A.get('nH', 150)), int(A.get('nF', 80)), int(A.get('n1', 300)), int(A.get('n2', 600))
WK = dict(eye_layout='retino')
t0 = time.time()


def log(s):
    print(f"[post {tag} s{seed} {time.time() - t0:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}MB] {s}",
          flush=True)


class FaultWorld(World):
    """a bench world whose free-reply prompts have one syllable replaced (acceptable replies unchanged)."""

    def __init__(self, *a, fmode=None, lex=None, sim=None, fseed=0, **kw):
        super().__init__(*a, **kw)
        self.fmode, self.lex, self.sim = fmode, lex, sim
        self.frng = np.random.default_rng(fseed)

    def _exchange(self, ex):
        if self.fmode and ex.get('kind') in ('probe', 'recast') and ex.get('prompt'):
            p = ex['prompt']
            i = int(self.frng.integers(len(p)))
            c = p[i]
            if self.fmode == 'sim':
                new = self.sim.get(c, c)
            else:
                new = c
                while new == c:
                    new = self.lex[int(self.frng.integers(len(self.lex)))]
            ex = dict(ex)
            ex['prompt'] = p[:i] + new + p[i + 1:]
        return super()._exchange(ex)


def run_probe(agent, stage, n, held_frac, fmode=None, lex=None, sim=None):
    ag = copy.deepcopy(agent)
    if stage in 'BC':
        ag.learning = False
    else:
        ag.learning = True
        ag.core.c.learn_listen = False
    w = FaultWorld(stage, seed=seed, tutor='recast', probe_frac=1.0, held_frac=held_frac, fmode=fmode, lex=lex,
                   sim=sim, fseed=seed * 31 + 7, **WK)
    w.run(ag, n)
    recs = [r for r in w.log if r['kind'] in ('probe', 'recast') and r['stage'] == stage]
    del ag
    return recs, w


def mech(recs):
    return dict(overlap=int(sum((r.get('mouth_while_partner') or 0) > 0 for r in recs)),
                garble=int(sum(r.get('garble') or 0 for r in recs)),
                not3=int(sum(sum(1 for x in (r.get('runs') or []) if x != 3) for r in recs)),
                extra=int(sum(r.get('extra_turns') or 0 for r in recs)))


def per_task(recs):
    out = {}
    for t in sorted(set(r['task'] for r in recs)):
        R = [r for r in recs if r['task'] == t]
        for lbl, S in (('seen', [r for r in R if not r['held_out']]), ('held', [r for r in R if r['held_out']])):
            if not S:
                continue
            acc = float(np.mean([r['reply'] in r['acceptable'] for r in S]))
            echo = float(np.mean([r['prompt'] in r['acceptable'] for r in S]))
            cands = sorted(set(a for r in S for a in r['acceptable']))
            best, bc = 0.0, None
            for c in cands:
                v = float(np.mean([c in r['acceptable'] for r in S]))
                if v > best:
                    best, bc = v, c
            silent = float(np.mean([r['reply'] == '' for r in S]))
            parrot = float(np.mean([r['reply'] != '' and r['reply'] in r['prompt'] for r in S]))
            out.setdefault(t, {})[lbl] = dict(n=len(S), acc=round(acc, 4), echo=round(echo, 4),
                                              const=round(best, 4), const_str=bc, silent=round(silent, 3),
                                              sub_prompt=round(parrot, 3))
    return out


def acc_all(recs):
    return round(float(np.mean([r['reply'] in r['acceptable'] for r in recs])), 4) if recs else None


with open(os.path.join(HERE, 'out', 'agents', f'{tag}_s{seed}.pkl'), 'rb') as f:
    agent = pickle.load(f)
log(f"loaded {type(agent.core).__name__} mix n {agent.core.mix.n}")
res = dict(tag=tag, seed=seed)
outp = os.path.join(HERE, 'out', 'post', f'{tag}_s{seed}.json')
os.makedirs(os.path.dirname(outp), exist_ok=True)


def save():
    res['secs'] = round(time.time() - t0, 1)
    res['rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    json.dump(res, open(outp, 'w'), ensure_ascii=False, indent=0, default=str)


if 'h' in parts:
    H = {}
    allrecs = []
    for st in 'BCD':
        recs, _ = run_probe(agent, st, nH, 0.5)
        allrecs += recs
        H[st] = per_task(recs)
        H[st]['_mech'] = mech(recs)
        log(f"held-out probe {st}: " + ", ".join(f"{t} {v['held']['acc']:.2f}/{v['held']['n']} (echo {v['held']['echo']:.2f}, "
                                                   f"const {v['held']['const']:.2f})" for t, v in H[st].items()
                                                   if not t.startswith('_') and 'held' in v))
    res['heldout'] = H
    save()

if 'f' in parts:
    lex = sorted(world_mod.LEXICON_CHARS)
    v = agent_voice = World('B', seed=seed, **WK).voice
    G = {c: v.hear(c, 'other').astype(np.float64).ravel() for c in lex}
    sim, simv = {}, {}
    for c in lex:
        best, bc = -1.0, None
        for d in lex:
            if d == c:
                continue
            s = float(G[c] @ G[d] / (np.linalg.norm(G[c]) * np.linalg.norm(G[d]) + 1e-9))
            if s > best:
                best, bc = s, d
        sim[c], simv[c] = bc, best
    # how the agent's ear treats the similar syllables: the same Z category or not
    FT = {}
    for mode in (None, 'rand', 'sim'):
        recs_all = []
        T = {}
        for st in 'BCD':
            recs, _ = run_probe(agent, st, nF, 0.3, fmode=mode, lex=lex, sim=sim)
            recs_all += recs
            T[st] = dict(all=acc_all(recs), seen=acc_all([r for r in recs if not r['held_out']]),
                         held=acc_all([r for r in recs if r['held_out']]), n=len(recs), mech=mech(recs),
                         tasks={t: acc_all([r for r in recs if r['task'] == t]) for t in sorted(set(r['task'] for r in recs))})
        T['ALL'] = acc_all(recs_all)
        FT[mode or 'clean'] = T
        log(f"fault {mode or 'clean'}: ALL {T['ALL']} B {T['B']['all']} C {T['C']['all']} D {T['D']['all']}")
    FT['sim_mean_cos'] = round(float(np.mean(list(simv.values()))), 3)
    FT['sim_examples'] = {c: [sim[c], round(simv[c], 3)] for c in list(lex)[:12]}
    res['fault'] = FT
    save()

if 'd' in parts:
    def cprobe(ag):
        recs, _ = run_probe(ag, 'C', 80, 0.3)
        return dict(all=acc_all(recs), tasks={t: acc_all([r for r in recs if r['task'] == t])
                                              for t in sorted(set(r['task'] for r in recs))}, mech=mech(recs))
    D = dict(before=cprobe(agent))
    agent.learning = True
    wf = World('C', seed=seed, tutor='recast', probe_frac=1.0, held_frac=0.0, **WK)
    wf.run(agent, n1)
    D[f'after{n1}'] = cprobe(agent)
    log(f"drift: {D['before']['all']} -> {D[f'after{n1}']['all']} after {n1}")
    wf.run(agent, n2)
    D[f'after{n1 + n2}'] = cprobe(agent)
    log(f"drift: -> {D[f'after{n1 + n2}']['all']} after {n1 + n2}")
    D['mech_untutored'] = mech([r for r in wf.log if r['kind'] in ('probe', 'recast')])
    res['drift'] = D
    save()
log('done')
