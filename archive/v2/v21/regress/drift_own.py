# -*- coding: utf-8 -*-
"""Skeptic's cheap test of the drift mechanism on the saved final agent (seed 0, agent_final_s0.pkl; read-only).
Condition A (as ARCH_v2): own speech never enters the thought (ct_own = 0): 300 untutored C dialogues, probe C, then
  600 more untutored C dialogues, probe C again (does the free-use loss keep growing?).
Condition B: the same agent reloaded, own speech enters the thought at 0.6 (ct_own = 0.6, above threshold 0.5), so the
  partner's next line is learned in a context that contains the agent's own answer; 300 untutored C dialogues, probe C.
Probes = run_j.py's C probe (80 dialogues, held_frac 0.3, probe_frac 1, learning off), done in place (no deepcopy, to
stay under the RAM limit).  usage: python3 drift_own.py [n1=300] [n2=600]"""
import os, sys, pickle, time, resource
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
SYN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # v21 copy
sys.path.insert(0, SYN)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
sys.path.insert(0, os.path.join(SYN, 'bench_copy'))
from world import World          # noqa: E402
from metrics import evaluate     # noqa: E402

A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
n1, n2 = int(A.get('n1', 300)), int(A.get('n2', 600))
PKL = os.environ.get('AGENT', os.path.join(SYN, '..', 'synth', 'out', 'agent_final_s0.pkl'))
import agent_synth, agent_v21  # noqa: E402,F401
t0 = time.time()


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def probe(ag):
    ag.learning = False
    w = World('C', seed=0, tutor='recast', probe_frac=1.0, held_frac=0.3, eye_layout='retino')
    w.run(ag, 80)
    S = evaluate(w.log)['C']
    ag.learning = True
    return {k: (round(v['acc'], 3), v['n']) for k, v in S.items() if isinstance(v, dict) and v.get('acc') is not None
            and k in ('ALL', 'colour', 'count', 'offer', 'taste', 'ritual')}


def load():
    with open(PKL, 'rb') as f:
        ag = pickle.load(f)
    ag.learning = True
    return ag


def log(s):
    print(f'[{time.time() - t0:.0f}s rss {rss()} MB] {s}', flush=True)


ag = load()
log(f'loaded; ct={ag.core.c.ct} ct_own={ag.core.c.ct_own} ctx={ag.core.c.ctx}')
log(f'A before: {probe(ag)}')
wf = World('C', seed=0, tutor='recast', probe_frac=1.0, held_frac=0.0, eye_layout='retino')
wf.run(ag, n1)
log(f'A after {n1} untutored: {probe(ag)}')
wf.run(ag, n2)
log(f'A after {n1 + n2} untutored: {probe(ag)}')
del ag, wf

ag = load()
ag.core.c.ct_own = 0.6
log(f'B (ct_own=0.6) before: {probe(ag)}')
wf = World('C', seed=0, tutor='recast', probe_frac=1.0, held_frac=0.0, eye_layout='retino')
wf.run(ag, n1)
log(f'B after {n1} untutored: {probe(ag)}')
log('done')
