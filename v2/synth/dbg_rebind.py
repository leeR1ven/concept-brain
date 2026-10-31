import sys; sys.path.insert(0, '.')
from agent_synth import SynthAgent
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from metrics import evaluate
from run_j import TUNE_BEST, THINK_FINAL
kw = dict(TUNE_BEST); kw.update(THINK_FINAL)
wA = World('A', seed=1, tutor='recast')
ag = SynthAgent(wA, seed=1, rebind=1, **kw)
wA.run(ag, 300); ag.babble(wA.voice, 35960)
print('after A: progs', len(ag.cat2prog), 'of', ag.Z.n)
w = World('C', seed=1, tutor='recast', prev=wA); w.run(ag, 150)
print('after C150: progs', len(ag.cat2prog), 'of', ag.Z.n, 'rebound', ag.stats['rebound'])
r = evaluate(w.log)['C']
print('ritual', r['ritual']['acc'], r['ritual']['n'])
