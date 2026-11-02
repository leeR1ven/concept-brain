# -*- coding: utf-8 -*-
"""Smoke run: the synthesized agent living in world2 (bench/world2) as a mouth-only body (MouthOnly), N agents, one
unbroken life, learning on.  It only shows that the agent runs there end to end; it is not a result.
The agent uses world2's (1600,) eye as a flat eye (its 3-slot parser treats the egocentric view as one scene).
usage: python3 world2_smoke.py n=4 days=1 seed=0 [agent/cfg keys]"""
import os, sys, time, json
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
from agent_synth import SynthAgent                   # noqa: E402
from run_j import TUNE_BEST, typed                   # noqa: E402
from life_synth import THINK_LIFE                    # noqa: E402
BENCH2 = os.path.join(HERE, '..', 'bench')
sys.path.insert(0, BENCH2)
from world2 import World2, PerAgent, MouthOnly, evaluate, headline   # noqa: E402


class Shim:
    """what the agent reads from a world at birth: seed, the warm-up windows, the voice (babbling), eye size."""
    def __init__(self, w2, n_eye):
        self.seed = w2.seed
        self.voice = w2.voice
        self.eye = np.zeros(n_eye, np.uint8)
        self.p = {'eye_layout': 'random'}
        self._w2 = w2

    def warmup_windows(self):
        return self._w2.warmup_windows()


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, days, seed = int(A.pop('n', 4)), float(A.pop('days', 1)), int(A.pop('seed', 0))
    kw = dict(TUNE_BEST); kw.update(THINK_LIFE); kw['rebind'] = 1
    for k, v in A.items():
        kw[k] = typed(k, v)
    t0 = time.time()
    w = World2(n_agents=n, seed=seed)
    sh = Shim(w, 1600)
    agents = []
    for j in range(n):
        a = SynthAgent(sh, seed=seed * 100 + j, **kw)
        a.babble(w.voice, 35960)
        agents.append(MouthOnly(a))
    print(f"[w2] {n} agents built + babbled in {time.time() - t0:.0f}s", flush=True)
    ticks = int(days * 30000)
    done = 0
    while done < ticks:
        step = min(15000, ticks - done)
        log = w.run(PerAgent(agents), ticks=step)
        done += step
        res = evaluate(w.log if hasattr(w, 'log') else log)
        print(f"[w2] t={done} {time.time() - t0:.0f}s {headline(res)} | progs {[len(a.agent.cat2prog) for a in agents]} "
              f"cats {[a.agent.Z.n for a in agents]} spoke {[a.agent.tstats['reply_spoke'] + a.agent.tstats['scene_spoke'] for a in agents]}",
              flush=True)
    json.dump(dict(n=n, days=days, secs=round(time.time() - t0), headline=headline(res)),
              open(os.path.join(HERE, 'out', 'world2_smoke.json'), 'w'), ensure_ascii=False)


if __name__ == "__main__":
    main()
