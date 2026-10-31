# -*- coding: utf-8 -*-
"""Embodied review probe: the real ARCH_v2 agent (synth/SynthAgent, final keys as in synth/world2_smoke.py) living in
world2 as a mouth-only body, with and without the bench's ear noise.  Counts what the agent itself registers:
heard lines (line ends), awake recalls (quiet moments), releases, hippocampal episodes.  Read-only use of synth/ and
bench/world2/.  usage: python3 -B noise_agent.py noise ticks n_agents"""
import os
import sys
import time
import json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..', '..', '..', 'v2', 'synth')
sys.path.insert(0, SYN)
import numpy as np                                   # noqa: E402
from agent_synth import SynthAgent                   # noqa: E402
from run_j import TUNE_BEST, typed                   # noqa: E402
from life_synth import THINK_LIFE                    # noqa: E402
sys.path.insert(0, os.path.join(SYN, '..', 'bench'))
from world2 import World2, PerAgent, MouthOnly, evaluate, headline   # noqa: E402
from world2_smoke import Shim                        # noqa: E402

FINAL = dict(ct=1, init_min_tau=10, hip_rho=2, ctx=0, ct_assoc=0.5, wander=50, wander_after=200)


def main():
    noise, ticks, n = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    kw = dict(TUNE_BEST); kw.update(THINK_LIFE); kw['rebind'] = 1
    for k, v in FINAL.items():
        kw[k] = typed(k, str(v))
    w = World2(n_agents=n, seed=0, noise=noise)
    sh = Shim(w, 1600)
    agents = []
    for j in range(n):
        a = SynthAgent(sh, seed=j, **kw)
        a.babble(w.voice, 35960)
        agents.append(MouthOnly(a))
    w.run(PerAgent(agents), ticks=ticks)
    res = evaluate(w.log)
    out = dict(noise=noise, ticks=ticks, n=n, secs=round(time.time() - t0),
               lines=[a.agent.stats['lines'] for a in agents],
               items=[a.agent.stats['items'] for a in agents],
               cats=[a.agent.Z.n for a in agents],
               quiet_recall_moments=[a.agent.core.stats['quiet'] for a in agents],
               awake_recalls=[a.agent.core.stats['awake_recalls'] for a in agents],
               hip=[a.agent.core.hip.n for a in agents],
               released=[a.agent.tstats.get('released', 0) for a in agents],
               blocked_nogo=[a.agent.tstats.get('blocked_nogo', 0) for a in agents],
               spoke=[a.agent.tstats['reply_spoke'] + a.agent.tstats['scene_spoke'] for a in agents],
               headline=headline(res))
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
