# -*- coding: utf-8 -*-
"""Robustness check in the bench's continuous life mode (bench v2.0 life.py, copied into bench_copy/).
One unbroken life from birth (tutor='recast'); offline babbling pass at birth (babbling memory -> later-heard syllables
are imitated at once); learning always on; nothing reset.  Prints evaluate_life summaries as it goes.
usage: python3 life_run.py tag=x seed=5 ticks=1000000 chunk=100000 [agent / cfg keys ...]"""
import os, sys, json, time
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import DevAgent, BENCH          # noqa: E402
sys.path.insert(0, BENCH)
from life import Life, evaluate_life, headline_life   # noqa: E402
from core import Cfg                       # noqa: E402

AGENT_KW = {'theta_z': float, 'z_rate': float, 'quiet_ticks': int}


def summary(res):
    out = dict(hours=res['hours'], ticks=res['ticks'], n_free=res['n_free'],
               acc=res['tasks'].get('ALL', {}).get('acc'),
               groups={k: dict(acc=v.get('acc'), n=v.get('n'), curve=v.get('curve')) for k, v in res['groups'].items()},
               tasks={k: dict(acc=v.get('acc'), n=v.get('n')) for k, v in res['tasks'].items() if isinstance(v, dict)},
               curves=res.get('curves'), retention=res.get('retention'), stability=res.get('stability'),
               flags=res.get('stability_flags'), overlap=res.get('overlap'), initiative=res.get('initiative'),
               monologue=res.get('monologue'), latency=res.get('latency'), mouth=res.get('mouth'))
    return out


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, seed = A.pop("tag", "life"), int(A.pop("seed", 5))
    ticks, chunk = int(float(A.pop("ticks", 10 ** 6))), int(float(A.pop("chunk", 10 ** 5)))
    akw = {k: AGENT_KW[k](A.pop(k)) for k in list(A) if k in AGENT_KW}
    cfgkw = {}
    for k, v in A.items():
        d = getattr(Cfg(), k)
        cfgkw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
    cfgkw.setdefault("floor", 0.3)
    t0 = time.time()
    L = Life(seed=seed, tutor='recast')
    agent = DevAgent(L, seed=seed, life_eye=True, **akw, **cfgkw)
    agent.babble(L.voice, 35960)
    print(f"[{tag}] life seed {seed} ticks {ticks} cfg {cfgkw} agent {akw} (babbling {time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(HERE, "out", "life"), exist_ok=True)
    outp = os.path.join(HERE, "out", "life", f"{tag}_s{seed}.json")
    done = 0
    while done < ticks:
        L.run(agent, ticks=min(chunk, ticks - done))
        done = L.t
        res = evaluate_life(L.log)
        print(f"[{tag}] {time.time() - t0:.0f}s " + headline_life(res) + f" | cats {agent.Z.n} progs {len(agent.cat2prog)} "
              f"mix {agent.core.mix.n} hip {agent.core.hip.n} recalls {agent.core.stats['awake_recalls']}", flush=True)
        json.dump(dict(tag=tag, seed=seed, cfg=cfgkw, agent=akw, secs=round(time.time() - t0, 1), summary=summary(res)),
                  open(outp, "w"), ensure_ascii=False, default=str)
    print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
