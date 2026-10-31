# -*- coding: utf-8 -*-
"""v2.1 copy of synth/life_synth.py: AgentV21 (ral=0 = ARCH_v2 final_noctx life config incl. ct_assoc 0.5 / wander 50),
+ the v2.1 life metrics (life_metrics.py).  Life mode (bench v2.0 life.py): one unbroken life from birth, learning always on, nothing reset, no frozen copies.
Babbling pass at birth (as tune's life_run.py: babbling memory -> later-heard syllables are imitated at once).
Prints evaluate_life summaries as it goes and saves the summary (tune's format) + the log.
usage: python3 life_synth.py tag=x seed=5 ticks=1000000 chunk=100000 [base=think|v1] [w_<world key>=v] [agent/cfg keys]"""
import os, sys, json, time, pickle
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from agent_v21 import AgentV21 as SynthAgent  # noqa: E402
from agent import BENCH                      # noqa: E402
sys.path.insert(0, BENCH)
from life import Life, evaluate_life, headline_life   # noqa: E402
from run_v21 import TUNE_BEST, typed, ARCH_V2  # noqa: E402
from life_metrics import metrics                # noqa: E402

THINK_LIFE = dict(policy='free', ior=0, think_k=3, urg_gate=False, brake_now=True, late_release=False, u_reply=0.15,
                  u_R1=1.0, own_tail=True, g0_scene=1.0, u_scene=0.02, oth_kind=True, life_eye=True, eye_feats=True,
                  no_react=True)


def summary(res):
    return dict(hours=res['hours'], ticks=res['ticks'], n_free=res['n_free'],
                acc=res['tasks'].get('ALL', {}).get('acc'),
                groups={k: dict(acc=v.get('acc'), n=v.get('n'), curve=v.get('curve')) for k, v in res['groups'].items()},
                tasks={k: dict(acc=v.get('acc'), n=v.get('n')) for k, v in res['tasks'].items() if isinstance(v, dict)},
                curves=res.get('curves'), retention=res.get('retention'), stability=res.get('stability'),
                flags=res.get('stability_flags'), overlap=res.get('overlap'), initiative=res.get('initiative'),
                monologue=res.get('monologue'), latency=res.get('latency'), mouth=res.get('mouth'))


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, seed = A.pop("tag", "life"), int(A.pop("seed", 5))
    ticks, chunk = int(float(A.pop("ticks", 10 ** 6))), int(float(A.pop("chunk", 10 ** 5)))
    base = A.pop("base", "think")
    tutor = A.pop("tutor", "recast")
    wkw = {'eye_layout': 'retino'}
    for k in list(A):
        if k.startswith("w_"):
            wkw[k[2:]] = A.pop(k)
    kw = dict(TUNE_BEST); kw.update(ARCH_V2); kw.update(ct_assoc=0.5, wander=50, wander_after=200)
    if base == "think":
        kw.update(THINK_LIFE)
    else:
        kw.update(policy='v1', think_k=3, ior=0, life_eye=True)
    for k, v in A.items():
        kw[k] = typed(k, v)
    t0 = time.time()
    L = Life(seed=seed, tutor=tutor, **wkw)
    agent = SynthAgent(L, seed=seed, **kw)
    agent.babble(L.voice, 35960)
    print(f"[{tag}] life seed {seed} ticks {ticks} base {base} world {wkw} kw {kw} (babbling {time.time() - t0:.0f}s)",
          flush=True)
    os.makedirs(os.path.join(HERE, "out", "life"), exist_ok=True)
    t_log = time.time()
    outp = os.path.join(HERE, "out", "life", f"{tag}_s{seed}.json")
    done = 0
    while done < ticks:
        L.run(agent, ticks=min(chunk, ticks - done))
        done = L.t
        res = evaluate_life(L.log)
        ini = res['initiative']
        print(f"[{tag}] {time.time() - t0:.0f}s " + headline_life(res) + f" | init {ini.get('n')} cats {ini.get('categories')} "
              f"| cats {agent.Z.n} progs {len(agent.cat2prog)} mix {agent.core.mix.n} hip {agent.core.hip.n} "
              f"recalls {agent.core.stats['awake_recalls']} | t {agent.tstats} ct {agent.core.ct_stats} "
              f"name {agent.core.name_stats}", flush=True)
        json.dump(dict(tag=tag, seed=seed, cfg={k: str(v) for k, v in kw.items()}, world=wkw, secs=round(time.time() - t0, 1),
                       tstats=agent.tstats, ct=agent.core.ct_stats, name=agent.core.name_stats, summary=summary(res)),
                  open(outp, "w"), ensure_ascii=False, default=str)
    with open(os.path.join(HERE, "out", "life", f"{tag}_s{seed}_log.pkl"), "wb") as f:
        pickle.dump(L.log, f)
    # favourite thoughts: what the thought holds at the first heard line after >= 2,000 silent ticks
    item_n = agent.core.item_n[:agent.core.c.V].astype(float)
    top = set(np.argsort(-item_n)[:20].tolist())
    fav = agent.fav_log
    held_top = [np.mean([i in top for i in it]) for (_, _, it) in fav if it]
    base_top = float(np.mean([i in top for i in range(agent.core.c.V) if item_n[i] > 0]))
    favs = dict(n=len(fav), nonempty=sum(1 for f in fav if f[2]), share_top20_held=float(np.mean(held_top)) if held_top else None,
                share_top20_vocab=base_top, mean_held=float(np.mean([len(f[2]) for f in fav])) if fav else None)
    print(f"[{tag}] favourites after long silences: {favs}", flush=True)
    d = json.load(open(outp)); d['favourites'] = favs; d['agent_stats'] = agent.stats
    d['v21'] = metrics(L.log); d['ral'] = agent.core.ral_stats; d['say'] = agent.core.say_stats; d['val'] = agent.core.val_stats
    print(f"[{tag}] v21 metrics {json.dumps(d['v21'], ensure_ascii=False)}", flush=True)
    json.dump(d, open(outp, "w"), ensure_ascii=False, default=str)
    print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
