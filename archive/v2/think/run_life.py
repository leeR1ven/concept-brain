# -*- coding: utf-8 -*-
"""One unbroken life (bench v2.0 life.py, copied into bench_copy/) for the develop agent (v1 GO timer) and the think
agent (speech from thought).  Both get the same two life adaptations (documented):
  * eye: extra eye bits that appear while the ear is silent (the partner's 'person' code, an object appearing) are
    absorbed into the scene baseline (develop's parser would otherwise take the person code for a pointing gesture);
  * babbling: develop's offline babbling pass (all muscle sets once; inherited shortcut) after the first pre_ticks of
    listening (before it the agent has no programs and cannot speak).
usage: python3 run_life.py agent=think|dev seed=0 tutor=recast pre=30000 ticks=600000 tag=x [agent keys] [cfg keys]"""
import os, sys, time, json, pickle, collections
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import DevAgent, BENCH           # noqa: E402
from agent_think import ThinkAgent          # noqa: E402
sys.path.insert(0, BENCH)
from life import Life, evaluate_life, headline_life   # noqa: E402
from core_think import TCfg                 # noqa: E402
import numpy as np                          # noqa: E402


class DevLifeAgent(DevAgent):
    """develop's agent unchanged except the life eye fix (absorb silent additions into the scene baseline)."""
    life_eye = True
    base_changed = False

    def step(self, obs):
        self._E_now = float(np.asarray(obs['ear']).sum())
        return DevAgent.step(self, obs)

    _parse_eye = ThinkAgent._parse_eye


A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
def arg(k, d):
    v = A.pop(k, None)
    if v is None:
        return d
    if isinstance(d, bool):
        return v in ("1", "True", "true")
    return type(d)(v)
kind, seed, tutor = arg("agent", "think"), arg("seed", 0), arg("tutor", "recast")
pre, ticks, tag, nbab = arg("pre", 30000), arg("ticks", 600000), arg("tag", "life"), arg("babbles", 35960)
chunk = arg("chunk", 100000)
AG_KEYS = dict(policy='free', theta_I=1.0, g0_reply=1.0, g0_scene=1.0, u_reply=0.05, u_scene=0.02, beta=2.0,
               g_max=3.0, nogo=0.25, nogo_late=0.05, L=10, eta_oth=0.2, eta_go=0.2, learn_turn=True, tau_w=7,
               urg_gate=True, close_on_scene=True, oth_kind=True, brake_now=True, life_eye=True, eye_feats=True,
               no_react=True, late_release=True, u_R1=1.0, own_tail=False)
akw, cfgkw = {}, {}
for k, v in A.items():
    if k in AG_KEYS:
        d = AG_KEYS[k]
        akw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
    else:
        d = getattr(TCfg(), k)
        cfgkw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
cfgkw.setdefault('floor', 0.3)
OUT = os.path.join(HERE, "out", f"life_{tag}")
logf = open(OUT + ".log", "w")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

t0 = time.time()
L = Life(seed=seed, tutor=tutor)
if kind == 'dev':
    agent = DevLifeAgent(L, seed=seed, **cfgkw)
else:
    kw = dict(AG_KEYS); kw.update(akw)
    agent = ThinkAgent(L, seed=seed, **kw, **cfgkw)
log(f"life seed {seed} tutor {tutor} agent {kind} {akw} cfg {cfgkw} pre {pre} ticks {ticks}")
L.run(agent, ticks=pre)
log(f"[pre] t={L.t}: categories {agent.Z.n}, items {agent.stats['items']}, lines {agent.stats['lines']} ({time.time()-t0:.0f}s)")
agent.babble(L.voice, nbab)
log(f"[pre] babbling: programs bound {len(agent.cat2prog)} of {agent.Z.n} categories")
t_start = L.t
while L.t < t_start + ticks:
    L.run(agent, ticks=min(chunk, t_start + ticks - L.t))
    res = evaluate_life([r for r in L.log])
    ini = res['initiative']
    log(f"[life t={L.t}] {headline_life(res)} | init {ini.get('n')} ({ini.get('per_1000_quiet_ticks', 0):.2f}/1000q, "
        f"appropriate {ini.get('appropriate_share')}, cats {ini.get('categories')}) ({time.time()-t0:.0f}s)")
    if hasattr(agent, 'tstats'):
        log(f"   thoughts {agent.tstats} body go_blocked {agent.body.go_blocked} now_blocked {getattr(agent.body, 'now_blocked', 0)}")
res = evaluate_life(L.log)
json.dump(res, open(OUT + ".json", "w"), ensure_ascii=False, default=str)
pickle.dump(L.log, open(OUT + "_log.pkl", "wb"))
if hasattr(agent, 'th_log'):
    pickle.dump(agent.th_log, open(OUT + "_thoughts.pkl", "wb"))
log(f"done {time.time()-t0:.0f}s")
