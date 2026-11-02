# -*- coding: utf-8 -*-
"""Staged driver for the meaning layer = v21's run_v21.py (frozen protocol, J / J2, per-task held-out with same-run
references, drift 300 / 900, role confusion, lesions; metric code untouched) with three changes, none in the metrics:
  1. the canonical tie rule (gpuport/run_j_canon.py): the DG k-WTA breaks exact ties by unit index (stable sort), and
     SparsePR sums in CPython 3.12's order;
  2. the agent is v2.1's AgentV21 with PFC-2 = the random-at-birth layer (rm=1, rm_* = RandMix arguments; default
     randmix's c_flat: train, prune, age 20000, regrow, flat) reading sound codes, plus (mean=<mode>, mean_* keys) the
     learned meaning codes of meanlayer.py; mean=none = the v2.1 + c_flat base;
  3. a side file out/runs/<tag>_s<seed>.meaning.json: code statistics on the bench lexicon after every training stage
     (within / between class Jaccard, NN purity over objects / colours / counts / slots / names, code drift between
     stages), layer size, cache clears.  save=1 pickles the agent for post_mean.py.
usage: python3 run_mean.py tag=x seed=0 rm=1 [mean=ri mean_km=16 ...] [run_v21 keys ...]"""
import os, sys, json, time
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _d in ('synth', 'randmix', 'v21'):   # repo: the unchanged ARCH_v2, randmix and v2.1 code
    if os.path.join(HERE, '..', _d) not in sys.path:
        sys.path.insert(1, os.path.join(HERE, '..', _d))
import rules  # noqa: E402


def dg(self, inputs):
    d = self.tie.copy()
    for g, x in inputs.items():
        if g in self.idx and x is not None and x.any():
            d += self.wg[g] * x[self.idx[g]].sum(1)
    code = np.zeros(self.n_dg, bool)
    code[np.argsort(-d, kind='stable')[:self.k_dg]] = True
    return code


rules.Hippo.dg = dg


def _sum312(xs):
    xs = list(xs)
    if not xs:
        return 0
    f = 0 + xs[0]; c = 0.0
    for x in xs[1:]:
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c and np.isfinite(c):
        f += c
    return f


import agent_think  # noqa: E402


def _p(self, keys, bias=None):
    b = self.init if bias is None else bias
    return float(np.clip(b + _sum312(self.w.get(k, 0.0) for k in keys), 0.0, 1.0))


agent_think.SparsePR.p = _p


def num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


RM, MK, MEAN, USE_RM = dict(train=1, prune=1, age=20000, regrow=1, alloc='flat'), {}, 'none', 0
rest = []
for a in sys.argv[1:]:
    k, _, v = a.partition('=')
    if k == 'rm':
        USE_RM = int(v)
    elif k.startswith('rm_'):
        RM[k[3:]] = num(v)
    elif k == 'mean':
        MEAN = v
    elif k.startswith('mean_'):
        MK[k[5:]] = num(v)
    else:
        rest.append(a)
sys.argv = [sys.argv[0]] + rest
ARGS = dict(a.split('=', 1) for a in rest if '=' in a)

import run_v21  # noqa: E402
run_v21.HERE = HERE   # repo: run_v21 lives in ../v21; its runs and pickles go to this folder's out/ as before
from mcore import make_agent  # noqa: E402
from meanlayer import code_stats, code_drift  # noqa: E402
from physics import Stream  # noqa: E402
import world as world_mod  # noqa: E402

AGENT = {}


def factory(world, seed=0, **kw):
    mean = None if MEAN == 'none' else dict(mode=MEAN, **MK)
    ag = make_agent(world, seed=seed, rm=RM if USE_RM else None, mean=mean, **kw)
    AGENT['a'] = ag
    return ag


run_v21.AgentV21 = factory
SNAP = []
_World = run_v21.World


class SnapWorld(_World):
    """training worlds (probe_frac != 1) snapshot the meaning codes after they ran"""

    def run(self, agent, n, *a, **kw):
        out = super().run(agent, n, *a, **kw)
        core = getattr(agent, 'core', None)
        if self.p.get('probe_frac') != 1.0 and getattr(core, 'mcodes', None) is not None and agent is AGENT.get('a'):
            if core.mcodes.mode in ('oracle', 'oracle2'):          # control: the bench's word classes, given
                G = lex_groups(agent, self.voice)
                cnt = {}
                for v in G.values():
                    for i in v:
                        cnt[i] = cnt.get(i, 0) + 1
                core.mcodes.set_classes({i: g for g, v in G.items() for i in v if cnt[i] == 1},
                                        core.mcodes.kcls if core.mcodes.mode == 'oracle2' else core.mcodes.km)
            core.mcodes.recompute()
            SNAP.append((self.stage if hasattr(self, 'stage') else '?', int(core.mcodes.stats['lines']),
                         core.mcodes.snapshot(), lex_groups(agent, self.voice)))
        return out


run_v21.World = SnapWorld


def lex_groups(agent, voice):
    """the agent's own categories of the lexicon's syllables, by word class and syllable position (frozen ear)."""
    G = {}
    for name, words in (('obj', world_mod.OBJECTS), ('col', world_mod.COLOURS), ('cnt', world_mod.COUNTS),
                        ('slot', world_mod.SLOTS), ('name', world_mod.NAMES)):
        for pos in (0, 1):
            cats = []
            for w in words:
                ch = w[pos]
                st = Stream(voice.F, voice.T, voice.T // 3, 3, voice.levels)
                e = agent._ear_copy(); none = (None, None, False); g = voice.hear(ch, 'other'); out = None
                for k in range(3):
                    out = e.step(st.tick(((ch, g, k == 0), none, none)))
                j, m = agent.Z.best(out[agent.F0])
                if m >= agent.theta_z:
                    cats.append(int(j))
            G[f'{name}{pos + 1}'] = sorted(set(cats))
    return G


def main():
    t0 = time.time()
    run_v21.main()
    ag = AGENT.get('a')
    core = ag.core if ag is not None else None
    tag, seed = ARGS.get('tag', 'x'), int(ARGS.get('seed', 0))
    meta = dict(tag=tag, seed=seed, rm=RM if USE_RM else None, mean=MEAN, mk=MK, secs=round(time.time() - t0, 1))
    if core is not None and hasattr(core.mix, 'snapshot'):
        meta['layer'] = core.mix.snapshot('end')
        meta['clears'] = getattr(core.mix, 'mstats', {}).get('clears')
    mc = getattr(core, 'mcodes', None)
    if mc is not None and SNAP:
        st = []
        for k, (stage, nl, snap, G) in enumerate(SNAP):
            cnt = {}
            for v in G.values():
                for i in v:
                    cnt[i] = cnt.get(i, 0) + 1
            groups = {g: [i for i in v if i in snap and cnt[i] == 1] for g, v in G.items() if v}   # unambiguous items
            groups = {g: v for g, v in groups.items() if len(v) >= 2}
            d = dict(stage=k, lines=nl, n_codes=len(snap), stats=code_stats(snap, groups),
                     drift_from_prev=code_drift(SNAP[k - 1][2], snap) if k else None)
            st.append(d)
        meta['codes'] = st
        meta['mstats'] = mc.stats
        meta['groups'] = SNAP[-1][3]
    out = os.path.join(HERE, 'out', 'runs', f'{tag}_s{seed}.meaning.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(meta, open(out, 'w'), ensure_ascii=False, default=str, indent=0)
    print(f"[{tag} s{seed}] meaning meta: " + json.dumps({k: meta[k] for k in meta if k in ('layer', 'clears', 'mstats')},
                                                         default=str), flush=True)
    if mc is not None and SNAP:
        for d in meta['codes']:
            print(f"[{tag} s{seed}] codes after stage {d['stage']}: {d['stats']} drift {d['drift_from_prev']}", flush=True)


if __name__ == '__main__':
    main()
