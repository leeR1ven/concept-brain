# -*- coding: utf-8 -*-
"""Test (review2/develop): role-aware listening, minimal form, on run_j's drift protocol.
Rule ("a line that gets answered was a first pair part, not an answer"): the readout and gate updates made while a
heard line is learned in the context of the held thought are kept on an eligibility trace until the line's role is
known; if the agent itself starts speaking within W ticks after the line ends (it treats the line as addressed to it),
the line was a question/initiation -> the trace is cancelled (the updates are undone) and the hippocampal NEXT link
from the previous line to it is cut (so awake recall cannot re-teach it as an answer).  Otherwise the updates stay.
Local: per synapse a one-line trace of its last change + one broadcast signal (own speech onset = efference copy).
usage: python3 drift_veto.py veto=0|1 [W=12] [n=300]"""
import collections, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..')  # v21 copy: code from arch/v2/v21; agent pickle from $AGENT (default synth's seed-0 final agent)
PKL = os.environ.get('AGENT', os.path.join(HERE, '..', '..', 'synth', 'out', 'agent_final_s0.pkl'))
sys.path.insert(0, SYN); sys.path.insert(0, os.path.join(SYN, 'bench_copy'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from world import World            # noqa
import agent_synth, agent_v21      # noqa
A = dict(a.split('=') for a in sys.argv[1:])
VETO, W, N = int(A.get('veto', 1)), int(A.get('W', 12)), int(A.get('n', 300))
agent = pickle.load(open(PKL, 'rb'))
core = agent.core
S = dict(rec_on=False, buf=[], pending=None, pv=None, prev_mouth=False, vetoed=0, kept=0, cut=0)
def wrap_learn(obj):
    orig = obj.learn
    def learn(Aa, target, total, J=None, **kw):
        if not (S['rec_on'] and not core._replaying and len(Aa)):
            return orig(Aa, target, total, J=J, **kw)
        Jj = np.arange(obj.W.shape[1]) if J is None else np.asarray(J)
        Ai = np.asarray(Aa)
        before = obj.W[np.ix_(Ai, Jj)].copy()
        out = orig(Aa, target, total, J=J, **kw)
        S['buf'].append((obj, Ai, Jj, obj.W[np.ix_(Ai, Jj)] - before))
        return out
    obj.learn = learn
wrap_learn(core.gen); wrap_learn(core.gate)
orig_step = agent.step
def step(obs):
    was = agent.in_line
    pv_before = core.last_rec
    m = orig_step(obs)
    now = agent.in_line
    if VETO and agent.learning:
        if not was and now:                          # a heard line starts: the previous pending line is committed
            if S['pending'] is not None:
                S['kept'] += 1
            S['pending'] = None; S['buf'] = []; S['rec_on'] = True; S['pv'] = pv_before
        if was and not now:                          # the heard line ended (its END was learned in this step)
            S['pending'] = dict(buf=S['buf'], t=agent.t, pv=S['pv'], rec=core.last_rec); S['buf'] = []; S['rec_on'] = False
        mouth = bool(np.any(np.asarray(m) > 0))
        p = S['pending']
        if mouth and not S['prev_mouth'] and p is not None:
            if agent.t - p['t'] <= W:                # I answered it: it was addressed to me (a first pair part)
                for obj, Ai, Jj, d in p['buf']:
                    obj.W[np.ix_(Ai, Jj)] -= d
                pv, rec = p['pv'], p['rec']
                if pv is not None and rec is not None and pv.get('ep', -1) >= 0 and pv.get('next') == rec.get('ep'):
                    pv['next'] = None; core.next_of[pv['ep']] = -1; S['cut'] += 1
                S['vetoed'] += 1
            else:
                S['kept'] += 1
            S['pending'] = None
        S['prev_mouth'] = mouth
    return m
agent.step = step
Q = set('什么吗哪几') | {'不'}
def kind(r, prompts):
    rep = r['reply']
    if rep in r['acceptable']:
        return 'ok'
    if not rep:
        return 'silent'
    if rep in prompts or any(ch in Q for ch in rep):
        return 'partner line'
    return 'other'
def run(kw, n, learning):
    agent.learning = learning
    w = World('C', seed=0, tutor='recast', eye_layout='retino', **kw)
    w.run(agent, n)
    return [r for r in w.log if r.get('kind') == 'probe']
before = run(dict(probe_frac=1.0, held_frac=0.3), 80, False)
drift = run(dict(probe_frac=1.0, held_frac=0.0), N, True)
after = run(dict(probe_frac=1.0, held_frac=0.3), 80, False)
prompts = {r['prompt'] for r in before + drift + after}
res = {}
for name, rs in (('before', before), ('drift last third', drift[-len(drift) // 3:]), ('after', after)):
    c = collections.Counter(kind(r, prompts) for r in rs)
    res[name] = {k: round(v / len(rs), 3) for k, v in c.items()}
    print(f'veto={VETO} {name:16s} n {len(rs):3d} ' + '  '.join(f'{k} {v / len(rs):.2f}' for k, v in sorted(c.items())), flush=True)
res['held_after'] = round(float(np.mean([r['reply'] in r['acceptable'] for r in after if r.get('held_out')])), 3)
res['seen_after'] = round(float(np.mean([r['reply'] in r['acceptable'] for r in after if not r.get('held_out')])), 3)
res['stats'] = {k: S[k] for k in ('vetoed', 'kept', 'cut')}
print(res['stats'], 'seen after', res['seen_after'], 'held after', res['held_after'])
json.dump(res, open(os.path.join(HERE, f'drift_veto{VETO}.json'), 'w'), ensure_ascii=False)
