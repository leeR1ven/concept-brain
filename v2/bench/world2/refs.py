# -*- coding: utf-8 -*-
"""Reference runs and speed table for world2.

    python3 refs.py [days] [seeds] [--eye fovea]   (default 10 days, seeds 0 1 2; ~6 min)
        uniform eye -> results/*.json, checks/refs.out;  --eye fovea -> results_fovea/*.json, checks/refs_fovea.out
"""
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from world2 import World2, REFERENCES, evaluate, headline  # noqa: E402

ARGS = [a for a in sys.argv[1:]]
EYE = 'uniform'
if '--eye' in ARGS:
    EYE = ARGS[ARGS.index('--eye') + 1]
    del ARGS[ARGS.index('--eye'):ARGS.index('--eye') + 2]
DAYS = float(ARGS[0]) if ARGS else 10.0
SEEDS = [int(x) for x in ARGS[1:]] or [0, 1, 2]
NAMES = ['silent', 'random', 'echo', 'lookup', 'namer', 'oracle'] + (['looker'] if EYE == 'fovea' else [])
RES_DIR = 'results' if EYE == 'uniform' else 'results_' + EYE
OUT_FILE = 'refs.out' if EYE == 'uniform' else f'refs_{EYE}.out'
os.makedirs(os.path.join(HERE, RES_DIR), exist_ok=True)
os.makedirs(os.path.join(HERE, 'checks'), exist_ok=True)
LINES = []


def out(s):
    print(s, flush=True)
    LINES.append(s)


def run(nm, n, seed, days):
    w = World2(n_agents=n, seed=seed, eye_mode=EYE)
    ticks = int(days * (w.p['day_ticks'] + w.p['night_ticks']))
    t0 = time.time()
    log = w.run(REFERENCES[nm](n), ticks=ticks)
    dt = time.time() - t0
    res = evaluate(log)
    days = max(res['days'], 1)
    says = [r for r in log if r.get('type') == 'say']
    mons = [r for r in log if r.get('type') == 'mon']
    aud = float(np.mean(mons[-1]['cg_audible_ticks'])) if mons else 0.0
    res['input'] = dict(cg_lines_per_day=len(says) / days,
                        lines_to_agent_per_day=sum(1 for r in says if r['to'] is not None) / n / days,
                        audible_cg_speech_ticks_per_agent_day=aud / days,
                        audible_share_of_daytime=aud / days / w.p['day_ticks'])
    res['secs'] = round(dt, 1)
    res['ticks_per_s'] = round(ticks / dt)
    res['diag'] = dict(w.diag)
    json.dump(res, open(os.path.join(HERE, RES_DIR, f'{nm}_n{n}_s{seed}.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, default=str)
    return res


out(f'# world2 reference runs (eye_mode={EYE!r}): {DAYS} days per run, seeds {SEEDS}, 4 agents (and 16 agents for seed '
    f'{SEEDS[0]})')
allres = {}
for seed in SEEDS:
    for nm in NAMES:
        r = run(nm, 4, seed, DAYS)
        allres[(nm, 4, seed)] = r
        out(f'{nm:7s} n=4 seed {seed}: {r["ticks_per_s"]:>7,} ticks/s | ' + headline(r))
for nm in NAMES:
    r = run(nm, 16, SEEDS[0], DAYS)
    allres[(nm, 16, SEEDS[0])] = r
    out(f'{nm:7s} n=16 seed {SEEDS[0]}: {r["ticks_per_s"]:>7,} ticks/s | ' + headline(r))


def f(x):
    return '-' if x is None else f'{x:.2f}'


def ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return '-'
    return f'{np.mean(v):.2f}' + (f'±{np.std(v):.2f}' if len(v) > 1 else '')


out('\n## Table (4 agents, mean ± sd over seeds)\n')
out('| agent | free acc | scene (new) | held-out | memory | social | requests | fetch | give | come | appropriate initiatives /agent/day | vocabulary /agent | a2a turns /day | talk over partner (ticks/day) |')
out('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for nm in NAMES:
    rs = [allres[(nm, 4, s)] for s in SEEDS]
    days = [max(r['days'], 1) for r in rs]
    out(f"| {nm} | {ms([r['tasks']['ALL'].get('acc') for r in rs])} | {ms([r['groups']['scene']['acc'] for r in rs])} | "
        f"{ms([r['groups']['held_out']['acc'] for r in rs])} | {ms([r['groups']['memory']['acc'] for r in rs])} | "
        f"{ms([r['groups']['social']['acc'] for r in rs])} | {ms([r['comprehension']['ALL']['success'] for r in rs])} | "
        f"{ms([r['comprehension']['fetch']['success'] for r in rs])} | {ms([r['comprehension']['give']['success'] for r in rs])} | "
        f"{ms([r['comprehension']['come']['success'] for r in rs])} | "
        f"{ms([r['initiative']['appropriate'] / 4 / d for r, d in zip(rs, days)])} | "
        f"{ms([r['vocabulary']['final_mean'] for r in rs])} | {ms([r['a2a']['turns'] / d for r, d in zip(rs, days)])} | "
        f"{ms([r['overlap']['partner_ticks'] / d for r, d in zip(rs, days)])} |")
r0 = [allres[(nm, 4, SEEDS[0])] for nm in NAMES]
out(f"\nScored items per run (seed {SEEDS[0]}, 4 agents, {DAYS} days): " + ', '.join(
    f"{nm}: free {r['n_free']}, scene {r['groups']['scene']['n']}, held-out {r['groups']['held_out']['n']}, memory "
    f"{r['groups']['memory']['n']}, requests {r['comprehension']['ALL']['n']}" for nm, r in zip(NAMES, r0)))
out('\nInput to the agents (seed %d, mean over agents): ' % SEEDS[0] + ', '.join(
    f"{nm} n={n}: {allres[(nm, n, SEEDS[0])]['input']['lines_to_agent_per_day']:.0f} caregiver lines addressed per agent "
    f"per day, caregiver speech audible {100 * allres[(nm, n, SEEDS[0])]['input']['audible_share_of_daytime']:.0f} % of "
    f"daytime" for nm in ('silent', 'oracle') for n in (4, 16)))
if EYE == 'fovea':
    out('\n## Active vision (fovea mode, 4 agents, mean over seeds): pointed questions whose pointed cell was in the fovea\n')
    out('| agent | pointed questions / run | in the fovea at the start | reached before the window closed | of those not '
        'at the start | median ticks to reach |')
    out('|---|---|---|---|---|---|')
    for nm in NAMES:
        av = [allres[(nm, 4, s)].get('active_vision') or {} for s in SEEDS]
        out(f"| {nm} | {np.mean([a.get('n', 0) for a in av]):.0f} | {ms([a.get('in_fovea_at_start') for a in av])} | "
            f"{ms([a.get('reached') for a in av])} | {ms([a.get('reached_if_not_at_start') for a in av])} | "
            f"{ms([a.get('median_wait') for a in av])} |")
o = allres[('oracle', 4, SEEDS[0])]
out('\nOracle per task (seed %d): ' % SEEDS[0] + ', '.join(f"{t} {f(v.get('acc'))} (n={v['n']})" for t, v in o['tasks'].items()))
out('\n## 16 agents (seed %d)\n' % SEEDS[0])
out('| agent | ticks/s | free acc | scene | held-out | memory | requests | questions per agent per day | initiative | a2a |')
out('|---|---|---|---|---|---|---|---|---|---|')
for nm in NAMES:
    r = allres[(nm, 16, SEEDS[0])]
    out(f"| {nm} | {r['ticks_per_s']:,} | {f(r['tasks']['ALL'].get('acc'))} | {f(r['groups']['scene']['acc'])} | "
        f"{f(r['groups']['held_out']['acc'])} | {f(r['groups']['memory']['acc'])} | {f(r['comprehension']['ALL']['success'])} | "
        f"{r['n_free'] / 16 / max(r['days'], 1):.1f} | {r['initiative']['n']} | {r['a2a']['turns']} |")
# speed table: trivial agents, N = 1, 4, 16, 64
out('\n## Speed (trivial agents, 5 days = 150,000 ticks each, best of 2)\n')
out('| agents | ticks/s | us/tick |')
out('|---|---|---|')
from world2 import Silent  # noqa: E402
for n in (1, 4, 16, 64):
    best = 0
    for _ in range(2):
        w = World2(n_agents=n, seed=0, eye_mode=EYE)
        t0 = time.time()
        w.run(Silent(n), ticks=150000)
        best = max(best, 150000 / (time.time() - t0))
    out(f'| {n} | {best:,.0f} | {1e6 / best:.1f} |')
with open(os.path.join(HERE, 'checks', OUT_FILE), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(LINES) + '\n')
out('REFS DONE')
