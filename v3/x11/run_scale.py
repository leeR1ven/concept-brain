"""X11 scaling test (overhearing only): one life per seed overhears LCCC dialogues continuously with learning on; at
each checkpoint (number of dialogues heard) a frozen copy is scored on a FIXED held-out set (probes11), next to
count models trained on the same dialogues; activity health per 10-dialogue window. Chains jobs of <= 90 min through a
life checkpoint in ~/ckpt/<run>_seed<seed>/ (resumes automatically; deleted after the last checkpoint).
usage: python3 run_scale.py --seed 0 --run x11_s1 [--s 1] [--checkpoints 1000,3000,10000,30000,100000]
       [--max_minutes 75] [--n_heldout 200] [--out out_scale] [--set key=value ...]"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from common import DEV, log, save_json
from life11 import Life11, load_bases, x11_config
from probes11 import Activity, codes, heldout_eval, ngram_eval
from world11 import Dialogues

HERE = os.path.dirname(os.path.abspath(__file__))


def val(v):
    if v in ('True', 'False'):
        return v == 'True'
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--s', type=int, default=1)
    ap.add_argument('--run', default='x11_s1')
    ap.add_argument('--checkpoints', default='1000,3000,10000,30000,100000')
    ap.add_argument('--max_minutes', type=float, default=75.0)
    ap.add_argument('--n_heldout', type=int, default=200)
    ap.add_argument('--out', default='out_scale')
    ap.add_argument('--keep_ckpt', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='newborn eval + checkpoint round trip only, learning off')
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    t0 = time.time()
    over = {k: val(v) for k, v in (x.split('=') for x in a.set)}
    c = x11_config(a.s, **over)
    bases = load_bases(HERE)
    K = len(bases) * 5
    train = Dialogues.load(os.path.join(HERE, 'data', 'train.npz'))
    birth = Dialogues.load(os.path.join(HERE, 'data', 'birth.npz'))
    test = Dialogues.load(os.path.join(HERE, 'data', 'heldout.npz'))
    uni = np.bincount(train.syl.astype(np.int64), minlength=K).astype(np.float64)
    sids = np.nonzero((uni > 0) | (np.bincount(test.syl.astype(np.int64), minlength=K) > 0))[0]
    ck = os.path.expanduser(f'~/ckpt/{a.run}_seed{a.seed}')
    os.makedirs(ck, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    name = f'{a.run}_seed{a.seed}'
    life_p, res_p, codes_p = (os.path.join(ck, f) for f in ('life.pt', 'res.json', 'codes.pt'))
    cps = [int(x) for x in a.checkpoints.split(',')]
    if os.path.exists(life_p):
        life, extra = Life11.load(life_p, c, a.seed, train, bases)
        res = json.load(open(res_p))
        cd = torch.load(codes_p, map_location=DEV, weights_only=False)
        log(f'resumed at {life.world.n_ex} dialogues, brain tick {life.brain.t}')
        res['jobs'].append(dict(start_dialogues=life.world.n_ex))
    else:
        life = Life11(c, a.seed, train, birth, bases)
        b = life.brain
        cd = codes(life, sids, uni, n_ctx=2)
        torch.save(cd, codes_p)
        res = dict(run=name, config={k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                   n1=b.n1, sound_units=b.sound.n, pfc_n=b.pfc.n, hip_n=b.hip.nh, a0_bits=life.coch.n,
                   n_candidates=len(sids), code_size3=cd['mean_code_size3'], calib=b.calib,
                   chain_report=b.chain_report, birth_ticks=life.birth_ticks, checkpoints={}, windows=[],
                   jobs=[dict(start_dialogues=0)])
        r, occ = heldout_eval(life, test, a.n_heldout, cd)
        r['ngram'] = None                                   # no dialogues heard yet
        r['minutes'] = (time.time() - t0) / 60
        res['checkpoints']['0'] = r
        log(name, 'newborn', 'pfc', r['pfc'], 'hip', r.get('hip'), 'oracle', r['heard_oracle'])
        if a.smoke:                                         # code check only: learning stays OFF
            r['ngram_smoke'] = ngram_eval(train, 50, occ, cd, K)
            log('ngram on 50 dialogues', r['ngram_smoke'])
            life.save(life_p)
            lf2, _ = Life11.load(life_p, c, a.seed, train, bases)
            same = [all(bool((t1.w == t2.w).all()) for t1, t2 in zip(life.brain.plastic(), lf2.brain.plastic()))]
            for _ in range(300):
                life.tick()
                lf2.tick()
                same.append(all(bool((x == y).all()) for x, y in zip(life.brain.sound.state, lf2.brain.sound.state))
                            and bool((life.brain.pfc.L4.on == lf2.brain.pfc.L4.on).all())
                            and bool((life.brain.hip.Hs.on == lf2.brain.hip.Hs.on).all()))
            r['resume_identical_300_ticks'] = bool(all(same))
            r['resume_tables_equal_at_load'] = same[0]
            r['resume_first_divergent_tick'] = next((i for i, x in enumerate(same) if not x), None)
            log('tables equal at load', same[0], 'first divergent tick', r['resume_first_divergent_tick'])
            log('resume identical over 300 ticks (learning off):', r['resume_identical_300_ticks'],
                'ckpt MB', os.path.getsize(life_p) / 1e6)
            save_json(os.path.join(a.out, name + '_smoke.json'), res)
            shutil.rmtree(ck, ignore_errors=True)
            return
    b = life.brain
    b.learning = True
    act = Activity(b)
    act.d0 = life.world.n_ex
    limit = a.max_minutes * 60
    t_job = time.time()
    ticks0 = b.t
    finished = True
    for cp in cps:
        done = life.world.n_ex
        if cp <= done:
            continue
        life.run_dialogues(cp - done, record=act, until=lambda: time.time() - t0 > limit)
        act.flush(life.world.n_ex)
        res['windows'] += act.wins
        act.wins = []
        if life.world.n_ex < cp:
            finished = False
            break
        b.learning = False
        r, occ = heldout_eval(life, test, a.n_heldout, cd)
        r['ngram'] = ngram_eval(train, cp, occ, cd, K)
        b.learning = True
        r['minutes'] = (time.time() - t0) / 60
        r['ticks'] = b.t
        r['child_muscle_ticks'] = life.world.child_ticks
        r['activity'] = dict(windows_since_last=[w for w in res['windows'] if w['dialogue'] > max(
            [int(k) for k in res['checkpoints']] + [0]) and w['dialogue'] <= cp])
        ws = r['activity']['windows_since_last']
        r['activity']['share_ok'] = float(np.mean([w['ok'] for w in ws])) if ws else None
        res['checkpoints'][str(cp)] = r
        log(name, cp, 'pfc', r['pfc'], 'late', r['pfc_late'], 'hip', r.get('hip'), 'ngram', {k: r['ngram'][k] for k in (
            'unigram', 'bigram', 'trigram', 'bigram_lag2', 'trigram_lag2', 'interp_via_codes')},
            'J', round(r['tick_J_pfc'], 4), 'response brain', r['response'], 'response ngram', r['ngram']['response'])
        save_json(res_p, res)
        save_json(os.path.join(a.out, name + '.json'), res)
        if cp != cps[-1]:
            life.save(life_p)
    dt = time.time() - t_job
    res['jobs'][-1].update(end_dialogues=life.world.n_ex, minutes=(time.time() - t0) / 60,
                           ms_per_tick=1000 * dt / max(1, b.t - ticks0), gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    save_json(res_p, res)
    save_json(os.path.join(a.out, name + '.json'), res)
    if not finished:
        life.save(life_p)
        log(name, f'time limit at {life.world.n_ex} dialogues: checkpoint saved, resume with the same command')
        with open(os.path.join(a.out, name + '.RESUME'), 'w') as f:
            f.write(str(life.world.n_ex))
    else:
        log(name, 'all checkpoints done')
        if not a.keep_ckpt:
            shutil.rmtree(ck, ignore_errors=True)


if __name__ == '__main__':
    main()
