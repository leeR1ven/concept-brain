"""Stage (c): systematic training (ARCH 2.11): overhear with learning on; frozen probes at checkpoints [PLAN 6c].
usage: python3 run_c.py --s 1 --seed 0 --out out_c [--checkpoints 300,1000,3000,10000] [--max_minutes 85]
       [--set key=value ...] (arms: hip=False, band=..., rho=4, post=any)"""
import argparse
import os
import time

from common import config, log, save_json
from life import Life
import probes
import probes2
import scaffold
from wiring import measure_codes, select_codes

ap = argparse.ArgumentParser()
ap.add_argument('--s', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out_c')
ap.add_argument('--checkpoints', default='300,1000,3000,10000')
ap.add_argument('--max_minutes', type=float, default=85.0)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--speech', default='g1p', help='g1p (hand-wired in the PFC) | F1 | F3 (learned: scaffold first)')
ap.add_argument('--oneshot_at', default='1000,10000')
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)


def val(v):
    if v in ('True', 'False'):
        return v == 'True'
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


over = {k: val(v) for k, v in (x.split('=') for x in a.set)}
if 'band_hi' in over:
    over['band'] = (0.002, over.pop('band_hi'))
c = config(a.s, birth_dialogues=30 if a.quick else 200, **over)
if a.speech == 'g1p':
    life = Life(c, a.seed)                                               # G1P (+ MC if mc=True)
    res_sc = None
else:
    life = Life(c, a.seed, wire=False)
    life.wire_mc()
    life.codes = select_codes(measure_codes(life.brain, life.coch))
    res_sc = scaffold.run(life, a.speech, per_syl=6 if a.quick else 30)  # gen-0 forced regular speaking
    log('scaffold', {k: v for k, v in res_sc.items() if k not in ('windows', 'budget')})
name = f's{a.s}_seed{a.seed}_{a.speech}' + ''.join(f'_{k}{v}' for k, v in sorted(over.items()))
res = dict(config=over, speech=a.speech, registry=life.registry, calib=life.brain.calib, scaffold=res_sc,
           checkpoints={})
life.brain.learning = True
done = 0
for cp in [int(x) for x in a.checkpoints.split(',')]:
    if (time.time() - t0) / 60 > a.max_minutes:
        log('time limit: stopping before checkpoint', cp)
        break
    life.run_dialogues(cp - done)
    done = cp
    r = dict(minutes=(time.time() - t0) / 60, ticks=life.brain.t)
    r['qa'] = probes.qa_probe(life, reps=1 if a.quick else 2)                         # answers (primary)
    r['codes'] = probes.newborn_codes(life, n_ctx=2, n_noise=1, lags=12)              # d1, d2 (a3: number only)
    r['tables'] = probes.table_stats(life)                                            # d6, d10
    r['qa_links'] = probes2.qa_link_stats(life)                                       # learned Q -> A link ratio
    r['returns'] = probes.return_share(life, 5 if a.quick else 30)                    # d8 (learning on)
    if str(cp) in a.oneshot_at.split(','):                                           # memory: one-shot by route
        r['one_shot'] = probes2.one_shot(life, Ds=(50,) if a.quick else (200, 1000))
    if cp == int(a.checkpoints.split(',')[-1]):
        r['d7'] = probes.lateral_sign(life, 200 if a.quick else 1000)                 # d7 (learning on)
    with probes.Frozen(life):                                                        # d9: G1 retention (b1)
        snap = life.brain.snapshot()
        said = []
        for s in range(34):
            life.brain.restore(snap)
            u, _ = probes.forced_say(life, [s])
            said.append(u == [probes.SYL[s]])
        r['d9_b1'] = sum(said) / 34
    res['checkpoints'][cp] = r
    save_json(os.path.join(a.out, name + '.json'), res)
    log(name, cp, 'seen exact', r['qa']['seen']['exact'], 'minority', r['qa']['minority_exact'], 'baseline',
        round(r['qa']['frame_baseline'], 2), 'silent', r['qa']['seen']['silent'], 'garble', r['qa']['seen']['garble'],
        'pass', r['qa']['pass'], 'Q->A links', r['qa_links'], 'one-shot', r.get('one_shot'))
