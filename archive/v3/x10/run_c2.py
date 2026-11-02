"""Stage (c), designer 11-02: a long OVERHEARING life ("可以先不管说的对不对，想让模型听大量别人对话……看看会怎么样").
Base B + P1 + G1P inner + MC pools from returns, learning on, no reward (r = 0); the child is free to speak or stay
silent and its correctness is ignored. The code is frozen for the whole run. Long runs chain jobs through a brain
checkpoint (--ckpt; the same life is rebuilt from the seed, the saved brain loaded, the world fast-forwarded).
Recorded all along: activity windows every 100 dialogues; the child's spontaneous speech per segment (what, how
clear, when). At each checkpoint (frozen copies, observation only): answers to the 32 registered questions incl. the
never-heard pairs (vs the frame baseline, a number only); the learned Q -> A link ratio; the PFC's next-syllable
prediction; one-shot Q -> A by route; what it says when it thinks a syllable (b1 form); table statistics.
usage: python3 run_c2.py --tag _notouch --ckpt ~/ckpt/c2_nt_s0.pt --until 30000 --max_minutes 80 --set ..."""
import argparse
import os
import time

import numpy as np
import torch

from b2lib import say_test
from common import DEV, config, log, save_json
from life import Life
import probes
import probes2

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='')
ap.add_argument('--out', default='out_c2')
ap.add_argument('--ckpt', default=None)
ap.add_argument('--until', type=int, default=30000)
ap.add_argument('--checkpoints', default='1000,3000,10000,30000')
ap.add_argument('--max_minutes', type=float, default=80.0)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--set', nargs='*', default=[])
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
c = config(1, birth_dialogues=30 if a.quick else 200, **over)
ck = os.path.expanduser(a.ckpt) if a.ckpt else None
if ck and os.path.exists(ck):
    with open(ck, 'rb') as f:
        st = torch.load(f, map_location=DEV, weights_only=False)
    life = Life(c, a.seed, wire=False)
    life.brain, life.codes, life.registry = st['brain'], st['codes'], st['registry']
    life.l4_sel = st.get('l4_sel')
    while life.env.w.n_ex < st['n_ex']:                                # the world continues (child mute while skipping)
        life.env.step(np.zeros(24, bool))
    res, done = st['res'], st['n_done']
    log('resumed at dialogue', done, 'world dialogue', life.env.w.n_ex)
else:
    life = Life(c, a.seed)                                             # G1P (+ MC) wired
    res, done = dict(config=over, tag=a.tag, registry=life.registry, calib=life.brain.calib, windows=[], speech=[],
                     checkpoints={}, rate=[]), 0
b = life.brain
w = life.env.w
names = b.count_names()
ix = {n: i for i, n in enumerate(names)}
cps = [int(x) for x in a.checkpoints.split(',')] if not a.quick else [20, 40]
until = a.until if not a.quick else 40
seg = 50 if not a.quick else 10


def window(rows, end):
    C = torch.stack(rows).float()
    m = C.mean(0).cpu().numpy()
    mx = C.max(0).values.cpu().numpy()
    mn = C.min(0).values.cpu().numpy()
    r = dict(dialogue=end, L2_L1=float(m[ix['L2']] / max(m[ix['L1']], 1)), L3_L2=float(m[ix['L3']] / max(m[ix['L2']], 1)),
             L4_L3=float(m[ix['L4']] / max(m[ix['L3']], 1)), L4=float(m[ix['L4']]), L4_frac_max=float(mx[ix['L4']] / b.pfc.n),
             Hs=float(m[ix['Hs']]) if 'Hs' in ix else None, Hs_min=float(mn[ix['Hs']]) if 'Hs' in ix else None,
             A4=float(m[ix['A4']]), force=float(m[ix['force']]), stop=float(m[ix['stop']]))
    if 'T1' in ix:
        r['T1'] = float(m[ix['T1']])
    return r


def save_ck(done):
    save_json(os.path.join(a.out, f'c2{a.tag}_seed{a.seed}.json'), res)
    if ck:
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        with open(ck + '.tmp', 'wb') as f:
            torch.save(dict(brain=b, codes=life.codes, registry=life.registry, l4_sel=getattr(life, 'l4_sel', None),
                            n_ex=w.n_ex, res=res, n_done=done), f, pickle_protocol=4)
        os.replace(ck + '.tmp', ck)


def observe(cp):
    r = dict(dialogues=cp, minutes=(time.time() - t0) / 60, ticks=b.t)
    r['qa'] = probes.qa_probe(life, reps=1 if a.quick else 2)                         # answers incl. never-heard pairs
    r['qa_links'] = probes2.qa_link_stats(life)
    r['next_syllable'] = probes2.next_syllable(life)
    r['say_when_thinking'] = say_test(life, list(range(34)))
    snap = b.snapshot_full()                                                          # one-shot on a copy
    r['one_shot'] = probes2.one_shot(life, Ds=(50,) if a.quick else (200, 1000))
    b.restore_full(snap)
    r['tables'] = probes.table_stats(life)
    log(a.tag, 'dialogue', cp, 'QA seen exact', r['qa']['seen']['exact'], 'held exact', r['qa']['held']['exact'],
        'baseline', round(r['qa']['frame_baseline'], 2), 'silent', r['qa']['seen']['silent'], 'garble',
        r['qa']['seen']['garble'], 'Q->A links', r['qa_links'], 'next syl', r['next_syllable'], 'say', r['say_when_thinking'],
        'one-shot', r['one_shot'])
    return r


b.learning = True
rows, last_w, seg_start = [], done, done
t_seg = time.time()
while done < until and (time.time() - t0) / 60 < a.max_minutes:
    target = done + seg
    life.run_dialogues(seg, record=lambda lf: rows.append(lf.brain.counts()))
    done = target
    if done - last_w >= (100 if not a.quick else 10):
        res['windows'].append(window(rows, done))
        rows, last_w = [], done
    if done % (1000 if not a.quick else 20) == 0 or done in cps:
        n_ex = w.n_ex
        sp = probes2.speech_summary(w, n_ex - (done - seg_start), n_ex)
        sp['dialogues'] = done
        res['speech'].append(sp)
        res['rate'].append(dict(dialogues=done, per_hour=(done - seg_start) / max(1e-6, (time.time() - t_seg) / 3600)))
        log(a.tag, 'dialogue', done, 'speech', {k: v for k, v in sp.items() if k != 'examples'}, 'eg', sp['examples'][-5:],
            'window', res['windows'][-1] if res['windows'] else None, 'rate/h', round(res['rate'][-1]['per_hour']))
        seg_start, t_seg = done, time.time()
    if done in cps:
        res['checkpoints'][done] = observe(done)
        b.learning = True
        save_ck(done)
save_ck(done)
log(a.tag, 'stopped at dialogue', done)
