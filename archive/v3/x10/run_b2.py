"""Stage (b2): self-correction training (designer 11-02, ARCH 2.7 / 2.8). Mechanisms already in ARCH only:
feel = own voice (+ touch arm); learn to stop / correct = the reward tool [R6] at the caregiver's reaction.
Practice trial: a caregiver (C1 / C2 alternating) says syllable x (3 ticks), then waits W ticks while the child may
speak through its gen-0 tendency (G1P). The caregiver's innate-value reaction, one per trial, at the tick she hears it:
  sad (r = -1) at the first garbled or wrong syllable (also after a clear x);
  happy (r = +1) when the child said clear x and then fell silent;
  nothing otherwise (silence, a mumble).
All 34 syllables are practised (designer 11-02: no generalisation test at this scale). Every 26 trials, 3 overheard
dialogues of ordinary life (learning on, no reward; the child's utterances there = babble). Every --every trials a
checkpoint (frozen): the say test (forced inner x as in b1, unmuted: said exactly [x], garble, tail ticks after the
first clear x, stop right after x); practice outcomes; babble per dialogue; b1-b7 by speech_basics; and the reaction
withdrawn: --withdrawn practice trials with r = 0 on a copy, then the say test (does it still stop by itself). The
life continues from the state before the copy. Long runs chain jobs through a checkpoint file (--ckpt).
usage: python3 run_b2.py --out out_b2 --ckpt ~/ckpt/b2_x --until 26000 --max_minutes 75 [--set key=value ...]"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from common import DEV, config, log, save_json
from life import Life
import probes
from probes import Frozen, code_mask
from b2lib import practice_trial as _pt, say_test as _st
from world_adapter import C1, C2, NSYL, SYL

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out_b2')
ap.add_argument('--until', type=int, default=2600, help='stop after this many practice trials in total')
ap.add_argument('--every', type=int, default=2600)
ap.add_argument('--withdrawn', type=int, default=260)
ap.add_argument('--ckpt', default=None, help='life checkpoint file (resume if it exists; saved at the end)')
ap.add_argument('--max_minutes', type=float, default=75.0)
ap.add_argument('--W', type=int, default=16)
ap.add_argument('--tag', default='')
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
ck = os.path.expanduser(a.ckpt) if a.ckpt else None
if ck and os.path.exists(ck):                                          # resume a chained run: the same life rebuilt
    with open(ck, 'rb') as f:                                          # (same seed, same world), then the saved brain
        st = torch.load(f, map_location=DEV, weights_only=False)
    c = config(1, reward=True, birth_dialogues=30 if a.quick else 200, **over)
    life = Life(c, a.seed, wire=False)
    life.brain, life.codes, life.registry = st['brain'], st['codes'], st['registry']
    life.l4_sel = st.get('l4_sel')
    while life.env.w.n_ex < st['n_ex']:                                # the world continues where it was (child mute)
        life.env.step(np.zeros(24, bool))
    rng, k, res, outcomes = st['rng'], st['k'], st['res'], st['outcomes']
    log('resumed at trial', k, 'from', ck, 'world dialogue', life.env.w.n_ex)
else:
    c = config(1, reward=True, birth_dialogues=30 if a.quick else 200, **over)
    life = Life(c, a.seed)                                             # G1P (+ MC) wired
    rng = np.random.default_rng(a.seed + 7100)
    k, outcomes = 0, []
    res = dict(config=over, tag=a.tag, registry=life.registry, checkpoints=[], rate=[])
b, ph, env = life.brain, life.ph, life.probe_env
train = list(range(NSYL))
practice_trial = lambda x, voice, react=True: _pt(life, rng, x, voice, react, a.W)
say_test = lambda syls: _st(life, syls)


def checkpoint(k, outcomes, babble):
    r = dict(trials=k, minutes=(time.time() - t0) / 60, ticks=b.t, say=say_test(train))
    if outcomes:
        n = len(outcomes)
        kinds = sorted(set(o for o, _ in outcomes))
        r['practice_outcomes'] = {kd: sum(o == kd for o, _ in outcomes) / n for kd in kinds}
        r['reactions'] = {str(rc): sum(x == rc for _, x in outcomes) / n for rc in ('happy', 'sad', None)}
    r['babble_utts_per_dialogue'] = babble
    snap = b.snapshot_full()                                   # on a copy: b1-b7, then the reaction withdrawn
    rng_state = rng.bit_generator.state
    sb = probes.speech_basics(life, n_overhear=20 if a.quick else 100)
    r['speech_basics'] = {kk: v for kk, v in sb.items()}
    b.restore_full(snap)
    b.learning = True
    wo = [practice_trial(int(x), C1 if i % 2 == 0 else C2, react=False)
          for i, x in enumerate(rng.permutation(train)[:a.withdrawn] if a.withdrawn <= NSYL else
                                np.resize(rng.permutation(train), a.withdrawn))]
    r['withdrawn'] = dict(trials=len(wo), outcomes={kd: sum(o == kd for o, _ in wo) / len(wo) for kd in sorted(set(o for o, _ in wo))},
                          say=say_test(train))
    b.restore_full(snap)
    rng.bit_generator.state = rng_state
    log(a.tag, 'trials', k, 'say', r['say'], 'babble', babble, 'outcomes', r.get('practice_outcomes'),
        'withdrawn', r['withdrawn'], 'b4/b7', sb.get('b4_echo_per_line'), sb.get('b7_utterances_per_dialogue'))
    return r


if a.quick:
    a.every, a.withdrawn = 26, 26
w = life.env.w
utt0, ex0 = len(w.utts), w.n_ex


def babble_rate():
    global utt0, ex0
    v = (len(w.utts) - utt0) / max(1, w.n_ex - ex0)
    utt0, ex0 = len(w.utts), w.n_ex
    return v


def save():
    save_json(os.path.join(a.out, f'b2{a.tag}_seed{a.seed}.json'), res)
    if ck:
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        with open(ck + '.tmp', 'wb') as f:
            torch.save(dict(brain=life.brain, codes=life.codes, registry=life.registry, l4_sel=getattr(life, 'l4_sel', None),
                            n_ex=life.env.w.n_ex, rng=rng, k=k, res=res, outcomes=outcomes), f, pickle_protocol=4)
        os.replace(ck + '.tmp', ck)


if k == 0:
    res['checkpoints'].append(checkpoint(0, [], None))
b.learning = True
k_start, t_start = k, time.time()
order = []
while k < a.until and (time.time() - t0) / 60 < a.max_minutes:
    if not order:
        order = list(rng.permutation(train))
    x = int(order.pop())
    outcomes.append(practice_trial(x, C1 if k % 2 == 0 else C2))
    k += 1
    if k % 26 == 0:
        life.run_dialogues(3)                                  # ordinary life in between (learning on, r = 0)
    if k % a.every == 0:
        res['rate'].append(dict(trials=k, trials_per_hour=(k - k_start) / max(1e-6, (time.time() - t_start) / 3600)))
        res['checkpoints'].append(checkpoint(k, outcomes, babble_rate()))
        outcomes = []
        b.learning = True
        utt0, ex0 = len(w.utts), w.n_ex                         # the copy's dialogues do not count as babble
        save()
res['rate'].append(dict(trials=k, trials_per_hour=(k - k_start) / max(1e-6, (time.time() - t_start) / 3600), end=True))
log('trials', k, 'rate per hour', res['rate'][-1]['trials_per_hour'])
save()
