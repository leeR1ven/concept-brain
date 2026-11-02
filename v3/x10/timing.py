"""Timing pilot: build a life at scale s (short birth), wire G1, then 2,000 ticks of overhearing with learning on.
Reports build time, ms per tick (synchronised), GPU memory, and sizes. usage: python3 timing.py --s 4 [--ticks 2000]"""
import argparse
import time

import torch

from common import DEV, Prof, config, log, save_json, sync
from life import Life

ap = argparse.ArgumentParser()
ap.add_argument('--s', type=int, default=1)
ap.add_argument('--ticks', type=int, default=2000)
ap.add_argument('--birth', type=int, default=50)
ap.add_argument('--out', default=None)
ap.add_argument('--prof_ticks', type=int, default=300)
ap.add_argument('--g_loop', type=float, default=None)
ap.add_argument('--dense_limit', type=float, default=None)
ap.add_argument('--pfc_size', default='medium')
ap.add_argument('--w44', type=float, default=None)
a = ap.parse_args()
t0 = time.time()
if DEV.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
over = dict(pfc_size=a.pfc_size)
if a.g_loop is not None:
    over['g_loop'] = a.g_loop
if a.w44 is not None:
    over['w44'] = a.w44
if a.dense_limit is not None:
    over['dense_limit'] = a.dense_limit
c = config(a.s, birth_dialogues=a.birth, **over)
life = Life(c, 0)
sync()
t_build = time.time() - t0
b = life.brain
mem_build = torch.cuda.max_memory_allocated() / 1e9 if DEV.type == 'cuda' else 0.0
b.learning = True
for _ in range(50):                                     # warm-up
    life.tick()
sync()
t1 = time.time()
for _ in range(a.ticks):
    life.tick()
sync()
dt = (time.time() - t1) / a.ticks
acc = torch.zeros_like(b.counts().float())
mx = torch.zeros_like(acc)
for _ in range(500):                                    # mean and max activity over 500 more ticks (learning on)
    life.tick()
    v = b.counts().float()
    acc += v
    mx = torch.maximum(mx, v)
counts = dict(mean=(acc / 500).tolist(), max=mx.tolist())
b.prof = Prof()
for _ in range(a.prof_ticks):                            # per-section profile (synchronised, so slower overall)
    life.tick()
prof = {k: 1000 * v / a.prof_ticks for k, v in b.prof.t.items()}
b.prof = None
res = dict(s=a.s, pfc_size=a.pfc_size, g_loop=c['g_loop'], w44=c['w44'], dense_limit=c['dense_limit'], device=str(DEV), ms_per_tick=1000 * dt, ticks=a.ticks, build_s=t_build,
           gpu_peak_GB=torch.cuda.max_memory_allocated() / 1e9 if DEV.type == 'cuda' else 0.0,
           gpu_peak_build_GB=mem_build,
           gpu_reserved_GB=torch.cuda.max_memory_reserved() / 1e9 if DEV.type == 'cuda' else 0.0,
           n1=b.n1, chains=[len(ix) for ix in b.chains.sound_idx], sound=b.sound.n, pfc_layer=b.pfc.n,
           hs=b.hip.nh if b.hip is not None else 0,
           tables={tb.name: list(tb.shape()) for tb in b.tables()},
           counts_after=counts, count_names=b.count_names(), profile_ms_per_tick=prof,
           chain_report=b.chain_report,
           life_10k_dialogues_min=dt * 406000 / 60, registry=life.registry, calib=b.calib)
log(f"s={a.s} {a.pfc_size} g_loop {c['g_loop']} w44 {c['w44']}: {res['ms_per_tick']:.2f} ms/tick, peak {res['gpu_peak_GB']:.2f} GB, build {t_build:.0f} s, "
    f"N1 {b.n1}, 10k dialogues ~ {res['life_10k_dialogues_min']:.0f} min")
log('mean counts', {k: round(v) for k, v in zip(b.count_names(), counts['mean'])})
log('max counts', {k: round(v) for k, v in zip(b.count_names(), counts['max'])})
log('profile ms/tick', {k: round(v, 2) for k, v in prof.items()})
save_json(a.out or f'timing_s{a.s}.json', res)
