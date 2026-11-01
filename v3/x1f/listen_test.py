"""main arm G0 with / without the innate listening signal: overhear N dialogues (learning on), then the
path metrics (phase-1 measure: same vs cross, question / frame decoding by lag) and probes.
  python3 -I listen_test.py seed=0 n=1000 g_listen=0.8"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common6, core7
common6.Brain = core7.Brain
from common6 import make_brain, run_until, GEN0, probe6, summ, spec
from world6 import OverhearWorld, overhear_mech
from wire import wire_speech
from run import parse
import phase1
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
seed = kv.pop('seed', 0); n = kv.pop('n', 1000)
t0 = time.time()
bk = dict(GEN0, **kv)
ph, b = make_brain(seed, bk)
wire_speech(b, ph)
ow = OverhearWorld(ph, seed)
out = dict(cfg=kv, seed=seed)
for tag, m in (('birth', 0), ('O', n)):
    if m:
        run_until(b, ow, m)
    pth = phase1.measure(bk, seed=seed, n_ctx=4, brain=(ph, ow, b), verbose=False, n_dmg_ctx=1)
    allp = [(f, x) for f in range(4) for x in range(8)]
    pr = probe6(b, ph, ow.props, ow.held, allp, seed)
    s = summ([r for r in pr if not r['held']]); sp = spec([r for r in pr if not r['held']])
    out[tag] = dict(same=pth['same'], cross=pth['cross'], cross_same_frame=pth['cross_same_frame'], dec_q=pth['dec_q'],
                    dec_frame=pth['dec_frame'], dec_fill_given_frame=pth['dec_fill_given_frame'], n3_lag=pth['n3_lag'],
                    within_J1=pth['within_J1'], probe=s, spec=sp, mech=overhear_mech(ow, 0, max(m, 1)) if m else None)
    L = (0, 3, 6, 12)
    print(kv, tag, m, '| same', [pth['same'][k] for k in L], 'cross', [pth['cross'][k] for k in L], 'cross_sf', [pth['cross_same_frame'][k] for k in L],
          '| decQ', [pth['dec_q'][k] for k in L], 'decF', [pth['dec_frame'][k] for k in L], 'fill|F', [pth['dec_fill_given_frame'][k] for k in L],
          '| acc', s['acc'], 'sil', s['silent'], '| spec', sp['own_max'], sp['other_max'], sp['hub_syl'], f'{time.time() - t0:.0f}s', flush=True)
os.makedirs('out7', exist_ok=True)
json.dump(out, open(f"out7/listen_g{kv.get('g_listen', 0)}_s{seed}.json", 'w'), indent=1, default=str)
