import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import make_brain, run_until, GEN0, probe6, summ, spec
from world6 import OverhearWorld, overhear_mech
from wire import wire_speech
from run import parse
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
seed = kv.pop('seed', 0); n = kv.pop('n', 300)
t0 = time.time()
ph, b = make_brain(seed, dict(GEN0, **kv))
wire_speech(b, ph)
ow = OverhearWorld(ph, seed)
rec = dict(n3=[], np=[])
out = []
for m in (n // 3, n):
    run_until(b, ow, m, rec)
    n3 = np.array(rec['n3'][-1500:])
    allp = [(f, x) for f in range(4) for x in range(8)]
    pr = probe6(b, ph, ow.props, ow.held, allp, seed)
    sp = spec([r for r in pr if not r['held']])
    s = summ([r for r in pr if not r['held']])
    out.append(f"O{m}: n3 mean {n3.mean():.0f} max {n3.max()} psat {(n3 >= 360).mean():.2f} pempty {(n3 == 0).mean():.2f} | acc {s['acc']} first {s['first']} sil {s['silent']} | spec {sp['own_max']}/{sp['other_max']} hub {sp['hub_syl']} {sp['hub_share']} | utts/dlg {overhear_mech(ow, 0, m)['child_utts_per_dlg']}")
print(kv, ' || '.join(out), f'{time.time() - t0:.0f}s', flush=True)
