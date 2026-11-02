import sys, time
X = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..')
sys.path.insert(0, X)
import numpy as np
import core8, common6
common6.Brain, common6.DEF = core8.Brain, core8.DEF
from common6 import make_brain, run_until
from run8 import build_cfg
from wire8 import wire_speech
from world6 import OverhearWorld
N = int(sys.argv[1]); n_dlg = int(sys.argv[2])
bk = build_cfg(dict(N=N))
ph, b = make_brain(0, bk, n_birth=50)
wire_speech(b, ph)
ow = OverhearWorld(ph, 0)
t0 = time.time()
run_until(b, ow, n_dlg)
L = b.L33_true()
d = np.diag(L)
print('ticks', b.t, 'time', round(time.time() - t0, 1), 'L33 pos frac', round(float((L > 1e-3).mean()), 3),
      'diag>1e-3', int((d > 1e-3).sum()), 'of', len(d), 'diag mean', round(float(d.mean()), 4), 'offdiag mean(pos)', round(float(L[L > 1e-3].mean()), 4),
      'diag max', round(float(d.max()), 3), 'L33 max', round(float(L.max()), 3))
# share of each unit's learned out-weight that is on itself
out = np.maximum(L, 0).sum(1)
print('self share of out (median/max)', round(float(np.median(d / np.maximum(out, 1e-9))), 4), round(float((d / np.maximum(out, 1e-9)).max()), 4))
print('L31 pos frac (within mask)', round(float((b.L31 > 1e-3).sum() / b.M31.sum()), 3), 'h3 range', round(float(b.h3.min()), 3), round(float(b.h3.max()), 3))
