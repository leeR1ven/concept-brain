"""X10 sizing v3: PFC L1 size from the chain-band knobs (form K6, per-unit thresholds), s = 1.
Re-uses sound_stats2.py's stream and layers (imported by exec), then counts chained units for band settings."""
import sys
import numpy as np
sys.argv = ['x', '5000', '1', 'K6']
src = open('sound_stats2.py').read().split("sil = np.array")[0]
exec(src)
rates = [act.mean(0) for act in layers]
sil = np.array([len(s) == 0 for s in SRC]); spk = lab >= 0
def size(bands, depth):
    n = 0; ls = ss = 0
    for l, (r, act) in enumerate(zip(rates, layers)):
        lo, hi = bands[l] if isinstance(bands, list) else bands
        m = np.ones_like(r, bool) if l >= depth else (r >= lo) & (r <= hi)
        n += m.sum(); c = act[:, m].sum(1); ls += c[sil].mean(); ss += c[spk].mean()
    return int(n), round(ls), round(ss)
print('rate quantiles per layer (10/25/50/75/90 %):')
for l, r in enumerate(rates):
    print(f'  A{l+1}', np.round(np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9]), 3))
for name, bands, depth in [
        ('large: band on A1 only', (0.002, 0.90), 1),
        ('band everywhere 0.002-0.90', (0.002, 0.90), 4),
        ('A1 0.002-0.9; deeper 0.02-0.9', [(0.002, 0.9), (0.02, 0.9), (0.02, 0.9), (0.02, 0.9)], 4),
        ('A1 0.002-0.9; deeper 0.05-0.9', [(0.002, 0.9), (0.05, 0.9), (0.05, 0.9), (0.05, 0.9)], 4),
        ('A1 0.002-0.9; deeper 0.08-0.9', [(0.002, 0.9), (0.08, 0.9), (0.08, 0.9), (0.08, 0.9)], 4),
        ('everywhere 0.05-0.9', (0.05, 0.90), 4),
        ('everywhere 0.08-0.9', (0.08, 0.90), 4),
        ('everywhere 0.10-0.9', (0.10, 0.90), 4),
        ('everywhere 0.12-0.9', (0.12, 0.90), 4),
        ('everywhere 0.15-0.9', (0.15, 0.90), 4)]:
    print(f'{name:40s} N1(sound) {size(bands, depth)}  (chained, lit silence, lit speech)')
print('cuts relative to each layer\'s own target rate r_l = (0.20, 0.12, 0.08, 0.05):')
R_ = [0.20, 0.12, 0.08, 0.05]
for name, fA1, f in [('A1 std, deeper 0.5 r_l', None, 0.5), ('A1 std, deeper 0.75 r_l', None, 0.75),
                     ('A1 std, deeper 0.9 r_l', None, 0.9), ('all 0.75 r_l', 0.75, 0.75), ('all 0.9 r_l', 0.9, 0.9),
                     ('all 0.95 r_l', 0.95, 0.95)]:
    bands = [(0.002, 0.9) if (l == 0 and fA1 is None) else (f if l else fA1) * R_[l] for l in range(4)]
    bands = [b if isinstance(b, tuple) else (b, 0.9) for b in bands]
    per = []
    for l, r in enumerate(rates):
        lo, hi = bands[l]; per.append(int(((r >= lo) & (r <= hi)).sum()))
    print(f'{name:28s} bands {[(round(b[0],3), b[1]) for b in bands]} per layer {per} -> {size(bands, 4)}')
