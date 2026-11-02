"""X10 sizing v3: units, synapses and GPU table memory for the three PFC-size presets (sizing/band_sizes.py,
s = 1: small 1,804, medium 4,054, large 8,150 sound copies; linear in s, as measured from s = 1 to s = 4)."""
import math
SND = {'small': 1804, 'medium': 4054, 'large': 8150}
print('preset  s | N1 (PFC L1) | PFC units | sound | hip (copies+Hs) | total units | PFC learned cand/unit | '
      'synapses | GPU tables GB (PFC learned + hip)')
for z, n_s1 in SND.items():
    for s in (1, 4, 8, 16):
        n1 = n_s1 * s + 32 * s
        A0 = 1080 * s; sound = A0 + 4 * 2 * A0
        cand = math.pi * 0.15 ** 2 * n1
        NH = 5000 * s
        pfc_l = 2 * n1 * cand
        hip = NH * 24 + NH * 700 * s
        syn = 4 * 2 * A0 * 6 + 3 * n1 * 24 + n1 * 50 + pfc_l + hip + 4 * n1
        gb = (pfc_l + NH * 700 * s) * 8 / 1e9
        print(f'{z:6s} {s:2d} | {n1:,} | {4 * n1:,} | {sound:,} | {n1 + NH:,} | {sound + 40 * s + 4 * n1 + n1 + NH:,} | '
              f'{cand:.0f} | {syn / 1e6:.0f}M | {pfc_l * 8 / 1e9:.2f} + {NH * 700 * s * 8 / 1e9:.2f} = {gb:.2f}')
