"""X1e arms.  Bases are X1d's R1e and RNpm (copied from regimes.py, unchanged).  Everything else is the top layer's
internal links learning by the one rule (core5 rule33='onset'), ARCH two-layer weights on them, the ARCH scale and
the shaping caregiver.  kappa = 0 everywhere (no self-hold); no fatigue / refractoriness / timers / event boundaries.
"""
from regimes import R1e, RNpm

# L3 -> L3 learning, same onset / held form, ARCH two-layer weights (scale-free), one global decay (dec_33)
L33 = dict(eta33=0.02, rule33='onset', nrm33='arch', a33=0.5)

# RNpm: birth lateral column sum is ~2 (w33 0.1 x fan 20); g33 sets how much the (learned) lateral drive weighs
# against the feed-forward drive (s23 = 16).  f3 is re-set so that the threshold level f3 * (16 + g33) stays at
# RNpm's 0.3 * 18 = 5.4 (a minor parameter, not a new part).
def rn(g33, **kw):
    return dict(RNpm, **L33, g33=g33, f3=round(5.4 / (16.0 + g33), 4), **kw)


# R1e: global quantile threshold (shortcut reference); birth lateral column sum is ~8 (w33 0.4 x fan 20)
def r1(g33, **kw):
    return dict(R1e, **L33, g33=g33, **kw)


X4 = dict(n2=1600, n3=1600)                    # ARCH scale x4 of L2 / L3 (L1 = sensory copies, unchanged)

ARMS = {
    # ---- references (X1d bases, L33 frozen at birth values), re-run in this code
    'RNpm': dict(care='format', b=dict(RNpm)),
    'R1e': dict(care='format', b=dict(R1e)),
    # ---- L33 learning on
    'RNpm_L2': dict(care='format', b=rn(2.0)),      # same lateral weight as at birth
    'RNpm_L8': dict(care='format', b=rn(8.0)),      # lateral ~1/3 of the in-weight
    'RNpm_L16': dict(care='format', b=rn(16.0)),    # lateral ~1/2 of the in-weight
    'R1e_L8': dict(care='format', b=r1(8.0)),
    'R1e_L16': dict(care='format', b=r1(16.0)),
    # ---- x4 scale (k for the global arm scaled x4, as X1d queued it)
    'RNpm_x4': dict(care='format', b=dict(RNpm, **X4)),
    'R1e_x4': dict(care='format', b=dict(R1e, **X4, k2=120, k3=160)),
    # hand-wired DIAGNOSTIC (X1d oracle): a fixed moving code per (question, lag) replaces L3 after each heard question
    'R1e_oracle': dict(care='format', oracle=True, b=dict(R1e)),
    'RNpm_oracle': dict(care='format', oracle=True, b=dict(RNpm)),
}


def _add_x4_and_shape():
    for name in list(ARMS):
        b = ARMS[name]['b']
        if name.endswith('_x4') or name in ('RNpm', 'R1e') or 'oracle' in name:
            continue
        bx = dict(b, **X4)
        if b.get('act') == 'global':
            bx.update(k2=120, k3=160)
        ARMS[name + '_x4'] = dict(care='format', b=bx)
    for name in list(ARMS):
        ARMS[name + '_shape'] = dict(ARMS[name], shaping=True)
        ARMS[name + '_real'] = dict(ARMS[name], care='real')


_add_x4_and_shape()
