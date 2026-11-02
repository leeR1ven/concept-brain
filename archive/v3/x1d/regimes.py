"""phase-2 arms.  All: kappa = 0 (no self-hold), no fatigue / refractoriness / time cells / event boundaries.
R1 = best realistic (own voice heard) phase-1 regime: global quantile threshold (shortcut reference), reciprocal birth
     loop (designer msg 84 form), lateral w33 0.4.
RN = no PFC inhibitory neurons (coordinator's main arm, designer 11-01): thresholds a fraction of each neuron's own
     in-weight (X1c arm C values), no birth loop (every birth loop tried either saturates or dies, phase 1); the learned
     top->L1 drive is bounded by the ARCH-form two-layer weights (nrm31='arch') and an absolute prediction floor.
RA = local feedback inhibition fallback (X1c arm A form, kappa 0, reciprocal loop).
"""
ONSET = dict(rule='onset')
R1 = dict(act='global', kappa=0.0, birth31='recip', g_loop=4.0, w33=0.4, pred_rel=False, **ONSET)
RN = dict(act='frac', kappa=0.0, g_loop=0.0, w33=0.1, f1=0.0, f2=0.22, f3=0.3, nrm31='arch', g31=1.0, floor_in=0.35,
          **ONSET)
RA = dict(act='local', kappa=0.0, birth31='recip', g_loop=4.0, w33=0.1, pred_rel=False, g_loc=0.3, **ONSET)

R1e = dict(R1, post='ext')
R1er = dict(R1e, pred_rel=True, f1=0.35)
RNe = dict(RN, post='ext')
# no-inhibition main arm, growth/decay attempt: learned gain set from the equilibrium L* (see report), strong decay
RNg = dict(RN, post='ext', g31=0.3, dec_31=0.99)
# no-inhibition + X3 patch-mean of the prediction / loop drive (fallback)
RNpm = dict(RN, post='ext', pm=1.0)
R1e_real = dict(R1e)
ARMS = {
    'R1e_oracle': dict(care='format', oracle=True, b=R1e),        # DIAGNOSTIC (hand-wired reproducible path)
    'R1e_oracle_real': dict(care='real', oracle=True, b=R1e),
    'RNpm_oracle': dict(care='format', oracle=True, b=RNpm),
    'RNpm_shape': dict(care='format', shaping=True, b=RNpm),
    'RNpm_shape_real': dict(care='real', shaping=True, b=RNpm),
    'RNg': dict(care='format', b=RNg),
    'RNpm': dict(care='format', b=RNpm),
    'RNpm_real': dict(care='real', b=RNpm),
    'R1e_real': dict(care='real', b=R1e),
    'R1e_shape': dict(care='format', shaping=True, b=R1e),
    'R1e_shape_real': dict(care='real', shaping=True, b=R1e),
    'R1e': dict(care='format', b=R1e),
    'R1er': dict(care='format', b=R1er),
    'RNe': dict(care='format', b=RNe),
    'R1': dict(care='format', b=R1),
    'R1_real': dict(care='real', b=R1),
    'R1_shape': dict(care='format', shaping=True, b=R1),
    'R1_shape_real': dict(care='real', shaping=True, b=R1),
    'R1_x1crule': dict(care='format', b=dict(R1, rule='x1c')),
    'RN': dict(care='format', b=RN),
    'RN_real': dict(care='real', b=RN),
    'RA': dict(care='format', b=RA),
}
