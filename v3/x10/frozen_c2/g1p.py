"""Generation-0 speech wiring G1P (designer 11-02; ARCH 2.7, 2.11.1/3, 2.12): sound and action are linked only
through the PFC. Registry entry G1P:
  from: PFC L4 units that represent syllable x, selected on the frozen newborn (after the birth period, loop and
        lateral on, learning off) with the probe world: they fire in >= g1p_p of the ticks where x's code reaches
        L4 (L4 ticks 4-9 after the onset: chains 1-4 ticks, then L1 -> L2 -> L3 -> L4) in the four adult voices,
        and >= g1p_margin more often than for any other syllable. Arm (ii) g1p_own_excl: also drop units that the
        child's own voice lights in >= g1p_own_cut of x's ticks.
  to:   the PFC L1 copies of x's force neurons (the motor chain copies already in L1).
  how:  non-negative least squares (ridge 0.3) from the selected-unit membership to x's 4-muscle set, the same
        weight to every force grade of a muscle; one gain set so that a syllable's full unit set drives its own force
        copies at g1p_rel x the L1 internal threshold.
  form: a table with the form of the learned prediction links L4(t) -> L1(t+1): acquired part, two-layer weights,
        decay gamma_pfc, and the same learning rule and post event as L41. No gate and no decay exemption.
  path: internally lit force copies -> 1:1 returns -> force neurons -> competition with tonic / ladder blocking."""
import numpy as np
import torch

from common import DEV
from learning import ByPre
from probes import Frozen, play
from wiring import nnls
from world_adapter import ADULTS, NSYL, OWN


def measure_l4(life, voices, reps=2, pre=4, post=10):
    """P[x, v, j]: firing probability of L4 unit j in the L4 ticks of syllable x heard in voice v (mouth muted)"""
    b = life.brain
    P = torch.zeros((NSYL, len(voices), b.pfc.n), device=DEV)
    with Frozen(life) as snap:
        for x in range(NSYL):
            for vi, v in enumerate(voices):
                for _ in range(reps):
                    b.restore(snap)
                    rec = []
                    play(life, [None] * pre + [(x, v, 1.0)] * 3 + [None] * post, mute=True,
                         rec=lambda lf, t: rec.append(lf.brain.pfc.L4.on.clone()))
                    H = torch.stack(rec).float()
                    P[x, vi] += H[pre + 4: pre + 10].mean(0) / reps
    return P


def measure_l4_inner(life, pre=4, post=10):
    """P[x, j]: firing probability of L4 unit j while the PFC 'thinks' x: x's sound code lit in PFC L1 for 3 ticks
    (inner hearing; mouth muted), L4 read 3-9 ticks after the first forced tick"""
    from probes import code_mask
    b = life.brain
    P = torch.zeros((NSYL, b.pfc.n), device=DEV)
    with Frozen(life) as snap:
        for x in range(NSYL):
            if life.codes[x].numel() == 0:
                continue
            b.restore(snap)
            rec = []
            play(life, [None] * (pre + 3 + post), mute=True, force={pre + k: code_mask(life, [x]) for k in range(3)},
                 rec=lambda lf, t: rec.append(lf.brain.pfc.L4.on.clone()))
            P[x] = torch.stack(rec).float()[pre + 3: pre + 10].mean(0)
    return P


def background_l4(life, n_lines=24, seed=0):
    """each L4 unit's firing rate in ordinary listening (random adult lines of 3 syllables with 6-12 silent ticks
    between them, learning off, state restored afterwards): the held / ordinary state"""
    b = life.brain
    rng = np.random.default_rng(seed + 4242)
    script = [None] * 8
    for _ in range(n_lines):
        v = ADULTS[int(rng.integers(len(ADULTS)))]
        for x in rng.integers(0, NSYL, 3):
            script += [(int(x), v, 1.0)] * 3
        script += [None] * int(rng.integers(6, 13))
    rec = []
    with Frozen(life):
        play(life, script, mute=True, rec=lambda lf, t: rec.append(lf.brain.pfc.L4.on.clone()))
    return torch.stack(rec).float().mean(0)


def select_l4(life):
    """L4 units representing each syllable (>= g1p_p of x's L4 ticks, and >= g1p_margin more often than for any
    other syllable). g1p_src: 'heard' (the four adult voices), 'inner' (the PFC thinking x), or 'both' (the unit
    passes in both). Returns bool [34, n4] and the inner measure (or None)."""
    c = life.c

    def pick(M):
        out = []
        for x in range(NSYL):
            other = torch.cat([M[:x], M[x + 1:]]).max(0).values
            out.append((M[x] >= c['g1p_p']) & (M[x] - other >= c['g1p_margin']))
        return torch.stack(out)
    Mi = measure_l4_inner(life) if c['g1p_src'] in ('inner', 'both') else None
    if c['g1p_src'] == 'inner':
        sel = pick(Mi)
        if c['g1p_evoked']:
            life.l4_background = background_l4(life)
            sel = sel & (life.l4_background < c['g1p_off'])[None, :]
        return sel, Mi
    sel = pick(measure_l4(life, ADULTS).mean(1))
    if c['g1p_src'] == 'both':
        sel = sel & pick(Mi)
    if c['g1p_evoked']:                     # only units evoked by x: rarely on in ordinary listening / the held state
        life.l4_background = background_l4(life)
        sel = sel & (life.l4_background < c['g1p_off'])[None, :]
    return sel, Mi


def wire_g1p(life):
    b, c, ph = life.brain, life.c, life.ph
    pfc, mot = b.pfc, b.motor
    sel, Mi = select_l4(life)                                                      # [34, n4] bool
    life.l4_sel = sel
    n_adult = sel.sum(1).tolist()
    own_drop = None
    if c['g1p_own_excl']:
        Pown = measure_l4(life, (OWN,))[:, 0]
        drop = sel & (Pown >= c['g1p_own_cut'])
        own_drop = drop.sum(1).tolist()
        sel = sel & ~drop
    counts = sel.sum(1).tolist()
    units = torch.nonzero(sel.any(0)).squeeze(1)
    A = sel[:, units].float()                                                      # [34, n_u]
    need = torch.as_tensor(ph.need_mat, dtype=torch.float32, device=DEV)            # [34, 24]
    W = nnls(A, need)                                                              # [n_u, 24]
    Wf = W[:, mot.muscle_of]                                                       # [n_u, 24 s]
    seg = [sg for sg in b.chains.segs if sg[0] == 'force'][0]
    fcopies = torch.arange(seg[2], seg[3], device=DEV, dtype=torch.int32)          # L1 indices of the force copies
    idx = fcopies.repeat(pfc.n, 1)
    w = torch.zeros((pfc.n, mot.nf), device=DEV)
    w[units] = Wf
    if c['gen0_strength'] > 0:              # store the inherited link at the strength of a well-used link (drive unchanged:
        nz = w[w > 1e-6]                    # two-layer weights are scale-free; only the concave plasticity changes)
        if nz.numel():
            w = w * (c['gen0_strength'] * c['w_ref'] / nz.median())
    tab = ByPre(idx, w, pfc.n, b.n1, 'PFC G1P L4 -> L1 force copies (gen-0, acquired)', birth=False, eta=c['eta'],
                eta_m=c['eta_m'], gain=1.0, gamma=b.gam['pfc'], c=c)
    # gain: a syllable's full unit set drives its own force copies at g1p_rel x the internal threshold theta_1
    own_d, exact = [], []
    for x in range(NSYL):
        if counts[x] == 0:
            exact.append(False)
            continue
        d = tab.drive(sel[x])[seg[2]:seg[3]]                                       # [24 s] drive into force copies
        own = torch.as_tensor(ph.need_mat[x], device=DEV)[mot.muscle_of]
        own_d.append(d[own].median())
        per_muscle = torch.zeros(24, device=DEV).index_add_(0, mot.muscle_of, d)
        exact.append(set(torch.topk(per_muscle, 4).indices.tolist()) == set(np.nonzero(ph.need_mat[x])[0].tolist()))
    gain = float(c['g1p_rel'] * c['theta1'] / torch.stack(own_d).median()) if own_d else 0.0
    if Mi is not None and own_d:            # gain from the inner state: thinking x drives x's own copies at g1p_rel x theta_1
        tab.gain = 1.0
        di = []
        for x in range(NSYL):
            if counts[x] == 0:
                continue
            d = tab.drive(Mi[x] >= 0.5)[seg[2]:seg[3]]
            own = torch.as_tensor(ph.need_mat[x], device=DEV)[mot.muscle_of]
            di.append(d[own].median())
        med = float(torch.stack(di).median())
        gain = float(c['g1p_rel'] * c['theta1'] / med) if med > 0 else gain
    tab.gain = gain
    pfc.G1P = tab
    return dict(set='G1P', src=c['g1p_src'], evoked=c['g1p_evoked'], off=c['g1p_off'], own_excl=c['g1p_own_excl'], units=int(len(units)), per_syllable=counts,
                per_syllable_adult_only=n_adult, own_dropped=own_drop,
                flagged_below_10=[int(x) for x in range(NSYL) if counts[x] < 10], top4_exact=float(np.mean(exact)),
                gain=gain, links=int((w > 1e-6).sum()))
