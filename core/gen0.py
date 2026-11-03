"""Generation-0 hand wiring (ARCH 五 训练顺序 1, 三 运动区 落定). Every set is registered in life.registry.
  codes: which sound copies each syllable lights (measured on the newborn sound area, adult voices)
  G1P:   PFC L4 units that are on while the PFC thinks syllable x -> the L1 copies of x's force neurons
         (a learned-type L4(t) -> L1(t+1) link inside the PFC; it keeps learning and decaying)
  pools: per syllable an excitatory and an inhibitory group of motor neurons fed by the lit returns, so a mixed
         tendency settles on the strongest whole syllable (collapse)
Links are stored at the strength of a well-used link (gen0_strength x w_ref), so everyday learning refines them
instead of washing them out (ARCH 六)."""
import numpy as np
import torch

from .net import DEV, Links, Pop
from .world.adapter import ADULTS, NSYL


def nnls(A, B, lam=0.3, it=3000):
    """min |A W - B|^2 + lam^2 |W|^2 with W >= 0 (projected gradient)"""
    L = torch.linalg.matrix_norm(A, ord=2) ** 2 + lam ** 2
    W = torch.zeros((A.shape[1], B.shape[1]), device=DEV)
    for _ in range(it):
        W = (W - (A.T @ (A @ W - B) + lam ** 2 * W) / L).clamp_min(0.0)
    return W


def sound_codes(brain, coch, reps=2, pre=4, post=8, p_rel=0.5):
    """for each syllable, the sound copies it lights in >= p_rel of its ticks (sound area alone, from silence; each
    layer read at its own delay), averaged over the adult voices. Indices into the PFC L1 / H1 copies."""
    snd, ch = brain.sound, brain.chains
    n = sum(len(ix) for ix in ch.sound_idx)
    P = torch.zeros((NSYL, n), device=DEV)
    no_ret = [torch.zeros(k, device=DEV) for k in snd.n]
    saved = [x.clone() for x in snd.state]
    for s in range(NSYL):
        for v in ADULTS:
            for _ in range(reps):
                snd.state = [torch.zeros_like(x) for x in saved]
                H = []
                for srcs in [[]] * pre + [[('s', s, v, 1.0)]] * 3 + [[]] * post:
                    st = snd.step(torch.as_tensor(coch.a0(srcs), device=DEV), no_ret)
                    H.append(torch.cat([st[l][ix] for l, ix in enumerate(ch.sound_idx)]).float())
                H = torch.stack(H)
                a = 0
                for l, ix in enumerate(ch.sound_idx):
                    P[s, a:a + len(ix)] += H[pre + l: pre + l + 3, a:a + len(ix)].mean(0) / (reps * len(ADULTS))
                    a += len(ix)
    snd.state = saved
    _, _, a0, _ = ch.seg('sound')[0]
    return [a0 + torch.nonzero(P[s] >= p_rel).squeeze(1) for s in range(NSYL)]


def inner_l4(life, pre=4, post=10):
    """M[x, j]: how often L4 unit j is on while the PFC thinks x (x's sound copies lit inside the PFC for 3 ticks,
    mouth muted, learning off), read 3-9 ticks after the thought began"""
    b = life.brain
    M = torch.zeros((NSYL, b.pfc.n), device=DEV)
    for x in range(NSYL):
        if life.codes[x].numel() == 0:
            continue
        rec = []
        life.play([None] * (pre + 3 + post), mute=True, think={pre + k: [x] for k in range(3)},
                  rec=lambda t: rec.append(b.pfc.L4.on.clone()))
        M[x] = torch.stack(rec).float()[pre + 3: pre + 10].mean(0)
    return M


def wire_g1p(life):
    b, c, need = life.brain, life.c, life.ph.need_mat
    pfc, mot = b.pfc, b.motor
    M = inner_l4(life)
    sel = []                                  # each syllable: its g1p_k most specific units that are on while thinking it
    for x in range(NSYL):
        other = torch.cat([M[:x], M[x + 1:]]).max(0).values
        spec = torch.where(M[x] >= c['g1p_p'], M[x] - other, torch.full_like(other, -9.0))
        top = torch.topk(spec, c['g1p_k'])
        s_ = torch.zeros_like(M[x], dtype=torch.bool)
        s_[top.indices[top.values > 0]] = True
        sel.append(s_)
    sel = torch.stack(sel)                                                      # [34, n4]
    units = torch.nonzero(sel.any(0)).squeeze(1)
    W = nnls(sel[:, units].float(), torch.as_tensor(need, dtype=torch.float32, device=DEV))   # [n_u, 24]
    _, _, f0, f1 = b.chains.seg('force')[0]
    idx = torch.arange(f0, f1, device=DEV).repeat(pfc.n, 1)
    w = torch.zeros((pfc.n, mot.nf), device=DEV)
    w[units] = W[:, mot.muscle_of]
    nz = w[w > 1e-6]
    if nz.numel():
        w = w * (c['gen0_strength'] * c['w_ref'] / nz.median())
    tab = Links(idx, w, b.n1, c, gamma=c['gamma_pfc'], name='G1P')
    d = []                                    # gain: thinking x drives x's own force copies at g1p_rel x theta1
    for x in range(NSYL):
        if sel[x].any():
            own = torch.as_tensor(need[x], device=DEV)[mot.muscle_of]
            d.append(tab.drive(M[x] >= 0.5)[f0:f1][own].median())
    tab.gain = float(c['g1p_rel'] * c['theta1'] / torch.stack(d).median()) if d else 0.0
    pfc.G1P = tab
    return dict(units=int(len(units)), per_syllable=sel.sum(1).tolist(), gain=tab.gain)


def wire_pools(life):
    """per syllable: nI excitatory pool neurons (-> the syllable's own force neurons) and nI inhibitory ones
    (-> the force neurons of the other muscles); each samples one force neuron of each of the syllable's 4 muscles
    and fires when all four are lit"""
    b, c, need = life.brain, life.c, np.asarray(life.ph.need_mat, bool)
    mot = b.motor
    m, nf, nI, ns = mot.m, mot.nf, c['mc_n'], need.shape[0]
    g = np.random.default_rng(c['lineage_seed'] + 950)
    mus_of = mot.muscle_of.cpu().numpy()
    g0 = c['gen0_strength'] * c['w_ref']

    def samples():
        return np.stack([np.nonzero(need[s])[0] * m + g.integers(0, m, 4) for s in range(ns) for _ in range(nI)])

    def table(lists, n_post, name):
        w_ = max(len(x) for x in lists)
        idx = np.full((len(lists), w_), n_post, np.int64)
        for i, x in enumerate(lists):
            idx[i, :len(x)] = x
        return Links(idx, (idx < n_post).astype(np.float32) * g0, n_post, c, gamma=c['gamma_motor'], name=name)

    def into_pool(S):
        lists = [[] for _ in range(nf)]
        for p, row in enumerate(S):
            for f in row:
                lists[f].append(p)
        return table([x or [S.shape[0]] for x in lists], S.shape[0], 'force -> pool')
    syl = np.repeat(np.arange(ns), nI)
    p = dict(in_e=into_pool(samples()), in_i=into_pool(samples()),
             exc=table([np.nonzero(need[s][mus_of])[0] for s in syl], nf, 'pool -> own force'),
             inh=table([np.nonzero(~need[s][mus_of])[0] for s in syl], nf, 'pool -> other force (inhibitory)'),
             pe=Pop(ns * nI), pi=Pop(ns * nI))
    rel = float((mot.thr_f + c['b_block'] * mot.n_tonic / (8.0 * m)).median())
    ge, gi = [], []
    for s in range(ns):                                                        # one fully active group
        act = torch.zeros(ns * nI, dtype=torch.bool, device=DEV)
        act[s * nI:(s + 1) * nI] = True
        own = torch.as_tensor(need[s][mus_of], device=DEV)
        ge.append(p['exc'].drive(act)[own].median())
        gi.append(p['inh'].drive(act)[~own].median())
    p['exc'].gain = float(c['mc_e'] * rel / torch.stack(ge).median())
    p['inh'].gain = float(c['mc_i'] * rel / torch.stack(gi).median())
    p['tables'] = [p['in_e'], p['in_i'], p['exc'], p['inh']]
    mot.pools = p
    return dict(groups=2 * ns, per_group=nI, gain_exc=p['exc'].gain, gain_inh=p['inh'].gain)
