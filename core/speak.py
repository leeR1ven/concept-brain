"""Speaking in the big dialogue world (designer 11-11: "把运动说话相关的加上，这样就能直接知道模型要表达什么信息").

ARCH 五 训练顺序 1 and 三 运动区: generation 0 hand-wires (registered) "sound <-> the muscles that make it" inside the PFC,
and the motor area's settling (a mixed tendency settles on the strongest whole syllable; none: no speech).
Big-world form (config speak, with pfc_from_concept and syl_ticks 1):
  - the mouth gives a whole syllable per tick, as the ear hears one: 72 channels = 24 muscles x (onset, nucleus,
    offset) in one tick (registered deviation; with three ticks per syllable it goes back to 24 muscles per tick);
  - the speech link: a THOUGHT concept copy (lit from inside) -> the force copies of the syllables whose concept code
    it belongs to (the PFC hears sounds only through the concept area now, so a sound's representation is its concept
    code). Measured on the life when the concept area is added; stored at the strength of a used link;
  - an intent to move (a force copy) is not a thought: it does not use the thought budget (brain.PFC.step);
  - pools: per syllable an excitatory and an inhibitory group (gen0.wire_pools, for 72 channels);
  - the mouth (corpus_life): the channels that move this tick, an exact match to a syllable's pattern -> that syllable
    in the child's own voice, heard by the child the next tick; otherwise nothing comes out but a garble."""
import numpy as np
import torch

from .gen0 import nnls
from .life import log
from .net import DEV, Links, Pop
from .world.voice import ADULTS


def patterns72(body):
    """bool [n_sid, 72]: channel = part x 24 + muscle"""
    return body.pat.reshape(len(body.pat), -1)


class SpeechLink:
    """thought concept copies -> force copies (generation 0, registered). drive(x_int) -> push into every L1 copy."""

    def __init__(self, units, W, f0, f1, n):
        self.units, self.W, self.f0, self.f1, self.n, self.gain = units, W, f0, f1, n, 1.0

    def drive(self, x):
        out = torch.zeros(self.n, device=DEV)
        out[self.f0:self.f1] = self.gain * (x[self.units].float() @ self.W)
        return out


def concept_codes(life, sids, p_rel=0.5):
    """per syllable: the concept copies lit from outside when it is heard alone, in >= p_rel of the adult voices"""
    b, p = life.brain, life.brain.pfc
    segs = {(k, l): (a, bb) for k, l, a, bb in b.chains.segs}
    ca, cb = segs[('concept', 0)]
    C = torch.zeros((len(sids), cb - ca), device=DEV)
    hear, n_ext, k = getattr(b, 'pfc_in', None), 0.0, 0
    for i, x in enumerate(sids):
        for v in ADULTS:
            rec = []
            life.play([None] * 4 + life.syl(x, v) + [None] * 2, rec=lambda t: rec.append(p.L1ext.on.clone()))
            C[i] += torch.stack(rec[5:7])[:, ca:cb].any(0).float() / len(ADULTS)   # its code comes a tick after the ear
            if hear is not None:
                n_ext += float((rec[5] & hear).sum())
                k += 1
    life._ext_heard = n_ext / max(1, k)                   # copies the PFC hears from outside while a syllable is heard
    return C >= p_rel, ca, cb


def wire_speech(life, n_syl=None):
    b, c = life.brain, life.c
    pfc, mot = b.pfc, b.motor
    n_syl = int(n_syl or c.get('speak_n', 600))
    cnt = np.bincount(life.train.syl.astype(np.int64), minlength=len(life.body.pat))
    sids = [int(x) for x in np.argsort(-cnt)[:n_syl] if cnt[x] > 0]
    code, ca, cb = concept_codes(life, sids)
    share = code.float().mean(0)                                  # units in many syllables' codes say little
    code = code & (share <= c.get('speak_unit_max', 0.05))[None, :]
    keep = code.sum(1) >= 3
    sids = [s for s, k in zip(sids, keep.tolist()) if k]
    code = code[keep]
    need = torch.as_tensor(patterns72(life.body)[sids], dtype=torch.float32, device=DEV)   # [nS, 72]
    units = torch.nonzero(code.any(0)).squeeze(1)
    W = nnls(code[:, units].float(), need)                        # [n_units, 72]
    Wf = W[:, mot.muscle_of]                                      # [n_units, nf]
    _, _, f0, f1 = b.chains.seg('force')[0]
    link = SpeechLink(ca + units, Wf, f0, f1, b.n1)
    link.code = code[:, units].float()                            # [nS, n_units]: each wired syllable's code
    link.size = link.code.sum(1).clamp_min(1.0)
    link.copies = need.bool()[:, mot.muscle_of]                   # [nS, nf]: its force copies (every grade)
    # gain: thinking a syllable lights only a few of its copies (the thought budget); a random `budget`-sized part of
    # its code drives its own channels' copies at speak_rel x theta1 (median over the wired syllables)
    g = torch.Generator(device='cpu').manual_seed(3)
    # the thought budget in life: context x the usual count of copies the PFC hears from outside. Just after the
    # concept area is added pfc.budget still counts every copy (192 in the first smoke test, 9-13 in life), so it is
    # measured on the copies it hears (pfc_in). Saying a syllable takes about half of the budget on its code.
    hear = getattr(b, 'pfc_in', None)
    n_ext = float(code.sum(1).float().mean()) * 3 if hear is None else life._ext_heard
    k = max(1, int(round(c['context'] * n_ext * c.get('speak_share', 0.5))))
    d = []
    for i in range(len(sids)):
        own = torch.nonzero(code[i]).squeeze(1)
        part = own[torch.randperm(len(own), generator=g)[:k].to(DEV)]
        x = torch.zeros(b.n1, dtype=torch.bool, device=DEV)
        x[ca + part] = True
        ch = need[i].bool()[mot.muscle_of]
        d.append(link.drive(x)[f0:f1][ch].mean())
    link.gain = float(c.get('speak_rel', 1.5) * c['theta1'] / torch.stack(d).median().clamp_min(1e-9))
    pfc.G1C = link
    mm = torch.zeros(b.n1, dtype=torch.bool, device=DEV)
    mm[f0:f1] = True
    pfc.motor_mask = mm
    pools = wire_pools72(life, sids, need.bool().cpu().numpy())
    life.speak_sids = sids
    life.registry = getattr(life, 'registry', []) + [('speech link (generation 0)', len(sids), int(len(units))),
                                                     ('motor pools (generation 0)', pools['groups'], pools['per_group'])]
    log(f'speech wired: {len(sids)} syllables, {int(len(units))} concept units, {code.sum(1).float().mean():.0f} per '
        f'syllable, copies heard {getattr(life, "_ext_heard", 0):.0f}, units thought to say one {k}, gain {link.gain:.3f}; '
        f'pools {pools}')
    return sids


def wire_pools72(life, sids, need):
    """gen0.wire_pools for the wired syllables and any channel count: per syllable mc_n excitatory pool neurons
    (-> its own channels' force neurons) and mc_n inhibitory ones (-> the other channels'); each samples one force
    neuron of each channel the syllable uses and fires when (nearly) all of them are lit"""
    b, c = life.brain, life.c
    mot = b.motor
    m, nf, nI, ns = mot.m, mot.nf, c['mc_n'], need.shape[0]
    g = np.random.default_rng(c['lineage_seed'] + 950)
    mus_of = mot.muscle_of.cpu().numpy()
    g0 = c['gen0_strength'] * c['w_ref']

    def samples():
        out = []
        for s in range(ns):
            ch = np.nonzero(need[s])[0]
            for _ in range(nI):
                out.append(ch * m + g.integers(0, m, len(ch)))
        return out

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
        return table([x or [len(S)] for x in lists], len(S), 'force -> pool')
    syl = np.repeat(np.arange(ns), nI)
    p = dict(in_e=into_pool(samples()), in_i=into_pool(samples()),
             exc=table([np.nonzero(need[s][mus_of])[0] for s in syl], nf, 'pool -> own force'),
             inh=table([np.nonzero(~need[s][mus_of])[0] for s in syl], nf, 'pool -> other force (inhibitory)'),
             pe=Pop(ns * nI), pi=Pop(ns * nI))
    rel = float((mot.thr_f + c['b_block'] * mot.n_tonic / (8.0 * m)).median())
    ge, gi = [], []
    for s in range(0, ns, max(1, ns // 50)):                      # one fully active group
        act = torch.zeros(ns * nI, dtype=torch.bool, device=DEV)
        act[s * nI:(s + 1) * nI] = True
        own = torch.as_tensor(need[s][mus_of], device=DEV)
        ge.append(p['exc'].drive(act)[own].median())
        gi.append(p['inh'].drive(act)[~own].median())
    p['exc'].gain = float(c['mc_e'] * rel / torch.stack(ge).median())
    p['inh'].gain = float(c['mc_i'] * rel / torch.stack(gi).median())
    p['tables'] = [p['in_e'], p['in_i'], p['exc'], p['inh']]
    mot.pools = p
    return dict(groups=2 * ns, per_group=nI)


class Mouth:
    """the channels that move -> a sound. An exact match to a syllable's 72-channel pattern says it; a near one says the
    nearest syllable (a mouth makes some sound for any posture: tol = the least Jaccard similarity, 1.0 = exact only);
    otherwise a garble (no syllable)."""

    def __init__(self, body, tol=1.0, sids=None):
        P = patterns72(body)
        self.lookup = {}
        for sid in range(len(P)):
            self.lookup.setdefault(P[sid].tobytes(), sid)
        self.tol = float(tol)
        self.sids = np.arange(len(P)) if sids is None else np.asarray(sids)
        self.P = P[self.sids].astype(np.float32)
        self.n = self.P.sum(1)

    def say(self, muscles):
        m = np.asarray(muscles, bool)
        if not m.any():
            return None
        x = self.lookup.get(m.tobytes())
        if x is not None or self.tol >= 1.0:
            return 'garble' if x is None else x
        inter = self.P @ m.astype(np.float32)
        jac = inter / (self.n + m.sum() - inter)
        i = int(jac.argmax())
        return int(self.sids[i]) if jac[i] >= self.tol else 'garble'
