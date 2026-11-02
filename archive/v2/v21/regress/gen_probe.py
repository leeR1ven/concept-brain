# -*- coding: utf-8 -*-
"""Learning-lens probe: does ARCH_v2's PFC readout generalise a frame to a filler never taught in that frame?

Symbolic harness (no audio, no body, no scene): the ARCH_v2 cognitive core (SynthCore with the final configuration's
core keys: continuous thought, recency, CTX off, awake recall 8 / 0.2 / uniform, 10k hippocampus, floor 0.3) hears
prompt lines, speaks through core.speak() (the same max/END/floor loop the body runs), gets the recast tutor (reward 1
if acceptable, then the correct reply is heard), and rehearses in a quiet moment after every exchange.

Four frames x 12 fillers; every filler is held out of exactly one frame and taught in the other three (bench style:
held-out pairs are only probed, never taught).
  offer  你要X吗     -> 要X          (copy the filler)
  like   你喜欢X吗   -> 喜欢X        (copy the filler)
  taste  X好不好吃   -> 好吃         (constant answer; filler first in the prompt)
  where  X在哪里     -> 不知道       (constant answer; filler first in the prompt)

Variants (all local; none reads anything the agent could not perceive):
  base      ARCH_v2 core as is
  noid      gate_noid=True: the BG copy gates read context only, never the candidate's identity
  K         + a learned distributional code: each item's 'meaning' unit accumulates (Hebbian) the fixed random index
              vectors of its left and right neighbours in heard lines (random indexing with order permutation);
              its k-WTA (8 of 512) is the item's class code.  The held thought's class bits join the context
              (group K) with pairs (P,K) (K,N) (K,L); gates as noid.
  Koracle   as K but the class code is given (one shared code for all fillers): an upper bound for any learned code.
  R         + a turn-role tag (an LLM's speaker tag, perceivable through the efference copy): every context key is
              tagged with R of the held line = 'the agent spoke between the previous heard line and this line'.
              A recast (heard after the own reply) is held with R=1, a prompt with R=0, so the transitions
              prompt -> answer and answer -> next prompt no longer share readout units.
  RK        R and K together.
  own       ct_own=1: the own reply enters the thought (efference) with gain 1, so the next heard line is learned in the
              context (prompt + own reply) instead of the prompt alone (applies to recasts too).
  Cor       ORACLE contingency (upper bound, not a mechanism): after an unanswered reply (probe) the own reply enters
              the thought as a line (displacing the prompt), so the next partner line is learned after the own reply,
              not as the prompt's answer; after a recast exchange nothing changes.
  Cor2      ORACLE as Cor, and the hippocampus does not link the next line as the probe's successor (NEXT).
  Hcut      ORACLE: only the hippocampal NEXT link is cut after an unanswered reply (cortex as base).
  Xcut      perceivable (no oracle): after ANY own utterance the own reply enters the thought as a line and the next
              heard line is not linked as the prompt's successor -- answers are then learned only from other
              speakers' answers (tutor=mixed: model replies), recasts no longer teach the prompt's answer.
Measurement (v2): live = held-out pairs probed inside the stream with learning on (as in life: the line after an
unanswered probe is learned in the probe's context, so live held-out mixes generalisation with drift);
frozen = at the end a deep copy with learning off answers every held-out pair 3 times (as run_j's frozen probes).
heldstream=0 removes held-out pairs from the stream (frozen = pure generalisation, nothing learned about them).
usage: python3 gen_probe.py variant=base n=900 seed=0 [heldstream=1]
"""
import os, sys, json, time, collections
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..', '..', 'synth')
sys.path.insert(0, SYN)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
import numpy as np                                   # noqa: E402
from core_synth import SynthCore, SCfg               # noqa: E402
from rules import pair_keys                          # noqa: E402
from core import BOS, NONE                           # noqa: E402

A = dict(a.split('=', 1) for a in sys.argv[1:])
VAR = A.get('variant', 'base')
N_DLG = int(A.get('n', 900))
SEED = int(A.get('seed', 0))
HELDSTREAM = int(A.get('heldstream', 1))
TUTOR = A.get('tutor', 'recast')      # 'mixed': 30 % of seen exchanges are model replies by another speaker

FRAME_CH = list('你要吗喜欢好不吃在哪里知道')
FILL = ['苹', '蕉', '瓜', '饼', '奶', '葡', '球', '车', '狗', '猫', '鸟', '鸭']   # 12 one-char fillers
VOC = FRAME_CH + FILL
ID = {c: i for i, c in enumerate(VOC)}
FILL_ID = set(ID[c] for c in FILL)
FRAMES = {
    'offer': (lambda x: ['你', '要', x, '吗'], lambda x: ['要', x]),
    'like': (lambda x: ['你', '喜', '欢', x, '吗'], lambda x: ['喜', '欢', x]),
    'taste': (lambda x: [x, '好', '不', '好', '吃'], lambda x: ['好', '吃']),
    'where': (lambda x: [x, '在', '哪', '里'], lambda x: ['不', '知', '道']),
}
FN = list(FRAMES)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(FILL))
HELD = {f: set(FILL[j] for j in perm[3 * k:3 * k + 3]) for k, f in enumerate(FN)}   # each filler held out once
PROMPTS = set(''.join(FRAMES[f][0](x)) for f in FN for x in FILL)


class RCore(SynthCore):
    """SynthCore + a turn-role tag R on every readout key (R of the held line, stored with the line)."""
    use_R = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.spoke = False            # efference: the agent spoke since the last heard line ended
        self.cur_R = 0

    def end_line(self, st, store=True):
        super().end_line(st, store=store)
        if self.last_rec is not None:
            self.last_rec['R'] = self.cur_R
        self.prev = self._ct_view()

    def _ct_view(self):
        v = super()._ct_view()
        if v is not None and self.T_dom is not None:
            v['R'] = self.T_dom.get('R', 0)
        return v

    def _keys(self, st, cands):
        keys = self._keys_base(st, cands)
        if not self.use_R:
            return keys
        r = 'r%d' % int((self.prev or {}).get('R', 0))
        return [(r,) + k if len(k) == 4 else (k[0] + '|' + r, k[1]) for k in keys]

    def _keys_base(self, st, cands):
        return SynthCore._keys(self, st, cands)


class KCore(RCore):
    """SynthCore + a class-code context group K (learned by random indexing, or oracle)."""
    D, KK = 512, 8

    def __init__(self, cfg, mode):
        super().__init__(cfg)
        self.mode = mode
        r = np.random.default_rng(cfg.seed + 4242)
        V = cfg.V
        self.idxL = np.zeros((V, self.D), np.float32)
        self.idxR = np.zeros((V, self.D), np.float32)
        for i in range(V):
            self.idxL[i, r.choice(self.D, 8, replace=False)] = 1
            self.idxR[i, r.choice(self.D, 8, replace=False)] = 1
        self.ctxv = np.zeros((V, self.D), np.float32)   # meaning unit i <- index units (Hebbian co-activity)
        self.nseen = np.zeros(V, np.int64)
        self.oracle = {}
        shared = tuple(range(1000, 1000 + self.KK))
        for i in range(V):
            self.oracle[i] = shared if i in FILL_ID else tuple(range(2000 + self.KK * i, 2000 + self.KK * (i + 1)))

    def observe(self, items):
        """Hebbian random indexing over a heard line (local: item unit x neighbour index units)."""
        for p, i in enumerate(items):
            self.nseen[i] += 1
            if p > 0:
                self.ctxv[i] += self.idxL[items[p - 1]]
            if p + 1 < len(items):
                self.ctxv[i] += self.idxR[items[p + 1]]

    def kbits(self, i):
        if not (0 <= i < self.c.V):
            return ()
        if self.mode == 'Koracle':
            return self.oracle[i]
        if self.nseen[i] < 3:
            return ()
        v = self.ctxv[i]
        return tuple(int(b) for b in np.argpartition(-v, self.KK)[:self.KK])

    def _keys_base(self, st, cands):
        keys = SynthCore._keys(self, st, cands)
        if self.mode not in ('K', 'Koracle', 'RK'):
            return keys
        P = list(dict.fromkeys(st.get('Pbag') if st.get('Pbag') is not None else st['P']))
        K = sorted(set(b for i in P for b in self.kbits(i)))
        if not K:
            return keys
        n = len(st['items'])
        L = st['items'][-1] if st['items'] else BOS
        g = {'K': K, 'N': [min(n, 6)], 'L': [L], 'P': P}
        extra = pair_keys(g, [('P', 'K'), ('K', 'N'), ('K', 'L')])
        extra = [k for k in extra if k[0] == 'K' or (len(k) == 4 and 'K' in (k[0], k[2]))]
        return keys + extra


def make_core():
    kw = dict(V=len(VOC), n_eye=48, seed=SEED, retino=False, ct=True, hip_rho=2.0, ctx=False,
              recall_n=8, eta_recall=0.2, rec_pow=0.0, hip_cap=10000, hip_ndg=6000, hip_kdg=120, floor=0.3,
              think_k=3, ior=0)
    if VAR in ('noid', 'K', 'Koracle', 'RK'):
        kw['gate_noid'] = True
    if VAR == 'own':
        kw['ct_own'] = 1.0
    cfg = SCfg(**kw)
    core = KCore(cfg, VAR)
    core.use_R = VAR in ('R', 'RK')
    return core


def ids(chars):
    return [ID[c] for c in chars]


def classify(y, ok, prompt):
    if ok:
        return 'correct'
    if y == '':
        return 'silent'
    if y == prompt:
        return 'echo_prompt'
    if y in PROMPTS or any(p.startswith(y) for p in PROMPTS if len(y) >= 2):
        return 'partner_line'
    return 'other'


def main():
    t0 = time.time()
    core = make_core()
    log = []
    for d in range(N_DLG):
        for _ in range(int(rng.integers(1, 4))):
            f = FN[int(rng.integers(len(FN)))]
            x = FILL[int(rng.integers(len(FILL)))]
            q, a = FRAMES[f][0](x), FRAMES[f][1](x)
            held = x in HELD[f]
            if held and not HELDSTREAM:
                continue
            u = rng.random()
            kind = 'probe' if (held or u < 0.15) else ('model' if (TUTOR == 'mixed' and u < 0.45) else 'recast')
            core.observe(ids(q))
            core.cur_R = int(core.spoke); core.spoke = False
            core.hear_line(ids(q))
            if kind == 'model':                   # another speaker answers at once (bench 'model'): not scored
                core.observe(ids(a))
                core.cur_R = 0
                core.hear_line(ids(a))
                core.awake_recall()
                continue
            out = core.speak()
            core.own_finish(core.own)
            core.spoke = len(out) > 0
            y = ''.join(VOC[i] for i in out)
            ok = y == ''.join(a)
            log.append(dict(d=d, frame=f, filler=x, held=held, kind=kind, reply=y, ok=ok,
                            cls=classify(y, ok, ''.join(q))))
            if VAR == 'Xcut' and out:             # perceivable rule: never learn across the own utterance
                core._ct_input(dict(items=list(out), eye=None, att=None, code=None, next=None, ep=-1))
                core.prev = core._ct_view()
                core.last_rec = None
            if kind == 'recast':
                core.reward(1.0 if ok else 0.0)
                core.observe(ids(a))
                core.cur_R = int(core.spoke); core.spoke = False
                core.hear_line(ids(a))
            else:
                core.actor_tr = []
                if VAR in ('Cor', 'Cor2') and out:
                    core._ct_input(dict(items=list(out), eye=None, att=None, code=None, next=None, ep=-1))
                    core.prev = core._ct_view()
                if VAR in ('Cor2', 'Hcut') and out:
                    core.last_rec = None      # the next heard line is not stored as the probe's successor (NEXT)
            core.awake_recall()
    import copy
    fz = copy.deepcopy(core)
    frozen = []
    for f in FN:
        for x in sorted(HELD[f]) + [y for y in FILL if y not in HELD[f]][:3]:
            q, a = FRAMES[f][0](x), FRAMES[f][1](x)
            for _ in range(3):
                fz.cur_R = int(fz.spoke); fz.spoke = False
                fz.hear_line(ids(q), learn=False, store=False)
                out = fz.speak(); fz.actor_tr = []
                fz.spoke = len(out) > 0
                y = ''.join(VOC[i] for i in out)
                ok = y == ''.join(a)
                frozen.append(dict(frame=f, held=x in HELD[f], reply=y, ok=ok, cls=classify(y, ok, ''.join(q))))
    last = [r for r in log if r['d'] >= 2 * N_DLG // 3]
    res = dict(variant=VAR, seed=SEED, n_dlg=N_DLG, heldstream=HELDSTREAM, tutor=TUTOR, secs=round(time.time() - t0),
               mix=int(core.mix.n))
    fh = [r for r in frozen if r['held']]; fs = [r for r in frozen if not r['held']]
    res['frozen'] = dict(seen=round(np.mean([r['ok'] for r in fs]), 3), held=round(np.mean([r['ok'] for r in fh]), 3),
                         n_held=len(fh), held_cls={k: round(v / len(fh), 2) for k, v in
                                                   collections.Counter(r['cls'] for r in fh).items()},
                         per_frame={f: round(np.mean([r['ok'] for r in fh if r['frame'] == f]), 2) for f in FN},
                         examples=[(r['frame'], r['reply']) for r in fh][::3][:8])
    for f in FN + ['ALL']:
        R = [r for r in last if f == 'ALL' or r['frame'] == f]
        h = [r for r in R if r['held']]
        s = [r for r in R if not r['held'] and r['kind'] == 'probe']
        res[f] = dict(seen=round(np.mean([r['ok'] for r in s]), 3) if s else None, n_seen=len(s),
                      held=round(np.mean([r['ok'] for r in h]), 3) if h else None, n_held=len(h),
                      held_cls={k: round(v / len(h), 2) for k, v in collections.Counter(r['cls'] for r in h).items()}
                      if h else None)
    res['examples'] = [(r['frame'], r['filler'], r['reply']) for r in last if r['held']][:12]
    print(json.dumps(res, ensure_ascii=False), flush=True)
    with open(os.path.join(HERE, 'gen_probe_out.jsonl'), 'a') as fo:
        fo.write(json.dumps(res, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
