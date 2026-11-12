"""The designer's few-hearings test (11-11: "模型应该只需要接收几遍就能记住"): a NEW pair "A B" (B almost never
follows A in the dialogues heard) is heard k times in one voice, learning on, with silence between. Then A alone (each
adult voice, learning off): is B thought of? Measured as in pair_test: the share of B's concept code lit from inside
(thought) in the ticks after A, against the other candidates, and how often B is the most lit. Every k starts from
the same saved brain (a deep copy), so the hearings do not add up across k.
A measuring tool, not part of the brain (it uses labels to pick the pairs and the codes).
usage: python -u -m core.fewshot <ckpt> [pairs=10] [reps=[1,3,5,10]] [gap=4] [after=3] [hip=1] [seed=0] [gstep=x]
       gstep: the PFC graded table's step for this test (designer 11-11: raise the PFC's learning for now, to see
       that it can remember within a few hearings; default = the life's own graded_step).
       hip=1 / 0: the hippocampus's recall lights its copies or not (config hip_recall), for this test only."""
import copy
import json
import sys
import time

import numpy as np
import torch

from .net import DEV
from .world.voice import ADULTS


def thought_shares(life, a, code, have, after):
    """A alone from silence in each adult voice (learning off): per candidate, the share of its code thought"""
    b, p = life.brain, life.brain.pfc
    segs = {(kind, l): (x, y) for kind, l, x, y in b.chains.segs}
    ca, cb = segs[('concept', 0)]
    n_code = code.sum(1).clamp_min(1).float()
    out = []
    prec = thought_shares.prec = []
    for v in ADULTS:
        rec = []
        # what is lit from inside: the PFC's thoughts and (config hip_recall) the hippocampus's recall
        inside = (lambda: p.x_int | b.hip.h_int) if life.c.get('hip_recall', True) else (lambda: p.x_int)
        life.play([None] * 6 + life.syl(a, v) + [None] * after, rec=lambda t: rec.append(inside().clone()))
        lit = torch.stack(rec[-after:]).any(0)
        sh = (lit[ca:cb][None, :] & code).sum(1).float() / n_code
        sh[~have] = -1
        out.append(sh)
        inner = lit[ca:cb]
        prec.append((inner[None, :] & code).sum(1).float() / max(1.0, float(inner.sum())))   # of what is thought
    return torch.stack(out)                                # [voices, candidates]


def teach(life, a, bb, k, gap, voice):
    """hear "A B" k times in one voice, learning on (states and weights change: run on a copy)"""
    b = life.brain
    was, b.learning = b.learning, True
    for _ in range(k):
        for x in [None] * gap + life.syl(a, voice) + life.syl(bb, voice):
            b.step(torch.as_tensor(life.with_sight(life.coch.a0([] if x is None else [x])), device=DEV), 0.0)
    b.learning = was


def main():
    ckpt = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:])}
    n_pairs, reps = int(over.get('pairs', 10)), list(over.get('reps', [1, 3, 5, 10]))
    gap, after, hip, seed = int(over.get('gap', 4)), int(over.get('after', 3)), int(over.get('hip', 1)), int(over.get('seed', 0))
    t0 = time.time()
    with torch.no_grad():
        life = torch.load(ckpt, weights_only=False)['life']
        life.c['hip_recall'] = bool(hip)
        if 'gstep' in over:
            life.c['graded_step'] = float(over['gstep'])
        if over.get('hip_rewire'):
            # test of a fix (11-11): with pfc_from_concept the hippocampus hears only the concept / force / stop copies,
            # but its birth fan-in (rec_fan 24) is drawn over all copies: about 0.07 lit inputs per unit, so its own
            # code is near random and its recall is generic. Draw its fan-in from the copies it hears (hip_fan of them)
            # and start its learned links from zero.
            from .net import Fixed
            hp, fan = life.brain.hip, int(over.get('hip_fan', 150))
            src = torch.nonzero(life.brain.pfc_in).squeeze(1).cpu().numpy()
            g = np.random.default_rng(11)
            hp.rec = Fixed(src[g.integers(0, len(src), (hp.nh, fan))], g.uniform(0.5, 1.5, (hp.nh, fan)))
            for tb in hp.plastic():
                tb.w.zero_()
            hp.usual.zero_()
            hp.usual_s = None
        life.next_ranks(n_dlg=1)                           # the candidates' concept codes, as pair_test
        S = life._nS[' concept']
        kc = max(1, int(round(float(S.sum(1).mean()))))
        code = torch.zeros_like(S, dtype=torch.bool).scatter_(1, torch.topk(S, kc, dim=1).indices, True) & (S > 0)
        have = life._nhave & code.any(1)
        cands = life._ncand
        seq = life.train.syl[:life.train.turn_off[life.train.dlg_off[min(life.env.n_ex, life.train.n)]]].tolist()
        import collections
        c2 = collections.Counter(zip(seq[:-1], seq[1:]))
        rng = np.random.default_rng(seed)
        ok = [i for i in range(len(cands)) if bool(have[i])]
        pairs = []
        while len(pairs) < n_pairs:
            i, j = rng.choice(ok, 2, replace=False)
            if c2[(cands[i], cands[j])] == 0 and c2[(cands[j], cands[i])] == 0:
                pairs.append((int(i), int(j)))
        print(f'{ckpt}: {life.env.n_ex} dialogues heard; hippocampus recall {"on" if hip else "off"}; PFC graded step '
              f'{life.c.get("graded_step")}; {len(pairs)} new '
              f'pairs (never adjacent in what was heard); code size {kc}; codes in {time.time() - t0:.0f} s', flush=True)
        res = {}
        k2s = list(over.get('then_c', []))                 # change test: after "A B" x reps, "A C" x each of these
        cc_ = None
        if k2s:
            cc_ = []
            for (i, j) in pairs:
                while True:
                    x = int(rng.choice(ok))
                    if x not in (i, j) and c2[(cands[i], cands[x])] == 0:
                        cc_.append(x)
                        break
        plan = [(k, 0) for k in [0] + reps] if not k2s else [(reps[0], k2) for k2 in [0] + k2s]
        for k, k2 in plan:
            sB, sO, top = [], [], []
            same, other = [], []                           # B most lit: in the voice it was taught in / the others
            pB, pO, pA, pC = [], [], [], []                # of the copies thought: in B's code / best other / A's / C's
            for n, (i, j) in enumerate(pairs):
                L = copy.copy(life)                        # the world and data shared; the brain copied
                L.brain = copy.deepcopy(life.brain)
                if k:
                    teach(L, cands[i], cands[j], k, gap, ADULTS[n % len(ADULTS)])
                if cc_ is not None and k2:                  # change: then "A C" k2 times
                    teach(L, cands[i], cands[cc_[n]], k2, gap, ADULTS[n % len(ADULTS)])
                sh = thought_shares(L, cands[i], code, have, after)
                for pr in thought_shares.prec:
                    pB.append(float(pr[j]))
                    pA.append(float(pr[i]))
                    m_ = torch.ones_like(pr, dtype=torch.bool)
                    m_[i] = m_[j] = False
                    pO.append(float(pr[m_].max()))
                    if cc_ is not None:
                        pC.append(float(pr[cc_[n]]))
                for vi, row in enumerate(sh):
                    others = torch.cat([row[:j], row[j + 1:]])
                    sB.append(float(row[j]))
                    sO.append(float(others[others >= 0].mean()))
                    top.append(bool(row[j] >= row.max()) and float(row[j]) > 0)
                    (same if vi == n % len(ADULTS) else other).append(top[-1])
                del L
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            res[k if not k2s else k2] = dict(B=round(float(np.mean(sB)), 3), others=round(float(np.mean(sO)), 3),
                          B_most=round(float(np.mean(top)), 2), same_voice=round(float(np.mean(same)), 2),
                          other_voices=round(float(np.mean(other)), 2), thought_in_B=round(float(np.mean(pB)), 3),
                          thought_in_A=round(float(np.mean(pA)), 3), thought_in_C=round(float(np.mean(pC)), 3) if pC else None,
                          thought_in_best_other=round(float(np.mean(pO)), 3))
            print(f'FEWSHOT heard {k:>2} times' + (f' then A C {k2:>2} times' if k2s else '') + f': B share {res[k if not k2s else k2]["B"]:.3f}, others {res[k if not k2s else k2]["others"]:.3f}, '
                  f'B most lit {res[k if not k2s else k2]["B_most"]:.2f} (taught voice {res[k if not k2s else k2]["same_voice"]:.2f}, other voices '
                  f'{res[k if not k2s else k2]["other_voices"]:.2f}) | of what is thought: in B {res[k if not k2s else k2]["thought_in_B"]:.3f}, in A '
                  f'{res[k if not k2s else k2]["thought_in_A"]:.3f}' + (f', in C {res[k if not k2s else k2]["thought_in_C"]:.3f}' if pC else '') +
                  f', in the best other {res[k if not k2s else k2]["thought_in_best_other"]:.3f} ({time.time() - t0:.0f} s)', flush=True)
        print('FEWSHOT', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
