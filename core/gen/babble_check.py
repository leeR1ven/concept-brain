"""The learned reverse path (grow.motor_reverse), checked before any life: the child alone in quiet babbles (innate
random syllables) and hears itself; only the reverse links learn (everything else frozen). Then, frozen: think a
syllable (its concept code lit inside the PFC for one tick) -> what the mouth says; hear a syllable -> must not repeat
it; an adult speaking -> silent.
usage: python -m core.gen.babble_check <ckpt> [ticks=20000] [n=40]"""
import collections
import json
import sys

import numpy as np
import torch


def main(ckpt, ticks=20000, n=40, step=0.2, pb=0.3, norm=False, dialogues=20, echo=0.5):
    import core.corpus_life  # noqa: F401
    from ..grow import motor_reverse
    from ..net import DEV
    from ..speak import concept_codes
    from ..world.voice import ADULTS
    life = torch.load(ckpt, weights_only=False)['life']
    motor_reverse(life, step=step, p_babble=pb, norm=norm)
    b, p, mot, cc = life.brain, life.brain.pfc, life.brain.motor, life.brain.concept
    b.learning = False
    life.mouth.tol = 1.0                                   # checks: the mouth says a syllable only at an exact posture
    own, said, driven = None, collections.Counter(), collections.Counter()
    pending = {}                                           # the world: a parent repeats a babbled syllable
    erng = np.random.default_rng(11)                       # two ticks later, half of the time
    for t in range(ticks):                                 # babbling in quiet; only the reverse links learn
        ad = pending.pop(t, None)
        a0 = life.coch.a0(([] if own is None else [(own, -1, 'own', 0.0, 0.0)]) +
                          ([] if ad is None else [(ad[0], -1, ad[1], 0.0, 0.0)]))
        mus = b.step(torch.as_tensor(life.with_sight(a0), device=DEV), 0.0).cpu().numpy()
        mot.rev.learn(mot.force.on, cc.below, bool(mot.braked.any()))
        if (t + 1) % life.c['decay_every'] == 0:
            mot.rev.decay()
        y = life.mouth.say(mus)
        own = y if (y is not None and y != 'garble') else None
        if own is not None and erng.random() < echo:
            pending[t + 2] = (own, ADULTS[int(erng.integers(len(ADULTS)))])
        if y is not None:
            said[y] += 1
            if not mot.babbled:                            # said by the reverse path, not babbled
                driven['garble' if y == 'garble' else 'syllable'] += 1
        if (t + 1) % 5000 == 0:
            print('BABBLING', t + 1, 'said by the reverse path so far:', dict(driven), flush=True)
    print('BABBLED', json.dumps(dict(ticks=ticks, utterances=sum(said.values()), distinct=len(said),
                                     garbles=said.get('garble', 0), by_reverse_path=dict(driven), W_max=round(float(mot.rev.W.max()), 3),
                                     W_min=round(float(mot.rev.W.min()), 3), lessons=mot.rev.n_learn)), flush=True)
    mot.p_babble = 0.0                                     # tests: no babbling
    sids = list(life.speak_sids)
    rng = np.random.default_rng(4)
    test = [int(x) for x in rng.choice(sids, n, replace=False)]
    code, _, _ = concept_codes(life, test)
    _, _, ca, cb = b.chains.seg('concept')[0]
    snap = b.snapshot()

    def run(thought=None, heard=None, brake=False, T=6):
        b.restore(snap)
        out, own_ = [], None
        for t in range(T):
            p.force_int = thought if (thought is not None and t == 0) else None
            hl = [] if heard is None else (heard if isinstance(heard, list) else [heard])
            src = ([] if t >= len(hl) else [(hl[t], -1, ADULTS[0], 0.0, 0.0)]) + \
                  ([(sids[0], -1, ADULTS[1], 0.0, 0.0)] if brake else []) + ([] if own_ is None else [(own_, -1, 'own', 0.0, 0.0)])
            mus = b.step(torch.as_tensor(life.with_sight(life.coch.a0(src)), device=DEV), 0.0).cpu().numpy()
            y = life.mouth.say(mus)
            out.append(y)
            own_ = y if (y is not None and y != 'garble') else None
        p.force_int = None
        b.restore(snap)
        return out

    def tally(name, cases):
        k = collections.Counter()
        for out, want in cases:
            s = [y for y in out if y is not None]
            k['nothing' if not s else 'garble' if s[0] == 'garble' else 'right' if s[0] == want else 'other'] += 1
        print('CHECK', name, json.dumps({kk: round(v / len(cases), 3) for kk, v in sorted(k.items())}), flush=True)

    def thought(i, share=1.0, k=None, m=None):
        m = torch.zeros(p.n, dtype=torch.bool, device=DEV) if m is None else m
        idx = torch.nonzero(code[i]).squeeze(1)
        if k is not None:                                  # k random units of its code (a thought is ~8 units)
            idx = idx[torch.as_tensor(rng.permutation(idx.numel())[:k], device=DEV)]
        m[ca + idx[:max(1, int(round(share * idx.numel())))]] = True
        return m
    P = mot.babble_pat                                     # [syllables, force neurons]
    sid_row = {s_: r_ for r_, s_ in enumerate(sids)}

    def diag(i, s_):
        want = P[sid_row[s_]] > 0
        x = code[i]
        push = (x.float() @ mot.rev.W)                     # in units of a fully predicted movement
        b.restore(snap)
        p.force_int = thought(i)
        b.step(torch.as_tensor(life.with_sight(life.coch.a0([])), device=DEV), 0.0)
        p.force_int = None
        rc = b.chains.ret_concept > 0
        pr = rc.float() @ mot.rev.W
        b.step(torch.as_tensor(life.with_sight(life.coch.a0([])), device=DEV), 0.0)
        f = mot.force.on.clone()
        b.restore(snap)
        r3 = lambda v: round(float(v), 2)
        b.restore(snap)                                    # the own voice of it: concept units lit from below
        heard_o, pushes = torch.zeros_like(x), []
        for t in range(5):
            b.step(torch.as_tensor(life.with_sight(life.coch.a0([(s_, -1, 'own', 0.0, 0.0)] if t == 0 else [])),
                                   device=DEV), 0.0)
            bl = cc.below if cc.below is not None else torch.zeros_like(x)
            heard_o |= bl
            pushes.append(r3((bl.float() @ mot.rev.W)[want].mean()))
        b.restore(snap)
        jac = float((heard_o & x).sum()) / max(1, float((heard_o | x).sum()))
        return dict(own_voice_units=int(heard_o.sum()), own_vs_code_jaccard=round(jac, 2), own_push_by_tick=pushes,
                    code=int(x.sum()), rc=int(rc.sum()), rc_in_code=int((rc & x).sum()),
                    heard_own=[r3(push[want].mean()), r3(push[want].min()), r3(push[~want].max())],
                    from_return=[r3(pr[want].mean()), r3(pr[want].min()), r3(pr[~want].max())],
                    fired_own=int((f & want).sum()), fired_other=int((f & ~want).sum()), own_n=int(want.sum()))
    for i in range(8):
        print('DIAG', test[i], json.dumps(diag(i, test[i])), flush=True)
    for sh in (1.0, 0.5, 0.25):
        tally(f'think a syllable ({sh:.0%} of its code)', [(run(thought(i, sh)), s) for i, s in enumerate(test)])
    tally('think 8 random units of its code', [(run(thought(i, k=8)), s) for i, s in enumerate(test)])
    for ka, kb in ((4, 4), (6, 2)):                        # two syllables in one thought: one of them or nothing
        k = collections.Counter()
        for i in range(n):
            j = (i + 1) % n
            out = [y for y in run(thought(j, k=kb, m=thought(i, k=ka))) if y is not None]
            k['nothing' if not out else 'garble' if out[0] == 'garble' else 'first' if out[0] == test[i] else
              'second' if out[0] == test[j] else 'other'] += 1
        print('CHECK', f'think {ka} units of one + {kb} of another', json.dumps({kk: round(v / n, 3) for kk, v in sorted(k.items())}), flush=True)
    tally('hear a syllable (must not repeat it)', [(run(heard=s), -1) for s in test])
    tally('think it while an adult speaks (brake)', [(run(thought(i), brake=True), s) for i, s in enumerate(test)])
    k = collections.Counter()                              # a heard line of 6 syllables, then 10 quiet ticks
    for j in range(n):
        line = [int(x) for x in rng.choice(sids, 6)]
        out = run(heard=line, T=16)
        for y in out[6:]:
            k['nothing' if y is None else 'garble' if y == 'garble' else 'syllable'] += 1
    from ..world import phon
    from ..world.corpus import script_dialogue             # never-heard dialogues, frozen, the child's own voice
    bases = phon.load_inventory()
    nm = lambda z: phon.toned_name(bases, int(z))
    kq, said_d = collections.Counter(), []                 # heard back; what it says in the pauses
    gt = tuple(life.c.get('gap_turn', (4, 10)))
    for di in range(min(dialogues, life.heldout.n)):
        src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9311, di]), gt, 8, 1, 1)
        src = [None if x is None else x[:5] for x in src]
        b.restore(snap)
        own_, last = None, []
        for t, x in enumerate(src):
            a0 = life.coch.a0(([] if x is None else [x]) + ([] if own_ is None else [(own_, -1, 'own', 0.0, 0.0)]))
            mus = b.step(torch.as_tensor(life.with_sight(a0), device=DEV), 0.0).cpu().numpy()
            y = life.mouth.say(mus)
            own_ = y if (y is not None and y != 'garble') else None
            if x is not None:
                last = (last + [int(x[0])])[-4:]
            else:
                kq['nothing' if y is None else 'garble' if y == 'garble' else 'syllable'] += 1
                if y is not None and y != 'garble':
                    said_d.append((' '.join(nm(z) for z in last), nm(y)))
        b.restore(snap)
    tq = max(1, sum(kq.values()))
    print('CHECK never-heard dialogues, pauses (per tick)', json.dumps({kk: round(v / tq, 3) for kk, v in sorted(kq.items())}),
          'said (after what):', said_d[:20], flush=True)
    print('CHECK quiet after a heard line (per tick)', json.dumps({kk: round(v / (10 * n), 3) for kk, v in sorted(k.items())}), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], ticks=int(kw.get('ticks', 20000)), n=int(kw.get('n', 40)), step=float(kw.get('step', 0.2)),
         pb=float(kw.get('pb', 0.3)), norm=bool(int(kw.get('norm', 0))), dialogues=int(kw.get('dialogues', 20)),
         echo=float(kw.get('echo', 0.5)))
