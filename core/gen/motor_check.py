"""Injection checks of the motor area's settling (Fable a39 A), frozen on a saved life with speech wired:
force copies lit as a tendency would light them (per channel, the lowest grades up to strength x m), the motor area
steps once in silence, the mouth reads the muscles. usage: python -m core.gen.motor_check <ckpt> [settle=1]"""
import json
import sys

import numpy as np
import torch


def main(ckpt, settle=1):
    import core.corpus_life  # noqa: F401
    from ..net import DEV
    from ..speak import patterns72
    from ..speak2 import install_settle
    life = torch.load(ckpt, weights_only=False)['life']
    b, mot = life.brain, life.brain.motor
    if settle:
        install_settle(life)
    sids = list(life.speak_sids)
    pat = torch.as_tensor(patterns72(life.body)[sids], dtype=torch.bool, device=DEV)[:, mot.muscle_of]   # [nS, nf]
    m = mot.m
    grade = (torch.arange(mot.nf, device=DEV) % m).float()
    a0 = torch.as_tensor(life.coch.a0([]), device=DEV)
    no_stop = torch.zeros(mot.ns, dtype=torch.bool, device=DEV)
    rng = np.random.default_rng(3)

    def say(level, stop=no_stop):
        """level: float [nf] the push per force copy's channel (0..1) -> what the mouth says"""
        lit = grade < (level * m).round()
        mus = mot.step(a0, lit.float(), stop)
        x = life.mouth.say(mus.cpu().numpy())
        return 'silent' if x is None else ('garble' if x == 'garble' else int(x))

    def tally(outs, want):
        n = len(outs)
        return dict(right=round(sum(o == w for o, w in zip(outs, want)) / n, 3),
                    other=round(sum(o not in ('silent', 'garble') and o != w for o, w in zip(outs, want)) / n, 3),
                    garble=round(sum(o == 'garble' for o in outs) / n, 3), silent=round(sum(o == 'silent' for o in outs) / n, 3))
    S = rng.choice(len(sids), 60, replace=False)
    out = {}
    for a in (0.2, 0.4, 0.6, 0.8, 1.0):                   # one syllable at strength a
        outs = [say(pat[i].float() * a) for i in S]
        out[f'single {a}'] = tally(outs, [sids[i] for i in S])
    for a in (0.5, 0.6, 0.7, 0.8):                        # a mixture of two
        outs, want = [], []
        for i in S:
            j = int(rng.integers(len(sids)))
            lv = (pat[i].float() * a + pat[j].float() * (1 - a)).clamp(max=1.0)
            outs.append(say(lv))
            want.append(sids[i])
        out[f'mix {a}/{1 - a:.1f}'] = tally(outs, want)
    sub = [(i, j) for i in range(len(sids)) for j in range(len(sids)) if i != j and bool((pat[i] <= pat[j]).all())][:60]
    if sub:                                               # s1's channels inside s2's: the injected one should win
        out['subset: the smaller injected'] = tally([say(pat[i].float()) for i, j in sub], [sids[i] for i, j in sub])
        out['subset: the larger injected'] = tally([say(pat[j].float()) for i, j in sub], [sids[j] for i, j in sub])
    outs = []
    for _ in range(40):                                   # a flat mixture of ten
        J = rng.choice(len(sids), 10, replace=False)
        outs.append(say(pat[J].float().sum(0).clamp(max=1.0) * 0.5))
    out['flat ten'] = tally(outs, [None] * len(outs))
    outs = [say(pat[i].float(), torch.ones(mot.ns, dtype=torch.bool, device=DEV)) for i in S]   # the brake on
    out['brake on'] = tally(outs, [sids[i] for i in S])
    print('MOTOR', 'settle' if settle else 'old pools', json.dumps(out), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], settle=int(kw.get('settle', 1)))
