"""Stage (b2) pieces shared by run_b2.py and diag_b2.py: one practice trial with the caregiver's reaction, and the
say test (b1 form, unmuted, frozen)."""
import numpy as np
import torch

from common import DEV
import probes
from probes import Frozen, code_mask
from world_adapter import SYL


def practice_trial(life, rng, x, voice, react=True, W=16):
    """one practice trial (learning on). Returns the outcome and the reaction given."""
    b, env = life.brain, life.probe_env
    script = [(x, voice, 1.0)] * 3 + [None] * W
    m = np.zeros(24, bool)
    env.reset_track()
    reacted, said_x, outcome = None, False, 'silent'
    for t, adult in enumerate(script):
        a0 = env.step(m, adult=adult)
        heard = env.ph.decode(m) if m.any() else None          # what the caregiver hears now (the last tick's mouth)
        r = 0.0
        if t >= 3 and reacted is None and heard is not None:
            if heard[0] == 'g' or (heard[0] in ('s', 'm') and heard[1] != x):
                outcome = 'garble' if heard[0] == 'g' else 'wrong'
                if said_x:
                    outcome = 'x then ' + outcome
                r, reacted = (-1.0 if react else 0.0), 'sad'
            elif heard[0] == 's' and heard[1] == x:
                said_x, outcome = True, 'x'
            elif heard[0] == 'm':
                outcome = 'mumble x'
        elif t >= 3 and reacted is None and heard is None and said_x:
            r, reacted = (1.0 if react else 0.0), 'happy'
        m = b.step(torch.as_tensor(a0, device=DEV), r).cpu().numpy()
    gap = int(rng.integers(8, 15))
    for _ in range(gap):
        a0 = env.step(m)
        m = b.step(torch.as_tensor(a0, device=DEV), 0.0).cpu().numpy()
    return outcome, reacted


def say_test(life, syls, T=22):
    """frozen: x's code lit in PFC L1 on ticks 6-8 (as b1), unmuted; per syllable the decoded mouth per tick"""
    b, ph = life.brain, life.ph
    rows = []
    with Frozen(life) as snap:
        for x in syls:
            if life.codes[x].numel() == 0:
                continue
            b.restore(snap)
            decs = []
            probes.play(life, [None] * T, force={6 + k: code_mask(life, [x]) for k in range(3)},
                        rec=lambda lf, t: decs.append(ph.decode(lf.last_mouth)))
            said = [d for d in decs if d is not None]
            first = next((i for i, d in enumerate(decs) if d is not None and d[0] == 's' and d[1] == x), None)
            tail = (sum(d is not None for d in decs[first + 1:]) - sum(d is not None and d[0] == 's' and d[1] == x
                                                                      for d in decs[first + 1:])) if first is not None else None
            keys, last = [], None                                 # utterance list as the world tracks it
            for d in decs:
                k = None if d is None else (('s' if d[0] == 'm' else d[0]), d[1])
                if k is not None and k != last:
                    keys.append('~' if d[0] == 'g' else SYL[d[1]])
                last = k
            stop = first is not None and all(d is None or (d[0] == 's' and d[1] == x) for d in decs[first:first + 4]) \
                and all(d is None for d in decs[first + 4:])
            rows.append(dict(said=keys == [SYL[x]], any_x=first is not None, garble=any(d[0] == 'g' for d in said),
                             tail=tail, stop=bool(stop), silent=not said))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ('said', 'any_x', 'garble', 'stop', 'silent')}
    tails = [r['tail'] for r in rows if r['tail'] is not None]
    agg['tail_ticks_after_x'] = float(np.mean(tails)) if tails else None
    agg['n'] = len(rows)
    return agg
