"""Gain-free readout margin: for each syllable, (weakest own muscle drive) / (strongest other muscle drive).
The X1c competition (release if >= 0.6 x max) needs >= ~1.67 for a clean syllable."""
import numpy as np
import babble
from world import Physics, NSYL

def margins(ph, W):
    D = ph.content.astype(np.float32) @ W[:ph.n_c]
    m = []
    for s in range(NSYL):
        own = D[s, ph.need_mat[s]]; oth = D[s, ~ph.need_mat[s]]
        m.append(own.min() / max(oth.max(), 1e-9))
    return np.array(m)

for phys in ('x1c', 'lawful'):
    ph = Physics() if phys == 'x1c' else babble.LawfulPhysics()
    S = babble.standin(ph)
    ms = margins(ph, S)
    print(phys, 'stand-in margin median %.2f, share>=1.67 %.2f, share>=1 %.2f' % (np.median(ms), (ms >= 1.67).mean(), (ms >= 1).mean()))
    for arm in ('same', 'redup'):
        for ratio in (1.0, 0.5):
            # rerun one life, keep W
            import types
            out = {}
            orig = babble.probes
            def grab(ph_, W, seed, mass_c, _o=out):
                _o['W'] = W.copy(); return orig(ph_, W, seed, mass_c)
            babble.probes = grab
            babble.life(arm, 0, n_babble=2000, ratio=ratio, checkpoints=(2000,), phys=phys)
            babble.probes = orig
            mm = margins(ph, out['W'])
            print(f"  {arm} r={ratio}: learned margin median {np.median(mm):.2f}, share>=1.67 {(mm >= 1.67).mean():.2f}, share>=1 {(mm >= 1).mean():.2f}")
