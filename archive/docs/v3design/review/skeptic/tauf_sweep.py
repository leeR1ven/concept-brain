"""Skeptic toy: is the anchored context's stability a property of the core, or of W1's 10-tick format delay sitting two
octaves before the bench's 45-tick give-up?  Event-level O1 (the reading under which drift_toy's ST is stable), both
decision rules, caregiver self-answer delay tau_f swept; give-up stays at Q_end + 45 (world.py gap_answer).
Run: python3 -I tauf_sweep.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tick_toy import run
U = ('free', 3000, [300, 900])
for decide in ('thr', 'acc'):
    for tf in (10, 16, 20, 24, 30):
        rows = [run(s, [('format', 600), U], mode='event', decide=decide, tau_f=tf) for s in range(5)]
        ks = ['format600', 'free300', 'free900', 'free3000']
        print(f'event-level, {decide}, tau_f {tf:2d}: ' + '  '.join(f'{k} {np.mean([r[k] for r in rows]):.2f}' for k in ks)
              + f"  | live free (right, silent, partner) {tuple(np.round(np.mean([r['free3000_live'] for r in rows], 0), 2))}",
              flush=True)
