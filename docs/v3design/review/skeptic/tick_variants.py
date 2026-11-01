"""Skeptic toy, follow-up: tick-level O1 collapse with and without forced silence, format (tau_f 10) vs model tutor
(other2 answers at tau 3).  Run: python3 -I tick_variants.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tick_toy import run
U = ('free', 3000, [300, 900])
for tutor in ('format', 'model'):
    for ps in (0.0, 0.3):
        for decide in ('thr', 'acc'):
            rows = [run(s, [(tutor, 600), U], mode='tick', decide=decide, p_sil=ps) for s in range(5)]
            ks = [f'{tutor}600', 'free300', 'free900', 'free3000']
            print(f'tick-level, {tutor}, p_sil {ps}, {decide}: ' + '  '.join(f'{k} {np.mean([r[k] for r in rows]):.2f}' for k in ks)
                  + f"  | live free (right, silent, partner) {tuple(np.round(np.mean([r['free3000_live'] for r in rows], 0), 2))}",
                  flush=True)
