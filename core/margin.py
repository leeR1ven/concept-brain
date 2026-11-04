"""How strong may the last layer's own links be before its state stops following what is heard? (For the Hebbian
readout to equal a trained readout, the state must be set mostly by the recent input: the echo state property of a
fixed random network with a trained readout, Jaeger 2001.) Measured per tick while dialogues are heard (nothing
learned unless learn=n dialogues first, to let a newborn's usual levels settle):
  sd_in     spread over last-layer units of the input push above each unit's usual level (what ranks the units)
  gap       the input push of the lit units minus the cut (how far in they are)
  own_lit   the push the lit units get from the last layer's own links (from the units lit a tick before)
  own_all   the same over all units (its spread)
usage: python -u -m core.margin <ckpt or 'newborn'> [learn=n] [key=value]"""
import json
import sys

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife


def main():
    src = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    learn = over.pop('learn', 0)
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    b, p = life.brain, life.brain.pfc
    if learn:
        b.learning = True
        n0 = life.env.n_ex
        while life.env.n_ex - n0 < learn:
            life.tick()
    b.learning = False
    rows = []
    n0 = life.env.n_ex
    while life.env.n_ex - n0 < 3:
        life.tick()
        lit = p.L4.on
        if lit.sum() < 50:
            continue
        ff = p.ff[2].drive(p.L3.prev) - p.usual4
        own = p.L44.drive(p.L4.prev)
        cut = ff[lit].min()
        rows.append([float(ff.std()), float((ff[lit] - cut).mean()), float(own[lit].mean()), float(own.std())])
    R = np.array(rows).mean(0)
    print('MARGIN', src, json.dumps(dict(over=over, learn=learn, sd_in=round(R[0], 3), gap=round(R[1], 3),
                                         own_lit=round(R[2], 3), own_sd=round(R[3], 3), g44=p.L44.gain, cap=getattr(p.L44, 'cap', None),
                                         w_max=round(float(p.L44.w.max()), 2), ticks=len(rows))), flush=True)


if __name__ == '__main__':
    main()
