"""World / sensory check on the GPU (no learning at any point): build the newborn (birth period, learning off), report
the chain bands and sizes (whole-brain rules of X10), then syllable confusion at A0, after each sound layer A1-A4 and
in PFC L1's chained sound copies (rises after onset, as the codes), by nearest centroid: same voices / new tokens and
new voice. Also the step time with learning off.
usage: python3 check_sound.py --out res_wcheck.json [--ctx 3] [--s 1]"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import DEV, log, save_json
from life11 import Life11, load_bases, x11_config
from probes11 import chain_rises, codes, n_sound_copies, sound_states, token_stream
from world11 import Dialogues

HERE = os.path.dirname(os.path.abspath(__file__))


def nc_eval(F, lab, folds, sids):
    """nearest centroid on +-1 features (GPU); returns predicted sids"""
    pos = {s: i for i, s in enumerate(sids)}
    li = torch.as_tensor([pos[s] for s in lab], device=DEV)
    X = F.float() * 2 - 1
    pred = torch.zeros(len(lab), dtype=torch.int64, device=DEV)
    fo = torch.as_tensor(folds, device=DEV)
    sid_t = torch.as_tensor(np.asarray(sids), device=DEV)
    for f in torch.unique(fo):
        tr, te = fo != f, fo == f
        C = torch.zeros((len(sids), X.shape[1]), device=DEV).index_add_(0, li[tr], X[tr])
        n = torch.zeros(len(sids), device=DEV).index_add_(0, li[tr], torch.ones(int(tr.sum()), device=DEV))
        C = C / n.clamp_min(1)[:, None]
        idx = torch.nonzero(te).squeeze(1)
        for a in range(0, len(idx), 2048):
            ii = idx[a:a + 2048]
            S = X[ii] @ C.T - 0.5 * (C * C).sum(1)[None]
            pred[ii] = sid_t[S.argmax(1)]
    return pred.cpu().numpy()


def acc(lab, pred):
    return dict(toned=float((lab == pred).mean()), toneless=float((lab // 5 == pred // 5).mean()),
                tone=float((lab % 5 == pred % 5).mean()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='res_wcheck.json')
    ap.add_argument('--ctx', type=int, default=3)
    ap.add_argument('--s', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--timing_ticks', type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    bases = load_bases(HERE)
    st = json.load(open(os.path.join(HERE, 'data', 'stats_valid.json')))
    cnt = np.array(st['syl_count'], np.float64)
    sids = np.nonzero(cnt > 0)[0]
    birth = Dialogues.load(os.path.join(HERE, 'data', 'birth.npz'))
    train = Dialogues.load(os.path.join(HERE, 'data', 'heldout.npz'))      # only for the timing run (learning off)
    c = x11_config(a.s)
    life = Life11(c, a.seed, train, birth, bases)
    b = life.brain
    assert not b.learning
    res = dict(config={k: v for k, v in c.items() if not isinstance(v, (list, tuple)) or len(v) < 10},
               chain_report=b.chain_report, n1=b.n1, n_sound_copies=n_sound_copies(b), pfc_n=b.pfc.n,
               hip_n=b.hip.nh, a0_bits=life.coch.n, sound_units=b.sound.n, calib=b.calib,
               birth_ticks=life.birth_ticks, minutes_build=(time.time() - t0) / 60)
    log('built', res['n1'], res['n_sound_copies'], res['sound_units'])
    # ---- confusion: A0, A1..A4, L1 chained sound copies (rises after onset)
    A0, lab, vox, rep = token_stream(life, sids, cnt, a.ctx, 991)
    states = sound_states(b, A0)
    _, R = chain_rises(b, states)
    T0 = torch.arange(len(lab), device=DEV) * 9 + 3
    feats = {'A0': torch.cat([A0[T0 + k] for k in range(3)], 1)}
    for l, stl in enumerate(states):
        o = 0 if l == 0 else l                                    # A1 window t0..t0+2, Al window t0+l..t0+l+2
        feats[f'A{l + 1}'] = torch.cat([stl[T0 + o + k] for k in range(3)], 1)
    feats['L1_sound_rises'] = torch.cat([R[T0 + 1 + k] for k in range(3)], 1)
    conf = {}
    for nm, F in feats.items():
        conf[nm] = dict(dims=int(F.shape[1]), on_frac=float(F.float().mean()),
                        same_voices=acc(lab, nc_eval(F, lab, rep, sids)),
                        new_voice=acc(lab, nc_eval(F, lab, vox, sids)))
        log(nm, conf[nm])
    res['confusion'] = conf
    res['confusion_tokens'] = dict(types=len(sids), voices=4, ctx=a.ctx)
    del feats, states, R, A0
    cd = codes(life, sids, cnt, n_ctx=2)
    res['code_size3'] = cd['mean_code_size3']
    # ---- step time, learning off (overhearing world)
    torch.cuda.synchronize()
    ts = time.time()
    for _ in range(a.timing_ticks):
        life.tick()
    torch.cuda.synchronize()
    res['ms_per_tick_learning_off'] = 1000 * (time.time() - ts) / a.timing_ticks
    res['minutes'] = (time.time() - t0) / 60
    res['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
    save_json(a.out, res)
    log('done', res['ms_per_tick_learning_off'], res['minutes'])
