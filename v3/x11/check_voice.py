"""World check (no brain): are the ~1,200 toned syllables of the corpus distinguishable at the input layer A0?
Nearest-centroid on the 3-tick A0 pattern, leave-one-voice-out (centroids from 3 adult voices, tested on the 4th),
with per-turn F0 offset and per-tick jitter / noise as in the running world. Reports toned / toneless / tone accuracy
and what the errors share (same base, same final, same initial).
usage: python3 check_voice.py [--db 35] [--levels 4] [--reps 2] [--stats data/stats_valid.json]"""
import argparse
import json
import os

import numpy as np

import phon
import voice as V

HERE = os.path.dirname(os.path.abspath(__file__))


def cfg(levels=4):
    return dict(levels_per_s=levels, s=1, th_lo=0.12, th_hi=1.0, noise_sd=0.04, e_max=1.5, own_level=0.6,
                lineage_seed=4321)


def used_ids(stats_path, min_count=1):
    st = json.load(open(stats_path))
    cnt = np.array(st['syl_count'])
    return np.nonzero(cnt >= min_count)[0], cnt


def tokens(co, sids, reps, seed=0):
    """-> X bool [n_tok, 3 * n_a0], labels (sid), voice index"""
    rng = np.random.default_rng(seed)
    X, lab, vox = [], [], []
    for vi, v in enumerate(V.ADULTS):
        for r in range(reps):
            for s in sids:
                off = rng.uniform(-1, 1)
                X.append(np.concatenate([co.a0([(int(s), k, v, off, 0.0)]) for k in range(3)]))
                lab.append(s)
                vox.append(vi)
    return np.array(X), np.array(lab), np.array(vox)


def nearest_centroid(X, lab, vox, sids, folds=None):
    """folds: array of fold ids (default: the voice = leave one voice out); returns predicted labels"""
    Xf = X.astype(np.float32) * 2 - 1
    pred = np.zeros(len(lab), np.int64)
    pos = {s: i for i, s in enumerate(sids)}
    li = np.array([pos[s] for s in lab])
    folds = vox if folds is None else folds
    for v in np.unique(folds):
        tr, te = folds != v, folds == v
        C = np.zeros((len(sids), X.shape[1]), np.float32)
        np.add.at(C, li[tr], Xf[tr])
        C /= np.maximum(np.bincount(li[tr], minlength=len(sids))[:, None], 1)
        S = Xf[te] @ C.T - 0.5 * (C * C).sum(1)[None]
        pred[te] = sids[S.argmax(1)]
    return pred


def summarise(lab, pred, bases):
    parts = [phon.parse(b) for b in bases]
    ok = pred == lab
    base_ok = pred // 5 == lab // 5
    tone_ok = pred % 5 == lab % 5
    err = ~ok
    same_fin = np.array([(parts[a // 5]['medial'], parts[a // 5]['nucleus'], parts[a // 5]['coda']) ==
                         (parts[b // 5]['medial'], parts[b // 5]['nucleus'], parts[b // 5]['coda'])
                         for a, b in zip(lab[err], pred[err])])
    same_ini = np.array([parts[a // 5]['initial'] == parts[b // 5]['initial'] for a, b in zip(lab[err], pred[err])])
    return dict(toned=float(ok.mean()), toneless=float(base_ok.mean()), tone=float(tone_ok.mean()),
                errors=int(err.sum()), err_same_base=float(base_ok[err].mean()) if err.any() else None,
                err_same_final=float(same_fin.mean()) if err.any() else None,
                err_same_initial=float(same_ini.mean()) if err.any() else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=float, default=30.0)
    ap.add_argument('--levels', type=int, default=4)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--stats', default=os.path.join(HERE, 'data', 'stats_valid.json'))
    a = ap.parse_args()
    bases = phon.load_inventory()
    sids, cnt = used_ids(a.stats)
    v = V.Voice(bases)
    co = V.Cochlea(v, cfg(a.levels), 0, db_range=a.db)
    X, lab, vox = tokens(co, sids, a.reps)
    rep = np.arange(len(lab)) // len(sids) % a.reps                      # token repetition within a voice
    out = dict(n_types=len(sids), db=a.db, levels=a.levels, a0_bits=co.n)
    w = cnt[lab] / cnt[lab].sum()
    for mode, folds in (('same_voices_new_tokens', rep), ('new_voice', vox)):
        pred = nearest_centroid(X, lab, vox, sids, folds)
        r = summarise(lab, pred, bases)
        r['toned_freq_weighted'] = float((w * (pred == lab)).sum())
        out[mode] = r
    print(json.dumps(out))
