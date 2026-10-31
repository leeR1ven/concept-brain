# -*- coding: utf-8 -*-
"""P5.1: one agent, fp32 — the torch brain against the recorded numpy reference trace (p0_trace.py).

Open loop: the observation stream recorded from the numpy reference (ear window, eye, reward per tick, from birth
through stage A, babbling, stage B) is fed to the torch brain (B = 1, exact mode); every tick its outputs are compared
with the reference's: mouth force (32 muscles), GO, released program, stop reason, the open thought (kind, step,
confidence, intention, released), the MIX unit count, the quiet counter; plus every detection and every awake-recall
pick.  Open-loop replay is exact as long as the outputs are identical (the world would have produced the same stream).
usage: python3 test_trace.py trace=out/p0_trace_s0_canon.npz [ticks=0] [device=cuda]"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                   # noqa: E402
from brain_torch import Brain, Cfg             # noqa: E402


def load_init(Z, seed=None):
    if seed is None:
        seed = int(Z['seed']) if 'seed' in Z else 0
    d = dict(F0=Z['init_F0'], ear_idx=Z['init_ear_idx'], ear_w=Z['init_ear_w'], ear_b=Z['init_ear_b'],
             ear_alive=Z['init_ear_alive'], hip_tie=Z['init_hip_tie'], seed=seed, n_eye=Z['eye'].shape[1] * 8)
    for g in ('bag', 'first', 'last', 'scene'):
        d['hip_idx_' + g] = Z['init_hip_idx_' + g]
    return d


def main():
    A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    path = A.get('trace', os.path.join(HERE, 'out', 'p0_trace_s0_canon.npz'))
    dev = A.get('device', 'cuda')
    Z = dict(np.load(path))
    meta = os.path.splitext(path)[0] + '.json'
    if os.path.exists(meta):
        Z['seed'] = json.load(open(meta))['seed']
    T = len(Z['force']) if not int(A.get('ticks', 0)) else int(A['ticks'])
    tA = int(Z['tA'])
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.3)
    br = Brain(1, [load_init(Z)], Cfg(U=1 << 16, logM=18, fused=bool(int(A.get('fused', 0)))), device=dev)
    br.trace = dict(dets=[], picks=[], learns=[])
    ref_learns = Z['learns']
    nl = 0
    if A.get('debug'):
        br.debug_t = set(int(x) for x in A['debug'].split(','))
    prof = {}
    if int(A.get('prof', 0)):
        import functools
        def wrap(name):
            f = getattr(br, name)
            @functools.wraps(f)
            def g(*a, **kw):
                t = time.perf_counter()
                r = f(*a, **kw)
                p = prof.setdefault(name, [0, 0.0]); p[0] += 1; p[1] += time.perf_counter() - t
                return r
            setattr(br, name, g)
        for nm in ('_drives', '_learn_step', '_confidence', '_recall_next', '_dg', '_detect', '_ear', '_zmatch', '_bind',
                   '_match', '_recognize', '_attd', '_awake_recall', '_replay', '_body', '_parse_eye', '_intend', '_store',
                   '_begin_line', '_line_end', '_line_item', '_open', '_think_step', '_scorer', '_view', '_ct_input',
                   '_pend_finalize', '_cands', '_vals', 'step'):
            wrap(nm)
    eye_all = np.unpackbits(Z['eye'], axis=1).astype(bool)
    fields = ('force', 'go', 'y', 'stop', 'th_kind', 'th_k', 'th_e', 'th_I', 'th_rel', 'mix', 'quiet', 'inline')
    first_bad = {k: None for k in fields}
    nbad = {k: 0 for k in fields}
    t0 = time.time()
    for t in range(T):
        if t == tA:
            pats = np.unpackbits(Z['bab_pat'], axis=1)[:, :br.NF].astype(bool)
            br.load_babble(np.array([0]), pats, Z['bab_set'])
        f = br.step(Z['ear'][t][None].astype(np.int64), eye_all[t][None], np.array([Z['rew'][t]]))
        mine = dict(force=f[0], go=int(br.bGO[0]), y=int(br.bY[0]), stop=int(br.bstop[0]),
                    th_kind=(int(br.th_kind[0]) + 1) if br.th_on[0] else 0, th_k=int(br.th_k[0]) if br.th_on[0] else -1,
                    th_e=float(br.th_e[0]) if br.th_on[0] else -1.0, th_I=float(br.th_I[0]) if br.th_on[0] else -1.0,
                    th_rel=int(br.th_rel[0]) if br.th_on[0] else 0, mix=int(br.conj.n[0].item()),
                    quiet=int(br.quiet[0]), inline=int(br.in_line[0]))
        for k in fields:
            ref = Z[k][t]
            ok = np.array_equal(np.asarray(mine[k]), ref) if k == 'force' else (mine[k] == ref)
            if not ok:
                nbad[k] += 1
                if first_bad[k] is None:
                    first_bad[k] = (t, str(mine[k]), str(ref))
        while nl < len(br.trace['learns']):
            mine_l = br.trace['learns'][nl]
            ref_l = ref_learns[nl]
            if (mine_l[1], mine_l[2]) != (int(ref_l[1]), int(ref_l[2])):
                print('FIRST LEARN MISMATCH #%d tick %d: torch (nA %d sumA %d target %d) ref (t %d nA %d sumA %d target %d)'
                      % (nl, t, mine_l[1], mine_l[2], mine_l[3], ref_l[0], ref_l[1], ref_l[2], ref_l[3]), flush=True)
                print('TORCH KEYS', mine_l[4], flush=True)
                T = t + 1
                break
            nl += 1
        if T == t + 1:
            break
        if t % 5000 == 0:
            print(f't {t} {time.time() - t0:.0f}s bad {nbad}', flush=True)
    secs = time.time() - t0
    dets_ref = [tuple(x) for x in Z['dets'] if x[0] < T]
    dets = br.trace['dets']
    nd = min(len(dets), len(dets_ref))
    det_first = next((i for i in range(nd) if tuple(dets[i]) != dets_ref[i]), None)
    pk_ref = [(int(t), [int(x) for x in e if x >= 0]) for t, e in zip(Z['pick_t'], Z['pick_e']) if t < T]
    pk = br.trace['picks']
    npk = min(len(pk), len(pk_ref))
    pk_first = next((i for i in range(npk) if pk[i] != pk_ref[i]), None)
    sp_ref = int((Z['force'][:T].sum(1) > 0).sum())
    res = dict(trace=os.path.basename(path), ticks=T, tA=tA, secs=round(secs, 1), ms_per_tick=round(1000 * secs / T, 3),
               first_mismatch=first_bad, n_mismatch=nbad, dets=[len(dets), len(dets_ref)], det_first_mismatch=det_first,
               picks=[len(pk), len(pk_ref)], pick_first_mismatch=pk_first, spoken_ticks_ref=sp_ref,
               identical=all(v == 0 for v in nbad.values()) and det_first is None and pk_first is None
               and len(dets) == len(dets_ref) and len(pk) == len(pk_ref),
               stats=br.stats, units=int(br.conj.n[0].item()), hip=int(br.hn[0]), Z=int(br.zn[0]),
               progs=int(br.n_prog[0]),
               prof={k: [v[0], round(v[1], 2)] for k, v in sorted(prof.items(), key=lambda x: -x[1][1])})
    print(json.dumps(res, default=str))


if __name__ == '__main__':
    main()
