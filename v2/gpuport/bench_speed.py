# -*- coding: utf-8 -*-
"""P5.3: speed and memory of the batched brain at B = 1 ... 128 agents.

One agent is grown through the recorded reference stream (stage A, babbling, warm stage B); the brain is then tiled to
B copies and every copy is fed the recorded stage-B observation stream from its own offset (open loop), so the agents'
events (hearing, thinking, speaking, awake recall) are desynchronised as in a population.  Measures agent-ticks per
second of the brain alone (no world), per-event counts, and peak GPU memory.
usage: python3 bench_speed.py [Bs=1,4,16,64] [ticks=600] [warm=1500]"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                   # noqa: E402
from brain_torch import Brain, Cfg, tile       # noqa: E402
from test_trace import load_init               # noqa: E402


def main():
    A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    Bs = [int(x) for x in A.get('Bs', '1,4,16,64').split(',')]
    N = int(A.get('ticks', 600)); warm = int(A.get('warm', 1500))
    Z = dict(np.load(os.path.join(HERE, 'out', 'p0_trace_s0_canon.npz')))
    torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.3)))
    tA = int(Z['tA']); T = len(Z['force'])
    eye_all = np.unpackbits(Z['eye'], axis=1).astype(bool)
    win_all = Z['ear'].astype(np.int64); rew_all = Z['rew']
    br = Brain(1, [load_init(Z)], Cfg(U=1 << 15, logM=17, fused=bool(int(A.get('fused', 0)))), device='cuda')
    t0 = time.time()
    for t in range(tA + warm):
        if t == tA:
            pats = np.unpackbits(Z['bab_pat'], axis=1)[:, :br.NF].astype(bool)
            br.load_babble(np.array([0]), pats, Z['bab_set'])
        br.step(win_all[t][None], eye_all[t][None], rew_all[t][None])
    torch.cuda.synchronize()
    grow = time.time() - t0
    res = dict(grow_ticks=tA + warm, grow_secs=round(grow, 1), grow_ms_per_tick=round(1000 * grow / (tA + warm), 3),
               units=int(br.conj.n[0].item()), runs=[])
    print(json.dumps(res), flush=True)
    t_start = tA + warm
    span = T - t_start - N
    for B in Bs:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        b = tile(br, B)
        prof = {}
        if int(A.get('prof', 0)):
            import functools
            def wrap(name):
                f = getattr(b, name)
                @functools.wraps(f)
                def g(*a, **kw):
                    t = time.perf_counter(); r = f(*a, **kw)
                    p = prof.setdefault(name, [0, 0.0]); p[0] += 1; p[1] += time.perf_counter() - t
                    return r
                setattr(b, name, g)
            for nm in ('_drives', '_front', '_learn_step', '_confidence', '_recall_next', '_dg', '_detect', '_match',
                       '_recognize', '_awake_recall', '_replay', '_body', '_parse_eye', '_intend', '_store', '_begin_line',
                       '_line_end', '_line_item', '_open', '_think_step', '_scorer', '_view', '_ct_input', '_pend_finalize',
                       '_cands', '_vals', '_other_onset', '_reaction', '_reward', '_store_act', '_ctx_new', '_step_recall',
                       '_step_compare', '_step_vr', '_end_line', '_advance', '_set_recall'):
                wrap(nm)
        off = (np.arange(B) * 997) % max(span, 1)
        s0 = dict(b.stats)
        torch.cuda.synchronize()
        t1 = time.time()
        for t in range(N):
            ix = t_start + off + t
            b.step(win_all[ix], eye_all[ix], rew_all[ix])
        torch.cuda.synchronize()
        dt = time.time() - t1
        ev = {k: b.stats[k] - s0.get(k, 0) for k in b.stats}
        r = dict(B=B, ticks=N, secs=round(dt, 2), ms_per_tick=round(1000 * dt / N, 2),
                 agent_ticks_per_s=round(B * N / dt), gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                 gpu_mb_per_agent=round(torch.cuda.max_memory_allocated() / 2 ** 20 / B, 1), events=ev,
                 prof={k: [v[0], round(v[1], 2)] for k, v in sorted(prof.items(), key=lambda x: -x[1][1])[:16]})
        res['runs'].append(r)
        print(json.dumps(r), flush=True)
        del b
    print('RESULT', json.dumps(res))


if __name__ == '__main__':
    main()
