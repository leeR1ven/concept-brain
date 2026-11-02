# -*- coding: utf-8 -*-
"""tune2 configurations (keys on top of ARCH_v2 = synth final_noctx).  Rung 1 = A 300 + 3 x 500, seeds 0 and 1."""
R1 = {
    'base': '',
    # A. context length / excitability (held lines = 1 + floor(log(theta/gain) / log(lambda)) while lambda >= theta)
    'lam60': 'ct_lam=0.6', 'lam75': 'ct_lam=0.75', 'lam85': 'ct_lam=0.85',                     # bag keys: 2 / 3 / 5 lines
    'lam60a': 'ct_lam=0.6 ctx_keys=age', 'lam75a': 'ct_lam=0.75 ctx_keys=age', 'lam85a': 'ct_lam=0.85 ctx_keys=age',
    'k4': 'ct_k=4',                                                                            # 1 line, at most 4 items
    'lam85ak24': 'ct_lam=0.85 ctx_keys=age ct_k=24',                                           # 5 lines uncapped
    'lam85atau': 'ct_lam=0.85 ctx_keys=age ct_tau=150',                                        # leaky in time
    'own60': 'ct_own=0.6',                                                                     # own speech enters
    # C. sensitive constants
    'eta25': 'eta=0.25', 'eta50': 'eta=0.5',
    'mthr50': 'match_thr=0.5', 'mthr70': 'match_thr=0.7',
    'cmin4': 'conf_min=4', 'cmin8': 'conf_min=8',
    'rmax30': 'rate_max=0.3', 'rmax50': 'rate_max=0.5',
    'rthr75': 'rec_thr=0.75', 'rthr95': 'rec_thr=0.95',
    'u05': 'u_reply=0.05', 'u30': 'u_reply=0.3',
    # B. wiring locality (last priority)
    'loc04': 'wiring=local loc_r=0.04', 'sw16': 'wiring=sw loc_r=0.04 n_lr=16', 'rand40': 'wiring=rand n_lr=40',
}
R1_ARGS = 'nS=500 evaln=80 drift=300 probes=min ctxp=1'


def lines(names, seeds, prefix='r1', args=R1_ARGS, table=R1):
    out = []
    for s in seeds:
        for n in names:
            out.append(f"run_t2.py tag={prefix}_{n} seed={s} {args} {table[n]}".strip())
    return out


if __name__ == '__main__':
    import sys
    names = sys.argv[2].split(',')
    seeds = [int(x) for x in sys.argv[3].split(',')]
    open(sys.argv[1], 'w').write('\n'.join(lines(names, seeds)) + '\n')
    print(open(sys.argv[1]).read())
