"""phase-1 tables and curves (markdown) from out1/*.json"""
import json, os, sys, glob
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out1')
LAGS = (0, 1, 2, 3, 4, 6, 8, 10, 12)


def load(n, seed=0):
    f = f'{D}/{n}_s{seed}.json'
    return json.load(open(f)) if os.path.exists(f) else None


def table(names):
    print('| regime | control | same@0 | cross@0 | same@12 | cross@12 | gap@12 | decQ@12 | decFrame@12 | noise-only@12 | damage10%@12 | within J1 | L3 mean (max) | empty | mouth | long silence stuck20 / turnover5 |')
    print('|' + '---|' * 16)
    for n in names:
        r = load(n)
        if r is None:
            continue
        s, c = np.array(r['same']), np.array(r['cross'])
        L = slice(10, 13)
        lo = r['motion_long_silence']
        print(f"| {n} | {r['cfg'].get('act')} | {s[0]:.2f} | {c[0]:.2f} | {s[L].mean():.2f} | {c[L].mean():.2f} | {(s - c)[L].mean():+.3f} | "
              f"{np.mean(r['dec_q'][10:13]):.3f} | {np.mean(r['dec_frame'][10:13]):.2f} | {np.mean(r['noise_same'][10:13]):.2f} | "
              f"{r['damage']['0.1'][12]:.2f} | {np.mean(r['within_J1'][1:13]):.2f} | {np.mean(r['n3_lag']):.0f} ({max(r['n3_lag_max']):.0f}) | "
              f"{np.mean(r['p_empty_lag']):.2f} | {np.mean(r['mouth_lag']):.2f} | {lo['stuck20']:.2f} / {lo['turnover5']:.2f} |")


def curves(n, seeds=(0,)):
    R = [load(n, s) for s in seeds]
    R = [r for r in R if r is not None]
    if not R:
        return
    print(f'\n**{n}** ({len(R)} seed{"s" if len(R) > 1 else ""}; cfg {R[0]["cfg"]})\n')
    print('| lag | ' + ' | '.join(str(k) for k in LAGS) + ' |')
    print('|' + '---|' * (len(LAGS) + 1))
    for key, lab in (('same', 'same Q, other context'), ('cross', 'other Q'), ('noise_same', 'same Q+context, new noise'),
                     ('dec_q', 'decode Q (chance .031)'), ('dec_q_win3', 'decode Q, 3-tick window'), ('dec_frame', 'decode frame (chance .25)'),
                     ('dec_fill_given_frame', 'decode filler | frame (chance .125)'), ('n3_lag', 'L3 active'), ('mouth_lag', 'mouth on')):
        if key not in R[0]:
            continue
        v = np.mean([r[key] for r in R], 0)
        print(f'| {lab} | ' + ' | '.join(f'{v[k]:.2f}' if k < len(v) else '' for k in LAGS) + ' |')
    w = np.mean([r['within_J1'] for r in R], 0)
    print('| J(lag, lag+1) within path | ' + ' | '.join(f'{w[k]:.2f}' if k < len(w) else '' for k in LAGS) + ' |')
    for p in ('0.1', '0.25'):
        v = np.mean([r['damage'][p] for r in R], 0)
        print(f'| damage {int(float(p) * 100)}% (J to unperturbed) | ' + ' | '.join(f'{v[k]:.2f}' for k in LAGS) + ' |')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'table':
        table(sys.argv[2].split(','))
    else:
        seeds = tuple(int(s) for s in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0,)
        for n in sys.argv[2].split(','):
            curves(n, seeds)
