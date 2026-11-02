"""short pilot lives (birth 200, teach N, no correction / untutored), one process, configs one after another.
  python3 -I pilot.py <listname> [seed] [n_teach]          -> out3/pilot_<name>_s<seed>.json + one SUMMARY line each
"""
import sys, os, json, time, io, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _x in ('x1d', 'x1c'):   # repo: core4/phase1/run2/regimes from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
import run3
from regimes5 import ARMS, rn, r1, RNpm, R1e, L33


def summ(res, tagp='teach'):
    p = res['path'][tagp]
    s, c = np.array(p['same']), np.array(p['cross'])
    tl = max((k for k in res['probes'] if k.startswith('teach')), key=lambda k: int(k[5:]))
    sp = res['spec'][tl]
    h = res['hub'][tagp]
    return dict(same0=round(s[0], 2), cross0=round(c[0], 2), same12=round(s[12], 2), cross12=round(c[12], 2),
                dq=[round(p['dec_q'][k], 3) for k in (0, 3, 6, 9, 12)], dfr12=round(p['dec_frame'][12], 2),
                within=round(float(np.mean(p['within_J1'][1:13])), 2), dmg12=p['damage']['0.1'][12],
                n3=[round(p['n3_lag'][k]) for k in (0, 6, 12)], pempty12=round(p['p_empty_lag'][12], 2),
                own=sp['own_max'], other=sp['other_max'], hub=sp['hub_syl'] + str(sp['hub_share']),
                first=res['probes'][tl]['seen']['first'], acc=res['probes'][tl]['seen']['acc'],
                silent=res['probes'][tl]['seen']['silent'], garble=res['probes'][tl]['seen']['garble'],
                L33pos=h['L33_pos_frac'], L33in=h['L33_in_w_pct'], forced=res['release']['FORCED_lag12'].get('said_when_pred'),
                lc=res['layer_counts']['n3'], t=res['time_s'])


LISTS = {
    'p1': [
        ('RNpm', dict(RNpm)),
        ('RNpm_L8', rn(8.0)),
        ('RNpm_L8_f3hi', dict(rn(8.0), f3=0.3)),
        ('RNpm_L8_a0', dict(rn(8.0), a33=0.0)),
        ('RNpm_L8_a.25', dict(rn(8.0), a33=0.25)),
        ('RNpm_L8_eta.005', dict(rn(8.0), eta33=0.005)),
        ('RNpm_L8_dec.99', dict(rn(8.0), dec_33=0.99)),
        ('RNpm_L2', rn(2.0)),
        ('RNpm_L16', rn(16.0)),
        ('R1e', dict(R1e)),
        ('R1e_L8', r1(8.0)),
        ('R1e_L8_a0', dict(r1(8.0), a33=0.0)),
    ],
}

if __name__ == '__main__':
    lst = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    only = sys.argv[4].split(',') if len(sys.argv) > 4 else None
    for name, b in LISTS[lst]:
        if only and name not in only:
            continue
        run3.ARMS['_p_' + name] = dict(care='format', b=b)
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            res = run3.main('_p_' + name, seed, 200, nt, 0, 0, dict(tag=f'_{lst}'))
        print('SUMMARY', name, json.dumps(summ(res), ensure_ascii=False), flush=True)
