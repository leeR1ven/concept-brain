"""one-line summaries of out3 lives matching a pattern: python3 -I qsum.py <glob>"""
import json, glob, sys, os
import numpy as np
for fn in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out3', sys.argv[1]))):
    d = json.load(open(fn))
    tl = max((k for k in d['probes'] if k.startswith('teach')), key=lambda k: int(k[5:]))
    p = d['path']['teach']; s = d['spec'][tl]; m = d['mech'][tl]
    pr = d['probes'][tl]['seen']
    print(os.path.basename(fn)[:-5], '| silent %.2f garble %.2f talk %.2f first %.2f acc %.2f' % (pr['silent'], pr['garble'], pr['talk'], pr['first'], pr['acc']),
          '| resp %.2f garb/ex %.2f talkover %.3f yieldf %.3f' % (m['resp_rate'], m['garble_per_ex'], m['talkover_per_ex'], m['yieldfail_per_ex']),
          '| spec %.3f-%.3f=%.3f hub %s%.2f' % (s['own_max'], s['other_max'], s['own_max'] - s['other_max'], s['hub_syl'], s['hub_share']),
          '| F6 first %.2f said %.2f' % (d['release']['FORCED_lag6']['first'], d['release']['FORCED_lag6']['said_when_pred']),
          '| dq0/6/12 %.2f/%.2f/%.2f' % tuple(p['dec_q'][k] for k in (0, 6, 12)), 'gap12 %.3f' % (p['same'][12] - p['cross'][12]),
          'within %.2f dmg12 %.2f n3 %s pe12 %.2f' % (np.mean(p['within_J1'][1:13]), p['damage']['0.1'][12], [round(p['n3_lag'][k]) for k in (0, 6, 12)], p['p_empty_lag'][12]))
