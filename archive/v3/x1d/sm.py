import sys, json
for l in open(sys.argv[1]):
    if not l.startswith('SUMMARY'): continue
    d = json.loads(l[8:]); lo = d.pop('long')
    print(' '.join(f'{k}={v}' for k, v in d.items() if k not in ('t',)), '| LONG to5=%s st20=%s n=%s pE=%s' % (lo['turnover5'], lo['stuck20'], lo['n_active'], lo['p_empty']))
