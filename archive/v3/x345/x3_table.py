"""compact tables from x3_results.json (horizon = upper edge of the last age bin before the first bin with n >= 3 and
mean < 0.9; bins with n < 3 are skipped)"""
import json, os
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'x3_results.json')))


def hor(rows, thr=0.9):
    h = 0
    for lo, hi, n, m in rows:
        if n < 3:
            continue
        if m >= thr:
            h = hi
        else:
            break
    return h


def at(rows, lo0):
    for lo, hi, n, m in rows:
        if lo == lo0:
            return m
    return None


print('%-32s %8s %8s %8s | r50 by age: 2-5k 5-10k 10-20k 20-50k | r20 5-10k 10-20k | i100 5-10k | fid50 5-10k' % ('run', 'h(r50)', 'h(r20)', 'h(i100)'))
for name, b in R.items():
    print('%-32s %8d %8d %8d |        %5s %5s %5s %5s | %5s %5s | %5s | %5s' % (name, hor(b['r50']), hor(b['r20']), hor(b['i100']),
          *[str(at(b['r50'], a)) for a in (2000, 5000, 10000, 20000)], str(at(b['r20'], 5000)), str(at(b['r20'], 10000)),
          str(at(b['i100'], 5000)), str(at(b['fid50'], 5000))))
b = R['base']
print('\nbase, unique vs repeated (r50 by age bin lo: n, mean):')
for k in ('u_r50_n1', 'rB_r50', 'rS_r50'):
    print('  ', k, [(lo, n, m) for lo, hi, n, m in b[k]])
print('separation (age < 20k, 50 % stored cue: right continuation / the similar episode\'s continuation):')
for name in ('base', 'arm_recode_no_birth_recurrence', 'arm_copy_no_recode'):
    print('  ', name, {k: (R[name][k]['right'], R[name][k]['other'], R[name][k]['n']) for k in ('sep_pS', 'sep_pV', 'sep_pD')})
print('random kicks (summed over test times):')
for name in ('base', 'arm_recode_no_birth_recurrence', 'arm_copy_no_recode'):
    K = R[name]['kicks']; tot = {k: sum(x[k] for x in K) for k in ('n', 'silent', 'decoded', 'world_seq', 'probe_seq')}
    print('  ', name, tot)
for name in ('pert_3pct', 'pert_5pct'):
    b = R[name]
    print(name, 'probes written before the split/delete, recalled after: r50', [(lo, n, m) for lo, hi, n, m in b['pert_r50']],
          '| i100', [(lo, n, m) for lo, hi, n, m in b['pert_i100']])
    print('   ', 'not crossing it: r50', [(lo, n, m) for lo, hi, n, m in b['nopert_r50']])
