# diagnose stage C of a run: ritual replies, silent rate, overlaps (where / which thought)
import sys, pickle, collections
p = sys.argv[1]
L = pickle.load(open(p + '_log.pkl', 'rb'))
TH = pickle.load(open(p + '_thoughts.pkl', 'rb')) if len(sys.argv) < 3 else None
C = [r for r in L if r['stage'] == 'C' and r['kind'] in ('probe', 'recast') and 2300 <= r['dlg'] < 4300]
for task in ('ritual', 'colour'):
    rs = [r for r in C if r['task'] == task]
    print(task, len(rs), 'replies:', collections.Counter((r['prompt'], r['reply']) for r in rs).most_common(14))
ov = [r for r in L if r.get('mouth_while_partner', 0) > 0]
print('overlap exchanges', len(ov))
for r in ov[:8]:
    print('  ', r['stage'], r['kind'], r['task'], r['prompt'], '| reply', repr(r['reply']), 'lat', r.get('latency'),
          'mwp', r['mouth_while_partner'], 'mdp', r.get('mouth_during_prompt'), 'after', repr(r.get('after_recast_reply')),
          't0', r['t_prompt_start'])
    if TH:
        for t in TH:
            if t['t_rel'] is not None and r['t_prompt_start'] - 14 <= t['t_rel'] <= r['t_prompt_start'] + 2:
                print('      thought', t['kind'], 'open', t['t_open'], 'rel', t['t_rel'], 'tau', t['tau_rel'], t['outcome'], t['p_rel'], t['feats'][:4])
