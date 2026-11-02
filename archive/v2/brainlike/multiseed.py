# summarise the GPU job's partial / complete pipeline JSONs: python3 -B multiseed.py RESULT_DIR
import sys, json, glob, os, numpy as np
D = sys.argv[1]
skip = ('ALL', 'SCENE', 'NOSCENE', 'mech', 'after_model', 'learning', 'one_shot')
rows = []
for f in sorted(glob.glob(os.path.join(D, 'res_*.json'))):
    try:
        R = json.load(open(f))
    except Exception as e:
        print(f, 'unreadable', e); continue
    name = os.path.basename(f)[4:-5]
    out = {'run': name}
    for st in 'BCD':
        a = R.get(st, {}).get(f'{st}/ALL')
        if a:
            out[st] = (a['acc'], a.get('held'))
            out[st + '_tasks'] = {k.split('/')[1]: round(v['acc'], 2) for k, v in R[st].items()
                                  if isinstance(v, dict) and v.get('acc') is not None and not k.endswith(skip)}
            l = R[st].get(f'{st}/learning')
            if l:
                out[st + '_learn'] = (round(l['acc_early'], 2), round(l['acc_late'], 2))
        p = R.get(st + '_predict')
        if p:
            out[st + '_pred'] = (round(p['readout'], 2), round(max(p['n1'], p['n2'], p['n3']), 2))
    if 'lesion' in R:
        out['lesions'] = {L: round(np.mean([v['acc'] for k, v in d.items() if isinstance(v, dict) and v.get('acc') is not None and not k.endswith(skip)]), 3)
                          for L, d in R['lesion'].items()}
    rows.append(out)
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
