# -*- coding: utf-8 -*-
"""Compare life runs (v2.1 metrics): usage python3 life_table.py tag1 tag2 ... (out/life/<tag>_s5.json)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for t in sys.argv[1:]:
    d = json.load(open(os.path.join(HERE, 'out', 'life', f'{t}_s5.json')))
    m = d['v21']; s = d['summary']
    f = m['free']
    print(f"== {t}: free {f['all']['agent']} (last tenth {f['all']['agent_last_tenth']}) | scene {f['scene']['agent']} (const {f['scene']['const_best']}) "
          f"held {f['held_out']['agent']} social {f['social']['agent']} (last {f['social']['agent_last_tenth']}) memory {f.get('memory', {}).get('agent')} | "
          f"role-conf {f['all']['role_conf']} | echo {f['all']['echo']} const {f['all']['const_best']} echo|const {f['all']['echo_or_const']}")
    print(f"   init {m['n_init']} {m['categories']} appropriate {m['appropriate']:.3f} (partner {m['appropriate_partner']:.3f}, last tenth {m['appropriate_last_tenth']})")
    print(f"   nihao empty/tenth {m['nihao_empty_per_tenth']}  nonsense/tenth {m['nonsense_per_tenth']}  greeting/tenth {m['greeting_per_tenth']}")
    print(f"   top init {m['top_init_text'][:8]}")
    pt = ', '.join(k + ': ' + str(v['acc']) + ('/' + str(v['held']) if v['held'] is not None else '') for k, v in m['per_task'].items())
    print('   per task (acc/held) ' + pt)
    print(f"   ral {d.get('ral')}  tstats {d.get('tstats')}")
