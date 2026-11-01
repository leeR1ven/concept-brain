"""Run a list of X6 specs (a JSON file: [{name, cfg, t_end, seed, kw}]) one after another; one JSON result per spec in
out/. Run: python3 -I x6_run.py specs/<file>.json"""
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x6_hippo as H6  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
specs = json.load(open(sys.argv[1]))
for sp in specs:
    path = os.path.join(HERE, 'out', sp['name'] + '.json')
    if os.path.exists(path):
        continue
    t0 = time.time()
    print('START', sp['name'], time.strftime('%T'), flush=True)
    cfg = dict(H6.BASE6, **sp['cfg'])
    r = H6.run6(cfg, sp['t_end'], seed=sp['seed'], verbose=True,
                log=lambda m: print(sp['name'], m, flush=True) if sp.get('verbose') else None, **sp.get('kw', {}))
    json.dump(dict(spec=sp, **r), open(path, 'w'), ensure_ascii=False)
    print('END', sp['name'], f'{time.time() - t0:.0f}s', 'rss', r['rss_mb'], flush=True)
