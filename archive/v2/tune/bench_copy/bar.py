# -*- coding: utf-8 -*-
"""The bar: RefV1 (ARCH_v1-equivalent) and all baselines on stages B, C, D (and E), several seeds, in parallel.
usage: python3 bar.py [n_dialogues] [seeds] [workers] [out_dir]     e.g. python3 bar.py 2000 0,1,2 12 results"""
import hashlib, json, multiprocessing as mp, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'                      # one thread per worker (BLAS would otherwise spawn one per core)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ND = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else '0,1,2').split(',')]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
OUT = sys.argv[4] if len(sys.argv) > 4 else 'results'


def fingerprint():
    from baselines import BASELINES, run_agent
    w, log = run_agent(BASELINES['oracle'](), 'BCD', 0, 60, 'model')
    h = hashlib.md5(json.dumps([(r['prompt'], r['reply'], r['t_prompt_start'], r['scene']) for r in log],
                               ensure_ascii=False, default=str).encode()).hexdigest()
    return dict(ticks=w.t, records=len(log), md5=h)


def job(spec):
    import numpy as np
    from world import World
    from metrics import evaluate, headline
    from baselines import BASELINES, Silent
    from ref_v1 import RefV1
    name, agent_name, stage, seed, tutor, nd = spec
    t0 = time.time()
    w = World(stage, seed=seed, tutor=tutor)
    if agent_name.startswith('refv1'):
        agent = RefV1(w, cfg=agent_name.split('_')[1], seed=seed)
    elif agent_name == 'silent':
        agent = Silent()
    else:
        agent = BASELINES[agent_name]()
    log = w.run(agent, nd)
    res = evaluate(log)
    out = dict(name=name, agent=agent_name, stage=stage, seed=seed, tutor=tutor, n_dialogues=nd, ticks=w.t,
               secs=round(time.time() - t0, 1), ms_per_tick=round((time.time() - t0) / w.t * 1e3, 3),
               headline=headline(res), res=res, timing_violations=dict(w.timing_violations))
    if agent_name.startswith('refv1'):
        out['programs'] = int(agent.body.n_prog)
        out['backterm_hits'] = int(agent.backterm_hits)
        out['examples'] = [(r['task'], r['prompt'], r['reply'], r['acceptable'][0], r['held_out'])
                           for r in log if r['kind'] == 'probe'][-30:]
    with open(os.path.join(OUT, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, default=str)
    return name, out['secs'], out['headline']


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    import numpy
    print('numpy', numpy.__version__, 'fingerprint', json.dumps(fingerprint()), flush=True)
    specs = []
    for seed in SEEDS:
        for stage in ('B', 'C', 'D'):
            for cfg in ('mem', 'base'):
                specs.append((f'refv1_{cfg}_{stage}_force_s{seed}', f'refv1_{cfg}', stage, seed, 'force', ND))
            for tutor in ('force', 'model'):
                for b in ('echo', 'last', 'prior', 'lookup', 'ngram', 'oracle'):
                    specs.append((f'{b}_{stage}_{tutor}_s{seed}', b, stage, seed, tutor, ND))
        specs.append((f'refv1_mem_E_force_s{seed}', 'refv1_mem', 'E', seed, 'force', ND))
        for b in ('echo', 'last', 'prior', 'lookup', 'ngram', 'oracle'):
            specs.append((f'{b}_E_force_s{seed}', b, 'E', seed, 'force', ND))
    specs.append(('refv1_mem_B_model_s0', 'refv1_mem', 'B', SEEDS[0], 'model', ND))
    specs.append(('refv1_mem_C_recast_s0', 'refv1_mem', 'C', SEEDS[0], 'recast', ND))
    specs.append(('silent_BC_model_s0', 'silent', 'BC', SEEDS[0], 'model', ND))
    specs.sort(key=lambda s: not s[1].startswith('refv1'))          # long jobs first
    done = {f[:-5] for f in os.listdir(OUT) if f.endswith('.json')}
    if done:
        print('skipping', len(done), 'finished jobs', flush=True)
        specs = [s for s in specs if s[0] not in done]
    print(len(specs), 'jobs,', WORKERS, 'workers', flush=True)
    t0 = time.time()
    with mp.Pool(WORKERS) as pool:
        for name, secs, hl in pool.imap_unordered(job, specs):
            print(f'[{time.time() - t0:6.0f}s] {name} ({secs}s)\n    ' + hl.replace('\n', '\n    '), flush=True)
    print('ALL DONE', round(time.time() - t0), 's', flush=True)
