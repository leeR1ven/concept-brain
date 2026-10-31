# Baselines on bench v0.2 (no cache), per stage, model tutor (as my agent), 2500 dialogues.
import sys, os, json, time, multiprocessing as mp
sys.path.insert(0, os.environ.get('PRED_BENCH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_snapshot')))
def one(job):
    st, seed = job
    from baselines import baselines
    from metrics import headline
    t0 = time.time(); res = baselines(st, seed=seed, n_dialogues=1500, tutor='model', cache=False)
    json.dump(res, open(f'bl2_{st}_s{seed}.json', 'w'), default=str)
    return '\n'.join([f'== {st} seed {seed} ({time.time()-t0:.0f}s)'] + [f'{k:7s} ' + headline(r, stage=st) for k, r in res.items()])
if __name__ == '__main__':
    jobs = [(st, s) for st in ('B', 'C', 'D') for s in (0,)]
    with mp.Pool(3) as p:
        for txt in p.imap_unordered(one, jobs): print(txt, flush=True)
