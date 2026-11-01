# -*- coding: utf-8 -*-
"""x2: re-run the (fixed) relearning probe on saved trained agents, one at a time in one process.
(numpy-2 server pickles do not load under local numpy 1.26: run those on the server.)
usage: python3 relearn_many.py out.json rounds tag:seed:path [tag:seed:path ...]"""
import os, sys, json, time, gzip, pickle, gc
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import drift_x2                                           # noqa: E402


def main():
    outp, rounds = sys.argv[1], int(sys.argv[2])
    import resource
    for spec in sys.argv[3:]:
        tag, seed, path = spec.split(':', 2)
        t0 = time.time()
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
        ag = obj['agent']
        rl = drift_x2.relearn(ag, int(seed), {'eye_layout': 'retino'}, rounds=rounds, log=lambda s: None)
        rec = dict(tag=tag, seed=int(seed), path=path, relearn=rl, secs=round(time.time() - t0, 1),
                   rss=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
        with open(outp, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        print(tag, seed, 'mean_first', rl['mean_first'], 'mean_stable', rl['mean_stable'],
              [(x['item'], x['before'], x['first'], x['stable_from'], x['last5']) for x in rl['items']],
              'secs', rec['secs'], 'rss', rec['rss'], flush=True)
        del obj, ag
        gc.collect()


if __name__ == '__main__':
    main()
