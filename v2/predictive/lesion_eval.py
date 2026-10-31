# Causal audit: a trained agent (pickle) runs the SAME content (new World with the same stage/seed/tutor/prev) with each
# learned part silenced; learning stays on (the agent is always online).  Prints acc per task for each lesion.
import sys, copy, pickle, json, argparse
import numpy as np
sys.path.insert(0, __import__('os').environ.get('PRED_HERE', __import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from run_lib import DiagWorld, Timer
from metrics import evaluate, headline
ap = argparse.ArgumentParser()
ap.add_argument('pickle'); ap.add_argument('--stage', default='B'); ap.add_argument('--n', type=int, default=300)
ap.add_argument('--tutor', default='model'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--lesions', default='none,x1,x2,start,gline,ghold,gtopic,B,V,V2,C2,H,Brec,go,fwd')
ap.add_argument('--out', default='lesion')
args = ap.parse_args()
T = Timer()
agent0, world0 = pickle.load(open(args.pickle, 'rb'))
res_all = {}
for L in args.lesions.split(','):
    a = copy.deepcopy(agent0)
    a.lesion = set() if (L == 'none' or L.startswith('floor=')) else set(L.split('+'))
    if L.startswith('floor='):
        a.body.theta = float(L.split('=')[1])      # behavioural readout threshold only (own speech does not train ZP)
    a.predict()
    w = DiagWorld(args.stage, seed=args.seed, tutor=args.tutor, prev=world0)
    log = w.run(a, n_dialogues=args.n)
    res = evaluate(log)[args.stage]
    row = {t: (r['n'], round(r['acc'], 3), round(r['held']['acc'], 3) if r['held']['acc'] is not None else None,
               round(r['seen']['acc'], 3) if r['seen']['acc'] is not None else None)
           for t, r in res.items() if isinstance(r, dict) and r.get('n') and 'held' in r}
    res_all[L] = row
    print(f"lesion {L:10s} ALL acc {row['ALL'][1]:.3f} held {row['ALL'][2]} seen {row['ALL'][3]} | " +
          ' '.join(f"{t} {v[1]:.2f}" for t, v in row.items() if t not in ('ALL', 'SCENE', 'NOSCENE')), T(), flush=True)
json.dump(res_all, open(args.out + '.json', 'w'), indent=1)
