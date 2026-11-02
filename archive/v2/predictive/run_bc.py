# Pipeline: (1) listen to stage-B/C model dialogues while unable to speak, (2) babble (no tutor), (3) bench stages.
import sys, json, time, pickle
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from run_lib import DiagWorld, new_agent, z_report, lexicon_chars, Timer
from metrics import evaluate, headline
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--listen', type=int, default=300)
ap.add_argument('--babble', type=int, default=120000); ap.add_argument('--B', type=int, default=1500)
ap.add_argument('--C', type=int, default=1500); ap.add_argument('--out', default='bc_s0')
ap.add_argument('--save', type=int, default=0)
args = ap.parse_args()
T = Timer()
w0 = DiagWorld('BC', seed=args.seed, tutor='model')
a = new_agent(w0); a.log_events = []
w0.run(a, n_dialogues=args.listen)
print('listen', z_report(a, w0, lexicon_chars()), T(), flush=True)
w0.babble(a, args.babble, bout=(4, 8))
print('babble', z_report(a, w0, lexicon_chars()), a.stats, T(), flush=True)
a.log_events = None
res_all = {}
prev = w0
for st, n in (('B', args.B), ('C', args.C)):
    if n <= 0: continue
    w = DiagWorld(st, seed=args.seed, tutor='model', prev=prev)
    log = w.run(a, n_dialogues=n)
    res = evaluate(log); res_all[st] = res
    print(headline(res), T(), flush=True)
    for task, r in res[st].items():
        if isinstance(r, dict) and 'acc' in r and r.get('n'):
            print(f'   {task:8s} n {r["n"]:4d} acc {r["acc"]:.2f} first {r["first"]:.2f} held {r["held"]["acc"]} seen {r["seen"]["acc"]}')
    print('   after_model', res[st]['after_model'], ' learning', res[st]['learning'], flush=True)
    prev = w
json.dump(res_all, open(f'{args.out}.json', 'w'), default=str, indent=1)
if args.save:
    pickle.dump(a, open(f'{args.out}_agent.pkl', 'wb'))
