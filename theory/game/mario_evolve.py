"""Parameter search by selection for Mario by trial and error (designer 11-08: many models with different parameters,
keep the best, make new ones around them, several rounds). The mechanism (mario_self.py) stays fixed; only its
parameters change. Every model starts newborn and plays 1-1 for the same number of frames; its score is how far it
gets on average in the tries of the last 40 % of its frames. Each round: the best few are run again with a new seed
(a lucky seed should not win) and each gets several children with changed parameters. A few models without reward
(round 0) give the baseline.
usage: python mario_evolve.py <brain.pt> <out dir> [rounds] [per round] [parallel] [frames]"""
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# name: (low, high, log scale, default); a value of 0 for critic / record means that part is off
SPACE = {
    'MARIO_STEP': (0.01, 1.0, True, 0.3),
    'MARIO_DEATH': (1.0, 40.0, True, 10.0),
    'MARIO_FLAG': (1.0, 50.0, True, 5.0),
    'MARIO_R_WIN': (15.0, 240.0, True, 60.0),
    'MARIO_NOISE': (0.2, 3.0, True, 1.0),
    'MARIO_NOISE_T': (3.0, 120.0, True, 30.0),
    'MARIO_KEY_R': (0.08, 0.6, True, 0.2),
    'MARIO_RECORD': (0.05, 5.0, True, 1.0),
    'MARIO_LONG_T': (120.0, 2400.0, True, 600.0),
    'MARIO_CRITIC': (0.003, 0.3, True, 0.1),
    'MARIO_GAMMA': (0.95, 0.998, False, 0.99),
}
OFF = ('MARIO_CRITIC', 'MARIO_RECORD')                  # may also be switched off (0)


def draw(rng):
    p = {}
    for k, (lo, hi, lg, _) in SPACE.items():
        p[k] = math.exp(rng.uniform(math.log(lo), math.log(hi))) if lg else rng.uniform(lo, hi)
    for k in OFF:
        if rng.random() < 0.3:
            p[k] = 0.0
    return p


def mutate(p, rng, s=0.35):
    q = dict(p)
    for k, (lo, hi, lg, d) in SPACE.items():
        if rng.random() < 0.5:
            continue
        v = q[k]
        if k in OFF and v == 0.0:
            if rng.random() < 0.2:
                q[k] = d
            continue
        if k in OFF and rng.random() < 0.1:
            q[k] = 0.0
            continue
        v = math.exp(math.log(v) + rng.gauss(0, s)) if lg else v + rng.gauss(0, s * (hi - lo) / 4)
        q[k] = min(hi, max(lo, v))
    return q


def score(log, budget):
    """mean x of the tries that started in the last 40 % of the frames (at least the last 5 tries)"""
    xs, late = [], []
    for line in open(log):
        if not line.startswith('SELF'):
            continue
        f = line.split()
        x, tot, t = int(f[5]), int(f[f.index('total') + 1]), int(f[f.index('frames') + 1])
        xs.append(x)
        if tot - t >= 0.6 * budget:
            late.append(x)
    if len(late) < 5:
        late = xs[-5:]
    if not late:
        return None, 0, 0.0
    return sum(late) / len(late), len(xs), sum(v > 450 for v in late) / len(late)


def run_all(jobs, brain, out, parallel, budget):
    env0 = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY='/tmp/mps', CUDA_MPS_LOG_DIRECTORY='/tmp/mps-log',
                MARIO_GAZE='1', MARIO_THREADS='2', MARIO_FRAMES=str(budget))
    todo, live = list(jobs), []
    while todo or live:
        while todo and len(live) < parallel:
            j = todo.pop(0)
            env = dict(env0, **{k: repr(v) for k, v in j['p'].items()})
            if j.get('control'):
                env['MARIO_REWARD'] = '0'
            log = open(f'{out}/{j["name"]}.log', 'w')
            pr = subprocess.Popen([sys.executable, '-W', 'ignore', '-u', f'{HERE}/mario_self.py', brain, '1-1', '100000',
                                   f'{out}/{j["name"]}.pt', str(j['seed'])], env=env, stdout=log,
                                  stderr=subprocess.DEVNULL)
            live.append((pr, j, log))
        time.sleep(10)
        for item in list(live):
            if item[0].poll() is not None:
                item[2].close()
                live.remove(item)
                try:
                    os.remove(f'{out}/{item[1]["name"]}.pt')         # only the scores are kept
                except OSError:
                    pass
    for j in jobs:
        j['score'], j['tries'], j['past'] = score(f'{out}/{j["name"]}.log', budget)


def main():
    brain, out = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    parallel = int(sys.argv[5]) if len(sys.argv) > 5 else 22
    budget = int(sys.argv[6]) if len(sys.argv) > 6 else 40000
    os.makedirs(out, exist_ok=True)
    rng = random.Random(7)
    seed = 100
    hist = {}                                          # params key -> list of scores (over seeds)
    keyof = lambda p: json.dumps(p, sort_keys=True)    # noqa: E731
    jobs = []
    for i in range(4):                                 # baseline: no reward
        jobs.append({'name': f'r0_control{i}', 'p': {}, 'seed': seed, 'control': True})
        seed += 1
    jobs.append({'name': 'r0_default', 'p': {k: v[3] for k, v in SPACE.items()}, 'seed': seed})
    seed += 1
    while len(jobs) < per + 4:
        jobs.append({'name': f'r0_m{len(jobs)}', 'p': draw(rng), 'seed': seed})
        seed += 1
    for r in range(rounds):
        t0 = time.time()
        run_all(jobs, brain, out, parallel, budget)
        for j in jobs:
            if not j.get('control') and j['score'] is not None:
                hist.setdefault(keyof(j['p']), []).append(j['score'])
        ranked = sorted(hist.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        with open(f'{out}/round{r}.json', 'w') as f:
            json.dump({'jobs': jobs, 'ranked': [(json.loads(k), v) for k, v in ranked[:10]],
                       'minutes': (time.time() - t0) / 60}, f, indent=1)
        ctrl = [j['score'] for j in jobs if j.get('control') and j['score'] is not None]
        print(f'round {r}: {len(jobs)} models, {(time.time() - t0) / 60:.0f} min'
              + (f'; controls {sum(ctrl) / len(ctrl):.0f}' if ctrl else ''), flush=True)
        for k, v in ranked[:5]:
            print(f'   mean {sum(v) / len(v):6.0f} over {len(v)} seeds {[round(x) for x in v]}  {k}', flush=True)
        if r == rounds - 1:
            break
        parents = [json.loads(k) for k, _ in ranked[:5]]
        jobs = []
        for i, p in enumerate(parents):                # the best again, with a new seed
            jobs.append({'name': f'r{r + 1}_best{i}', 'p': p, 'seed': seed})
            seed += 1
        i = 0
        while len(jobs) < per:
            jobs.append({'name': f'r{r + 1}_c{i}', 'p': mutate(parents[i % len(parents)], rng), 'seed': seed})
            seed += 1
            i += 1


if __name__ == '__main__':
    main()
