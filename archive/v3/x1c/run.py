"""X1c driver: phases (teach -> correction -> untutored use), frozen probes and all measurements.
Run:  python3 -I run.py <arm> <seed> [key=value ...]     (one process; writes out/<arm>_s<seed>.json)
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import (Physics, World, render, question, answer, SID, SYL, OWN, C1, FILL, COLS, SIZES)  # noqa
from core3 import Brain

ARMS = {
    # activity-control forms of the PFC (coordinator's arms)
    'C':        dict(care='format', b=dict(act='frac', f2=0.22, f3=0.3)),   # no PFC inhibition, depth-rising fraction
    'C2':       dict(care='format', b=dict(act='own')),                    # no PFC inhibition, own-signal, count kept
    'A':        dict(care='format', b=dict(act='local')),                  # local feedback inhibition + homeo + partners
    'B':        dict(care='format', b=dict(act='global')),                 # global quantile threshold (shortcut)
    # comparisons on B
    'B_nlms':   dict(care='format', b=dict(act='global', nlms=True)),
    'B_nopart': dict(care='format', b=dict(act='global', partners=False)),
    'B_linear': dict(care='format', b=dict(act='global', concave=False)),
    'B_fast':   dict(care='format', b=dict(act='global', fast=True)),
    'B_hip':    dict(care='format', b=dict(act='global', hip=True)),
    'B_dist':   dict(care='format', distract=True, b=dict(act='global')),
    'B_real':   dict(care='real', b=dict(act='global')),
    # coordinator's arm: no self-hold, random birth loop top->L1, stronger random lateral
    'B_loop':   dict(care='format', b=dict(act='global', kappa=0.0, w33=0.4, pred_rel=False, g_loop=8.0)),
}


def birth_rates(ph, world, n=3000):
    """birth listening period: firing rate of every A0 unit (on/off pairs); always-on units send no chain line."""
    w = World(ph, world.seed + 77, care=world.care, distract=world.distract, empty_every=0)
    acc = np.zeros(2 * ph.n_a)
    z = np.zeros(ph.n_mus, bool)
    for _ in range(n):
        snd, _r = w.step(z)
        acc += np.concatenate([snd, ~snd])
    return acc / n


def make_brain(ph, world, seed, bk):
    return Brain(ph, seed, rates=birth_rates(ph, world), **bk)


def probe(brain, ph, world, pairs, seed=0, wait=16):
    snap = brain.snapshot()
    rng = np.random.default_rng(seed)
    out = []
    for (f, x) in pairs:
        brain.restore(snap)
        Q, A = question(f, x), answer(f, x, world.props)
        sched = [None] * 6 + [SID[s] for s in Q for _ in range(3)]
        q_end = len(sched)
        sched += [None] * wait
        mprev = None
        utts, cur, sil, last = [], None, 99, None
        for t, cg in enumerate(sched + [None, None, None]):
            srcs = []
            if mprev is not None:
                srcs.append((mprev[0], mprev[1], OWN, 1.0))
            if cg is not None:
                srcs.append(('s', cg, C1, 1.0))
            mus = brain.step(render(ph, srcs, rng), 0.0, learn=False)
            m = ph.decode(mus)
            if m is None:
                sil += 1
                if cur is not None and sil >= 3:
                    utts.append(cur)
                    cur = None
                last = None
            else:
                key = ('s' if m[0] != 'g' else 'g', m[1])
                if cur is None:
                    cur = dict(t0=t, syl=[], garble=0)
                if key != last:
                    cur['syl'].append(SYL[m[1]] if m[0] != 'g' else '~')
                    cur['garble'] += m[0] == 'g'
                last = key
                sil = 0
            mprev = m
        if cur is not None:
            utts.append(cur)
        during = [u for u in utts if u['t0'] < q_end - 1]
        resp = [u for u in utts if q_end - 1 <= u['t0'] < q_end + 14]
        syl = resp[0]['syl'] if resp else None
        out.append(dict(f=f, x=x, held=(f, x) in world.held, A=A, syl=syl, lat=(resp[0]['t0'] - q_end) if resp else None,
                        correct=syl == A, first=bool(syl) and syl[0] == A[0], echo=bool(syl) and syl[0] in Q,
                        silent=syl is None, garble=bool(syl) and '~' in syl, talk=len(during) > 0))
    brain.restore(snap)
    return out


def summarize_probe(rs):
    def agg(sub):
        n = max(1, len(sub))
        return dict(n=len(sub), acc=round(sum(r['correct'] for r in sub) / n, 3),
                    first=round(sum(r['first'] for r in sub) / n, 3), echo=round(sum(r['echo'] for r in sub) / n, 3),
                    silent=round(sum(r['silent'] for r in sub) / n, 3), garble=round(sum(r['garble'] for r in sub) / n, 3),
                    talk=round(sum(r['talk'] for r in sub) / n, 3))
    d = dict(seen=agg([r for r in rs if not r['held']]), held=agg([r for r in rs if r['held']]))
    d['per_frame_seen'] = {f: agg([r for r in rs if r['f'] == f and not r['held']])['acc'] for f in range(4)}
    d['per_frame_held'] = {f: agg([r for r in rs if r['f'] == f and r['held']])['acc'] for f in range(4)}
    return d


def mechanics(world, ex_from, ex_to):
    exs = [e for e in world.exchanges if ex_from < e['n'] <= ex_to]
    n = max(1, len(exs))
    utts = [u for e in exs for u in e['utts'] if not u['empty']]
    resp = [e for e in exs if e['resp'] is not None]
    talk = sum(1 for u in utts if u['cg_speaking'])
    yfail = sum(1 for u in utts if u['yield_fail'])
    garb = sum(1 for u in utts if u['garble'] > 0)
    extra = sum(1 for e in exs for u in e['utts'] if not u['empty'] and u is not e['resp'] and not u['cg_speaking'])
    acc = sum(1 for e in resp if e['resp']['syl'] == e['A'])
    echo = 0
    for e in resp:
        Q = question(e['f'], e['x'])
        echo += e['resp']['syl'][:1] and e['resp']['syl'][0] in Q
    return dict(n_ex=len(exs), resp_rate=round(len(resp) / n, 3), online_acc=round(acc / n, 3),
                echo_per_ex=round(echo / n, 3), talkover_per_ex=round(talk / n, 4), yieldfail_per_ex=round(yfail / n, 4),
                garble_per_ex=round(garb / n, 4), extra_per_ex=round(extra / n, 4),
                self_answer=round(sum(e['self_answer'] for e in exs) / n, 3),
                lat_mean=round(float(np.mean([e['lat'] for e in resp])), 2) if resp else None)


def run_until(brain, world, n_target, rec=None):
    while not (world.n_ex >= n_target and world.ex is not None and world.ex['state'] == 'gap'):
        mus = brain.step(world.sound, getattr(world, 'r_last', 0.0))
        snd, r = world.step(mus)
        world.r_last = r
        if rec is not None and world.empty:
            rec['empty_ticks'] += 1
            if len(rec['pas']) < 4000:
                rec['pas'].append(brain.l3.copy())


def thought_motion(pas):
    if len(pas) < 60:
        return None
    P = np.array(pas, bool)
    def jac(a, b):
        u = (a | b).sum(1)
        return np.where(u > 0, (a & b).sum(1) / np.maximum(u, 1), 1.0)
    j1 = jac(P[:-1], P[1:])
    j5 = jac(P[:-5], P[5:])
    j20 = jac(P[:-20], P[20:])
    return dict(n_active=round(float(P.sum(1).mean()), 1), turnover1=round(float(1 - j1.mean()), 3),
                turnover5=round(float(1 - j5.mean()), 3), stuck20=round(float((j20 > 0.9).mean()), 3),
                distinct=int(len({p.tobytes() for p in P})), ticks=len(P))


def empty_speech(world, ex_from, ex_to, empty_ticks):
    u = [x for x in world.utts if x['empty'] and ex_from <= x['ex'] <= ex_to]
    return dict(utts=len(u), per100=round(100 * len(u) / max(1, empty_ticks), 3), syl=sum(len(x['syl']) for x in u),
                sample=[''.join(x['syl']) for x in u[:8]])


def main(arm, seed, n_teach=1500, n_corr=800, n_free=3000, extra=None):
    t0 = time.time()
    spec = dict(ARMS[arm])
    bk = dict(spec.get('b', {}))
    wk = dict(care=spec.get('care', 'format'), distract=spec.get('distract', False))
    if extra:
        for k, v in extra.items():
            if k.startswith('w_'):
                wk[k[2:]] = v
            else:
                bk[k] = v
    ph = Physics()
    world = World(ph, seed, **{k: v for k, v in wk.items() if k in ('care', 'distract', 'p_narr', 'wait')})
    if 'p_tut_free' in wk:
        world.p_tut_free = wk['p_tut_free']
    brain = make_brain(ph, world, seed, bk)
    allp = [(f, x) for f in range(4) for x in range(8)]
    res = dict(arm=arm, seed=seed, cfg=dict(world=wk, brain=bk), held=sorted(world.held), probes={}, mech={}, empty={},
               motion={})
    rec = dict(empty_ticks=0, pas=[])
    # ---------------- teaching
    marks = sorted({min(300, n_teach), n_teach})
    last = 0
    for m in marks:
        run_until(brain, world, m, rec)
        res['probes'][f'teach{m}'] = summarize_probe(probe(brain, ph, world, allp, seed))
        res['mech'][f'teach{last}-{m}'] = mechanics(world, last, m)
        print(arm, seed, f'teach {m}', res['probes'][f'teach{m}']['seen'], res['probes'][f'teach{m}']['held']['acc'],
              res['mech'][f'teach{last}-{m}'], f'{time.time() - t0:.0f}s', flush=True)
        last = m
    res['empty']['teach'] = empty_speech(world, 0, n_teach, rec['empty_ticks'])
    res['motion']['teach'] = thought_motion(rec['pas'])
    res['probe_detail_teach'] = probe(brain, ph, world, allp, seed)
    # ---------------- correction: the world changes 4 taught answers (2 colours, 2 sizes)
    rng = np.random.default_rng(seed + 3000)
    taught2 = [x for x in range(8) if (2, x) not in world.held]
    taught3 = [x for x in range(8) if (3, x) not in world.held]
    xc = list(rng.choice(taught2, 2, replace=False))
    xs = list(rng.choice([x for x in taught3 if x not in xc], 2, replace=False))
    changed = [(2, int(x)) for x in xc] + [(3, int(x)) for x in xs]
    old = {p: answer(p[0], p[1], world.props) for p in changed}
    for x in xc:
        cur = world.props[int(x)]['col']
        world.props[int(x)]['col'] = COLS[(COLS.index(cur) + 1 + int(rng.integers(2))) % 3]
    for x in xs:
        world.props[int(x)]['size'] = SIZES[1 - SIZES.index(world.props[int(x)]['size'])]
    new = {p: answer(p[0], p[1], world.props) for p in changed}
    n0 = world.n_ex
    rec = dict(empty_ticks=0, pas=[])
    run_until(brain, world, n0 + n_corr, rec)
    lat = {}
    for p in changed:
        exs = [e for e in world.exchanges if e['n'] > n0 and (e['f'], e['x']) == p]
        k = 0
        first = None
        seq = []
        for e in exs:
            syl = e['resp']['syl'] if e['resp'] is not None else None
            seq.append('N' if syl == new[p] else ('O' if syl == old[p] else ('-' if syl is None else 'x')))
            if first is None and syl == new[p]:
                first = k
            k += e['teach_new']
        lat[str(p)] = dict(first_new_after_teachings=first, n_asks=len(exs), seq=''.join(seq))
    pr = probe(brain, ph, world, allp, seed)
    res['correction'] = dict(changed=[list(p) for p in changed], latency=lat,
                             probe_new=round(float(np.mean([r['syl'] == new[(r['f'], r['x'])] for r in pr if (r['f'], r['x']) in new])), 3),
                             probe_old=round(float(np.mean([r['syl'] == old[(r['f'], r['x'])] for r in pr if (r['f'], r['x']) in old])), 3))
    res['probes']['corr'] = summarize_probe(pr)
    res['mech']['corr'] = mechanics(world, n0, world.n_ex)
    print(arm, seed, 'corr', res['correction'], res['probes']['corr']['seen'], f'{time.time() - t0:.0f}s', flush=True)
    # ---------------- untutored use
    world.phase = 'free'
    n1 = world.n_ex
    rec = dict(empty_ticks=0, pas=[])
    last = n1
    for m in (300, 1000, n_free):
        run_until(brain, world, n1 + m, rec)
        res['probes'][f'free{m}'] = summarize_probe(probe(brain, ph, world, allp, seed))
        res['mech'][f'free{m}'] = mechanics(world, last, n1 + m)
        print(arm, seed, f'free {m}', res['probes'][f'free{m}']['seen'], res['probes'][f'free{m}']['held']['acc'],
              res['mech'][f'free{m}'], f'{time.time() - t0:.0f}s', flush=True)
        last = n1 + m
    res['empty']['free'] = empty_speech(world, n1, world.n_ex, rec['empty_ticks'])
    res['motion']['free'] = thought_motion(rec['pas'])
    res['time_s'] = round(time.time() - t0, 1)
    res['speech_route'] = {k: (round(v / max(1, brain.route['onsets']), 3) if k != 'onsets' else v) for k, v in brain.route.items()}
    res['ms_per_tick'] = round(1000 * res['time_s'] / max(1, brain.t), 3)
    res['layer_counts'] = {k: dict(mean=round(float(np.mean(v)), 1), min=int(np.min(v)), max=int(np.max(v)),
                                   p_zero=round(float(np.mean(np.array(v) == 0)), 3),
                                   p_sat=round(float(np.mean(np.array(v) >= 0.9 * {'n1': brain.n1, 'n2': brain.c['n2'], 'n3': brain.c['n3']}[k])), 3),
                                   size={'n1': brain.n1, 'n2': brain.c['n2'], 'n3': brain.c['n3']}[k])
                           for k, v in brain.cnt.items()}
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'), exist_ok=True)
    tag = extra.get('tag', '') if extra else ''
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', f'{arm}{tag}_s{seed}.json')
    json.dump(res, open(fn, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('wrote', fn, res['time_s'], flush=True)
    return res


def parse(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    if v in ('True', 'False'):
        return v == 'True'
    return v


if __name__ == '__main__':
    arm, seed = sys.argv[1], int(sys.argv[2])
    kv = dict(a.split('=', 1) for a in sys.argv[3:])
    kv = {k: parse(v) for k, v in kv.items()}
    nt = kv.pop('n_teach', 1500)
    nc = kv.pop('n_corr', 800)
    nf = kv.pop('n_free', 3000)
    main(arm, seed, nt, nc, nf, kv)
