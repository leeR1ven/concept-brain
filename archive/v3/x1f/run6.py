"""X1f driver.  python3 -I run6.py <arm> <seed> [key=value ...]  -> out6/<arm><tag>_s<seed>.json
arms: G0 (main: sound<->action map A1+A2 only), G0T (diagnostic: + hand-wired thinking B + responding C on a base set).
Stages: test 1 (learning off) -> O overhearing (marks) -> S speaking (format caregiver) -> F untutored -> amnesia ->
corrections.  All probes frozen (learning off)."""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _x in ('x1d', 'x1c'):   # repo: core4/phase1/run2/regimes from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
from common6 import (make_brain, run_until, probe6, summ, spec, mechanics6, acc_by_set, GEN0, Physics, World,
                     question, answer, SYL)
from world import COLS, SIZES
from world6 import OverhearWorld, overhear_mech
from wire import wire_speech, wire_thinking
from run import parse
import phase1

ARMS = {'G0': dict(think=False), 'G0T': dict(think=True),
        'G0P': dict(think=True, paths_all=True), 'G0P_noL33': dict(think=True, paths_all=True, b=dict(eta33=0.0)), 'G0_noL33': dict(think=False, b=dict(eta33=0.0)),
        'G0T_noL33': dict(think=True, b=dict(eta33=0.0))}


def decode_lag(pr_a, pr_b, idx, lags=(0, 3, 6, 9, 12)):
    """question decoding from two noise repetitions of the muted probe: nearest top-layer state (Jaccard) at the same
    lag among all 32 questions; reported for the questions in idx (chance 1/32)"""
    out = {}
    A = np.array([r['l3'] for r in pr_a])
    B = np.array([r['l3'] for r in pr_b])
    for k in lags:
        a, b = A[:, k].astype(np.float32), B[:, k].astype(np.float32)
        inter = a @ b.T
        uni = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
        J = np.where(uni > 0, inter / np.maximum(uni, 1), 0)
        ok = [J[i].argmax() == i and J[i].max() > 0 for i in idx]
        out[k] = round(float(np.mean(ok)), 3) if len(idx) else None
    return out


def wstats(brain):
    L33 = brain.L33
    return dict(L31_max=round(float(brain.L31.max()), 3), L31_mean=round(float(brain.L31.mean()), 5),
                L33_max=round(float(L33.max()), 3), L33_pos=round(float((L33 > 1e-3).mean()), 4),
                L33_neg=round(float((L33 < -1e-3).mean()), 4))


def main(arm, seed, marks=(100, 300, 1000, 3000), n_speak=300, n_free=1000, n_corr=300, extra=None):
    t0 = time.time()
    spec_arm = ARMS[arm]
    extra = dict(extra or {})
    tag = extra.pop('tag', '')
    wk = {k[2:]: extra.pop(k) for k in list(extra) if k.startswith('w_')}
    tk = {k[3:]: extra.pop(k) for k in list(extra) if k.startswith('th_')}
    bk = dict(GEN0, **spec_arm.get('b', {}), **extra)
    ph, brain = make_brain(seed, bk)
    world = World(ph, seed, care='format')
    props, held = world.props, world.held
    allp = [(f, x) for f in range(4) for x in range(8)]
    seen = list(world.pairs)
    rng = np.random.default_rng(seed + 4000)
    perm = rng.permutation(len(seen))
    base = [seen[i] for i in sorted(perm[:len(seen) // 2])]
    new = [p for p in seen if p not in base]
    if not spec_arm['think']:
        new, base = seen, []
    sets = dict(base=set(base), new=set(new), held=set(held))
    reg = [wire_speech(brain, ph)]
    if spec_arm['think'] and spec_arm.get('paths_all'):   # DIAGNOSTIC: paths (B) for every seen pair, C for base only
        reg.append(wire_thinking(brain, ph, props, seen, seed, say_pairs=set(base), **tk))
    elif spec_arm['think']:
        reg.append(wire_thinking(brain, ph, props, base, seed, **tk))
    res = dict(arm=arm, seed=seed, cfg=dict(brain=bk, world=wk, think=tk), registry=reg, held=sorted(held),
               base=base, new=new, probes={}, mech={}, spec={}, path={}, weights={}, overhear={}, counts={})
    rec = dict(n3=[], np=[])
    idx = {k: [allp.index(p) for p in v] for k, v in sets.items()}

    def checkpoint(name, live=None):
        pr = probe6(brain, ph, props, held, allp, seed)
        res['probes'][name] = {k: summ([r for r in pr if (r['f'], r['x']) in v]) for k, v in sets.items()}
        res['spec'][name] = {k: spec([r for r in pr if (r['f'], r['x']) in v]) for k, v in sets.items() if len(v) > 3}
        pa = probe6(brain, ph, props, held, allp, seed + 1, mute=True, rec_l3=True)
        pb = probe6(brain, ph, props, held, allp, seed + 2, mute=True, rec_l3=True)
        res['path'][name] = {k: decode_lag(pa, pb, v) for k, v in idx.items() if v}
        res['path'][name]['n3_lag'] = np.round(np.mean([r['l3'].sum(1) for r in pa], 0), 1).tolist()
        pd = probe6(brain, ph, props, held, [p for p in allp if p in sets['new'] or p in sets['base']], seed, mute=True, drive=True)
        res.setdefault('drive', {})[name] = {k: dict(ans_l1_6=round(float(np.mean([r['drive_ans'][1:7].max() for r in pd if (r['f'], r['x']) in v])), 4),
                                                      other_l1_6=round(float(np.mean([r['drive_other'][1:7].mean() for r in pd if (r['f'], r['x']) in v])), 4),
                                                      best_other_syl=round(float(np.mean([r['drive_max_other_syl'] for r in pd if (r['f'], r['x']) in v])), 4),
                                                      floor=brain.c['floor_in'])
                                             for k, v in sets.items() if k in ('base', 'new') and v}
        res['weights'][name] = wstats(brain)
        if rec['n3']:
            n3 = np.array(rec['n3'][-2000:])
            res['counts'][name] = dict(n3_mean=round(float(n3.mean()), 1), n3_max=int(n3.max()),
                                       p_sat=round(float((n3 >= 0.9 * brain.c['n3']).mean()), 4),
                                       p_empty=round(float((n3 == 0).mean()), 3))
        ps = res['probes'][name]
        print(arm, seed, name, '| ' + ' '.join(f"{k}: acc {v['acc']} first {v['first']} sil {v['silent']} garb {v['garble']}"
                                               for k, v in ps.items() if v), '| path12',
              {k: v[12] for k, v in res['path'][name].items() if k != 'n3_lag'}, '| spec',
              {k: (v['own_max'], v['other_max'], v['hub_syl'], v['hub_share']) for k, v in res['spec'][name].items()},
              '|', res['weights'][name], res['counts'].get(name, ''), f'{time.time() - t0:.0f}s', flush=True)
        return pr

    # ------------------------------------------------ test 1: sanity, learning off
    checkpoint('t1')
    prf = probe6(brain, ph, props, held, seen, seed, force=1)
    res['probes']['t1_forced_answer'] = summ(prf)
    lw = World(ph, seed, care='format')
    run_until(brain, lw, 60, learn=False)
    res['mech']['t1_live'] = mechanics6(lw, 0, 60)
    res['mech']['t1_live_sets'] = acc_by_set(lw, 0, 60, sets)
    lr = World(ph, seed, care='real')
    run_until(brain, lr, 60, learn=False)
    res['mech']['t1_live_real'] = mechanics6(lr, 0, 60)
    res['mech']['t1_live_real_sets'] = acc_by_set(lr, 0, 60, sets)
    print(arm, seed, 't1 forced', res['probes']['t1_forced_answer'], '| live', res['mech']['t1_live'], res['mech']['t1_live_sets'],
          '| real', res['mech']['t1_live_real'], res['mech']['t1_live_real_sets'], flush=True)
    if extra.get('only_t1') or tk.get('only_t1'):
        return finish(res, brain, arm, tag, seed, t0)
    pth = phase1.measure(bk, seed=seed, n_ctx=4, brain=(ph, world, brain), verbose=False, n_dmg_ctx=1)
    res['path']['t1_phase1'] = {k: pth[k] for k in ('same', 'cross', 'dec_q', 'dec_frame', 'within_J1', 'n3_lag')}
    # ------------------------------------------------ stage O: overhearing (learning on)
    ow = OverhearWorld(ph, seed, **wk)
    last = 0
    for m in marks:
        run_until(brain, ow, m, rec)
        res['overhear'][f'O{m}'] = overhear_mech(ow, last, m)
        checkpoint(f'O{m}')
        print('   overhear mech', res['overhear'][f'O{m}'], flush=True)
        last = m
    pth = phase1.measure(bk, seed=seed, n_ctx=4, brain=(ph, world, brain), verbose=False, n_dmg_ctx=1)
    res['path']['O_phase1'] = {k: pth[k] for k in ('same', 'cross', 'dec_q', 'dec_frame', 'within_J1', 'n3_lag')}
    # ------------------------------------------------ stage S: speaking (format caregiver), learning on
    sw = World(ph, seed + 300, care='format')
    sw.props, sw.held, sw.pairs = props, held, seen
    last = 0
    for m in sorted({min(100, n_speak), n_speak}):
        run_until(brain, sw, m, rec)
        res['mech'][f'S{m}'] = mechanics6(sw, last, m)
        res['mech'][f'S{m}_sets'] = acc_by_set(sw, last, m, sets)
        checkpoint(f'S{m}')
        print('   speak mech', res['mech'][f'S{m}'], res['mech'][f'S{m}_sets'], flush=True)
        last = m
    # ------------------------------------------------ test 3: untutored use
    sw.phase = 'free'
    n1 = sw.n_ex
    last = n1
    for m in sorted({300, n_free}):
        run_until(brain, sw, n1 + m, rec)
        res['mech'][f'F{m}'] = mechanics6(sw, last, n1 + m)
        res['mech'][f'F{m}_sets'] = acc_by_set(sw, last, n1 + m, sets)
        checkpoint(f'F{m}')
        last = n1 + m
    # amnesia: no hippocampus in this build; wipe the fastest learned store (L33) and re-probe
    keep33 = brain.L33.copy()
    brain.L33[:] = 0
    brain._norm()
    checkpoint('F_amnesia_L33')
    brain.L33[:] = keep33
    brain._norm()
    # ------------------------------------------------ test 4: corrections (colour / size answers changed)
    pr = probe6(brain, ph, props, held, seen, seed)
    good = {(r['f'], r['x']) for r in pr if r['correct']}
    crng = np.random.default_rng(seed + 3000)
    changed = []
    for S_ in (base, new):
        cand = [p for p in S_ if p[0] in (2, 3)]
        cand = sorted(cand, key=lambda p: (p not in good, crng.random()))
        changed += cand[:2 if spec_arm['think'] else 4]
    old = {p: answer(p[0], p[1], props) for p in changed}
    for (f, x) in changed:
        if f == 2:
            cur = props[x]['col']
            props[x]['col'] = COLS[(COLS.index(cur) + 1 + int(crng.integers(2))) % 3]
        else:
            props[x]['size'] = SIZES[1 - SIZES.index(props[x]['size'])]
    newA = {p: answer(p[0], p[1], props) for p in changed}
    side = {(2 if f == 3 else 3, x) for (f, x) in changed}      # same filler, other frame: unchanged on purpose
    sw.phase = 'teach'
    n0 = sw.n_ex
    run_until(brain, sw, n0 + n_corr, rec)
    lat = {}
    for p in changed:
        exs = [e for e in sw.exchanges if e['n'] > n0 and (e['f'], e['x']) == p]
        k, first, seq = 0, None, []
        for e in exs:
            syl = e['resp']['syl'] if e['resp'] is not None else None
            seq.append('N' if syl == newA[p] else ('O' if syl == old[p] else ('-' if syl is None else 'x')))
            if first is None and syl == newA[p]:
                first = k
            k += e['teach_new']
        lat[str(p)] = dict(set='base' if p in sets['base'] else 'new', was_correct=p in good,
                           first_new_after_teachings=first, n_asks=len(exs), seq=''.join(seq))
    pr2 = probe6(brain, ph, props, held, seen, seed)
    unchanged = [r for r in pr2 if (r['f'], r['x']) not in newA]
    res['correction'] = dict(changed=[list(p) for p in changed], latency=lat,
                             probe_new=round(float(np.mean([r['syl'] == newA[(r['f'], r['x'])] for r in pr2 if (r['f'], r['x']) in newA])), 3),
                             probe_old=round(float(np.mean([r['syl'] == old[(r['f'], r['x'])] for r in pr2 if (r['f'], r['x']) in old])), 3),
                             unchanged_before={k: round(float(np.mean([(r['f'], r['x']) in good for r in pr if (r['f'], r['x']) in v and (r['f'], r['x']) not in newA])), 3)
                                               for k, v in sets.items() if k != 'held'},
                             unchanged_after={k: round(float(np.mean([r['correct'] for r in unchanged if (r['f'], r['x']) in v])), 3)
                                              for k, v in sets.items() if k != 'held'})
    print(arm, seed, 'correction', res['correction'], flush=True)
    return finish(res, brain, arm, tag, seed, t0)


def finish(res, brain, arm, tag, seed, t0):
    res['time_s'] = round(time.time() - t0, 1)
    res['ticks'] = brain.t
    res['speech_route'] = {k: (round(v / max(1, brain.route['onsets']), 3) if k not in ('onsets', 'gated') else v)
                           for k, v in brain.route.items()}
    os.makedirs(os.path.join(HERE, 'out6'), exist_ok=True)
    fn = os.path.join(HERE, 'out6', f'{arm}{tag}_s{seed}.json')
    json.dump(res, open(fn, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('wrote', fn, res['time_s'], flush=True)
    return res


if __name__ == '__main__':
    arm, seed = sys.argv[1], int(sys.argv[2])
    kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
    marks = tuple(int(v) for v in str(kv.pop('marks', '100,300,1000,3000')).split(','))
    main(arm, seed, marks, kv.pop('n_speak', 300), kv.pop('n_free', 1000), kv.pop('n_corr', 300), kv)
