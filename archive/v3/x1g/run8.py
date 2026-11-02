"""X1g driver.  python3 -I run8.py <name> <seed> [key=value ...]  -> out8/<name>_s<seed>.json

Generation 0 = X1f G0 (hand-wired speech basics A1 + A2, quiet wait; RNpm + pm_ff, no PFC inhibitory neurons), then an
overhearing stage (OverhearWorld, 4 adult voices), learning on, checkpoints at `marks` overheard dialogues.  All probes
are frozen (learning off).

arm keys (all optional):
  N        L2 = L3 size (400 / 1600 / 6400); fan-ins fixed and local (fan12 24, fan23 30, M31 radius 0.3), patches ~33 units
  sparse   1: each L2 / L3 unit's threshold follows its own long-run activity toward k = 30 active (absolute) -> the
           active fraction falls as N grows;  0: X1f's thresholds (f2 0.18, f3 0.08), no homeostasis (X1f regime)
  kprop    with sparse=1: target k = kprop * N instead of 30 (proportional k)
  L33      1: L3 -> L3 learns (X1e form);  0: off
  a        two-layer split for top -> L1 and L3 -> L3: propagation = w / (out^a in^(1-a))
  rel      1: release by sharpness (divisive gain + max-relative, core8 rel1);  0: X1f absolute floor 0.35
  pc       1: POSITIVE CONTROL (X1f G0P): hand-wired held, question-specific paths for every seen pair (B), inner-hearing
           links C for the base half only; assemblies of m units (m=30 realistic; overlap allowed in a WM pool of
           wm_units); never-wired-C pairs ('new') must be learned by the Hebbian readout
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _x in ('x1f', 'x1d', 'x1c'):   # repo: core5/core6/common6/world6 from ../x1f, core4/phase1 from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
import core8
import common6
common6.Brain, common6.DEF = core8.Brain, core8.DEF        # common6's helpers build core8 brains
from common6 import make_brain, run_until, probe6, summ, spec, mechanics6, acc_by_set, GEN0, World, SYL
from world6 import OverhearWorld, overhear_mech
from wire8 import wire_speech, wire_thinking   # X1g's copy of X1f's wire.py (+ overlap option)
from run import parse
import phase1

SWAPS = [('a0.5_abs', dict(a31=0.5, rel1=False)), ('a0.5_rel', dict(a31=0.5, rel1=True)),
         ('a0.75_rel', dict(a31=0.75, rel1=True)), ('a1_abs', dict(a31=1.0, rel1=False)),
         ('a1_rel', dict(a31=1.0, rel1=True)), ('a1_rel_th.008', dict(a31=1.0, rel1=True, th1=0.008)),
         ('a1_rel_th.03', dict(a31=1.0, rel1=True, th1=0.03)),
         # post-hoc sensitivity (added after the seed-0 pilot showed mixtures silenced by A2): stricter max-relative rho1
         ('a1_rel_rho.8', dict(a31=1.0, rel1=True, rho1=0.8)), ('a0.5_rel_rho.8', dict(a31=0.5, rel1=True, rho1=0.8)),
         ('a0.5_rel_rho.9', dict(a31=0.5, rel1=True, rho1=0.9))]


def build_cfg(kv):
    N = int(kv.pop('N', 400))
    sparse = int(kv.pop('sparse', 1))
    kprop = kv.pop('kprop', None)
    L33 = int(kv.pop('L33', 1))
    a = float(kv.pop('a', 1.0))
    rel = int(kv.pop('rel', 1))
    bk = dict(GEN0, n2=N, n3=N, lat_fly=True, a31=a, a33=a, rel1=bool(rel))
    if not L33:
        bk['eta33'] = 0.0
    if sparse:
        k = int(round(kprop * N)) if kprop is not None else 30
        bk.update(homeo2=True, homeo3=True, k2=k, k3=k, homeo_tau=1000.0)
    bk.update(kv)
    return bk


def pred_specificity(pr):
    """X1e (run3.py) metric; fair = own_max - other1_max (one other question, unbiased); hub over lags 1-5"""
    S = np.array([r['allsc'] for r in pr])
    a1 = np.array([r['a1'] for r in pr])
    nq = len(pr)
    own = np.array([S[i, :, a1[i]] for i in range(nq)])
    o1 = []
    for i in range(nq):
        for k in range(1, nq):
            j = (i + k) % nq
            if a1[j] != a1[i]:
                o1.append(S[j, :, a1[i]].max())
                break
    best = S.argmax(2)
    hub = np.bincount(best[:, 1:6].ravel(), minlength=S.shape[2])
    has = S[:, 1:6].max(2) > 0
    return dict(own_max=float(own.max(1).mean()), other1_max=float(np.mean(o1)),
                fair=float(own.max(1).mean() - np.mean(o1)),
                top_is_answer=float(np.mean((best[:, 1:6] == a1[:, None])[has])) if has.any() else 0.0,
                hub_syl=SYL[int(hub.argmax())], hub_share=float(hub.max() / max(1, hub.sum())) if has.any() else 0.0)


def fair_with_null(pr, n_perm=200, seed=1):
    real = pred_specificity(pr)
    rng = np.random.default_rng(seed)
    nul = []
    for _ in range(n_perm):
        perm = rng.permutation(len(pr))
        nul.append(pred_specificity([dict(r, a1=pr[perm[i]]['a1']) for i, r in enumerate(pr)])['fair'])
    real.update(null_mean=float(np.mean(nul)), null_sd=float(np.std(nul)), null_95=float(np.percentile(nul, 95)))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in real.items()}


def decode_lag(pr_a, pr_b, idx, lags=(0, 3, 6, 9, 12)):
    """X1f: nearest top-layer state (Jaccard) between two noise repetitions of the muted probe, among all 32 questions"""
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


def within_motion(pr):
    """within-path motion: mean Jaccard of the L3 state between consecutive lags (1 = stuck), and P(empty) per lag"""
    L = np.array([r['l3'] for r in pr]).astype(bool)          # (q, lag, n3)
    inter = (L[:, :-1] & L[:, 1:]).sum(2)
    uni = (L[:, :-1] | L[:, 1:]).sum(2)
    J = np.where(uni > 0, inter / np.maximum(uni, 1), np.nan)
    return dict(within_J1=[round(float(x), 3) if not np.isnan(x) else None for x in np.nanmean(J, 0)],
                within_J1_mean=round(float(np.nanmean(J)), 3) if np.isfinite(J).any() else None,
                n3_lag=np.round(L.sum(2).mean(0), 1).tolist(), p_empty_lag=np.round((L.sum(2) == 0).mean(0), 3).tolist())


def drive_stats(brain, ph, pr):
    """learned top -> L1 drive (pre-threshold, P31 only) into each syllable's L1 copies, lags 1-6 after question end"""
    from common6 import syl_lines
    lines = [syl_lines(brain, ph, s) for s in range(len(ph.content))]
    ans, wrong, top, share, top1, topm, w_a2 = [], [], [], [], [], [], []
    for r in pr:
        D = r['D'][1:7]                                        # (lag, n1)
        Ds = np.stack([D[:, l].mean(1) for l in lines], 1)     # (lag, syl)
        a = r['a1']
        da = Ds[:, a].max()
        dw = np.delete(Ds, a, 1).max()
        ans.append(da)
        wrong.append(dw)
        top.append(float(da > dw))
        (top1 if len(r['A']) == 1 else topm).append(float(da > dw))
        if len(r['A']) > 1:
            bw = int(np.argmax(np.where(np.arange(Ds.shape[1]) == a, -1, Ds.max(0))))
            w_a2.append(float(bw == r['A'][1]))
        tot = D.sum(1).max()
        share.append(float(da / tot) if tot > 0 else 0.0)
    return dict(ans=round(float(np.mean(ans)), 4), best_wrong=round(float(np.mean(wrong)), 4),
                answer_is_top=round(float(np.mean(top)), 3), ans_share_per_copy=round(float(np.mean(share)), 4),
                top_single=round(float(np.mean(top1)), 3) if top1 else None, top_multi=round(float(np.mean(topm)), 3) if topm else None,
                best_wrong_is_a2=round(float(np.mean(w_a2)), 3) if w_a2 else None)


def probe_drive(brain, ph, props, held, pairs, seed):
    """muted frozen probe (as probe6, question in voice C1) recording the learned top -> L1 drive vector at every lag"""
    from common6 import question, answer, SID, render, C1
    res = []
    rng = np.random.default_rng(seed)
    snap = brain.snapshot()
    for (f, x) in pairs:
        brain.restore(snap)
        Q = question(f, x)
        sched = [None] * 6 + [SID[s] for s in Q for _ in range(3)]
        q_end = len(sched)
        sched += [None] * 16
        D = []
        for t, cg in enumerate(sched):
            mus = brain.step(render(ph, [('s', cg, C1, 1.0)] if cg is not None else [], rng), 0.0, learn=False)
            brain.mus = np.zeros_like(mus)
            a3 = np.nonzero(brain.l3)[0]
            D.append(brain.P31[a3].sum(0) if len(a3) else np.zeros(brain.n1, np.float32))
        res.append(dict(a1=SID[answer(f, x, props)[0]], A=[SID[c] for c in answer(f, x, props)],
                        D=np.array(D[q_end - 1: q_end + 15]), f=f, x=x))
    brain.restore(snap)
    return res


def main(name, seed, marks, kv):
    t0 = time.time()
    pc = int(kv.pop('pc', 0))
    m_asm = int(kv.pop('m', 30))
    wm_units = kv.pop('wm_units', None)
    overlap = int(kv.pop('overlap', 1))
    n_live = int(kv.pop('n_live', 60))
    do_phase1 = int(kv.pop('phase1', 1))
    swaps = int(kv.pop('swaps', 1))
    bk = build_cfg(dict(kv))
    ph, brain = make_brain(seed, bk)
    world = World(ph, seed, care='format')
    props, held = world.props, world.held
    allp = [(f, x) for f in range(4) for x in range(8)]
    seen = list(world.pairs)
    rng = np.random.default_rng(seed + 4000)
    perm = rng.permutation(len(seen))
    base = [seen[i] for i in sorted(perm[:len(seen) // 2])] if pc else []
    new = [p for p in seen if p not in base]
    sets = dict(seen=set(seen), held=set(held))
    if pc:
        sets.update(base=set(base), new=set(new))
    reg = [wire_speech(brain, ph)]
    if pc:
        wmu = int(wm_units) if wm_units is not None else bk['n3'] // 2
        reg.append(wire_thinking(brain, ph, props, seen, seed, m=m_asm, say_pairs=set(base), overlap=bool(overlap),
                                 wm_units=wmu))
    res = dict(name=name, seed=seed, cfg=dict(brain={k: v for k, v in bk.items()}, pc=pc, m=m_asm, overlap=overlap,
                                              wm_units=wm_units),
               registry=reg, held=sorted(held), base=base, new=new, ck={}, t1={})
    rec = dict(n2=[], n3=[], np=[])
    idx = {k: [allp.index(p) for p in v] for k, v in sets.items()}

    def run_rec(world_, n_target):
        while not (world_.n_ex >= n_target and world_.ex is not None and world_.ex['state'] == 'gap'):
            mus = brain.step(world_.sound, getattr(world_, 'r_last', 0.0))
            snd, r = world_.step(mus)
            world_.r_last = r
            if brain.t % 5 == 0:
                rec['n2'].append(int(brain.l2.sum()))
                rec['n3'].append(int(brain.l3.sum()))
                rec['np'].append(int(brain.pred.sum()))

    def checkpoint(cname):
        ck = {}
        pr = probe6(brain, ph, props, held, allp, seed)
        ck['probe'] = {k: summ([r for r in pr if (r['f'], r['x']) in v]) for k, v in sets.items()}
        ck['pairs'] = [[r['f'], r['x'], ''.join(r['A']), ''.join(r['syl']) if r['syl'] else None] for r in pr]
        ck['spec_x1d'] = spec([r for r in pr if (r['f'], r['x']) in sets['seen']])
        ck['fair'] = {k: fair_with_null([r for r in pr if (r['f'], r['x']) in v]) for k, v in sets.items() if len(v) > 3}
        pa = probe6(brain, ph, props, held, allp, seed + 1, mute=True, rec_l3=True)
        pb = probe6(brain, ph, props, held, allp, seed + 2, mute=True, rec_l3=True)
        ck['path'] = {k: decode_lag(pa, pb, v) for k, v in idx.items() if v}
        ck['path']['all'] = decode_lag(pa, pb, list(range(32)))
        ck['motion'] = within_motion(pa)
        pdv = probe_drive(brain, ph, props, held, [p for p in allp if p in sets['seen']], seed)
        ck['drive'] = {k: drive_stats(brain, ph, [r for r in pdv if (r['f'], r['x']) in v])
                       for k, v in sets.items() if k != 'held' and v}
        if swaps:
            keep = {k: brain.c[k] for k in ('a31', 'rel1', 'th1', 'rho1')}
            ck['swap'] = {}
            for sn, sw in SWAPS:
                brain.c.update(sw)
                brain._norm()
                ps = probe6(brain, ph, props, held, allp, seed)
                ck['swap'][sn] = {k: dict(summ([r for r in ps if (r['f'], r['x']) in v]),
                                          fair=round(pred_specificity([r for r in ps if (r['f'], r['x']) in v])['fair'], 4))
                                  for k, v in sets.items() if k != 'held' and len(v) > 3}
                ck['swap'][sn]['held'] = summ([r for r in ps if (r['f'], r['x']) in sets['held']])
                ck['swap'][sn]['pairs'] = [[r['f'], r['x'], ''.join(r['syl']) if r['syl'] else None] for r in ps]
                brain.c.update(keep)
            brain._norm()
        L31 = brain.L31
        L33 = brain.L33_true()
        ck['weights'] = dict(L31_max=round(float(L31.max()), 3), L31_pos=round(float((L31 > 1e-3).mean()), 4),
                             L33_pos=round(float((L33 > 1e-3).mean()), 4), L33_max=round(float(L33.max()), 3))
        if rec['n3']:
            n3 = np.array(rec['n3'][-4000:])
            n2 = np.array(rec['n2'][-4000:])
            npr = np.array(rec['np'][-4000:])
            ck['counts'] = dict(n3_mean=round(float(n3.mean()), 1), n3_p90=float(np.percentile(n3, 90)), n3_max=int(n3.max()),
                                p_sat=round(float((n3 >= 0.9 * bk['n3']).mean()), 4), p_empty=round(float((n3 == 0).mean()), 3),
                                n2_mean=round(float(n2.mean()), 1), n2_max=int(n2.max()), pred_mean=round(float(npr.mean()), 1))
        ck['time_s'] = round(time.time() - t0, 1)
        res['ck'][cname] = ck
        p = ck['probe']
        print(name, seed, cname, '| ' + ' '.join(f"{k}: acc {v['acc']} first {v['first']} sil {v['silent']} garb {v['garble']}"
                                                for k, v in p.items() if v),
              '| fair', {k: (v['fair'], v['null_95'], v['hub_syl'], round(v['hub_share'], 2)) for k, v in ck['fair'].items()},
              '| path all', ck['path']['all'], 'J', ck['motion']['within_J1_mean'],
              '| drive', ck['drive'], '|', ck['weights'], ck.get('counts', ''), f"{ck['time_s']}s", flush=True)
        if swaps:
            print('   swaps', {sn: (v.get('seen', v.get('new', {})).get('first'), v.get('new', {}).get('first')) for sn, v in ck['swap'].items()}, flush=True)

    def phase1_path(tag):
        pth = phase1.measure(bk, seed=seed, n_ctx=4, brain=(ph, world, brain), verbose=False, n_dmg_ctx=1)
        res['path_ctx_' + tag] = {k: pth[k] for k in ('same', 'cross', 'cross_same_frame', 'dec_q', 'dec_q_win3', 'dec_frame',
                                                      'dec_fill_given_frame', 'within_J1', 'n3_lag', 'p_empty_lag',
                                                      'motion_long_silence')}
        d = pth['dec_q']
        print(name, seed, 'phase1', tag, 'dec_q lag0/6/12', d[0], d[6], d[12], 'win3 lag0/6/12', pth['dec_q_win3'][0], pth['dec_q_win3'][6],
              pth['dec_q_win3'][12], 'same/cross@12', pth['same'][12], pth['cross'][12],
              'within_J1@0-11', round(float(np.mean(pth['within_J1'][:12])), 3), flush=True)

    # ---------------- test 1: newborn with speech basics (learning off)
    prf = probe6(brain, ph, props, held, seen, seed, force=1)
    res['t1']['forced_answer'] = summ(prf)
    checkpoint('t1')
    if do_phase1:
        phase1_path('t1')
    # ---------------- overhearing (learning on)
    ow = OverhearWorld(ph, seed)
    last = 0
    for mk in marks:
        run_rec(ow, mk)
        res['ck'].setdefault('_overhear', {})[f'O{mk}'] = overhear_mech(ow, last, mk)
        print('   overhear mech', mk, res['ck']['_overhear'][f'O{mk}'], flush=True)
        checkpoint(f'O{mk}')
        last = mk
        json.dump(res, open(out_fn(name, seed), 'w'), ensure_ascii=False, default=jdef)
    if do_phase1:
        phase1_path('end')
    # ---------------- live world (frozen): talk-over, failure to yield, garble, live accuracy
    lw = World(ph, seed, care='format')
    run_until(brain, lw, n_live, learn=False)
    res['live'] = dict(mech=mechanics6(lw, 0, n_live), sets=acc_by_set(lw, 0, n_live, sets))
    lr = World(ph, seed, care='real')
    run_until(brain, lr, n_live, learn=False)
    res['live_real'] = dict(mech=mechanics6(lr, 0, n_live), sets=acc_by_set(lr, 0, n_live, sets))
    print(name, seed, 'live', res['live'], '| real', res['live_real'], flush=True)
    res['time_s'] = round(time.time() - t0, 1)
    res['ticks'] = brain.t
    res['speech_route'] = {k: (round(v / max(1, brain.route['onsets']), 3) if k not in ('onsets', 'gated') else v)
                           for k, v in brain.route.items()}
    json.dump(res, open(out_fn(name, seed), 'w'), ensure_ascii=False, default=jdef)
    print('wrote', out_fn(name, seed), res['time_s'], flush=True)
    return res


def jdef(o):
    if hasattr(o, 'item'):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def out_fn(name, seed):
    os.makedirs(os.path.join(HERE, 'out8'), exist_ok=True)
    return os.path.join(HERE, 'out8', f'{name}_s{seed}.json')


if __name__ == '__main__':
    name, seed = sys.argv[1], int(sys.argv[2])
    kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
    marks = tuple(int(v) for v in str(kv.pop('marks', '300,1000,3000,10000')).split(','))
    main(name, seed, marks, kv)
