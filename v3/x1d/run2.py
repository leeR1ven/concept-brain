"""X1d phase 2 driver: X1c's teaching protocol (teach -> correction -> untutored use) on core4.Brain, plus
  * fix of X1c's mark bug (n_free = 1000 repeated the 1000 mark);
  * caregiver-shaping world (graded praise of close attempts; world property only);
  * specificity of the top->L1 predictions (is the answer's L1 code predicted after its question, not after others?);
  * chain-break diagnosis: reproducible path (phase-1 metrics on the trained, frozen brain) -> predicted inner
    hearing of the answer -> motor release; plus a hand-wired diagnostic arm that forces the answer prediction.
  python3 -I run2.py <arm> <seed> [key=value ...]   -> out2/<arm><tag>_s<seed>.json
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'x1c'))   # repo: core3.py, world.py, run.py from ../x1c
from world import Physics, World, render, question, answer, SID, SYL, OWN, C1, COLS, SIZES  # noqa
from core4 import Brain
from run import birth_rates, mechanics, thought_motion, empty_speech, summarize_probe, parse
import phase1

ARMS = {}          # filled from regimes.py
try:
    from regimes import ARMS  # noqa
except ImportError:
    pass


class ShapingWorld(World):
    """the caregiver is pleased by close attempts (designer 11-01): full answer > right first syllable / any answer
    syllable said clearly in the answer window > any clear syllable > garble (never).  Standards rise slowly with age."""
    P = dict(full=0.8, first=0.5, ans_syl=0.35, clear=0.15)
    age_tau = 2500.0

    def _react(self, ex, u, A, voice, f, x):
        rng = self.rng
        ex['state'] = 'react'
        yield from self._wait(int(rng.integers(2, 4)))
        syl = u['syl']
        correct = syl == A
        clear = [s for s in syl if s != '~']
        if correct:
            kind = 'full'
        elif syl[:1] == A[:1]:
            kind = 'first'
        elif any(s in A for s in clear):
            kind = 'ans_syl'
        elif clear:
            kind = 'clear'
        else:
            kind = None
        p = 0.0 if kind is None else self.P[kind] * (1.0 if kind == 'full' else np.exp(-self.n_ex / self.age_tau))
        ex['shape_kind'] = kind
        if rng.random() < p:
            self.r_pending += 1.0                                      # pleased
            ex['pleased'] = True
        if f == 1 and self.props[x]['food'] and syl[:1] == ['要']:
            self.r_pending += self.hunger
            self.hunger = 0.0
        if correct:
            yield from self._say(['对'] + A, voice)
        else:
            if not ex.get('pleased') and rng.random() < 0.25:
                self.r_pending -= 1.0                                  # upset
            yield from self._say(question(f, x), voice)                # recast
            yield from self._wait(2)
            yield from self._say(A, voice)
        ex['teach_new'] += 1


def count_pleased(world, a, b):
    """pleased reactions per exchange in (a, b]; X1c's world does not mark them, so it is inferred from the
    reaction (correct or right first syllable in the exact-only world)."""
    exs = [e for e in world.exchanges if a < e['n'] <= b]
    n = max(1, len(exs))
    pl = sum(1 for e in exs if e.get('pleased'))
    return round(pl / n, 4)


def speech_quality(world, a, b):
    """answer-window speech: garble / clear syllables / answer-syllable hits / first syllable right."""
    exs = [e for e in world.exchanges if a < e['n'] <= b]
    n = max(1, len(exs))
    resp = [e for e in exs if e['resp'] is not None]
    garb = sum(1 for e in resp if '~' in e['resp']['syl'])
    clear = sum(1 for e in resp if any(s != '~' for s in e['resp']['syl']))
    hit = sum(1 for e in resp if any(s in e['A'] for s in e['resp']['syl'] if s != '~'))
    first = sum(1 for e in resp if e['resp']['syl'][:1] == e['A'][:1])
    full = sum(1 for e in resp if e['resp']['syl'] == e['A'])
    return dict(resp=round(len(resp) / n, 3), garble=round(garb / n, 3), clear=round(clear / n, 3),
                ans_syl_hit=round(hit / n, 3), first=round(first / n, 3), full=round(full / n, 3))


def probe2(brain, ph, world, pairs, seed=0, wait=16, force=None):
    """X1c's frozen probe + per-tick predicted L1 content (inner hearing) and mouth.  force=(lag, gain): hand-wired
    DIAGNOSTIC: at that lag after the question, the answer's first-syllable L1 content units are switched on as
    predicted (inner hearing), to test whether the motor side then releases the answer."""
    snap = brain.snapshot()
    rng = np.random.default_rng(seed)
    out = []
    C = ph.content
    for (f, x) in pairs:
        brain.restore(snap)
        Q, A = question(f, x), answer(f, x, world.props)
        sched = [None] * 6 + [SID[s] for s in Q for _ in range(3)]
        q_end = len(sched)
        sched += [None] * wait
        mprev = None
        utts, cur, sil, last = [], None, 99, None
        predc = []
        mouth = []
        for t, cg in enumerate(sched + [None, None, None]):
            srcs = []
            if mprev is not None:
                srcs.append((mprev[0], mprev[1], OWN, 1.0))
            if cg is not None:
                srcs.append(('s', cg, C1, 1.0))
            om = getattr(brain, '_oracle_map', None)
            if om is not None and 0 <= t - q_end < om[(f, x)].shape[0]:
                brain._oracle = om[(f, x)][t - q_end]
            if force is not None and 0 <= t - (q_end - 1 + force[0]) < 3:          # 3 ticks, like a heard syllable
                a1 = SID[A[0]]
                lines = np.nonzero(brain.l1_content & np.isin(brain.l1_feat, np.nonzero(C[a1])[0]))[0]
                brain._force = lines
            mus = brain.step(render(ph, srcs, rng), 0.0, learn=False)
            v = np.zeros(ph.n_c, bool)
            idx = np.nonzero(brain.pred & brain.l1_content)[0]
            v[brain.l1_feat[idx]] = True
            predc.append(v)
            m = ph.decode(mus)
            mouth.append(m)
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
        P = np.array(predc[q_end - 1: q_end + 15])                     # lag 0 .. 15
        a1 = SID[A[0]]
        score = (P & C[a1]).sum(1) / C[a1].sum()                       # fraction of the answer's 1st-syllable code
        allsc = (P[:, None, :] & C[None, :, :]).sum(2) / C.sum(1)[None, :]   # (lag, syllable)
        out.append(dict(f=f, x=x, held=(f, x) in world.held, A=A, syl=syl, lat=(resp[0]['t0'] - q_end) if resp else None,
                        correct=syl == A, first=bool(syl) and syl[0] == A[0], echo=bool(syl) and syl[0] in Q,
                        silent=syl is None, garble=bool(syl) and '~' in syl, talk=len(during) > 0,
                        a1=a1, score=score.tolist(), allsc=allsc, n_pred=P.sum(1).tolist(),
                        mouth_ans=[(m is not None and m[0] != 'g' and m[1] == a1) for m in mouth[q_end - 1: q_end + 15]]))
    brain.restore(snap)
    return out


def pred_specificity(pr):
    """does the answer's first-syllable code become predicted after its question and not after the others?"""
    S = np.array([r['allsc'] for r in pr])                          # (q, lag, syl)
    a1 = np.array([r['a1'] for r in pr])
    nq, nl, ns = S.shape
    own = np.array([S[i, :, a1[i]] for i in range(nq)])            # (q, lag)
    other = np.array([[S[j, :, a1[i]] for j in range(nq) if a1[j] != a1[i]] for i in range(nq)], dtype=object)
    oth = np.array([np.mean(np.stack(o), 0) for o in other])       # (q, lag): same syllable after other questions
    best = S.argmax(2)                                             # most predicted syllable per lag
    top_is_a1 = np.mean(best == a1[:, None], 0)
    hub = np.bincount(best[:, 10:14].ravel(), minlength=ns)
    return dict(own=np.round(own.mean(0), 3).tolist(), other=np.round(oth.mean(0), 3).tolist(),
                top_is_answer=np.round(top_is_a1, 3).tolist(),
                own_max=round(float(own.max(1).mean()), 3), other_max=round(float(oth.max(1).mean()), 3),
                hub_syl=SYL[int(hub.argmax())], hub_share=round(float(hub.max() / max(1, hub.sum())), 3),
                n_pred=np.round(np.mean([r['n_pred'] for r in pr], 0), 1).tolist())


def release_stats(pr, thr=0.6):
    """motor link: when the answer's first syllable is predicted (score >= thr) at some lag, is it said within 3 ticks?"""
    n_pred = n_said = 0
    for r in pr:
        sc, ma = r['score'], r['mouth_ans']
        for k in range(len(sc) - 3):
            if sc[k] >= thr:
                n_pred += 1
                n_said += any(ma[k:k + 4])
                break
    return dict(q_with_answer_pred=round(n_pred / max(1, len(pr)), 3), said_when_pred=round(n_said / max(1, n_pred), 3))


def light(pr):
    keys = ('f', 'x', 'held', 'A', 'syl', 'lat', 'correct', 'first')
    return [{k: r[k] for k in keys} for r in pr]


def oracle_codes(seed, n3, k, lags=16):
    """DIAGNOSTIC: a perfectly reproducible, question-specific, moving top-layer path: a fixed random code of k units
    for every (question, lag after the question).  Nothing about the answers is wired."""
    rng = np.random.default_rng(seed * 31 + 7)
    codes = {}
    for f in range(4):
        for x in range(8):
            C = np.zeros((lags, n3), bool)
            for l in range(lags):
                C[l, rng.choice(n3, k, replace=False)] = True
            codes[(f, x)] = C
    return codes


def run_until(brain, world, n_target, rec=None):
    om = getattr(brain, '_oracle_map', None)
    while not (world.n_ex >= n_target and world.ex is not None and world.ex['state'] == 'gap'):
        if om is not None:                                     # oracle path after every heard question line
            st = world._orc if hasattr(world, '_orc') else None
            if st is not None and st[1] < om[st[0]].shape[0]:
                brain._oracle = om[st[0]][st[1]]
        mus = brain.step(world.sound, getattr(world, 'r_last', 0.0))
        snd, r = world.step(mus)
        if om is not None:
            cur = world.cg_now
            line = getattr(world, '_line', [])
            if cur is not None:
                if not line or line[-1] != cur[0]:
                    line = line + [cur[0]]
                world._line = line
                world._orc = None
            else:
                ex = world.ex
                if line and ex is not None and not ex.get('overheard') and \
                        [SYL[s] for s in line] == question(ex['f'], ex['x']):
                    world._orc = ((ex['f'], ex['x']), 0)
                elif getattr(world, '_orc', None) is not None:
                    world._orc = (world._orc[0], world._orc[1] + 1)
                world._line = []
        world.r_last = r
        if r > 0 and world.ex is not None:
            world.ex['pleased'] = True                                # mark for the exact-only world too
        if rec is not None and world.empty:
            rec['empty_ticks'] += 1
            if len(rec['pas']) < 4000:
                rec['pas'].append(brain.l3.copy())
        if rec is not None and brain.t % 10 == 0:
            rec['n3'].append(int(brain.l3.sum()))
            rec['n2'].append(int(brain.l2.sum()))
            rec['np'].append(int(brain.pred.sum()))


def main(arm, seed, n_birth=200, n_teach=600, n_corr=300, n_free=1000, extra=None):
    t0 = time.time()
    spec = dict(ARMS[arm])
    bk = dict(spec.get('b', {}))
    wk = dict(care=spec.get('care', 'format'), shaping=spec.get('shaping', False), oracle=spec.get('oracle', False))
    extra = dict(extra or {})
    tag = extra.pop('tag', '')
    for k, v in extra.items():
        if k.startswith('w_'):
            wk[k[2:]] = v
        else:
            bk[k] = v
    ph = Physics()
    Wc = ShapingWorld if wk['shaping'] else World
    world = Wc(ph, seed, care=wk['care'])
    # birth period (homeostasis only, all Hebbian / value rates 0), as in phase 1
    nol = dict(bk)
    keep = {k: bk.get(k) for k in phase1.NOLEARN}
    nol.update(phase1.NOLEARN)
    brain = Brain(ph, seed, rates=birth_rates(ph, world), **nol)
    bw = World(ph, seed + 50, care=wk['care'])
    if n_birth:
        run_until(brain, bw, n_birth)
    from core4 import DEF
    for k in phase1.NOLEARN:
        brain.c[k] = keep[k] if keep[k] is not None else DEF[k]
    allp = [(f, x) for f in range(4) for x in range(8)]
    if wk['oracle']:
        brain._oracle_map = oracle_codes(seed, brain.c['n3'], brain.c['k3'])
    res = dict(arm=arm, seed=seed, cfg=dict(world=wk, brain=bk), held=sorted(world.held), probes={}, mech={}, empty={},
               motion={}, spec={}, release={}, quality={}, pleased={}, path={})
    rec = dict(empty_ticks=0, pas=[], n3=[], n2=[], np=[])

    def checkpoint(tagm, a, b):
        pr = probe2(brain, ph, world, allp, seed)
        res['probes'][tagm] = summarize_probe(pr)
        res['spec'][tagm] = pred_specificity(pr)
        res['release'][tagm] = release_stats(pr)
        res['quality'][tagm] = speech_quality(world, a, b)
        res['pleased'][tagm] = count_pleased(world, a, b)
        res['mech'][tagm] = mechanics(world, a, b)
        sp = res['spec'][tagm]
        print(arm, seed, tagm, 'probe', res['probes'][tagm]['seen'], 'held', res['probes'][tagm]['held']['acc'],
              '| spec own/other max %.2f/%.2f top=ans@12 %.2f hub %s %.2f' % (sp['own_max'], sp['other_max'],
              sp['top_is_answer'][12], sp['hub_syl'], sp['hub_share']), '| rel', res['release'][tagm],
              '| q', res['quality'][tagm], 'pleased', res['pleased'][tagm],
              '| n3 %.0f max %d n2 %.0f pred %.1f' % (np.mean(rec['n3'][-500:]), max(rec['n3'][-500:]), np.mean(rec['n2'][-500:]),
                                                     np.mean(rec['np'][-500:])),
              '| L31 max %.2f mean %.4f' % (brain.L31.max(), brain.L31.mean()), f'{time.time() - t0:.0f}s', flush=True)
        return pr

    # ---------------- teaching
    last = 0
    marks = [m for m in (100, 300, 600, 1000, 1500) if m < n_teach] + [n_teach]
    for m in marks:
        run_until(brain, world, m, rec)
        pr = checkpoint(f'teach{m}', last, m)
        last = m
    res['empty']['teach'] = empty_speech(world, 0, n_teach, rec['empty_ticks'])
    res['motion']['teach'] = thought_motion(rec['pas'])
    res['probe_detail_teach'] = light(pr)
    # reproducible path on the trained brain (learning frozen inside measure)
    snap = brain.snapshot()
    om = getattr(brain, '_oracle_map', None)
    brain._oracle_map = None
    pth = phase1.measure(bk, seed=seed, n_ctx=4, brain=(ph, world, brain), verbose=False, n_dmg_ctx=1)
    brain.restore(snap)
    brain._oracle_map = om
    res['path']['teach'] = {k: pth[k] for k in ('same', 'cross', 'cross_same_frame', 'dec_q', 'dec_q_win3', 'same_win3', 'cross_win3', 'dec_frame',
                                                'dec_fill_given_frame', 'noise_same', 'within_J1', 'n3_lag', 'damage',
                                                'motion_long_silence', 'mouth_lag', 'pred_lag')}
    print(arm, seed, 'PATH same', pth['same'][:13:2], 'cross', pth['cross'][:13:2], 'decq', pth['dec_q'][:13:2],
          'within', pth['within_J1'][:12:2], flush=True)
    # hand-wired DIAGNOSTIC: force the answer's inner hearing at lag 12; does the motor side release it?
    for lagf in (6, 12):
        prf = probe2(brain, ph, world, allp, seed, force=(lagf, 1.0))
        res['release'][f'FORCED_lag{lagf}'] = dict(summarize_probe(prf)['seen'], **release_stats(prf))
    print(arm, seed, 'DIAG forced inner hearing', res['release']['FORCED_lag6'], res['release']['FORCED_lag12'], flush=True)
    # ---------------- correction
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
    run_until(brain, world, n0 + n_corr, rec)
    pr = checkpoint('corr', n0, world.n_ex)
    res['correction'] = dict(changed=[list(p) for p in changed],
                             probe_new=round(float(np.mean([r['syl'] == new[(r['f'], r['x'])] for r in pr if (r['f'], r['x']) in new])), 3),
                             probe_old=round(float(np.mean([r['syl'] == old[(r['f'], r['x'])] for r in pr if (r['f'], r['x']) in old])), 3))
    # ---------------- untutored use (mark bug fixed: unique, increasing marks)
    world.phase = 'free'
    n1 = world.n_ex
    rec['pas'] = []
    rec['empty_ticks'] = 0
    last = n1
    for m in sorted({x for x in (300, 1000, n_free) if x <= n_free}):
        run_until(brain, world, n1 + m, rec)
        checkpoint(f'free{m}', last, n1 + m)
        last = n1 + m
    res['empty']['free'] = empty_speech(world, n1, world.n_ex, rec['empty_ticks'])
    res['motion']['free'] = thought_motion(rec['pas'])
    res['time_s'] = round(time.time() - t0, 1)
    res['speech_route'] = {k: (round(v / max(1, brain.route['onsets']), 3) if k != 'onsets' else v) for k, v in brain.route.items()}
    res['ms_per_tick'] = round(1000 * res['time_s'] / max(1, brain.t), 3)
    res['layer_counts'] = {k: dict(mean=round(float(np.mean(v)), 1), max=int(np.max(v)), p_zero=round(float(np.mean(np.array(v) == 0)), 3),
                                   p_sat=round(float(np.mean(np.array(v) >= 0.9 * s)), 3))
                           for k, v, s in (('n2', rec['n2'], brain.c['n2']), ('n3', rec['n3'], brain.c['n3']))}
    os.makedirs(os.path.join(HERE, 'out2'), exist_ok=True)
    fn = os.path.join(HERE, 'out2', f'{arm}{tag}_s{seed}.json')
    json.dump(res, open(fn, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('wrote', fn, res['time_s'], flush=True)
    return res


if __name__ == '__main__':
    arm, seed = sys.argv[1], int(sys.argv[2])
    kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
    nb = kv.pop('n_birth', 200)
    nt = kv.pop('n_teach', 600)
    nc = kv.pop('n_corr', 300)
    nf = kv.pop('n_free', 1000)
    main(arm, seed, nb, nt, nc, nf, kv)
