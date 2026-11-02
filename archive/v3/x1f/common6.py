"""X1f shared helpers: brain birth, frozen probes (with a forced whole-answer inner-hearing diagnostic), world runner."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _x in ('x1d', 'x1c'):   # repo: core4/phase1/run2/regimes from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
from world import Physics, World, render, question, answer, SID, SYL, OWN, C1, C2  # noqa
from core6 import Brain, DEF
from run import birth_rates, summarize_probe  # noqa
import phase1

# generation-0 base: X1d RNpm (no PFC inhibitory neurons, X3 patch-mean, held-neuron rule everywhere) with the top
# layer's internal links learning by the same rule in X1e's form (RNpm_L2: ARCH two-layer weights, g33 = 2 keeps the
# birth lateral strength of RNpm, f3 keeps RNpm's threshold level 5.4)
RNPM = dict(act='frac', kappa=0.0, g_loop=0.0, w33=0.1, f1=0.0, f2=0.22, f3=0.3, nrm31='arch', g31=1.0, floor_in=0.35,
            rule='onset', post='ext', pm=1.0)
L33 = dict(eta33=0.02, rule33='onset', nrm33='arch', a33=0.5, g33=2.0, f3=0.3, dec_33=0.95)
BASE = dict(RNPM, **L33)
# generation 0 (PLAN.md): thresholds so every syllable reaches the top layer; speech basics knobs (A2 gate, no latch)
GEN0 = dict(BASE, f2=0.18, f3=0.08, pm_ff=1.0, c_mb=0.0, g_dir=0.25, gate_lo=3, gate_hi=5)


def make_brain(seed, bk, n_birth=200, care='format'):
    """birth: chains from the birth listening period, then n_birth world exchanges with every Hebbian / value rate at
    0 (X1d phase-1 / run2 protocol), then the real rates are given back."""
    ph = Physics()
    w0 = World(ph, seed, care=care)
    nol = dict(bk)
    keep = {k: bk.get(k) for k in phase1.NOLEARN}
    nol.update(phase1.NOLEARN)
    brain = Brain(ph, seed, rates=birth_rates(ph, w0), **nol)
    if n_birth:
        bw = World(ph, seed + 50, care=care)
        run_until(brain, bw, n_birth)
    for k in phase1.NOLEARN:
        brain.c[k] = keep[k] if keep[k] is not None else DEF[k]
    return ph, brain


def run_until(brain, world, n_target, rec=None, learn=None):
    while not (world.n_ex >= n_target and world.ex is not None and world.ex['state'] == 'gap'):
        mus = brain.step(world.sound, getattr(world, 'r_last', 0.0), learn=learn)
        snd, r = world.step(mus)
        world.r_last = r
        if r > 0 and world.ex is not None:
            world.ex['pleased'] = True
        if rec is not None and brain.t % 10 == 0:
            rec.setdefault('n3', []).append(int(brain.l3.sum()))
            rec.setdefault('np', []).append(int(brain.pred.sum()))


def syl_lines(brain, ph, s):
    """L1 content lines (chain copies) of syllable s = its inner-hearing units"""
    return np.nonzero(brain.l1_content & np.isin(brain.l1_feat, np.nonzero(ph.content[s])[0]))[0]


def probe6(brain, ph, props, held, pairs, seed=0, wait=16, voice=C1, force=None, mute=False, rec_l3=False, force_mix=False,
           drive=False):
    """frozen probe (learning off): 6 silent ticks, the question (3 ticks per syllable), `wait` silent ticks; the own
    mouth output is heard one tick later.  force=lag: DIAGNOSTIC: from that lag on, the whole answer is made inner-heard
    (each syllable's L1 content lines for 3 ticks, in order) -- tests the sound -> action map (ARCH 2.11 step 1)."""
    snap = brain.snapshot()
    rng = np.random.default_rng(seed)
    out = []
    C = ph.content
    for (f, x) in pairs:
        brain.restore(snap)
        Q, A = question(f, x), answer(f, x, props)
        sched = [None] * 6 + [SID[s] for s in Q for _ in range(3)]
        q_end = len(sched)
        sched += [None] * wait
        fseq = {}
        if force is not None:
            t0 = q_end - 1 + force
            for i, s in enumerate(A):
                for k in range(3):
                    fseq[t0 + 3 * i + k] = syl_lines(brain, ph, SID[s])
            if force_mix:                                  # DIAGNOSTIC: two syllables inner-heard at once
                o = (SID[A[0]] + 7) % len(ph.content)
                for k in range(3):
                    fseq[t0 + k] = np.union1d(syl_lines(brain, ph, SID[A[0]]), syl_lines(brain, ph, o))
                for i in range(1, len(A)):
                    for k in range(3):
                        fseq.pop(t0 + 3 * i + k, None)
        mprev = None
        utts, cur, sil, last = [], None, 99, None
        predc, mouth, l3s, drv = [], [], [], []
        for t, cg in enumerate(sched + [None, None, None]):
            srcs = []
            if mprev is not None and not mute:
                srcs.append((mprev[0], mprev[1], OWN, 1.0))
            if cg is not None:
                srcs.append(('s', cg, voice, 1.0))
            if t in fseq:
                brain._force_lines = fseq[t]
            mus = brain.step(render(ph, srcs, rng), 0.0, learn=False)
            if mute:
                mus = np.zeros_like(mus)
                brain.mus = mus
            v = np.zeros(ph.n_c, bool)
            idx = np.nonzero(brain.pred & brain.l1_content)[0]
            v[brain.l1_feat[idx]] = True
            predc.append(v)
            if rec_l3:
                l3s.append(brain.l3.copy())
            if drive:                                          # learned top -> L1 drive for the next tick (pre-threshold)
                a3 = np.nonzero(brain.l3)[0]
                dv = brain.P31[a3].sum(0) if len(a3) else np.zeros(brain.n1, np.float32)
                drv.append(dv)
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
        P = np.array(predc[q_end - 1: q_end + 15])
        a1 = SID[A[0]]
        allsc = (P[:, None, :] & C[None, :, :]).sum(2) / C.sum(1)[None, :]
        r = dict(f=f, x=x, held=(f, x) in held, A=A, syl=syl, lat=(resp[0]['t0'] - q_end) if resp else None,
                 correct=syl == A, first=bool(syl) and syl[0] == A[0], echo=bool(syl) and syl[0] in Q,
                 silent=syl is None, garble=bool(syl) and '~' in syl, talk=len(during) > 0, a1=a1, allsc=allsc,
                 n_extra=len(utts) - len(during) - (1 if resp else 0))
        if rec_l3:
            r['l3'] = np.array(l3s[q_end - 1: q_end + 15])
        if drive:
            D = np.array(drv[q_end - 1: q_end + 15])
            la = syl_lines(brain, ph, a1)
            lc = np.nonzero(brain.l1_content)[0]
            r['drive_ans'] = D[:, la].mean(1)                 # mean learned drive into the answer's 1st-syllable copies
            r['drive_other'] = D[:, lc].mean(1)                # into all content copies
            r['drive_max_other_syl'] = max(D[:, syl_lines(brain, ph, s)].mean(1).max() for s in range(len(C)) if s != a1)
        out.append(r)
    brain.restore(snap)
    return out


def agg(rs, key):
    return round(float(np.mean([r[key] for r in rs])), 3) if rs else None


def summ(rs):
    """accuracy etc. on a list of probe results"""
    if not rs:
        return {}
    return dict(n=len(rs), acc=agg(rs, 'correct'), first=agg(rs, 'first'), silent=agg(rs, 'silent'),
                garble=agg(rs, 'garble'), talk=agg(rs, 'talk'), echo=agg(rs, 'echo'),
                extra=round(float(np.mean([r['n_extra'] for r in rs])), 3))


def spec(pr):
    """is the answer's first-syllable code predicted after its own question more than after others? (X1d metric)"""
    S = np.array([r['allsc'] for r in pr])
    a1 = np.array([r['a1'] for r in pr])
    nq = len(pr)
    own = np.array([S[i, :, a1[i]] for i in range(nq)])
    oth = np.array([np.mean([S[j, :, a1[i]] for j in range(nq) if a1[j] != a1[i]], 0) for i in range(nq)])
    best = S.argmax(2)
    hub = np.bincount(best[:, 1:6].ravel(), minlength=S.shape[2])
    return dict(own_max=round(float(own.max(1).mean()), 3), other_max=round(float(oth.max(1).mean()), 3),
                own_l1_5=round(float(own[:, 1:6].mean()), 3), other_l1_5=round(float(oth[:, 1:6].mean()), 3),
                top_is_answer_l1_5=round(float(np.mean(best[:, 1:6] == a1[:, None])), 3),
                hub_syl=SYL[int(hub.argmax())], hub_share=round(float(hub.max() / max(1, hub.sum())), 3))


def mechanics6(world, a, b):
    """live-world mechanics per exchange in (a, b] (X1c definitions) plus accuracy split by pair set"""
    exs = [e for e in world.exchanges if a < e['n'] <= b and not e.get('overheard')]
    n = max(1, len(exs))
    utts = [u for e in exs for u in e['utts'] if not u['empty']]
    resp = [e for e in exs if e['resp'] is not None]
    return dict(n_ex=len(exs), resp_rate=round(len(resp) / n, 3),
                acc=round(sum(1 for e in resp if e['resp']['syl'] == e['A']) / n, 3),
                talkover_per_ex=round(sum(1 for u in utts if u['cg_speaking']) / n, 4),
                yieldfail_per_ex=round(sum(1 for u in utts if u['yield_fail']) / n, 4),
                garble_per_ex=round(sum(1 for u in utts if u['garble'] > 0) / n, 4),
                extra_per_ex=round(sum(1 for e in exs for u in e['utts'] if not u['empty'] and u is not e['resp']
                                       and not u['cg_speaking']) / n, 4),
                lat_mean=round(float(np.mean([e['lat'] for e in resp])), 2) if resp else None)


def acc_by_set(world, a, b, sets):
    """online accuracy per pair set (dict name -> set of pairs) in (a, b]"""
    out = {}
    for name, S in sets.items():
        exs = [e for e in world.exchanges if a < e['n'] <= b and (e['f'], e['x']) in S and not e.get('overheard')]
        if exs:
            out[name] = dict(n=len(exs), acc=round(sum(1 for e in exs if e['resp'] is not None and e['resp']['syl'] == e['A']) / len(exs), 3),
                             resp=round(sum(1 for e in exs if e['resp'] is not None) / len(exs), 3))
    return out
