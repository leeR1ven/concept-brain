# -*- coding: utf-8 -*-
"""x2: long untutored use (drift) with attribution arms, diagnostics, checkpoints and the relearning probe.

run_drift(agent, ...) continues ONE copy of the trained agent in an untutored C world (probe-only dialogues, learning
on) exactly as run_v22's drift block does (same world, same order of world runs, probes on frozen deep copies, so the
"as is" arm reproduces run_v22's drift numbers bit for bit), with:
  * marks: the dialogue counts at which C is probed (default 300, 900, 1500, 2100, 3000);
  * dset:  'k:v;k:v' settings applied to the drifting copy at the start of untutored use only (attribution arms):
           c.<cfg key>  (core cfg, e.g. c.recall_n:0 = no awake recall in untutored use; c.ral_hip_own:1)
           mix.<attr>   (random layer, e.g. mix.prune_on:0)
           u.reply      (ThinkAgent urgency of reply thoughts)   <attr> (agent attribute, e.g. nogo:1.01, veto_R1:0)
  * diag:  per mark, read-only counters (hippocampus composition, replay pool, layer pruning, thought outcomes, RAL
           resolutions, online reply classes of the untutored segment) and test-time lesion probes
           (noN: hippocampal NEXT copy route off; nohip: no hippocampal recall at test) at the marks in les_marks;
  * save:  gzip pickles of (agent, world) at chosen marks (to continue to 10,000 in a later job).
relearn(agent, ...): after training, change the taught answer of a few items and count the tutored exposures (recasts)
until a frozen copy says the new answer.
usage (from a saved agent): python3 drift_x2.py load=ck/x.pkl.gz tag=x seed=0 [marks=...] [dset=...] [diag=1]
       [save_at=3000] [save_dir=ck] [start=0 (dialogues already done in the loaded world)] [relearn=1] [out=...]"""
import os, sys, json, time, copy, pickle, gzip, collections
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import BENCH                                              # noqa: E402
sys.path.insert(0, BENCH)
from world import World                                              # noqa: E402
from metrics import evaluate                                         # noqa: E402

QW = set('什么吗哪几谁')


# ------------------------------------------------------------------ settings applied at the start of untutored use
def parse_dset(spec):
    out = []
    for part in [p for p in (spec or '').replace(',', ';').split(';') if p]:
        k, v = part.split(':', 1)
        out.append((k, v))
    return out


def _cast(old, v):
    if isinstance(old, bool):
        return v in ('1', 'True', 'true')
    if isinstance(old, (int, np.integer)) and not isinstance(old, bool):
        return int(float(v))
    if isinstance(old, (float, np.floating)):
        return float(v)
    return v


def apply_dset(agent, spec):
    done = []
    for k, v in parse_dset(spec):
        if k.startswith('c.'):
            obj, nm = agent.core.c, k[2:]
        elif k.startswith('mix.'):
            obj, nm = agent.core.mix, k[4:]
        elif k.startswith('u.'):
            agent.u[k[2:]] = float(v); done.append((k, float(v))); continue
        elif k.startswith('hip.'):                   # x2 horizon arms: e.g. hip.gamma:0.99975 (slower trace decay)
            obj, nm = agent.core.hip, k[4:]
        elif k == 'amnesia':                         # x2 horizon arm: every trace stored so far -> sigma 0
            h = agent.core.hip                       # (never recalled / replayed again; overwritten first)
            if int(float(v)):
                h.sig[:h.n] = 0.0
            done.append((k, int(h.n))); continue
        else:
            obj, nm = agent, k
        assert hasattr(obj, nm), k
        setattr(obj, nm, _cast(getattr(obj, nm), v))
        done.append((k, getattr(obj, nm)))
    return done


# ------------------------------------------------------------------ probes
def classify(r, prompts):
    rep, pr, acc = r['reply'], r['prompt'], r.get('acceptable') or []
    if rep in acc:
        return 'ok'
    if not rep:
        return 'silent'
    if rep == pr or (len(rep) >= 2 and rep in pr):
        return 'echo'
    if rep in prompts or (len(rep) >= 2 and any(p.startswith(rep) for p in prompts)) or (set(rep) & QW):
        return 'partner'
    return 'other'


def role_summary(recs, prompts):
    cls = collections.Counter(classify(r, prompts) for r in recs)
    n = max(1, len(recs))
    out = dict(n=len(recs), **{k: round(v / n, 4) for k, v in cls.items()},
               role_conf=round((cls['echo'] + cls['partner']) / n, 4))
    wrong = collections.Counter(r['reply'] for r in recs if r['reply'] not in (r.get('acceptable') or []))
    out['top_wrong'] = wrong.most_common(6)
    return out


def probe_C(agent, n, seed, mech, wkw, keep=None, spy=None, set_attr=None, mech_add=None):
    """run_v22.probe for stage C (frozen copy), optionally with a test-time lesion and a recall spy."""
    ag = copy.deepcopy(agent)
    ag.learning = False
    if set_attr:
        for k, v in set_attr:
            setattr(ag.core.c, k, v)
    rec_src = None
    if spy is not None:
        core = ag.core
        orig = core._set_recall
        rec_src = collections.Counter()
        clock0 = spy

        def _spy(st, e, ov, _o=orig, _c=core):
            rec = _c.hip.content[e]
            old = int(_c.hip.born[e]) < clock0
            nxt = rec.get('next')
            rec_src['old' if old else 'new'] += 1
            rec_src[('old' if old else 'new') + ('_next' if nxt is not None else '_nonext')] += 1
            return _o(st, e, ov)
        core._set_recall = _spy
    w = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3, **wkw)
    w.run(ag, n)
    res = evaluate(w.log)
    if mech_add is not None:
        mech_add(mech, res)
    S = res["C"]
    out = {k: [round(v["acc"], 4) if v.get("acc") is not None else None, v.get("n"),
               (round(v["held"]["acc"], 4) if (v.get("held") or {}).get("acc") is not None else None),
               (v.get("held") or {}).get("n"), round(v["spoke"], 3) if v.get("spoke") is not None else None]
           for k, v in S.items() if isinstance(v, dict) and "acc" in v}
    if keep is not None:
        for r in w.log:
            if r.get('kind') == 'probe' and r.get('stage') == 'C':
                keep.append(dict(stage='C', task=r['task'], held=bool(r.get('held_out')), prompt=r['prompt'],
                                 reply=r['reply'], ok=r['reply'] in (r.get('acceptable') or []),
                                 acceptable=list(r.get('acceptable') or [])))
    if rec_src is not None:
        out['_recall_src'] = dict(rec_src)
    return out


# ------------------------------------------------------------------ read-only diagnostics of the drifting copy
def hip_diag(core, clock0):
    h = core.hip
    n = h.n
    sig = h.sig[:n]
    born = h.born[:n]
    old = born < clock0
    live = sig > 0.05
    nx = core.next_of[:n]
    ok = nx >= 0
    ok[ok] = h.born[nx[ok]] == core.next_born[:n][ok]
    ok &= live
    if core.c.ral and core.c.ral_hip_own:
        ok &= ~core.own_ep[:n]
        if core.c.ral_replay_own <= 0:
            ok &= ~core.own_ep[np.where(nx >= 0, nx, 0)]
    nxt_old = np.zeros(n, bool)
    nxt_old[ok] = h.born[nx[ok]] < clock0
    return dict(n=int(n), clock=int(h.clock), live=int(live.sum()), live_old=int((live & old).sum()),
                pool=int(ok.sum()), pool_old=int((ok & old).sum()), pool_old_next_old=int((ok & old & nxt_old).sum()),
                own_cells=int(core.own_ep[:n].sum()) if hasattr(core, 'own_ep') else 0,
                sig_min_live=round(float(sig[live].min()), 4) if live.any() else None)


def _delta(now, before):
    return {k: (now[k] - before.get(k, 0)) for k in now if isinstance(now[k], (int, float, np.integer))}


def seg_classes(log, prompts):
    recs = [r for r in log if r.get('kind') == 'probe']
    cls = collections.Counter(classify(dict(reply=r['reply'], prompt=r['prompt'], acceptable=r.get('acceptable')), prompts)
                              for r in recs)
    n = max(1, len(recs))
    return dict(n=len(recs), **{k: round(v / n, 4) for k, v in cls.items()})


def state_counters(ag):
    core = ag.core
    d = dict(learn_events=int(core.stats['learn_events']), awake_recalls=int(core.stats['awake_recalls']),
             lines=int(core.stats['lines']))
    for k, v in core.ral_stats.items():
        d['ral_' + k] = int(v)
    for k, v in ag.tstats.items():
        if isinstance(v, (int, np.integer)):
            d['t_' + k] = int(v)
    if getattr(core, 'rmix', False):
        mx = core.mix
        d.update(rm_ev=int(mx.ev), rm_pruned=int(mx.stats['pruned']), rm_regrown=int(mx.stats['regrown']))
    d['rbar'] = float(ag.rbar); d['rr'] = float(getattr(ag, 'rr', 0.0))
    return d


def gen_diag(core):
    G = core.gen
    n = core.mix.n
    W = G.W[:n]
    out = dict(gen_abs_mean=float(np.abs(W).mean()), gen_pos_sum=float(W[W > 0].sum()), gen_neg_sum=float(W[W < 0].sum()))
    if G.U is not None:
        U = G.U[:n]
        m = U > 0
        out.update(u_mean=float(U[m].mean()) if m.any() else 0.0, u_max=float(U.max()),
                   uw_p50=float(np.median((U * np.abs(W))[m])) if m.any() else 0.0,
                   uw_p90=float(np.percentile((U * np.abs(W))[m], 90)) if m.any() else 0.0)
        if G.stab > 0:
            f = G._stab_f(U[m], W[m])
            out.update(f_mean=float(f.mean()), f_p10=float(np.percentile(f, 10)), f_floor=float((f <= G.stab_min + 1e-6).mean()))
    return out


# ------------------------------------------------------------------ the drift loop
def run_drift(ag2, seed, wkw, marks, prompts, mech, mech_add, log, evaln=80, dset='', diag=0, les_marks=(),
              save_at=(), save_fn=None, wf=None, done_n=0, clock0=None, keep_marks=None, trim=0):
    """ag2: the drifting copy (learning on).  Returns dict(marks={n: {...}}, ...).  keep_marks: marks whose probe
    replies are kept (list of dicts) in out['replies'][mark]."""
    ag2.learning = True
    if wf is None:
        wf = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0, **wkw)
    applied = apply_dset(ag2, dset) if dset else []
    if applied:
        log(f"drift settings {applied}")
    if clock0 is None:
        clock0 = int(ag2.core.hip.clock)
    out = dict(marks={}, dset=dset, applied=[(k, str(v)) for k, v in applied], clock0=clock0, replies={})
    cnt0 = state_counters(ag2)
    for mk in marks:
        if mk <= done_n:
            continue
        n0 = len(wf.log)
        t_s = time.time()
        wf.run(ag2, mk - done_n)
        done_n = mk
        seg = wf.log[n0:]
        mech_add(mech, evaluate(seg))
        if trim:                                      # x2 local memory cap: drop non-state logs (as save_fn does)
            slim(ag2)
            wf.log = wf.log[-1:]
        rep = []
        pr = probe_C(ag2, evaln, seed, mech, wkw, keep=rep, spy=clock0 if diag else None, mech_add=mech_add)
        rs = role_summary(rep, prompts)
        m = dict(acc=pr['ALL'][0], probe={k: v for k, v in pr.items() if not k.startswith('_')}, role=rs,
                 secs=round(time.time() - t_s, 1))
        if keep_marks and mk in keep_marks:
            out['replies'][str(mk)] = rep
        if diag:
            cnt = state_counters(ag2)
            m['recall_src'] = pr.get('_recall_src')
            m['seg'] = seg_classes(seg, prompts)
            m['delta'] = _delta(cnt, cnt0)
            cnt0 = cnt
            m['hip'] = hip_diag(ag2.core, clock0)
            m['gen'] = gen_diag(ag2.core)
            if mk in les_marks:
                les = {}
                for nm, sa in (('noN', [('copy_next', False)]), ('nohip', [('hippo', False)])):
                    rp = []
                    o = probe_C(ag2, evaln, seed, {}, wkw, keep=rp, set_attr=sa)
                    les[nm] = dict(acc=o['ALL'][0], role={k: v for k, v in role_summary(rp, prompts).items() if k != 'top_wrong'})
                m['lesion'] = les
        out['marks'][str(mk)] = m
        log(f"drift C: -> {m['acc']} after {mk} untutored dialogues | role {dict((k, v) for k, v in rs.items() if k != 'top_wrong')}"
            + (f" | seg {m['seg']} | hip {m['hip']} | rsrc {m.get('recall_src')} | les {m.get('lesion')}" if diag else ''))
        if mk in save_at and save_fn is not None:
            save_fn(ag2, wf, mk)
    out['done_n'] = done_n
    return out, wf


def save_pickle(path, obj):
    with gzip.open(path, 'wb', compresslevel=6) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return os.path.getsize(path)


def load_pickle(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def slim(agent):
    """drop logs that are not state (thought log, per-line diagnostic logs) before pickling"""
    agent.th_log = []
    if hasattr(agent, 'fav_log'):
        agent.fav_log = []
    if hasattr(agent, 'voice_log'):
        agent.voice_log = []
    if hasattr(agent.core, 'ral_log'):
        agent.core.ral_log = []


# ------------------------------------------------------------------ relearning probe
RELEARN = [('ritual', '谢谢', '再见'), ('ritual', '晚安', '早上好'), ('taste', 0, '不好吃'), ('offer', 1, '不要')]


def _relearn_items(w, rng_items):
    """exchanges with a changed taught answer: rituals (no scene) and one taste / offer item (scene + pointing)."""
    from world import FOODS
    items = []
    for task, x, new in rng_items:
        if task == 'ritual':
            items.append(dict(task='ritual', prompt=x, scene=None, attend=None, new=new, X=None))
        else:                                    # the x-th food whose (task, food) pair is taught (not held out)
            ok = [f for f in FOODS if f not in w.cur.held.get(task, ())]
            items.append(dict(task=task, X=ok[int(x) % len(ok)], new=new))
    return items


def _run_ex(w, agent, it, kind, accept):
    """one dialogue: (scene) + one exchange of `kind` with acceptable = accept (the new taught answer)."""
    p = w.p
    if it['task'] == 'ritual':
        ex = dict(stage='C', task='ritual', held_out=False, needs_scene=False, prompt=it['prompt'], acceptable=[accept],
                  model=accept, item=('relearn', it['prompt']), attend=None)
        scene = None
    else:
        scene = w._scene_with([it['X']])
        ex = dict(stage='C', task=it['task'], held_out=False, needs_scene=False, X=it['X'], item=('relearn', it['X']))
        w._fill(ex, scene)
        ex.update(acceptable=[accept], model=accept)
    ex['kind'] = kind
    w._cur_stage = 'C'
    w._set_scene(None)
    w._silence(max(0, p['gap_dialogue'] - p['scene_lead']))
    w._set_scene(scene)
    w._silence(p['scene_lead'])
    rec = w._exchange(ex)
    w.n_dialogues += 1
    return rec


def relearn(agent, seed, wkw, rounds=20, log=print, items=RELEARN):
    """copy the trained agent; per round: one recast exchange per item with the NEW answer (learning on), then a frozen
    probe of every item.  Returns per item: the replies per round, the first round the new answer is said, and how many
    of the last 5 rounds say it."""
    ag = copy.deepcopy(agent)
    ag.learning = True
    w = World("C", seed=seed, tutor="recast", **wkw)        # x2 fix: same seed = same voice + eye codes
    w.agent = ag
    w._hook('bench_attach', w)
    w._silence(w.p['lead_in'])
    its = _relearn_items(w, items)
    # before: what does the trained agent say (frozen)?
    def frozen_probe():
        fz = copy.deepcopy(ag); fz.learning = False
        wp = World("C", seed=seed, tutor="recast", **wkw)
        wp.agent = fz; wp._hook('bench_attach', wp); wp._silence(wp.p['lead_in'])
        return [_run_ex(wp, fz, it, 'probe', it['new'])['reply'] for it in its]
    hist = [frozen_probe()]
    online = []
    for r in range(rounds):
        recs = [_run_ex(w, ag, it, 'recast', it['new']) for it in its]
        online.append([x['reply'] for x in recs])
        hist.append(frozen_probe())
        log(f"relearn round {r + 1}: online {online[-1]} frozen {hist[-1]}")
    res = []
    for i, it in enumerate(its):
        said = [h[i] == it['new'] for h in hist]
        first = next((k for k, s in enumerate(said) if s), None)
        stable = next((k for k in range(len(said)) if all(said[k:])), None)
        res.append(dict(item=it.get('prompt') or f"{it['task']}:{it['X']}", new=it['new'], before=hist[0][i],
                        first=first, stable_from=stable, last5=int(sum(said[-5:])), replies=[h[i] for h in hist]))
    mech = sum(int(r.get('mouth_while_partner') or 0) for r in w.log)
    return dict(items=res, rounds=rounds, overlap=mech,
                mean_first=float(np.mean([x['first'] if x['first'] is not None else rounds + 1 for x in res])),
                mean_stable=float(np.mean([x['stable_from'] if x['stable_from'] is not None else rounds + 1 for x in res])))


# ------------------------------------------------------------------ CLI: continue a saved agent
def main():
    import run_x2
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, seed = A.pop('tag', 'x'), int(A.pop('seed', 0))
    load = A.pop('load')
    marks = [int(x) for x in A.pop('marks', '300,900,1500,2100,3000').split(',')]
    dset = A.pop('dset', '')
    diag = int(A.pop('diag', 1))
    les_marks = [int(x) for x in A.pop('les', '2100,3000').split(',') if x]
    save_at = [int(x) for x in A.pop('save_at', '').split(',') if x]
    save_dir = A.pop('save_dir', os.path.join(HERE, 'ck'))
    evaln = int(A.pop('evaln', 80))
    trim = int(A.pop('trim', 0))
    do_relearn = int(A.pop('relearn', 0))
    outp = A.pop('out', os.path.join(HERE, 'out', 'runs', f'drift_{tag}_s{seed}.json'))
    wkw = {'eye_layout': 'retino'}
    t0 = time.time()

    def log(s):
        print(f"[{tag} s{seed} {time.time() - t0:.0f}s] {s}", flush=True)
    obj = load_pickle(load)
    if isinstance(obj, dict):
        ag, wf, done_n, prompts, clock0 = obj['agent'], obj.get('world'), obj.get('done_n', 0), obj['prompts'], obj.get('clock0')
    else:
        raise ValueError('pickle format')
    if 'clock0' in A:                                 # x2: resume from a pickle saved without clock0
        clock0 = int(A.pop('clock0'))
    if clock0 is None and done_n == 0:                # so that saved drift pickles carry the start of untutored use
        clock0 = int(ag.core.hip.clock)
    log(f"loaded {load} done_n {done_n} hip {ag.core.hip.n} clock {ag.core.hip.clock} clock0 {clock0} ({time.time() - t0:.0f}s)")
    import resource
    mech = {}
    rec = dict(tag=tag, seed=seed, load=load, marks=marks, dset=dset)
    if do_relearn:
        t_r = time.time()
        rec['relearn'] = relearn(ag, seed, wkw, rounds=int(A.pop('rounds', 20)), log=log)
        rec['relearn_secs'] = round(time.time() - t_r, 1)
        log(f"relearn {json.dumps({k: v for k, v in rec['relearn'].items() if k != 'items'})} "
            + str([(x['item'], x['before'], x['first'], x['stable_from']) for x in rec['relearn']['items']]))

    def save_fn(a, w, mk):
        slim(a)
        n = w.log[-1:] if w.log else []
        w.log = n
        p = os.path.join(save_dir, f'{tag}_s{seed}_d{mk}.pkl.gz')
        sz = save_pickle(p, dict(agent=a, world=w, done_n=mk, prompts=prompts, clock0=clock0))
        log(f"saved {p} {sz / 2 ** 20:.1f} MB")
    if marks and marks[-1] > done_n:
        dr, wf = run_drift(ag, seed, wkw, marks, prompts, mech, run_x2.mech_add, log, evaln=evaln, dset=dset, diag=diag,
                           les_marks=les_marks, save_at=save_at, save_fn=save_fn, wf=wf, done_n=done_n, clock0=clock0,
                           trim=trim)
        rec['drift'] = dr
    rec['viol'] = sum(mech.get(k, 0) for k in run_x2.MECH_KEYS)
    rec['mech'] = mech
    rec['secs'] = round(time.time() - t0, 1)
    rec['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    log(f"done viol={rec['viol']} secs={rec['secs']} rss={rec['peak_rss_mb']}")


if __name__ == "__main__":
    main()
