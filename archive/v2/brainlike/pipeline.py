# Full staged run on the bench + lesion ablations.
#   python3 -B pipeline.py SEED NA NB NC ND OUT.json [key=value ...]
# A (babbling, CDS, overheard, play) -> B -> C -> D, tutor 'model' (learning by listening; probes are scored free
# replies; no muscle cue ever).  Optional: tutor=recast for B-D.  Then, on frozen deep copies, a fixed probe stream
# 'BCD' (same world seed for every lesion) with each learned part removed.
import sys, os, time, json, copy, pickle, collections
import numpy as np
argv = sys.argv; sys.argv = ['x']
import bench_run as br
from metrics import evaluate
from agent import Agent, DEFAULT, NSM, NLM


def summ(res, stages):
    out = {}
    for st in stages:
        r = res.get(st, {})
        for task, d in r.items():
            if isinstance(d, dict) and 'acc' in d and d.get('n'):
                h = d.get('held') or {}; s = d.get('seen') or {}
                out[f"{st}/{task}"] = dict(acc=d['acc'], n=d['n'], first=d.get('first'), echo_free=d.get('echo_free'),
                                          parrot=d.get('parrot'), seen=s.get('acc'), n_seen=s.get('n'),
                                          held=h.get('acc'), n_held=h.get('n'))
        m = r.get('mech')
        if m:
            out[f"{st}/mech"] = {k: m.get(k) for k in ('mouth_while_partner', 'exchanges_with_overlap', 'latency_median',
                                                    'frac_not_3', 'garble', 'extra_turns', 'spoke')}
        for k in ('one_shot', 'learning'):
            if k in r:
                out[f"{st}/{k}"] = r[k]
    return out


PROBE = [dict(prompt='这是什么', reply='苹果', kind='probe', scene=(('苹果', 0, 1), ('小狗', 2, 0), ('汽车', 1, 2)),
              attend=0, task='what'),
         dict(prompt='小狗在哪里', reply='在中间', kind='probe', scene=(('苹果', 0, 1), ('小狗', 2, 0), ('汽车', 1, 2)),
              task='where')]


def codes(agent, prev, seed):
    """drift probe: on a frozen copy, the PFC / hippocampal codes at the end of two fixed prompts."""
    a = copy.deepcopy(agent); a.c['freeze'] = True; a.c['babble'] = False; a.c['babble_idle'] = False
    snaps = []
    orig = a.other_line_end
    def hook(eye):
        orig(eye)
        snaps.append(dict(sup=a.sup.copy(), dq=a.dq.copy(), dg=a.dg_code(a.ec_vector(a.slot_maps(eye)[0]))))
    a.other_line_end = hook
    w = br.mk('B', seed, prev=prev)            # same seed = same language (the Voice is built from the seed)
    w.run_script(a, PROBE, stage='B')
    return snaps[:2]


def predict_vs_ngram(log, start):
    """next-syllable prediction while listening: the readout's argmax vs an ONLINE exact n-gram (1, 2, 3 previous
    syllables of the same line, START-padded, counts from everything heard before) on the identical event stream.
    Scored on events with index >= start."""
    import collections
    cnt = [collections.defaultdict(collections.Counter) for _ in range(3)]
    hist, cur = [], None
    hits = {'readout': [], 'n1': [], 'n2': [], 'n3': []}
    for i, (line, syl, hit) in enumerate(log):
        if line != cur:
            cur, hist = line, []
        ctx = tuple([-1] * 3 + hist)[-3:]
        for n in range(3):
            c = cnt[n].get(ctx[3 - n - 1:])
            pred = c.most_common(1)[0][0] if c else None
            if i >= start:
                hits[f'n{n + 1}'].append(pred == syl)
        if i >= start:
            hits['readout'].append(bool(hit))
        for n in range(3):
            cnt[n][ctx[3 - n - 1:]][syl] += 1
        hist.append(syl)
    return {k: (float(np.mean(v)) if v else None) for k, v in hits.items()} | {'n_events': len(hits['readout'])}


def J(x, y):
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 1.0


def lesion(agent, what):
    a = copy.deepcopy(agent)
    a.c['freeze'] = True; a.c['babble'] = False; a.c['babble_idle'] = False
    if what == 'none':
        pass
    elif what == 'W':
        a.W.w[:] = 0
    elif what.startswith('G_'):
        for s in what[2:].split('+'):
            a.G[s].w[:] = 0
    elif what == 'VL':
        for V in a.VL.values():
            V.raw[:] = 0
    elif what == 'LA':
        a.LA.raw[:] = 0
    elif what == 'chunks':
        a.chunks = {}; a.chunk_syl = {}; a.n_lex = NSM
    elif what == 'REC':
        if a.REC is not None:
            a.REC.raw[:] = 0
    elif what == 'hip':
        a.c['hip'] = False
    elif what == 'echo':
        a.c['no_echo'] = True
    elif what == 'vis':
        a.c['no_vis'] = True
    elif what in ('no_sup', 'no_dq', 'no_dc', 'no_ctx'):
        a.c[what] = True
    else:
        raise ValueError(what)
    return a


def motor(a, w):
    """motor inventory: which chars the self-found programs produce (decoded by the world's own Voice)."""
    from world import OBJECTS, COLOURS, COUNTS, SLOTS, RITUALS, NAMES, NOVEL
    v = w.voice
    have = collections.Counter()
    for y in range(a.body.n_prog):
        s = tuple(np.nonzero(a.body.prog[y])[0].tolist()); ch = v.decode(s)
        if ch and ch != '~':
            have[ch] += 1
    bc = set(''.join(OBJECTS + COLOURS + COUNTS + SLOTS + [q + ''.join(x) for q, x in RITUALS] + ['这是什么在哪里你要吗好不吃颜色有几个很的']))
    names = set(''.join(NAMES)); novel = set(''.join(NOVEL))
    hs = set(have)
    return dict(n_prog=int(a.body.n_prog), pool=int(a.body.NP), distinct=len(hs), bc=f"{len(hs & bc)}/{len(bc)}",
                names=f"{len(hs & names)}/{len(names)}", novel=f"{len(hs & novel)}/{len(novel)}",
                per_char=round(a.body.n_prog / max(len(hs), 1), 2), max_per_char=max(have.values()) if have else 0,
                n_syl=int(a.syl_n))


def main():
    seed, NA, NB, NC, ND = (int(x) for x in argv[1:6])
    out_path = argv[6]
    opt = dict(x.split('=', 1) for x in argv[7:] if '=' in x)
    tutor = opt.pop('tutor', 'model')
    n_test = int(opt.pop('ntest', '300'))
    les_opt = opt.pop('lesions', None)
    save = opt.pop('save', '1') == '1'
    tag = opt.pop('tag', None)
    cfg = {'seed': seed, 'p_go': 0.5, 'p_new': 0.8}
    for k, v in opt.items():
        cfg[k] = (v in ('1', 'True')) if isinstance(DEFAULT[k], bool) else type(DEFAULT[k])(v)
    t0 = time.time()
    a = Agent(cfg)
    w = br.mk('A', seed)
    for win in w.warmup_windows():
        a.warm(win)
    a.end_warmup()
    w.run(a, NA)
    R = {'tag': tag or f's{seed}', 'seed': seed, 'cfg': cfg, 'tutor': tutor, 'stats_A': dict(a.stats), 'secs_A': time.time() - t0}
    a.c['babble'] = False
    prev = w
    drift = [codes(a, w, seed)]
    n_ev0 = len(a.pred_log)
    for st, N in (('B', NB), ('C', NC), ('D', ND)):
        if N <= 0:
            continue
        ws = br.mk(st, seed, prev=prev, tutor=tutor)
        ws.run(a, N)
        R[st] = summ(evaluate(ws.log), [st])
        try:                                     # late third of the stage per task, and the memory-task replies
            recs = [r for r in ws.log if r['kind'] in ('probe', 'recast')]
            late = recs[-max(1, len(recs) // 3):]
            R[st + '_late'] = {t: round(float(np.mean([r['reply'] in r['acceptable'] for r in late if r['task'] == t])), 3)
                               for t in sorted({r['task'] for r in late})}
            if st == 'D':
                R['D_replies'] = {t: [(r['acceptable'][0], r['reply']) for r in recs if r['task'] == t][-40:]
                                  for t in ('name', 'oneshot', 'before')}
        except Exception as e:
            R[st + '_late'] = repr(e)
        R[st + '_examples'] = [f"{r['prompt']}->{r['reply']}" for r in ws.log if r['kind'] in ('probe', 'recast')][-20:]
        R[st + '_predict'] = predict_vs_ngram(a.pred_log, start=len(a.pred_log) - max(1, (len(a.pred_log) - n_ev0) // 2))
        n_ev0 = len(a.pred_log)
        print(f"[s{seed}] {st} listening next-syllable prediction (2nd half of the stage): {R[st + '_predict']}", flush=True)
        json.dump(R, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
        print(f"[s{seed}] {st} done {time.time()-t0:.0f}s",
              {k: (round(v['acc'], 2) if isinstance(v, dict) and v.get('acc') is not None else None)
               for k, v in R[st].items()}, flush=True)
        prev = ws
        drift.append(codes(a, ws, seed))
    R['drift'] = {k: [[J(drift[i][q][k], drift[i + 1][q][k]) for q in range(len(drift[0]))] for i in range(len(drift) - 1)]
                  for k in ('sup', 'dq', 'dg')}
    R['drift_A_to_end'] = {k: [J(drift[0][q][k], drift[-1][q][k]) for q in range(len(drift[0]))] for k in ('sup', 'dq', 'dg')}
    print(f"[s{seed}] drift stage-to-stage {R['drift']} A->end {R['drift_A_to_end']}", flush=True)
    if save:
        pickle.dump((a, prev), open(out_path.replace('.json', '_agent.pkl'), 'wb'))
    try:
        R['motor'] = motor(a, prev)
    except Exception as e:
        R['motor'] = repr(e)
    print(f"[s{seed}] motor {R['motor']}", flush=True)
    R['stats'] = dict(a.stats); R['n_prog'] = a.body.n_prog; R['n_syl'] = a.syl_n; R['n_chunks'] = len(a.chunks)
    # ---------------- lesions on frozen copies, same probe stream
    stages = ''.join(s for s, N in (('B', NB), ('C', NC), ('D', ND)) if N > 0)
    les = ['none', 'W', 'G_H', 'G_Vo+Vc+Vn+Vl', 'G_Vh+Vr', 'VL', 'LA', 'chunks', 'REC', 'hip', 'echo', 'vis', 'no_sup',
           'no_dq', 'no_dc', 'no_ctx']
    if les_opt:
        les = les_opt.split(',')
    R['lesion'] = {}
    for L in les:
        al = lesion(a, L)
        wt = br.mk(stages, seed, prev=prev, tutor='model')   # same physics; fresh content continuing from prev
        wt.p['probe_frac'] = 1.0                                  # every exchange is a probe (no model reply)
        wt.run(al, n_test)
        res = evaluate(wt.log)
        R['lesion'][L] = summ(res, list(stages))
        allacc = [v['acc'] for k, v in R['lesion'][L].items() if isinstance(v, dict) and v.get('acc') is not None
                  and '/' in k and not k.endswith('mech')]
        json.dump(R, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
        print(f"[s{seed}] lesion {L}: mean task acc {np.mean(allacc) if allacc else float('nan'):.3f} {time.time()-t0:.0f}s", flush=True)
        del al, wt
    R['secs'] = time.time() - t0
    json.dump(R, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
