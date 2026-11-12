"""Does the OLD line's trained PFC have a rough order of activation? (designer 11-12: "你不需要重新训练啊，拿原来训练过的
模型直接检测前额叶是不是有大致的循环激活顺序不就行了。") The frozen parts of core/expect.py (commit 49e444b), printed
part by part: for lines heard in training, never-heard lines, and real replies, the share of each newly arriving
copy that the PFC had already lit from inside (thought: same tick or one tick before), in the lines' own order
against the same syllables shuffled (Saffran, Aslin & Newport 1996). Nothing is read out by an outside reader and
nothing assumes what it should think. No learning: every hearing starts from the same restored state.

Runs only in a checkout of 49e444b (core/expect.py, core/exam.py):  python -u -m core.gen.old_order <ckpt> [n=40]"""
import json
import sys
import time

import numpy as np
import torch

from ..exam import first_lines
from ..expect import compare, listen, share
from ..world.corpus import common_order


def calib(life, p, FC, sids, ca, cb, lag, nd=60, d0=0):
    """Fable a38 #3, frozen: in held-out dialogues heard as in life, at every tick the most expected syllable (the
    push above usual summed over each syllable's current code; T1 its score) against the next non-silent syllable;
    precision by confidence (T1 quintile) and by position (mid-turn: speech now and next; gap k: k ticks after the
    last speech tick); the babbler (the commonest syllable) at the same ticks"""
    from ..world.corpus import script_dialogue
    gt = tuple(life.c.get('gap_turn', (4, 10)))
    pos = {x: i for i, x in enumerate(sids)}
    cnt = np.bincount(life.train.syl[:200000].astype(np.int64))
    common = int(np.argmax(cnt))
    rows = []                                             # (T1, top==next, top in next 3, babbler==next, babbler in 3, where)
    size = FC.sum(1).clamp_min(1.0)
    for di in range(d0, d0 + nd):
        src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9300, di]), gt, 8, 1, 1)
        src = src + [None] * 4
        E = []
        life.play([None if x is None else x[:5] for x in src], rec=lambda t: E.append(p.e1[ca:cb].clamp_min(0.0)))
        lab = [(-1 if x is None else int(x[0])) for x in src]
        last = -99
        for t in range(len(src) - 1):
            if lab[t] >= 0:
                last = t
            ahead = [y for y in lab[t + 1:] if y >= 0][:3]
            if not ahead or last < 0:
                continue
            # the expectation for what arrives at t + 1 + lag is the push at that tick (made before it arrives)
            te = t + 1 + lag
            if te >= len(E):
                continue
            sc = (FC @ E[te]) / size
            T1, top = float(sc.max()), int(sids[int(sc.argmax())])
            where = 'mid' if lab[t] >= 0 and lab[t + 1] >= 0 else (f'gap{min(3, t - last + 1)}' if lab[t + 1] < 0
                                                                    or lab[t] < 0 else 'other')
            rows.append((T1, top == ahead[0], top in ahead, common == ahead[0], common in ahead, where))
    if not rows:
        return
    T = np.array([r[0] for r in rows])
    qs = np.quantile(T, [0.2, 0.4, 0.6, 0.8])
    out = {}
    for where in ('mid', 'gap1', 'gap2', 'gap3', 'all'):
        R = [r for r in rows if where == 'all' or r[5] == where]
        if not R:
            continue
        t1 = np.array([r[0] for r in R])
        b = np.digitize(t1, qs)
        out[where] = {'n': len(R), 'babbler next / next3': [round(float(np.mean([r[3] for r in R])), 3),
                                                             round(float(np.mean([r[4] for r in R])), 3)],
                      'by confidence quintile (next, next3)': [
                          [round(float(np.mean([r[1] for r, bb in zip(R, b) if bb == k])), 3) if (b == k).any() else None,
                           round(float(np.mean([r[2] for r, bb in zip(R, b) if bb == k])), 3) if (b == k).any() else None]
                          for k in range(5)]}
    print('CALIB', json.dumps(out), flush=True)


def main(ckpt, n=40, loop_scale=1.0, gt_scale=1.0, parts='push,lit,reply', gt_no_motor=0, concept_gt_only=0, cset=None,
         fresh_codes=0):
    t0 = time.time()
    life = torch.load(ckpt, weights_only=False)['life']
    c = life.c
    for k_, v_ in (cset or {}).items():                   # c.<key>=<json>: a config value for this test only
        c[k_] = v_
    p = life.brain.pfc
    # Fable a35 #3 (frozen, this process only): the hold (the loop: last layer -> first layer) against the link to the
    # next content (the graded table): scale either, and see whether the lit thoughts then follow the order
    if loop_scale != 1.0:
        p.loop.w = p.loop.w * loop_scale
    th_m = c['theta1'] + c.get('check_margin', 0.0)
    if gt_scale != 1.0:
        c['graded_gain'] = gt_scale * c.get('graded_gain', th_m)
    if gt_no_motor:                                       # the graded table without the motor copies (its own
        from ..grow import gt_drop_motor                  # babble predicted itself: full10's thoughts were 57
        gt_drop_motor(life)                               # force copies, the same every time)

    def push_by_source():                                 # PFC.push, with each source kept (same sum)
        a4 = p.L4.on
        comp = {'loop': p.loop.drive(a4), 'L41': p.L41.drive(a4)}
        for nm, tb in getattr(p, 'P', {}).items():
            comp['P_' + nm] = tb.drive(p.pred_src(nm, getattr(p, nm).on))
        if getattr(p, 'G1P', None) is not None:
            comp['G1P'] = p.G1P.drive(a4)
        if getattr(p, 'G1C', None) is not None:
            comp['G1C'] = p.G1C.drive(p.x_int)
        if getattr(p, 'GT', None) is not None:
            p.gt_p = p.GT.drive(p.L1.on)
            comp['GT'] = c.get('graded_gain', th_m) * p.gt_p
        if concept_gt_only:                               # Fable a36 what-if: concept copies driven by the graded
            for k_ in comp:                               # table only (the saturating, stale sources off for them)
                if k_ != 'GT' and k_ != 'G1C':
                    comp[k_] = comp[k_].clone()
                    comp[k_][ca:cb] = 0.0
        p._comp = comp
        return sum(comp.values())
    p.push = push_by_source
    src_share, src_share_c = {}, {}
    lag_ov = {k_: [] for k_ in (-3, -2, -1, 0, 1)}        # lit concept thoughts at t vs the concept copies heard at t+k
    conc = []                                             # ticks where one syllable has enough code units thought
    G1C = getattr(p, 'G1C', None)
    k_say = max(1, int(round(c['context'] * getattr(life, '_ext_heard', 30.0) * c.get('speak_share', 0.5))))
    segs_ = {k_: (a_, b_) for k_, l_, a_, b_ in life.brain.chains.segs}
    ca, cb = segs_.get('concept', (0, 0))
    kinds = []                                            # the copies by region (contiguous blocks of PFC layer 1)
    for k_, l_, a_, b_ in life.brain.chains.segs:
        if kinds and kinds[-1][0] == k_ and kinds[-1][2] == a_:
            kinds[-1] = (k_, kinds[-1][1], b_)
        else:
            kinds.append((k_, a_, b_))
    print('ORDER setting', json.dumps(dict(loop_scale=loop_scale, gt_scale=gt_scale,
                                           graded_gain=c.get('graded_gain', th_m))), flush=True)
    # the life's own sound format (core/expect.py assumed 3 ticks per syllable; this checkpoint hears one whole
    # syllable per tick): each syllable as life.syl gives it; a copy reaches the PFC `lag` ticks after the ear
    T = len(life.syl(0, 'C1'))
    lag = 2 if (getattr(life.brain, 'concept', None) is not None and not c.get('concept_sync')) else 1
    lead = 4

    def srcs_of(ln, v):
        return [None] * lead + [x for s_ in ln for x in life.syl(s_, v)] + [None] * 4

    def window(ln):                                           # the 2nd syllable's arrival .. the last's + 1
        return slice(lead + T + lag, lead + T * (len(ln) - 1) + lag + T + 1)
    print('ORDER format', json.dumps(dict(ticks_per_syllable=T, lag=lag, concept_sync=c.get('concept_sync'))),
          flush=True)
    rng = np.random.default_rng(7)
    order = getattr(life, 'order', None)
    order = common_order(life.train) if order is None else order

    def shuffled(ln):
        for _ in range(20):
            s = [ln[i] for i in rng.permutation(len(ln))]
            if s != ln:
                return s
        return None

    # Fable a34's two checks: (1) are the thoughts generic (the inside-lit sets after different lines overlap a lot)?
    # (2) order in the push itself: each arriving copy's push above its own usual a tick before it arrives, as a
    # percentile among the copies that can be thought (0.5 = chance), and the share of arriving copies inside the
    # top-n of that push (n = the number arriving); own order against shuffled
    b = life.brain
    p = b.pfc
    cand = getattr(p, 'pfc_in', None)

    sids_ = list(getattr(life, 'speak_sids', []))
    pos_ = {x: i for i, x in enumerate(sids_)}
    think = {'own': [], 'shuffled': []}
    ctrl = []                                             # the readout's check: the copies that do arrive, ranked
    tend = {'own': [], 'shuffled': []}                    # Fable a37: the tendency (push above usual, summed over
    freq_rank = None                                      # each syllable's code) instead of the binary thoughts

    def think_of(rec, ln, key):
        """the syllable the lit thoughts are closest to, through the brain's own speech-link codes (the concept
        copies each wired syllable lights), at the tick before each next syllable's concept copies arrive"""
        if (G1C is None and FC is None) or not sids_:
            return
        for j in range(1, len(ln)):
            tt = lead + T * j + lag - 1
            if tt >= len(rec) or ln[j] not in pos_:
                continue
            # the push a copy gets at the tick it arrives is computed before it arrives (from the state a tick
            # earlier): the expectation of the syllable arriving at tt + 1 is e1 at tt + 1 (11-12 night: e1 at tt was
            # the expectation of the syllable arriving at tt, the one just heard)
            if tt + 1 >= len(rec):
                continue
            if FC is not None:                            # the current codes
                e_ = rec[tt + 1][1][ca:cb].clamp_min(0.0)
                sc_ = (FC @ e_) / FC.sum(1).clamp_min(1.0)
            else:
                e_ = rec[tt + 1][1][G1C.units].clamp_min(0.0)
                sc_ = (G1C.code @ e_) / G1C.size          # mean push above usual over the syllable's code
            v_ = sc_[pos_[ln[j]]]
            tend[key].append((float((sc_ > v_).sum()) + 0.0, float(freq_pos.get(ln[j], len(sids_)))))
            if FC is not None and tt + 1 < len(rec):     # the same readout on what actually arrives (should be ~top)
                x_ = rec[tt + 1][3][ca:cb].float()
                s2 = (FC @ x_) / FC.sum(1).clamp_min(1.0)
                ctrl.append(float((s2 > s2[pos_[ln[j]]]).sum()))
            if G1C is None:
                continue
            cnt_ = G1C.code @ rec[tt][2][G1C.units].float()
            if float(cnt_.max()) <= 0:
                think[key].append((0.0, 0.0, float(len(sids_))))
                continue
            top = int(cnt_.argmax())
            v_ = cnt_[pos_[ln[j]]]
            rank = float((cnt_ > v_).sum() + (cnt_ == v_).sum() - 1)
            think[key].append((float(sids_[top] == ln[j]), float(sids_[top] == ln[j - 1]), rank))

    FC = None
    if fresh_codes:                                       # each syllable's concept code as the concept area lights it
        if not sids_:                                     # NOW (heard alone, >= half of the adult voices; the speech
            cnt0 = np.bincount(life.train.syl[:200000].astype(np.int64))     # link's codes were measured once)
            sids_ = [int(x) for x in np.argsort(-cnt0)[:600] if cnt0[x] > 0]
            pos_.update({x: i for i, x in enumerate(sids_)})
        from ..world.voice import ADULTS
        FC = torch.zeros((len(sids_), cb - ca), device=p.L1ext.on.device)
        for i_, x_ in enumerate(sids_):
            for v_ in ADULTS:
                rr = []
                life.play([None] * 4 + life.syl(x_, v_) + [None] * 2, rec=lambda t: rr.append(p.L1ext.on.clone()))
                FC[i_] += torch.stack(rr[4 + lag:4 + lag + 2])[:, ca:cb].any(0).float() / len(ADULTS)
        FC = (FC >= 0.5).float()
        msg = ['CODES now: units per syllable', round(float(FC.sum(1).mean()), 1)]
        if G1C is not None:
            old_ = torch.zeros_like(FC)
            old_[:, (G1C.units - ca)] = G1C.code
            msg += ['at wiring', round(float(old_.sum(1).mean()), 1), 'share of the old code still lit',
                    round(float(((FC > 0) & (old_ > 0)).sum() / old_.sum().clamp_min(1)), 3)]
        print(*msg, flush=True)
    cnt_all = np.bincount(life.train.syl[:200000].astype(np.int64))
    freq_order = sorted(sids_, key=lambda x: -cnt_all[x] if x < len(cnt_all) else 0)
    freq_pos = {x: i for i, x in enumerate(freq_order)}   # the next syllable's rank by frequency alone

    def push_rec(srcs, w, ln=None, key=None):
        rec = []

        def r_(t):
            rec.append((p.L1ext.rose().clone(), p.e1.clone(), p.x_int.clone(), p.L1ext.on.clone()))
            xi = p.x_int
            if xi.any():                                  # which source drives the lit thoughts
                for k_, v_ in p._comp.items():
                    src_share[k_] = src_share.get(k_, 0.0) + float(v_[xi].clamp_min(0).sum())
            xc = xi[ca:cb]
            if xc.any():                                  # ... the lit concept thoughts only
                for k_, v_ in p._comp.items():
                    src_share_c[k_] = src_share_c.get(k_, 0.0) + float(v_[ca:cb][xc].clamp_min(0).sum())
            if G1C is not None:
                cnt_ = G1C.code @ xi[G1C.units].float()
                conc.append(float(cnt_.max()) >= k_say)
        life.play(srcs, rec=r_)
        if ln is not None:
            think_of(rec, ln, key)
        out = []
        for t_ in range(w.start, min(w.stop, len(rec))):  # the lag profile of the lit concept thoughts
            xc = rec[t_][2][ca:cb]
            if int(xc.sum()):
                for k_ in lag_ov:
                    if 0 <= t_ + k_ < len(rec):
                        lag_ov[k_].append(float((xc & rec[t_ + k_][3][ca:cb]).sum()) / float(xc.sum()))
        lit_c = []                                        # arriving concept copies lit inside the tick before
        for t_ in range(w.start, min(w.stop, len(rec))):
            rose = rec[t_][0][ca:cb]
            if t_ > 0 and int(rose.sum()):
                lit_c.append(float((rose & rec[t_ - 1][2][ca:cb]).sum()) / float(rose.sum()))
        for rose, e1, xi, _ in rec[w]:
            m = rose if cand is None else (rose & cand)
            e = e1 if cand is None else e1[cand]
            r_ = m if cand is None else m[cand]
            if int(r_.sum()) == 0:
                continue
            srt = torch.sort(e).values
            pct = torch.searchsorted(srt, e[r_]).float() / len(e)
            top = torch.topk(e, int(r_.sum())).indices
            out.append((float(pct.mean()), float(r_[top].float().mean())))
        th = torch.stack([x[2] for x in rec[w]]).any(0)
        r_ = np.array(out).mean(0) if out else np.array([np.nan, np.nan])
        return np.append(r_, np.mean(lit_c) if lit_c else np.nan), th

    for nm, lines in (('heard', first_lines(life.train, order, n, lo=4)),
                      ('novel', first_lines(life.heldout, range(life.heldout.n), n, lo=4))):
        R, C, TH = [], [], []
        for ln in lines:
            sh = shuffled(ln)
            if sh is None:
                continue
            w = window(ln)
            for v in ('C1', 'D2'):
                a_, th = push_rec(srcs_of(ln, v), w, ln, 'own')
                c_, _ = push_rec(srcs_of(sh, v), w, sh, 'shuffled')
                R.append(a_)
                C.append(c_)
                TH.append(th)
        R, C = np.array(R), np.array(C)
        jac = [float((TH[i] & TH[j]).sum()) / max(1.0, float((TH[i] | TH[j]).sum()))
               for i in range(0, len(TH), 2) for j in range(i + 2, len(TH), 2)]
        print('PUSH', nm, json.dumps(dict(
            percentile_own_shuffled=[round(float(np.nanmean(R[:, 0])), 4), round(float(np.nanmean(C[:, 0])), 4),
                                     round(float(np.nanmean(R[:, 0] - C[:, 0])), 4),
                                     round(float(np.nanstd(R[:, 0] - C[:, 0]) / np.sqrt(len(R))), 4)],
            in_top_own_shuffled=[round(float(np.nanmean(R[:, 1])), 4), round(float(np.nanmean(C[:, 1])), 4),
                                 round(float(np.nanmean(R[:, 1] - C[:, 1])), 4),
                                 round(float(np.nanstd(R[:, 1] - C[:, 1]) / np.sqrt(len(R))), 4)],
            concept_lit_before_own_shuffled=[round(float(np.nanmean(R[:, 2])), 4), round(float(np.nanmean(C[:, 2])), 4),
                                             round(float(np.nanmean(R[:, 2] - C[:, 2])), 4),
                                             round(float(np.nanstd(R[:, 2] - C[:, 2]) / np.sqrt(len(R))), 4)],
            thoughts_jaccard_between_lines=round(float(np.mean(jac)), 3),
            thought_set_size=round(float(np.mean([float(t.sum()) for t in TH])), 1),
            thought_kinds={k_: round(float(np.mean([float(t[a_:b_].sum()) for t in TH])), 1)
                           for k_, a_, b_ in kinds},
            jaccard_by_kind={k_: round(float(np.mean([
                float((TH[i][a_:b_] & TH[j][a_:b_]).sum()) / max(1.0, float((TH[i][a_:b_] | TH[j][a_:b_]).sum()))
                for i in range(0, len(TH), 2) for j in range(i + 2, len(TH), 2)])), 3) for k_, a_, b_ in kinds},
            secs=round(time.time() - t0))), flush=True)
    tot = max(1e-9, sum(src_share.values()))
    print('SOURCES of the lit thoughts\' drive', json.dumps({k_: round(v_ / tot, 3) for k_, v_ in src_share.items()}),
          flush=True)
    tot = max(1e-9, sum(src_share_c.values()))
    print('SOURCES of the lit concept thoughts\' drive', json.dumps({k_: round(v_ / tot, 3) for k_, v_ in
                                                                     src_share_c.items()}), flush=True)
    print('LAG lit concept thoughts at t vs concept copies heard at t+k', json.dumps(
        {k_: round(float(np.mean(v_)), 4) for k_, v_ in lag_ov.items() if v_}), flush=True)
    if tend['own']:
        st_ = lambda r: dict(top1=round(float(np.mean(r == 0)), 4), top10=round(float(np.mean(r < 10)), 4),
                             mean_rank=round(float(np.mean(r)), 1))
        print('TENDENCY (rank of the next syllable among', len(sids_), 'wired by the summed push)', json.dumps(
            {k_: st_(np.array(v_)[:, 0]) for k_, v_ in tend.items() if v_} |
            {'frequency (own)': st_(np.array(tend['own'])[:, 1])} |
            ({'check: the arriving copies themselves': st_(np.array(ctrl))} if ctrl else {})), flush=True)
    if think['own']:
        print('THINK (thought closest to: the next syllable, the one just heard; rank of the next among',
              len(sids_), 'wired)', json.dumps({k_: [round(float(x), 4) for x in np.mean(np.array(v_), 0)]
                                                   for k_, v_ in think.items() if v_}), flush=True)
    if conc:
        print('CONCENTRATED one syllable has >=', k_say, 'code units thought, share of ticks',
              round(float(np.mean(conc)), 4), flush=True)
    if 'calib' in parts and FC is not None:
        calib(life, p, FC, sids_, ca, cb, lag)
    if 'say' in parts and hasattr(life, 'speak_test'):
        print('SAY frozen', json.dumps(life.speak_test()), flush=True)
    if 'lit' not in parts:
        return

    for nm, lines in (('heard', first_lines(life.train, order, n, lo=4)),
                      ('novel', first_lines(life.heldout, range(life.heldout.n), n, lo=4))):
        R, C = [], []
        for ln in lines:
            sh = shuffled(ln)
            if sh is None:
                continue
            w = window(ln)
            for v in ('C1', 'D2'):
                R.append(share(listen(life, srcs_of(ln, v), w)))
                C.append(share(listen(life, srcs_of(sh, v), w)))
        print('ORDER', nm, json.dumps(dict(own_shuffled=compare(np.array(R), np.array(C)), items=len(R),
                                           secs=round(time.time() - t0))), flush=True)
    pairs = []
    for d in order:
        tr = life.train.turns(int(d))
        for j in range(len(tr) - 1):
            if 2 <= len(tr[j]) <= 10 and len(tr[j + 1]) >= 2:
                pairs.append(([int(x) for x in tr[j]], [int(x) for x in tr[j + 1][:4]]))
                break
        if len(pairs) >= n:
            break
    R, C = [], []
    for i, (pr, rp) in enumerate(pairs):
        for reply in (rp, pairs[(i + 1) % len(pairs)][1]):
            srcs = [None] * lead + [x for s_ in pr for x in life.syl(s_, 'C1')] + [None] * 6 + \
                [x for s_ in reply for x in life.syl(s_, 'D1')] + [None] * 3
            e = lead + T * len(pr) + 6 + lag                  # from the reply's first arrival
            (R if reply is rp else C).append(share(listen(life, srcs, slice(e, len(srcs)))))
    print('ORDER reply', json.dumps(dict(real_other=compare(np.array(R), np.array(C)), items=len(R),
                                         secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], n=int(kw.get('n', 40)), loop_scale=float(kw.get('loop_scale', 1.0)),
         gt_scale=float(kw.get('gt_scale', 1.0)), parts=kw.get('parts', 'push,lit,reply'),
         gt_no_motor=int(kw.get('gt_no_motor', 0)), concept_gt_only=int(kw.get('concept_gt_only', 0)),
         cset={k_[2:]: json.loads(v_) for k_, v_ in kw.items() if k_.startswith('c.')},
         fresh_codes=int(kw.get('fresh_codes', 0)))
