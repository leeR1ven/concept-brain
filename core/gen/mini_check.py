"""Frozen checks on a saved minimal-generator brain (no learning, no training; designer 11-12: "你不需要重新训练啊，拿原来
训练过的模型直接检测…"). Fable a34's table:

  W's code sent on = units whose LEARNED push is >= thr above their own usual (capped at k), against the fixed top-k:
    completeness  share of the sign's set (what the sight of the syllable lights) with learned push >= thr
    purity        share of the sent-on cells inside the sign's set
    recurrence    share of hearing 1's cells lit again at the same place, the same dialogues heard again in the same
                  voices (only the ear's noise differs) and in fresh voices
    empty         share of sound ticks with nothing sent on: taught syllables / held syllables / the held voice
    ridge         the linear bound to the next W code (class-mean readout), from the sent-on code and from the top-k
  the top's order (the designer's test): the share of the actual next W code inside the top k of the top's
    expectation above its own usual, dialogues in their own order against the same turns with syllables shuffled,
    and against the average W pattern; 'next 3': inside the union of the next three codes

usage: python -m core.gen.mini_check <ckpt> [nd=384] [thr=1.0] [d0=15000]"""
import json
import os
import sys

import numpy as np
import torch


def main():
    ck = sys.argv[1]
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    nd, thr, d0 = int(kw.get('nd', 384)), float(kw.get('thr', 1.0)), int(kw.get('d0', 15000))
    saved = torch.load(ck, weights_only=False, map_location='cpu')
    os.environ['GEN_ARGS'] = ' '.join(f'{k}={v}' if isinstance(v, str) else f"{k}={json.dumps(v, separators=(',', ':'))}"
                                      for k, v in saved['P'].items() if k != 'save' and v != '')
    from . import mini as M
    M.load(ck)
    B, W, DEV = M.B, M.W, M.DEV
    per = nd // B
    lists = [range(d0 + b * per, d0 + (b + 1) * per) for b in range(B)]
    sign = {}

    def sign_of(sid):
        if sid not in sign:
            e = torch.as_tensor(M.sight(sid), device=DEV).float()[None].expand(B, -1)
            sign[sid] = M.topk_b(W.branch('eye', e, False), W.k)[0]
        return sign[sid]

    def hear(tag, dl=None):
        """one frozen hearing of the lists: per tick the top-k code, the sent-on code, the top's expectation"""
        if dl is not None:
            was = M.TRAIN
            M.TRAIN = dl
        S = [M.stream(lists[b], tag, False, False) for b in range(B)]
        if dl is not None:
            M.TRAIN = was
        T = min(len(s) for s in S)
        M.reset()
        rec = dict(top=[], sent=[], Ex=[], lp1=[])
        for t in range(T):
            ear, eye, seen = M.batch_in(S, t)
            M.tick(ear, eye, seen, False, False)
            lp = W.learned(M.A1.on.float(), False)
            rec['top'].append(W.on.clone())
            rec['sent'].append((lp >= thr) & M.topk_b(lp, W.k))
            rec['lp1'].append(lp >= thr)
            rec['Ex'].append((M.ST['ExT'] - M.ST['usualE']).half())
        meta = [[S[b][t][2:6] for t in range(T)] for b in range(B)]     # syllable, voice, dialogue, (turn, place)
        return {k: torch.stack(v, 1) for k, v in rec.items()}, meta, T

    out = {'thr': thr, 'nd': nd}
    R1, m1, T1 = hear(9301)
    # completeness / purity / empty (taught, held syllables, held voice)
    comp, pur, empty = [], [], dict(taught=[], held_syllable=[], held_voice=[])
    for b in range(B):
        for t in range(T1):
            sid, vo = m1[b][t][0], m1[b][t][1]
            if sid < 0:
                continue
            s_ = R1['sent'][b, t]
            grp = 'held_voice' if vo == M.P['held_voice'] else ('held_syllable' if sid in M.HELD else 'taught')
            empty[grp].append(float(s_.sum() == 0))
            if grp == 'taught':
                sg = sign_of(sid)
                comp.append(float((R1['lp1'][b, t] & sg).sum()) / max(1.0, float(sg.sum())))
                if s_.any():
                    pur.append(float((s_ & sg).sum()) / float(s_.sum()))
    out['completeness'] = round(float(np.mean(comp)), 3)
    out['purity'] = round(float(np.mean(pur)), 3)
    out['sent per sound tick (taught)'] = round(float(np.mean(
        [float(R1['sent'][b, t].sum()) for b in range(B) for t in range(T1)
         if m1[b][t][0] >= 0 and m1[b][t][0] not in M.HELD and m1[b][t][1] != M.P['held_voice']])), 1)
    out['empty'] = {k: round(float(np.mean(v)), 3) if v else None for k, v in empty.items()}
    print('CHECK', json.dumps(out), flush=True)
    # recurrence: the same dialogues again, same voices (tag) and fresh voices
    for nm, tag in (('same voices', 9301), ('fresh voices', 9302)):
        R2_, m2, T2 = hear(tag)
        res = {k: [] for k in ('top', 'sent')}
        for b in range(B):
            key = lambda x: (x[2],) + x[3]
            at = {key(m2[b][t]): t for t in range(T2) if m2[b][t][3] is not None}
            for t in range(T1):
                if m1[b][t][3] is None or m1[b][t][0] in M.HELD or m1[b][t][1] == M.P['held_voice']:
                    continue
                t2 = at.get(key(m1[b][t]))
                if t2 is None:
                    continue
                for k in res:
                    a_ = R1[k][b, t]
                    if a_.any():
                        res[k].append(float((a_ & R2_[k][b, t2]).sum()) / float(a_.sum()))
        out[f'recurrence, {nm}'] = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
        del R2_
        print('CHECK', json.dumps(out), flush=True)
    # the linear bound to the next top-k code, from the sent-on code and from the top-k code (as in mini.probe)
    lab = np.array([[m1[b][t][0] for t in range(T1)] for b in range(B)])
    dl = np.array([[m1[b][t][2] for t in range(T1)] for b in range(B)])
    TG = R1['top'].float()
    items = [(b, t) for b in range(B) for t in range(T1 - 1) if lab[b, t] >= 0 and lab[b, t + 1] >= 0]
    cnt = np.bincount(np.array([lab[b, t + 1] for b, t in items]))
    cands = [int(c_) for c_ in np.argsort(-cnt)[:300] if cnt[c_] > 0]
    pos = {c_: i for i, c_ in enumerate(cands)}
    its = [(b, t) for b, t in items if int(lab[b, t + 1]) in pos]
    nxt = np.array([lab[b, t + 1] for b, t in its])
    M_ = torch.stack([TG[[b for (b, t), l_ in zip(its, nxt) if l_ == c_],
                         [t + 1 for (b, t), l_ in zip(its, nxt) if l_ == c_]].mean(0) for c_ in cands]).to(DEV)
    M_ = M_ - M_.mean(0, keepdim=True)
    Mn = M_ / M_.norm(dim=1, keepdim=True).clamp_min(1e-9)
    tr_ = [i for i, (b, t) in enumerate(its) if dl[b, t] % 2 == 0]
    te_ = [i for i, (b, t) in enumerate(its) if dl[b, t] % 2 == 1]
    Yr = torch.stack([TG[b, t + 1] for b, t in its]).to(DEV)
    tru = torch.as_tensor([pos[int(nxt[i])] for i in te_], device=DEV)

    def ridge(Xr, lam):
        Xa = Xr[tr_]
        Wm = torch.linalg.solve(Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=DEV), Xa.T @ Yr[tr_])
        Yh = Xr[te_] @ Wm - (Xa @ Wm).mean(0, keepdim=True)
        sc = (Yh / Yh.norm(dim=1, keepdim=True).clamp_min(1e-9)) @ Mn.T
        st_ = sc.gather(1, tru[:, None])
        rk = ((sc > st_).sum(1) + (sc == st_).sum(1) - 1).cpu().numpy()
        return dict(top1=round(float(np.mean(rk == 0)), 3), top10=round(float(np.mean(rk < 10)), 3))
    out['ridge to the next code'] = {
        'from the sent-on code': ridge(torch.stack([R1['sent'][b, t] for b, t in its]).float().to(DEV), 10.0),
        'from the top-k code': ridge(torch.stack([R1['top'][b, t] for b, t in its]).float().to(DEV), 10.0)}
    oh = torch.zeros(len(its), len(cands) + 1, device=DEV)
    for i, (b, t) in enumerate(its):
        oh[i, pos.get(int(lab[b, t]), len(cands))] = 1.0
    out['ridge to the next code']['from the one-hot syllable'] = ridge(oh, 1.0)
    print('CHECK', json.dumps(out), flush=True)

    # the designer's test on the top: own order against the same turns shuffled
    class Shuffled:
        def __init__(self, D):
            self.D, self.rng = D, np.random.default_rng(11)

        def turns(self, d):
            return [t[self.rng.permutation(len(t))] for t in self.D.turns(d)]

        def __getattr__(self, k):
            return getattr(self.D, k)
    wmean = M.ST['wmean'].cpu()
    for nm, dl_ in (('own order', None), ('shuffled', Shuffled(M.TRAIN))):
        R_, m_, T_ = hear(9301, dl_)
        lift, nx3, avg = [], [], []
        for b in range(B):
            for t in range(1, T_):
                if m_[b][t][0] < 0 or m_[b][t - 1][0] < 0:
                    continue
                top = torch.topk(R_['Ex'][b, t].float(), W.k).indices
                act = R_['top'][b, t]
                lift.append(float(act[top].float().mean()) * W.k / max(1.0, float(act.sum())))
                avg.append(float(act[torch.topk(wmean, W.k).indices].float().mean()) * W.k / max(1.0, float(act.sum())))
                u = R_['top'][b, t:t + 3].any(0)
                nx3.append(float(u[top].float().mean()))
        out[f'top, {nm}'] = dict(next=round(float(np.mean(lift)), 4), next3=round(float(np.mean(nx3)), 4),
                                 average_pattern=round(float(np.mean(avg)), 4), n=len(lift))
        print('CHECK', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
