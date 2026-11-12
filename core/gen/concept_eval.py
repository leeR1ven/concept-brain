"""Concept evaluation by the designer's definition (GENERATOR.md 11-12; Fable a23).
Designer: "概念就是总是激活的神经元，而特征就是偶尔激活的神经元"; "音色不同但是意思相同的两句话".
A concept sheet gives the same cells for the same syllable in different voices; a feature sheet does not.

Input per sheet: hearing codes X [n, cells] (spike counts over the held ticks, lag 0), with per hearing the syllable
y, the voice v (0..3) and the dialogue d. Splits are by dialogue (even / odd), never by hearing; training counts are
equal (n_tr hearings per class and voice); classes = the commonest syllables with enough hearings in every voice.

Measures:
  within / cross   nearest-class-mean accuracy, means from one half of the dialogues: within = same voice,
                   cross = the other three voices (leave one voice out); tested on the other half in that voice
  voice decoding   which voice spoke (4 balanced classes), from the same codes
  VI               [sim(same syl, other voice) - sim(other syl)] / [sim(same syl, same voice) - sim(other syl)],
                   sim = Jaccard of the firing sets of two hearings from different dialogues
  core             cells active in >= core_q of a syllable's hearings (n_tr per voice, all voices), minus the same
                   with shuffled labels (busy cells); specificity = in how many cores a core cell lies
  core retention   a core from three voices: share of it firing on the left-out voice's hearings of that syllable,
                   against the share of other syllables' cores firing
Sanity: every accuracy with shuffled labels (should be chance), and a silent code scores 0."""
import numpy as np
import torch


def _ncm_acc(Xtr, ytr, Xte, yte, classes):
    """nearest class mean with cosine; a silent test code is wrong"""
    if len(yte) == 0:
        return float('nan')
    M = torch.stack([Xtr[ytr == c].mean(0) for c in classes])
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Xn = Xte / Xte.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos = {c: i for i, c in enumerate(classes)}
    ti = torch.as_tensor([pos[int(c)] for c in yte.tolist()], device=Xte.device)
    ok = ((Xn @ Mn.T).argmax(1) == ti) & (Xte.norm(dim=1) > 0)
    return ok.float().mean().item()


def _pick(idx, n, rng):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def evaluate(X, y, v, d, n_tr=8, max_cls=60, core_q=0.8, seed=0, half=0):
    """X: dict sheet -> tensor [n, cells]; y, v, d: numpy arrays. Returns dict sheet -> measures (one dialogue split;
    half=1 swaps the roles of the two halves)"""
    rng = np.random.default_rng(seed)
    voices = sorted(set(v.tolist()))
    tr_half = (d % 2) == half
    te_half = ~tr_half
    # classes: enough training hearings in every voice in the training half, and test hearings in every voice
    cnt = {}
    for c in np.unique(y):
        ok = all(((y == c) & (v == vo) & tr_half).sum() >= n_tr and ((y == c) & (v == vo) & te_half).sum() >= 1
                 for vo in voices)
        if ok:
            cnt[int(c)] = int((y == c).sum())
    classes = sorted(cnt, key=lambda c: -cnt[c])[:max_cls]
    out = {'_classes': len(classes), '_voices': len(voices)}
    if len(classes) < 5:
        return out
    # training sets: exactly n_tr per class and voice from the training half
    tr_idx = {(c, vo): _pick(np.nonzero((y == c) & (v == vo) & tr_half)[0], n_tr, rng) for c in classes for vo in voices}
    te_mask = te_half & np.isin(y, classes)
    ys = rng.permutation(y)                                       # shuffled labels (sanity)
    for sh, Xs in X.items():
        r = {}
        Xf = Xs.float()
        w_acc, c_acc, w_sh = [], [], []
        for vo in voices:
            te = np.nonzero(te_mask & (v == vo))[0]
            Xte, yte = Xf[te], y[te]
            tw = np.concatenate([tr_idx[(c, vo)] for c in classes])
            # equal counts: cross uses n_tr per class too (drawn evenly from the other voices)
            tc = np.concatenate([_pick(np.concatenate([tr_idx[(c, o)] for o in voices if o != vo]), n_tr, rng)
                                 for c in classes])
            w_acc.append(_ncm_acc(Xf[tw], y[tw], Xte, yte, classes))
            c_acc.append(_ncm_acc(Xf[tc], y[tc], Xte, yte, classes))
            w_sh.append(_ncm_acc(Xf[tw], ys[tw], Xte, yte, classes))
        r['within'] = round(float(np.nanmean(w_acc)), 3)
        r['cross'] = round(float(np.nanmean(c_acc)), 3)
        r['shuffled'] = round(float(np.nanmean(w_sh)), 3)
        r['within by voice'] = [round(float(x), 3) for x in w_acc]     # per left-out voice (a voice never taught
        r['cross by voice'] = [round(float(x), 3) for x in c_acc]      # with signs is the generalisation test)
        # voice decoding: balanced, training half -> test half
        nv = min(int(((v == vo) & tr_half & np.isin(y, classes)).sum()) for vo in voices)
        tv = np.concatenate([_pick(np.nonzero((v == vo) & tr_half & np.isin(y, classes))[0], min(nv, 200), rng)
                             for vo in voices])
        tev = np.nonzero(te_mask)[0]
        r['voice'] = round(_ncm_acc(Xf[tv], v[tv], Xf[tev], v[tev], voices), 3)
        # VI from firing-set overlaps between hearings of different dialogues
        B_ = (Xf > 0).float()
        sel = np.nonzero(np.isin(y, classes))[0]
        sel = _pick(sel, 1500, rng)
        S = B_[sel]
        inter = S @ S.T
        uni = S.sum(1)[:, None] + S.sum(1)[None] - inter
        J = (inter / uni.clamp_min(1)).cpu().numpy()
        ysel, vsel, dsel = y[sel], v[sel], d[sel]
        dd = dsel[:, None] != dsel[None]
        ss = (ysel[:, None] == ysel[None]) & dd
        same_v = ss & (vsel[:, None] == vsel[None])
        other_v = ss & (vsel[:, None] != vsel[None])
        diff = (ysel[:, None] != ysel[None]) & dd
        if same_v.any() and other_v.any() and diff.any():
            a, b, c = J[same_v].mean(), J[other_v].mean(), J[diff].mean()
            r['VI'] = round(float((b - c) / (a - c)), 3) if (a - c) > 1e-6 else None
            r['sim same/other-voice/diff'] = [round(float(a), 3), round(float(b), 3), round(float(c), 3)]
        # cores: n_tr hearings per voice per class
        allc = {c: np.concatenate([tr_idx[(c, vo)] for vo in voices]) for c in classes}
        cores = {c: (B_[allc[c]].mean(0) >= core_q) for c in classes}
        sizes = [int(cores[c].sum()) for c in classes]
        # the null: random groups of hearings of the same size (busy cells are in every 'core')
        pool = np.concatenate(list(allc.values()))
        rg_sizes = [int((B_[_pick(pool, len(allc[classes[0]]), rng)].mean(0) >= core_q).sum()) for _ in classes]
        r['core size'] = round(float(np.mean(sizes)), 1)
        r['core size random groups'] = round(float(np.mean(rg_sizes)), 1)
        C = torch.stack([cores[c] for c in classes]).float()
        member = C.sum(0)
        incore = member > 0
        r['core cell in n cores'] = round(float(member[incore].mean()), 2) if bool(incore.any()) else None
        # leave-one-voice-out core retention
        keep, oth = [], []
        for vo in voices:
            cs = {c: (B_[np.concatenate([tr_idx[(c, o)] for o in voices if o != vo])].mean(0) >= core_q) for c in classes}
            Cm = torch.stack([cs[c] for c in classes]).float()
            sz = Cm.sum(1).clamp_min(1)
            te = np.nonzero(te_mask & (v == vo))[0]
            if len(te) == 0:
                continue
            F = B_[te]                                              # firing sets of the left-out voice
            fr = (F @ Cm.T) / sz[None]                              # [n_te, n_cls] share of each core firing
            pos = {c: i for i, c in enumerate(classes)}
            ti = torch.as_tensor([pos[int(c)] for c in y[te].tolist()], device=F.device)
            own = fr.gather(1, ti[:, None]).squeeze(1)
            other = (fr.sum(1) - own) / max(len(classes) - 1, 1)
            valid = Cm.sum(1)[ti] > 0
            if bool(valid.any()):
                keep.append(own[valid].mean().item())
                oth.append(other[valid].mean().item())
        r['core retention own/other'] = [round(float(np.mean(keep)), 3), round(float(np.mean(oth)), 3)] if keep else None
        out[sh] = r
    return out


def random_projection(XA, n_out, k_active, seed=0):
    """the 'no concept' control: a fixed random projection of A's codes to n_out cells, the k most driven firing"""
    g = torch.Generator(device='cpu').manual_seed(seed)
    W = torch.randn(XA.shape[1], n_out, generator=g).to(XA.device)
    Z = XA.float() @ W
    k = max(1, int(k_active))
    top = Z.topk(k, dim=1).indices
    out = torch.zeros(XA.shape[0], n_out, device=XA.device)
    out.scatter_(1, top, 1.0)
    out[XA.float().sum(1) == 0] = 0.0
    return out
