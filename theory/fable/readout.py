"""The readout, computed from the recorded codes (codes.py) with the real timing and the real selection. numpy only.
Learning events in the stream "A then B" (B's onset 3 ticks after A's): a copy of B that rises at B-tick tau is paired
with every last-layer unit on at A-tick tau+3 or last on within the 3 ticks before (the rule's window), so with
A's code at A-ticks 4..6 and, for tau >= 4, with B's own early code. The readout after the corpus is a sum of such
pair counts weighted by N(A, B), turned into weights by the rule:
  count   w = count                        concave  w = sqrt(1 + 2 eta count) - 1   (the rule's slowing growth)
  cov     pre and post centred on their usual levels (signed weights)
  ls      the prediction-error rule at convergence: least squares, W = argmin sum_A pi_A |Ybar_A - W h_A|^2 + lam|W|^2
  online  the local prediction-error rule run event by event: dw_ij = eta a_i (x_j - yhat_j), yhat = the thought set
Test: cue A (another hearing: ear noise differs, or voice D1), the real selection: at ticks 5..8 the thought drive
e = W a(t-1) + G a(t-1) - usual, thoughts = the `budget` most driven above theta1. Decode: which syllable's copies the
thoughts match best (share of its copies lit, copies weighted by how specific they are); rank of the best follower B.
usage: python3 readout.py codes.npz '{"rule": "count|concave|cov|ls|online", "links": "net|all", ...}'"""
import json
import sys
import time

import numpy as np

TAUS = (1, 2, 3, 4, 5)        # B-ticks on which copies rise


def unpack(a, n):
    return np.unpackbits(a, axis=-1)[..., :n].astype(bool)


def load(path):
    z = np.load(path)
    n1 = int(z['n1'])
    # saved as [syllable, hearing, tick, n]; used as [hearing, syllable, tick, n]
    d = dict(syl=z['syl'], N=z['N'], n1=n1, ext=unpack(z['ext'], n1).transpose(1, 0, 2, 3),
             l4=unpack(z['l4'], n1).transpose(1, 0, 2, 3), push=z['push'].transpose(1, 0, 2, 3), idx=z['idx'], n_post=int(z['n_post']), loop_idx=z['loop_idx'], loop_w=z['loop_w'],
             budget=float(z['budget']), theta1=float(z['theta1']), eta=float(z['eta']))
    if 'ctx_l4' in z.files:
        d['ctx_l4'] = unpack(z['ctx_l4'], n1)
        d['ctx_ext'] = unpack(z['ctx_ext'], n1)
    return d


def rises(ext):
    """ext [S, T, n] -> per tau the copies rising at B-tick tau (index tau + 1)"""
    return {tau: ext[:, tau + 1] & ~ext[:, tau] for tau in TAUS}


def pre_sets(l4, tau):
    """last-layer units paired with a copy rising at B-tick tau: on at A-tick tau+3 or last on within 3 ticks before.
    A-ticks 1..3 are silent (nothing on), A-tick 7 and later belong to B's own code (B-tick 4...)"""
    a_ticks = [tau + 3 - k for k in range(4)]
    own = np.zeros_like(l4[:, 0])
    for t in a_ticks:
        if 4 <= t <= 6:
            own |= l4[:, t + 1]
    return own                       # [S, n4] A's part; B's own part is added by the caller


def self_pre(l4, tau):
    """B's own last-layer units paired with its copies rising at B-tick tau (A-tick tau + 3 = B-tick tau)"""
    out = np.zeros_like(l4[:, 0])
    for k in range(4):
        t = tau - k
        if t >= 4:
            out |= l4[:, t + 1]
    return out


def loop_drive(z, a):
    """birth loop G: each lit last-layer unit pushes its `fan` copies"""
    n = z['n1']
    out = np.zeros(a.shape[:-1] + (n,), np.float32)
    for row in range(a.shape[0]) if a.ndim == 2 else [None]:
        on = np.nonzero(a[row] if row is not None else a)[0]
        tgt = out[row] if row is not None else out
        np.add.at(tgt, z['loop_idx'][on].ravel(), z['loop_w'][on].ravel())
    return out


def link_mask(z, links):
    return None if links == 'all' else z['idx']


def apply_mask(W, idx, n_post):
    """keep only the links that exist (idx rows, padded with the sink n_post), in place, row chunks (low memory)"""
    for a in range(0, W.shape[0], 500):
        m = np.zeros((min(500, W.shape[0] - a), n_post + 1), bool)
        np.put_along_axis(m, idx[a:a + 500], True, axis=1)
        W[a:a + 500] *= m[:, :-1]
    return W


def counts(z, N, h=0, self_pairs=True):
    """pair counts [n4, n1] after the corpus (hearing h for the codes) = sum_tau pre_tau^T N rise_tau (+ self pairs)"""
    l4, ext = z['l4'][h], z['ext'][h]
    R = rises(ext)
    C = np.zeros((z['n1'], z['n1']), np.float32)
    nB = N.sum(0)                                  # how often each B is heard (as a follower)
    for tau in TAUS:
        P = pre_sets(l4, tau).astype(np.float32)
        C += P.T @ (N.astype(np.float32) @ R[tau].astype(np.float32))
        if self_pairs:
            Q = self_pre(l4, tau).astype(np.float32)
            C += (Q * nB[:, None]).T @ R[tau].astype(np.float32)
    return C


def cue_codes(z, h):
    """the cue's last-layer code per tick 4..7 -> drives thoughts at 5..8"""
    return [z['l4'][h][:, t + 1] for t in (4, 5, 6, 7)]


def specific_copies(z, h=0):
    """copies(X) weighted by specificity: a copy lit by many syllables says little. weight = 1 / (#syllables lighting it)"""
    cp = z['ext'][h][:, 2:7].any(1)                 # lit on B-ticks 1..5
    cnt = cp.sum(0).astype(np.float32)
    return cp, 1.0 / np.maximum(cnt, 1.0)


def evaluate(z, W, h_test=1, duty=0.75, tag=''):
    """the real selection; returns rank-based scores for the best follower"""
    N, n1 = z['N'], z['n1']
    S = N.shape[0]
    piA = N.sum(1) / N.sum()
    best = N.argmax(1)
    cp, spec = specific_copies(z, 0)
    codes = cue_codes(z, h_test)
    thoughts = np.zeros((S, n1), bool)
    drives = []
    for a in codes[:3]:                            # thoughts at ticks 5, 6, 7
        d = a.astype(np.float32) @ W + loop_drive(z, a)
        drives.append(d)
    usual = duty * sum(piA @ d for d in drives) / len(drives)
    budget = int(round(z['budget']))
    n_over = []
    for d in drives:
        e = d - usual
        sel = np.argpartition(-e, budget - 1, axis=1)[:, :budget]
        row = np.arange(S)[:, None]
        ok = e[row, sel] > z['theta1']
        n_over.append(float(ok.mean(1).mean() * budget))
        t = np.zeros_like(thoughts)
        t[row, sel] = True
        thoughts |= t
    # decode: score(X) = sum over thoughts of spec weight for copies of X / sum spec over copies of X
    M = thoughts.astype(np.float32) @ (cp.astype(np.float32) * spec[None]).T          # [cue, X]
    M /= (cp.astype(np.float32) * spec[None]).sum(1)[None]
    rank = (M > M[np.arange(S), best][:, None]).sum(1)
    hit_B = (thoughts & cp[best]).sum(1) / np.maximum(cp[best].sum(1), 1)              # share of B's copies lit
    chance_B = thoughts.sum(1) / n1
    self_hit = (thoughts & cp[np.arange(S)]).sum(1) / np.maximum(cp.sum(1), 1)
    return dict(tag=tag, top1=round(float((piA * (rank == 0)).sum()), 3), top5=round(float((piA * (rank < 5)).sum()), 3),
                mean_rank=round(float((piA * rank).sum()), 1), chance_top1=round(1.0 / S, 4),
                B_copies_lit=round(float((piA * hit_B).sum()), 3), chance_copies=round(float((piA * chance_B).sum()), 3),
                own_copies_lit=round(float((piA * self_hit).sum()), 3),
                thoughts_per_cue=round(float(thoughts.sum(1).mean()), 1), over_theta_per_tick=round(float(np.mean(n_over)), 1))


def ls_readout(z, N, lam=1e-2, h=0):
    """prediction-error rule at convergence: per tau, W_tau = argmin sum_A pi_A |Ybar_A - W h_A|^2 (kernel form), and the
    pre code is the union over the rule's window; the targets are the copies rising at tau. Summed over tau."""
    l4, ext = z['l4'][h], z['ext'][h]
    R = rises(ext)
    n1 = z['n1']
    W = np.zeros((n1, n1), np.float32)
    for tau in TAUS:
        H = pre_sets(l4, tau).astype(np.float64)            # [S, n4]
        piA = N.sum(1).astype(np.float64)
        Yb = (N @ R[tau].astype(np.float64)) / np.maximum(piA, 1)[:, None]
        K = H @ H.T
        lam_ = lam * np.trace(K) / len(K)
        A = (K * piA[None, :]) + lam_ * np.eye(len(K))
        W += H.astype(np.float32).T @ (piA[:, None] * np.linalg.solve(A, Yb)).astype(np.float32)
    return W


def online(z, N, eta=0.05, events=60000, seed=0, h=0, mask=None, signed=True, log=None):
    """the local prediction-error rule, event by event, on the stream's real timing (simplified to one tick per
    syllable pair): sample (A, B) ~ N; thoughts from A's code; grow a_i -> x_j for copies that rise unpredicted,
    weaken a_i -> yhat_j for thoughts that did not come. Returns W."""
    rng = np.random.default_rng(seed)
    n1 = z['n1']
    l4, ext = z['l4'][h], z['ext'][h]
    pre = pre_sets(l4, 2) | pre_sets(l4, 3)                  # A's code in the window (ticks 4..6)
    cpB = ext[:, 2:7].any(1)                                 # copies of B (ticks 1..5)
    W = np.zeros((n1, n1), np.float32)
    usual = np.zeros(n1, np.float32)
    p = (N / N.sum()).ravel()
    pairs = rng.choice(len(p), events, p=p)
    budget = int(round(z['budget']))
    for k, pr in enumerate(pairs):
        A, B = divmod(int(pr), N.shape[1])
        a = np.nonzero(pre[A])[0]
        d = W[a].sum(0) + loop_drive(z, pre[A])
        usual += (d - usual) / 1000.0
        e = d - usual
        sel = np.argpartition(-e, budget - 1)[:budget]
        yhat = np.zeros(n1, bool)
        yhat[sel[e[sel] > z['theta1']]] = True
        x = cpB[B]
        err = x.astype(np.float32) - (yhat.astype(np.float32) if signed else 0.0)
        err[x & yhat] = 0.0
        dW = eta * err[None, :]
        W[a] += dW
        if not signed:
            np.maximum(W[a], 0.0, out=W[a])
        if log is not None and (k + 1) % log == 0:
            print(json.dumps(dict(events=k + 1, **evaluate(z, W, tag='online'))), flush=True)
    return W


def main():
    t0 = time.time()
    z = load(sys.argv[1])
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    rule, links = kw.get('rule', 'count'), kw.get('links', 'net')
    N = z['N']
    mask = link_mask(z, links)
    if rule in ('count', 'concave', 'cov'):
        if rule == 'cov':
            piA = N.sum(1) / N.sum()
            l4, ext = z['l4'][0], z['ext'][0]
            R = rises(ext)
            W = np.zeros((z['n1'], z['n1']), np.float32)
            for tau in TAUS:
                P = pre_sets(l4, tau).astype(np.float32)
                P -= (piA @ P)[None]
                X = R[tau].astype(np.float32)
                X -= (piA @ X)[None]
                W += P.T @ (N.astype(np.float32) @ X)
        else:
            W = counts(z, N, self_pairs=kw.get('self', 1))
            if rule == 'concave':
                W = np.sqrt(1.0 + 2.0 * z['eta'] * W) - 1.0
            else:
                W = z['eta'] * W
    elif rule == 'ls':
        W = ls_readout(z, N, lam=kw.get('lam', 1e-2))
    elif rule == 'online':
        W = online(z, N, eta=kw.get('eta', 0.05), events=kw.get('events', 60000), mask=mask,
                   signed=kw.get('signed', 1), log=kw.get('log', 20000))
    if mask is not None:
        apply_mask(W, mask, z['n_post'])
    for h, name in ((1, 'ear noise'), (2, 'voice D1')):
        r = evaluate(z, W, h_test=h, duty=kw.get('duty', 0.75), tag=f'{rule}/{links}/{name}')
        print(json.dumps(dict(**r, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
