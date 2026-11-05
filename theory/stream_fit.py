"""Readouts computed from a recording (stream_record.py). At the start of each syllable of a test line: of the copies that
really come on (m of them), what share are among the m copies a predictor rates highest? No labels in any readout.
  frequency  how often each copy comes on, whatever happened before: the chance level
  ideal      knows the syllable and has counted which syllable follows which in the same stream: the ceiling
  copies     regression from the outside copies themselves, how much of the order the input carries linearly
  last layer regression from the last layer, at each delay between it and the copies (the architecture has delay 1)
own order vs shuffled lines of the same syllables, on heard lines and never-heard lines.
usage: python stream_fit.py <rec.npz>"""
import sys

import numpy as np
import torch

torch.set_num_threads(4)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LAM = (10.0, 100.0, 1000.0)
DELAY = (1, 2)


class Rows:
    """binary tick x unit matrix kept as index lists"""

    def __init__(self, ix, off, n):
        self.ix, self.off, self.n, self.t = ix.astype(np.int64), off.astype(np.int64), n, len(off) - 1

    def row(self, t):
        return self.ix[self.off[t]:self.off[t + 1]]

    def dense(self, rows, use=None):
        rows = np.asarray(rows)
        M = np.zeros((len(rows), self.n), np.float32)
        for i, t in enumerate(rows):
            M[i, self.row(t)] = 1.0
        return M if use is None else M[:, use]

    def total(self, rows):
        return int(sum(self.off[t + 1] - self.off[t] for t in rows))


def rising(X):
    ix, off = [], [0]
    prev = np.zeros(0, np.int64)
    for t in range(X.t):
        cur = X.row(t)
        new = np.setdiff1d(cur, prev, assume_unique=True)
        ix.append(new)
        off.append(off[-1] + len(new))
        prev = cur
    return Rows(np.concatenate(ix), np.array(off), X.n)


def chunks(a, size=4000):
    return [a[i:i + size] for i in range(0, len(a), size)]


def main():
    z = np.load(sys.argv[1])
    n, T0 = int(z['n']), int(z['train_ticks'])
    lab = z['lab']
    A, X = Rows(z['a'], z['a_off'], n), Rows(z['x'], z['x_off'], n)
    N = lab.shape[0]
    R = rising(X)
    sid, k, tag, pos = lab[:, 0], lab[:, 1], lab[:, 2], lab[:, 4]
    tr = np.arange(N) < T0
    print('ticks', T0, 'of', N, 'speech', float((sid[tr] >= 0).mean()), 'last layer lit per tick', A.total(range(T0)) / T0,
          'copies on', X.total(range(T0)) / T0, 'coming on', R.total(range(T0)) / T0, flush=True)

    # only the copies that answer a sound at once: at a syllable's start the rest are still the echo of the syllable before
    q = np.nonzero(tr & (k == 0))[0]
    q = q[(q >= 4) & (q < T0 - 5)]
    q = q[[bool((sid[i - 4:i] < 0).all()) for i in q]]
    c1 = np.bincount(np.concatenate([R.row(i + 1) for i in q]), minlength=n)
    c2 = np.bincount(np.concatenate([R.row(i + d) for i in q for d in (2, 3, 4)]), minlength=n)
    early = (c1 >= 3) & (c1 > c2) if 'early' in sys.argv else np.ones(n, bool)
    keep, off = [], [0]
    for i in range(N):
        r = R.row(i)
        r = r[early[r]]
        keep.append(r)
        off.append(off[-1] + len(r))
    R = Rows(np.concatenate(keep), np.array(off), n)
    print('starts after silence', len(q), 'copies that answer at once', int(early.sum()), flush=True)

    # the lag from a syllable's first sound tick to the copies: the tick at which most of them come on
    on = np.nonzero(tr & (k == 0))[0]
    prof = [R.total(np.minimum(on + d, N - 1)) / len(on) for d in range(0, 6)]
    print('copies coming on, by ticks after the syllable begins', np.round(prof, 1), flush=True)
    hit = int(np.argmax(prof))
    for a_ in sys.argv[2:]:
        if a_.startswith('hit='):
            hit = int(a_[4:])
    print('scoring the copies that come on', hit, 'ticks after a syllable begins', flush=True)

    # labelled symbol sequence of the heard stream: within a turn only
    onset = np.nonzero(k == 0)[0]
    otr = onset[onset < T0]
    seq, turn = sid[otr], lab[otr, 2] * 1000 + lab[otr, 3]
    S = int(sid.max()) + 1
    uni = np.bincount(seq, minlength=S).astype(float)
    pairs = [(a, b) for a, b, ta, tb in zip(seq[:-1], seq[1:], turn[:-1], turn[1:]) if ta == tb]
    bi = {}
    for a_, b_ in pairs:
        bi.setdefault(int(a_), {}).setdefault(int(b_), 0)
        bi[int(a_)][int(b_)] += 1
    nb = sum(len(v) for v in bi.values())
    print('heard syllables', len(seq), 'distinct', int((uni > 0).sum()), 'pairs', len(pairs),
          'distinct pairs', nb, 'mean hearings of a pair', len(pairs) / max(1, nb), flush=True)

    # mean code per syllable: copies coming on at its start, heard stream
    heard = np.nonzero(uni > 0)[0]
    mp = {s: i for i, s in enumerate(heard)}
    rows = onset[(onset < T0)]
    C = np.zeros((len(heard), n), np.float32)
    for s, r in zip(sid[rows], rows):
        C[mp[s], R.row(min(r + hit, N - 1))] += 1.0
    C /= np.maximum(1.0, uni[heard])[:, None]

    # test: for each line, each syllable from the 2nd on. tag = 1000 + 2000 kind + 4 line + order
    te = np.nonzero((~tr) & (k == 0) & (pos >= 1))[0]
    prev = {}
    for i in np.nonzero((~tr) & (k == 0))[0]:
        prev[(tag[i], pos[i])] = sid[i]
    cur = np.array([prev[(tag[i], pos[i] - 1)] for i in te])
    want = sid[te]
    kind = ((tag[te] - 1000) >= 2000).astype(int)
    order = (tag[te] - 1000) % 2
    line = ((tag[te] - 1000) % 2000) // 4 + 1000 * kind
    actual = R.dense(te + hit)
    m_true = actual.sum(1).astype(int)
    ok = m_true > 0
    te, cur, want, kind, order, line, actual, m_true = te[ok], cur[ok], want[ok], kind[ok], order[ok], line[ok], actual[ok], m_true[ok]
    print('test syllables', len(te), 'copies coming on at each', float(m_true.mean()), flush=True)

    def score(P):
        """P: test rows x copies, numpy or torch"""
        P = torch.as_tensor(P, device=DEV)
        idx = torch.topk(P, int(m_true.max()), dim=1).indices.cpu().numpy()
        return np.array([actual[i, idx[i, :m_true[i]]].mean() for i in range(len(te))])

    def report(name, right, mask=None):
        for kd, kn in ((0, 'heard lines'), (1, 'new lines')):
            m = (kind == kd) if mask is None else (kind == kd) & mask
            own, sh = right[m & (order == 0)], right[m & (order == 1)]
            lo, ls = line[m & (order == 0)], line[m & (order == 1)]
            ids = np.unique(lo)
            d = np.array([own[lo == i].mean() - sh[ls == i].mean() for i in ids if (lo == i).any() and (ls == i).any()])
            print(f'  {name:22s} {kn:12s} own {100 * own.mean():5.1f}%  shuffled {100 * sh.mean():5.1f}%  gap '
                  f'{100 * d.mean():+5.1f} +- {100 * d.std() / np.sqrt(len(d)):.1f}', flush=True)

    freq = C.T @ (uni[heard] / uni[heard].sum()).astype(np.float32)
    top = heard[np.argsort(-uni[heard])[:100]]
    D = C[[mp[int(s)] for s in top]] - freq
    D /= np.maximum(1e-6, np.linalg.norm(D, axis=1, keepdims=True))
    Dt, fq = torch.as_tensor(D, device=DEV), torch.as_tensor(freq, device=DEV)
    in_top = np.isin(want, top)
    print('share of test syllables among the 100 most heard', float(in_top.mean()), flush=True)

    def pick(P):
        """which of the 100 syllables does the predicted pattern look most like"""
        P = torch.as_tensor(P, device=DEV)
        return (top[((P - fq) @ Dt.T).argmax(1).cpu().numpy()] == want).astype(float)

    print('chance and ceiling:')
    report('frequency', score(np.tile(freq, (len(te), 1))))
    ideal = np.zeros((len(te), n), np.float32)
    for i, c in enumerate(cur):
        d = bi.get(int(c))
        if d is None:
            ideal[i] = freq
            continue
        tot = sum(d.values())
        for s, k_ in d.items():
            ideal[i] += C[mp[s]] * (k_ / tot)
    report('ideal reader', score(ideal))
    bt = {c: {s: k_ for s, k_ in d.items() if s in set(top.tolist())} for c, d in bi.items()}
    guess = np.array([max(bt[int(c)], key=bt[int(c)].get) if bt.get(int(c)) else int(top[0]) for c in cur])
    report('choice: most frequent', (want == top[0]).astype(float), in_top)
    report('choice: counted pairs', (guess == want).astype(float), in_top)
    report('choice: ideal pattern', pick(ideal), in_top)
    del ideal

    tens = lambda M: torch.from_numpy(M).to(DEV)

    locked = onset[onset < T0 - hit - 1] + hit      # the ticks at which a new syllable's copies come on, heard stream

    def fit(F, name, delays):
        """regression trained only on those ticks: does this activity carry the next syllable at all?"""
        cnt = np.bincount(F.ix[:F.off[T0]], minlength=n)
        use = np.nonzero(cnt > 0)[0]
        m = len(use)
        print(name, 'active units', m, 'training ticks', len(locked), flush=True)
        for dl in delays:
            rows = locked[locked >= dl]
            G, B = torch.zeros(m, m, device=DEV), torch.zeros(m, n, device=DEV)
            for c in chunks(rows):
                D = tens(F.dense(c - dl, use))
                G += D.T @ D
                B += D.T @ tens(R.dense(c))
            Ft = tens(F.dense(np.maximum(te + hit - dl, 0), use))
            I = torch.eye(m, device=DEV)
            for lam in LAM:
                L = torch.linalg.cholesky((G + lam * I).double())
                P = (Ft.double() @ torch.cholesky_solve(B.double(), L)).float()
                print(f' delay {dl} lambda {lam:g}', flush=True)
                report(name, score(P))
                report('choice: ' + name, pick(P), in_top)

    # pipeline check: features that simply name the syllable heard before this tick
    last, ix, off = -1, [], [0]
    for i in range(N):
        if last >= 0:
            ix.append(last)
        off.append(len(ix))
        if k[i] == 0:
            last = int(sid[i])
    fit(Rows(np.array(ix), np.array(off), n), 'syllable named', (1,))
    fit(A, 'last layer', DELAY)


if __name__ == '__main__':
    main()
