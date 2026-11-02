"""Neuro review toy 1 (local, < 300 MB, ~1-2 min): capacity and spontaneous-replay fidelity of the pure-neuron CA3
that ARCH_v3_core 7.2 / S6 proposes: N = 10,000 binary units, fixed fan-in C = 1,000 (dilution 0.1), k = 100 active,
one-shot clipped Hebbian (Willshaw) storage, recall = k-WTA iterations from a partial DG cue.

Measures, as a function of the number of stored episodes P:
  recall: cue = 50 % of a stored pattern's units (+ 0 noise), 3 k-WTA iterations, final overlap with the target;
  replay: start from k random units (spontaneous kick), 5 iterations, best overlap with ANY stored pattern
          (overlap < 0.9 = a spurious / mixture state that M3 would replay into the cortex at the normal rate).
Variants:
  design   : N 1e4, C 1000, k 100  (about 10 connected active inputs per unit)
  denser-k : N 1e4, C 1000, k 300  (about 30 connected active inputs per unit; biology CA3 ~ 300, Rolls 2007)
  palimp   : design + stochastic LTD (pre inactive, post active, p 0.01) so old episodes fade (Amit & Fusi 1994)
Run: python3 -I ca3_capacity.py
"""
import numpy as np, time


def run(N, C, k, Ps, ltd=0.0, seed=0, ntest=40):
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(N, C, replace=False) for _ in range(N)]).astype(np.int16)  # presyn partners per unit
    W = np.zeros((N, C), dtype=bool)
    pats = []
    out = []
    Pmax = max(Ps)

    def kwta(h):
        th = np.partition(h, N - k)[N - k]
        on = h >= th
        if on.sum() > k:  # random tie break
            cand = np.flatnonzero(on); keep = rng.choice(cand, k, replace=False)
            on = np.zeros(N, bool); on[keep] = True
        return on

    def drive(x):
        return (W & x[idx]).sum(1).astype(np.float32) + 1e-3 * rng.random(N)

    for p in range(1, Pmax + 1):
        a = np.zeros(N, bool); a[rng.choice(N, k, replace=False)] = True
        pats.append(np.flatnonzero(a))
        post = np.flatnonzero(a)
        pre_on = a[idx[post]]
        if ltd > 0:
            dep = (~pre_on) & (rng.random(pre_on.shape) < ltd)
            W[post] &= ~dep
        W[post] |= pre_on
        if p in Ps:
            # recall at different ages: recent (last 10 %), middle, oldest 10 %
            res = {}
            for name, lo, hi in (('recent', int(0.9 * p), p), ('mid', int(0.45 * p), int(0.55 * p) + 1), ('old', 0, max(1, int(0.1 * p)))):
                ov = []
                for j in rng.integers(lo, hi, ntest):
                    tgt = pats[j]
                    x = np.zeros(N, bool); x[rng.choice(tgt, k // 2, replace=False)] = True
                    for _ in range(3):
                        x = kwta(drive(x))
                    ov.append(x[tgt].mean())
                res[name] = round(float(np.mean(ov)), 3)
            # spontaneous replay fidelity
            P_arr = np.stack(pats).astype(np.int32)
            good, spur = 0, 0
            for _ in range(ntest):
                x = np.zeros(N, bool); x[rng.choice(N, k, replace=False)] = True
                for _ in range(5):
                    x = kwta(drive(x))
                best = x[P_arr].sum(1).max() / k
                good += best >= 0.9
            res['replay_clean'] = round(good / ntest, 3)
            # mean connected active inputs per unit for a stored pattern
            res['conn_inputs'] = round(k * C / N, 1)
            res['frac_W_on'] = round(float(W.mean()), 3)
            out.append((p, res))
            print(f'  P={p:6d}', res, flush=True)
    return out


if __name__ == '__main__':
    t0 = time.time()
    Ps = [250, 1000, 2500, 5000, 10000]
    print('== design: N 1e4, C 1000, k 100 (Willshaw, no forgetting)')
    run(10000, 1000, 100, Ps)
    print(f'   [{time.time() - t0:.0f} s]')
    print('== denser-k: N 1e4, C 1000, k 300')
    run(10000, 1000, 300, [250, 1000, 2500, 5000])
    print(f'   [{time.time() - t0:.0f} s]')
    print('== palimpsest: design + stochastic LTD p 0.01')
    run(10000, 1000, 100, Ps, ltd=0.01)
    print(f'   [{time.time() - t0:.0f} s]')
