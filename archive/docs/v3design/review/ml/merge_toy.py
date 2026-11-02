"""Two checks of ARCH_v3_core §2.4 (lineage merge), local numpy, < 200 MB.

Part 1. Are 'order-independent unit claims' order-independent?  SYL area of N units with random birth templates;
        a novel syllable claims the k free units with the best birth match (§2.4a-3).  Two siblings, same birth
        wiring, hear the same syllables in different first-encounter orders (+ optional junk recruitments from noisy
        windows).  Agreement = mean Jaccard of a syllable's assembly across siblings.
Part 2. Merge operators on error-driven (NLMS) readouts over SHARED random conjunction features (GEN-like).
        Parents learn overlapping context sets; the child is probed zero-shot.  Operators: plain mean of deltas,
        TIES (trim, elect sign by summed delta, disjoint mean: the §2.4a-5 operator), use-weighted mean
        (Fisher-like, weights = each parent's per-input use count), keep-larger.  Also feature drift in one parent.
Run: python3 -I merge_toy.py
"""
import numpy as np, json

# ---------------------------------------------------------------- Part 1: unit claims
def claims(N, k, protos, birth, order, junk_at, rng):
    free = np.ones(N, dtype=bool); asm = {}
    for step, s in enumerate(order):
        for _ in range(junk_at.get(step, 0)):                # a junk window recruits k units too
            v = rng.standard_normal(protos.shape[1]); v /= np.linalg.norm(v)
            m = birth @ v; m[~free] = -9; u = np.argsort(-m)[:k]; free[u] = False
        m = birth @ protos[s]; m[~free] = -9
        u = np.argsort(-m)[:k]; free[u] = False; asm[s] = set(u.tolist())
    return asm


def part1(seed=0):
    rng = np.random.default_rng(seed); D = 48; out = {}
    for grouped in (False, True):
        for N, S in ((2048, 32), (2048, 64), (2048, 120), (8192, 141)):
            if grouped:                                    # sound-alike groups of 4 (within-group cosine ~0.67)
                cent = rng.standard_normal((S // 4 + 1, D))
                protos = cent[np.arange(S) // 4] + 0.7 * rng.standard_normal((S, D))
            else:
                protos = rng.standard_normal((S, D))
            protos /= np.linalg.norm(protos, axis=1, keepdims=True)
            birth = rng.standard_normal((N, D)); birth /= np.linalg.norm(birth, axis=1, keepdims=True)
            for junk in (0, 2):
                js, jh = [], []
                for rep in range(5):
                    oa, ob = rng.permutation(S), rng.permutation(S)
                    ja = {int(t): junk for t in rng.choice(S, S // 4, replace=False)} if junk else {}
                    jb = {int(t): junk for t in rng.choice(S, S // 4, replace=False)} if junk else {}
                    # siblings' learned templates differ slightly (voices, noise): proto + 0.1 noise each
                    pa = protos + 0.1 * rng.standard_normal(protos.shape) / np.sqrt(D) * np.sqrt(D) / 7
                    pb = protos + 0.1 * rng.standard_normal(protos.shape) / np.sqrt(D) * np.sqrt(D) / 7
                    pa /= np.linalg.norm(pa, axis=1, keepdims=True); pb /= np.linalg.norm(pb, axis=1, keepdims=True)
                    A = claims(N, 16, pa, birth, oa, ja, np.random.default_rng(rep))
                    B = claims(N, 16, pb, birth, ob, jb, np.random.default_rng(rep + 99))
                    js.append(np.mean([len(A[s] & B[s]) / len(A[s] | B[s]) for s in range(S)]))
                    # order-free alternative: assembly = top-16 birth match, no exclusivity (a fly-hash code)
                    ha = np.argsort(-(pa @ birth.T), 1)[:, :16]; hb = np.argsort(-(pb @ birth.T), 1)[:, :16]
                    jh.append(np.mean([len(set(ha[s]) & set(hb[s])) / len(set(ha[s]) | set(hb[s])) for s in range(S)]))
                key = f'{"grouped" if grouped else "independent"}_N{N}_S{S}_junk{junk}'
                out[key] = dict(claims=round(float(np.mean(js)), 3), flyhash=round(float(np.mean(jh)), 3))
                fill = S * 16 * (1 + junk * 0.25) / N
                print(f'  {"sound-alike groups" if grouped else "independent syllables":22s} N={N} S={S} junk={junk}: '
                      f'fill {fill:.2f}; Jaccard across siblings: free-unit claims {np.mean(js):.3f} | '
                      f'order-free top-k code {np.mean(jh):.3f}', flush=True)
    return out


# ---------------------------------------------------------------- Part 2: merging NLMS readouts
NF, NX, V, NU = 6, 20, 30, 3000


def make_units(rng, n=NU):
    Fs = [set(rng.choice(NF, rng.integers(1, 3), replace=False).tolist()) for _ in range(n)]
    Xs = [set(rng.choice(NX, rng.integers(2, 4), replace=False).tolist()) for _ in range(n)]
    return Fs, Xs


def phi(units, f, x):
    Fs, Xs = units
    return np.array([i for i in range(len(Fs)) if f in Fs[i] and x in Xs[i]], dtype=np.int64)


def train(units, ctxs, table, rng, epochs=10, eta=0.35):
    W = np.zeros((NU, V)); use = np.zeros(NU)
    cache = {c: phi(units, *c) for c in ctxs}
    for ep in range(epochs):
        for c in rng.permutation(len(ctxs)):
            c = ctxs[c]; A = cache[c]
            if len(A) == 0:
                continue
            p = np.clip(W[A].sum(0), 0, 1); t = np.zeros(V); t[table[c]] = 1
            W[A] += eta * (t - p) / len(A); use[A] += 1
    return W, use


def acc(W, units, ctxs, table, theta=0.3):
    ok = 0
    for c in ctxs:
        A = phi(units, *c)
        p = np.clip(W[A].sum(0), 0, 1) if len(A) else np.zeros(V)
        ok += (p.max() >= theta and int(np.argmax(p)) == table[c])
    return ok / max(1, len(ctxs))


def merge(Ds, uses, how, eps=0.01):
    D = np.stack(Ds)                                  # parents x NU x V (deltas from birth = 0)
    if how == 'mean':
        return D.mean(0)
    if how == 'keep-larger':
        i = np.abs(D).argmax(0); return np.take_along_axis(D, i[None], 0)[0]
    if how == 'ties':
        Dt = np.where(np.abs(D) >= eps * np.abs(D).max(), D, 0.0)
        sgn = np.sign(Dt.sum(0)); agree = (np.sign(Dt) == sgn[None]) & (Dt != 0)
        n = agree.sum(0); return np.where(n > 0, (Dt * agree).sum(0) / np.maximum(n, 1), 0.0)
    if how == 'use-weighted':
        U = np.stack(uses)
        return (D * U).sum(0) / np.maximum(U.sum(0), 1e-9)
    raise ValueError(how)


def part2(seed=0):
    rng = np.random.default_rng(seed); out = {}
    allc = [(f, x) for f in range(NF) for x in range(NX)]
    res = {}
    for scen in ('disjoint+shared', 'with-conflict', 'pool4', 'drift0.1', 'drift0.3'):
        rows = {m: [] for m in ('mean', 'ties', 'use-weighted', 'keep-larger')}; parents = []
        for rep in range(4):
            r = np.random.default_rng(1000 * rep + seed)
            units = make_units(r)
            table = {c: int(r.integers(V)) for c in allc}
            perm = [allc[i] for i in r.permutation(len(allc))]
            if scen == 'pool4':
                shared = perm[:24]; own = [perm[24 + 24 * j: 48 + 24 * j] for j in range(4)]
                sets = [shared + own[j] for j in range(4)]; tabs = [table] * 4; unitsP = [units] * 4
                probes = {'shared': shared, 'one-parent': sum(own, [])}
            else:
                shared, a_only, b_only = perm[:40], perm[40:80], perm[80:120]
                sets = [shared + a_only, shared + b_only]
                tabB = dict(table)
                conf = []
                if scen == 'with-conflict':
                    conf = shared[:10]
                    for c in conf:
                        tabB[c] = (table[c] + 1 + int(r.integers(V - 1))) % V
                tabs = [table, tabB]
                unitsP = [units, units]
                if scen.startswith('drift'):              # parent B's prune/regrow (or instar) moved a fraction rho
                    rho = float(scen[5:]); Fs, Xs = list(units[0]), list(units[1])
                    nf, nx = make_units(r)
                    for i in r.choice(NU, int(rho * NU), replace=False):
                        Fs[i], Xs[i] = nf[i], nx[i]
                    unitsP = [units, (Fs, Xs)]
                probes = {'shared(same answer)': [c for c in shared if c not in conf], 'A-only': a_only,
                          'B-only': b_only}
                if conf:
                    probes['conflict:A answer'] = conf
            Ws, Us = [], []
            for j, S in enumerate(sets):
                W, U = train(unitsP[j], S, tabs[j], np.random.default_rng(r.integers(1 << 30)))
                Ws.append(W); Us.append(np.repeat(U[:, None], V, 1))
            parents.append({j: acc(Ws[j], unitsP[j], sets[j], tabs[j]) for j in range(len(sets))})
            for m in rows:
                Wc = merge(Ws, Us, m)
                # the child is born with parent A's wiring (keep-larger-move picks one parent per moved unit)
                rows[m].append({k: acc(Wc, units, v, table) for k, v in probes.items()})
        res[scen] = {m: {k: round(float(np.mean([d[k] for d in v])), 3) for k in v[0]} for m, v in rows.items()}
        res[scen]['parents_own'] = round(float(np.mean([np.mean(list(p.values())) for p in parents])), 3)
        print(f'  scenario {scen}: parents on their own sets {res[scen]["parents_own"]}', flush=True)
        for m in rows:
            print(f'     {m:13s}', res[scen][m], flush=True)
    return res


if __name__ == '__main__':
    print('== Part 1: sibling assembly agreement under first-encounter-order claims')
    p1 = part1()
    print('== Part 2: merging NLMS readouts (zero-shot child accuracy)')
    p2 = part2()
    json.dump({'part1': p1, 'part2': p2}, open('merge_toy.json', 'w'), indent=1)
