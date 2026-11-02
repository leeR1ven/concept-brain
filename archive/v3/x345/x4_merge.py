"""X4: merging siblings that learned different lessons with the designer's Hebbian rule (local numpy, 1 process).

Lineage: one birth wiring (lineage seed) and one fixed physics seed (the char -> AUD assembly codes), shared by all
siblings. Each sibling hears its own lessons as a tick stream: "q1 q2 a" (3 ticks per char) then silence, shuffled
with filler lines. Areas (binary units):
  AUD : 1,024 units, a char = a fixed 16-unit assembly (physics seed)
  PFC : 2,048 units, a smoothed quantile threshold (about 2 % on, count varies, ceiling 2x; ARCHITECTURE.md 2.1);
        input AUD -> PFC by random birth fan-in (64), plus random birth recurrence PFC -> PFC (fan-out 128), so a
        PFC code carries the previous code (order) without any time cell
  PFC -> AUD prediction links ("the PFC predicts the next moment", ARCHITECTURE.md 2.4): born random and small,
        trained.
Rule (ARCHITECTURE.md 2.8): only units that just changed; falling -> rising +eta*g(w); rising <-> rising +eta_co*g(w);
  reverse order (pre rising while post falls) -eta*g(w); g(w) = 1/(1 + w/w_ref)^pow; w = birth + learned, the total
  never below 0; one global decay of the learned part (cortex gamma) every T ticks; birth part never decays.
Scenarios:
  fixed_in : AUD -> PFC stays at birth (near-input layer), only PFC -> AUD learns
  train_in : AUD -> PFC is trained too (ARCHITECTURE.md 3: inter-area links are born random and trained), so the
             siblings' PFC codes for the same input drift apart
  pool4    : 4 siblings, each with its own lessons, merged together
Merge of the learned parts (birth wiring shared, so the child = birth + merged learned part):
  mean      : plain mean
  sign      : sign-elected mean (TIES: trim |d| < 1 % of max, elect the sign by the summed d, mean over agreeing)
  align+mean, align+sign : first match sibling k's PFC units to sibling 0's by activation on a shared probe stream
             (greedy max-cosine assignment; the operator lives outside the agent), permute, then merge
Readout (motor competition, ARCHITECTURE.md 2.7): the answer is released only if the best char's predicted drive
  beats the inhibitory default (theta_inh, calibrated on the parents) and is sharp (best >= 1.5 x second);
  otherwise no answer.
Run: python3 -I x4_merge.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_A, K_A, V = 1024, 16, 60
N_P, F_P, CA, CB = 2048, 0.02, 64, 128
N_ADD = 31                     # spare stable IDs N_A .. N_A + 30 for children of AUD splits (3 %)
N_AX = N_A + N_ADD
SYL, GAP = 3, 15


class Thr:
    def __init__(self, f, alpha, ceil=2.0):
        self.f, self.a, self.theta, self.ceil = f, alpha, None, ceil

    def fire(self, d, rng):
        n = len(d); k = max(1, int(round(self.f * n)))
        q = float(np.partition(d, n - k)[n - k])
        self.theta = q if self.theta is None else self.theta + self.a * (q - self.theta)
        dd = d + 1e-7 * rng.random(n)
        on = np.flatnonzero((dd >= max(self.theta, 1e-9)) & (d > 0))
        kmax = int(self.ceil * k)
        if len(on) > kmax:
            on = np.sort(on[np.argpartition(-dd[on], kmax)[:kmax]])
        return on


class Lineage:
    """everything drawn from the lineage seed (birth wiring) and the physics seed (codes)"""

    def __init__(self, seed_lin=1, seed_phys=7):
        rp = np.random.default_rng(seed_phys)
        self.asm = np.stack([np.sort(rp.choice(N_A, K_A, replace=False)) for _ in range(V)])
        rl = np.random.default_rng(seed_lin)
        self.B_ap = np.zeros((N_A, N_P), np.float32)
        for j in range(N_P):
            self.B_ap[rl.choice(N_A, CA, replace=False), j] = rl.uniform(0.05, 0.20, CA)
        self.B_pp = np.zeros((N_P, N_P), np.float32)
        for i in range(N_P):
            tg = rl.choice(N_P - 1, CB, replace=False); tg[tg >= i] += 1
            self.B_pp[i, tg] = rl.uniform(0.05, 0.20, CB)
        self.B_pa = rl.uniform(0.0, 0.02, (N_P, N_A)).astype(np.float32)
        self.mask_ap = self.B_ap > 0


class Agent:
    def __init__(self, lin, cfg, seed):
        self.L, self.c = lin, cfg
        self.rng = np.random.default_rng(seed)
        # own copy of the AUD-side birth wiring: base IDs from the lineage, spare IDs filled by this sibling's own
        # splits (stable IDs as in the chain repo: a deleted unit keeps its ID and is dead; children get new IDs)
        self.B_ap = np.zeros((N_AX, N_P), np.float32); self.B_ap[:N_A] = lin.B_ap
        self.B_pa = np.zeros((N_P, N_AX), np.float32); self.B_pa[:, :N_A] = lin.B_pa
        self.mask_ap = self.B_ap > 0
        self.dead = np.zeros(N_AX, bool); self.dead[N_A:] = True
        self.parent = -np.ones(N_ADD, int)
        if cfg.get('stable'):
            r = np.random.default_rng(seed + 7)
            feat = np.unique(lin.asm)
            self.dead[r.choice(feat, N_ADD, replace=False)] = True
            alive = feat[~self.dead[feat]]
            self.parent = r.choice(alive, N_ADD, replace=False)
            for j in range(N_ADD):                       # a child's new chain: new random links (sibling-specific)
                tg = r.choice(N_P, int(CA * N_P / N_A), replace=False)
                self.B_ap[N_A + j, tg] = r.uniform(0.05, 0.20, len(tg))
                self.B_pa[:, N_A + j] = r.uniform(0.0, 0.02, N_P)
            self.dead[N_A:] = False
            self.mask_ap = self.B_ap > 0
            self.B_ap[self.dead] = 0.0; self.B_pa[:, self.dead] = 0.0
        self.D_ap = np.zeros((N_AX, N_P), np.float32)
        self.D_pa = np.zeros((N_P, N_AX), np.float32)
        self.thr = Thr(F_P, cfg['alpha'])
        self.prev_p = np.zeros(0, int); self.prev_a = np.zeros(0, int)
        self.t = 0; self.t_dec = 0
        self._norm = None

    def pattern(self, c):
        """the AUD units a char turns on in this sibling: the surviving assembly plus children of its members"""
        a = self.L.asm[c]
        a = a[~self.dead[a]]
        kids = N_A + np.flatnonzero(np.isin(self.parent, a))
        return np.concatenate([a, kids])

    # ---- dynamics
    def pfc(self, aud, prev, thr, D_ap=None):
        if self.c['train_in']:
            W = np.maximum(self.B_ap + (self.D_ap if D_ap is None else D_ap), 0.0)
        else:
            W = self.B_ap
        if len(aud) == 0:
            d = np.zeros(N_P, np.float32)
        elif self.c.get('tl_in'):                       # two-layer weights on the input path too
            so, si = W.sum(1) + 1e-6, W.sum(0) + 1e-6
            d = (W[aud] / np.sqrt(so[aud])[:, None]).sum(0) / np.sqrt(si) * np.sqrt(si.mean() * so.mean())
        else:
            d = W[aud].sum(0)
        if len(prev):
            r = self.L.B_pp[prev].sum(0)
            d = d + self.c['rho'] * (r - r.mean())          # gain control: inhibition grows with the total activity
        return thr.fire(d, self.rng)

    def g(self, w):
        return 1.0 / (1.0 + np.maximum(w, 0.0) / self.c['w_ref']) ** self.c['pow']

    def _upd(self, D, B, rows, cols, eta, sign=1.0, mask=None):
        if len(rows) == 0 or len(cols) == 0 or eta == 0:
            return
        ix = np.ix_(rows, cols)
        w = B[ix] + D[ix]
        new = D[ix] + sign * eta * self.g(w)
        new = np.maximum(new, -B[ix])
        if mask is not None:
            new = np.where(mask[ix], new, D[ix])
        D[ix] = new

    def step(self, t, aud):
        nd = t // self.c['T'] - self.t_dec
        if nd > 0:
            f = self.c['gamma'] ** nd
            for D in (self.D_pa, self.D_ap):
                D *= f
                D[np.abs(D) < self.c['w_min']] = 0.0
            self.t_dec += nd
        x = self.pfc(aud, self.prev_p, self.thr)
        rp, fp = np.setdiff1d(x, self.prev_p), np.setdiff1d(self.prev_p, x)
        ra, fa = np.setdiff1d(aud, self.prev_a), np.setdiff1d(self.prev_a, aud)
        c = self.c
        # PFC -> AUD: the previous PFC state predicts the input that comes now
        self._upd(self.D_pa, self.B_pa, fp, ra, c['eta'])
        self._upd(self.D_pa, self.B_pa, rp, ra, c['eta_co'])
        self._upd(self.D_pa, self.B_pa, rp, fa, c['eta'], -1.0)
        if c['train_in']:
            self._upd(self.D_ap, self.B_ap, fa, rp, c['eta_in'], mask=self.mask_ap)
            self._upd(self.D_ap, self.B_ap, ra, rp, c['eta_in'], mask=self.mask_ap)
            self._upd(self.D_ap, self.B_ap, ra, fp, c['eta_in'], -1.0, mask=self.mask_ap)
        self.prev_p, self.prev_a = x, aud
        return x

    def hear(self, chars):
        """a line of chars then silence, tick-event driven (state changes at char onsets and at the line end)"""
        for ch in chars:
            self.step(self.t, self.pattern(ch)); self.t += SYL
        self.step(self.t, np.zeros(0, int)); self.t += GAP

    # ---- tests (no learning)
    def code(self, chars, D_ap=None):
        th = Thr(F_P, self.c['alpha']); th.theta = self.thr.theta
        x = np.zeros(0, int)
        xs = []
        for ch in chars:
            x = self.pfc(self.pattern(ch), x, th, D_ap); xs.append(x)
        return xs

    def scores(self, q, D_pa=None, D_ap=None):
        """predicted drive of each char's AUD units from the PFC state after the question; with two-layer weights
        (ARCHITECTURE.md 2.4) the propagation is the strength divided by sqrt(out-strength x in-strength)"""
        x = self.code(q, D_ap)[-1]
        W = np.maximum(self.B_pa + (self.D_pa if D_pa is None else D_pa), 0.0)
        if self.c['twolayer']:
            so, si = W.sum(1) + 1e-6, W.sum(0) + 1e-6
            d = (W[x] / np.sqrt(so[x])[:, None]).sum(0) / np.sqrt(si)
        else:
            d = W[x].sum(0)
        return np.array([d[self.pattern(c)].mean() for c in range(V)])


def answer(sc, theta_inh, sharp=1.5):
    o = np.argsort(-sc)
    b, s = sc[o[0]], sc[o[1]]
    return int(o[0]) if (b >= theta_inh and b >= sharp * s) else None


# ----------------------------------------------------------------------------------------------- merge operators
def ties(Ds, trim=0.01):
    D = np.stack(Ds)
    mx = np.abs(D).max()
    D = np.where(np.abs(D) >= trim * mx, D, 0.0)
    sg = np.sign(D.sum(0))
    agree = (np.sign(D) == sg[None]) & (D != 0)
    n = agree.sum(0)
    return np.where(n > 0, (D * agree).sum(0) / np.maximum(n, 1), 0.0).astype(np.float32)


def align(ref, other, probes):
    """greedy max-cosine matching of other's PFC units to ref's by activity on a shared probe stream; returns perm
    with perm[i_ref] = i_other"""
    A = np.zeros((len(probes) * 2, N_P), np.float32); B = np.zeros_like(A)
    r = 0
    for q in probes:
        for xa, xb in zip(ref.code(q), other.code(q)):
            A[r, xa] = 1; B[r, xb] = 1; r += 1
    A, B = A[:r], B[:r]
    na, nb = np.linalg.norm(A, axis=0) + 1e-9, np.linalg.norm(B, axis=0) + 1e-9
    S = (A.T @ B) / na[:, None] / nb[None, :]
    perm = np.arange(N_P)
    act_a, act_b = np.flatnonzero(na > 1e-6), np.flatnonzero(nb > 1e-6)
    sub = S[np.ix_(act_a, act_b)]
    order = np.argsort(-sub, axis=None)
    used_a, used_b = set(), set()
    pairs = {}
    for flat in order:
        i, j = divmod(int(flat), sub.shape[1])
        if sub[i, j] <= 0:
            break
        a, b = int(act_a[i]), int(act_b[j])
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b); pairs[a] = b
        if len(used_a) == len(act_a) or len(used_b) == len(act_b):
            break
    free_b = [b for b in range(N_P) if b not in used_b]
    free_a = [a for a in range(N_P) if a not in pairs]
    # unmatched units: keep identity where possible, else fill
    fb = set(free_b)
    rest = []
    for a in free_a:
        if a in fb:
            pairs[a] = a; fb.discard(a)
        else:
            rest.append(a)
    for a, b in zip(rest, sorted(fb)):
        pairs[a] = b
    for a, b in pairs.items():
        perm[a] = b
    ident = float(np.mean([pairs[a] == a for a in act_a])) if len(act_a) else 1.0
    return perm, ident


def aud_map(ref, ag, aligned):
    """AUD IDs of ag -> IDs of ref. Base IDs map to themselves (stable IDs); a unit dead in either maps nowhere.
    Spare IDs: by ID (naive), or by activation matching over all chars (aligned): ag's child j goes to the ref child
    whose activity profile over the probe chars is the same (cosine >= 0.99), else nowhere."""
    m = -np.ones(N_AX, int)
    for u in range(N_A):
        if not ref.dead[u] and not ag.dead[u]:
            m[u] = u
    prof = lambda a: np.array([[u in set(a.pattern(c).tolist()) for c in range(V)] for u in range(N_A, N_AX)], float)
    if not aligned:
        for u in range(N_A, N_AX):
            if not ref.dead[u] and not ag.dead[u]:
                m[u] = u
    else:
        pr, pa = prof(ref), prof(ag)
        nr, na = np.linalg.norm(pr, axis=1) + 1e-9, np.linalg.norm(pa, axis=1) + 1e-9
        S = (pa @ pr.T) / na[:, None] / nr[None, :]
        used = set()
        for j in np.argsort(-S.max(1)):
            i = int(np.argmax(np.where([k in used for k in range(N_ADD)], -1, S[j])))
            if S[j, i] >= 0.99 and i not in used:
                m[N_A + j] = N_A + i; used.add(i)
    return m


def permute(ref, ag, perm, aligned_aud=False):
    """sibling's learned parts in the reference's PFC and AUD index space"""
    am = aud_map(ref, ag, aligned_aud) if ag is not ref else np.arange(N_AX)
    D_pa = np.zeros_like(ag.D_pa); D_ap = np.zeros_like(ag.D_ap)
    ok = am >= 0
    D_pa[:, am[ok]] = ag.D_pa[perm][:, ok]
    D_ap[am[ok]] = ag.D_ap[:, perm][ok]
    return D_pa, D_ap


# ----------------------------------------------------------------------------------------------- experiment
def make_lessons(rng, n_s=20, n_own=30, n_c=15, n_sib=2):
    qs = set()
    def newq():
        while True:
            q = tuple(int(v) for v in rng.choice(V, 2, replace=False))
            if q not in qs:
                qs.add(q); return q
    S = [(newq(), int(rng.integers(V))) for _ in range(n_s)]
    own = [[(newq(), int(rng.integers(V))) for _ in range(n_own)] for _ in range(n_sib)]
    C = []
    for _ in range(n_c):
        q = newq(); a = rng.choice(V, n_sib, replace=False)
        C.append((q, [int(v) for v in a]))
    return S, own, C


def raise_sib(lin, cfg, seed, lessons, n_rep, n_fill):
    ag = Agent(lin, cfg, seed)
    rng = np.random.default_rng(seed + 1000)
    items = [list(q) + [a] for q, a in lessons] * n_rep
    items += [list(rng.choice(V, 3, replace=False)) for _ in range(n_fill)]
    for i in rng.permutation(len(items)):
        ag.hear([int(v) for v in items[i]])
    return ag


def evaluate(child_scores, sets, theta, sharp=1.25):
    out = {}
    for name, items in sets.items():
        res = []
        for q, a in items:
            got = answer(child_scores(q), theta, sharp)
            if isinstance(a, list):                        # conflict: any released answer is a failure
                res.append(dict(none=got is None, a0=got == a[0], a1=got == a[1]))
            else:
                res.append(dict(right=got == a, none=got is None))
        out[name] = {k: round(float(np.mean([r[k] for r in res])), 3) for k in res[0]}
    return out


def run_scenario(name, cfg, n_sib=2, seed=0):
    n_rep, n_fill, sh = cfg['n_rep'], cfg['n_fill'], cfg['sharp']
    t0 = time.time()
    lin = Lineage(seed_lin=1 + seed)
    rng = np.random.default_rng(50 + seed)
    S, own, C = make_lessons(rng, n_sib=n_sib)
    sibs = []
    for k in range(n_sib):
        lessons = S + own[k] + [(q, a[k]) for q, a in C]
        sibs.append(raise_sib(lin, cfg, 100 * seed + 11 * (k + 1), lessons, n_rep, n_fill))
    # calibrate the inhibitory default on the parents' own trained items
    best = [ag.scores(q).max() for k, ag in enumerate(sibs) for q, a in S + own[k]]
    theta = 0.3 * float(np.median(best))
    parents = {}
    for k, ag in enumerate(sibs):
        sets = {'shared': S, 'own': own[k], 'conflict_own': [(q, a[k]) for q, a in C]}
        parents[k] = evaluate(lambda q, ag=ag: ag.scores(q), sets, theta, sh)
    sets = {'shared': S, 'conflict': C}
    for k in range(n_sib):
        sets[f'only{k}'] = own[k]
    # code drift between siblings (same input, same birth wiring)
    probes = [list(q) for q, a in S + sum(own, []) + C] + [[int(v)] for v in range(V)]
    probes = probes[:400]
    jac = []
    for q in probes[:80]:
        xa, xb = sibs[0].code(q)[-1], sibs[1].code(q)[-1]
        jac.append(len(np.intersect1d(xa, xb)) / max(1, len(np.union1d(xa, xb))))
    res = dict(scenario=name, parents=parents, theta=theta, code_jaccard_sib0_sib1=round(float(np.mean(jac)), 3))
    ref = sibs[0]
    perms, idents = [np.arange(N_P)], [1.0]
    for ag in sibs[1:]:
        p, ident = align(ref, ag, probes)
        perms.append(p); idents.append(ident)
    res['align_identity_frac'] = [round(v, 3) for v in idents[1:]]
    for op in ('mean', 'sign', 'align+mean', 'align+sign'):
        if op.startswith('align'):
            pa = [permute(ref, ag, p, True) for ag, p in zip(sibs, perms)]
        else:
            pa = [permute(ref, ag, np.arange(N_P), False) for ag in sibs]
        f = (lambda Ds: np.mean(np.stack(Ds), 0).astype(np.float32)) if op.endswith('mean') else (lambda Ds: ties(Ds, cfg['trim']))
        D_pa = f([a for a, b in pa]); D_ap = f([b for a, b in pa]) if cfg['train_in'] else None
        res[op] = evaluate(lambda q: ref.scores(q, D_pa=D_pa, D_ap=D_ap), sets, theta, sh)
    res['secs'] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
    return res


BASE = dict(alpha=0.3, rho=0.5, eta=0.3, eta_co=0.01, w_ref=1.0, pow=1.0, gamma=0.999, T=200, w_min=0.002,
            train_in=False, twolayer=True, sharp=1.25, n_rep=10, n_fill=300, eta_in=0.02, trim=0.01)

if __name__ == '__main__':
    out = []
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for seed in range(n_seed):
        for name, ov, nsib in (('fixed_in', dict(trim=0.1), 2), ('fixed_in+stable_ids', dict(trim=0.1, stable=True), 2),
                               ('fixed_in pool4', dict(trim=0.1), 4)):
            out.append(run_scenario(name, dict(BASE, **ov), n_sib=nsib, seed=seed))
    out.append(run_scenario('train_in (two-layer input)', dict(BASE, trim=0.1, train_in=True, tl_in=True), seed=0))
    json.dump(out, open(os.path.join(HERE, 'x4_merge.json'), 'w'), indent=1)
