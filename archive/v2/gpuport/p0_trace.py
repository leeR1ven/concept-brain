# -*- coding: utf-8 -*-
"""P0: freeze the numpy ARCH_v2 reference (SynthAgent, tag final_noctx) and record a fixed-seed trace.

Runs the frozen copy in ./ref (byte-identical to arch/v2/synth): stage A (nA dialogues) -> babbling -> stage B (nB
dialogues), the exact J protocol start (run_j.py, final_noctx keys).  Records per tick from birth: mouth force, ear
energy, body GO / REL / Y / stop, the open thought (kind, step k, confidence e, intention I, released), detections,
line ends, awake recalls; state checksums at the end of A (after babbling) and at the end.  Also measures caps for the
batched port (line length, keys per event, VR / F sizes, pending turn predictions) and tie statistics of the DG k-WTA
and of the syllable matcher (where exact float identity can break).
Also records the observation stream (ear window, eye, reward per tick) so that the port can be replayed open-loop on
the GPU server (whose numpy 2.1 world may differ), the fixed random init (FEAT0, ear wiring, DG wiring), the babbling
memory, the awake-recall picks, and a digest of every learning event and boundary score.
canon=1: the DG k-WTA breaks exact ties by unit index (stable sort) instead of np.argpartition's introselect order
(the only change; it makes the reference platform-independent).
usage: python3 p0_trace.py seed=0 nA=300 nB=40 [canon=0] out=out/p0_trace_s0"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', 'synth')
sys.path.insert(0, REF)
from agent_synth import SynthAgent          # noqa: E402
from agent import BENCH                     # noqa: E402
sys.path.insert(0, BENCH)
from world import World                     # noqa: E402

TUNE_BEST = dict(recall_n=8, eta_recall=0.2, rec_pow=0.0, hip_cap=10000, hip_ndg=6000, hip_kdg=120, floor=0.3)
THINK_FINAL = dict(policy='free', ior=0, think_k=3, urg_gate=False, brake_now=True, late_release=False, u_reply=0.15,
                   u_R1=1.0, own_tail=True)
FINAL = dict(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)
STOPS = {None: 0, 'END': 1, 'breath': 2, 'floor': 3, 'silent': 4, 'babble': 5}


def final_kw():
    kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(FINAL)
    return kw


def canon_dg(hip):
    """DG k-WTA with exact ties broken by unit index (stable sort): the canonical, platform-independent rule."""
    def dg(inputs):
        d = hip.tie.copy()
        for g, x in inputs.items():
            if g in hip.idx and x is not None and x.any():
                d += hip.wg[g] * x[hip.idx[g]].sum(1)
        code = np.zeros(hip.n_dg, bool)
        code[np.argsort(-d, kind='stable')[:hip.k_dg]] = True
        return code
    return dg


class Rec(SynthAgent):
    """the reference agent + recording (no behavioural change)."""

    def setup_rec(self, canon=False):
        self.R = dict(force=[], E=[], go=[], rel=[], y=[], stop=[], th_kind=[], th_k=[], th_e=[], th_I=[], th_rel=[],
                      inline=[], mix=[], ear=[], eye=[], rew=[], quiet=[])
        self.dets, self.lines, self.recalls, self.picks, self.learns, self.scores = [], [], [], [], [], []
        self.caps = dict(line_len=0, P=0, keys=0, VR=0, F=0, feats=0, pend=0, chainN=0, chainC=0)
        self.ties = dict(dg_calls=0, dg_ties=0, z_calls=0, z_close=0, z_thr=0, ear_calls=0, ear_ties=0)
        core = self.core
        hip = core.hip
        caps, ties, me = self.caps, self.ties, self
        if canon:
            hip.dg = canon_dg(hip)
        dg0 = hip.dg

        def dg(inputs):
            code = dg0(inputs)
            d = hip.tie.copy()
            for g, x in inputs.items():
                if g in hip.idx and x is not None and x.any():
                    d += hip.wg[g] * x[hip.idx[g]].sum(1)
            s = np.sort(d)[::-1]
            ties['dg_calls'] += 1
            if s[hip.k_dg - 1] == s[hip.k_dg]:
                ties['dg_ties'] += 1
            return code
        hip.dg = dg
        keys0 = core._keys

        def keys(st, cands):
            k = keys0(st, cands)
            caps['keys'] = max(caps['keys'], len(k))
            caps['P'] = max(caps['P'], len(st.get('Pbag') or st['P']))
            caps['VR'] = max(caps['VR'], len(st.get('VR') or []))
            caps['F'] = max(caps['F'], sum(1 for x in k if x[0] == 'F'))
            caps['line_len'] = max(caps['line_len'], len(st['items']))
            for nm in ('N', 'C'):
                ch = st['src'].get(nm)
                if ch is not None:
                    caps['chain' + nm] = max(caps['chain' + nm], len(ch.items))
            return k
        core._keys = keys
        gl0 = core.gen.learn

        def glearn(A, target, total, J=None, g=1.0, eta=None, grp=None):
            me.learns.append((me.t, len(A), int(np.sum(A)), int(np.argmax(target)), float(np.sum(total, dtype=np.float64)),
                              len(J) if J is not None else -1))
            return gl0(A, target, total, J=J, g=g, eta=eta, grp=grp)
        core.gen.learn = glearn
        os0 = core.own_scores

        def own_scores(st):
            prog, s_end = os0(st)
            me.scores.append((me.t, float(np.sum(prog, dtype=np.float64)), int(np.argmax(prog)), float(s_end)))
            return prog, s_end
        core.own_scores = own_scores
        ch0 = core.rng.choice

        class RngProxy:
            def __getattr__(self, k):
                return getattr(core.rng_real, k)

            def choice(self, *a, **kw):
                r = ch0(*a, **kw)
                me.picks.append((me.t, [int(x) for x in r]))
                return r
        core.rng_real = core.rng
        core.rng = RngProxy()
        ear = self.ear
        resp0 = ear.respond

        def respond(lines):
            s = (lines.astype(np.float32)[ear.idx] * ear.w).sum(1) + ear.b
            s[~ear.alive] = -np.inf
            ss = np.sort(s)[::-1]
            ties['ear_calls'] += 1
            if ss[ear.k - 1] == ss[ear.k]:
                ties['ear_ties'] += 1
            return resp0(lines)
        ear.respond = respond

    def _detect(self, f):
        m = self.Z.match(f) if self.Z.n else np.zeros(0)
        if len(m) >= 2:
            s = np.sort(m)[::-1]
            self.ties['z_calls'] += 1
            if s[0] - s[1] < 1e-5:
                self.ties['z_close'] += 1
            if abs(s[0] - self.theta_z) < 1e-5:
                self.ties['z_thr'] += 1
        j = super()._detect(f)
        self.dets.append((self.t, -1 if j is None else int(j)))
        return j

    def step(self, obs):
        n_lines = self.stats['lines']
        n_rec = self.core.stats['awake_recalls']
        t = self.t
        R = self.R
        R['ear'].append(np.asarray(obs['ear'], np.uint8))
        R['eye'].append(np.packbits(np.asarray(obs['eye']).astype(bool)))
        R['rew'].append(float(obs.get('reward', 0.0)))
        force = super().step(obs)
        b, th = self.body, self.th
        R['force'].append(np.asarray(force, np.uint8))
        R['E'].append(float(np.asarray(obs['ear']).sum()))
        R['go'].append(int(b.GO)); R['rel'].append(int(b.REL)); R['y'].append(int(b.Y))
        R['stop'].append(STOPS.get(b.stop, 9)); R['inline'].append(int(self.in_line))
        R['mix'].append(int(self.core.mix.n)); R['quiet'].append(int(self.quiet))
        if th is not None:
            R['th_kind'].append(1 if th['kind'] == 'reply' else 2); R['th_k'].append(th['k'])
            R['th_e'].append(float(th['e'][-1]) if th['e'] else -1.0); R['th_I'].append(float(th['I']))
            R['th_rel'].append(int(th['released']))
        else:
            R['th_kind'].append(0); R['th_k'].append(-1); R['th_e'].append(-1.0); R['th_I'].append(-1.0)
            R['th_rel'].append(0)
        if self.stats['lines'] > n_lines:
            lr = self.core.last_rec
            self.lines.append((t, int(self.core.mix.n), int(self.core.hip.n), list(map(int, lr['items'])) if lr else []))
            self.caps['line_len'] = max(self.caps['line_len'], len(lr['items']) if lr else 0)
        if self.core.stats['awake_recalls'] > n_rec:
            self.recalls.append((t, self.core.stats['awake_recalls'] - n_rec))
        if th is not None:
            self.caps['feats'] = max(self.caps['feats'], len(th['feats']))
        self.caps['pend'] = max(self.caps['pend'], len(self.pend))
        return force


def checksums(ag):
    c = ag.core
    n = c.mix.n
    d = dict(t=ag.t, mix=int(n), hip=int(c.hip.n), Z=int(ag.Z.n), cat2prog=len(ag.cat2prog), progs=int(ag.body.n_prog),
             gen_sum=float(c.gen.W[:n].astype(np.float64).sum()), gen_abs=float(np.abs(c.gen.W[:n]).astype(np.float64).sum()),
             gate_sum=float(c.gate.W[:n].astype(np.float64).sum()), critic=float(c.critic.v[:n].astype(np.float64).sum()),
             prime=float(c.prime.W.astype(np.float64).sum()), attD=float(c.attD.W.astype(np.float64).sum()),
             Zsum=float(ag.Z.W[:, :ag.Z.n].astype(np.float64).sum()), sig=float(c.hip.sig[:c.hip.n].astype(np.float64).sum()),
             P_oth=float(sum(ag.P_oth.w.values())), P_oth_n=len(ag.P_oth.w), V_go=float(sum(ag.V_go.w.values())),
             V_go_n=len(ag.V_go.w), rbar=float(ag.rbar), T_a=float(c.T_a.sum()),
             T_W=float(c.T_W.sum()) if c.T_W is not None else None, bab=int(len(ag.bab_set)),
             stats=dict(ag.stats), core_stats={k: int(v) for k, v in c.stats.items()}, tstats=dict(ag.tstats))
    return d


def main():
    A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    seed = int(A.get('seed', 0)); nA = int(A.get('nA', 300)); nB = int(A.get('nB', 40)); canon = int(A.get('canon', 0))
    out = A.get('out', os.path.join(HERE, 'out', f'p0_trace_s{seed}' + ('_canon' if canon else '')))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    wA = World('A', seed=seed, tutor='recast', eye_layout='retino')
    ag = Rec(wA, seed=seed, **final_kw())
    init = dict(F0=ag.F0.astype(np.int64), ear_idx=ag.ear.idx, ear_w=ag.ear.w, ear_b=ag.ear.b, ear_alive=ag.ear.alive,
                hip_tie=ag.core.hip.tie, **{f'hip_idx_{g}': v for g, v in ag.core.hip.idx.items()})
    ag.setup_rec(canon=bool(canon))
    wA.run(ag, nA)
    tA = ag.t
    ag.babble(wA.voice, 35960)
    csA = checksums(ag)
    print(f'A: {nA} dialogues {tA} ticks, categories {ag.Z.n} programs {len(ag.cat2prog)} ({time.time() - t0:.0f}s)',
          flush=True)
    bab = dict(bab_pat=np.packbits(ag.bab_pat, axis=1), bab_set=np.asarray(ag.bab_set, np.int64))
    wB = World('B', seed=seed, tutor='recast', prev=wA, eye_layout='retino')
    wB.run(ag, nB)
    csE = checksums(ag)
    R = {k: np.asarray(v) for k, v in ag.R.items()}
    pk = [(t, p) for t, p in ag.picks]
    np.savez_compressed(out + '.npz', tA=tA, dets=np.asarray(ag.dets, np.int64), recalls=np.asarray(ag.recalls, np.int64),
                        learns=np.asarray(ag.learns, np.float64), scores=np.asarray(ag.scores, np.float64),
                        pick_t=np.asarray([t for t, _ in pk], np.int64),
                        pick_e=np.asarray([p + [-1] * (8 - len(p)) for _, p in pk], np.int64).reshape(-1, 8),
                        **R, **{'init_' + k: v for k, v in init.items()}, **bab)
    meta = dict(seed=seed, nA=nA, nB=nB, canon=canon, tA=tA, t_end=ag.t, ticks_B=ag.t - tA, secs=round(time.time() - t0, 1),
                cs_A=csA, cs_end=csE, caps=ag.caps, ties=ag.ties, kw={k: str(v) for k, v in final_kw().items()},
                lines=ag.lines, n_spoken_ticks=int((R['force'].sum(1) > 0).sum()),
                n_spoken_ticks_B=int((R['force'][tA:].sum(1) > 0).sum()),
                releases_B=int((R['go'][tA:]).sum()), n_learn=len(ag.learns), n_scores=len(ag.scores),
                numpy=np.__version__)
    with open(out + '.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('lines', 'cs_A', 'cs_end')}, default=str))
    print('cs_end', json.dumps(csE, default=str))


if __name__ == '__main__':
    main()
