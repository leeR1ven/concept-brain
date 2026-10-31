# -*- coding: utf-8 -*-
"""ARCH_v2 (SynthAgent, final_noctx) as one batched brain for B agents: torch on the GPU for the data plane,
vectorised numpy for the control plane.

Design (ARCH_v2 §10-11, P1-P3):
  * every per-agent quantity is an array with a leading agent axis B.  Control state (body turn units, the thought,
    the intention, the turn learners, line contexts, the continuous thought, hippocampal record contents) is small
    and lives in numpy on the host; everything that scales (ear layer 0, syllable templates, babbling memory, the
    conjunction table MIX, the readouts GEN / gates / critic, the hippocampal keys and strengths, the DG, primes,
    word->slot attention) lives in torch tensors on the device.
  * events are masks: each tick computes, for all agents at once, which agents have a line onset, a syllable sample,
    a line end, a thought step, a body boundary, a reward, a quiet moment; each handler runs once on the subset that
    needs it (batched over agents, never a per-agent loop over the heavy data).
  * MIX = a per-agent hash table (open addressing, no collisions) from a key code to a unit id, ids assigned in the
    reference's creation order; GEN (V+1 outputs), gates (3) and critic rows are dense per unit; learning is an
    index_add_ of the NLMS step over the active units (no duplicate rows, hence exact).
  * the conjunction schema is DATA: GROUPS (name, max values) and PAIRS (group a, group b); key slots are generated
    from them, so a new group (an item-code layer, own-speech / role groups) is one line each.
Exact mode (default): drives are sequential fp32 sums (Triton) in the reference's unit order, f64 where the reference
computes in f64, the reference's rng algorithm; the only rule that differs from the numpy prototype is that exact
ties in the DG k-WTA are broken by unit index (canonical; numpy's argpartition order is platform dependent).
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from kernels import seqsum, HashTable, topk_mask_canon, learn_fused, HAVE_TRITON

BOS, NONE, INV = -1, -2, -99
F32 = np.float32

# ------------------------------------------------------------------------------------------------ conjunction schema
GROUPS = [('P', 16), ('L', 1), ('N', 1), ('FAM', 1), ('P1', 1), ('PL', 1), ('NOP', 1), ('CP', 1), ('CN', 1),
          ('CC', 1), ('F', 32), ('A', 1), ('VR', 16), ('M', 1)]
PAIRS = [('P', 'M'), ('L', 'M'), ('P', 'L'), ('P', 'N'), ('L', 'N'), ('P1', 'N'), ('PL', 'N'),
         ('P', 'CP'), ('L', 'CP'), ('N', 'CP'), ('P', 'CN'), ('L', 'CN'), ('FAM', 'CN'),
         ('P', 'CC'), ('L', 'CC'), ('FAM', 'CC'), ('P', 'FAM'), ('N', 'FAM'),
         ('P', 'A'), ('L', 'A'), ('N', 'A'), ('P', 'F'), ('L', 'F'), ('P', 'VR'), ('L', 'VR'), ('N', 'VR'),
         ('NOP', 'VR'), ('NOP', 'F'), ('NOP', 'A'), ('NOP', 'L'), ('NOP', 'CC')]


class HashConj:
    """The conjunction layer (PFC-2 MIX) behind one interface:
        active(rows, vals, create) -> A (n, K) unit ids sorted ascending (-1 pad), nA (n,)
    vals (n, G+1) are the group values of each event (column G = 0); the schema (GROUPS, PAIRS) decides which value
    pairs become keys.  This implementation recruits a unit on the first co-occurrence (R3) through a per-agent hash
    table; a fixed random sparse projection (arch/v2/randmix) would implement the same call."""

    def __init__(self, B, U, logM, dev, groups=GROUPS, pairs=PAIRS):
        self.B, self.U, self.dev = B, U, dev
        self.table = HashTable(B, logM, dev)
        off, o = {}, 0
        for g, n in groups:
            off[g] = (o, n); o += n
        self.goff, self.G = off, o
        ga, gb, pid = [], [], []
        for gi, (g, n) in enumerate(groups):
            for i in range(n):
                ga.append(off[g][0] + i); gb.append(o); pid.append(40 + gi)
        for pi, (a, b) in enumerate(pairs):
            for i in range(off[a][1]):
                for j in range(off[b][1]):
                    ga.append(off[a][0] + i); gb.append(off[b][0] + j); pid.append(pi)
        self.k_ga = torch.tensor(ga, device=dev); self.k_gb = torch.tensor(gb, device=dev)
        self.k_pid = torch.tensor(pid, device=dev, dtype=torch.int64) << 44
        self.K = len(ga)

    def active(self, rows, vals, create):
        va, vb = vals[:, self.k_ga], vals[:, self.k_gb]
        valid = (va > -50) & (vb > -50)
        keys = self.k_pid | ((va + 8) << 22) | (vb + 8)
        uid, lost = self.table.lookup(rows, keys, valid, create)
        big = 1 << 40
        A = torch.sort(torch.where(uid >= 0, uid, torch.full_like(uid, big)), 1).values
        nA = (uid >= 0).sum(1)
        st = torch.stack([nA.max(), lost.any().long(), self.table.n.max()]).cpu().numpy()
        if st[1]:
            self.table.place_losers(rows, keys, uid, lost)
        if st[2] > self.U:
            raise RuntimeError('MIX unit capacity exceeded')
        A = A[:, :max(int(st[0]), 1)]
        return torch.where(A == big, torch.full_like(A, -1), A), nA

    @property
    def n(self):
        return self.table.n


PER_AGENT = ('eta', 'eta_g', 'eta_recall', 'eta_actor', 'eta_critic', 'theta_I', 'nogo', 'kappa', 'eta_oth', 'eta_go')


class Cfg:
    """final_noctx (run_j.py defaults + TUNE_BEST + THINK_FINAL + rebind ct init_min_tau hip_rho ctx=0)."""

    def __init__(self, **kw):
        self.V = 256; self.LM = 16; self.NP = 320
        self.eta = 0.35; self.eta_g = 0.5; self.gain = 1.0; self.floor = 0.3
        self.kappa = 2.0; self.eta_prime = 0.02; self.eta_attd = 0.02; self.match_thr = 0.6; self.rec_thr = 0.85
        self.rate_max = 0.4; self.conf_min = 6; self.match_n = 6; self.attd_n = 2
        self.hip_gamma = 0.9995; self.hip_cap = 10000; self.hip_ndg = 6000; self.hip_kdg = 120; self.hip_theta = 0.2
        self.hip_rho = 2.0
        self.eta_critic = 0.2; self.eta_actor = 0.3; self.recall_n = 8; self.eta_recall = 0.2
        self.match_min = 0.5; self.match_gap = 0.3; self.att_thr = 0.5
        self.think_k = 3; self.scene_recall = True; self.nop_keys = True
        self.ct_lam = 0.3; self.ct_theta = 0.5; self.ct_k = 12; self.ct_assoc = 0.0; self.ct_assoc_eta = 0.02
        self.ct_fat = 0.1; self.ct_fat_tau = 20.0
        self.seg = 240
        # agent (think + synth)
        self.theta_z = 0.75; self.z_rate = 0.1; self.theta_bind = 0.85
        self.theta_I = 1.0; self.g0 = (1.0, 0.3); self.u = (0.15, 0.0); self.beta = 2.0; self.g_max = 3.0
        self.nogo = 0.25; self.nogo_late = 0.05; self.L = 10; self.eta_oth = 0.2; self.eta_go = 0.2; self.tau_w = 7
        self.late_release = False; self.u_R1 = 1.0; self.T_MAX = 40; self.init_min_tau = 10; self.quiet_ticks = 12
        self.wander = 0; self.wander_after = 200; self.rebind = True; self.life_eye = False
        # body
        self.breath = 12; self.f_w = 0.01; self.f_thr = 0.1
        # port
        self.U = 1 << 17          # unit capacity per agent (MIX); the reference grows to ~49k in a full J run
        self.logM = 19            # hash slots per agent
        self.exact = True
        self.fused = False        # one Triton kernel per learning event (validated by test_trace before use)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)


def pairwise_sum_f32(a):
    """numpy's float32 1-D add.reduce (pairwise, 8-way unrolled, blocks of 128), exactly."""
    n = len(a)
    if n < 8:
        res = F32(-0.0)
        for x in a:
            res = F32(res + x)
        return res
    if n <= 128:
        r = [F32(a[j]) for j in range(8)]
        i = 8
        while i < n - (n % 8):
            for j in range(8):
                r[j] = F32(r[j] + a[i + j])
            i += 8
        res = F32(F32(F32(r[0] + r[1]) + F32(r[2] + r[3])) + F32(F32(r[4] + r[5]) + F32(r[6] + r[7])))
        while i < n:
            res = F32(res + a[i]); i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return F32(pairwise_sum_f32(a[:n2]) + pairwise_sum_f32(a[n2:]))


def choice_np(rng, a, size):
    """numpy Generator.choice(a, size, replace=False, p=uniform) re-implemented on the same random stream."""
    p = np.full(len(a), 1.0 / len(a)) if False else np.ones(len(a)) / len(a)
    n_uniq = 0
    found = np.zeros(size, np.int64)
    while n_uniq < size:
        x = rng.random((size - n_uniq,))
        if n_uniq > 0:
            p[found[0:n_uniq]] = 0
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        new = cdf.searchsorted(x, side='right')
        _, ui = np.unique(new, return_index=True)
        ui.sort()
        new = new.take(ui)
        found[n_uniq:n_uniq + new.size] = new
        n_uniq += new.size
    return a[found]


class LCtx:
    """a line context (develop's st dict) for B agents."""

    def __init__(self, B, c, NE):
        LM, GP = c.LM, 16
        self.items = np.full((B, LM), -1, np.int64); self.ni = np.zeros(B, np.int64)
        self.P = np.full((B, LM), -1, np.int64); self.nP = np.zeros(B, np.int64)
        self.Pk = np.full((B, GP), INV, np.int64)                  # values of the P key group
        self.ch = np.full((B, 3, LM), -1, np.int64); self.nch = np.zeros((B, 3), np.int64)
        self.used = np.zeros((B, 3, LM), bool)
        self.fam = np.zeros(B, np.int64)
        self.att = np.full(B, -1, np.int64); self.att_tutor = np.full(B, -1, np.int64)
        self.att_match = np.full(B, -1, np.int64)
        self.m = np.full(B, np.nan); self.has_eye = np.zeros(B, bool); self.eye = np.zeros((B, NE), bool)
        self.kind = np.zeros(B, np.int64)                           # 0 reply, 1 scene
        self.no_vr = np.ones(B, bool); self.has_VR = np.zeros(B, bool)
        self.VR = np.full((B, 16), INV, np.int64)
        self.ep = np.full(B, -1, np.int64)

    def copy_rows(self, src, ix, six=None):
        six = ix if six is None else six
        for k, v in self.__dict__.items():
            v[ix] = getattr(src, k)[six]

    def reset_rows(self, ix):
        self.items[ix] = -1; self.ni[ix] = 0; self.used[ix] = False


class RecSlot:
    """a line record (items, eye, att, ep, born) for B agents; its DG code lives on the device (code)."""

    def __init__(self, B, c, NE, dev, ndg):
        self.valid = np.zeros(B, bool)
        self.items = np.full((B, c.LM), -1, np.int64); self.n = np.zeros(B, np.int64)
        self.has_eye = np.zeros(B, bool); self.eye = np.zeros((B, NE), bool)
        self.att = np.full(B, -1, np.int64); self.ep = np.full(B, -1, np.int64); self.born = np.full(B, -1, np.int64)
        self.code = torch.zeros(B, ndg, dtype=torch.bool, device=dev)

    def copy_rows(self, src, ix):
        for k in ('valid', 'items', 'n', 'has_eye', 'eye', 'att', 'ep', 'born'):
            getattr(self, k)[ix] = getattr(src, k)[ix]
        tix = torch.as_tensor(ix, device=self.code.device)
        self.code[tix] = src.code[tix]


def dedup(row, n):
    out, seen = [], set()
    for x in row[:n]:
        x = int(x)
        if x not in seen:
            seen.add(x); out.append(x)
    return out


class Brain:
    def __init__(self, B, inits, cfg=None, device='cuda'):
        """inits: list of B dicts with F0, ear_idx, ear_w, ear_b, ear_alive, hip_tie, hip_idx_{bag,first,last,scene}
        (the fixed random wiring of each agent; numpy), seed, n_eye."""
        self.c = c = cfg or Cfg()
        self.B = B
        self.dev = dev = torch.device(device)
        V = c.V
        self.V, self.END = V, V
        self.NE = NE = int(inits[0]['n_eye'])
        # FEAT0 differs a little per world seed: pad every agent to the longest with a dead ear unit (never on), so
        # padded features are always 0 (templates, matches and babbling counts are unchanged: exact)
        self.NF = NF = max(len(i['F0']) for i in inits)
        self.nf = np.array([len(i['F0']) for i in inits])
        F0p = []
        for i in inits:
            dead = int(np.flatnonzero(~np.asarray(i['ear_alive']))[0])
            F0p.append(np.r_[np.asarray(i['F0'], np.int64), np.full(NF - len(i['F0']), dead, np.int64)])
        t = lambda a, dt=None: torch.as_tensor(np.stack(a), device=dev, dtype=dt)
        # ---- ear layer 0 (fixed)
        self.ear_idx = t([i['ear_idx'] for i in inits], torch.int64)
        self.ear_w = t([i['ear_w'] for i in inits], torch.float32)
        self.ear_b = t([i['ear_b'] for i in inits], torch.float32)
        self.ear_alive = t([i['ear_alive'] for i in inits], torch.bool)
        self.ear_k = int(round(0.4 * int(inits[0]['ear_alive'].sum())))
        self.F0 = t(F0p, torch.int64)
        self.lv = torch.arange(10, device=dev)
        # ---- Z templates (NF x V per agent) + babbling memory
        self.ZW = torch.zeros(B, NF, V, dtype=torch.float32, device=dev)
        self.zn = np.zeros(B, np.int64)
        self.zrows = torch.arange(NF, device=dev)
        self.bab = None; self.bab_n = np.zeros(B, np.int64); self.bab_sets = [None] * B
        # ---- body programs
        self.progs = np.zeros((B, c.NP, 32), bool); self.n_prog = np.zeros(B, np.int64)
        self.prog2cat = np.full((B, c.NP), -1, np.int64); self.cat2prog = np.full((B, V), -1, np.int64)
        # ---- MIX + readouts
        self.conj = HashConj(B, c.U, c.logM, dev)
        self.goff, self.G = self.conj.goff, self.conj.G
        self.W = torch.zeros(B * c.U + 1, V + 4, dtype=torch.float32, device=dev)   # GEN | 3 BG gates; last row = sink
        self.SINK = B * c.U
        self.CRIT = torch.zeros(B * c.U, dtype=torch.float32, device=dev)
        # ---- hippocampus
        self.NDG, self.KDG, cap = c.hip_ndg, c.hip_kdg, c.hip_cap
        self.hidx = {g: t([i['hip_idx_' + g] for i in inits], torch.int64) for g in ('bag', 'first', 'last', 'scene')}
        self.htie = t([i['hip_tie'] for i in inits], torch.float32)
        self.hkeys = torch.zeros(B, cap, self.KDG, dtype=torch.int16, device=dev)
        self.hsig = torch.zeros(B, cap, dtype=torch.float32, device=dev)
        self.hn = np.zeros(B, np.int64); self.hclock = np.zeros(B, np.int64)
        self.hborn = np.zeros((B, cap), np.int64); self.hitems = np.full((B, cap, c.LM), -1, np.int16)
        self.hni = np.zeros((B, cap), np.int64); self.heye = np.zeros((B, cap, NE), bool)
        self.hhas = np.zeros((B, cap), bool)
        self.hatt = np.full((B, cap), -1, np.int64); self.hnext = np.full((B, cap), -1, np.int64)
        self.next_of = np.full((B, cap), -1, np.int64); self.next_born = np.full((B, cap), -1, np.int64)
        self.rng = [np.random.default_rng(int(i['seed'])) for i in inits]
        # ---- primes (slot-relative) and word -> slot
        S = c.seg
        self.prW = torch.zeros(B, V + 1, S, dtype=torch.float32, device=dev)
        self.prN = torch.zeros(B, V + 1, dtype=torch.float32, device=dev)
        self.adW = torch.zeros(B, V + 1, 3, dtype=torch.float32, device=dev)
        self.adN = torch.zeros(B, V + 1, dtype=torch.float32, device=dev)
        self.eye_rate = torch.zeros(B, S, dtype=torch.float32, device=dev)
        self.eye_n = np.zeros(B, np.int64)
        # the reference caches recognition per (slot, eye) and invalidates only when the primes learn
        # (prime_version), although the result also depends on eye_rate / eye_n: replicated exactly
        self.prime_version = np.zeros(B, np.int64)
        self.rec_cache = [dict() for _ in range(B)]
        # ---- continuous thought + records
        self.T_a = np.zeros((B, V), F32)
        self.T_f = np.zeros((B, V), F32)
        self.T_W = np.zeros((B, V, V), F32) if c.ct_assoc > 0 else None
        self.DOM = RecSlot(B, c, NE, dev, self.NDG)
        self.LAST = RecSlot(B, c, NE, dev, self.NDG)
        self.PREV = RecSlot(B, c, NE, dev, self.NDG)          # core.prev: the thought's view (or a scene context)
        self.prev_mode = np.zeros(B, np.int64)                  # 0 None, 1 view, 2 scene_thought
        self.prev_bag = np.full((B, 16), INV, np.int64); self.prev_nbag = np.zeros(B, np.int64)
        # ---- line contexts
        self.ST = LCtx(B, c, NE); self.TH = LCtx(B, c, NE); self.OWN = LCtx(B, c, NE); self.RP = LCtx(B, c, NE)
        self.TMP = LCtx(B, c, NE)
        # ---- the thought
        self.th_on = np.zeros(B, bool); self.th_kind = np.zeros(B, np.int64); self.th_t0 = np.zeros(B, np.int64)
        self.th_k = np.zeros(B, np.int64); self.th_done = np.zeros(B, bool); self.th_I = np.zeros(B)
        self.th_G = np.zeros(B); self.th_V = np.zeros(B); self.th_e = np.zeros(B); self.th_rel = np.zeros(B, bool)
        self.th_spoke = np.zeros(B, bool); self.th_fb = np.zeros(B, bool)
        self.th_vis = np.full((B, 4), -1, np.int64)
        self.th_hascode = np.zeros(B, bool)
        self.th_code = torch.zeros(B, self.NDG, dtype=torch.bool, device=dev)
        self.th_feats = np.full((B, 40), -1, np.int64); self.th_nf = np.zeros(B, np.int64)
        # ---- turn learners (direct-indexed key tables: feature id x time cell)
        self.NFEAT = 6 + 3 * V
        self.oth_w = np.zeros((B, self.NFEAT * 31)); self.go_w = np.zeros((B, self.NFEAT))
        self.oth_used = np.zeros((B, self.NFEAT * 31), bool); self.go_used = np.zeros((B, self.NFEAT), bool)
        PQ = 16
        self.pend_t = np.full((B, PQ), -1, np.int64); self.pend_k = np.full((B, PQ, 41), -1, np.int64)
        self.pend_n = np.zeros((B, PQ), np.int64); self.pend_p = np.zeros((B, PQ)); self.pend_b = np.zeros((B, PQ))
        self.pend_len = np.zeros(B, np.int64)
        self.elig_on = np.zeros(B, bool); self.elig_k = np.full((B, 40), -1, np.int64)
        self.elig_n = np.zeros(B, np.int64); self.elig_t = np.zeros(B, np.int64)
        self.rbar = np.full(B, 0.5); self.rbar_n = np.zeros(B, np.int64)
        self.R_line = np.zeros(B, bool); self.spoke_since = np.zeros(B, bool)
        self.sr_on = np.zeros(B, bool); self.sr_t = np.zeros(B, np.int64); self.sr_eye = np.zeros((B, NE), bool)
        self.sr_att = np.full(B, -1, np.int64)
        # ---- agent state
        self.t = np.zeros(B, np.int64); self.E_prev = np.zeros(B); self.in_line = np.zeros(B, bool)
        self.clk = np.zeros(B, np.int64); self.quiet = np.zeros(B, np.int64); self.mouth_ago = np.full(B, 99, np.int64)
        self.learning = np.ones(B, bool); self.learn_listen = np.ones(B, bool)
        self.base = np.zeros((B, NE), bool); self.has_base = np.zeros(B, bool); self.base_changed = np.zeros(B, bool)
        self.prev_eye = np.zeros((B, NE), bool); self.has_prev_eye = np.zeros(B, bool)
        self.att_codes = np.zeros((B, 3, NE), bool); self.n_codes = np.zeros(B, np.int64)
        self.att_now = np.full(B, -1, np.int64); self.prev_att = np.full(B, -1, np.int64)
        self.own_on = np.zeros(B, bool)
        self.REL = np.zeros(B, bool)
        # ---- body
        self.bF = np.zeros(B, bool); self.bF2 = np.zeros(B, bool); self.bGO = np.zeros(B, bool)
        self.bR1 = np.zeros(B, bool); self.bR2 = np.zeros(B, bool); self.bCD = np.zeros((B, 4), bool)
        self.bE = np.zeros(B); self.Hp = np.full(B, -1, np.int64); self.phase = np.zeros(B, np.int64)
        self.SPK = np.zeros(B, bool); self.nsyl = np.zeros(B, np.int64); self.bY = np.full(B, -1, np.int64)
        self.bstop = np.zeros(B, np.int64)                     # 0 none 1 END 2 breath 3 floor 4 silent
        # ---- actor trace (A of each own syllable) on the device
        self.act_A = None; self.act_item = np.full((B, 16), -1, np.int64); self.act_n = np.zeros(B, np.int64)
        self.own_A = None; self.own_nA = np.zeros(B, np.int64)
        self.own_cand = np.full((B, 3), -1, np.int64); self.own_pos = np.full((B, 3), -1, np.int64)
        self.stats = dict(learn_events=0, awake_recalls=0, quiet=0, recall=0, lines=0, items=0, new_cats=0, own=0,
                          released=0, rebound=0, imitated=0)
        self.trace = None
        self.debug_t = set()
        self.prm = {k: np.full(B, float(getattr(c, k))) for k in PER_AGENT}
        for k in ('u_reply', 'u_scene', 'g0_reply', 'g0_scene'):
            pass
        self._kdg_t = torch.tensor(float(self.KDG), dtype=torch.float32, device=dev)
        self._1e4 = float(F32(1e-4))
        self._three_t = torch.tensor(3.0, dtype=torch.float32, device=dev)

    def key_tuples(self, vals_row):
        """the conjunction keys of one event as the reference's tuples (debugging)."""
        names = [g for g, n in GROUPS for _ in range(n)]
        out = []
        o = self.conj
        ga, gb, pid = o.k_ga.cpu().numpy(), o.k_gb.cpu().numpy(), (o.k_pid >> 44).cpu().numpy()
        for a, b, p in zip(ga, gb, pid):
            va = vals_row[a]; vb = vals_row[b]
            if va <= -50 or vb <= -50:
                continue
            fa = ('r', int(va)) if names[a] == 'F' else int(va)
            if p >= 40:
                out.append((names[a], fa))
            else:
                fb = ('r', int(vb)) if names[b] == 'F' else int(vb)
                out.append((names[a], fa, names[b], fb))
        return out

    def _debug(self, tag, C, ix, vals, A, nAt, total, gd, cand):
        hit = [r for r, b in enumerate(ix) if int(self.t[b]) in self.debug_t]
        for r in hit:
            b = ix[r]
            d = dict(tag=tag, t=int(self.t[b]), items=C.items[b, :C.ni[b]].tolist(), P=C.P[b, :C.nP[b]].tolist(),
                     Pk=[int(x) for x in C.Pk[b] if x > -50], ch=[C.ch[b, k, :C.nch[b, k]].tolist() for k in range(3)],
                     used=[np.flatnonzero(C.used[b, k]).tolist() for k in range(3)], fam=int(C.fam[b]),
                     att=int(C.att[b]), att_tutor=int(C.att_tutor[b]), att_match=int(C.att_match[b]), m=float(C.m[b]),
                     kind=int(C.kind[b]), no_vr=bool(C.no_vr[b]), VR=[int(x) for x in C.VR[b] if x > -50],
                     ep=int(C.ep[b]), cand=cand[r].tolist(), nA=int(nAt[r]), units=int(self.conj.n[b].item()))
            tot = total[r].cpu().numpy()
            top = np.argsort(-tot[:self.V], kind='stable')[:5]
            d['top'] = [(int(i), float(tot[i])) for i in top]
            d['end'] = float(tot[self.V]); d['gd'] = gd[r].cpu().numpy().tolist()
            d['keys'] = self.key_tuples(vals[r])
            print('DBG', d, flush=True)

    # ================================================================ data plane helpers
    def _t(self, a, dt=torch.int64):
        return torch.as_tensor(np.ascontiguousarray(a), device=self.dev, dtype=dt)

    def _ear(self, ix, win):
        """layer-0 response of the ear windows (n, 8, 6) -> FEAT0 pattern (n, NF) bool (device)."""
        tix = self._t(ix)
        w = self._t(win.reshape(len(ix), 8, 6, 1))
        up = w > self.lv
        lines = torch.stack([up, ~up], -1).reshape(len(ix), -1).to(torch.float32)
        idx, ww = self.ear_idx[tix], self.ear_w[tix]
        p = torch.gather(lines, 1, idx.reshape(len(ix), -1)).reshape(idx.shape) * ww
        s = p[..., 0] + p[..., 1]
        s = s + p[..., 2]
        s = s + p[..., 3]
        s = s + self.ear_b[tix]
        s = torch.where(self.ear_alive[tix], s, torch.full_like(s, -float('inf')))
        top = torch.topk(s, self.ear_k, dim=1).indices
        on = torch.zeros_like(s, dtype=torch.bool).scatter_(1, top, True)
        return torch.gather(on, 1, self.F0[tix])

    def _zmatch(self, ix, f):
        """Templates.match: num/den with sequential fp32 sums -> (n, V) f32 (device)"""
        n, NF, V = len(ix), self.NF, self.V
        tix = self._t(ix)
        base = (tix * NF)[:, None]
        rows = torch.where(f, base + self.zrows, torch.full_like(base + self.zrows, -1))   # ascending, -1 skipped
        W2 = self.ZW.view(-1, V)
        num = seqsum(W2, rows)
        den = seqsum(W2, base + self.zrows)
        return num / torch.clamp(den, min=F32(1e-6).item())

    def _bind(self, ix, cats, f):
        """_bind_from_memory for rows ix: category cats (n,), heard pattern f (n, NF) bool (device)."""
        if self.bab is None:
            return
        ok = (self.bab_n[ix] > 0) & (self.cat2prog[ix, cats] < 0)
        if not ok.any():
            return
        ix, cats, f = ix[ok], cats[ok], f[self._t(np.flatnonzero(ok))]
        tix = self._t(ix)
        P = self.bab[tix]                                                      # (n, NB, NF) fp16
        cnt = torch.bmm(P, f.to(P.dtype)[:, :, None])[..., 0].double()
        den = torch.maximum(torch.maximum(self.bab_sum[tix], f.sum(1, keepdim=True).double()), torch.ones_like(cnt))
        m = cnt / den
        m = torch.where(self.bab_valid[tix], m, torch.full_like(m, -1.0))
        k = torch.argmax(m, 1)
        mk = torch.gather(m, 1, k[:, None])[:, 0]
        k, mk = k.cpu().numpy(), mk.cpu().numpy()
        for r, b in enumerate(ix):
            if mk[r] >= self.c.theta_bind and self.cat2prog[b, cats[r]] < 0:
                y = self._commit(b, self.bab_sets[b][k[r]])
                if y is not None:
                    self.cat2prog[b, cats[r]] = y; self.prog2cat[b, y] = cats[r]; self.stats['imitated'] += 1

    def _commit(self, b, mus):
        v = np.zeros(32, bool); v[list(mus)] = True
        n = self.n_prog[b]
        if n:
            hit = np.flatnonzero((self.progs[b, :n] == v).all(1))
            if len(hit):
                return int(hit[0])
        if n >= self.c.NP:
            return None
        self.progs[b, n] = v; self.n_prog[b] = n + 1
        return int(n)

    def load_babble(self, ix, pats, sets):
        """babbling memory of agents ix: pats (NB, NF) bool, sets (NB, 4).  Then every heard category binds its best
        babbled sound (DevAgent.babble's tail)."""
        if pats.shape[1] < self.NF:
            pats = np.concatenate([pats, np.zeros((len(pats), self.NF - pats.shape[1]), bool)], 1)
        NB = max(len(pats), 1)
        if self.bab is None or self.bab.shape[1] < NB:
            old = self.bab
            self.bab = torch.zeros(self.B, NB, self.NF, dtype=torch.float16, device=self.dev)
            self.bab_sum = torch.zeros(self.B, NB, dtype=torch.float64, device=self.dev)
            self.bab_valid = torch.zeros(self.B, NB, dtype=torch.bool, device=self.dev)
            if old is not None:
                self.bab[:, :old.shape[1]] = old
        for b in ix:
            self.bab[b, :len(pats)] = torch.as_tensor(pats, device=self.dev).to(torch.float16)
            self.bab_sum[b, :len(pats)] = torch.as_tensor(pats.sum(1), device=self.dev).double()
            self.bab_valid[b, :len(pats)] = True
            self.bab_n[b] = len(pats); self.bab_sets[b] = [tuple(int(x) for x in s) for s in sets]
        for b in ix:
            for j in range(int(self.zn[b])):
                if self.cat2prog[b, j] < 0:
                    f = (self.ZW[b, :, j] >= 0.5)[None]
                    self._bind(np.array([b]), np.array([j]), f)

    # ---------------------------------------------------------------- keys and drives
    def _vals(self, C, ix, cand):
        """group values (n, G+1) of the contexts C[ix] with the copy candidates cand (n, 3)."""
        n, o = len(ix), self.goff
        V = self.V
        vals = np.full((n, self.G + 1), INV, np.int64)
        vals[:, self.G] = 0
        vals[:, o['P'][0]:o['P'][0] + 16] = C.Pk[ix]
        ni = C.ni[ix]
        last = np.where(ni > 0, C.items[ix, np.maximum(ni - 1, 0)], BOS)
        vals[:, o['L'][0]] = last
        vals[:, o['N'][0]] = np.minimum(ni, 6)
        vals[:, o['FAM'][0]] = C.fam[ix]
        nP = C.nP[ix]
        hp = nP > 0
        vals[hp, o['P1'][0]] = C.P[ix[hp], 0]
        vals[hp, o['PL'][0]] = C.P[ix[hp], nP[hp] - 1]
        nop = ~hp & (C.kind[ix] == 1) & self.c.nop_keys
        vals[nop, o['NOP'][0]] = 1
        for j, g in enumerate(('CP', 'CN', 'CC')):
            vals[:, o[g][0]] = np.where(cand[:, j] >= 0, cand[:, j], NONE)
        att = C.att[ix]
        foc = (att >= 0) & C.has_eye[ix]
        vals[:, o['A'][0]] = np.where(foc, att, NONE)
        S = self.c.seg
        fr = np.flatnonzero(foc)
        if len(fr):
            seg = C.eye[ix[fr], :3 * S].reshape(len(fr), 3, S)[np.arange(len(fr)), att[fr]]   # attended block
            pos = np.where(seg, np.arange(S), S + 1000)
            pos.sort(1)
            pos = pos[:, :32]
            assert not (seg.sum(1) > 32).any()
            vals[fr, o['F'][0]:o['F'][0] + 32] = np.where(pos >= S, INV, pos)
        vr = foc & ~C.no_vr[ix]
        need = np.flatnonzero(vr & ~C.has_VR[ix])
        if len(need):                                   # lazily, at the first key building (as SynthCore._keys)
            rx = ix[need]
            got = self._recognize(rx, C.eye[rx], C.att[rx])
            for r, b in enumerate(rx):
                assert len(got[r]) <= 16
                C.VR[b] = INV
                C.VR[b, :len(got[r])] = got[r]
                C.has_VR[b] = True
        vals[vr, o['VR'][0]:o['VR'][0] + 16] = C.VR[ix[vr]]
        m = C.m[ix]
        hm = ~np.isnan(m)
        vals[hm, o['M'][0]] = (m[hm] >= 0.5).astype(np.int64) + (m[hm] >= 0.8)
        return vals

    def _cands(self, C, ix):
        """OneShotChain.candidate for the 3 copy chains (P, N, C): (cand (n,3), pos (n,3)), -1 = None."""
        n, LM = len(ix), self.c.LM
        ch, nch, used = C.ch[ix], C.nch[ix], C.used[ix]
        ni = C.ni[ix]
        last = np.where(ni > 0, C.items[ix, np.maximum(ni - 1, 0)], -9)
        cand = np.full((n, 3), -1, np.int64); pos = np.full((n, 3), -1, np.int64)
        start = (ni == 0)[:, None] & (nch > 0) & ~used[:, :, 0]
        cand[start] = ch[:, :, 0][start]; pos[start] = 0
        ar = np.arange(LM - 1)
        eq = (ch[:, :, :-1] == last[:, None, None]) & (ar[None, None] < (nch - 1)[:, :, None]) & ~used[:, :, 1:]
        eq &= (ni > 0)[:, None, None]
        hit = eq.any(2)
        k = np.argmax(eq, 2)
        cand[hit] = np.take_along_axis(ch[:, :, 1:], k[:, :, None], 2)[:, :, 0][hit]
        pos[hit] = (k + 1)[hit]
        return cand, pos

    def _front(self, C, ix, create, target=None):
        """candidates, keys and active units of contexts C[ix] -> A, rows, nA, cand, pos, device cand, target."""
        cand, pos = self._cands(C, ix)
        vals_np = self._vals(C, ix, cand)
        n, G1 = vals_np.shape
        pack = np.empty((n, G1 + 5), np.int64)
        pack[:, :G1] = vals_np; pack[:, G1:G1 + 3] = cand; pack[:, G1 + 3] = ix
        pack[:, G1 + 4] = 0 if target is None else target
        P = self._t(pack)
        vals, candt, tix, tg = P[:, :G1], P[:, G1:G1 + 3], P[:, G1 + 3], P[:, G1 + 4]
        A, nAt = self.conj.active(tix, vals, create)
        rows = torch.where(A >= 0, A + (tix * self.c.U)[:, None], torch.full_like(A, -1))
        return A, rows, nAt, cand, pos, candt, tg

    def _drives(self, C, ix, create, target=None):
        """develop's _drives / _total_ro for contexts C[ix] -> A (n, K) sorted unit ids (-1 pad), rows, nA, total
        (n, V+1), gd (n, 3), cand, pos, (device cand, target).  create: absent conjunctions are recruited (R3)."""
        cand, pos = self._cands(C, ix)
        vals_np = self._vals(C, ix, cand)
        self._last_vals = vals_np
        n, G1 = vals_np.shape
        pack = np.empty((n, G1 + 5), np.int64)
        pack[:, :G1] = vals_np; pack[:, G1:G1 + 3] = cand; pack[:, G1 + 3] = ix
        pack[:, G1 + 4] = 0 if target is None else target
        P = self._t(pack)
        vals, candt, tix, tg = P[:, :G1], P[:, G1:G1 + 3], P[:, G1 + 3], P[:, G1 + 4]
        A, nAt = self.conj.active(tix, vals, create)
        rows = torch.where(A >= 0, A + (tix * self.c.U)[:, None], torch.full_like(A, -1))
        out = seqsum(self.W, rows)                                             # sequential fp32 = numpy's order
        V = self.V
        gd = out[:, V + 1:]
        pg = torch.clamp(gd, 0.0, 1.0)
        total = out[:, :V + 2].clone()                                         # column V+1 = scratch
        for j in range(3):                                                     # copy routes P, N, C (in order)
            cj = candt[:, j]
            ok = (cj >= 0) & (cj < V)
            cc = torch.where(ok, cj, torch.full_like(cj, V + 1))[:, None]
            cur = total.gather(1, cc)
            nw = (cur.double() + pg[:, j:j + 1].double()).float()
            total.scatter_(1, cc, torch.where(ok[:, None], nw, cur))
        total = total[:, :V + 1]
        if self.debug_t:
            self._debug('create' if create else 'ro', C, ix, vals_np, A, nAt, total, gd, cand)
        return A, rows, nAt, total, gd, cand, pos, candt, tg

    def _learn_step(self, C, ix, target, eta=None, g=1.0):
        """R2 on GEN (all routes' summed drive as the prediction) and the BG gates, for contexts C[ix]."""
        c = self.c
        n = len(ix)
        e_gen = self.prm['eta'][ix] if eta is None else self.prm['eta_recall'][ix]
        e_gate = self.prm['eta_g'][ix] if eta is None else self.prm['eta_recall'][ix] * self.prm['eta_g'][ix] / self.prm['eta'][ix]
        coef = self._t(np.stack([(e_gen * g).astype(F32), (e_gate * g).astype(F32)], 1), torch.float32)
        if c.fused and HAVE_TRITON and self.dev.type == 'cuda' and not self.debug_t:
            A, rows, nAt, cand, pos, candt, tg = self._front(C, ix, True, target)
            norm = (1.0 / nAt.double()).float()
            learn_fused(self.W, rows, candt, tg, coef * norm[:, None], self.V)
            self.stats['learn_events'] += n
            return cand, pos
        A, rows, nAt, total, gd, cand, pos, candt, tg = self._drives(C, ix, True, target)
        if self.trace is not None and 'learns' in self.trace:
            dg = torch.stack([nAt, torch.where(A >= 0, A, torch.zeros_like(A)).sum(1)], 1).cpu().numpy()
            for r, b in enumerate(ix):
                self.trace['learns'].append((int(self.t[b]), int(dg[r, 0]), int(dg[r, 1]), int(target[r]),
                                             self.key_tuples(self._last_vals[r])))
        norm = (1.0 / nAt.double()).float()                                    # np.full(len(A), 1/len(A), f32)
        tv = torch.zeros(n, self.V + 1, dtype=torch.float32, device=self.dev)
        tv.scatter_(1, tg[:, None], 1.0)
        p = torch.clamp(total, 0.0, 1.0)
        err = tv - p
        keep = ((tv > 0) | (p > self._1e4)) & (err.abs() > self._1e4)
        cg = coef[:, 0] * norm
        step = torch.where(keep, cg[:, None] * err, torch.zeros_like(err))
        tgt = (candt == tg[:, None]).float()
        e2 = tgt - torch.clamp(gd, 0.0, 1.0)
        k2 = (candt >= 0) & (e2.abs() > self._1e4)
        c2 = coef[:, 1] * norm
        s2 = torch.where(k2, c2[:, None] * e2, torch.zeros_like(e2))
        S = torch.cat([step, s2], 1)
        K = rows.shape[1]
        rr = torch.where(rows >= 0, rows, torch.full_like(rows, self.SINK)).reshape(-1)
        self.W.index_add_(0, rr, S[:, None, :].expand(n, K, S.shape[1]).reshape(-1, S.shape[1]))
        self.stats['learn_events'] += n
        return cand, pos

    def _confidence(self, C, ix):
        A, rows, nAt, total, gd, cand, pos, candt, tg = self._drives(C, ix, False)
        prog = total[:, :self.V]
        j = torch.argmax(prog, 1)
        tk = torch.topk(prog, 2, dim=1).values
        out = torch.stack([tk[:, 0], tk[:, 1], j.float()], 1).cpu().numpy()
        s1, s2 = out[:, 0].astype(np.float64), out[:, 1].astype(np.float64)
        e = np.clip(s1 - np.maximum(s2, 0.0), 0.0, 1.0)
        return e

    def _advance(self, C, ix, item, cand, pos):
        r = np.arange(len(ix))
        C.items[ix, C.ni[ix]] = item
        C.ni[ix] += 1
        for k in range(3):
            hit = (cand[:, k] >= 0) & (cand[:, k] == item)
            if hit.any():
                C.used[ix[hit], k, pos[hit, k]] = True
        return r

    # ---------------------------------------------------------------- hippocampus
    def _dg(self, ix, items, ni, has_eye, eye):
        """DG k-WTA code (n, NDG) bool (device) of the cue (line bag, first, last, scene)."""
        n, V = len(ix), self.V
        tix = self._t(ix)
        bag = np.zeros((n, V), bool); fi = np.zeros((n, V), bool); la = np.zeros((n, V), bool)
        for r in range(n):
            it = items[r, :ni[r]]
            it = it[(it >= 0) & (it < V)]
            bag[r, it] = True
            if ni[r]:
                if 0 <= items[r, 0] < V:
                    fi[r, items[r, 0]] = True
                if 0 <= items[r, ni[r] - 1] < V:
                    la[r, items[r, ni[r] - 1]] = True
        sc = np.where(has_eye[:, None], eye, False)
        d = self.htie[tix]
        for g, x in (('bag', bag), ('first', fi), ('last', la), ('scene', sc)):
            xt = self._t(x, torch.float32)
            idx = self.hidx[g][tix]
            cnt = torch.gather(xt, 1, idx.reshape(n, -1)).reshape(idx.shape).sum(-1)
            d = (d.double() + cnt.double()).float()
        return topk_mask_canon(d, self.KDG)

    def _recall_next(self, ix, code, visited):
        """best not-yet-visited episode for the key code (n, NDG): score overlap * sigma^2 (rho = 2)."""
        n = len(ix)
        e_out = np.full(n, -1, np.int64); ov_out = np.zeros(n)
        hn = self.hn[ix]
        ok = hn > 0
        if not ok.any():
            return e_out, ov_out
        r = np.flatnonzero(ok)
        tix = self._t(ix[r])
        cap = self.c.hip_cap
        nmax = int(hn[r].max())
        keys = self.hkeys[tix, :nmax].long()
        cd = code[self._t(r)]
        cnt = torch.gather(cd, 1, keys.reshape(len(r), -1)).reshape(keys.shape).sum(-1)
        ov = cnt.float() / self._kdg_t          # a device tensor: torch turns x / python_scalar into x * (1/s)
        sig = torch.clamp(self.hsig[tix, :nmax], min=0.0)
        assert self.c.hip_rho == 2.0
        sc = ov * (sig * sig)
        inr = torch.arange(nmax, device=self.dev)[None] < self._t(hn[r])[:, None]
        sc = torch.where(inr, sc, torch.full_like(sc, -2.0))
        vis = self._t(visited[r])
        vm = (vis >= 0) & (vis < self._t(hn[r])[:, None])
        rr = torch.arange(len(r), device=self.dev)[:, None].expand_as(vis)
        sc[rr[vm], vis[vm]] = -1.0
        e = torch.argmax(sc, 1)
        g = lambda x: torch.gather(x, 1, e[:, None])[:, 0]
        res = torch.stack([e.double(), g(sc).double(), g(ov).double(), g(self.hsig[tix, :nmax]).double()], 1).cpu().numpy()
        for q, row in enumerate(r):
            ee, s, o, sg = int(res[q, 0]), res[q, 1], res[q, 2], res[q, 3]
            ov_out[row] = o
            if s < 0 or o < F32(self.c.hip_theta) or sg <= F32(0.05):
                continue
            e_out[row] = ee
        return e_out, ov_out

    def _store(self, ix, code, rec):
        """Hippo.store + tick(1) for rows ix with DG codes code (n, NDG) and record slot rec -> episode ids."""
        n = len(ix)
        cap = self.c.hip_cap
        ep = np.zeros(n, np.int64)
        full = self.hn[ix] >= cap
        ep[~full] = self.hn[ix[~full]]
        if full.any():
            fx = ix[full]
            ep[full] = torch.argmin(self.hsig[self._t(fx)], 1).cpu().numpy()
        self.hn[ix[~full]] += 1
        tix = self._t(ix)
        tep = self._t(ep)
        kk = code.nonzero()[:, 1].reshape(n, self.KDG)
        self.hkeys[tix, tep] = kk.to(torch.int16)
        self.hsig[tix, tep] = 1.0
        self.hborn[ix, ep] = self.hclock[ix]
        self.hitems[ix, ep] = rec.items[ix]
        self.hni[ix, ep] = rec.n[ix]
        self.heye[ix, ep] = np.where(rec.has_eye[ix][:, None], rec.eye[ix], False)
        self.hhas[ix, ep] = rec.has_eye[ix]
        self.hatt[ix, ep] = rec.att[ix]
        self.hnext[ix, ep] = -1
        self.next_of[ix, ep] = -1
        self.hclock[ix] += 1
        inr = torch.arange(cap, device=self.dev)[None] < self._t(self.hn[ix])[:, None]
        g = torch.tensor(F32(self.c.hip_gamma), device=self.dev)
        self.hsig[tix] = torch.where(inr, self.hsig[tix] * g, self.hsig[tix])
        return ep

    # ---------------------------------------------------------------- vision (slot-relative primes)
    def _conf(self, tix):
        """(n, V, S) bool confident expectations of every word (prime.conf & eye-rate filter)."""
        c = self.c
        W = self.prW[tix, :self.V]
        cf = (W >= F32(c.match_thr).item()) & (self.prN[tix, :self.V] >= c.match_n)[..., None]
        old = self._t(self.eye_n[tix.cpu().numpy()] >= 50, torch.bool)
        cf = cf & (~old[:, None, None] | (self.eye_rate[tix] < F32(c.rate_max).item())[:, None, :])
        return cf

    def _slots(self, eye_rows):
        S = self.c.seg
        e = self._t(eye_rows[:, :3 * S].reshape(len(eye_rows), 3, S), torch.float32)
        return e

    def _match(self, ix, Pmask, eye_rows):
        """per slot: max over the heard words of the fraction of the word's confident expectation present."""
        tix = self._t(ix)
        cf = self._conf(tix).float()                                          # (n, V, S)
        nn = cf.sum(-1)
        good = (nn >= self.c.conf_min) & self._t(Pmask, torch.bool)
        e3 = self._slots(eye_rows)
        cnt = torch.bmm(cf, e3.transpose(1, 2))                               # (n, V, 3)
        frac = cnt.double() / torch.clamp(nn.double(), min=1.0)[..., None]
        frac = torch.where(good[..., None], frac, torch.full_like(frac, -1.0))
        mx = frac.max(1).values
        out = torch.where(mx < 0, torch.zeros_like(mx), mx).float()
        return out.cpu().numpy()

    def _attd(self, ix, Pmask):
        tix = self._t(ix)
        ok = (self.adN[tix, :self.V] >= self.c.attd_n) & self._t(Pmask, torch.bool)
        W = torch.where(ok[..., None], self.adW[tix, :self.V], torch.full_like(self.adW[tix, :self.V], -1.0))
        mx = W.max(1).values
        anyok = ok.any(1)[:, None]
        return torch.where(anyok, mx, torch.zeros_like(mx)).cpu().numpy()

    def _recognize(self, ix, eye_rows, att):
        """SynthCore._recognize with its cache: key (slot, eye bytes), valid while prime_version is unchanged."""
        out = [None] * len(ix)
        keys = [(int(att[r]), eye_rows[r].tobytes()) for r in range(len(ix))]
        miss = []
        for r, b in enumerate(ix):
            hit = self.rec_cache[b].get(keys[r])
            if hit is not None and hit[0] == self.prime_version[b]:
                out[r] = hit[1]
            else:
                miss.append(r)
        if miss:
            mr = np.array(miss)
            fresh = self._recognize_now(ix[mr], eye_rows[mr], att[mr])
            for q, r in enumerate(miss):
                b = ix[r]
                if len(self.rec_cache[b]) > 5000:
                    self.rec_cache[b] = {}
                self.rec_cache[b][keys[r]] = (int(self.prime_version[b]), fresh[q])
                out[r] = fresh[q]
        return out

    def _recognize_now(self, ix, eye_rows, att):
        """words whose confident expected appearance is present (>= rec_thr) at slot att -> list per row."""
        tix = self._t(ix)
        S = self.c.seg
        cf = self._conf(tix).float()
        nn = cf.sum(-1)
        good = (nn >= self.c.conf_min) & (self.prN[tix, :self.V] >= self.c.match_n)
        es = np.stack([eye_rows[r, att[r] * S:(att[r] + 1) * S] for r in range(len(ix))])
        e = self._t(es, torch.float32)
        cnt = torch.bmm(cf, e[:, :, None])[..., 0]
        m = cnt.double() / torch.clamp(nn.double(), min=1.0)
        hit = (good & (m >= self.c.rec_thr)).cpu().numpy()
        return [np.flatnonzero(h) for h in hit]

    def _outstar(self, W, N, tix, Pmask_t, tgt, m_min):
        """Outstar.learn for the words in Pmask (n, V) with targets tgt (n, X): w += max(1/n, m_min) (t - w)."""
        n = len(tix)
        r, w = Pmask_t.nonzero(as_tuple=True)
        b = tix[r]
        N[b, w] += 1.0
        m = torch.maximum(1.0 / N[b, w], torch.tensor(F32(m_min), device=self.dev))
        Wo = W[b, w]
        W[b, w] = Wo + m[:, None] * (tgt[r] - Wo)

    # ================================================================ the continuous thought and records
    def _view(self, ix):
        """core.prev = _ct_view() for rows ix (mode 1 = view, 0 = None)."""
        c, V = self.c, self.V
        for b in ix:
            held = self.T_a[b] >= F32(c.ct_theta)
            if not held.any() or not self.DOM.valid[b]:
                self.prev_mode[b] = 0
                continue
            dn = self.DOM.n[b]
            items = [int(i) for i in self.DOM.items[b, :dn] if 0 <= i < V and held[i]]
            inl = set(items)
            extra = [int(i) for i in np.flatnonzero(held) if int(i) not in inl]
            extra.sort(key=lambda i: -float(self.T_a[b, i]))
            bag = list(dict.fromkeys(items)) + extra
            if not bag:
                self.prev_mode[b] = 0
                continue
            it = items if items else bag
            self.prev_mode[b] = 1
            self.PREV.items[b] = -1; self.PREV.items[b, :len(it)] = it; self.PREV.n[b] = len(it)
            self.prev_bag[b] = INV; self.prev_bag[b, :len(bag)] = bag; self.prev_nbag[b] = len(bag)
            self.PREV.has_eye[b] = self.DOM.has_eye[b]; self.PREV.eye[b] = self.DOM.eye[b]
            self.PREV.att[b] = self.DOM.att[b]; self.PREV.ep[b] = self.DOM.ep[b]; self.PREV.born[b] = self.DOM.born[b]
        if len(ix):
            tix = self._t(ix)
            self.PREV.code[tix] = self.DOM.code[tix]

    def _ct_input(self, ix):
        c = self.c
        fac = F32(1.0 - (1.0 - c.ct_lam) * 1.0)
        for b in ix:
            n = self.LAST.n[b]
            items = [int(i) for i in self.LAST.items[b, :n] if 0 <= i < self.V]
            if not items:
                continue
            self.T_a[b] *= fac
            for i in items:
                if F32(1.0) > self.T_a[b, i]:
                    self.T_a[b, i] = F32(1.0)
            self.DOM.valid[b] = True
            for k in ('items', 'n', 'has_eye', 'eye', 'att', 'ep', 'born'):
                getattr(self.DOM, k)[b] = getattr(self.LAST, k)[b]
            self.DOM.code[b] = self.LAST.code[b]
            held = np.flatnonzero(self.T_a[b] >= F32(c.ct_theta))
            if len(held) > c.ct_k:
                order = held[np.argsort(-self.T_a[b, held], kind='stable')]
                self.T_a[b, order[c.ct_k:]] = np.minimum(self.T_a[b, order[c.ct_k:]], c.ct_theta * 0.99)
            if self.T_W is not None:
                hi = np.flatnonzero(self.T_a[b] >= F32(c.ct_theta))
                if len(hi) > 1:
                    W = self.T_W[b]
                    W[np.ix_(hi, hi)] += c.ct_assoc_eta
                    np.fill_diagonal(W, 0)
                    s = W[hi].sum(1, keepdims=True)
                    W[hi] /= np.maximum(s, 1.0)

    def _tick_thought(self, ix):
        c = self.c
        for b in ix:
            a = self.T_a[b]
            if a.max() <= 0:
                continue
            self.T_f[b] += (a - self.T_f[b]) / c.ct_fat_tau
            a *= (1.0 - c.ct_fat * self.T_f[b])
            spread = c.ct_assoc * (a @ self.T_W[b])
            np.maximum(a, spread, out=a)
            m = float(a.max())
            if m > 0:
                a /= m
            held = np.flatnonzero(a >= c.ct_theta)
            if len(held) > c.ct_k:
                order = held[np.argsort(-a[held], kind='stable')]
                a[order[c.ct_k:]] = np.minimum(a[order[c.ct_k:]], c.ct_theta * 0.99)

    # ================================================================ thoughts (steps of ThinkCore)
    def _ctx_new(self, C, ix, kind, eye, has_eye, att, use_prev):
        """ThinkCore.think_new's st for rows ix; use_prev: rows whose prompt is core.prev (reply kind)."""
        C.reset_rows(ix)
        C.kind[ix] = kind
        C.has_eye[ix] = has_eye; C.eye[ix] = np.where(has_eye[:, None], eye, False)
        C.att_tutor[ix] = att; C.att[ix] = att; C.att_match[ix] = -1
        C.no_vr[ix] = True; C.has_VR[ix] = False; C.VR[ix] = INV
        C.fam[ix] = 0; C.m[ix] = np.nan; C.ep[ix] = -1
        C.P[ix] = -1; C.nP[ix] = 0; C.Pk[ix] = INV
        C.ch[ix] = -1; C.nch[ix] = 0
        for r, b in enumerate(ix):
            if use_prev[r]:
                n = self.PREV.n[b]
                C.P[b, :n] = self.PREV.items[b, :n]; C.nP[b] = n
            dd = dedup(C.P[b], C.nP[b])
            C.Pk[b, :len(dd)] = dd
            C.ch[b, 0, :C.nP[b]] = C.P[b, :C.nP[b]]; C.nch[b, 0] = C.nP[b]

    def _use_bag(self, C, ix, use_prev):
        for r, b in enumerate(ix):
            if use_prev[r] and self.prev_mode[b] == 1:
                C.Pk[b] = self.prev_bag[b]

    def _step_recall(self, C, ix, code, visited, ok):
        """k=1: hippocampal recall #1 (best unvisited episode for the thought's key) -> N / C chains, FAM."""
        if not ok.any():
            return visited
        r = np.flatnonzero(ok)
        e, ov = self._recall_next(ix[r], code[self._t(r)], visited[r])
        for q, row in enumerate(r):
            if e[q] < 0:
                continue
            b = ix[row]
            ep = int(e[q])
            vv = visited[row]
            vv[np.argmax(vv < 0)] = ep
            self.stats['recall'] += 1
            self._set_recall(C, b, ep, ov[q])
        return visited

    def _set_recall(self, C, b, e, ov):
        C.fam[b] = 1 + int(ov >= 0.45) + int(ov >= 0.7)
        nxt = self.hnext[b, e]
        C.ch[b, 1:] = -1; C.nch[b, 1:] = 0; C.used[b, 1:] = False
        if nxt >= 0:
            n = self.hni[b, nxt]
            C.ch[b, 1, :n] = self.hitems[b, nxt, :n]; C.nch[b, 1] = n
        pb = set(int(x) for x in C.P[b, :C.nP[b]])
        n = self.hni[b, e]
        comp = [int(i) for i in self.hitems[b, e, :n] if int(i) not in pb]
        C.ch[b, 2, :len(comp)] = comp; C.nch[b, 2] = len(comp)
        C.ep[b] = e

    def _step_compare(self, C, ix, scene):
        """k=2: comparator -> internal attention (reply) / innate scan (scene)."""
        c = self.c
        sc = scene & (C.att_tutor[ix] < 0) & C.has_eye[ix]
        C.att[ix[sc]] = 0
        rp = ~scene & C.has_eye[ix] & (C.nP[ix] > 0)
        if not rp.any():
            return
        rx = ix[rp]
        Pm = np.zeros((len(rx), self.V), bool)
        for r, b in enumerate(rx):
            it = C.P[b, :C.nP[b]]
            Pm[r, it[(it >= 0) & (it < self.V)]] = True
        ms = self._match(rx, Pm, C.eye[rx])
        srt = np.sort(ms, 1)
        gm = (srt[:, -1] >= 0.5) & (srt[:, -1] - srt[:, -2] >= 0.3)
        C.att_match[rx[gm]] = np.argmax(ms[gm], 1)
        nt = C.att_tutor[rx] < 0
        if nt.any():
            ad = self._attd(rx[nt], Pm[nt])
            d = ad + self.prm['kappa'][rx[nt]].astype(F32)[:, None] * ms[nt]
            j = np.argmax(d, 1)
            ok = d[np.arange(len(j)), j] > 0.5
            C.att[rx[nt][ok]] = j[ok]
        ha = C.att[rx] >= 0
        C.m[rx[ha]] = ms[ha, C.att[rx[ha]]].astype(np.float64)

    def _step_vr(self, C, ix):
        """k=3: visual recognition at the attended slot becomes available; it is computed lazily at the next key
        building (SynthCore._keys), which matters for the reference's recognition cache."""
        C.no_vr[ix] = False
        C.has_VR[ix] = False
        C.VR[ix] = INV

    def _think_complete(self, C, ix, code, visited, scene):
        visited = self._step_recall(C, ix, code, visited, np.ones(len(ix), bool) & self._hascode)
        self._step_compare(C, ix, scene)
        self._step_vr(C, ix)
        return visited

    # ================================================================ listening
    def _begin_line(self, ix):
        """a heard line starts: its context = the completed thought about core.prev (ThinkCore.begin_line)."""
        notst = self.prev_mode[ix] != 2
        self._view(ix[notst])
        scene = self.prev_mode[ix] == 2
        use = ~scene & (self.prev_mode[ix] == 1)
        C = self.ST
        self._ctx_new(C, ix, scene.astype(np.int64), self.base[ix], self.has_base[ix], self.att_now[ix], use)
        code = torch.zeros(len(ix), self.NDG, dtype=torch.bool, device=self.dev)
        hascode = use.copy()
        if use.any():
            code[self._t(np.flatnonzero(use))] = self.PREV.code[self._t(ix[use])]
        sc = scene & self.has_base[ix]
        if sc.any():
            r = np.flatnonzero(sc)
            n = len(r)
            code[self._t(r)] = self._dg(ix[r], np.full((n, self.c.LM), -1), np.zeros(n, np.int64),
                                        np.ones(n, bool), self.base[ix[r]])
            hascode[r] = True
        visited = np.full((len(ix), 4), -1, np.int64)
        vp = use & (self.PREV.ep[ix] >= 0)
        visited[vp, 0] = self.PREV.ep[ix[vp]]
        self._hascode = hascode
        self._think_complete(C, ix, code, visited, scene)
        self._use_bag(C, ix, use)

    def _detect(self, ix, win):
        """perception: syllable category per sample (vigilance; new categories bind babbled sounds)."""
        c = self.c
        f = self._ear(ix, win)
        out = np.full(len(ix), -1, np.int64)
        has = self.zn[ix] > 0
        m = np.zeros(len(ix), F32); j = np.full(len(ix), -1, np.int64)
        if has.any():
            r = np.flatnonzero(has)
            mm = self._zmatch(ix[r], f[self._t(r)])
            V = self.V
            colok = torch.arange(V, device=self.dev)[None] < self._t(self.zn[ix[r]])[:, None]
            mm = torch.where(colok, mm, torch.full_like(mm, -float('inf')))
            jj = torch.argmax(mm, 1)
            mv = torch.gather(mm, 1, jj[:, None])[:, 0]
            j[r] = jj.cpu().numpy(); m[r] = mv.cpu().numpy()
        lrn = self.learning[ix]
        hit = (j >= 0) & (m >= F32(c.theta_z))
        out[hit] = j[hit]
        hl = hit & lrn
        if hl.any():
            r = np.flatnonzero(hl)
            tix = self._t(ix[r]); tj = self._t(j[r])
            fv = f[self._t(r)].float()
            wj = self.ZW[tix, :, tj]
            self.ZW[tix, :, tj] = float(F32(1 - c.z_rate)) * wj + float(F32(c.z_rate)) * fv
        fs = f.sum(1).cpu().numpy()
        new = ~hit & lrn & (self.zn[ix] < self.V) & (fs > 40)
        if new.any():
            r = np.flatnonzero(new)
            for q in r:
                b = ix[q]
                jn = int(self.zn[b]); self.zn[b] += 1
                self.ZW[b, :, jn] = f[q].float()
                out[q] = jn
                self.stats['new_cats'] += 1
            self._bind(ix[r], out[r], f[self._t(r)])
        if c.rebind:
            rb = (out >= 0) & lrn
            rb &= self.cat2prog[ix, np.maximum(out, 0)] < 0
            rb &= self.bab_n[ix] > 0
            if rb.any():
                r = np.flatnonzero(rb)
                before = (self.cat2prog[ix[r], out[r]] >= 0).sum()
                self._bind(ix[r], out[r], f[self._t(r)])
                self.stats['rebound'] += int((self.cat2prog[ix[r], out[r]] >= 0).sum() - before)
        return out

    def _line_item(self, ix, it):
        lrn = self.learning[ix] & self.learn_listen[ix] & (self.prev_mode[ix] != 0)
        cand = np.full((len(ix), 3), -1, np.int64); pos = np.full((len(ix), 3), -1, np.int64)
        if lrn.any():
            r = np.flatnonzero(lrn)
            cand[r], pos[r] = self._learn_step(self.ST, ix[r], it[r])
        if (~lrn).any():
            r = np.flatnonzero(~lrn)
            cand[r], pos[r] = self._cands(self.ST, ix[r])
        self._advance(self.ST, ix, it, cand, pos)

    def _line_end(self, ix):
        c, C = self.c, self.ST
        lrn = self.learning[ix]
        l2 = lrn & self.learn_listen[ix] & (self.prev_mode[ix] != 0) & (C.ni[ix] > 0)
        if l2.any():
            r = np.flatnonzero(l2)
            self._learn_step(C, ix[r], np.full(len(r), self.END))
        ey = lrn & C.has_eye[ix]
        S = c.seg
        ex = ix[ey]
        if len(ex):
            self.eye_n[ex] += 1
            e3 = self._t(C.eye[ex, :3 * S].reshape(len(ex), 3, S), torch.float32)
            ev = ((e3[:, 0] + e3[:, 1]) + e3[:, 2]) / self._three_t
            fac = self._t(np.maximum(1.0 / self.eye_n[ex], 0.01).astype(F32)[:, None], torch.float32)
            tx = self._t(ex)
            er = self.eye_rate[tx]
            self.eye_rate[tx] = er + (ev - er) * fac
        gl = lrn & C.has_eye[ix] & (C.ni[ix] > 0)
        if gl.any():
            gx = ix[gl]
            Pm = np.zeros((len(gx), self.V), bool)
            for r, b in enumerate(gx):
                it = C.items[b, :C.ni[b]]
                Pm[r, it[(it >= 0) & (it < self.V)]] = True
            a_ext = C.att_tutor[gx].copy(); a_m = C.att_match[gx].copy()
            sg = (a_ext < 0) & (a_m < 0)
            if sg.any():
                ms = self._match(gx[sg], Pm[sg], C.eye[gx[sg]])
                srt = np.sort(ms, 1)
                ok = (srt[:, -1] >= c.match_min) & (srt[:, -1] - srt[:, -2] >= c.match_gap)
                am = np.where(ok, np.argmax(ms, 1), -1)
                a_m[sg] = am
            a = np.where(a_ext >= 0, a_ext, a_m)
            la = a >= 0
            if la.any():
                tix = self._t(gx[la])
                oh = torch.eye(3, device=self.dev)[self._t(a[la])]
                self._outstar(self.adW, self.adN, tix, self._t(Pm[la], torch.bool), oh, c.eta_attd)
            lp = a_ext >= 0
            if lp.any():
                px = gx[lp]
                es = np.stack([C.eye[b, a_ext[lp][r] * S:(a_ext[lp][r] + 1) * S] for r, b in enumerate(px)])
                tgt = self._t(es, torch.float32)
                self._outstar(self.prW, self.prN, self._t(px), self._t(Pm[lp], torch.bool), tgt, c.eta_prime)
                self.prime_version[px] += 1
        self._end_line(ix, lrn)

    def _end_line(self, ix, store):
        C = self.ST
        code = self._dg(ix, C.items[ix], C.ni[ix], C.has_eye[ix], C.eye[ix])
        L = self.LAST
        old_valid, old_ep, old_born = L.valid[ix].copy(), L.ep[ix].copy(), L.born[ix].copy()
        L.valid[ix] = True
        L.items[ix] = C.items[ix]; L.n[ix] = C.ni[ix]; L.has_eye[ix] = C.has_eye[ix]; L.eye[ix] = C.eye[ix]
        L.att[ix] = C.att[ix]; L.ep[ix] = -1; L.born[ix] = -1
        L.code[self._t(ix)] = code
        st = store & (C.ni[ix] > 0)
        if st.any():
            r = np.flatnonzero(st)
            sx = ix[r]
            ep = self._store(sx, code[self._t(r)], L)
            L.ep[sx] = ep; L.born[sx] = self.hborn[sx, ep]
            for q, b in enumerate(sx):
                pe = old_ep[r[q]]
                if old_valid[r[q]] and pe >= 0 and self.hborn[b, pe] == old_born[r[q]]:
                    self.hnext[b, pe] = ep[q]; self.next_of[b, pe] = ep[q]; self.next_born[b, pe] = self.hborn[b, ep[q]]
        self._ct_input(ix)
        self._view(ix)
        self.stats['lines'] += len(ix)

    # ================================================================ the agent's thought
    def _open(self, ix, kind):
        c = self.c
        if not len(ix):
            return
        C = self.TH
        if kind == 0:
            sc = self.prev_mode[ix] != 2
            self._view(ix[sc])
            use = self.prev_mode[ix] == 1
        else:
            use = np.zeros(len(ix), bool)
        self._ctx_new(C, ix, np.full(len(ix), kind), self.base[ix], self.has_base[ix], self.att_now[ix], use)
        self.th_on[ix] = True; self.th_kind[ix] = kind; self.th_t0[ix] = self.t[ix]; self.th_k[ix] = 0
        self.th_done[ix] = False; self.th_I[ix] = 0.0; self.th_rel[ix] = False; self.th_spoke[ix] = False
        self.th_fb[ix] = False
        self.th_vis[ix] = -1
        vp = use & (self.PREV.ep[ix] >= 0)
        self.th_vis[ix[vp], 0] = self.PREV.ep[ix[vp]]
        tix = self._t(ix)
        self.th_hascode[ix] = use
        if use.any():
            self.th_code[self._t(ix[use])] = self.PREV.code[self._t(ix[use])]
        if kind == 1:
            sc = self.has_base[ix]
            if sc.any():
                r = ix[sc]
                n = len(r)
                self.th_code[self._t(r)] = self._dg(r, np.full((n, c.LM), -1), np.zeros(n, np.int64),
                                                    np.ones(n, bool), self.base[r])
                self.th_hascode[r] = True
        self.th_e[ix] = self._confidence(C, ix)
        self._use_bag(C, ix, use)
        # features of the thought for the turn learners
        V = self.V
        for b in ix:
            P = C.P[b, :C.nP[b]]
            f = [1 + kind, 3 + int(self.R_line[b])]
            f += [5 + int(i) for i in sorted(set(int(x) for x in P))]
            if len(P):
                f += [5 + V + int(P[0]), 5 + 2 * V + int(P[-1])]
            if C.att_tutor[b] >= 0:
                f.append(5 + 3 * V)
            self.th_feats[b] = -1; self.th_feats[b, :len(f)] = f; self.th_nf[b] = len(f)
            v = self._pr_p(self.go_w[b], f, 0.5)
            self.th_V[b] = v
            self.th_G[b] = float(np.clip(c.g0[kind] + c.beta * (v - self.rbar[b]), 0.0, c.g_max))

    def _think_step(self, ix):
        """one internal step of the agent's open thought for rows ix (k advances by one) -> confidence e."""
        if not len(ix):
            return np.zeros(0)
        C = self.TH
        k = self.th_k[ix] + 1
        self.th_k[ix] = k
        k1 = k == 1
        if k1.any():
            r = ix[k1]
            ok = (self.th_kind[r] == 0) | self.c.scene_recall
            ok &= self.th_hascode[r]
            vis = self.th_vis[r].copy()
            vis = self._step_recall(C, r, self.th_code[self._t(r)], vis, ok)
            self.th_vis[r] = vis
        k2 = k == 2
        if k2.any():
            r = ix[k2]
            self._step_compare(C, r, self.th_kind[r] == 1)
        k3 = k == 3
        if k3.any():
            self._step_vr(C, ix[k3])
        self.th_done[ix[k >= self.c.think_k]] = True
        e = self._confidence(C, ix)
        self.th_e[ix] = e
        return e

    def _close(self, ix):
        self.th_on[ix] = False
        self.REL[ix] = False

    @staticmethod
    def _pr_p(w, keys, bias):
        """SparsePR.p: clip(b + sum(w[k])); sum() exactly as CPython 3.12 adds floats (Neumaier compensation; the
        reference ran on 3.12, whose sum() differs from 3.11's plain left-to-right sum in the last bit)."""
        if not len(keys):
            return float(np.clip(bias + 0, 0.0, 1.0))
        f = 0 + float(w[keys[0]])
        c = 0.0
        for k in keys[1:]:
            x = float(w[k])
            t = f + x
            if abs(f) >= abs(x):
                c += (f - t) + x
            else:
                c += (x - t) + f
            f = t
        if c and np.isfinite(c):
            f += c
        return float(np.clip(bias + f, 0.0, 1.0))

    def _pr_learn(self, w, used, keys, y, p, eta):
        if not len(keys):
            return
        d = eta * 1.0 * (y - p) / len(keys)
        if d == 0:
            return
        for k in keys:
            w[k] = w[k] + d
            used[k] = True

    def _oth_keys(self, b, tau):
        tc = min(int(tau), 30)
        f = self.th_feats[b, :self.th_nf[b]]
        return [tc] + [int(x) * 31 + tc for x in f]

    def _intend(self, own, act):
        c = self.c
        busy = self.SPK | (self.Hp != -1) | ((self.bGO | self.bR1 | self.bCD.any(1)) & ~(self.th_on & self.th_rel))
        cand = self.th_on & ~busy & ~self.in_line & act
        tau = self.t - self.th_t0
        tm = cand & (tau > c.T_MAX)
        if tm.any():
            self._close(np.flatnonzero(tm))
        cand &= ~tm
        rl = cand & self.th_rel
        st = rl & (tau > 0)
        if st.any():
            sx = np.flatnonzero(st & ~self.th_done)
            self._think_step(sx)
        self.REL[rl] = True
        fr = np.flatnonzero(cand & ~self.th_rel)
        if not len(fr):
            return
        need = ~self.th_done[fr] & (tau[fr] > 0)
        e = self.th_e[fr].copy()
        if need.any():
            e[need] = self._think_step(fr[need])
        for q, b in enumerate(fr):
            kind = self.th_kind[b]
            u = c.u[kind]
            if kind == 0 and self.R_line[b]:
                u *= c.u_R1
            self.th_I[b] += self.th_G[b] * e[q] + u
            keys = self._oth_keys(b, tau[b])
            prior = 0.0 if (kind == 0 and tau[b] <= c.tau_w) else 1.0
            p_o = self._pr_p(self.oth_w[b], keys, prior)
            if self.learning[b]:
                n = self.pend_len[b]
                self.pend_t[b, n] = self.t[b]; self.pend_k[b, n] = -1; self.pend_k[b, n, :len(keys)] = keys
                self.pend_n[b, n] = len(keys); self.pend_p[b, n] = p_o; self.pend_b[b, n] = prior
                self.pend_len[b] = n + 1
            late = kind == 0 and tau[b] > c.tau_w
            if late and not c.late_release:
                continue
            if self.th_I[b] >= self.prm['theta_I'][b]:
                if p_o < (self.prm['nogo'][b] if prior == 0.0 else c.nogo_late):
                    self.REL[b] = True
            if self.REL[b] and kind == 1 and self.t[b] - self.th_t0[b] < c.init_min_tau:
                self.REL[b] = False

    def _pend_finalize(self, act):
        c = self.c
        for b in np.flatnonzero((self.pend_len > 0) & act):
            if self.t[b] - self.pend_t[b, 0] <= c.L:
                continue
            keep = []
            for q in range(self.pend_len[b]):
                if self.t[b] - self.pend_t[b, q] > c.L:
                    if self.learning[b]:
                        self._pr_learn(self.oth_w[b], self.oth_used[b], self.pend_k[b, q, :self.pend_n[b, q]], 0.0,
                                       self.pend_p[b, q], self.prm['eta_oth'][b])
                else:
                    keep.append(q)
            self._pend_keep(b, keep)

    def _pend_keep(self, b, keep):
        for a in ('pend_t', 'pend_k', 'pend_n', 'pend_p', 'pend_b'):
            arr = getattr(self, a)
            arr[b, :len(keep)] = arr[b, keep]
        self.pend_len[b] = len(keep)

    def _other_onset(self, ix):
        c = self.c
        for b in ix:
            if self.learning[b]:
                for q in range(self.pend_len[b]):
                    if self.t[b] - self.pend_t[b, q] <= c.L:
                        self._pr_learn(self.oth_w[b], self.oth_used[b], self.pend_k[b, q, :self.pend_n[b, q]], 1.0,
                                       self.pend_p[b, q], self.prm['eta_oth'][b])
            self.pend_len[b] = 0
        self._close(ix[self.th_on[ix]])
        self.R_line[ix] = self.spoke_since[ix]
        self.spoke_since[ix] = False

    def _reaction(self, ix, r):
        c = self.c
        for b in ix:
            if not self.elig_on[b]:
                continue
            self.elig_on[b] = False
            if self.t[b] - self.elig_t[b] > 50:
                continue
            if self.learning[b]:
                keys = self.elig_k[b, :self.elig_n[b]]
                y = float(r[b] > 0)
                p = self._pr_p(self.go_w[b], keys, 0.5)
                self._pr_learn(self.go_w[b], self.go_used[b], keys, y, p, self.prm['eta_go'][b])
                self.rbar_n[b] += 1
                self.rbar[b] += (y - self.rbar[b]) / min(self.rbar_n[b], 200)

    # ================================================================ eye
    def _parse_eye(self, eye, E, own, act):
        c = self.c
        e = eye
        self.base_changed[act] = False
        anye = e.any(1) & act
        no = ~e.any(1) & act
        self.has_base[no] = False
        self.att_now[act] = -1
        if c.life_eye:
            self.prev_eye[no] = e[no]; self.has_prev_eye[no] = True
        nb = anye & ~self.has_base
        self.base[nb] = e[nb]; self.has_base[nb] = True
        extra = e & ~self.base
        missing = self.base & ~e
        chg = anye & (missing.sum(1) > 2)
        self.base[chg] = e[chg]; extra[chg] = False; self.base_changed[chg] = True
        if c.life_eye:
            le = anye & ~chg & (E == 0) & ~self.in_line & ~own
            pe = np.where(self.has_prev_eye[:, None], self.prev_eye, False)
            nbits = np.where(self.has_prev_eye[:, None], extra & ~pe, extra)
            nw = le & (nbits.sum(1) > 2)
            self.base[nw] |= nbits[nw]; extra[nw] &= ~nbits[nw]; self.base_changed[nw] = True
            self.prev_eye[anye] = e[anye]; self.has_prev_eye[anye] = True
        S = c.seg
        for b in np.flatnonzero(anye & (extra.sum(1) >= 4)):
            ex = extra[b]
            best, bj = 0.0, -1
            for j in range(self.n_codes[b]):
                cc = self.att_codes[b, j]
                jac = (cc & ex).sum() / max((cc | ex).sum(), 1)
                if jac > best:
                    best, bj = jac, j
            att = -1
            if best >= 0.6:
                att = bj
            elif self.n_codes[b] < 3:
                self.att_codes[b, self.n_codes[b]] = ex; att = int(self.n_codes[b]); self.n_codes[b] += 1
            if att >= 0:
                ex2 = e[b] & ~self.base[b]
                cnt = [int(ex2[s * S:(s + 1) * S].sum()) for s in range(3)]
                att = int(np.argmax(cnt)) if max(cnt) >= 4 else att
            self.att_now[b] = att

    # ================================================================ body (ThinkBody, go_mode gate, brake_now)
    def _body(self, E, act):
        c = self.c
        B = self.B
        ncd = self.bCD.sum(1)
        F_new = (c.f_w * self.bE - 10.0 * ncd) >= c.f_thr
        F2_new = self.bF.copy()
        free = ~self.bF & ~self.SPK & (ncd == 0) & ~self.bGO & ~self.bR1 & (self.Hp == -1)
        GO_new = self.REL & free
        CD_new = np.concatenate([(self.bGO | self.SPK)[:, None], self.bCD[:, :-1]], 1)
        R1_new = self.bGO.copy()
        R2_new = self.bGO | (self.SPK & (self.phase == 1))
        bnd = (self.bR1 & ~self.bF) | (self.SPK & (self.phase == 2))
        bnd &= ~(~self.SPK & (c.f_w * E >= c.f_thr))
        bnd &= act
        first = bnd & ~self.SPK
        self.bstop[:] = 0
        Y_new = np.full(B, -1, np.int64)
        bx = np.flatnonzero(bnd)
        if len(bx):
            self.nsyl[bx[first[bx]]] = 0
            sc_ix = bx[self.n_prog[bx] > 0]
            prog_sc = {}
            if len(sc_ix):
                prog_sc = self._scorer(sc_ix, first[sc_ix])
            for b in bx:
                fb = first[b]
                if b in prog_sc:
                    progsc, s_end = prog_sc[b]
                    n = self.n_prog[b]
                    pc = self.prog2cat[b, :n]
                    sc = np.where(pc >= 0, progsc[np.maximum(pc, 0)].astype(np.float64), -1.0)
                else:
                    sc, s_end = np.zeros(0), 0.0
                best = int(np.argmax(sc)) if len(sc) else -1
                s_best = float(sc[best]) if best >= 0 else 0.0
                comp = 1.0 * s_best
                if not fb and (s_end > 0 and s_end >= comp):
                    self.bstop[b] = 1
                elif not fb and self.nsyl[b] >= c.breath:
                    self.bstop[b] = 2
                elif best < 0 or s_best < c.floor:
                    self.bstop[b] = 4 if fb else 3
                else:
                    Y_new[b] = best; self.Hp[b] = best; self.phase[b] = 0; self.nsyl[b] += 1
                if self.bstop[b]:
                    self.Hp[b] = -1; self.phase[b] = 0
        sp = ~bnd & self.SPK & act
        self.phase[sp] += 1
        SPK_new = self.Hp != -1
        force = np.zeros((B, 32), np.uint8)
        hp = np.flatnonzero(self.Hp >= 0)
        if len(hp):
            force[hp] = self.progs[hp, self.Hp[hp]] * (self.phase[hp] + 1)[:, None]
        w = lambda new, old: np.where(act if new.ndim == 1 else act[:, None], new, old)
        self.bF, self.bF2, self.bGO, self.bR1 = w(F_new, self.bF), w(F2_new, self.bF2), w(GO_new, self.bGO), w(R1_new, self.bR1)
        self.bCD, self.bR2 = w(CD_new, self.bCD), w(R2_new, self.bR2)
        self.bY = Y_new
        self.SPK = w(SPK_new, self.SPK)
        self.bE = w(E, self.bE)
        return force

    def _scorer(self, ix, first):
        """the body asks for program scores at a boundary: own speech context (built from the thought at the
        first boundary) -> GEN + copy routes (MIX units are recruited, as in own_scores)."""
        fx = ix[first | ~self.own_on[ix]]
        if len(fx):
            nt = fx[~self.th_on[fx]]
            if len(nt):                                         # no open thought: a fresh reply thought (no steps)
                sc = self.prev_mode[nt] != 2
                self._view(nt[sc])
                use = self.prev_mode[nt] == 1
                self._ctx_new(self.TMP, nt, np.zeros(len(nt), np.int64), self.base[nt], self.has_base[nt],
                              self.att_now[nt], use)
                self._use_bag(self.TMP, nt, use)
            self.act_n[fx] = 0
            for b in fx:
                src = self.TH if self.th_on[b] else self.TMP
                self.OWN.copy_rows(src, np.array([b]))
                self.OWN.reset_rows(np.array([b]))
                self.OWN.att_tutor[b] = src.att_tutor[b]
            self.own_on[fx] = True
            th = fx[self.th_on[fx]]
            self.th_fb[th] = True
        A, rows, nAt, total, gd, cand, pos, _, _ = self._drives(self.OWN, ix, True)
        self._own_last = (ix, A, nAt, cand, pos)
        tot = total.cpu().numpy()
        return {b: (tot[r, :self.V], float(tot[r, self.V])) for r, b in enumerate(ix)}

    def _store_acts(self, bs, items):
        """actor trace: append (A of this boundary, chosen item) for agents bs -> their rows in _own_last."""
        ix, A, nAt, cand, pos = self._own_last
        pos_of = {int(b): r for r, b in enumerate(ix)}
        rs = np.array([pos_of[int(b)] for b in bs], np.int64)
        if self.act_A is None or self.act_A.shape[2] < A.shape[1]:
            old = self.act_A
            self.act_A = torch.full((self.B, 16, max(A.shape[1], 512)), -1, dtype=torch.int64, device=self.dev)
            if old is not None:
                self.act_A[:, :, :old.shape[2]] = old
        k = self.act_n[bs]
        tb, tk = self._t(bs), self._t(k)
        self.act_A[tb, tk] = -1
        self.act_A[tb, tk, :A.shape[1]] = A[self._t(rs)]
        self.act_item[bs, k] = items
        self.act_n[bs] = k + 1
        return rs

    def _store_act(self, b, item):
        ix, A, nAt, cand, pos = self._own_last
        r = int(np.flatnonzero(ix == b)[0])
        if self.act_A is None or self.act_A.shape[2] < A.shape[1]:
            old = self.act_A
            self.act_A = torch.full((self.B, 16, max(A.shape[1], 512)), -1, dtype=torch.int64, device=self.dev)
            if old is not None:
                self.act_A[:, :, :old.shape[2]] = old
        k = self.act_n[b]
        self.act_A[b, k] = -1
        self.act_A[b, k, :A.shape[1]] = A[r]
        self.act_item[b, k] = item
        self.act_n[b] = k + 1
        return r

    def _reward(self, ix, rew):
        c = self.c
        for b in ix:
            n = self.act_n[b]
            if n == 0:
                continue
            A0 = self.act_A[b, 0]
            A0 = A0[A0 >= 0]
            v = self.CRIT[b * c.U + A0]
            V0 = float(pairwise_sum_f32(v.cpu().numpy())) if len(A0) else 0.0
            delta = float(rew[b]) - V0
            if len(A0):
                self.CRIT[b * c.U + A0] = v + torch.tensor(F32(self.prm['eta_critic'][b] * delta / len(A0)),
                                                           device=self.dev)
            if delta != 0:
                for q in range(n):
                    A = self.act_A[b, q]
                    A = A[A >= 0]
                    if len(A):
                        y = int(self.act_item[b, q])
                        idx = b * c.U + A
                        self.W[idx, y] = self.W[idx, y] + torch.tensor(F32(self.prm['eta_actor'][b] * delta / len(A)),
                                                                       device=self.dev)
            self.act_n[b] = 0

    # ================================================================ awake recall (R10)
    def _awake_recall(self, ix):
        c = self.c
        picks = {}
        sigs = self.hsig[self._t(ix), :int(self.hn[ix].max())].cpu().numpy() if len(ix) else None
        for q, b in enumerate(ix):
            n = self.hn[b]
            if n == 0:
                continue
            nx = self.next_of[b, :n]
            ok = nx >= 0
            ok[ok] = self.hborn[b, nx[ok]] == self.next_born[b, :n][ok]
            sig = sigs[q, :n]
            ok &= sig > F32(0.05)
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            k = min(c.recall_n, len(idx))
            picks[b] = choice_np(self.rng[b], idx, k)
        if not picks:
            return
        if self.trace is not None:
            for b, p in picks.items():
                self.trace['picks'].append((int(self.t[b]), [int(x) for x in p]))
        P = self.PREV
        saved = (self.prev_mode.copy(), {k: getattr(P, k).copy() for k in ('valid', 'items', 'n', 'has_eye', 'eye', 'att',
                                                                        'ep', 'born')},
                 self.prev_bag.copy(), self.prev_nbag.copy())
        sv_code = P.code.clone()
        bs = np.array(sorted(picks))
        for s in range(c.recall_n):
            rows = np.array([b for b in bs if len(picks[b]) > s], np.int64)
            if not len(rows):
                break
            e = np.array([picks[b][s] for b in rows], np.int64)
            self._replay(rows, e)
            self.stats['awake_recalls'] += len(rows)
        self.prev_mode[:] = saved[0]
        for k, v in saved[1].items():
            setattr(self.PREV, k, v)
        self.PREV.code = sv_code
        self.prev_bag[:], self.prev_nbag[:] = saved[2], saved[3]

    def _replay(self, rows, e):
        """reinstate episode e as the prompt and learn its successor line as if heard again (eta_recall)."""
        c, C = self.c, self.RP
        nxt = self.hnext[rows, e]
        n = len(rows)
        P = self.PREV
        P.items[rows] = self.hitems[rows, e]; P.n[rows] = self.hni[rows, e]
        P.ep[rows] = e
        self.prev_mode[rows] = 3                                  # a stored record (no bag)
        self._ctx_new(C, rows, np.zeros(n, np.int64), self.heye[rows, nxt], self.hhas[rows, nxt],
                      self.hatt[rows, nxt], np.ones(n, bool))
        code = torch.zeros(n, self.NDG, dtype=torch.bool, device=self.dev)
        kk = self.hkeys[self._t(rows), self._t(e)].long()
        code.scatter_(1, kk, True)
        visited = np.full((n, 4), -1, np.int64)
        visited[:, 0] = e
        self._hascode = np.ones(n, bool)
        self._think_complete(C, rows, code, visited, np.zeros(n, bool))
        ln = self.hni[rows, nxt]
        for q in range(int(ln.max()) + 1):
            act = ln >= q
            rx = rows[act]
            tgt = np.where(ln[act] > q, self.hitems[rx, nxt[act], np.minimum(q, c.LM - 1)].astype(np.int64), self.END)
            cand, pos = self._learn_step(C, rx, tgt, eta=c.eta_recall)
            it = ln[act] > q
            if it.any():
                self._advance(C, rx[it], tgt[it], cand[it], pos[it])

    # ================================================================ one tick
    def own_voice(self):
        return self.SPK | (self.mouth_ago <= 2)

    def step(self, win, eye, rew, active=None):
        """win (B, 8, 6) ints, eye (B, NE) bool, rew (B,) floats -> force (B, 32) uint8.
        active (B,) bool: agents that live this tick; the others are frozen (nothing of theirs changes)."""
        c = self.c
        act = np.ones(self.B, bool) if active is None else np.asarray(active, bool)
        E = np.asarray(win).reshape(self.B, -1).sum(1).astype(np.float64)
        rew = np.where(act, rew, 0.0)
        had_scene = self.has_base.copy()
        own = self.own_voice()
        self._parse_eye(np.asarray(eye, bool), E, own, act)
        self._pend_finalize(act)
        lis = ~own & act
        onset = lis & (E > 0) & (self.E_prev == 0) & ~self.in_line
        ox = np.flatnonzero(onset)
        if len(ox):
            self._reaction(ox, rew)
            self._other_onset(ox)
            sr = ox[self.sr_on[ox] & (self.t[ox] - self.sr_t[ox] <= 10)]
            self.prev_mode[sr] = 2
            self.sr_on[ox] = False
            self.in_line[ox] = True; self.clk[ox] = 0
            self._begin_line(ox)
        cont = lis & ~onset & self.in_line & (E > 0)
        self.clk[cont] += 1
        samp = lis & self.in_line & (E > 0) & (self.clk % 3 == 2)
        sx = np.flatnonzero(samp)
        if len(sx):
            cat = self._detect(sx, np.asarray(win)[sx])
            ok = cat >= 0
            if ok.any():
                kx = sx[ok]
                self.stats['items'] += len(kx)
                upd = kx[(self.ST.att_tutor[kx] < 0) & (self.att_now[kx] >= 0)]
                self.ST.att_tutor[upd] = self.att_now[upd]
                if self.trace is not None:
                    for b, cc in zip(sx, cat):
                        self.trace['dets'].append((int(self.t[b]), int(cc)))
                self._line_item(kx, cat[ok])
            if self.trace is not None:
                for b, cc in zip(sx[~ok], cat[~ok]):
                    self.trace['dets'].append((int(self.t[b]), -1))
        le = lis & self.in_line & (E == 0)
        lx = np.flatnonzero(le)
        if len(lx):
            self.in_line[lx] = False
            self._line_end(lx)
            self._open(lx, 0)
        oi = np.flatnonzero(own & self.in_line & act)
        if len(oi):
            self.in_line[oi] = False
            self._line_end(oi)
        cl = self.th_on & ((had_scene & ~self.has_base) | self.base_changed) & ~self.th_rel & act
        self._close(np.flatnonzero(cl))
        new_scene = self.has_base & (~had_scene | self.base_changed)
        new_point = (self.att_now >= 0) & (self.prev_att < 0)
        so = (new_scene | new_point) & ~self.th_on & ~self.in_line & ~own & (E == 0) & act
        self._open(np.flatnonzero(so), 1)
        self.prev_att[act] = self.att_now[act]
        self.REL[:] = False
        self._intend(own, act)
        force = self._body(E, act)
        gr = self.bGO & self.th_on & ~self.th_rel & act
        self.th_rel[gr] = True
        self.stats['released'] += int(gr.sum())
        yx = np.flatnonzero((self.bY >= 0) & self.own_on & act)
        if len(yx):
            cc = self.prog2cat[yx, self.bY[yx]]
            ok = cc >= 0
            if ok.any():
                bs, cs = yx[ok], cc[ok]
                rs = self._store_acts(bs, cs)
                ix_, A_, nAt_, cand_, pos_ = self._own_last
                self._advance(self.OWN, bs, cs, cand_[rs], pos_[rs])
                self.stats['own'] += len(bs)
            self.th_spoke[yx[self.th_on[yx]]] = True
        stx = np.flatnonzero((self.bstop > 0) & act)
        ex = stx[self.own_on[stx] & (self.bstop[stx] == 1)]
        if len(ex):
            self._store_acts(ex, np.full(len(ex), self.END))
        for b in stx:
            spoke = self.own_on[b] and self.OWN.ni[b] > 0
            if spoke:
                self.spoke_since[b] = True
                if self.th_on[b]:
                    self.elig_on[b] = True; self.elig_t[b] = self.t[b]
                    self.elig_k[b] = -1; self.elig_k[b, :self.th_nf[b]] = self.th_feats[b, :self.th_nf[b]]
                    self.elig_n[b] = self.th_nf[b]
                    if self.th_kind[b] == 1 and self.TH.has_eye[b]:
                        self.sr_on[b] = True; self.sr_t[b] = self.t[b]; self.sr_eye[b] = self.TH.eye[b]
                        self.sr_att[b] = self.TH.att[b]
            self.own_on[b] = False
            if self.th_on[b]:
                self._close(np.array([b]))
        rx = np.flatnonzero((rew != 0) & self.learning & act)
        if len(rx):
            self._reward(rx, rew)
        self.mouth_ago = np.where(act, np.where(force.any(1), 0, self.mouth_ago + 1), self.mouth_ago)
        self.E_prev = np.where(act, E, self.E_prev)
        self.t += act
        own2 = self.own_voice()
        q = (E == 0) & ~self.in_line & ~own2
        self.quiet = np.where(act, np.where(q, self.quiet + 1, 0), self.quiet)
        qx = np.flatnonzero((self.quiet == c.quiet_ticks) & self.learning & self.learn_listen & act)
        if len(qx) and c.recall_n > 0:
            self.stats['quiet'] += len(qx)
            self._awake_recall(qx)
        if c.wander and self.T_W is not None:
            wx = np.flatnonzero((self.quiet > c.wander_after) & (self.quiet % c.wander == 0) & act)
            self._tick_thought(wx)
        return force

    # ================================================================ misc
    def n_units(self):
        return self.conj.n.cpu().numpy()

    def clone(self):
        tr, self.trace = self.trace, None
        b = copy.deepcopy(self)
        self.trace = tr
        return b


CONST = {'_last_vals', 'nf', 'trace', '_own_last', 'c', 'dev', 'lv', 'zrows', 'goff', 'G', 'V', 'END', 'NE', 'NF', 'NDG', 'KDG', 'NFEAT',
         'ear_k', 'B', 'SINK', '_kdg_t', '_three_t', '_1e4', 'debug_t', 'stats', 'own_A', '_hascode'}


def _row_items(br):
    """(name, kind, value) of every per-agent state item of a brain: kind 'a' = leading agent axis, 'u' = flattened
    (B*U [+ sink], ...) readouts, 'o' = an LCtx / RecSlot object, 'h' = the conjunction hash table, 'l' = list."""
    for k, v in br.__dict__.items():
        if k in CONST or v is None:
            continue
        if k in ('W', 'CRIT'):
            yield k, 'u', v
        elif k == 'conj':
            yield k, 'h', v.table
        elif k == 'hidx':
            for g, x in v.items():
                yield ('hidx', g), 'a', x
        elif k == 'prm':
            for g, x in v.items():
                yield ('prm', g), 'a', x
        elif isinstance(v, (LCtx, RecSlot)):
            yield k, 'o', v
        elif k in ('rng', 'bab_sets', 'rec_cache'):
            yield k, 'l', v
        elif isinstance(v, np.ndarray) or torch.is_tensor(v):
            assert v.shape[0] == br.B, k
            yield k, 'a', v


def copy_row(br, src, dst):
    """agent dst := an exact copy of agent src (a probe copy = copy.deepcopy of one numpy agent)."""
    U = br.c.U
    for name, kind, v in _row_items(br):
        if kind == 'a':
            v[dst] = v[src]
        elif kind == 'u':
            v[dst * U:(dst + 1) * U] = v[src * U:(src + 1) * U]
        elif kind == 'o':
            for kk, vv in v.__dict__.items():
                vv[dst] = vv[src]
        elif kind == 'h':
            M = v.M
            v.key_f[dst * M:(dst + 1) * M] = v.key_f[src * M:(src + 1) * M]
            v.uid_f[dst * M:(dst + 1) * M] = v.uid_f[src * M:(src + 1) * M]
            v.n[dst] = v.n[src]
        elif name == 'rng':
            v[dst] = copy.deepcopy(v[src])
        elif name == 'bab_sets':
            v[dst] = v[src]
        elif name == 'rec_cache':
            v[dst] = dict(v[src])


def tile(src: Brain, B: int) -> Brain:
    """a B-agent brain whose every agent is a copy of src's agent 0 (speed tests, populations)."""
    assert src.B == 1
    U = src.c.U
    dst = copy.copy(src)
    dst.B, dst.SINK = B, B * U
    dst.stats = dict(src.stats)
    rp = lambda v: np.repeat(v, B, 0) if isinstance(v, np.ndarray) else v.repeat(B, *([1] * (v.dim() - 1)))
    for name, kind, v in _row_items(src):
        if kind == 'a':
            nv = rp(v)
        elif kind == 'u':
            nv = rp(v[:U])
            if name == 'W':
                nv = torch.cat([nv, torch.zeros_like(v[:1])], 0)
        elif kind == 'o':
            nv = copy.copy(v)
            for kk, vv in v.__dict__.items():
                setattr(nv, kk, rp(vv))
        elif kind == 'h':
            tb = v.clone()
            M = v.M
            tb.B = B
            tb.key_f = torch.cat([v.key_f[:M].repeat(B), v.key_f[-1:]])
            tb.uid_f = torch.cat([v.uid_f[:M].repeat(B), v.uid_f[-1:]])
            tb.n = v.n.repeat(B)
            cj = copy.copy(src.conj); cj.table = tb; cj.B = B
            dst.conj = cj
            continue
        elif name == 'rng':
            nv = [copy.deepcopy(v[0]) for _ in range(B)]
        elif name == 'rec_cache':
            nv = [dict(v[0]) for _ in range(B)]
        else:
            nv = [v[0]] * B
        if isinstance(name, tuple):
            d = dict(getattr(dst, name[0])) if getattr(dst, name[0]) is getattr(src, name[0]) else getattr(dst, name[0])
            d[name[1]] = nv
            setattr(dst, name[0], d)
        else:
            setattr(dst, name, nv)
    dst.trace = None
    return dst
