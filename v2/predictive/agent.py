# -*- coding: utf-8 -*-
"""ARCH_v2-P prototype agent: one general predictive learner on the kept body.

Every learned synapse uses the one rule PR (pr.py).  Areas differ only in wiring, sizes, timescales and plasticity.

  ear window -> EAR0/FEAT0 (fixed) -> Z: perceptual syllable units (localist, recruited when nothing explains a heard
      syllable; competitive PR: each unit predicts its own firing from FEAT0; own voice trains it with the efference
      copy as the target = the inverse model)
  perceptual clock (innate theta-like phase reset at speech onset): one char event per 3 ticks for other speakers;
      own syllables are events at their release (efference copy)
  context (presynaptic traces, all binary codes):
      x1, x2   the last two chars of the current line (phonological buffer)      start  = line-initial position
      Gline    union of fixed random codes of the line's unigrams + bigrams (PFC line layer, persistent over the line)
      Ghold    Gline of the last COMPLETED line of another speaker (BG-gated working memory: loads at the line end,
               holds during replies -- own, overheard or recast)                  B = its chars (localist bag)
      Gtopic   slow leaky mix of the Ghold loads (dialogue layer)
      V        the eye code                                                         C = conjunctions of Ghold x V
      H        hippocampal DG code of (Ghold, V, Gline, x1): sparse expansion, fast plasticity (pi_H)
  ONE predictor ZP over {END} + Z units, trained at every event of another speaker with target = what is heard
      (learning by listening, overheard replies included); own events: M = 0 (or the reward prediction error)
  speaking = saying the prediction: at a body boundary, program scores = ZP of its Z unit, END score = ZP[END]
  babbling: random muscle sets on a spontaneous beat; a set is kept as the program of the Z unit its own voice
      activates (no teacher, no cue)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

import os
HERE = os.environ.get('PRED_HERE', __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, HERE)
from pr import Predictor, F32  # noqa: E402
from frontend import Ear0  # noqa: E402
from body import Body  # noqa: E402


@dataclass
class Cfg:
    seed: int = 0
    n_eye: int = 1024
    zmax: int = 256            # capacity of the perceptual syllable layer (localist, recruited)
    # Z (perceptual categories)
    eta_z: float = 0.2
    z_detect: float = 0.7      # a Z unit "fires" if its prediction p >= this
    z_recruit: float = 0.7     # recruit a fresh unit when no unit reaches this on a sampled syllable
    z_link: float = 0.75       # babbling: my voice must drive a heard unit this well to become its program
    z_margin: float = 0.2      # ... and beat every other unit by this much
    z_min_level: int = 5       # a sampled window is speech-like if some cell >= this level (innate)
    z_confuse: float = 0.45    # rivals whose prediction exceeds this on the pattern are pushed down (competition)
    # predictor
    eta: float = 0.5
    pi_h: float = 8.0          # hippocampal plasticity factor (fast path); cortex = 1
    pi_conj: float = 1.0       # plasticity factor of the conjunction paths V2, C2
    release: float = 1.0       # synaptic release probability at learning events (1 = deterministic; <1 = stochastic)
    floor: float = 0.3         # body floor: say nothing if the best prediction is below it
    # PFC / codes
    NG: int = 2048
    m_code: int = 8
    topic_leak: float = 0.5
    NV2: int = 65536           # visual conjunctions: AND of 2 random eye units (mixed selectivity, fixed wiring)
    NC2: int = 65536           # PFC conjunctions: AND of (1 random Ghold unit, 1 random eye unit)
    NR: int = 4096             # hippocampal recall index: DG of (Ghold, V) at line ends (k-WTA)
    kR: int = 40
    r_fan: int = 12
    r_thr: int = 3
    eta_r: float = 1.0         # one-shot
    ep_cells: int = 6000       # episode cells (recruited one per completed line of another speaker; oldest freed)
    recall_min: float = 0.3    # an episode is reinstated only if its key matches the cue this well (mean cosine)
    ep_max: int = 6000         # hippocampal event store for replay (bounded; oldest dropped)
    ep_surprise: float = 0.3   # store an event only if it was not predicted (1 - p(target) >= this): novelty gate
    sleep_decay: float = 0.1   # global downscaling of the fast (hippocampal) paths per sleep
    eta_gate: float = 0.3      # copy gates (one PR unit per buffer): 'the next heard char is in this buffer'
    kappa: float = 1.0         # strength of the gated one-to-one mirror path buffer_c -> ZP_c (fixed)
    go_win: int = 4            # Go/NoGo: does another speaker start within this many ticks after a line ends?
    go_open: int = 30          # ... or others stay silent this long (the floor is left to me) -> Go target 1
    go_thr: float = 0.5
    eta_go: float = 0.3
    NH: int = 4096
    kH: int = 32
    h_fan: int = 12
    h_thr: int = 3
    # switches (ablations)
    use: dict = field(default_factory=lambda: dict(x1=True, x2=True, start=True, gline=True, ghold=True, gtopic=True,
                                                   B=True, V=True, V2=True, C2=True, H=True, H2=True, go=True, Brec=True, Bnext=True, gate=True, S=True))
    learn: bool = True
    learn_z: bool = True
    use_fwd: bool = True       # babble memory (forward model, lookup form)
    own_M: str = 'zero'        # 'zero' | 'rpe'


def _code(rng_seed, n, m):
    return np.random.default_rng(rng_seed).choice(n, m, replace=False)


class PredAgent:
    privileged = False

    def __init__(self, cfg: Cfg | None = None):
        self.c = c = cfg or Cfg()
        self.ear = Ear0(seed=c.seed)
        self.body = Body(theta=c.floor)
        self.rng = np.random.default_rng(c.seed + 99)
        self.NF = None
        # ---------------- Z perceptual units (created after FEAT0 is known)
        self.Z = None
        self.nz = 0
        self.z_prog = [None] * c.zmax          # program (muscle tuple) of each Z unit
        self.z_heard = np.zeros(c.zmax, int)
        self.z_label = [None] * c.zmax         # diagnostics only (never read by the agent): what the world said
        # ---------------- predictor ZP: index 0 = END, 1 + z = char z
        nT = 1 + c.zmax
        P = self.P = Predictor(nT, eta=c.eta, theta=1.0, bias=True, name='ZP')
        u = c.use
        if u['x1']: P.add('x1', c.zmax)
        if u['x2']: P.add('x2', c.zmax)
        if u['start']: P.add('start', 1)
        if u['gline']: P.add('gline', c.NG)
        if u['ghold']: P.add('ghold', c.NG)
        if u['gtopic']: P.add('gtopic', c.NG)
        if u['B']:
            mask = np.zeros((c.zmax, nT), bool); mask[np.arange(c.zmax), 1 + np.arange(c.zmax)] = True
            P.add('B', c.zmax, mask)
        if u['V']: P.add('V', c.n_eye)
        if u['V2']: P.add('V2', c.NV2, pi=c.pi_conj)
        if u['C2']: P.add('C2', c.NC2, pi=c.pi_conj)
        if u['H']: P.add('H', c.NH, pi=c.pi_h)
        if u.get('H2'): P.add('H2', c.NH, pi=c.pi_h)   # DG without the held prompt: re-enacts what was said here
        if u['Brec']:
            mask = np.zeros((c.zmax, nT), bool); mask[np.arange(c.zmax), 1 + np.arange(c.zmax)] = True
            P.add('Brec', c.zmax, mask)
        if u.get('Bnext'):
            mask = np.zeros((c.zmax, nT), bool); mask[np.arange(c.zmax), 1 + np.arange(c.zmax)] = True
            P.add('Bnext', c.zmax, mask)
        if u.get('S'): P.add('S', 6)            # surprise code of the held line (how expected was what I heard)
        # ---------------- fixed random wiring of C and H
        r = np.random.default_rng(c.seed + 7)
        self.R_in = np.c_[r.integers(0, c.NG, (c.NR, c.r_fan - 3)), c.NG + r.integers(0, c.n_eye, (c.NR, 3))]
        self.R_tie = r.random(c.NR).astype(F32) * 1e-3
        # hippocampal episode cells: one recruited per completed line of another speaker (one-shot); key = sparse codes
        self.ep_size = {'R': c.NR, 'g': c.NG, 'e': c.n_eye, 'c2': c.NC2}
        self.ep_len = {'R': c.kR, 'g': 192, 'e': 128, 'c2': 384}
        self.ep_K = {nm: np.full((c.ep_cells, L), self.ep_size[nm], np.int32) for nm, L in self.ep_len.items()}
        self.ep_n = {nm: np.zeros(c.ep_cells, np.float32) for nm in self.ep_len}
        self.ep_born = np.full(c.ep_cells, -1, np.int64)   # recruitment order (-1 = free cell)
        self.ep_self = [None] * c.ep_cells                  # the line's own chars
        self.ep_next = [None] * c.ep_cells                  # the next other-speaker line's chars (filled when it ends)
        self.ep_count = 0
        self.ep_last = -1
        self.pred_log = []              # (stage tag, hit) at every heard char: next-char prediction accuracy
        self.gates = {}
        for gname in ('gB', 'gR', 'gN'):            # copy gates: prompt buffer B, recalled line Brec, recalled next line
            G = Predictor(1, eta=c.eta_gate, theta=1.0, bias=True, name=gname)
            for nm, n_src in (('x1', c.zmax), ('start', 1), ('gline', c.NG), ('ghold', c.NG)):
                if u.get(nm):
                    G.add(nm, n_src)
            self.gates[gname] = G
        self.p_gate = {'gB': 0.0, 'gR': 0.0, 'gN': 0.0}
        self.episodes = []              # replay store: (target index, {path: (idx, vals)}) without the H path
        self.V2_in = r.integers(0, c.n_eye, (c.NV2, 2))
        self.C2_inG = r.integers(0, c.NG, c.NC2); self.C2_inV = r.integers(0, c.n_eye, c.NC2)
        # learned Go/NoGo (BG): predicts whether another speaker takes the floor right after a line
        self.GoP = Predictor(1, eta=c.eta_go, theta=1.0, bias=True, name='go')
        for nm in ('ghold', 'B', 'V'):
            if u.get(nm):
                self.GoP.add(nm, {'ghold': c.NG, 'B': c.zmax, 'V': c.n_eye}[nm])
        n_ctx = 3 * c.NG + c.n_eye + 1
        self.H_in = r.integers(0, n_ctx, (c.NH, c.h_fan))
        self.H_tie = r.random(c.NH).astype(F32) * 1e-3
        self.H2_in = r.integers(0, 2 * c.NG + c.n_eye + 1, (c.NH, c.h_fan))
        self.H2s = np.zeros(c.NH, bool)
        self._ucode, self._bcode = {}, {}
        self.lesion = set()
        self.reset_state()
        self.t = 0
        self.babbling = False
        self.bab_plan = []
        self.bab_bout = (1, 4)
        self.stats = dict(events_other=0, events_own=0, recruits=0, links=0, learn_steps=0)
        self.log_events = None

    # ------------------------------------------------------------------ state
    def reset_state(self):
        c = self.c
        self.E_prev = 0.0
        self.clock = -1                 # perceptual clock phase (-1 = no speech running)
        self.line_by = None             # 'other' | 'self' | None
        self.x1 = -1; self.x2 = -1; self.start = True
        self.line_chars = []
        self.gline = np.zeros(c.NG, bool)
        self.ghold = np.zeros(c.NG, bool)
        self.gtopic_v = np.zeros(c.NG, F32)
        self.gtopic = np.zeros(c.NG, bool)
        self.B = np.zeros(c.zmax, bool)
        self.eye = np.zeros(c.n_eye, bool)
        self.V2s = np.zeros(c.NV2, bool)
        self.C2s = np.zeros(c.NC2, bool)
        self.Brec = np.zeros(c.zmax, F32)
        self.Bnext = np.zeros(c.zmax, F32)
        self.recall_sim = 0.0
        self.line_surp = []             # prediction errors of the current line's chars (after the first)
        self.Shold = np.zeros(6, F32)   # thermometer of the held line's mean surprise
        self.R_prev_key = None          # recall-index code of the context before the current line
        self.go_pending = None          # (deadline tick, context) after another speaker's line end
        self.p_go = 1.0
        self.Hs = np.zeros(c.NH, bool)
        self.p_next = np.zeros(1 + c.zmax, F32)
        self.own_pending = []           # own syllables waiting for their full-window tick: (tick, prog tuple, z or -1)
        self.last_rel = None

    def attach(self, warm_windows):
        """birth calibration (innate): FEAT0 from a warm-up of random sounds."""
        self.NF = self.ear.warmup(warm_windows)
        self.Z = Predictor(self.c.zmax, eta=self.c.eta_z, theta=1.0, bias=False, name='Z')
        self.Z.add('f', self.NF)
        return self.NF

    # ------------------------------------------------------------------ codes
    def ucode(self, z):
        v = self._ucode.get(z)
        if v is None:
            v = self._ucode[z] = _code([self.c.seed, 1, z], self.c.NG, self.c.m_code)
        return v

    def bcode(self, a, b):
        v = self._bcode.get((a, b))
        if v is None:
            v = self._bcode[(a, b)] = _code([self.c.seed, 2, a, b], self.c.NG, self.c.m_code)
        return v

    @staticmethod
    def _norm(x):
        x = x.astype(F32)
        n = float(np.sqrt((x * x).sum()))
        return x / n if n > 0 else x

    # ------------------------------------------------------------------ context -> prediction
    def context(self):
        xs = self._context()
        if self.lesion:
            for nm in list(xs):
                if nm in self.lesion:
                    del xs[nm]
        return xs

    def _context(self):
        c, u = self.c, self.c.use
        xs = {}
        z1 = np.zeros(c.zmax, F32)
        if self.x1 >= 0: z1[self.x1] = 1
        if u['x1']: xs['x1'] = z1
        if u['x2']:
            z2 = np.zeros(c.zmax, F32)
            if self.x2 >= 0: z2[self.x2] = 1
            xs['x2'] = z2
        if u['start']: xs['start'] = np.array([1.0 if self.start else 0.0], F32)
        if u['gline']: xs['gline'] = self._norm(self.gline)
        if u['ghold']: xs['ghold'] = self._norm(self.ghold)
        if u['gtopic']: xs['gtopic'] = self._norm(self.gtopic)
        if u['B']: xs['B'] = self._norm(self.B)
        if u.get('Brec'): xs['Brec'] = self._norm(self.Brec)
        if u.get('Bnext'): xs['Bnext'] = self._norm(self.Bnext)
        if u.get('S'): xs['S'] = self._norm(self.Shold)
        if u['V']: xs['V'] = self._norm(self.eye)
        if u['V2']: xs['V2'] = self._norm(self.V2s)
        if u['C2']: xs['C2'] = self._norm(self.C2s)
        if u['H']: xs['H'] = self._norm(self.Hs)
        if u.get('H2'): xs['H2'] = self._norm(self.H2s)
        return xs

    def learn_P(self, y):
        """one PR step of the predictor.  With release < 1 every presynaptic input is transmitted with probability q
        (stochastic vesicle release; expectation kept), and the error is that of the transmitted input."""
        q = self.c.release
        if q >= 1.0 or self.P._xs is None:
            self.P.learn(y, M=1.0)
            return
        xs = {}
        for nm, x in self.P._xs.items():
            m = self.rng.random(x.shape[0]) < q
            xs[nm] = (x * m / q).astype(F32)
        extra = self.P.d - self.P.b - sum(P.drive(self.P._xs[nm]) for nm, P in self.P.paths.items() if nm in self.P._xs)
        p = self.P.predict(xs, extra=extra).copy()
        self.P.learn(y, xs=xs, p=p, M=1.0)

    def remember(self, tgt):
        """hippocampal event store (novelty-gated): the cortical context of an unpredicted event and what happened."""
        if 1.0 - float(self.p_next[tgt]) < self.c.ep_surprise or self.P._xs is None:
            return
        snap = {}
        for nm, x in self.P._xs.items():
            if nm in ('H', 'H2'):
                continue
            idx = np.flatnonzero(x)
            snap[nm] = (idx.astype(np.int32), x[idx].astype(F32))
        self.episodes.append((tgt, snap))
        if len(self.episodes) > self.c.ep_max:
            del self.episodes[:len(self.episodes) - self.c.ep_max]

    def sleep(self, n_replay=2000, rng=None):
        """offline: replay stored events in random (interleaved) order into the slow cortical paths with the
        hippocampal path silent (its prediction is not part of the cortical prediction), then downscale the fast
        hippocampal paths (one global decay).  No world ticks pass."""
        rng = rng or self.rng
        if not self.episodes:
            return 0
        n = 0
        for i in rng.integers(0, len(self.episodes), n_replay):
            tgt, snap = self.episodes[i]
            xs = {}
            for nm, (idx, vals) in snap.items():
                P = self.P.paths.get(nm)
                if P is None:
                    continue
                v = np.zeros(P.W.shape[0], F32); v[idx] = vals; xs[nm] = v
            p = self.P.predict(xs).copy()
            y = np.zeros(self.P.n, F32); y[tgt] = 1
            self.P.learn(y, xs=xs, p=p)
            n += 1
        for nm in ('H', 'H2'):
            if nm in self.P.paths:
                self.P.paths[nm].W *= (1.0 - self.c.sleep_decay)
        self.stats['replayed'] = self.stats.get('replayed', 0) + n
        self.predict()
        return n

    def rkey(self):
        ctx = np.r_[self.ghold, self.eye]
        I = ctx[self.R_in].sum(1)
        idx = np.flatnonzero(I >= self.c.r_thr)
        s = np.zeros(self.c.NR, bool)
        if len(idx) > self.c.kR:
            idx = idx[np.argpartition(-(I[idx] + self.R_tie[idx]), self.c.kR)[:self.c.kR]]
        s[idx] = True
        c2 = self.ghold[self.C2_inG] & self.eye[self.C2_inV]
        return {'R': np.flatnonzero(s), 'g': np.flatnonzero(self.ghold), 'e': np.flatnonzero(self.eye),
                'c2': np.flatnonzero(c2)}

    def recall(self, key):
        """winner-take-all over episode cells: mean cosine of the binary key codes; the newest wins ties."""
        used = self.ep_born >= 0
        if not used.any():
            return -1, 0.0
        sims = np.zeros(self.c.ep_cells, np.float32)
        for nm, cue in key.items():
            if not len(cue):
                continue
            v = np.zeros(self.ep_size[nm] + 1, np.float32); v[cue] = 1.0
            ov = v[self.ep_K[nm]].sum(1)
            sims += ov / np.sqrt(np.maximum(self.ep_n[nm], 1) * len(cue))
        sims = sims / len(key) + 1e-4 * (self.ep_born.astype(np.float32) / max(self.ep_count, 1))
        sims[~used] = -1
        b = int(np.argmax(sims))
        return (b, float(sims[b])) if sims[b] >= self.c.recall_min else (-1, float(sims[b]))

    def store_episode(self, key, chars):
        i = self.ep_count % self.c.ep_cells                 # ring: the oldest cell is freed
        for nm, idx in key.items():
            L = self.ep_len[nm]
            self.ep_K[nm][i] = self.ep_size[nm]
            self.ep_K[nm][i, :min(L, len(idx))] = idx[:L]
            self.ep_n[nm][i] = min(L, len(idx))
        self.ep_born[i] = self.ep_count
        self.ep_self[i] = np.array(chars, int); self.ep_next[i] = None
        self.ep_last = i
        self.ep_count += 1

    def recompute_conj(self):
        if self.c.use['V2']:
            self.V2s = self.eye[self.V2_in[:, 0]] & self.eye[self.V2_in[:, 1]]
        if self.c.use['C2']:
            self.C2s = self.ghold[self.C2_inG] & self.eye[self.C2_inV]

    def recompute_H(self):
        c = self.c
        x1c = np.zeros(c.NG, bool)
        if self.x1 >= 0: x1c[self.ucode(self.x1)] = True
        ctx = np.r_[self.ghold, self.gline, x1c, self.eye, [self.start]]
        self.Hs = self._dg(ctx[self.H_in].sum(1))
        if c.use.get('H2'):
            ctx2 = np.r_[self.gline, x1c, self.eye, [self.start]]
            self.H2s = self._dg(ctx2[self.H2_in].sum(1))

    def _dg(self, I):
        c = self.c
        idx = np.flatnonzero(I >= c.h_thr)
        s = np.zeros(c.NH, bool)
        if len(idx) > c.kH:
            v = I[idx] + self.H_tie[idx]
            idx = idx[np.argpartition(-v, c.kH)[:c.kH]]
        s[idx] = True
        return s

    def predict(self):
        self.recompute_conj()
        if self.c.use['H']: self.recompute_H()
        xs = self.context()
        extra = None
        if self.c.use.get('gate') and 'gate' not in self.lesion:
            extra = np.zeros(self.P.n, F32)
            for gname, buf in (('gB', self.B.astype(F32)), ('gR', (self.Brec > 0.3).astype(F32)),
                               ('gN', (self.Bnext > 0.3).astype(F32))):
                if gname == 'gB' and 'B' in self.lesion or gname in ('gR', 'gN') and 'Brec' in self.lesion:
                    continue
                G = self.gates[gname]
                pg = float(G.predict({k: v for k, v in xs.items() if k in G.paths})[0]) if buf.any() else 0.0
                self.p_gate[gname] = pg
                extra[1:] += self.c.kappa * pg * buf
        self.p_next = self.P.predict(xs, extra=extra).copy()

    # ------------------------------------------------------------------ events
    def char_event(self, z, who):
        """a char of the current line: z heard from another speaker, or said by me (efference copy)."""
        if who == 'other':
            self.stats['events_other'] += 1
            self.pred_log.append(int(np.argmax(self.p_next[:1 + self.nz]) == 1 + z))
            if not self.start:
                self.line_surp.append(1.0 - float(self.p_next[1 + z]))
            if self.c.learn and self.c.use.get('gate'):
                for gname, buf in (('gB', self.B), ('gR', self.Brec > 0.3), ('gN', self.Bnext > 0.3)):
                    if buf.any() and self.gates[gname]._xs is not None:
                        self.gates[gname].learn(np.array([1.0 if buf[z] else 0.0], F32))
            if self.c.learn:
                self.remember(1 + z)
                y = np.zeros(self.P.n, F32); y[1 + z] = 1
                self.learn_P(y)
                self.stats['learn_steps'] += 1
        else:
            self.stats['events_own'] += 1
        if self.line_by is None:
            self.line_by = who
        if self.x1 >= 0:
            self.gline[self.bcode(self.x1, z)] = True
        self.gline[self.ucode(z)] = True
        self.line_chars.append(z)
        self.x2, self.x1 = self.x1, z
        self.start = False
        if self.log_events is not None:
            self.log_events.append((self.t, who, z))
        self.predict()

    def end_event(self, who):
        """the current line ended (another speaker fell silent / I stopped)."""
        if who == 'other' and self.c.learn and not self.start:
            self.remember(0)
            y = np.zeros(self.P.n, F32); y[0] = 1
            self.learn_P(y)
            self.stats['learn_steps'] += 1
        if who == 'other' and not self.start:
            # BG gate: the completed line of another speaker is loaded into the held working memory
            self.ghold = self.gline.copy()
            self.B = np.zeros(self.c.zmax, bool)
            self.B[self.line_chars] = True
            self.gtopic_v = self.c.topic_leak * self.gtopic_v + self.ghold
            self.gtopic = self.gtopic_v >= 0.75
            ms = float(np.mean(self.line_surp)) if self.line_surp else 0.0
            self.Shold = (ms >= np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.95])).astype(F32)
            if self.c.use.get('Brec'):
                # hippocampus: recall the most similar past line (reinstate its chars and the chars of the line that
                # followed it), then store this line in a fresh episode cell (one-shot)
                key = self.rkey()
                b, sim = self.recall(key) if 'Brec' not in self.lesion else (-1, 0.0)
                self.recall_sim = sim
                self.Brec = np.zeros(self.c.zmax, F32); self.Bnext = np.zeros(self.c.zmax, F32)
                if b >= 0:
                    self.Brec[self.ep_self[b]] = 1
                    if self.ep_next[b] is not None and self.c.use.get('Bnext'):
                        self.Bnext[self.ep_next[b]] = 1
                self.Brec[self.line_chars] = 0             # what I just heard is already in B
                self.Bnext[self.line_chars] = 0
                if self.c.learn:
                    if self.ep_last >= 0 and self.ep_next[self.ep_last] is None:
                        self.ep_next[self.ep_last] = np.array(self.line_chars, int)
                    self.store_episode(key, self.line_chars)
        if self.log_events is not None:
            self.log_events.append((self.t, who, 'END'))
        self.gline = np.zeros(self.c.NG, bool)
        self.line_chars = []
        self.line_surp = []
        self.x1 = self.x2 = -1
        self.start = True
        was = self.line_by
        self.line_by = None
        self.predict()
        if who == 'other' and was == 'other' and self.c.use['go']:
            xg = {k: v for k, v in self.context().items() if k in self.GoP.paths}
            self.p_go = float(self.GoP.predict(xg)[0])
            self.go_pending = (self.t, xg, self.p_go)

    # ------------------------------------------------------------------ Z (perceptual units)
    def z_drive(self, f):
        x = self._norm(f)
        p = self.Z.predict({'f': x}).copy()
        p[self.nz:] = 0
        return x, p

    def z_sample(self, f, target_z=None):
        """a sampled syllable window: returns the Z unit (recruited if needed) or -1.  target_z: efference copy."""
        x, p = self.z_drive(f)
        if target_z is not None:
            if self.c.learn_z and target_z >= 0:
                y = np.zeros(self.c.zmax, F32); y[target_z] = 1
                cols = np.flatnonzero((p > self.c.z_confuse) | (y > 0))
                self.Z.learn(y, xs={'f': x}, p=p, cols=cols)
            return target_z
        w = int(np.argmax(p)) if self.nz else -1
        if w >= 0 and p[w] >= self.c.z_recruit:
            if self.c.learn_z:                             # competitive PR: the winner and its confusable rivals
                y = np.zeros(self.c.zmax, F32); y[w] = 1
                cols = np.flatnonzero((p > self.c.z_confuse) | (y > 0))
                self.Z.learn(y, xs={'f': x}, p=p, cols=cols)
            self.z_heard[w] += 1
            return w
        if not self.c.learn_z or self.nz >= self.c.zmax:
            return w if (w >= 0 and p[w] >= self.c.z_detect) else -1
        z = self.nz; self.nz += 1                         # recruitment: a fresh unit for an unexplained syllable
        y = np.zeros(self.c.zmax, F32); y[z] = 1
        self.Z.learn(y, xs={'f': x}, p=p, cols=[z], eta=1.0)
        self.z_heard[z] += 1
        if self.c.use_fwd and self.z_prog[z] is None and 'fwd' not in self.lesion:
            self.fwd_lookup(z)
        self.stats['recruits'] += 1
        return z

    # ------------------------------------------------------------------ body interface
    def prog_list(self):
        self._progs = [(z, self.z_prog[z]) for z in range(self.nz) if self.z_prog[z] is not None]
        return [pg for _, pg in self._progs]

    def scores(self, first):
        sc = np.array([self.p_next[1 + z] for z, _ in self._progs], F32)
        if first and self.c.use['go'] and 'go' not in self.lesion and self.p_go < self.c.go_thr:
            sc = np.zeros_like(sc)                         # NoGo: nobody is expected to take this turn
        return sc, float(self.p_next[0])

    def fwd_store(self, prog, f):
        """babble memory = a forward model in lookup form: gesture -> FEAT0 code of its own voice (one-shot)."""
        if not hasattr(self, 'fwd_keys'):
            self.fwd_keys, self.fwd_pats, self.fwd_seen = [], [], set()
        pk = np.packbits(f.astype(bool))
        h = pk.tobytes()
        if h in self.fwd_seen:                             # the same sound is already remembered
            return
        self.fwd_seen.add(h)
        self.fwd_keys.append(prog)
        self.fwd_pats.append(pk)
        self._fwd_mat = None

    def fwd_lookup(self, z):
        """a newly heard syllable unit z: which remembered gesture made a sound that z would recognise?"""
        if not getattr(self, 'fwd_keys', None):
            return False
        if self._fwd_mat is None:
            self._fwd_mat = np.unpackbits(np.stack(self.fwd_pats), axis=1)[:, :self.NF].astype(F32)
            nrm = np.sqrt(self._fwd_mat.sum(1, keepdims=True)); nrm[nrm == 0] = 1
            self._fwd_mat /= nrm
        d = self._fwd_mat @ self.Z.paths['f'].W[:, z]
        b = int(np.argmax(d))
        if d[b] >= self.c.z_link:
            self.z_prog[z] = self.fwd_keys[b]
            self.stats['links_fwd'] = self.stats.get('links_fwd', 0) + 1
            del self.fwd_keys[b]; del self.fwd_pats[b]; self._fwd_mat = None
            return True
        return False

    def new_gesture(self):
        """innate habituation: a gesture already babbled is not tried again until every 4-set has been tried."""
        if not hasattr(self, '_order') or self._oi >= len(self._order):
            import itertools
            allsets = list(itertools.combinations(range(32), 4))
            self._order = [allsets[i] for i in self.rng.permutation(len(allsets))]
            self._oi = 0
        g = self._order[self._oi]; self._oi += 1
        return g

    def babble_fn(self):
        if not self.babbling:
            return None
        if not self.bab_plan:
            return 'STOP'
        return self.bab_plan.pop()

    # ------------------------------------------------------------------ one tick
    def step(self, obs):
        c = self.c
        win = obs['ear']
        E = float(np.asarray(win).sum())
        self.ear.step(win)
        f = self.ear.feat0()
        eye = np.asarray(obs['eye']).astype(bool)
        if not np.array_equal(eye, self.eye):
            self.eye = eye
            self.predict()
        # own voice in the ear = I spoke at t-1 or within the last 3 ticks (heard 1 tick late, 6-slice window);
        # NOT the corollary discharge itself, which also starts at a GO that never became speech
        self.own_tail = 3 if self.body.SPK else max(0, getattr(self, 'own_tail', 0) - 1)
        own_active = self.body.SPK or self.own_tail > 0
        # ---- own syllables: full own window 3 ticks after the release -> Z learns own voice (efference target)
        while self.own_pending and self.own_pending[0][0] == self.t:
            _, prog, zt = self.own_pending.pop(0)
            if zt >= 0:
                self.z_sample(f, target_z=zt)
            else:                                          # babble: which heard unit does my voice activate?
                if np.asarray(win).max() >= c.z_min_level:
                    x, p = self.z_drive(f)
                    w = int(np.argmax(p)) if self.nz else -1
                    p2 = float(np.partition(p, -2)[-2]) if self.nz > 1 else 0.0
                    # link only -- no Z learning here: a babble has no trustworthy target (self-supervision on the
                    # winner runs away into hub units); an unambiguous match is required
                    if w >= 0 and p[w] >= c.z_link and p[w] - p2 >= c.z_margin:
                        if self.z_prog[w] is None:
                            self.z_prog[w] = prog
                            self.stats['links'] += 1
                    elif c.use_fwd and (w < 0 or p[w] < c.z_link):  # unfamiliar sound: keep (gesture -> own sound)
                        self.fwd_store(prog, f)
        # ---- Go/NoGo learning: did another speaker take the floor within go_win ticks of the last line end?
        if self.go_pending is not None:
            t_end, xg, pg = self.go_pending
            onset_other = (E > 0 and self.E_prev == 0 and not own_active)
            el = self.t - t_end
            tgt = None
            if onset_other:
                tgt = 1.0 if el <= self.c.go_win else (0.0 if el < self.c.go_open else 1.0)
            elif el >= self.c.go_open:
                tgt = 1.0
            if tgt is not None:
                if self.c.learn:
                    self.GoP.learn(np.array([tgt], F32), xs=xg, p=np.array([pg], F32))
                self.go_pending = None
        # ---- other speakers: perceptual clock, char events, line end
        if not own_active and self.line_by != 'self':
            if E > 0:
                if self.clock < 0 or self.E_prev == 0:
                    self.clock = 0                         # phase reset at speech onset
                else:
                    self.clock += 1
                if self.clock % 3 == 2 and np.asarray(win).max() >= c.z_min_level:
                    z = self.z_sample(f)
                    if z >= 0:
                        self.char_event(z, 'other')
            elif self.clock >= 0:
                self.clock = -1
                if self.line_by == 'other':
                    self.end_event('other')
        else:
            self.clock = -1
        # ---- body (babbling: an innate spontaneous beat plays bouts of random 4-muscle sets when idle)
        if self.babbling and not self.body.SPK and not self.body.CD.any() and not self.bab_plan:
            n = int(self.rng.integers(self.bab_bout[0], self.bab_bout[1] + 1))
            self.bab_plan = [self.new_gesture() for _ in range(n)]
        progs = self.prog_list()
        force, ev = self.body.step(E, self.scores, progs, spont=self.babbling and bool(self.bab_plan) and not self.body.SPK,
                                   babble_fn=self.babble_fn if self.babbling else None)
        if 'rel' in ev:
            kind, val = ev['rel']
            if kind == 'prog':
                z, prog = self._progs[val]
                if self.line_by == 'other':                 # I start while a line was open: close it first
                    self.end_event('other')
                self.line_by = 'self'
                self.char_event(z, 'self')
                self.own_pending.append((self.t + 3, prog, z))
            else:
                self.line_by = 'self'
                self.own_pending.append((self.t + 3, val, -1))
            self.last_rel = ev['rel']
        if 'stop' in ev and ev['stop'] not in ('silent',) and self.line_by == 'self':
            self.end_event('self')
        self.E_prev = E
        self.t += 1
        return force
