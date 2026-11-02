# ARCH_v2-brainlike prototype mind (CPU, numpy).  Areas (see report.md for the full spec):
#   EAR-0 (fixed) -> SYL (STG syllable units; association rule, template form; recruited at theta-clean ticks)
#   SYL events -> ECHO (phonological loop; order rule, one-shot) and S (SYL->SYL transition statistics; order rule,
#   slow, presynaptic normalisation) -> LEX (word-form units = syllables + chunks recruited from high transition
#   probability) -> LEXtr (current-line word trace)
#   EYE (retinotopic, fixed) -> attention priority map A (exogenous pointing + learned top-down from words) ->
#   IT_att (gain-gated pooling of the attended slot's shape/colour/number maps); VL_m word<->visual templates and
#   LA word<->location (association rule)
#   PFC: SUP (superficial; mixed selectivity: fixed random fan-in over LEXtr, IT_att, A, DQ, DC, CTX, CA1, PREV, POS;
#        k-WTA; learned recurrent co-activity assemblies), DC (deep, current-line gist, leaky), DQ (deep, held line gist:
#        BG gate loads it at the end of another speaker's line, holds it through the reply; leaky units + MD loop),
#        CTX (rostral, dialogue topic, leaky over lines)
#   HIPPOCAMPUS: EC -> DG (fixed sparse expansion) -> CA3 (one-shot co-activity index + recurrent completion) ->
#        CA1 (one-shot heteroassociation back to EC); recall at line ends; offline replay in sleep()
#   READOUT (premotor / BG decision path): next syllable or END from the context x (decision rule: eligibility tag x
#        third factor, delta form, signed) + copy sources (heard line ECHO, visual naming V_o/V_c/V_n, location V_l,
#        hippocampal recall V_h) each through a learned BG copy gate (decision rule)
#   BODY (innate, ARCH_v1): F/GO/CD, beat, program WTA, END/counter, D/I/M.  Programs created by babbling; the dorsal
#        mirror Mir (SYL <-> program, association rule) is learned from the own voice; no muscle puppeting.
import numpy as np
from front import Ear
from body import Body
from rules import OrderPath, AssocPath, DecisionPath

NSM = 256           # max SYL units (index NSM = START for the echo store)
NLM = 640           # max LEX units (syllables 0..NSM-1, chunks NSM..)


def kwta(v, k, noise=None):
    if noise is not None:
        v = v + noise
    out = np.zeros(len(v), bool)
    if k > 0:
        out[np.argpartition(-v, k)[:k]] = True
    return out


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


DEFAULT = Cfg(
    seed=0,
    # SYL
    th_det=0.80, th_new=0.75, eta_syl=0.05,
    # LEX
    eta_S=0.05, th_tp=0.75, n_chunk=6, lam_lex=0.8, fast_map=True,
    # PFC
    n_sup=3000, k_sup=150, n_d=1000, k_d=50, n_x=500, k_x=25, lam_dc=0.85, lam_dq=0.9, md=0.3, dq_noise=0.0,
    lam_x=0.6, sup_rec=True, eta_rec=0.002, B_rec=1.0, g_rec=0.5, sp_beta=0.0, sp_tau=200.0, dc_src='sup', word_assembly=False,
    # vision
    eta_vl=0.2, eta_vl_all=0.02, th_vl=0.55, eta_la=0.2, th_att=0.4, att_keep=0.9,
    # hippocampus
    hip=True, n_dg=2000, k_dg=40, dg_fan=30, eta_hip=0.0005, th_recall=0.3, max_episodes=3000, hip_recency=0.0, hip_tau=150.0,
    # readout
    eta_w=0.3, eta_g=0.3, g0_echo=0.3, tag_life=12, eta_reward=0.5, block_norm=True,
    # babbling / mirror
    babble=True, n_prog=512, prog_cap=0, babble_recruit=False, p_new=0.6, th_babble=0.25, eta_mir=0.5, p_go=0.2, p_explore=0.3, babble_idle=True, p_go_idle=0.7,
    # ablations
    no_echo=False, no_vis=False, no_sup=False, no_dq=False, no_dc=False, no_ctx=False, no_lex_chunks=False,
    no_copyV=False, no_W=False, freeze=False,
)


class Agent:
    def __init__(self, cfg=None, slot_bits=240, n_slots=3, maps=(("o", 0, 160), ("c", 160, 48), ("n", 208, 24)),
                 att_off=232):
        self.c = c = Cfg(DEFAULT); c.update(cfg or {})
        self.rng = np.random.default_rng(c.seed)
        rng = np.random.default_rng(c.seed + 1)
        self.ear = Ear(); self.use = np.zeros(self.ear.n); self.nwarm = 0
        self.body = Body(NP=c.n_prog)
        self.t = 0
        # ---------------- vision layout (innate retinotopy: per-slot replicated feature maps)
        self.slot_bits, self.n_slots, self.maps, self.att_off = slot_bits, n_slots, maps, att_off
        self.n_feat = sum(m[2] for m in maps)
        # ---------------- SYL
        self.nf = None
        self.syl_n = 0
        self.syl_heard = np.zeros(NSM, int)
        # ---------------- sequence stores
        self.echo = OrderPath(NSM + 1, eta_f=1.0, eta_b=1.0, W=4)            # current line (phonological loop)
        self.echo_prev = np.zeros((NSM + 1, NSM + 1), np.float32)            # last other's line, frozen at its end
        self.S = OrderPath(NSM, eta_f=c.eta_S, eta_b=0.0, W=4, norm='pre')   # forward TP  P(b | a)
        self.Sb = OrderPath(NSM, eta_f=c.eta_S, eta_b=0.0, W=4, norm='post') # backward TP P(a | b)
        self.chunks = {}; self.chunk_syl = {}                                 # (a,b) -> lex id ; lex id -> (a,b)
        self.n_lex = NSM
        self.lextr = np.zeros(NLM, np.float32)
        # ---------------- vision association
        self.VL = {m: AssocPath(nb, NLM, eta=c.eta_vl, B=1.0) for m, _, nb in maps}
        self.vl_n = np.zeros(NLM, int)                                        # exposures per word
        self.LA = AssocPath(n_slots, NLM, eta=c.eta_la, B=1.0)
        self.td = np.zeros(n_slots)
        self.A = -1; self.A_hold = -1; self.rel_hold = 0.0; self.rel = 0.0; self.last_feats = None
        self.it_att = np.zeros(self.n_feat, np.float32)
        self.match = np.zeros(2, np.float32)
        # ---------------- PFC
        self.blocks = [("lex", 512 if c.word_assembly else NLM), ("it", self.n_feat), ("A", n_slots + 1), ("match", 2), ("dq", c.n_d),
                       ("dc", c.n_d), ("ctx", c.n_x), ("ca1", c.n_d), ("prevc", 512), ("role", 40), ("pos", 8)]
        fan = {"lex": 4, "it": 4, "A": 1, "match": 1, "dq": 6, "dc": 3, "ctx": 2, "ca1": 2, "prevc": 4, "role": 2, "pos": 1}
        gain = {"lex": 1.0, "it": 1.0, "A": 0.5, "match": 0.5, "dq": 1.0, "dc": 1.0, "ctx": 0.5, "ca1": 0.5, "prevc": 1.0,
                "role": 1.0, "pos": 0.5}
        # the previous syllable as a distributed assembly (20 of 512) so that mixed units can conjoin it with context
        self.prev_code = np.zeros((NSM + 1, 512), np.float32)
        for s in range(NSM + 1):
            self.prev_code[s, rng.choice(512, 20, replace=False)] = 1.0
        rw = np.random.default_rng(c.seed + 9)
        self.word_code = np.zeros((NLM, 512), np.float32)          # words as distributed assemblies
        for s in range(NLM):
            self.word_code[s, rw.choice(512, 20, replace=False)] = 1.0
        self.sup_fan = {b: (rng.integers(0, n, (c.n_sup, fan[b])), gain[b] / fan[b]) for b, n in self.blocks}
        self.sup = np.zeros(c.n_sup, bool)
        self.sup_tie = rng.random(c.n_sup) * 1e-3
        self.REC = AssocPath(c.n_sup, c.n_sup, eta=c.eta_rec, B=c.B_rec) if c.sup_rec else None
        self.dc_fan = rng.integers(0, c.n_sup, (c.n_d, 20))
        self.v_dc = np.zeros(c.n_d, np.float32); self.dc = np.zeros(c.n_d, bool)
        self.v_dq = np.zeros(c.n_d, np.float32); self.dq = np.zeros(c.n_d, bool)
        self.x_fan = rng.integers(0, c.n_d, (c.n_x, 20))
        self.v_x = np.zeros(c.n_x, np.float32); self.ctx = np.zeros(c.n_x, bool)
        self.ca1 = np.zeros(c.n_d, bool)
        # ---------------- hippocampus (EC = dq + lex set + it_all + ctx)
        self.n_ec = c.n_d + NLM + self.n_feat + n_slots * self.n_feat + c.n_x
        self.ec_blocks = []; o = 0
        for n in (c.n_d, NLM, self.n_feat, n_slots * self.n_feat, c.n_x):
            self.ec_blocks.append((o, o + n)); o += n
        fan_b = max(2, c.dg_fan // len(self.ec_blocks))
        self.dg_fan = [rng.integers(o0, o1, (c.n_dg, fan_b)) for o0, o1 in self.ec_blocks]
        self.EC3 = AssocPath(self.n_ec, c.n_dg, eta=c.eta_hip, B=1.0)        # perforant path EC -> CA3 index
        self.C3 = AssocPath(c.n_dg, c.n_dg, eta=c.eta_hip, B=1.0)            # CA3 recurrent
        self.episodes = []
        # ---------------- readout + copy gates
        self.lay = {}; o = 0
        for b, n in [("sup", c.n_sup), ("dq", c.n_d), ("dc", c.n_d), ("ctx", c.n_x), ("role", 2), ("prev", NSM),
                     ("lex", NLM), ("echo", NSM), ("pos", 8), ("it", self.n_feat), ("A", n_slots + 1), ("match", 2),
                     ("done", 8), ("bias", 1)]:
            self.lay[b] = (o, o + n); o += n
        self.nx = o
        self.W = DecisionPath(self.nx, NSM + 1, eta=c.eta_w)                  # index NSM = END
        self.srcs = ["H", "Vo", "Vc", "Vn", "Vl", "Vh", "Vr"]
        self.G = {s: DecisionPath(self.nx, 1, eta=c.eta_g) for s in self.srcs}
        b0 = self.lay["bias"][0]
        self.G["H"].w[b0, 0] = c.g0_echo                                     # innate imitation tendency
        # ---------------- motor mirror (dorsal stream): SYL c <-> program y
        self.Mir = AssocPath(NSM, self.body.NP, eta=c.eta_mir, B=1.0)
        self.pending_own = []                                                # (tick heard clean, program)
        self.pending_trial = []
        self.quiet = 0; self.self_go = -100; self.blank_n = 0
        self.pred_log = []; self.line_no = 0
        self.babble_tries = {}
        # ---------------- line state
        self.E_prev = 0.0; self.theta_next = -1; self.heard_line = False
        self.t_echo = 0; self.t_S = 0; self.t_mouth = -100; self.t_line_end = -100; self.role = False
        self.prev = -1; self.pos = 0
        self.tag = None
        self.own_tags = []
        self.mode_listen = True
        self.babbled = False
        self.stats = {k: 0 for k in ["heard", "recruit", "own_det", "mir_learn", "babble", "chunks", "learn_ev",
                                     "end_ev", "recall", "encode", "reward_learn"]}

    # ================================================================== warm-up (FEAT0)
    def warm(self, win):
        self.use += self.ear.layer0(win); self.nwarm += 1

    def end_warmup(self):
        self.ear.feat = np.nonzero(self.ear.alive & (self.use / max(self.nwarm, 1) <= 0.6))[0]
        self.nf = len(self.ear.feat)
        self.SW = np.zeros((self.nf, NSM), np.float32)
        self.SWn = np.ones(NSM, np.float32)

    # ================================================================== SYL
    def syl_detect(self, f):
        if self.syl_n == 0:
            return -1, 0.0
        idx = np.nonzero(f)[0]
        if not len(idx):
            return -1, 0.0
        d = self.SW[idx, :self.syl_n].sum(0)
        cs = d / (np.sqrt(len(idx)) * self.SWn[:self.syl_n])
        j = int(np.argmax(cs))
        return j, float(cs[j])

    def syl_recruit(self, f):
        if self.syl_n >= NSM:
            return -1
        j = self.syl_n; self.syl_n += 1
        v = f.astype(np.float32); self.SW[:, j] = v / v.sum(); self.SWn[j] = np.linalg.norm(self.SW[:, j])
        self.stats["recruit"] += 1
        return j

    def syl_learn(self, f, j, eta):
        v = f.astype(np.float32); n = v.sum()
        if n > 0:
            self.SW[:, j] = (1 - eta) * self.SW[:, j] + eta * v / n
            self.SWn[j] = np.linalg.norm(self.SW[:, j])

    # ================================================================== vision
    def slot_maps(self, eye):
        """per-slot feature maps and pointing (innate retinotopic organisation)."""
        sl = np.asarray(eye[:self.n_slots * self.slot_bits]).reshape(self.n_slots, self.slot_bits)
        feats = sl[:, :self.n_feat].astype(np.float32)
        point = sl[:, self.att_off:].any(1)
        return feats, point

    def update_attention(self, eye):
        c = self.c
        feats, point = self.slot_maps(eye)
        self.feats = feats; self.point = point
        if c.no_vis or not feats.any():
            self.A = -1; self.it_att[:] = 0; self.match[:] = 0; self.td[:] = 0; self.rel = 0.0; self.A_hold = -1
            return
        if self.last_feats is None or not np.array_equal(feats, self.last_feats):
            self.A_hold = -1; self.rel_hold = 0.0                            # new scene: the old referent is gone
            self.last_feats = feats.copy()
        words = np.nonzero(self.lextr > 0.2)[0]
        td = self.c.att_keep * self.td
        _, o0, nb = self.maps[0]
        best_match = 0.0
        if len(words):
            Wo = self.VL["o"].w[:, words]                                    # (64, nw) shape templates
            m = feats[:, o0:o0 + nb] @ Wo                                    # (slots, nw) match = template mass present
            sel = (self.vl_n[words] >= 2)
            m = m * sel[None, :]
            # a word attracts attention to a slot only if it is shape-selective (its template matches one slot)
            td = td + (m * self.lextr[words][None, :]).max(1)
            best_match = m.max() if m.size else 0.0
            td = td + 0.8 * (self.LA.w[:, words] * self.lextr[words][None, :] * (self.LA.w[:, words].max(0) > 0.6)).max(1)
        self.td = td
        if point.any():
            s = int(np.argmax(point)); rel = 1.0; src = 'point'              # exogenous: the tutor points
        elif td.max() > c.th_att:
            s = int(np.argmax(td)); rel = float(td.max()); src = 'td'        # top-down from heard words
        elif self.A_hold >= 0:
            s = self.A_hold; rel = self.rel_hold; src = 'hold'               # referent held from the partner's line
        else:
            s = -1; rel = 0.0; src = None
        self.rel = rel
        self.att_src = src
        self.A = s
        self.it_att = feats[s].copy() if s >= 0 else np.zeros(self.n_feat, np.float32)
        # comparator (stage D 'check'): a heard object-like word (sharp shape template) vs the pointed object
        self.match[:] = 0
        if point.any() and len(words):
            Wo = self.VL["o"].w[:, words]
            sharp = (np.sort(Wo, 0)[-8:].sum(0) >= 0.6) & (self.vl_n[words] >= 4)
            if sharp.any():
                mp = feats[s, o0:o0 + nb] @ Wo[:, sharp]
                self.match[0 if mp.max() > c.th_vl else 1] = 1.0

    def vis_learn(self, lex_on, eye):
        """word <-> visual templates (association rule, per-word normalisation) and word <-> location."""
        c = self.c
        if c.no_vis or not len(lex_on):
            return
        feats, point = self.slot_maps(eye)
        if not feats.any():
            return
        x = np.zeros(NLM, bool); x[lex_on] = True
        if self.A >= 0 and self.rel >= 0.8:
            src, eta = feats[self.A], c.eta_vl                                  # reliable attention
        else:
            src, eta = feats.max(0), c.eta_vl_all                               # cross-situational, weak
        if eta == c.eta_vl:
            # comparator gate: a well-learned, sharp word whose shape template clearly mismatches the attended
            # object refers to something else ("这是X吗" pointing at Y) -> not re-bound
            o0s, nbs = self.maps[0][1], self.maps[0][2]
            for wd in list(lex_on):
                tmpl = self.VL["o"].col(wd)
                if self.vl_n[wd] >= 10 and np.sort(tmpl)[-8:].sum() >= 0.6 and \
                        float(src[o0s:o0s + nbs] @ tmpl) < 0.3:
                    x[wd] = False
        for m, o0, nb in self.maps:
            V = self.VL[m]; V.eta = eta
            V.learn(src[o0:o0 + nb] > 0, x)
        self.vl_n[lex_on] += 1
        if self.A >= 0 and self.rel >= 0.8:
            a = np.zeros(self.n_slots, bool); a[self.A] = True
            self.LA.learn(a, x)

    def evoked(self, m, key):
        """copy source from vision: the word whose map-m template best matches the attended map; its syllables."""
        o0 = [mm[1] for mm in self.maps if mm[0] == m][0]; nb = [mm[2] for mm in self.maps if mm[0] == m][0]
        if self.A < 0:
            return None
        f = self.it_att[o0:o0 + nb]
        Wm = self.VL[m].w
        ok = (self.vl_n >= 4) | ((self.vl_n >= 1) & (np.sort(Wm, 0)[-8:].sum(0) >= 0.8))
        sc = (f @ Wm) * ok * self.word_units()
        w = int(np.argmax(sc))
        if sc[w] < self.c.th_vl:
            return None
        return self.word_succ(w, key)

    def word_succ(self, w, key):
        if w >= NSM and w not in self.chunk_syl:
            return None
        """the copy pointer of a word source: inside the word -> the next syllable; at a word onset (the previous
        syllable is not part of the word) -> the word's first syllable; after the word's last syllable -> nothing."""
        seq = list(self.chunk_syl[w]) if w >= NSM else [w]
        v = np.zeros(NSM, np.float32)
        if key >= 0 and key in seq[:-1]:
            v[seq[seq.index(key) + 1]] = 1
        elif key >= 0 and key == seq[-1]:
            self._done = True
            return None
        else:
            v[seq[0]] = 1
        return v

    def word_units(self):
        """LEX word units: chunks, plus syllables that are not part of any chunk (single-syllable words)."""
        m = np.zeros(NLM, bool)
        m[NSM:self.n_lex] = True
        m[:self.syl_n] = True
        for (p, q) in self.chunks:
            m[p] = False; m[q] = False
        return m

    def evoked_loc(self, key):
        if self.A < 0:
            return None
        sc = self.LA.w[self.A] * (self.LA.w.max(0) > 0.6) * (np.arange(NLM) >= NSM) * (self.vl_n >= 4) \
            * self.word_units()                                                # location words
        w = int(np.argmax(sc))
        if sc[w] < 0.6:
            return None
        return self.word_succ(w, key)

    # ================================================================== PFC
    def sup_inputs(self):
        A = np.zeros(self.n_slots + 1, np.float32); A[self.A] = 1.0
        prev = np.zeros(NSM, np.float32)
        if self.prev >= 0:
            prev[self.prev] = 1.0
        pos = np.zeros(8, np.float32); pos[min(self.pos, 7)] = 1.0
        c = self.c
        role = np.zeros(40, np.float32); role[(0 if self.role else 20):(20 if self.role else 40)] = 1.0
        lexc = np.zeros(512, np.float32)
        if c.word_assembly:
            nz = np.nonzero(self.lextr)[0]
            if len(nz):
                lexc = np.minimum(1.0, (self.lextr[nz, None] * self.word_code[nz]).sum(0))
        return {"lex": (lexc if c.word_assembly else self.lextr), "it": self.it_att, "A": A, "match": self.match, "role": role,
                "prevc": self.prev_code[self.prev if self.prev >= 0 else NSM],
                "dq": (self.dq if not c.no_dq else np.zeros(c.n_d, bool)).astype(np.float32),
                "dc": (self.dc if not c.no_dc else np.zeros(c.n_d, bool)).astype(np.float32),
                "ctx": (self.ctx if not c.no_ctx else np.zeros(c.n_x, bool)).astype(np.float32),
                "ca1": self.ca1.astype(np.float32), "pos": pos}

    def pfc_event(self):
        c = self.c
        inp = self.sup_inputs()
        d = np.zeros(c.n_sup, np.float32)
        for b, (idx, g) in self.sup_fan.items():
            d += g * inp[b][idx].sum(1)
        if self.REC is not None and self.sup.any():
            d += c.g_rec * self.REC.current(self.sup)
        if c.sp_beta > 0:
            if not hasattr(self, 'u_sup'):
                self.u_sup = np.full(c.n_sup, c.k_sup / c.n_sup, np.float32); self.u_dc = np.full(c.n_d, c.k_d / c.n_d, np.float32)
            d = d - c.sp_beta * np.maximum(self.u_sup / (c.k_sup / c.n_sup) - 1.0, 0) * d.mean()
        new = kwta(d + self.sup_tie, c.k_sup)
        if c.sp_beta > 0:
            self.u_sup += (new - self.u_sup) / c.sp_tau
        if self.REC is not None and not c.freeze:
            self.REC.learn(new, new)                                             # co-activity assemblies
        self.sup = new
        # DC: current-line gist.  dc_src 'sup': leaky integration of SUP (which also carries the held context);
        # 'content': leaky integration of the line's CONTENT only (words + attended object + location), so that a held
        # gist does not inherit the previous one (the reverberation / generic-state failure)
        if c.dc_src == 'content':
            if not hasattr(self, 'dcc_fan'):
                r = np.random.default_rng(c.seed + 5)
                self.dcc_fan = {b: r.integers(0, len(inp[b]), (c.n_d, 6)) for b in ('lex', 'it', 'A')}
            u = 0
            for b in ('lex', 'it', 'A'):
                x = inp[b]; m = max(float((x > 0).mean()), 1e-3)
                u = u + x[self.dcc_fan[b]].mean(1) / m * (0.3 if b == 'A' else 1.0)
            self.v_dc = c.lam_dc * self.v_dc + u.astype(np.float32)
        else:
            self.v_dc = c.lam_dc * self.v_dc + self.sup[self.dc_fan].sum(1) / 20.0
        vd = self.v_dc
        if c.sp_beta > 0 and self.v_dc.any():
            vd = vd - c.sp_beta * np.maximum(self.u_dc / (c.k_d / c.n_d) - 1.0, 0) * vd.mean()
        self.dc = kwta(vd + self.sup_tie[:c.n_d], c.k_d) if self.v_dc.any() else np.zeros(c.n_d, bool)
        if c.sp_beta > 0:
            self.u_dc += (self.dc - self.u_dc) / c.sp_tau
        self.inp = inp

    def dq_tick(self):
        """deep hold layer: leaky units + MD-thalamus loop (amplifies the active pattern); k-WTA output."""
        c = self.c
        if not self.dq.any():
            return
        noise = self.rng.normal(0, c.dq_noise, c.n_d) if c.dq_noise > 0 else None
        self.v_dq = c.lam_dq * self.v_dq + c.md * self.dq
        self.dq = kwta(self.v_dq, c.k_d, noise)

    def load_dq(self):
        """BG gate opens at the end of another speaker's line: DQ <- DC (1:1), CTX integrates DQ."""
        c = self.c
        if not self.dc.any():
            return
        self.v_dq = self.dc.astype(np.float32) * (c.md / (1 - c.lam_dq))
        self.dq = self.dc.copy()
        self.v_x = c.lam_x * self.v_x + self.dq[self.x_fan].sum(1) / 20.0
        self.ctx = kwta(self.v_x, c.k_x)

    def reset_line(self):
        self.v_dc[:] = 0; self.dc[:] = False
        self.prev = -1; self.pos = 0
        self.lextr[:] = 0

    # ================================================================== hippocampus
    def ec_vector(self, feats=None):
        """EC input of the hippocampus: held line gist DQ, the line's words, pooled (object identity) and per-slot
        (retinotopic snapshot) scene features, dialogue context CTX."""
        c = self.c
        ec = np.zeros(self.n_ec, bool)
        o = 0
        ec[o:o + c.n_d] = self.dq; o += c.n_d
        ec[o:o + NLM] = self.line_words > 0; o += NLM
        if feats is not None:
            ec[o:o + self.n_feat] = feats.max(0) > 0; o += self.n_feat
            ec[o:o + self.n_slots * self.n_feat] = feats.reshape(-1) > 0; o += self.n_slots * self.n_feat
        else:
            o += self.n_feat * (1 + self.n_slots)
        ec[o:] = self.ctx
        return ec

    def ec_weights(self, ec):
        """each EC input stream (gist, words, scene, scene per slot, context) contributes comparably."""
        x = ec.astype(np.float32)
        for o0, o1 in self.ec_blocks:
            n = float(ec[o0:o1].sum())
            if n > 0:
                x[o0:o1] /= np.sqrt(n)
        return x

    def dg_code(self, ec):
        x = self.ec_weights(ec)
        d = sum(x[f].sum(1) for f in self.dg_fan) + self.sup_tie[:self.c.n_dg]
        return kwta(d.astype(np.float32), self.c.k_dg)

    def hip_line_end(self, eye):
        """at the end of another speaker's line: recall with the current EC as a cue (perforant path -> CA3 index ->
        recurrent completion -> the stored episode's content via CA1), then store the new episode one-shot."""
        c = self.c
        self.ca1[:] = False
        self.rec_words = np.zeros(NLM, bool); self.rec_seq = []; self.rec_feats = None
        if not c.hip:
            return
        feats, _ = self.slot_maps(eye)
        ec = self.ec_vector(feats)
        if len(self.episodes):
            d = self.EC3.current(self.ec_weights(ec))
            idx = kwta(d, c.k_dg)
            for _ in range(2):
                idx = kwta(self.C3.current(idx) + 0.5 * d / max(d.max(), 1e-9), c.k_dg)
            E = np.array([e["idx"] for e in self.episodes])                 # (n_episodes, k_dg) CA3 indices
            ov = idx[E].sum(1)
            if c.hip_recency > 0:
                # temporal context (lateral EC time code): recent episodes are closer to the current context
                T = np.array([e.get("t", -10 ** 9) for e in self.episodes], float)
                k = int(np.argmax(ov / c.k_dg + c.hip_recency * np.exp(-(self.t - T) / c.hip_tau)))
            else:
                k = int(np.argmax(ov))
            if ov[k] >= c.th_recall * c.k_dg:
                e = self.episodes[k]
                self.ca1 = e["dq"].copy()
                self.rec_words = e["words"].copy(); self.rec_seq = list(e["seq"]); self.rec_feats = e["feats"]
                self.stats["recall"] += 1
        if not c.freeze:
            g = self.dg_code(ec)
            self.EC3.learn(ec, g); self.C3.learn(g, g)
            self.episodes.append({"idx": np.nonzero(g)[0], "dq": self.dq.copy(), "words": self.line_words > 0,
                                  "seq": list(self.line_seq), "feats": feats.copy(), "t": self.t})
            if len(self.episodes) > c.max_episodes:
                self.episodes.pop(0)
            self.stats["encode"] += 1

    def sleep(self, n_replay=200):
        """offline phase: replay stored episodes (CA3 index -> content) into slow cortex: the word <-> vision
        templates learn from the recalled words and the recalled scene (slow eta), interleaving old and new."""
        c = self.c
        if not self.episodes:
            return 0
        n = 0
        for i in self.rng.integers(0, len(self.episodes), n_replay):
            e = self.episodes[i]
            words = np.nonzero(e["words"])[0]
            if len(words) and e["feats"].any():
                x = np.zeros(NLM, bool); x[words] = True
                src = e["feats"].max(0) > 0
                for m, o0, nb in self.maps:
                    V = self.VL[m]; V.eta = c.eta_vl_all
                    V.learn(src[o0:o0 + nb], x)
                n += 1
        return n

    # ================================================================== readout
    def context(self):
        c = self.c
        x = np.zeros(self.nx, np.float32)
        L = self.lay
        if not c.no_sup:
            x[L["sup"][0]:L["sup"][1]] = self.sup
        x[L["dq"][0]:L["dq"][1]] = self.dq if not c.no_dq else 0
        x[L["dc"][0]:L["dc"][1]] = self.dc if not c.no_dc else 0
        x[L["ctx"][0]:L["ctx"][1]] = self.ctx if not c.no_ctx else 0
        x[L["role"][0] + (0 if self.role else 1)] = 1
        if self.prev >= 0:
            x[L["prev"][0] + self.prev] = 1
        x[L["lex"][0]:L["lex"][1]] = self.lextr
        key = self.prev if self.prev >= 0 else NSM
        ech = self.echo_prev[key, :NSM] if not c.no_echo else np.zeros(NSM, np.float32)
        x[L["echo"][0]:L["echo"][1]] = ech
        x[L["pos"][0] + min(self.pos, 7)] = 1
        if not c.no_vis:
            x[L["it"][0]:L["it"][1]] = self.it_att
            x[L["A"][0] + (self.A if self.A >= 0 else self.n_slots)] = 1
            x[L["match"][0]:L["match"][1]] = self.match
        db = getattr(self, "done_bits", {})
        for i, s in enumerate(["H", "Vo", "Vc", "Vn", "Vl", "Vh", "Vr"]):
            if db.get(s):
                x[L["done"][0] + i] = 1
        x[L["bias"][0]] = 1
        if c.block_norm:
            for b, (o0, o1) in L.items():
                n = float((x[o0:o1] > 0).sum())
                if n > 1:
                    x[o0:o1] /= np.sqrt(n)
        return x

    def cands(self):
        key = self.prev
        out = {}
        e = self.echo_prev[key if key >= 0 else NSM, :NSM]
        out["H"] = e.copy() if (e.any() and not self.c.no_echo) else None
        done = {}
        if not self.c.no_vis and not self.c.no_copyV:
            for m, s in (("o", "Vo"), ("c", "Vc"), ("n", "Vn")):
                self._done = False; out[s] = self.evoked(m, key); done[s] = self._done
            self._done = False; out["Vl"] = self.evoked_loc(key); done["Vl"] = self._done
        else:
            out.update({"Vo": None, "Vc": None, "Vn": None, "Vl": None})
        out["Vh"] = None; out["Vr"] = None
        # CA1 comparison: a recalled line answers the question only if it partially matches it (>= 2 shared
        # syllables) and differs in the rest ("我叫什么" vs "我叫小明" -> 小明)
        shared = sum(1 for s in set(self.rec_seq) if s >= 0 and self.line_words_q[s]) if len(self.rec_seq) else 0
        partial = shared >= 2
        if self.c.hip and len(self.rec_seq) and partial:
            new_info = [s for s in self.rec_seq if s >= 0 and not self.line_words_q[s]]   # recalled, not just heard
            v = np.zeros(NSM, np.float32)
            if new_info:
                if key >= 0 and key in new_info[:-1]:
                    v[new_info[new_info.index(key) + 1]] = 1
                elif not (key >= 0 and key == new_info[-1]):
                    v[new_info[0]] = 1
            out["Vh"] = v if v.any() else None
            done["Vh"] = bool(new_info) and key >= 0 and key == new_info[-1]
        if self.c.hip and self.rec_feats is not None and self.A >= 0 and not self.c.no_vis and partial:
            o0 = self.maps[0][1]; nb = self.maps[0][2]
            f = self.rec_feats[self.A, o0:o0 + nb]
            sc = (f @ self.VL["o"].w) * (self.vl_n >= 4) * self.word_units()
            w = int(np.argmax(sc))
            if sc[w] >= self.c.th_vl:
                self._done = False; out["Vr"] = self.word_succ(w, key); done["Vr"] = self._done
        self.done_bits = done
        return out

    def set_tag(self):
        cd = self.cands()
        x = self.context()
        self.tag = {"x": x, "cands": cd, "t": self.t}

    def predict(self, tag):
        x = tag["x"]
        d = np.zeros(NSM + 1, np.float32) if self.c.no_W else self.W.drive(x)
        g = {}
        extra = np.zeros(NSM + 1, np.float32)
        for s, v in tag["cands"].items():
            if v is None:
                continue
            gs = float(np.clip(self.G[s].drive(x)[0], 0, 1))
            g[s] = gs
            extra[:NSM] = np.maximum(extra[:NSM], gs * v)
        return d + extra, extra, g

    def learn_readout(self, target):
        """learning by listening: M = 1 for another speaker's syllable (or END)."""
        tag = self.tag
        if tag is None or self.t - tag["t"] > self.c.tag_life or self.c.freeze:
            return
        tgt = np.zeros(NSM + 1, np.float32); tgt[target] = 1.0
        d, extra, g = self.predict(tag)
        if target < NSM:                       # listening prediction log (recurrent-predictor comparison)
            self.pred_log.append((self.line_no, int(target), int(np.argmax(np.clip(d[:NSM], 0, 1))) == target))
        self.W.e = tag["x"]
        if not self.c.no_W:
            self.W.learn(tgt, M=1.0, extra=extra, cols=np.nonzero((tgt > 0) | (d > 0))[0])
        for s, v in tag["cands"].items():
            if v is None:
                continue
            tG = 1.0 if (target < NSM and v[target] > 0) else 0.0
            self.G[s].e = tag["x"]; self.G[s].learn(np.array([tG], np.float32))
        self.stats["learn_ev" if target < NSM else "end_ev"] += 1

    def boundary(self, body, first):
        c = self.c
        if self.tag is None:
            self.set_tag()
        d, extra, g = self.predict(self.tag)
        p = np.clip(d, 0, 1)
        n = body.n_prog
        sc = np.zeros(n)
        if n and self.syl_n:
            M = self.Mir.w[:self.syl_n, :n]
            sc = (p[:self.syl_n, None] * M).max(0)
        self.last_pred = (p, g)
        self_init = first and (self.t - self.self_go) <= 4                   # this turn was the agent's own GO
        if first and ((c.babble and self.babble_on and (n == 0 or sc.max() < c.th_babble or self.rng.random() < c.p_explore))
                      or (self_init and c.babble_idle and not c.freeze)):
            linked = np.nonzero(self.Mir.w[:, :n].max(0) > 0.3)[0] if n else []
            if len(linked) == 0 or self.rng.random() < c.p_new:
                for _ in range(20):
                    ms = frozenset(int(m) for m in self.rng.choice(32, 4, replace=False))
                    if self.babble_tries.get(ms, 0) < 2:
                        break
                self.babble_tries[ms] = self.babble_tries.get(ms, 0) + 1
                body.trial = ms
                self.pending_trial.append((self.t + 3, ms))
                self.stats["babble"] += 1
                return "trial", 0.0
            y = int(self.rng.choice(linked))                                 # practice a known program, ONE beat
            ms = frozenset(int(m) for m in np.nonzero(body.prog[y])[0])
            body.trial = ms
            self.pending_trial.append((self.t + 3, ms))
            self.stats["practice"] = self.stats.get("practice", 0) + 1
            return "trial", 0.0
        return sc, (float(p[NSM]) if not first else 0.0)

    # ================================================================== babbling / mirror
    def pick_babble(self, body):
        linked = np.nonzero(self.Mir.w[:, :body.n_prog].max(0) > 0.3)[0] if body.n_prog else []
        if len(linked) == 0 or self.rng.random() < self.c.p_new:
            for _ in range(20):
                ms = frozenset(int(m) for m in self.rng.choice(32, 4, replace=False))
                if self.babble_tries.get(ms, 0) < 2:
                    break
            self.babble_tries[ms] = self.babble_tries.get(ms, 0) + 1
            return body.add_program(ms)
        return int(self.rng.choice(linked))

    def mirror_syl(self, y):
        if y < 0 or self.syl_n == 0:
            return -1
        col = self.Mir.w[:self.syl_n, y]
        j = int(np.argmax(col))
        return j if col[j] > 0.3 else -1

    # ================================================================== events
    def on_heard(self, j, eye):
        """another speaker's syllable j (clean tick)."""
        c = self.c
        self.stats["heard"] += 1
        self.syl_heard[j] += 1
        self.learn_readout(j)
        self.echo.step([j], dt=self.t - self.t_echo); self.t_echo = self.t
        # LEX: slow statistics + chunk recruitment
        p = self.prev
        if not c.freeze:
            self.S.step([j], dt=self.t - self.t_S); self.Sb.step([j], dt=self.t - self.t_S); self.t_S = self.t
        lex_on = [j]
        if p >= 0:
            ch = self.chunks.get((p, j))
            novel = (self.syl_heard[p] <= 2 and self.syl_heard[j] <= 2 and c.fast_map
                     and self.stats['heard'] > 2000)                      # after the vocabulary spurt
            if ch is None and not c.no_lex_chunks and novel and self.n_lex < NLM:
                ch = self.n_lex; self.n_lex += 1                                 # fast mapping of a novel word
                self.chunks[(p, j)] = ch; self.chunk_syl[ch] = (p, j); self.stats["fastmap"] = self.stats.get("fastmap", 0) + 1
            if ch is None and not c.no_lex_chunks and self.S.w[p, j] >= c.th_tp and self.Sb.w[p, j] >= c.th_tp \
                    and self.syl_heard[p] >= c.n_chunk and self.syl_heard[j] >= c.n_chunk \
                    and self.n_lex < NLM:
                ch = self.n_lex; self.n_lex += 1
                self.chunks[(p, j)] = ch; self.chunk_syl[ch] = (p, j); self.stats["chunks"] += 1
            if ch is not None:
                lex_on.append(ch)
        self.lextr *= c.lam_lex
        self.lextr[lex_on] = 1.0
        self.line_words[lex_on] = 1
        self.line_seq.append(j)
        self.update_attention(eye)
        if not c.freeze:
            self.vis_learn(lex_on, eye)
        self.prev = j; self.pos += 1
        self.pfc_event()
        self.set_tag()

    def other_line_onset(self):
        self.line_no += 1
        self.theta_next = self.t + 2
        self.role = (self.t - self.t_line_end) <= 8
        if not self.role:                                   # a new initiative: the previous exchange's referent ends
            self.A_hold = -1; self.rel_hold = 0.0
        if not self.line_reset:
            self.reset_line()
        self.echo.w[:] = 0
        self.echo.step([NSM], dt=self.t - self.t_echo); self.t_echo = self.t      # START -> first syllable
        self.td[:] = 0
        self.line_reset = False
        self.heard_line = True
        self.line_words = np.zeros(NLM, np.float32)
        self.line_seq = []

    def other_line_end(self, eye):
        c = self.c
        self.t_line_end = self.t
        if self.prev >= 0:
            self.learn_readout(NSM)                                          # END target
        self.echo_prev = self.echo.w.copy()
        self.line_words_q = self.line_words.copy() > 0
        self.load_dq()
        self.update_attention(eye)
        self.A_hold, self.rel_hold = self.A, self.rel                        # BG gate: hold the referent
        self.hip_line_end(eye)
        self.reset_line(); self.line_reset = True
        self.heard_line = False
        self.role = True                                                     # the next line (if soon) is a reply
        self.update_attention(eye)
        self.pfc_event()
        self.set_tag()

    def own_release(self, y, eye):
        c = self.c
        cs = self.mirror_syl(y)
        if self.pos == 0:
            self.own_tags = []
        if self.tag is not None and not self.babbled:
            self.own_tags.append((self.tag, cs))
        self.babbled = False
        self.pending_own.append((self.t + 3, y))
        if cs < 0:
            return
        if self.pos == 0 and not self.line_reset:
            self.reset_line()
        self.line_reset = False
        self.lextr *= c.lam_lex; self.lextr[cs] = 1.0
        p = self.prev
        if p >= 0 and (p, cs) in self.chunks:
            self.lextr[self.chunks[(p, cs)]] = 1.0
        self.prev = cs; self.pos += 1
        self.update_attention(eye)
        self.pfc_event()
        self.set_tag()

    def own_end(self, eye):
        self.t_line_end = self.t
        self.reset_line(); self.line_reset = True
        self.update_attention(eye)
        self.pfc_event()
        self.set_tag()
        self.own_t_end = self.t

    def on_reward(self, r):
        """contingent partner reaction: third factor for the eligibility tags of the own reply (and babble links)."""
        c = self.c
        if c.freeze or r <= 0:
            self.own_tags = []
            return
        for tag, cs in self.own_tags[-12:]:
            if cs < 0:
                continue
            tgt = np.zeros(NSM + 1, np.float32); tgt[cs] = 1.0
            self.W.e = tag["x"]
            d, extra, g = self.predict(tag)
            self.W.learn(tgt, M=c.eta_reward * r, extra=extra, cols=[cs])
            self.stats["reward_learn"] += 1
        self.own_tags = []

    # ================================================================== tick
    def step(self, obs):
        c = self.c
        win = obs["ear"]; eye = obs["eye"]; r = obs.get("reward", 0.0)
        E = float(win.sum())
        f = self.ear.feat0(win)
        # own voice in the ear: the mouth was active within the last 3 ticks (efference copy), not the CD timer
        own = bool(self.body.SPK or (self.t - self.t_mouth) <= 3)
        self.babble_on = c.babble
        if not hasattr(self, "line_reset"):
            self.line_reset = True; self.line_words = np.zeros(NLM, np.float32)
            self.line_words_q = np.zeros(NLM, bool); self.rec_words = np.zeros(NLM, bool)
            self.line_seq = []; self.rec_seq = []; self.rec_feats = None
            self.feats = np.zeros((self.n_slots, self.n_feat), np.float32); self.point = np.zeros(self.n_slots, bool)
        if E > 0 and self.E_prev == 0 and not own:
            self.other_line_onset()
        if E == 0 and self.E_prev > 0 and not own and self.heard_line:
            self.other_line_end(eye)
        j, cs = self.syl_detect(f)
        if own:
            for (tc, y) in list(self.pending_own):
                if tc == self.t:
                    self.pending_own.remove((tc, y))
                    if cs >= c.th_det and not c.freeze:
                        a = np.zeros(NSM, bool); a[j] = True; b = np.zeros(self.body.NP, bool); b[y] = True
                        self.Mir.learn(a, b, M=1.0); self.stats["mir_learn"] += 1
                        self.syl_learn(f, j, c.eta_syl * 0.5)
                        self.stats["own_det"] += 1
            self.pending_own = [(tc, y) for tc, y in self.pending_own if tc > self.t]
            for (tc, ms) in list(self.pending_trial):
                if tc != self.t or c.freeze:
                    continue
                jj = j if cs >= c.th_det else -1
                if jj < 0 and c.babble_recruit and cs < c.th_new and float(win.max()) >= 5 and self.syl_n < NSM:
                    # a clear own syllable that matches no known unit: recruit it (the infant's own repertoire)
                    jj = self.syl_recruit(f); self.stats["self_recruit"] = self.stats.get("self_recruit", 0) + 1
                if jj < 0:
                    continue
                if c.prog_cap and self.body.n_prog:
                    v_ = np.zeros(self.body.prog.shape[1], bool); v_[list(ms)] = True
                    exists = bool((self.body.prog[:self.body.n_prog] == v_).all(1).any())
                    if not exists and int((self.Mir.w[jj, :self.body.n_prog] > 0.3).sum()) >= c.prog_cap:
                        self.stats["capped"] = self.stats.get("capped", 0) + 1
                        continue                      # novelty-gated: this syllable already has enough programs
                y = self.body.add_program(ms)
                if y >= 0:
                    a_ = np.zeros(NSM, bool); a_[jj] = True; b_ = np.zeros(self.body.NP, bool); b_[y] = True
                    self.Mir.learn(a_, b_, M=1.0); self.stats["kept"] = self.stats.get("kept", 0) + 1
            self.pending_trial = [(tc, ms) for tc, ms in self.pending_trial if tc > self.t]
        else:
            if cs >= c.th_det:
                if not c.freeze:
                    self.syl_learn(f, j, c.eta_syl)
                self.on_heard(j, eye)
            elif self.t == self.theta_next and E > 0:
                if self.syl_n < NSM and cs < c.th_new and not c.freeze:
                    jj = self.syl_recruit(f)
                    if jj >= 0:
                        self.on_heard(jj, eye)
            if self.t == self.theta_next:
                self.theta_next += 3
        if r != 0:
            self.on_reward(r)
        self.dq_tick()
        self.quiet = self.quiet + 1 if (E == 0 and not own) else 0
        blank = not np.asarray(eye).any()
        self.blank_n = self.blank_n + 1 if blank else 0
        if c.babble and self.quiet >= 3 and self.rng.random() < c.p_go:
            self.body.go_req = True; self.self_go = self.t
        elif c.babble_idle and not c.freeze and self.blank_n == 3 and self.quiet >= 3 and self.rng.random() < c.p_go_idle:
            # lifelong vocal play: nothing shown, nobody speaks; started at the 3rd blank tick so that the 6-tick
            # babble (GO, boundary, 3-tick beat) always ends before the next line
            self.body.go_req = True; self.self_go = self.t
        ev = self.body.step(E, self)
        if "rel" in ev:
            self.own_release(ev["rel"], eye)
        if "stop" in ev and not ev.get("first", False):
            self.own_end(eye)
        if self.body.force.any():
            self.t_mouth = self.t
        self.E_prev = E
        self.t += 1
        return self.body.force
