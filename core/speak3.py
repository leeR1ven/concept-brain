"""Concept-space syllable pools: what the child says (ARCH 三 运动区落定; Fable a39 C, a40 B), generation 0 (registered).

Why: settling in muscle space (core/speak2.py SettlePools) works on one injected syllable, but the PFC's real push is
a superposition over several expected syllables; in muscle space their channels mix and the best onset, nucleus and
offset taken separately make a "ghost" syllable nobody expected (v9, 11-13 night: never the next syllable). Choosing
among whole syllables where the expectation is held - the concept copies - and only then moving the muscles said the
next syllable of a cut line on 2.7 % of lines, 0 % in shuffled order (core/gen/v9_speak_sim.py mode concept).

- One pool per wired syllable, in the motor area (a premotor stage). It reads the PFC's push above usual on the concept
  copies (pfc.e1, the tendency: the PFC only expects, it selects nothing) through its own learned links, its code
  (rows start at the syllable's present concept code, weight 1/size each). support = links . push.
- Competition (one settling of the motor area's local loop): each pool is held down by its strongest competitor; a pool
  stays on if its support is at least `ratio` x the strongest other's and above the bar. The bar is the "say nothing"
  competitor; both are set once, on a frozen stretch of never-heard dialogue, so that a pool wins on about `share` of
  the ticks without the brake (afterwards only learning moves them). The stop neurons (the brake while others speak,
  returns of stopping) hold all pools down.
- The winner drives every force neuron of its own channels (completion) and holds the others down. A return on a
  force copy only prepares its force neuron (as with SettlePools), so a mixture can never move muscles directly.
- Learning the codes from the own voice (DIVA; designer 11-02: hearing oneself links the concept units with the muscles
  just used): the pool that spoke keeps a trace; 2-3 ticks later the concept units lit from below by the own voice
  correct the links by the graded rule: that pool toward 1, the other pools those units push toward 0, the error on
  each pool's summed link from them, shared by them (a unit shared by many syllables is explained away).
- Reward on the pool choice (Fable a40 B4: pool copies and a last-layer -> pool-copy table) is not built yet; the reward
  still replays last layer -> force / stop copies, which no longer choose what is said (registered, open)."""
import numpy as np
import torch

from .life import log
from .net import DEV
from .speak import concept_codes, patterns72


class ConceptPools:
    def __init__(self, code, pat_f, ca, cb, step=0.02):
        c = code.float()
        self.W = (c / c.sum(1, keepdim=True).clamp_min(1.0)).contiguous()   # [nS, concept units] links unit -> pool
        self.P = pat_f.float()                                    # [nS, nf] the force neurons of each syllable
        self.ca, self.cb = ca, cb
        self.bar, self.ratio = float('inf'), 1.5
        self.step = step
        self.winner, self.n_learn, self.n_won = -1, 0, 0
        self.last_support = None
        self.overlaps()

    def overlaps(self):
        """how much each two pools' codes overlap (cosine of the code sets): a rival is weighed by 1 - overlap in the lead
        test (Fable a41 C; ARCH: neurons seldom lit together inhibit each other, related ones excite each other), so
        a tone pair with near-identical codes does not veto the brain's choice exactly when it is most agreed"""
        M = (self.W > 0).float()
        n = M.sum(1).clamp_min(1.0)
        self.O = (M @ M.T) / torch.sqrt(n[:, None] * n[None, :])
        self.O.fill_diagonal_(1.0)

    def lead(self, sup):
        """the top pool, its support, and the strongest rival's support weighed by 1 - overlap with it"""
        i = int(torch.argmax(sup))
        if not getattr(self, 'discount', False):
            v = torch.topk(sup, 2).values
            return i, float(v[0]), float(v[1])
        r = sup * (1.0 - self.O[i])
        r[i] = 0.0
        return i, float(sup[i]), float(r.max())

    def support(self, e1, heard=None):
        x = e1[self.ca:self.cb].float().clamp_min(0.0)
        if heard is not None and getattr(self, 'mask_heard', False):
            x = x * (~heard).float()                          # what arrives from below now meets its expectation:
        return self.W @ x                                     # it is being heard, not something to say (11-13)

    # ---- reward on the choice (Fable a41 B, cheaper first version; pool copies in PFC layer 1 / hippocampus still owed)
    def enable_reward(self, n4, cap=0.3, step=0.5, win=24):
        """a learned table last layer -> pools, changed only by reward. It can only tilt a tendency that is there: it
        counts only for pools pushed above their own usual support, at most +cap x bar (a pool still needs 0.7 of the bar
        from the tendency), and the lead test comes after it."""
        nS = self.W.shape[0]
        self.T = torch.zeros((n4, nS), device=DEV)
        self.mu = torch.zeros(nS, device=DEV)                     # each pool's usual support
        self.rate4 = torch.zeros(n4, device=DEV)                  # each last-layer unit's usual rate
        self.r_cap, self.r_step, self.r_win = cap, step, win
        self.trace, self.tick = [], 0

    def bias(self, l4, sup):
        b = (l4.float() @ self.T).clamp(-self.bar, self.r_cap * self.bar)
        return torch.where(sup > (0.0 if getattr(self, 'lift', False) else self.mu), b, torch.zeros_like(b))

    def reward(self, r):
        """pleased (r > 0): the (state, pool) choices of the last win ticks gain toward the cap, fading with age"""
        if getattr(self, 'T', None) is None or r <= 0:
            return
        for t0, units, pool in self.trace:
            f = self.r_step * r * max(0.0, 1.0 - (self.tick - t0) / self.r_win)
            if units.numel() == 0 or f <= 0:
                continue
            now = float(self.T[units, pool].sum())
            room = max(0.0, 1.0 - now / (self.r_cap * self.bar))
            self.T[units, pool] += f * room * self.r_cap * self.bar / units.numel()
        self.trace = []

    def settle(self, e1, stop_share=0.0, heard=None, l4=None, adapt=False):
        """e1: the PFC's push above usual [n copies]; heard: bool [concept units] lit from below now; l4: the last layer
        now (reward table) -> the winning pool's row, or -1"""
        sup = self.support(e1, heard)
        lift = getattr(self, 'lift', False)
        if lift:                                                  # ARCH 二.4: only support above the pool's own usual
            if adapt:                                             # (11-13 smoke: without it the child drifted to the
                self.us += (sup - self.us) / 1000.0               # commonest syllable, 'de5' 23 % -> 55 % of what it
            sup = sup - self.us                                   # said in 400 dialogues)
        T = getattr(self, 'T', None)
        if T is not None and l4 is not None:
            self.tick += 1
            if adapt and not lift:
                self.mu += (sup - self.mu) / 1000.0
                self.rate4 += (l4.float() - self.rate4) / 1000.0
            sup = sup + self.bias(l4, sup)
        self.last_support = sup
        i, s1, s2 = self.lead(sup)
        ok = stop_share < 0.5 and s1 / (1.0 + 4.0 * stop_share) >= self.bar and s1 >= self.ratio * s2
        self.winner = i if ok else -1
        self.n_won += ok
        if ok and T is not None and l4 is not None:              # credit only units not always on (a context-free
            u = torch.nonzero(l4 & (self.rate4 < 0.5)).squeeze(1)  # habit would be the frequency loop again)
            self.trace = [x for x in self.trace if self.tick - x[0] < self.r_win] + [(self.tick, u, i)]
        return self.winner

    def learn(self, heard, pool):
        """heard: bool [concept units] lit from below by the own voice; pool: the row that spoke"""
        a = torch.nonzero(heard).squeeze(1)
        if a.numel() == 0 or pool < 0:
            return
        push = self.W[:, a].sum(1)                                # each pool's summed link from the heard units
        tgt = torch.zeros_like(push)
        tgt[pool] = 1.0
        rows = torch.nonzero((push > 0) | (tgt > 0)).squeeze(1)
        err = tgt[rows] - push[rows]
        d = (self.step / a.numel()) * err[:, None]
        self.W[rows[:, None], a[None, :]] = (self.W[rows[:, None], a[None, :]] + d).clamp_min(0.0)
        # each pool's links keep their total (1): learning moves weight within a code, toward the units its own voice
        # lights, and never grows a pool that then wins for everything (11-13: life_C's 'wo3' row grew to 41 over
        # 7968 units and said 15331 of 15353 syllables)
        self.W[rows] = self.W[rows] / self.W[rows].sum(1, keepdim=True).clamp_min(1e-9)
        self.n_learn += 1
        if self.n_learn % 500 == 0:                               # the codes' overlaps follow what was learned
            self.overlaps()


def install_concept_pools(life, share=0.1, cal_dialogues=30, retire_link=True, min_ratio=1.05, mask_heard=False, fresh=False, discount=False,
                          lift=False):
    """the pools for the wired syllables, seeded from the present concept codes; bar and ratio from a frozen stretch of
    never-heard dialogue (ticks without the brake)"""
    b, c = life.brain, life.c
    mot, pfc = b.motor, b.pfc
    sids = list(getattr(life, 'speak_sids', []))
    code, ca, cb = concept_codes(life, sids)
    pat = torch.as_tensor(patterns72(life.body)[sids], dtype=torch.bool, device=DEV)
    cp = ConceptPools(code, pat[:, mot.muscle_of], ca, cb)
    cp.mask_heard, cp.fresh, cp.discount, cp.lift = mask_heard, fresh, discount, False
    mot.cpools, mot.cpool_sids = cp, sids
    mot.ret_prep = 0.6 * (mot.thr_f + c['b_block'] * mot.n_tonic / (8.0 * mot.m))
    mot.complete = mot.thr_f.max() + c['b_block'] * mot.n_tonic / (8.0 * mot.m) + 0.5
    if retire_link and getattr(pfc, 'G1C', None) is not None:
        pfc.G1C = None                                            # the old link into force copies no longer chooses
    from .world.corpus import script_dialogue
    S, RAW = [], []
    gt = tuple(c.get('gap_turn', (4, 10)))

    def rec(t):                                                   # what the pools would see next tick
        ss = float(mot.stop.on.float().mean())
        e1 = getattr(mot, 'e1_fresh', None) if fresh else getattr(pfc, 'e1', None)
        if ss < 0.5 and e1 is not None:
            cc = getattr(b, 'concept', None)
            RAW.append((cp.support(e1, None if cc is None else cc.below), ss))
    for di in range(min(cal_dialogues, life.heldout.n)):
        src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9311, di]), gt, 8, 1, 1)
        life.play([None if x is None else x[:5] for x in src], rec=rec)
    if lift:                                                      # each pool's usual support, then above it
        cp.us = torch.stack([x for x, _ in RAW]).mean(0)
        cp.lift = True
    for x, ss in RAW:
        _, s1, s2 = cp.lead(x - cp.us if lift else x)
        S.append((s1 / (1.0 + 4.0 * ss), s2))
    S = np.array(S)
    # the lead the winner needs over the strongest other: the usual lead, but at least min_ratio (11-13: on life_A's
    # overlapping codes the usual lead was 1.004, and a flat mix of ten said one of them 85 % of the time; on life_B
    # 12000 1.05 passes the injection checks, 1.2 silences most mixtures because of near-duplicate codes)
    cp.ratio = max(min_ratio, float(np.quantile(S[:, 0] / np.maximum(S[:, 1], 1e-9), 0.5)))
    lead = S[:, 0] >= cp.ratio * S[:, 1]
    # the bar: among ticks where the top pool leads, the top support passing it on `share` of all unbraked ticks
    q = 1.0 - min(1.0, share / max(lead.mean(), 1e-9))
    cp.bar = float(np.quantile(S[lead, 0], q)) if lead.any() else float('inf')
    life.registry = getattr(life, 'registry', []) + [('concept-space syllable pools (generation 0)', len(sids),
                                                      int(code.any(0).sum()))]
    log(f'concept-space pools: {len(sids)} syllables, {float(code.sum(1).float().mean()):.0f} concept units per code; '
        f'ratio {cp.ratio:.3f}, bar {cp.bar:.4f} (a pool wins on ~{share:.0%} of {len(S)} unbraked ticks)')
    return cp
