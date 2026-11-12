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

    def support(self, e1):
        return self.W @ e1[self.ca:self.cb].float().clamp_min(0.0)

    def settle(self, e1, stop_share=0.0):
        """e1: the PFC's push above usual [n copies] -> the winning pool's row, or -1"""
        sup = self.support(e1)
        self.last_support = sup
        v = torch.topk(sup, 2)
        s1, s2 = float(v.values[0]), float(v.values[1])
        ok = stop_share < 0.5 and s1 / (1.0 + 4.0 * stop_share) >= self.bar and s1 >= self.ratio * s2
        self.winner = int(v.indices[0]) if ok else -1
        self.n_won += ok
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
        self.n_learn += 1


def install_concept_pools(life, share=0.1, cal_dialogues=30, retire_link=True):
    """the pools for the wired syllables, seeded from the present concept codes; bar and ratio from a frozen stretch of
    never-heard dialogue (ticks without the brake)"""
    b, c = life.brain, life.c
    mot, pfc = b.motor, b.pfc
    sids = list(getattr(life, 'speak_sids', []))
    code, ca, cb = concept_codes(life, sids)
    pat = torch.as_tensor(patterns72(life.body)[sids], dtype=torch.bool, device=DEV)
    cp = ConceptPools(code, pat[:, mot.muscle_of], ca, cb)
    mot.cpools, mot.cpool_sids = cp, sids
    mot.ret_prep = 0.6 * (mot.thr_f + c['b_block'] * mot.n_tonic / (8.0 * mot.m))
    mot.complete = mot.thr_f.max() + c['b_block'] * mot.n_tonic / (8.0 * mot.m) + 0.5
    if retire_link and getattr(pfc, 'G1C', None) is not None:
        pfc.G1C = None                                            # the old link into force copies no longer chooses
    from .world.corpus import script_dialogue
    S = []
    gt = tuple(c.get('gap_turn', (4, 10)))

    def rec(t):                                                   # what the pools would see next tick
        ss = float(mot.stop.on.float().mean())
        if ss < 0.5 and getattr(pfc, 'e1', None) is not None:
            v = torch.topk(cp.support(pfc.e1), 2).values
            S.append((float(v[0]) / (1.0 + 4.0 * ss), float(v[1])))
    for di in range(min(cal_dialogues, life.heldout.n)):
        src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9311, di]), gt, 8, 1, 1)
        life.play([None if x is None else x[:5] for x in src], rec=rec)
    S = np.array(S)
    cp.ratio = float(np.quantile(S[:, 0] / np.maximum(S[:, 1], 1e-9), 0.5))
    lead = S[:, 0] >= cp.ratio * S[:, 1]
    # the bar: among ticks where the top pool leads, the top support passing it on `share` of all unbraked ticks
    q = 1.0 - min(1.0, share / max(lead.mean(), 1e-9))
    cp.bar = float(np.quantile(S[lead, 0], q)) if lead.any() else float('inf')
    life.registry = getattr(life, 'registry', []) + [('concept-space syllable pools (generation 0)', len(sids),
                                                      int(code.any(0).sum()))]
    log(f'concept-space pools: {len(sids)} syllables, {float(code.sum(1).float().mean()):.0f} concept units per code; '
        f'ratio {cp.ratio:.3f}, bar {cp.bar:.4f} (a pool wins on ~{share:.0%} of {len(S)} unbraked ticks)')
    return cp
