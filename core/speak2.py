"""The speech link that reads the PFC's tendency and learns from the child's own voice (11-12 night; Fable a37 b, a38).

ARCH 三 运动区: the PFC -> force path only gives a tendency; the motor area settles on the strongest whole syllable,
or says nothing. The 11-11 link (speak.SpeechLink) read the binary thoughts through codes measured once at wiring;
by 11-12 only 36 % of those codes were still lit (the concept area keeps learning), and a handful of scattered thought
units rarely added up to one syllable (2.9 % of ticks), so the child babbled or kept silent.

- Input: the tendency of every concept copy - the PFC's push above the copy's own usual level, clamped at 0 (the
  push of the tick before, i.e. what the PFC expects to hear now).
- Links concept copy -> force copy, never changing sign: excitatory E >= 0 and an inhibitory partner I >= 0 (ARCH
  二.1), as the graded table.
- Generation 0 (registered): re-wired from the concept codes as they are now (each wired syllable heard alone in the
  adult voices), by the same least-squares fit as speak.wire_speech.
- Learning from the own voice (designer 11-02: "it hears its own voice and links sound and action"): when the mouth
  said a syllable, the next tick the child hears itself; the concept units lit from below then correct their links
  toward the force copies that moved: error = moved - (their summed push), shared by them (graded, like the graded
  table, so a unit shared by many syllables is explained away and the specific units carry the map). Only ticks on
  which the mouth said a syllable teach (a thought not said weakens nothing; ARCH 六).
- Gain: set so the drive passes theta1 only on about the most confident fifth of ticks (没把握时就不说)."""
import numpy as np
import torch

from .gen0 import nnls
from .life import log
from .net import DEV
from .speak import concept_codes, patterns72


class TendencyLink:
    def __init__(self, ca, cb, f0, f1, n, muscle_of, Wf):
        self.ca, self.cb, self.f0, self.f1, self.n = ca, cb, f0, f1, n
        self.E = Wf.clamp_min(0.0).contiguous()          # [concept units, force neurons]
        self.I = torch.zeros_like(self.E)
        self.muscle_of = muscle_of
        self.gain, self.step = 1.0, 0.02
        self.tendency = True
        self.n_learn = 0

    def drive(self, e1):
        """e1: the PFC's push above usual, float [n] (or a bool thought vector, for the old call) -> push [n]"""
        out = torch.zeros(self.n, device=DEV)
        x = e1[self.ca:self.cb].float().clamp_min(0.0)
        y = self.gain * (x @ (self.E - self.I))
        g = getattr(self, 'grade', None)
        out[self.f0:self.f1] = y if g is None else y * g
        return out

    def learn(self, heard, moved):
        """heard: bool [concept units] lit from below by the own voice; moved: bool [force neurons] that said it"""
        a = torch.nonzero(heard).squeeze(1)
        if a.numel() == 0:
            return
        push = (self.E[a] - self.I[a]).sum(0)
        err = moved.float() - push
        d = (0.5 * self.step / a.numel()) * err[None, :]
        self.E[a] = (self.E[a] + d).clamp_min(0.0)
        self.I[a] = (self.I[a] - d).clamp_min(0.0)
        self.n_learn += 1


def install_tendency_speech(life, cal_dialogues=30, top_share=0.2):
    """replaces the PFC's speech link (pfc.G1C) by a TendencyLink, wired from the present concept codes; the gain is
    set on a short frozen stretch of held-out dialogue"""
    b, c = life.brain, life.c
    pfc, mot = b.pfc, b.motor
    sids = list(getattr(life, 'speak_sids', []))
    code, ca, cb = concept_codes(life, sids)
    share = code.float().mean(0)
    code_f = code & (share <= c.get('speak_unit_max', 0.05))[None, :]
    need = torch.as_tensor(patterns72(life.body)[sids], dtype=torch.float32, device=DEV)
    W = nnls(code_f.float(), need)                                # [concept units, 72]
    Wf = W[:, mot.muscle_of]                                      # [concept units, force neurons]
    _, _, f0, f1 = b.chains.seg('force')[0]
    link = TendencyLink(ca, cb, f0, f1, b.n1, mot.muscle_of, Wf)
    old = getattr(pfc, 'G1C', None)
    pfc.G1C = link
    # the gain: the drive's top force value per tick, on held-out dialogue heard frozen; theta1 at its (1 - top_share)
    # quantile
    from .world.corpus import script_dialogue
    tops = []
    gt = tuple(c.get('gap_turn', (4, 10)))
    for di in range(min(cal_dialogues, life.heldout.n)):   # never-heard dialogues
        src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9311, di]), gt, 8, 1, 1)
        life.play([None if x is None else x[:5] for x in src],
                  rec=lambda t: tops.append(float(link.drive(pfc.e1)[f0:f1].max())))
    q = float(np.quantile(np.array(tops), 1.0 - top_share))
    link.gain = float(c['theta1'] / max(q, 1e-9))
    life.registry = getattr(life, 'registry', []) + [('speech link from the tendency (generation 0, re-wired)',
                                                      len(sids), int(code_f.any(0).sum()))]
    log(f'tendency speech link: {len(sids)} syllables, {int(code_f.any(0).sum())} concept units wired, '
        f'{float(code.sum(1).float().mean()):.0f} per syllable now; gain {link.gain:.3f} (theta1 at the '
        f'{1 - top_share:.0%} quantile of the drive); replaced {type(old).__name__}')
    return link


class SettlePools:
    """The motor area's settling (ARCH 三 运动区落定; Fable a39 A), generation 0 (registered): per wired syllable one
    pool; its support = the share of its own channels' force copies that are lit (prepared by returns), minus half of
    the share lit on channels it does not use (feedforward inhibition, so a syllable whose channels are a subset of
    another's does not tie with it); pools excite themselves and drive one shared inhibitory group that inhibits all
    of them; `passes` settling passes inside the tick (the motor area's fast local loop). A pool that stays on drives
    every force neuron of its own channels past its threshold (completion) and inhibits the rest; none on: nothing
    is said. A return alone only prepares its force neuron (Motor.step with settle: prep x threshold), so a mixture
    can never move muscles directly (the 11-12 garbles)."""

    def __init__(self, pat_f, thr=0.5, g_self=1.0, g_inh=1.5, passes=4, other=0.5):
        self.P = pat_f.float()                                    # [nS, nf] the force neurons of each syllable
        self.n_own = self.P.sum(1).clamp_min(1.0)
        self.n_oth = (1.0 - self.P).sum(1).clamp_min(1.0)
        self.thr, self.g_self, self.g_inh, self.passes, self.other = thr, g_self, g_inh, passes, other
        self.winner = -1

    def settle(self, lit):
        """lit: bool [nf] the force copies lit by returns -> the winning syllable's row, or -1. With channel_of (the
        generator, core/model.py): a pool reads which of its muscles are prepared (any of a muscle's force neurons),
        so an intention is one copy per muscle and fits the thought budget; the winner still drives all its force
        neurons (Pf)"""
        ch = getattr(self, 'channel_of', None)
        if ch is not None:
            lit = torch.zeros(self.P.shape[1], device=lit.device).index_add_(0, ch, lit.float()) > 0
        x = lit.float()
        own = (self.P @ x) / self.n_own
        oth = ((1.0 - self.P) @ x) / self.n_own                  # counted against the pool's own size
        sup = (own - self.other * oth).clamp_min(0.0)
        a = sup > self.thr
        for _ in range(self.passes):
            # each pool is held down by its strongest active competitor (11-12 night: inhibition by the sum of all
            # active pools silenced every case - many syllables share most of their channels, so dozens of pools
            # have high support together)
            act = sup * a.float()
            top2 = torch.topk(act, 2).values
            other_max = torch.where(act >= top2[0], top2[1], top2[0])
            a = sup * (1.0 + self.g_self * a.float()) - self.g_inh * other_max > self.thr
        on = torch.nonzero(a).squeeze(1)
        self.winner = int(on[0]) if on.numel() == 1 else -1
        self.support_max = float(sup.max())
        return self.winner


def install_settle(life, prep=0.6):
    """the settling pools for the wired syllables; returns only prepare (prep x what a force neuron needs)"""
    b = life.brain
    mot = b.motor
    sids = list(getattr(life, 'speak_sids', []))
    pat = torch.as_tensor(patterns72(life.body)[sids], dtype=torch.bool, device=DEV)   # [nS, 72]
    mot.settle = SettlePools(pat[:, mot.muscle_of])
    mot.settle_sids = sids
    c = mot.c
    mot.ret_prep = prep * (mot.thr_f + c['b_block'] * mot.n_tonic / (8.0 * mot.m))
    mot.complete = mot.thr_f.max() + c['b_block'] * mot.n_tonic / (8.0 * mot.m) + 0.5
    life.registry = getattr(life, 'registry', []) + [('motor settling pools (generation 0)', len(sids), 1)]
    log(f'motor settling: {len(sids)} syllable pools; a return prepares {prep:.0%} of a force neuron\'s need')
    return mot.settle


def grade_scale(link, m):
    """strength as a count (Fable a39 A2): the m force copies of a channel need more push the higher their grade"""
    k = (torch.arange(link.f1 - link.f0, device=DEV) % m).float()
    link.grade = 1.0 / (0.5 + 1.5 * k / max(1, m - 1))
    return link.grade
