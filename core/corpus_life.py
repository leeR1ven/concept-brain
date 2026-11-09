"""One life in the big dialogue world: real Chinese dialogues (LCCC, converted to toned syllables) spoken by four
synthetic adult voices; the child overhears (ARCH 五 训练顺序 2). The brain is the same core brain; only the world
differs. Data files (not in the repo): <data>/birth.npz, train.npz, heldout.npz (core/world/corpus.py format)."""
import os

import numpy as np
import torch

from .brain import Brain
from .life import log
from .net import DEV
from .world import phon
from .world.adapter import Skin
from .world.body import Body
from .world.corpus import DialogueWorld, Dialogues, script_dialogue
from .world.voice import ADULTS, Cochlea, Voice


class CorpusLife:
    """(next_ranks: the next-syllable test, see there)"""
    def __init__(self, c, seed, data):
        self.c, self.seed = c, seed
        bases = phon.load_inventory()
        self.voice = Voice(bases)
        self.coch = Cochlea(self.voice, c, seed)
        self.body = Body(bases)
        self.skin = Skin(c['m'] * c['s'], self.body.need_mat)
        self.train = Dialogues.load(os.path.join(data, 'train.npz'))
        self.heldout = Dialogues.load(os.path.join(data, 'heldout.npz'))
        birth = DialogueWorld(Dialogues.load(os.path.join(data, 'birth.npz')), self.coch, seed)
        A0 = []
        while birth.n_ex < c['birth_dialogues']:
            A0.append(birth.step())
        log(f'birth: {len(A0)} ticks, {birth.n_ex} dialogues')
        self.brain = Brain(c, seed, self.coch, self.skin, np.array(A0))
        del A0
        b = self.brain
        log(f'brain: sound {b.sound.n}, touch {b.touch.n}, PFC {b.n1} x 4, hippocampus {b.hip.nh}, '
            f'motor {b.motor.nf} force; birth density {b.birth_density:.3f}')
        self.env = DialogueWorld(self.train, self.coch, seed)
        self.muscles = np.zeros(24, bool)

    def tick(self):
        a0 = self.env.step(self.muscles)
        self.muscles = self.brain.step(torch.as_tensor(a0, device=DEV), 0.0).cpu().numpy()

    def play(self, srcs, rec=None):
        """probe: one source (sid, tick, voice, f0_offset, level) or None per tick; learning off, neuron states
        restored afterwards; the child's mouth is muted"""
        b = self.brain
        was, b.learning = b.learning, False
        snap = b.snapshot()
        for t, x in enumerate(srcs):
            b.step(torch.as_tensor(self.coch.a0([] if x is None else [x]), device=DEV), 0.0)
            if rec is not None:
                rec(t)
        b.restore(snap)
        b.learning = was

    def syl(self, x, v):
        """the per-tick sources of one syllable: its 3 ticks, or the whole syllable in one tick (syl_ticks 1)"""
        if self.c.get('syl_ticks', 3) == 1:
            return [(int(x), -1, v, 0.0, 0.0)]
        return [(int(x), k, v, 0.0, 0.0) for k in range(3)]

    def code(self, sid, p_rel=0.5, layers=None):
        """the sound copies a syllable lights (sound area alone, from silence, the four adult voices)"""
        snd, ch = self.brain.sound, self.brain.chains
        n = sum(len(ix) for ix in ch.sound_idx)
        P = torch.zeros(n, device=DEV)
        no_ret = [torch.zeros(k, device=DEV) for k in snd.n]
        saved = [x.clone() for x in snd.state]
        for v in ADULTS:
            snd.state = [torch.zeros_like(x) for x in saved]
            H = []
            ks = self.syl(sid, v)
            for x in [None] * 4 + ks + [None] * 6:
                st = snd.step(torch.as_tensor(self.coch.a0([] if x is None else [x]), device=DEV), no_ret)
                H.append(torch.cat([st[l][ix] for l, ix in enumerate(ch.sound_idx)]).float())
            H = torch.stack(H)
            a = 0
            for l, ix in enumerate(ch.sound_idx):
                if layers is None or l in layers:
                    P[a:a + len(ix)] += H[4 + l: 4 + len(ks) + l, a:a + len(ix)].mean(0) / len(ADULTS)
                a += len(ix)
        snd.state = saved
        _, _, a0, _ = ch.seg('sound')[0]
        return a0 + torch.nonzero(P >= p_rel).squeeze(1)

    def reply_ranks(self, n=60):
        """held-out dialogues: after a turn ends (silence before the reply), the rank of the reply's first syllable
        among the first syllables of all probed replies, by the share of its sound copies lit inside the PFC and the
        hippocampus during the 8 silent ticks. A record of what the model thinks of, not a goal."""
        b, D = self.brain, self.heldout
        items = []
        for d in range(D.n):
            tr = D.turns(d)
            for j in range(len(tr) - 1):
                if len(tr[j]) and len(tr[j + 1]):
                    items.append((tr[j], int(tr[j + 1][0])))
            if len(items) >= n:
                break
        cands = sorted({s for _, s in items})
        if not hasattr(self, '_codes') or set(self._codes) != set(cands):
            self._codes = {s: self.code(s) for s in cands}
        out = {'pfc': [], 'hip': []}
        for prompt, s in items:
            src = [None] * 4 + [y for x in prompt for y in self.syl(x, ADULTS[0])] + [None] * 8
            rec = {'pfc': [], 'hip': []}
            self.play(src, rec=lambda t: (rec['pfc'].append(b.pfc.x_int.clone()), rec['hip'].append(b.hip.h_int.clone())))
            e = len(src) - 8
            for k in out:
                H = torch.stack(rec[k])[e:].any(0)
                sc = {x: float(H[cd].float().mean()) if cd.numel() else 0.0 for x, cd in self._codes.items()}
                out[k].append(sum(v > sc[s] for v in sc.values()) + sum(v == sc[s] for v in sc.values()))
        return {**{k: round(float(np.mean(v)), 1) for k, v in out.items()}, 'chance': (len(cands) + 1) / 2}

    def next_ranks(self, n_dlg=40, n_cands=300):
        """The next-syllable test (11-10): held-out dialogues heard as in life (learning off, states restored). Each
        tick, the last layer's push into each candidate syllable's first-layer sound copies (above their usual level)
        is what the PFC expects to arrive next; the syllable heard this tick is what arrives. The rank of the true
        syllable among the n_cands most common ones (1 = best, chance about n_cands / 2), and the same items ranked by
        counts over the training dialogues heard so far (ngram_ranks: frequency, previous 1, previous 2 syllables)."""
        from .world.corpus import script_dialogue
        b, D, p = self.brain, self.heldout, self.brain.pfc
        if getattr(self, '_ncands', None) != n_cands:
            cnt = np.bincount(self.train.syl.astype(np.int64))
            self._cands = [int(x) for x in np.argsort(-cnt)[:n_cands]]
            codes = [self.code(x, layers=[0]) for x in self._cands]
            L = max(1, max(len(cd) for cd in codes))
            idx = torch.full((n_cands, L), p.n, dtype=torch.int64, device=DEV)
            for i, cd in enumerate(codes):
                idx[i, :len(cd)] = cd
            self._cidx, self._clen = idx, torch.tensor([max(1, len(cd)) for cd in codes], device=DEV).float()
            self._ncands = n_cands
        pos = {x: i for i, x in enumerate(self._cands)}
        ranks, items = [], []
        for d in range(min(n_dlg, D.n)):
            src = script_dialogue(D, d, np.random.default_rng([self.seed, 8800, d]), tuple(self.c.get('gap_turn', (4, 10))),
                                  8, self.c.get('syl_ticks', 3))
            ctx, rows = [], []

            def rec(t):
                x = src[t]
                if x is None or (self.c.get('syl_ticks', 3) == 3 and x[1] != 0):
                    return
                if x[0] in pos:
                    e = p.loop.drive(p.L4.on) + p.L41.drive(p.L4.on) - p.usual
                    e = torch.cat([e, torch.zeros(1, device=DEV)])
                    sc = e[self._cidx].sum(1) / self._clen
                    i = pos[x[0]]
                    ranks.append(float((sc > sc[i]).sum() + 0.5 * ((sc == sc[i]).sum() - 1) + 1))
                    items.append((list(ctx), x[0], self._cands))
                ctx.append(x[0])
            self.play([None if x is None else x[:5] for x in src], rec=rec)
        r = np.array(ranks)
        out = {'brain': r}
        out.update({k: np.array(v) for k, v in ngram_ranks(self.train, items, self.env.n_ex).items()})
        return {k: {'mean rank': round(float(v.mean()), 1), 'top1': round(float((v <= 1).mean()), 3),
                    'top10': round(float((v <= 10).mean()), 3)} for k, v in out.items()} | \
            {'items': len(r), 'chance': (n_cands + 1) / 2}


def ngram_ranks(train, items, upto):
    """yardsticks for next_ranks: the rank of the true next syllable by its count after the same 0, 1 or 2 syllables
    in the first `upto` training dialogues (ties: average rank); candidates = those of next_ranks"""
    import collections
    seq = train.syl[:train.dlg_off[min(upto, train.n)]]
    c1 = collections.Counter(seq.tolist())
    c2 = collections.Counter(zip(seq[:-1].tolist(), seq[1:].tolist()))
    c3 = collections.Counter(zip(seq[:-2].tolist(), seq[1:-1].tolist(), seq[2:].tolist()))
    out = {'freq': [], 'prev 1': [], 'prev 2': []}
    for ctx, true, cands in items:
        for name, f in (('freq', lambda x: c1[x]), ('prev 1', lambda x: c2[(ctx[-1], x)] if len(ctx) >= 1 else c1[x]),
                        ('prev 2', lambda x: c3[(ctx[-2], ctx[-1], x)] if len(ctx) >= 2 else 0)):
            sc = np.array([f(x) for x in cands], float)
            st = sc[cands.index(true)]
            out[name].append(float((sc > st).sum() + 0.5 * ((sc == st).sum() - 1) + 1))
    return out
