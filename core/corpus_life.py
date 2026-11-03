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

    def code(self, sid, p_rel=0.5):
        """the sound copies a syllable lights (sound area alone, from silence, the four adult voices)"""
        snd, ch = self.brain.sound, self.brain.chains
        n = sum(len(ix) for ix in ch.sound_idx)
        P = torch.zeros(n, device=DEV)
        no_ret = [torch.zeros(k, device=DEV) for k in snd.n]
        saved = [x.clone() for x in snd.state]
        for v in ADULTS:
            snd.state = [torch.zeros_like(x) for x in saved]
            H = []
            for k in [None] * 4 + [0, 1, 2] + [None] * 6:
                st = snd.step(torch.as_tensor(self.coch.a0([] if k is None else [(sid, k, v, 0.0, 0.0)]),
                                              device=DEV), no_ret)
                H.append(torch.cat([st[l][ix] for l, ix in enumerate(ch.sound_idx)]).float())
            H = torch.stack(H)
            a = 0
            for l, ix in enumerate(ch.sound_idx):
                P[a:a + len(ix)] += H[4 + l: 7 + l, a:a + len(ix)].mean(0) / len(ADULTS)
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
            src = [None] * 4 + [(int(x), k, ADULTS[0], 0.0, 0.0) for x in prompt for k in range(3)] + [None] * 8
            rec = {'pfc': [], 'hip': []}
            self.play(src, rec=lambda t: (rec['pfc'].append(b.pfc.x_int.clone()), rec['hip'].append(b.hip.h_int.clone())))
            e = len(src) - 8
            for k in out:
                H = torch.stack(rec[k])[e:].any(0)
                sc = {x: float(H[cd].float().mean()) if cd.numel() else 0.0 for x, cd in self._codes.items()}
                out[k].append(sum(v > sc[s] for v in sc.values()) + sum(v == sc[s] for v in sc.values()))
        return {**{k: round(float(np.mean(v)), 1) for k, v in out.items()}, 'chance': (len(cands) + 1) / 2}
