"""One life: the world, the birth period (child silent, learning off), the brain, generation-0 wiring, and the loop
that runs the world tick by tick. The brain gets only the input-layer bits (and r)."""
import time

import numpy as np
import torch

from . import gen0
from .brain import Brain
from .net import DEV
from .world import world as W
from .world.adapter import Cochlea, ScriptedEnv, Skin, make_overhear


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class Life:
    def __init__(self, c, seed, birth_dialogues=None, wire=True):
        self.c, self.seed = c, seed
        self.ph = W.Physics()
        self.coch = Cochlea(self.ph, c, seed)
        self.skin = Skin(c['m'] * c['s'], self.ph.need_mat)
        self.env = make_overhear(self.ph, self.coch, seed)          # ARCH 五 训练顺序 2: overhear others' dialogues
        nb = c['birth_dialogues'] if birth_dialogues is None else birth_dialogues
        mute = np.zeros(24, bool)
        A0 = []
        while self.env.w.n_ex < nb:
            A0.append(self.env.step(mute))
        log(f'birth: {len(A0)} ticks, {nb} dialogues')
        self.brain = Brain(c, seed, self.coch, self.skin, np.array(A0))
        del A0
        b = self.brain
        log(f'brain: sound {b.sound.n}, touch {b.touch.n}, PFC {b.n1} x 4, hippocampus {b.hip.nh}, '
            f'motor {b.motor.nf} force; birth density {b.birth_density:.3f}')
        self.probe_env = ScriptedEnv(self.ph, self.coch)
        self.muscles = mute
        self.registry = {}
        self.codes = gen0.sound_codes(b, self.coch)
        if wire:
            self.registry['G1P'] = gen0.wire_g1p(self)
            self.registry['pools'] = gen0.wire_pools(self)
            log(f'generation 0: {self.registry}')

    def tick(self):
        a0 = self.env.step(self.muscles)
        self.muscles = self.brain.step(torch.as_tensor(a0, device=DEV), 0.0).cpu().numpy()
        return a0

    def run_dialogues(self, n, every=None):
        target = self.env.w.n_ex + n
        while self.env.w.n_ex < target:
            self.tick()
            if every is not None:
                every(self)

    def play(self, script, mute=False, think=None, rec=None, learn=False):
        """Probe: a scripted adult source per tick (None = silence). think: {tick: [syllables]} lights those
        syllables' sound copies inside the PFC on that tick (as if thought). Learning off unless learn=True; the
        neuron states are restored afterwards (weights change only when learn=True)."""
        b, env = self.brain, self.probe_env
        was, b.learning = b.learning, learn
        snap = b.snapshot()
        m = np.zeros(24, bool)
        env.reset_track()
        for t, adult in enumerate(script):
            if think and t in think:
                f = torch.zeros(b.n1, dtype=torch.bool, device=DEV)
                for x in think[t]:
                    f[self.codes[x]] = True
                b.pfc.force_int = f
            a0 = env.step(None if mute else m, adult=adult, mute=mute)
            m = b.step(torch.as_tensor(a0, device=DEV), 0.0).cpu().numpy()
            b.pfc.force_int = None
            if rec is not None:
                rec(t)
        b.restore(snap)
        b.learning = was
        return list(env.syl), list(env.events)
