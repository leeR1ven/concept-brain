"""One life: the physics, the cochlea, the birth period (child silent, learning off), the brain, G1, and the loop that
runs a world tick by tick. Everything the brain gets is A0 (and r = 0.0)."""
import numpy as np
import torch

import world as W
from brain import Brain
from common import DEV, log
from wiring import wire_g1
from world_adapter import Cochlea, ScriptedEnv, make_overhear


class Life:
    def __init__(self, c, seed, birth_dialogues=None, wire=True):
        self.c, self.seed = c, seed
        self.ph = W.Physics()                                             # lineage-fixed body and sound physics
        self.coch = Cochlea(self.ph, c, seed)
        self.env = make_overhear(self.ph, self.coch, seed)               # the overhearing world (ARCH 2.11 step 2)
        nb = c['birth_dialogues'] if birth_dialogues is None else birth_dialogues
        A0, speech = [], []
        mute = np.zeros(24, bool)
        while self.env.w.n_ex < nb:                                       # birth: the newborn is silent
            A0.append(self.env.step(mute))
            speech.append(self.env.adult_speaking())
        self.birth_A0 = np.array(A0)
        self.birth_speech = np.array(speech)
        log(f'birth: {len(A0)} ticks, {nb} dialogues')
        self.brain = Brain(c, seed, self.coch, self.birth_A0, self.birth_speech)
        self.brain.calibrate_online(self.birth_A0[-c['calib_online_ticks']:])
        self.registry = {}
        self.codes = None
        self.muscles = mute
        if wire:
            self.registry['G1'], self.codes = wire_g1(self.brain, self.coch, self.ph)
            log(f"G1: {self.registry['G1']}")
        self.probe_env = ScriptedEnv(self.ph, self.coch)

    def tick(self):
        """one tick of the running world: the brain's muscles go into the world, the world's sound comes back"""
        a0 = self.env.step(self.muscles)
        m = self.brain.step(torch.as_tensor(a0, device=DEV), 0.0)
        self.muscles = m.cpu().numpy()
        return a0

    def run_dialogues(self, n, record=None):
        target = self.env.w.n_ex + n
        while self.env.w.n_ex < target:
            self.tick()
            if record is not None:
                record(self)
