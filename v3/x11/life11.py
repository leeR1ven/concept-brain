"""X11 life: the X10 brain (brain.py and its modules, unchanged copies) in the X11 world (voice.py cochlea, LCCC
dialogues). Birth: the newborn overhears `birth` dialogues (a separate set) with learning off; the sound thresholds,
chain bands and PFC / hippocampus thresholds are set from that stream exactly as in X10. Then one continuous life of
overhearing with learning on. No speech wiring (overhearing only). Checkpoints: the whole Brain object + world state."""
import os

import numpy as np
import torch

import phon
from body import Body
from brain import Brain
from common import DEV, config, log
from voice import Cochlea, Voice
from world11 import DialogueWorld

# X10 base B + P1 (coordinator 11-02): P1 patch inhibition, reciprocal loop g 1.0, w44 0, learned gains 0.1 (R2),
# A1/A2 chain band upper cut 0.6, medium PFC. X11 world: 4 levels per channel per s, brake scale for this cochlea.
BASE = dict(inhib='P1', loop='recip', g_loop=1.0, w44=0.0, g44=0.1, g_hh=0.1, g_h1=0.1, band_cut_a12=0.6,
            pfc_size='medium', levels_per_s=4, n_brake_per_s=20.0)


def x11_config(s=1, **over):
    c = dict(BASE)
    c.update(over)
    return config(s, **c)


class Life11:
    def __init__(self, c, seed, train, birth, bases, build=True):
        self.c, self.seed, self.bases = c, seed, bases
        self.voice = Voice(bases)
        self.coch = Cochlea(self.voice, c, seed)
        self.body = Body(bases)
        self.world = DialogueWorld(train, self.coch, seed)
        self.muscles = np.zeros(24, bool)
        self.brain = None
        if build:
            bw = DialogueWorld(birth, self.coch, seed + 500)
            A0, sp = [], []
            while bw.n_ex < birth.n:
                A0.append(bw.step())
                sp.append(bw.adult_speaking())
            A0 = np.array(A0)
            log(f'birth: {len(A0)} ticks, {birth.n} dialogues, speech share {np.mean(sp):.2f}')
            self.brain = Brain(c, seed, self.coch, A0, np.array(sp))
            self.brain.calibrate_online(A0[-c['calib_online_ticks']:])
            self.birth_ticks = len(A0)

    def tick(self):
        a0 = self.world.step(self.muscles)
        m = self.brain.step(torch.as_tensor(a0, device=DEV), 0.0)
        self.muscles = m.cpu().numpy()
        return a0

    def run_dialogues(self, n, record=None, until=None):
        """overhear n more dialogues (learning as set); stops early when until() returns True at a dialogue end"""
        target = self.world.n_ex + n
        while self.world.n_ex < target:
            k = self.world.n_ex
            self.tick()
            if record is not None:
                record(self)
            if until is not None and self.world.n_ex > k and until():
                break

    # ---------------------------------------------------------------- checkpoints (server: ~/ckpt/x11_<run>/)
    def save(self, path, extra=None):
        tmp = path + '.tmp'
        torch.save(dict(brain=self.brain, world=self.world.state(), muscles=self.muscles, extra=extra or {}), tmp)
        os.replace(tmp, path)

    @staticmethod
    def load(path, c, seed, train, bases):
        d = torch.load(path, map_location=DEV, weights_only=False)
        lf = Life11(c, seed, train, None, bases, build=False)
        lf.brain = d['brain']
        lf.world.restore(d['world'])
        lf.muscles = d['muscles']
        return lf, d['extra']


def load_bases(here):
    return phon.load_inventory(os.path.join(here, 'data', 'inventory.json'))
