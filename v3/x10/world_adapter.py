"""World adapter (CPU, numpy). The X1c/X1f dialogue worlds (world.py, world6.py, unchanged copies) with the binary
render replaced by a graded cochlea [W1-W3]. The brain receives only the A0 bit vector produced here; every label
(who speaks, which syllable, which question) stays in this module and is used only by scorers."""
import numpy as np

import world as W
import world6 as W6

SYL, SID, NSYL = W.SYL, W.SID, W.NSYL
OWN, C1, C2, D1, D2 = W.OWN, W.C1, W.C2, W.D1, W.D2
ADULTS = (C1, C2, D1, D2)


class Cochlea:
    """[W1] graded energy on the world's 180 channels; [W2] own voice content at 0.6; [W3] thermometer pairs.
    A0 sheet: y = channel (0..179), z = level*2 + polarity (0 = up line, 1 = down line). Exactly half on."""

    def __init__(self, ph, c, seed):
        self.ph, self.c = ph, c
        self.L = c['levels_per_s'] * c['s']
        lrng = np.random.default_rng(c['lineage_seed'] + 11)          # lineage-fixed thresholds
        self.th = np.sort(lrng.uniform(c['th_lo'], c['th_hi'], (ph.n_a, self.L)), 1).astype(np.float32)
        self.rng = np.random.default_rng(seed + 31)
        self.NC, self.Z = ph.n_a, 2 * self.L
        self.n = self.NC * self.Z
        # [M4] brake lines: up lines of the four OTHER voices' timbre channels (own timbre left out: birth wiring)
        lines = []
        for v in ADULTS:
            for ch in range(ph.n_c + v * ph.n_vu, ph.n_c + (v + 1) * ph.n_vu):
                lines += [ch * self.Z + lv * 2 for lv in range(self.L)]
        self.brake_lines = np.array(lines, np.int64)

    def energy(self, srcs):
        ph, c, rng = self.ph, self.c, self.rng
        e = np.zeros(self.NC, np.float32)
        for kind, payload, voice, loud in srcs:
            if kind == 's':
                cont = ph.content[payload]
            elif kind == 'm':
                cont = ph.content[payload].copy()
                on = np.nonzero(cont)[0]
                cont[rng.choice(on, 3, replace=False)] = False
            else:
                cont = ph.garble_content(payload)
            g = c['own_level'] if voice == OWN else 1.0
            e[:ph.n_c] += loud * g * cont * rng.uniform(0.85, 1.0, ph.n_c).astype(np.float32)
            e[ph.n_c + voice * ph.n_vu: ph.n_c + (voice + 1) * ph.n_vu] += loud * c['voice_level']
        e += np.abs(rng.normal(0.0, c['noise_sd'], self.NC)).astype(np.float32)
        return np.minimum(e, c['e_max'])

    def code(self, e):
        up = e[:, None] >= self.th
        a0 = np.empty((self.NC, self.L, 2), bool)
        a0[:, :, 0] = up
        a0[:, :, 1] = ~up
        a0 = a0.reshape(-1)
        assert 2 * int(a0.sum()) == a0.size, 'A0 must be exactly half on'
        return a0

    def a0(self, srcs):
        return self.code(self.energy(srcs))


class Env:
    """A running world (World or OverhearWorld) whose sound goes through the cochlea. step(muscles) -> A0 bits."""

    def __init__(self, world, coch):
        self.w, self.coch = world, coch
        self.srcs = []
        world._render = self._capture          # instance attribute shadows World._render

    def _capture(self, srcs):
        self.srcs = list(srcs)
        return np.zeros(self.w.ph.n_a, bool)   # the binary render is not used

    def step(self, muscles):
        self.w.step(muscles)                   # the world's reward is not passed on [R5]
        return self.coch.a0(self.srcs)

    def adult_speaking(self):
        return self.w.cg_now is not None


def make_overhear(ph, coch, seed, **kw):
    return Env(W6.OverhearWorld(ph, seed, **kw), coch)


def make_caregiver(ph, coch, seed, care='real', **kw):
    return Env(W.World(ph, seed, care=care, **kw), coch)


class ScriptedEnv:
    """Probe world: the same physics; an adult script per tick; the child's mouth decoded and heard (unless muted).
    Tracks the child's syllables like World._track (a new entry whenever the decoded key changes)."""

    def __init__(self, ph, coch):
        self.ph, self.coch = ph, coch
        self.reset_track()

    def reset_track(self):
        self.syl, self.events, self.last_key, self.t = [], [], None, 0

    def step(self, muscles, adult=None, extra=(), mute=False):
        srcs = []
        mouth = None if (mute or muscles is None) else self.ph.decode(muscles)
        if mouth is not None:
            srcs.append((mouth[0], mouth[1], OWN, 1.0))
            key = ('s' if mouth[0] == 'm' else mouth[0], mouth[1])
            if key != self.last_key:
                self.syl.append('~' if mouth[0] == 'g' else SYL[mouth[1]])
                self.events.append((self.t, adult is not None, mouth[0]))
            self.last_key = key
        else:
            self.last_key = None
        if adult is not None:
            srcs.append(('s', adult[0], adult[1], adult[2] if len(adult) > 2 else 1.0))
        srcs += list(extra)
        self.t += 1
        return self.coch.a0(srcs)


def line_script(syls, voice, loud=1.0):
    """an adult line as a per-tick script, 3 ticks per syllable (as World._say)"""
    return [(SID[x], voice, loud) for x in syls for _ in range(3)]


def contexts(rng, props):
    """the four preceding contexts of the newborn probes [PLAN 6a]: silence, narration, another question, TV line"""
    x = int(rng.integers(8))
    f2, x2 = int(rng.integers(4)), int(rng.integers(8))
    tv = [SYL[int(rng.integers(NSYL))] for _ in range(3)]
    return [[None] * 20,
            line_script(W.narration(x, props, rng), C2) + [None] * 4,
            line_script(W.question(f2, x2), C2) + [None] * 4 + line_script(W.answer(f2, x2, props), D1),
            line_script(tv, D1, 0.4)]
