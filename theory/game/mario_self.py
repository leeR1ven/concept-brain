"""Mario by itself: a brain plays a level again and again, learning all the while (one life, ARCH 五), by trial and
error (designer 11-08: no data set needed). The buttons it presses go back in as copies, so it sees what it did; the
links into the button copies change only by reward.

Choosing: each button is pressed when the push on its press copies beats the push on its stop copies, plus motor noise
that wanders over about half a second (a young brain tries things, ARCH 五).

Reward (designer 11-08: right is rewarded, wrong is punished, the nearer the goal the better), every frame:
  progress     Mario's move toward the flag this frame, dx / 16 (forward pleases, going back upsets)
  death        upset (DEATH: about what the last second of running earned, so walking into an enemy does not pay)
Standing still is neither.
  record       a win: ground never reached before on this level, per 16 pixels (RECORD), on the long trace
  flag         a win (FLAG), on the long trace
Good or bad changes only the links that carried an action tendency (last layer -> button
copies; designer 11-08), those of the last second or so: each tick's choice leaves a trace on the links from the lit
last-layer units to the chosen side of each button (press when held, stop when not; not jumping is a choice too) and
the opposite on the other side, fading over R_WIN ticks. A reward r moves the links by check_step x r x trace
(x r_pleased when pleased, x (1 + r_upset) when upset, ARCH 四), shared by the lit units in proportion to how seldom
each is lit (a unit lit nearly always, like Mario himself at the centre of the eye, carries no feature: ARCH 三),
slowed by the concave factor. Pressing right all the time is fine if it keeps paying. A win acts on a second, long
trace that fades over LONG_T ticks (10 s), so a long stretch that was mostly right grows, the moments nearer the win
more (designer 11-08; like synaptic tags that wait seconds for dopamine in real brains).
Expectation (MARIO_CRITIC > 0; dopamine in real brains signals reward against what was expected): the brain learns how
good the scene now is (links from the lit heard units); progress and death move the choices only by their surprise,
what came plus what is expected now minus what was expected a moment ago. A death that was coming hurts less, getting
past where one often dies pleases more, and scenes on the way to wins come to please by themselves.
usage: python mario_self.py <brain.pt> <levels e.g. 1-1> <tries> <out.pt> [seed]"""
import collections
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_brain                                         # noqa: E402,F401  (pickles)
from game_brain import DEV, MARIO_KEYS                   # noqa: E402
from game_brain import UNITS_PER_KEY as U                # noqa: E402
from mario_data import Eye                               # noqa: E402
from mario_life import next_push                         # noqa: E402
from mario_play import cells_of, pad_of                  # noqa: E402

torch.set_num_threads(int(os.environ.get('MARIO_THREADS', 4)))
NOISE = float(os.environ.get('MARIO_NOISE', 1.0))      # motor noise (push units; theta1 = 1) that wanders over about
NOISE_T = float(os.environ.get('MARIO_NOISE_T', 30))   # NOISE_T frames
R_WIN = float(os.environ.get('MARIO_R_WIN', 60))       # the trace fades over this many ticks (1 s; ARCH 四: 1-2 s)
DEATH = float(os.environ.get('MARIO_DEATH', 10.0))    # a death costs about what a second of running earns
FLAG = float(os.environ.get('MARIO_FLAG', 5.0))
KEY_R = float(os.environ.get('MARIO_KEY_R', 0.2))      # buttons hear the last layer within this distance of the
                                                        # sheet's centre, around Mario (0: all of it)
RATE_T = float(os.environ.get('MARIO_RATE_T', 2000))  # ticks over which a last-layer unit's firing rate is followed
REWARD = os.environ.get('MARIO_REWARD', '1') == '1'    # 0: the same play and learning without reward (control)
LONG_T = float(os.environ.get('MARIO_LONG_T', 600))   # the long trace fades over 10 s: a win (ground never reached
RECORD = float(os.environ.get('MARIO_RECORD', 1.0))    # before on this level, per 16 pixels; the flag) strengthens a
                                                        # long stretch, the moments nearer the win more (designer 11-08)
GAMMA = float(os.environ.get('MARIO_GAMMA', 0.99))    # how far ahead the expectation looks (per tick: ~1.7 s)
CRITIC = float(os.environ.get('MARIO_CRITIC', 0.1))   # how fast the expectation learns; 0: no expectation (raw reward)
STUCK = float(os.environ.get('MARIO_STUCK', 0.0))     # stuck (no new ground in this try for STUCK_T frames) is wrong
STUCK_T = int(os.environ.get('MARIO_STUCK_T', 30))    # too: this much less each frame, so what it does while stuck fades
PAIN_FULL = os.environ.get('MARIO_PAIN_FULL') == '1'  # with SURPRISE: a pleasure strengthens a choice by its surprise,
                                                        # a punishment weakens what it did, sure or not (doing wrong weakens)
GO_ON = os.environ.get('MARIO_GO_ON') == '1'         # a flag: the next try is the next level; a death: the same
ORDER = [f'{w}-{v}' for w in range(1, 9) for v in range(1, 5)]   # level again (designer 11-08: how many levels it clears)
UNDO_T = float(os.environ.get('MARIO_UNDO_T', 0))    # >0: a death takes back what pleasure strengthened over about the
                                                        # last UNDO_T frames (ARCH 四: running forward and then dying, those
                                                        # seconds still count as wrong; without it the progress paid on the
                                                        # way into an enemy about cancels the death for the choices that
                                                        # mattered, 10-20 frames before)
KEY_SRC = os.environ.get('MARIO_KEY_SRC', '')          # the PFC layer the button copies hear, e.g. L3 (default the
                                                        # last layer, L4: its own usual level (二.4) squeezes out what is
                                                        # always near Mario, an enemy coming at him too; L3 keeps it)
A_REPEAT = os.environ.get('MARIO_A_REPEAT') == '1'   # the body: A held = jump again on landing (the game wants A let
                                                        # go first; without it a held A never jumps again, and a death
                                                        # while holding A teaches 'pressing A was wrong'). How long it is
                                                        # held in the air still sets the height.
GAZE_MEM = os.environ.get('MARIO_GAZE_MEM') == '1'   # the eye is placed by the game's memory (Eye.from_memory)
SURPRISE = os.environ.get('MARIO_SURPRISE') == '1'   # a choice is credited by how far it was from what the brain would
                                                        # have done anyway (chosen minus its own chance of choosing it):
                                                        # a button it presses every time it is lit earns nothing more
NK = len(MARIO_KEYS)


class Learner:
    """learning tick by tick (the copies' own check, the timing rule, the one decay) plus the reward on the trace of
    recent choices"""

    def __init__(self, b):
        self.b, self.p, self.c = b, b.pfc, b.c
        self.K = K = self.p.L41.keys
        hear = getattr(K, 'hear', None)
        self.ids = torch.nonzero(hear).squeeze(1) if hear is not None else torch.arange(K.n_pre, device=DEV)
        self.local = torch.full((K.n_pre,), -1, dtype=torch.int64, device=DEV)
        self.local[self.ids] = torch.arange(len(self.ids), device=DEV)
        self.E = torch.zeros(len(self.ids), K.n_post, device=DEV)     # the trace: heard units x button copies
        self.EL = torch.zeros_like(self.E)                               # the long trace (wins)
        self.EF = torch.zeros_like(self.E)                               # every choice in full, for punishments
        self.EP = torch.zeros_like(self.E)                               # what pleasure strengthened lately (UNDO_T)
        # the expectation (dopamine neurons, designer 11-08: what is linked to wins lights them by itself): how good
        # the scene now is, from the lit heard units; it learns from its own surprises (a 'good' and a 'bad' side,
        # kept here as one signed link per unit)
        self.wV = torch.zeros(len(self.ids), device=DEV)
        self.eV = torch.zeros(len(self.ids), device=DEV)
        self.V, self.pending = 0.0, None
        self.lit = 1.0
        self.src = getattr(self.p.L41, 'src', None) or self.p.L4    # the layer the button copies hear
        if getattr(b, 'rate4', None) is None:              # how often each last-layer unit is lit
            b.rate4 = torch.full((self.p.n,), 0.01, device=DEV)
        self.rate4 = b.rate4
        self.t = 0
        self.total = collections.Counter()

    def tick(self):
        self.p.learn()
        self.rate4 += (self.src.on.float() - self.rate4) / min(RATE_T, self.t + 1.0)   # a plain mean at first
        self.t += 1
        if self.t % self.c['decay_every'] == 0:
            for tb in self.p.plastic():
                tb.decay()

    def did(self, keys, push=None):
        """the last layer now drove this choice of buttons: it joins the trace, older choices fade"""
        self.E *= 1.0 - 1.0 / R_WIN
        self.EL *= 1.0 - 1.0 / LONG_T
        if PAIN_FULL:
            self.EF *= 1.0 - 1.0 / R_WIN
        if UNDO_T:
            self.EP *= 1.0 - 1.0 / UNDO_T
        rows = torch.nonzero(self.src.on).squeeze(1)
        loc = self.local[rows]
        rows, loc = rows[loc >= 0], loc[loc >= 0]
        if rows.numel() == 0:
            return
        sel1 = (1.0 - self.rate4[rows]).clamp_min(0.0)
        V = float((self.wV[loc] * sel1).sum() / sel1.sum().clamp_min(1e-6))
        if self.pending is not None:                       # surprise: what came, plus what is expected now, against
            self.learn(self.pending + GAMMA * V - self.V)  # what was expected a moment ago
        self.eV *= GAMMA * (1.0 - 1.0 / R_WIN)
        self.eV[loc] += sel1 / sel1.sum().clamp_min(1e-6)
        self.V, self.pending = V, 0.0
        k = torch.as_tensor(np.asarray(keys, bool), device=DEV)
        if SURPRISE and push is not None:                  # its chance of pressing: the push against the noise
            pr = 0.5 * (1.0 + torch.erf(torch.as_tensor(push, dtype=torch.float32, device=DEV) / (NOISE * 2 ** 0.5)))
            a = 2.0 * (k.float() - pr)                     # +-1 at an even chance, 0 when it was sure anyway
            side = torch.stack([a, -a])
        else:
            side = (torch.stack([k, ~k]).float() - torch.stack([~k, k]).float())  # chosen +1, other side -1
        m = side[:, :, None].expand(2, NK, U).reshape(-1)
        sel = (1.0 - self.rate4[rows]).clamp_min(0.0)[:, None]                     # how seldom each unit is lit
        self.E[loc] += sel * m[None, :] / R_WIN          # each choice is judged by about R_WIN later frames
        self.EL[loc] += sel * m[None, :] / LONG_T
        if PAIN_FULL:
            full = (torch.stack([k, ~k]).float() - torch.stack([~k, k]).float())[:, :, None].expand(2, NK, U).reshape(-1)
            self.EF[loc] += sel * full[None, :] / R_WIN
        self.lit = float(rows.numel())

    def reward(self, r, what, long=False):
        """long: a win, on the long trace; otherwise, with an expectation, it waits to be compared with it"""
        self.total[what] += r
        if not REWARD or r == 0:
            return
        if CRITIC and not long:
            self.pending = (self.pending or 0.0) + r
            return
        self.apply(r, self.EL if long else (self.EF if PAIN_FULL and r < 0 else self.E))

    def end(self):
        """the try is over: nothing more is to be expected"""
        if REWARD and CRITIC and self.pending is not None:
            self.learn(self.pending - self.V)
        self.pending, self.V = None, 0.0

    def learn(self, delta):
        """the surprise moves the expectation and the recent choices"""
        delta = float(np.clip(delta, -2.0 * DEATH, 2.0 * DEATH))
        self.wV += CRITIC * delta * self.eV
        self.total['surprise'] += abs(delta)
        self.apply(delta, self.E)

    def apply(self, r, E):
        if r == 0:
            return
        c, K = self.c, self.K
        g = c['r_pleased'] * r if r > 0 else -(1.0 + c['r_upset']) * (-r)
        w = K.w[self.ids]
        K.w[self.ids] = (w + c['check_step'] * g * K.plast(w) * E / (K.gain * self.lit)).clamp_min(0.0)
        if UNDO_T and g > 0:
            self.EP += g * E / (K.gain * self.lit)

    def undo(self):
        """a death: what pleasure strengthened in the last moments is taken back"""
        c, K = self.c, self.K
        w = K.w[self.ids]
        K.w[self.ids] = (w - c['check_step'] * K.plast(w) * self.EP).clamp_min(0.0)
        self.total['undo'] += float(self.EP.abs().sum())
        self.EP.zero_()

    def clear(self):
        self.E.zero_()
        self.EL.zero_()
        self.EP.zero_()
        self.EF.zero_()
        self.eV.zero_()
        self.V, self.pending = 0.0, None


def body_keys(keys, env, held):
    """what the hands send: with A_REPEAT, A held on the ground right after a frame it was sent is let go for one
    frame, so the next frame starts a new jump"""
    sent = np.asarray(keys, bool).copy()
    if A_REPEAT and sent[0] and held and int(env.ram[0x1D]) == 0:
        sent[0] = False
    return sent


def one_try(env, b, learner, rng):
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    b.reset()
    learner.clear()                                        # a new try: the last one's choices are not judged by it
    learner.total.clear()
    eye = Eye()
    keys = np.zeros(NK, bool)
    best, t, x0, still = 0, 0, None, 0
    z = NOISE * rng.standard_normal(NK)
    pressed = np.zeros(NK)
    last = collections.deque(maxlen=30)                    # the last half second: what it did just before the end
    held = False
    while True:
        if GAZE_MEM:
            eye.from_memory(env.ram)
        b.step(cells_of(obs, eye), keys, learner.tick)
        z += -z / NOISE_T + NOISE * np.sqrt(2.0 / NOISE_T) * rng.standard_normal(NK)
        push = next_push(b)
        keys = push + z > 0
        learner.did(keys, push)
        pressed += keys
        last.append((keys, push))
        sent = body_keys(keys, env, held)
        held = bool(sent[0])
        obs, _, done, info = env.step(pad_of(sent))
        t += 1
        x = info['x_pos']
        if x0 is not None:                                 # toward the flag pleases, back upsets
            learner.reward(float(np.clip(x - x0, -8, 8)) / 16.0, 'progress')
        x0 = x
        still = 0 if x > best else still + 1
        best = max(best, x)
        if STUCK and still > STUCK_T:
            learner.reward(-STUCK, 'stuck')
        while best >= learner.record + 16:                 # ground never reached before on this level: a win
            learner.record += 16
            learner.reward(RECORD, 'record', long=True)
        if info['flag_get']:
            learner.reward(FLAG, 'flag', long=True)
            learner.end()
            return best, True, t, False, pressed / t, last
        if done:                                           # death (the game clock running out kills him too)
            learner.reward(-DEATH, 'death')
            if UNDO_T:
                learner.undo()
            learner.end()
            return best, False, t, True, pressed / t, last


def main():
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
    path, levels, tries, out = sys.argv[1], sys.argv[2].split(','), int(sys.argv[3]), sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = np.random.default_rng(seed)
    b = torch.load(path, weights_only=False, map_location=DEV)
    if KEY_R:
        game_brain.keys_hear_centre(b, KEY_R)
    game_brain.keys_hear_all(b)
    if KEY_SRC and getattr(b.pfc.L41, 'src_name', '') != KEY_SRC:   # the buttons start hearing another layer: what
        b.pfc.L41.src, b.pfc.L41.src_name = getattr(b.pfc, KEY_SRC), KEY_SRC   # was learned from the old one means
        b.pfc.L41.keys.w.zero_()                           # nothing there, and its firing rates are new
        b.rate4 = None
        if KEY_R:
            game_brain.keys_rehear(b, KEY_R)
    b.pfc.adapt = True
    b.pfc.L41.check_keys = False                           # its own presses are not the world teaching
    if os.environ.get('MARIO_STEP'):                       # a different change per reward (parameter sweeps)
        b.c['check_step'] = float(os.environ['MARIO_STEP'])
    budget = int(os.environ.get('MARIO_FRAMES', 0))        # >0: stop after this many frames in all (sweeps)
    learner = Learner(b)
    records, envs = {}, {}                                 # each level's ground reached so far; its emulator

    def env_of(lv):
        if lv not in envs:
            envs[lv] = SuperMarioBrosEnv(rom_mode='vanilla', target=tuple(int(v) for v in lv.split('-')))
        return envs[lv]
    cur = ORDER.index(levels[0])
    print('step', b.c['check_step'], 'frames', budget, 'critic', CRITIC, 'gamma', GAMMA, 'long', LONG_T, 'record', RECORD, 'reward', REWARD, 'window', R_WIN, 'death', DEATH, 'flag', FLAG, 'noise', NOISE, 'over', NOISE_T, 'key r', KEY_R, 'surprise', SURPRISE, 'go on', GO_ON, 'stuck', STUCK, STUCK_T, 'pain full', PAIN_FULL, 'gaze mem', GAZE_MEM, 'undo', UNDO_T, 'keys hear', KEY_SRC or 'L4', 'A repeat', A_REPEAT, flush=True)
    import time
    t0, frames = time.time(), 0
    for i in range(tries):
        if budget and frames >= budget:
            break
        for lv in ([ORDER[cur]] if GO_ON else levels):
            learner.record = records.get(lv, 40)           # 40: where Mario starts
            x, flag, t, died, pr, last = one_try(env_of(lv), b, learner, rng)
            records[lv] = learner.record
            frames += t
            if GO_ON and flag and cur + 1 < len(ORDER):
                cur += 1
            w = learner.K.w[learner.ids]
            gw = [float(w[:, j * U:(j + 1) * U].mean()) for j in range(2 * NK)]
            L = np.array([k for k, _ in last], float)
            print(f'SELF try {i} {lv} x {x} flag {flag} died {died} frames {t} progress {learner.total["progress"]:+.0f}'
                  f' total {frames} secs {time.time() - t0:.0f}'
                  f' record {learner.record} expect {float(learner.wV.mean()):+.2f}'
                  f' | pressed ' + ' '.join(f'{v:.2f}' for v in pr) +
                  f' | last 0.5 s: jump {L[:, 0].mean():.2f} right {L[:, 3].mean():.2f} | links ' +
                  ' '.join(f'{p:.2f}/{q:.2f}' for p, q in zip(gw[:NK], gw[NK:])), flush=True)
        if (i + 1) % 25 == 0 or i == tries - 1:
            torch.save(b, out)


if __name__ == '__main__':
    main()
