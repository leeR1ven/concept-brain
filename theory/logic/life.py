"""The logic life: one brain lives through the logic world (world.py), answering with two muscles (YES, NO), rewarded
when the answer fits the sentence's logic and punished when it does not (ARCH 四 reward tool; 五 the world keeps
teaching). The stage goes up when most recent answers are right.

The brain is the game brain (a sense area made by the one generator, chain copies, the muscles as press / stop copies,
the PFC; pfc_depth layers). The muscles move with motor noise (a young brain tries things). The reward changes only
the links from the layer the muscle copies hear (the last by default) to the muscle copies, by a trace of the choices
(press group when pressed, stop group when not; ARCH 四):
  dw = step x g x plast(w) x trace / (gain x lit),   g = r_pleased x r (pleased) or -(1 + r_upset) x |r| (upset)
With a trace of one frame (the choice at the answer) and full credit, one reward moves that input's push by about
3.6 x step (step 0.5: about twice the motor noise, so one mistake should be enough; designer 11-09: compute it, fail
once, then not again).
  python life.py build <net.pt> [seed]                      a newborn (LOGIC_DEPTH: PFC layers, default 4, as the
                                                             real cortex's 3-4 steps; 2 is too few, ARCH 三)
  python life.py live <net.pt> <out.pt> <sentences> [seed]  (LOGIC_STEP, LOGIC_R_WIN, LOGIC_CREDIT full|surprise,
                                                             LOGIC_NOISE, LOGIC_NOISE_T, LOGIC_KEY_SRC L4|L3|L2, LOGIC_STAGE,
                                                             LOGIC_PFC_LEARN 0|1, LOGIC_LOOP_X)"""
import collections
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'game'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import game_brain                                  # noqa: E402  (pickles)
from game_brain import DEV, GameBrain, UNITS_PER_KEY as U   # noqa: E402
from core.config import config                     # noqa: E402
import world as W                                  # noqa: E402

STEP = float(os.environ.get('LOGIC_STEP', 0.5))      # the reward's step (see above)
R_WIN = float(os.environ.get('LOGIC_R_WIN', 1))      # frames the trace lasts (1: only the choice at the answer)
CREDIT = os.environ.get('LOGIC_CREDIT', 'mixed')     # full: chosen side +1, other -1; surprise: chosen - its chance;
                                                    # mixed: a right answer by its surprise (sure and right: nothing
                                                    # changes), a wrong one in full (sure and wrong is still corrected)
NOISE = float(os.environ.get('LOGIC_NOISE', 1.0))    # motor noise (push units)
NOISE_T = float(os.environ.get('LOGIC_NOISE_T', 4))  # frames over which it wanders
KEY_SRC = os.environ.get('LOGIC_KEY_SRC', 'L4')      # the layer the muscle copies hear
PFC_LEARN = os.environ.get('LOGIC_PFC_LEARN', '1') == '1'   # 0: the PFC's own links stay as born (only the reward learns)
LOOP_X = float(os.environ.get('LOGIC_LOOP_X', 1.0))  # the birth loop's strength x this (holding a thought)
W_REF = float(os.environ.get('LOGIC_W_REF', 0))      # >0: the reward links' own working size in the concave factor
                                                        # (push ~2 needs a press-stop difference ~2 per unit at gain
                                                        # 0.0025 x ~400 lit: w_ref 1 makes them stiff where they work)
CANCEL = os.environ.get('LOGIC_CANCEL') == '1'      # take back what a unit's press and stop links share (the push,
                                                        # their difference, stays): growth both ways stiffens them
COMPETE = os.environ.get('LOGIC_COMPETE', '0') == '1'   # 1: when both would move only the more driven one does. Off
                                                        # (designer 11-09): each muscle has its own press and stop
                                                        # (inhibiting) neurons; none, one or both is four outputs, and
                                                        # only exactly the right muscle is right
R_OK = float(os.environ.get('LOGIC_R_OK', 1.0))      # right: exactly the right muscle
R_BAD = float(os.environ.get('LOGIC_R_BAD', -1.0))   # wrong: the wrong muscle, both, or none (designer 11-09: reward
                                                    # more, punish less; most answers of a young brain are wrong)
R_SURE = float(os.environ.get('LOGIC_R_SURE', R_BAD))   # very wrong: it was sure of the wrong muscle
SURE = float(os.environ.get('LOGIC_SURE', 1.28))    # sure = that muscle's push above this x the noise (90 %)
R_NONE = R_BAD
NK = 2
GEN0 = os.environ.get('LOGIC_GEN0') == '1'           # generation 0 (ARCH 五, registered): the YES word <-> the YES muscle,
                                                    # NO <-> NO: each copy of a word lit from inside (thought of; a word
                                                    # heard is not thought: ARCH 六) pushes its muscle's press group;
                                                    # the tonic blockers hold every muscle still by default (ARCH 三);
                                                    # the innate brake: each lit input cell pushes the stop groups
G0, BLOCK = 8.8, 2.0                                # per share of the thought budget: the right word took ~31 % of it
                                                    # (-> +0.75), the wrong ~13 % (-> -0.86), measured (4x brain)
DIALOGUE = os.environ.get('LOGIC_DIALOGUE') == '1'   # half the time others ask and answer (it listens), half the time
                                                    # it is asked; the word it says shows on the sheet (it hears itself)


def next_push(b):
    """the push the heard layer gives the muscle copies at the next tick: press minus stop (above their usual); with
    generation 0 also the thought words, the tonic blockers and the brake (all fixed links)"""
    p, nv = b.pfc, b.ch.n_vision
    d = p.loop.drive(p.L4.on) + p.L41.drive(p.L4.on)
    e = (d - p.usual)[nv:].reshape(2 * NK, U).mean(1)
    push = (e[:NK] - e[NK:]).cpu().numpy()
    if GEN0:
        thought = np.array([int(p.x_int[ix].sum()) for ix in b.gen0_words])
        push = push + G0 * thought / max(float(p.budget), 1.0) - BLOCK - b.g_brake * b.input_lit
    return push


class Learner:
    """the reward on the trace of recent choices; tick-by-tick learning of the PFC as always"""

    def __init__(self, b):
        self.b, self.p, self.c = b, b.pfc, b.c
        self.K = b.pfc.L41.keys
        self.src = getattr(b.pfc.L41, 'src', None) or b.pfc.L4
        self.E = torch.zeros(self.K.n_pre, self.K.n_post, device=DEV)    # by surprise (or full)
        self.EF = torch.zeros_like(self.E) if CREDIT == 'mixed' else self.E   # in full, for punishments
        if getattr(b, 'rate4', None) is None:
            b.rate4 = torch.full((self.K.n_pre,), 0.01, device=DEV)
        self.rate, self.t, self.lit = b.rate4, 0, 1.0

    def tick(self):
        if PFC_LEARN:
            self.p.learn()
        self.rate += (self.src.on.float() - self.rate) / min(2000.0, self.t + 1.0)
        self.t += 1
        if self.t % self.c['decay_every'] == 0:
            for tb in self.p.plastic():
                tb.decay()

    def did(self, keys, push):
        self.E *= 1.0 - 1.0 / R_WIN
        if self.EF is not self.E:
            self.EF *= 1.0 - 1.0 / R_WIN
        rows = torch.nonzero(self.src.on).squeeze(1)
        if rows.numel() == 0:
            return
        k = torch.as_tensor(np.asarray(keys, bool), device=DEV)
        full = 2.0 * k.float() - 1.0
        pr = 0.5 * (1.0 + torch.erf(torch.as_tensor(push, dtype=torch.float32, device=DEV) / (NOISE * 2 ** 0.5)))
        a = 2.0 * (k.float() - pr) if CREDIT in ('surprise', 'mixed') else full
        if CREDIT == 'pleased_full':                       # every choice in full, pleased or upset
            a = full
        side = lambda v: torch.stack([v, -v])[:, :, None].expand(2, NK, U).reshape(-1)[None, :]   # noqa: E731
        sel = (1.0 - self.rate[rows]).clamp_min(0.0)[:, None]
        self.E[rows] += sel * side(a) / R_WIN
        if self.EF is not self.E:
            self.EF[rows] += sel * side(full) / R_WIN
        self.lit = float(rows.numel())

    def reward(self, r):
        c, K = self.c, self.K
        g = c['r_pleased'] * r if r > 0 else -(1.0 + c['r_upset']) * (-r)
        E = self.EF if r < 0 else self.E
        K.w = (K.w + STEP * g * K.plast(K.w) * E / (K.gain * self.lit)).clamp_min(0.0)
        if CANCEL:
            h = K.w.shape[1] // 2                      # press copies, then stop copies, pair by pair
            m = torch.minimum(K.w[:, :h], K.w[:, h:])
            K.w[:, :h] -= m
            K.w[:, h:] -= m


def build(out, seed):
    rng = np.random.default_rng(seed)
    G = W.glyphs()
    C, K = W.birth_stream(3000, rng, G)
    c = config(s=1, dz=(2, 4, 8, 16), levels_per_s=4, r_learn=float(os.environ.get('LOGIC_R_LEARN', 0.05)),
               pfc_depth=int(os.environ.get('LOGIC_DEPTH', 4)))
    b = GameBrain(c, seed, C, K, game='logic')
    game_brain.keys_hear_all(b)
    print('newborn: pfc depth', c['pfc_depth'], 'vision units', b.area.n, 'copies', b.ch.n, flush=True)
    torch.save(b, out)


def live(path, out, n, seed):
    rng = np.random.default_rng(seed)
    b = torch.load(path, weights_only=False, map_location=DEV)
    game_brain.keys_hear_all(b)
    if KEY_SRC != 'L4' and getattr(b.pfc.L41, 'src_name', 'L4') != KEY_SRC:
        b.pfc.L41.src, b.pfc.L41.src_name = getattr(b.pfc, KEY_SRC), KEY_SRC
        b.pfc.L41.keys.w.zero_()
        b.rate4 = None
    b.pfc.adapt = True
    b.pfc.L41.check_keys = os.environ.get('LOGIC_CHECK_KEYS') == '1'   # 1: the muscle copies also check what the last
    # layer lit in them, like every copy (designer 11-09: while others talk the brake lights the stop groups, the PFC comes
    # to expect that and keeps still by itself; a tendency, learned). 0: only the reward changes them
    if os.environ.get('LOGIC_L4_USUAL') == '1':
        b.pfc.c['l4_usual'] = True                 # the last layer keeps its usual size in quiet moments
    if LOOP_X != 1.0 and not getattr(b, 'loop_scaled', False):
        b.pfc.loop.w *= LOOP_X
        b.loop_scaled = True
    if W_REF:
        b.pfc.L41.keys.w_ref = W_REF
    L = Learner(b)
    G = W.glyphs()
    b.input_lit = 0.0
    if GEN0 and getattr(b, 'gen0_words', None) is None:
        from listen import word_copies
        b.gen0_words = word_copies(b, G)              # the copies each answer word lights (registered gen-0 links)
        b.g_brake = 5.0 / (0.3 * W.ROWS * W.COLS)     # someone saying a word (30 % of the cells) pushes stop by 5
    stats = collections.defaultdict(lambda: collections.deque(maxlen=200))
    stage = int(os.environ.get('LOGIC_STAGE', 1))
    print(f'right {R_OK} wrong {R_BAD} sure-wrong {R_SURE} (sure at {SURE} x noise) | step {STEP} r_win {R_WIN} credit {CREDIT} noise {NOISE} over {NOISE_T} keys hear {KEY_SRC} depth '
          f'{getattr(b.pfc, "depth", 4)} pfc learns {PFC_LEARN} muscle copies check {b.pfc.L41.check_keys} gen0 {GEN0} dialogue {DIALOGUE} loop x{LOOP_X} w_ref {W_REF or b.c["w_ref"]} cancel {CANCEL} compete {COMPETE} l4 usual {b.pfc.c.get("l4_usual", False)} stage {stage}', flush=True)
    kinds = collections.deque(maxlen=200)           # how the latest answers ended: right, wrong, both, none
    keys, z = np.zeros(NK, bool), NOISE * rng.standard_normal(NK)
    recent = collections.deque(maxlen=40)          # (stage of the sentence, right?) of the latest answers
    by = collections.defaultdict(lambda: collections.deque(maxlen=20))
    t0, first = time.time(), {}
    voice = []                                     # frames of its own word still to show (it hears itself)

    def frame(cells):
        """one frame of life: what it sees (and hears of itself), the brain's tick, the muscles"""
        nonlocal keys, z, voice
        if voice:
            cells = cells | voice.pop(0)
        lit = float(cells.sum())                   # the brake hears the input layer and, a tick later, the next
        b.input_lit = max(lit, getattr(b, 'lit_before', 0.0))   # layer: it holds over the gap between two words
        b.lit_before = lit
        b.step(cells, keys, L.tick)
        z += -z / NOISE_T + NOISE * np.sqrt(2.0 / NOISE_T) * rng.standard_normal(NK)
        push = next_push(b)
        drive = push + z
        keys = drive > 0
        if COMPETE and keys.all():
            keys = drive >= drive.max()
        L.did(keys, push)
        return keys, push

    for i in range(n):
        toks, truth, s = W.sentence(stage, rng)
        if DIALOGUE and i % 2 == 0:                # others ask and answer; it listens
            over = echo = 0
            for cells, phase in W.dialogue(toks, truth, G):
                k, _ = frame(cells)
                over += int(k.any() and phase == 'say')
                echo += int(k.any() and phase == 'answer')
            stats['talks over others'].append(over > 0)
            stats['echoes their answer'].append(echo > 0)
            continue
        answer = None
        for cells, phase in (W.turn(toks, G) if DIALOGUE else W.frames(toks, G)):
            keys, push = frame(cells)
            if phase == 'ask' and answer is None and keys.any():
                answer = 'both' if keys.all() else ('yes' if keys[W.YES] else 'no')
                right = answer == ('yes' if truth else 'no')
                k = W.YES if answer == 'yes' else W.NO
                sure_wrong = not right and answer != 'both' and push[k] > SURE * NOISE
                L.reward(R_OK if right else (R_SURE if sure_wrong else R_BAD))
                if DIALOGUE:                       # it hears itself, softer than others (ARCH 二.6: similar, not
                    own = W.own_voice(G)           # the same; the brake goes by loudness)
                    word = own['YES'] if answer == 'yes' else (own['NO'] if answer == 'no' else own['YES'] | own['NO'])
                    voice = [word] * 3
            if phase == 'rest' and answer is None:
                answer = 'none'
                L.reward(R_NONE)
        right = answer == ('yes' if truth else 'no')
        recent.append((s, right))
        kinds.append('right' if right else ('wrong' if answer in ('yes', 'no') else answer))
        by[s].append(right)
        acc = np.mean([r for _, r in recent])
        if len(recent) == recent.maxlen and acc >= 0.9 and stage not in first:
            first[stage] = i + 1
            print(f'STAGE {stage} learned after {i + 1} sentences (last {recent.maxlen}: {acc:.2f})', flush=True)
            if stage < W.STAGES:
                stage += 1
                recent.clear()
        if (i + 1) % (40 if DIALOGUE else 20) == 0:
            print(f'LIFE {i + 1} stage {stage} right (last 40) {acc:.2f} | by stage ' +
                  ' '.join(f'{k}:{np.mean(v):.2f}' for k, v in sorted(by.items())) +
                  f' | last 200: ' + ' '.join(f'{k} {kinds.count(k)}' for k in ('right', 'wrong', 'both', 'none')) +
                  ''.join(f' | {k} {np.mean(v):.2f}' for k, v in stats.items()) +
                  f' | {" ".join(toks)} -> {answer} | {time.time() - t0:.0f} s', flush=True)
        if (i + 1) % 500 == 0 or i == n - 1:
            torch.save(b, out)
    print('stages learned at sentence:', first, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        live(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 0)
