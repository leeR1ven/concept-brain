"""Speaking whole sentences (designer 11-10): every word has its own muscle.

Generation 0 (ARCH 五, registered): the copies only a word lights (its own copies, think.own_copies), when lit from
inside (thought of), push that word's muscle (G0 x their share of the thought budget); tonic blockers hold every muscle
still by default (BLOCK); the innate brake: what is heard pushes stop (others are louder than its own voice). Each
muscle has its own wandering noise (motor noise: a young brain tries things, and it gives thinking the randomness it
needs: a word said by chance is heard and thought on). When several muscles would move, the strongest wins (the motor
area settles on one word, ARCH 二.7). With one muscle per word (a brain built with LOGIC_MUSCLES=15) it feels the muscle the tick it moves (its copies
stand in for the touch area's force sense, ARCH 二.2), and hears the word a tick later, in its own softer voice on the
sheet. A word is one muscle here, so the motor area's settling (ARCH 二.7) is stood in for by "the strongest wins".

Two tests, nothing learns:
  ask   someone says a sentence (stages 1..LOGIC_STAGE), then quiet: the first word it says, right if it is the answer
  free  nobody talks: what it says by itself, cut into utterances at pauses: a whole sentence of the language with its
        answer (grammatical), the answer right (true), or a question only; random strings of the same lengths are the
        yardstick
  life  (SPEAK_LIFE=<out.pt>; asks = dialogues): half listening, half asked; it hears and feels itself; learning on
  python speak.py <brain.pt> [asks] [free ticks] [seed]   (LOGIC_SCALE, LOGIC_TOK, LOGIC_GAP, LOGIC_WAIT as it heard;
         SPEAK_CHOICE acc|tick, SPEAK_G0, SPEAK_LEAK, SPEAK_INHIB, SPEAK_THETA, SPEAK_STOP, SPEAK_NOISE, ...)"""
import collections
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'game'))
import game_brain                                  # noqa: E402
from game_brain import DEV                         # noqa: E402
import world as W                                  # noqa: E402
from think import own_copies                       # noqa: E402

G0 = float(os.environ.get('SPEAK_G0', 8.8))
BLOCK = float(os.environ.get('SPEAK_BLOCK', 2.0))
NOISE = float(os.environ.get('SPEAK_NOISE', 1.0))
NOISE_T = float(os.environ.get('SPEAK_NOISE_T', 4))
TOK = int(os.environ.get('LOGIC_TOK', 1))
GAP = int(os.environ.get('LOGIC_GAP', 0))
WAIT = int(os.environ.get('LOGIC_WAIT', 3))
PAUSE = 3                                          # quiet ticks that end an utterance
MARGIN = float(os.environ.get('SPEAK_MARGIN', 0.5))   # the motor area settles on a word only if it clearly leads
                                                    # (ARCH 二.7: one word, else silence); stood in for here
CORRECT = float(os.environ.get('SPEAK_CORRECT', 0.2))   # life: how often the other then says the right answer (at
CORRECT0 = float(os.environ.get('SPEAK_CORRECT0', CORRECT))   # the start CORRECT0, falling to CORRECT by the end)
LISTEN = int(os.environ.get('SPEAK_LISTEN', 1))     # life: dialogues it only listens to per one it is asked
READ = os.environ.get('SPEAK_READ', 'thought')   # what drives a word's muscle (Mouth.shares)
CHOICE = os.environ.get('SPEAK_CHOICE', 'acc')     # acc: the motor area's choice builds up over ticks (below); tick:
                                                    # the strongest this tick, if it clearly leads
LEAK = float(os.environ.get('SPEAK_LEAK', 0.3))     # acc: each word's build-up leaks this share per tick
INHIB = float(os.environ.get('SPEAK_INHIB', 0.3))   # acc: and is pressed down by this x the others' build-up
THETA = float(os.environ.get('SPEAK_THETA', 3.0))   # acc: the build-up at which the muscle moves
STOP = float(os.environ.get('SPEAK_STOP', 3.0))     # acc: a word just said: its stop neurons (lit through touch)
STOP_FADE = float(os.environ.get('SPEAK_STOP_FADE', 0.7))   # press it down, fading by this per tick
TAU = float(os.environ.get('SPEAK_TAU', 1000))                                       # ticks over which a word's usual thought share follows (二.4)
WORDS = list(W.TOKENS)


class Mouth:
    def __init__(self, b, G, rng, learn=False):
        self.b, self.rng, self.learn, self.t = b, rng, learn, 0
        self.own = own_copies(copy.deepcopy(b), G)
        self.voice_of = W.own_voice(G)
        self.g_brake = 5.0 / (0.3 * W.ROWS * W.COLS)          # someone saying a word (30 % of the cells): stop by 5
        self.z = NOISE * rng.standard_normal(len(WORDS))
        self.voice, self.lit_before = [], 0.0
        self.touch = len(b.still()) == len(WORDS)      # one muscle per word: it feels its muscles (their copies)
        self.moves = []                                # the muscle moving on the next ticks (felt at once)
        self.usual = self.usual_shares(G, rng)
        self.acc = np.zeros(len(WORDS))                # each word's build-up in the motor area
        self.stop = np.zeros(len(WORDS))               # each word's stop neurons

    def shares(self):
        """what moves each word's muscle: READ=thought: its own copies lit from inside, as a share of the thought budget;
        READ=push: the last layer's push into its own copies above their usual level (graded, not cut by the budget)"""
        p = self.b.pfc
        if READ == 'push':
            return np.array([float(p.e1[ix].mean()) if len(ix) else 0.0 for ix in self.own.values()])
        return np.array([float(p.x_int[ix].sum()) for ix in self.own.values()]) / max(float(p.budget), 1.0)

    def usual_shares(self, G, rng, n=100):
        """each word's usual thought share (ARCH 二.4: only what is above a unit's own usual level counts): over
        dialogues heard and the quiet between them, no learning"""
        bb = copy.deepcopy(self.b)
        keep, self.b = self.b, bb
        tot, k = np.zeros(len(WORDS)), 0
        for _ in range(n):
            toks, truth, _ = W.sentence(int(os.environ.get('LOGIC_STAGE', 3)), rng)
            for cells, _ in W.dialogue(toks, truth, G):
                bb.step(cells, bb.still())
                tot += self.shares()
                k += 1
        self.b = keep
        return tot / k

    def tick(self, cells):
        """one tick: what it sees and hears of itself, the brain, the muscles; returns the word said (or None)"""
        b = self.b
        if self.voice:
            cells = cells | self.voice.pop(0)
        lit = float(cells.sum())
        heard = max(lit, self.lit_before)                     # the brake holds over the gap between two words
        self.lit_before = lit
        keys = b.still()
        if self.moves:                                        # the muscle moving now: felt this tick (touch area,
            k = self.moves.pop(0)                             # ARCH 二.2), heard only on the next
            if self.touch:
                keys[k] = True
        b.step(cells, keys, b.pfc.learn if self.learn else None)
        self.t += 1
        sh = self.shares()
        if self.learn:                                        # it lives: the usual levels follow, the one decay
            self.usual += (sh - self.usual) / TAU
            if self.t % b.c['decay_every'] == 0:
                for tb in b.pfc.plastic():
                    tb.decay()
        if CHOICE == 'acc':
            # the motor area's choice (Usher & McClelland 2001; Wang 2002; competitive queuing, Bohland et al. 2010):
            # each word's force group builds up what its word gets above usual over ticks (its own recurrent links),
            # leaks, and is pressed down by the others' build-up (shared inhibitory neurons) and by the brake (others
            # talking) and its own stop neurons (lit through touch when it was just said). The first to reach THETA
            # moves; a small, steady lead wins this way where one tick's absolute level does not.
            # Stood in for by this mean-field; in the brain: binary force groups with recurrent links (ARCH 二.7).
            inp = G0 * (sh - self.usual) - self.g_brake * heard - self.stop + NOISE * self.rng.standard_normal(len(WORDS))
            others = self.acc.sum() - self.acc
            self.acc = np.maximum(0.0, (1.0 - LEAK) * self.acc + inp - INHIB * others)
            self.stop *= STOP_FADE
            k = int(np.argmax(self.acc))
            if self.acc[k] < THETA:
                return None
            self.acc[k] = 0.0
            self.stop[k] += STOP
            w = WORDS[k]
            self.moves = [k] * TOK
            self.voice = [np.zeros_like(cells)] + [self.voice_of[w]] * TOK + [np.zeros_like(cells)] * GAP
            return w
        push = G0 * (sh - self.usual) - BLOCK - self.g_brake * heard
        self.z += -self.z / NOISE_T + NOISE * np.sqrt(2.0 / NOISE_T) * self.rng.standard_normal(len(WORDS))
        drive = push + self.z
        o = np.argsort(drive)[::-1]
        if drive[o[0]] > 0 and drive[o[0]] - drive[o[1]] > MARGIN:
            w = WORDS[int(o[0])]
            self.moves = [WORDS.index(w)] * TOK
            self.voice = [np.zeros_like(cells)] + [self.voice_of[w]] * TOK + [np.zeros_like(cells)] * GAP
            return w
        return None


def frames(toks, G):
    blank = np.zeros((W.ROWS, W.COLS), bool)
    out = []
    for t in toks:
        out += [G[t]] * TOK + [blank] * GAP
    return out + [blank] * WAIT


def judge(u):
    """an utterance -> ('full', true?) for a whole sentence with its answer, ('question', None) for a question alone,
    (None, None) otherwise"""
    TV, NM, AN = ('T', 'F'), W.NAMES, ('YES', 'NO')
    q, a = (u[:-1], u[-1]) if u and u[-1] in AN else (u, None)
    val = None
    if len(q) == 2 and q[0] in TV and q[1] == '?':
        val = q[0] == 'T'
    elif len(q) == 3 and q[0] == 'NOT' and q[1] in TV and q[2] == '?':
        val = q[1] == 'F'
    elif len(q) == 4 and q[0] in TV and q[1] in ('AND', 'OR') and q[2] in TV and q[3] == '?':
        x, y = q[0] == 'T', q[2] == 'T'
        val = (x and y) if q[1] == 'AND' else (x or y)
    elif len(q) == 4 and q[0] in NM and q[1] == 'SAME' and q[2] in NM and q[3] == '?':
        val = q[0] == q[2]
    elif len(q) == 6 and q[0] in NM and q[1] == 'IS' and q[2] in TV and q[3] == '.' and q[4] == q[0] and q[5] == '?':
        val = q[2] == 'T'
    if val is None:
        return None, None
    if a is None:
        return 'question', None
    return 'full', (a == 'YES') == val


def share_test(b0, G, n, stage, rng):
    """no muscles: after a question, tick by tick, the thought share of the right answer word, the wrong one, and the
    largest of the other words; and in silence, the largest share of any word (to set G0 and BLOCK from)"""
    b = copy.deepcopy(b0)
    m = Mouth(b, G, rng)
    blank = np.zeros((W.ROWS, W.COLS), bool)

    def shares():
        return m.shares() - m.usual
    T = WAIT + 8
    right, wrong, other = np.zeros(T), np.zeros(T), np.zeros(T)
    top, beats = np.zeros(T), np.zeros(T)
    for i in range(n):
        toks, truth, s = W.sentence(stage, rng)
        for _ in range(6):
            b.step(blank, b.still())
        for cells in frames(toks, G)[:-WAIT]:
            b.step(cells, b.still())
        r, w = WORDS.index('YES' if truth else 'NO'), WORDS.index('NO' if truth else 'YES')
        for t in range(T):
            b.step(blank, b.still())
            sh = shares()
            right[t] += sh[r] / n
            wrong[t] += sh[w] / n
            other[t] += np.delete(sh, [r, w]).max() / n
            top[t] += (int(np.argmax(sh)) == r) / n
            beats[t] += (sh[r] > sh[w]) / n
    quiet = []
    for t in range(300):
        b.step(blank, b.still())
        quiet.append(shares().max())
    print('SHARES above each word usual level, after the question, ticks 1..' + str(T) + ' (usual: '
          + ' '.join(f'{w}:{u:.2f}' for w, u in zip(WORDS, m.usual)) + ')', flush=True)
    for name, v in (('right answer', right), ('wrong answer', wrong), ('largest other', other),
                    ('right is top', top), ('right > wrong', beats)):
        print(f'  {name:14s} ' + ' '.join(f'{x:.2f}' for x in v), flush=True)
    q = np.array(quiet)
    print(f'  in silence, the largest share above usual of any word: mean {q.mean():.2f}, 90 % {np.quantile(q, 0.9):.2f}, '
          f'99 % {np.quantile(q, 0.99):.2f}', flush=True)


def ask_test(b0, G, n, stage, rng):
    b = copy.deepcopy(b0)
    m = Mouth(b, G, rng)
    blank = np.zeros((W.ROWS, W.COLS), bool)
    res, over = collections.defaultdict(list), 0
    said_first = collections.Counter()
    for i in range(n):
        toks, truth, s = W.sentence(stage, rng)
        for _ in range(6):
            m.tick(blank)
        for cells in frames(toks, G)[:-WAIT]:
            over += m.tick(cells) is not None
        first = None
        for _ in range(WAIT + 10):
            w = m.tick(blank)
            if w is not None and first is None:
                first = w
        said_first[first] += 1
        res[s].append(first == ('YES' if truth else 'NO'))
    print(f'ASK {n}: says the right answer first: ' + ' '.join(f'stage {k}: {np.mean(v):.2f}' for k, v in sorted(res.items()))
          + f' | words said while asked: {over} | first words: ' + ' '.join(f'{k}:{v}' for k, v in said_first.most_common(8)),
          flush=True)


def free_test(b0, G, ticks, rng):
    b = copy.deepcopy(b0)
    m = Mouth(b, G, rng)
    blank = np.zeros((W.ROWS, W.COLS), bool)
    stream, quiet, utt, out = [], 0, [], []
    for t in range(ticks):
        w = m.tick(blank)
        stream.append(w or '_')
        if w is None:
            quiet += 1
            if quiet >= PAUSE and utt:
                out.append(utt)
                utt = []
        else:
            quiet = 0
            utt.append(w)
    if utt:
        out.append(utt)
    kinds = collections.Counter()
    for u in out:
        k, tv = judge(u)
        kinds['full true' if k == 'full' and tv else ('full false' if k == 'full' else (k or 'other'))] += 1
    # yardstick: random strings of the same lengths
    rk = collections.Counter()
    for _ in range(50):
        for u in out:
            r = [WORDS[int(i)] for i in rng.integers(len(WORDS), size=len(u))]
            k, tv = judge(r)
            rk['full true' if k == 'full' and tv else ('full false' if k == 'full' else (k or 'other'))] += 1
    n = max(len(out), 1)
    print(f'FREE {ticks} ticks: {len(out)} utterances, words said {sum(len(u) for u in out)}; lengths '
          + str(collections.Counter(len(u) for u in out).most_common(6)), flush=True)
    print('  kinds: ' + ' '.join(f'{k} {v / n:.2f}' for k, v in kinds.most_common())
          + ' | random strings: ' + ' '.join(f'{k} {v / (50 * n):.3f}' for k, v in rk.most_common()), flush=True)
    print('  most said: ' + ' '.join(f'{" ".join(u)} x{c}' for u, c in collections.Counter(tuple(u) for u in out).most_common(12)),
          flush=True)
    print('  stream: ' + ' '.join(stream[:200]), flush=True)


def life(b, G, n, stage, rng, out):
    """half the time others ask and answer (it listens), half the time it is asked and answers; it hears and feels
    itself, learning is on; after its answer the other sometimes says the right one (rare correction)"""
    m = Mouth(b, G, rng, learn=True)
    blank = np.zeros((W.ROWS, W.COLS), bool)
    stats = collections.defaultdict(lambda: collections.deque(maxlen=200))
    for i in range(n):
        toks, truth, s = W.sentence(stage, rng)
        ans = 'YES' if truth else 'NO'
        if i % (LISTEN + 1) != LISTEN:
            said = [m.tick(cells) for cells, _ in W.dialogue(toks, truth, G)]
            stats['talks while others talk'].append(any(w is not None for w in said))
        else:
            for _ in range(4):
                m.tick(blank)
            said = [m.tick(cells) for cells in frames(toks, G)[:-WAIT]]
            first = None
            for _ in range(WAIT + 10):
                w = m.tick(blank)
                first = first or w
            stats['answers right'].append(first == ans)
            stats['answers at all'].append(first is not None)
            if rng.random() < CORRECT0 + (CORRECT - CORRECT0) * i / max(n - 1, 1):
                for cells in [G[ans]] * TOK + [blank] * 4:
                    m.tick(cells)
        if (i + 1) % int(os.environ.get('SPEAK_TEST_EVERY', 500)) == 0 or i == n - 1:
            print(f'LIFE {i + 1}: ' + ' | '.join(f'{k} {np.mean(v):.2f}' for k, v in stats.items()), flush=True)
            bb = copy.deepcopy(b)
            bb.pfc.adapt = False
            ask_test(bb, G, 60, stage, np.random.default_rng(100 + i))
            free_test(bb, G, 600, np.random.default_rng(200 + i))
            torch.save(b, out)


def main():
    path = sys.argv[1]
    asks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    b0 = torch.load(path, weights_only=False, map_location=DEV)
    game_brain.keys_hear_all(b0)
    b0.pfc.adapt = False                           # nothing changes while we look
    G = W.glyphs()
    stage = int(os.environ.get('LOGIC_STAGE', 3))
    print(f'brain {os.path.basename(path)}: G0 {G0} block {BLOCK} noise {NOISE} over {NOISE_T}; pace tok {TOK} gap {GAP} '
          f'wait {WAIT}; stages <= {stage}', flush=True)
    rng = np.random.default_rng(seed)
    if os.environ.get('SPEAK_SHARES') == '1':
        share_test(b0, G, 60, stage, rng)
        return
    if os.environ.get('SPEAK_LIFE'):                  # a life of listening and talking: SPEAK_LIFE=<out.pt>
        b0.pfc.adapt = True
        life(b0, G, asks, stage, rng, os.environ['SPEAK_LIFE'])
        return
    if asks:
        ask_test(b0, G, asks, stage, rng)
    if ticks:
        free_test(b0, G, ticks, rng)


if __name__ == '__main__':
    main()
