"""The real game for closed-loop tests. One step = one game frame, as in the recordings. The harness presses fire when a ball has
to be served: serving is rare in the recordings and is not what is being tested.
Checks when run directly: a player that does nothing, a random one, and a hand-written ball follower (the last two only to show
the harness can score)."""
import os
import sys

import ale_py
import ale_py.roms
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_data import locate, to_cells          # noqa: E402

NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3              # the emulator's minimal action set for Breakout


class Breakout:
    def __init__(self, seed=0):
        self.ale = ale_py.ALEInterface()
        self.ale.setInt('random_seed', seed)
        self.ale.setFloat('repeat_action_probability', 0.0)
        self.ale.setInt('frame_skip', 1)
        self.ale.loadROM(ale_py.roms.get_rom_path('breakout'))
        self.acts = self.ale.getMinimalActionSet()
        self.rng = np.random.default_rng(seed)

    def reset(self, wait=None):
        self.ale.reset_game()
        for _ in range(int(self.rng.integers(0, 30)) if wait is None else wait):   # a different start each game
            self.ale.act(self.acts[NOOP])
        self.score, self.lives, self.since = 0, self.ale.lives(), 0
        self.serve()
        return self.gray()

    def serve(self):
        self.ale.act(self.acts[FIRE])
        self.since = 0

    def gray(self):
        return self.ale.getScreenGrayscale()

    def step(self, a):
        self.score += self.ale.act(self.acts[a])
        self.since += 1
        if self.ale.lives() < self.lives or self.since > 600:                # a life lost, or a ball that never left
            self.lives = self.ale.lives()
            self.serve()
        return self.gray(), self.ale.game_over()


def play(policy, games, seed=0, limit=20000):
    out = []
    for g in range(games):
        e = Breakout(seed + g)
        f = e.reset()
        for t in range(limit):
            f, over = e.step(policy(f, t))
            if over:
                break
        out.append(e.score)
    return out


def main():
    rng = np.random.default_rng(1)

    def follow(f, t):
        bx, by, px = locate(f)
        if bx < 0 or px < 0:
            return NOOP
        d = bx - (px - 0)
        return RIGHT if d > 1 else LEFT if d < -1 else NOOP

    print('nothing', play(lambda f, t: NOOP, 5))
    print('random', play(lambda f, t: int(rng.choice((NOOP, RIGHT, LEFT))), 10))
    print('follower', play(follow, 5))
    f = Breakout(0).reset()
    print('cells lit', int(to_cells(f).sum()), 'of', 41 * 36, 'position', locate(f))


if __name__ == '__main__':
    main()
