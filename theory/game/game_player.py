"""Game data made by code (a data source, not part of the model): the real game played by a hand-written ball follower
with some random slips, so the model also sees mistakes and recoveries a perfect player never shows. Same file format as
game_data.py.
usage: python game_player.py <out dir> <first seed> <games> [noise] [style]
style slip (default): now and then the wrong key held for 5-25 ticks.
style lapse: now and then the player looks away and presses nothing for 10-60 ticks, so the ball gets far from the paddle
  and the player then chases it; every key pressed points toward the ball.
GAME_SEE_LAG=n (env): the player sees the ball and paddle as they were n frames ago, like the model, and adds its own key
  presses since then to where it thinks the paddle is (GAME_SPEED cells per frame held); so what it does depends only on
  what the model can also know."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_data import locate, to_cells          # noqa: E402
from game_env import LEFT, NOOP, RIGHT, Breakout  # noqa: E402

GAME = {NOOP: 0, RIGHT: 3, LEFT: 4}
SEE_LAG = int(os.environ.get('GAME_SEE_LAG', 0))
SPEED = float(os.environ.get('GAME_SPEED', 1.0))


def main():
    out, seed, games = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    style = sys.argv[5] if len(sys.argv) > 5 else 'slip'
    os.makedirs(out, exist_ok=True)
    for g in range(seed, seed + games):
        e = Breakout(g)
        rng = np.random.default_rng(g + 77)
        f = e.reset()
        C, A, P = [], [], []
        hold, held = 0, NOOP
        seen, did = [], []
        for t in range(12000):
            seen.append(locate(f))
            bx, by, px = seen[max(0, len(seen) - 1 - SEE_LAG)]
            if SEE_LAG and px >= 0:
                px = px + SPEED * sum(did[-SEE_LAG:])
            if hold > 0:                                 # a slip: the wrong key held for a moment
                a, hold = held, hold - 1
            elif rng.random() < noise / 10:
                if style == 'lapse':
                    held, hold = NOOP, int(rng.integers(10, 60))
                else:
                    held, hold = int(rng.choice((RIGHT, LEFT))), int(rng.integers(5, 25))
                a = held
            elif bx < 0 or px < 0:
                a = NOOP
            else:
                d = bx - px
                a = RIGHT if d > 1 else LEFT if d < -1 else NOOP
            C.append(to_cells(f))
            A.append(GAME[a])
            P.append(seen[-1])                               # where things really are now (for measuring)
            did.append({NOOP: 0, RIGHT: 1, LEFT: -1}[a])
            f, over = e.step(a)
            if over:
                break
        p = np.array(P, np.float32)
        np.savez_compressed(f'{out}/t{g}.npz', cells=np.array(C), action=np.array(A, np.uint8), ball_x=p[:, 0], ball_y=p[:, 1], paddle_x=p[:, 2])
        print(g, 'score', e.score, 'frames', len(A), flush=True)


if __name__ == '__main__':
    main()
