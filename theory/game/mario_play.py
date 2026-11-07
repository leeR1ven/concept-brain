"""Mario closed loop: the brain plays a level in the emulator (nes-py, gym-super-mario-bros), one tick per frame; the buttons
it presses go back in as copies. A button is pressed when the drive of its press copies minus its stop copies, from the
last layer through the learned links, is above that button's own bar: the bar follows the button's recent drive so that it
fires about as often as it did in the recordings (each neuron's own threshold from its firing rate, ARCH 二.4).
Measured: how far Mario gets (x position), and whether he reaches the flag. A run ends at death, at the flag, or when he has
not got further for 600 frames.
Baselines (no brain): nothing, run right, run right and jump every 30 frames.
usage: python mario_play.py <net.pt> <learned.npz | baseline name> <levels e.g. 1-1,2-1> <runs per level> [seed]"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_brain import MARIO_KEYS, UNITS_PER_KEY as U   # noqa: E402
from mario_data import CELL, COLS, ROWS, TOP             # noqa: E402

torch.set_num_threads(1)
PAD = {128: 0b00000001, 16: 0b00000010, 32: 0b01000000, 4: 0b10000000, 2: 0b00100000}   # dataset bit -> nes-py joypad bit
RATE = os.environ.get('MARIO_RATE')                      # override the per-button firing shares, e.g. "0.3,0.9,0.05,0.8,0.01"
MAX_STILL = 600


def cells_of(rgb):
    """same as mario_data.to_cells, from an RGB frame: the commonest colour is the background"""
    a = rgb[TOP:].astype(np.int32)
    a = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    vals, cnt = np.unique(a, return_counts=True)
    on = a != vals[np.argmax(cnt)]
    return on.reshape(ROWS, CELL, COLS, CELL).any(axis=(1, 3))


def pad_of(keys):
    """our five keys -> the emulator's joypad byte; left and right together: neither"""
    k = np.asarray(keys, bool).copy()
    if k[2] and k[3]:
        k[2] = k[3] = False
    return sum(PAD[b] for b, on in zip(MARIO_KEYS, k) if on)


def run(env, policy, brain, rng):
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    keys = np.zeros(len(MARIO_KEYS), bool)
    best, still, t, info = 0, 0, 0, {'x_pos': 0, 'flag_get': False}
    hist = []
    while True:
        c = cells_of(obs)
        if brain is not None:
            b, W, rate = brain
            b.step(c, keys)
            on = torch.nonzero(b.pfc.L4.on).squeeze(1).numpy()
            d = W[on].sum(0) if len(on) else np.zeros(W.shape[1], np.float32)
            nk = len(MARIO_KEYS)
            s = np.array([d[k * U:(k + 1) * U].mean() - d[(nk + k) * U:(nk + k + 1) * U].mean() for k in range(nk)])
            hist.append(s)
            H = np.array(hist[-600:])
            bar = np.array([np.quantile(H[:, k], 1.0 - rate[k]) for k in range(nk)]) if len(hist) > 60 else np.zeros(nk)
            keys = s > bar
        elif policy == 'nothing':
            keys = np.zeros(len(MARIO_KEYS), bool)
        elif policy == 'run':
            keys = np.array([False, True, False, True, False])
        elif policy == 'runjump':
            keys = np.array([(t // 30) % 2 == 0, True, False, True, False])
        obs, _, done, info = env.step(pad_of(keys))
        t += 1
        if info['x_pos'] > best:
            best, still = info['x_pos'], 0
        else:
            still += 1
        if done or info['flag_get'] or still > MAX_STILL:
            return best, bool(info['flag_get']), t


def main():
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
    net, what, levels, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(','), int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = np.random.default_rng(seed)
    W, rate = None, None
    if what.endswith('.npz'):
        z = np.load(what)
        W = z['w']
        rate = np.array([float(v) for v in RATE.split(',')]) if RATE else z['rate']
    for lv in levels:
        w, s = (int(v) for v in lv.split('-'))
        env = SuperMarioBrosEnv(rom_mode='vanilla', target=(w, s))
        for r in range(runs):
            brain = (torch.load(net, weights_only=False), W, rate) if W is not None else None
            x, flag, t = run(env, what if W is None else None, brain, rng)
            print('MARIO', os.path.basename(what), lv, 'run', r, 'x', x, 'flag', flag, 'frames', t, flush=True)
        env.close()


if __name__ == '__main__':
    main()
