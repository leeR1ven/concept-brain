"""Breakout seen by the model: a vision area made by the one generator, chain copies of its feature units, the keys as copies
too (ARCH: motor signals are PFC inputs like any other), and the same PFC class as the speech model. Newborn: nothing learned.
Every key has press neurons and stop neurons; one of the two groups is always lit (designer 11-07: when a key is not pressed,
its inhibiting neurons light up, so that "not now" is something the PFC can learn too).
Gaze (GAME_GAZE=1 when building): the eye looks at what moves in the picture (not counting the own paddle's rows) and the
sheet is centred on that column, twice as wide so the whole screen stays in view; where the paddle lies on the sheet then
says where it is from the ball.
  python game_brain.py <cells dir> <birth trial> <net.pt>   builds and calibrates once, every later run loads the same brain"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.brain import PFC, SenseArea          # noqa: E402
from core.config import config                 # noqa: E402
from core.net import DEV, Links                # noqa: E402

ROWS, COLS = 41, 36
KEYS = {3: 0, 4: 1, 11: 0, 12: 1}              # game action -> key: 0 right, 1 left (fire keeps its direction)
N_KEY, UNITS_PER_KEY = 2, 24
TPF = int(os.environ.get('GAME_TPF', 1))         # brain ticks per game frame
PADDLE_ROW = 39                                 # cell rows from here down hold the paddle


class Eye:
    """input sheet: y = column, z = row * 2 + polarity. Each cell is a pair, exactly one of the two is on (ARCH 二)."""
    NC, Z = COLS, 2 * ROWS

    @staticmethod
    def bits(cells):
        """cells: bool [T, 41, 36] or [41, 36]; returns bool [T, n] or [n]"""
        c = np.moveaxis(np.asarray(cells), -1, -2)         # [.., col, row]
        out = np.stack([c, ~c], -1)                        # [.., col, row, polarity]
        return out.reshape(*c.shape[:-2], -1)


class Gaze:
    """the eye's column: it jumps to what changed since the last frame above the paddle rows, else stays"""
    W = 2 * COLS - 1

    def __init__(self):
        self.reset()

    def reset(self):
        self.g, self.prev = COLS // 2, None

    def look(self, cells):
        cells = np.asarray(cells, bool)
        if self.prev is not None:
            ch = np.nonzero((cells[:PADDLE_ROW] != self.prev[:PADDLE_ROW]).any(0))[0]
            if len(ch):
                self.g = int(round(ch.mean()))
        self.prev = cells
        out = np.zeros((ROWS, self.W), bool)
        out[:, COLS - 1 - self.g:COLS - 1 - self.g + COLS] = cells
        return out


class Copies:
    """stands in for Chains: feature units of the vision area, then the key copies, each with a place on the PFC sheet"""

    def __init__(self, area, rates, band, n_key=N_KEY, key_box=None):
        lo, hi = band
        self.idx = [torch.nonzero((r >= lo) & (r <= hi)).squeeze(1) for r in rates]
        grid, pos, a = [], [], 0
        for l, ix in enumerate(self.idx):
            g = area.grid[l][ix.cpu().numpy()]
            grid.append(g)
            pos.append(np.stack([g[:, 1] / (area.NC - 1), g[:, 2] / (area.Z - 1)], 1))
            a += len(ix)
        self.n_vision = a
        u = np.arange(2 * n_key * UNITS_PER_KEY)          # press key 0, 1, .., then stop key 0, 1, ..
        if key_box is None:
            g = np.stack([np.full_like(u, 1), u // UNITS_PER_KEY, u % UNITS_PER_KEY], 1)
            p = np.stack([g[:, 1] / (2 * n_key - 1), g[:, 2] / (UNITS_PER_KEY - 1)], 1)
        else:                                             # spread over a box of the sheet (where the eye looks), so
            lo_, hi_ = key_box                            # each key hears the last layer all over it; a key's press
            r = np.random.default_rng(11).uniform(lo_, hi_, (n_key * UNITS_PER_KEY, 2))   # and stop units share places
            p = np.concatenate([r, r])
            g = np.stack([np.full(len(u), 1), np.round(p[:, 0] * (area.NC - 1)), np.round(p[:, 1] * (area.Z - 1))],
                         1).astype(np.int64)
        grid.append(g)
        pos.append(p)
        self.n = a + len(u)
        self.grid = np.concatenate(grid)
        jit = np.random.default_rng(7).normal(0, 0.005, (self.n, 2))
        self.pos = torch.as_tensor(np.clip(np.concatenate(pos) + jit, 0, 1), dtype=torch.float32, device=DEV)

    def gather(self, state, keys):
        """state: the vision layers; keys: bool [n_key], the keys held on the tick before"""
        k = torch.as_tensor(np.concatenate([keys, ~np.asarray(keys)]), dtype=torch.bool, device=DEV)
        k = k.repeat_interleave(UNITS_PER_KEY)
        return torch.cat([s[ix] for s, ix in zip(state, self.idx)] + [k])


class KeyLinks:
    """the last layer -> first layer learned links (ARCH 四, each copy's own check) where the button copies hear the
    whole last layer: whether to press depends on the whole scene, while near links let a button copy see only a few
    patches of it. The other copies keep their near links. Each part has its own gain, so one link has the usual size
    (1 / the number of lit last-layer units linked to a copy)."""

    def __init__(self, near, n_vision):
        nv, n = n_vision, near.n_post
        indeg = torch.bincount(near.idx.reshape(-1), minlength=n + 1)[:nv].float().mean()
        keep = near.idx < nv
        near.w *= keep
        near.idx = torch.where(keep, near.idx, torch.full_like(near.idx, n))
        nk = n - nv
        idx = torch.arange(nk, device=DEV).expand(near.n_pre, nk)
        self.near = near
        self.keys = Links(idx, 0.0, nk, near.c, gain=near.gain * float(indeg) / near.n_pre, gamma=near.gamma,
                          name='L4->keys')
        self.nv, self.learned, self.name, self.log = nv, True, near.name, None

    @property
    def w(self):
        return self.near.w

    def drive(self, act):
        out = self.near.drive(act)
        src = getattr(self, 'src', None)               # the layer the button copies hear (default the last layer)
        out[self.nv:] += self.keys.drive(act if src is None else src.on)
        return out

    def learn_check(self, pre, ext, e, theta, margin, step):
        self.near.learn_check(pre, ext, e, theta, margin, step)
        if getattr(self, 'check_keys', True):             # off while the brain plays itself: its own choices then
            self.keys.learn_check(pre, ext[self.nv:], e[self.nv:], theta, margin, step)   # change only by reward

    def decay(self):
        self.near.decay()
        self.keys.decay()


def keys_hear_centre(b, r):
    """the button copies hear only the last-layer units within r of the sheet's centre, where the eye looks (the
    vision is denser in the middle, ARCH 三): what decides a press is around Mario, and the far background would only
    dilute what a reward teaches. K.hear marks the units heard."""
    keys_hear_all(b)
    K, near = b.pfc.L41.keys, b.pfc.L41.near
    pos = b.ch.pos
    hear = ((pos - 0.5) ** 2).sum(1).sqrt() < r
    if getattr(K, 'hear', None) is not None:
        return b
    full = float(K.n_pre)
    K.idx = torch.where(hear[:, None], K.idx, torch.full_like(K.idx, K.n_post))
    K.w *= hear[:, None].float()
    K.gain *= full / float(hear.sum())                 # 1 / the number of lit heard units, as before
    K.hear = hear
    return b


def keys_rehear(b, r):
    """hear another radius: undo keys_hear_centre (all units heard, the gain back), then hear within r again; the
    links of units no longer heard are dropped"""
    K = b.pfc.L41.keys
    if getattr(K, 'hear', None) is not None:
        K.gain *= float(K.hear.sum()) / float(K.n_pre)
        K.idx = torch.arange(K.n_post, device=DEV).expand(K.n_pre, K.n_post).clone()
        K.hear = None
    return keys_hear_centre(b, r)


def keys_hear_all(b):
    """a newborn brain's button copies get links from every last-layer unit (KeyLinks)"""
    if not isinstance(b.pfc.L41, KeyLinks):
        b.pfc.L41 = KeyLinks(b.pfc.L41, b.ch.n_vision)
    return b


class GameBrain:
    def __init__(self, c, seed, birth_cells, birth_keys, gaze=False, game='breakout', key_box=None):
        self.c, self.game = c, game
        rng = np.random.default_rng(seed)
        self.gaze = Gaze() if gaze else None
        if gaze:
            birth_cells = np.array([self.gaze.look(f) for f in birth_cells])
        rows, cols = np.asarray(birth_cells).shape[-2:]
        self.area = SenseArea(cols, 2 * rows, c['sound_rates'], c, rng)
        A = torch.as_tensor(Eye.bits(birth_cells), device=DEV)
        S = self.area.calibrate(A)
        rates = [s.float().mean(0) for s in S]
        self.ch = Copies(self.area, rates, c['band'], birth_keys.shape[1], key_box)
        K = np.concatenate([birth_keys, ~birth_keys], 1)
        keys = torch.as_tensor(K, device=DEV).repeat_interleave(UNITS_PER_KEY, 1)
        X = torch.cat([s[:, ix] for s, ix in zip(S, self.ch.idx)] + [keys], 1)
        self.pfc = PFC(self.ch, c, rng)
        self.pfc.calibrate(X)
        self.pfc.adapt = True
        self.reset()

    def still(self):
        """the muscle copies' input when no muscle moves (as many keys as this brain was born with)"""
        return np.zeros((self.ch.n - self.ch.n_vision) // (2 * UNITS_PER_KEY), bool)

    def reset(self):
        if getattr(self, 'gaze', None) is not None:
            self.gaze.reset()
        self.area.state = [torch.zeros(n, dtype=torch.bool, device=DEV) for n in self.area.n]
        self.no_ret = [torch.zeros(n, device=DEV) for n in self.area.n]

    def step(self, cells, keys, each=None):
        """one game frame: the brain runs TPF ticks on it (GAME_TPF, default 1), so the last layer lags fewer frames;
        each(): called after every tick (learning lives tick by tick)"""
        if getattr(self, 'gaze', None) is not None:
            cells = self.gaze.look(cells)
        a = torch.as_tensor(Eye.bits(cells), device=DEV)
        for _ in range(TPF):
            st = self.area.step(a, self.no_ret)
            x = self.ch.gather(st, keys)
            self.pfc.step(x)
            if each is not None:
                each()
        return x


    def keys_of(self, action):
        """the game's action code -> which keys are held"""
        return mario_bits(action) if getattr(self, 'game', 'breakout') == 'mario' else key_bits(action)


MARIO_KEYS = (128, 16, 32, 4, 2)                 # Mario buttons kept: A (jump), B (run, fire), left, right, down


def mario_bits(action):
    return np.array([bool(int(action) & b) for b in MARIO_KEYS])


def key_bits(action):
    b = np.zeros(N_KEY, bool)
    k = KEYS.get(int(action))
    if k is not None:
        b[k] = True
    return b


def main():
    src, birth, out = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.load(f'{src}/{birth}.npz')
    c = config(s=1, dz=(2, 4, 8, 16), levels_per_s=4, r_learn=0.01)     # no learned links in this first test
    torch.set_num_threads(8)
    keys = np.array([key_bits(a) for a in z['action']])
    b = GameBrain(c, 0, z['cells'][::3], np.concatenate([np.zeros((1, N_KEY), bool), keys[:-1]])[::3],
                  gaze=os.environ.get('GAME_GAZE') == '1')
    ch = b.ch
    print('vision units', b.area.n, 'feature units', [len(i) for i in ch.idx], 'copies', ch.n, flush=True)
    b.ch_rates = None
    torch.save(b, out)
    print('saved', os.path.getsize(out) // 1000000, 'MB')


if __name__ == '__main__':
    import game_brain                               # so the saved brain loads from any script
    game_brain.main()
