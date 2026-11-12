"""Vision, step B1 (ARCH 三 感觉区 视觉; Fable a35 #2), grown into the living brain: an eye that turns to what moves.

World side (Scene):
- the sign board: where the written sign of the syllable being heard appears; at rest it is where the gaze rests;
  at the start of a turn, with chance p_drift, it drifts away one block (4 px) per tick for 2-4 ticks and stays
  there for the turn; the next turn it is back.
- a fixed random texture on the world in blocks (the room), so that when the eye moves the whole picture changes.
- the retina, centred on the gaze: the fovea is the existing 16 x 16 input, unchanged (when the gaze is on the
  sign the trained visual pathway sees exactly what it saw before; elsewhere it sees the sign shifted or partly);
  around it a 12 x 12 grid of coarse cells (4 x 4 px each, a 48 x 48 window): a cell is lit when something (the
  texture or the sign) is in it. Centre dense, edge sparse.
- the gaze moves one block per tick by the eye muscles.

Brain side (EyeMotion, innate, ARCH 三):
- local motion neurons on the coarse cells, 4 directions: A = the neighbour lit a tick ago (it arrives a tick
  late); B = this cell newly lit now, with A: something moved here, in this direction;
- they inhibit each other as a whole: when more than motion_inhib of the cells report motion (the eye itself moved,
  the whole picture changed) none fires;
- born linked to the eye muscles (up, down, left, right): motion at a place pushes the muscle that turns the gaze
  toward that place; each muscle has m force neurons with graded thresholds and moves when half of them are lit.
Registered deviations for B1: the motion neurons have no chain copies yet, and the eye muscles' force is not yet
felt by the touch area; position invariance in the concept area (B2) is not built."""
import numpy as np
import torch

from .life import log
from .net import DEV

FOV, BLK, GRID = 16, 4, 12                      # fovea px, coarse block px, coarse grid (48 px around the gaze)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))       # up, down, left, right


class Scene:
    def __init__(self, seed, p_drift=0.1, max_off=16, texture=0.4):
        self.rng = np.random.default_rng(seed + 7070)
        R = (max_off + BLK + FOV + BLK * GRID) // BLK + 2
        self.R = R                                           # world blocks: -R .. R-1 in each direction
        self.tex = np.random.default_rng(seed + 7071).random((2 * R, 2 * R)) < texture
        self.board, self.gaze = np.zeros(2, int), np.zeros(2, int)   # px offsets from the resting place
        self.p_drift, self.max_off = p_drift, max_off
        self.dir, self.steps, self.turn = None, 0, None
        self.stats = dict(shown=0, on_fovea=0, shown_away=0, on_fovea_away=0, gaze_moves=0, drifts=0)

    def world_step(self, cur):
        """cur: what the adults say this tick (dialogue at [5], turn at [6]) or None"""
        key = None if cur is None else (int(cur[5]), int(cur[6]))
        if key is not None and key != self.turn:            # a new turn: the board is back; sometimes it drifts
            self.turn, self.board[:] = key, 0
            self.dir, self.steps = None, 0
            if self.rng.random() < self.p_drift:
                self.dir, self.steps = DIRS[int(self.rng.integers(4))], int(self.rng.integers(2, 5))
                self.stats['drifts'] += 1
        if self.dir is not None and self.steps > 0:
            nb = self.board + BLK * np.array(self.dir)
            if np.all(np.abs(nb) <= self.max_off):
                self.board = nb
            self.steps -= 1

    def retina(self, sign, background, count=True):
        """sign: bool [256] the sign's pixels, or None (nothing shown); background: bool [256] what the fovea shows
        when nothing is there (the 'nothing seen' reading). -> fovea [256], coarse [144]"""
        fov = background.reshape(FOV, FOV).copy()
        g = self.gaze // BLK
        u = np.arange(GRID) - GRID // 2 + g[0] + self.R - FOV // BLK // 2 + 2
        v = np.arange(GRID) - GRID // 2 + g[1] + self.R - FOV // BLK // 2 + 2
        coarse = self.tex[np.clip(u, 0, 2 * self.R - 1)][:, np.clip(v, 0, 2 * self.R - 1)].copy()
        y0, x0 = self.board - self.gaze                      # the board's top-left in fovea coordinates
        # the board (the speaker holding it) is always there for the periphery: coarse block (i, j) covers fovea
        # rows [-16 + 4i, -12 + 4i); the board covers 4 x 4 blocks. (11-12 smoke: drawn only with a sign, every
        # sign's onset beside the texture looked like motion and the gaze wandered: 86 % on the fovea vs 90 % still)
        bi, bj = (y0 + FOV) // BLK, (x0 + FOV) // BLK
        coarse[max(0, bi):max(0, bi + FOV // BLK), max(0, bj):max(0, bj + FOV // BLK)] = True
        if sign is not None:                                 # the fovea sees the sign itself, where it is
            P = sign.reshape(FOV, FOV)
            ys, xs, ye, xe = max(0, y0), max(0, x0), min(FOV, y0 + FOV), min(FOV, x0 + FOV)
            if ys < ye and xs < xe:
                fov[ys:ye, xs:xe] = P[ys - y0:ye - y0, xs - x0:xe - x0]
            if count:
                on = bool(np.all(self.board == self.gaze))
                away = bool(np.any(self.board != 0))
                self.stats['shown'] += 1
                self.stats['on_fovea'] += on
                self.stats['shown_away'] += away
                self.stats['on_fovea_away'] += on and away
        return fov.reshape(-1), coarse.reshape(-1)

    def move(self, force):
        """force: [4] lit force neurons of the up, down, left, right muscles; the gaze moves that many blocks"""
        if force is None:
            return
        f = np.asarray(force, int)
        d = np.array([f[1] - f[0], f[3] - f[2]])
        g = np.clip(self.gaze + BLK * d, -(self.max_off + BLK), self.max_off + BLK)
        if np.any(g != self.gaze):
            self.gaze = g
            self.stats['gaze_moves'] += 1


class EyeMotion:
    def __init__(self, c, m=8):
        self.m, self.inhib = m, float(c.get('motion_inhib', 0.1))
        self.prev = torch.zeros(GRID, GRID, dtype=torch.bool, device=DEV)
        # born links: a B neuron at a cell -> the muscle toward that cell, with the strength of how far the cell is
        # outside the fovea's blocks (4..7): rows above push 'up' by 4 - row, below push 'down' by row - 7, etc.
        r = torch.arange(GRID, device=DEV).float()
        lo, hi = (r < 4).float() * (4 - r), (r > 7).float() * (r - 7)
        self.ecc = torch.stack([lo[:, None].expand(GRID, GRID), hi[:, None].expand(GRID, GRID),
                                lo[None, :].expand(GRID, GRID), hi[None, :].expand(GRID, GRID)])
        self.thr = torch.arange(m, device=DEV).float() + 0.5    # graded force neurons: k lit = a k-block turn
        self.force = torch.zeros(4, m, dtype=torch.bool, device=DEV)
        self.B = torch.zeros(4, GRID, GRID, dtype=torch.bool, device=DEV)
        self.n_motion = 0
        self.last = (0, 0)

    def step(self, coarse):
        C = coarse.view(GRID, GRID)
        # the copy of the eye's own last turn (corollary discharge; registered addition to ARCH 三 视觉): the picture
        # a tick ago is shifted by it, so after a turn the room stands still and only what moved in it is motion
        # (11-12 smoke: without it the turn's whole-picture change hid the next step of a drifting sign; 10 % followed)
        dy, dx = self.last
        known = torch.ones_like(C)
        if dy or dx:
            P = torch.zeros_like(self.prev)
            K = torch.zeros_like(C)
            sy, sx = -dy, -dx                                 # the picture moves opposite to the gaze
            ys, ye, xs, xe = max(0, sy), GRID + min(0, sy), max(0, sx), GRID + min(0, sx)
            if ys < ye and xs < xe:
                P[ys:ye, xs:xe] = self.prev[ys - sy:ye - sy, xs - sx:xe - sx]
                K[ys:ye, xs:xe] = True
            self.prev, known = P, K
        new = C & ~self.prev & known
        B = []
        for dy, dx in DIRS:                                   # A: the neighbour behind, lit a tick ago
            A = torch.zeros_like(C)
            src = self.prev[max(0, -dy):GRID - max(0, dy), max(0, -dx):GRID - max(0, dx)]
            A[max(0, dy):GRID - max(0, -dy), max(0, dx):GRID - max(0, -dx)] = src
            B.append(new & A)
        B = torch.stack(B)
        if float(B.any(0).float().mean()) > self.inhib:      # the whole picture changed: none (mutual inhibition)
            B = torch.zeros_like(B)
        self.B, self.prev = B, C
        where = B.any(0)
        # each muscle is pushed as hard as its farthest motion neuron (as many force neurons light: ARCH 三 运动区
        # 亮得越多，力越大): one turn brings the moving thing to the fovea (a saccade)
        drive = (self.ecc * where[None].float()).flatten(1).max(1).values
        net = drive - drive[[1, 0, 3, 2]]                     # against the opposite muscle
        self.force = net[:, None] >= self.thr[None, :]
        self.n_motion += int(where.any())
        f = self.force.sum(1)
        self.last = (int(f[1] - f[0]), int(f[3] - f[2]))
        return f


def pairs(v):
    """thermometer pairs (ARCH 二.3): on, off"""
    v = np.asarray(v, bool)
    return np.stack([v, ~v], 1).reshape(-1)


def add_eye(life, p_drift=0.1):
    """grows the eye's periphery, the motion neurons and the eye muscles into the living brain (the fovea is the
    existing visual input, unchanged)"""
    b = life.brain
    assert life.c.get('sight_area'), 'the eye extends the visual area (config sight_area)'
    life.scene = Scene(life.seed, p_drift=p_drift)
    b.eye_motion = EyeMotion(life.c)
    b.n_periph = 2 * GRID * GRID
    life.registry = getattr(life, 'registry', []) + [('eye: motion neurons -> eye muscles (innate)', 4, GRID * GRID)]
    log(f'eye added: fovea {FOV}x{FOV} (unchanged), periphery {GRID}x{GRID} coarse cells, 4 eye muscles, '
        f'drift on {p_drift:.0%} of turns')
