# -*- coding: utf-8 -*-
"""world2 core: an open 2-D house + garden where N models live continuously with scripted caregivers.

    import sys; sys.path.insert(0, '<scratchpad>/arch/v2/bench')
    from world2 import World2, PerAgent, evaluate, headline
    w = World2(n_agents=4, seed=0)
    log = w.run(agents, ticks=100000)        # agents.step(obs) -> (N, 41) actions; obs = dict of (N, ...) arrays
    print(headline(evaluate(log)))

This module holds the mechanics (map, bodies, objects, sound field, eye, actions, the event loop, logging and the
caregiver op driver); care.py holds the caregivers' scripts.  See README.md and DESIGN.md.
"""
from __future__ import annotations

import collections
import copy
import heapq
import importlib.util
import os
import sys

import numpy as np

from . import lex
from .lex import (ANIMALS, EDIBLE, EYE_B, H, KIND_ID, KINDS, N_VIEW, T_OUT, TASTE_N, TO_BUMP, TO_HAND, TOUCH_N, W,
                  VIEW_FRONT)

# ------------------------------------------------------------------------------------------ the bench's physics
_BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench_module(name):
    """load a bench module (physics, metrics) by path under a private name (no clash with agents' own copies)."""
    key = 'w2bench_' + name
    m = sys.modules.get(key)
    if m is None:
        spec = importlib.util.spec_from_file_location(key, os.path.join(_BENCH, name + '.py'))
        m = importlib.util.module_from_spec(spec)
        sys.modules[key] = m
        spec.loader.exec_module(m)
    return m


phys = bench_module('physics')
SOUND, PAUSE = phys.SOUND, phys.PAUSE          # 3 ticks per char; a bout ends after 5 silent ticks
_VOICES = {}


def _voice(seed, physics):
    """the bench's Voice over the sorted corpus chars (= the real model's sounds and muscle sets), deep-copied from a
    cached pristine instance."""
    key = (int(seed), physics)
    if key not in _VOICES:
        if len(_VOICES) > 4:
            _VOICES.clear()
        _VOICES[key] = phys.Voice(phys.corpus_chars(), seed=int(seed), physics=physics)
    return copy.deepcopy(_VOICES[key])


# ------------------------------------------------------------------------------------------ static map tables
PAD = 4
PW = W + 2 * PAD                 # padded width (views never leave the padded grid)
NC = PW * PW
DIRS = (-PW, 1, PW, -1)          # headings 0 N, 1 E, 2 S, 3 W
DXY = ((0, -1), (1, 0), (0, 1), (-1, 0))
VOID = 0                         # a padding cell: occluded view cells point here; it never holds anything
INAUD = 6                        # attenuation index meaning "not heard"; 0..5 = levels subtracted
MOTOR = ('forward', 'back', 'left', 'right', 'turn_left', 'turn_right', 'take', 'put', 'eat')
N_MOUTH, N_MOTOR = 32, 9
N_ACT = N_MOUTH + N_MOTOR
POW2 = (1 << np.arange(N_MOUTH, dtype=np.int64))


def cell_of(x, y):
    return (y + PAD) * PW + (x + PAD)


def xy_of(c):
    return c % PW - PAD, c // PW - PAD


class Static:
    """layout-only tables (built once per process)."""
    _inst = None

    def __new__(cls):
        if cls._inst is not None:
            return cls._inst
        s = super().__new__(cls)
        ter = np.full(NC, T_OUT, np.int8)
        room = np.full(NC, -1, np.int8)
        for y, row in enumerate(lex.LAYOUT):
            for x, ch in enumerate(row):
                c = cell_of(x, y)
                ter[c] = lex.TERRAIN_CHARS.index(ch)
                room[c] = lex.room_of_xy(x, y)
        s.TER, s.ROOM = ter, room
        s.WALK = np.isin(ter, list(lex.WALKABLE_T))
        s.HOLD = np.isin(ter, list(lex.HOLDS_OBJ_T))
        s.OPAQUE = np.isin(ter, list(lex.OPAQUE_T))
        s.walk_cells = np.flatnonzero(s.WALK)
        doors = [int(c) for c in s.walk_cells if ter[c] == lex.T_DOOR]
        s.doorzone = set(doors) | {c + d for c in doors for d in DIRS}     # doorways and the cells in front of them
        # open cells: walkable floor with all 8 neighbours walkable (a person standing there never cuts a passage)
        s.open_cells = [int(c) for c in s.walk_cells if ter[c] != lex.T_DOOR and
                        all(s.WALK[c + d1 + d2] and ter[c + d1 + d2] != lex.T_DOOR
                            for d1 in (-PW, 0, PW) for d2 in (-1, 0, 1))]
        s.map_cells = np.array([cell_of(x, y) for y in range(H) for x in range(W)])
        # egocentric views with line-of-sight occlusion by walls
        view = np.full((NC, 4, N_VIEW), VOID, np.int32)
        for p in s.map_cells:
            px, py = xy_of(int(p))
            for h in range(4):
                fx, fy = DXY[h]
                rx, ry = DXY[(h + 1) % 4]
                for r in range(5):
                    for cc in range(-2, 3):
                        x, y = px + r * fx + cc * rx, py + r * fy + cc * ry
                        hidden = False
                        for f in (0.25, 0.5, 0.75):
                            rr, c2 = round(r * f), round(cc * f)
                            if (rr, c2) in ((0, 0), (r, cc)):
                                continue
                            if s.OPAQUE[cell_of(px + rr * fx + c2 * rx, py + rr * fy + c2 * ry)]:
                                hidden = True
                                break
                        if not hidden:
                            view[p, h, r * 5 + cc + 2] = cell_of(x, y)
        s.VIEW = view
        invk = np.full((NC, NC * 4), -1, np.int8)         # INVK[c, p*4+h] = view index of cell c from (p, h), or -1
        P, Hh, K = np.meshgrid(np.arange(NC), np.arange(4), np.arange(N_VIEW), indexing='ij')
        invk[view.reshape(-1), (P * 4 + Hh).reshape(-1)] = K.reshape(-1)
        invk[VOID] = -1
        s.INVK = invk
        s.walk_list = s.WALK.tolist()
        s.xy = np.array([xy_of(c) for c in range(NC)], np.float64)
        s._dist = {}
        cls._inst = s
        return s

    def att_table(self, rng_same, rng_wall, wall_att):
        """ATT[listener cell, source cell]: levels subtracted (0 at <= 2 cells, +1 per 2 cells, + wall_att between
        rooms), INAUD beyond range (rng_same in the same room, rng_wall through walls / doorways)."""
        key = ('att', rng_same, rng_wall, wall_att)
        if key in self._dist:
            return self._dist[key]
        wc = self.walk_cells
        xy = self.xy[wc]
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        a = np.where(d <= 2.0, 0, np.ceil((d - 2.0) / 2.0 - 1e-9)).astype(np.int64)
        same = self.ROOM[wc][:, None] == self.ROOM[wc][None, :]
        a = np.where(same, np.where(d <= rng_same, a, INAUD), np.where(d <= rng_wall, a + wall_att, INAUD))
        a = np.minimum(a, INAUD)
        att = np.full((NC, NC), INAUD, np.int8)
        att[wc[:, None], wc[None, :]] = a
        self._dist[key] = att
        return att

    def dist_to(self, goals):
        """BFS distance (over statically walkable cells) to the nearest of `goals` (tuple of cells); cached."""
        key = goals
        d = self._dist.get(key)
        if d is None:
            d = [255] * NC                                   # bytes: 255 = unreachable
            q = collections.deque()
            walk = self.walk_list
            for g in goals:
                if walk[g]:
                    d[g] = 0
                    q.append(g)
            while q:
                c = q.popleft()
                nd = d[c] + 1
                for dd in DIRS:
                    n = c + dd
                    if walk[n] and d[n] > nd:
                        d[n] = nd
                        q.append(n)
            d = bytes(d)
            if len(self._dist) > 20000:
                self._dist = {k: v for k, v in self._dist.items() if k[0] == 'att'}
            self._dist[key] = d
        return d


# ------------------------------------------------------------------------------------------ defaults
DEFAULTS = dict(
    day_ticks=24000, night_ticks=6000,       # 1 tick = 75 ms: a 30-min day and a 7.5-min night
    bedtime=900,                             # caregivers start the bedtime routine this many ticks before the night
    hear_range=8.0, hear_range_wall=6.0, wall_att=2,
    motor_ticks=3,                           # a motor action occupies the body for 3 ticks (one beat)
    cg_step_ticks=3,                         # caregivers walk one cell per 3 ticks
    hunger_rate=1 / 20000, hunger_eat=0.5, hunger0=0.3, taste_ticks=6,
    gap_answer=45, pause_after=5, reply_cap=60, gap_model=3, model_wait=12,   # the bench's answer timing
    request_ticks=400,                       # a request is scored by its outcome within this window
    p_model=0.3,                             # seen items: the caregiver answers her own question (model, not scored)
    probe_frac=0.15,                         # seen items asked with no reaction at all (held-out items: always)
    held_ratio=0.3,
    reward_ok=1.0, reward_request=1.0, reward_init=1.0, reward_social=1.0, reward_imitate=0.5,
    turns=(2, 6), turn_gap=(20, 150),
    w_teach=0.6, w_tidy=0.14, w_chat=0.12, w_rest=0.14, rest_ticks=(150, 700),
    w_turn=dict(name=0.22, question=0.33, memory=0.15, request=0.15, novel=0.06, fact=0.06, offer=0.0),
    p_call=0.5,                              # a far agent is called (名字过来) instead of walked to
    p_corpus=0.2,                            # share of caregiver chat lines taken from the adult corpus
    naming_cooldown=3000, call_cooldown=600, ask_cooldown=300,
    meals=(0.06, 0.45, 0.80), food_per_meal=(2, 4), cook_ticks=300,
    mon_every=10000,
    ear_keep=1024,                           # obs['ear'] views stay valid for >= this many ticks (copy to keep longer)
    motor_shuffle=False,                     # per-agent random permutation of the 9 motor units
    mouth_vocab=None,                        # None | 'lexicon' | list of chars (bench option)
    min_mem_delay=300,                       # memory probes: the fact was last told / seen >= this many ticks ago
    mem_spacing=3000,                        # ... and the same item was not asked to this agent in the last 3000 ticks
)


class _Agents:
    """placeholder so World2 can be built before agents are attached."""
    privileged = False

    @staticmethod
    def step(obs):
        return None


class CG:
    """a caregiver's runtime state (scripts live in care.py)."""
    __slots__ = ('pid', 'name', 'vkey', 'home', 'stack', 'op', 'token', 'say_wait', 'partner', 'held_by', 'received',
                 'busy_kind', 'greeted', 'prefer', 'n_acts', 'idx', 'speaking', 'bed_done', 'act', 'ep_t0')

    def __init__(self, idx, pid, name, vkey, home):
        self.idx, self.pid, self.name, self.vkey, self.home = idx, pid, name, vkey, home
        self.stack = []; self.op = None; self.token = 0; self.say_wait = 0
        self.partner = None; self.held_by = None; self.received = None; self.busy_kind = None
        self.greeted = set(); self.prefer = None; self.n_acts = 0; self.speaking = False; self.bed_done = False
        self.act = None; self.ep_t0 = 0


class PyRng:
    """seeded Python random with the few numpy-Generator methods the scripts use (scalar draws ~20x faster)."""

    def __init__(self, *seed):
        import random
        self.r = random.Random(repr(seed))

    def random(self):
        return self.r.random()

    def integers(self, a, b=None):
        if b is None:
            a, b = 0, a
        return self.r.randrange(int(a), int(b))

    def uniform(self, a, b):
        return self.r.uniform(a, b)

    def choice(self, n, p=None):
        if p is None:
            return self.r.randrange(n)
        u = self.r.random() * float(sum(p))
        acc = 0.0
        for i, w in enumerate(p):
            acc += w
            if u < acc:
                return i
        return n - 1

    def shuffle(self, x):
        self.r.shuffle(x)

    def permutation(self, n):
        x = list(range(n))
        self.r.shuffle(x)
        return x


PENDING = object()
_NO_SND = ((), ())
PREEMPTIBLE = ('wait', 'goto')


from .care import CareMixin   # noqa: E402  (scripts; uses the names above)


class World2(CareMixin):
    """see the module docstring.  Persons: agents 0..N-1, caregivers N..N+2 (妈妈 爸爸 奶奶)."""

    def __init__(self, n_agents=4, seed=0, physics='random', noise=0.0, tutor='mixed', **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise TypeError(f'unknown World2 params {sorted(unknown)}')
        if not 1 <= n_agents <= len(lex.AGENT_NAMES):
            raise ValueError(f'n_agents must be 1..{len(lex.AGENT_NAMES)}')
        self.p = p = dict(DEFAULTS)
        p.update(params)
        if tutor == 'recast':
            p['p_model'] = 0.0
        elif tutor == 'model':
            p['p_model'] = 1.0 - p['probe_frac']
        elif tutor != 'mixed':
            raise ValueError(tutor)
        self.tutor = tutor
        self.N = N = int(n_agents)
        self.seed, self.physics, self.noise = int(seed), physics, float(noise)
        self.S = S = Static()
        self.ATT = S.att_table(p['hear_range'], p['hear_range_wall'], p['wall_att'])
        # ---------------- physics (the bench's Voice: same chars, sounds, muscle sets, mouth rule)
        self.voice = v = _voice(self.seed, physics)
        if p['mouth_vocab'] is not None:
            v.set_mouth_vocab(set(''.join(lex.all_lexicon_texts())) if p['mouth_vocab'] == 'lexicon' else p['mouth_vocab'])
        rv = np.random.default_rng(self.seed + 3000)
        v.voice['w2_cg2'] = rv.choice([-1, 0, 1], p=[0.15, 0.7, 0.15], size=(v.F, v.T))
        for j in range(N):
            ra = np.random.default_rng([self.seed, 4000, j])
            v.voice[f'w2_a{j}'] = ra.choice([-1, 0, 1], p=[0.15, 0.7, 0.15], size=(v.F, v.T))
        self.cg_vkeys = ['other', 'other2', 'w2_cg2']          # 妈妈 = the bench partner, 爸爸 = other2, 奶奶 = new
        self.lawful = physics == 'lawful'
        self._tabs = {}
        self._dec = {}
        self._zero_tab = np.zeros((INAUD + 1, v.F, 2), np.int64)
        # ---------------- rngs
        self.rng = PyRng(self.seed, 61)                             # caregivers' choices (content)
        self.rng_anim = PyRng(self.seed, 62)
        self.rng_noise = np.random.default_rng([self.seed, 63])     # garble noise and ear noise (arrays)
        self.rng_place = PyRng(self.seed, 64)
        # ---------------- codes, held-out pairs, homes
        self.P = P = N + 3
        self.codes = lex.Codes(self.seed, P)
        self.held = lex.held_out_sets(self.seed, p['held_ratio'])
        self.home = lex.home_rooms(self.seed)
        self.names = [lex.AGENT_NAMES[j] for j in range(N)] + list(lex.CAREGIVERS)
        # ---------------- grid layers
        self.OBJ = [0] * NC                               # object id per cell (Python lists: scalar access is hot)
        self.PER = [0] * NC                               # person id + 1 per cell
        self.PTR = [0] * NC                               # number of caregivers pointing at the cell
        self._ter = S.TER.tolist()
        self._room = S.ROOM.tolist()
        self.CS = np.zeros(NC, np.int32)                  # cell state id -> row of self._ctab (composed eye block)
        self._walk = bytearray(S.WALK.astype(np.uint8).tobytes())   # Python-speed occupancy maps
        self._hold = bytearray(S.HOLD.astype(np.uint8).tobytes())
        self._blk = bytearray((~S.WALK).astype(np.uint8).tobytes())   # agents: not walkable, or a person / object there
        self._cblk = bytearray((~S.WALK).astype(np.uint8).tobytes())  # caregivers: not walkable, or a person (adults
                                                                      # step over objects and may stand on them)
        self._ob = bytearray(NC)
        self._pb = bytearray(NC)
        self._room_cells = {r: [int(c) for c in S.map_cells if S.ROOM[c] == r and S.HOLD[c]] for r in range(4)}
        self._room_tables = {r: [c for c in self._room_cells[r] if S.TER[c] == lex.T_TABLE] for r in range(4)}
        self._hver = 0
        self._aph_c = (-1, None)
        self._cstate = {}
        self._ctab = np.zeros((1024, EYE_B), np.uint8)
        self._n_cs = 0
        # ---------------- persons
        self.pos = np.zeros(P, np.int64)
        self.head = np.zeros(P, np.int64)
        self.hold = [0] * P                               # held object id (0 = none)
        self.present = [True] * P
        self.adult = [False] * N + [True] * 3
        self._pblock = [self.codes.person_block(i, self.adult[i]) for i in range(P)]
        # ---------------- objects (id 0 = none)
        self.o_kind, self.o_col, self.o_cnt, self.o_size = [0], [0], [0], [0]
        self.o_cell, self.o_holder, self.o_alive, self.o_tmove, self.o_t0 = [-1], [-1], [False], [0], [0]
        self.kind_objs = collections.defaultdict(set)     # kind index -> alive object ids
        # ---------------- agents' bodies and senses
        self.motor_perm = np.tile(np.arange(N_MOTOR), (N, 1))
        if p['motor_shuffle']:
            rp = np.random.default_rng([self.seed, 65])
            self.motor_perm = np.stack([rp.permutation(N_MOTOR) for _ in range(N)])
        self._busy = [0] * N
        self._mperm = self.motor_perm.tolist()
        self.EYE = np.zeros((N, N_VIEW * EYE_B), np.uint8)
        self._E3 = self.EYE.reshape(N, N_VIEW, EYE_B)
        self._seek = [None] * NC                          # cell -> {agent: view index} for agents that see it
        self._vcells = [[] for _ in range(N)]             # cells agent i currently sees
        self.TOUCH = np.zeros((N, TOUCH_N), np.uint8)
        self.TASTE = np.zeros((N, TASTE_N), np.uint8)
        self.HUNGER = np.full(N, p['hunger0'], np.float32)
        self.REW = np.zeros(N, np.float32)
        self._EL = 6 + 2 * int(p['ear_keep'])
        self._EB = np.zeros((N, v.F, self._EL), np.int64)
        self._EBro = self._EB.view()
        self._EBro.flags.writeable = False                # slices of it are read-only views for the agents
        self._zero_new = np.zeros((N, v.F, 2), np.int64)
        self._es = 0
        self.obs = {}
        for k, a in (('eye', self.EYE), ('touch', self.TOUCH), ('taste', self.TASTE), ('hunger', self.HUNGER),
                     ('reward', self.REW)):
            vw = a.view()
            vw.flags.writeable = False
            self.obs[k] = vw
        self._set_ear_view()
        # ---------------- mouth / bout state
        self._pany = [False] * N; self._pmask = [0] * N; self._ppmax = [0.0] * N
        self._atok = [None] * N; self._aph = [0] * N
        self._spoke = set()                # agents whose mouth was active at the previous tick
        self._asrc = {}                    # agent sources for the next tick: j -> (tok, ph, ch)
        self.last_mouth = [-10 ** 9] * N
        self._open = {}                    # j -> bout dict (open bouts)
        self._last_bout = [None] * N       # (t_end, text, listeners) of the agent's last closed bout
        self.listening = [None] * N        # j -> listen state (answer window) or None
        # ---------------- sound sources
        self._say = {}                     # caregiver pid -> (text, t0, vkey)
        self._ear_live = 0
        self._attc = {}
        self._aver = 0
        self._snd_vecs = _NO_SND           # this tick's (caregiver att vecs, (agent, att vec) list) for overlap stats
        self._taste_t = {}
        self.diag = collections.Counter()      # walk outcomes etc. (diagnostics)
        self._in_act = False
        self.meal_due = []
        # ---------------- eye bookkeeping
        self._chg = []
        self._ochg = []                    # (cell, object, added) object appearances / disappearances this tick
        self._moved = set()
        self._vis = [set() for _ in range(N)]
        self.seen_until = [dict() for _ in range(N)]   # object id -> last tick seen (while not in view)
        # ---------------- knowledge / exposure bookkeeping per agent (measurement only; agents never see it)
        self.exposures = [collections.Counter() for _ in range(N)]
        self.last_told = [dict() for _ in range(N)]
        self.first_told = [dict() for _ in range(N)]
        self.told_obj = [dict() for _ in range(N)]     # object id -> tick its place was told
        self.heard_sit = [set() for _ in range(N)]
        self.asked = [collections.Counter() for _ in range(N)]
        self.last_asked = [dict() for _ in range(N)]
        self.gave_food = [-10 ** 9] * N
        self.engaged = [None] * N
        self.last_attended = [-1] * N
        self.cooldown = [dict() for _ in range(N)]
        # ---------------- monitors (per agent counters)
        self.mon = {k: [0] * N for k in (
            'mouth_ticks', 'syllables', 'garble', 'bouts', 'overlap_cg', 'overlap_partner', 'overlap_ag',
            'interruptions', 'interruptions_partner', 'moves', 'bumps', 'turns', 'takes', 'puts', 'gives', 'eats',
            'fails', 'initiatives', 'cg_audible_ticks')}
        self.mon_f = {k: [0.0] * N for k in ('reward',)}
        # ---------------- time, log, events
        self.t = 0
        self.log = []
        self._heap = []
        self._seq = 0
        self.day_i = 0
        self.night = False
        self.bedtime_on = False
        self.t_day0 = 0
        self.agents = _Agents()
        self._priv = False
        self._pulse = False
        self._born = False
        self._ex_id = 0
        self.cgs = []
        homes = [cell_of(14, 3), cell_of(16, 4), cell_of(18, 3)]
        for k, nm in enumerate(lex.CAREGIVERS):
            self.cgs.append(CG(k, N + k, nm, self.cg_vkeys[k], homes[k]))
        self._corpus = None
        if p['p_corpus'] > 0 and phys.corpus_path() is not None:
            pool = phys.load_corpus()
            self._corpus = pool[:-3000]                  # the bench's training part of the corpus
            self._corpus_pos = int(self.rng.integers(len(self._corpus)))
        self._build_world()

    # ================================================================ building the world
    def _build_world(self):
        S, r = self.S, self.rng_place
        for c in range(NC):                                # every cell starts with its terrain-only eye block
            self._cell_changed(c)
        self._chg = []
        free = [int(c) for c in S.walk_cells if S.TER[c] != lex.T_DOOR]
        self._blk = bytearray((~S.WALK).astype(np.uint8).tobytes())
        self._cblk = bytearray((~S.WALK).astype(np.uint8).tobytes())
        # agents in the living room and the garden
        cand = [c for c in free if S.ROOM[c] in (1, 3)]
        order = r.permutation(len(cand))
        k = 0
        for j in range(self.N):
            while True:
                c = cand[int(order[k])]
                k += 1
                if self.PER[c] == 0 and all(self.PER[c + d] == 0 for d in DIRS):
                    break
            self.pos[j] = c
            self.head[j] = int(r.integers(4))
            self.PER[c] = j + 1
            self._cell_changed(c)
        for cg in self.cgs:
            self.pos[cg.pid] = cg.home
            self.PER[cg.home] = cg.pid + 1
            self._cell_changed(cg.home)
        # objects: food on the kitchen table, toys, animals
        for kind, room, n in (('苹果', 0, 1), ('香蕉', 0, 1), ('饼干', 0, 1), ('牛奶', 0, 1), ('皮球', 1, 1),
                              ('汽车', 2, 1), ('小狗', 1, 1), ('小猫', 2, 1), ('小鸟', 3, 1), ('鸭子', 3, 1)):
            c = self._free_cell_in_room(room, prefer_table=kind in EDIBLE)
            if c is not None:
                self._spawn(KIND_ID[kind], cell=c)
        self._moved = set(range(self.N))

    def _free_cell_in_room(self, room, prefer_table=False, rng=None):
        r = rng or self.rng_place
        ob, pb = self._ob, self._pb
        cells = None
        if prefer_table:
            cells = [c for c in self._room_tables[room] if not ob[c] and not pb[c]]
        if not cells:
            cells = [c for c in self._room_cells[room] if not ob[c] and not pb[c]]
        if not cells:
            return None
        return cells[int(r.integers(len(cells)))]

    # ================================================================ cells and eye codes
    def _cell_changed(self, c):
        """recompose the eye block of cell c (state-id table) and mark it changed."""
        o = self.OBJ[c]
        pp = self.PER[c]
        okey = None
        if o:
            okey = (self.o_kind[o], self.o_col[o], self.o_cnt[o], self.o_size[o])
        pkey = None
        if pp:
            h = self.hold[pp - 1]
            pkey = (pp - 1, (self.o_kind[h], self.o_col[h], self.o_cnt[h], self.o_size[h]) if h else None)
        key = (self._ter[c], okey, pkey, self.PTR[c])
        sid = self._cstate.get(key)
        if sid is None:
            sid = self._n_cs
            if sid >= len(self._ctab):
                self._ctab = np.concatenate([self._ctab, np.zeros_like(self._ctab)])
            b = self.codes.ter[key[0]].copy()
            if okey is not None:
                b |= self.codes.obj_block(*okey)
            if pkey is not None:
                b |= self._pblock[pkey[0]]
                if pkey[1] is not None:
                    b |= self.codes.obj_block(*pkey[1])
                    b[lex.E_HELD] = 1
            if key[3]:
                b[lex.E_PTR:lex.E_PTR + 2] = 1
            self._ctab[sid] = b
            self._cstate[key] = sid
            self._n_cs += 1
        self.CS[c] = sid
        self._chg.append(c)
        self._ob[c] = 1 if o else 0
        self._pb[c] = 1 if pp else 0
        self._blk[c] = 1 if (o or pp or not self._walk[c]) else 0
        self._cblk[c] = 1 if (pp or not self._walk[c]) else 0

    def _eyes(self):
        """update the eyes: a full gather for agents that moved / turned; for every other agent only the 64-unit
        blocks of the changed cells it sees (self._seek[c] = {agent: view index})."""
        if self.night:
            self._chg = []
            self._ochg = []
            self._moved = set()
            return
        moved = self._moved
        t = self.t
        seek = self._seek
        if moved:
            ii = sorted(moved)
            ids = self.S.VIEW[self.pos[ii], self.head[ii]]
            self.EYE[ii] = self._ctab[self.CS[ids]].reshape(len(ii), -1)
            OBJ = self.OBJ
            for k, i in enumerate(ii):
                cells = ids[k].tolist()
                for c in self._vcells[i]:                    # re-register what agent i sees
                    dct = seek[c]
                    if dct is not None:
                        dct.pop(i, None)
                for kk, c in enumerate(cells):
                    if c != VOID:
                        dct = seek[c]
                        if dct is None:
                            dct = seek[c] = {}
                        dct[i] = kk
                self._vcells[i] = cells
                new = {o for o in [OBJ[c] for c in cells] if o}
                old = self._vis[i]
                if old != new:
                    su = self.seen_until[i]
                    for o in old - new:
                        su[o] = t - 1
                    self._vis[i] = new
        if self._chg:
            E3, ctab, CS = self._E3, self._ctab, self.CS
            done = set()
            for c in self._chg:
                dct = seek[c]
                if not dct or c in done:
                    continue
                done.add(c)
                row = ctab[CS[c]]
                for a, k in dct.items():
                    if a not in moved:
                        E3[a, k] = row
            if self._ochg:
                for c, o, added in self._ochg:
                    dct = seek[c]
                    if not dct:
                        continue
                    for a in dct:
                        if a in moved:
                            continue
                        if added:
                            self._vis[a].add(o)
                        elif o in self._vis[a]:
                            self._vis[a].discard(o)
                            self.seen_until[a][o] = t - 1
        self._chg = []
        self._ochg = []
        self._moved = set()

    def view_cells(self, i):
        return self.S.VIEW[self.pos[i], self.head[i]]

    # ================================================================ objects
    def _spawn(self, kind, cell=None, holder=None, colour=None, count=None, size=None):
        r = self.rng_place
        o = len(self.o_kind)
        k = KINDS[kind]
        self.o_kind.append(kind)
        self.o_col.append(int(r.integers(6)) if colour is None else colour)
        self.o_cnt.append((1 if k in ANIMALS else int(r.integers(1, 4))) if count is None else count)
        self.o_size.append(int(r.integers(2)) if size is None else size)
        self.o_cell.append(-1); self.o_holder.append(-1); self.o_alive.append(True)
        self.o_tmove.append(self.t); self.o_t0.append(self.t)
        self.kind_objs[kind].add(o)
        if cell is not None:
            self._place(o, cell)
        elif holder is not None:
            self._give_to(holder, o)
        return o

    def _place(self, o, c):
        self.OBJ[c] = o
        self._ochg.append((c, o, True))
        self.o_cell[o] = c
        self.o_holder[o] = -1
        self.o_tmove[o] = self.t
        self._cell_changed(c)

    def _lift(self, o):
        c = self.o_cell[o]
        if c >= 0:
            self.OBJ[c] = 0
            self._ochg.append((c, o, False))
            self.o_cell[o] = -1
            self.o_tmove[o] = self.t
            self._cell_changed(c)

    def _give_to(self, pid, o):
        """object o goes into the hands of person pid (empty hands)."""
        self.hold[pid] = o
        self.o_holder[o] = pid
        self.o_cell[o] = -1
        self.o_tmove[o] = self.t
        if pid < self.N:
            self._touch_hold(pid)
        if self.present[pid]:
            self._cell_changed(int(self.pos[pid]))

    def _take_from(self, pid):
        o = self.hold[pid]
        self.hold[pid] = 0
        if o:
            self.o_holder[o] = -1
            self.o_tmove[o] = self.t
        if pid < self.N:
            self._touch_hold(pid)
        if self.present[pid]:
            self._cell_changed(int(self.pos[pid]))
        return o

    def _remove_obj(self, o):
        if self.o_cell[o] >= 0:
            self._lift(o)
        h = self.o_holder[o]
        if h >= 0 and self.hold[h] == o:
            self._take_from(h)
        self.o_alive[o] = False
        self.kind_objs[self.o_kind[o]].discard(o)

    def _touch_hold(self, j):
        T = self.TOUCH
        T[j, :14] = 0
        o = self.hold[j]
        if o:
            T[j, self.codes.touch[self.o_kind[o]]] = 1
            T[j, 12 + self.o_size[o]] = 1
        T[j, TO_HAND] = 1
        self._pulse = True

    def obj_room(self, o):
        c = self.o_cell[o]
        if c >= 0:
            return int(self.S.ROOM[c])
        h = self.o_holder[o]
        return int(self.S.ROOM[self.pos[h]]) if h >= 0 and self.present[h] else -1

    # ================================================================ persons moving
    def _move_person(self, pid, c):
        old = int(self.pos[pid])
        self.PER[old] = 0
        self.PER[c] = pid + 1
        self.pos[pid] = c
        self._cell_changed(old)
        self._cell_changed(c)
        if pid < self.N:
            self._moved.add(pid)
            self._aver += 1
            self._hver += 1

    def _swap_persons(self, a, b):
        ca, cb = int(self.pos[a]), int(self.pos[b])
        self.PER[ca], self.PER[cb] = b + 1, a + 1
        self.pos[a], self.pos[b] = cb, ca
        self._cell_changed(ca)
        self._cell_changed(cb)
        for x in (a, b):
            if x < self.N:
                self._moved.add(x)
                self._aver += 1
                self._hver += 1

    def free(self, c):
        return not self._blk[c]

    # ================================================================ sound
    def _set_ear_view(self):
        self.obs['ear'] = self._EBro[:, :, self._es:self._es + 6]

    def _tab(self, ch, vkey, ph):
        """attenuation table (INAUD+1, F, 2) of the 2 slices at phase ph of char ch in voice vkey."""
        key = (ch, vkey, ph)
        tb = self._tabs.get(key)
        if tb is None:
            g = self.voice.hear(ch, vkey)
            tb = self._mk_tab(g, ph)
            self._tabs[key] = tb
        return tb

    @staticmethod
    def _cols(g, ph):
        if ph + 2 <= g.shape[1]:
            return g[:, ph:ph + 2]
        last = g.shape[1] - 1
        return g[:, [min(ph, last), min(ph + 1, last)]]

    def _mk_tab(self, g, ph):
        cols = self._cols(np.asarray(g, np.int64), ph)
        a = np.arange(INAUD + 1)[:, None, None]
        tb = np.maximum(cols[None] - a, 0)
        tb[INAUD] = 0
        return tb

    def _agent_tabs(self, j, tok, ph, ch):
        """(table for the other listeners, own-voice cols) of agent j's mouth sound at phase ph."""
        php = min(ph, 6)
        if ch == '~' and not self.lawful:
            g = self.rng_noise.integers(0, 4, size=(self.voice.F, self.voice.T))   # garble: fresh noise (voice.py)
            tb = self._mk_tab(g, php)
            return tb, tb[0]
        if self.lawful:
            key = ('m', tok, j, php)
            r = self._tabs.get(key)
            if r is None:
                act = tuple(i for i in range(N_MOUTH) if tok >> i & 1)
                ext = np.clip(self.voice.set_sound(act) + self.voice.voice[f'w2_a{j}'], 0, self.voice.levels)
                own = self.voice.hear_set(act)
                r = (self._mk_tab(ext, php), self._cols(np.asarray(own, np.int64), php).copy())
                self._tabs[key] = r
            return r
        key = ('a', ch, j, php)
        r = self._tabs.get(key)
        if r is None:
            r = (self._mk_tab(self.voice.hear(ch, f'w2_a{j}'), php),
                 self._cols(np.asarray(self.voice.hear(ch, 'self'), np.int64), php).copy())
            self._tabs[key] = r
        return r

    def attv(self, cell):
        """(N,) attenuation indices of a source at `cell` for every agent (cached until an agent moves)."""
        e = self._attc.get(cell)
        if e is None or e[0] != self._aver:
            e = (self._aver, self.ATT[cell][self.pos[:self.N]])
            self._attc[cell] = e
        return e[1]

    def _sound(self, t):
        cgv, agv = [], []
        new = None
        for e in self._say.values():
            if e[3] != self._aver:
                e[3], e[4] = self._aver, self.ATT[e[2]][self.pos[:self.N]]
            av = e[4]
            cgv.append(av)
            v = e[0][t - e[1]][av]
            new = v if new is None else np.maximum(new, v, out=new)
        own = None
        if self._asrc:
            own = []
            for j, (tok, ph, ch) in self._asrc.items():
                tb, oc = self._agent_tabs(j, tok, ph, ch)
                av = self.attv(int(self.pos[j])).copy()
                av[j] = INAUD
                agv.append((j, av))
                v = tb[av]
                new = v if new is None else np.maximum(new, v, out=new)
                own.append((j, oc))
        if new is None:
            new = self._zero_new
            self._ear_live -= 1
        else:
            self._ear_live = 3
            for j, oc in own or ():
                np.maximum(new[j], oc, out=new[j])
        if self.noise > 0:                                # the bench's noise: max(signal, clip(rint(N(0, s)), 0, 10))
            nz = np.clip(np.rint(self.rng_noise.normal(0.0, self.noise, new.shape)), 0, self.voice.levels).astype(np.int64)
            new = nz if new is self._zero_new else np.maximum(new, nz)    # never write into the shared zero block
        s = self._es + 2
        if s + 6 > self._EL:
            self._EB[:, :, 0:4] = self._EB[:, :, s:s + 4]
            s = 0
        self._EB[:, :, s + 4:s + 6] = new
        self._es = s
        self._set_ear_view()
        self._snd_vecs = (cgv, agv)

    def audible(self, src_cell, dst_cell):
        return self.ATT[src_cell, dst_cell] < INAUD

    def heard_by(self, cell):
        """agents that hear a source at `cell`."""
        return np.flatnonzero(self.attv(cell) < INAUD).tolist()

    # ================================================================ the agents' actions
    def _act(self, A, t, nz):
        sp, mv = {}, {}
        if nz:
            rr, cc = np.nonzero(A > 0)
            if len(rr):
                for j, c, v in zip(rr.tolist(), cc.tolist(), A[rr, cc].tolist()):
                    if c < N_MOUTH:
                        e = sp.get(j)
                        if e is None:
                            sp[j] = [1 << c, v]
                        else:
                            e[0] |= 1 << c
                            if v > e[1]:
                                e[1] = v
                    else:
                        e = mv.get(j)
                        if e is None or v > e[0]:
                            mv[j] = (v, c - N_MOUTH)
        if sp or self._spoke:
            self._mouth(sp, t)
        if self._open:
            self._bout_check(t)
        if mv:
            busy = self._busy
            for j, (v, u) in mv.items():
                if busy[j] <= t:
                    self._motor(j, u, t)

    def _decode(self, mask):
        ch = self._dec.get(mask, 0)
        if ch == 0:
            ch = self.voice.decode(tuple(i for i in range(N_MOUTH) if mask >> i & 1))
            self._dec[mask] = ch
        return ch

    def _mouth(self, sp, t):
        """mouth output of this tick: decode (the bench's mouth and onset rules), own-voice source for the next tick,
        bouts, reply windows, overlaps."""
        for j in self._spoke.difference(sp):                  # stopped: the next sound restarts its phase
            self._pany[j] = False; self._pmask[j] = 0; self._ppmax[j] = 0.0; self._atok[j] = None
        self._asrc = {}
        self._spoke = set(sp)
        if not sp:
            return
        mon = self.mon
        cgv, agv = self._snd_vecs
        lawful = self.lawful
        say = self._say
        for j, (mask, pm) in sp.items():
            ch = self._decode(mask)
            onset = (not self._pany[j]) or mask != self._pmask[j] or pm < self._ppmax[j]
            self._pany[j] = True; self._pmask[j] = mask; self._ppmax[j] = pm
            tok = mask if lawful else ch
            ph = 0 if (onset or tok != self._atok[j]) else self._aph[j] + 2
            self._atok[j] = tok; self._aph[j] = ph
            self._asrc[j] = (tok, ph, ch)
            # ---- bookkeeping: bouts, replies, overlaps
            mon['mouth_ticks'][j] += 1
            aud_cg = any(av[j] < INAUD for av in cgv) if cgv else False
            aud_ag = any(av[j] < INAUD for jj, av in agv if jj != j) if agv else False
            e = self.engaged[j]
            aud_pt = e is not None and e in say
            if aud_cg:
                mon['overlap_cg'][j] += 1
            if aud_pt:
                mon['overlap_partner'][j] += 1
            if aud_ag:
                mon['overlap_ag'][j] += 1
            b = self._open.get(j)
            if b is None:
                b = dict(t0=t, chars=[], overlap=0, interrupt=aud_pt, interrupt_any=aud_cg, garble=0,
                         reply=self.listening[j] is not None)
                self._open[j] = b
                mon['bouts'][j] += 1
                if aud_cg:
                    mon['interruptions'][j] += 1
                if aud_pt:
                    mon['interruptions_partner'][j] += 1
                if not b['reply']:
                    self._a2a_check(j, t)
            if aud_cg or aud_ag:
                b['overlap'] += 1
            self.last_mouth[j] = t
            L = self.listening[j]
            if L is not None:
                if L['first'] is None:
                    L['first'] = t
                L['mouth'] += 1
                if aud_pt:
                    L['overlap'] += 1
            if onset:
                mon['syllables'][j] += 1
                b['chars'].append(ch)
                if ch == '~':
                    mon['garble'][j] += 1
                    b['garble'] += 1
                if L is not None:
                    L['chars'].append(ch)

    def _bout_check(self, t):
        for j in [j for j, b in self._open.items() if t - self.last_mouth[j] >= PAUSE]:
            b = self._open.pop(j)
            b['t1'] = self.last_mouth[j] + 1
            b['text'] = ''.join(b['chars'])
            if not b['reply'] and b['chars']:
                self._last_bout[j] = (b['t1'], self.heard_by(int(self.pos[j])))
                self._initiative(j, b)

    def _a2a_check(self, j, t):
        """agent j starts an utterance (not a reply to a caregiver) within 30 ticks after another agent's utterance
        that j could hear: an agent-agent turn."""
        for i in range(self.N):
            lb = self._last_bout[i]
            if i != j and lb is not None and 0 <= t - lb[0] <= 30 and j in lb[1]:
                self.log.append(dict(type='a2a', t=t, src=i, dst=j, gap=t - lb[0]))
                self._last_bout[i] = None
                break

    def _motor(self, j, u, t):
        S = self.S
        k = self._mperm[j][u]
        self._busy[j] = t + self.p['motor_ticks']
        mon = self.mon
        h = int(self.head[j])
        c = int(self.pos[j])
        if k < 4:                                             # forward, back, left, right (no turning)
            d = DIRS[(h + (0, 2, 3, 1)[k]) % 4]
            n = c + d
            if not self._blk[n]:
                self._move_person(j, n)
                mon['moves'][j] += 1
            else:
                self.TOUCH[j, TO_BUMP] = 1
                self._pulse = True
                mon['bumps'][j] += 1
        elif k < 6:                                           # turn left / right
            self.head[j] = (h + (3 if k == 4 else 1)) % 4
            self._moved.add(j)
            self._hver += 1
            mon['turns'][j] += 1
        elif k == 6:                                          # take the object in front
            f = c + DIRS[h]
            o = self.OBJ[f]
            if o and not self.hold[j] and KINDS[self.o_kind[o]] not in ANIMALS:
                self._lift(o)
                self._give_to(j, o)
                mon['takes'][j] += 1
                self._on_take(j, o)
            else:
                mon['fails'][j] += 1
        elif k == 7:                                          # put down / give to the person in front
            o = self.hold[j]
            f = c + DIRS[h]
            pp = self.PER[f]
            if o and pp and not self.hold[pp - 1] and self.present[pp - 1]:
                self._take_from(j)
                self._give_to(pp - 1, o)
                mon['gives'][j] += 1
                self._on_give(j, pp - 1, o)
            elif o and S.HOLD[f] and not pp and not self.OBJ[f]:
                self._take_from(j)
                self._place(o, f)
                mon['puts'][j] += 1
            else:
                mon['fails'][j] += 1
        else:                                                 # eat what is held
            o = self.hold[j]
            if o and KINDS[self.o_kind[o]] in EDIBLE:
                before = float(self.HUNGER[j])
                self.HUNGER[j] = max(0.0, before - self.p['hunger_eat'])
                self.REW[j] += before
                self.mon_f['reward'][j] += before
                self.TASTE[j] = 0
                self.TASTE[j, self.codes.taste[self.o_kind[o]]] = 1
                self._pulse = True
                self._schedule(t + self.p['taste_ticks'], self._taste_off, (j, t))
                self._taste_t[j] = t
                mon['eats'][j] += 1
                self.o_cnt[o] -= 1
                self.log.append(dict(type='eat', t=t, agent=j, kind=KINDS[self.o_kind[o]], hunger=before))
                if self.o_cnt[o] <= 0:
                    self._remove_obj(o)
                    self._touch_hold(j)
                else:
                    self._cell_changed(c)
            else:
                mon['fails'][j] += 1

    def _taste_off(self, arg):
        j, t0 = arg
        if self._taste_t.get(j) == t0:
            self.TASTE[j] = 0

    # ================================================================ events
    def _schedule(self, t, fn, arg=None):
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, fn, arg))

    def _events(self, t):
        heap = self._heap
        while heap and heap[0][0] <= t:
            _, _, fn, arg = heapq.heappop(heap)
            fn(arg)

    def _clock100(self, _):
        self.HUNGER += np.float32(100 * self.p['hunger_rate'])
        np.minimum(self.HUNGER, 1.0, out=self.HUNGER)
        self._schedule(self.t + 100, self._clock100)

    def _monitor(self, _):
        self.log.append(dict(type='mon', t=self.t, day=self.day_i, hunger=self.HUNGER.round(3).tolist(),
                             **{k: list(v) for k, v in self.mon.items()},
                             **{k: [round(x, 3) for x in v] for k, v in self.mon_f.items()}))
        self._schedule(self.t + self.p['mon_every'], self._monitor)

    # ---------------- days and nights
    def _morning(self, _=None):
        p = self.p
        self.night = False
        self.bedtime_on = False
        self.day_i += 1
        self.t_day0 = self.t
        for cg in self.cgs:
            cg.greeted = set(); cg.bed_done = False; cg.stack = []; cg.op = None; cg.token += 1
            cg.partner = None; cg.held_by = None
            c = cg.home
            if self.PER[c]:
                cands = [int(x) for x in self.S.walk_cells if self.S.ROOM[x] == 2 and self.free(int(x))]
                c = cands[0]
            self.present[cg.pid] = True
            self.pos[cg.pid] = c
            self.PER[c] = cg.pid + 1
            self._cell_changed(c)
        self._moved = set(range(self.N))
        self.log.append(dict(type='event', t=self.t, what='day_start', day=self.day_i))
        d0 = self.t
        self._schedule(d0 + p['day_ticks'] - p['bedtime'], self._bedtime)
        self._schedule(d0 + p['day_ticks'], self._night)
        self.meal_due = []
        for f in p['meals']:
            self._schedule(d0 + int(f * p['day_ticks']), self._meal_due)
        for k, cg in enumerate(self.cgs):
            self._schedule(self.t + 1 + 7 * k, self._cg_start, cg)

    def _meal_due(self, _):
        self.meal_due.append(self.t)

    def _bedtime(self, _):
        self.bedtime_on = True
        for cg in self.cgs:
            if cg.op is not None and cg.op[0] in PREEMPTIBLE and cg.held_by is None:
                self._abandon(cg)
                self._cg_resume(cg, None)

    def _night(self, _):
        self.night = True
        t = self.t
        for cg in self.cgs:
            self._abandon(cg)
            self._say.pop(cg.pid, None)
            o = self.hold[cg.pid]
            c = int(self.pos[cg.pid])
            self.PER[c] = 0
            self.present[cg.pid] = False
            if o:
                self._take_from(cg.pid)
                cc = c if self.S.HOLD[c] else self._free_cell_in_room(int(self.S.ROOM[c]) if self.S.ROOM[c] >= 0 else 1)
                if cc is not None:
                    self._place(o, cc)
                else:
                    self._remove_obj(o)
            self._cell_changed(c)
        for j in range(self.N):
            self.engaged[j] = None
            if self.listening[j] is not None:
                self.listening[j] = None
            su = self.seen_until[j]
            for o in self._vis[j]:
                su[o] = t - 1
            self._vis[j] = set()
        self.EYE[:] = 0
        self._chg = []
        self._ochg = []
        self._moved = set()
        self.log.append(dict(type='event', t=t, what='night_start', day=self.day_i))
        self._schedule(t + self.p['night_ticks'], self._morning)

    # ---------------- animals wander
    def _animal(self, o):
        if not self.o_alive[o]:
            return
        r = self.rng_anim
        c = self.o_cell[o]
        if c >= 0 and not self.night:
            d = DIRS[int(r.integers(4))]
            n = c + d
            garden_only = KINDS[self.o_kind[o]] in ('小鸟', '鸭子')
            if self.free(n) and self._ter[n] != lex.T_DOOR and (not garden_only or self._room[n] == 3):
                self.OBJ[c] = 0
                self._ochg.append((c, o, False))
                self._cell_changed(c)
                self._place(o, n)
        self._schedule(self.t + int(r.integers(30, 120)), self._animal, o)

    # ================================================================ caregiver op driver
    def _cg_start(self, cg):
        if self.night or cg.stack or cg.op is not None:
            return
        self._cg_resume(cg, None)

    def _abandon(self, cg):
        """drop every script of cg (generators closed: their finally clauses release agents, pointing, ...)."""
        cg.token += 1
        st, cg.stack, cg.op = cg.stack, [], None
        self._say.pop(cg.pid, None)
        cg.speaking = False
        self._point(cg, None)
        for g in reversed(st):
            try:
                g.close()
            except Exception:
                pass
        for j in range(self.N):
            L = self.listening[j]
            if L is not None and L.get('cg') is cg:
                self.listening[j] = None

    def _cg_resume(self, cg, result):
        while True:
            if self.night or not self.present[cg.pid]:
                return
            if not cg.stack:
                g = self._new_activity(cg)
                if g is None:
                    self._cg_op(cg, ('wait', 50))
                    return
                cg.stack.append(g)
                result = None
            g = cg.stack[-1]
            try:
                op = g.send(result)
            except StopIteration:
                cg.stack.pop()
                result = None
                continue
            result = self._cg_op(cg, op)
            if result is PENDING:
                return

    def _cg_wake(self, arg):
        cg, token = arg
        if token != cg.token or self.night:
            return
        op = cg.op
        kind = op[0]
        t = self.t
        if kind == 'wait':
            cg.op = None
            self._cg_resume(cg, True)
        elif kind == 'say':
            if cg.speaking:
                self._say_end(cg)
            else:
                self._say_start(cg)
        elif kind == 'goto':
            self._goto_step(cg)
        elif kind == 'listen':
            self._listen_check(cg)
        elif kind == 'watch':
            st = op[3]
            ok = bool(op[1]())
            if ok or t >= st['deadline']:
                cg.op = None
                self._cg_resume(cg, (ok, t - st['t0']))
            else:
                self._cg_later(cg, 3)

    def _cg_later(self, cg, dt):
        self._schedule(self.t + max(1, int(dt)), self._cg_wake, (cg, cg.token))

    def _cg_op(self, cg, op):
        """start op; returns PENDING for timed ops, else the immediate result."""
        kind = op[0]
        t = self.t
        if kind == 'wait':
            cg.op = op
            self._cg_later(cg, op[1])
            return PENDING
        if kind == 'say':
            cg.op = op
            cg.say_wait = 0
            cg.speaking = False
            self._say_start(cg)
            return PENDING
        if kind == 'goto':
            cg.op = (op[0], op[1], dict(t0=t, blocked=0, stuck=0, nobfs=0, path=None,
                                        deadline=t + op[2] if len(op) > 2 else t + 600))
            r = self._goto_step(cg, first=True)
            return PENDING if r is None else r
        if kind == 'listen':
            agents, kw = op[1], op[2]
            st = dict(t0=t, agents=list(agents), min_gap=kw.get('min_gap', self.p['gap_answer']),
                      cap=kw.get('cap', self.p['reply_cap']), after=self.p['pause_after'] + 1, res={})
            for j in st['agents']:
                self.listening[j] = dict(cg=cg, t0=t, first=None, chars=[], mouth=0, overlap=0,
                                         garble0=int(self.mon['garble'][j]))
            cg.op = ('listen', st)
            self._cg_later(cg, st['min_gap'])
            return PENDING
        if kind == 'watch':
            cg.op = ('watch', op[1], op[2], dict(t0=t, deadline=t + op[2]))
            self._cg_later(cg, 3)
            return PENDING
        if kind == 'point':
            self._point(cg, op[1])
            return True
        if kind == 'call':                                     # an instantaneous world function
            return op[1](*op[2:])
        raise ValueError(kind)

    def _point(self, cg, c):
        """cg points at cell c (None: stops pointing)."""
        old = cg.act
        if old is not None and old >= 0:
            self.PTR[old] = max(0, self.PTR[old] - 1)
            self._cell_changed(old)
        cg.act = c
        if c is not None and c >= 0:
            self.PTR[c] += 1
            self._cell_changed(c)

    # ---------------- say
    def _say_start(self, cg):
        op = cg.op
        text, kw = op[1], op[2]
        speaker = kw.get('speaker', cg)
        cell = int(self.pos[speaker.pid])
        if cg.say_wait < kw.get('max_wait', 30) and not self._quiet_at(cell, speaker.pid):
            cg.say_wait += 3
            self._cg_later(cg, 3)
            return
        t = self.t + (1 if self._in_act else 0)
        cg.speaking = True
        vk = speaker.vkey
        self._say[speaker.pid] = [[self._tab(ch, vk, ph) for ch in text for ph in (0, 2, 4)], t, int(self.pos[speaker.pid]),
                                  -1, None]
        to = kw.get('to')
        if kw.get('reward') and to is not None:
            self.REW[to] += kw['reward']
            self.mon_f['reward'][to] += kw['reward']
            self._pulse = True
        kw['_t0'] = t
        kw['_speaker'] = speaker
        self._ear_live = 3
        self._cg_later(cg, t + SOUND * len(text) - self.t)

    def _say_end(self, cg):
        op = cg.op
        text, kw = op[1], op[2]
        speaker = kw['_speaker']
        self._say.pop(speaker.pid, None)
        cg.speaking = False
        cg.op = None
        cell = int(self.pos[speaker.pid])
        rec = dict(type='say', t=kw['_t0'], by=speaker.name, text=text, to=kw.get('to'), kind=kw.get('kind', 'stmt'))
        self.log.append(rec)
        listeners = self.heard_by(cell)
        cat = self.mon['cg_audible_ticks']
        for j in listeners:
            cat[j] += SOUND * len(text)
        if kw.get('teach') is not None:
            self._taught(listeners, kw['teach'])
        if kw.get('tell') is not None:
            self._told(listeners, kw['tell'], kw.get('to'))
        self._cg_resume(cg, True)

    def _quiet_at(self, cell, me):
        for pid in self._say:
            if pid != me and self.ATT[cell, self.pos[pid]] < INAUD:
                return False
        t = self.t
        for j in self._open:
            if t - self.last_mouth[j] <= 2 and self.ATT[cell, self.pos[j]] < INAUD:
                return False
        return True

    # ---------------- listen (answer windows; the bench's timing)
    def _listen_check(self, cg):
        st = cg.op[1]
        t = self.t
        pending = []
        for j in st['agents']:
            if j in st['res']:
                continue
            L = self.listening[j]
            if L is None:                                     # cleared (night)
                st['res'][j] = dict(reply='', first=None, stop='none')
                continue
            el = t - st['t0']
            if L['first'] is None:
                done = el >= st['min_gap']
                stop = 'silent'
            else:
                silent = t - 1 - self.last_mouth[j]
                done = (silent >= st['after'] and el >= st['min_gap'])
                stop = 'self'
                if not done and el >= st['cap']:
                    done, stop = True, 'cut'
            if done:
                st['res'][j] = dict(reply=''.join(L['chars']), first=L['first'], stop=stop, mouth=L['mouth'],
                                    overlap=L['overlap'], garble=int(self.mon['garble'][j]) - L['garble0'],
                                    t_close=t)
                self.listening[j] = None
                if j in self._open:
                    self._open[j]['reply'] = True
            else:
                pending.append(j)
                if L['first'] is None:
                    nxt = st['t0'] + st['min_gap']
                else:
                    nxt = min(max(self.last_mouth[j] + 1 + st['after'], st['t0'] + st['min_gap']), st['t0'] + st['cap'])
                st.setdefault('next', []).append(nxt)
        if not pending:
            cg.op = None
            self._cg_resume(cg, st['res'])
        else:
            nx = [x for x in st.pop('next', []) if x > t]
            self._cg_later(cg, (min(nx) - t) if nx else 1)

    # ---------------- goto (greedy descent on the static distance map, BFS around dynamic obstacles)
    def _goal_cells(self, spec):
        """the set of cells that satisfy a goto spec: ('adj4', cell) 4-neighbours, ('adj8', cell) 8-neighbourhood,
        ('cell', cell), ('person', pid, mode) next to a person ('front' = front/left/right cells of an agent)."""
        kind = spec[0]
        if kind == 'person':
            pid = spec[1]
            c = int(self.pos[pid])
            if spec[2] == 'front' and pid < self.N:
                h = int(self.head[pid])
                g = tuple(c + DIRS[(h + k) % 4] for k in (0, 3, 1))
            else:
                g = tuple(c + d for d in DIRS)
            walk = self._walk                                        # do not stand in a doorway if avoidable
            g2 = tuple(x for x in g if walk[x] and x not in self.S.doorzone)
            return g2 or tuple(x for x in g if walk[x]) or g
        c = spec[1]
        if kind == 'cell':
            return (c,)
        if kind == 'adj4':
            return tuple(c + d for d in DIRS)
        return tuple(c + d1 + d2 for d1 in (-PW, 0, PW) for d2 in (-1, 0, 1) if d1 or d2)

    def _goto_step(self, cg, first=False):
        """one step of a walk: follow a planned path, else greedy descent on the static distance map; after 2
        consecutive blocked steps plan around the obstacles (A*, bidirectional BFS fallback); a 'front' target
        accepts any side of the person after 6 blocked steps in total; the walk fails after 12 consecutive blocked
        steps or at its deadline."""
        op = cg.op
        spec, st = op[1], op[2]
        pid = cg.pid
        c = int(self.pos[pid])
        tkey = (int(self.pos[spec[1]]), int(self.head[spec[1]])) if spec[0] == 'person' else 0
        if st.get('tkey') != tkey:
            st['tkey'] = tkey
            st['goals'] = goals = self._goal_cells(spec)
            st['goals_any'] = self._goal_cells(('person', spec[1], 'any')) \
                if (spec[0] == 'person' and spec[2] == 'front') else goals
            st['d'] = st['d_any'] = None
            st['path'] = None
        use_any = st['stuck'] > 6
        goals = st['goals_any'] if use_any else st['goals']
        if c in goals:
            cg.op = None
            self.diag['goto_ok'] += 1
            if first:
                return True
            self._cg_resume(cg, True)
            return None
        if self.t >= st['deadline'] or st['blocked'] > 12:
            cg.op = None
            self.diag['goto_fail_deadline' if self.t >= st['deadline'] else 'goto_fail_blocked'] += 1
            if first:
                return False
            self._cg_resume(cg, False)
            return None
        nxt = None
        path = st.get('path')
        blk = self._cblk
        if path:
            n = path[0]
            if not blk[n]:
                nxt = n
                path.pop(0)
            else:
                st['path'] = None
        if nxt is None:
            key = 'd_any' if use_any else 'd'
            d = st[key] or self.S.dist_to(goals)
            st[key] = d
            dc = d[c]
            best = None
            for dd in DIRS:
                n = c + dd
                if d[n] < dc and not blk[n] and (best is None or d[n] < d[best]):
                    best = n
            if best is None:                                  # someone in the way: swap places
                for dd in DIRS:
                    n = c + dd
                    q = self.PER[n] - 1
                    if d[n] >= dc or q < 0:
                        continue
                    if q >= self.N:                           # an idle caregiver steps aside
                        o = self.cgs[q - self.N]
                        if o.op is None or o.op[0] != 'wait' or o.pid in self._say:
                            continue
                        self.diag['swaps'] += 1
                    elif st['blocked'] >= 2:                  # squeezing past a child who stays in the way: the
                        self.diag['nudges'] += 1              # child is nudged one cell (logged)
                        self.log.append(dict(type='event', t=self.t, what='nudge', agent=q, by=cg.name))
                    else:
                        continue
                    self._swap_persons(pid, q)
                    st['blocked'] = 0
                    self._cg_later(cg, self.p['cg_step_ticks'])
                    return None
            if best is not None:
                nxt = best
            else:
                st['blocked'] += 1
                st['stuck'] += 1
                if st['blocked'] >= 2 and st['nobfs'] <= 0:
                    path = self._astar(c, goals, d, blk=blk)
                    if path == 'budget':
                        path = self._bfs(c, goals, blk=blk)
                    self.diag['plans'] += 1
                    if path:
                        st['path'] = path
                        nxt = path.pop(0)
                    else:
                        st['nobfs'] = 3
                else:
                    st['nobfs'] -= 1
        if nxt is not None:
            st['blocked'] = 0
            self._move_person(pid, nxt)
        self._cg_later(cg, self.p['cg_step_ticks'])
        return None

    def _astar(self, start, goals, d, budget=200, blk=None):
        """A* over currently free cells with the static distance map d (to these goals) as the heuristic; returns
        a path, None (no free goal) or 'budget' (gave up after `budget` expansions).  blk: the blocking map (agents'
        by default)."""
        blk = self._blk if blk is None else blk
        gs = {g for g in goals if not blk[g] and g != start}
        if not gs:
            return None
        hp = heapq.heappush
        op = [(d[start], 0, start)]
        best = {start: 0}
        prev = {start: None}
        n = 0
        while op:
            f, g, c = heapq.heappop(op)
            if g > best[c]:
                continue
            if c in gs:
                path = []
                while c != start:
                    path.append(c)
                    c = prev[c]
                return path[::-1]
            n += 1
            if n > budget:
                return 'budget'
            g1 = g + 1
            for dd in DIRS:
                m = c + dd
                if blk[m] and m not in gs:
                    continue
                if g1 < best.get(m, 10 ** 9):
                    best[m] = g1
                    prev[m] = c
                    hp(op, (g1 + d[m], g1, m))
        return None

    def _bfs(self, start, goals, blk=None):
        """shortest path (list of cells, excluding start) to any free goal cell over currently free cells.
        Bidirectional: the smaller frontier is expanded first, so a search into (or out of) a closed pocket fails after
        exploring only the pocket."""
        blk = self._blk if blk is None else blk
        gs = [g for g in goals if not blk[g] and g != start]
        if not gs:
            return None
        pf = {start: None}
        pb = dict.fromkeys(gs)
        ff, fb = [start], gs
        meet = None
        while ff and fb and meet is None:
            if len(ff) <= len(fb):
                nf = []
                for c in ff:
                    for dd in DIRS:
                        n = c + dd
                        if n in pf or blk[n]:
                            continue
                        pf[n] = c
                        if n in pb:
                            meet = n
                            break
                        nf.append(n)
                    if meet is not None:
                        break
                ff = nf
            else:
                nb = []
                for c in fb:
                    for dd in DIRS:
                        n = c + dd
                        if n in pb:
                            continue
                        if n == start or (not blk[n] and n in pf):
                            pb[n] = c
                            meet = n
                            break
                        if blk[n]:
                            continue
                        pb[n] = c
                        nb.append(n)
                    if meet is not None:
                        break
                fb = nb
        if meet is None:
            return None
        path = []
        c = meet
        while c is not None and c != start:
            path.append(c)
            c = pf.get(c)
        path.reverse()
        c = pb.get(meet)
        while c is not None:
            path.append(c)
            c = pb.get(c)
        if meet == start:                       # met at the start: the backward chain is the whole path
            path = []
            c = pb[start]
            while c is not None:
                path.append(c)
                c = pb.get(c)
        return path

    def _preempt(self, cg, gen, walks=True, hold_by=None):
        """push a reaction script on top of cg's current script if cg is interruptible: waiting or walking, or (for a
        reaction that does not walk away) watching for a request's outcome (the watch then resumes).  hold_by: cg is
        taken into another caregiver's chat (held until released)."""
        if self.night or not self.present[cg.pid] or cg.held_by is not None or cg.pid in self._say:
            return False
        if cg.op is not None and cg.op[0] not in PREEMPTIBLE and not (cg.op[0] == 'watch' and not walks):
            return False
        if cg.op is None and cg.stack:
            return False
        cg.token += 1
        cg.op = None
        if hold_by is not None:
            cg.held_by = hold_by
        cg.stack.append(gen)
        self._cg_resume(cg, None)
        return True

    # ================================================================ measurement bookkeeping
    def eye_hash(self, j):
        return hash(self.EYE[j].tobytes())

    def _taught(self, listeners, teach):
        """an answer was said: (item, prompt, reply).  Every agent in hearing range experienced it."""
        item, prompt, reply = teach
        t = self.t
        for j in listeners:
            self.exposures[j][item] += 1
            self.last_told[j][item] = t
            self.first_told[j].setdefault(item, t)
            eh = self.eye_hash(j)
            self.heard_sit[j].add((prompt, eh))
            if self._priv:
                f = getattr(self.agents, 'bench_taught', None)
                if f is not None:
                    f(j, dict(prompt=prompt, reply=reply, item=item, eye_hash=eh))

    def _told(self, listeners, tell, to):
        """a fact was stated: tell = (item, extra) ; item counts as an experience for every listener."""
        item, extra = tell
        t = self.t
        for j in listeners:
            it = item(j) if callable(item) else item
            if it is None:
                continue
            self.exposures[j][it] += 1
            self.last_told[j][it] = t
            self.first_told[j].setdefault(it, t)
            if extra is not None and extra[0] == 'obj':
                self.told_obj[j][extra[1]] = t

    def _hook_prompt(self, j, info):
        if self._priv:
            f = getattr(self.agents, 'bench_prompt', None)
            if f is not None:
                f(j, info)

    # ================================================================ running
    def warmup_windows(self, n=60, seed=1, pause=4):
        """the bench's birth warm-up (60 random chars in the partner's voice, 4 silent ticks each) as ear windows: the
        only non-obs data an agent may take from the world (innate calibration)."""
        v = self.voice
        st = phys.Stream(v.F, v.T, v.T // SOUND, 3, v.levels, self.noise, noise_seed=self.seed * 7919 + 17)
        rng = np.random.default_rng(seed)
        out, none = [], (None, None, False)
        for _ in range(n):
            ch = v.chars[rng.integers(len(v.chars))]
            for k in range(SOUND):
                out.append(st.tick(((ch, v.hear(ch, 'other'), k == 0), none, none)).copy())
            for _ in range(pause):
                out.append(st.tick((none, none, none)).copy())
        return out

    def attach(self, agents):
        self.agents = agents
        self._priv = bool(getattr(agents, 'privileged', False))
        if self._priv and hasattr(agents, 'bench_attach'):
            agents.bench_attach(self)

    def run(self, agents, ticks=None, days=None, hours=None, verbose=0):
        """live `ticks` more ticks (or days = day+night, or hours of 48,000 ticks).  May be called repeatedly (life
        goes on); the same agents object should be passed every time."""
        if ticks is None:
            ticks = int(days * (self.p['day_ticks'] + self.p['night_ticks'])) if days is not None else \
                int((hours or 1.0) * 48000)
        if agents is not self.agents:
            self.attach(agents)
        if not self._born:
            self._born = True
            self._schedule(self.t, self._morning)
            self._schedule(self.t + 100, self._clock100)
            self._schedule(self.t + self.p['mon_every'], self._monitor)
            for o in range(1, len(self.o_kind)):
                if KINDS[self.o_kind[o]] in ANIMALS:
                    self._schedule(self.t + int(self.rng_anim.integers(30, 120)), self._animal, o)
        step = agents.step
        obs = self.obs
        heap = self._heap
        N = self.N
        zb32 = bytes(N * N_ACT * 4)
        t = self.t
        t_end = t + int(ticks)
        noise = self.noise > 0
        lastA = None
        next_v = t + (verbose or 10 ** 12)
        while t < t_end:
            self.t = t
            if heap[0][0] <= t:
                self._events(t)
            if self._say or self._asrc or self._ear_live > 0 or noise:
                self._sound(t)
            elif self._snd_vecs is not _NO_SND:
                self._snd_vecs = _NO_SND
            if self._chg or self._moved:
                self._eyes()
            A = step(obs)
            if self._pulse:
                self.REW[:] = 0
                self.TOUCH[:, TO_BUMP] = 0
                self.TOUCH[:, TO_HAND] = 0
                self._pulse = False
            if A is not lastA:
                if type(A) is not np.ndarray or A.dtype != np.float32 or not A.flags.c_contiguous:
                    A = np.ascontiguousarray(A, dtype=np.float32)
                    lastA = None
                else:
                    lastA = A
                if A.shape != (N, N_ACT):
                    raise ValueError(f'agents.step must return ({N}, {N_ACT}) actions, got {A.shape}')
            nz = A.tobytes() != zb32
            if nz or self._spoke or self._open:
                self._in_act = True
                self._act(A, t, nz)
                self._in_act = False
            t += 1
            if t >= next_v:
                next_v += verbose
                print(f'  t={t} day {self.day_i} log {len(self.log)}', flush=True)
        self.t = t
        return self.log
