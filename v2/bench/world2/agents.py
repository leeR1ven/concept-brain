# -*- coding: utf-8 -*-
"""Agent wrappers and reference agents for world2.

Batched API (one model drives many bodies):   agents.step(obs) -> (N, 41) float32, obs = dict of (N, ...) arrays.
Per-agent loops:                               PerAgent([a_0, ..., a_{N-1}]),  a_j.step(obs_j) -> (41,)
Bench agents (mouth only, 32 muscles):         PerAgent([MouthOnly(bench_agent), ...])

Reference agents (privileged ones get world hooks that ordinary agents never get: bench_attach(world),
bench_prompt(j, info) when a caregiver addresses agent j, bench_taught(j, info) when agent j hears a taught answer):
    Silent        the trivial agent
    RandomActor   random motor units and random 4-muscle babbles (chance level for every measure)
    Echo          repeats whatever is addressed to it (questions, requests, greetings), never acts
    Lookup        (prompt, exact eye input) -> the reply it heard taught for exactly that input, never acts
    Oracle        answers from the world state and carries out requests by path planning
    Namer         every K quiet ticks names a random object it sees (initiative reference)
Optional innate reflex (documented, off unless used):  Explore(agent) = random motor twitch after `idle` ticks
without any motor output of the wrapped agent.
"""
from __future__ import annotations

import collections

import numpy as np

from .core import DIRS, N_ACT, N_MOTOR, N_MOUTH, MOTOR
from .lex import ANIMALS, KINDS

LATENCY = 6


# ------------------------------------------------------------------------------------------ wrappers
class PerAgent:
    """run a list of single agents (a_j.step(obs_j) -> (41,)); obs_j holds row views (ear (8, 6), eye (1600,),
    touch (16,), taste (8,)) and floats (hunger, reward)."""

    def __init__(self, agents):
        self.agents = list(agents)
        self.out = None
        self._obs = None

    def step(self, obs):
        n = len(self.agents)
        if self.out is None:
            self.out = np.zeros((n, N_ACT), np.float32)
            self._obs = [dict() for _ in range(n)]
        ear, eye, touch, taste = obs['ear'], obs['eye'], obs['touch'], obs['taste']
        hun, rew = obs['hunger'].tolist(), obs['reward'].tolist()
        out = self.out
        for j, a in enumerate(self.agents):
            o = self._obs[j]
            o['ear'] = ear[j]; o['eye'] = eye[j]; o['touch'] = touch[j]; o['taste'] = taste[j]
            o['hunger'] = hun[j]; o['reward'] = rew[j]
            out[j] = a.step(o)
        return out


class MouthOnly:
    """a bench agent (step(obs) -> (32,) muscles, obs = ear / eye / reward / cue) living in world2 as a body that
    cannot move.  The ear window is copied (bench agents may keep it); eye = world2's (1600,) code."""

    def __init__(self, agent):
        self.agent = agent
        self.out = np.zeros(N_ACT, np.float32)

    def step(self, o):
        m = self.agent.step({'ear': np.array(o['ear']), 'eye': o['eye'], 'reward': o['reward'], 'cue': None})
        self.out[:N_MOUTH] = m
        return self.out


class Explore:
    """optional generic innate reflex: when the wrapped (batched) agent has produced no motor output for `idle`
    ticks, one random motor unit is twitched (force 1, one tick).  No unit has a built-in meaning for the reflex."""

    def __init__(self, agents, n_agents, idle=40, seed=0):
        self.inner = agents
        self.idle = idle
        self.quiet = np.zeros(n_agents, np.int64)
        self.rng = np.random.default_rng(seed)
        self.privileged = getattr(agents, 'privileged', False)

    def __getattr__(self, k):                         # pass hooks through
        return getattr(self.inner, k)

    def step(self, obs):
        A = np.array(self.inner.step(obs), np.float32)
        mot = A[:, N_MOUTH:].max(1) > 0
        self.quiet = np.where(mot, 0, self.quiet + 1)
        due = np.flatnonzero(self.quiet >= self.idle)
        for j in due:
            A[j, N_MOUTH + int(self.rng.integers(N_MOTOR))] = 1.0
            self.quiet[j] = 0
        return A


# ------------------------------------------------------------------------------------------ simple references
class Silent:
    """the trivial agent."""

    def __init__(self, n_agents):
        self.A = np.zeros((n_agents, N_ACT), np.float32)

    def step(self, obs):
        return self.A


class RandomActor:
    """each tick: with p_motor a random motor unit (force 1, one tick); with p_babble (when not babbling) a syllable of
    a random 4-of-32 muscle set held 3 ticks with force 1-2-3 (re-articulated each time)."""

    def __init__(self, n_agents, p_motor=0.1, p_babble=0.01, seed=0):
        import random
        self.N = n_agents
        self.A = np.zeros((n_agents, N_ACT), np.float32)
        self.r = random.Random(seed)
        self.p_motor, self.p_babble = p_motor, p_babble
        self.bab = [0] * n_agents
        self.sets = [None] * n_agents
        self.rows = []

    def step(self, obs):
        A, r = self.A, self.r
        for j in self.rows:
            A[j] = 0
        rows = []
        rnd, pm, pb = r.random, self.p_motor, self.p_babble
        for j in range(self.N):
            if rnd() < pm:
                A[j, N_MOUTH + r.randrange(N_MOTOR)] = 1.0
                rows.append(j)
            b = self.bab[j]
            if b == 0 and rnd() < pb:
                self.sets[j] = r.sample(range(N_MOUTH), 4)
                b = 3
            if b > 0:
                A[j, self.sets[j]] = float(4 - b)
                self.bab[j] = b - 1
                rows.append(j)
        self.rows = rows
        return A


class Looker:
    """eye_mode='fovea' reference that uses ONLY obs (the documented retinotopic layout and the motor order): when a
    pointed cell is seen outside the fovea it plans, inside its visible 5 x 5 frame, a path over the cells its eye shows
    as free to a cell next to the pointed one, walks there (forward / back / sideways) and turns to face it, so that the
    pointed cell comes into the fovea.  It never speaks."""

    def __init__(self, n_agents):
        from . import lex as L
        self.N = n_agents
        self.A = np.zeros((n_agents, N_ACT), np.float32)
        nm = len(L.MID_K)
        self.f_ptr = L.F_PTR
        ks = list(L.MID_K) + list(L.OUT_K)
        self.rc = [(k // 5, k % 5 - 2) for k in ks]
        self.ptr_cols = [L.FB + i * L.MB + L.M_PTR for i in range(nm)] + \
            [L.FB + nm * L.MB + i * L.OB + L.O_PTR for i in range(len(L.OUT_K))]
        walk_t = [L.T_DOOR, L.T_K, L.T_L, L.T_B, L.T_G]
        # per view cell: (units meaning walkable terrain, units meaning blocked: object / person / obstacle)
        self.cells = {(1, 0): ([L.F_TER + x for x in walk_t],
                               list(range(L.F_CLS, L.F_CLS + 8)) + [L.F_ADULT, L.F_CHILD])}
        for i, k in enumerate(L.MID_K):
            b = L.FB + i * L.MB
            self.cells[(k // 5, k % 5 - 2)] = ([b + L.M_TER + x for x in walk_t],
                                              list(range(b + L.M_CLS, b + L.M_CLS + 8)) + [b + L.M_ADULT, b + L.M_CHILD])
        for i, k in enumerate(L.OUT_K):
            b = L.FB + nm * L.MB + i * L.OB
            self.cells[(k // 5, k % 5 - 2)] = ([b + L.O_DOOR, b + L.O_FLOOR], [b + L.O_OBST, b + L.O_OBJ, b + L.O_PERSON])
        self.order = sorted(range(len(ks)), key=lambda x: (self.rc[x][0] + abs(self.rc[x][1]), abs(self.rc[x][1])))
        self.unit = {a: N_MOUTH + MOTOR.index(a) for a in MOTOR}
        self.step_act = {(1, 0): 'forward', (-1, 0): 'back', (0, 1): 'right', (0, -1): 'left'}

    def _free(self, e):
        return {rc: bool(e[w].any()) and not e[b].any() for rc, (w, b) in self.cells.items() if rc != (0, 0)}

    def _act(self, e, target):
        tr, tc = target
        free = self._free(e)
        goals = {(tr + dr, tc + dc) for dr, dc in self.step_act}
        if (0, 0) in goals:                                   # next to it: turn to face it
            return {(0, 1): 'turn_right', (0, -1): 'turn_left'}.get((tr, tc))
        goals = {x for x in goals if free.get(x)}
        prev = {(0, 0): None}
        q = collections.deque([(0, 0)])
        while q:
            cur = q.popleft()
            if cur in goals:
                while prev[cur] != (0, 0):
                    cur = prev[cur]
                return self.step_act[cur]
            for d in self.step_act:
                nx = (cur[0] + d[0], cur[1] + d[1])
                if nx not in prev and free.get(nx):
                    prev[nx] = cur
                    q.append(nx)
        return 'turn_right' if tc > 0 else 'turn_left' if tc < 0 else None

    def step(self, obs):
        A, E = self.A, obs['eye']
        A[:, N_MOUTH:] = 0
        P = E[:, self.ptr_cols]
        for j in np.flatnonzero(P.any(1) & (E[:, self.f_ptr] == 0)).tolist():
            x = next(x for x in self.order if P[j, x])
            act = self._act(E[j], self.rc[x])
            if act is not None:
                A[j, self.unit[act]] = 1.0
        return A


# ------------------------------------------------------------------------------------------ privileged references
class Scripted:
    """says planned texts through the mouth (exact muscle sets, force 1-2-3 per char, first syllable LATENCY ticks
    after the prompt) and, for the oracle, drives the motor units."""
    privileged = True
    answer_kinds = ('probe', 'recast', 'social')

    def __init__(self, n_agents):
        self.N = n_agents
        self.A = np.zeros((n_agents, N_ACT), np.float32)
        self.plans = [collections.deque() for _ in range(n_agents)]
        self.active = set()
        self.w = None

    def bench_attach(self, world):
        self.w = world
        self.need = {c: sorted(s) for c, s in world.voice.need.items()}
        self.inv_perm = [np.argsort(p) for p in world.motor_perm]

    def say(self, j, text, delay=LATENCY):
        q = self.plans[j]
        q.clear()
        for _ in range(delay):
            q.append(None)
        for ch in text:
            if ch in self.need:
                for lv in (1.0, 2.0, 3.0):
                    q.append((ch, lv))
        self.active.add(j)

    def answer(self, j, info):
        return ''

    def bench_prompt(self, j, info):
        if info['kind'] in ('q', 'social') and info.get('exkind', 'social') in self.answer_kinds:
            txt = self.answer(j, info)
            if txt:
                self.say(j, txt)
        elif info['kind'] == 'req':
            self.request(j, info)

    def request(self, j, info):
        pass

    def bench_taught(self, j, info):
        pass

    def step(self, obs):
        A = self.A
        for j in list(self.active):
            q = self.plans[j]
            A[j, :N_MOUTH] = 0
            if not q:
                self.active.discard(j)
                continue
            it = q.popleft()
            if it is not None:
                A[j, self.need[it[0]]] = it[1]
        self.motor(A)
        return A

    def motor(self, A):
        pass


class Echo(Scripted):
    def bench_prompt(self, j, info):
        if info['kind'] == 'q' and info.get('exkind') not in ('probe', 'recast'):
            return
        self.say(j, info['prompt'])


class Lookup(Scripted):
    """(prompt, exact eye input) -> the reply heard taught for exactly that input (per agent)."""

    def __init__(self, n_agents):
        super().__init__(n_agents)
        self.tab = [dict() for _ in range(n_agents)]

    def bench_taught(self, j, info):
        self.tab[j][(info['prompt'], info['eye_hash'])] = info['reply']

    def answer(self, j, info):
        return self.tab[j].get((info['prompt'], info.get('eye_hash')), '')


class Namer(Scripted):
    """every `every` ticks with a silent ear, says the name of a random object it sees (privileged)."""

    def __init__(self, n_agents, every=300, seed=0):
        super().__init__(n_agents)
        self.every = every
        self.quiet = np.zeros(n_agents, np.int64)
        self.rng = np.random.default_rng(seed)

    def bench_prompt(self, j, info):
        pass

    def step(self, obs):
        loud = obs['ear'].reshape(self.N, -1).max(1) > 0
        self.quiet = np.where(loud, 0, self.quiet + 1)
        for j in np.flatnonzero(self.quiet >= self.every):
            self.quiet[j] = 0
            objs = self.w.visible_objs(j)
            if objs and j not in self.active:
                o = objs[int(self.rng.integers(len(objs)))][0]
                self.say(j, KINDS[self.w.o_kind[o]], delay=0)
        return super().step(obs)


class Oracle(Scripted):
    """answers every scored question with acceptable[0]; carries out requests (fetch / give / come) by BFS path
    planning over the current grid, strafing without turning and turning only to face the target."""

    def __init__(self, n_agents):
        super().__init__(n_agents)
        self.goal = [None] * n_agents

    def answer(self, j, info):
        return info['acceptable'][0] if info.get('acceptable') else ''

    def request(self, j, info):
        self.goal[j] = (info['req'], info['target'], info['cg'])

    def _unit(self, j, action):
        return N_MOUTH + int(self.inv_perm[j][MOTOR.index(action)])

    def motor(self, A):
        w = self.w
        A[:, N_MOUTH:] = 0
        t = w.t
        for j in range(self.N):
            g = self.goal[j]
            if g is None or w._busy[j] > t or w.night:
                continue
            act = self._plan(j, g)
            if act == 'done':
                self.goal[j] = None
            elif act is not None:
                A[j, self._unit(j, act)] = 1.0

    def _plan(self, j, g):
        w = self.w
        typ, X, cg = g
        c, h = int(w.pos[j]), int(w.head[j])
        if typ == 'fetch':
            if w.hold[j]:
                return 'done'
            objs = [o for o in w.kind_objs[KINDS.index(X)] if w.o_cell[o] >= 0 and KINDS[w.o_kind[o]] not in ANIMALS]
            if not objs:
                return 'done'
            targets = {}
            for o in objs:
                oc = w.o_cell[o]
                for d in DIRS:
                    targets.setdefault(oc - d, oc)       # stand at oc - d, facing +d
            return self._approach(j, c, h, targets, 'take')
        cc = int(w.pos[cg])
        if typ == 'give':
            if not w.hold[j]:
                return 'done'
            targets = {cc - d: cc for d in DIRS}
            return self._approach(j, c, h, targets, 'put')
        # come
        near = {cc + d1 + d2 for d1 in (-DIRS[2], 0, DIRS[2]) for d2 in (-1, 0, 1) if d1 or d2}
        if c in near:
            return 'done'
        path = w._bfs(c, {x for x in near if w.free(x) or x == c})
        if not path:
            return None
        return self._step_to(c, h, path[0])

    def _approach(self, j, c, h, targets, final):
        w = self.w
        if c in targets:
            d = targets[c] - c
            want = DIRS.index(d)
            if want == h:
                return final
            return 'turn_right' if want == (h + 1) % 4 else 'turn_left'
        path = w._bfs(c, {x for x in targets if w.free(x)})
        if not path:
            return None
        return self._step_to(c, h, path[0])

    @staticmethod
    def _step_to(c, h, n):
        d = DIRS.index(n - c)
        return {0: 'forward', 2: 'back', 3: 'left', 1: 'right'}[(d - h) % 4]


REFERENCES = {'silent': Silent, 'random': RandomActor, 'echo': Echo, 'lookup': Lookup, 'oracle': Oracle,
              'namer': Namer, 'looker': Looker}
