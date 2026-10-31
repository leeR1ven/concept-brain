# -*- coding: utf-8 -*-
"""world2 self-test: physics reuse, sound field, bodies, eye, determinism, wrappers, caregivers, anti-cheat, speed.

    python3 selftest.py            (~3 min on one core)  -> checks/selftest.out
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

import numpy as np  # noqa: E402

from world2 import World2, PerAgent, Silent, RandomActor, Echo, Lookup, Oracle, Namer, evaluate, lex  # noqa: E402
from world2.core import (DIRS, N_ACT, N_MOUTH, MOTOR, cell_of, xy_of, phys, INAUD, SOUND, VOID)  # noqa: E402
from world2.lex import EYE_B, E_KIND, E_COL, E_CNT, E_SIZE, E_PTR, KIND_ID, KINDS  # noqa: E402

RESULTS, OUT = [], []


def out(s):
    print(s, flush=True)
    OUT.append(s)


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    out(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))


# ------------------------------------------------------------------------------------------ helpers
def _forever(cg):
    while True:
        yield ('wait', 10 ** 7)


def quiet_world(n=1, seed=0, **kw):
    """a world whose caregivers stand still forever (scripts replaced), animals removed: a lab for mechanics."""
    w = World2(n_agents=n, seed=seed, **kw)
    w.attach(_Zero(n))
    w._born = True                                   # no morning / clock events: we drive everything by hand
    w._schedule(10 ** 12, lambda a: None)
    for o in range(1, len(w.o_kind)):
        w._remove_obj(o)
    for cg in w.cgs:
        w._abandon(cg)
        cg.stack = [_forever(cg)]
        w._cg_resume(cg, None)
    return w


class _Zero:
    def __init__(self, n):
        self.A = np.zeros((n, N_ACT), np.float32)

    def step(self, obs):
        return self.A


class _Plan:
    """batched agent playing a fixed plan: {tick offset: (agent, action vector)}; records obs copies."""

    def __init__(self, n, plan=None):
        self.A = np.zeros((n, N_ACT), np.float32)
        self.plan = plan or {}
        self.k = 0
        self.rec = []

    def step(self, obs):
        self.A[:] = 0
        for j, a in self.plan.get(self.k, ()):
            self.A[j] = a
        self.rec.append({k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()})
        self.k += 1
        return self.A


def move(w, pid, x, y, head=None):
    c = cell_of(x, y)
    old = int(w.pos[pid])
    if w.PER[old] == pid + 1:
        w.PER[old] = 0
        w._cell_changed(old)
    w.PER[c] = pid + 1
    w.pos[pid] = c
    if head is not None:
        w.head[pid] = head
    w._cell_changed(c)
    if pid < w.N:
        w._moved.add(pid)
        w._aver += 1
        w._hver += 1


def say_as(w, k, text):
    """caregiver k says text now (then stands still)."""
    cg = w.cgs[k]

    def g(cg):
        yield ('say', text, {})
        while True:
            yield ('wait', 10 ** 7)
    w._abandon(cg)
    cg.stack = [g(cg)]
    w._cg_resume(cg, None)


def mus(ch, w, lv=1.0):
    a = np.zeros(N_ACT, np.float32)
    a[sorted(w.voice.need[ch])] = lv
    return a


def mot(name, v=1.0):
    a = np.zeros(N_ACT, np.float32)
    a[N_MOUTH + MOTOR.index(name)] = v
    return a


# ------------------------------------------------------------------------------------------ physics and sound
def t_physics():
    w = World2(n_agents=2, seed=0)
    ref = phys.Voice(phys.corpus_chars(), seed=0)
    same = all((w.voice.sound[c] == ref.sound[c]).all() and w.voice.need[c] == ref.need[c] for c in ref.chars) and \
        (w.voice.voice['self'] == ref.voice['self']).all() and (w.voice.voice['other'] == 0).all()
    rng = np.random.default_rng(5)
    bad = 0
    for _ in range(2000):
        k = int(rng.integers(1, 7))
        s = tuple(sorted(int(x) for x in rng.choice(32, k, replace=False)))
        mask = sum(1 << i for i in s)
        bad += w._decode(mask) != ref.decode(s)
    check('physics: the bench Voice (3,970 corpus chars: sounds, muscle sets, self offset) and its mouth rule',
          same and bad == 0, f'{len(ref.chars)} chars, 2,000 random muscle sets, {bad} decode mismatches')
    other2 = phys.Voice(phys.corpus_chars(), seed=0)
    ok = (w.voice.voice['other2'] == other2.voice['other2']).all() and not (w.voice.voice['w2_cg2'] == 0).all() \
        and len({w.voice.voice[f'w2_a{j}'].tobytes() for j in range(2)} | {w.voice.voice['w2_cg2'].tobytes()}) == 3
    check('physics: voices 妈妈 = bench other (original sound), 爸爸 = bench other2, 奶奶 and every agent own offsets', ok)


def t_ear():
    # one caregiver next to the agent, nothing else sounding: the ear == the bench's Stream with 'other' only
    w = quiet_world(1)
    move(w, 0, 10, 9, head=0)
    move(w, w.N + 0, 11, 9)
    text = '你好苹果哈哈'
    ag = _Plan(1)
    w.attach(ag)
    say_as(w, 0, text)
    w.run(ag, ticks=SOUND * len(text) + 6)
    v = w.voice
    st = phys.Stream(v.F, v.T, v.T // SOUND, 3, v.levels, 0.0)
    none = (None, None, False)
    bad = 0
    for k, o in enumerate(ag.rec):
        ci = k // SOUND
        src = (text[ci], v.hear(text[ci], 'other'), k % SOUND == 0) if ci < len(text) else none
        win = st.tick((src, none, none))
        bad += not (o['ear'][0] == win).all()
    check("ear: an agent next to 妈妈 hears exactly the bench Stream window (other voice, 2 slices/tick, 3 ticks/char)",
          bad == 0, f'{len(ag.rec)} ticks compared, {bad} differ')
    # attenuation: louder nearer, nothing beyond range, walls attenuate
    en = {}
    for d in (1, 3, 5, 7, 9):
        w = quiet_world(1)
        move(w, 0, 1 + d, 18, head=1)
        move(w, w.N, 1, 18)
        ag = _Plan(1)
        w.attach(ag)
        say_as(w, 0, '苹果苹果')
        w.run(ag, ticks=12)
        en[d] = int(sum(o['ear'].sum() for o in ag.rec))
    w = quiet_world(1)                               # through the kitchen / living-room wall, distance 2
    move(w, 0, 8, 7, head=0)
    move(w, w.N, 8, 5)
    ag = _Plan(1)
    w.attach(ag)
    say_as(w, 0, '苹果苹果')
    w.run(ag, ticks=12)
    wall = int(sum(o['ear'].sum() for o in ag.rec))
    ok = en[1] > en[3] > en[5] > en[7] > 0 and en[9] == 0 and 0 < wall < en[1]
    w = quiet_world(2, noise=1.0)                     # background noise as the bench (silent-window energy ~19)
    move(w, 0, 3, 18, head=0)
    move(w, 1, 20, 20, head=0)
    ag = _Plan(2)
    w.attach(ag)
    w.run(ag, ticks=1500)
    noise_e = float(np.mean([o['ear'][0].sum() for o in ag.rec[10:]]))
    ok = ok and 12 < noise_e < 26
    check('ear: louder nearer (levels -1 per 2 cells beyond 2), silent beyond 8 cells, -2 through a wall; noise=1 as '
          f'the bench (silent-window energy {noise_e:.1f}, bench ~19)',
          ok, f'energy by distance {en}, through a wall at distance 2: {wall}')


def t_voices():
    # agent 0 says a char: it hears its own voice one tick later (the bench self offset), agent 1 next to it hears
    # agent 0's voice, agent 2 far away hears nothing
    w = quiet_world(3)
    move(w, 0, 5, 9, head=1)
    move(w, 1, 6, 9, head=3)
    move(w, 2, 20, 20, head=0)
    ch = '球'
    plan = {k: [(0, mus(ch, w, float(k + 1)))] for k in range(3)}
    ag = _Plan(3, plan)
    w.attach(ag)
    w.run(ag, ticks=6)
    v = w.voice
    own = v.hear(ch, 'self')
    ext = v.hear(ch, 'w2_a0')
    e0 = [o['ear'][0] for o in ag.rec]
    e1 = [o['ear'][1] for o in ag.rec]
    ok_self = e0[0].sum() == 0 and (e0[1][:, 4:] == own[:, 0:2]).all() and (e0[3][:, :] == own).all()
    ok_other = (e1[1][:, 4:] == ext[:, 0:2]).all() and (e1[3] == ext).all()
    ok_far = all(o['ear'][2].sum() == 0 for o in ag.rec)
    check('ear: own voice heard one tick later (self offset, as talk.py); a neighbour hears it in the agent\'s own voice; '
          'far agents hear nothing', ok_self and ok_other and ok_far, f'self {ok_self}, neighbour {ok_other}, far {ok_far}')
    # mouth rule + onset rule: 苹果 then 哈哈 (force drop re-articulates) -> an initiative record with the text
    w = quiet_world(1)
    move(w, 0, 5, 9, head=1)
    seq = []
    for ch in '苹果哈哈':
        seq += [mus(ch, w, 1.0), mus(ch, w, 2.0), mus(ch, w, 3.0)]
    plan = {k: [(0, a)] for k, a in enumerate(seq)}
    ag = _Plan(1, plan)
    w.attach(ag)
    w.run(ag, ticks=len(seq) + 10)
    ini = [r for r in w.log if r['type'] == 'init']
    check('mouth: the bench mouth and onset rules (a force drop re-articulates 哈哈); an utterance outside an answer window '
          'is logged as an initiative', len(ini) == 1 and ini[0]['text'] == '苹果哈哈' and ini[0]['category'] == 'alone',
          f"{[(r['text'], r['category']) for r in ini]}")


# ------------------------------------------------------------------------------------------ body and eye
def t_body():
    w = quiet_world(1)
    move(w, 0, 10, 9, head=1)                         # facing east
    seq = ['forward', None, None, 'turn_right', None, None, 'left', None, None, 'back', None, None]
    plan = {k: [(0, mot(a))] for k, a in enumerate(seq) if a}
    ag = _Plan(1, plan)
    w.attach(ag)
    w.run(ag, ticks=len(seq))
    ok_move = xy_of(int(w.pos[0])) == (12, 8) and int(w.head[0]) == 2
    # busy: two commands on consecutive ticks -> only the first executes
    w2 = quiet_world(1)
    move(w2, 0, 10, 9, head=1)
    ag = _Plan(1, {0: [(0, mot('forward'))], 1: [(0, mot('forward'))], 2: [(0, mot('forward'))]})
    w2.attach(ag)
    w2.run(ag, ticks=4)
    ok_busy = xy_of(int(w2.pos[0])) == (11, 9)
    # bump into a wall
    w3 = quiet_world(1)
    move(w3, 0, 1, 9, head=3)
    ag = _Plan(1, {0: [(0, mot('forward'))]})
    w3.attach(ag)
    w3.run(ag, ticks=3)
    ok_bump = xy_of(int(w3.pos[0])) == (1, 9) and ag.rec[1]['touch'][0, lex.TO_BUMP] == 1 and ag.rec[2]['touch'][0, lex.TO_BUMP] == 0
    check('body: forward / back / left / right move relative to the heading, turns rotate; one action per 3 ticks; '
          'a blocked move is felt as a bump', ok_move and ok_busy and ok_bump,
          f'moves {ok_move} (at {xy_of(int(w.pos[0]))}, heading {int(w.head[0])}), busy {ok_busy}, bump {ok_bump}')
    # take / put / give / eat
    w = quiet_world(1)
    move(w, 0, 10, 9, head=1)
    o = w._spawn(KIND_ID['苹果'], cell=cell_of(11, 9), colour=0, count=2, size=1)
    move(w, w.N, 10, 10)                              # 妈妈 south of the agent
    w.HUNGER[0] = 0.8
    seq = ['take', None, None, 'eat', None, None, 'turn_right', None, None, 'put', None, None, None]
    ag = _Plan(1, {k: [(0, mot(a))] for k, a in enumerate(seq) if a})
    w.attach(ag)
    w.run(ag, ticks=len(seq))
    r = ag.rec
    touch_ok = r[1]['touch'][0, list(w.codes.touch[KIND_ID['苹果']])].all() and r[1]['touch'][0, lex.TO_HAND] == 1
    eat_ok = abs(r[4]['reward'][0] - 0.8) < 1e-6 and abs(float(r[4]['hunger'][0]) - 0.3) < 1e-6 and \
        r[4]['taste'][0].sum() == 3 and r[9]['taste'][0].sum() == 0 and w.o_cnt[o] == 1
    give_ok = w.hold[w.N] == o and w.hold[0] == 0 and r[10]['touch'][0, :12].sum() == 0
    check('body: take the object in front (touch code), eat it (reward = hunger before, hunger -0.5, taste for 6 ticks, '
          'one of two eaten), give it to the person in front', touch_ok and eat_ok and give_ok,
          f'touch {touch_ok}, eat {eat_ok}, give {give_ok}')


def t_eye():
    w = quiet_world(1)
    move(w, 0, 10, 9, head=1)                         # living room, facing east
    o = w._spawn(KIND_ID['皮球'], cell=cell_of(11, 9), colour=2, count=3, size=0)
    w.attach(_Zero(1))
    w.run(w.agents, ticks=1)
    E = w.EYE[0].reshape(25, EYE_B)
    blk = E[lex.VIEW_FRONT]
    code_ok = blk[E_KIND + w.codes.kind[KIND_ID['皮球']]].all() and blk[E_KIND:E_KIND + 24].sum() == 4 and \
        blk[E_COL + 2] == 1 and blk[E_CNT + 2] == 1 and blk[E_SIZE + 0] == 1 and blk[lex.E_TER + lex.T_L] == 1
    self_ok = E[lex.VIEW_SELF, lex.E_CHILD] == 1
    w._point(w.cgs[0], cell_of(11, 9))
    w.run(w.agents, ticks=1)
    ptr_ok = w.EYE[0].reshape(25, EYE_B)[lex.VIEW_FRONT, E_PTR:E_PTR + 2].all()
    w.head[0] = 3
    w._moved.add(0); w._hver += 1
    w.run(w.agents, ticks=1)
    E = w.EYE[0].reshape(25, EYE_B)
    gone_ok = not E[:, E_KIND:E_KIND + 24].any()
    # occlusion: agent in the living room just below the kitchen wall facing north: kitchen cells hidden
    w = quiet_world(1)
    move(w, 0, 8, 7, head=0)
    w._spawn(KIND_ID['香蕉'], cell=cell_of(8, 5))
    w.run(w.agents, ticks=1)
    E = w.EYE[0].reshape(25, EYE_B)
    occl_ok = E[7, lex.E_TER + lex.T_WALL] == 1 and E[12].sum() == 0 and not E[:, E_KIND:E_KIND + 24].any()
    w = quiet_world(1)
    move(w, 0, 3, 3, head=1)                          # kitchen, facing east: floor, the table, nothing else
    w.run(w.agents, ticks=1)
    E = w.EYE[0].reshape(25, EYE_B)
    vc = w.view_cells(0).tolist()
    terr_ok = all((E[k] == w.codes.ter[w._ter[c]]).all() for k, c in enumerate(vc) if k != lex.VIEW_SELF) and \
        E[12, lex.E_TER + lex.T_TABLE] == 1
    code_ok = code_ok and terr_ok
    check('eye: egocentric 5 x 5 view (own row + 4 ahead), one 64-unit block per cell with the same code everywhere '
          '(kind 4/24, colour, count, size, terrain, person, pointing); turning changes the view; walls occlude',
          code_ok and self_ok and ptr_ok and gone_ok and occl_ok,
          f'code {code_ok}, self {self_ok}, pointing {ptr_ok}, turned away {gone_ok}, occlusion {occl_ok}')


def t_night():
    w = World2(n_agents=2, seed=1)

    class Spy:
        def __init__(self):
            self.A = np.zeros((2, N_ACT), np.float32)
            self.n = 0
            self.bad = 0

        def step(self, obs):
            if w.night:
                self.n += 1
                self.bad += bool(obs['eye'].any())
            return self.A
    s = Spy()
    w.run(s, ticks=w.p['day_ticks'] + 3000)
    said = [r for r in w.log if r['type'] == 'say' and w.p['day_ticks'] <= r['t'] < w.p['day_ticks'] + 3000]
    check('day / night: at night the eye is dark and the caregivers sleep (silent)', s.n > 2000 and s.bad == 0 and not said,
          f'{s.n} night ticks, {s.bad} with light, {len(said)} caregiver lines')


# ------------------------------------------------------------------------------------------ determinism, wrappers
def _sig(log):
    return hashlib.md5(json.dumps([[r.get(k) for k in ('type', 't', 'agent', 'task', 'prompt', 'reply', 'text', 'success')]
                                   for r in log], ensure_ascii=False, default=str).encode()).hexdigest()


class _Bab1:
    """single agent: says one fixed char every `every` ticks and steps forward every 7 ticks."""

    def __init__(self, ms, every):
        self.ms, self.every, self.k = ms, every, 0
        self.a = np.zeros(N_ACT, np.float32)

    def step(self, o):
        self.a[:] = 0
        k = self.k % self.every
        if k < 3:
            self.a[self.ms] = k + 1.0
        if self.k % 7 == 0:
            self.a[N_MOUTH + MOTOR.index('forward' if (self.k // 7) % 3 else 'turn_left')] = 1.0
        self.k += 1
        return self.a


class _BabN:
    def __init__(self, singles):
        self.s = singles
        self.A = np.zeros((len(singles), N_ACT), np.float32)

    def step(self, obs):
        for j, a in enumerate(self.s):
            self.A[j] = a.step(None)
        return self.A


def t_determinism():
    a = World2(n_agents=3, seed=3)
    la = a.run(Oracle(3), ticks=60000)
    b = World2(n_agents=3, seed=3)
    lb = b.run(Oracle(3), ticks=60000)
    c = World2(n_agents=3, seed=3)
    lc = c.run(RandomActor(3, seed=1), ticks=30000)
    d = World2(n_agents=3, seed=3)
    ld = d.run(RandomActor(3, seed=1), ticks=30000)
    check('determinism: a fixed seed and the same agents give identical lives (oracle, random actor)',
          _sig(la) == _sig(lb) and _sig(lc) == _sig(ld), f'{len(la)} / {len(lc)} records')
    v = World2(n_agents=3, seed=4).voice
    sets = [sorted(v.need[ch]) for ch in '苹果球']
    w1 = World2(n_agents=3, seed=4)
    l1 = w1.run(_BabN([_Bab1(s, 37 + 11 * j) for j, s in enumerate(sets)]), ticks=30000)
    w2 = World2(n_agents=3, seed=4)
    l2 = w2.run(PerAgent([_Bab1(s, 37 + 11 * j) for j, s in enumerate(sets)]), ticks=30000)
    check('API: a batched agent and PerAgent([...]) of the same single agents give identical lives',
          _sig(l1) == _sig(l2) and len(l1) > 100, f'{len(l1)} records')


# ------------------------------------------------------------------------------------------ caregivers and anti-cheat
def t_life():
    ticks = 150000
    res, logs, worlds = {}, {}, {}
    for nm, mk in (('silent', Silent), ('random', RandomActor), ('echo', Echo), ('lookup', Lookup), ('oracle', Oracle)):
        w = World2(n_agents=4, seed=1)
        logs[nm] = w.run(mk(4), ticks=ticks)
        res[nm] = evaluate(logs[nm])
        worlds[nm] = w
    G = lambda nm, g: res[nm]['groups'][g]['acc'] or 0.0          # noqa: E731
    Cm = lambda nm: res[nm]['comprehension']['ALL']['success'] or 0.0   # noqa: E731
    ok = all(G(n, g) == 0 for n in ('echo', 'lookup') for g in ('scene', 'held_out', 'memory')) and \
        all(Cm(n) == 0 for n in ('echo', 'lookup', 'silent'))
    check('anti-cheat: echo and lookup score 0 on scene questions (new situations), held-out pairs, memory probes and '
          'requests', ok, ', '.join(f"{n}: scene {G(n, 'scene'):.2f} held {G(n, 'held_out'):.2f} memory "
                                    f"{G(n, 'memory'):.2f} requests {Cm(n):.2f}" for n in ('echo', 'lookup')))
    o = res['oracle']
    ok = all(o['groups'][g]['acc'] == 1.0 for g in ('scene', 'held_out', 'memory', 'social')) and Cm('oracle') >= 0.95
    check('anti-cheat: the scripted oracle (world state + path planning) scores 1.0 on every question group and >= 0.95 '
          'on requests', ok, ', '.join(f"{g} {o['groups'][g]['acc']:.2f} (n={o['groups'][g]['n']})"
                                      for g in ('scene', 'held_out', 'memory', 'social')) +
          f", requests {Cm('oracle'):.2f} (n={o['comprehension']['ALL']['n']})")
    r = res['random']
    check('chance: the random actor answers nothing and carries out few requests', (r['tasks']['ALL']['acc'] or 0) < 0.02
          and Cm('random') < 0.15, f"acc {r['tasks']['ALL']['acc']:.2f}, requests {Cm('random'):.2f}")
    # held-out pairs are never answered (no taught exposure ever), all held-out questions are probes
    w, lg = worlds['oracle'], logs['oracle']
    leaks = sum(w.exposures[j][(t, X)] for j in range(w.N) for t, xs in w.held.items() for X in xs)
    hq = [x for x in lg if x['type'] == 'q' and x.get('held_out')]
    check('held-out (task, kind) pairs are never answered by anyone; they are always probes',
          leaks == 0 and hq and all(x['kind'] == 'probe' for x in hq), f'{len(hq)} held-out questions, {leaks} answers heard')
    # here-and-now: object questions point at an object in the agent's view
    oq = [x for x in lg if x['type'] == 'q' and x['task'] in ('what', 'colour', 'count', 'size', 'check')]
    check('caregivers talk about the here-and-now: every object question points at an object in the agent\'s view',
          oq and all(x.get('view_k') is not None and 0 <= x['view_k'] < 25 and x['view_k'] != 2 for x in oq),
          f'{len(oq)} object questions')
    # memory probes: facts told / seen >= 300 ticks earlier, never answered afterwards
    mq = [x for x in lg if x['type'] == 'q' and x['task'] in ('where', 'novel', 'person', 'myname')]
    d = [x['ticks_since_told'] for x in mq]
    tasks = collections.Counter(x['task'] for x in mq)
    check('memory probes: about facts seen or told >= 300 ticks earlier (hundreds to tens of thousands), always probes',
          len(mq) > 30 and min(d) >= 300 and max(d) > 10000 and all(x['kind'] == 'probe' for x in mq) and len(tasks) >= 3,
          f'{len(mq)} probes {dict(tasks)}, delay min {min(d) if d else None} median {int(np.median(d)) if d else None} '
          f'max {max(d) if d else None}')
    # requests carried out by the oracle with all three types
    rq = collections.Counter(x['req'] for x in lg if x['type'] == 'req')
    check('requests (去拿X, 把X给我, 名字过来) are scored by the outcome within 400 ticks', len(rq) == 3 and min(rq.values()) > 5,
          f'{dict(rq)}')
    # a caregiver says one thing at a time (no overlapping lines of the same person); the oracle does not talk over
    # the caregiver it is engaged with
    bad = 0
    for nm in lex.CAREGIVERS:
        ss = sorted((x['t'], len(x['text'])) for x in lg if x['type'] == 'say' and x['by'] == nm)
        bad += sum(1 for (t1, n1), (t2, _) in zip(ss, ss[1:]) if t1 + SOUND * n1 > t2)
    po = res['oracle']['overlap']
    check('caregivers say one line at a time; the oracle never talks over the caregiver engaged with it',
          bad == 0 and po['partner_ticks'] <= 3, f"{bad} overlapping lines; oracle over its partner: "
          f"{po['partner_ticks']} ticks / {po['partner_interruptions']} interruptions in {res['oracle']['days']} days")
    # many agents: caregivers attend to everyone
    att = collections.Counter(x['agent'] for x in lg if x['type'] == 'q')
    check('many agents: attention rotates fairly (every agent gets >= half the questions of the most-asked one)',
          len(att) == 4 and min(att.values()) >= 0.5 * max(att.values()), f'{dict(sorted(att.items()))}')


def t_initiative():
    w = World2(n_agents=4, seed=2)
    lg = w.run(Namer(4, every=200), ticks=120000)
    ini = [x for x in lg if x['type'] == 'init']
    nam = [x for x in ini if x['category'] == 'naming']
    ok = nam and all(x['reward'] > 0 and x['by'] for x in nam) and len({x['agent'] for x in nam}) >= 3 \
        and len({x['by'] for x in nam}) >= 2 and all(x['reward'] == 0 for x in ini if x['category'] in ('alone', 'repeat'))
    cats = collections.Counter(x['category'] for x in ini)
    check('initiative: naming a visible object while a caregiver hears earns a reply (对是X) and reward from the nearest '
          'caregiver, whoever speaks; alone / repeated: nothing', ok,
          f"{dict(cats)}; rewarded agents {sorted({x['agent'] for x in nam})} by {sorted({x['by'] for x in nam})}")
    # agents hear each other: agent 1 answers agent 0 after 15 ticks (in earshot) -> one agent-agent turn; a third
    # agent far away answering at the same time is not a turn
    w = quiet_world(3)
    move(w, 0, 5, 10, head=1)
    move(w, 1, 7, 10, head=3)
    move(w, 2, 20, 20, head=0)
    plan = collections.defaultdict(list)
    for k, lv in enumerate((1.0, 2.0, 3.0)):
        plan[k].append((0, mus('球', w, lv)))
        plan[15 + k].append((1, mus('好', w, lv)))
        plan[16 + k].append((2, mus('好', w, lv)))
    ag = _Plan(3, dict(plan))
    w.attach(ag)
    lg = w.run(ag, ticks=40)
    a2a = [x for x in lg if x['type'] == 'a2a']
    check('agent-agent talk: an utterance answered by another agent in earshot within 30 ticks is an agent-agent turn '
          '(a far agent speaking at the same time is not)', len(a2a) == 1 and a2a[0]['src'] == 0 and a2a[0]['dst'] == 1,
          f'{a2a}')


def t_speed():
    best = 0.0
    for _ in range(2):
        w = World2(n_agents=16, seed=0)
        t0 = time.time()
        w.run(Silent(16), ticks=150000)
        best = max(best, 150000 / (time.time() - t0))
    check('speed: 16 trivial agents >= 1e5 world ticks/s on one core (best of 2 runs of 150,000 ticks = 5 days)',
          best >= 1e5, f'{best:,.0f} ticks/s')


def t_bench_unchanged():
    md5 = {'life.py': '4d60a4adaef703df58b3d6f6063f15f8', 'life_refs.py': '5f89f739822a01ec0f0222362ed8cc13',
           'selftest.py': '1e72bc3770b58a2f2c09445e7040bea2', 'README.md': '4124d67d3bfd190fe4b8e114df92a530'}
    same = all(hashlib.md5(open(os.path.join(BENCH, f), 'rb').read()).hexdigest() == h for f, h in md5.items())
    core = all(open(os.path.join(BENCH, f + '.py'), 'rb').read() == open(os.path.join(BENCH, 'v1.2', f + '.py'), 'rb').read()
               for f in ('physics', 'world', 'metrics', 'baselines', 'frontend', 'body', 'ref_v1'))
    check('the bench is untouched: v1.2 core modules == v1.2/, life mode files unchanged (world2 only imports them)',
          same and core)


if __name__ == '__main__':
    t0 = time.time()
    for f in (t_bench_unchanged, t_physics, t_ear, t_voices, t_body, t_eye, t_night, t_determinism, t_life,
              t_initiative, t_speed):
        f()
    n_ok = sum(ok for _, ok in RESULTS)
    out(f'\n{n_ok}/{len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
    os.makedirs(os.path.join(HERE, 'checks'), exist_ok=True)
    with open(os.path.join(HERE, 'checks', 'selftest.out'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(OUT) + '\n')
    sys.exit(0 if n_ok == len(RESULTS) else 1)
