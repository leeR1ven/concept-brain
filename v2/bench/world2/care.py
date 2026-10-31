# -*- coding: utf-8 -*-
"""The caregivers' scripts (妈妈 爸爸 奶奶): generators that yield ops to the driver in core.py.

Every line is generated from the current world state: what the addressed agent can see right now, what it holds, where
it is, what it was told or saw earlier, whether it is hungry.  Ops: ('say', text, opts), ('goto', spec, max_ticks),
('wait', n), ('listen', agents, opts), ('watch', condition, max_ticks), ('point', cell | None), ('call', fn, *args).
A preempted 'wait' / 'goto' returns None to the script (it re-issues what it needs).
"""
from __future__ import annotations

import numpy as np

from . import lex
from .lex import (ANIMALS, CAREGIVERS, COLOURS, COUNTS, EDIBLE, FOODS, GOODNIGHT, GREET, GREET_MORNING, GREET_WORDS,
                  HUNGRY_WORDS, KIND_ID, KINDS, NOVEL, OBJECTS, ROOMS)

# view indices ordered by distance from the own cell (the own cell itself excluded)
VIEW_ORDER = sorted([k for k in range(25) if k != 2],          # nearest first; straight ahead before the sides
                    key=lambda k: (k // 5 + abs(k % 5 - 2), abs(k % 5 - 2), k // 5, k))
PRAISE = ['真棒', '对', '很好']
QTASK_W = dict(what=0.25, colour=0.15, count=0.12, size=0.1, check=0.13, holding=0.08, room=0.08, who=0.09)
MEMORY_TASKS = ('where', 'novel', 'person', 'myname')
SCENE_TASKS = ('what', 'colour', 'count', 'size', 'check', 'holding', 'room', 'who', 'offer')
APPROPRIATE = ('greeting', 'naming', 'calling', 'asking', 'request', 'hungry')
_PW = lex.W + 8                  # padded grid width (core.PW)
_DIRS = (-_PW, 1, _PW, -1)       # = core.DIRS


def _say(text, **kw):
    return ('say', text, kw)


class CareMixin:
    # ================================================================ small helpers
    def cheb(self, a, b):
        ax, ay = self.S.xy[self.pos[a]]
        bx, by = self.S.xy[self.pos[b]]
        return max(abs(ax - bx), abs(ay - by))

    def adj4(self, a, b):
        return abs(int(self.pos[a]) - int(self.pos[b])) in (1, _PW)

    def hears(self, listener, speaker):
        return self.ATT[self.pos[speaker], self.pos[listener]] < 6

    def room_at(self, pid):
        return int(self.S.ROOM[self.pos[pid]])

    def visible_objs(self, i):
        """[(object, cell, view index)] lying in agent i's view, nearest first."""
        if self.night:
            return []
        ids = self.view_cells(i)
        out = []
        for k in VIEW_ORDER:
            c = int(ids[k])
            o = int(self.OBJ[c])
            if o:
                out.append((o, c, k))
        return out

    def visible_persons(self, i):
        if self.night:
            return []
        ids = self.view_cells(i)
        out = []
        for k in VIEW_ORDER:
            c = int(ids[k])
            pp = int(self.PER[c])
            if pp and pp - 1 != i:
                out.append((pp - 1, c, k))
        return out

    def visible_kinds(self, i):
        """kinds agent i can see: lying in view, held by a person in view, held by itself."""
        ks = {self.o_kind[o] for o, _, _ in self.visible_objs(i)}
        for pid, _, _ in self.visible_persons(i):
            if self.hold[pid]:
                ks.add(self.o_kind[self.hold[pid]])
        if self.hold[i]:
            ks.add(self.o_kind[self.hold[i]])
        return ks

    def _pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def _wchoice(self, d):
        ks = [k for k, v in d.items() if v > 0]
        if not ks:
            return None
        w = np.array([d[k] for k in ks], float)
        return ks[int(self.rng.choice(len(ks), p=w / w.sum()))]

    def _reward(self, j, v):
        self.REW[j] += v
        self.mon_f['reward'][j] += v
        self._pulse = True
        return True

    def _kind_for(self, held_out, memory=False):
        if held_out or memory:
            return 'probe'
        u = self.rng.random()
        if u < self.p['probe_frac']:
            return 'probe'
        if u < self.p['probe_frac'] + self.p['p_model']:
            return 'model'
        return 'recast'

    def _go_to_agent(self, cg, i, max_ticks=500):
        for _ in range(4):
            r = yield ('goto', ('person', i, 'front'), max_ticks)
            if r is not None:
                return r
        return False

    def _go(self, cg, spec, max_ticks=600):
        for _ in range(4):
            r = yield ('goto', spec, max_ticks)
            if r is not None:
                return r
        return False

    def _wait(self, n):
        yield ('wait', int(n))

    def _watch(self, cond, ticks):
        """wait until cond() or `ticks` passed; survives an interruption by a reaction (re-issued with the rest)."""
        t0 = self.t
        end = t0 + int(ticks)
        while True:
            r = yield ('watch', cond, max(1, end - self.t))
            if r is not None:
                return bool(r[0]), self.t - t0
            if self.t >= end or cond():
                return bool(cond()), self.t - t0

    # ================================================================ choosing what to do
    def _new_activity(self, cg):
        if self.night:
            return None
        cg.n_acts += 1
        if self.bedtime_on:
            return self._act_bedtime(cg) if not cg.bed_done else self._act_rest(cg, 200, 400)
        if self.hold[cg.pid]:
            return self._act_putaway(cg)
        if self.meal_due and (cg.idx == 2 or (cg.idx == 0 and self.t - self.meal_due[0] > 800)):
            return self._act_cook(cg)
        if cg.prefer is not None:
            i, cg.prefer = cg.prefer, None
            if self.engaged[i] is None:
                return self._act_teach(cg, i)
        p = self.p
        w = dict(teach=p['w_teach'], tidy=p['w_tidy'], chat=p['w_chat'], rest=p['w_rest'])
        if cg.idx == 2:
            w['tidy'] *= 1.5; w['teach'] *= 0.8
        if cg.idx == 1:
            w['chat'] *= 1.3
        a = self._wchoice(w)
        if a == 'teach':
            return self._act_teach(cg)
        if a == 'tidy':
            return self._act_tidy(cg)
        if a == 'chat':
            return self._act_chat(cg)
        return self._act_rest(cg)

    def _pick_agent(self, cg):
        """the available agent attended least recently (ties at random) that cg can reach."""
        cand = [j for j in range(self.N) if self.engaged[j] is None and self.listening[j] is None]
        if not cand:
            return None
        self.rng.shuffle(cand)
        cand.sort(key=lambda j: self.last_attended[j])
        for j in cand[:3]:
            d = self.S.dist_to(self._goal_cells(('person', j, 'any')))
            if d[int(self.pos[cg.pid])] < 255:
                return j
        return None

    # ================================================================ activities
    def _act_rest(self, cg, lo=None, hi=None):
        r = self.rng
        if lo is None:
            lo, hi = self.p['rest_ticks']
        room = 2 if (self.bedtime_on or r.random() < 0.3) else 1
        cells = [x for x in self.S.open_cells if self._room[x] == room and not self._cblk[x]]
        if cells:
            yield from self._go(cg, ('cell', self._pick(cells)), 400)
        yield ('wait', int(r.integers(lo, hi)))

    def _act_putaway(self, cg):
        o = self.hold[cg.pid]
        if not o:
            return
        room = self.home[KINDS[self.o_kind[o]]]
        if room is None:
            room = self.room_at(cg.pid) if self.room_at(cg.pid) >= 0 else 1
        c = self._free_cell_in_room(room, prefer_table=KINDS[self.o_kind[o]] in EDIBLE, rng=self.rng)
        if c is None:
            yield ('wait', 50)
            return
        ok = yield from self._go(cg, ('adj8', c), 500)
        if ok and self.OBJ[c] == 0 and self.PER[c] == 0 and self.hold[cg.pid] == o:
            yield ('call', self._cg_put, cg, o, c)
            if self.heard_by(int(self.pos[cg.pid])):
                X, R = KINDS[self.o_kind[o]], ROOMS[int(self.S.ROOM[c])]
                yield _say('我把' + X + '放在' + R + '了', kind='fact', tell=(('where', X), ('obj', o)))
        else:
            yield ('wait', 30)

    def _cg_put(self, cg, o, c):
        self._take_from(cg.pid)
        self._place(o, c)
        return True

    def _cg_take(self, cg, o):
        if self.o_alive[o] and self.o_cell[o] >= 0 and not self.hold[cg.pid]:
            self._lift(o)
            self._give_to(cg.pid, o)
            return True
        return False

    def _act_tidy(self, cg):
        """carry a misplaced object back to its usual room, then say where it was put (if anyone hears)."""
        cand = []
        for o in range(1, len(self.o_kind)):
            if not self.o_alive[o] or self.o_cell[o] < 0:
                continue
            k = KINDS[self.o_kind[o]]
            home = self.home[k]
            if home is None or int(self.S.ROOM[self.o_cell[o]]) == home:
                continue
            cand.append(o)
        if not cand:
            yield ('wait', 100)
            return
        cx, cy = self.S.xy[self.pos[cg.pid]]
        cand.sort(key=lambda o: abs(self.S.xy[self.o_cell[o]][0] - cx) + abs(self.S.xy[self.o_cell[o]][1] - cy))
        o = cand[0] if self.rng.random() < 0.7 else self._pick(cand)
        ok = yield from self._go(cg, ('adj8', self.o_cell[o]), 500)
        if not ok:
            yield ('wait', 50)
            return
        got = yield ('call', self._cg_take, cg, o)
        if got:
            self.log.append(dict(type='event', t=self.t, what='tidy', by=cg.name, kind=KINDS[self.o_kind[o]]))
            yield from self._act_putaway(cg)

    def _act_cook(self, cg):
        """go to the stove, cook, clear old food, put fresh food on the kitchen table, call 吃饭了."""
        due = self.meal_due.pop(0) if self.meal_due else self.t
        stove = [c for c in self.S.map_cells if self.S.TER[c] == lex.T_STOVE]
        ok = yield from self._go(cg, ('adj8', int(stove[0])), 600)
        yield ('wait', self.p['cook_ticks'])
        old = [o for o in range(1, len(self.o_kind)) if self.o_alive[o] and self.o_cell[o] >= 0
               and KINDS[self.o_kind[o]] in FOODS and int(self.S.ROOM[self.o_cell[o]]) == 0 and self.o_t0[o] < due]
        for o in old:
            self._remove_obj(o)
        n = int(self.rng.integers(self.p['food_per_meal'][0], self.p['food_per_meal'][1] + 1))
        made = []
        for _ in range(n):
            c = self._free_cell_in_room(0, prefer_table=True, rng=self.rng)
            if c is None:
                break
            made.append(KINDS[self._spawn_kind(self._pick(FOODS), c)])
        self.log.append(dict(type='event', t=self.t, what='meal', by=cg.name, food=made, cleared=len(old), due=due))
        yield _say('吃饭了', kind='stmt')

    def _spawn_kind(self, kind_name, c):
        o = self._spawn(KIND_ID[kind_name], cell=c)
        return self.o_kind[o]

    def _act_chat(self, cg):
        """talk with another caregiver: questions about the world answered truthfully; overheard by agents nearby."""
        others = [x for x in self.cgs if x is not cg and self.present[x.pid] and x.held_by is None
                  and x.partner is None and x.op is not None and x.op[0] == 'wait' and x.pid not in self._say]
        if not others:
            yield ('wait', 60)
            return
        ot = self._pick(others)
        if not self._preempt(ot, self._g_held(ot), walks=False, hold_by=cg):     # held until released below
            yield ('wait', 30)
            return
        try:
            ok = yield from self._go(cg, ('adj8', int(self.pos[ot.pid])), 400)
            if not ok:
                return
            for k in range(int(self.rng.integers(2, 5))):
                a, b = (cg, ot) if k % 2 == 0 else (ot, cg)
                q = self._chat_pair(a, b)
                if q is None:
                    continue
                prompt, reply, item, tell = q
                yield _say(prompt, speaker=a, kind='chat')
                yield ('wait', self.p['gap_model'])
                yield _say(reply, speaker=b, kind='chat', teach=(item, prompt, reply) if item else None, tell=tell)
                yield ('wait', int(self.rng.integers(10, 40)))
        finally:
            ot.held_by = None

    def _g_held(self, cg):
        while cg.held_by is not None:
            yield ('wait', 10)

    def _chat_pair(self, a, b):
        r = self.rng
        u = r.random()
        if self._corpus is not None and u < self.p['p_corpus']:
            dlg = self._corpus[self._corpus_pos % len(self._corpus)]
            self._corpus_pos += 1
            return dlg[0], dlg[1], ('corpus', dlg[0]), None
        u = r.random()
        if u < 0.4:                                            # X在哪里 -> 在R (a uniquely placed object)
            lying = [o for o in range(1, len(self.o_kind)) if self.o_alive[o] and self.o_cell[o] >= 0
                     and len(self.kind_objs[self.o_kind[o]]) == 1]
            if lying:
                o = self._pick(lying)
                X, R = KINDS[self.o_kind[o]], ROOMS[int(self.S.ROOM[self.o_cell[o]])]
                return X + '在哪里', '在' + R, ('where', X), (('where', X), ('obj', o))
        if u < 0.6 and self.N > 0:                             # 小明在哪里 -> 在R
            j = int(r.integers(self.N))
            R = ROOMS[self.room_at(j)] if self.room_at(j) >= 0 else '客厅'
            return self.names[j] + '在哪里', '在' + R, ('where_person', self.names[j]), None
        if u < 0.75:
            food = any(self.o_alive[o] and self.o_cell[o] >= 0 and KINDS[self.o_kind[o]] in FOODS
                       and int(self.S.ROOM[self.o_cell[o]]) == 0 for o in range(1, len(self.o_kind)))
            return '饭好了吗', '好了' if food else '还没有', ('meal',), None
        # a question about an object next to the speaker (never a held-out pair)
        near = []
        ac = int(self.pos[a.pid])
        for d1 in (-_PW, 0, _PW):
            for d2 in (-1, 0, 1):
                o = int(self.OBJ[ac + d1 + d2])
                if o:
                    near.append(o)
        if near:
            o = self._pick(near)
            X = KINDS[self.o_kind[o]]
            if r.random() < 0.5 and X not in self.held['what']:
                return '这是什么', '这是' + X, ('what', X), None
            if X not in self.held['colour']:
                return X + '是什么颜色', COLOURS[self.o_col[o]], ('colour', X), None
        return '饭好了吗', '还没有', ('meal',), None

    # ================================================================ bedtime
    def _act_bedtime(self, cg):
        cg.bed_done = True
        if self.hold[cg.pid]:
            yield from self._act_putaway(cg)
        yield _say('睡觉了', kind='stmt')
        near = [j for j in self.heard_by(int(self.pos[cg.pid])) if self.listening[j] is None
                and self.engaged[j] is None]
        if near:
            for j in near:
                self.engaged[j] = cg.pid
            try:
                yield from self._social(cg, near, 'goodnight')
            finally:
                for j in near:
                    if self.engaged[j] == cg.pid:
                        self.engaged[j] = None
        cells = [x for x in self.S.open_cells if self._room[x] == 2 and not self._cblk[x]]
        if cells:
            yield from self._go(cg, ('cell', self._pick(cells)), 400)
        yield ('wait', 2000)

    # ================================================================ teaching episodes
    def _act_teach(self, cg, i=None):
        if i is None:
            i = self._pick_agent(cg)
        if i is None or self.engaged[i] is not None:
            yield ('wait', 100)
            return
        self.engaged[i] = cg.pid
        cg.partner = i
        cg.ep_t0 = self.t
        try:
            yield from self._episode(cg, i)
        finally:
            if self.engaged[i] == cg.pid:
                self.engaged[i] = None
            cg.partner = None
            self.last_attended[i] = self.t
            self._point(cg, None)

    def _episode(self, cg, i):
        p, r = self.p, self.rng
        if self.cheb(cg.pid, i) >= 3 and self.hears(i, cg.pid) and r.random() < p['p_call']:
            yield from self._req_come(cg, i)
        if not self.adj4(cg.pid, i):
            ok = yield from self._go_to_agent(cg, i)
            if not ok:
                return
        if i not in cg.greeted:
            cg.greeted.add(i)
            morning = self.t - self.t_day0 < p['day_ticks'] // 3
            yield from self._social(cg, [i], 'morning' if morning else 'greet')
        for _ in range(int(r.integers(p['turns'][0], p['turns'][1] + 1))):
            if self.bedtime_on or self.night:
                return
            if not self.adj4(cg.pid, i):
                ok = yield from self._go_to_agent(cg, i)
                if not ok:
                    return
            yield from self._turn(cg, i)
            g0, g1 = p['turn_gap']
            yield ('wait', int(np.exp(r.uniform(np.log(g0), np.log(g1)))))

    def _turn(self, cg, i):
        w = dict(self.p['w_turn'])
        if self.HUNGER[i] > 0.4 and self.hold[i] == 0:
            w['offer'] = max(w.get('offer', 0.0), 0.3)
        for _ in range(4):
            kind = self._wchoice(w)
            if kind is None:
                return
            done = yield from getattr(self, '_t_' + kind)(cg, i)
            if done:
                return
            w[kind] = 0.0
        yield ('wait', 20)

    # ---------------- exchanges (a question to one agent, scored when its answer window closes)
    def _exchange(self, cg, i, ex):
        p = self.p
        item, prompt = ex['item'], ex['prompt']
        kind = ex.get('kind') or self._kind_for(ex['held_out'], ex['task'] in MEMORY_TASKS)
        eh = self.eye_hash(i)
        rec = dict(type='q', t=self.t, agent=i, by=cg.name, task=ex['task'], prompt=prompt,
                   acceptable=list(ex['acceptable']), kind=kind, held_out=bool(ex['held_out']),
                   needs_scene=ex['task'] in SCENE_TASKS, item=item,
                   new_situation=(prompt, eh) not in self.heard_sit[i],
                   n_experienced=int(self.exposures[i][item]), n_asked_before=int(self.asked[i][item]),
                   day=self.day_i, room=self.room_at(i), dist=float(self.cheb(cg.pid, i)), n_agents_near=len(
                       self.heard_by(int(self.pos[cg.pid]))))
        for k_, v_ in ex.get('tags', {}).items():
            assert k_ not in rec, k_
            rec[k_] = v_
        self.asked[i][item] += 1
        self.last_asked[i][item] = self.t
        if ex.get('point') is not None:
            yield ('point', ex['point'])
        try:
            yield _say(prompt, to=i, kind='q')
            t_end = self.t
            rec['t_prompt_end'] = t_end
            self._hook_prompt(i, dict(kind='q', exkind=kind, task=ex['task'], prompt=prompt, item=item,
                                      acceptable=list(ex['acceptable']), eye_hash=eh, cg=cg.pid))
            if kind == 'model':
                yield ('wait', p['gap_model'])
                yield _say(ex['model'], to=i, kind='model', teach=(item, prompt, ex['model']))
                res = yield ('listen', [i], dict(min_gap=p['model_wait']))
                rec['reply'] = ''
                rec['model_reply'] = ex['model']
                rec['after_model_reply'] = res[i]['reply']
            else:
                res = yield ('listen', [i], {})
                R = res[i]
                rec.update(reply=R['reply'], t_first=R['first'], stop=R['stop'], garble=R.get('garble', 0),
                           overlap=R.get('overlap', 0),
                           latency=(R['first'] - t_end) if R['first'] is not None else None)
                rec['acc'] = R['reply'] in rec['acceptable']
                rec['first_ok'] = bool(R['reply']) and R['reply'][0] in {a[0] for a in rec['acceptable'] if a}
                if kind == 'recast':
                    if rec['acc']:
                        yield _say(self._pick(PRAISE), to=i, reward=p['reward_ok'], kind='praise')
                    else:
                        yield _say(ex['model'], to=i, kind='recast', teach=(item, prompt, ex['model']))
            self.log.append(rec)
        finally:
            if ex.get('point') is not None:
                self._point(cg, None)
        return rec

    def _t_question(self, cg, i):
        """a question about the here-and-now (what the agent can see or holds), with pointing."""
        H = self.held
        objs = self.visible_objs(i)
        everyday = [x for x in objs if KINDS[self.o_kind[x[0]]] in OBJECTS]
        pers = [x for x in self.visible_persons(i) if x[0] != cg.pid]
        w = dict(QTASK_W)
        if not everyday:
            w['what'] = 0
        if not objs:
            w['colour'] = w['size'] = w['check'] = 0
        countable = [x for x in objs if KINDS[self.o_kind[x[0]]] not in ANIMALS]
        if not countable:
            w['count'] = 0
        if not self.hold[i]:
            w['holding'] = 0
        if self.S.TER[self.pos[i]] == lex.T_DOOR:
            w['room'] = 0
        if not pers:
            w['who'] = 0
        task = self._wchoice(w)
        if task is None:
            return False
        near = lambda lst: lst[int(self.rng.integers(min(3, len(lst))))]   # noqa: E731
        ex = dict(task=task, held_out=False)
        if task in ('what', 'colour', 'size', 'check', 'count'):
            o, c, k = near(everyday if task == 'what' else countable if task == 'count' else objs)
            X = KINDS[self.o_kind[o]]
            ex['point'] = c
            ex['tags'] = dict(view_k=k, obj=X)
            if task == 'what':
                ex.update(prompt='这是什么', acceptable=[X, '是' + X, '这是' + X],
                          model=X if self.rng.random() < 0.7 else '这是' + X)
            elif task == 'colour':
                C = COLOURS[self.o_col[o]]
                ex.update(prompt=X + '是什么颜色', acceptable=[C, '是' + C, C[0] + '的'], model=C)
            elif task == 'count':
                Nn = COUNTS[self.o_cnt[o] - 1]
                ex.update(prompt='有几个' + X, acceptable=[Nn, '有' + Nn], model=Nn)
            elif task == 'size':
                acc = ['大', '大的', '很大'] if self.o_size[o] else ['小', '小的', '不大']
                ex.update(prompt=X + '大不大', acceptable=acc, model=acc[1])
            else:
                Y = X
                if self.rng.random() < 0.5:
                    X = self._pick([q for q in OBJECTS if q != Y])
                acc = ['是', '是的', '对'] if X == Y else ['不是', '不对']
                ex.update(prompt='这是' + X + '吗', acceptable=acc, model=acc[0])
            ex['item'] = (task, X)
            ex['held_out'] = X in H.get(task, ())
        elif task == 'holding':
            Y = KINDS[self.o_kind[self.hold[i]]]
            ex.update(prompt='你拿着什么', acceptable=[Y, '是' + Y, '我拿着' + Y], model=Y, item=('holding', Y),
                      held_out=Y in H['holding'])
        elif task == 'room':
            R = ROOMS[self.room_at(i)]
            ex.update(prompt='这是哪里', acceptable=[R, '是' + R, '在' + R, '这是' + R], model=R, item=('room', R))
        else:
            pid, c, k = near(pers)
            Nm = self.names[pid]
            ex.update(prompt='这是谁', acceptable=[Nm, '是' + Nm, '这是' + Nm], model=Nm, item=('who', Nm), point=c)
        yield from self._exchange(cg, i, ex)
        return True

    # ---------------- naming statements (no answer expected; an imitation is praised)
    def _t_name(self, cg, i):
        objs = self.visible_objs(i)
        if not objs:
            return False
        o, c, k = objs[int(self.rng.integers(min(3, len(objs))))]
        X = KINDS[self.o_kind[o]]
        forms = ['这是' + X, '看' + X, X, COLOURS[self.o_col[o]] + '的' + X]
        if X not in ANIMALS:
            forms += [COUNTS[self.o_cnt[o] - 1] + X, ('大' if self.o_size[o] else '小') + X]
        text = self._pick(forms)
        yield ('point', c)
        try:
            yield _say(text, to=i, kind='name', tell=(('name', X), None))
            res = yield ('listen', [i], dict(min_gap=self.p['model_wait']))
            reply = res[i]['reply']
            ok = reply in (X, '这是' + X, '是' + X, '看' + X)
            self.log.append(dict(type='q', t=self.t, agent=i, by=cg.name, task='name', kind='statement', prompt=text,
                                 acceptable=[X], reply=reply, acc=ok, item=('name', X), held_out=False,
                                 needs_scene=True, day=self.day_i))
            if ok:
                yield _say('对', to=i, reward=self.p['reward_imitate'], kind='praise')
        finally:
            self._point(cg, None)
        return True

    # ---------------- facts: who is who (told now, asked much later)
    def _t_fact(self, cg, i):
        r = self.rng
        opts = ['person', 'myname']
        pers = [x for x in self.visible_persons(i) if x[0] < self.N]
        if pers:
            opts.append('who')
        f = self._pick(opts)
        if f == 'person':
            nm = cg.name
            yield _say('我是' + nm, to=i, kind='fact', tell=(('person', nm), None))
        elif f == 'myname':
            nm = self.names[i]
            yield _say('你叫' + nm, to=i, kind='fact',
                       tell=((lambda j, i=i, nm=nm: ('myname', nm) if j == i else ('who', nm)), None))
        else:
            j, c, k = pers[int(r.integers(len(pers)))]
            nm = self.names[j]
            yield ('point', c)
            try:
                yield _say('这是' + nm, to=i, kind='fact',
                           tell=((lambda q, j=j, nm=nm: ('myname', nm) if q == j else ('who', nm)), None))
            finally:
                self._point(cg, None)
        return True

    # ---------------- novel objects: named once, asked much later
    def _place_for(self, cg, i):
        """a free cell next to cg that agent i can see (else any free cell next to cg)."""
        vc = set(int(c) for c in self.view_cells(i))
        c0 = int(self.pos[cg.pid])
        near = [c0 + d1 + d2 for d1 in (-_PW, 0, _PW) for d2 in (-1, 0, 1) if d1 or d2]
        ok = [c for c in near if self._hold[c] and not self._ob[c] and not self._pb[c]
              and self.S.TER[c] != lex.T_DOOR]
        vis = [c for c in ok if c in vc]
        if vis:
            return vis[0]
        return ok[0] if ok else None

    def _bring(self, cg, o, i):
        """fetch object o (lying somewhere) and put it down where agent i can see it; returns its cell or None."""
        if self.hold[cg.pid] or self.o_cell[o] < 0:
            return None
        ok = yield from self._go(cg, ('adj8', self.o_cell[o]), 500)
        if not ok:
            return None
        got = yield ('call', self._cg_take, cg, o)
        if not got:
            return None
        ok = yield from self._go_to_agent(cg, i)
        c = self._place_for(cg, i)
        if c is None:
            return None
        yield ('call', self._cg_put, cg, o, c)
        return c if ok else None

    def _t_novel(self, cg, i):
        cand = [X for X in NOVEL if self.exposures[i][('novel', X)] == 0]
        if not cand or self.hold[cg.pid]:
            return False
        fresh = [X for X in cand if not self.kind_objs[KIND_ID[X]]]
        if fresh:
            X = self._pick(fresh)
            c = self._place_for(cg, i)
            if c is None:
                return False
            o = self._spawn(KIND_ID[X], cell=c)
            self.log.append(dict(type='event', t=self.t, what='novel_appears', kind=X, agent=i))
        else:
            X = self._pick(cand)
            objs = [o for o in self.kind_objs[KIND_ID[X]] if self.o_cell[o] >= 0]
            if not objs:
                return False
            o = objs[0]
            vis = {x[0] for x in self.visible_objs(i)}
            if o in vis:
                c = self.o_cell[o]
            else:
                c = yield from self._bring(cg, o, i)
                if c is None:
                    return True
        yield ('point', c)
        try:
            cell = c
            yield _say('这是' + X, to=i, kind='novel',
                       tell=((lambda j, X=X, cell=cell: ('novel', X) if cell in set(self.view_cells(j).tolist())
                              else None), None))
            res = yield ('listen', [i], dict(min_gap=self.p['model_wait']))
            reply = res[i]['reply']
            ok = reply in (X, '这是' + X, '是' + X)
            self.log.append(dict(type='q', t=self.t, agent=i, by=cg.name, task='novel_intro', kind='statement',
                                 prompt='这是' + X, acceptable=[X], reply=reply, acc=ok, item=('novel', X),
                                 held_out=False, needs_scene=True, day=self.day_i))
            if ok:
                yield _say('对', to=i, reward=self.p['reward_imitate'], kind='praise')
        finally:
            self._point(cg, None)
        return True

    # ---------------- memory probes (always probes: never answered afterwards)
    def _t_memory(self, cg, i):
        t, md, sp = self.t, self.p['min_mem_delay'], self.p['mem_spacing']
        ex_i, lt, la = self.exposures[i], self.last_told[i], self.last_asked[i]
        ep0 = cg.ep_t0

        def ok_told(item):
            return ex_i[item] >= 1 and lt.get(item, t) < ep0 and t - lt.get(item, t) >= md and t - la.get(item, -10 ** 9) >= sp
        opts = {}
        # where: an object out of view whose current place the agent saw or was told after its last move
        where = []
        for o in range(1, len(self.o_kind)):
            if not self.o_alive[o] or self.o_cell[o] < 0 or o in self._vis[i]:
                continue
            if len(self.kind_objs[self.o_kind[o]]) != 1:
                continue
            seen = self.seen_until[i].get(o, -1)
            told = self.told_obj[i].get(o, -1)
            known = max(seen, told)
            if known >= self.o_tmove[o] and t - known >= md and self.S.ROOM[self.o_cell[o]] >= 0 \
                    and t - la.get(('where', KINDS[self.o_kind[o]]), -10 ** 9) >= sp:
                where.append((o, known, 'seen' if seen >= self.o_tmove[o] else 'told'))
        if where:
            opts['where'] = where
        nov = [X for X in NOVEL if ok_told(('novel', X)) and any(self.o_cell[o] >= 0 for o in self.kind_objs[KIND_ID[X]])]
        if nov:
            opts['novel'] = nov
        vc = set(self.view_cells(i).tolist())
        if ok_told(('person', cg.name)) and int(self.pos[cg.pid]) in vc:
            opts['person'] = [cg.name]
        nm = self.names[i]
        if ok_told(('myname', nm)):
            opts['myname'] = [nm]
        if not opts:
            return False
        typ = self._pick(sorted(opts))
        if typ == 'where':
            o, known, src = self._pick(opts['where'])
            X, R = KINDS[self.o_kind[o]], ROOMS[int(self.S.ROOM[self.o_cell[o]])]
            item = ('where', X)
            ex = dict(task='where', prompt=X + '在哪里', acceptable=['在' + R, R], model='在' + R, item=item,
                      held_out=False, kind='probe',
                      tags=dict(ticks_since_told=t - known, source=src, obj=X, ticks_since_moved=t - self.o_tmove[o],
                                same_room=self.room_at(i) == int(self.S.ROOM[self.o_cell[o]])))
        elif typ == 'novel':
            X = self._pick(opts['novel'])
            objs = [o for o in self.kind_objs[KIND_ID[X]] if self.o_cell[o] >= 0]
            o = objs[0]
            vis = {x[0] for x in self.visible_objs(i)}
            if o in vis:
                c = self.o_cell[o]
            else:
                c = yield from self._bring(cg, o, i)
                if c is None:
                    return True
            item = ('novel', X)
            ex = dict(task='novel', prompt='这是什么', acceptable=[X, '是' + X, '这是' + X], model=X, item=item,
                      held_out=False, kind='probe', point=c,
                      tags=dict(ticks_since_told=self.t - lt[item], ticks_since_first_told=self.t - self.first_told[i][item]))
        elif typ == 'person':
            item = ('person', cg.name)
            ex = dict(task='person', prompt='我是谁', acceptable=[cg.name, '你是' + cg.name, '是' + cg.name],
                      model=cg.name, item=item, held_out=False, kind='probe',
                      tags=dict(ticks_since_told=t - lt[item], ticks_since_first_told=t - self.first_told[i][item]))
        else:
            item = ('myname', nm)
            ex = dict(task='myname', prompt='你叫什么', acceptable=[nm, '我叫' + nm, '叫' + nm], model=nm, item=item,
                      held_out=False, kind='probe',
                      tags=dict(ticks_since_told=t - lt[item], ticks_since_first_told=t - self.first_told[i][item]))
        yield from self._exchange(cg, i, ex)
        return True

    # ---------------- offering food to a hungry agent (你要X吗 -> 要; then it gets the food)
    def _t_offer(self, cg, i):
        if self.HUNGER[i] <= 0.4 or self.hold[i] or self.hold[cg.pid]:
            return False
        foods = [o for o in range(1, len(self.o_kind)) if self.o_alive[o] and self.o_cell[o] >= 0
                 and KINDS[self.o_kind[o]] in FOODS]
        if not foods:
            return False
        cx, cy = self.S.xy[self.pos[cg.pid]]
        foods.sort(key=lambda o: abs(self.S.xy[self.o_cell[o]][0] - cx) + abs(self.S.xy[self.o_cell[o]][1] - cy))
        o = foods[0]
        ok = yield from self._go(cg, ('adj8', self.o_cell[o]), 500)
        if not ok:
            return True
        got = yield ('call', self._cg_take, cg, o)
        if not got:
            return True
        ok = yield from self._go_to_agent(cg, i)
        if not ok:
            return True
        X = KINDS[self.o_kind[o]]
        ex = dict(task='offer', prompt='你要' + X + '吗', acceptable=['要', '要' + X, '我要' + X], model='要' + X,
                  item=('offer', X), held_out=X in self.held['offer'], point=int(self.pos[cg.pid]))
        rec = yield from self._exchange(cg, i, ex)
        if self.hold[cg.pid] == o and not self.hold[i] and self.adj4(cg.pid, i):
            yield _say('给你', to=i, kind='give')
            yield ('call', self._cg_give, cg, i)
        return True

    def _cg_give(self, cg, i):
        o = self.hold[cg.pid]
        if o and not self.hold[i] and self.adj4(cg.pid, i):
            self._take_from(cg.pid)
            self._give_to(i, o)
            if KINDS[self.o_kind[o]] in EDIBLE:
                self.gave_food[i] = self.t
            return True
        return False

    # ---------------- requests: comprehension shown by action (scored by the outcome)
    def _req_rec(self, cg, i, typ, X, text):
        return dict(type='req', t=self.t, agent=i, by=cg.name, req=typ, target=X, prompt=text, day=self.day_i,
                    dist0=float(self.cheb(cg.pid, i)), n_asked_before=int(self.asked[i][('req', typ, X)]))

    def _agent_can_reach(self, i, cells):
        c0 = int(self.pos[i])
        if c0 in cells:
            return True
        return self._bfs(c0, tuple(cells)) is not None

    def _near8(self, pid):
        c = int(self.pos[pid])
        return {c + d1 + d2 for d1 in (-_PW, 0, _PW) for d2 in (-1, 0, 1) if d1 or d2}

    def _req_come(self, cg, i):
        p = self.p
        if not self._agent_can_reach(i, self._near8(cg.pid)):
            return False
        others = [j for j in self.heard_by(int(self.pos[cg.pid])) if j != i]
        text = self.names[i] + '过来' if (others or self.rng.random() < 0.5) else '过来'
        rec = self._req_rec(cg, i, 'come', None, text)
        self.asked[i][('req', 'come', None)] += 1
        start = {j: self.cheb(j, cg.pid) for j in others}
        yield _say(text, to=i, kind='req')
        self._hook_prompt(i, dict(kind='req', req='come', prompt=text, target=None, cg=cg.pid))
        ok, dt = yield from self._watch(lambda: self.cheb(i, cg.pid) <= 1, p['request_ticks'])
        rec.update(success=bool(ok), ticks=dt, named=text != '过来',
                   others=len(others), others_came=sum(1 for j in others if start[j] >= 2 and self.cheb(j, cg.pid) <= 1))
        self.log.append(rec)
        if ok:
            yield _say(self._pick(PRAISE), to=i, reward=p['reward_request'], kind='praise')
        return ok

    def _t_request(self, cg, i):
        p = self.p
        name = self.names[i]
        others = [j for j in self.heard_by(int(self.pos[cg.pid])) if j != i]
        pre = name if (others or self.rng.random() < 0.5) else ''
        if self.hold[i]:
            X = KINDS[self.o_kind[self.hold[i]]]
            if self.hold[cg.pid] or (X in EDIBLE and self.t - self.gave_food[i] < 600):   # time to eat it first
                return False
            yield from self._req_give(cg, i, X, pre)
            return True
        cand = [x for x in self.visible_objs(i) if KINDS[self.o_kind[x[0]]] not in ANIMALS]
        if not cand:
            room = self.room_at(i)
            cand = [(o, self.o_cell[o], -1) for o in range(1, len(self.o_kind)) if self.o_alive[o]
                    and self.o_cell[o] >= 0 and KINDS[self.o_kind[o]] not in ANIMALS
                    and int(self.S.ROOM[self.o_cell[o]]) == room]
        if not cand:
            return False
        self.rng.shuffle(cand)
        for o, c, k in cand:                       # a kind the agent can actually get to
            X = KINDS[self.o_kind[o]]
            reach = {self.o_cell[q] + d for q in self.kind_objs[KIND_ID[X]] if self.o_cell[q] >= 0 for d in _DIRS}
            if self._agent_can_reach(i, reach):
                break
        else:
            return False
        text = pre + '去拿' + X
        rec = self._req_rec(cg, i, 'fetch', X, text)
        rec.update(visible=k >= 0)
        self.asked[i][('req', 'fetch', X)] += 1
        yield _say(text, to=i, kind='req')
        self._hook_prompt(i, dict(kind='req', req='fetch', prompt=text, target=X, cg=cg.pid))
        kid = KIND_ID[X]
        ok, dt = yield from self._watch(lambda: bool(self.hold[i]) and self.o_kind[self.hold[i]] == kid,
                                        p['request_ticks'])
        rec.update(success=bool(ok), ticks=dt)
        self.log.append(rec)
        if ok:
            yield _say(self._pick(PRAISE), to=i, reward=p['reward_request'], kind='praise')
            if self.rng.random() < 0.5 and not self.hold[cg.pid]:
                yield ('wait', 10)
                yield from self._req_give(cg, i, X, '')
        return True

    def _req_give(self, cg, i, X, pre):
        p = self.p
        text = pre + '把' + X + '给我'
        rec = self._req_rec(cg, i, 'give', X, text)
        self.asked[i][('req', 'give', X)] += 1
        cg.received = None
        yield _say(text, to=i, kind='req')
        self._hook_prompt(i, dict(kind='req', req='give', prompt=text, target=X, cg=cg.pid))
        kid = KIND_ID[X]
        ok, dt = yield from self._watch(lambda: cg.received is not None and cg.received[0] == i
                                        and self.o_kind[cg.received[1]] == kid, p['request_ticks'])
        rec.update(success=bool(ok), ticks=dt)
        self.log.append(rec)
        if ok:
            yield _say('谢谢', to=i, reward=p['reward_request'], kind='praise')
        return ok

    # ---------------- greetings and goodnights (scored social replies)
    def _social(self, cg, agents, what):
        prompt, acc = {'morning': ('早上好', GREET_MORNING), 'greet': ('你好', GREET),
                       'goodnight': ('晚安', GOODNIGHT)}[what]
        task = 'goodnight' if what == 'goodnight' else 'greet'
        recs = {}
        for j in agents:
            item = ('social', prompt)
            recs[j] = dict(type='q', t=self.t, agent=j, by=cg.name, task=task, prompt=prompt, acceptable=list(acc),
                           kind='social', held_out=False, needs_scene=False, item=item, new_situation=True,
                           n_experienced=int(self.exposures[j][item]), n_asked_before=int(self.asked[j][item]),
                           day=self.day_i)
            self.asked[j][item] += 1
        yield _say(prompt, to=agents[0] if len(agents) == 1 else None, kind='social')
        t_end = self.t
        for j in agents:
            self._hook_prompt(j, dict(kind='social', exkind='social', task=task, prompt=prompt, item=('social', prompt),
                                      acceptable=list(acc), eye_hash=None, cg=cg.pid))
        res = yield ('listen', list(agents), {})
        for j in agents:
            R = res.get(j, dict(reply='', first=None, stop='none'))
            rec = recs[j]
            rec.update(reply=R['reply'], t_first=R['first'], stop=R['stop'], t_prompt_end=t_end,
                       latency=(R['first'] - t_end) if R['first'] is not None else None)
            rec['acc'] = R['reply'] in acc
            rec['first_ok'] = bool(R['reply']) and R['reply'][0] in {a[0] for a in acc}
            if rec['acc']:
                self._reward(j, self.p['reward_social'])
            self.log.append(rec)

    # ================================================================ the agents' initiatives
    def _on_take(self, j, o):
        pass

    def _on_give(self, j, pid, o):
        """agent j handed object o to person pid."""
        if pid < self.N:
            self.log.append(dict(type='event', t=self.t, what='a2a_give', src=j, dst=pid, kind=KINDS[self.o_kind[o]]))
            return
        cg = self.cgs[pid - self.N]
        cg.received = (j, o, self.t)
        waiting = cg.op is not None and cg.op[0] == 'watch'
        if not waiting:
            self._preempt(cg, self._r_say(cg, j, '谢谢', 0.0), walks=False)

    def _r_say(self, cg, j, text, reward, **kw):
        yield _say(text, to=j, reward=reward, kind='react', **kw)

    def _classify(self, j, text):
        """category of an agent's utterance outside an answer window (see README)."""
        if text in GREET_WORDS:
            return 'greeting', None
        ks = self.visible_kinds(j)
        for kd in ks:
            X = KINDS[kd]
            if text in (X, '这是' + X, '是' + X, '看' + X, '有' + X):
                return 'naming', X
        if text in CAREGIVERS:
            return 'calling', text
        if text == '这是什么' and self.visible_objs(j):
            return 'asking', None
        if text in HUNGRY_WORDS:
            return ('hungry', None) if self.HUNGER[j] > 0.3 else ('other', None)
        for pre in ('我要', '要'):
            if text.startswith(pre) and text[len(pre):] in KIND_ID:
                X = text[len(pre):]
                if any(self.o_cell[o] >= 0 for o in self.kind_objs[KIND_ID[X]]) and X not in ANIMALS:
                    return 'request', X
        return 'other', None

    def _initiative(self, j, b):
        text = b['text']
        t = self.t
        self.mon['initiatives'][j] += 1
        rec = dict(type='init', t=b['t0'], t_end=b['t1'], agent=j, text=text, syllables=len(b['chars']),
                   overlap=b['overlap'], interrupt=bool(b['interrupt']), interrupt_any=bool(b['interrupt_any']),
                   night=self.night, appropriate=False, reward=0.0, by=None, day=self.day_i)
        cell = int(self.pos[j])
        hearers = [cg for cg in self.cgs if self.present[cg.pid] and self.ATT[self.pos[cg.pid], cell] < 6
                   and cg.pid not in self._say and cg.held_by is None]
        rec['heard'] = bool(hearers)
        rec['agents_heard'] = len(self._last_bout[j][1]) - 1 if self._last_bout[j] else 0
        cat, x = self._classify(j, text)
        if self.night or not hearers:
            rec['category'] = 'alone'
            rec['said'] = cat
            self.log.append(rec)
            return
        cd = self.cooldown[j]
        if cat == 'greeting':
            fresh = [cg for cg in hearers if ('greeted_by', j) not in cg.greeted]
            if not fresh:
                cat = 'repeat'
        elif cat == 'naming':
            if t - cd.get(('name', x), -10 ** 9) < self.p['naming_cooldown']:
                cat = 'repeat'
        elif cat == 'calling':
            if t - cd.get('call', -10 ** 9) < self.p['call_cooldown'] or x not in [c.name for c in hearers]:
                cat = 'repeat' if x in [c.name for c in hearers] else 'other'
        elif cat == 'asking':
            if t - cd.get('ask', -10 ** 9) < self.p['ask_cooldown']:
                cat = 'repeat'
        elif cat in ('request', 'hungry'):
            if t - cd.get('req', -10 ** 9) < 600:
                cat = 'repeat'
        rec['category'] = cat
        rec['object'] = x
        rec['appropriate'] = cat in APPROPRIATE and not b['interrupt']
        if b['interrupt'] and cat in APPROPRIATE:          # started while its own caregiver was talking: no reaction
            rec['category'], rec['said'] = 'interrupt', cat
        if rec['appropriate']:
            if cat == 'calling':
                order = [cg for cg in hearers if cg.name == x]
            else:
                cx = self.S.xy[cell]
                order = sorted(hearers, key=lambda cg: float(np.abs(self.S.xy[self.pos[cg.pid]] - cx).max()))
            R = self.p['reward_init']
            for cg in order:
                if cat == 'greeting':
                    g = self._r_say(cg, j, '你好', R)
                elif cat == 'naming':
                    g = self._r_say(cg, j, '对是' + x, R, tell=(('name', x), None))
                elif cat == 'calling':
                    g = self._r_call(cg, j, R)
                elif cat == 'asking':
                    g = self._r_answer(cg, j)
                else:
                    g = self._r_bring(cg, j, x, R)
                if self._preempt(cg, g, walks=cat in ('calling', 'request', 'hungry')):
                    rec['by'] = cg.name
                    rec['reward'] = R if cat != 'asking' else 0.0
                    if cat == 'greeting':
                        cg.greeted.add(('greeted_by', j))
                        cg.greeted.add(j)
                    elif cat == 'naming':
                        cd[('name', x)] = t
                    elif cat == 'calling':
                        cd['call'] = t
                    elif cat == 'asking':
                        cd['ask'] = t
                    else:
                        cd['req'] = t
                    break
                g.close()
            if rec['by'] is None:
                rec['category'] = 'unattended'
                rec['said'] = cat
        self.log.append(rec)

    def _r_call(self, cg, j, R):
        yield _say('怎么了', to=j, reward=R, kind='react')
        if self.engaged[j] is None and cg.partner is None:
            self.engaged[j] = cg.pid
            cg.partner = j
            try:
                if not self.adj4(cg.pid, j):
                    ok = yield from self._go_to_agent(cg, j, 300)
                    if not ok:
                        return
                yield from self._turn(cg, j)
            finally:
                if self.engaged[j] == cg.pid:
                    self.engaged[j] = None
                cg.partner = None
                self._point(cg, None)

    def _r_answer(self, cg, j):
        """the agent asked 这是什么: name the nearest object it sees (not for held-out 'what' kinds)."""
        objs = self.visible_objs(j)
        if not objs:
            return
        o, c, k = objs[0]
        X = KINDS[self.o_kind[o]]
        yield ('point', c)
        try:
            if X in self.held['what']:
                yield _say('你说呢', to=j, kind='react')
            else:
                yield _say('这是' + X, to=j, kind='react', teach=(('what', X), '这是什么', '这是' + X))
        finally:
            self._point(cg, None)

    def _r_bring(self, cg, j, X, R):
        """the agent asked for X (or said it is hungry): fetch it and hand it over."""
        yield _say('好', to=j, reward=R, kind='react')
        if self.hold[cg.pid] or (self.engaged[j] is not None and self.engaged[j] != cg.pid):
            return
        if X is None:
            cand = [o for o in range(1, len(self.o_kind)) if self.o_alive[o] and self.o_cell[o] >= 0
                    and KINDS[self.o_kind[o]] in EDIBLE]
        else:
            cand = [o for o in self.kind_objs[KIND_ID[X]] if self.o_cell[o] >= 0]
        if not cand:
            yield _say('没有了', to=j, kind='react')
            return
        cx, cy = self.S.xy[self.pos[cg.pid]]
        cand.sort(key=lambda o: abs(self.S.xy[self.o_cell[o]][0] - cx) + abs(self.S.xy[self.o_cell[o]][1] - cy))
        o = cand[0]
        ok = yield from self._go(cg, ('adj8', self.o_cell[o]), 500)
        if not ok:
            return
        got = yield ('call', self._cg_take, cg, o)
        if not got:
            return
        ok = yield from self._go_to_agent(cg, j, 400)
        if ok and not self.hold[j]:
            yield _say('给你' + KINDS[self.o_kind[o]], to=j, kind='give', tell=(('name', KINDS[self.o_kind[o]]), None))
            yield ('call', self._cg_give, cg, j)
