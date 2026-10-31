# -*- coding: utf-8 -*-
"""Continuous 'life' mode: one unbroken life from birth, learning always on, probes embedded and scored as they happen.

    from life import Life, evaluate_life
    L = Life(seed=0, tutor='model')                 # same physics / curriculum / held-out sets / eye codes as World
    log = L.run(agent, hours=6)                     # or ticks=...; 1 h = 48,000 ticks (75 ms per tick); run() may be repeated
    res = evaluate_life(log)

The life (content drawn from its own rng, so it does not depend on the agent's behaviour; only timing does):
  day (day_ticks) = visits separated by alone periods, then night (night_ticks): a long dark, silent stretch (eye all
           zero, ear silent).  It has no sleep semantics: the world tells the agent nothing, and no metric is about it.
  visit  = the partner arrives (a 'person' code appears in the eye), waits greet_wait ticks (the agent may greet first),
           greets (你好 / 早上好 in the morning) unless greeted, 2-6 topics separated by gaps of 20-300 silent ticks, says
           再见 (晚安 before the night) and leaves.  other2 is present in 30 % of visits and chats with the partner.
  topics = a B/C/D question about the CURRENT scene (taught per tutor mode / probe, held-out pairs only probed) |
           a stable fact comes up (again): 我的小狗叫N (4 pets, fixed names), 我叫N / 他叫N (partner, other2), 我把X放在P了
           (X leaves the scene; objects have habitual places) | a novel object is introduced (这是X) or, when it is around
           again, named again | a memory question about a fact last told in an EARLIER visit (always a probe) |
           刚才S是什么 after S's object changed by itself, if the old object was never named since it appeared |
           an overheard question/answer between other2 and the partner | (topics='BCDE') an adult-corpus pair.
  alone  = 100-2,000 ticks (10 %: up to 6,000) with no partner; objects appear, move and disappear by themselves.
  initiative: any agent utterance outside an answer window.  Partner present and silent: a greeting in the greet window,
           or naming a visible object (X / 这是X / 是X / 看X / 有X; once per object per visit) -> reply (你好 / 对是X) with
           reward_init on its first tick; anything else -> no reward and the partner pauses pause_nonsense ticks.
           Speech while a partner speaks is an overlap / interruption.  Alone or at night: logged, no reaction.
"""
from __future__ import annotations

import collections

import numpy as np

from physics import PAUSE
from world import (COLOURS, COUNTS, FOODS, NAMES, NOVEL, OBJECTS, RITUALS, SLOTS, STAGE_TASKS, TASKS, Track, World,
                   _rng)

PETS = ['小狗', '小猫', '小鸟', '鸭子']
PLACES = ['盒子里', '桌子上', '床上', '外面', '包里']
GREETINGS = {'你好', '你好呀', '早上好', '早'}
TICKS_PER_HOUR = 48000                  # 75 ms per tick

LIFE_DEFAULTS = dict(
    day_ticks=20000, night_ticks=5000,
    visits_per_day=(8, 14),
    alone_min=100, alone_max=2000, p_long_alone=0.1, long_alone_max=6000,
    visit_topics=(2, 6), gap_min=20, gap_max=300,
    scene_event_every=600,      # mean ticks between scene events while alone
    p_scene_event_gap=0.15,     # per topic gap in a visit
    p_other2=0.3,               # other2 present in a visit
    w_qa=0.45, w_fact=0.15, w_novel=0.08, w_memq=0.2, w_before=0.07, w_overheard=0.15, w_corpus=0.15,
    greet_wait=30, reward_init=1.0, reward_social=1.0, pause_nonsense=30,
    p_novel_back=0.15,          # an 'appear' event brings back a novel object named earlier
    p_home=0.7,                 # an object is put away at its habitual place (else a random one)
    qa_stage_w={'B': 0.4, 'C': 0.4, 'D': 0.2},   # stage of a scene question (D = 这是X吗)
    topics='BCD',               # 'BCDE' adds adult-corpus pairs as topics
    mon_every=10000,            # monitor snapshot interval (ticks)
)


class Life(World):
    def __init__(self, seed=0, physics='random', noise=0.0, tutor='model', vocab=None, **params):
        lp = {k: params.pop(k) for k in list(params) if k in LIFE_DEFAULTS}
        self.lp = dict(LIFE_DEFAULTS); self.lp.update(lp)
        topics = self.lp['topics']
        if not set(topics) <= set('BCDE'):
            raise ValueError(topics)
        super().__init__(stage=topics, seed=seed, physics=physics, noise=noise, tutor=tutor, vocab=vocab, **params)
        self.stage = 'life'
        self.rng = _rng(self.seed, 77, {'model': 1, 'force': 2, 'recast': 3}[tutor])
        rp = _rng(self.seed, 14)
        n_eye = self.p['n_eye']
        self.person_code = {'other': rp.choice(n_eye, 8, replace=False), 'other2': rp.choice(n_eye, 8, replace=False)}
        self.present = {'other': False, 'other2': False}
        self.night = False
        self._dark = np.zeros(n_eye, np.uint8); self._dark.setflags(write=False)
        self._eye_life = {}
        self._zero_win = np.zeros((self.voice.F, self.voice.T), int); self._zero_win.setflags(write=False)
        self._quiet = self.noise == 0
        rf = _rng(self.seed, 15)         # life constants (seed only): stable names and habitual places
        nm = [str(x) for x in rf.permutation(NAMES)]
        self.pet_name = dict(zip(PETS, nm[:4]))
        self.person_name = {'other': nm[4], 'other2': nm[5]}
        self.home = {o: PLACES[int(rf.integers(len(PLACES)))] for o in OBJECTS}
        self.scene = self._random_scene()
        self.facts = {}                  # key -> dict(task, prompt, acceptable, item, told_t, first_t, told_visit)
        self.asked = collections.Counter()   # item -> times asked so far (any kind)
        self.mentioned = {}              # object -> last tick named by the world (content-driven only)
        self.appeared = {}               # object -> tick it appeared in its current slot
        self.changes = []                # (t, slot, old, new) scene changes
        self.visit_i = 0; self.day_i = 0
        self.named_this_visit = set()
        self._greet_open = False; self._greeted = False
        self._pause_debt = 0
        self.mon = collections.Counter()
        self._bout_syl = 0; self._bout_ticks = 0; self._bout_silent = 99; self._bout_overlap = 0
        self._next_mon = self.p['lead_in'] + self.lp['mon_every']
        self._cur_stage = 'life'
        self.taught_situations = set()   # (prompt, scene, attend) whose reply was heard taught
        self._before_from = 0

    # ================================================================ eye with persons, darkness at night
    def _set_scene(self, scene, attend=None):
        self.scene, self.attend = scene, attend
        if self.night:
            self.eye = self._dark
        else:
            key = (scene, attend, self.present['other'], self.present['other2'])
            e = self._eye_life.get(key)
            if e is None:
                e = self.cur.eye(scene, attend).copy()
                for who, on in self.present.items():
                    if on:
                        e[self.person_code[who]] = 1
                e.setflags(write=False)
                if len(self._eye_life) > 200000:
                    self._eye_life.clear()
                self._eye_life[key] = e
            self.eye = e
        self._hook('bench_scene', dict(scene=scene, attend=attend, present=dict(self.present), night=self.night))

    # ================================================================ one tick (fast when the world is silent)
    def _tick(self, other=None, other2=None, cue=None, reward=0.0):
        v = self.voice
        tok, g, fresh = self._own
        partner = other is not None or other2 is not None
        if not partner and tok is None and self._quiet:
            win = self._zero_win                                 # silent world: the window is already all zero
        else:
            s1 = (other[0], v.hear(other[0], 'other'), other[1]) if other is not None else (None, None, False)
            s2 = (other2[0], v.hear(other2[0], 'other2'), other2[1]) if other2 is not None else (None, None, False)
            if tok == '~':
                g = v.hear('~', 'self')
            win = self.stream.tick((s1, s2, (tok, g, fresh)))
            self._quiet = (self.noise == 0 and not partner and tok is None and not win.any())
        obs = {'ear': win, 'eye': self.eye, 'reward': float(reward), 'cue': cue}
        m = self.agent.step(obs)
        if not isinstance(m, np.ndarray):
            m = np.asarray(m)
        if not m.any():
            said, onset = None, False
            self._pact, self._pmax = (), 0.0
            self._own = (None, None, False)
        else:
            m = np.asarray(m)
            act = np.flatnonzero(m > 0)
            tup = tuple(act.tolist())
            said = v.decode(tup)
            if said is None:
                onset = False; pmax = 0.0
            else:
                pmax = float(m[act].max())
                onset = (not self._pact) or tup != self._pact or pmax < self._pmax
            self._pact, self._pmax = tup, pmax
            if said is None:
                self._own = (None, None, False)
            elif self.physics == 'lawful':
                self._own = (tup, v.hear_set(tup), onset)
            else:
                self._own = (said, None if said == '~' else v.hear(said, 'self'), onset)
        # ---- life monitors
        mon = self.mon
        if partner:
            mon['partner_ticks'] += 1
        else:
            mon['quiet_ticks'] += 1
        if said is not None:
            mon['mouth_ticks'] += 1
            if self._bout_silent >= PAUSE:
                self._bout_syl = 0; self._bout_ticks = 0; self._bout_overlap = 0
                mon['bouts'] += 1
            self._bout_ticks += 1
            if onset:
                self._bout_syl += 1; mon['syllables'] += 1
                if said == '~':
                    mon['garble'] += 1
            if partner:
                mon['overlap_ticks'] += 1
                if self._bout_overlap == 0:
                    mon['interruptions'] += 1
                self._bout_overlap += 1
            if self._bout_syl > mon['longest_syl']:
                mon['longest_syl'] = self._bout_syl; mon['longest_syl_t'] = self.t
            if self._bout_ticks > mon['longest_ticks']:
                mon['longest_ticks'] = self._bout_ticks
            self._bout_silent = 0
        else:
            self._bout_silent += 1
        if self.trk:
            ev = getattr(self.agent, 'tick_event', None)
            for tr in self.trk:
                tr.tick(self.t, said, onset, partner, ev)
        self.t += 1
        if self.t >= self._next_mon:
            self._next_mon += self.lp['mon_every']
            self.log.append(dict(type='mon', t=self.t, **{k: int(x) for k, x in mon.items()}))
        return said, onset

    # ================================================================ scene dynamics (content rng)
    def _random_scene(self):
        r = self.rng
        sc = [None, None, None]
        objs = list(r.permutation(OBJECTS))
        for s in range(3):
            if r.random() < 0.75:
                sc[s] = self._rand_item(str(objs.pop()))
        return tuple(sc)

    def _visible(self, scene=None):
        return [it[0] for it in (scene or self.scene) if it is not None]

    def _scene_event(self):
        """an object appears, moves or disappears by itself."""
        r = self.rng
        sc = list(self.scene)
        empty = [s for s in range(3) if sc[s] is None]
        full = [s for s in range(3) if sc[s] is not None]
        kinds = (['appear'] if empty else []) + (['move'] if empty and full else []) + (['disappear'] if full else [])
        kind = kinds[int(r.integers(len(kinds)))]
        if kind == 'appear':
            s = empty[int(r.integers(len(empty)))]
            back = [o for o in self.novel_used if o not in self._visible()]
            if back and r.random() < self.lp['p_novel_back']:
                o = back[int(r.integers(len(back)))]           # a novel object named earlier comes back
            else:
                cand = [o for o in OBJECTS if o not in self._visible()]
                o = cand[int(r.integers(len(cand)))]
            sc[s] = self._rand_item(o)
            self._appear(o, s, None)
        elif kind == 'move':
            a = full[int(r.integers(len(full)))]; b = empty[int(r.integers(len(empty)))]
            sc[b], sc[a] = sc[a], None
            self.changes.append((self.t, a, sc[b][0], None)); self.changes.append((self.t, b, None, sc[b][0]))
            self.appeared[sc[b][0]] = self.t
        else:
            s = full[int(r.integers(len(full)))]
            self.changes.append((self.t, s, sc[s][0], None))
            self._gone(sc[s][0])
            sc[s] = None
        self._set_scene(tuple(sc), self.attend)
        self.log.append(dict(type='event', t=self.t, what='scene_' + kind, scene=self.scene))

    def _appear(self, o, s, old):
        self.changes.append((self.t, s, old, o))
        self.appeared[o] = self.t
        f = self.facts.get(('place', o))
        if f is not None:                       # it is back in view: "where was it put" no longer applies
            del self.facts[('place', o)]

    def _gone(self, o):
        pass

    def _put_in_slot(self, o):
        """make object o visible (replacing a slot if all are full); returns its slot."""
        r = self.rng
        sc = list(self.scene)
        empty = [s for s in range(3) if sc[s] is None]
        s = empty[int(r.integers(len(empty)))] if empty else int(r.integers(3))
        old = sc[s][0] if sc[s] is not None else None
        sc[s] = self._rand_item(o)
        self._appear(o, s, old)
        self._set_scene(tuple(sc), self.attend)
        return s

    # ================================================================ free time and initiative
    def _free(self, n):
        """n ticks in which the world says nothing; agent utterances are initiatives (the partner may react)."""
        t_end = self.t + n
        tr = Track(); self.trk.append(tr)
        while self.t < t_end or (tr.first is not None and tr.silent < PAUSE and self.t < t_end + 60):
            self._tick()
            if tr.first is not None and tr.silent >= PAUSE:
                self.trk.remove(tr)
                self._initiative(tr)
                tr = Track(); self.trk.append(tr)
        self.trk.remove(tr)
        if tr.first is not None:
            self._initiative(tr)

    def _initiative(self, tr):
        tr.close()
        text = tr.text
        rec = dict(type='init', t=tr.first, t_end=(tr.last or tr.first) + 1, text=text, syllables=len(tr.chars),
                   overlap=tr.overlap, partner=self.present['other'], other2=self.present['other2'], night=self.night,
                   visit=self.visit_i, day=self.day_i, reward=0.0, appropriate=False)
        vis = self._visible()
        if self.night or not self.present['other']:
            rec['category'] = 'alone'
        elif self._greet_open and text in GREETINGS:
            rec.update(category='greeting', appropriate=True, reward=self.lp['reward_init'])
            self._greet_open = False; self._greeted = True
            self.log.append(rec)
            self._silence(2)
            self._say('other', '你好', reward_first=self.lp['reward_init'])
            return
        else:
            X = None
            for o in vis:
                if text in (o, '这是' + o, '是' + o, '看' + o, '有' + o):
                    X = o; break
            if X is not None and X not in self.named_this_visit:
                rec.update(category='naming', appropriate=True, reward=self.lp['reward_init'], object=X)
                self.named_this_visit.add(X)
                self.log.append(rec)
                self._silence(2)
                self._say('other', '对是' + X, reward_first=self.lp['reward_init'])
                return
            rec['category'] = 'repeat' if X is not None else 'nonsense'
            if X is None:
                self._pause_debt += self.lp['pause_nonsense']
        self.log.append(rec)

    # ================================================================ partner lines that are not B-D exchanges
    def _statement(self, st):
        super()._statement(st)
        self._life_tags(self.log[-1])

    def _mention(self, *objs):
        for o in objs:
            if o:
                self.mentioned[o] = self.t

    def _social(self, prompt, acceptable, task):
        """greeting / goodbye: the agent's free reply is scored; an acceptable one gets reward (a smile)."""
        p = self.p
        rec = self._record(dict(stage='life', task=task, prompt=prompt, acceptable=acceptable, item=(task, prompt)),
                           'social')
        rec['type'] = 'ex'
        tr_all = Track(); self.trk.append(tr_all)
        self._say('other', prompt)
        rec['t_prompt_end'] = self.t
        prompt_overlap = tr_all.overlap
        info = dict(prompt=prompt, task=task, item=(task, prompt), scene=self.scene, attend=None,
                    acceptable=acceptable, kind='social', stage='life')
        self._hook('bench_prompt', info)
        tr = Track(); self.trk.append(tr)
        cut = self._wait(tr, p['gap_answer'], p['pause_after'], p['reply_cap'])
        self._finish(rec, tr, cut)
        self.trk.remove(tr); self.trk.remove(tr_all)
        good = rec['reply'] in acceptable
        rec['reward'] = self.lp['reward_social'] if good else 0.0
        self._tick(reward=rec['reward'])
        rec.update(mouth_while_partner=tr_all.overlap, mouth_during_prompt=prompt_overlap, t_end=self.t)
        self._life_tags(rec)
        self.log.append(rec)

    def _overheard(self, ex):
        """other2 asks the partner a question about the scene and the partner answers (not directed at the agent)."""
        p = self.p
        rec = self._record(ex, 'overheard')
        rec.update(type='ex', prompt_by='other2', model_reply=ex['model'], model_by='other')
        tr_all = Track(); self.trk.append(tr_all)
        self._say('other2', ex['prompt'])
        rec['t_prompt_end'] = self.t
        self._silence(p['gap_model'])
        self._say('other', ex['model'])
        self._taught(ex, dict(prompt=ex['prompt'], task=ex['task'], item=ex.get('item'), scene=self.scene,
                              attend=ex.get('attend'), acceptable=ex['acceptable'], kind='overheard', stage=ex['stage']),
                     ex['model'])
        tr = Track(); self.trk.append(tr)
        cut = self._wait(tr, p['model_wait'], p['pause_after'], p['reply_cap'])
        self._finish(rec, tr, cut)
        self.trk.remove(tr); self.trk.remove(tr_all)
        rec.update(mouth_while_partner=tr_all.overlap, mouth_during_prompt=0, t_end=self.t)
        self._life_tags(rec)
        self.log.append(rec)

    def _life_tags(self, rec):
        rec.setdefault('type', 'ex')
        rec['life_t'] = rec['t_prompt_start']
        rec['visit'] = self.visit_i; rec['day'] = self.day_i
        if rec.get('acceptable'):
            rec['acc'] = rec['reply'] in rec['acceptable']
            rec['first_ok'] = bool(rec['reply']) and rec['reply'][0] in {a[0] for a in rec['acceptable'] if a}
            # how often this item was experienced before: its answer heard (taught, overheard, recast, or the fact
            # stated), and how often it was asked before (any kind)
            item = rec.get('item')
            rec['n_experienced'] = rec.get('n_exposures', 0)
            rec['n_asked_before'] = self.asked[item]
            self.asked[item] += 1

    def _taught(self, ex, info, reply):
        self.taught_situations.add((info['prompt'], info['scene'], info['attend']))
        super()._taught(ex, info, reply)

    def _ask(self, ex, kind=None):
        """a question exchange through the staged runner (taught per tutor mode, or a probe), with life tags.
        new_situation: this prompt + scene + pointing was never taught before (exact lookup cannot answer it)."""
        new_sit = (ex['prompt'], self.scene, ex.get('attend')) not in self.taught_situations
        if kind is not None:
            ex['kind'] = kind
        else:
            self._kind(ex)
            if ex['kind'] == 'model':
                ex['model_by'] = 'other2' if self.present['other2'] and ex.get('model_by') == 'other2' else 'other'
        rec = self._exchange(ex)
        rec['type'] = 'ex'
        rec['new_situation'] = new_sit
        if rec.get('mouth_during_prompt', 0) > 0:                # interrupted the partner: no reward, a pause
            self._pause_debt += self.lp['pause_nonsense']
        self._life_tags(rec)
        return rec

    # ================================================================ topics
    def _qa_now(self):
        """a B / C / D-check question about the current scene; None if nothing fits."""
        r, p = self.rng, self.p
        stages = [s for s in self.lp['topics'] if s in 'BCD']
        sw = np.array([self.lp['qa_stage_w'][s] for s in stages])
        st = stages[int(r.choice(len(stages), p=sw / sw.sum()))]
        tw = STAGE_TASKS[st] if st != 'D' else {'check': 1.0}
        tasks, w = list(tw), np.array(list(tw.values()))
        vis = self._visible()
        for _ in range(4):
            task = tasks[int(r.choice(len(tasks), p=w / w.sum()))]
            if TASKS[task][2] is None:
                q, acc = RITUALS[int(r.integers(len(RITUALS)))]
                return dict(stage='C', task=task, held_out=False, needs_scene=False, prompt=q, acceptable=list(acc),
                            model=acc[0], item=(task, q))
            held = r.random() < p['held_frac']
            if task == 'check':
                if not vis:
                    continue
                Y = vis[int(r.integers(len(vis)))]
                pool = [x for x in OBJECTS if (x in self.cur.held['check']) == held] or OBJECTS
                X = Y if r.random() < 0.5 and ((Y in self.cur.held['check']) == held) else pool[int(r.integers(len(pool)))]
                held = X in self.cur.held['check']
                ex = dict(stage='D', task='check', held_out=held, needs_scene=True, X=X, Y=Y, item=('check', X))
            else:
                dom = [o for o in vis if (TASKS[task][2] != 'food' or o in FOODS) and o in OBJECTS]
                cand = [o for o in dom if (o in self.cur.held[task]) == held] or dom
                if not cand:
                    continue
                X = cand[int(r.integers(len(cand)))]
                ex = dict(stage=TASKS[task][0], task=task, held_out=X in self.cur.held[task], needs_scene=True, X=X,
                          item=(task, X))
            self._fill(ex, self.scene)
            return ex
        return None

    def _topic_qa(self):
        r = self.rng
        n = int(r.integers(1, 4))
        for _ in range(n):
            ex = self._qa_now()
            if ex is None:
                return
            self._ask(ex)
            self._mention(ex.get('X'), ex.get('Y'))
            if _ < n - 1:
                self._free(int(r.integers(10, 40)))

    def _tell(self, key, item, text, task, prompt, acceptable, attend=None):
        """the partner states a fact (a statement; item counts the encounters)."""
        self._statement(dict(task=task + '_tell', text=text, item=item, attend=attend))
        f = self.facts.get(key)
        first = f['first_t'] if (f is not None and f['item'] == item) else self.t
        self.facts[key] = dict(task=task, prompt=prompt, acceptable=list(acceptable), item=item, told_t=self.t,
                               first_t=first, told_visit=self.visit_i)

    def _topic_fact(self):
        """a stable fact comes up again (or for the first time): a pet's name, a person's name, where something was
        put (objects have habitual places).  Facts recur at irregular intervals over the whole life."""
        r, lp = self.rng, self.lp
        u = r.random()
        if u < 0.35:
            X = PETS[int(r.integers(len(PETS)))]; N = self.pet_name[X]
            self._tell(('name', X), ('name', X), '我的' + X + '叫' + N, 'fact_name', '我的' + X + '叫什么',
                       [N, '叫' + N, '它叫' + N])
        elif u < 0.6:
            if self.present['other2'] and r.random() < 0.5:
                N = self.person_name['other2']
                self._tell(('pname', 'other2'), ('pname', 'other2'), '他叫' + N, 'fact_person', '他叫什么',
                           [N, '他叫' + N, '叫' + N])
            else:
                N = self.person_name['other']
                self._tell(('pname', 'other'), ('pname', 'other'), '我叫' + N, 'fact_person', '我叫什么',
                           [N, '你叫' + N])
        else:
            X = OBJECTS[int(r.integers(len(OBJECTS)))]
            P = self.home[X] if r.random() < lp['p_home'] else PLACES[int(r.integers(len(PLACES)))]
            self._tell(('place', X), ('place', X, P), '我把' + X + '放在' + P + '了', 'fact_place', X + '放在哪里了',
                       [P, '在' + P, '放在' + P])
            if X in self._visible():                      # put away: it leaves the scene
                s = [i for i, it in enumerate(self.scene) if it is not None and it[0] == X][0]
                sc = tuple(None if i == s else it for i, it in enumerate(self.scene))
                self.changes.append((self.t, s, X, None))
                self._set_scene(sc)
            self._mention(X)

    def _topic_novel(self):
        """a novel object: introduced once (这是X), and later named again when it is around (re-encounters)."""
        r = self.rng
        vis_old = [o for o in self._visible() if o in NOVEL and ('novel', o) in self.facts]
        if vis_old and r.random() < 0.5:
            X = vis_old[int(r.integers(len(vis_old)))]
        else:
            fresh = [o for o in NOVEL if o not in self.novel_used]
            if fresh:
                X = fresh[int(r.integers(len(fresh)))]
                self.novel_used.append(X)
            else:
                X = self.novel_used[int(r.integers(len(self.novel_used)))]
            if X not in self._visible():
                self._put_in_slot(X)
                self._free(int(r.integers(10, 30)))
        s = [i for i, it in enumerate(self.scene) if it is not None and it[0] == X][0]
        self._tell(('novel', X), ('novel', X), '这是' + X, 'novel', '这是什么', [X, '是' + X, '这是' + X], attend=s)
        self._mention(X)

    def _topic_memq(self):
        """a probe about a fact last told in an EARLIER visit (never answered by the tutor afterwards)."""
        r = self.rng
        vis = self._visible()
        keys = sorted(k for k, f in self.facts.items() if f['told_visit'] < self.visit_i
                      and (k != ('pname', 'other2') or self.present['other2'])
                      and (k[0] != 'place' or k[1] not in vis))
        if not keys:
            return self._topic_qa()
        types = sorted({k[0] for k in keys})                    # balance: a fact type first, then a fact of it
        ty = types[int(r.integers(len(types)))]
        keys = [k for k in keys if k[0] == ty]
        key = keys[int(r.integers(len(keys)))]
        f = self.facts[key]
        attend = None
        if key[0] == 'novel':
            if key[1] not in self._visible():
                self._put_in_slot(key[1])
                self._free(int(r.integers(10, 30)))
            attend = [i for i, it in enumerate(self.scene) if it is not None and it[0] == key[1]][0]
        ex = dict(stage='life', task=f['task'], held_out=False, needs_scene=key[0] == 'novel', prompt=f['prompt'],
                  acceptable=list(f['acceptable']), model=f['acceptable'][0], item=f['item'], attend=attend)
        rec = self._ask(ex, kind='probe')
        rec['ticks_since_told'] = rec['t_prompt_start'] - f['told_t']
        rec['ticks_since_first_told'] = rec['t_prompt_start'] - f['first_t']
        rec['visits_since_told'] = self.visit_i - f['told_visit']
        rec['one_shot'] = rec['n_experienced'] == 1

    def _topic_before(self):
        """刚才S是什么: S's object changed by itself since the previous alone period and the old object was never named by
        the world since it appeared (so the answer needs a memory of the scene, not of speech)."""
        r = self.rng
        cand = []
        for ci in range(len(self.changes) - 1, self._before_from - 1, -1):     # changes since the previous alone period
            t, s, old, new = self.changes[ci]
            if old is None or old not in OBJECTS:
                continue
            cur = self.scene[s][0] if self.scene[s] is not None else None
            if cur == old:
                continue
            if self.mentioned.get(old, -1) >= self.appeared.get(old, 0):
                continue
            if any(c[1] == s and c[0] > t for c in self.changes):      # only the latest change of that slot
                continue
            cand.append((t, s, old))
        if not cand:
            return self._topic_qa()
        t0, s, old = cand[int(r.integers(len(cand)))]
        ex = dict(stage='D', task='before', held_out=False, needs_scene=True, prompt='刚才' + SLOTS[s] + '是什么',
                  acceptable=[old, '是' + old], model=old, item=('before', old))
        rec = self._ask(ex, kind='probe')
        rec['ticks_since_told'] = rec['t_prompt_start'] - t0
        self._mention(old)

    def _topic_overheard(self):
        ex = self._qa_now()
        if ex is None or ex['held_out'] or ex['task'] == 'check':
            return
        self._overheard(ex)
        self._mention(ex.get('X'))

    def _topic_corpus(self):
        C = self._corpus
        r = self.rng
        if r.random() < self.p['e_probe']:
            q, a = C.test[int(r.integers(len(C.test)))]
            ex = dict(stage='E', task='corpus', held_out=True, needs_scene=False, prompt=q, acceptable=[a], model=a,
                      item=('corpus', q))
            self._ask(ex, kind='probe')
            return
        dlg = C.train[self.corpus_pos % len(C.train)]
        self.corpus_pos += 1
        q, a = dlg[0], dlg[1]
        ex = dict(stage='E', task='corpus', held_out=False, needs_scene=False, prompt=q, acceptable=[a], model=a,
                  item=('corpus', q))
        self._ask(ex, kind=self.tutor)

    # ================================================================ visits, alone, night, days
    def _visit(self, morning=False, last=False):
        r, lp = self.rng, self.lp
        self.visit_i += 1
        self.named_this_visit = set()
        other2 = r.random() < lp['p_other2']
        n_topics = int(r.integers(lp['visit_topics'][0], lp['visit_topics'][1] + 1))
        names = ['qa', 'fact', 'novel', 'memq', 'before', 'overheard'] + (['corpus'] if 'E' in lp['topics'] else [])
        w = np.array([lp['w_' + n] for n in names])
        if not other2:
            w[names.index('overheard')] = 0
        plan = [names[int(r.choice(len(names), p=w / w.sum()))] for _ in range(n_topics)]
        gaps = [int(np.exp(r.uniform(np.log(lp['gap_min']), np.log(lp['gap_max'])))) for _ in range(n_topics + 1)]
        ev_gap = [r.random() < lp['p_scene_event_gap'] for _ in range(n_topics)]
        self.present['other'] = True; self.present['other2'] = other2
        self._set_scene(self.scene)
        self.log.append(dict(type='event', t=self.t, what='arrive', other2=other2, visit=self.visit_i))
        self._greet_open, self._greeted = True, False
        self._free(lp['greet_wait'])
        self._greet_open = False
        if not self._greeted:
            if morning:
                self._social('早上好', ['早上好', '早'], 'greet')
            else:
                self._social('你好', ['你好', '你好呀'], 'greet')
        for i, topic in enumerate(plan):
            g = gaps[i] + self._pause_debt; self._pause_debt = 0
            if ev_gap[i]:
                self._free(g // 2); self._scene_event(); self._free(g - g // 2)
            else:
                self._free(g)
            getattr(self, '_topic_' + topic)()
        self._free(gaps[-1] + self._pause_debt); self._pause_debt = 0
        if last:
            self._social('晚安', ['晚安'], 'goodnight')
        else:
            self._social('再见', ['再见', '拜拜'], 'bye')
        self.present['other'] = False; self.present['other2'] = False
        self._set_scene(self.scene)
        self.log.append(dict(type='event', t=self.t, what='leave', visit=self.visit_i))

    def _alone(self, n, events):
        self._before_from = len(self.changes)        # 刚才 refers to changes since here (content-defined, not ticks)
        if not events:
            self._free(n)
            return
        cuts = sorted(events)
        t0 = 0
        for c in cuts:
            self._free(max(0, c - t0)); self._scene_event(); t0 = c
        self._free(max(0, n - t0))

    def _night(self):
        self.night = True
        self._set_scene(self.scene)
        self.log.append(dict(type='event', t=self.t, what='night_start', day=self.day_i))
        self._free(self.lp['night_ticks'])
        self.night = False
        if self.rng.random() < 0.5:                     # the room changes overnight
            self.scene = self._random_scene()
            for s, it in enumerate(self.scene):
                if it is not None:
                    self.appeared[it[0]] = self.t
        self._set_scene(self.scene)
        self.log.append(dict(type='event', t=self.t, what='night_end', day=self.day_i))

    def _day(self):
        r, lp = self.rng, self.lp
        self.day_i += 1
        n_vis = int(r.integers(lp['visits_per_day'][0], lp['visits_per_day'][1] + 1))
        plan = []
        for _ in range(n_vis - 1):
            if r.random() < lp['p_long_alone']:
                d = int(r.uniform(lp['alone_max'], lp['long_alone_max']))
            else:
                d = int(np.exp(r.uniform(np.log(lp['alone_min']), np.log(lp['alone_max']))))
            k = r.poisson(d / lp['scene_event_every'])
            plan.append((d, sorted(int(x) for x in r.integers(0, max(d, 1), k))))
        self.log.append(dict(type='event', t=self.t, what='day_start', day=self.day_i, visits=n_vis))
        self._visit(morning=True, last=n_vis == 1)
        for i, (d, evs) in enumerate(plan):
            self._alone(d, evs)
            self._visit(morning=False, last=i == len(plan) - 1)

    def run(self, agent, ticks=None, hours=None, verbose=0):
        """live for `ticks` (or `hours`) more ticks; stops at the end of a day or night that crosses the limit
        (days are planned in advance, so the content never depends on the agent).  May be called repeatedly."""
        if ticks is None:
            ticks = int((hours or 1.0) * TICKS_PER_HOUR)
        self.agent = agent
        self._hook('bench_attach', self)
        if self.t == 0:
            self._silence(self.p['lead_in'])
            self._set_scene(self.scene)
        t_end = self.t + ticks
        while self.t < t_end:
            self._day()
            if verbose:
                print(f'  day {self.day_i}: t={self.t}, visits {self.visit_i}', flush=True)
            if self.t >= t_end:
                break
            self._night()
        self.log.append(dict(type='mon', t=self.t, **{k: int(x) for k, x in self.mon.items()}))
        return self.log


def run_life(agent, hours=None, ticks=None, seed=0, tutor='model', **kw):
    L = Life(seed=seed, tutor=tutor, **kw)
    log = L.run(agent, ticks=ticks, hours=hours)
    return L, log


# ==================================================================================================== metrics
LIFE_FREE = ('probe', 'recast', 'social')
MEMORY_TASKS = ('fact_name', 'fact_person', 'fact_place', 'novel', 'before')
ENC_BINS = (0, 1, 2, 3, 5, 9, 10 ** 9)   # encounters: 0, 1, 2, 3-4, 5-8, 9+


def _bins_of(t, edges):
    return int(np.clip(np.searchsorted(edges, t, side='right') - 1, 0, len(edges) - 2))


def evaluate_life(log, bins=10, delay_bins=(0, 300, 1000, 3000, 10000, 30000, 100000, 10 ** 9)):
    """in-life metrics as functions of life time (no frozen copy): per-task accuracy curves, retention vs delay,
    initiative, overlaps / interruptions, longest monologue, latency, stability."""
    from metrics import evaluate
    ex = [r for r in log if r.get('type') == 'ex']
    free = [r for r in ex if r['kind'] in LIFE_FREE and r.get('acceptable')]
    inits = [r for r in log if r.get('type') == 'init']
    mons = [r for r in log if r.get('type') == 'mon']
    t_max = max([r['t'] for r in mons] + [r.get('t_end', 0) for r in ex] + [1])
    edges = np.linspace(0, t_max, bins + 1)
    out = dict(ticks=int(t_max), hours=t_max / TICKS_PER_HOUR, n_free=len(free), n_init=len(inits))
    # ---- per task: overall (staged evaluate, life kinds) + learning curve
    out['tasks'] = {k: v for k, v in evaluate(free, kinds=LIFE_FREE).get('ALL', {}).items()}
    by_task = collections.defaultdict(list)
    for r in free:
        by_task[r['task']].append(r)
    curves = {}
    for task, rs in sorted(by_task.items()):
        c = [[0, 0] for _ in range(bins)]
        for r in rs:
            b = _bins_of(r['t_prompt_start'], edges)
            c[b][0] += 1; c[b][1] += r['reply'] in r['acceptable']
        curves[task] = [dict(t=float((edges[i] + edges[i + 1]) / 2), n=n, acc=(k / n if n else None))
                        for i, (n, k) in enumerate(c)]
    out['curves'] = curves
    groups = {'scene': [r for r in free if r.get('needs_scene') and r.get('new_situation', True)],
              'scene_repeat': [r for r in free if r.get('needs_scene') and not r.get('new_situation', True)],
              'held_out': [r for r in free if r.get('held_out')],
              'memory': [r for r in free if r['task'] in MEMORY_TASKS], 'social': [r for r in free if r['kind'] == 'social']}
    out['groups'] = {}
    for g, rs in groups.items():
        c = [[0, 0] for _ in range(bins)]
        for r in rs:
            b = _bins_of(r['t_prompt_start'], edges); c[b][0] += 1; c[b][1] += r['reply'] in r['acceptable']
        out['groups'][g] = dict(n=len(rs), acc=(np.mean([r['reply'] in r['acceptable'] for r in rs]) if rs else None),
                                curve=[(k / n if n else None) for n, k in c])
    # ---- retention vs delay AND vs number of encounters (the key memory metric)
    def _acc(sel):
        return float(np.mean([r['reply'] in r['acceptable'] for r in sel])) if sel else None

    def _lab(a, b):
        return f'{a}' if b == a + 1 else (f'{a}+' if b > 10 ** 8 else f'{a}-{b - 1}')
    ret = {}
    for task in MEMORY_TASKS:
        rs = [r for r in free if r['task'] == task and r.get('ticks_since_told') is not None]
        by_d, by_e, grid = [], [], []
        for a, b in zip(delay_bins, delay_bins[1:]):
            sel = [r for r in rs if a <= r['ticks_since_told'] < b]
            if sel:
                by_d.append(dict(delay=f'{a}-{b}', n=len(sel), acc=_acc(sel)))
            for c, d in zip(ENC_BINS, ENC_BINS[1:]):
                sel2 = [r for r in sel if c <= r.get('n_experienced', 0) < d]
                if sel2:
                    grid.append(dict(delay=f'{a}-{b}', encounters=_lab(c, d), n=len(sel2), acc=_acc(sel2)))
        for c, d in zip(ENC_BINS, ENC_BINS[1:]):
            sel = [r for r in rs if c <= r.get('n_experienced', 0) < d]
            if sel:
                by_e.append(dict(encounters=_lab(c, d), n=len(sel), acc=_acc(sel),
                                 median_delay=float(np.median([r['ticks_since_told'] for r in sel]))))
        ret[task] = dict(n=len(rs), acc=_acc(rs), by_delay=by_d, by_encounters=by_e, delay_x_encounters=grid)
    # B-D items: accuracy against how often the answer was heard before (learning by re-encounter)
    qa = [r for r in free if r['task'] not in MEMORY_TASKS and r['kind'] != 'social']
    ret['qa_by_encounters'] = [dict(encounters=_lab(c, d), n=len(sel), acc=_acc(sel)) for c, d in zip(ENC_BINS, ENC_BINS[1:])
                               for sel in [[r for r in qa if c <= r.get('n_experienced', 0) < d]] if sel]
    out['retention'] = ret
    # ---- initiative, overlaps, monologue
    last = mons[-1] if mons else {}
    quiet = max(last.get('quiet_ticks', 0), 1)
    cats = collections.Counter(r['category'] for r in inits)
    out['initiative'] = dict(n=len(inits), per_1000_quiet_ticks=1000 * len(inits) / quiet,
                             appropriate_share=(float(np.mean([r['appropriate'] for r in inits])) if inits else None),
                             appropriate_share_partner_present=(float(np.mean([r['appropriate'] for r in inits if r['partner']]))
                                                                if any(r['partner'] for r in inits) else None),
                             reward=float(sum(r['reward'] for r in inits)), categories=dict(cats))
    out['overlap'] = dict(overlap_ticks=last.get('overlap_ticks', 0), interruptions=last.get('interruptions', 0),
                          partner_ticks=last.get('partner_ticks', 0),
                          overlap_share_of_partner_speech=last.get('overlap_ticks', 0) / max(last.get('partner_ticks', 0), 1))
    out['monologue'] = dict(longest_syllables=last.get('longest_syl', 0), longest_ticks=last.get('longest_ticks', 0),
                            at_tick=last.get('longest_syl_t'))
    lat = collections.Counter(r['latency'] for r in free if r.get('latency') is not None)
    out['latency'] = dict(hist=dict(sorted(lat.items())), median=(float(np.median(list(lat.elements()))) if lat else None))
    out['mouth'] = dict(mouth_ticks=last.get('mouth_ticks', 0), syllables=last.get('syllables', 0),
                        garble=last.get('garble', 0), bouts=last.get('bouts', 0), quiet_ticks=last.get('quiet_ticks', 0))
    # ---- stability per time bin (from monitor snapshots + records)
    stab = []
    prev = {}
    snaps = sorted(mons, key=lambda r: r['t'])
    for i in range(bins):
        a, b = edges[i], edges[i + 1]
        fr = [r for r in free if a <= r['t_prompt_start'] < b]
        it = [r for r in inits if a <= r['t'] < b]
        cur = next((s for s in snaps if s['t'] >= b), snaps[-1] if snaps else {})
        d = {k: cur.get(k, 0) - prev.get(k, 0) for k in ('quiet_ticks', 'mouth_ticks', 'overlap_ticks', 'partner_ticks',
                                                      'syllables', 'garble')}
        prev = cur
        stab.append(dict(t=float((a + b) / 2), free=len(fr), spoke=(float(np.mean([bool(r['reply']) for r in fr])) if fr else None),
                         acc=(float(np.mean([r['reply'] in r['acceptable'] for r in fr])) if fr else None),
                         mean_len=(float(np.mean([len(r['reply']) for r in fr if r['reply']])) if any(r['reply'] for r in fr) else None),
                         init_per_1000_quiet=1000 * len(it) / max(d['quiet_ticks'], 1),
                         mouth_share=d['mouth_ticks'] / max(b - a, 1), overlap_ticks=d['overlap_ticks'],
                         garble_share=d['garble'] / max(d['syllables'], 1)))
    out['stability'] = stab
    sp = [s['spoke'] for s in stab if s['spoke'] is not None]
    flags = []
    if sp and max(sp) > 0.2 and sp[-1] < 0.5 * max(sp):
        flags.append('answer rate collapsed (last bin < half of the best)')
    if stab and stab[-1]['mouth_share'] > 0.5:
        flags.append('mouth active > 50 % of the last bin (runaway)')
    if stab and stab[-1]['overlap_ticks'] > 0.2 * max(1, sum(s['overlap_ticks'] for s in stab)) and len(stab) > 3 \
            and stab[-1]['overlap_ticks'] > 2 * np.mean([s['overlap_ticks'] for s in stab[:-1]]) + 10:
        flags.append('overlap rising at the end')
    out['stability_flags'] = flags
    return out


def headline_life(res):
    f = lambda x: '-' if x is None else f'{x:.2f}'   # noqa: E731
    g = res['groups']
    i = res['initiative']
    return (f"life {res['hours']:.1f} h ({res['ticks']} ticks): free replies {res['n_free']} acc "
            f"{f(res['tasks'].get('ALL', {}).get('acc'))} | scene {f(g['scene']['acc'])} held-out {f(g['held_out']['acc'])} memory "
            f"{f(g['memory']['acc'])} social {f(g['social']['acc'])} | initiative {i['n']} "
            f"({i['per_1000_quiet_ticks']:.2f}/1000 quiet ticks, appropriate {f(i['appropriate_share'])}, reward "
            f"{i['reward']:.0f}) | overlap {res['overlap']['overlap_ticks']} ticks, {res['overlap']['interruptions']} "
            f"interruptions | longest monologue {res['monologue']['longest_syllables']} syl | latency median "
            f"{res['latency']['median']} | flags {res['stability_flags'] or 'none'}")


# ==================================================================================================== reference agents
from baselines import BASELINES, Scripted  # noqa: E402


class RandomNamer(Scripted):
    """reference: after every `every` silent ticks (nothing in the ear), say the name of a random visible object
    (privileged: it is told the scene).  It never answers questions on purpose."""

    def __init__(self, every=200, seed=0):
        super().__init__()
        self.every, self.rng = every, np.random.default_rng(seed)
        self.quiet = 0
        self.visible = []

    def bench_scene(self, info):
        sc = info['scene']
        self.visible = [] if info['night'] or sc is None else [it[0] for it in sc if it is not None]

    def bench_prompt(self, info):
        pass

    def step(self, obs):
        if self.plan:
            return self.plan.popleft()
        if obs['ear'].any():
            self.quiet = 0
        else:
            self.quiet += 1
        if self.quiet >= self.every and self.visible:
            self.quiet = 0
            self.say(self.visible[int(self.rng.integers(len(self.visible)))], delay=0)
            return self.plan.popleft()
        return self.zero


def life_baseline(name):
    """a staged baseline that also answers the social exchanges of life (greetings, goodbyes)."""
    a = BASELINES[name]()
    a.answer_kinds = ('probe', 'recast', 'social')
    return a
