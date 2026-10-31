# -*- coding: utf-8 -*-
"""The ARCH_v2 bench world: a staged, grounded micro-world with a tutor, on the real model's sound/mouth physics.

    from world import World
    w = World('B', seed=0, tutor='model')
    log = w.run(agent, n_dialogues=2000)          # agent.step(obs) -> (32,) muscles, > 0 = active
    from metrics import evaluate; print(evaluate(log))

See README.md for the full contract as implemented (API.md + clarifications).
"""
from __future__ import annotations

import collections

import numpy as np

from physics import PAUSE, SOUND, Stream, Voice, corpus_chars, load_corpus

# ------------------------------------------------------------------------------------------ lexicon (all in corpus)
OBJECTS = ['苹果', '香蕉', '饼干', '牛奶', '西瓜', '葡萄', '小狗', '小猫', '皮球', '汽车', '小鸟', '鸭子']
FOODS = OBJECTS[:6]
COLOURS = ['红色', '黄色', '蓝色', '绿色', '白色', '黑色']
COUNTS = ['一个', '两个', '三个']
SLOTS = ['左边', '中间', '右边']
NOVEL = ['熊猫', '兔子', '气球', '积木', '蛋糕', '面包', '帽子', '杯子', '书包', '飞机', '火车', '雨伞', '鞋子', '袜子',
         '椅子', '桌子', '手机', '电脑', '眼镜', '钥匙', '铅笔', '衣服', '裙子', '星星', '月亮', '太阳', '花朵', '草莓',
         '橘子', '柠檬', '番茄', '土豆', '老虎', '狮子', '猴子', '大象', '青蛙', '乌龟', '蝴蝶', '蜜蜂']
NAMES = ['小明', '小红', '小华', '小刚', '小丽', '小军', '小美', '小强']
RITUALS = [('你好', ['你好', '你好呀']), ('谢谢', ['不客气', '不谢']), ('晚安', ['晚安']), ('再见', ['再见', '拜拜']),
           ('早上好', ['早上好', '早'])]

LEXICON_CHARS = set(''.join(OBJECTS + COLOURS + COUNTS + SLOTS + NOVEL + NAMES + [q for q, _ in RITUALS] +
                            [a for _, acc in RITUALS for a in acc]) + '这是什么在哪里你要吗我好不吃颜色有几个刚才叫看宝的对')

# task -> (stage, needs_scene, filler domain or None)
TASKS = {
    'what': ('B', True, 'obj'),       # 这是什么 (pointing) -> X
    'slot': ('B', True, 'obj'),       # 左边是什么 -> X
    'where': ('B', True, 'obj'),      # X在哪里 -> 在左边
    'offer': ('C', False, 'obj'),     # 你要X吗 (pointing) -> 要X / 我要X
    'taste': ('C', False, 'food'),    # X好不好吃 (pointing) -> 好吃
    'colour': ('C', True, 'obj'),     # X是什么颜色 -> 红色
    'count': ('C', True, 'obj'),      # 有几个X -> 两个
    'ritual': ('C', False, None),     # 你好 -> 你好 ...
    'check': ('D', True, 'obj'),      # 这是X吗 (pointing at Y) -> 是 / 不是
    'name': ('D', False, None),       # 我叫N ... 我叫什么 -> N
    'before': ('D', True, None),      # (the slot's object changes) 刚才左边是什么 -> the old object
    'oneshot': ('D', True, None),     # 这是X (novel object, once) ... 这是什么 -> X
    'corpus': ('E', False, None),     # adult corpus pair
    'corpus_fam': ('E', False, None),  # adult corpus pair from the familiar set (taught repeatedly)
}
STAGE_TASKS = {
    'B': {'what': .4, 'slot': .3, 'where': .3},
    'C': {'offer': .25, 'taste': .15, 'colour': .2, 'count': .2, 'ritual': .2},
    'D': {'check': .4, 'name': .2, 'before': .2, 'oneshot': .2},
}
QA_TASKS = ('what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'ritual', 'check')
SCENE_TASKS = {t for t, v in TASKS.items() if v[1]}

DEFAULTS = dict(
    probe_frac=0.15,       # seen items asked WITHOUT any help (scored as free replies)
    held_frac=0.15,        # exchanges that probe a held-out (task, filler) pair (always a probe; never taught)
    held_ratio=0.3,        # fraction of (task, filler) pairs held out
    probe_frac_mem=0.4,    # stage D memory questions (name, before) asked as probes; one-shot questions are always probes
    gap_answer=45,         # >= 45 ticks from the end of a line that wants an answer to the next line, if the agent is silent
    after_stop_test=12,    # free replies: ticks kept after the agent stops (ARCH_v1 free(): 12)  -> extra turns visible
    pause_after=5,         # taught replies: silent ticks after the agent's stop before the next line (PAUSE >= 5)
    reply_cap=60,          # the agent's answer window ends at most this many ticks after the prompt (ARCH_v1: 60)
    gap_model=3,           # model mode: ticks from the prompt's end to the model reply (3 blocks a v1 body's GO)
    model_wait=12,         # after a model reply / statement / recast: ticks for the agent to start (else next line)
    gap_dialogue=10,       # silent ticks between dialogues (scene changes inside)
    scene_lead=5,          # the new scene appears this many ticks before the first line of a dialogue
    gap_change=10,         # stage D 'before': silent ticks around a scene change
    force_timeout=45,      # force mode: the teacher gives up after this many ticks without an onset
    lead_in=10,            # silent ticks at the start of run()
    n_eye=1024, eye_k=8, eye_shared=4,
    eye_layout='random',   # brainlike copy: 'retino' = innate retinotopy (per-slot blocks, same feature code per slot)
       # eye: units; units per (slot, value) code; of these, shared across slots
    p_empty=0.15,          # an unrequired slot is empty
    ex_per_dialogue=(1, 3),
    other_models=0.3,      # model mode: share of model replies said by the partner itself (else by other2)
    reward_ok=1.0, reward_wrong=0.0, reward_imitate=0.2,
    # stage A
    a_weights=(0.4, 0.3, 0.3),   # cds, overheard, play
    play_ticks=60,         # one free-play period
    imitate_gap=2,         # the tutor imitates a babble bout this many ticks after it ended (bout end = 2 silent ticks)
    imitate_max=3,         # at most this many syllables imitated
    # stage E
    e_corpus=0.5,          # share of E dialogues taken from the adult corpus (the rest: B, C, D dialogues)
    e_probe=0.3,           # per corpus dialogue: probability of an extra probe with a never-heard corpus pair
    e_fam=0.1,             # per corpus dialogue: probability of a taught pair from the familiar set
    e_fam_probe=0.1,       # per corpus dialogue: probability of a probe of a familiar-set pair
    corpus_start=0,        # first training dialogue of the corpus (to continue a curriculum)
    n_test_pairs=500, n_fam_pairs=40,
)


_VOICES = {}


def _voice(chars, seed, physics):
    """a fresh Voice (deep copy of a cached pristine one: identical to Voice(chars, seed, physics), 15x faster)."""
    import copy
    key = (chars, seed, physics)
    if key not in _VOICES:
        if len(_VOICES) > 8:
            _VOICES.clear()
        _VOICES[key] = Voice(list(chars), seed=seed, physics=physics)
    return copy.deepcopy(_VOICES[key])


def _rng(seed, *salt):
    return np.random.default_rng([int(seed)] + [int(s) for s in salt])


# ------------------------------------------------------------------------------------------ curriculum (seed only)
class Curriculum:
    """everything that depends on the world seed only: held-out pairs, eye codes.  Same for every stage."""

    def __init__(self, seed, held_ratio=0.3, n_eye=1024, eye_k=8, eye_shared=4, eye_layout='random'):
        r = _rng(seed, 11)
        self.held = {}
        for group in ('B', 'C', 'D'):
            tasks = [t for t in STAGE_TASKS[group] if TASKS[t][2] is not None]
            cnt = collections.Counter()
            for t in tasks:
                dom = FOODS if TASKS[t][2] == 'food' else OBJECTS
                n = max(1, int(round(held_ratio * len(dom))))
                order = sorted(dom, key=lambda x: (cnt[x], r.random()))     # least-held fillers first
                self.held[t] = set(order[:n])
                for x in order[:n]:
                    cnt[x] += 1
        self.train = {t: [x for x in (FOODS if TASKS[t][2] == 'food' else OBJECTS) if x not in self.held[t]]
                      for t in self.held}
        # eye codes: per (slot, value) a sparse code = a part shared by the value across slots + a slot-specific part
        re = _rng(seed, 12)
        self.n_eye = n_eye
        vals = [('obj', o) for o in OBJECTS + NOVEL] + [('col', c) for c in COLOURS] + [('cnt', n) for n in COUNTS]
        shared = {v: re.choice(n_eye, eye_shared, replace=False) for v in vals}
        self.code = {}
        for s in range(3):
            for v in vals:
                spec = re.choice(n_eye, eye_k - eye_shared, replace=False)
                self.code[(s, v)] = np.union1d(shared[v], spec)
        self.attend_code = [re.choice(n_eye, eye_k, replace=False) for _ in range(3)]
        if eye_layout == 'retino':
            # innate retinotopy (brainlike copy only): slot s owns units [s*240, (s+1)*240); inside a block the same
            # feature maps are replicated: shape map 0..159 (8 of 160 per object), colour 160..207, number 208..231,
            # pointing 232..239.  Same held-out sets and scenes; only the eye code layout differs.
            rr = _rng(seed, 13)
            mp = {'obj': (0, 160), 'col': (160, 48), 'cnt': (208, 24)}
            base = {v: mp[v[0]][0] + rr.choice(mp[v[0]][1], eye_k, replace=False) for v in vals}
            self.code = {(s, v): s * 240 + base[v] for s in range(3) for v in vals}
            self.attend_code = [s * 240 + np.arange(232, 240) for s in range(3)]
        self._eye_cache = {}

    def eye(self, scene, attend):
        key = (scene, attend)
        e = self._eye_cache.get(key)
        if e is None:
            e = np.zeros(self.n_eye, np.uint8)
            if scene is not None:
                for s, it in enumerate(scene):
                    if it is None:
                        continue
                    o, c, n = it
                    for v in (('obj', o), ('col', COLOURS[c]), ('cnt', COUNTS[n])):
                        e[self.code[(s, v)]] = 1
                if attend is not None:
                    e[self.attend_code[attend]] = 1
            e.setflags(write=False)
            if len(self._eye_cache) > 200000:
                self._eye_cache.clear()
            self._eye_cache[key] = e
        return e


class CorpusSets:
    """stage E: training dialogues = corpus[:-3000] in order; test pairs from corpus[-3000:] whose prompt never occurs
    in the training part (never heard); a familiar set (taught repeatedly) from the same pool."""
    _cache = {}

    def __new__(cls, seed, n_test, n_fam):
        key = (seed, n_test, n_fam)
        if key not in cls._cache:
            obj = super().__new__(cls)
            pool = load_corpus()
            obj.train = pool[:-3000]
            train_q = {c[i] for c in obj.train for i in range(0, len(c) - 1, 2)}
            cand = [(c[i], c[i + 1]) for c in pool[-3000:] for i in range(0, len(c) - 1, 2) if c[i] not in train_q]
            perm = _rng(seed, 31).permutation(len(cand))
            obj.test = [cand[i] for i in perm[:n_test]]
            obj.fam = [cand[i] for i in perm[n_test:n_test + n_fam]]
            cls._cache[key] = obj
        return cls._cache[key]


# ------------------------------------------------------------------------------------------ exchange tracker
class Track:
    """mouth activity in a segment: syllables (one per onset), run lengths, bouts, overlap with partner speech."""
    __slots__ = ('chars', 'runs', 'cur', 'first', 'last', 'silent', 'bouts', 'garble', 'overlap', 'stop', 'ticks',
                 'agent_stop', 'trunc')

    def __init__(self):
        self.chars, self.runs = [], []
        self.cur = 0; self.first = None; self.last = None; self.silent = 0; self.bouts = 0
        self.garble = 0; self.overlap = 0; self.stop = None; self.ticks = 0; self.agent_stop = None
        self.trunc = False                 # the current syllable began before this window (its length is unknown)

    def tick(self, t, said, onset, partner, ev=None):
        self.ticks += 1
        if ev is not None and self.first is not None and self.agent_stop is None:
            self.agent_stop = ev
        if said is not None:
            if partner:
                self.overlap += 1
            if self.first is None:
                self.first = t; self.bouts = 1
            elif self.silent >= PAUSE:
                self.bouts += 1
            if onset or self.cur == 0:
                if self.cur and not self.trunc:
                    self.runs.append(self.cur)
                self.trunc = self.ticks == 1 and not onset
                self.chars.append(said); self.cur = 1
                if said == '~':
                    self.garble += 1
            else:
                self.cur += 1
            self.last = t; self.silent = 0
        else:
            if self.cur:
                if not self.trunc:
                    self.runs.append(self.cur)
                self.cur = 0; self.trunc = False
                if self.stop is None:
                    self.stop = t
            self.silent += 1

    def close(self):
        """a syllable still sounding when the window closes is not counted in runs (its length is unknown)."""
        self.cur = 0; self.trunc = False

    @property
    def text(self):
        return ''.join(self.chars)


# ------------------------------------------------------------------------------------------ the world
class World:
    def __init__(self, stage='B', seed=0, physics='random', noise=0.0, tutor='model', vocab=None, prev=None,
                 **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise TypeError(f'unknown World params {sorted(unknown)}')
        self.p = dict(DEFAULTS); self.p.update(params)
        if tutor not in ('model', 'force', 'recast'):
            raise ValueError(tutor)
        self.stage, self.seed, self.physics, self.noise, self.tutor = stage, int(seed), physics, float(noise), tutor
        stages = [s for s in stage if s in 'ABCDE']
        if not stages or len(stages) != len(stage):
            raise ValueError(f'stage must be letters from ABCDE, got {stage!r}')
        self.stages = stages
        chars = corpus_chars() if vocab is None else sorted(set(vocab))
        self.voice = _voice(tuple(chars), self.seed, physics)
        self.cur = Curriculum(self.seed, self.p['held_ratio'], self.p['n_eye'], self.p['eye_k'], self.p['eye_shared'],
                              self.p['eye_layout'])
        self.need_vec = {}
        self.stream = Stream(self.voice.F, self.voice.T, self.voice.T // SOUND, 3, self.voice.levels, noise,
                             noise_seed=self.seed * 7919 + 13)
        self.dlg_offset = (prev.dlg_offset + prev.n_dialogues) if prev is not None else 0
        self.rng = _rng(self.seed, 21, *[ord(s) for s in stage], {'model': 1, 'force': 2, 'recast': 3}[tutor],
                        *([self.dlg_offset] if self.dlg_offset else []))
        self.log = []
        self.t = 0
        self.scene = None; self.attend = None
        self.eye = self.cur.eye(None, None)
        self._own = (None, None, False)
        self._pact = (); self._pmax = 0.0
        self.exposures = collections.Counter()
        self.last_taught = {}
        self.novel_used = []                                   # novel objects introduced so far (stage D one-shot)
        self.corpus_pos = self.p['corpus_start']
        self.n_dialogues = 0
        if prev is not None:                                   # continue a curriculum: keep the clock and the history
            self.t = prev.t
            self.exposures = prev.exposures.copy()
            self.last_taught = dict(prev.last_taught)
            self.novel_used = list(prev.novel_used)
            self.corpus_pos = max(self.corpus_pos, prev.corpus_pos)
        self._corpus = CorpusSets(self.seed, self.p['n_test_pairs'], self.p['n_fam_pairs']) if 'E' in stages else None
        self.trk = []                                          # active trackers (list of Track)
        self.agent = None
        self.timing_violations = collections.Counter()
        self._refs = {}

    # ---------------------------------------------------------------- low level: one tick
    def cue_vec(self, c):
        v = self.need_vec.get(c)
        if v is None:
            v = np.zeros(self.voice.n_muscles, np.float32)
            v[list(self.voice.need[c])] = 1.0
            v.setflags(write=False)
            self.need_vec[c] = v
        return v

    def _tick(self, other=None, other2=None, cue=None, reward=0.0):
        """other / other2: None or (char, fresh).  Returns (said, onset) of the agent's mouth at this tick."""
        v = self.voice
        s1 = (other[0], v.hear(other[0], 'other'), other[1]) if other is not None else (None, None, False)
        s2 = (other2[0], v.hear(other2[0], 'other2'), other2[1]) if other2 is not None else (None, None, False)
        tok, g, fresh = self._own
        if tok == '~':
            g = v.hear('~', 'self')                            # garble: fresh noise every tick (as voice.py)
        win = self.stream.tick((s1, s2, (tok, g, fresh)))
        obs = {'ear': win, 'eye': self.eye, 'reward': float(reward), 'cue': cue}
        m = self.agent.step(obs)
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
        partner = other is not None or other2 is not None
        ev = getattr(self.agent, 'tick_event', None)
        for tr in self.trk:
            tr.tick(self.t, said, onset, partner, ev)
        self.t += 1
        return said, onset

    def _silence(self, n, **kw):
        for _ in range(n):
            self._tick(**kw)

    def _say(self, who, text, reward_first=0.0):
        first = True
        for ch in text:
            for k in range(SOUND):
                src = (ch, k == 0)
                self._tick(other=src if who == 'other' else None, other2=src if who == 'other2' else None,
                           reward=reward_first if first else 0.0)
                first = False

    def _set_scene(self, scene, attend=None):
        self.scene, self.attend = scene, attend
        self.eye = self.cur.eye(scene, attend)

    # ---------------------------------------------------------------- waiting for the agent
    def _wait(self, tr, min_gap, after_stop, cap):
        """run silent ticks until: agent never spoke and min_gap ticks passed; or it spoke, then stayed silent for
        after_stop+1 ticks and min_gap passed; or cap ticks passed (agent cut)."""
        t0 = self.t
        cut = False
        while True:
            self._tick()
            el = self.t - t0
            if tr.first is None:
                if el >= min_gap:
                    break
            elif tr.silent >= after_stop + 1 and el >= min_gap:
                break
            if el >= cap:
                cut = tr.first is not None and tr.silent == 0
                break
        return cut

    # ---------------------------------------------------------------- privileged hooks (baselines only)
    def _hook(self, name, *a):
        f = getattr(self.agent, name, None) if getattr(self.agent, 'privileged', False) else None
        if f is not None:
            f(*a)

    # ================================================================ dialogue generation (content rng only)
    def _pick_filler(self, task, held, avoid):
        dom = self.cur.held[task] if held else self.cur.train[task]
        cand = [x for x in sorted(dom) if x not in avoid]
        if not cand:
            return None
        return cand[int(self.rng.integers(len(cand)))]

    def _rand_item(self, o):
        r = self.rng
        return (o, int(r.integers(len(COLOURS))), int(r.integers(len(COUNTS))))

    def _scene_with(self, req, slots=None):
        """a scene with each required object (or (object, colour, count) item) in its own slot; the rest random."""
        r, p = self.rng, self.p
        slots = list(r.permutation(3)) if slots is None else slots
        scene = [None, None, None]
        names = []
        for x, s in zip(req, slots):
            it = x if isinstance(x, tuple) else self._rand_item(x)
            scene[s] = it; names.append(it[0])
        rest = [o for o in OBJECTS if o not in names]
        for s in [s for s in range(3) if scene[s] is None]:
            if r.random() >= p['p_empty']:
                scene[s] = self._rand_item(rest.pop(int(r.integers(len(rest)))))
        return tuple(scene)

    def _qa_exchanges(self, stage, n_ex, used):
        """1..n filler-task exchanges of a stage (B, C or D-check); returns exchanges with 'X' set (scene pending)."""
        r, p = self.rng, self.p
        tw = STAGE_TASKS[stage] if stage != 'D' else {'check': 1.0}
        tasks, w = list(tw), np.array(list(tw.values()))
        exs = []
        for _ in range(n_ex):
            task = tasks[int(r.choice(len(tasks), p=w / w.sum()))]
            ex = dict(stage=TASKS[task][0], task=task, held_out=False, needs_scene=TASKS[task][1])
            if TASKS[task][2] is not None:
                held = r.random() < p['held_frac']
                X = self._pick_filler(task, held, used)
                if X is None:
                    held = not held
                    X = self._pick_filler(task, held, used)
                if X is None:
                    continue
                ex.update(X=X, held_out=held, item=(task, X))
                if task == 'check':                           # pointing at Y; Y == X half of the time
                    if r.random() < 0.5:
                        ex['Y'] = X
                    else:
                        cand = [o for o in OBJECTS if o != X and o not in used]
                        ex['Y'] = cand[int(r.integers(len(cand)))]
                    used.add(ex['Y'])
                used.add(X)
            else:
                q, acc = RITUALS[int(r.integers(len(RITUALS)))]
                ex.update(prompt=q, acceptable=list(acc), model=acc[0], item=(task, q))
            exs.append(ex)
        return exs

    def _kind(self, ex, probe_p=None):
        r = self.rng
        pp = self.p['probe_frac'] if probe_p is None else probe_p
        ex['kind'] = 'probe' if ex['held_out'] or r.random() < pp else self.tutor
        if ex['kind'] == 'model':
            ex['model_by'] = 'other' if r.random() < self.p['other_models'] else 'other2'
        return ex

    def _dlg_qa(self, stage):
        r, p = self.rng, self.p
        n_ex = int(r.integers(p['ex_per_dialogue'][0], p['ex_per_dialogue'][1] + 1))
        used = set()
        exs = self._qa_exchanges(stage, n_ex, used)
        req = []
        for ex in exs:
            if 'X' in ex:
                req.append(ex['Y'] if ex['task'] == 'check' else ex['X'])
        req = list(dict.fromkeys(req))[:3]
        scene = self._scene_with(req)
        events = [('scene', scene)]
        for ex in exs:
            if 'X' in ex:
                self._fill(ex, scene)
            events.append(('ex', self._kind(ex)))
        return dict(stage=stage, events=events)

    def _fill(self, ex, scene):
        """prompt, acceptable replies, the model reply and the attended slot of a filler task, from the scene."""
        r, task, X = self.rng, ex['task'], ex['X']
        where = {it[0]: s for s, it in enumerate(scene) if it is not None}
        att = None
        if task == 'check':
            Y = ex['Y']; s = where[Y]; att = s
            prompt = '这是' + X + '吗'
            acc = ['是', '是的', '对'] if X == Y else ['不是', '不对']
            ex.update(prompt=prompt, acceptable=acc, model=acc[0], attend=att)
            return
        s = where[X]
        S = SLOTS[s]
        o, c, n = scene[s]
        if task == 'what':
            prompt, acc, att = '这是什么', [X, '是' + X, '这是' + X], s
            model = X if r.random() < .7 else '这是' + X
        elif task == 'slot':
            prompt, acc = S + '是什么', [X, '是' + X]
            model = X
        elif task == 'where':
            prompt, acc = X + '在哪里', ['在' + S, S]
            model = '在' + S if r.random() < .8 else S
        elif task == 'offer':
            prompt, acc, att = '你要' + X + '吗', ['要' + X, '我要' + X, '要'], s
            model = '要' + X if r.random() < .6 else '我要' + X
        elif task == 'taste':
            prompt, acc, att = X + '好不好吃', ['好吃', X + '好吃', '很好吃'], s
            model = '好吃' if r.random() < .7 else X + '好吃'
        elif task == 'colour':
            C = COLOURS[c]
            prompt, acc = X + '是什么颜色', [C, '是' + C, C[0] + '的']
            model = C
        elif task == 'count':
            N = COUNTS[n]
            prompt, acc = '有几个' + X, [N, '有' + N]
            model = N
        else:
            raise ValueError(task)
        ex.update(prompt=prompt, acceptable=acc, model=model, attend=att)

    def _qa_about(self, X, scene, tasks=('what', 'slot', 'where')):
        """a B question about object X (present in the scene); held-out pairs become probes as everywhere."""
        task = tasks[int(self.rng.integers(len(tasks)))]
        ex = dict(stage='B', task=task, held_out=X in self.cur.held[task], needs_scene=True, X=X, item=(task, X))
        self._fill(ex, scene)
        return ex

    # ---------------- stage D routines
    def _dlg_D(self):
        r, p = self.rng, self.p
        tw = STAGE_TASKS['D']
        tasks, w = list(tw), np.array(list(tw.values()))
        task = tasks[int(r.choice(len(tasks), p=w / w.sum()))]
        if task == 'check':
            return self._dlg_qa('D')
        if task == 'name':
            N = NAMES[int(r.integers(len(NAMES)))]
            fill = self._qa_exchanges('BC'[int(r.integers(2))], int(r.integers(1, 3)), set())
            scene = self._scene_with(list(dict.fromkeys(ex['X'] for ex in fill if 'X' in ex))[:3])
            events = [('scene', scene), ('stmt', dict(task='name_intro', text='我叫' + N, ref='name', item=('name', N)))]
            for ex in fill:
                if 'X' in ex:
                    self._fill(ex, scene)
                events.append(('ex', self._kind(ex)))
            q = dict(stage='D', task='name', held_out=False, needs_scene=False, prompt='我叫什么',
                     acceptable=[N, '你叫' + N], model=N, item=('name', N), ref='name')
            events.append(('ex', self._kind(q, p['probe_frac_mem'])))
            return dict(stage='D', events=events)
        if task == 'before':
            # the changed slot is never named before the change (the old object was only SEEN); fillers ask about the
            # other slots; with p=0.5 the NEW object is named right before the question, so repeating the last heard
            # answer is wrong.  The answer needs a memory of the scene.
            s = int(r.integers(3))
            scene = self._scene_with([OBJECTS[int(r.integers(len(OBJECTS)))]], slots=[s])
            old = scene[s][0]
            S = SLOTS[s]
            events = [('scene', scene)]
            others = [it[0] for i, it in enumerate(scene) if i != s and it is not None]
            for X in list(r.permutation(others))[:int(r.integers(0, 2))]:
                events.append(('ex', self._kind(self._qa_about(str(X), scene))))
            present = {it[0] for it in scene if it is not None}
            new_obj = [o for o in OBJECTS if o not in present]
            new = self._rand_item(new_obj[int(r.integers(len(new_obj)))])
            scene2 = tuple(new if i == s else it for i, it in enumerate(scene))
            events.append(('change', scene2))
            if r.random() < 0.5:
                now = dict(stage='B', task='slot', held_out=new[0] in self.cur.held['slot'], needs_scene=True,
                           X=new[0], item=('slot', new[0]))
                self._fill(now, scene2)
                events.append(('ex', self._kind(now)))
            q = dict(stage='D', task='before', held_out=False, needs_scene=True, prompt='刚才' + S + '是什么',
                     acceptable=[old, '是' + old], model=old, item=('before', old), ref='change')
            events.append(('ex', self._kind(q, p['probe_frac_mem'])))
            return dict(stage='D', events=events)
        # one-shot new object: 这是X (novel, pointing) once ... later 这是什么 (pointing at it) -> X
        fresh = [o for o in NOVEL if o not in self.novel_used]
        if self.novel_used and (not fresh or r.random() < 0.3):         # retention: an object named in an earlier dialogue
            X = self.novel_used[int(r.integers(len(self.novel_used)))]
            intro = False
        else:
            X = (fresh or NOVEL)[int(r.integers(len(fresh or NOVEL)))]
            intro = True
            if X not in self.novel_used:
                self.novel_used.append(X)
        s = int(r.integers(3))
        scene = self._scene_with([X], slots=[s])
        events = [('scene', scene)]
        if intro:
            events.append(('stmt', dict(task='oneshot_intro', text='这是' + X, ref=('novel', X), item=('oneshot', X),
                                        attend=s)))
            fill = self._qa_exchanges('B', int(r.integers(0, 3)), {X})
            for ex in fill:
                if ex['X'] not in {it[0] for it in scene if it is not None}:
                    continue                                  # fillers must be about objects in this scene
                self._fill(ex, scene)
                events.append(('ex', self._kind(ex)))
        q = dict(stage='D', task='oneshot', held_out=False, needs_scene=True, prompt='这是什么',
                 acceptable=[X, '是' + X, '这是' + X], model=X, item=('oneshot', X), ref=('novel', X), attend=s,
                 one_shot_q=True, kind='probe')
        events.append(('ex', q))
        return dict(stage='D', events=events)

    # ---------------- stage A
    def _dlg_A(self):
        r, p = self.rng, self.p
        w = np.array(p['a_weights'], float)
        kind = ['cds', 'overheard', 'play'][int(r.choice(3, p=w / w.sum()))]
        X = OBJECTS[int(r.integers(len(OBJECTS)))]
        scene = self._scene_with([X])
        s = [i for i, it in enumerate(scene) if it is not None and it[0] == X][0]
        events = [('scene', scene)]
        if kind == 'cds':
            o, c, n = scene[s]
            forms = [(X, s), ('这是' + X, s), ('看' + X, s), (X + '在' + SLOTS[s], None), (COLOURS[c][0] + '色的' + X, s),
                     (COUNTS[n] + X, s), ('宝宝看' + X, s)]
            for _ in range(int(r.integers(1, 4))):
                txt, att = forms[int(r.integers(len(forms)))]
                events.append(('stmt', dict(task='cds', text=txt, attend=att, item=None)))
        elif kind == 'overheard':
            exs = self._qa_exchanges('BC'[int(r.integers(2))], int(r.integers(1, 3)), {X} - {X})
            req = list(dict.fromkeys([X] + [ex['X'] for ex in exs if 'X' in ex]))[:3]
            scene = self._scene_with(req)
            events = [('scene', scene)]
            for ex in exs:
                if 'X' in ex:
                    if ex['X'] not in {it[0] for it in scene if it is not None}:
                        continue
                    self._fill(ex, scene)
                ex['stage'] = 'A'; ex['kind'] = 'overheard'; ex['model_by'] = 'other2'
                ex['held_out'] = False if not ex['held_out'] else True
                if ex['held_out']:
                    continue                                   # held-out pairs are never heard answered
                events.append(('ex', ex))
        else:
            events.append(('play', p['play_ticks']))
        return dict(stage='A', events=events)

    # ---------------- stage E
    def _dlg_E(self):
        r, p = self.rng, self.p
        if r.random() >= p['e_corpus']:
            st = 'BCD'[int(r.integers(3))]
            d = self._dlg_D() if st == 'D' else self._dlg_qa(st)
            return d
        C = self._corpus
        dlg = C.train[self.corpus_pos % len(C.train)]
        self.corpus_pos += 1
        events = [('scene', None)]
        for i in range(0, len(dlg) - 1, 2):
            ex = dict(stage='E', task='corpus', held_out=False, needs_scene=False, prompt=dlg[i],
                      acceptable=[dlg[i + 1]], model=dlg[i + 1], item=('corpus', dlg[i]))
            events.append(('ex', self._kind(ex, 0.0)))
        if r.random() < p['e_fam']:
            q, a = C.fam[int(r.integers(len(C.fam)))]
            ex = dict(stage='E', task='corpus_fam', held_out=False, needs_scene=False, prompt=q, acceptable=[a],
                      model=a, item=('corpus_fam', q))
            events.append(('ex', self._kind(ex, 0.0)))
        if r.random() < p['e_fam_probe']:
            q, a = C.fam[int(r.integers(len(C.fam)))]
            events.append(('ex', dict(stage='E', task='corpus_fam', held_out=False, needs_scene=False, prompt=q,
                                      acceptable=[a], model=a, item=('corpus_fam', q), kind='probe')))
        if r.random() < p['e_probe']:
            q, a = C.test[int(r.integers(len(C.test)))]
            events.append(('ex', dict(stage='E', task='corpus', held_out=True, needs_scene=False, prompt=q,
                                      acceptable=[a], model=a, item=('corpus', q), kind='probe')))
        return dict(stage='E', events=events)

    def _draw_dialogue(self):
        st = self.stages[int(self.rng.integers(len(self.stages)))]
        if st == 'A':
            return self._dlg_A()
        if st == 'D':
            return self._dlg_D()
        if st == 'E':
            return self._dlg_E()
        return self._dlg_qa(st)

    # ================================================================ running
    def _record(self, ex, kind):
        item = ex.get('item')
        n = self.exposures.get(item, 0)
        ref = ex.get('ref')
        lt = self._refs.get(ref) if ref is not None else self.last_taught.get(item)
        if isinstance(ref, tuple):
            lt = self.last_taught.get(('oneshot', ref[1]))
        return dict(i=len(self.log), dlg=self.dlg_offset + self.n_dialogues, stage=ex['stage'], task=ex['task'], kind=kind,
                    prompt=ex['prompt'], prompt_by='other', scene=self.scene, attend=ex.get('attend'),
                    acceptable=list(ex.get('acceptable') or []), item=item, held_out=bool(ex.get('held_out')),
                    needs_scene=bool(ex.get('needs_scene')), familiar=n > 0, n_exposures=n,
                    one_shot=bool(ex.get('one_shot_q')) and n == 1,
                    ticks_since_taught=(self.t - lt) if lt is not None else None,
                    model_reply=None, model_by=None, reply='', t_prompt_start=self.t)

    def _finish(self, rec, tr_rep, cut=False, prefix=''):
        tr_rep.close()
        rec[prefix + 'reply'] = tr_rep.text
        rec[prefix + 't_first_mouth'] = tr_rep.first
        rec[prefix + 't_stop'] = tr_rep.stop if tr_rep.stop is not None else (tr_rep.last + 1 if tr_rep.last is not None else None)
        rec[prefix + 'latency'] = (tr_rep.first - rec['t_prompt_end']) if tr_rep.first is not None else None
        rec[prefix + 'runs'] = list(tr_rep.runs)
        rec[prefix + 'garble'] = tr_rep.garble
        rec[prefix + 'extra_turns'] = max(0, tr_rep.bouts - 1)
        rec[prefix + 'stop'] = 'silent' if tr_rep.first is None else ('cut' if cut else 'self')
        rec[prefix + 'agent_stop'] = tr_rep.agent_stop

    def _exchange(self, ex):
        p = self.p
        kind = ex['kind']
        rec = self._record(ex, kind)
        self._set_scene(self.scene, ex.get('attend'))
        rec['scene'] = self.scene
        tr_all = Track(); self.trk.append(tr_all)
        self._say('other', ex['prompt'])
        rec['t_prompt_end'] = self.t
        prompt_overlap = tr_all.overlap
        info = dict(prompt=ex['prompt'], task=ex['task'], item=ex.get('item'), scene=self.scene,
                    attend=ex.get('attend'), acceptable=ex.get('acceptable'), kind=kind, stage=ex['stage'])
        self._hook('bench_prompt', info)
        tr = Track(); self.trk.append(tr)
        if kind in ('probe', 'recast'):
            cut = self._wait(tr, p['gap_answer'], p['after_stop_test'] if kind == 'probe' else p['pause_after'],
                             p['reply_cap'])
            self._finish(rec, tr, cut)
            if kind == 'recast':
                good = rec['reply'] in rec['acceptable']
                rew = p['reward_ok'] if good else (p['reward_wrong'] if rec['reply'] else 0.0)
                text = rec['reply'] if good else ex['model']
                rec['reward'], rec['recast'] = rew, text
                self.trk.remove(tr)
                tr = Track(); self.trk.append(tr)
                t0 = self.t
                self._say('other', text, reward_first=rew)
                rec['t_recast_end'] = self.t
                self._taught(ex, info, text)
                cut = self._wait(tr, p['model_wait'], p['pause_after'], p['reply_cap'])
                self._finish(rec, tr, cut, prefix='after_recast_')
                rec['after_recast_latency'] = (tr.first - rec['t_recast_end']) if tr.first is not None else None
                del t0
        elif kind in ('model', 'overheard'):
            self._silence(p['gap_model'])
            by = ex.get('model_by', 'other2')
            rec['model_reply'], rec['model_by'] = ex['model'], by
            self._say(by, ex['model'])
            rec['t_model_end'] = self.t
            self._taught(ex, info, ex['model'])
            tr2 = Track(); self.trk.append(tr2)
            cut = self._wait(tr2, p['model_wait'], p['pause_after'], p['reply_cap'])
            self._finish(rec, tr, cut)                         # everything the agent said after the prompt
            self._finish(rec, tr2, cut, prefix='after_model_')
            self.trk.remove(tr2)
        elif kind == 'force':
            rec['cued'] = ex['model']
            self._forced(ex['model'])
            self._taught(ex, info, ex['model'])
            self._wait(tr, 0, p['pause_after'], p['reply_cap'])
            self._finish(rec, tr)
        else:
            raise ValueError(kind)
        self.trk.remove(tr)
        self.trk.remove(tr_all)
        rec['mouth_while_partner'] = tr_all.overlap
        rec['mouth_during_prompt'] = prompt_overlap
        rec['t_end'] = self.t
        self._set_scene(self.scene, None)
        self.log.append(rec)
        return rec

    def _statement(self, st):
        """the partner says something that wants no reply (CDS, 我叫N, 这是X); the agent may speak after it."""
        p = self.p
        ex = dict(stage=self._cur_stage, task=st['task'], prompt=st['text'], item=st.get('item'), attend=st.get('attend'))
        rec = self._record(ex, 'statement')
        self._set_scene(self.scene, st.get('attend'))
        tr_all = Track(); self.trk.append(tr_all)
        self._say('other', st['text'])
        rec['t_prompt_end'] = self.t
        if st.get('ref') is not None:
            self._refs[st['ref']] = self.t
        if st.get('item') is not None:
            self.exposures[st['item']] += 1
            self.last_taught[st['item']] = self.t
        self._hook('bench_statement', dict(text=st['text'], task=st['task'], scene=self.scene, attend=st.get('attend')))
        tr = Track(); self.trk.append(tr)
        cut = self._wait(tr, p['model_wait'], p['pause_after'], p['reply_cap'])
        self._finish(rec, tr, cut)
        self.trk.remove(tr); self.trk.remove(tr_all)
        rec['mouth_while_partner'] = tr_all.overlap
        rec['mouth_during_prompt'] = tr_all.overlap
        rec['t_end'] = self.t
        self._set_scene(self.scene, None)
        self.log.append(rec)

    def _play(self, ticks):
        """free play: whatever the mouth produces, the tutor imitates it (as the partner) with a small reward."""
        p = self.p
        rec = self._record(dict(stage='A', task='play', prompt=''), 'play')
        rec['t_prompt_end'] = self.t
        tr_all = Track(); self.trk.append(tr_all)
        t_end = self.t + ticks
        bout, imits, reward_total = [], [], 0.0
        while self.t < t_end:
            said, onset = self._tick()
            if said is not None and onset:
                bout.append(said)
            if said is None and bout and tr_all.silent >= 2:
                txt = ''.join(c for c in bout if c != '~')[-p['imitate_max']:]
                if txt:
                    self._silence(max(0, p['imitate_gap'] - 2))
                    self._say('other', txt, reward_first=p['reward_imitate'])
                    reward_total += p['reward_imitate']
                imits.append((''.join(bout), txt))
                bout = []
        tr_all.close()
        self.trk.remove(tr_all)
        rec.update(reply=tr_all.text, runs=list(tr_all.runs), garble=tr_all.garble, imitations=imits,
                   reward=reward_total, mouth_while_partner=tr_all.overlap, mouth_during_prompt=0, t_end=self.t,
                   t_first_mouth=tr_all.first, latency=None, extra_turns=0, stop=None, agent_stop=None)
        self.log.append(rec)

    def _taught(self, ex, info, reply):
        item = ex.get('item')
        if item is not None:
            self.exposures[item] += 1
            self.last_taught[item] = self.t
        self._hook('bench_taught', info, reply)

    def _forced(self, text):
        """v1 teacher forcing: cue syllable i on every tick until the mouth makes an onset; after the last syllable
        an all-zero cue (= END push) until the mouth is silent."""
        i, t_last = 0, self.t
        end_cue = np.zeros(self.voice.n_muscles, np.float32); end_cue.setflags(write=False)
        while True:
            cue = self.cue_vec(text[i]) if i < len(text) else end_cue
            said, onset = self._tick(cue=cue)
            if i < len(text):
                if onset:
                    i += 1; t_last = self.t
            elif said is None:
                break
            if self.t - t_last > self.p['force_timeout']:
                self.timing_violations['teacher_timeout'] += 1
                break

    def _run_dialogue(self, dlg):
        p = self.p
        self._cur_stage = dlg['stage']
        first_scene = True
        for ev in dlg['events']:
            typ = ev[0]
            if typ == 'scene':
                self._set_scene(None)
                self._silence(max(0, p['gap_dialogue'] - p['scene_lead']))
                self._set_scene(ev[1])
                self._silence(p['scene_lead'] if first_scene else 0)
                first_scene = False
            elif typ == 'change':
                self._silence(p['gap_change'] // 2)
                self._set_scene(ev[1])
                self._refs['change'] = self.t
                self._silence(p['gap_change'] - p['gap_change'] // 2)
            elif typ == 'ex':
                self._exchange(ev[1])
            elif typ == 'stmt':
                self._statement(ev[1])
            elif typ == 'play':
                self._play(ev[1])
        self.n_dialogues += 1

    def run(self, agent, n_dialogues=100, verbose=0):
        self.agent = agent
        self._hook('bench_attach', self)
        if self.n_dialogues == 0:
            self._silence(self.p['lead_in'])
        for _ in range(n_dialogues):
            self._run_dialogue(self._draw_dialogue())
            if verbose and self.n_dialogues % verbose == 0:
                print(f'  {self.n_dialogues} dialogues, {self.t} ticks', flush=True)
        return self.log

    def run_script(self, agent, items, stage='S'):
        """run a given list of exchanges: each item is a dict with prompt, reply (the taught/model reply) or
        acceptable (list), kind ('force'|'model'|'probe'|'recast'), optional task, held_out, scene, attend.
        A silent gap of gap_dialogue ticks precedes each item (0 for ARCH_v1's protocol)."""
        self.agent = agent
        self._hook('bench_attach', self)
        self._cur_stage = stage
        if self.n_dialogues == 0:
            self._silence(self.p['lead_in'])
        for it in items:
            acc = list(it.get('acceptable') or [it['reply']])
            ex = dict(stage=it.get('stage', stage), task=it.get('task', 'script'), held_out=bool(it.get('held_out')),
                      needs_scene=bool(it.get('needs_scene')), prompt=it['prompt'], acceptable=acc,
                      model=it.get('reply', acc[0]), kind=it.get('kind', self.tutor),
                      item=it.get('item', ('script', it['prompt'])), attend=it.get('attend'),
                      model_by=it.get('model_by', 'other2'))
            if self.p['gap_dialogue'] > 0:
                self._set_scene(None)
                self._silence(self.p['gap_dialogue'])
            self._set_scene(it.get('scene'))
            self._exchange(ex)
            self.n_dialogues += 1
        return self.log

    # ---------------------------------------------------------------- helpers for agents (innate calibration)
    def warmup_windows(self, n=60, seed=1, pause=4):
        """ARCH_v1's birth warm-up as ear windows: n random chars (from the whole voice), each followed by `pause`
        silent ticks, own voice silent.  Uses a separate stream; the world's state is not touched."""
        v = self.voice
        st = Stream(v.F, v.T, v.T // SOUND, 3, v.levels, self.noise, noise_seed=self.seed * 7919 + 17)
        rng = np.random.default_rng(seed)
        out = []
        none = (None, None, False)
        for _ in range(n):
            ch = v.chars[rng.integers(len(v.chars))]
            for k in range(SOUND):
                out.append(st.tick(((ch, v.hear(ch, 'other'), k == 0), none, none)).copy())
            for _ in range(pause):
                out.append(st.tick((none, none, none)).copy())
        return out


def run_curriculum(agent, plan, seed=0, tutor='model', physics='random', noise=0.0, **params):
    """plan: list of (stage, n_dialogues[, tutor]), e.g. [('A', 300), ('B', 2000), ('C', 2000), ('D', 2000), ('E', 2000)].
    One agent goes through all stages; each World continues the previous one (clock, exposure history, novel objects,
    corpus position; a fresh content stream).  Returns (worlds, log) with the logs concatenated in time order."""
    worlds, log, prev = [], [], None
    for step in plan:
        stage, n = step[0], step[1]
        tut = step[2] if len(step) > 2 else tutor
        w = World(stage, seed=seed, physics=physics, noise=noise, tutor=tut, prev=prev, **params)
        w.run(agent, n)
        worlds.append(w); log += w.log; prev = w
    return worlds, log
