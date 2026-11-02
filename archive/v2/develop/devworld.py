# -*- coding: utf-8 -*-
"""Symbolic developmental micro-world (my module-test harness; the shared bench replaces it for comparable numbers).

Same task list as arch/v2/bench/API.md stages B-D, but at the level of DETECTED ITEMS (chars), so the cognitive core
can be developed in seconds.  The sound front end, Z detection and the body are tested separately and on the bench.

Scene: 3 slots (left/middle/right), each an object (12 kinds) with a colour (6) and a count (1-3), or empty.
Eye code: a fixed sparse random code (8 of N_EYE bits) per (slot, attribute, value), unioned, plus an attended-slot
code when the tutor points or looks.  NOT retinotopic: bits of different slots are drawn from the same space.
"""
from __future__ import annotations

import numpy as np

OBJ = ["苹果", "香蕉", "西瓜", "葡萄", "饼干", "牛奶", "小狗", "小猫", "皮球", "汽车", "气球", "积木"]
FOOD = OBJ[:6]
NEW_OBJ = ["熊猫", "兔子", "鸭子", "老虎", "飞机", "火车"]          # one-shot objects (stage D only)
COL = ["红色", "绿色", "蓝色", "黄色", "白色", "黑色"]
CNT = ["一个", "两个", "三个"]
SLOT = ["左边", "中间", "右边"]
NAMES = ["小明", "小红", "小华", "小刚", "小丽", "小军", "小强", "小雪"]
RITUALS = [("你好", "你好"), ("谢谢", "不客气"), ("晚安", "晚安"), ("再见", "再见"), ("早上好", "早上好")]
N_EYE = 400


class Scene:
    def __init__(self, objs, cols, cnts):
        self.objs, self.cols, self.cnts = objs, cols, cnts      # per slot (None = empty)

    def slot_of(self, o):
        return [i for i, x in enumerate(self.objs) if x == o]


class DevWorld:
    FRAMES = {   # name: (prompt template, fillers, answers (templates), tutor's choice weights)
        "offer": ("你要{X}吗", OBJ, ["要{X}", "我要{X}"], [0.7, 0.3]),
        "taste": ("{X}好不好吃", FOOD, ["好吃", "{X}好吃"], [0.6, 0.4]),
        "like": ("你喜欢{X}吗", OBJ, ["喜欢{X}", "喜欢"], [0.6, 0.4]),
    }

    def __init__(self, seed=0, held_frac=0.3, retino=True):
        self.rng = np.random.default_rng(seed)
        r = np.random.default_rng(seed + 100)
        self.code = {}
        self.retino = retino
        seg = N_EYE // 3
        allv = [("obj", o) for o in OBJ + NEW_OBJ] + [("col", c) for c in COL] + [("cnt", c) for c in CNT]
        for s in range(3):
            lo, hi = (s * seg, (s + 1) * seg) if retino else (0, N_EYE)
            for a in allv:
                self.code[(s,) + a] = lo + r.choice(hi - lo, 8, replace=False)
            self.code[(s, "att")] = lo + r.choice(hi - lo, 8, replace=False)
        # held-out frame x filler pairs: 30 % of each frame's fillers never heard in that frame
        self.held = set()
        for f, (tpl, fill, ans, w) in self.FRAMES.items():
            for x in r.choice(fill, size=int(round(held_frac * len(fill))), replace=False):
                self.held.add((f, x))
        # colour / count held-out object x property combos (never seen together in training scenes)
        self.held_col = {(o, COL[int(r.integers(6))]) for o in OBJ}
        self.scene = self.new_scene()
        self.dialog_names = None

    # ---------------------------------------------------------------- scenes
    def new_scene(self, test=False, force=None):
        r = self.rng
        objs = list(r.choice(OBJ, 3, replace=False))
        if force is not None:
            objs[int(r.integers(3))] = force
            if len(set(objs)) < 3:
                objs = [force] + [o for o in r.choice([o for o in OBJ if o != force], 2, replace=False)]
                r.shuffle(objs)
        cols, cnts = [], []
        for o in objs:
            while True:
                c = COL[int(r.integers(6))]
                if test or (o, c) not in self.held_col:
                    break
            cols.append(c); cnts.append(CNT[int(r.integers(3))])
        return Scene(objs, cols, cnts)

    def eye(self, sc: Scene, att=None):
        v = np.zeros(N_EYE, bool)
        for s in range(3):
            if sc.objs[s] is None:
                continue
            v[self.code[(s, "obj", sc.objs[s])]] = True
            v[self.code[(s, "col", sc.cols[s])]] = True
            v[self.code[(s, "cnt", sc.cnts[s])]] = True
        if att is not None:
            v[self.code[(att, "att")]] = True
        return v

    # ---------------------------------------------------------------- exchanges
    def exchange(self, stage, test=False, task=None):
        """returns dict(task, prompt, scene, att, acceptable, tutor (the reply the tutor says), held, pre=[lines])."""
        r = self.rng
        if task is None:
            tasks = {"B": ["name", "slot_name", "where"],
                     "C": ["offer", "taste", "like", "colour", "count", "ritual", "name", "slot_name", "where"],
                     "D": ["check", "name_mem", "changed", "oneshot", "offer", "colour", "name", "ritual"]}[stage]
            task = tasks[int(r.integers(len(tasks)))]
        sc = self.scene = self.new_scene(test=test)
        ex = dict(task=task, scene=sc, att=None, held=False, pre=[], stage=stage)
        if task == "name":
            s = int(r.integers(3)); o = sc.objs[s]
            ex.update(prompt="这是什么", att=s, acceptable=[o, "是" + o], tutor=o if r.random() < 0.6 else "是" + o)
        elif task == "slot_name":
            s = int(r.integers(3)); o = sc.objs[s]
            ex.update(prompt=SLOT[s] + "是什么", att=(s if r.random() < 0.5 else None), acceptable=[o], tutor=o)
        elif task == "where":
            s = int(r.integers(3)); o = sc.objs[s]
            ex.update(prompt=o + "在哪里", acceptable=["在" + SLOT[s]], tutor="在" + SLOT[s])
        elif task in self.FRAMES:
            tpl, fill, ans, w = self.FRAMES[task]
            pool = [x for x in fill if ((task, x) in self.held) == test]
            x = pool[int(r.integers(len(pool)))]
            acc = [a.format(X=x) for a in ans]
            ex.update(prompt=tpl.format(X=x), acceptable=acc, tutor=acc[int(r.choice(len(acc), p=w))], held=test)
        elif task == "colour":
            s = int(r.integers(3)); o = sc.objs[s]
            ex.update(prompt=o + "是什么颜色", att=s, acceptable=[sc.cols[s]], tutor=sc.cols[s],
                      held=(o, sc.cols[s]) in self.held_col)
        elif task == "count":
            s = int(r.integers(3)); o = sc.objs[s]
            ex.update(prompt="有几个" + o, att=s, acceptable=[sc.cnts[s]], tutor=sc.cnts[s])
        elif task == "ritual":
            q, a = RITUALS[int(r.integers(len(RITUALS)))]
            ex.update(prompt=q, acceptable=[a], tutor=a)
        elif task == "check":
            s = int(r.integers(3)); o = sc.objs[s]
            x = o if r.random() < 0.5 else [y for y in OBJ if y != o][int(r.integers(len(OBJ) - 1))]
            ex.update(prompt="这是" + x + "吗", att=s, acceptable=(["是", "是的"] if x == o else ["不是"]),
                      tutor=("是" if x == o else "不是"))
        elif task == "name_mem":
            n = NAMES[int(r.integers(len(NAMES)))]
            fill = [self.exchange("C", test=False, task=t) for t in r.choice(["ritual", "name", "offer"], int(r.integers(1, 4)))]
            ex.update(pre=[("say", "我叫" + n, "你好" + n)] + [("ex", f) for f in fill], prompt="我叫什么",
                      acceptable=[n], tutor=n)
        elif task == "changed":
            old = sc; s = int(r.integers(3)); o = old.objs[s]
            ex.update(pre=[("look", old)], prompt="刚才" + SLOT[s] + "是什么", acceptable=[o], tutor=o)
            ex["scene"] = self.scene = self.new_scene(test=test)
        elif task == "oneshot":
            x = NEW_OBJ[int(r.integers(len(NEW_OBJ)))]
            sc = self.scene = self.new_scene(test=test, force=x)
            s = sc.slot_of(x)[0]
            fill = [self.exchange("C", test=False, task=t) for t in r.choice(["ritual", "offer", "colour"], int(r.integers(1, 4)))]
            sc2 = self.new_scene(test=test, force=x); s2 = sc2.slot_of(x)[0]
            ex.update(pre=[("teach", "这是" + x, sc, s)] + [("ex", f) for f in fill], prompt="这是什么", scene=sc2, att=s2,
                      acceptable=[x, "是" + x], tutor=x)
            self.scene = sc2
        else:
            raise ValueError(task)
        return ex
