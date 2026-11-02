# A small bench-like world (my own, until arch/v2/bench is READY): same sound physics as world/voice.py (3 ticks per
# char, speakers mixed by max, own voice heard 1 tick later), three voices (other, other2, self), a 3-slot scene seen
# through a fixed sparse random eye code, a contingent reward, and stages A-D of the bench contract.
# The agent sees only obs = {ear, eye, reward, cue=None}; it returns muscles (32,) (> 0 = active).
import json, numpy as np
from voice_copy import Voice

SOUND = 3
OBJ = ["苹果", "香蕉", "葡萄", "西瓜", "饼干", "牛奶", "小狗", "小猫", "皮球", "汽车", "气球", "鸭子"]
FOOD = OBJ[:6]
COL = ["红", "绿", "黄", "蓝", "白", "黑"]
CNT = ["一", "两", "三"]
SLOTW = ["左边", "中间", "右边"]
NAMES = ["小明", "小红", "小兰", "小刚", "小东", "小天"]
NEWOBJ = ["熊猫", "兔子", "积木", "包子"]          # one-shot objects (stage D)
RITUAL = [("你好", "你好"), ("谢谢", "不客气"), ("晚安", "晚安"), ("再见", "再见")]


def corpus_chars():
    pool = [json.loads(l) for l in open("/root/work/corpus/dialogs.jsonl", encoding="utf-8")]
    return sorted({ch for c in pool if len(c) >= 2 for u in c for ch in u})


class Stream3:
    """world/voice.py Stream with three sources (other, other2, self); max-mixed; re-articulation flag for self."""

    def __init__(self, v, step=2):
        self.v, self.step = v, step
        self.win = np.zeros((v.F, v.T), int)
        self.src = {k: [None, 0] for k in ("other", "other2", "self")}

    def tick(self, srcs):
        v, T = self.v, self.v.T
        new = np.zeros((v.F, self.step), int)
        for who in ("other", "other2", "self"):
            ch, fresh = srcs.get(who, (None, False))
            prev, ph = self.src[who]
            if ch != prev or fresh:
                ph = 0
            if ch is not None:
                g = v.hear(ch, who)
                cols = [min(ph + j, T - 1) for j in range(self.step)]
                new = np.maximum(new, g[:, cols])
            self.src[who] = [ch, ph + self.step]
        self.win = np.concatenate([self.win[:, self.step:], new], 1)
        return self.win


class World:
    def __init__(self, seed=0, code_k=8, noise=0.0, held_frac=0.3, vocab_voice=True):
        self.vocab = sorted(set("".join(OBJ + NEWOBJ + COL + CNT + SLOTW + NAMES + [q + a for q, a in RITUAL] +
                                        ["这是什么在哪里你要吗好不吃颜色有几个我叫刚才的"])))
        self.chars = self.vocab if vocab_voice else corpus_chars()
        self.v = Voice(self.chars, seed=0)
        r2 = np.random.default_rng(1234)
        self.v.voice["other2"] = r2.choice([-1, 0, 1], p=[0.15, 0.7, 0.15], size=(self.v.F, self.v.T))
        self.rng = np.random.default_rng(seed)
        self.stream = Stream3(self.v)
        # RETINOTOPIC eye (innate visual organisation): per slot the same feature maps replicated:
        #   shape map (OBJ+NEWOBJ codes, 8 of 64 bits), colour map (8 of 48), number map (8 of 24), pointing (8 bits)
        er = np.random.default_rng(99)
        self.maps = {"o": (64, OBJ + NEWOBJ), "c": (48, COL), "n": (24, CNT)}
        self.fcode = {a: {x: er.choice(nb, code_k, replace=False) for x in vals} for a, (nb, vals) in self.maps.items()}
        self.slot_bits = 64 + 48 + 24 + 8
        self.n_eye = 3 * self.slot_bits
        self.off = {"o": 0, "c": 64, "n": 112, "att": 136}
        self.noise = noise
        hr = np.random.default_rng(500 + seed)
        self.held = {"offer": set(hr.choice(OBJ, int(round(held_frac * len(OBJ))), replace=False)),
                     "colour": set(), "count": set()}
        # held-out (object, slot) scene combinations for naming / where (stage B)
        combos = [(o, s) for o in OBJ for s in range(3)]
        idx = hr.choice(len(combos), int(round(held_frac * len(combos))), replace=False)
        self.held_os = {combos[i] for i in idx}
        self.held["colour"] = set(map(tuple, [(o, c) for o in OBJ for c in COL if hr.random() < held_frac]))
        self.scene = None; self.att = None
        self.reward = 0.0
        self.log = []
        self.t = 0
        self.prev_force = np.zeros(32, int); self.saying = None; self.onset = False

    # ------------------------------------------------------------------ scene
    def new_scene(self, objs=None):
        o = list(self.rng.choice(OBJ, 3, replace=False)) if objs is None else list(objs)
        self.scene = [(o[s], COL[int(self.rng.integers(6))], CNT[int(self.rng.integers(3))]) for s in range(3)]
        return self.scene

    def pick_slot(self, test, held_fn):
        """a queried slot: in training never a held-out combination; at test 50 % held-out when possible.
        The scene itself may contain held-out combinations (they are seen, just never asked about)."""
        for _ in range(100):
            sc = self.new_scene()
            ok = [s for s in range(3) if not held_fn(sc, s)]
            hs = [s for s in range(3) if held_fn(sc, s)]
            if test and hs and self.rng.random() < 0.5:
                return sc, int(self.rng.choice(hs)), True
            if ok:
                return sc, int(self.rng.choice(ok)), False
        raise RuntimeError

    def eye(self):
        e = np.zeros(self.n_eye, bool)
        if self.scene is not None:
            for s, (o, c, n) in enumerate(self.scene):
                b = s * self.slot_bits
                for a, x in (("o", o), ("c", c), ("n", n)):
                    e[b + self.off[a] + self.fcode[a][x]] = True
            if self.att is not None:
                e[self.att * self.slot_bits + self.off["att"]: (self.att + 1) * self.slot_bits] = True
        return e

    # ------------------------------------------------------------------ one tick
    def tick(self, agent, other=None, new_other=False, other2=None, new_o2=False):
        win = self.stream.tick({"other": (other, new_other), "other2": (other2, new_o2), "self": (self.saying, self.onset)})
        if self.noise > 0:
            win = np.clip(win + (self.rng.random(win.shape) < self.noise), 0, self.v.levels)
        obs = {"ear": win.copy(), "eye": self.eye(), "reward": self.reward, "cue": None}
        self.reward = 0.0
        f = np.asarray(agent.step(obs), int)
        said = self.v.mouth({f"m{i}": int(x) for i, x in enumerate(f)})
        act, pact = f > 0, self.prev_force > 0
        if said is None:
            onset = False
        elif not pact.any() or not np.array_equal(act, pact):
            onset = True
        else:
            onset = bool(f[act].max() < self.prev_force[act].max())
        self.prev_force = f.copy()
        self.saying, self.onset = said, onset
        self.t += 1
        return said, onset

    def say(self, agent, line, who="other"):
        """a line by another speaker; returns what the agent's mouth produced meanwhile (should be nothing)."""
        out = []
        for ch in line:
            for k in range(SOUND):
                kw = {"other": ch, "new_other": k == 0} if who == "other" else {"other2": ch, "new_o2": k == 0}
                s, on = self.tick(agent, **kw)
                if on and s is not None:
                    out.append(s)
        return out

    def silence(self, agent, n):
        out = []
        for _ in range(n):
            s, on = self.tick(agent)
            if on and s is not None:
                out.append(s)
        return out

    def wait_reply(self, agent, max_wait=12, max_len=60):
        """free ticks until the agent has spoken and been silent 5 ticks, or never started within max_wait."""
        out = []; started = None; quiet = 0
        for k in range(max_len):
            s, on = self.tick(agent)
            if s is not None:
                quiet = 0
                if started is None:
                    started = k
                if on:
                    out.append(s)
            else:
                quiet += 1
            if started is None and k >= max_wait:
                break
            if started is not None and quiet >= 5:
                break
        return "".join(out), started

    # ------------------------------------------------------------------ tasks
    def task(self, stage, kind=None, test=False):
        """returns (kind, prompt, acceptable list, model reply, tags)."""
        r = self.rng
        kinds = {"B": ["name", "slot", "where"], "C": ["offer", "taste", "colour", "count", "ritual"],
                 "D": ["check", "colour", "name", "offer"]}[stage]
        kind = kind or kinds[int(r.integers(len(kinds)))]
        tags = {"held_out": False}
        hos = lambda sc, s: (sc[s][0], s) in self.held_os
        if kind in ("name", "check", "slot", "where"):
            sc, s, h = self.pick_slot(test, hos); tags["held_out"] = h
            o = sc[s][0]
            if kind == "name":
                self.att = s; return kind, "这是什么", [o], o, tags
            if kind == "check":
                self.att = s
                x = o if r.random() < 0.5 else str(r.choice([q for q in OBJ if q not in [w[0] for w in sc]]))
                return kind, "这是" + x + "吗", (["是", "是的"] if x == o else ["不是"]), ("是" if x == o else "不是"), tags
            self.att = None
            if kind == "slot":
                return kind, SLOTW[s] + "是什么", [o], o, tags
            return kind, o + "在哪里", ["在" + SLOTW[s]], "在" + SLOTW[s], tags
        if kind == "offer":
            self.scene = None; self.att = None
            if test and r.random() < 0.5:
                x = str(r.choice(sorted(self.held["offer"]))); tags["held_out"] = True
            else:
                x = str(r.choice([o for o in OBJ if o not in self.held["offer"]]))
            rep = "要" + x if r.random() < 0.7 else "我要" + x
            return kind, "你要" + x + "吗", ["要" + x, "我要" + x], rep, tags
        if kind == "taste":
            self.scene = None; self.att = None
            x = str(r.choice(FOOD)); return kind, x + "好不好吃", ["好吃"], "好吃", tags
        if kind == "colour":
            sc, s, h = self.pick_slot(test, lambda sc, s: (sc[s][0], sc[s][1]) in self.held["colour"])
            tags["held_out"] = h; self.att = None
            o, c, _ = sc[s]
            return kind, o + "是什么颜色", [c + "色"], c + "色", tags
        if kind == "count":
            sc, s, h = self.pick_slot(test, lambda sc, s: False); self.att = None
            o, _, n = sc[s]; return kind, "有几个" + o, [n + "个"], n + "个", tags
        if kind == "ritual":
            self.scene = None; self.att = None
            q, a = RITUAL[int(r.integers(len(RITUAL)))]; return kind, q, [a], a, tags
        raise ValueError(kind)

    def exchange(self, agent, stage, mode="model", kind=None, test=False, pre=6):
        """one exchange.  mode: 'model' (other2 says the reply; agent overhears), 'recast' (agent answers, then
        reward + recast by other2), 'test' (agent answers; no reward, no recast)."""
        kind, q, acc, rep, tags = self.task(stage, kind, test)
        overlap = self.silence(agent, pre)
        overlap += self.say(agent, q, "other")
        rec = {"stage": stage, "task": kind, "prompt": q, "scene": self.scene, "att": self.att, "acc": acc,
               "mode": mode, **tags}
        if mode == "model":
            overlap += self.silence(agent, 4)
            overlap += self.say(agent, rep, "other2")
            self.silence(agent, 4)
            rec["reply"] = None
        else:
            reply, started = self.wait_reply(agent)
            rec["reply"] = reply; rec["latency"] = started
            ok = reply in acc
            if mode == "recast":
                self.reward = 1.0 if ok else 0.0
                self.silence(agent, 2)
                self.say(agent, rep if not ok else reply, "other2")
                self.silence(agent, 4)
        rec["overlap"] = "".join(overlap)
        self.att = None
        self.log.append(rec)
        return rec

    def warmup(self, agent, n=60):
        rng = np.random.default_rng(1)
        for _ in range(n):
            ch = self.chars[rng.integers(len(self.chars))]
            for j in range(SOUND):
                agent.warm(self.stream.tick({"other": (ch, j == 0)}))
            for _ in range(4):
                agent.warm(self.stream.tick({}))
        agent.end_warmup()

    # ------------------------------------------------------------------ stage A pieces
    def cds(self, agent):
        """child-directed utterance or an overheard two-speaker routine (no reply demanded)."""
        st = "BC"[int(self.rng.integers(2))]
        kind, q, acc, rep, tags = self.task(st)
        self.silence(agent, 6); self.say(agent, q, "other"); self.silence(agent, 4)
        self.say(agent, rep, "other2"); self.silence(agent, 4)
        self.att = None

    def play(self, agent, n_turns=6):
        """play period: the tutor says a short syllable; whatever the agent produces is imitated back (+ small reward)."""
        voc = sorted(set("".join(OBJ + COL + CNT + SLOTW + ["这是什么在哪里你要吗好不吃颜色有几个我叫刚才谢客气晚安再见"])))
        self.silence(agent, 6)
        self.say(agent, voc[int(self.rng.integers(len(voc)))], "other")
        n = 0
        for _ in range(n_turns):
            reply, started = self.wait_reply(agent, max_wait=8)
            if reply and reply[0] != "~":
                self.reward = 0.2
                self.silence(agent, 2); self.say(agent, reply[:1], "other"); n += 1
            else:
                self.silence(agent, 3); self.say(agent, voc[int(self.rng.integers(len(voc)))], "other")
        self.silence(agent, 4)
        return n
