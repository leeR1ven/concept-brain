# -*- coding: utf-8 -*-
"""各部分单独验证（用户：先把各个部分做好、独立验证能否可用，再组合起来）。

  python tests/test_parts.py sense     听觉区：分得清字 / 分得清是谁说的 / 记住连续的声音（给开头顺着放出来）
  python tests/test_parts.py thought   前额叶：想法范围大致不变、慢慢移动；A 之后想到 B
  python tests/test_parts.py hippo     海马体：听一遍就记住，过一阵忘掉
  python tests/test_parts.py motor     运动区：平时放松；用数据训练后，给个开头就做完一串动作并停下
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.areas import Circuit, SenseArea, SenseCfg  # noqa: E402
from brain.net3d import Net3D, Net3DCfg  # noqa: E402
from world.voice import Voice  # noqa: E402

SENT = "今天天气很好"
OTHER = "我们去公园散步吧你吃饭了吗"
CHARS = sorted(set(SENT + OTHER))
SOUND = 2          # 每个字响几拍；一句话里字连着说，句子之间停 PAUSE 拍
PAUSE = 4


def jac(x, y):
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 0.0


def lcs(a: str, b: str) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            d[i + 1][j + 1] = d[i][j] + 1 if x == y else max(d[i][j + 1], d[i + 1][j])
    return d[-1][-1]


def runs(seq: list) -> str:
    """去掉空拍、同一个字连续几拍只算一次。"""
    return "".join(c for i, c in enumerate(seq) if c != "_" and (i == 0 or c != seq[i - 1]))


# ================================================================ 听觉区
class Ear:
    def __init__(self, cfg: SenseCfg | None = None, net: Net3DCfg | None = None, seed: int = 0, chars=CHARS):
        self.chars = list(chars)
        self.voice = v = Voice(self.chars, seed=seed)
        self.area = SenseArea("ear", Net3D(v.F, v.T, net, seed=seed), cfg, seed=seed)
        self.c = Circuit([self.area])
        self.sil = self.ref(None)

    def lines(self, ch, who="other"):
        return self.area.net.encode(self.voice.hear(ch, who))

    def ref(self, ch, who="other"):
        return self.area.pattern(self.lines(ch, who))

    def tick(self, ch=None, who="other"):
        self.c.step(lines={"ear": self.lines(ch, who)})

    def hear(self, utt: str):
        for ch in utt:
            for _ in range(SOUND):
                self.tick(ch)
        for _ in range(PAUSE):
            self.tick()

    def decode(self, chars=None) -> str:
        """皮层里比安静时多亮的那一整片，最像哪个字（比较整片激活）。"""
        extra = self.area.region.s & ~self.sil
        if extra.sum() < 5:
            return "_"
        sc = {c: jac(extra, self.ref(c) & ~self.sil) for c in (chars or self.chars)}
        return max(sc, key=sc.get)

    def warmup(self, n=60, seed=1):
        """出生后随便听一阵（不学），经常亮、没特征的退出皮层。"""
        self.c.learning = False
        rng = np.random.default_rng(seed)
        for _ in range(n):
            self.hear(self.chars[rng.integers(len(self.chars))])
        a = self.area
        a.mute(a.featureless())
        a.clear_usage()
        self.c.learning = True


SENTS = ["今天天气很好", "我们去公园散步", "你吃饭了吗", "明年再来看花"]


def sense(reps: int = 80, seed: int = 0, cfg: SenseCfg | None = None, verbose: bool = True):
    """几句话各听 reps 遍（顺序打乱），之后分别只给每句的第一个字，看听觉区想起的范围偏向哪里。
    读法看整片激活、看范围（不取第一名）：给开头后几拍里，比安静多亮的那一片
    和"这句后面的字"的相似度  −  和"别的句子的字"的相似度。> 0 就是偏向自己那句。"""
    chars = sorted(set("".join(SENTS)))
    e = Ear(cfg, seed=seed, chars=chars)
    e.warmup()
    refs = {c: e.ref(c) & ~e.sil for c in chars}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        for i in rng.permutation(len(SENTS)):
            e.hear(SENTS[i])
    e.c.learning = False
    got = []
    for u in SENTS:                                    # 学过之后还听不听得清真实的声音
        for ch in u:
            for _ in range(SOUND):
                e.tick(ch)
            got.append(e.decode(chars) == ch)
        for _ in range(PAUSE):
            e.tick()
    bias, lines = [], []
    for u in SENTS:
        own = sorted(set(u[1:]) - {u[0]})
        other = sorted({c for v in SENTS if v != u for c in v} - set(u))
        for _ in range(12):
            e.tick()
        base = e.area.region.s.copy()
        for _ in range(SOUND):
            e.tick(u[0])
        d = []
        for _ in range(6):
            e.tick()
            extra = e.area.region.s & ~base
            d.append(np.mean([jac(extra, refs[c]) for c in own]) - np.mean([jac(extra, refs[c]) for c in other]))
        bias.append(np.mean(d))
        lines.append(f"  给'{u[0]}'：每拍偏向自己那句 " + " ".join(f"{x:+.3f}" for x in d))
    score = float(np.mean(bias))
    if verbose:
        print(f"听 {reps} 遍：真实声音听得清 {np.mean(got):.2f}；想起的范围偏向自己那句 {score:+.3f}")
        print("\n".join(lines))
    return score, float(np.mean(got))


# ================================================================ 海马体
def hippo(seed: int = 0, cfg=None, verbose: bool = True, after: int = 300, chain_w: float = 3.0):
    """听觉区（不学，只当输入）→ 链（一对一）→ 海马体。几句话各只听一遍，之后给每句的开头：
    海马体想起的范围偏不偏向自己那句；再随便听 after 拍别的字之后，还记不记得（短期记忆会忘）。
    读法：海马体每个神经元对应听觉区的一个神经元，把亮的那片映回听觉区，和每个字的整片激活比。"""
    from brain.areas import ThoughtArea, hip_cfg
    from brain.net import Path, PathCfg
    chars = sorted(set("".join(SENTS)))
    e = Ear(SenseCfg(internal=PathCfg(eta=0.0)), seed=seed, chars=chars)
    e.warmup()
    ear = e.area
    cx = np.nonzero(ear.cortex)[0]
    hip = ThoughtArea("hip", ear.region.n, cfg or hip_cfg(), seed=seed + 1, ids=cx)
    # 有真实输入时以输入为准：链的强度要明显大于海马体内部学来的推力；安静时空出来的名额才是回放
    chain = Path("ear⇒hip", ear.region, hip.region, PathCfg(w_init=chain_w), np.random.default_rng(seed),
                 pairs=[(int(i), [int(i)]) for i in cx])
    e.c = Circuit([ear, hip], [chain])
    refs = {c: e.ref(c) & ~e.sil for c in chars}
    H = hip.region

    def recall():
        bias = []
        for u in SENTS:
            own = sorted(set(u[1:]) - {u[0]})
            other = sorted({c for v in SENTS if v != u for c in v} - set(u))
            for _ in range(12):
                e.tick()
            for _ in range(SOUND):
                e.tick(u[0])
            d = []
            for _ in range(6):
                e.tick()
                extra = H.s & ~e.sil & ~e.ref(u[0])          # 扣掉安静本身和开头那个字本身
                d.append(np.mean([jac(extra, refs[c]) for c in own]) - np.mean([jac(extra, refs[c]) for c in other]))
            bias.append(float(np.mean(d)))
        return bias

    for u in SENTS:
        e.hear(u)
    e.c.learning = False
    b1 = recall()
    e.c.learning = True
    rng = np.random.default_rng(seed + 7)
    for _ in range(after // (SOUND + PAUSE)):
        e.hear(chars[rng.integers(len(chars))])
    e.c.learning = False
    b2 = recall()
    if verbose:
        print(f"海马体：每句只听一遍后，给开头想起的范围偏向自己那句 {np.mean(b1):+.3f}  （各句 " +
              " ".join(f"{x:+.3f}" for x in b1) + "）")
        print(f"        又随便听了 {after} 拍别的之后            {np.mean(b2):+.3f}  （各句 " +
              " ".join(f"{x:+.3f}" for x in b2) + "）")
    return float(np.mean(b1)), float(np.mean(b2))


# ================================================================ 运动区
def motor(reps: int = 30, seed: int = 0, verbose: bool = True):
    """运动区单独：平时放松；用数据训练几串嘴部动作（每个字 = 一组肌肉，说完推抑制神经元停下）；
    之后只轻推第一个字，看运动区自己能不能把这一串做完、做完停下。"""
    from brain.areas import MotorArea
    words = ["今天天气", "我们去", "你吃饭了吗"]
    v = Voice(sorted(set("".join(words))), seed=seed)
    m = MotorArea(seed=seed)
    c = Circuit([m])

    def run(drive_seq):
        said = []
        for d in drive_seq:
            c.step(ext=None if d is None else {"motor": d})
            said.append(v.mouth({f"m{i}": int(f) for i, f in enumerate(m.force())}))
        return said

    def say_plan(w):
        plan = []
        for ch in w:
            plan += [m.push(v.need[ch])] * SOUND
        plan += [m.stop()] * 2 + [None] * PAUSE                 # 说完：先停下（推抑制神经元），再安静
        return plan

    rest = run([None] * 10)
    for _ in range(reps):
        for w in words:
            run(say_plan(w))
    c.learning = False
    out = []
    for w in words:
        run([None] * 6)
        said = run([m.push(v.need[w[0]], level=1)] + [None] * 16)
        out.append((w, runs([s or "_" for s in said])))
    if verbose:
        print(f"运动区：平时（没有输入）{'放松' if all(s is None for s in rest) else '在动'}")
        for w, s in out:
            print(f"  训练 {reps} 遍后只轻推第一个字（1 档） '{w[0]}'：说出 {s or '（没说）'}    （训练的是 {w}）")
    return out


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "sense"
    if part == "sense":
        for r in (80, 0):
            sense(r)
    elif part == "hippo":
        hippo()
    elif part == "motor":
        motor()
