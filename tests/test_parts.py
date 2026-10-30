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
        for i in a.featureless():
            a.inner.cut_src([int(i)])
            a.inner.redirect(int(i), [])
            a.cortex[i] = False
        a.clear_usage()
        self.c.learning = True


SENTS = ["今天天气很好", "我们去公园散步", "你吃饭了吗", "明年再来看花"]


def sense(reps: int = 80, seed: int = 0, cfg: SenseCfg | None = None, verbose: bool = True):
    """几句话各听 reps 遍（顺序打乱），之后分别只给每句的第一个字，看听觉区想起的是不是这句话后面的字。
    读法看整片激活：每一拍比安静时多亮的那一片，和每个字（比安静多亮的那一片）有多像。"""
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
    own = []
    lines = []
    for u in SENTS:
        for _ in range(12):
            e.tick()
        base = e.area.region.s.copy()
        for _ in range(SOUND):
            e.tick(u[0])
        seq = []
        for _ in range(6):
            e.tick()
            extra = e.area.region.s & ~base
            if extra.sum() < 20:
                seq.append("_")
                continue
            sims = {c: jac(extra, refs[c]) for c in chars if c != u[0]}
            best = max(sims, key=sims.get)
            seq.append(best)
            own.append(best in u[1:])
        lines.append(f"  给'{u[0]}'：{''.join(seq)}   （这句后面是 {u[1:]}）")
    score = float(np.mean(own)) if own else 0.0
    if verbose:
        print(f"听 {reps} 遍：真实声音听得清 {np.mean(got):.2f}；想起的是自己那句后面的字 {score:.2f}"
              f"（{len(own)} 拍有回忆；随机猜约 {np.mean([len(u) - 1 for u in SENTS]) / (len(chars) - 1):.2f}）")
        print("\n".join(lines))
    return score, float(np.mean(got))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "sense"
    if part == "sense":
        for r in (80, 0):
            sense(r)
