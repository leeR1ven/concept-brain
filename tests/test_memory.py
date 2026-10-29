# -*- coding: utf-8 -*-
"""记住一段连续的声音：反复听同一串字，之后只给开头，看前额叶能不能顺着"放"下去。

  学：反复听"今天天气很好"（别人的声音，每字 2 拍 + 1 拍安静，句末安静 4 拍）。
  测（不学习）：只听"今"，然后一直安静；记录之后每拍前额叶的状态。
  比：听这句话时，每个字那几拍前额叶的状态（参照）。安静后每拍的状态最像哪个字的参照 → 解码出来的序列。
  对照：没听过这句话的同一个脑。

python tests/test_memory.py [重复遍数]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talk import Talker  # noqa: E402

SENT = "今天天气很好"
OTHER = "我们去公园散步吧你吃饭了吗"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def jac(x, y):
    u = (x | y).sum()
    return (x & y).sum() / u if u else 0.0


def run(train: bool):
    t = Talker(sorted(set(SENT + OTHER)))
    b = t.brain
    t.warmup()
    if train:
        for _ in range(REPS):
            t.listen(SENT)
            t.listen(OTHER[: 1 + np.random.default_rng(_).integers(3, 8)])   # 中间穿插别的话
    b.learning = False
    # 参照：听这句话时每个字那几拍的前额叶状态（只看前额叶自己的活动里和耳朵无关的部分也行，这里看整体）
    ref = {}
    for ch in SENT:
        states = []
        for _ in range(2):
            t.tick(other=ch)
            states.append(b.pfc.a > 0)
        t.tick()
        ref.setdefault(ch, []).append(np.any(states, axis=0))
    for _ in range(6):
        t.tick()
    ref = {ch: np.any(v, axis=0) for ch, v in ref.items()}
    # 测：只给开头的字，然后安静
    for _ in range(2):
        t.tick(other=SENT[0])
    out = []
    for _ in range(18):
        t.tick()
        a = b.pfc.a > 0
        scores = {ch: jac(a, r) for ch, r in ref.items()}
        best = max(scores, key=scores.get)
        out.append((best, round(scores[best], 2), int(a.sum())))
    return out


if __name__ == "__main__":
    for name, tr in (("听过 %d 遍" % REPS, True), ("没听过", False)):
        out = run(tr)
        print(f"{name}：只给'{SENT[0]}'之后每拍最像的字(相似度,前额叶亮几个)：")
        print("   ", "  ".join(f"{c}({s},{n})" for c, s, n in out))
