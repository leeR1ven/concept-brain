# -*- coding: utf-8 -*-
"""记住一段连续的声音：反复听同一句话，之后只给开头的字，看听觉区能不能"在心里"顺着放下去。

  学：反复听"今天天气很好"（别人的声音，每字 2 拍 + 1 拍安静，句末安静；中间穿插别的话）。
  测（不学习）：只听"今"，然后一直安静；海马体 / 前额叶经回程送回听觉区（在心里听到）。
  读：听觉区第 2、3 层每一拍的状态，和每个字真的听到时的状态比（Jaccard），最像哪个字 / 安静。
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


def run(train: bool, steps: int = 24):
    t = Talker(sorted(set(SENT + OTHER)))
    b = t.brain
    ear = b.sensors["ear"]
    t.warmup()
    deep = np.nonzero(ear._layer >= 1)[0]
    ref = {c: ear.respond(ear.encode(t.voice.hear(c, "other")))[deep] for c in sorted(set(SENT + OTHER))}
    ref["_"] = ear.respond(ear.encode(t.voice.hear(None, "other")))[deep]          # 安静
    rng = np.random.default_rng(0)
    if train:
        for _ in range(REPS):
            t.listen(SENT)
            t.listen(OTHER[: rng.integers(3, 8)])
    b.learning = False
    for _ in range(6):
        t.tick()
    for _ in range(2):
        t.tick(other=SENT[0])
    out = []
    for _ in range(steps):
        t.tick()
        a = ear.respond(ear.encode(t.voice.hear(None, "other")), b._sense_topdown()["ear"])[deep]
        extra = a & ~ref["_"]                       # 回程在安静之外额外点亮的
        if extra.sum() < 3:
            out.append(("_", int(extra.sum())))
            continue
        sc = {c: jac(extra, r & ~ref["_"]) for c, r in ref.items() if c != "_"}
        best = max(sc, key=sc.get)
        out.append((best, int(extra.sum())))
    return out


if __name__ == "__main__":
    for name, tr in ((f"听过 {REPS} 遍", True), ("没听过", False)):
        out = run(tr)
        print(f"{name}：只给'{SENT[0]}'之后，听觉区每拍在心里听到的（_ = 安静；括号里是额外亮了几个）：")
        print("   ", " ".join(f"{c}({n})" for c, n in out))
