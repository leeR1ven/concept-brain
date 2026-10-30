# -*- coding: utf-8 -*-
"""记住一段连续的声音：反复听一句话，之后只给开头，看听觉区能不能顺着往下放（表征坍缩），
前额叶的大致想法是不是跟着慢慢移动。

读法（宽松，看范围不逐字卡死）：
  听觉区 —— 每一拍皮层里"比安静时多亮"的那些神经元，和每个字自己的声音特征最像哪个字；
  前额叶 —— 每一拍前额叶亮着的那片，和听这句话时每个字那一拍前额叶亮的那片最像哪个字。
得分：给"今"之后，放出来的字里按顺序对上"天气很好"几个（最长公共子序列）。对照：没听过这句话的脑。

  python tests/test_memory.py [遍数]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talk import Talker  # noqa: E402

SENT = "今天天气很好"
OTHER = "我们去公园散步吧你吃饭了吗"


def jac(x, y):
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 0.0


def lcs(a: str, b: str) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            d[i + 1][j + 1] = d[i][j] + 1 if x == y else max(d[i][j + 1], d[i + 1][j])
    return d[-1][-1]


def run(reps: int, steps: int = 24, seed: int = 0, cfg=None):
    chars = sorted(set(SENT + OTHER))
    t = Talker(chars, cfg=cfg, seed=seed)
    b, v = t.brain, t.voice
    t.warmup(150)
    cx = b.cortex
    ref = {c: b.sheet.respond(b.sheet.encode(v.hear(c, "other")))[cx] for c in chars}
    sil = b.sheet.respond(b.sheet.encode(v.hear(None, "other")))[cx]
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        t.listen(SENT)
        t.listen("".join(rng.choice(list(OTHER), rng.integers(3, 7))))
    b.learning = False
    pref = {}
    for _ in range(8):
        t.tick()
    for ch in SENT:                                   # 前额叶参照：听这句话时每个字那一拍的前额叶
        t.tick(other=ch)
        t.tick(other=ch)
        pref.setdefault(ch, b.pfc.s.copy())
        t.tick()
    for _ in range(12):
        t.tick()
    t.tick(other=SENT[0])
    t.tick(other=SENT[0])
    ear_seq, pfc_seq = [], []
    for _ in range(steps):
        t.tick()
        extra = b.ear.s[cx] & ~sil
        if extra.sum() < 5:
            ear_seq.append("_")
        else:
            sc = {c: jac(extra, r & ~sil) for c, r in ref.items()}
            ear_seq.append(max(sc, key=sc.get))
        ps = {c: jac(b.pfc.s, r) for c, r in pref.items()}
        pfc_seq.append(max(ps, key=ps.get) if max(ps.values()) > 0.05 else "_")
    flat = "".join(c for i, c in enumerate(ear_seq) if c != "_" and (i == 0 or c != ear_seq[i - 1]))
    return lcs(flat, SENT[1:]), "".join(ear_seq), "".join(pfc_seq)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for r in (reps, 0):
        s, e, p = run(r)
        print(f"听 {r:>2} 遍后只给'今'：得分 {s}/5   听觉区 {e}   前额叶 {p}")
