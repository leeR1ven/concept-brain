# -*- coding: utf-8 -*-
"""实验 F：条件反射——前额叶自己学会"看到 C 之后要缩回"，经回程提前动肌肉。

不加任何新机制：本能（强光 → 缩回）+ 运动神经元经链进前额叶 + 先亮 → 后亮的实时学习 + 回程。
  训练：灰 6 拍 → C 3 拍 → 强光 3 拍（本能缩回）→ 灰 8 拍；重复 REPS 遍。
  测试：灰 10 拍 → C 3 拍 → 灰 10 拍（**不给强光**）；记录每拍"缩回"力度。
对照：
  untrained   没训练过的同一个脑看 C
  other       训练过的脑看一张没和强光配对过的画面 D
另外记录：强光撤掉后肌肉能不能自己放松（回程会不会把肌肉锁住）。
先天连线（用户建议手动安排）："缩回"的肌肉神经元 → 缩回组自己的抑制神经元（缩一下就停）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.brain import Brain  # noqa: E402
from brain.motor import Motor  # noqa: E402
from brain.pfc import PFCCfg  # noqa: E402
from brain.vision import SheetCfg, VisionSheet  # noqa: E402

H = W = 12
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SELF_STOP = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0   # 缩回肌肉 → 缩回抑制 的先天连线权重（0 = 不接）
ATTN = sys.argv[3] if len(sys.argv) > 3 else "inhib_pair"
ETA_H = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0      # 海马体学习速度


def make(seed):
    v = VisionSheet(H, W, SheetCfg(layers=(6,)), seed=seed)
    m = Motor(["withdraw", "approach"], rivals=[("withdraw", "approach")])
    if SELF_STOP:                                          # 先天："缩一下就停"
        m.wire(m.muscle_ids("withdraw"), m.inh_id("withdraw"), SELF_STOP)
    gray = np.full((H, W), 4, dtype=int)
    base = int(v.respond(v.encode(gray)).sum()) + len(m.act)
    b = Brain(v, m, pfc_cfg=PFCCfg(attention=ATTN, target=base + 80, eta_hippo=ETA_H))
    srcs = []
    for i, inp in enumerate(v.inputs):
        t, pol = (inp // 2) % v.cfg.levels, inp % 2
        if v.pos[i][1] < W // 2 and ((pol == 0) & (t >= 6)).sum() >= v.need[i]:
            srcs.append(i)
    b.wire_reflex("withdraw", srcs, gain=1.5)
    return b, gray


def pattern(rng):
    low = rng.integers(0, 4, size=(3, 3))                  # 暗的花纹（不触发强光本能）
    return np.kron(low, np.ones((4, 4), dtype=int))


def run(b, seq):
    return [b.step(img)["withdraw"] for img in seq]


def main():
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        b, gray = make(seed)
        C, D = pattern(rng), pattern(rng)
        flash = gray.copy()
        flash[:, : W // 2] = 10
        test_C = [gray] * 10 + [C] * 3 + [gray] * 10
        test_D = [gray] * 10 + [D] * 3 + [gray] * 10
        b0, _ = make(seed)
        b0.pfc.cfg.learn = False
        untrained = run(b0, test_C)
        after_flash = []
        for r in range(REPS):
            f = run(b, [gray] * 6 + [C] * 3 + [flash] * 3 + [gray] * 8)
            if r == 0:
                after_flash = f[12:]                         # 强光撤掉后的 8 拍
        b.pfc.cfg.learn = False
        tc = run(b, test_C)
        td = run(b, test_D)
        rows.append((untrained, tc, td, after_flash))
        print(f"种子 {seed}")
        print(f"  没训练看 C   : {untrained[8:]}")
        print(f"  训练后看 C   : {tc[8:]}      ← C 在第 3~5 个数")
        print(f"  训练后看 D   : {td[8:]}")
        print(f"  第一次强光撤掉后 8 拍的缩回力度: {after_flash}")
    hitC = np.mean([max(r[1][10:]) > 0 for r in rows])
    hitD = np.mean([max(r[2][10:]) > 0 for r in rows])
    hit0 = np.mean([max(r[0][10:]) > 0 for r in rows])
    print(f"\n看 C 后提前缩回：训练后 {hitC:.2f}，没训练 {hit0:.2f}；看 D 后缩回 {hitD:.2f}")


if __name__ == "__main__":
    main()
