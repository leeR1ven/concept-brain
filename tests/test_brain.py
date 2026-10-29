# -*- coding: utf-8 -*-
"""整脑冒烟测试（几秒）：python tests/test_brain.py

  1. 没有刺激：肌肉一直放松（抑制占上风）。
  2. 强光（左半边很亮）：本能"缩回"被触发。
  3. 本能和理性都想缩回：协作，力度更大。
  4. 理性强烈想"靠近"（和缩回冲突）：理性压过本能；理性只是弱弱地想靠近：本能赢。
  5. 力度单调分级。
  6. 结构调整后（3 层，所有层都可分裂 / 删除）：每个活着的皮层神经元恰好有一套活着的跟随者
     （链 / 前额叶 / 海马体），删掉 / 被拆的父单元的跟随者都没了；本能来源由子单元继承；
     没有任何入边指向死神经元（下一层已改接子单元或断开）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.brain import Brain  # noqa: E402
from brain.motor import Motor  # noqa: E402
from brain.vision import SheetCfg, VisionSheet  # noqa: E402

H = W = 8


def make():
    v = VisionSheet(H, W, SheetCfg(layers=(4, 4, 4), split_frac=(0.1, 0.2, 0.3)), seed=0)
    m = Motor(["withdraw", "approach"], rivals=[("withdraw", "approach")])
    b = Brain(v, m)
    # 先天：第 1 层里左半边、对"很亮"敏感的神经元（入边里至少 need 条是高档位的正线）直连"缩回"
    srcs = []
    for i, inp in enumerate(v.inputs):
        if v.layer[i] != 0:
            continue
        y, x = v.pos[i]
        t = (inp // 2) % v.cfg.levels
        pol = inp % 2
        if x < W // 2 and ((pol == 0) & (t >= 6)).sum() >= v.need[i]:
            srcs.append(i)
    b.wire_reflex("withdraw", srcs, gain=1.5)
    return b, srcs


def gray():
    return np.full((H, W), 4, dtype=int)


def flash():
    img = gray()
    img[:, : W // 2] = 10
    return img


def test_relaxed_and_reflex():
    b, srcs = make()
    assert len(srcs) > 5, f"本能来源太少：{len(srcs)}"
    for _ in range(30):
        f = b.step(gray())
    assert b.motor.relaxed, f"没刺激时肌肉应放松：{f}"
    f = b.step(flash())
    assert f["withdraw"] > 0 and f["approach"] == 0, f"强光应触发缩回：{f}"
    return f["withdraw"]


def test_cooperate_and_compete(base_force):
    b, _ = make()
    f = b.step(flash(), reason={"withdraw": 0.5})
    assert f["withdraw"] > base_force, f"协作应更有力：{f} vs {base_force}"
    b, _ = make()
    f = b.step(flash(), reason={"approach": 2.5})
    assert f["approach"] > 0 and f["withdraw"] == 0, f"理性强时应压过本能：{f}"
    b, _ = make()
    f = b.step(flash(), reason={"approach": 0.4})
    assert f["withdraw"] > 0 and f["approach"] == 0, f"理性弱时本能赢：{f}"


def test_graded():
    m = Motor(["a"])
    forces = [m.step({"a": d})["a"] for d in np.linspace(0, 2, 21)]
    assert all(x <= y for x, y in zip(forces, forces[1:])) and forces[0] == 0 and forces[-1] > 5, forces


def test_followers():
    b, _ = make()
    rng = np.random.default_rng(0)
    for _ in range(3):
        for _ in range(100):
            img = gray() + (rng.random((H, W)) < 0.1) * rng.choice([-1, 1], (H, W))
            b.step(img)
        info = b.restructure()
        assert info["split"] or info["deleted"], "这一轮应有结构变化"
    v = b.vision
    alive_c = set(np.nonzero(v._alive)[0].tolist())
    alive_f = {b.f_src[f][1] for f in range(len(b.f_src)) if b.f_alive[f] and b.f_src[f][0] == "vision"}
    assert alive_c == alive_f, f"跟随者对不上：皮层 {len(alive_c)} vs 跟随者 {len(alive_f)}"
    assert b.pfc.n == len(b.f_src) == b.pfc.hippo.W.shape[0] == b.pfc.cortex.W.shape[0]
    for f in range(len(b.f_src)):
        if not b.f_alive[f]:
            assert not b.pfc.alive[f]
            assert not b.pfc.hippo.W[f].any() and not b.pfc.hippo.W[:, f].any()
            assert not b.pfc.cortex.W[f].any() and not b.pfc.cortex.W[:, f].any()
    for srcs in b.reflex.values():
        assert srcs <= alive_c, "本能来源里不能有死神经元"
    refs = v._idx[v._alive]
    neur = refs[refs >= v.off] - v.off
    assert v._alive[neur].all(), "有入边指向死神经元"
    layers_changed = {v.layer[p] for p, _ in info["split"]} | {v.layer[i] for i in info["deleted"]}
    assert len(layers_changed) >= 2, f"分裂 / 删除应发生在多层：{layers_changed}"
    b.step(flash())   # 结构变了之后还能正常走
    return len(alive_c)


if __name__ == "__main__":
    force = test_relaxed_and_reflex()
    print(f"1-2 放松 / 本能缩回：ok（力度 {force}）")
    test_cooperate_and_compete(force)
    print("3-4 协作 / 竞争：ok")
    test_graded()
    print("5   力度分级：ok")
    n = test_followers()
    print(f"6   跟随增删（3 层）：ok（结构调整后皮层神经元 {n}，跟随者一一对应）")
