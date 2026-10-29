# -*- coding: utf-8 -*-
"""实验 D：长时间看相似画面 → 常亮的神经元分裂 → 对这类画面分辨得更细。

场景：一张平滑的底图；20 个变体，各自只在一个 3×3 小块上亮一档（细微差别）。
每次"看"= 某个变体 + 噪声（每个像素以 noise 的概率 ±1 档）。
经历：每轮看 400 次（随机变体），统计使用率，做一次结构调整；共 ROUNDS 轮。
测量（每轮后）：
  - 相似变体：每个变体 20 次"看"做原型（活动均值），另 20 次测；最近原型（汉明距离）的准确率；
    分离度 = 最近的别的原型距离 / 本类测试样本到自己原型的平均距离（无量纲，不受神经元数影响）；
  - 别的场景：20 张没看过的不相关底图（同样噪声），同样测——看结构调整有没有伤到别的。
方案：
  none          不做结构调整
  split_on      用户机制：常亮的分裂（每轮拆常亮者的 10%）+ 从不亮的删除
  split_on_keep 同上，但不删除（看删除的作用）
  split_random  对照：拆同样多个随机挑的神经元（+ 删除）
  add_random    对照：新加同样多个随机神经元（+ 删除）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.vision import SheetCfg, VisionSheet  # noqa: E402

H = W = 16
LEVELS = 10
N_VAR, VIEWS_PROTO, VIEWS_TEST = 20, 20, 20
ROUNDS, VIEWS_PER_ROUND = 20, 400
NOISE = 0.15
# 命令行：python exp_refine.py [每层每像素神经元数...]，例如 `8 8 8` = 3 层（默认单层 8）
LAYERS = tuple(int(a) for a in sys.argv[1:]) or (8,)
MODES = ["none", "split_on", "split_on_keep", "split_random", "add_random"] if len(LAYERS) == 1 \
    else ["none", "split_on", "split_random"]
SEEDS = 5 if len(LAYERS) == 1 else 3


def smooth_base(rng):
    low = rng.uniform(2, 8, size=(4, 4))
    img = np.kron(low, np.ones((4, 4)))
    return np.clip(np.round(img + rng.normal(0, 0.5, img.shape)), 1, LEVELS - 1).astype(int)


def variants(base, rng):
    out = []
    for _ in range(N_VAR):
        v = base.copy()
        y, x = rng.integers(0, H - 3), rng.integers(0, W - 3)
        v[y:y + 3, x:x + 3] = np.clip(v[y:y + 3, x:x + 3] + 1, 0, LEVELS)
        out.append(v)
    return out


def view(img, rng):
    flip = rng.random(img.shape) < NOISE
    return np.clip(img + flip * rng.choice([-1, 1], img.shape), 0, LEVELS)


def evaluate(sheet, imgs, rng):
    alive = sheet._alive
    protos, tests = [], []
    for im in imgs:
        protos.append(np.mean([sheet.respond(sheet.encode(view(im, rng)))[alive] for _ in range(VIEWS_PROTO)], axis=0))
        tests.append([sheet.respond(sheet.encode(view(im, rng)))[alive].astype(float) for _ in range(VIEWS_TEST)])
    P = np.array(protos)
    hit, within, between = [], [], []
    for i, ts in enumerate(tests):
        for a in ts:
            d = np.abs(P - a).sum(axis=1)
            hit.append(np.argmin(d) == i)
            within.append(d[i])
        dp = np.abs(P - P[i]).sum(axis=1)
        dp[i] = np.inf
        between.append(dp.min())
    return float(np.mean(hit)), float(np.mean(between) / np.mean(within))


def run(mode, seed):
    rng = np.random.default_rng(seed)
    base = smooth_base(rng)
    sim = variants(base, rng)
    other = [smooth_base(rng) for _ in range(N_VAR)]
    sheet = VisionSheet(H, W, SheetCfg(levels=LEVELS, layers=LAYERS, delete=(mode != "split_on_keep"),
                                       split_frac=(0.1,) if len(LAYERS) == 1 else (0.02, 0.05, 0.1)), seed=seed)
    rows = []

    def record(r, info):
        acc, sep = evaluate(sheet, sim, np.random.default_rng(1000 + seed))
        acc_o, sep_o = evaluate(sheet, other, np.random.default_rng(2000 + seed))
        rows.append({"round": r, "sim_acc": acc, "sim_sep": sep, "other_acc": acc_o, "other_sep": sep_o,
                     "neurons": sheet.n_alive, **info})

    record(0, {})
    for r in range(1, ROUNDS + 1):
        use = np.zeros(len(sheet.inputs))
        for _ in range(VIEWS_PER_ROUND):
            use += sheet.respond(sheet.encode(view(sim[rng.integers(N_VAR)], rng)))
        info = {} if mode == "none" else sheet.restructure(use / VIEWS_PER_ROUND, mode.replace("_keep", ""))
        record(r, info)
    return rows


def main():
    modes = MODES
    res = {}
    for mode in modes:
        t0 = time.perf_counter()
        runs = [run(mode, s) for s in range(SEEDS)]
        keys = [k for k in runs[0][-1] if isinstance(runs[0][-1][k], (int, float))]
        rows = []
        for i in range(len(runs[0])):
            rows.append({k: float(np.mean([r[i].get(k, 0) for r in runs])) for k in keys})
        res[mode] = rows
        print(f"\n{mode}  ({time.perf_counter() - t0:.0f}s)")
        print("  轮   相似:准确率 分离度   别的场景:准确率 分离度   神经元   常亮  拆/加  删")
        for r in rows[::4] + ([rows[-1]] if len(rows) % 4 != 1 else []):
            print(f"  {int(r['round'])}     {r['sim_acc']:.3f}  {r['sim_sep']:.3f}        {r['other_acc']:.3f}  {r['other_sep']:.3f}"
                  f"   {int(r['neurons']):6d}  {int(r.get('always_on', 0)):5d}  {int(r.get('changed', 0)):4d}  {int(r.get('never', 0)):4d}")
    name = "refine.json" if len(LAYERS) == 1 else f"refine_{len(LAYERS)}layer.json"
    out = Path(__file__).resolve().parents[1] / "results" / name
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
