# -*- coding: utf-8 -*-
"""实验 C：短期记忆（海马体，快）与长期记忆（皮层自己学，慢）。

用户的机制：海马体和皮层是同一种学习，只是海马体权重涨得快、降得快；
长期记忆就是皮层自己慢慢学，没有别的规则。

世界：4,000 个概念，每个 32 个老搭档（先天 / 早已学会的知识，0.5~0.95）。
流：一句 10 个概念、一拍一个（每拍刚亮一个）；共 3,000 句。
  - 一次对 20 对：只在开头出现一次；
  - 重复对 20 对：第 0~1,990 句里每 100 句各出现一次（共 20 次），之后不再出现；
  - 其余是随机填充句。
  每对 (A,B) 放在句中第 4、第 8 位（中间隔 3 个）。
测：只给 A，驱动 = 老知识 + 皮层新学的 + 海马体回程；B 进前 25 算想起来。
老知识：50 个探针概念原来最强的 10 个老搭档还有几个在前 25。

方案：
  cortex-only   只有皮层慢学（η=0.05）
  hippo-only    只有海马体快学（η=1.0，半衰期 ~2,000 拍），皮层不学
  hippo+cortex  两个都有（用户的架构）
  fast-cortex   对照：没有海马体，皮层直接快学、不降
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.global_decay import GlobalDecay  # noqa: E402
from brain.plastic import Plastic, PlasticCfg  # noqa: E402

N, PRIOR_K, TOP, N_PAIRS, SENT = 4000, 32, 25, 20, 10
TOTAL, REP_EVERY, REP_UNTIL = 3000, 100, 2000
CHECK = [1, 50, 500, 1990, 2100, 3000]
HIPPO_HALF = 2000
VARIANTS = {   # 名字: (皮层 η, 海马体 η)
    "cortex-only": (0.05, None),
    "hippo-only": (None, 1.0),
    "hippo+cortex": (0.05, 1.0),
    "fast-cortex": (1.0, None),
}


def run(variant, seed):
    rng = np.random.default_rng(seed)
    prior = np.zeros((N, N), dtype=np.float32)
    for w in range(N):
        nb = rng.choice(N, PRIOR_K, replace=False)
        nb = nb[nb != w]
        prior[w, nb] = rng.uniform(0.5, 0.95, len(nb))
    gd = GlobalDecay(T=200)
    eta_c, eta_h = VARIANTS[variant]
    cortex = hippo = None
    if eta_c is not None:
        cortex = Plastic(N, PlasticCfg(eta=eta_c))
        gd.register("cortex", cortex.edge_rows, 1.0)          # 皮层降得慢：实验时长内视为不降
    if eta_h is not None:
        hippo = Plastic(N, PlasticCfg(eta=eta_h))
        gd.register("hippocampus", hippo.edge_rows, GlobalDecay.gamma_for_half_life(HIPPO_HALF))
    parts = [x for x in (cortex, hippo) if x is not None]

    pairs = rng.choice(N, size=(2 * N_PAIRS, 2), replace=False)
    once, rep = pairs[:N_PAIRS], pairs[N_PAIRS:]
    probes = rng.choice(N, 50, replace=False)
    old_top = {int(p): np.argsort(-prior[p])[:10] for p in probes}

    def scores(a):
        s = prior[a].astype(np.float64).copy()
        for part in parts:
            for j, w in part.recall(a).items():
                s[j] += w
        s[a] = -np.inf
        return s

    def recall(ps):
        hit = []
        for a, b in ps:
            s = scores(int(a))
            hit.append((s > s[int(b)]).sum() < TOP)
        return float(np.mean(hit))

    def sentence(pair=None):
        f = rng.choice(N, SENT, replace=False)
        if pair is not None:
            f[3], f[7] = pair
        for x in f:
            for part in parts:
                part.step([int(x)])
            gd.tick()

    out, t0 = [], time.perf_counter()
    for k in range(TOTAL):
        if k < N_PAIRS:
            sentence(once[k])
        elif k < REP_UNTIL and (k % REP_EVERY) < N_PAIRS:
            sentence(rep[k % REP_EVERY])
        else:
            sentence()
        if k + 1 in CHECK:
            kept = []
            for p in probes:
                top = set(np.argsort(-scores(int(p)))[:TOP].tolist())
                kept.append(sum(int(o) in top for o in old_top[int(p)]))
            out.append({"sentence": k + 1, "once": recall(once), "repeated": recall(rep),
                        "old_kept_of10": float(np.mean(kept))})
    return out, (time.perf_counter() - t0) / (TOTAL * SENT) * 1e3


def main():
    res = {}
    for v in VARIANTS:
        runs = [run(v, s) for s in range(3)]
        rows = [{k: float(np.mean([r[0][i][k] for r in runs])) for k in runs[0][0][i]} for i in range(len(CHECK))]
        res[v] = {"rows": rows, "ms_per_step": float(np.mean([r[1] for r in runs]))}
        print(f"\n{v}  ({res[v]['ms_per_step']:.3f} ms/拍，含测试)")
        print("  第几句   一次对想起   重复对想起   老搭档保留")
        for r in rows:
            print(f"  {int(r['sentence']):6d}     {r['once']:.2f}         {r['repeated']:.2f}       {r['old_kept_of10']:4.1f}/10")
    out = Path(__file__).resolve().parents[1] / "results" / "hippocampus.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
