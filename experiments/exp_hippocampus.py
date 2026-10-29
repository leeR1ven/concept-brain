# -*- coding: utf-8 -*-
"""实验 C：海马体短期记忆（快涨快降）vs 皮层直接快学（不忘）。

世界：4,000 个概念，每个 32 个先天老搭档（慢权重 0.5~0.95，皮层里，实验中不改）。
流：一句 10 个概念、一拍一个（每拍刚亮一个）。
  - 开头 20 句各藏一对新 (A,B)（A 在第 4 位，B 在第 8 位，中间隔 3 个）；
  - 之后是随机填充句。
在第 0/10/50/200/1000/3000 句（学完之后）测：
  - 新记忆：只给 A，驱动 = 慢权重 + 海马回程；B 进前 25 算想起来；
  - 老知识：50 个探针概念，原来最强的 10 个老搭档还有几个在前 25（被新边挤掉就掉）。
方案：
  no-hippo        只有皮层慢权重（什么也不学）
  cortex-fast     没有海马体，快学习直接写进皮层、不忘（decay=1.0）
  hippo-hl500     海马体，整体降权 T=200，γ 折合半衰期约 500 拍（50 句）
  hippo-hl5000    海马体，整体降权 T=200，γ 折合半衰期约 5,000 拍（500 句）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.global_decay import GlobalDecay  # noqa: E402
from brain.hippocampus import Hippocampus  # noqa: E402

N, PRIOR_K, TOP, N_PAIRS, SENT = 4000, 32, 25, 20, 10
CHECK = [0, 10, 50, 200, 1000, 3000]




def scores(prior, hip, a):
    s = prior[a].astype(np.float64).copy()
    if hip is not None:
        for j, w in hip.recall(a).items():
            s[j] += w
    s[a] = -np.inf
    return s


def run(variant, seed):
    rng = np.random.default_rng(seed)
    prior = np.zeros((N, N), dtype=np.float32)
    for w in range(N):
        nb = rng.choice(N, PRIOR_K, replace=False)
        nb = nb[nb != w]
        prior[w, nb] = rng.uniform(0.5, 0.95, len(nb))
    hip, gd = None, GlobalDecay(T=200)
    half = {"cortex-fast": None, "hippo-hl500": 500, "hippo-hl5000": 5000}
    if variant in half:
        hip = Hippocampus(N)
        g = 1.0 if half[variant] is None else GlobalDecay.gamma_for_half_life(half[variant])
        gd.register("hippocampus", hip.edge_rows, g)

    pairs = rng.choice(N, size=(N_PAIRS, 2), replace=False)
    probes = rng.choice(N, 50, replace=False)
    old_top = {int(p): np.argsort(-prior[p])[:10] for p in probes}

    def feed(sent):
        if hip is not None:
            for x in sent:
                hip.step([int(x)])
                gd.tick()

    t0, steps = time.perf_counter(), 0
    for a, b in pairs:
        f = rng.choice(N, SENT - 2, replace=False)
        feed(list(f[:3]) + [a] + list(f[3:6]) + [b] + list(f[6:]))
        steps += SENT
    out, done = [], 0
    for chk in CHECK:
        while done < chk:
            feed(rng.choice(N, SENT, replace=False))
            done += 1
            steps += SENT
        rk = []
        for a, b in pairs:
            s = scores(prior, hip, int(a))
            rk.append(int((s > s[int(b)]).sum()))
        rec = float(np.mean(np.array(rk) < TOP))
        kept = []
        for p in probes:
            top = set(np.argsort(-scores(prior, hip, int(p)))[:TOP].tolist())
            kept.append(sum(int(o) in top for o in old_top[int(p)]))
        out.append({"after_sentences": chk, "recall@25": rec, "median_rank": float(np.median(rk)), "old_kept_of10": float(np.mean(kept)),
                    "hippo_edges": hip.n_edges() if hip else 0})
    ms = (time.perf_counter() - t0) / steps * 1e3
    return out, ms


def main():
    res = {}
    for v in ("no-hippo", "cortex-fast", "hippo-hl500", "hippo-hl5000"):
        runs = [run(v, s) for s in range(3)]
        rows = []
        for k, chk in enumerate(CHECK):
            rows.append({"after_sentences": chk,
                         "recall@25": float(np.mean([r[0][k]["recall@25"] for r in runs])),
                         "median_rank": float(np.mean([r[0][k]["median_rank"] for r in runs])),
                         "old_kept_of10": float(np.mean([r[0][k]["old_kept_of10"] for r in runs])),
                         "hippo_edges": int(np.mean([r[0][k]["hippo_edges"] for r in runs]))})
        res[v] = {"rows": rows, "ms_per_step": float(np.mean([r[1] for r in runs]))}
        print(f"\n{v}  ({res[v]['ms_per_step']:.3f} ms/拍，含测试)")
        print("  学完后第几句   想起来   中位名次   老搭档保留   海马边数")
        for r in rows:
            print(f"  {r['after_sentences']:8d}     {r['recall@25']:.2f}   {r['median_rank']:7.1f}   {r['old_kept_of10']:5.1f}/10   {r['hippo_edges']:8d}")
    out = Path(__file__).resolve().parents[1] / "results" / "hippocampus.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
