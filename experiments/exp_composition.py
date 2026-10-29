# -*- coding: utf-8 -*-
"""实验 A：组合（要两个符号同时在才能决定下一个）。

数据：三元组 (a, b, c) 连续流，a,b 是 0..9 里随机的，c = 10 + (a+b) mod 10。
  - 只看 b 时，c 是 10 选 1 均匀 → 两两关联矩阵的上限 = 10%。
  - 看 (a,b) 时 c 唯一确定 → 上限 100%。
只在 c 的位置记准确率（a 本来就不可预测）。

第二段换规则：c = 10 + (a−b) mod 10，看能不能实时改过来（需要计数衰减）。

对照：
  pairwise      只有感觉单元（不长概念单元）= 两两关联矩阵
  concept         意外时长、没用就删，不忘
  concept-noprune 意外时长、不删（看删除的作用）
  concept+decay   意外时长、没用就删，按使用衰减
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.sequence import SeqCfg, SequenceCortex  # noqa: E402


def stream(n_triples: int, rule: str, rng) -> list[tuple[int, bool]]:
    out = []
    for _ in range(n_triples):
        a, b = rng.integers(0, 10, 2)
        c = 10 + ((a + b) % 10 if rule == "add" else (a - b) % 10)
        out += [(int(a), False), (int(b), False), (int(c), True)]
    return out


def run(cfg: SeqCfg, seed: int, n1: int, n2: int, window: int) -> dict:
    rng = np.random.default_rng(seed)
    data = stream(n1, "add", rng) + stream(n2, "sub", rng)
    m = SequenceCortex(V=20, cfg=cfg)
    hits, curve = [], []
    t0 = time.perf_counter()
    for s, scored in data:
        info = m.step(s)
        if scored:
            hits.append(info["hit"])
            if len(hits) % window == 0:
                curve.append(float(np.mean(hits[-window:])))
    dt = (time.perf_counter() - t0) / len(data) * 1e3
    h = np.array(hits, dtype=float)
    return {
        "curve": curve,
        "rule1_last": float(h[n1 - window:n1].mean()),
        "rule2_last": float(h[-window:].mean()),
        # 换规则后多少个三元组才回到 90%
        "relearn_triples": next((i * window for i, v in enumerate(curve[n1 // window:]) if v >= 0.9), None),
        "units": m.n_units,
        "proven": m.n_proven,
        "grown": m.n_grown,
        "ms_per_step": dt,
    }


def main():
    n1, n2, window = 3000, 3000, 100
    variants = {
        "pairwise": SeqCfg(grow=False),
        "concept": SeqCfg(),
        "concept-noprune": SeqCfg(prune=False),
        "concept+decay": SeqCfg(use_decay=0.7),
    }
    res = {}
    for name, cfg in variants.items():
        runs = [run(cfg, seed, n1, n2, window) for seed in range(3)]
        res[name] = {
            "rule1_last": [r["rule1_last"] for r in runs],
            "rule2_last": [r["rule2_last"] for r in runs],
            "relearn_triples": [r["relearn_triples"] for r in runs],
            "units": [r["units"] for r in runs],
            "proven": [r["proven"] for r in runs],
            "ms_per_step": float(np.mean([r["ms_per_step"] for r in runs])),
            "curve_seed0": runs[0]["curve"],
        }
        r = res[name]
        print(f"{name:14s} 规则1末段 {np.mean(r['rule1_last']):.3f}  规则2末段 {np.mean(r['rule2_last']):.3f}  "
              f"换规则后回到90%: {r['relearn_triples']}  存活 {r['units']} 转正 {r['proven']}  {r['ms_per_step']:.3f} ms/拍")
    out = Path(__file__).resolve().parents[1] / "results" / "composition.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
