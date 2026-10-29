# -*- coding: utf-8 -*-
"""海马体：短期记忆。

机制（用户 2026-10-29 给出，取代旧的"时间神经元"方案）：
  - 海马体和前额叶一样，经**链式专用线**接收各区信号：每个概念单元在海马体里有一个
    对应单元（1:1），回程同样走链回到概念层。
  - 海马体内部神经元之间的权重**涨得快、降得也快**。
  - 不负责逻辑思考，只负责记住短期记忆；不改皮层的慢权重。

实现：
  - 只有"刚亮"的单元参与学习：刚亮的 j ← 最近刚亮过、痕迹还在的 i（先 → 后），
    痕迹每拍乘 trace_decay（常亮的不会反复"刚亮"，所以不会和所有人连上）。
  - 增长：Δw = eta · trace_i / (1 + w)（越涨越难涨、无上限）。
  - 下降：不自己做——走全脑共用的**整体降权**（brain/global_decay.py），
    只是海马体那一组的 γ 更小（降得更快）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HippoCfg:
    eta: float = 1.0
    trace_decay: float = 0.95    # "刚亮过"的痕迹每拍保留多少（约覆盖一句 / 一段经历）
                                 # 0.8 时隔 4 拍只剩 0.41，打不过 0.5~0.95 的老搭档，刚学完就想不起来（实测 0%）
    trace_min: float = 0.05      # 痕迹低于它就不算


class Hippocampus:
    def __init__(self, n: int, cfg: HippoCfg | None = None):
        self.n = n
        self.cfg = cfg or HippoCfg()
        self.rows: list[dict[int, float]] = [dict() for _ in range(n)]   # rows[i][j] = w
        self.trace: dict[int, float] = {}

    def step(self, rising) -> None:
        """一拍：rising = 这一拍刚亮的单元（经专用线从概念层送来）。"""
        c = self.cfg
        for j in rising:
            for i, tr in self.trace.items():
                if i != j:
                    w = self.rows[i].get(j, 0.0)
                    self.rows[i][j] = w + c.eta * tr / (1.0 + w)
        nxt = {}
        for i, tr in self.trace.items():
            tr *= c.trace_decay
            if tr >= c.trace_min:
                nxt[i] = tr
        for j in rising:
            nxt[j] = 1.0
        self.trace = nxt

    def recall(self, i: int) -> dict[int, float]:
        """经回程送回概念层的驱动：i → 各单元的当前权重。"""
        return self.rows[i]

    def edge_rows(self):
        """给整体降权用：这一组的全部可塑边。"""
        return self.rows

    def n_edges(self) -> int:
        return sum(len(r) for r in self.rows)
