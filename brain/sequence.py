# -*- coding: utf-8 -*-
"""序列皮层：意外时长出概念单元，用来存"组合"。

问题来源（char-neuron-voice）：一张 V×V 关联矩阵只存两两关系，
(人,工)→智 这种"要两个音同时在才成立"的信息存不下——下一个音准确率卡在 9~10%。

做法（一个概念单元 = 一个巧合检测器）：
  - 感觉单元 0..V-1：一个符号一个神经元（深度 1）。
  - 概念单元 c = (父单元 p, 符号 s)：上一拍 p 亮、这一拍 s 亮 → c 亮。
    父单元到 c 的这一拍延迟就是一段中继链（链式架构里的 1:1 专用线）。
  - 每个亮着的单元对"下一拍来的符号"做计数（赫布：前亮 → 后亮）。
  - **只在意外时长新单元**：预测给实际符号的概率 < surprise_p 时，
    在"最浅的、还没有这个子单元"的亮着的单元下面长一个（一次就长）。
    从浅往深长：先试两个符号的组合，不够再加长。
    （第一版从最深处长，结果长出一堆把无关前文也算进去的过细组合：4500 个单元、65%。）
  - **长了没用就删（只考核一次 = 试用期）**：子单元和同一拍亮着的更短上下文比预测
    （对数概率差），用满 prune_after 次后平均增益 < prune_gain 就删掉，且不再重长；
    通过的转正，以后不再考核，内容靠衰减更新。
    （一直考核不行：换规则后有用的单元暂时变差，被删掉、拉黑，重学要 2200~2700 个三元组。）
    —— 本来就随机的位置（谁也预测不了）长出来的单元会被删掉。
  - 遗忘：默认走全脑共用的**整体降权**（brain/global_decay.py，登记 edge_rows）。
    另留一个对照开关 use_decay：单元每被用一次，旧计数先乘它（实验 A 里比较两者）。
  - 稳定 ID + alive：删除只改 alive，不移位。

读出：亮着的、见过足够多次 (≥ n_min) 的单元按**最近的可靠度**竞争（每个单元记一个
"最近预测对没对"的滑动平均），最可靠的胜出，平手时深的胜。
（第一版"最深的优先"：深层单元很少被用到，换规则后还拿着旧计数，用过时内容压住已经更新的浅层。）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeqCfg:
    max_depth: int = 4          # 最长记几个符号的组合
    surprise_p: float = 0.5     # 实际符号的预测概率低于它 → 意外 → 长新单元
    n_min: float = 2.0          # 单元至少见过这么多次才拿来预测
    use_decay: float = 1.0      # 每次被用到时旧计数先乘它（1.0 = 不忘）
    grow: bool = True           # 关掉 = 只有感觉单元（等价于两两关联矩阵）
    prune: bool = True          # 长了没用就删
    prune_after: int = 12       # 比较满这么多次再判
    prune_gain: float = 0.1     # 平均对数概率增益（nats）低于它就删
    smooth: float = 0.05        # 比较预测时的平滑
    rel_rate: float = 0.3       # 可靠度滑动平均的速度


class SequenceCortex:
    def __init__(self, V: int, cfg: SeqCfg | None = None):
        self.V = V
        self.cfg = cfg or SeqCfg()
        self.parent: list[int] = []
        self.sym: list[int] = []
        self.depth: list[int] = []
        self.alive: list[bool] = []
        self.counts: list[dict[int, float]] = []
        self.gain: list[float] = []
        self.uses: list[int] = []
        self.rel: list[float] = []
        self.children: dict[tuple[int, int], int] = {}
        self.t = 0
        for s in range(V):
            self._new(-1, s, 1)
        self.active: list[int] = []     # 当前亮着的单元，按深度从浅到深
        self.n_grown = 0
        self.n_proven = 0

    # ------------------------------------------------------------ 结构
    def _new(self, parent: int, sym: int, depth: int) -> int:
        i = len(self.parent)
        self.parent.append(parent)
        self.sym.append(sym)
        self.depth.append(depth)
        self.alive.append(True)
        self.counts.append({})
        self.gain.append(0.0)
        self.uses.append(0)
        self.rel.append(0.5)
        if parent >= 0:
            self.children[(parent, sym)] = i
        return i

    @property
    def n_units(self) -> int:
        return int(sum(self.alive))

    # ------------------------------------------------------------ 读
    def predict(self, active: list[int] | None = None) -> tuple[np.ndarray, int]:
        """返回 (下一个符号的分布, 用来预测的单元)。"""
        active = self.active if active is None else active
        best, score = -1, -1.0
        for c in active:
            if self._tot(c) >= self.cfg.n_min:
                sc = self.rel[c] + 1e-3 * self.depth[c]
                if sc > score:
                    best, score = c, sc
        if best < 0:
            return np.full(self.V, 1.0 / self.V), -1
        p = np.zeros(self.V)
        for s, n in self.counts[best].items():
            p[s] = n
        return p / p.sum(), best

    def _tot(self, c: int) -> float:
        return sum(self.counts[c].values())

    def _p(self, c: int, s: int) -> float:
        a = self.cfg.smooth
        return (self.counts[c].get(s, 0.0) + a) / (self._tot(c) + a * self.V)

    def edge_rows(self):
        """给整体降权用：每个单元的"下一拍符号"计数就是它的可塑出边。"""
        return self.counts

    # ------------------------------------------------------------ 走一拍
    def step(self, s: int, learn: bool = True) -> dict:
        """输入一个符号。先用上一拍的状态预测它、学习，再更新亮着的单元。"""
        prev = self.active
        p, used = self.predict(prev)
        info = {"p": float(p[s]), "hit": bool(np.argmax(p) == s), "used": used, "grew": -1}
        if learn and prev:
            c_ = self.cfg
            if c_.prune:                        # 子单元 vs 同一拍更短的上下文
                for q, c in zip(prev[:-1], prev[1:]):
                    if self.uses[c] >= 0 and self._tot(c) >= c_.n_min and self._tot(q) >= c_.n_min:
                        self.gain[c] += np.log(self._p(c, s)) - np.log(self._p(q, s))
                        self.uses[c] += 1
                        if self.uses[c] >= c_.prune_after:
                            if self.gain[c] / self.uses[c] < c_.prune_gain:
                                self.alive[c] = False   # 留在 children 里 → 不再重长
                            else:
                                self.n_proven += 1      # 转正：以后不再考核
                            self.uses[c] = -10 ** 9
            for c in prev:                      # 可靠度：这个单元自己的最大计数猜对没有
                if self._tot(c) >= c_.n_min:
                    cs = self.counts[c]
                    ok = float(max(cs, key=cs.get) == s)
                    self.rel[c] += c_.rel_rate * (ok - self.rel[c])
            for c in prev:                      # 前亮 → 后亮：计数（按使用衰减）
                if c_.use_decay < 1.0:
                    cs = self.counts[c]
                    for k in cs:
                        cs[k] *= c_.use_decay
                self.counts[c][s] = self.counts[c].get(s, 0.0) + 1.0
            if c_.grow and p[s] < c_.surprise_p:
                for c in prev:                  # 从浅往深，找第一个缺口
                    if self.depth[c] < c_.max_depth and (c, s) not in self.children:
                        info["grew"] = self._new(c, s, self.depth[c] + 1)
                        self.n_grown += 1
                        break
        self.t += 1
        # 这一拍亮：感觉单元 s，以及父单元上一拍亮、符号是 s 的概念单元
        nxt = [s]
        for c in prev:
            ch = self.children.get((c, s))
            if ch is not None and self.alive[ch]:
                nxt.append(ch)
        nxt.sort(key=lambda c: self.depth[c])
        self.active = nxt
        return info

    def reset_context(self) -> None:
        self.active = []
