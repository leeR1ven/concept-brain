# -*- coding: utf-8 -*-
"""一群互相连接的二值神经元：前额叶、海马体、各感觉区内部都用它（机制相同，参数不同）。

用户的框架：
  - 神经元只有激活 / 不激活。一开始就随机充满连接，训练只是修改它们。
  - 感觉区内部也有赫布连接：被激活后会沿学好的顺序一个接一个亮（表征坍缩——瞬间回忆起一段连续的声音 /
    做出一套连续动作）。
  - 前额叶：想法是一个大范围，代表很多信息的综合；单个信号进来最多点亮一个神经元，很难改变整体范围。
  - 海马体：记细节，变得快、忘得快；前额叶（皮层）变得慢。

每拍（s = 0/1 状态）：
  I = s @ W − inh_gain · (s @ V0) − pool_gain · (亮着的比例) + bias + ext
  s' = I ≥ thr
  V0：先天随机的抑制（经抑制神经元，不学）；pool：抑制神经元按总活跃度压所有单元；
  bias：动态平衡（去抑制神经元慢慢调整，让活跃比例稳定在 target 附近；0 = 关）。
学习（三条，只看相邻两拍）：
  1. 先亮 → 后亮：i 的痕迹还在、j 这一拍刚亮 → W[i,j] 加强，越涨越难涨。同一拍一起亮的不互相加强。
  2. 亮了却没带亮对方：i 上一拍亮、j 这一拍没亮 → W[i,j] 往出生权重退一点（退到出生权重就停）。
  3. 输入预算：每个神经元学来的输入总量（W − 出生权重）有上限，超了按比例缩。
  整体降权（global_decay）：学来的部分往出生权重退。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PFCCfg:
    thr: float = 1.0
    w_init: float = 0.004        # 出生兴奋权重 ~ U(0, w_init)，两两之间都有
    v_init: float = 0.004        # 出生抑制权重 ~ U(0, v_init)（不学）
    inh_gain: float = 1.0
    pool_gain: float = 4.0       # 抑制神经元按总活跃度压：pool_gain × 亮着的比例
    target: float = 0.1          # 动态平衡的目标活跃比例（0 = 关）
    balance_rate: float = 0.05
    eta: float = 0.01            # 先亮 → 后亮 加强
    w_ref: float = 0.2
    fail: float = 0.002          # 亮了却没带亮对方：往出生权重退的比例
    budget: float = 1.5          # 每个神经元学来的输入总量上限
    trace_decay: float = 0.7
    seed: int = 0
    learn: bool = True


def apply_budget(M: np.ndarray, cols: np.ndarray, budget: float, M0: np.ndarray | None = None) -> None:
    """输入预算：cols 这些目标收到的（学来的）连接总量超过 budget 就按比例缩回去。"""
    if budget <= 0 or not len(cols):
        return
    base = 0.0 if M0 is None else M0[:, cols]
    learned = M[:, cols] - base
    tot = learned.sum(axis=0)
    over = tot > budget
    if over.any():
        cc = cols[over]
        b0 = 0.0 if M0 is None else M0[:, cc]
        M[:, cc] = b0 + learned[:, over] * (budget / tot[over])[None, :]


class PFC:
    def __init__(self, n: int, cfg: PFCCfg | None = None):
        self.cfg = cfg or PFCCfg()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.n = 0
        self.W = self.W0 = self.V0 = np.zeros((0, 0), dtype=np.float32)
        self.a = np.zeros(0, dtype=np.float32)
        self.trace = np.zeros(0, dtype=np.float32)
        self.alive = np.zeros(0, dtype=bool)
        self.bias = 0.0
        self.grow(n)

    # ------------------------------------------------------------ 跟随增删
    def grow(self, k: int) -> list[int]:
        """加 k 个神经元：它们和所有神经元之间都有随机的出生连接。"""
        c, n0 = self.cfg, self.n
        n = n0 + k
        dead = np.nonzero(~self.alive)[0]
        for name, hi in (("W0", c.w_init), ("V0", c.v_init)):
            M = np.zeros((n, n), dtype=np.float32)
            M[:n0, :n0] = getattr(self, name)
            M[n0:, :] = self.rng.uniform(0, hi, (k, n))
            M[:, n0:] = self.rng.uniform(0, hi, (n, k))
            M[dead, :] = 0.0
            M[:, dead] = 0.0
            np.fill_diagonal(M, 0.0)
            setattr(self, name, M)
        W = self.W0.copy()
        W[:n0, :n0] = self.W
        self.W = W
        self.a = np.r_[self.a, np.zeros(k, dtype=np.float32)]
        self.trace = np.r_[self.trace, np.zeros(k, dtype=np.float32)]
        self.alive = np.r_[self.alive, np.ones(k, dtype=bool)]
        self.n = n
        return list(range(n0, n))

    def remove(self, i: int) -> None:
        self.alive[i] = False
        self.a[i] = self.trace[i] = 0.0
        for M in (self.W, self.W0, self.V0):
            M[i, :] = 0.0
            M[:, i] = 0.0

    def global_decay(self, gamma: float) -> None:
        """整体降权：学来的部分往出生权重退。"""
        if gamma < 1.0:
            self.W[:] = self.W0 + (self.W - self.W0) * gamma

    # ------------------------------------------------------------ 一拍
    def currents(self, s: np.ndarray | None = None) -> np.ndarray:
        """当前状态 s 送给每个神经元的电流（不含外部输入）。"""
        c = self.cfg
        s = self.a if s is None else s
        frac = float(s[self.alive].mean()) if self.alive.any() else 0.0
        return s @ self.W - c.inh_gain * (s @ self.V0) - c.pool_gain * frac + self.bias

    def learn_from(self, s: np.ndarray, new: np.ndarray) -> None:
        """上一拍 s → 这一拍 new：三条学习规则 + 痕迹 + 动态平衡。"""
        c = self.cfg
        if c.learn:
            rising = np.nonzero((new > 0) & (s == 0))[0]
            pre = np.nonzero(self.trace > 0)[0]
            if len(rising) and len(pre):                         # 1. 先亮 → 后亮
                ix = np.ix_(pre, rising)
                sub = self.W[ix]
                sub += c.eta * self.trace[pre][:, None] / (1.0 + sub / c.w_ref)
                self.W[ix] = sub
                apply_budget(self.W, rising, c.budget, self.W0)  # 3. 输入预算
            was = np.nonzero(s > 0)[0]
            quiet = np.nonzero((new == 0) & self.alive)[0]
            if len(was) and len(quiet):                          # 2. 亮了却没带亮对方
                ix = np.ix_(was, quiet)
                self.W[ix] -= c.fail * (self.W[ix] - self.W0[ix])
        if c.target > 0:
            now = float(new[self.alive].mean()) if self.alive.any() else 0.0
            self.bias += c.balance_rate * (c.target - now)
        self.trace *= c.trace_decay
        self.trace[self.trace < 0.05] = 0.0
        self.trace[new > 0] = 1.0
        self.a = new

    def step(self, ext: np.ndarray) -> np.ndarray:
        s = self.a
        new = ((self.currents(s) + ext >= self.cfg.thr) & self.alive).astype(np.float32)
        self.learn_from(s, new)
        return new

    def reset(self) -> None:
        self.a[:] = 0.0
        self.trace[:] = 0.0
        self.bias = 0.0
