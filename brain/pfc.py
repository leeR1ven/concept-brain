# -*- coding: utf-8 -*-
"""一群互相连接的神经元：前额叶用它，海马体也用它（机制很像，各是各的神经元，只是参数不同）。

用户的框架：
  - 神经元只有激活 / 不激活两种状态。
  - 一开始就**随机充满连接**（兴奋，以及经抑制神经元的抑制），训练只是**修改**它们，不是从零长。
  - 前额叶内部充满互相激活的连接，可能想了很多才说话；只要求慢慢形成连接规律。
  - 海马体：权重变得快、忘得快，只管短期记忆；前额叶（皮层）变得慢。
  - 单元经链式专用线接收各区神经元（brain/brain.py 的跟随者），ext 就是链送来的 0/1。

每拍（s = 这一群的 0/1 状态）：
  I = s @ W − inh_gain · (s @ V) − pool_gain · (亮着的比例) + ext
  s' = I ≥ thr
  W：兴奋连接；V：到抑制神经元的连接（"A 亮 → 压 B"）；pool：一群抑制神经元按总活跃度压所有单元。
  + bias：一群去抑制神经元的活动（动态平衡）——整体亮得比 target 少就慢慢多放开一点，多就少放开一点。
    （用户：输入层亮的数量始终一样，前额叶亮的数量也应始终差不多，不该听字时多、安静时少。）
学习（只看相邻两拍）：
  - 先亮 → 后亮：i 上一拍亮（痕迹还在）、j 这一拍刚亮 → W[i,j] 加强，越涨越难涨（Δ = η / (1 + W/w_ref)）。
    同一拍一起亮的不互相加强。每个单元学来的输入总量有上限（budget），超了按比例缩。
  - 亮了却没带亮对方：i 上一拍亮、j 这一拍没亮 → W[i,j] 往**出生时的权重**退一点（退到出生权重就停：
    很少用的不会一直掉，背景的随机连接始终在）。
  - 抑制：i 亮、j 不亮 → V[i,j] 加强；**两拍里一起亮**（同一拍一起亮，或 j 刚亮时 i 的痕迹还在）→ 算相关，
    V[i,j] 往出生权重退。（只看同一拍时，一前一后的字从不同时亮，"今"学会了压它的下一个字"天"，实测。）
  - 整体降权：学来的部分（W − 出生权重）每隔一段整体 ×γ（global_decay）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PFCCfg:
    thr: float = 1.0
    w_init: float = 0.004        # 出生兴奋权重 ~ U(0, w_init)，所有单元两两之间都有
    v_init: float = 0.004        # 出生抑制权重 ~ U(0, v_init)
    inh_gain: float = 1.0
    pool_gain: float = 4.0       # 一群抑制神经元：压的电流 = pool_gain × 亮着的比例
    target: float = 0.1          # 动态平衡：一群去抑制神经元慢慢调整自己的活动，让整体活跃比例稳定在这附近（0 = 关）
    balance_rate: float = 0.05
    eta: float = 0.01            # 先亮 → 后亮 加强
    budget: float = 1.5          # 每个单元学来的兴奋输入（W − 出生权重）总量上限：一个单元同时从几百个来源学，总量不会失控
    w_ref: float = 0.2
    fail: float = 0.002          # 亮了却没带亮对方：往出生权重退的比例（1% 时学不过削：40 遍后"今→天"反而变弱，实测）
    eta_inhib: float = 0.0002    # 抑制每拍都在学、兴奋只在刚亮时学：抑制要慢得多，否则常亮的"安静"单元很快压死一切
    inh_budget: float = 0.5      # 每个单元学来的抑制（V − 出生权重）总量上限（不限时 40 遍后压在一个单元上的抑制达 215，实测）
    trace_decay: float = 0.7     # 0.5 时隔一拍安静后痕迹只剩 1/4，"今 → 天"几乎学不到；0.8 时跨度覆盖整句，隔几个字的也学
    seed: int = 0
    learn: bool = True


def apply_budget(M: np.ndarray, cols: np.ndarray, budget: float) -> None:
    """输入预算：cols 这些目标收到的连接总量超过 budget 就按比例缩回去（新学的挤掉一部分旧的）。"""
    if budget <= 0 or not len(cols):
        return
    tot = M[:, cols].sum(axis=0)
    over = tot > budget
    if over.any():
        cc = cols[over]
        M[:, cc] *= (budget / tot[over])[None, :]


class PFC:
    def __init__(self, n: int, cfg: PFCCfg | None = None):
        self.cfg = cfg or PFCCfg()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.n = 0
        self.W = self.W0 = self.V = self.V0 = np.zeros((0, 0), dtype=np.float32)
        self.a = np.zeros(0, dtype=np.float32)
        self.trace = np.zeros(0, dtype=np.float32)
        self.alive = np.zeros(0, dtype=bool)
        self.bias = 0.0
        self.grow(n)

    # ------------------------------------------------------------ 跟随增删
    def grow(self, k: int) -> list[int]:
        """加 k 个单元：它们和所有单元之间都有随机的出生连接。"""
        c, n0 = self.cfg, self.n
        n = n0 + k
        for name, hi in (("W0", c.w_init), ("V0", c.v_init)):
            M = np.zeros((n, n), dtype=np.float32)
            M[:n0, :n0] = getattr(self, name)
            M[n0:, :] = self.rng.uniform(0, hi, (k, n))
            M[:, n0:] = self.rng.uniform(0, hi, (n, k))
            dead = np.nonzero(~self.alive)[0]                    # 已删掉的单元不连新单元
            M[dead, :] = 0.0
            M[:, dead] = 0.0
            np.fill_diagonal(M, 0.0)
            setattr(self, name, M)
        for name, birth in (("W", "W0"), ("V", "V0")):
            M = getattr(self, birth).copy()
            M[:n0, :n0] = getattr(self, name)
            setattr(self, name, M)
        self.a = np.r_[self.a, np.zeros(k, dtype=np.float32)]
        self.trace = np.r_[self.trace, np.zeros(k, dtype=np.float32)]
        self.alive = np.r_[self.alive, np.ones(k, dtype=bool)]
        self.n = n
        return list(range(n0, n))

    def remove(self, i: int) -> None:
        self.alive[i] = False
        self.a[i] = self.trace[i] = 0.0
        for M in (self.W, self.W0, self.V, self.V0):
            M[i, :] = 0.0
            M[:, i] = 0.0

    def global_decay(self, gamma: float) -> None:
        """整体降权：学来的部分往出生权重退。"""
        if gamma < 1.0:
            self.W[:] = self.W0 + (self.W - self.W0) * gamma
            self.V[:] = self.V0 + (self.V - self.V0) * gamma

    # ------------------------------------------------------------ 一拍
    def step(self, ext: np.ndarray) -> np.ndarray:
        c = self.cfg
        s = self.a
        frac = float(s[self.alive].mean()) if self.alive.any() else 0.0
        I = s @ self.W - c.inh_gain * (s @ self.V) - c.pool_gain * frac + ext + self.bias
        new = ((I >= c.thr) & self.alive).astype(np.float32)
        if c.target > 0:                                         # 动态平衡（去抑制神经元的活动慢慢调整）
            now = float(new[self.alive].mean()) if self.alive.any() else 0.0
            self.bias += c.balance_rate * (c.target - now)
        if c.learn:
            rising = np.nonzero((new > 0) & (s == 0))[0]
            pre = np.nonzero(self.trace > 0)[0]
            if len(rising) and len(pre):                         # 先亮 → 后亮
                ix = np.ix_(pre, rising)
                sub = self.W[ix]
                sub += c.eta * self.trace[pre][:, None] / (1.0 + sub / c.w_ref)
                self.W[ix] = sub
                if c.budget > 0:                                 # 输入预算：只管学来的那部分
                    learned = self.W[:, rising] - self.W0[:, rising]
                    tot = learned.sum(axis=0)
                    over = tot > c.budget
                    if over.any():
                        cc = rising[over]
                        self.W[:, cc] = self.W0[:, cc] + learned[:, over] * (c.budget / tot[over])[None, :]
            was = np.nonzero(s > 0)[0]
            quiet = np.nonzero((new == 0) & self.alive)[0]
            if len(was) and len(quiet):                          # 亮了却没带亮对方：往出生权重退
                ix = np.ix_(was, quiet)
                self.W[ix] -= c.fail * (self.W[ix] - self.W0[ix])
                self.V[ix] += c.eta_inhib                        # 一个亮、一个不亮：抑制变强
                if c.inh_budget > 0:
                    learned = self.V[:, quiet] - self.V0[:, quiet]
                    tot = learned.sum(axis=0)
                    over = tot > c.inh_budget
                    if over.any():
                        cc = quiet[over]
                        self.V[:, cc] = self.V0[:, cc] + learned[:, over] * (c.inh_budget / tot[over])[None, :]
            both = np.nonzero(new > 0)[0]
            if len(was) and len(both):                           # 一起亮：抑制往出生权重退
                ix = np.ix_(was, both)
                self.V[ix] -= 0.5 * (self.V[ix] - self.V0[ix])
            if len(rising) and len(pre):                         # 先亮 → 后亮也算相关：抑制按痕迹往出生权重退
                ix = np.ix_(pre, rising)
                self.V[ix] -= self.trace[pre][:, None] * (self.V[ix] - self.V0[ix])
        self.trace *= c.trace_decay
        self.trace[self.trace < 0.05] = 0.0
        self.trace[new > 0] = 1.0
        self.a = new
        return new

    def reset(self) -> None:
        self.a[:] = 0.0
        self.trace[:] = 0.0
        self.bias = 0.0
