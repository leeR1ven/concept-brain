# -*- coding: utf-8 -*-
"""前额叶：在分布式单元上思考。

用户的框架：
  - 不是"一个字一个神经元"。看到 / 听到一个字会激活很多神经元；每次都亮的那批就是它的特征神经元。
  - 前额叶单元 = 各区每个神经元经链式专用线送来的接收端（brain/brain.py 的跟随者）。
  - 文字网络证明了这套动态能让前额叶真正思考；这里把它搬到分布式单元上：
      扩散 + 全局阈值（注意力带宽）+ 增益控制 + 先亮 → 后亮的实时学习。
  - 海马体：同一批单元（经链 1:1 对应）、同一种学习，只是快涨快降；回程送回前额叶。
  - 长期记忆 = 皮层（前额叶之间的慢权重）自己学。下降都走整体降权。

每拍（a = 前额叶活动，0..1）：
  drive = (a @ Wc + hippo_gain · a @ Wh) / (1 + γ_g · Σa)      增益控制：总激活越多越难推高
  pot   = decay · a + drive + ext                               ext = 链送来的感觉活动
  θ*    = 第 target 大的 pot − act_eps·soft；θ ← θ + θ_lr (θ* − θ)   全局阈值：分位数反推 + 平滑跟进
  a     = clip((pot − θ) / soft, 0, 1)
学习（Wc 慢、Wh 快，同一规则）：
  刚亮的 j（a 刚越过 on_thr）← 痕迹还在的 i：ΔW[i,j] = η · trace_i / (1 + W[i,j])
  痕迹每拍 ×trace_decay；刚亮的痕迹在**学完之后**才置 1。
  → **只有先亮的指向后亮的**；同一拍一起亮的不互相加强（用户：否则容易来回不停地激活）。
  常亮的不会反复"刚亮"，不和所有人连。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PFCCfg:
    target: int = 200            # 注意力带宽：大约多少个单元同时亮
    decay: float = 0.45
    soft: float = 0.15
    act_eps: float = 0.35
    theta_lr: float = 0.5
    theta_min: float = 1e-3      # 门槛下限：有正电位的单元比带宽少时，分位数会反推出负门槛 → 全亮（实测）
    gain_g: float = 0.01         # 增益控制
    on_thr: float = 0.35         # 判"亮"
    trace_decay: float = 0.8
    eta_cortex: float = 0.02     # 慢
    eta_hippo: float = 1.0       # 快
    hippo_gain: float = 1.0      # 海马体回程强度
    learn: bool = True


class DenseWeights:
    """一组可塑连接（稠密存放）：皮层慢权重 / 海马体快权重各一份。
    W = 连接强度（越涨越难涨）。"""

    def __init__(self, n: int, eta: float):
        self.W = np.zeros((n, n), dtype=np.float32)
        self.eta = eta

    def learn(self, trace: np.ndarray, rising: np.ndarray) -> None:
        if not len(rising) or self.eta <= 0:
            return
        pre = np.nonzero(trace > 0)[0]
        if not len(pre):
            return
        ix = np.ix_(pre, rising)
        sub = self.W[ix]
        sub += self.eta * trace[pre][:, None] / (1.0 + sub)
        self.W[ix] = sub

    def grow(self, k: int) -> None:
        n = self.W.shape[0]
        W = np.zeros((n + k, n + k), dtype=np.float32)
        W[:n, :n] = self.W
        self.W = W

    def remove(self, i: int) -> None:
        self.W[i, :] = 0.0
        self.W[:, i] = 0.0

    def rows(self):
        return self.W


class PFC:
    def __init__(self, n: int, cfg: PFCCfg | None = None):
        self.cfg = cfg or PFCCfg()
        self.n = n
        self.cortex = DenseWeights(n, self.cfg.eta_cortex)
        self.hippo = DenseWeights(n, self.cfg.eta_hippo)
        self.a = np.zeros(n, dtype=np.float32)
        self.trace = np.zeros(n, dtype=np.float32)
        self.alive = np.ones(n, dtype=bool)
        self.theta = None

    # ------------------------------------------------------------ 跟随增删
    def grow(self, k: int) -> list[int]:
        start = self.n
        self.n += k
        self.cortex.grow(k)
        self.hippo.grow(k)
        self.a = np.r_[self.a, np.zeros(k, dtype=np.float32)]
        self.trace = np.r_[self.trace, np.zeros(k, dtype=np.float32)]
        self.alive = np.r_[self.alive, np.ones(k, dtype=bool)]
        return list(range(start, self.n))

    def remove(self, i: int) -> None:
        self.alive[i] = False
        self.a[i] = self.trace[i] = 0.0
        self.cortex.remove(i)
        self.hippo.remove(i)


    # ------------------------------------------------------------ 一拍
    def step(self, ext: np.ndarray) -> np.ndarray:
        c = self.cfg
        a = self.a
        drive = a @ self.cortex.W + c.hippo_gain * (a @ self.hippo.W)
        drive = drive / (1.0 + c.gain_g * float(a.sum()))
        pot = c.decay * a + drive + ext
        pot[~self.alive] = -np.inf
        k = min(c.target, int(self.alive.sum()) - 1)
        th_star = float(np.partition(pot, -k)[-k]) - c.act_eps * c.soft
        self.theta = th_star if self.theta is None else self.theta + c.theta_lr * (th_star - self.theta)
        self.theta = max(self.theta, c.theta_min)
        new = np.clip((pot - self.theta) / c.soft, 0.0, 1.0).astype(np.float32)
        new[~self.alive] = 0.0
        rising = np.nonzero((new >= c.on_thr) & (a < c.on_thr))[0]
        if c.learn:
            self.cortex.learn(self.trace, rising)
            self.hippo.learn(self.trace, rising)

        self.trace *= c.trace_decay
        self.trace[self.trace < 0.05] = 0.0
        self.trace[rising] = 1.0
        self.a = new
        return new

    def reset(self) -> None:
        self.a[:] = 0.0
        self.trace[:] = 0.0
        self.theta = None
