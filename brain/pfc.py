# -*- coding: utf-8 -*-
"""前额叶：在分布式单元上思考。

用户的框架：
  - 不是"一个字一个神经元"。看到 / 听到一个字会激活很多神经元；每次都亮的那批就是它的特征神经元。
  - 前额叶单元 = 各区每个神经元经链式专用线送来的接收端（brain/brain.py 的跟随者）。
  - 文字网络证明了这套动态能让前额叶真正思考；这里搬到分布式单元上：
      扩散 + 增益控制 + 先亮 → 后亮的实时学习。
  - 海马体：同一批单元（经链 1:1 对应）、同一种学习，只是快涨快降；回程送回前额叶。
  - 长期记忆 = 皮层（前额叶之间的慢权重）自己学。下降都走整体降权。
  - **没有全局门槛**（用户 2026-10-29）：注意力靠抑制神经元——思考中的大量神经元抑制无关的，
    有关的互相激活，让思想总在一个范围里。

抑制（attention="inhib_pair"，默认；用户描述的第二种方法）：
  - 每个前额叶单元 B 身边一个抑制神经元 I_B（独立细胞，不在连接上写负权重）：B 不亮时 I_B 亮，B 亮时 I_B 不亮。
  - 学习：A 刚亮、而 B 没亮（I_B 亮着）→ A → I_B 的连接加强：ΔV[A,B] = η / (1 + V[A,B])。
  - 以后 A 一亮就推 I_B，把还没亮的 B 压住；B 已经亮着时 I_B 不亮，压不到它。
    → 无关的互相抑制、有关的（正连接）互相激活；和当前思考无关的输入进不来。
  - 抑制连接只由皮层慢慢学（速度沿用字网络 learn_inh_eta=0.01），海马体不学抑制，同样走整体降权；
    每个抑制神经元最多接收 24 条来源、抑制总增益 0.6（都沿用字网络）。
    （不限来源数时：分布式表示里同时亮约 500 个单元，每条 0.01 加起来也有 5，任何新输入都进不来。）
    （曾让海马体也用 η=1 快学抑制：一开始看到的灰底一下子把其他一切挡在前额叶外，
    连 C 本身都进不来，压在肌肉对应单元上的抑制达 514——那是我自己加的，不是用户框架，已去掉。）
  （attention="global" 保留旧的全局阈值作对照：第 target 大的电位反推门槛。）

每拍（a = 前额叶活动，0..1）：
  exc  = a @ (Wc + hippo_gain·Wh)                 正连接（先亮 → 后亮学来的）
  inh  = (1 − a) ⊙ (a @ (Vc + hippo_gain·Vh))     抑制神经元：只在自己的伙伴没亮时起作用
  pot  = decay·a + (exc − inh) / (1 + γ_g·Σa) + ext      增益控制
  a'   = clip((pot − thr) / soft, 0, 1)            每个单元自己的门槛 thr（不随全局变）
学习（快慢同一规则）：
  正连接：刚亮的 j ← 痕迹还在的 i：ΔW[i,j] = η · trace_i / (1 + W[i,j])
          痕迹每拍 ×trace_decay；刚亮的痕迹在**学完之后**才置 1
          → **只有先亮的指向后亮的**；同一拍一起亮的不互相加强（用户：否则容易来回不停地激活）。
  抑制：  刚亮的 A → 此刻没亮的 B 的抑制神经元。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PFCCfg:
    attention: str = "inhib_pair"   # "inhib_pair"（抑制神经元）/ "global"（旧：全局阈值，对照用）
    thr: float = 0.5             # 每个单元自己的门槛
    decay: float = 0.45
    soft: float = 0.15
    gain_g: float = 0.01         # 增益控制
    on_thr: float = 0.35         # 判"亮"
    trace_decay: float = 0.8
    eta_cortex: float = 0.02     # 慢
    eta_hippo: float = 1.0       # 快
    eta_inhib: float = 0.01      # 抑制连接（只在皮层，慢）
    inh_per_target: int = 24     # 每个抑制神经元最多接收多少条来源（沿用字网络）
    inh_gain: float = 0.6        # 抑制总增益（沿用字网络）
    hippo_gain: float = 1.0      # 海马体回程强度
    learn: bool = True
    # 以下只给 attention="global" 用
    target: int = 200
    act_eps: float = 0.35
    theta_lr: float = 0.5
    theta_min: float = 1e-3


class DenseWeights:
    """一组可塑连接（稠密存放）：皮层慢 / 海马体快各一份。
    W = 正连接（先亮 → 后亮），V = 到抑制神经元的连接（A → I_B）。都是越涨越难涨。"""

    def __init__(self, n: int, eta: float):
        self.W = np.zeros((n, n), dtype=np.float32)
        self.V = np.zeros((n, n), dtype=np.float32)
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

    def learn_inhib(self, rising: np.ndarray, quiet: np.ndarray, eta: float, per_target: int) -> None:
        if not len(rising) or not len(quiet) or eta <= 0:
            return
        ix = np.ix_(rising, quiet)
        sub = self.V[ix]
        sub += eta / (1.0 + sub)
        self.V[ix] = sub
        # 每个抑制神经元（每一列）只留最强的 per_target 条来源
        cols = self.V[:, quiet]
        if cols.shape[0] > per_target:
            cut = np.partition(cols, -per_target, axis=0)[-per_target]
            cols[cols < cut[None, :]] = 0.0
            self.V[:, quiet] = cols

    def grow(self, k: int) -> None:
        n = self.W.shape[0]
        for name in ("W", "V"):
            M = np.zeros((n + k, n + k), dtype=np.float32)
            M[:n, :n] = getattr(self, name)
            setattr(self, name, M)

    def remove(self, i: int) -> None:
        for M in (self.W, self.V):
            M[i, :] = 0.0
            M[:, i] = 0.0


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

    def decay_targets(self):
        """给整体降权登记：(皮层的所有可塑矩阵, 海马体的所有可塑矩阵)。"""
        return (lambda: [self.cortex.W, self.cortex.V]), (lambda: [self.hippo.W, self.hippo.V])

    # ------------------------------------------------------------ 一拍
    def step(self, ext: np.ndarray) -> np.ndarray:
        c = self.cfg
        a = self.a
        g = c.hippo_gain
        exc = a @ self.cortex.W + g * (a @ self.hippo.W)
        if c.attention == "inhib_pair":
            inh = c.inh_gain * (1.0 - a) * (a @ self.cortex.V)
        else:
            inh = 0.0
        pot = c.decay * a + (exc - inh) / (1.0 + c.gain_g * float(a.sum())) + ext
        pot[~self.alive] = -np.inf
        if c.attention == "global":
            k = min(c.target, int(self.alive.sum()) - 1)
            th_star = float(np.partition(pot, -k)[-k]) - c.act_eps * c.soft
            self.theta = th_star if self.theta is None else self.theta + c.theta_lr * (th_star - self.theta)
            self.theta = max(self.theta, c.theta_min)
            thr = self.theta
        else:
            thr = c.thr
        new = np.clip((pot - thr) / c.soft, 0.0, 1.0).astype(np.float32)
        new[~self.alive] = 0.0
        rising = np.nonzero((new >= c.on_thr) & (a < c.on_thr))[0]
        if c.learn:
            self.cortex.learn(self.trace, rising)
            self.hippo.learn(self.trace, rising)
            if c.attention == "inhib_pair":
                quiet = np.nonzero((new < c.on_thr) & self.alive)[0]
                self.cortex.learn_inhib(rising, quiet, c.eta_inhib, c.inh_per_target)
        self.trace *= c.trace_decay
        self.trace[self.trace < 0.05] = 0.0
        self.trace[rising] = 1.0
        self.a = new
        return new

    def reset(self) -> None:
        self.a[:] = 0.0
        self.trace[:] = 0.0
        self.theta = None
