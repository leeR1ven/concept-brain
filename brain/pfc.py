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
  - 以后 A 一亮就推 I_B，I_B 压住 B（不管 B 当时亮没亮）。"B 亮时 I_B 不亮"说的是**什么时候长连接**，
    不是 I_B 放不放电——I_B 放不放电只看它收到的输入。
    （曾把它做成"只压还没亮的 B"——那是 Claude 的过度解读：已经亮着的念头永远压不下去，已去掉。）
    → 无关的互相抑制、有关的（正连接）互相激活。
  - 抑制连接只由皮层慢慢学（速度沿用字网络 learn_inh_eta=0.01），海马体不学抑制，同样走整体降权；
    抑制总增益 0.6（沿用字网络）。

输入预算（旧架构"输入兴奋预算"的概念 / 突触缩放；可关）：
  每个单元（及每个抑制神经元）**学来的**输入总量有上限，超过就按比例缩回——新学的挤掉一部分旧的。
  分布式表示里一个目标同时从几百个来源学（看灰底时约 500 个单元同时亮），不限的话总量失控：
  抑制压死一切新输入（5 以上）、兴奋把念头锁住（约 240）、回程让静息的肌肉全力收缩（实测）。
  （之前用"每个抑制神经元最多 24 条来源"只治了抑制那一处，已由预算统一替代。）
    （曾让海马体也用 η=1 快学抑制：一开始看到的灰底一下子把其他一切挡在前额叶外，
    连 C 本身都进不来，压在肌肉对应单元上的抑制达 514——那是我自己加的，不是用户框架，已去掉。）
  （attention="global" 保留旧的全局阈值作对照：第 target 大的电位反推门槛。）

先天随机连接（用户：现在的模型还没有丰富连接，思考没法持续变化，B 亮起来后自然就一直亮；
像儿童时期那样给它丰富的随机连接；born-wired-cortex 里每个细胞出生就有随机兴奋 / 抑制出边）：
  W0：每个单元随机连 innate_exc_k 个别的单元，权重 U(innate_w)；
  V0：每个单元随机连 innate_inh_k 个别的单元的抑制神经元，权重 U(innate_w)。
  先天连接不学、不参与整体降权；学到的连接叠加在上面。数量 / 强度是搜索空间（用户 2026-09-29），先给能跑的默认值。

每拍（a = 前额叶活动，0..1）：
  exc  = a @ (W0 + Wc + hippo_gain·Wh)            正连接（先天 + 先亮 → 后亮学来的）
  inh  = inh_gain · (a @ (V0 + Vc))               抑制神经元：被 A 推亮就压它的伙伴
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
    inh_gain: float = 0.6        # 抑制总增益（沿用字网络）
    hippo_gain: float = 1.0      # 海马体回程强度
    in_budget: float = 2.0       # 每个单元学来的兴奋输入总量上限（每份：皮层 / 海马体各一份）；0 = 不限
    inh_budget: float = 2.0      # 每个抑制神经元学来的输入总量上限
    innate_exc_k: int = 20       # 先天随机兴奋出边数
    innate_inh_k: int = 20       # 先天随机抑制出边数
    innate_w: tuple = (0.02, 0.1)
    seed: int = 0
    learn: bool = True
    # 以下只给 attention="global" 用
    target: int = 200
    act_eps: float = 0.35
    theta_lr: float = 0.5
    theta_min: float = 1e-3


def apply_budget(M: np.ndarray, cols: np.ndarray, budget: float) -> None:
    """输入预算：cols 这些目标收到的（学来的）连接总量超过 budget 就按比例缩回去。
    新学的挤掉一部分旧的；分布式表示里一个目标同时从几百个来源学，总量不会失控。"""
    if budget <= 0 or not len(cols):
        return
    tot = M[:, cols].sum(axis=0)
    over = tot > budget
    if over.any():
        cc = cols[over]
        M[:, cc] *= (budget / tot[over])[None, :]


class DenseWeights:
    """一组可塑连接（稠密存放）：皮层慢 / 海马体快各一份。
    W = 正连接（先亮 → 后亮），V = 到抑制神经元的连接（A → I_B）。都是越涨越难涨。"""

    def __init__(self, n: int, eta: float):
        self.W = np.zeros((n, n), dtype=np.float32)
        self.V = np.zeros((n, n), dtype=np.float32)
        self.eta = eta

    def learn(self, trace: np.ndarray, rising: np.ndarray, budget: float = 0.0) -> None:
        if not len(rising) or self.eta <= 0:
            return
        pre = np.nonzero(trace > 0)[0]
        if not len(pre):
            return
        ix = np.ix_(pre, rising)
        sub = self.W[ix]
        sub += self.eta * trace[pre][:, None] / (1.0 + sub)
        self.W[ix] = sub
        apply_budget(self.W, rising, budget)

    def learn_inhib(self, rising: np.ndarray, quiet: np.ndarray, eta: float, budget: float) -> None:
        if not len(rising) or not len(quiet) or eta <= 0:
            return
        ix = np.ix_(rising, quiet)
        sub = self.V[ix]
        sub += eta / (1.0 + sub)
        self.V[ix] = sub
        apply_budget(self.V, quiet, budget)

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
        self.rng = np.random.default_rng(self.cfg.seed)
        self.W0 = np.zeros((0, 0), dtype=np.float32)
        self.V0 = np.zeros((0, 0), dtype=np.float32)
        self._innate(0, n)

    def _innate(self, start: int, stop: int) -> None:
        """给编号 [start, stop) 的新单元接先天随机连接（出边 + 从老单元来的入边）。"""
        c, n = self.cfg, stop
        for name in ("W0", "V0"):
            M = np.zeros((n, n), dtype=np.float32)
            old = getattr(self, name)
            M[: old.shape[0], : old.shape[1]] = old
            setattr(self, name, M)
        if n < 2:
            return
        for M, k in ((self.W0, c.innate_exc_k), (self.V0, c.innate_inh_k)):
            k = min(k, n - 1)
            for i in range(start, stop):                 # 新单元的出边
                tgt = self.rng.choice(n - 1, k, replace=False)
                tgt[tgt >= i] += 1
                M[i, tgt] = self.rng.uniform(*c.innate_w, k)
            if start > 0:                                # 老单元 → 新单元：按同样的密度补
                p = k / (n - 1)
                for j in range(start, stop):
                    src = np.nonzero(self.rng.random(start) < p)[0]
                    M[src, j] = self.rng.uniform(*c.innate_w, len(src))

    # ------------------------------------------------------------ 跟随增删
    def grow(self, k: int) -> list[int]:
        start = self.n
        self.n += k
        self.cortex.grow(k)
        self.hippo.grow(k)
        self.a = np.r_[self.a, np.zeros(k, dtype=np.float32)]
        self.trace = np.r_[self.trace, np.zeros(k, dtype=np.float32)]
        self.alive = np.r_[self.alive, np.ones(k, dtype=bool)]
        self._innate(start, self.n)
        return list(range(start, self.n))

    def remove(self, i: int) -> None:
        self.alive[i] = False
        self.a[i] = self.trace[i] = 0.0
        self.cortex.remove(i)
        self.hippo.remove(i)
        for M in (self.W0, self.V0):
            M[i, :] = 0.0
            M[:, i] = 0.0

    def decay_targets(self):
        """给整体降权登记：(皮层的所有可塑矩阵, 海马体的所有可塑矩阵)。"""
        return (lambda: [self.cortex.W, self.cortex.V]), (lambda: [self.hippo.W, self.hippo.V])

    # ------------------------------------------------------------ 一拍
    def step(self, ext: np.ndarray) -> np.ndarray:
        c = self.cfg
        a = self.a
        g = c.hippo_gain
        exc = a @ (self.W0 + self.cortex.W) + g * (a @ self.hippo.W)
        if c.attention == "inhib_pair":
            inh = c.inh_gain * (a @ (self.V0 + self.cortex.V))
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
            self.cortex.learn(self.trace, rising, c.in_budget)
            self.hippo.learn(self.trace, rising, c.in_budget)
            if c.attention == "inhib_pair":
                quiet = np.nonzero((new < c.on_thr) & self.alive)[0]
                self.cortex.learn_inhib(rising, quiet, c.eta_inhib, c.inh_budget)
        self.trace *= c.trace_decay
        self.trace[self.trace < 0.05] = 0.0
        self.trace[rising] = 1.0
        self.a = new
        return new

    def reset(self) -> None:
        self.a[:] = 0.0
        self.trace[:] = 0.0
        self.theta = None
