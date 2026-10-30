# -*- coding: utf-8 -*-
"""二值神经元的"区"（Region）和连接（Path）。规则尽量少，全部来自用户的框架：

  1. 神经元只有亮 / 不亮。每个区亮的数量大致不变：收到电流最多的那 frac 比例的神经元亮
     （frac = 0 时改用门槛：电流 ≥ thr 才亮，运动区这种平时不动的区用）。
  2. 一出生就随机充满连接（稀疏但丰富），学习只修改它们。
  3. 学习只有一条：上一拍亮着的 i、这一拍 j 刚亮 → i→j 加强；越强越难涨，不封顶。同一拍一起亮的不加强。
  4. 清理只靠整体降权：每隔一段时间，学来的部分整体往出生强度退（海马体退得快，皮层退得慢）。
抑制 / 去抑制神经元发出的连接是抑制性的（sign = −1）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegionCfg:
    frac: float = 0.1           # 每拍亮的比例（0 = 用门槛 thr）
    thr: float = 1.0


class Region:
    def __init__(self, name: str, n: int, cfg: RegionCfg | None = None):
        self.name = name
        self.cfg = cfg or RegionCfg()
        self.n = 0
        self.s = np.zeros(0, dtype=bool)          # 这一拍
        self.prev = np.zeros(0, dtype=bool)       # 上一拍
        self.alive = np.zeros(0, dtype=bool)
        self.thr = np.zeros(0, dtype=np.float32)
        self.base = np.zeros(0, dtype=np.float32)
        self.grow(n)

    def grow(self, k: int) -> np.ndarray:
        n0 = self.n
        self.n += k
        self.s = np.r_[self.s, np.zeros(k, bool)]
        self.prev = np.r_[self.prev, np.zeros(k, bool)]
        self.alive = np.r_[self.alive, np.ones(k, bool)]
        self.thr = np.r_[self.thr, np.full(k, self.cfg.thr, np.float32)]
        self.base = np.r_[self.base, np.zeros(k, np.float32)]
        return np.arange(n0, self.n)

    def kill(self, ids) -> None:
        self.alive[ids] = False
        self.s[ids] = self.prev[ids] = False

    def fire(self, current: np.ndarray) -> None:
        cur = current + self.base
        if self.cfg.frac > 0:
            k = int(round(self.cfg.frac * self.alive.sum()))
            cur = np.where(self.alive, cur, -np.inf)
            new = np.zeros(self.n, dtype=bool)
            if k:
                new[np.argpartition(-cur, k)[:k]] = True
        else:
            new = (cur >= self.thr) & self.alive
        self.set(new)

    def set(self, new: np.ndarray) -> None:
        self.prev, self.s = self.s, new

    def reset(self) -> None:
        self.s[:] = self.prev[:] = False


@dataclass
class PathCfg:
    k: int = 32               # 每个源出生连多少根（随机挑目标）
    w_init: float = 0.02      # 出生强度 ~ U(0, w_init)；一对一的连接就是这个定值
    sign: float = 1.0         # 抑制神经元发出的连接为 −1
    eta: float = 0.0          # 先亮 → 后亮 的加强速度（0 = 固定连接，不学）
    w_ref: float = 0.2        # 越强越难涨：Δ = eta / (1 + 学来的部分 / w_ref)
    gamma: float = 1.0        # 整体降权：每次学来的部分 × gamma


class Path:
    """src 区 → dst 区的一组连接。每个源最多 K 根，存成 (源数 × K) 的表；没用的位置 ok = False。"""

    def __init__(self, name: str, src: Region, dst: Region, cfg: PathCfg, rng: np.random.Generator,
                 src_ids=None, dst_ids=None, pairs=None):
        """随机：src_ids 里每个源从 dst_ids 里随机连 k 根。
        一对一 / 指定：pairs = [(源, [目标, ...]), ...]，强度都是 w_init。"""
        self.name, self.src, self.dst, self.cfg, self.rng = name, src, dst, cfg, rng
        if pairs is not None:
            K = max([len(t) for _, t in pairs] + [1])
        else:
            K = min(cfg.k, len(dst_ids)) if dst_ids is not None and len(dst_ids) else 1
        self.K = K
        self.to = np.zeros((src.n, K), dtype=np.int64)
        self.w = np.zeros((src.n, K), dtype=np.float32)
        self.ok = np.zeros((src.n, K), dtype=bool)
        self.w0 = self.w
        if pairs is not None:
            for i, ts in pairs:
                self.set_row(i, ts, cfg.w_init)
        elif src_ids is not None:
            for i in src_ids:
                self.random_row(int(i), dst_ids)
        self.w0 = self.w.copy()

    # ------------------------------------------------------------ 建线
    def _fit(self) -> None:
        """源区长了新神经元 → 表跟着加行。"""
        extra = self.src.n - self.to.shape[0]
        if extra > 0:
            for name in ("to", "w", "ok", "w0"):
                a = getattr(self, name)
                setattr(self, name, np.r_[a, np.zeros((extra, self.K), a.dtype)])

    def set_row(self, i: int, targets, w) -> None:
        self._fit()
        t = np.asarray(targets, dtype=np.int64)[: self.K]
        self.to[i] = 0
        self.w[i] = 0.0
        self.ok[i] = False
        self.to[i, : len(t)] = t
        self.w[i, : len(t)] = w
        self.ok[i, : len(t)] = True
        if self.w0 is not self.w:
            self.w0[i] = self.w[i]

    def random_row(self, i: int, dst_ids) -> None:
        t = self.rng.choice(dst_ids, size=min(self.K, len(dst_ids)), replace=False)
        self.set_row(i, t, self.rng.uniform(0, self.cfg.w_init, len(t)).astype(np.float32))

    def copy_row(self, i: int, j: int) -> None:
        """j 继承 i 的出边（分裂出的子单元继承父单元）。"""
        self._fit()
        for name in ("to", "w", "ok", "w0"):
            getattr(self, name)[j] = getattr(self, name)[i]

    def cut_src(self, ids) -> None:
        self._fit()
        self.ok[ids] = False
        self.w[ids] = self.w0[ids] = 0.0

    def redirect(self, old: int, new) -> None:
        """指向 old 的连接改指 new 里随机一个；new 为空就断开。"""
        hit = (self.to == old) & self.ok
        if not hit.any():
            return
        if len(new):
            self.to[hit] = self.rng.choice(np.asarray(new), size=int(hit.sum()))
        else:
            self.ok[hit] = False
            self.w[hit] = self.w0[hit] = 0.0

    # ------------------------------------------------------------ 一拍
    def current(self) -> np.ndarray:
        """上一拍亮着的源（src.s 还没更新时调用）送给 dst 每个神经元的电流。"""
        self._fit()
        act = np.nonzero(self.src.s)[0]
        if not len(act):
            return np.zeros(self.dst.n, dtype=np.float32)
        m = self.ok[act]
        out = np.bincount(self.to[act][m], weights=self.w[act][m], minlength=self.dst.n)
        return (self.cfg.sign * out).astype(np.float32)

    def learn(self) -> None:
        """两个区都已更新完（prev = 上一拍，s = 这一拍）：上一拍亮着的源 → 这一拍刚亮的目标，加强。"""
        c = self.cfg
        if c.eta <= 0:
            return
        rising = self.dst.s & ~self.dst.prev
        rows = np.nonzero(self.src.prev)[0]
        if len(rows) and rising.any():
            R = rising[self.to[rows]] & self.ok[rows]
            if R.any():
                W, W0 = self.w[rows], self.w0[rows]
                W += R * (c.eta / (1.0 + (W - W0) / c.w_ref))
                self.w[rows] = W

    def decay(self) -> None:
        """整体降权：学来的部分往出生强度退。"""
        if self.cfg.eta > 0 and self.cfg.gamma < 1.0:
            self.w[:] = self.w0 + (self.w - self.w0) * self.cfg.gamma

    def learned(self) -> float:
        return float(np.abs(self.w - self.w0)[self.ok].sum())
