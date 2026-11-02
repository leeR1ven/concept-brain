# -*- coding: utf-8 -*-
"""二值神经元的"区"（Region）和连接（Path）。规则尽量少，全部来自用户的框架：

  1. 神经元只有亮 / 不亮。每个区亮的数量大致不变：收到电流最多的那 frac 比例的神经元亮
     （frac = 0 时改用门槛：电流 ≥ thr 才亮，运动区这种平时不动的区用）。
  2. 一出生就随机充满连接（稀疏但丰富），学习只修改它们。
  3. 学习：i 先亮、j 后亮 → i→j 加强（越强越难涨，不封顶）；反过来 j 先亮、i 后亮（i 一直亮着也算）→ i→j 减弱
     （最多减回出生强度）。同一拍一起亮的不加强。
  4. 清理：每隔一段时间，学来的部分整体往出生强度退（海马体退得快，皮层退得慢）。
  5. 一个神经元从某一类连接学来的总量有上限（真实大脑里的异突触可塑性：一些输入加强时，同一个神经元的
     其他输入跟着变弱）。超过时这个神经元收到的这类连接一起按比例缩回——学到的只改变谁强谁弱，
     不会无限放大，出生就有的固定结构（环路、抑制门槛）一直有分量。
  6. 用掉就弱（真实大脑里的短时压抑和稳态，同一条原理的两个时间尺度）：
     - 输出余量：神经元每亮一拍，它对别人的推力就掉一截，几拍内恢复——一直不变的输入（安静、静止的身体）
       推力很快变小，变化的、新的输入照样全力送出；学到的量也乘这个余量（越新鲜学得越多）。不是不应期：
       它自己照样能亮。
     - 设定点：神经元记着自己长期亮得多频繁；比本区目标比例亮得多的，慢慢变得更难赢——不会有"永远赢的一批"。
抑制 / 去抑制神经元发出的连接是抑制性的（sign = −1）。

用 torch 存和算：有显卡（或环境变量 BRAIN_DEVICE 指定）就在显卡上，否则在 CPU 上；规则完全一样。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

DEV = torch.device(os.environ.get("BRAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
DENSE = DEV.type == "cuda"      # 显卡上整张表一起算更快；CPU 上只碰亮着的那些更快（结果一样）
if DEV.type == "cpu":
    torch.set_num_threads(int(os.environ.get("BRAIN_THREADS", "1")))   # 批量测试时每个进程一个核


def T(x, dtype=None) -> torch.Tensor:
    """numpy / 列表 → 设备上的张量。"""
    if isinstance(x, torch.Tensor):
        return x.to(DEV) if dtype is None else x.to(DEV, dtype)
    return torch.as_tensor(np.asarray(x), device=DEV, dtype=dtype)


def N(x) -> np.ndarray:
    """张量 → numpy（给读数、检查用）。"""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@dataclass
class RegionCfg:
    frac: float = 0.1           # 每拍亮的比例（0 = 用门槛 thr）
    thr: float = 1.0
    std_U: float = 0.0          # 输出余量：每亮一拍余量 × (1 − std_U)（0 = 没有）
    std_tau: float = 6.0        #           余量往 1 恢复的快慢（拍）
    sp_beta: float = 0.0        # 设定点：亮得比目标多的，每多 1 倍减 sp_beta 电流（0 = 没有）
    sp_tau: float = 1000.0      #         记"亮得多频繁"的时间尺度（拍）


class Region:
    def __init__(self, name: str, n: int, cfg: RegionCfg | None = None):
        self.name = name
        self.cfg = cfg or RegionCfg()
        self.n = n
        self.s = torch.zeros(n, dtype=torch.bool, device=DEV)          # 这一拍
        self.prev = torch.zeros(n, dtype=torch.bool, device=DEV)       # 上一拍
        self.alive = torch.ones(n, dtype=torch.bool, device=DEV)
        self.thr = torch.full((n,), self.cfg.thr, dtype=torch.float32, device=DEV)
        self.base = torch.zeros(n, dtype=torch.float32, device=DEV)
        self.x = torch.ones(n, dtype=torch.float32, device=DEV)                   # 输出余量
        self.u = torch.full((n,), self.cfg.frac, dtype=torch.float32, device=DEV)  # 长期亮的频率

    def kill(self, ids) -> None:
        ids = T(ids, torch.long)
        self.alive[ids] = False
        self.s[ids] = False
        self.prev[ids] = False
        self._alive_ref = None

    def fire(self, current: torch.Tensor) -> None:
        cur = current + self.base
        if self.cfg.frac > 0:
            if getattr(self, "_alive_ref", None) is not self.alive:          # 活着的数目只在增删时变
                self._alive_ref, self._n_alive = self.alive, int(self.alive.sum())
            k = int(round(self.cfg.frac * self._n_alive))
            new = torch.zeros(self.n, dtype=torch.bool, device=DEV)
            if k:
                cur = torch.where(self.alive, cur, torch.full_like(cur, -torch.inf))
                new[torch.topk(cur, k).indices] = True
        else:
            new = (cur >= self.thr) & self.alive
        self.set(new)

    def set(self, new: torch.Tensor) -> None:
        self.prev, self.s = self.s, new

    def deplete(self) -> None:
        """这一拍送出过电流的（上一拍亮着的）余量掉一截；所有余量往 1 恢复。"""
        c = self.cfg
        if c.std_U > 0:
            self.x = torch.where(self.prev, self.x * (1.0 - c.std_U), self.x)
            self.x = self.x + (1.0 - self.x) / c.std_tau

    def homeostasis(self) -> None:
        """设定点：只罚亮得比目标比例多的那部分。"""
        c = self.cfg
        if c.sp_beta > 0 and c.frac > 0:
            self.u = self.u + (self.s.float() - self.u) / c.sp_tau
            self.base = -c.sp_beta * torch.clamp(self.u / c.frac - 1.0, min=0.0)

    def reset(self) -> None:
        self.s = torch.zeros_like(self.s)
        self.prev = torch.zeros_like(self.prev)
        self.x = torch.ones_like(self.x)


@dataclass
class PathCfg:
    k: int = 32               # 每个源出生连多少根（随机挑目标）
    w_init: float = 0.02      # 出生强度 ~ U(0, w_init)；一对一的连接就是这个定值
    sign: float = 1.0         # 抑制神经元发出的连接为 −1
    eta: float = 0.0          # 先亮 → 后亮 的加强速度（0 = 固定连接，不学）
    w_ref: float = 0.2        # 越强越难涨：Δ = eta / (1 + 学来的部分 / w_ref)
    gamma: float = 1.0        # 整体降权：每次学来的部分 × gamma
    budget: float = 0.0       # 每个目标神经元从这类连接学来的总量上限（0 = 不限）
    use_x: bool = True        # 推力乘源的输出余量（源区有余量时）；海马要照实记录，链到海马不乘
    back: bool = True         # "后亮的源 → 先亮的目标"减弱（False：只加强，一直亮着的源也能学会"它在时目标会亮"）


class Path:
    """src 区 → dst 区的一组连接。每个源最多 K 根，存成 (源数 × K) 的表；没用的位置 ok = False、权重 0。"""

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
        to = np.zeros((src.n, K), dtype=np.int64)
        w = np.zeros((src.n, K), dtype=np.float32)
        ok = np.zeros((src.n, K), dtype=bool)
        if pairs is not None:
            for i, ts in pairs:
                ts = np.asarray(ts)[:K]
                to[i, :len(ts)] = ts
                w[i, :len(ts)] = cfg.w_init
                ok[i, :len(ts)] = True
        elif src_ids is not None:
            dst_ids = np.asarray(dst_ids)
            n = len(dst_ids)
            for i in src_ids:
                t = dst_ids[rng.choice(n, size=K, replace=False)] if n > K else dst_ids
                to[i, :len(t)] = t
                w[i, :len(t)] = rng.uniform(0, cfg.w_init, len(t))
                ok[i, :len(t)] = True
        self.to, self.w, self.ok = T(to), T(w), T(ok)
        self.w0 = self.w.clone()

    # ------------------------------------------------------------ 改线（很少用：分裂、删除、没特征的退出）
    def copy_row(self, i: int, j: int) -> None:
        """j 继承 i 的出边（分裂出的子单元继承父单元）。"""
        for name in ("to", "w", "ok", "w0"):
            getattr(self, name)[j] = getattr(self, name)[i]

    def cut_src(self, ids) -> None:
        ids = T(ids, torch.long)
        self.ok[ids] = False
        self.w[ids] = 0.0
        self.w0[ids] = 0.0

    def cut_dst(self, ids) -> None:
        """指向 ids 的连接断开。"""
        hit = torch.isin(self.to, T(ids, torch.long)) & self.ok
        self.ok[hit] = False
        self.w[hit] = 0.0
        self.w0[hit] = 0.0

    def move_dst(self, old: int, new: int, frac: float = 0.5) -> None:
        """指向 old 的连接里，约 frac 比例改指 new。"""
        hit = (self.to == old) & self.ok
        move = hit & (torch.rand(hit.shape, device=DEV) < frac)
        self.to[move] = new

    # ------------------------------------------------------------ 一拍
    def current(self) -> torch.Tensor:
        """上一拍亮着的源（src.s 还没更新时调用）送给 dst 每个神经元的电流。"""
        out = torch.zeros(self.dst.n, dtype=torch.float32, device=DEV)
        xs = self._xs()
        if DENSE:                                   # 显卡上：整张表一起算，不用先挑出亮着的（省得来回等）
            a = self.src.s.float() if xs is None else self.src.s * xs
            out.index_add_(0, self.to.reshape(-1), (self.w * a[:, None]).reshape(-1))
        else:
            act = self.src.s.nonzero().squeeze(1)
            if len(act):
                w = self.w[act] if xs is None else self.w[act] * xs[act, None]
                out.index_add_(0, self.to[act].reshape(-1), w.reshape(-1))   # 没用的位置权重为 0
        return out if self.cfg.sign == 1.0 else self.cfg.sign * out

    def _xs(self):
        """源的输出余量（源区没有余量、或这类连接不乘时为 None）。"""
        return self.src.x if self.cfg.use_x and self.src.cfg.std_U > 0 else None

    def learn(self) -> None:
        """两个区都已更新完（prev = 上一拍，s = 这一拍）。i→j：
          i 先亮、j 后亮（上一拍 i 亮着，这一拍 j 刚亮）→ 加强，越强越难涨；
          j 先亮、i 后亮（上一拍 j 亮着，这一拍 i 亮着）→ 减弱，最多减回出生强度。
          i 一直亮着、j 中途亮了一下，就等于"i 后 j"又"j 后 i"，两边抵消（用户）。
        所以只有总是"i 灭了、j 接着亮"的先后关系留得住；一直亮着的共有特征连不出很多。"""
        c = self.cfg
        if c.eta <= 0:
            return
        rising = self.dst.s & ~self.dst.prev
        xs = self._xs()                             # 传递那一拍的余量（Circuit 在学完之后才更新余量）
        if DENSE:                                   # 显卡上：整张表一起算
            fwd = self.src.prev[:, None] & rising[self.to] & self.ok
            back = self.src.s[:, None] & self.dst.prev[self.to] & self.ok
            inc = c.eta / (1.0 + (self.w - self.w0) / c.w_ref)
            if xs is not None:
                inc = inc * xs[:, None]
            w = torch.where(fwd, self.w + inc, self.w)
            self.w = torch.where(back, torch.maximum(w - c.eta, self.w0), w) if c.back else w
            return
        rows = self.src.prev.nonzero().squeeze(1)
        if len(rows) and bool(rising.any()):                             # i 先亮、j 后亮
            r, k = (rising[self.to[rows]] & self.ok[rows]).nonzero(as_tuple=True)
            if len(r):
                rr = rows[r]
                w, w0 = self.w[rr, k], self.w0[rr, k]
                inc = c.eta / (1.0 + (w - w0) / c.w_ref)
                self.w[rr, k] = w + (inc if xs is None else inc * xs[rr])
        rows = self.src.s.nonzero().squeeze(1)
        if c.back and len(rows) and bool(self.dst.prev.any()):                      # j 先亮、i 后亮（i 一直亮着也算）
            r, k = (self.dst.prev[self.to[rows]] & self.ok[rows]).nonzero(as_tuple=True)
            if len(r):
                rr = rows[r]
                self.w[rr, k] = torch.maximum(self.w[rr, k] - c.eta, self.w0[rr, k])

    def decay(self) -> None:
        """整体降权：学来的部分往出生强度退；每个目标神经元学来的总量超过上限就按比例缩回。"""
        c = self.cfg
        if c.eta <= 0:
            return
        L = self.w - self.w0
        if c.gamma < 1.0:
            L = L * c.gamma
        if c.budget > 0:
            tot = torch.zeros(self.dst.n, dtype=torch.float32, device=DEV)
            tot.index_add_(0, self.to.reshape(-1), (L * self.ok).reshape(-1))
            scale = torch.clamp(c.budget / torch.clamp(tot, min=1e-9), max=1.0)
            L = L * scale[self.to]
        self.w = self.w0 + L

    def learned(self) -> tuple[float, float]:
        """学来的部分：平均、最大（给检查用）。"""
        d = (self.w - self.w0).abs()[self.ok]
        return (float(d.mean()), float(d.max())) if len(d) else (0.0, 0.0)
