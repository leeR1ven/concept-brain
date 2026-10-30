# -*- coding: utf-8 -*-
"""感觉区的多层网络：立体（3D）局部网络（照 local3d-similarity 的结构）。

  - 每层是一个 H×W×Z 的立体：H×W 是位置（像素 / 频率×时间），Z 是同一位置上叠的神经元。
  - 输入层：每个位置每个通道 levels 档温度计，正反一对，每对恰好亮一个 → 输入层永远一半亮。
  - 每个神经元 4 个输入：上一层**自己那个位置**（权重 1）+ 3 个随机近邻（dx,dy ∈ {-1,0,1}，dz ∈ {-2..2}，
    权重很小）+ 一点随机偏置。权重出生定好，不学。
  - **每层亮的比例固定**（frac，例如 40/30/20/10%）：得分最高的那些亮。外部（区内连接、回程）送来的电流
    加在得分上一起比 —— 它们只能改变"哪些亮"，改变不了"亮多少"。
相似的输入 → 相似的激活。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Net3DCfg:
    levels: int = 10
    channels: int = 1
    zs: tuple = (40, 60, 120, 240)        # 第 1 层起，每个位置叠几个神经元
    frac: tuple = (0.4, 0.3, 0.2, 0.1)    # 每层亮的比例（固定）
    fan_in: int = 4
    alpha: float = 0.01                   # 近邻的权重 ~ U(0.5, 1)·alpha
    bias_scale: float = 0.5               # 偏置 ~ U(0, bias_scale)


class Net3D:
    def __init__(self, H: int, W: int, cfg: Net3DCfg | None = None, seed: int = 0):
        self.H, self.W = H, W
        self.cfg = c = cfg or Net3DCfg()
        rng = np.random.default_rng(seed)
        z0 = c.channels * c.levels * 2
        self.dims = [(H, W, z0)] + [(H, W, z) for z in c.zs]
        self.n_lines = H * W * z0
        self.idx, self.w, self.b, self.k = [], [], [], []
        self.start = []
        n_total = 0
        for l in range(1, len(self.dims)):
            prev, out = self.dims[l - 1], self.dims[l]
            x, y, z = np.meshgrid(np.arange(out[0]), np.arange(out[1]), np.arange(out[2]), indexing="ij")
            oc = np.stack([x, y, z], -1).reshape(-1, 3)
            n = len(oc)
            selfc = oc.copy()
            selfc[:, 2] = np.round(oc[:, 2] * (prev[2] - 1) / max(out[2] - 1, 1)).astype(int)
            idx = np.zeros((n, c.fan_in), dtype=np.int64)
            w = np.zeros((n, c.fan_in), dtype=np.float32)
            idx[:, 0] = np.ravel_multi_index(selfc.T, prev)
            w[:, 0] = 1.0
            for j in range(1, c.fan_in):
                nb = selfc.copy()
                nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
                nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
                nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
                idx[:, j] = np.ravel_multi_index(nb.T, prev)
                w[:, j] = (rng.random(n) * 0.5 + 0.5) * c.alpha
            self.idx.append(idx)
            self.w.append(w)
            self.b.append((rng.random(n) * c.bias_scale).astype(np.float32))
            self.k.append(int(round(n * c.frac[l - 1])))
            self.start.append(n_total)
            n_total += n
        self.n = n_total                                       # 不含输入层
        self.layer = np.concatenate([np.full(len(i), l) for l, i in enumerate(self.idx)])
        self.alive = np.ones(self.n, dtype=bool)

    def encode(self, img: np.ndarray) -> np.ndarray:
        """img: H×W（或 H×W×通道）整数 0..levels → 输入线（恰好一半亮）。"""
        img = img.reshape(self.H, self.W, -1)
        up = img[..., None] > np.arange(self.cfg.levels)
        return np.stack([up, ~up], -1).reshape(-1)

    def respond(self, lines: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        """输入线 → 各层亮不亮（拼成一条，长度 self.n）。extra：外部送给每个神经元的电流，加在得分上。"""
        h = lines.astype(np.float32)
        out = np.zeros(self.n, dtype=bool)
        for l, (idx, w, b, k, s0) in enumerate(zip(self.idx, self.w, self.b, self.k, self.start)):
            n = len(idx)
            s = (h[idx] * w).sum(1) + b
            if extra is not None:
                s = s + extra[s0:s0 + n]
            s[~self.alive[s0:s0 + n]] = -np.inf
            on = np.zeros(n, dtype=bool)
            on[np.argpartition(-s, k)[:k]] = True
            out[s0:s0 + n] = on
            h = on.astype(np.float32)
        return out
