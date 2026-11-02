# -*- coding: utf-8 -*-
"""感觉区的多层网络：立体（3D）局部网络（照 local3d-similarity 的结构）。

  - 每层是一个 H×W×Z 的立体：H×W 是位置（像素 / 频率×时间），Z 是同一位置上叠的神经元。
  - 输入层：每个位置每个通道 levels 档温度计，正反一对，每对恰好亮一个 → 输入层永远一半亮。
  - 每个神经元 4 个输入：上一层**自己那个位置**（权重 1）+ 3 个随机近邻（dx,dy ∈ {-1,0,1}，dz ∈ {-2..2}，
    权重很小）+ 一点随机偏置。权重出生定好，不学。
  - **每层亮的比例固定**（frac）：得分最高的那些亮。外部（区内连接、回程）送来的电流加在得分上一起比 ——
    它们只能改变"哪些亮"，改变不了"亮多少"。
相似的输入 → 相似的激活。

神经元的增加和删除（用户的机制，慢）：
  - 经常亮的神经元特征少（几乎什么都亮），从不亮的也没有特征；激活频率不多不少的特征最多。
  - 常亮的分裂：父神经元的 4 个输入分成两半，一半留给自己、一半给一个新神经元（新的输入权重放大到能亮），
    下一层原来接父神经元的，一半改接新神经元 → 两个都只对一部分情况亮，变得能区分。
  - 从不亮的删除。越靠近输入层分裂得越少（浅层一改，后面全跟着变）。
  - 每层预留空位（spare）：新神经元用空位，删除只是让它不再亮，编号都不动；
    跟着它的连接、链、前额叶 / 海马体里对应的神经元也都预先按空位分配好，跟着亮不亮的开关走。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .net import DEV, T


@dataclass
class Net3DCfg:
    levels: int = 10
    channels: int = 1
    zs: tuple = (40, 60, 120, 240)        # 第 1 层起，每个位置叠几个神经元
    frac: tuple = (0.4, 0.3, 0.2, 0.1)    # 每层亮的比例（固定，按活着的神经元算）
    fan_in: int = 4
    alpha: float = 0.01                   # 近邻的权重 ~ U(0.5, 1)·alpha
    bias_scale: float = 0.5               # 偏置 ~ U(0, bias_scale)
    spare: float = 0.25                   # 每层预留的空位比例（给分裂出来的新神经元）
    on_thr: float = 0.9                   # 使用率 ≥ 它 = 常亮
    split_frac: tuple = (0.0, 0.02, 0.05, 0.1)   # 每次每层拆常亮者中的这么多（越靠近输入越少）
    vary_from: int = 99                   # 从第几层起（0 = 第一个立体层），亮的个数跟着新鲜驱动变（99 = 都固定）
    fresh_thr: float = 1.0                # 新鲜驱动得分 ≥ 它，算"被新鲜地驱动"（自己那格刚亮起来就够）
    sigma: float = 0.1                    # 半饱和：被新鲜驱动的个数 = 活着的 × sigma 时，亮名额的一半
    n_min: float = 0.15                   # 保底：至少亮名额的这么多（安静时留给"心里听见"）


class Net3D:
    def __init__(self, H: int, W: int, cfg: Net3DCfg | None = None, seed: int = 0):
        self.H, self.W = H, W
        self.cfg = c = cfg or Net3DCfg()
        self.rng = rng = np.random.default_rng(seed)
        z0 = c.channels * c.levels * 2
        self.dims = [(H, W, z0)] + [(H, W, z) for z in c.zs]
        self.n_lines = H * W * z0
        self.idx, self.w, self.b, self.cap, self.start = [], [], [], [], []
        alive = []
        n_total = 0
        for l in range(1, len(self.dims)):
            prev, out = self.dims[l - 1], self.dims[l]
            x, y, z = np.meshgrid(np.arange(out[0]), np.arange(out[1]), np.arange(out[2]), indexing="ij")
            oc = np.stack([x, y, z], -1).reshape(-1, 3)
            n = len(oc)
            cap = n + int(round(n * c.spare))
            selfc = oc.copy()
            selfc[:, 2] = np.round(oc[:, 2] * (prev[2] - 1) / max(out[2] - 1, 1)).astype(int)
            idx = np.zeros((cap, c.fan_in), dtype=np.int64)
            w = np.zeros((cap, c.fan_in), dtype=np.float32)
            b = np.zeros(cap, dtype=np.float32)
            # 上一层在本层的输入下标是"上一层内的编号"：上一层的出生神经元编号 = 它在立体里的位置
            idx[:n, 0] = np.ravel_multi_index(selfc.T, prev)
            w[:n, 0] = 1.0
            for j in range(1, c.fan_in):
                nb = selfc.copy()
                nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
                nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
                nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
                idx[:n, j] = np.ravel_multi_index(nb.T, prev)
                w[:n, j] = (rng.random(n) * 0.5 + 0.5) * c.alpha
            b[:n] = rng.random(n) * c.bias_scale
            a = np.zeros(cap, dtype=bool)
            a[:n] = True
            self.idx.append(idx)
            self.w.append(w)
            self.b.append(b)
            alive.append(a)
            self.cap.append(cap)
            self.start.append(n_total)
            n_total += cap
        self.n = n_total                                       # 不含输入层，含空位
        self.layer = np.concatenate([np.full(cap, l) for l, cap in enumerate(self.cap)])
        self.alive = np.concatenate(alive)
        self.used = self.alive.copy()                          # 空位用过了就不再用
        self.n_split = self.n_deleted = 0
        self.last_F = [0] * len(self.cap)                    # 每层上一拍"被新鲜驱动"的个数（量用）
        self._sync()

    def _sync(self) -> None:
        """结构（numpy 底本）→ 每拍计算用的张量（结构改动之后调用）。"""
        self._t = [(T(i), T(w), T(b)) for i, w, b in zip(self.idx, self.w, self.b)]
        self._alive = T(self.alive)

    def encode(self, img: np.ndarray) -> np.ndarray:
        """img: H×W（或 H×W×通道）整数 0..levels → 输入线（恰好一半亮）。"""
        img = img.reshape(self.H, self.W, -1)
        up = img[..., None] > np.arange(self.cfg.levels)
        return np.stack([up, ~up], -1).reshape(-1)

    def respond(self, lines, extra: torch.Tensor | None = None, x: torch.Tensor | None = None) -> torch.Tensor:
        """输入线 → 各层亮不亮（拼成一条，长度 self.n）。extra：外部送给每个神经元的电流，加在得分上。
        x：每个神经元的输出余量（一直亮的推力变小）。给了 x，第 vary_from 层起：
          新鲜驱动得分 = 上一层每个源的贡献 × 它的余量；亮的个数 = 名额 × F/(F + sigma·活着的)，有保底、封顶，
          F = 新鲜驱动得分 ≥ fresh_thr 的个数 → 新的、变化的输入亮得多（强烈），不变的、安静的亮得少（平静）。
          学来的 / 回程的电流只决定亮哪些，改变不了亮多少。"""
        c = self.cfg
        h = T(lines, torch.float32)
        out = torch.zeros(self.n, dtype=torch.bool, device=DEV)
        for l, ((idx, w, b), s0, cap) in enumerate(zip(self._t, self.start, self.cap)):
            s = (h[idx] * w).sum(1) + b
            al = self._alive[s0:s0 + cap]
            n_al = int(self.alive[s0:s0 + cap].sum())
            k = int(round(c.frac[l] * n_al))
            if x is not None and l >= 1:
                xp = x[self.start[l - 1]:self.start[l - 1] + self.cap[l - 1]]
                f = (h[idx] * xp[idx] * w).sum(1) + b
                F = int(((f >= c.fresh_thr) & al).sum())
                self.last_F[l] = F
                if l >= c.vary_from:
                    k = min(k, max(round(c.n_min * k), round(k * F / (F + c.sigma * n_al))))
                    s = f
            if extra is not None:
                s = s + extra[s0:s0 + cap]
            s = torch.where(al, s, torch.full_like(s, -torch.inf))
            on = torch.zeros(cap, dtype=torch.bool, device=DEV)
            if k:
                on[torch.topk(s, k).indices] = True
            out[s0:s0 + cap] = on
            h = on.to(torch.float32)
        return out

    # ------------------------------------------------------------ 增删
    def _free_slot(self, l: int) -> int | None:
        s0, cap = self.start[l], self.cap[l]
        free = np.nonzero(~self.used[s0:s0 + cap])[0]
        return int(s0 + free[0]) if len(free) else None

    def split(self, i: int) -> int | None:
        """常亮的 i 分裂：输入一半留给自己、一半给新神经元；下一层接 i 的，一半改接新神经元。返回新神经元编号。"""
        l = int(self.layer[i])
        f = self._free_slot(l)
        if f is None:
            return None
        s0 = self.start[l]
        a, fa = i - s0, f - s0
        idx, w, b = self.idx[l], self.w[l], self.b[l]
        h = self.cfg.fan_in // 2
        parent_w = float(w[a].sum())
        idx[fa] = idx[a]
        w[fa] = 0.0
        w[fa, h:] = w[a, h:]
        w[a, h:] = 0.0
        for row in (a, fa):                       # 两半各自放大到父神经元原来的总强度，才亮得起来
            tot = float(w[row].sum())
            if tot > 0:
                w[row] *= parent_w / tot
        b[fa] = b[a]
        self.alive[f] = self.used[f] = True
        if l + 1 < len(self.idx):                 # 下一层：接 i 的，一半改接新神经元
            nxt = self.idx[l + 1]
            hit = np.argwhere(nxt == a)
            move = hit[self.rng.random(len(hit)) < 0.5]
            nxt[move[:, 0], move[:, 1]] = fa
        self.n_split += 1
        return f

    def delete(self, i: int) -> None:
        """从不亮的删除：不再亮（下一层接它的输入就一直是 0）。"""
        self.alive[i] = False
        self.n_deleted += 1

    def restructure(self, usage: np.ndarray) -> dict:
        """usage：这段时间每个神经元的使用率（numpy）。常亮的按层分裂一部分，从不亮的删除。"""
        c = self.cfg
        split, deleted = [], []
        for l in range(len(self.idx)):
            s0, cap = self.start[l], self.cap[l]
            ids = np.arange(s0, s0 + cap)
            al = ids[self.alive[ids]]
            frac = c.split_frac[min(l, len(c.split_frac) - 1)]       # 增删都慢：每次每层只动一小部分
            on = al[usage[al] >= c.on_thr]
            m = int(round(frac * len(on)))
            for i in self.rng.choice(on, m, replace=False) if m else []:
                f = self.split(int(i))
                if f is not None:
                    split.append((int(i), f))
            never = al[usage[al] == 0.0]
            m = int(round(frac * len(never)))
            for i in self.rng.choice(never, m, replace=False) if m else []:
                self.delete(int(i))
                deleted.append(int(i))
        self._sync()
        return {"split": split, "deleted": deleted}
