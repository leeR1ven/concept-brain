# -*- coding: utf-8 -*-
"""视觉区（多层）：温度计编码 + 逐层就近随机连线 + 神经元的分裂与删除。

"皮层"指感觉区里所有层的神经元，不只是最后一层：每一层都带局部特征，每一层都往前额叶送
（每个神经元一根链式专用线，见 brain/brain.py）。

输入：每个像素 levels 档温度计 + 正反一对（第 t 档：正 = 亮度 > t，反 = 亮度 ≤ t），
      每对恰好亮一个 → 输入线永远一半亮。
第 1 层：位于某个像素，从附近（radius 内）像素的输入线里随机接 k 条。
第 d 层（d ≥ 2）：从上一层附近位置的神经元里随机接 k 个。
权重都是 1（"自己"和邻居同一量级，不会退化成像素复制）；亮的入边数 ≥ need 才亮。

结构可塑性（用户机制，慢；所有层都适用）：
  - **常亮的分裂**：长时间看相同 / 相似画面时，很多神经元总是亮——分不出这些画面的细节。
    拆成 2 个子单元，入边一分为二，**门槛保持父单元原来的 need**（入边少了、门槛不变 → 更挑剔）
    → 子单元只对部分细节亮，变得能区分。父单元由子单元取代；
    下一层原来接父单元的入边，改接其中一个子单元。
  - **从不亮的删除**：没有特征。下一层接它的入边断开（指向哑线）。
  - 本来就有时亮有时不亮的不动。拆得太窄、从不亮的子单元，下一轮自然被删。
  - 慢：每次只拆常亮者中的 split_frac；**越靠近输入层拆得越少**（用户：浅层的改动会大幅改变
    后面的神经元和权重）。
  - 稳定 ID + alive：删除只改 alive，不移位；结构事件（split / added / deleted）交给跟随者。

状态向量：S = [输入线 n_lines 条, 哑线 1 条(永远 False), 神经元 0..N-1]；入边都是 S 里的下标。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SheetCfg:
    levels: int = 10
    layers: tuple = (8,)         # 每层每个像素位置先天几个神经元（长度 = 层数）
    k: int = 8                   # 先天入边数
    need_frac: float = 0.5       # 先天门槛 = ceil(need_frac · k)
    radius: int = 1
    on_thr: float = 0.9          # 使用率 ≥ 它 = 常亮
    split_frac: tuple = (0.02, 0.05, 0.1)
                                 # 每层每次拆常亮者中的这么多（越靠近输入越少：浅层一改，后面所有神经元和权重都跟着变）；
                                 # 层数多于元组长度时沿用最后一个
    delete: bool = True


class VisionSheet:
    def __init__(self, H: int, W: int, cfg: SheetCfg | None = None, seed: int = 0):
        self.H, self.W = H, W
        self.cfg = cfg or SheetCfg()
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.n_lines = H * W * c.levels * 2
        self.off = self.n_lines + 1                   # 神经元 i 在状态向量里的下标 = off + i
        self.inputs: list[np.ndarray] = []
        self.need: list[int] = []
        self.alive: list[bool] = []
        self.pos: list[tuple[int, int]] = []
        self.layer: list[int] = []
        need0 = int(np.ceil(c.need_frac * c.k))
        for d, per in enumerate(c.layers):
            self._pack()
            for y in range(H):
                for x in range(W):
                    for _ in range(per):
                        self._new((y, x), d, self.random_inputs(y, x, c.k, d), need0)
        self.n_split = self.n_deleted = 0
        self._pack()

    # ------------------------------------------------------------ 编码
    def encode(self, img: np.ndarray) -> np.ndarray:
        """img: H×W 整数亮度 0..levels。返回所有输入线 + 一条永远不亮的哑线。"""
        L = self.cfg.levels
        up = img[:, :, None] > np.arange(L)[None, None, :]          # H×W×L
        lines = np.stack([up, ~up], axis=-1).reshape(-1)
        return np.r_[lines, False]

    def line(self, y: int, x: int, t: int, pol: int) -> int:
        return ((y * self.W + x) * self.cfg.levels + t) * 2 + pol

    def random_inputs(self, y: int, x: int, k: int, layer: int) -> np.ndarray:
        r = self.cfg.radius
        if layer == 0:
            out = set()
            while len(out) < k:
                yy = int(np.clip(y + self.rng.integers(-r, r + 1), 0, self.H - 1))
                xx = int(np.clip(x + self.rng.integers(-r, r + 1), 0, self.W - 1))
                out.add(self.line(yy, xx, int(self.rng.integers(self.cfg.levels)), int(self.rng.integers(2))))
            return np.array(sorted(out))
        P = self._pos
        cand = np.nonzero((self._layer == layer - 1) & self._alive
                          & (np.abs(P[:, 0] - y) <= r) & (np.abs(P[:, 1] - x) <= r))[0]
        pick = self.rng.choice(cand, size=min(k, len(cand)), replace=False)
        return np.sort(self.off + pick)

    # ------------------------------------------------------------ 结构
    def _new(self, pos, layer, inp, need) -> int:
        self.pos.append(pos)
        self.layer.append(layer)
        self.inputs.append(np.asarray(inp, dtype=int))
        self.need.append(int(need))
        self.alive.append(True)
        return len(self.inputs) - 1

    def _pack(self) -> None:
        n = len(self.inputs)
        kmax = max([len(x) for x in self.inputs] + [1])
        idx = np.full((n, kmax), self.n_lines, dtype=int)        # 空位指向哑线
        for i, x in enumerate(self.inputs):
            idx[i, :len(x)] = x
        self._idx = idx
        self._need = np.asarray(self.need, dtype=int)
        self._alive = np.asarray(self.alive, dtype=bool)
        self._layer = np.asarray(self.layer, dtype=int)
        self._pos = np.asarray(self.pos, dtype=int).reshape(-1, 2)
        self._by_layer = [np.nonzero(self._layer == d)[0] for d in range(len(self.cfg.layers))]

    @property
    def n_alive(self) -> int:
        return int(self._alive.sum())

    def respond(self, lines: np.ndarray) -> np.ndarray:
        """一张画面 → 所有层所有神经元的亮 / 不亮（删掉的永远不亮），逐层往前算。"""
        S = np.zeros(self.off + len(self.inputs), dtype=bool)
        S[: self.off] = lines
        for ids in self._by_layer:
            S[self.off + ids] = (S[self._idx[ids]].sum(axis=1) >= self._need[ids]) & self._alive[ids]
        return S[self.off:]

    def _rewire_downstream(self, old: int, new_choices) -> None:
        """下一层接 old 的入边：改接 new_choices 里随机一个；没有可选的就断开（指向哑线）。"""
        # 扫当前的入边（不能用打包好的 _idx：同一轮里新拆出的子单元继承了父单元的入边，_idx 里没有它们）
        ref = self.off + old
        for q, x in enumerate(self.inputs):
            pos = np.nonzero(x == ref)[0]
            if not len(pos):
                continue
            x = x.copy()
            if len(new_choices):
                x[pos] = self.off + int(self.rng.choice(new_choices))
            else:
                x[pos] = self.n_lines
            self.inputs[q] = x

    def restructure(self, usage: np.ndarray, mode: str = "split_on") -> dict:
        """usage：这段时间每个神经元的使用率。
        mode：split_on（用户机制）/ split_random（对照：拆同样多个随机挑的）/
              add_random（对照：新加同样多个随机神经元）。
        返回的 info 里带结构事件，给跟随者（链 / 前额叶 / 海马体）用：
          split   = [(父, [子, 子]), ...]   父单元已删、子单元新建
          added   = [新神经元, ...]
          deleted = [删掉的, ...]"""
        c = self.cfg
        alive = np.nonzero(self._alive)[0]
        on = alive[usage[alive] >= c.on_thr]
        fr = tuple(c.split_frac) if isinstance(c.split_frac, (tuple, list)) else (c.split_frac,)
        picked = []                                   # 每层按自己的比例挑
        for d in range(len(c.layers)):
            f = fr[min(d, len(fr) - 1)]
            on_d = on[self._layer[on] == d]
            m_d = int(round(f * len(on_d)))
            if not m_d:
                continue
            pool = on_d if mode == "split_on" else alive[self._layer[alive] == d]
            picked += list(self.rng.choice(pool, m_d, replace=False))
        m = len(picked)
        info = {"always_on": int(len(on)), "never": 0, "changed": m, "split": [], "added": [], "deleted": []}
        if mode in ("split_on", "split_random"):
            parents = picked
        else:
            parents = []
            for i in picked:
                (y, x), d = self.pos[i], self.layer[i]
                info["added"].append(self._new((y, x), d, self.random_inputs(y, x, c.k, d),
                                               int(np.ceil(c.need_frac * c.k))))
        for p in parents:
            x = self.rng.permutation(self.inputs[p])
            h = len(x) // 2
            kids = [self._new(self.pos[p], self.layer[p], np.sort(part), min(self.need[p], len(part)))
                    for part in (x[:h], x[h:])]
            self.alive[p] = False
            self._rewire_downstream(int(p), kids)
            info["split"].append((int(p), kids))
            self.n_split += 1
        if c.delete:
            never = alive[usage[alive] == 0.0]
            for i in never:
                self.alive[i] = False
                self._rewire_downstream(int(i), [])
            info["never"] = int(len(never))
            info["deleted"] = [int(i) for i in never]
            self.n_deleted += len(never)
        self._pack()
        return info
