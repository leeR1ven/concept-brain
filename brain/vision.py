# -*- coding: utf-8 -*-
"""视觉皮层一层：温度计编码 + 就近随机连线 + 神经元的分裂与删除。

输入：每个像素 levels 档温度计 + 正反一对（第 t 档：正 = 亮度 > t，反 = 亮度 ≤ t），
      每对恰好亮一个 → 输入线永远一半亮。
皮层神经元：位于某个像素，从自己附近（radius 内）的像素里随机接 k 条输入线，权重都是 1
      （"自己"和邻居同一量级，不会退化成像素复制）；亮的入边数 ≥ need 才亮。

结构可塑性（用户机制，慢）：
  - **常亮的分裂**：长时间看相同 / 相似画面时，很多神经元总是亮——它们分不出这些画面的细节。
    把它拆成 2 个子单元，入边一分为二，**门槛保持父单元原来的 need**（子单元入边少了，同样的
    门槛就更挑剔）→ 子单元只对部分细节亮，变得能区分。父单元由子单元取代。
  - **从不亮的删除**：没有特征。
  - 本来就有时亮有时不亮的不动。拆得太窄、从不亮的子单元，下一轮自然被删。
  - 慢：每次只拆常亮者中的 split_frac。
  - 稳定 ID + alive：删除只改 alive，不移位；跟在后面的读出按 alive 取。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SheetCfg:
    levels: int = 10
    per_pixel: int = 8           # 每个像素位置先天几个神经元
    k: int = 8                   # 先天入边数
    need_frac: float = 0.5       # 先天门槛 = ceil(need_frac · k)
    radius: int = 1
    on_thr: float = 0.9          # 使用率 ≥ 它 = 常亮
    split_frac: float = 0.1      # 每次拆常亮者中的这么多
    delete: bool = True


class VisionSheet:
    def __init__(self, H: int, W: int, cfg: SheetCfg | None = None, seed: int = 0):
        self.H, self.W = H, W
        self.cfg = cfg or SheetCfg()
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.n_lines = H * W * c.levels * 2
        self.inputs: list[np.ndarray] = []
        self.need: list[int] = []
        self.alive: list[bool] = []
        self.pos: list[tuple[int, int]] = []
        for y in range(H):
            for x in range(W):
                for _ in range(c.per_pixel):
                    self._new((y, x), self.random_inputs(y, x, c.k), int(np.ceil(c.need_frac * c.k)))
        self.n_split = self.n_deleted = 0
        self._pack()

    # ------------------------------------------------------------ 编码
    def encode(self, img: np.ndarray) -> np.ndarray:
        """img: H×W 整数亮度 0..levels。返回所有输入线（加一条永远不亮的哑线）。"""
        L = self.cfg.levels
        up = img[:, :, None] > np.arange(L)[None, None, :]          # H×W×L
        lines = np.stack([up, ~up], axis=-1).reshape(-1)
        return np.r_[lines, False]

    def line(self, y: int, x: int, t: int, pol: int) -> int:
        return ((y * self.W + x) * self.cfg.levels + t) * 2 + pol

    def random_inputs(self, y: int, x: int, k: int) -> np.ndarray:
        r, out = self.cfg.radius, set()
        while len(out) < k:
            yy = int(np.clip(y + self.rng.integers(-r, r + 1), 0, self.H - 1))
            xx = int(np.clip(x + self.rng.integers(-r, r + 1), 0, self.W - 1))
            out.add(self.line(yy, xx, int(self.rng.integers(self.cfg.levels)), int(self.rng.integers(2))))
        return np.array(sorted(out))

    # ------------------------------------------------------------ 结构
    def _new(self, pos, inp, need) -> int:
        self.pos.append(pos)
        self.inputs.append(np.asarray(inp, dtype=int))
        self.need.append(int(need))
        self.alive.append(True)
        return len(self.inputs) - 1

    def _pack(self) -> None:
        n, kmax = len(self.inputs), max(len(x) for x in self.inputs)
        idx = np.full((n, kmax), self.n_lines, dtype=int)        # 空位指向哑线
        for i, x in enumerate(self.inputs):
            idx[i, :len(x)] = x
        self._idx = idx
        self._need = np.asarray(self.need)
        self._alive = np.asarray(self.alive)

    @property
    def n_alive(self) -> int:
        return int(self._alive.sum())

    def respond(self, lines: np.ndarray) -> np.ndarray:
        """一张画面 → 所有神经元的亮 / 不亮（删掉的永远不亮）。"""
        return (lines[self._idx].sum(axis=1) >= self._need) & self._alive

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
        m = int(round(c.split_frac * len(on)))
        info = {"always_on": int(len(on)), "never": 0, "changed": m, "split": [], "added": [], "deleted": []}
        if mode == "split_on":
            parents = self.rng.choice(on, m, replace=False) if m else []
        elif mode == "split_random":
            parents = self.rng.choice(alive, m, replace=False) if m else []
        else:
            parents = []
            for i in self.rng.choice(alive, m, replace=False) if m else []:
                y, x = self.pos[i]
                info["added"].append(
                    self._new((y, x), self.random_inputs(y, x, c.k), int(np.ceil(c.need_frac * c.k))))
        for p in parents:
            x = self.rng.permutation(self.inputs[p])
            h = len(x) // 2
            kids = [self._new(self.pos[p], np.sort(part), min(self.need[p], len(part))) for part in (x[:h], x[h:])]
            self.alive[p] = False
            info["split"].append((int(p), kids))
            self.n_split += 1
        if c.delete:
            never = alive[usage[alive] == 0.0]
            for i in never:
                self.alive[i] = False
            info["never"] = int(len(never))
            info["deleted"] = [int(i) for i in never]
            self.n_deleted += len(never)
        self._pack()
        return info
