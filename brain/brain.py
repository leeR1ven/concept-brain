# -*- coding: utf-8 -*-
"""整脑骨架：感觉区 → 链 → 前额叶 / 海马体；本能直连运动区；前额叶驱动运动区；只经肌肉输出。

  感觉区（现在只有视觉，brain/vision.py）
     │  每个皮层神经元一根链式专用线（1:1，延迟 chain_delay 拍）
     ├──────────────► 前额叶接收端（每个皮层神经元一个）── 皮层慢学（Plastic，η 小）
     ├──────────────► 海马体单元（每个皮层神经元一个）──── 快涨快降（Plastic，η 大）
     │
     └─ 本能：先天直连运动区（不走链，比理性快）
                                        前额叶运动相关组（理性）
                                              │
  运动区（brain/motor.py）：肌肉控制神经元 vs 抑制神经元 → 力度 → 肌肉（唯一输出）

跟随增删：皮层神经元分裂 / 删除时，它的链、前额叶接收端、海马体单元一起增删
（子单元各配一套新的；本能直连的来源身份由子单元继承）。
跟随者编号 f：前额叶接收端 f 与海马体单元 f 是同一根链送来的同一个皮层神经元。

还没做（如实标出）：
  - 前额叶 → 运动区的权重怎么学：现在由外部给"前额叶运动相关组"的活动（step 的 reason 参数）。
  - 前额叶内部的协调 / 思考循环、自 / 他标记。
  - 视觉以外的区。
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .global_decay import GlobalDecay
from .motor import Motor
from .plastic import Plastic, PlasticCfg
from .vision import VisionSheet


class Brain:
    def __init__(self, vision: VisionSheet, motor: Motor, chain_delay: int = 2,
                 hippo_eta: float = 1.0, cortex_eta: float = 0.05,
                 hippo_gamma: float = 0.5, cortex_gamma: float = 1.0, decay_T: int = 200):
        self.vision = vision
        self.motor = motor
        self.chain_delay = chain_delay
        # 跟随者：f → 皮层神经元编号；皮层神经元 → f
        self.f_src: list[int] = []
        self.f_alive: list[bool] = []
        self.f_of: dict[int, int] = {}
        self.hippo = Plastic(0, PlasticCfg(eta=hippo_eta))
        self.cortex = Plastic(0, PlasticCfg(eta=cortex_eta))
        for cid in np.nonzero(vision._alive)[0]:
            self._attach(int(cid))
        self.gd = GlobalDecay(T=decay_T)
        self.gd.register("hippocampus", self.hippo.edge_rows, hippo_gamma)
        self.gd.register("cortex", self.cortex.edge_rows, cortex_gamma)
        self.chain = deque([np.zeros(len(vision.inputs), dtype=bool)] * chain_delay, maxlen=chain_delay)
        self.pfc_prev = np.zeros(0, dtype=bool)
        self.pfc = np.zeros(0, dtype=bool)
        self.reflex: dict[str, set[int]] = {}      # 动作 → 先天直连它的皮层神经元
        self.reflex_gain: dict[str, float] = {}
        self.usage = np.zeros(len(vision.inputs))
        self.n_seen = 0
        self.t = 0

    # ------------------------------------------------------------ 跟随者
    def _attach(self, cid: int) -> int:
        f = len(self.f_src)
        self.f_src.append(cid)
        self.f_alive.append(True)
        self.f_of[cid] = f
        self.hippo.add_units(1)
        self.cortex.add_units(1)
        return f

    def _detach(self, cid: int) -> None:
        f = self.f_of.pop(cid, None)
        if f is None:
            return
        self.f_alive[f] = False
        self.hippo.remove(f)
        self.cortex.remove(f)

    def apply_events(self, info: dict) -> None:
        for parent, kids in info.get("split", []):
            for k in kids:
                self._attach(k)
            for a, srcs in self.reflex.items():        # 本能来源身份由子单元继承
                if parent in srcs:
                    srcs.discard(parent)
                    srcs.update(kids)
            self._detach(parent)
        for cid in info.get("added", []):
            self._attach(cid)
        for cid in info.get("deleted", []):
            for srcs in self.reflex.values():
                srcs.discard(cid)
            self._detach(cid)

    def restructure(self) -> dict:
        """慢：用这段时间的使用率做一次结构调整，跟随者一起增删。"""
        info = self.vision.restructure(self.usage[: len(self.vision.inputs)] / max(1, self.n_seen))
        self.apply_events(info)
        self.usage = np.zeros(len(self.vision.inputs))
        self.n_seen = 0
        self.chain = deque([np.r_[a, np.zeros(len(self.vision.inputs) - len(a), dtype=bool)] for a in self.chain],
                           maxlen=self.chain_delay)
        return info

    # ------------------------------------------------------------ 本能
    def wire_reflex(self, action: str, sources, gain: float = 1.5) -> None:
        """先天直连：这批皮层神经元一起亮 → 驱动这个动作（驱动 = gain × 亮的比例）。"""
        self.reflex[action] = set(int(i) for i in sources)
        self.reflex_gain[action] = gain

    # ------------------------------------------------------------ 走一拍
    def step(self, image: np.ndarray, reason: dict[str, float] | None = None) -> dict[str, int]:
        act = self.vision.respond(self.vision.encode(image))
        self.usage[: len(act)] += act
        self.n_seen += 1
        # 链：延迟 chain_delay 拍后到前额叶 / 海马体
        self.chain.append(act)
        arrived = self.chain[0]
        n_f = len(self.f_src)
        pfc = np.zeros(n_f, dtype=bool)
        src = np.asarray(self.f_src)
        ok = np.asarray(self.f_alive) & (src < len(arrived))
        pfc[ok] = arrived[src[ok]]
        prev = np.r_[self.pfc_prev, np.zeros(n_f - len(self.pfc_prev), dtype=bool)]
        rising = np.nonzero(pfc & ~prev)[0].tolist()
        self.hippo.step(rising)
        self.cortex.step(rising)
        self.pfc_prev, self.pfc = pfc, pfc
        # 运动区：本能（不走链）+ 理性（前额叶运动相关组）
        drive: dict[str, float] = {}
        for a, srcs in self.reflex.items():
            if srcs:
                ids = np.fromiter(srcs, dtype=int)
                drive[a] = drive.get(a, 0.0) + self.reflex_gain[a] * float(act[ids].mean())
        for a, v in (reason or {}).items():
            drive[a] = drive.get(a, 0.0) + v
        self.gd.tick()
        self.t += 1
        return self.motor.step(drive)

    # ------------------------------------------------------------ 看
    def n_followers(self) -> int:
        return int(sum(self.f_alive))
