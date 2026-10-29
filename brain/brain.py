# -*- coding: utf-8 -*-
"""整脑骨架：各区 → 链 → 前额叶（思考 + 海马体）→ 回程 → 运动区 → 肌肉（唯一输出）。

  视觉区（brain/vision.py）──┐
  运动区的每个神经元 ─────────┤ 每个神经元一根链式专用线（1:1，延迟 chain_delay 拍）
  （控制肌肉 / 抑制 / 去抑制）  ├──► 前额叶接收端（在上面思考，brain/pfc.py：皮层慢权重 + 海马体快权重）
                              │
  本能：视觉先天直连运动区 ────┼──► 运动区（brain/motor.py）
                              │        ▲
  前额叶里运动神经元的对应单元 ── 回程（链，1:1，延迟 chain_delay 拍）──► 同一个运动神经元

  - 前额叶在想别的事、没激活运动对应单元时：运动完全由其他区（本能）决定。
  - 前额叶想动 / 想压住某些肌肉：靠它自己内部的相互激活 / 竞争把对应单元点亮。
  - 前额叶会自己学"看到什么之后哪些肌肉动了"：运动神经元的活动和感觉一样经链进来，
    先亮 → 后亮的实时学习把它们接起来。
  - 前额叶 ↔ 各区之间的回路是正常现象（用户 2026-10-29），不当作问题去改。

跟随增删：各区神经元分裂 / 删除时，它的链、前额叶接收端、海马体单元一起增删
（子单元各配一套新的；本能直连的来源身份由子单元继承）。
跟随者 f ↔ (区, 神经元编号)。

还没做（如实标出）：视觉以外的感觉区；前额叶影响其他感觉区（回程到感觉层）。
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .global_decay import GlobalDecay
from .motor import Motor
from .pfc import PFC, PFCCfg
from .vision import VisionSheet


class Brain:
    def __init__(self, vision: VisionSheet, motor: Motor, chain_delay: int = 2,
                 pfc_cfg: PFCCfg | None = None,
                 hippo_gamma: float = 0.5, cortex_gamma: float = 1.0, decay_T: int = 200):
        self.vision = vision
        self.motor = motor
        self.chain_delay = chain_delay
        self.f_src: list[tuple[str, int]] = []
        self.f_alive: list[bool] = []
        self.f_of: dict[tuple[str, int], int] = {}
        self.pfc = PFC(0, pfc_cfg)
        for cid in np.nonzero(vision._alive)[0]:
            self._attach("vision", int(cid))
        for mid in range(motor.n_neurons):
            self._attach("motor", mid)
        self.motor_f = np.array([self.f_of[("motor", m)] for m in range(motor.n_neurons)])
        self.gd = GlobalDecay(T=decay_T)
        cortex_m, hippo_m = self.pfc.decay_targets()
        self.gd.register("hippocampus", hippo_m, hippo_gamma)
        self.gd.register("cortex", cortex_m, cortex_gamma)
        z = (np.zeros(len(vision.inputs), dtype=bool), np.zeros(motor.n_neurons, dtype=np.float32))
        self.chain = deque([z] * chain_delay, maxlen=chain_delay)          # 去程
        self.ret = deque([np.zeros(motor.n_neurons, dtype=np.float32)] * chain_delay,
                         maxlen=chain_delay)                                 # 回程
        self.reflex: dict[str, set[int]] = {}      # 动作 → 先天直连它的视觉神经元
        self.reflex_gain: dict[str, float] = {}
        self.usage = np.zeros(len(vision.inputs))
        self.n_seen = 0
        self.t = 0

    # ------------------------------------------------------------ 跟随者
    def _attach(self, region: str, nid: int) -> int:
        f = len(self.f_src)
        self.f_src.append((region, nid))
        self.f_alive.append(True)
        self.f_of[(region, nid)] = f
        self.pfc.grow(1)
        return f

    def _detach(self, region: str, nid: int) -> None:
        f = self.f_of.pop((region, nid), None)
        if f is None:
            return
        self.f_alive[f] = False
        self.pfc.remove(f)

    def apply_events(self, info: dict) -> None:
        for parent, kids in info.get("split", []):
            for k in kids:
                self._attach("vision", k)
            for srcs in self.reflex.values():           # 本能来源身份由子单元继承
                if parent in srcs:
                    srcs.discard(parent)
                    srcs.update(kids)
            self._detach("vision", parent)
        for cid in info.get("added", []):
            self._attach("vision", cid)
        for cid in info.get("deleted", []):
            for srcs in self.reflex.values():
                srcs.discard(cid)
            self._detach("vision", cid)

    def restructure(self) -> dict:
        """慢：用这段时间的使用率做一次结构调整，跟随者一起增删。"""
        info = self.vision.restructure(self.usage[: len(self.vision.inputs)] / max(1, self.n_seen))
        self.apply_events(info)
        n = len(self.vision.inputs)
        self.usage = np.zeros(n)
        self.n_seen = 0
        self.chain = deque([(np.r_[v, np.zeros(n - len(v), dtype=bool)], m) for v, m in self.chain],
                           maxlen=self.chain_delay)
        return info

    # ------------------------------------------------------------ 本能
    def wire_reflex(self, action: str, sources, gain: float = 1.5) -> None:
        """先天直连：这批视觉神经元一起亮 → 驱动这个动作（驱动 = gain × 亮的比例）。"""
        self.reflex[action] = set(int(i) for i in sources)
        self.reflex_gain[action] = gain

    # ------------------------------------------------------------ 走一拍
    def step(self, image: np.ndarray, reason: dict[str, float] | None = None) -> dict[str, int]:
        """reason：测试用的外部"理性驱动"（直接加到运动区）；正常运行时不给，由前额叶经回程自己驱动。"""
        act = self.vision.respond(self.vision.encode(image))
        self.usage[: len(act)] += act
        self.n_seen += 1
        # 去程：视觉 + 运动神经元（上一拍的）→ 链 → 前额叶
        self.chain.append((act, self.motor.act.copy()))
        v_arr, m_arr = self.chain[0]
        ext = np.zeros(len(self.f_src), dtype=np.float32)
        for f, (region, nid) in enumerate(self.f_src):
            if not self.f_alive[f]:
                continue
            if region == "vision":
                if nid < len(v_arr):
                    ext[f] = v_arr[nid]
            else:
                ext[f] = m_arr[nid]
        a = self.pfc.step(ext)
        # 回程：前额叶里运动神经元的对应单元 → 链 → 同一个运动神经元
        self.ret.append(a[self.motor_f].copy())
        ret = self.ret[0]
        # 运动区：本能（不走链）+ 回程 +（测试用）外部理性驱动
        drive: dict[str, float] = {}
        for name, srcs in self.reflex.items():
            if srcs:
                ids = np.fromiter(srcs, dtype=int)
                drive[name] = drive.get(name, 0.0) + self.reflex_gain[name] * float(act[ids].mean())
        for name, v in (reason or {}).items():
            drive[name] = drive.get(name, 0.0) + v
        self.gd.tick()
        self.t += 1
        out = self.motor.step(drive, ret)
        return out

    # ------------------------------------------------------------ 看
    def n_followers(self) -> int:
        return int(sum(self.f_alive))
