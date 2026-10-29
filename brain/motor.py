# -*- coding: utf-8 -*-
"""运动区：肌肉控制神经元 vs 抑制神经元的比较决定动不动；整个系统只能经肌肉输出。

用户的框架：
  - 是否做某个动作 = 运动区里控制肌肉的神经元与抑制神经元之间的比较。
  - 大部分时间抑制神经元比肌肉控制神经元更活跃 → 肌肉大部分时间是放松的。
  - 运动区做什么由其他各区控制；各区（本能）和前额叶（理性）之间协调或竞争：
    两边想做同一件事就是协作（驱动相加），想做不同的事才争夺控制权。

实现（每个动作一组）：
  - 驱动 d：本能（感觉区先天直连）+ 理性（前额叶）送来的总量，两边同向就相加。
  - 抑制神经元（独立细胞，不在连接上写负权重）：
        inh = max(0, tonic + lateral · 对手的驱动 − disinh · 自己的驱动)
    tonic > 0 → 没有驱动时抑制占上风、肌肉放松；
    对手（互相冲突的动作）越想动，自己被压得越狠；自己被大量驱动时打开自己的抑制（去抑制）。
  - 肌肉控制神经元：levels 个力度档，第 l 档在 d − inh > l · step 时亮；力度 = 亮的档数（单调分级）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotorCfg:
    levels: int = 10
    tonic: float = 1.0          # 抑制神经元的基础活动
    lateral: float = 1.0        # 对手的驱动 → 我的抑制
    disinh: float = 1.0         # 我的驱动 → 打开我的抑制
    step: float = 0.2           # 每档力度要多出多少驱动


class Motor:
    def __init__(self, actions: list[str], rivals: list[tuple[str, str]] = (), cfg: MotorCfg | None = None):
        self.actions = list(actions)
        self.cfg = cfg or MotorCfg()
        self.idx = {a: i for i, a in enumerate(self.actions)}
        n = len(self.actions)
        self.rival = np.zeros((n, n))
        for a, b in rivals:
            self.rival[self.idx[a], self.idx[b]] = self.rival[self.idx[b], self.idx[a]] = 1.0
        self.inh = np.full(n, self.cfg.tonic)
        self.force = np.zeros(n, dtype=int)

    def step(self, drive: dict[str, float]) -> dict[str, int]:
        c = self.cfg
        d = np.zeros(len(self.actions))
        for a, v in drive.items():
            d[self.idx[a]] += v
        self.inh = np.maximum(0.0, c.tonic + c.lateral * (self.rival @ d) - c.disinh * d)
        margin = d - self.inh
        lv = np.arange(1, c.levels + 1)
        self.force = (margin[:, None] > lv[None, :] * c.step).sum(axis=1)
        return {a: int(self.force[i]) for i, a in enumerate(self.actions)}

    @property
    def relaxed(self) -> bool:
        return bool((self.force == 0).all())
