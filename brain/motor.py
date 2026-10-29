# -*- coding: utf-8 -*-
"""运动区：肌肉控制神经元 vs 抑制神经元的比较决定动不动；整个系统只能经肌肉输出。

用户的框架：
  - 是否做某个动作 = 运动区里控制肌肉的神经元与抑制神经元之间的比较。
  - 大部分时间抑制神经元比肌肉控制神经元更活跃 → 肌肉大部分时间是放松的。
  - 运动区由整个大脑（甚至运动区自己）整体调控：
      · 各区（本能）直接驱动；
      · 运动区的**每个神经元**（控制肌肉的、抑制的、去抑制的）都经链式神经元接到前额叶，
        前额叶经回程（链）1:1 送回同一个神经元。前额叶在想别的事、没激活这些对应单元时，
        运动完全由其他区决定（边走路边想别的事）；前额叶想动 / 想压住某些肌肉时，
        靠它自己内部的相互激活 / 竞争把对应单元点亮。
  - 本能和理性想做同一件事就是协作（驱动相加），想做冲突的事才竞争。

每个动作一组神经元：
  muscle[0..levels-1]  控制肌肉的神经元（力度分档）
  I                    抑制神经元（独立细胞，有基础活动 tonic）
  D                    去抑制神经元（被自己这组的驱动点亮，压住 I）
每拍：
  d    = 本能驱动 + ret_gain · mean(回程到本组 muscle 的活动)
  D    = disinh · d + ret_gain · 回程到 D 的活动
  I    = max(0, tonic + lateral · 对手的 d + ret_gain · 回程到 I 的活动 − D)
  第 l 档 muscle 在 d − I > l · step 时亮；力度 = 亮的档数
没有回程时等于：I = tonic + lateral·对手驱动 − disinh·自己驱动（与之前一致）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotorCfg:
    levels: int = 10
    tonic: float = 1.0          # 抑制神经元的基础活动
    lateral: float = 1.0        # 对手的驱动 → 我的抑制
    disinh: float = 1.0         # 我的驱动 → 去抑制
    step: float = 0.2           # 每档力度要多出多少驱动
    ret_gain: float = 2.0       # 前额叶回程的强度


class Motor:
    def __init__(self, actions: list[str], rivals: list[tuple[str, str]] = (), cfg: MotorCfg | None = None):
        self.actions = list(actions)
        self.cfg = cfg or MotorCfg()
        self.idx = {a: i for i, a in enumerate(self.actions)}
        n = len(self.actions)
        self.rival = np.zeros((n, n))
        for a, b in rivals:
            self.rival[self.idx[a], self.idx[b]] = self.rival[self.idx[b], self.idx[a]] = 1.0
        self.per = self.cfg.levels + 2                 # 每组神经元数：muscle × levels + I + D
        self.n_neurons = n * self.per
        self.act = np.zeros(self.n_neurons, dtype=np.float32)
        self.inh = np.full(n, self.cfg.tonic)
        self.force = np.zeros(n, dtype=int)

    # 神经元编号
    def muscle_ids(self, a: str) -> np.ndarray:
        b = self.idx[a] * self.per
        return np.arange(b, b + self.cfg.levels)

    def inh_id(self, a: str) -> int:
        return self.idx[a] * self.per + self.cfg.levels

    def dis_id(self, a: str) -> int:
        return self.idx[a] * self.per + self.cfg.levels + 1

    def step(self, drive: dict[str, float], ret: np.ndarray | None = None) -> dict[str, int]:
        """drive：各区（本能）送来的驱动；ret：前额叶经回程送回每个运动神经元的活动（0..1）。"""
        c = self.cfg
        n = len(self.actions)
        d = np.zeros(n)
        for a, v in drive.items():
            d[self.idx[a]] += v
        r = np.zeros(self.n_neurons) if ret is None else np.asarray(ret, dtype=float)
        R = r.reshape(n, self.per)
        d = d + c.ret_gain * R[:, : c.levels].mean(axis=1)
        D = c.disinh * d + c.ret_gain * R[:, c.levels + 1]
        self.inh = np.maximum(0.0, c.tonic + c.lateral * (self.rival @ d) + c.ret_gain * R[:, c.levels] - D)
        margin = d - self.inh
        lv = np.arange(1, c.levels + 1)
        on = margin[:, None] > lv[None, :] * c.step
        self.force = on.sum(axis=1)
        A = np.zeros((n, self.per), dtype=np.float32)
        A[:, : c.levels] = on
        A[:, c.levels] = np.clip(self.inh / max(c.tonic, 1e-6), 0, 1)
        A[:, c.levels + 1] = np.clip(D / max(c.tonic, 1e-6), 0, 1)
        self.act = A.reshape(-1)
        return {a: int(self.force[i]) for i, a in enumerate(self.actions)}

    @property
    def relaxed(self) -> bool:
        return bool((self.force == 0).all())
