# -*- coding: utf-8 -*-
"""完整模型：听觉区、海马体、前额叶、运动区，以及链、回程、区间直连（同一拍里全部用上一拍的状态）。

  听觉区（SenseArea）：立体局部网络把声音压成特征；皮层有自己的可塑连接（顺着放出连续的声音）。
  链：听觉区皮层、运动区的每个神经元，各一根线到海马体的一个神经元、一根到前额叶的一个神经元。
      海马体的链强（有真实输入时以输入为准，安静时空出的名额是回放）；
      前额叶的链弱（单独进不来，进不进得来看前额叶在想什么）。
  回程：海马体 / 前额叶 → 听觉区（心里听到）、→ 运动区（想动）。强度是可调的数值（理性 vs 本能的旋钮）。
  运动区：控制肌肉的神经元 vs 抑制神经元；收到各路信号的总和；连续动作记在自己的连接里。
  区间直连：听觉区 ↔ 运动区，出生随机、很弱，一起亮得多就变强。
  只能经肌肉输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .areas import (Circuit, MotorArea, MotorCfg, SenseArea, SenseCfg, ThoughtArea, ThoughtCfg, hip_cfg,
                    pfc_cfg)
from .net import Path, PathCfg
from .net3d import Net3D, Net3DCfg


@dataclass
class BrainCfg:
    net3d: Net3DCfg = field(default_factory=Net3DCfg)
    # 皮层学得慢、忘得也慢（整体降权接近 1）；海马体学得快、忘得快
    sense: SenseCfg = field(default_factory=lambda: SenseCfg(internal=PathCfg(k=96, w_init=0.01, eta=0.03, gamma=0.999)))
    hip: ThoughtCfg = field(default_factory=hip_cfg)
    pfc: ThoughtCfg = field(default_factory=pfc_cfg)
    motor: MotorCfg = field(default_factory=lambda: MotorCfg(internal=PathCfg(k=64, w_init=0.02, eta=0.01, gamma=0.999)))
    chain_hip: float = 3.0
    chain_pfc: float = 0.5
    ret_hip: float = 0.5          # 回程强度：可调旋钮
    ret_pfc: float = 0.3
    ret_motor: float = 0.5        # 想法推嘴：单靠一路推不开抑制（去抑制门槛 2.5），要几路一起
    ear_mot: PathCfg = field(default_factory=lambda: PathCfg(k=16, w_init=0.002, eta=0.001, gamma=0.999))   # 每个运动神经元收 ~90 根，学习要慢
    mot_ear: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.005, eta=0.02, gamma=0.999))
    decay_T: int = 100
    seed: int = 0


class Brain:
    def __init__(self, H: int, W: int, n_muscles: int, cfg: BrainCfg | None = None):
        self.cfg = c = cfg or BrainCfg()
        c.motor.n_muscles = n_muscles
        rng = np.random.default_rng(c.seed)
        self.ear = ear = SenseArea("ear", Net3D(H, W, c.net3d, seed=c.seed), c.sense, seed=c.seed)
        self.mot = mot = MotorArea("motor", c.motor, seed=c.seed)
        ne, nm = ear.region.n, mot.region.n
        cand = np.nonzero(ear.net.layer >= c.sense.cortex_from_layer)[0]   # 皮层候选（含预留空位）
        self.mirror_ids = np.r_[cand, ne + np.arange(nm)]
        self.hip = ThoughtArea("hip", ne + nm, c.hip, seed=c.seed + 1, ids=self.mirror_ids)
        self.pfc = ThoughtArea("pfc", ne + nm, c.pfc, seed=c.seed + 2, ids=self.mirror_ids)
        paths = []
        for T, cg, rg in ((self.hip, c.chain_hip, c.ret_hip), (self.pfc, c.chain_pfc, c.ret_pfc)):
            nm_ = T.name
            paths += [
                Path(f"ear⇒{nm_}", ear.region, T.region, PathCfg(w_init=cg), rng, pairs=[(int(i), [int(i)]) for i in cand]),
                Path(f"mot⇒{nm_}", mot.region, T.region, PathCfg(w_init=cg), rng, pairs=[(j, [ne + j]) for j in range(nm)]),
                Path(f"{nm_}⇒ear", T.region, ear.region, PathCfg(w_init=rg), rng, pairs=[(int(i), [int(i)]) for i in cand]),
                Path(f"{nm_}⇒mot", T.region, mot.region, PathCfg(w_init=c.ret_motor), rng,
                     pairs=[(ne + j, [j]) for j in range(nm)]),
            ]
        paths += [Path("ear→mot", ear.region, mot.region, c.ear_mot, rng, src_ids=cand, dst_ids=np.arange(nm)),
                  Path("mot→ear", mot.region, ear.region, c.mot_ear, rng, src_ids=np.arange(nm), dst_ids=cand)]
        self.c = Circuit([ear, self.hip, self.pfc, mot], paths, decay_T=c.decay_T)
        self._sync_mirrors()

    def _sync_mirrors(self) -> None:
        """前额叶 / 海马体里对应听觉区的那些神经元，跟着听觉区皮层神经元亮不亮的开关走（没特征的、删掉的不送链）。"""
        ne = self.ear.region.n
        for T in (self.hip, self.pfc):
            T.region.alive[:ne] = self.ear.cortex
            T.region.alive[ne:] = True

    @property
    def learning(self) -> bool:
        return self.c.learning

    @learning.setter
    def learning(self, v: bool) -> None:
        self.c.learning = v

    def step(self, lines: np.ndarray, drive: np.ndarray | None = None) -> np.ndarray:
        """lines：耳朵输入线；drive：外部送进运动区的电流（外力 / 本能）。返回每块肌肉的力度。"""
        self.c.step(ext=None if drive is None else {"motor": drive}, lines={"ear": lines})
        return self.mot.force()

    def mute_featureless(self) -> int:
        ids = self.ear.featureless()
        self.ear.mute(ids)
        self._sync_mirrors()
        return len(ids)

    def restructure(self) -> dict:
        """常亮的分裂、从不亮的删除（很慢）；链和前额叶 / 海马体里对应的神经元跟着开关。"""
        info = self.ear.restructure()
        self._sync_mirrors()
        return info

    def reset(self) -> None:
        self.c.reset()
