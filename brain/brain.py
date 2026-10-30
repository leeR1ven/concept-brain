# -*- coding: utf-8 -*-
"""完整模型：听觉区、运动区（运动皮层 + 肌肉神经元）、海马体、前额叶，以及链、回程、区间直连
（同一拍里全部用上一拍的状态）。

  听觉区：立体局部网络把声音压成特征；皮层有自己的可塑连接（顺着放出连续的声音）。
  运动区：多层网络。运动皮层（输入层）表示"肌肉的组合"——同一个字的那组肌肉对应一组特定的神经元，
          用同一种立体网络从肌肉的状态压出来；别的区都连到运动皮层。运动皮层自己的连接学"哪个组合后面接哪个组合"，
          运动皮层 → 肌肉神经元的连接学"这个组合之后要动哪些肌肉"。
          肌肉神经元 vs 抑制神经元同一拍比较，平时抑制占上风；收到的是各路信号的总和。
  链：听觉区皮层、运动皮层、肌肉神经元的每个神经元，各一根线到海马体的一个神经元、一根到前额叶的一个神经元。
      海马体的链强（有真实输入时以输入为准，安静时空出的名额是回放）；前额叶的链弱（进不进得来看它在想什么）。
  回程：海马体 / 前额叶 → 听觉区（心里听到）、→ 运动皮层（想动）。强度是可调的数值。
  区间直连：听觉区 ↔ 运动皮层，出生随机、很弱，一起亮得多就变强。
  只能经肌肉输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .areas import (Circuit, MotorArea, MotorCfg, SenseArea, SenseCfg, ThoughtArea, ThoughtCfg, hip_cfg,
                    pfc_cfg)
from .net import Path, PathCfg, T
from .net3d import Net3D, Net3DCfg


@dataclass
class BrainCfg:
    net3d: Net3DCfg = field(default_factory=Net3DCfg)
    # 皮层学得慢、忘得也慢（整体降权接近 1）；海马体学得快、忘得快
    sense: SenseCfg = field(default_factory=lambda: SenseCfg(internal=PathCfg(k=96, w_init=0.01, eta=0.03, gamma=0.999)))
    # 运动皮层：肌肉状态（每块肌肉的力度）排成 8×4 的图，用同一种立体网络压成"组合特征"
    mctx_net: Net3DCfg = field(default_factory=lambda: Net3DCfg(levels=3, zs=(20, 40, 80), frac=(0.4, 0.3, 0.2)))
    mctx: SenseCfg = field(default_factory=lambda: SenseCfg(internal=PathCfg(k=96, w_init=0.01, eta=0.03, gamma=0.999)))
    hip: ThoughtCfg = field(default_factory=hip_cfg)
    pfc: ThoughtCfg = field(default_factory=pfc_cfg)
    motor: MotorCfg = field(default_factory=lambda: MotorCfg(internal=PathCfg(k=64, w_init=0.02, eta=0.01, gamma=0.999)))
    mctx_mot: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.002, eta=0.003, gamma=0.999))
    chain_hip: float = 3.0
    chain_pfc: float = 0.5
    ret_hip: float = 0.5          # 回程强度：可调旋钮
    ret_pfc: float = 0.5
    ear_mctx: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.005, eta=0.01, gamma=0.999))
    mctx_ear: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.005, eta=0.01, gamma=0.999))
    decay_T: int = 100
    seed: int = 0


class Brain:
    def __init__(self, H: int, W: int, n_muscles: int, cfg: BrainCfg | None = None):
        self.cfg = c = cfg or BrainCfg()
        c.motor.n_muscles = n_muscles
        rng = np.random.default_rng(c.seed)
        self.ear = ear = SenseArea("ear", Net3D(H, W, c.net3d, seed=c.seed), c.sense, seed=c.seed)
        self.mH = 8
        self.mW = (n_muscles + self.mH - 1) // self.mH
        self.mctx = mctx = SenseArea("mctx", Net3D(self.mH, self.mW, c.mctx_net, seed=c.seed + 5), c.mctx, seed=c.seed + 5)
        self.mot = mot = MotorArea("motor", c.motor, seed=c.seed)
        ne, nc, nm = ear.region.n, mctx.region.n, mot.region.n
        self.off = {"ear": 0, "mctx": ne, "motor": ne + nc}          # 在前额叶 / 海马体里各自对应的那一段
        cand_e = np.nonzero(ear.net.layer >= c.sense.cortex_from_layer)[0]   # 皮层候选（含预留空位）
        cand_c = np.nonzero(mctx.net.layer >= c.mctx.cortex_from_layer)[0]
        self.cand = {"ear": cand_e, "mctx": cand_c, "motor": np.arange(nm)}
        ids = np.r_[cand_e, ne + cand_c, ne + nc + np.arange(nm)]
        self.hip = ThoughtArea("hip", ne + nc + nm, c.hip, seed=c.seed + 1, ids=ids)
        self.pfc = ThoughtArea("pfc", ne + nc + nm, c.pfc, seed=c.seed + 2, ids=ids)
        src = {"ear": ear.region, "mctx": mctx.region, "motor": mot.region}
        paths = []
        for T, cg, rg in ((self.hip, c.chain_hip, c.ret_hip), (self.pfc, c.chain_pfc, c.ret_pfc)):
            for k, R in src.items():                                   # 链：每个神经元一根线到 T 里对应的那一个
                o = self.off[k]
                paths.append(Path(f"{k}⇒{T.name}", R, T.region, PathCfg(w_init=cg), rng,
                                  pairs=[(int(i), [o + int(i)]) for i in self.cand[k]]))
            for k in ("ear", "mctx"):                                  # 回程：心里听到 / 想动
                o = self.off[k]
                paths.append(Path(f"{T.name}⇒{k}", T.region, src[k], PathCfg(w_init=rg), rng,
                                  pairs=[(o + int(i), [int(i)]) for i in self.cand[k]]))
        paths += [Path("mctx→motor", mctx.region, mot.region, c.mctx_mot, rng, src_ids=cand_c, dst_ids=np.arange(nm)),
                  Path("ear→mctx", ear.region, mctx.region, c.ear_mctx, rng, src_ids=cand_e, dst_ids=cand_c),
                  Path("mctx→ear", mctx.region, ear.region, c.mctx_ear, rng, src_ids=cand_c, dst_ids=cand_e)]
        self.c = Circuit([ear, mctx, self.hip, self.pfc, mot], paths, decay_T=c.decay_T)
        self._sync_mirrors()

    def _sync_mirrors(self) -> None:
        """前额叶 / 海马体里对应的神经元，跟着听觉区 / 运动皮层神经元亮不亮的开关走（没特征的、删掉的不送链）。"""
        for X in (self.hip, self.pfc):
            for k, A in (("ear", self.ear), ("mctx", self.mctx)):
                o = self.off[k]
                X.region.alive[o:o + A.region.n] = T(A.cortex)
            X.region.alive[self.off["motor"]:] = True
            X.region._alive_ref = None

    @property
    def learning(self) -> bool:
        return self.c.learning

    @learning.setter
    def learning(self, v: bool) -> None:
        self.c.learning = v

    def muscle_lines(self) -> np.ndarray:
        """运动皮层看到的：上一拍每块肌肉的力度（排成 mH×mW 的图）。"""
        f = np.zeros(self.mH * self.mW, dtype=int)
        f[: self.mot.cfg.n_muscles] = self.mot.force()
        return self.mctx.net.encode(f.reshape(self.mH, self.mW))

    def step(self, lines: np.ndarray, drive: np.ndarray | None = None) -> np.ndarray:
        """lines：耳朵输入线；drive：外部送进运动区的电流（外力 / 本能）。返回每块肌肉的力度。"""
        self.c.step(ext=None if drive is None else {"motor": drive},
                    lines={"ear": lines, "mctx": self.muscle_lines()})
        return self.mot.force()

    def mute_featureless(self) -> int:
        n = 0
        for A in (self.ear, self.mctx):
            ids = A.featureless()
            A.mute(ids)
            n += len(ids)
        self._sync_mirrors()
        return n

    def restructure(self) -> dict:
        """常亮的分裂、从不亮的删除（很慢）；链和前额叶 / 海马体里对应的神经元跟着开关。"""
        info = self.ear.restructure()
        self.mctx.restructure()
        self._sync_mirrors()
        return info

    def reset(self) -> None:
        self.c.reset()
