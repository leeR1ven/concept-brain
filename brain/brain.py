# -*- coding: utf-8 -*-
"""整脑：感觉区（耳朵）、前额叶、海马体、运动区，以及它们之间的链、回程、直连。

  感觉区：多层就近网络（brain/vision.py，权重出生固定）把输入压成特征。
          第 1 层以后、有特征的神经元 = 这个区的"皮层"：它们之间有自己的可塑连接（连续的声音记在这里，
          给个开头就顺着放出来 = 表征坍缩），也和别的区直连。
  链：皮层每个神经元一根专线送到前额叶的一个神经元、一根送到海马体的一个神经元（信号再强也只点亮那一个）；
      运动区所有神经元也有链。链单独点不亮前额叶神经元——进不进得来，看前额叶在想什么。
  前额叶：自己的一群神经元，内部出生就随机连满（稀疏），学得慢；活跃数靠动态平衡大致不变；
          回程送回各区（心里听到 / 想动），强度 ret_* 是可调的数值。
  海马体：自己的一群神经元，机制和前额叶一样（经链进来、有回程），只是连接涨得快、降得快 → 短期记忆。
  运动区：控制肌肉的神经元（每块肌肉 levels 个，亮几个 = 多大力）、抑制神经元（平时一直亮，压住肌肉）、
          去抑制神经元（压住抑制）。肌肉神经元收到的是各区信号的总和；连续动作记在运动区自己的连接里。
  区与区直连：耳朵 ↔ 运动区，出生随机、很弱，一起亮得多就变强。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .net import Path, PathCfg, Region, RegionCfg
from .vision import VisionSheet


@dataclass
class BrainCfg:
    # 感觉区皮层
    cortex_from_layer: int = 1        # 第 1 层（直接收输入线）不算皮层：不送链、不参加可塑连接
    mute_thr: float = 0.6             # 经常亮（使用率高于它）又没特征的：不送链、不参加可塑连接
    ear_pool: float = 1.0             # 耳朵皮层的整体抑制：抑制神经元收到和皮层一样的输入，按平均输入压全片
    ear_ear: PathCfg = field(default_factory=lambda: PathCfg(k=96, w_init=0.01, eta=0.05, fail=0.005, gamma=0.99, push=1.2))
    # 前额叶 / 海马体
    pfc: RegionCfg = field(default_factory=lambda: RegionCfg(pool_gain=1.0, target=0.08, balance_rate=0.02))
    hip: RegionCfg = field(default_factory=lambda: RegionCfg(pool_gain=1.0, target=0.08, balance_rate=0.02))
    pfc_pfc: PathCfg = field(default_factory=lambda: PathCfg(k=128, w_init=0.05, eta=0.02, fail=0.002, gamma=0.995, push=1.0))
    hip_hip: PathCfg = field(default_factory=lambda: PathCfg(k=128, w_init=0.05, eta=0.3, fail=0.03, gamma=0.8, push=1.0))
    chain_pfc: float = 0.5            # 链送进前额叶的强度（单独不够点亮）
    chain_hip: float = 0.5
    ret_pfc: float = 0.15             # 回程强度（可调旋钮）
    ret_hip: float = 0.15
    ret_motor: float = 0.5
    # 运动区
    n_muscles: int = 32
    levels: int = 3
    level_step: float = 0.25          # 第 l 档肌肉神经元的门槛 = 1 + l·level_step
    tonic: float = 1.2                # 抑制神经元的基础电流（平时一直亮）
    inh_w: float = 1.5                # 抑制神经元 → 肌肉神经元
    dis_w: float = 2.0                # 去抑制神经元 → 抑制神经元
    mot_mot: PathCfg = field(default_factory=lambda: PathCfg(k=64, w_init=0.02, eta=0.05, fail=0.005, gamma=0.99, push=1.2))
    # 区与区直连
    ear_mot: PathCfg = field(default_factory=lambda: PathCfg(k=16, w_init=0.02, eta=0.02, fail=0.002, gamma=0.995, push=0.5))
    mot_ear: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.01, eta=0.02, fail=0.002, gamma=0.995, push=0.5))
    decay_T: int = 100                # 每隔多少拍整体降权一次
    seed: int = 0


class Brain:
    def __init__(self, ear: VisionSheet, cfg: BrainCfg | None = None):
        self.cfg = c = cfg or BrainCfg()
        self.rng = np.random.default_rng(c.seed)
        self.sheet = ear
        self.ear = Region("ear", len(ear.inputs))
        self.ear.alive[:] = ear._alive
        lay = np.asarray(ear.layer)
        self.cortex = (lay >= c.cortex_from_layer) & self.ear.alive

        # ---- 运动区：肌肉神经元 M[m, l]、抑制神经元 I[m]、去抑制神经元 D[m]
        nm, L = c.n_muscles, c.levels
        self.M = np.arange(nm * L).reshape(nm, L)
        self.I = nm * L + np.arange(nm)
        self.D = nm * L + nm + np.arange(nm)
        self.mot = Region("motor", nm * L + 2 * nm)
        self.mot.thr[self.M] = 1.0 + c.level_step * np.arange(L)[None, :]
        self.mot.base[self.I] = c.tonic

        # ---- 前额叶 / 海马体：每个皮层候选神经元、每个运动神经元各一个
        ear_ids = np.nonzero(self.cortex)[0]
        mot_ids = np.arange(self.mot.n)
        self.src_of: dict[str, list] = {}
        self.of: dict[str, dict] = {}
        self.pfc = Region("pfc", len(ear_ids) + len(mot_ids), c.pfc)
        self.hip = Region("hip", len(ear_ids) + len(mot_ids), c.hip)
        for R in (self.pfc, self.hip):
            src = [("ear", int(i)) for i in ear_ids] + [("motor", int(i)) for i in mot_ids]
            self.src_of[R.name] = src
            self.of[R.name] = {s: j for j, s in enumerate(src)}

        self.paths: list[Path] = []
        mk = lambda *a, **k: self._add(Path(*a, rng=self.rng, **k))  # noqa: E731
        ear_cx = ear_ids
        M_all = self.M.reshape(-1)
        # 区内可塑连接
        mk("ear→ear", self.ear, self.ear, c.ear_ear, src_ids=ear_cx, dst_ids=ear_cx)
        mk("pfc→pfc", self.pfc, self.pfc, c.pfc_pfc, src_ids=np.arange(self.pfc.n), dst_ids=np.arange(self.pfc.n))
        mk("hip→hip", self.hip, self.hip, c.hip_hip, src_ids=np.arange(self.hip.n), dst_ids=np.arange(self.hip.n))
        mk("mot→mot", self.mot, self.mot, c.mot_mot, src_ids=M_all, dst_ids=mot_ids)
        # 运动区先天的抑制 / 去抑制
        mk("I⊣M", self.mot, self.mot, PathCfg(w_init=c.inh_w, sign=-1.0),
           pairs=[(int(self.I[m]), self.M[m]) for m in range(nm)])
        mk("D⊣I", self.mot, self.mot, PathCfg(w_init=c.dis_w, sign=-1.0),
           pairs=[(int(self.D[m]), [int(self.I[m])]) for m in range(nm)])
        # 区与区直连
        mk("ear→mot", self.ear, self.mot, c.ear_mot, src_ids=ear_cx, dst_ids=mot_ids)
        mk("mot→ear", self.mot, self.ear, c.mot_ear, src_ids=mot_ids, dst_ids=ear_cx)
        # 链与回程（一对一，固定）
        for R, cg, rg in ((self.pfc, c.chain_pfc, c.ret_pfc), (self.hip, c.chain_hip, c.ret_hip)):
            of = self.of[R.name]
            mk(f"ear⇒{R.name}", self.ear, R, PathCfg(w_init=cg), pairs=[(i, [of[("ear", int(i))]]) for i in ear_ids])
            mk(f"mot⇒{R.name}", self.mot, R, PathCfg(w_init=cg), pairs=[(i, [of[("motor", int(i))]]) for i in mot_ids])
            mk(f"{R.name}⇒ear", R, self.ear, PathCfg(w_init=rg), pairs=[(of[("ear", int(i))], [int(i)]) for i in ear_ids])
            mk(f"{R.name}⇒mot", R, self.mot, PathCfg(w_init=c.ret_motor),
               pairs=[(of[("motor", int(i))], [int(i)]) for i in mot_ids])

        self.regions = [self.ear, self.pfc, self.hip, self.mot]
        self.learning = True
        self.t = 0
        self.usage = np.zeros(self.ear.n)
        self.n_seen = 0

    def _add(self, p: Path) -> Path:
        self.paths.append(p)
        return p

    def path(self, name: str) -> Path:
        return next(p for p in self.paths if p.name == name)

    # ------------------------------------------------------------ 一拍
    def step(self, lines: np.ndarray, drive: np.ndarray | None = None) -> np.ndarray:
        """lines：耳朵输入线（sheet.encode 的结果）。drive：外部送进运动区每个神经元的电流（本能 / 外力）。
        返回每块肌肉的力度（亮了几档）。"""
        cur = {R.name: np.zeros(R.n, np.float32) for R in self.regions}
        for p in self.paths:                               # 全部用上一拍的状态
            cur[p.dst.name] += p.current()
        # 耳朵：前向逐层算；皮层神经元再加上 区内 / 回程 / 直连 送来的（以门槛为单位）− 整体抑制
        cx = self.cortex
        td = np.zeros(self.ear.n, np.float32)
        if cx.any():
            td[cx] = cur["ear"][cx] - self.cfg.ear_pool * float(cur["ear"][cx].mean())
        new = self.sheet.respond(lines, td) & self.ear.alive
        self.ear.prev, self.ear.s = self.ear.s, new
        self.pfc.fire(cur["pfc"])
        self.hip.fire(cur["hip"])
        m = cur["motor"] if drive is None else cur["motor"] + drive
        self.mot.fire(m)
        if self.learning:
            for p in self.paths:
                p.learn()
        for R in self.regions:
            R.settle()
        self.t += 1
        if self.learning and self.t % self.cfg.decay_T == 0:
            for p in self.paths:
                p.decay()
        self.usage += new
        self.n_seen += 1
        return self.mot.s[self.M].sum(axis=1)

    def reset(self) -> None:
        for R in self.regions:
            R.reset()

    # ------------------------------------------------------------ 没特征的不送链
    def _cut_ear(self, i: int) -> None:
        """耳朵神经元 i 退出皮层：它的出边、指向它的连接、它在前额叶 / 海马体的那个神经元都撤掉。"""
        for p in self.paths:
            if p.src is self.ear:
                p.cut_src([i])
            if p.dst is self.ear:
                p.redirect(i, [])
        for R in (self.pfc, self.hip):
            j = self.of[R.name].get(("ear", i))
            if j is not None:
                for p in self.paths:
                    if p.src is R:
                        p.cut_src([j])
                    if p.dst is R:
                        p.redirect(j, [])
                R.kill([j])
        self.cortex[i] = False

    def mute_featureless(self) -> int:
        """经常亮、没特征的皮层神经元（使用率 > mute_thr）撤出皮层。"""
        u = self.usage / max(self.n_seen, 1)
        ids = np.nonzero(self.cortex & (u > self.cfg.mute_thr))[0]
        for i in ids:
            self._cut_ear(int(i))
        return len(ids)

    # ------------------------------------------------------------ 神经元增删（慢）
    def restructure(self) -> dict:
        """耳朵常亮的分裂、从不亮的删除；它的链、前额叶 / 海马体神经元、可塑连接跟着变。"""
        u = self.usage / max(self.n_seen, 1)
        info = self.sheet.restructure(u)
        k = len(self.sheet.inputs) - self.ear.n
        if k > 0:
            self.ear.grow(k)
            self.cortex = np.r_[self.cortex, np.zeros(k, bool)]
            self.usage = np.r_[self.usage, np.zeros(k)]
        c = self.cfg
        for parent, kids in info["split"]:
            if self.cortex[parent]:
                for kid in kids:
                    self.cortex[kid] = True
                    for p in self.paths:
                        if p.src is self.ear and p.cfg.eta > 0:
                            p.copy_row(parent, kid)
                for p in self.paths:
                    if p.dst is self.ear and p.cfg.eta > 0:
                        p.redirect(parent, kids)
                for R, cg, rg in ((self.pfc, c.chain_pfc, c.ret_pfc), (self.hip, c.chain_hip, c.ret_hip)):
                    old = self.of[R.name][("ear", parent)]
                    new = R.grow(len(kids))
                    for kid, j in zip(kids, new):
                        self.src_of[R.name].append(("ear", kid))
                        self.of[R.name][("ear", kid)] = int(j)
                        self.path(f"ear⇒{R.name}").set_row(kid, [j], cg)
                        self.path(f"{R.name}⇒ear").set_row(int(j), [kid], rg)
                        self.path(f"{R.name}→{R.name}").copy_row(old, int(j))
                    self.path(f"{R.name}→{R.name}").redirect(old, new)
            self._cut_ear(parent)
            self.ear.kill([parent])
        for i in info["deleted"]:
            if self.cortex[i]:
                self._cut_ear(i)
            self.ear.kill([i])
        self.usage[:] = 0
        self.n_seen = 0
        return info
