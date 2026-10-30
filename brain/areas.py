# -*- coding: utf-8 -*-
"""脑的几个部分，各自能单独运行、单独验证，再用 Circuit 组合起来（同一拍里全部用上一拍的状态）。

  SenseArea   感觉区：立体局部网络（net3d.py，权重出生固定，每层亮的比例固定）把输入压成特征；
              有特征的神经元 = 皮层，它们之间有自己的可塑连接 → 听到开头顺着放出一段连续的声音（表征坍缩）。
  ThoughtArea 一群互相连接的神经元（亮的数量大致不变）。前额叶、海马体现在都从它开始：
              海马体只管短期记忆，就是它本身（连接涨得快、降得快）；前额叶以后会多出别的东西（比前额叶复杂）。
  MotorArea   运动区：控制肌肉的神经元（亮几个 = 多大力）、抑制神经元（平时一直亮）、去抑制神经元；
              肌肉神经元收到的是各路信号的总和；连续动作记在运动区自己的连接里。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .net import DEV, N, Path, PathCfg, Region, RegionCfg, T
from .net3d import Net3D


# ================================================================ 感觉区
@dataclass
class SenseCfg:
    cortex_from_layer: int = 1        # 第 1 层（几乎是输入的复制）不算皮层：不送链、不参加可塑连接
    mute_thr: float = 0.6             # 经常亮（使用率高于它）又没特征的：退出皮层
    internal: PathCfg = field(default_factory=lambda: PathCfg(k=96, w_init=0.01, eta=0.001, gamma=0.95))


class SenseArea:
    def __init__(self, name: str, net: Net3D, cfg: SenseCfg | None = None, seed: int = 0):
        self.name, self.net = name, net
        self.cfg = c = cfg or SenseCfg()
        self.rng = np.random.default_rng(seed)
        self.region = R = Region(name, net.n)
        R.alive = T(net.alive)
        cand = np.nonzero(net.layer >= c.cortex_from_layer)[0]      # 含预留的空位：分裂出的新神经元一出生就有连接
        self.cortex = (net.layer >= c.cortex_from_layer) & net.alive  # numpy（结构）；_cx 是每拍用的张量
        self._cx = T(self.cortex)
        self.inner = Path(f"{name}→{name}", R, R, c.internal, self.rng, src_ids=cand, dst_ids=cand)
        self.regions, self.paths = [R], [self.inner]
        self.clear_usage()

    def fire(self, cur: dict, lines: dict) -> None:
        """外部送来的电流（区内连接、回程、直连）加在得分上；每层亮多少由网络自己定（固定比例，或随新鲜驱动变）。"""
        R = self.region
        x = R.x if R.cfg.std_U > 0 else None
        new = self.net.respond(lines[self.name], torch.where(self._cx, cur[R.name], 0.0), x) & R.alive
        R.set(new)
        self.usage += new
        self.n_seen += 1

    def pattern(self, lines) -> np.ndarray:
        """只看外界输入、没有任何内部推动时，皮层亮哪些（numpy，量参照用）。"""
        return N(self.net.respond(lines)) & self.cortex

    def featureless(self) -> np.ndarray:
        u = N(self.usage) / max(self.n_seen, 1)
        return np.nonzero(self.cortex & (u > self.cfg.mute_thr))[0]

    def clear_usage(self) -> None:
        self.usage = torch.zeros(self.region.n, dtype=torch.int32, device=DEV)
        self.n_seen = 0

    def mute(self, ids) -> None:
        """没特征的（常亮的）退出皮层：不送链、不参加可塑连接。"""
        if len(ids):
            self.inner.cut_src(ids)
            self.inner.cut_dst(ids)
            self.cortex[ids] = False
            self._cx = T(self.cortex)

    def restructure(self) -> dict:
        """常亮的分裂、从不亮的删除（慢）。分裂出的新神经元继承父神经元学来的出边；指向父神经元的连接一半改指它。"""
        R = self.region
        info = self.net.restructure(N(self.usage) / max(self.n_seen, 1))
        for p, f in info["split"]:
            R.alive[f] = True
            self.cortex[f] = bool(self.net.layer[f] >= self.cfg.cortex_from_layer)
            if self.cortex[f]:
                self.inner.copy_row(p, f)
                self.inner.move_dst(p, f, 0.5)
        if info["deleted"]:
            R.kill(info["deleted"])
            self.cortex[info["deleted"]] = False
        self._cx = T(self.cortex)
        R._alive_ref = None
        self.clear_usage()
        return info


# ================================================================ 前额叶 / 海马体
@dataclass
class ThoughtCfg:
    region: RegionCfg = field(default_factory=lambda: RegionCfg(frac=0.08))
    internal: PathCfg = field(default_factory=lambda: PathCfg(k=128, w_init=0.05, eta=0.02, gamma=0.995))


def pfc_cfg() -> ThoughtCfg:
    """前额叶：亮的范围大、学得慢。出生的随机连接丰富一些，让它有自己的活动，不只是输入的复印件。"""
    return ThoughtCfg(region=RegionCfg(frac=0.08), internal=PathCfg(k=96, w_init=0.05, eta=0.02, gamma=0.999, budget=1.0))


def hip_cfg() -> ThoughtCfg:
    """海马体：只管短期记忆，连接涨得快、降得快。"""
    return ThoughtCfg(region=RegionCfg(frac=0.05), internal=PathCfg(k=96, w_init=0.01, eta=0.5, gamma=0.7, budget=2.0))


class ThoughtArea:
    def __init__(self, name: str, n: int, cfg: ThoughtCfg | None = None, seed: int = 0, ids=None):
        """ids：用到的神经元（其余的位置先空着，不亮）。"""
        self.name = name
        self.cfg = c = cfg or ThoughtCfg()
        self.rng = np.random.default_rng(seed)
        self.region = R = Region(name, n, c.region)
        ids = np.arange(n) if ids is None else np.asarray(ids)
        alive = np.zeros(n, dtype=bool)
        alive[ids] = True
        R.alive = T(alive)
        self.inner = Path(f"{name}→{name}", R, R, c.internal, self.rng, src_ids=ids, dst_ids=ids)
        self.regions, self.paths = [R], [self.inner]

    def fire(self, cur: dict, lines: dict) -> None:
        self.region.fire(cur[self.name])


# ================================================================ 运动区
@dataclass
class MotorCfg:
    n_muscles: int = 32
    levels: int = 3
    level_step: float = 0.25          # 第 l 档肌肉神经元的门槛 = 1 + l·level_step
    tonic: float = 1.2                # 抑制神经元的基础电流（平时一直亮）
    inh_w: float = 1.5                # 抑制神经元 → 肌肉神经元
    dis_w: float = 2.0                # 去抑制神经元 → 抑制神经元
    dis_thr: float = 1.2              # 去抑制神经元的门槛：零散的推力撤不掉抑制，要几路一起（和练熟后学得到的推力相当）
    beat: int = 3                     # 节拍：一个音节几拍（= 世界里一个字的长度）
    slots: int = 4                    # 每个节拍边界最多放开几块肌肉（一个字用几块）
    go_w: float = 1.2                 # 起动 → 所有去抑制神经元（只在边界那一拍）
    go_thr: float = 1.0
    internal: PathCfg = field(default_factory=lambda: PathCfg(k=64, w_init=0.02, eta=0.05, gamma=0.99))


class MotorArea:
    """肌肉控制神经元 vs 抑制神经元：同一拍里比较（抑制是即时的，不往后拖一拍）。
      去抑制神经元 D：收到的电流 ≥ dis_thr 就亮；
      抑制神经元 I：基础电流 tonic + 收到的 − dis_w·（这一拍它的 D 亮没亮）≥ 1 就亮（平时一直亮）；
      肌肉神经元 M：收到的 − inh_w·（这一拍它的 I 亮没亮）≥ 1 + 档·level_step 就亮；亮几档 = 多大力。
    "什么时候动"和"动哪些"分开（真实大脑：基底节 / 丘脑的起动信号与内容无关；鸣禽每个音节开头丘脑给一下）：
      起动神经元 go：与内容无关，亮了给所有 D 加 go_w——只有被内容支持的那几块肌肉越过门槛（谁先过由内容定）；
      节拍：go 一亮就开始数拍，每 beat 拍一个边界。D 只在边界上换：边界那一拍放开旧的、按内容选新的
            （最多 slots 块，挑推力最大的），其余拍子 D 自己撑着（出生就有的保持）；边界上没有任何 D 过门槛 = 说完。
      说话时力度按拍子加上去（第 1 拍 1 档、第 2 拍 2 档……），也是出生就有的结构，不用学。"""

    def __init__(self, name: str = "motor", cfg: MotorCfg | None = None, seed: int = 0):
        self.name = name
        self.cfg = c = cfg or MotorCfg()
        self.rng = np.random.default_rng(seed)
        nm, L = c.n_muscles, c.levels
        self.M = np.arange(nm * L).reshape(nm, L)        # 控制肌肉的神经元 M[肌肉, 档]
        self.I = nm * L + np.arange(nm)                  # 抑制神经元
        self.D = nm * L + nm + np.arange(nm)             # 去抑制神经元
        self.region = R = Region(name, nm * L + 2 * nm, RegionCfg(frac=0.0))   # 平时不动：用门槛
        thr = np.ones(R.n, dtype=np.float32)
        thr[self.M] = 1.0 + c.level_step * np.arange(L)[None, :]
        R.thr = T(thr)
        self.inner = Path(f"{name}→{name}", R, R, c.internal, self.rng,     # 学来的连接只从肌肉神经元出发；
                          src_ids=self.M.reshape(-1), dst_ids=np.arange(R.n))     # 抑制 / 去抑制的作用是上面固定的
        self.go = Region(f"{name}_go", 1, RegionCfg(frac=0.0, thr=c.go_thr))
        self.regions, self.paths = [R, self.go], [self.inner]
        self._M, self._I, self._D = T(self.M), T(self.I), T(self.D)
        self._lvl = T(np.arange(L))
        self.phase, self.speaking = 0, False

    def fire(self, cur: dict, lines: dict) -> None:
        c, R = self.cfg, self.region
        self.go.fire(cur[self.go.name])
        x = cur[self.name]
        dx = x[self._D]
        if (bool(self.go.s.any()) and not self.speaking) or (self.speaking and self.phase == c.beat - 1):   # 节拍边界（正说着时起动不打断）
            d_on = dx + c.go_w >= c.dis_thr
            if int(d_on.sum()) > c.slots:                                             # 竞争：只放开推力最大的几块
                keep = torch.zeros_like(d_on)
                keep[torch.topk(torch.where(d_on, dx, torch.full_like(dx, -torch.inf)), c.slots).indices] = True
                d_on = keep
            self.speaking, self.phase = bool(d_on.any()), 0
        elif self.speaking:                                                            # 字的中间：撑着
            d_on = R.s[self._D].clone()
            self.phase += 1
        else:
            d_on = dx >= c.dis_thr
        i_on = x[self._I] + c.tonic - c.dis_w * d_on.float() >= 1.0
        m_on = x[self._M] - c.inh_w * i_on.float()[:, None] >= R.thr[self._M]
        if self.speaking:                                                              # 力度按拍子加上去
            m_on = m_on | (d_on[:, None] & (self._lvl <= self.phase)[None, :])
        new = torch.zeros(R.n, dtype=torch.bool, device=DEV)
        new[self._M] = m_on
        new[self._I] = i_on
        new[self._D] = d_on
        R.set(new)

    def force(self) -> np.ndarray:
        """每块肌肉这一拍的力度 = 亮了几档（numpy，给身体用）。"""
        return N(self.region.s[self._M].sum(dim=1))

    def push(self, muscles, level: int | None = None) -> np.ndarray:
        """外力带着这些肌肉动：推它们的肌肉神经元 + 去抑制神经元。level：推到第几档（None = 满档）；
        一个字里力是逐拍加上去的（第 1 拍 1 档、第 2 拍 2 档……），运动区里就有了字内部的先后。"""
        c = self.cfg
        L = c.levels if level is None else level
        d = np.zeros(self.region.n, np.float32)
        for m in muscles:
            d[self.M[m, :L]] += 1.0 + c.level_step * (L - 1) + 0.05
            d[self.D[m]] += c.dis_thr + 0.5
        return d

    def cue(self, muscles) -> np.ndarray:
        """外力（教的人）在节拍边界那一拍指出这个字用哪几块肌肉：推它们的去抑制神经元。"""
        d = np.zeros(self.region.n, np.float32)
        for m in muscles:
            d[self.D[m]] += self.cfg.dis_thr + 0.5
        return d

    def reset(self) -> None:
        self.phase, self.speaking = 0, False

    def stop(self, strength: float = 2.0) -> np.ndarray:
        """外力让嘴停下：推所有抑制神经元。"""
        d = np.zeros(self.region.n, np.float32)
        d[self.I] += strength
        return d


class Unit:
    """几个门槛型神经元（电流 ≥ thr 就亮），比如"新鲜输入"小群。"""

    def __init__(self, name: str, n: int, thr: float):
        self.name = name
        self.region = Region(name, n, RegionCfg(frac=0.0, thr=thr))
        self.regions, self.paths = [self.region], []

    def fire(self, cur: dict, lines: dict) -> None:
        self.region.fire(cur[self.name])


# ================================================================ 组合
class Circuit:
    """几个部分 + 它们之间的连接，同一拍里全部用上一拍的状态。"""

    def __init__(self, parts: list, paths: list | None = None, decay_T: int = 100):
        self.parts = list(parts)
        self.paths = [p for part in self.parts for p in part.paths] + list(paths or [])
        self.decay_T = decay_T
        self.learning = True
        self.t = 0

    @property
    def regions(self):
        return [R for part in self.parts for R in part.regions]

    def step(self, ext: dict | None = None, lines: dict | None = None) -> None:
        """ext：外部送进各区的电流 {区名: 数组}；lines：各感觉区的输入线 {区名: 数组}。"""
        cur = {R.name: torch.zeros(R.n, dtype=torch.float32, device=DEV) for R in self.regions}
        for p in self.paths:
            cur[p.dst.name] += p.current()
        for k, v in (ext or {}).items():
            cur[k] = cur[k] + T(v, torch.float32)
        for part in self.parts:
            part.fire(cur, lines or {})
        if self.learning:
            for p in self.paths:
                p.learn()
        for R in self.regions:                      # 学完再更新余量（学习用的是传递那一拍的余量）
            R.deplete()
            if self.learning:
                R.homeostasis()
        self.t += 1
        if self.learning and self.t % self.decay_T == 0:
            for p in self.paths:
                p.decay()

    def reset(self) -> None:
        for R in self.regions:
            R.reset()
        for part in self.parts:
            if hasattr(part, "reset"):
                part.reset()

    def path(self, name: str) -> Path:
        return next(p for p in self.paths if p.name == name)
