# -*- coding: utf-8 -*-
"""整脑：各感觉区 → 链 → 前额叶 / 海马体 → 回程 → 运动区 → 肌肉（唯一输出）。

  感觉区（任意多个，brain/vision.py 的多层感觉片）──┐ 每个有特征的神经元一根链式专用线（1:1，延迟 chain_delay 拍）；
  运动区的每个神经元（控制肌肉 / 抑制 / 去抑制）──────┤ 常亮、没特征的神经元不送链
                                                    ├──► 前额叶（brain/pfc.py，变得慢）
                                                    ├──► 海马体（同样的机制、自己的神经元，变得快、忘得快）
                                                    │      └── 输出 1:1 送回前额叶
  本能：感觉区先天直连运动区（不走链）──────────────┼──► 运动区（brain/motor.py）
  前额叶里运动神经元的对应单元 ── 回程（链，1:1）────┘   维持：只扶住正在动的神经元
  海马体 / 前额叶里感觉神经元的对应单元 ── 回程 1:1 ──► 同一个感觉神经元（在心里看到 / 听到；
      回出来的东西再像平常一样经链送进去，回忆就沿着"听的时候学到的顺序"往下走）
  前额叶 ── 发起（出生就随机铺满的弱连接，训练中修改）──┘   发起：前额叶的大致意图点亮哪些运动神经元

  - 前额叶在想别的事、没激活运动相关的单元时：运动完全由其他区（本能）决定。
  - 前额叶 ↔ 各区之间的回路是正常现象。
  - 跟随增删：感觉区神经元分裂 / 删除时，它的链、前额叶单元、海马体单元一起增删；
    本能直连的来源身份由子单元继承。跟随者 f ↔ (区, 神经元编号)。
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .motor import Motor
from .pfc import PFC, PFCCfg, apply_budget
from .vision import VisionSheet


class Brain:
    def __init__(self, sensors, motor: Motor, chain_delay: int = 2,
                 pfc_cfg: PFCCfg | None = None, hippo_cfg: PFCCfg | None = None, hippo_gain: float = 1.0,
                 init_w: float = 0.0005, eta_init: float = 0.05, fail_init: float = 0.01, init_budget: float = 1.0,
                 decay_T: int = 200, cortex_gamma: float = 1.0, hippo_gamma: float = 0.9,
                 ext_gain: float = 1.5, pfc_ext_gain: float = 0.3, mute_thr: float = 0.6, chain_from_layer: int = 1,
                 sense_cfg: PFCCfg | None = None, sense_from_layer: int = 1,
                 sense_ret_hippo: float = 1.0, sense_ret_pfc: float = 0.5):
        if isinstance(sensors, VisionSheet):
            sensors = {"vision": sensors}
        self.sensors: dict[str, VisionSheet] = dict(sensors)
        self.first = next(iter(self.sensors))
        self.motor = motor
        self.chain_delay = chain_delay
        self.f_src: list[tuple[str, int]] = []
        self.f_alive: list[bool] = []
        self.f_of: dict[tuple[str, int], int] = {}
        # 靠近输入层的神经元经常亮、没特征：第 chain_from_layer 层（从 0 数）之前的层不送链
        self.chain_from_layer = chain_from_layer
        srcs = [(name, int(cid)) for name, sheet in self.sensors.items() for cid in np.nonzero(sheet._alive)[0]
                if sheet.layer[cid] >= chain_from_layer]
        srcs += [("motor", mid) for mid in range(motor.n_neurons)]
        pfc_cfg = pfc_cfg or PFCCfg()
        self.pfc = PFC(len(srcs), pfc_cfg)
        # 海马体：同样的机制，记细节——学得快、上限大；"亮了没带亮对方"削得慢（削得快会把刚学的马上削掉，实测）；
        # 忘得快靠整体降权（hippo_gamma）
        self.hippo = PFC(len(srcs), hippo_cfg or PFCCfg(eta=0.5, budget=3.0, fail=0.002, seed=pfc_cfg.seed + 1))
        self.hippo_gain = hippo_gain
        for f, key in enumerate(srcs):
            self.f_src.append(key)
            self.f_alive.append(True)
            self.f_of[key] = f
        self._index = None
        self.motor_f = np.array([self.f_of[("motor", m)] for m in range(motor.n_neurons)])
        # 发起：前额叶 → 运动神经元，出生就随机铺满弱连接，训练中修改（加强 / 往出生权重退）
        rng = np.random.default_rng(pfc_cfg.seed + 7)
        self.init_w = init_w
        self.R0 = rng.uniform(0, init_w, (self.pfc.n, motor.n_neurons)).astype(np.float32)
        self.R = self.R0.copy()
        self.eta_init, self.fail_init, self.init_budget = eta_init, fail_init, init_budget
        self.decay_T, self.cortex_gamma, self.hippo_gamma = decay_T, cortex_gamma, hippo_gamma
        self.ext_gain = ext_gain          # 链送进海马体的信号强度（记细节：单靠它就能点亮）
        self.pfc_ext_gain = pfc_ext_gain  # 链送进前额叶的信号强度（< 门槛：单个信号点不亮，只在当前想法范围上推一把）
        # 感觉区内部的赫布连接（链之后的那几层）：被激活后沿学好的顺序一个接一个亮（表征坍缩）
        # 只在第 sense_from_layer 层及更深的层之间（第 1 层靠近输入、一半常亮：压它会把整个感觉区压垮，实测）
        self.sense_from_layer = sense_from_layer
        self.sense_cfg = sense_cfg or PFCCfg(target=0.0, pool_gain=2.0, w_init=0.0002, v_init=0.0, eta=0.05,
                                             budget=1.5, seed=pfc_cfg.seed + 3)
        self.sense_pop = {}
        for name, sheet in self.sensors.items():
            pop = PFC(len(sheet.inputs), self.sense_cfg)
            for i in np.nonzero((sheet._layer < sense_from_layer) | ~sheet._alive)[0]:
                pop.remove(int(i))
            self.sense_pop[name] = pop
        self.mute_thr = mute_thr        # 使用率 ≥ 它的感觉神经元算"经常亮、没特征"，不送链
        # 回程到感觉区（在心里看到 / 听到）：海马体（细节）、前额叶（大致）里感觉神经元的对应单元 1:1 送回同一个神经元
        self.sense_ret_hippo, self.sense_ret_pfc = sense_ret_hippo, sense_ret_pfc
        z = ({k: np.zeros(len(s.inputs), dtype=bool) for k, s in self.sensors.items()},
             np.zeros(motor.n_neurons, dtype=np.float32))
        self.chain = deque([z] * chain_delay, maxlen=chain_delay)          # 去程
        zm = np.zeros(motor.n_neurons, dtype=np.float32)
        self.ret = deque([(zm, zm)] * chain_delay, maxlen=chain_delay)      # 回程：(维持, 发起)
        self.reflex: dict[str, tuple[str, set[int]]] = {}
        self.reflex_gain: dict[str, float] = {}
        self.usage = {k: np.zeros(len(s.inputs)) for k, s in self.sensors.items()}
        self.n_seen = 0
        self.t = 0

    @property
    def vision(self) -> VisionSheet:
        return self.sensors[self.first]

    # ------------------------------------------------------------ 跟随者
    def _attach(self, region: str, nid: int) -> int:
        f = len(self.f_src)
        self.f_src.append((region, nid))
        self.f_alive.append(True)
        self.f_of[(region, nid)] = f
        self.pfc.grow(1)
        self.hippo.grow(1)
        rng = np.random.default_rng(self.t * 7919 + f)
        row = rng.uniform(0, self.init_w, (1, self.R0.shape[1])).astype(np.float32)
        self.R0 = np.r_[self.R0, row]
        self.R = np.r_[self.R, row]
        self._index = None
        return f

    def _detach(self, region: str, nid: int) -> None:
        f = self.f_of.pop((region, nid), None)
        if f is None:
            return
        self.f_alive[f] = False
        self.pfc.remove(f)
        self.hippo.remove(f)
        self.R0[f] = self.R[f] = 0.0
        self._index = None

    def _ext_index(self):
        if self._index is None:
            idx = {}
            for f, (region, nid) in enumerate(self.f_src):
                if self.f_alive[f]:
                    idx.setdefault(region, ([], []))
                    idx[region][0].append(f)
                    idx[region][1].append(nid)
            self._index = {k: (np.array(a), np.array(b)) for k, (a, b) in idx.items()}
        return self._index

    def apply_events(self, region: str, info: dict) -> None:
        sheet = self.sensors[region]
        for parent, kids in info.get("split", []):
            for k in kids:
                if sheet.layer[k] >= self.chain_from_layer:
                    self._attach(region, k)
            for rg, srcs in self.reflex.values():
                if rg == region and parent in srcs:
                    srcs.discard(parent)
                    srcs.update(kids)
            self._detach(region, parent)
        for cid in info.get("added", []):
            if sheet.layer[cid] >= self.chain_from_layer:
                self._attach(region, cid)
        for cid in info.get("deleted", []):
            for rg, srcs in self.reflex.values():
                if rg == region:
                    srcs.discard(cid)
            self._detach(region, cid)

    def mute_featureless(self, on_thr: float | None = None) -> int:
        """经常亮、没特征的感觉神经元：不给它链式神经元（它的前额叶 / 海马体单元撤掉）。"""
        on_thr = self.mute_thr if on_thr is None else on_thr
        n = 0
        for name, sheet in self.sensors.items():
            u = np.zeros(len(sheet.inputs))
            k = min(len(u), len(self.usage[name]))
            u[:k] = self.usage[name][:k] / max(1, self.n_seen)       # 刚分裂出的新神经元还没有使用记录
            for cid in np.nonzero((u >= on_thr) & sheet._alive)[0]:
                if (name, int(cid)) in self.f_of:
                    self._detach(name, int(cid))
                    n += 1
        return n

    def restructure(self) -> dict:
        """慢：给每个感觉区做一次结构调整（常亮的分裂、从不亮的删除），跟随者一起增删；常亮没特征的不送链。"""
        infos = {}
        for name, sheet in self.sensors.items():
            info = sheet.restructure(self.usage[name][: len(sheet.inputs)] / max(1, self.n_seen))
            self.apply_events(name, info)
            infos[name] = info
        self.mute_featureless()
        for name, sheet in self.sensors.items():
            pop = self.sense_pop[name]
            if len(sheet.inputs) > pop.n:
                pop.grow(len(sheet.inputs) - pop.n)
            for i in np.nonzero(pop.alive & (~sheet._alive | (sheet._layer < self.sense_from_layer)))[0]:
                pop.remove(int(i))
            self.usage[name] = np.zeros(len(sheet.inputs))
        self.n_seen = 0
        self.chain = deque([({k: np.r_[v[k], np.zeros(len(self.sensors[k].inputs) - len(v[k]), dtype=bool)]
                              for k in v}, m) for v, m in self.chain], maxlen=self.chain_delay)
        return infos if len(infos) > 1 else infos[self.first]

    def _sense_topdown(self) -> dict:
        """感觉神经元这一拍收到的、来自"下面一层"以外的输入（以门槛为单位）：
        本区内部的赫布连接（上一拍的状态）+ 海马体 / 前额叶经回程送回的（在心里听到 / 看到）。"""
        out = {}
        for name, sheet in self.sensors.items():
            pop = self.sense_pop[name]
            td = np.zeros(len(sheet.inputs), dtype=np.float32)
            td[: pop.n] = np.where(pop.alive, pop.currents(), 0.0)[: len(td)]
            out[name] = td
        if self.sense_ret_hippo <= 0 and self.sense_ret_pfc <= 0:
            return out
        r = self.sense_ret_hippo * self.hippo.a + self.sense_ret_pfc * self.pfc.a
        for region, (fs, nids) in self._ext_index().items():
            if region == "motor":
                continue
            out[region][nids] += r[fs]
        return out

    # ------------------------------------------------------------ 本能
    def wire_reflex(self, action: str, sources, gain: float = 1.5, region: str | None = None) -> None:
        self.reflex[action] = (region or self.first, set(int(i) for i in sources))
        self.reflex_gain[action] = gain

    # ------------------------------------------------------------ 走一拍
    @property
    def learning(self) -> bool:
        return self.pfc.cfg.learn

    @learning.setter
    def learning(self, on: bool) -> None:
        self.pfc.cfg.learn = self.hippo.cfg.learn = on
        for pop in self.sense_pop.values():
            pop.cfg.learn = on

    def step(self, inputs, reason: dict[str, float] | None = None) -> dict[str, int]:
        """inputs：{感觉区: 这一拍的输入图}（只有一个区时也可以直接给图）。
        reason：外部直接加到运动区的驱动（键前加 "!" = 驱动该组抑制神经元）。"""
        if not isinstance(inputs, dict):
            inputs = {self.first: inputs}
        acts = {}
        topdown = self._sense_topdown()
        for name, sheet in self.sensors.items():
            act = sheet.respond(sheet.encode(inputs[name]), topdown.get(name))
            pop = self.sense_pop[name]
            pop.learn_from(pop.a, (act[: pop.n] & pop.alive).astype(np.float32))   # 本区内部：先亮 → 后亮
            self.usage[name][: len(act)] += act
            acts[name] = act
        self.n_seen += 1
        # 去程：各区 + 运动神经元（上一拍的）→ 链 → 前额叶 / 海马体
        self.chain.append((acts, (self.motor.act >= 0.5).astype(np.float32)))
        s_arr, m_arr = self.chain[0]
        ext = np.zeros(len(self.f_src), dtype=np.float32)
        for region, (fs, nids) in self._ext_index().items():
            src = m_arr if region == "motor" else s_arr[region]
            ok = nids < len(src)
            ext[fs[ok]] = src[nids[ok]]
        h_prev = self.hippo.a.copy()
        self.hippo.step(ext * self.ext_gain)
        trace_before = self.pfc.trace.copy()
        a = self.pfc.step(ext * self.pfc_ext_gain + self.hippo_gain * h_prev)
        # 回程（维持）：对应单元 → 同一个运动神经元；发起：前额叶的状态经 R 点亮运动神经元
        self.ret.append((a[self.motor_f].copy(), (a @ self.R).astype(np.float32)))
        mirror, init = self.ret[0]
        on = self.motor.act > 0
        ret = np.where(self.motor.muscle_mask, mirror * on, mirror) + init
        drive: dict[str, float] = {}
        for name, (region, srcs) in self.reflex.items():
            if srcs:
                ids = np.fromiter(srcs, dtype=int)
                drive[name] = drive.get(name, 0.0) + self.reflex_gain[name] * float(acts[region][ids].mean())
        for name, v in (reason or {}).items():
            drive[name] = drive.get(name, 0.0) + v
        self.t += 1
        if self.t % self.decay_T == 0:                        # 整体降权：学来的部分往出生权重退
            self.pfc.global_decay(self.cortex_gamma)
            self.hippo.global_decay(self.hippo_gamma)
        prev = self.motor.act.copy()
        out = self.motor.step(drive, ret)
        if self.learning:                                     # 发起连接：先亮的前额叶单元 → 刚亮的运动神经元
            now = self.motor.act
            rising = np.nonzero((now >= 0.35) & (prev < 0.35))[0]
            pre = np.nonzero(trace_before > 0)[0]
            if len(rising) and len(pre):
                ix = np.ix_(pre, rising)
                sub = self.R[ix]
                sub += self.eta_init * trace_before[pre][:, None] / (1.0 + sub / 0.2)
                self.R[ix] = sub
                apply_budget(self.R, rising, self.init_budget)
            was = np.nonzero(a > 0)[0]
            idle = np.nonzero(now < 0.35)[0]
            if len(was) and len(idle):                        # 亮了却没带动：往出生权重退
                ix = np.ix_(was, idle)
                self.R[ix] -= self.fail_init * (self.R[ix] - self.R0[ix])
        return out

    # ------------------------------------------------------------ 看
    def n_followers(self) -> int:
        return int(sum(self.f_alive))
