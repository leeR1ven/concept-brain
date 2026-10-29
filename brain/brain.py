# -*- coding: utf-8 -*-
"""整脑：各感觉区 → 链 → 前额叶（思考 + 海马体）→ 回程 → 运动区 → 肌肉（唯一输出）。

  感觉区（任意多个，brain/vision.py 的多层感觉片）──┐
  运动区的每个神经元（控制肌肉 / 抑制 / 去抑制）──────┤ 每个神经元一根链式专用线（1:1，延迟 chain_delay 拍）
                                                    ├──► 前额叶接收端（brain/pfc.py：思考 + 皮层慢权重 + 海马体快权重）
  本能：感觉区先天直连运动区（不走链）──────────────┼──► 运动区（brain/motor.py）
                                                    │        ▲
  前额叶里运动神经元的对应单元 ── 回程（链，1:1）────┘────────┘   维持：只作用在正在动的神经元上
  前额叶 ── 发起（先天空连接 + 先亮 → 后亮学来的）──────────────┘   发起：想做什么就点亮哪些运动神经元

  - 前额叶在想别的事、没激活运动对应单元时：运动完全由其他区（本能）决定。
  - 前额叶想动 / 想压住某些肌肉：靠它自己内部的相互激活 / 竞争把对应单元点亮。
  - 前额叶 ↔ 各区之间的回路是正常现象。
  - 运动的"维持"和"发起"是两条路（born-wired 的概念）：回程 1:1 只扶住正在做的动作；
    发起靠前额叶 → 运动神经元的可塑连接（先天铺好权重极小的空连接，学习只写在这些路线上）。
  - 跟随增删：感觉区神经元分裂 / 删除时，它的链、前额叶接收端、海马体单元一起增删；
    本能直连的来源身份由子单元继承。跟随者 f ↔ (区, 神经元编号)。
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .global_decay import GlobalDecay
from .motor import Motor
from .pfc import PFC, PFCCfg, apply_budget
from .vision import VisionSheet


class Brain:
    def __init__(self, sensors, motor: Motor, chain_delay: int = 2,
                 pfc_cfg: PFCCfg | None = None,
                 init_routes: int = 100, init_empty_w: float = 1e-4, init_budget: float = 0.5,
                 hippo_gamma: float = 0.5, cortex_gamma: float = 1.0, decay_T: int = 200):
        if isinstance(sensors, VisionSheet):
            sensors = {"vision": sensors}
        self.sensors: dict[str, VisionSheet] = dict(sensors)
        self.first = next(iter(self.sensors))
        self.motor = motor
        self.chain_delay = chain_delay
        self.f_src: list[tuple[str, int]] = []
        self.f_alive: list[bool] = []
        self.f_of: dict[tuple[str, int], int] = {}
        # 一次性登记全部跟随者（逐个登记要反复重新分配前额叶的大矩阵，规模一大就很慢）
        srcs = [(name, int(cid)) for name, sheet in self.sensors.items() for cid in np.nonzero(sheet._alive)[0]]
        srcs += [("motor", mid) for mid in range(motor.n_neurons)]
        self.pfc = PFC(len(srcs), pfc_cfg)
        for f, key in enumerate(srcs):
            self.f_src.append(key)
            self.f_alive.append(True)
            self.f_of[key] = f
        self._index = None
        self.motor_f = np.array([self.f_of[("motor", m)] for m in range(motor.n_neurons)])
        # 发起：前额叶 → 运动神经元。先天空连接 R0（定路线），学来的 Rc（慢）/ Rh（快）只写在路线上
        rng = np.random.default_rng(self.pfc.cfg.seed + 7)
        n_p, n_m = self.pfc.n, motor.n_neurons
        self.R0 = np.zeros((n_p, n_m), dtype=np.float32)
        for mid in range(n_m):
            self.R0[rng.choice(n_p, min(init_routes, n_p), replace=False), mid] = init_empty_w
        self.Rc = np.zeros_like(self.R0)
        self.Rh = np.zeros_like(self.R0)
        self.init_budget = init_budget          # 每个运动神经元学来的发起输入总量上限（可调数值）
        self.gd = GlobalDecay(T=decay_T)
        cortex_m, hippo_m = self.pfc.decay_targets()
        self.gd.register("hippocampus", lambda: hippo_m() + [self.Rh], hippo_gamma)
        self.gd.register("cortex", lambda: cortex_m() + [self.Rc], cortex_gamma)
        z = ({k: np.zeros(len(s.inputs), dtype=bool) for k, s in self.sensors.items()},
             np.zeros(motor.n_neurons, dtype=np.float32))
        self.chain = deque([z] * chain_delay, maxlen=chain_delay)          # 去程
        zm = np.zeros(motor.n_neurons, dtype=np.float32)
        self.ret = deque([(zm, zm)] * chain_delay, maxlen=chain_delay)      # 回程：(维持, 发起)
        self.reflex: dict[str, tuple[str, set[int]]] = {}   # 动作 → (感觉区, 先天直连它的神经元)
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
        self._index = None
        for name in ("R0", "Rc", "Rh"):              # 新的前额叶单元暂时没有发起路线
            if hasattr(self, name):
                M = getattr(self, name)
                setattr(self, name, np.r_[M, np.zeros((1, M.shape[1]), dtype=np.float32)])
        return f

    def _detach(self, region: str, nid: int) -> None:
        f = self.f_of.pop((region, nid), None)
        if f is None:
            return
        self.f_alive[f] = False
        self.pfc.remove(f)
        self._index = None
        for name in ("R0", "Rc", "Rh"):
            if hasattr(self, name):
                getattr(self, name)[f, :] = 0.0

    def _ext_index(self):
        """每个区：(跟随者编号, 该区神经元编号)，用来把链送来的活动摊到前额叶接收端。"""
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
        for parent, kids in info.get("split", []):
            for k in kids:
                self._attach(region, k)
            for rg, srcs in self.reflex.values():          # 本能来源身份由子单元继承
                if rg == region and parent in srcs:
                    srcs.discard(parent)
                    srcs.update(kids)
            self._detach(region, parent)
        for cid in info.get("added", []):
            self._attach(region, cid)
        for cid in info.get("deleted", []):
            for rg, srcs in self.reflex.values():
                if rg == region:
                    srcs.discard(cid)
            self._detach(region, cid)

    def restructure(self) -> dict:
        """慢：用这段时间的使用率给每个感觉区做一次结构调整，跟随者一起增删。"""
        infos = {}
        for name, sheet in self.sensors.items():
            info = sheet.restructure(self.usage[name][: len(sheet.inputs)] / max(1, self.n_seen))
            self.apply_events(name, info)
            infos[name] = info
            self.usage[name] = np.zeros(len(sheet.inputs))
        self.n_seen = 0
        self.chain = deque([({k: np.r_[v[k], np.zeros(len(self.sensors[k].inputs) - len(v[k]), dtype=bool)]
                              for k in v}, m) for v, m in self.chain], maxlen=self.chain_delay)
        return infos if len(infos) > 1 else infos[self.first]

    # ------------------------------------------------------------ 本能
    def wire_reflex(self, action: str, sources, gain: float = 1.5, region: str | None = None) -> None:
        """先天直连：这批感觉神经元一起亮 → 驱动这个动作（驱动 = gain × 亮的比例）。"""
        self.reflex[action] = (region or self.first, set(int(i) for i in sources))
        self.reflex_gain[action] = gain

    # ------------------------------------------------------------ 走一拍
    def step(self, inputs, reason: dict[str, float] | None = None) -> dict[str, int]:
        """inputs：{感觉区: 这一拍的输入图}（只有一个区时也可以直接给图）。
        reason：外部直接加到运动区的驱动（测试 / 身体自发的动作，例如咿呀学语时随机动肌肉）。"""
        if not isinstance(inputs, dict):
            inputs = {self.first: inputs}
        acts = {}
        for name, sheet in self.sensors.items():
            act = sheet.respond(sheet.encode(inputs[name]))
            self.usage[name][: len(act)] += act
            acts[name] = act
        self.n_seen += 1
        # 去程：各区 + 运动神经元（上一拍的）→ 链 → 前额叶
        self.chain.append((acts, self.motor.act.copy()))
        s_arr, m_arr = self.chain[0]
        ext = np.zeros(len(self.f_src), dtype=np.float32)
        for region, (fs, nids) in self._ext_index().items():
            src = m_arr if region == "motor" else s_arr[region]
            ok = nids < len(src)
            ext[fs[ok]] = src[nids[ok]]
        trace_before = self.pfc.trace.copy()
        a = self.pfc.step(ext)
        # 回程（维持）：对应单元 → 链 → 同一个运动神经元；发起：前额叶 → 运动神经元的可塑连接
        mirror = a[self.motor_f].copy()
        init = (a @ (self.R0 + self.Rc + self.pfc.cfg.hippo_gain * self.Rh)).astype(np.float32)
        self.ret.append((mirror, init))
        mirror, init = self.ret[0]
        on = self.motor.act > 0
        mus = self.motor.muscle_mask
        ret = np.where(mus, mirror * on, mirror) + init      # 肌肉神经元：回程只扶住正在动的
        # 运动区：本能（不走链）+ 回程 + 外部驱动
        drive: dict[str, float] = {}
        for name, (region, srcs) in self.reflex.items():
            if srcs:
                ids = np.fromiter(srcs, dtype=int)
                drive[name] = drive.get(name, 0.0) + self.reflex_gain[name] * float(acts[region][ids].mean())
        for name, v in (reason or {}).items():
            drive[name] = drive.get(name, 0.0) + v
        self.gd.tick()
        self.t += 1
        prev = self.motor.act.copy()
        out = self.motor.step(drive, ret)
        if self.pfc.cfg.learn:                               # 发起路线：亮了却没带动运动神经元的 → 变弱
            was_on = np.nonzero(self.pfc.a >= self.pfc.cfg.on_thr)[0]
            idle = np.nonzero(self.motor.act < 0.35)[0]
            if len(was_on) and len(idle):
                ix = np.ix_(was_on, idle)
                self.Rc[ix] *= 1.0 - self.pfc.cfg.fail_cortex
                self.Rh[ix] *= 1.0 - self.pfc.cfg.fail_hippo
        if self.pfc.cfg.learn:                               # 学发起：先亮的前额叶单元 → 刚亮的运动神经元
            rising = np.nonzero((self.motor.act >= 0.35) & (prev < 0.35))[0]
            pre = np.nonzero(trace_before > 0)[0]
            if len(rising) and len(pre):
                ix = np.ix_(pre, rising)
                route = self.R0[ix] > 0
                for M, eta in ((self.Rc, self.pfc.cfg.eta_cortex), (self.Rh, self.pfc.cfg.eta_hippo)):
                    sub = M[ix]
                    sub += route * (eta * trace_before[pre][:, None] / (1.0 + sub))
                    M[ix] = sub
                    apply_budget(M, rising, self.init_budget)
        return out

    # ------------------------------------------------------------ 看
    def n_followers(self) -> int:
        return int(sum(self.f_alive))
