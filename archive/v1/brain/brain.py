# -*- coding: utf-8 -*-
"""完整模型：听觉区、运动区（运动皮层 + 肌肉神经元）、海马体、前额叶，以及链、回程、区间直连
（同一拍里全部用上一拍的状态）。

  听觉区：立体局部网络把声音压成特征；皮层有自己的可塑连接（顺着放出连续的声音）。
  运动区：多层网络。运动皮层（输入层）表示"肌肉的组合"——同一个字的那组肌肉对应一组特定的神经元，
          用同一种立体网络从肌肉的状态压出来；别的区都连到运动皮层。运动皮层自己的连接学"哪个组合后面接哪个组合"，
          运动皮层 → 肌肉神经元的连接学"这个组合之后要动哪些肌肉"。
          肌肉神经元 vs 抑制神经元同一拍比较，平时抑制占上风；收到的是各路信号的总和。
  链：听觉区皮层、运动皮层、肌肉神经元的每个神经元，各一根线到海马体的一个神经元、一根到前额叶的一个神经元。
  前额叶分两层，外加一个和它来回连的丘脑小区：
      接收层：上面那些"对应者"（收链；内部互相连；集中的活动经回程中继送出去）；
      思考层：前额叶自己的神经元，没有链进来、也没有回程出去。从接收层收信号，层内互相连，
              再送回接收层（输出又返回成为上一层的输入：出生就有一条沿原路回到来源的固定连接，另有学来的）
              ——接收层的名额由想法有关的来源填，而不是随便凑；慢慢学成概括 / 概念；在这里想事情不推动别的区；
      丘脑：和思考层一小组一小组地来回连（出生固定），给想法一个回路托着，能撑过一段间隔。
      海马体的链强（有真实输入时以输入为准，安静时空出的名额是回放）；前额叶的链弱（进不进得来看它在想什么）。
  回程：海马体 / 前额叶 → 回程中继 → 听觉区（心里听到）、运动皮层（想动）。强度是可调的数值。
        只让集中的想法回送：每个中继读"它负责的那个神经元 + 同一位置上下相邻几个"在前额叶里对应的神经元，
        亮够几个（想法集中在这一小簇）才亮；零散的、随便亮的不回送。焦点一换，原来那一簇散了，回程就关了。
  区间直连：听觉区 ↔ 运动皮层，出生随机、很弱，一起亮得多就变强。
  只能经肌肉输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .areas import (Circuit, MotorArea, MotorCfg, SenseArea, SenseCfg, ThoughtArea, ThoughtCfg, Unit, hip_cfg,
                    pfc_cfg)
from .net import N, Path, PathCfg, Region, RegionCfg, T
from .net3d import Net3D, Net3DCfg


@dataclass
class BrainCfg:
    net3d: Net3DCfg = field(default_factory=Net3DCfg)
    # 皮层学得慢、忘得也慢（整体降权接近 1）；海马体学得快、忘得快
    sense: SenseCfg = field(default_factory=lambda: SenseCfg(internal=PathCfg(k=96, w_init=0.01, eta=0.03, gamma=0.999, budget=1.0)))
    # 运动皮层：肌肉状态（每块肌肉的力度）排成 8×4 的图，用同一种立体网络压成"组合特征"
    mctx_net: Net3DCfg = field(default_factory=lambda: Net3DCfg(levels=3, zs=(20, 40, 80), frac=(0.4, 0.3, 0.2)))
    mctx: SenseCfg = field(default_factory=lambda: SenseCfg(internal=PathCfg(k=96, w_init=0.01, eta=0.03, gamma=0.999, budget=1.0)))
    hip: ThoughtCfg = field(default_factory=hip_cfg)
    pfc: ThoughtCfg = field(default_factory=lambda: ThoughtCfg(region=RegionCfg(frac=0.08, sp_beta=0.05),
                                                               internal=pfc_cfg().internal))
    motor: MotorCfg = field(default_factory=lambda: MotorCfg(dis_thr=2.0, internal=PathCfg(k=64, w_init=0.02, eta=0.01, gamma=0.999, budget=1.0)))
    mctx_mot: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.002, eta=0.003, gamma=0.999, budget=1.5))
    chain_hip: float = 3.0
    chain_pfc: float = 0.3
    ret_hip: float = 0.5          # 回程强度：可调旋钮
    ret_pfc: float = 0.5
    relay_span: int = 3           # 回程中继读同一位置上下各几个邻居
    relay_need: float = 3.0       # 这一小簇里至少亮几个，中继才亮
    pfc_own: int = 10000          # 前额叶思考层（自己的神经元，不对应任何区，没有链也没有回程）
    think: ThoughtCfg = field(default_factory=lambda: ThoughtCfg(region=RegionCfg(frac=0.05, sp_beta=0.05),
                                                                 internal=PathCfg(k=96, w_init=0.05, eta=0.02, gamma=0.999, budget=0.5)))
    in_think: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.03, eta=0.01, gamma=0.999, budget=1.0))   # 接收层 → 思考层
    think_in: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.01, eta=0.01, gamma=0.999, budget=0.3))   # 思考层 → 接收层
    w_fb: float = 0.1             # 想法回投：接收层 i → 思考层 j 的每条出生连接，都有一条固定的 j → i
    thal_group: int = 5           # 丘脑每个神经元和思考层几个神经元来回连
    thal_w: float = 0.3           # 丘脑 → 思考层的强度（托住想法的旋钮：太强会锁死，太弱撑不住）
    thal_need: float = 2.0        # 丘脑神经元：它那组思考层神经元里至少亮几个才亮；亮了就一直亮（托住），直到有人说话
    fresh_thal: float = 10.0      # 有人在说话（新鲜输入小群亮）时压住丘脑（压得过整组亮 + 自己撑着）：听的时候思考层跟着输入走，停下来才托住
    ear_mctx: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.005, eta=0.01, gamma=0.999, budget=1.0))
    mctx_ear: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.005, eta=0.01, gamma=0.999, budget=0.5))
    fresh_w: float = 0.001        # "新鲜输入"小群：听觉皮层每个神经元（乘余量）给它的电流
    fresh_thr: float = 0.4        #   亮的门槛（≈ 有多少新鲜的听觉特征算"有人在说话"）
    ret_motor: float = 0.3        # 前额叶 / 海马体里肌肉神经元的对应者 → 回到肌肉神经元（不回到去抑制神经元）
    fresh_go: float = 5.0         #   它压住起动（别人在说时不开口）
    to_go: PathCfg = field(default_factory=lambda: PathCfg(k=1, w_init=0.0, eta=0.0001, gamma=0.999, budget=2.0, back=False))
    # ↑ 思考层 / 听觉 → 起动（学）：托住的想法也能学会"该我说了"；两路上限加起来 4 < 起动门槛 + 刹车，别人在说时永远压得住
    think_mot: PathCfg = field(default_factory=lambda: PathCfg(k=32, w_init=0.0, eta=0.00001, gamma=0.999, budget=1.0, back=False))
    # ↑ 思考层 → 去抑制神经元（学）：想法决定先放开哪几块肌肉（第一个字）。学得慢 = 记的是很多段对话的平均
    ear_mot: PathCfg = field(default_factory=lambda: PathCfg(k=8, w_init=0.0, eta=0.0001, gamma=0.999, budget=1.0))
    # ↑ 听觉区 → 去抑制神经元（学）：听到自己刚说的音节 → 下一个放开哪几块肌肉（听觉区的码是一个音节一个样；
    #   运动皮层看的是每块肌肉松 / 紧，各音节几乎一样，学不出"这个音节后面是那个"）
    std_U: float = 0.5            # 输出余量（听觉区、运动皮层、肌肉区）：每亮一拍掉一半，
    std_tau: float = 6.0          #   约 6 拍恢复——一直不变的输入推力很快变小
    decay_T: int = 100
    seed: int = 0


class Relay:
    """回程中继：一群神经元，每个对应目标区的一个皮层神经元；收到的 ≥ need 才亮（门槛，不按名额）。"""

    def __init__(self, name: str, area: SenseArea, cand, need: float):
        self.name, self.area = name, area
        self.region = Region(name, area.region.n, RegionCfg(frac=0.0, thr=need))
        self.region.alive = T(area.cortex)
        self.regions, self.paths = [self.region], []

    def fire(self, cur: dict, lines: dict) -> None:
        self.region.fire(cur[self.name])


def neighbours(net: Net3D, cand, span: int) -> dict:
    """每个皮层神经元：它自己 + 同一位置、同一层上下各 span 个（立体网络里同一位置叠着的那一列）。
    返回 {神经元: [以它为邻居的中继们]}——前额叶里它对应的神经元，送给这些中继。"""
    owner = {int(i): [] for i in cand}
    for l, (s0, cap) in enumerate(zip(net.start, net.cap)):
        Z = net.dims[l + 1][2]
        n_born = net.dims[l + 1][0] * net.dims[l + 1][1] * Z
        for a in range(n_born):
            i = s0 + a
            if i not in owner:
                continue
            col0 = a - a % Z
            for b in range(max(col0, a - span), min(col0 + Z, a + span + 1)):
                if s0 + b in owner:
                    owner[s0 + b].append(i)          # 神经元 b 的前额叶对应者 → 中继 i
    for i in cand:                                   # 预留的空位（分裂出来的）：只有自己
        if not owner[int(i)]:
            owner[int(i)] = [int(i)]
    return owner


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
        self.pfc = ThoughtArea("pfc", ne + nc + nm, c.pfc, seed=c.seed + 2, ids=ids)          # 接收层
        self.think = ThoughtArea("think", c.pfc_own, c.think, seed=c.seed + 3)              # 思考层
        n_th = c.pfc_own // c.thal_group
        self.thal = Unit("thal", n_th, c.thal_need)                                     # 丘脑：托住想法的闸门
        src = {"ear": ear.region, "mctx": mctx.region, "motor": mot.region}
        for R in src.values():
            R.cfg.std_U, R.cfg.std_tau = c.std_U, c.std_tau
        paths, relays = [], []
        for X, cg, rg in ((self.hip, c.chain_hip, c.ret_hip), (self.pfc, c.chain_pfc, c.ret_pfc)):
            for k, R in src.items():                                   # 链：每个神经元一根线到 T 里对应的那一个
                o = self.off[k]                                        # （海马体照实记录：不乘余量）
                paths.append(Path(f"{k}⇒{X.name}", R, X.region, PathCfg(w_init=cg, use_x=X is self.pfc), rng,
                                  pairs=[(int(i), [o + int(i)]) for i in self.cand[k]]))
            for k, A in (("ear", ear), ("mctx", mctx)):                # 回程：经中继，只有集中的想法回送
                relay = Relay(f"{X.name}→{k}中继", A, self.cand[k], c.relay_need)
                relays.append(relay)
                o = self.off[k]
                nb = neighbours(A.net, self.cand[k], c.relay_span)
                paths.append(Path(f"{X.name}⇒{k}中继", X.region, relay.region, PathCfg(w_init=1.0), rng,
                                  pairs=[(o + int(i), nb[int(i)]) for i in self.cand[k]]))
                paths.append(Path(f"{k}中继⇒{k}", relay.region, A.region, PathCfg(w_init=rg), rng,
                                  pairs=[(int(i), [int(i)]) for i in self.cand[k]]))
        grp = np.arange(c.pfc_own) % n_th                          # 思考层第 i 个 ↔ 丘脑第 i % n_th 个
        p_in = Path("pfc→think", self.pfc.region, self.think.region, c.in_think, rng, src_ids=ids,
                    dst_ids=np.arange(c.pfc_own))
        to, ok = N(p_in.to), N(p_in.ok)
        r, k = np.nonzero(ok)
        j = to[r, k]
        order = np.argsort(j, kind="stable")
        cut = np.searchsorted(j[order], np.arange(c.pfc_own + 1))
        paths += [p_in,                                            # 想法回投：沿出生连接反着送回它的来源
                  Path("think⇒pfc", self.think.region, self.pfc.region, PathCfg(w_init=c.w_fb), rng,
                       pairs=[(q, r[order[cut[q]:cut[q + 1]]]) for q in range(c.pfc_own)]),
                  Path("think→pfc", self.think.region, self.pfc.region, c.think_in, rng, src_ids=np.arange(c.pfc_own),
                       dst_ids=ids),
                  Path("think⇒thal", self.think.region, self.thal.region, PathCfg(w_init=1.0), rng,
                       pairs=[(i, [int(grp[i])]) for i in range(c.pfc_own)]),
                  Path("thal⇒think", self.thal.region, self.think.region, PathCfg(w_init=c.thal_w), rng,
                       pairs=[(j, np.nonzero(grp == j)[0]) for j in range(n_th)]),
                  Path("thal⇒thal", self.thal.region, self.thal.region, PathCfg(w_init=c.thal_need), rng,   # 亮了就自己撑着
                       pairs=[(j, [j]) for j in range(n_th)])]
        om = self.off["motor"]                                     # 前额叶 / 海马体里肌肉区神经元的对应者 → 回去（一对一）
        paths += [Path(f"{X.name}⇒motor", X.region, mot.region, PathCfg(w_init=c.ret_motor), rng,     # 只回到肌肉神经元：
                       pairs=[(om + int(i), [int(i)]) for i in mot.M.reshape(-1)]) for X in (self.hip, self.pfc)]   # 回到 D 会一直重复刚说的字
        self.fresh = Unit("fresh", 1, c.fresh_thr)                 # 新鲜输入小群：有人在说话时亮 → 压住起动和丘脑
        go = mot.go
        paths += [Path("ear⇒fresh", ear.region, self.fresh.region, PathCfg(w_init=c.fresh_w), rng,
                       pairs=[(int(i), [0]) for i in cand_e]),
                  Path("fresh⇒go", self.fresh.region, go, PathCfg(w_init=c.fresh_go, sign=-1.0), rng, pairs=[(0, [0])]),
                  Path("fresh⇒thal", self.fresh.region, self.thal.region, PathCfg(w_init=c.fresh_thal, sign=-1.0), rng,
                       pairs=[(0, np.arange(n_th))]),
                  Path("think→go", self.think.region, go, c.to_go, rng, src_ids=np.arange(c.pfc_own), dst_ids=[0]),
                  Path("ear→go", ear.region, go, c.to_go, rng, src_ids=cand_e, dst_ids=[0]),
                  Path("think→motor", self.think.region, mot.region, c.think_mot, rng, src_ids=np.arange(c.pfc_own),
                       dst_ids=mot.D),
                  Path("ear→motor", ear.region, mot.region, c.ear_mot, rng, src_ids=cand_e, dst_ids=mot.D)]
        paths += [Path("mctx→motor", mctx.region, mot.region, c.mctx_mot, rng, src_ids=cand_c, dst_ids=np.arange(nm)),
                  Path("ear→mctx", ear.region, mctx.region, c.ear_mctx, rng, src_ids=cand_e, dst_ids=cand_c),
                  Path("mctx→ear", mctx.region, ear.region, c.mctx_ear, rng, src_ids=cand_c, dst_ids=cand_e)]
        self.relays = relays
        self.c = Circuit([ear, mctx, self.hip, self.pfc, self.think, self.thal, self.fresh, mot] + relays, paths,
                         decay_T=c.decay_T)
        self._sync_mirrors()

    def _sync_mirrors(self) -> None:
        """前额叶 / 海马体里对应的神经元，跟着听觉区 / 运动皮层神经元亮不亮的开关走（没特征的、删掉的不送链）。"""
        for X in (self.hip, self.pfc):
            for k, A in (("ear", self.ear), ("mctx", self.mctx)):
                o = self.off[k]
                X.region.alive[o:o + A.region.n] = T(A.cortex)
            X.region.alive[self.off["motor"]:self.off["motor"] + self.mot.region.n] = True
            X.region._alive_ref = None
        for r in self.relays:
            r.region.alive = T(r.area.cortex)

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

    def step(self, lines: np.ndarray, drive: np.ndarray | None = None, go: bool = False) -> np.ndarray:
        """lines：耳朵输入线；drive：外部送进运动区的电流（外力 / 本能）；go：外力推一下起动（"轮到你了"）。
        返回每块肌肉的力度。"""
        ext = {} if drive is None else {"motor": drive}
        if go:
            ext[self.mot.go.name] = np.array([self.mot.cfg.go_thr + self.cfg.fresh_go + 1.0], np.float32)   # 外力压过刹车（和 cue 一样）
        self.c.step(ext=ext, lines={"ear": lines, "mctx": self.muscle_lines()})
        return self.mot.force()

    def mute_featureless(self) -> int:
        """没特征的神经元（几乎一直亮）不往任何地方送信号，也不收学来的连接。"""
        n = 0
        for A in (self.ear, self.mctx):
            ids = A.featureless()
            A.mute(ids)
            for p in self.c.paths:
                if p in A.paths or not len(ids):
                    continue
                if p.src is A.region:
                    p.cut_src(ids)
                if p.dst is A.region:
                    p.cut_dst(ids)
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
