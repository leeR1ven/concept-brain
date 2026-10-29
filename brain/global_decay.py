# -*- coding: utf-8 -*-
"""整体降权：全脑共用的清理机制。

每隔 T 拍，一组可塑连接整体乘 γ；低于 prune 的删掉。先天连接不登记、不动。
常用的边一直在涨，能抵消降权；少用的变弱；从不用的自然消失。
不同部位用不同的 γ：海马体 γ 小（降得快），皮层 γ 接近 1（降得慢）。

登记方式：register(名字, 取边函数, γ)。取边函数返回一组 dict（每个 dict 是一行：目标 → 权重），
降权时就地修改。
"""
from __future__ import annotations


class GlobalDecay:
    def __init__(self, T: int = 200, prune: float = 1e-3):
        self.T = T
        self.prune = prune
        self.groups: dict[str, tuple] = {}
        self.t = 0

    def register(self, name: str, get_rows, gamma: float) -> None:
        self.groups[name] = (get_rows, gamma)

    def tick(self) -> None:
        self.t += 1
        if self.t % self.T:
            return
        for get_rows, gamma in self.groups.values():
            if gamma >= 1.0:
                continue
            for row in get_rows():
                if not row:
                    continue
                dead = []
                for k in row:
                    row[k] *= gamma
                    if row[k] < self.prune:
                        dead.append(k)
                for k in dead:
                    del row[k]

    @staticmethod
    def gamma_for_half_life(half_life_ticks: float, T: int = 200) -> float:
        return 0.5 ** (T / half_life_ticks)
