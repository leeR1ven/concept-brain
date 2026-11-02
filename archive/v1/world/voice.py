# -*- coding: utf-8 -*-
"""声音世界（身体的物理，大脑不知道这些规则）。

- 每个字有一个固定的声音：F 个频段 × T 个时间片的响度格子（0..levels）。
  像语音一样，每个时间片只有少数几个频段（共振峰）响，其余几乎安静；不同的字响的频段不同。
- 说话人不同，声音略有差异：别人 = 原声；自己 = 原声 + 一张固定的小偏差（约 30% 的格子 ±1）。
  → 同一句话，自己说和别人说激活的神经元相似但不同。
- 嘴：n_muscles 块肌肉；每个字要特定 k 块肌肉同时到位才发得清楚。
  动的肌肉覆盖某个字的 k 块 → 清楚地发出那个字；覆盖 k−1 块 → 含糊地发出那个字（嘟囔，听得出是哪个字）；
  动得太多或凑不成 → 杂音 "~"；不动 → 安静。
"""
from __future__ import annotations

import numpy as np


class Voice:
    def __init__(self, chars, F: int = 8, T: int = 6, levels: int = 10,
                 n_muscles: int = 32, k: int = 4, peaks: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.chars = list(chars)
        self.F, self.T, self.levels = F, T, levels
        self.n_muscles, self.k = n_muscles, k
        self.sound = {}
        for c in self.chars:
            g = rng.integers(0, 2, size=(F, T))                       # 底噪
            for t in range(T):
                g[rng.choice(F, size=peaks, replace=False), t] = rng.integers(levels // 2, levels, size=peaks)
            self.sound[c] = g
        self.voice = {"other": np.zeros((F, T), dtype=int),
                      "self": rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15], size=(F, T))}
        self.need: dict[str, frozenset] = {}
        used = set()
        for c in self.chars:
            while True:
                s = frozenset(int(m) for m in rng.choice(n_muscles, k, replace=False))
                if s not in used:
                    used.add(s)
                    self.need[c] = s
                    break
        self._noise_rng = np.random.default_rng(seed + 1)
        self.silence = np.zeros((F, T), dtype=int)

    def hear(self, c: str | None, who: str) -> np.ndarray:
        """c 这个字被 who 说出来时耳朵收到的格子；None = 安静；"~" = 含糊的杂音。"""
        if c is None:
            return self.silence.copy()
        if c == "~":
            return self._noise_rng.integers(0, 4, size=(self.F, self.T))
        return np.clip(self.sound[c] + self.voice[who], 0, self.levels)

    def mouth(self, force: dict[str, int]) -> str | None:
        """嘴的物理：哪几块肌肉在用力 → 发出什么。"""
        active = frozenset(int(name[1:]) for name, f in force.items() if f > 0)
        if not active:
            return None
        if len(active) <= self.k + 2:
            best, hit = None, 0
            for c in self.chars:
                h = len(self.need[c] & active)
                if h > hit:
                    best, hit = c, h
                    if h == self.k:
                        break
            if hit >= self.k - 1:
                return best
        return "~"


class Stream:
    """耳朵收到的声音是连续流动的：每拍推进 step 个时间片，耳朵看到最近 T 个时间片（一个字那么长）。
    一个字的 T 个时间片在 T/step 拍里放完；前一个字的尾巴和后一个字的开头在时间窗里连在一起。
    别人的声音和自己的声音叠在一起（取大）。"""

    def __init__(self, voice: Voice, step: int):
        self.v, self.step = voice, step
        self.win = np.zeros((voice.F, voice.T), dtype=int)
        self.src = {"other": [None, 0], "self": [None, 0]}       # 每个声源：正在放的字、放到第几片

    def tick(self, other: str | None, self_: str | None, new_other: bool = False) -> np.ndarray:
        """other / self_：这一拍别人 / 自己在发的字（None = 不出声）；new_other：别人这一拍开始说一个新字。"""
        v, T = self.v, self.v.T
        new = np.zeros((v.F, self.step), dtype=int)
        for who, ch, fresh in (("other", other, new_other), ("self", self_, False)):
            prev, ph = self.src[who]
            if ch != prev or fresh:
                ph = 0
            if ch is not None:
                g = v.hear(ch, who)
                cols = [min(ph + j, T - 1) for j in range(self.step)]   # 拖得比本来长：停在最后一片
                new = np.maximum(new, g[:, cols])
            self.src[who] = [ch, ph + self.step]
        self.win = np.concatenate([self.win[:, self.step:], new], axis=1)
        return self.win
