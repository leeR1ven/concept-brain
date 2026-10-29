# -*- coding: utf-8 -*-
"""会听、会想、会说的模型：对话跟读 → 和它说话。

  耳朵（听觉感觉区）→ 链 → 前额叶 → 回程 → 嘴部肌肉 → 说出的声音再被自己听到（自己的声音）。

学：对话里别人说单数句、它说双数句。轮到它时，用数据驱动它的发音肌肉一个字一个字说完，
    说完再驱动抑制神经元让嘴停下（用户：肌肉动作本来是连续的，说完一句就停——先激活肌肉神经元，再激活抑制神经元）。
    前额叶按"先亮 → 后亮"学到：听完对方之后说什么、每个字之后动哪些肌肉、一句说完就停。

用法：
  python talk.py                 # 学，然后用几句话测试
  python talk.py --epochs 3      # 语料多学几遍
语料：/root/work/corpus/chatterbot/*.yml（chatterbot 中文对话）
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain.brain import Brain  # noqa: E402
from brain.global_decay import GlobalDecay  # noqa: E402
from brain.motor import Motor, MotorCfg  # noqa: E402
from brain.pfc import PFCCfg  # noqa: E402
from brain.vision import SheetCfg, VisionSheet  # noqa: E402
from world.voice import Voice  # noqa: E402

CORPUS = "/root/work/corpus/chatterbot/*.yml"
SOUND_STEPS, GAP_STEPS, TURN_GAP = 2, 1, 4


def load_corpus(pattern=CORPUS):
    convs = []
    for f in sorted(glob.glob(pattern)):
        cur = None
        for line in open(f, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("- - "):
                if cur:
                    convs.append(cur)
                cur = [line[4:].strip()]
            elif line.startswith("  - ") and cur is not None:
                cur.append(line[4:].strip())
        if cur:
            convs.append(cur)
    clean = lambda s: re.sub(r"[^一-鿿A-Za-z0-9]", "", s)  # noqa: E731  标点 = 停顿，不是声音
    return [[clean(u) for u in c if clean(u)] for c in convs]


class Talker:
    def __init__(self, chars, seed=0):
        self.voice = Voice(chars, seed=seed)
        v = self.voice
        # 耳朵 3 层，越深越挑剔：第 2、3 层每个字只亮约 14% / 11%，不同字几乎不重叠；
        # 同一个字自己说 / 别人说在第 3 层重叠约 54%（相似但不同）
        ear = VisionSheet(v.F, v.T, SheetCfg(levels=v.levels, layers=(8, 8, 8), need_frac=(0.5, 0.875, 0.375),
                                             split_frac=(0.01, 0.02, 0.05)), seed=seed)
        mouth = Motor([f"m{i}" for i in range(v.n_muscles)], cfg=MotorCfg(levels=3))
        self.brain = Brain({"ear": ear}, mouth, pfc_cfg=PFCCfg(eta_hippo=0.3, seed=seed),
                           hippo_gamma=GlobalDecay.gamma_for_half_life(2000))
        self.rng = np.random.default_rng(seed)
        self.saying = None           # 嘴这一拍发出的声音（下一拍自己听到）
        self.said: list[str] = []

    def tick(self, other: str | None = None, reason=None) -> str | None:
        """一拍：耳朵听到 别人这一拍说的 + 自己上一拍说的（叠在一起取大）。返回嘴这一拍发出的。"""
        v = self.voice
        heard = np.maximum(v.hear(other, "other"), v.hear(self.saying, "self"))
        force = self.brain.step({"ear": heard}, reason)
        self.saying = v.mouth(force)
        if self.saying is not None:
            self.said.append(self.saying)
        return self.saying

    # ------------------------------------------------------------ 学
    def babble(self, n: int):
        """重复的咿呀学语：随机凑一组肌肉（这里直接用某个字的那组，等于身体碰巧发出一个音节），
        重复 3 遍 → 自己听到。重复让前额叶同时学到"肌肉 → 声音"和"声音 → 下一次的肌肉"。"""
        v = self.voice
        for _ in range(n):
            c = v.chars[self.rng.integers(len(v.chars))]
            drive = {f"m{i}": 2.0 for i in v.need[c]}
            for _ in range(3):
                for _ in range(SOUND_STEPS):
                    self.tick(reason=drive)
                for _ in range(GAP_STEPS):
                    self.tick()
            for _ in range(TURN_GAP):
                self.tick()

    def speak_forced(self, utterance: str):
        """用数据驱动嘴说一句（跟读），说完驱动抑制神经元停下。"""
        v = self.voice
        for ch in utterance:
            drive = {f"m{i}": 2.0 for i in v.need[ch]}
            for _ in range(SOUND_STEPS):
                self.tick(reason=drive)
            for _ in range(GAP_STEPS):
                self.tick()
        stop = {f"!m{i}": 3.0 for i in range(v.n_muscles)}
        for _ in range(2):
            self.tick(reason=stop)
        for _ in range(TURN_GAP):
            self.tick()

    def listen(self, utterance: str):
        for ch in utterance:
            for _ in range(SOUND_STEPS):
                self.tick(other=ch)
            for _ in range(GAP_STEPS):
                self.tick()
        for _ in range(TURN_GAP):
            self.tick()

    # ------------------------------------------------------------ 说
    def reply(self, prompt: str, steps: int = 40) -> str:
        self.listen(prompt)
        self.said = []
        for _ in range(steps):
            self.tick()
        out, last = [], None
        for s in self.said:                      # 同一个音拉长几拍只算一次
            if s != last:
                out.append(s)
            last = s
        return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--corpus", default=CORPUS, help="语料文件（glob），可用逗号分隔多个")
    ap.add_argument("--babble", type=int, default=0)
    ap.add_argument("--restructure_every", type=int, default=5000)
    args = ap.parse_args()
    convs = [c for pat in args.corpus.split(",") for c in load_corpus(pat)]
    chars = sorted({ch for c in convs for u in c for ch in u})
    print(f"语料：{len(convs)} 段对话，{sum(len(u) for c in convs for u in c)} 字，{len(chars)} 个不同的字")
    t = Talker(chars)
    b = t.brain
    print(f"脑：耳朵 {b.sensors['ear'].n_alive} 个神经元，运动区 {b.motor.n_neurons}，前额叶 {b.pfc.n} 个单元")

    if args.babble:
        t0 = time.perf_counter()
        t.babble(args.babble)
        print(f"咿呀学语 {args.babble} 次：{b.t} 拍，{time.perf_counter() - t0:.0f}s")
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        for conv in convs:
            for i, u in enumerate(conv):
                if i % 2 == 0:
                    t.listen(u)            # 别人说
                else:
                    t.speak_forced(u)      # 轮到它：跟读
                if b.t % args.restructure_every < 40:
                    b.restructure()
        print(f"听语料第 {ep + 1} 遍：累计 {b.t} 拍，{time.perf_counter() - t0:.0f}s，前额叶 {b.pfc.n} 个单元")

    b.pfc.cfg.learn = False
    for p in ["你好", "早上好你好吗", "你好吗", "你最喜欢什么颜色", "谢谢", "你是谁"]:
        print(f"  我：{p}\n  它：{t.reply(p) or '（不说话）'}")


if __name__ == "__main__":
    main()
