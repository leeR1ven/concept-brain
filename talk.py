# -*- coding: utf-8 -*-
"""会听、会想、会说的模型：对话训练 → 和它说话。

  耳朵 → 听觉皮层（自己的连接放出连续的声音）→ 链 → 前额叶 / 海马体 → 回程 → 运动区 → 嘴 → 自己的声音又被耳朵听到。

训练：对话里别人说一句（别人的声音），模型回一句。模型那一句训练时由外力带着嘴说出来
      （"手搭连接"的一种：外力推动该动的肌肉神经元和去抑制神经元），它听到的是自己的声音。
      全脑照常学：听觉区记住声音的顺序，运动区记住嘴的动作顺序，前额叶记住大致关系，区间直连记住"听到 → 说出"。
测试：别人说一句，然后安静，看它会不会开口、说了什么。

用法：
  python talk.py --convs 20 --epochs 10
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
from brain.brain import Brain, BrainCfg  # noqa: E402
from brain.vision import SheetCfg, VisionSheet  # noqa: E402
from world.voice import Voice  # noqa: E402

CORPUS = "/root/work/corpus/chatterbot/*.yml"
SOUND_STEPS, GAP_STEPS, TURN_GAP = 2, 0, 4   # 一句话里字连着说；句子之间停顿


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
    def __init__(self, chars, cfg: BrainCfg | None = None, per: int = 16, seed: int = 0):
        self.voice = v = Voice(chars, seed=seed)
        ear = VisionSheet(v.F, v.T, SheetCfg(levels=v.levels, layers=(per, per, per), need_frac=(0.5, 0.875, 0.375),
                                             split_frac=(0.01, 0.02, 0.05)), seed=seed)
        cfg = cfg or BrainCfg(seed=seed)
        cfg.n_muscles = v.n_muscles
        self.brain = Brain(ear, cfg)
        self.saying = None            # 嘴这一拍发出的声音（下一拍自己听到）
        self.said: list = []

    def tick(self, other: str | None = None, drive: np.ndarray | None = None):
        """一拍：耳朵听到 别人这一拍说的 + 自己上一拍说的（叠在一起取大）。返回嘴这一拍发出的。"""
        v, b = self.voice, self.brain
        heard = np.maximum(v.hear(other, "other"), v.hear(self.saying, "self"))
        force = b.step(b.sheet.encode(heard), drive)
        self.saying = v.mouth({f"m{i}": int(f) for i, f in enumerate(force)})
        self.said.append(self.saying)
        return self.saying

    def push(self, ch: str, strength: float = 3.0) -> np.ndarray:
        """外力带着嘴说 ch：推该动的肌肉神经元 + 它们的去抑制神经元。"""
        b = self.brain
        d = np.zeros(b.mot.n, np.float32)
        for m in self.voice.need[ch]:
            d[b.M[m]] += strength
            d[b.D[m]] += 1.5
        return d

    def listen(self, utt: str):
        for ch in utt:
            for _ in range(SOUND_STEPS):
                self.tick(other=ch)
            for _ in range(GAP_STEPS):
                self.tick()
        for _ in range(TURN_GAP):
            self.tick()

    def say(self, utt: str):
        """训练时模型的那一句：外力带着嘴说出来。"""
        for ch in utt:
            d = self.push(ch)
            for _ in range(SOUND_STEPS):
                self.tick(drive=d)
            for _ in range(GAP_STEPS):
                self.tick()
        for _ in range(TURN_GAP):
            self.tick()

    def warmup(self, n: int = 300, seed: int = 1):
        """出生后先随便听一阵（不学），找出经常亮、没特征的皮层神经元，让它们不送链。"""
        b = self.brain
        b.learning = False
        rng = np.random.default_rng(seed)
        for _ in range(n):
            self.listen(self.voice.chars[rng.integers(len(self.voice.chars))])
        muted = b.mute_featureless()
        b.usage[:] = 0
        b.n_seen = 0
        b.reset()
        b.learning = True
        return muted

    def reply(self, prompt: str, steps: int = 40) -> str:
        self.listen(prompt)
        self.said = []
        for _ in range(steps):
            self.tick()
        out, last = [], None
        for s in self.said:                        # 同一个音拉长几拍只算一次
            if s is not None and s != last:
                out.append(s)
            last = s
        return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=20, help="用前多少段对话")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--restructure_every", type=int, default=0)
    args = ap.parse_args()
    convs = load_corpus()[: args.convs]
    chars = sorted({ch for c in convs for u in c for ch in u})
    print(f"语料：{len(convs)} 段对话，{sum(len(u) for c in convs for u in c)} 字，{len(chars)} 个不同的字")
    t = Talker(chars)
    b = t.brain
    print(f"脑：耳朵皮层 {int(b.cortex.sum())}，前额叶 {b.pfc.n}，海马体 {b.hip.n}，运动区 {b.mot.n}")
    print(f"没特征的不送链：{t.warmup()} 个")
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        for conv in convs:
            for i, u in enumerate(conv):
                (t.listen if i % 2 == 0 else t.say)(u)
            if args.restructure_every and b.t % args.restructure_every < 60:
                b.restructure()
        print(f"第 {ep + 1} 遍：累计 {b.t} 拍，{time.perf_counter() - t0:.0f}s")
    b.learning = False
    for conv in convs[:8]:
        b.reset()
        print(f"  别人：{conv[0]}\n  它：{t.reply(conv[0]) or '（不说话）'}    （该回：{conv[1] if len(conv) > 1 else ''}）")


if __name__ == "__main__":
    main()
