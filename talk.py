# -*- coding: utf-8 -*-
"""完整模型：对话训练 → 和它说话。以模型经肌肉说出来的东西为准。

  耳朵 → 听觉区 → 链 → 海马体 / 前额叶 → 回程 → 运动区 → 嘴 → 自己的声音又被耳朵听到。

训练：对话里别人说一句（别人的声音），模型回一句。模型那一句由外力带着嘴说出来
      （推该动的肌肉神经元 + 去抑制神经元），说完推抑制神经元让嘴停下（用户：先激活肌肉神经元，再激活抑制神经元）；
      它听到的是自己的声音。全脑照常学。
测试：别人说一句，然后安静，看嘴说出什么。

  python talk.py --convs 20 --epochs 10
语料：/root/work/corpus/chatterbot/*.yml（chatterbot 中文对话）
"""
from __future__ import annotations

import argparse
import glob
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain.brain import Brain, BrainCfg  # noqa: E402
from world.voice import Stream, Voice  # noqa: E402

CORPUS = "/root/work/corpus/chatterbot/*.yml"
SOUND, PAUSE = 3, 4          # 每个字的声音持续几拍（一句话里字连着说）；对话和对话之间停几拍
REPLY_GAP = 1                # 别人说完到模型接话的停顿（真实对话里接话间隔很短）


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
    def __init__(self, chars, cfg: BrainCfg | None = None, seed: int = 0):
        self.voice = v = Voice(chars, seed=seed)
        self.brain = Brain(v.F, v.T, v.n_muscles, cfg or BrainCfg(seed=seed))
        self.stream = Stream(v, step=v.T // SOUND)            # 声音连续流动：每个字 SOUND 拍放完
        self.saying = None
        self.said: list = []

    def tick(self, other: str | None = None, drive: np.ndarray | None = None, new_other: bool = False):
        """一拍：耳朵听到 别人这一拍说的 + 自己上一拍说的（连续的声音流）。返回嘴这一拍发出的。"""
        v, b = self.voice, self.brain
        heard = self.stream.tick(other, self.saying, new_other)
        force = b.step(b.ear.net.encode(heard), drive)
        self.saying = v.mouth({f"m{i}": int(f) for i, f in enumerate(force)})
        self.said.append(self.saying)
        return self.saying

    def listen(self, utt: str, pause: int = PAUSE):
        for ch in utt:
            for k in range(SOUND):
                self.tick(other=ch, new_other=(k == 0))
        for _ in range(pause):
            self.tick()

    def say(self, utt: str):
        """模型的那一句：外力带着嘴说，说完推抑制神经元停下。"""
        m = self.brain.mot
        for ch in utt:
            for k in range(SOUND):                                    # 力逐拍加上去
                self.tick(drive=m.push(self.voice.need[ch], level=min(k + 1, m.cfg.levels)))
        for _ in range(2):
            self.tick(drive=m.stop())
        for _ in range(PAUSE):
            self.tick()

    def warmup(self, n: int = 60, seed: int = 1) -> int:
        """出生后先随便听一阵（不学），经常亮、没特征的不送链。"""
        b = self.brain
        b.learning = False
        rng = np.random.default_rng(seed)
        chars = self.voice.chars
        for _ in range(n):
            self.listen(chars[rng.integers(len(chars))])
        muted = b.mute_featureless()
        b.ear.clear_usage()
        b.mctx.clear_usage()
        b.learning = True
        return muted

    def reply(self, prompt: str, steps: int = 40) -> str:
        self.listen(prompt, pause=REPLY_GAP)
        self.said = []
        for _ in range(steps):
            self.tick()
        out, last = [], None
        for s in self.said:                        # 同一个音拉长几拍只算一次
            if s is not None and s != last:
                out.append(s)
            last = s
        return "".join(out)


def overlap(a: str, b: str) -> float:
    """说出的字里有多少是该回的那句里的字（宽松：不看顺序）。"""
    a = [c for c in a if c != "~"]
    return sum(c in b for c in a) / len(a) if a else 0.0


def lcs(a: str, b: str) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            d[i + 1][j + 1] = d[i][j] + 1 if x == y else max(d[i][j + 1], d[i + 1][j])
    return d[-1][-1]


def check(t: Talker, convs, tag: str) -> None:
    """定期检查（不学）：参数是否平衡、有没有慢慢学到规律。"""
    b = t.brain
    b.learning = False
    m = b.mot
    # 1. 平衡：每条可塑连接学到多少；只听别人说话时嘴是否安静
    bal = "  ".join(f"{p.name} {np.abs(p.w - p.w0)[p.ok].mean():.4f}/{np.abs(p.w - p.w0).max():.2f}"
                    for p in b.c.paths if p.cfg.eta > 0)
    t.said = []
    own = []
    ne = b.ear.region.n
    for conv in convs[:3]:
        for ch in conv[0]:
            for k in range(SOUND):
                t.tick(other=ch, new_other=(k == 0))
                on = np.nonzero(b.pfc.region.s[:ne])[0]
                if len(on):
                    own.append(1.0 - b.ear.region.s[on].mean())      # 前额叶亮着的里，不是当前输入对应的比例
        for _ in range(PAUSE):
            t.tick()
    moved = sum(s is not None for s in t.said)
    # 3. 对话：听到别人那句，会不会开口、说出的和该回的有多少重合
    spoke, ov, ex = 0, [], []
    for conv in convs[:10]:
        out = t.reply(conv[0])
        spoke += bool(out.replace("~", ""))
        ov.append(overlap(out, conv[1]))
        if len(ex) < 4:
            ex.append(f"别人：{conv[0]}　它：{out or '（不说话）'}　（该回：{conv[1]}）")
    print(f"[{tag}] 只听时嘴动了 {moved} 拍 | 前额叶里不是当前输入的 {np.mean(own):.2f} | "
          f"听到一句后开口 {spoke}/10，说出的字属于该回那句 {np.mean(ov):.2f}", flush=True)
    print(f"    学到的量（均值/最大）：{bal}", flush=True)
    for e in ex:
        print("    " + e, flush=True)
    b.learning = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=20, help="用前多少段对话")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--check_every", type=int, default=5)
    ap.add_argument("--restructure_every", type=int, default=20)
    ap.add_argument("--save", default="", help="训练完把整个模型存到这个文件（pickle）")
    args = ap.parse_args()
    convs = [c for c in load_corpus() if len(c) >= 2][: args.convs]
    chars = sorted({ch for c in convs for u in c for ch in u})
    print(f"语料：{len(convs)} 段对话，{sum(len(u) for c in convs for u in c)} 字，{len(chars)} 个不同的字", flush=True)
    t = Talker(chars)
    b = t.brain
    print(f"脑：听觉区皮层 {int(b.ear.cortex.sum())}，海马体 / 前额叶 各 {int(b.pfc.region.alive.sum())}，"
          f"运动区 {b.mot.region.n}；没特征的不送链：{t.warmup()} 个", flush=True)
    check(t, convs, "训练前")
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        for conv in convs:
            for i, u in enumerate(conv):
                if i % 2 == 0:
                    t.listen(u, pause=REPLY_GAP if i + 1 < len(conv) else PAUSE)   # 别人说；后面紧接着模型回
                else:
                    t.say(u)
        if (ep + 1) % args.restructure_every == 0:            # 很慢：隔很久才动一点点
            info = b.restructure()
            muted = b.mute_featureless()
            print(f"  神经元增删：分裂 {len(info['split'])}、删除 {len(info['deleted'])}；没特征的不送链 +{muted}", flush=True)
        if (ep + 1) % args.check_every == 0:
            print(f"第 {ep + 1} 遍（这一遍 {time.perf_counter() - t0:.0f}s）", flush=True)
            check(t, convs, f"第 {ep + 1} 遍后")
    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump((t, convs), f)


if __name__ == "__main__":
    main()
