# -*- coding: utf-8 -*-
"""完整模型：对话训练 → 和它说话。以模型经肌肉说出来的东西为准。

  耳朵 → 听觉区 → 链 → 海马体 / 前额叶 → 回程 → 运动区 → 嘴 → 自己的声音又被耳朵听到。

训练：对话里别人说一句（别人的声音），模型回一句。模型那一句由外力带着嘴说出来
      （推该动的肌肉神经元 + 去抑制神经元），说完推抑制神经元让嘴停下（用户：先激活肌肉神经元，再激活抑制神经元）；
      它听到的是自己的声音。全脑照常学。
测试：别人说一句，然后安静，看嘴说出什么。

  python talk.py --convs 6000            # 一段段听新对话（不反复死记同几段）
语料：python corpus.py 合成的中文对话（chatterbot 问答 + LCCC / PersonalDialog 日常闲聊，只留短句）
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain.brain import Brain, BrainCfg  # noqa: E402
from brain.net import N  # noqa: E402
from world.voice import Stream, Voice  # noqa: E402

CORPUS = "/root/work/corpus/dialogs.jsonl"     # python corpus.py 生成（chatterbot + LCCC + PersonalDialog）
SOUND, PAUSE = 3, 4          # 每个字的声音持续几拍（一句话里字连着说）；对话和对话之间停几拍
REPLY_GAP = 3                # 训练时别人说完到模型接话的停顿（人对话的空隙很短；测试时什么时候开口由模型自己定）


def load_corpus(path: str = CORPUS) -> list:
    """corpus.py 生成的对话（每行一段：一串句子，别人和模型轮流说）。"""
    return [json.loads(line) for line in open(path, encoding="utf-8")]


class Talker:
    def __init__(self, chars, cfg: BrainCfg | None = None, seed: int = 0):
        self.voice = v = Voice(chars, seed=seed)
        self.brain = Brain(v.F, v.T, v.n_muscles, cfg or BrainCfg(seed=seed))
        self.stream = Stream(v, step=v.T // SOUND)            # 声音连续流动：每个字 SOUND 拍放完
        self.saying = None
        self.said: list = []

    def tick(self, other: str | None = None, drive: np.ndarray | None = None, new_other: bool = False,
             go: bool = False):
        """一拍：耳朵听到 别人这一拍说的 + 自己上一拍说的（连续的声音流）。返回嘴这一拍发出的。"""
        v, b = self.voice, self.brain
        heard = self.stream.tick(other, self.saying, new_other)
        force = b.step(b.ear.net.encode(heard), drive, go)
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
        """模型的那一句：外力在开头推一下起动（"轮到你了"），每个字开头那一拍指出用哪几块肌肉；
        字的保持、力度、换字、说完停下都由运动区自己的节拍完成。"""
        m = self.brain.mot
        for i, ch in enumerate(utt):
            for k in range(SOUND):
                self.tick(drive=m.cue(self.voice.need[ch]) if k == 0 else None, go=(i == 0 and k == 0))
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

    def reply(self, prompt: str, steps: int = 60) -> str:
        self.listen(prompt, pause=0)
        self.said = []
        for _ in range(steps):
            self.tick()
        moved = [i for i, s in enumerate(self.said) if s is not None]
        self.latency = moved[0] if moved else None                # 别人说完后第几拍开口
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
    bal = "  ".join(f"{p.name} {p.learned()[0]:.4f}/{p.learned()[1]:.2f}" for p in b.c.paths if p.cfg.eta > 0)
    t.said = []
    own = []
    ne = b.ear.region.n
    for conv in convs[:3]:
        for ch in conv[0]:
            for k in range(SOUND):
                t.tick(other=ch, new_other=(k == 0))
                on = np.nonzero(N(b.pfc.region.s[:ne]))[0]
                if len(on):
                    own.append(1.0 - N(b.ear.region.s)[on].mean())   # 前额叶亮着的里，不是当前输入对应的比例
        for _ in range(PAUSE):
            t.tick()
    moved = sum(s is not None for s in t.said)
    # 2. 想法：听完一句、停顿之后，思考层 / 海马体的状态还认不认得出是哪句（同一句在两种不同前文之后各听一次）
    jac = lambda x, y: (x & y).sum() / max(1, (x | y).sum())    # noqa: E731
    n = len(convs)
    st = {"think": [[], []], "hip": [[], []]}
    hold, ends = [], []
    for rep in (0, 1):
        for i, conv in enumerate(convs):
            t.listen(convs[(i + 1 + rep) % n][1])                 # 前文
            t.listen(conv[0], pause=0)
            a0 = N(b.think.region.s).copy()
            for _ in range(REPLY_GAP):
                t.tick()
            a1 = N(b.think.region.s).copy()
            if rep == 0:
                hold.append(jac(a0, a1))
                ends.append(a0)
            st["think"][rep].append(a1)
            st["hip"][rep].append(N(b.hip.region.s).copy())
            for _ in range(PAUSE):
                t.tick()
    across = [jac(ends[i], ends[j]) for i in range(n) for j in range(i + 1, n)]
    dec = {k: np.mean([int(np.argmax([jac(v[1][i], v[0][j]) for j in range(n)]) == i) for i in range(n)])
           for k, v in st.items()}
    # 3. 对话：听到别人那句，会不会开口、说出的和该回的有多少重合
    spoke, ov, ex, lat, mess = 0, [], [], [], []
    for conv in convs[:10]:
        out = t.reply(conv[0])
        spoke += bool(out.replace("~", ""))
        if t.latency is not None:
            lat.append(t.latency)
        said = [s for s in t.said if s is not None]
        if said:
            mess.append(said.count("~") / len(said))
        ov.append(overlap(out, conv[1]))
        if len(ex) < 4:
            ex.append(f"别人：{conv[0]}　它：{out or '（不说话）'}　（该回：{conv[1]}）")
    print(f"[{tag}] 只听时嘴动了 {moved} 拍 | 前额叶里不是当前输入的 {np.mean(own):.2f} | "
          f"思考层停顿后还保持 {np.mean(hold):.2f}、不同句子之间 {np.mean(across):.2f}、"
          f"停顿后认出是哪句 思考层 {dec['think']:.2f} 海马体 {dec['hip']:.2f}（瞎猜 {1 / n:.2f}） | "
          f"听到一句后开口 {spoke}/10（说完后第 {np.median(lat) if lat else '-'} 拍开口，含糊音占 "
          f"{np.mean(mess) if mess else 0:.2f}），说出的字属于该回那句 {np.mean(ov):.2f}", flush=True)
    print(f"    学到的量（均值/最大）：{bal}", flush=True)
    for e in ex:
        print("    " + e, flush=True)
    b.learning = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=6000, help="一共听多少段对话（从语料里依次取，每段只听一次）")
    ap.add_argument("--check_every", type=int, default=200, help="每听多少段对话检查一次")
    ap.add_argument("--restructure_every", type=int, default=1000, help="每听多少段对话做一次神经元增删（很慢）")
    ap.add_argument("--save", default="", help="每次检查后把整个模型存到这个文件（pickle）")
    ap.add_argument("--set", nargs="*", default=[], help="改参数，如 sense.internal.gamma=0.99 motor.dis_thr=2.0")
    args = ap.parse_args()
    pool = [c for c in load_corpus() if len(c) >= 2]
    probe = pool[:20]                                       # 检查用的一组（也会被听到，像熟悉的对话）
    convs = pool[: args.convs]
    chars = sorted({ch for c in pool for u in c for ch in u})
    print(f"语料：{len(pool)} 段对话、{len(chars)} 个不同的字；这次听前 {len(convs)} 段", flush=True)
    cfg = BrainCfg()
    for kv in args.set:                                     # 批量测试参数用
        key, val = kv.split("=")
        *path, last = key.split(".")
        target = cfg
        for name in path:
            target = getattr(target, name)
        setattr(target, last, float(val))
    if args.set:
        print("参数：", " ".join(args.set), flush=True)
    t = Talker(chars, cfg)
    b = t.brain
    print(f"脑：听觉区皮层 {int(b.ear.cortex.sum())}，运动皮层 {int(b.mctx.cortex.sum())}，"
          f"海马体 / 前额叶 各 {int(b.pfc.region.alive.sum())}，肌肉区 {b.mot.region.n}；"
          f"没特征的不送链：{t.warmup()} 个", flush=True)
    check(t, probe, "训练前")
    t0 = time.perf_counter()
    for done, conv in enumerate(convs, 1):
        for i, u in enumerate(conv):
            if i % 2 == 0:
                t.listen(u, pause=REPLY_GAP if i + 1 < len(conv) else PAUSE)   # 别人说；模型过一会儿回
            else:
                t.say(u)
        if done % args.restructure_every == 0:                    # 很慢：隔很久才动一点点
            info = b.restructure()
            muted = b.mute_featureless()
            print(f"  神经元增删：分裂 {len(info['split'])}、删除 {len(info['deleted'])}；没特征的不送链 +{muted}", flush=True)
        if done % args.check_every == 0:
            print(f"已听 {done} 段对话（{(time.perf_counter() - t0) / 60:.0f} 分钟）", flush=True)
            check(t, probe, f"{done} 段后")
            if args.save:
                with open(args.save, "wb") as f:
                    pickle.dump((t, probe), f)


if __name__ == "__main__":
    main()
