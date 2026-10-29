# -*- coding: utf-8 -*-
"""实验 E：分布式联想回忆——前额叶看过"A 之后是 B"，只给 A 能不能想起 B 的特征神经元。

没有"一个画面一个神经元"：每张画面激活视觉层里很多神经元，经链送到前额叶接收端。
某张画面的**特征神经元** = 看它时亮、看灰底（什么都没有）时不亮的那些。

流程（一拍 = 看一眼）：
  学习：灰 5 拍 → A 4 拍 → B 4 拍 → 灰 5 拍。
        一次对 5 对：只学 1 遍；重复对 5 对：学 10 遍（交错）。
  干扰：随后看 FILLER 对别的画面（同样的节奏），让海马体的快权重衰减。
  测试（不学习）：灰 10 拍 → A 4 拍 → 灰 8 拍（撤掉输入，前额叶自己想）。
        在"撤掉输入后"的 8 拍里：对每张从没出现过的对照画面 O，
        B 独有的特征神经元（不属于 A、灰底、O）最多亮了几成 → 想起率；
        O 独有的特征神经元（不属于 A、灰底、B）最多亮了几成 → 误想率。
方案：
  none     不学习
  hippo    只有海马体（快涨快降）
  cortex   只有皮层慢学
  both     两个都有（用户的架构）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.global_decay import GlobalDecay  # noqa: E402
from brain.pfc import PFC, PFCCfg  # noqa: E402
from brain.vision import SheetCfg, VisionSheet  # noqa: E402

H = W = 16
N_ONCE, N_REP, REPS = 5, 5, 10
FILLER = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ATTN = sys.argv[3] if len(sys.argv) > 3 else "inhib_pair"   # 前额叶注意力：抑制神经元 / global（旧全局阈值）
HIPPO_HALF = 300              # 拍
VARIANTS = {"none": (0.0, 0.0), "hippo": (0.0, 1.0), "cortex": (0.02, 0.0), "both": (0.02, 1.0)}


def smooth(rng):
    low = rng.uniform(1, 9, size=(4, 4))
    return np.clip(np.round(np.kron(low, np.ones((4, 4))) + rng.normal(0, 0.4, (H, W))), 0, 10).astype(int)


def run(variant, seed):
    rng = np.random.default_rng(seed)
    sheet = VisionSheet(H, W, SheetCfg(), seed=seed)
    gray = np.full((H, W), 5, dtype=int)
    n_img = 2 * (N_ONCE + N_REP) + 2 * FILLER + 10
    imgs = [smooth(rng) for _ in range(n_img)]
    resp = lambda im: sheet.respond(sheet.encode(im)).astype(np.float32)  # noqa: E731
    g_act = resp(gray)
    feats = [(resp(im) > 0) & (g_act == 0) for im in imgs]           # 特征神经元

    eta_c, eta_h = VARIANTS[variant]
    target = int(g_act.sum()) + 150                                   # 带宽：感觉驱动的 + 150 个"想"的名额
    pfc = PFC(len(g_act), PFCCfg(attention=ATTN, target=target, eta_cortex=eta_c, eta_hippo=eta_h))
    gd = GlobalDecay(T=50)
    cortex_m, hippo_m = pfc.decay_targets()
    gd.register("hippocampus", hippo_m, GlobalDecay.gamma_for_half_life(HIPPO_HALF, T=50))
    gd.register("cortex", cortex_m, 1.0)

    def show(act, n, learn=True, rec=None):
        pfc.cfg.learn = learn
        for _ in range(n):
            a = pfc.step(act)
            gd.tick()
            if rec is not None:
                rec.append(a.copy())

    def episode(ia, ib):
        show(g_act, 5)
        show(resp(imgs[ia]), 4)
        show(resp(imgs[ib]), 4)
        show(g_act, 5)

    pairs = [(2 * i, 2 * i + 1) for i in range(N_ONCE + N_REP)]
    once, rep = pairs[:N_ONCE], pairs[N_ONCE:]
    for p in once:
        episode(*p)
    for _ in range(REPS):
        for p in rep:
            episode(*p)
    base = 2 * (N_ONCE + N_REP)
    for k in range(FILLER):
        episode(base + 2 * k, base + 2 * k + 1)
    others = list(range(n_img - 10, n_img))                          # 从没出现过的画面：误想对照

    def test(ia, ib):
        show(g_act, 10, learn=False)
        show(resp(imgs[ia]), 4, learn=False)
        rec = []
        show(g_act, 8, learn=False, rec=rec)
        on = np.array(rec) >= pfc.cfg.on_thr
        hit, fa = [], []
        for io in others:                    # 只比"B 独有"和"对照独有"的特征（随机画面本来就共享很多特征）
            fb = feats[ib] & ~feats[ia] & ~feats[io]
            fo = feats[io] & ~feats[ia] & ~feats[ib]
            if fb.any() and fo.any():
                hit.append(float(on[:, fb].mean(axis=1).max()))
                fa.append(float(on[:, fo].mean(axis=1).max()))
        return float(np.mean(hit)), float(np.mean(fa))

    out = {}
    for name, ps in (("once", once), ("repeated", rep)):
        r = [test(*p) for p in ps]
        out[name] = {"recall": float(np.mean([x[0] for x in r])), "false": float(np.mean([x[1] for x in r]))}
    out["n_feat_mean"] = float(np.mean([f.sum() for f in feats]))
    out["target"] = target
    return out


def main():
    res = {}
    print(f"注意力 {ATTN}；干扰 {FILLER} 对（约 {FILLER * 18} 拍），海马体半衰期 {HIPPO_HALF} 拍，{SEEDS} 个种子")
    for v in VARIANTS:
        t0 = time.perf_counter()
        rs = [run(v, s) for s in range(SEEDS)]
        res[v] = {k: {kk: float(np.mean([r[k][kk] for r in rs])) for kk in rs[0][k]} for k in ("once", "repeated")}
        print(f"{v:7s} 一次对: 想起 {res[v]['once']['recall']:.2f} / 误想 {res[v]['once']['false']:.2f}   "
              f"重复对: 想起 {res[v]['repeated']['recall']:.2f} / 误想 {res[v]['repeated']['false']:.2f}   "
              f"(特征神经元均 {rs[0]['n_feat_mean']:.0f}，带宽 {rs[0]['target']}，{time.perf_counter() - t0:.0f}s)")
    out = Path(__file__).resolve().parents[1] / "results" / f"recall_filler{FILLER}_{ATTN}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
