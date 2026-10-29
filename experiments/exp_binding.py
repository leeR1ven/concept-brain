# -*- coding: utf-8 -*-
"""实验 B：新绑定能不能被想起来（字级弥散 vs 概念单元），2×2 拆因素。

问题来源（char-neuron-network）：「医院」「战争」在句子里一起读 5 遍，新边长进去了，
但只给「医院」时「战争」挤不进激活集——对方最强单字驱动 0.108，阈值 0.372。
作者诊断：绑定散成几十条字-字弱边，没有一条单独够强；概念应是独立单元。

这里用合成数据把两个因素拆开：
  表示：字级（一个词 = 2 个字，字被很多词共用，边在字-字之间）
        vs 概念（一个词 = 一个单元，边在词-词之间）
  速度：慢 η=0.05（字网络的量级） vs 快 η=0.5（海马式一次学习）
学习规则两边完全相同：同一句里先出现 → 后出现 Δw = η/(1+w)（越涨越难涨、无上限），
后 → 先 乘 0.5。

先天知识：每个词 32 个随机"老搭档"，权重 0.5~0.95（相当于语料学来的）。
新经历：50 对互不相关的 (A,B)，每对放进一句 10 个词的句子（另 8 个随机填充词），
         读 n 遍；50 对依次学完后统一测（含互相干扰）。
回忆：只给 A，按驱动给所有词排名，B 进前 25（激活带宽）算想起来。
老知识：A 原来最强的 10 个老搭档，学完后还有几个在前 25。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_WORDS, N_CHARS, PRIOR_K, TOP = 4000, 600, 32, 25


def build(rng):
    chars = rng.integers(0, N_CHARS, size=(N_WORDS, 2))
    prior = np.zeros((N_WORDS, N_WORDS), dtype=np.float32)
    for w in range(N_WORDS):
        nb = rng.choice(N_WORDS, PRIOR_K, replace=False)
        nb = nb[nb != w]
        prior[w, nb] = rng.uniform(0.5, 0.95, len(nb))
    # 字级先天边：把词-词知识摊到字-字上（同一对字可能被多对词共用 → 累加后凹变换），
    # 再加词内的字→字
    raw = np.zeros((N_CHARS, N_CHARS), dtype=np.float64)
    src, dst = np.nonzero(prior)
    for a in range(2):
        for b in range(2):
            np.add.at(raw, (chars[src, a], chars[dst, b]), prior[src, dst])
    np.add.at(raw, (chars[:, 0], chars[:, 1]), 1.0)
    cprior = (1.0 - np.exp(-raw)).astype(np.float32)
    np.fill_diagonal(cprior, 0.0)
    return chars, prior, cprior


def hebb(W, units_in_order, eta):
    """同一句：先 → 后 Δ=η/(1+w)；后 → 先 乘 0.5。"""
    u = np.asarray(units_in_order)
    for i in range(len(u)):
        for j in range(len(u)):
            if u[i] == u[j]:
                continue
            k = 1.0 if i < j else 0.5
            W[u[i], u[j]] += k * eta / (1.0 + W[u[i], u[j]])


def word_scores(rep, W, chars, a):
    if rep == "concept":
        s = W[a].astype(np.float64).copy()
    else:
        drive = W[chars[a]].sum(axis=0)                 # A 的两个字一起往外推
        s = drive[chars].mean(axis=1)                   # 一个词的分 = 它两个字的平均驱动
    s[a] = -np.inf
    return s


def rank_of(s, b):
    return int((s > s[b]).sum())


def run(rep, eta, reps, seed):
    rng = np.random.default_rng(seed)
    chars, prior, cprior = build(rng)
    W = prior.copy() if rep == "concept" else cprior.copy()
    pairs = rng.choice(N_WORDS, size=(50, 2), replace=False)
    old_top = {int(a): np.argsort(-prior[a])[:10] for a, _ in pairs}
    for a, b in pairs:
        fill = rng.choice(N_WORDS, 8, replace=False)
        sent = list(fill[:3]) + [a] + list(fill[3:6]) + [b] + list(fill[6:])
        for _ in range(reps):
            if rep == "concept":
                hebb(W, sent, eta)
            else:
                hebb(W, chars[sent].ravel(), eta)       # 句子 = 一串字
    ranks, kept = [], []
    for a, b in pairs:
        s = word_scores(rep, W, chars, int(a))
        ranks.append(rank_of(s, int(b)))
        top = set(np.argsort(-s)[:TOP].tolist())
        kept.append(sum(int(o) in top for o in old_top[int(a)]))
    ranks = np.array(ranks)
    return {"recall@25": float((ranks < TOP).mean()), "median_rank": float(np.median(ranks)),
            "old_kept_of10": float(np.mean(kept))}


def main():
    res = {}
    for rep in ("char", "concept"):
        for eta in (0.05, 0.5, 1.0):
            for reps in (1, 5):
                rs = [run(rep, eta, reps, seed) for seed in range(3)]
                key = f"{rep}/eta={eta}/reps={reps}"
                res[key] = {k: [r[k] for r in rs] for k in rs[0]}
                m = {k: np.mean(v) for k, v in res[key].items()}
                print(f"{key:28s} 想起来 {m['recall@25']:.2f}  中位名次 {m['median_rank']:7.1f}  "
                      f"老搭档保留 {m['old_kept_of10']:.1f}/10")
    out = Path(__file__).resolve().parents[1] / "results" / "binding.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
