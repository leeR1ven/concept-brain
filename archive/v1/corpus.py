# -*- coding: utf-8 -*-
"""把几种中文对话语料合成一份：每行一段对话（一串句子，别人和模型轮流说）。

  chatterbot  /root/work/corpus/chatterbot/*.yml            问答式
  LCCC        /root/work/corpus/hf/silver_lccc_*.jsonl.gz    日常闲聊（微博等，清华整理）
  PersonalDialog /root/work/corpus/hf/silver_personal_dialog_*.jsonl.gz  日常闲聊

只留短句（贴近日常说话）：去掉标点和空格，每句 2..max_len 个字，每段最多前 max_turns 轮。

  python corpus.py            # 生成 /root/work/corpus/dialogs.jsonl
"""
from __future__ import annotations

import glob
import gzip
import json
import random
import re
import sys

ROOT = "/root/work/corpus"
OUT = f"{ROOT}/dialogs.jsonl"
CLEAN = re.compile(r"[^一-鿿A-Za-z0-9]")


def clean(s: str) -> str:
    return CLEAN.sub("", s)


def chatterbot():
    for f in sorted(glob.glob(f"{ROOT}/chatterbot/*.yml")):
        cur = None
        for line in open(f, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("- - "):
                if cur:
                    yield cur
                cur = [line[4:].strip()]
            elif line.startswith("  - ") and cur is not None:
                cur.append(line[4:].strip())
        if cur:
            yield cur


def lccc():
    for f in sorted(glob.glob(f"{ROOT}/hf/silver_lccc_*.jsonl.gz")):
        for line in gzip.open(f, "rt", encoding="utf-8"):
            yield json.loads(line)


def personal_dialog():
    for f in sorted(glob.glob(f"{ROOT}/hf/silver_personal_dialog_*.jsonl.gz")):
        for line in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(line)
            yield [u[0] if isinstance(u, list) else u for u in r["dialog"]] + list(r.get("golden_response") or [])


def build(max_len: int = 12, max_turns: int = 4, seed: int = 0) -> list:
    out = []
    for src in (chatterbot(), lccc(), personal_dialog()):
        for conv in src:
            turns = []
            for u in conv:
                u = clean(u)
                if not 2 <= len(u) <= max_len:
                    break                                    # 一句不合适，这段对话就到此为止
                turns.append(u)
                if len(turns) == max_turns:
                    break
            if len(turns) >= 2:
                out.append(turns)
    random.Random(seed).shuffle(out)
    return out


if __name__ == "__main__":
    convs = build()
    with open(OUT, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    chars = {ch for c in convs for u in c for ch in u}
    print(f"{len(convs)} 段对话，{sum(len(u) for c in convs for u in c)} 字，{len(chars)} 个不同的字 → {OUT}",
          file=sys.stderr)
    for c in convs[:8]:
        print("  " + " / ".join(c), file=sys.stderr)
