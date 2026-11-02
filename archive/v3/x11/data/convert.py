"""X11 data pipeline, step B: LCCC dialogues (text) -> Mandarin syllables (toneless base id + tone), sharded int arrays.

usage: python3 -I convert.py --src <file.jsonl.gz> --split train --out <dir> [--procs 4] [--shard 100000]
       python3 -I convert.py --inventory <dir>          (writes inventory.json only)
Needs pypinyin on PYTHONPATH (pure-python wheel unpacked next to this script; no install).

Choices (registered in PLAN.md, section D):
- syllable = toneless base (canonical list: every toneless reading in pypinyin's character dictionary, 426 bases)
  + tone 1..5 (5 = neutral); stored as one uint16 id = base * 5 + (tone - 1). Surface tones: pypinyin tone_sandhi
  (3-3 -> 2-3, yi / bu sandhi), as heard.
- cleaning per utterance: drop spaces (LCCC is char-tokenised), digits -> Chinese reading (runs <= 4 digits without a
  leading 0 read as a cardinal number, e.g. 31 -> san shi yi; longer runs and years before 年 read digit by digit),
  punctuation and symbols dropped (no pause inserted), a dialogue with any Latin letter is dropped, a dialogue with a
  character pypinyin cannot read is dropped, a dialogue with an utterance that ends up empty is dropped.
- erhua (儿 read as er5 after a syllable) stays a separate syllable 'er'.
Shard format (npz): syl uint16 [n_syllables]; turn_off int64 [n_turns + 1] (into syl); dlg_off int64 [n_dlg + 1]
(into turns); src int64 [n_dlg] (source line). Speakers alternate by turn within a dialogue (A, B, A, ...)."""
import argparse
import gzip
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'pylib'))

from pypinyin import Style, lazy_pinyin  # noqa: E402
from pypinyin.contrib import tone_convert as tc  # noqa: E402
from pypinyin.pinyin_dict import pinyin_dict  # noqa: E402

HAN = re.compile(r'[㐀-䶿一-鿿豈-﫿]')
LATIN = re.compile(r'[A-Za-zＡ-Ｚａ-ｚ]')
DIG = re.compile(r'[0-9０-９]+')
NUM = '零一二三四五六七八九'


def inventory():
    bases = set()
    for v in pinyin_dict.values():
        for p in v.split(','):
            bases.add(tc.to_normal(p, v_to_u=False))
    return sorted(bases)


BASES = inventory()
BID = {b: i for i, b in enumerate(BASES)}


def read_int(s):
    """cardinal reading of a digit string of length <= 4 with no leading zero"""
    n = int(s)
    if n == 0:
        return '零'
    units = ['', '十', '百', '千']
    digs = [int(c) for c in s]
    out, zero = [], False
    L = len(digs)
    for i, d in enumerate(digs):
        u = units[L - 1 - i]
        if d == 0:
            zero = True
            continue
        if zero and out:
            out.append('零')
        zero = False
        if d == 1 and u == '十' and i == 0:
            out.append('十')
        else:
            out.append(NUM[d] + u)
    return ''.join(out)


def digits(text):
    def rep(m):
        s = m.group(0).translate({0xff10 + k: 0x30 + k for k in range(10)})
        after = text[m.end():m.end() + 1]
        if len(s) <= 4 and s[0] != '0' and not (len(s) == 4 and after == '年'):
            return read_int(s)
        return ''.join(NUM[int(c)] for c in s)
    return DIG.sub(rep, text)


def clean(utt):
    """-> (han-only string, reason or None)"""
    t = utt.replace(' ', '')
    if LATIN.search(t):
        return None, 'latin'
    t = digits(t)
    t = ''.join(HAN.findall(t))
    if not t:
        return None, 'empty'
    return t, None


def convert_dialogue(utts):
    """-> (list of uint16 arrays per turn, None) or (None, reason)"""
    turns = []
    for u in utts:
        t, why = clean(u)
        if why:
            return None, why
        py = lazy_pinyin(t, style=Style.TONE3, neutral_tone_with_five=True, tone_sandhi=True,
                         errors=lambda x: ['?'] * len(x))
        ids = []
        for p in py:
            if len(p) < 2 or not p[-1].isdigit():
                return None, 'unreadable'
            b, tone = p[:-1], int(p[-1])
            if b not in BID or not 1 <= tone <= 5:
                return None, 'unreadable'
            ids.append(BID[b] * 5 + tone - 1)
        if len(ids) != len(t):
            return None, 'unreadable'
        turns.append(np.array(ids, np.uint16))
    return turns, None


def work(chunk):
    a0, lines = chunk
    out, why = [], {}
    for k, line in enumerate(lines):
        try:
            utts = json.loads(line)
        except ValueError:
            why['json'] = why.get('json', 0) + 1
            continue
        if isinstance(utts, dict):
            utts = utts.get('dialog') or utts.get('conversation') or []
        if len(utts) < 2:
            why['one_turn'] = why.get('one_turn', 0) + 1
            continue
        turns, r = convert_dialogue(utts)
        if r:
            why[r] = why.get(r, 0) + 1
            continue
        out.append((a0 + k, turns))
    return out, why


def chunks(path, n=20000):
    buf, a0 = [], 0
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            buf.append(line)
            if len(buf) == n:
                yield a0, buf
                a0, buf = i + 1, []
    if buf:
        yield a0, buf


class Writer:
    def __init__(self, out, split, shard):
        self.out, self.split, self.shard = out, split, shard
        self.k, self.buf = 0, []
        self.syl_count = np.zeros(len(BASES) * 5, np.int64)
        self.turns_hist = np.zeros(64, np.int64)
        self.len_hist = np.zeros(256, np.int64)
        self.n_dlg = self.n_turn = self.n_syl = 0
        self.files = []

    def add(self, src, turns):
        self.buf.append((src, turns))
        if len(self.buf) == self.shard:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        syl = np.concatenate([t for _, ts in self.buf for t in ts])
        tl = np.array([len(t) for _, ts in self.buf for t in ts], np.int64)
        nt = np.array([len(ts) for _, ts in self.buf], np.int64)
        turn_off = np.concatenate([[0], np.cumsum(tl)])
        dlg_off = np.concatenate([[0], np.cumsum(nt)])
        src = np.array([s for s, _ in self.buf], np.int64)
        name = f'{self.split}_{self.k:03d}.npz'
        np.savez(os.path.join(self.out, name), syl=syl, turn_off=turn_off, dlg_off=dlg_off, src=src)
        self.files.append(name)
        self.syl_count += np.bincount(syl, minlength=len(self.syl_count))
        self.turns_hist += np.bincount(np.minimum(nt, 63), minlength=64)
        self.len_hist += np.bincount(np.minimum(tl, 255), minlength=256)
        self.n_dlg += len(nt)
        self.n_turn += len(tl)
        self.n_syl += len(syl)
        self.k += 1
        self.buf = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--shard', type=int, default=100000)
    ap.add_argument('--inventory', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inv = dict(bases=BASES, n_bases=len(BASES), id_rule='id = base * 5 + (tone - 1); tone 5 = neutral',
               split={b: [tc.to_initials(b, strict=True), tc.to_finals(b, strict=True)] for b in BASES})
    with open(os.path.join(a.out, 'inventory.json'), 'w') as f:
        json.dump(inv, f, ensure_ascii=False, indent=0)
    if a.inventory:
        return
    t0 = time.time()
    W = Writer(a.out, a.split, a.shard)
    why_all, n_in = {}, 0
    with Pool(a.procs) as pool:
        for out, why in pool.imap(work, chunks(a.src)):
            for k, v in why.items():
                why_all[k] = why_all.get(k, 0) + v
            n_in += len(out) + sum(why.values())
            for src, turns in out:
                W.add(src, turns)
            if W.k % 5 == 0 and W.buf == []:
                pass
            print(f'{time.time() - t0:7.0f}s lines {n_in} kept {W.n_dlg + len(W.buf)}', flush=True)
    W.flush()
    st = dict(split=a.split, src=os.path.basename(a.src), lines=n_in, kept=W.n_dlg, dropped=why_all,
              turns=W.n_turn, syllables=W.n_syl, shards=W.files, seconds=round(time.time() - t0, 1),
              syl_count=W.syl_count.tolist(), turns_hist=W.turns_hist.tolist(), len_hist=W.len_hist.tolist())
    with open(os.path.join(a.out, f'stats_{a.split}.json'), 'w') as f:
        json.dump(st, f)
    print(json.dumps({k: v for k, v in st.items() if k not in ('syl_count', 'turns_hist', 'len_hist', 'shards')}))


if __name__ == '__main__':
    main()
