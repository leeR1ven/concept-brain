"""Non-dialogue Chinese text (e.g. Wikipedia: a JSON list of {"completion": text}) -> the syllable format of
corpus.py, so it can be read aloud to the model. Needs pypinyin on PYTHONPATH (only for preparing data).
Each article is cut at punctuation into clauses; consecutive clauses are joined into turns of at most 30
syllables; up to 3 turns form one item (read in turn by two adult voices, like the dialogues). Clauses with Latin
letters or unreadable characters are dropped. Same syllable ids as the dialogue data (toneless base x 5 + tone - 1).
usage: python3 -m core.world.convert_text <wiki.json> <out.npz> [n_articles] [procs]"""
import json
import re
import sys
from multiprocessing import Pool

import numpy as np

from . import phon

HAN = re.compile(r'[㐀-䶿一-鿿豈-﫿]')
CUT = re.compile(r'[，。！？；：、,.!?;:\n（）()「」“”"《》]+')
LATIN = re.compile(r'[A-Za-zＡ-Ｚａ-ｚ]')
BID = None


def init():
    global BID
    BID = {b: i for i, b in enumerate(phon.load_inventory())}


def clause_ids(t):
    from pypinyin import Style, lazy_pinyin
    from .convert_digits import digits
    if LATIN.search(t):
        return None
    t = ''.join(HAN.findall(digits(t)))
    if not t:
        return None
    py = lazy_pinyin(t, style=Style.TONE3, neutral_tone_with_five=True, tone_sandhi=True,
                     errors=lambda x: ['?'] * len(x))
    ids = []
    for p in py:
        if len(p) < 2 or not p[-1].isdigit() or p[:-1] not in BID:
            return None
        ids.append(BID[p[:-1]] * 5 + int(p[-1]) - 1)
    return ids if len(ids) == len(t) else None


def article(text, max_turn=30, per_item=3):
    items, turns, cur = [], [], []
    for cl in CUT.split(text):
        ids = clause_ids(cl)
        if not ids or len(ids) > max_turn:
            continue
        if cur and len(cur) + len(ids) > max_turn:
            turns.append(cur)
            cur = []
        cur = cur + ids
        if len(turns) == per_item:
            items.append(turns)
            turns = []
    if cur:
        turns.append(cur)
    if turns:
        items.append(turns)
    return items


def work(texts):
    return [it for t in texts for it in article(t)]


def main():
    src, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    texts = [d['completion'] for d in json.load(open(src, encoding='utf-8'))[:n]]
    chunks = [texts[a:a + 500] for a in range(0, len(texts), 500)]
    with Pool(procs, initializer=init) as pool:
        items = [it for part in pool.imap(work, chunks) for it in part]
    tl = [len(t) for it in items for t in it]
    syl = np.concatenate([np.array(t, np.uint16) for it in items for t in it])
    np.savez(out, syl=syl, turn_off=np.concatenate([[0], np.cumsum(tl)]).astype(np.int64),
             dlg_off=np.concatenate([[0], np.cumsum([len(it) for it in items])]).astype(np.int64))
    print(f'{len(texts)} articles -> {len(items)} items, {len(tl)} turns, {len(syl)} syllables')


if __name__ == '__main__':
    main()
