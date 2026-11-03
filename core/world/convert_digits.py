"""Digits read the Chinese way (as in the dialogue conversion): runs of at most 4 digits without a leading 0 as a
cardinal number (31 -> 三十一); longer runs, and years before 年, digit by digit."""
import re

DIG = re.compile(r'[0-9０-９]+')
NUM = '零一二三四五六七八九'


def read_int(s):
    n = int(s)
    if n == 0:
        return '零'
    units = ['', '十', '百', '千']
    digs = [int(c) for c in s]
    out, zero = [], False
    for i, d in enumerate(digs):
        u = units[len(digs) - 1 - i]
        if d == 0:
            zero = True
            continue
        if zero and out:
            out.append('零')
        zero = False
        out.append('十' if d == 1 and u == '十' and i == 0 else NUM[d] + u)
    return ''.join(out)


def digits(text):
    def rep(m):
        s = m.group(0).translate({0xff10 + k: 0x30 + k for k in range(10)})
        after = text[m.end():m.end() + 1]
        if len(s) <= 4 and s[0] != '0' and not (len(s) == 4 and after == '年'):
            return read_int(s)
        return ''.join(NUM[int(c)] for c in s)
    return DIG.sub(rep, text)
