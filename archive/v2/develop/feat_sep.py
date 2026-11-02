# Aligned-window FEAT0 patterns per (char, voice): within-char cross-voice match vs between-char match (bench physics).
import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_copy"))
from physics import Voice, Stream, corpus_chars
from frontend import Ear0, feat0
import world as BW
w = BW.World('B', seed=0)
v = w.voice
ear = Ear0(seed=0); F0 = feat0(Ear0(seed=0), w.warmup_windows()); ear.set_feat0(F0)
need = sorted(set("".join(BW.OBJECTS + BW.COLOURS + BW.COUNTS + BW.SLOTS + BW.NAMES)) | set("这是什么在哪里你要吗我好不吃很有几颜色的谢客气晚安再见拜早上呀"))
print("FEAT0", len(F0), "needed chars", len(need))
def pattern(seq, who):
    st = Stream(v.F, v.T, v.T // 3, 3, v.levels); ear.reset(); pats = []
    none = (None, None, False)
    for j, ch in enumerate(seq):
        for k in range(3):
            src = (ch, v.hear(ch, who), k == 0)
            srcs = [none, none, none]; srcs[{'other': 0, 'other2': 1, 'self': 2}[who]] = src
            win = st.tick(tuple(srcs)); s = ear.step(win)
            if k == 2: pats.append(s[F0].copy())
    return pats
rng = np.random.default_rng(0)
P = {}
for who in ('other', 'other2', 'self'):
    for c in need:
        # the char in a random context (previous char random) -> aligned window is the char alone
        pre = need[rng.integers(len(need))]
        P[(c, who)] = pattern([pre, c], who)[1]
def match(t, f): return (t & f).sum() / max(t.sum(), 1)
same = [match(P[(c, 'other')], P[(c, w2)]) for c in need for w2 in ('other2', 'self')]
diff = [match(P[(a, 'other')], P[(b, 'other')]) for a in need for b in need if a != b]
print("active FEAT0 per aligned window: %.0f" % np.mean([p.sum() for p in P.values()]))
print("within-char cross-voice match: mean %.3f min %.3f" % (np.mean(same), np.min(same)))
print("between-char match: mean %.3f  p99 %.3f max %.3f" % (np.mean(diff), np.percentile(diff, 99), np.max(diff)))
# context dependence: same char after different previous chars (aligned window should be identical)
ctx = [match(pattern([need[i], '苹'], 'other')[1], pattern([need[j], '苹'], 'other')[1]) for i, j in rng.integers(0, len(need), (20, 2))]
print("same char, different previous char (reserve/aligned): mean %.3f min %.3f" % (np.mean(ctx), np.min(ctx)))
# misaligned windows (tick 1, 2 of a char) vs the aligned template of the same char
st_mis = []
for c in need[:30]:
    st = Stream(v.F, v.T, v.T // 3, 3, v.levels); ear.reset(); none = (None, None, False)
    seq = [need[rng.integers(len(need))], c, need[rng.integers(len(need))]]
    pats = []
    for j, ch in enumerate(seq):
        for k in range(3):
            win = st.tick(((ch, v.hear(ch, 'other'), k == 0), none, none)); pats.append(ear.step(win)[F0].copy())
    al = pats[5]
    st_mis += [match(al, pats[3]), match(al, pats[4]), match(al, pats[6]), match(al, pats[7])]
print("aligned template vs misaligned ticks of the same/next char: mean %.3f max %.3f" % (np.mean(st_mis), np.max(st_mis)))
