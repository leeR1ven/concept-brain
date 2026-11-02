# Sound-level ordered memory audit: random lines over the B/C lexicon, said by other / other2, through the agent's ear
import os
# (FEAT0) + syllable clock + categories (learning off).  Exact in-order recovery of the line's item sequence.
import sys, pickle, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
from physics import Stream
import world as BW
ag = pickle.load(open(sys.argv[1], "rb"))
v = BW.World("B", seed=0).voice
chars = sorted(set("".join(BW.OBJECTS + BW.SLOTS + BW.COLOURS + BW.COUNTS) + "这是什么在哪里你要吗我好不吃有几颜的"))
lab = {}
for ch in chars:
    j, m = ag.Z.best(ag.own_pattern(v, ch))
    if m >= 0.75: lab[ch] = j
chars = [c for c in chars if c in lab]
rng = np.random.default_rng(1)
ok = {"other": 0, "other2": 0}; n = {"other": 0, "other2": 0}
for t in range(400):
    who = ("other", "other2")[t % 2]
    L = int(rng.integers(2, 9)); line = [chars[int(i)] for i in rng.integers(0, len(chars), L)]
    st = Stream(v.F, v.T, v.T // 3, 3, v.levels); e = ag._ear_copy(); none = (None, None, False)
    det, clk, E_prev, inl = [], 0, 0.0, False
    seq = [(c, k == 0) for c in line for k in range(3)] + [(None, False)] * 3
    for c, fresh in seq:
        src = (c, v.hear(c, who), fresh) if c is not None else none
        win = st.tick((src, none, none) if who == "other" else (none, src, none))
        E = float(win.sum()); f = e.step(win)[ag.F0]
        if E > 0 and E_prev == 0: inl, clk = True, 0
        elif inl and E > 0: clk += 1
        if inl and E > 0 and clk % 3 == 2:
            j, m = ag.Z.best(f); det.append(j if m >= ag.theta_z else None)
        if E == 0: inl = False
        E_prev = E
    n[who] += 1; ok[who] += int(det == [lab[c] for c in line])
print("exact in-order recovery of random lines (len 2-8, %d chars):" % len(chars), {w: f"{ok[w]}/{n[w]}" for w in ok})
