# -*- coding: utf-8 -*-
import os
"""Symbolic curriculum run of the cognitive core: stages B -> C -> D, prequential scores, held-out tests, ablations.
usage: python3 run_dev.py [key=value ...]   (cfg keys of core.Cfg, plus nB nC nD mode seed sleep_every test_n)"""
import sys, time, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Core, Cfg
import devworld as DW

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
def arg(k, d):
    v = ARGS.pop(k, None)
    return d if v is None else type(d)(v) if not isinstance(d, bool) else v in ("1", "True", "true")

nB, nC, nD = arg("nB", 1500), arg("nC", 4000), arg("nD", 3000)
MODE = arg("mode", "recast")
SEED = arg("seed", 0)
SLEEP_EVERY = arg("sleep_every", 300)
TEST_N = arg("test_n", 300)
TAG = arg("tag", "")
cfgkw = {}
for k, v in ARGS.items():
    d = getattr(Cfg(), k)
    cfgkw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
cfg = Cfg(seed=SEED, **cfgkw)

CH = {}
def ids(s):
    out = []
    for ch in s:
        if ch not in CH:
            CH[ch] = len(CH)
        out.append(CH[ch])
    return out
def txt(seq):
    inv = {v: k for k, v in CH.items()}
    return "".join(inv.get(i, "?") for i in seq)

W = DW.DevWorld(seed=SEED, retino=cfg.retino)
for s in DW.OBJ + DW.NEW_OBJ + DW.COL + DW.CNT + DW.SLOT + DW.NAMES + ["这是什么", "在哪里", "你要吗我要", "好不好吃",
          "你喜欢吗", "是什么颜色", "有几个", "不是的", "我叫什么", "刚才", "你好谢谢不客气晚安再见早上好"]:
    ids(s)
core = Core(cfg)


def exchange(ex, learn=True, score=True):
    """one exchange in the chosen tutor mode; returns (ok, reply text)."""
    for p in ex["pre"]:
        if p[0] == "say":
            line, rep = p[1], p[2]
            eye = W.eye(ex["scene"], None)
            core.hear_line(ids(line), eye, None, learn=learn, store=True)
            out = core.speak(eye, None)
            if MODE == "recast":
                core.reward(1.0 if txt(out) == rep else 0.0)
            core.hear_line(ids(rep), eye, None, learn=learn, store=True)
        elif p[0] == "ex":
            exchange(p[1], learn=learn, score=False)
        elif p[0] == "look":
            old = p[1]; s = int(W.rng.integers(3))
            eye = W.eye(old, None)
            core.hear_line(ids("这是什么"), eye, s, learn=learn, store=True)
            core.speak(eye, s)
            core.hear_line(ids(old.objs[s]), eye, s, learn=learn, store=True)
        elif p[0] == "teach":
            line, sc, s = p[1], p[2], p[3]
            eye = W.eye(sc, None)
            core.hear_line(ids(line), eye, s, learn=learn, store=True)
    eye = W.eye(ex["scene"], None)                 # the pointing code is parsed into att, not part of the scene
    core.hear_line(ids(ex["prompt"]), eye, ex["att"], learn=learn, store=learn)
    out = core.speak(eye, ex["att"]); core.own_finish(core.own)
    t = txt(out)
    ok = t in ex["acceptable"]
    if learn:
        if MODE == "recast":
            core.reward(1.0 if ok else 0.0)
            core.hear_line(ids(ex["tutor"]), eye, ex["att"], learn=True, store=True, contingent=True)
        elif MODE == "model":
            core.hear_line(ids(ex["tutor"]), eye, ex["att"], learn=True, store=True)
    return ok, t


def snapshot():
    return (core.prev, core.ctx.u.copy(), core.ctx.s.copy(), W.rng.bit_generator.state)
def restore(s):
    core.prev, core.ctx.u, core.ctx.s, st = s
    W.rng.bit_generator.state = st


def test(stage, n, held):
    """learning OFF, nothing stored; held=True -> only held-out frame x filler pairs / held-out colour combos."""
    snap = snapshot()
    res = collections.defaultdict(list); ex_shown = {}
    tasks = {"B": ["name", "slot_name", "where"],
             "C": ["offer", "taste", "like", "colour", "count", "ritual"],
             "D": ["check", "name_mem", "changed", "oneshot"]}[stage]
    for i in range(n):
        task = tasks[i % len(tasks)]
        ex = W.exchange(stage, test=held, task=task)
        if held and task in ("colour",) and not ex["held"]:
            continue
        if held and task in ("count", "ritual", "name", "slot_name", "where"):
            continue
        ok, t = exchange(ex, learn=(stage == "D" and False), score=True)
        res[task].append(ok)
        if task not in ex_shown or (not ok and len(ex_shown) < 30):
            ex_shown[task] = f"{ex['prompt']}->{t}{'' if ok else ' (want ' + '/'.join(ex['acceptable']) + ')'}"
    restore(snap)
    return {k: (round(float(np.mean(v)), 3), len(v)) for k, v in res.items()}, ex_shown


def run_stage(stage, n, log):
    win = collections.defaultdict(lambda: collections.deque(maxlen=200))
    t0 = time.time()
    for i in range(1, n + 1):
        ex = W.exchange(stage)
        ok, t = exchange(ex)
        win[ex["task"]].append(ok)
        if SLEEP_EVERY and i % SLEEP_EVERY == 0:
            core.sleep(n=200)
        if i % 500 == 0 or i == n:
            pre = {k: round(float(np.mean(v)), 2) for k, v in sorted(win.items())}
            log(f"[{stage}] {i:5d} ex  prequential(last 200 per task) {pre}  mix={core.mix.n} hip={core.hip.n} {time.time()-t0:.0f}s")
    return {k: float(np.mean(v)) for k, v in win.items()}


out = []
def log(s):
    print(s, flush=True); out.append(s)

log(f"cfg: {json.dumps({k: v for k, v in vars(cfg).items()})} mode={MODE} nB={nB} nC={nC} nD={nD} sleep_every={SLEEP_EVERY}")
results = {}
if nB:
    results["B_pre"] = run_stage("B", nB, log)
    results["B_test"] = test("B", TEST_N, held=False); log(f"[B] test seen: {results['B_test']}")
if nC:
    results["C_pre"] = run_stage("C", nC, log)
    results["C_test_seen"] = test("C", TEST_N, held=False); log(f"[C] test seen: {results['C_test_seen']}")
    results["C_test_held"] = test("C", TEST_N, held=True); log(f"[C] test HELD-OUT: {results['C_test_held']}")
if nD:
    results["D_pre"] = run_stage("D", nD, log)
    results["D_test"] = test("D", TEST_N, held=False); log(f"[D] test: {results['D_test']}")
    results["C_test_held_after_D"] = test("C", TEST_N, held=True); log(f"[C after D] test HELD-OUT: {results['C_test_held_after_D']}")
if TAG:
    json.dump(results, open(f"{os.path.dirname(os.path.abspath(__file__))}/out/dev_{TAG}.json", "w"), ensure_ascii=False, indent=1, default=str)
