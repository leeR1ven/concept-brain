# Staged training / testing of the brainlike agent on the toy world.
#   python3 -B run.py SEED NA NB NC [key=value ...]
# Stage A: NA blocks of (CDS + overheard routine, play period with tutor imitation)
# Stage B / C: model-mode exchanges (other2 says the reply; the agent only listens) mixed with recast exchanges
#   (the agent answers, gets a contingent reward, hears the recast), then a test (agent answers, no feedback).
import sys, time, copy, json, collections, numpy as np
from world import World
from agent import Agent, NSM


def test(world, agent, stage, n=60, kinds=None, freeze=True):
    """test on a deep copy, learning frozen (pure readout of what was learned)."""
    w = copy.deepcopy(world); a = copy.deepcopy(agent)
    a.c["freeze"] = freeze; a.c["babble"] = False
    res = collections.defaultdict(list)
    for i in range(n):
        kind = kinds[i % len(kinds)] if kinds else None
        rec = w.exchange(a, stage, mode="test", kind=kind, test=True)
        ok = rec["reply"] in rec["acc"]
        res[(rec["task"], "held" if rec.get("held_out") else "seen")].append(ok)
        res["_first"].append(bool(rec["reply"]) and any(x[0] == rec["reply"][0] for x in rec["acc"]))
        res["_overlap"].append(len(rec["overlap"]))
        if i < 12:
            res["_ex"].append(f"{rec['prompt']}->{rec['reply']}({'/'.join(rec['acc'])})")
    return res


def summary(res):
    out = []
    for k in sorted(k for k in res if not str(k).startswith("_")):
        out.append(f"{k[0]}/{k[1]} {np.mean(res[k]):.2f} (n={len(res[k])})")
    tot = [x for k in res if not str(k).startswith("_") for x in res[k]]
    return f"ALL {np.mean(tot):.2f} | first-char {np.mean(res['_first']):.2f} | " + " ; ".join(out)


def main():
    seed, NA, NB, NC = (int(x) for x in sys.argv[1:5])
    opt = dict(a.split("=") for a in sys.argv[5:] if "=" in a)
    cfg = {k: (type(v)(opt[k]) if not isinstance(v, bool) else opt[k] in ("1", "True", "true"))
           for k, v in __import__("agent").DEFAULT.items() if k in opt}
    cfg["seed"] = seed
    w = World(seed=seed); a = Agent(cfg)
    t0 = time.time()
    w.warmup(a)
    print(f"FEAT0 {a.nf}", flush=True)
    # ---------------- stage A
    nplay = 0
    for i in range(NA):
        w.cds(a)
        nplay += w.play(a)
        if (i + 1) % max(1, NA // 4) == 0:
            lk = int((a.Mir.w[:a.syl_n, :a.body.n_prog].max(1) > 0.3).sum()) if a.body.n_prog else 0
            print(f"A {i+1}: syl {a.syl_n} (syl with a program {lk}) chunks {a.stats['chunks']} programs {a.body.n_prog} "
                  f"linked {int((a.Mir.w[:, :a.body.n_prog].max(0) > 0.3).sum())} babbles {a.stats['babble']} "
                  f"imitated {nplay} t {time.time()-t0:.0f}s", flush=True)
    a.c["babble"] = False
    for stage, N in (("B", NB), ("C", NC)):
        for i in range(N):
            mode = "model" if (i % 3) else "recast"
            w.exchange(a, stage, mode=mode)
            if (i + 1) % max(1, N // 4) == 0:
                r = test(w, a, stage)
                print(f"{stage} {i+1} ({time.time()-t0:.0f}s): {summary(r)}", flush=True)
        print("  examples:", " ".join(r["_ex"]), flush=True)
    print("stats", a.stats, flush=True)


if __name__ == "__main__":
    main()
