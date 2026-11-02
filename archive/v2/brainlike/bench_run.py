# Run the brainlike agent on the shared bench (copied in benchcopy/, metrics unchanged).
#   python3 -B bench_run.py SEED NA NB NC [ND] [key=value ...]
# World options used: vocab = the bench's own 142 chars (so babbling can cover the vocabulary with random physics),
# eye_layout = 'retino' (innate retinotopy; my copy's only world change) unless eye=random.
# Stages: A (CDS, overheard routines, free play with tutor imitation; babbling on) -> B -> C (-> D), tutor 'model'
# (the agent only overhears replies; no muscle puppeting, no reward except play).  Learning stays on everywhere.
import os, sys, re, time, json, pickle, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "benchcopy"))
from world import World                    # noqa: E402
from metrics import evaluate, headline     # noqa: E402
sys.path.insert(0, HERE)
from agent import Agent, DEFAULT           # noqa: E402

VOCAB = sorted(set(re.findall(r'[一-鿿]', open(os.path.join(HERE, "benchcopy", "world.py"), encoding="utf-8").read())))


def parse(argv):
    pos = [x for x in argv if "=" not in x]
    opt = dict(x.split("=", 1) for x in argv if "=" in x)
    cfg = {}
    for k, v in DEFAULT.items():
        if k in opt:
            cfg[k] = (opt[k] in ("1", "True", "true")) if isinstance(v, bool) else type(v)(opt[k])
    return pos, opt, cfg


def mk(stage, seed, prev=None, tutor="model", eye="retino", vocab=None):
    """bench world; env BL_PHYSICS (random|lawful) and BL_VOCAB (1 = the bench's 142 chars, 0 = the corpus Voice)."""
    if vocab is None:
        vocab = os.environ.get("BL_VOCAB", "1") == "1"
    kw = dict(seed=seed, tutor=tutor, prev=prev, eye_layout=eye, physics=os.environ.get("BL_PHYSICS", "random"))
    if vocab:
        kw["vocab"] = VOCAB
    return World(stage, **kw)


def brief(res, stage):
    r = res.get(stage, {})
    out = []
    f = lambda x: "  - " if x is None else f"{x:.2f}"
    for task in sorted(k for k in r if isinstance(r[k], dict) and "acc" in r[k]):
        d = r[task]
        h = d.get("held", {}) or {}; s = d.get("seen", {}) or {}
        out.append(f"{task}: {f(d['acc'])} (seen {f(s.get('acc'))} n{s.get('n', 0)} | held {f(h.get('acc'))} n{h.get('n', 0)})")
    return " ; ".join(out)


def main():
    pos, opt, cfg = parse(sys.argv[1:])
    seed, NA, NB, NC = (int(x) for x in pos[:4])
    ND = int(pos[4]) if len(pos) > 4 else 0
    eye = opt.get("eye", "retino"); vocab = opt.get("vocab", "1") == "1"
    tag = opt.get("tag", "run")
    cfg["seed"] = seed
    if eye != "retino":
        cfg["no_vis"] = True
    t0 = time.time()
    agent = Agent(cfg)
    wA = mk("A", seed, eye=eye, vocab=vocab)
    for win in wA.warmup_windows():
        agent.warm(win)
    agent.end_warmup()
    print(f"[{tag}] seed {seed} FEAT0 {agent.nf} vocab {len(VOCAB) if vocab else 'corpus'} eye {eye} cfg {cfg}", flush=True)
    logs = {}
    prev = None
    for stage, N in (("A", NA), ("B", NB), ("C", NC), ("D", ND)):
        if N <= 0:
            continue
        if stage != "A":
            agent.c["babble"] = False
        w = wA if stage == "A" else mk(stage, seed, prev=prev, eye=eye, vocab=vocab)
        chunk = max(1, N // 4)
        done = 0
        while done < N:
            n = min(chunk, N - done)
            w.run(agent, n)
            done += n
            if stage == "A":
                nl = int((agent.Mir.w[:agent.syl_n, :agent.body.n_prog].max(1) > 0.3).sum()) if agent.body.n_prog else 0
                print(f"[{tag}] A {done}: syl {agent.syl_n} with-program {nl} programs {agent.body.n_prog} "
                      f"babbles {agent.stats['babble']} kept {agent.stats.get('kept', 0)} chunks {agent.stats['chunks']} "
                      f"t {time.time()-t0:.0f}s", flush=True)
            else:
                res = evaluate(w.log[-3000:])
                print(f"[{tag}] {stage} {done} t {time.time()-t0:.0f}s :: {headline(res).splitlines()[0] if headline(res) else ''}", flush=True)
                print(f"[{tag}]    {brief(res, stage)}", flush=True)
        logs[stage] = w.log
        prev = w
        if stage != "A":
            res = evaluate(w.log)
            ex = [f"{r['prompt']}->{r['reply']}" for r in w.log if r['kind'] == 'probe'][-12:]
            print(f"[{tag}] {stage} final: {brief(res, stage)}", flush=True)
            print(f"[{tag}]    examples: {' '.join(ex)}", flush=True)
    print(f"[{tag}] stats {agent.stats}", flush=True)
    if opt.get("save"):
        with open(opt["save"], "wb") as f:
            pickle.dump({"logs": logs, "stats": agent.stats, "cfg": cfg}, f)


if __name__ == "__main__":
    main()
