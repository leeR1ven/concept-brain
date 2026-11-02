# -*- coding: utf-8 -*-
"""tune2 life run = synth's life_synth.py (bench v2.0 life: one unbroken life, learning always on, nothing reset) with
the T2Agent, plus
  * trivial references on the same scored replies (echo, best constant answer per task type) per life group;
  * favourites / diversity of the wandering thought against the TOKEN-frequency baseline of what was heard:
      time share of the top-20 heard items among held items (wandering steps) vs their share of heard tokens,
      KL(held || heard) in bits, entropies, distinct items held / dominant, top-1 share (collapse), content share
      (objects, foods, names, colours, counts: the 'food' of a foodie) vs function syllables;
      synth's measure (thought at the first heard line after >= 2,000 silent ticks) on both baselines;
  * an experimenter-side category -> char map (each char's sound played through a fresh copy of the ear, the agent's
    templates read only) to name the favourites;
  * npz with T_W, heard counts and the thought at every wandering onset (offline wandering sims).
usage: python3 life_t2.py tag=x seed=5 ticks=1000000 [agent/cfg keys]   (defaults: ARCH_v2 life = synth final_noctx)"""
import os, sys, json, time, collections
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent_t2 import T2Agent                 # noqa: E402
from agent import BENCH                      # noqa: E402
sys.path.insert(0, BENCH)
from life import Life, evaluate_life, headline_life, LIFE_FREE, MEMORY_TASKS   # noqa: E402
from world import OBJECTS, NOVEL, NAMES, COLOURS, COUNTS, SLOTS                 # noqa: E402
from physics import Stream, SOUND                                              # noqa: E402
from frontend import Ear0                                                      # noqa: E402
from run_j import TUNE_BEST                  # noqa: E402
from run_t2 import typed, ARCH_V2            # noqa: E402
from life_synth import THINK_LIFE, summary   # noqa: E402

WANDER = dict(ct_assoc=0.5, wander=50, wander_after=200)
FRAME = set('这是什么在哪里你要吗我好不吃颜色有几个刚才叫看宝的对呀谢客气晚安再见拜早上')
CONTENT = set(''.join(OBJECTS + NOVEL + NAMES + COLOURS + COUNTS + SLOTS)) - FRAME


def life_refs(log):
    ex = [r for r in log if r.get('type') == 'ex']
    free = [r for r in ex if r['kind'] in LIFE_FREE and r.get('acceptable')]
    groups = {'scene': [r for r in free if r.get('needs_scene') and r.get('new_situation', True)],
              'scene_repeat': [r for r in free if r.get('needs_scene') and not r.get('new_situation', True)],
              'held_out': [r for r in free if r.get('held_out')],
              'memory': [r for r in free if r['task'] in MEMORY_TASKS], 'social': [r for r in free if r['kind'] == 'social'],
              'ALL': free}
    out = {}
    for g, rs in groups.items():
        if not rs:
            continue
        by = collections.defaultdict(list)
        for r in rs:
            by[r['task']].append(r)
        const = 0
        for t, L in by.items():
            cnt = collections.Counter(a for r in L for a in set(r['acceptable']))
            const += cnt.most_common(1)[0][1] if cnt else 0
        n = len(rs)
        out[g] = dict(n=n, agent=round(sum(r['reply'] in r['acceptable'] for r in rs) / n, 4),
                      echo=round(sum(r['prompt'] in r['acceptable'] for r in rs) / n, 4), const=round(const / n, 4))
    return out


def char_map(agent, L, chars):
    """category -> chars: each char's sound (the partner's voice) through a fresh ear, sampled like the agent does
    (third tick of the syllable), classified by the agent's templates (read only).  Experimenter-side analysis."""
    v = L.voice
    ear = Ear0(seed=L.seed)
    F0 = np.asarray(agent.F0)
    none = (None, None, False)
    cat = collections.defaultdict(list)
    for ch in chars:
        st = Stream(v.F, v.T, v.T // SOUND, 3, v.levels, 0.0, noise_seed=1)
        ear.reset()
        for _ in range(3):
            ear.step(st.tick((none, none, none)))
        f = None
        for k in range(SOUND):
            s = ear.step(st.tick(((ch, v.hear(ch, 'other'), k == 0), none, none)))
            if k == 2:
                f = s[F0]
        j, m = agent.Z.best(f)
        if j >= 0 and m >= agent.theta_z:
            cat[int(j)].append(ch)
    return dict(cat)


def favourites(agent, cmap):
    core = agent.core
    V = core.c.V
    heard = core.item_n[:V].astype(float)
    p = heard / max(heard.sum(), 1)
    top = np.argsort(-heard)[:20]
    topset = set(int(i) for i in top)
    is_content = np.zeros(V, bool)
    for j, chs in cmap.items():
        if j < V and any(ch in CONTENT for ch in chs):
            is_content[j] = True
    name = lambda j: ''.join(cmap.get(int(j), ['?']))
    out = dict(top20_heard=[name(j) for j in top], top20_token_share=round(float(p[top].sum()), 4),
               top20_type_share=round(20 / max(int((heard > 0).sum()), 1), 4), n_types=int((heard > 0).sum()),
               content_token_share=round(float(p[is_content].sum()), 4), H_heard=round(float(-(p[p > 0] * np.log2(p[p > 0])).sum()), 3))
    wl = core.wander_log
    if wl:
        q = np.zeros(V)
        dom = collections.Counter()
        for d, held in wl:
            dom[d] += 1
            for i in held:
                q[i] += 1
        qn = q / max(q.sum(), 1)
        nz = qn > 0
        kl = float((qn[nz] * np.log2(qn[nz] / np.maximum(p[nz], 1e-9))).sum())
        Hq = float(-(qn[nz] * np.log2(qn[nz])).sum())
        domn = np.array(list(dom.values()), float); domn /= domn.sum()
        favs = np.argsort(-q)[:10]
        out.update(wander_steps=len(wl), held_top20_share=round(float(qn[top].sum()), 4),
                   excess_top20=round(float(qn[top].sum() - p[top].sum()), 4), KL_held_heard_bits=round(kl, 3),
                   H_held=round(Hq, 3), eff_topics=round(2 ** Hq, 2), distinct_held=int(nz.sum()),
                   distinct_dominant=len(dom), top1_dominant_share=round(float(domn.max()), 4),
                   H_dominant=round(float(-(domn * np.log2(domn)).sum()), 3),
                   held_content_share=round(float(qn[is_content].sum()), 4),
                   fav10=[(name(j), round(float(qn[j]), 4), round(float(p[j]), 4)) for j in favs],
                   fav10_content=[name(j) for j in np.argsort(-(q * is_content))[:10] if q[j] > 0],
                   mean_held=round(float(np.mean([len(h) for _, h in wl])), 2))
    fav = agent.fav_log
    held_top = [np.mean([i in topset for i in it]) for (_, _, it) in fav if it]
    out['after_silence'] = dict(n=len(fav), share_top20_held=round(float(np.mean(held_top)), 4) if held_top else None,
                                ends_distinct=len(set(tuple(sorted(it)) for (_, _, it) in fav)),
                                mean_held=round(float(np.mean([len(f[2]) for f in fav])), 2) if fav else None)
    return out


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, seed = A.pop("tag", "life"), int(A.pop("seed", 5))
    ticks, chunk = int(float(A.pop("ticks", 10 ** 6))), int(float(A.pop("chunk", 10 ** 5)))
    tutor = A.pop("tutor", "recast")
    outdir = A.pop("outdir", os.path.join(HERE, "out", "life"))
    wkw = {'eye_layout': 'retino'}
    for k in list(A):
        if k.startswith("w_"):
            wkw[k[2:]] = A.pop(k)
    kw = dict(TUNE_BEST); kw.update(THINK_LIFE); kw.update(ARCH_V2); kw.update(WANDER)
    for k, v in A.items():
        kw[k] = typed(k, v)
    t0 = time.time()
    L = Life(seed=seed, tutor=tutor, **wkw)
    agent = T2Agent(L, seed=seed, **kw)
    snaps = []
    orig_tick = agent.core.tick_thought
    def tick_spy():                                       # thought at every wandering onset (offline sims)
        if agent.quiet == (agent.wander_after // agent.wander + 1) * agent.wander and len(snaps) < 20000:
            snaps.append((agent.t, agent.core.T_a.copy(), len(agent.core.wander_log), agent.core.T_f.copy()))
        orig_tick()
    agent.core.tick_thought = tick_spy
    agent.babble(L.voice, 35960)
    print(f"[{tag}] life seed {seed} ticks {ticks} world {wkw} kw {kw} (babbling {time.time() - t0:.0f}s)", flush=True)
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"{tag}_s{seed}.json")
    done = 0
    while done < ticks:
        L.run(agent, ticks=min(chunk, ticks - done))
        done = L.t
        res = evaluate_life(L.log)
        print(f"[{tag}] {time.time() - t0:.0f}s " + headline_life(res) + f" | cats {agent.Z.n} mix {agent.core.mix.n} "
              f"wander {len(agent.core.wander_log)}", flush=True)
    res = evaluate_life(L.log)
    chars = sorted(set(''.join(r.get('prompt', '') or '' for r in L.log if r.get('type') == 'ex')) |
                   set(''.join(OBJECTS + NOVEL + NAMES + COLOURS + COUNTS + SLOTS)) | FRAME)
    cmap = char_map(agent, L, chars)
    favs = favourites(agent, cmap)
    print(f"[{tag}] favourites: {json.dumps(favs, ensure_ascii=False)}", flush=True)
    rec = dict(tag=tag, seed=seed, cfg={k: str(v) for k, v in kw.items()}, world=wkw, secs=round(time.time() - t0, 1),
               tstats=agent.tstats, ct=agent.core.ct_stats, ctx_stats=agent.core.ctx_stats, summary=summary(res),
               refs=life_refs(L.log), favourites=favs, agent_stats=agent.stats, mix=int(agent.core.mix.n),
               numpy=np.__version__)
    json.dump(rec, open(outp, "w"), ensure_ascii=False, default=str)
    W = agent.core.T_W if agent.core.T_W is not None else np.zeros((1, 1), np.float32)
    np.savez_compressed(os.path.join(outdir, f"{tag}_s{seed}_wander.npz"), T_W=W.astype(np.float32),
                        item_n=agent.core.item_n, snap_t=np.array([s[0] for s in snaps]),
                        snaps=np.array([s[1] for s in snaps], np.float32).reshape(len(snaps), -1),
                        snap_w=np.array([s[2] for s in snaps]), n_wander=len(agent.core.wander_log),
                        snap_f=np.array([s[3] for s in snaps], np.float32).reshape(len(snaps), -1),
                        cmap=json.dumps(cmap, ensure_ascii=False))
    print(f"[{tag}] refs {json.dumps(rec['refs'])}", flush=True)
    print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
