# -*- coding: utf-8 -*-
"""P5.5: batched sweep mode — B independent agents (different seeds and per-agent parameters) run the full staged J
protocol (ref/run_j.py's, unchanged) in ONE process on ONE batched brain.

Each protocol runs in its own thread with its own bench worlds; the worlds call agent.step(obs) as usual, and a hub
gathers the observations of every protocol that is waiting and advances all of them with one batched brain step
(agents that are not waiting are frozen by the active mask).  Each protocol owns 3 brain rows: its agent, a probe copy
(the reference's copy.deepcopy for frozen probes) and the drift copy.  Writes one result JSON per agent in the format
of run_j.py (analyze.py scores it unchanged).
usage: python3 sweep_j.py jobs=jobs.txt [out=out/sweep] [nA=300 nS=1000 evaln=80 drift=300]
  jobs.txt: one protocol per line: tag seed [param=value ...]   (params: brain_torch.PER_AGENT)"""
import os, sys, json, time, copy, threading
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                                  # noqa: E402
from agents_torch import birth_init, babble_patterns          # noqa: E402
from brain_torch import Brain, Cfg, copy_row, PER_AGENT       # noqa: E402
from world import World                                       # noqa: E402
from metrics import evaluate                                  # noqa: E402
import run_j_torch as RJ                                      # noqa: E402


class Hub:
    def __init__(self, br):
        self.br = br
        self.cv = threading.Condition()
        self.waiting, self.result = {}, {}
        self.alive = 0
        self.ticks = 0
        self.t_step = 0.0
        self.err = None

    def step(self, row, obs):
        with self.cv:
            self.waiting[row] = obs
            self.cv.notify_all()
            while row not in self.result:
                self.cv.wait()
            return self.result.pop(row)

    def loop(self):
        br = self.br
        B = br.B
        win = np.zeros((B, 8, 6), np.int64); eye = np.zeros((B, br.NE), bool); rew = np.zeros(B)
        with self.cv:
            while True:
                while self.alive and len(self.waiting) < self.alive:
                    self.cv.wait()
                if not self.alive:
                    return
                act = np.zeros(B, bool)
                for r, o in self.waiting.items():
                    act[r] = True
                    win[r] = o['ear']; eye[r] = np.asarray(o['eye']).astype(bool); rew[r] = float(o.get('reward', 0.0))
                t0 = time.time()
                f = br.step(win, eye, np.where(act, rew, 0.0), act)
                self.t_step += time.time() - t0
                self.ticks += 1
                for r in self.waiting:
                    self.result[r] = f[r].astype(np.float32)
                self.waiting = {}
                self.cv.notify_all()


class Agent:
    """one brain row seen as a bench agent (the interface run_j.py's protocol uses)."""

    def __init__(self, hub, row, ear=None, F0=None, seed=0, rows=None):
        self.hub, self.row, self.ear, self.F0, self.seed, self.rows = hub, row, ear, F0, seed, rows

    @property
    def learning(self):
        return bool(self.hub.br.learning[self.row])

    @learning.setter
    def learning(self, v):
        with self.hub.cv:
            self.hub.br.learning[self.row] = bool(v)

    def set_learn_listen(self, v):
        with self.hub.cv:
            self.hub.br.learn_listen[self.row] = bool(v)

    def step(self, obs):
        return self.hub.step(self.row, obs)

    def babble(self, voice, n):
        pats, sets = babble_patterns(self.ear, self.F0, voice, n, self.seed)
        with self.hub.cv:
            self.hub.br.load_babble(np.array([self.row]), pats, sets)

    def copy_to(self, slot):
        dst = self.rows[slot]
        with self.hub.cv:
            copy_row(self.hub.br, self.row, dst)
        return Agent(self.hub, dst, self.ear, self.F0, self.seed, self.rows)

    def summary(self):
        b, r = self.hub.br, self.row
        return dict(categories=int(b.zn[r]), programs=int((b.cat2prog[r] >= 0).sum()), mix=int(b.conj.n[r].item()),
                    hip=int(b.hn[r]))


def probe(agent, stage, n, seed, mech, wkw):
    ag = agent.copy_to('P')
    if stage in "BC":
        ag.learning = False
    else:
        ag.learning = True
        ag.set_learn_listen(False)
    w = World(stage, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3, **wkw)
    w.run(ag, n)
    res = evaluate(w.log)
    RJ.mech_add(mech, res)
    S = res[stage]
    out = {k: [round(v["acc"], 4) if v.get("acc") is not None else None, v.get("n"),
               (round(v["held"]["acc"], 4) if (v.get("held") or {}).get("acc") is not None else None),
               (v.get("held") or {}).get("n"), round(v["spoke"], 3) if v.get("spoke") is not None else None]
           for k, v in S.items() if isinstance(v, dict) and "acc" in v}
    ov = [dict(task=r.get('task'), kind=r.get('kind'), prompt=r.get('prompt'), reply=r.get('reply'), lat=r.get('latency'),
               mwp=r.get('mouth_while_partner'), stop=r.get('stop')) for r in w.log if (r.get('mouth_while_partner') or 0) > 0]
    if ov:
        mech.setdefault('overlap_records', []).extend(ov[:5])
    fr = [r for r in w.log if r['kind'] in ('probe', 'recast')]
    first, later, last_dlg = [], [], None
    for r in fr:
        (first if r['dlg'] != last_dlg else later).append(bool(r['reply']))
        last_dlg = r['dlg']
    out['_spoke_pos'] = [round(float(np.mean(first)), 3) if first else None, round(float(np.mean(later)), 3) if later else None]
    return out


def protocol(hub, agent, tag, seed, params, P, outdir, wA):
    t0 = time.time()
    try:
        nA, nS, evaln, drift_n = P['nA'], P['nS'], P['evaln'], P['drift']
        stages = ["B", "BC", "BCDD"]
        wkw = dict(eye_layout='retino')
        log = lambda s: print(f"[{tag} s{seed} {time.time() - t0:.0f}s] {s}", flush=True)
        wA.run(agent, nA); agent.babble(wA.voice, 35960)
        log(f"A: {agent.summary()}")
        mech, mech_train = {}, {}
        prev, table, train_res = wA, {}, {}
        for i, st in enumerate(stages):
            w = World(st, seed=seed, tutor="recast", prev=prev, **wkw)
            w.run(agent, nS); prev = w
            rs = evaluate(w.log)
            RJ.mech_add(mech_train, rs)
            train_res[f"s{i + 1}"] = {s: (round(rs[s]['ALL']['acc'], 4) if rs.get(s, {}).get('ALL', {}).get('acc') is not None
                                         else None) for s in "BCD" if s in rs}
            ev = {es: probe(agent, es, evaln, seed, mech, wkw) for es in "BCD"}
            table[f"s{i + 1}"] = ev
            log(f"after stage {i + 1} ({st}): " + "  ".join(f"{es} {ev[es]['ALL'][0]} (held {ev[es]['ALL'][2]})" for es in ev)
                + f" | online {train_res[f's{i + 1}']} | {agent.summary()}")
        drift = None
        if drift_n:
            base_p = probe(agent, "C", evaln, seed, mech, wkw)
            ag2 = agent.copy_to('X'); ag2.learning = True
            wf = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0, **wkw)
            wf.run(ag2, drift_n)
            RJ.mech_add(mech, evaluate(wf.log))
            after = probe(ag2, "C", evaln, seed, mech, wkw)
            drift = dict(stage="C", before=base_p, after=after)
            log(f"drift C: {base_p['ALL'][0]} -> {after['ALL'][0]} after {drift_n} untutored dialogues")
        for m in (mech, mech_train):
            if 'lat' in m:
                m['lat'] = sorted(set(m['lat']))
        viol = sum(mech.get(k, 0) + mech_train.get(k, 0) for k in RJ.MECH_KEYS)
        s = agent.summary()
        rec = dict(tag=tag, seed=seed, R=3, cur="mix", stages=stages, nA=nA, nS=nS, evaln=evaln, drift_n=drift_n,
                   agent={}, cfg=dict(port='torch-sweep', **params), world=wkw, table=table, train=train_res, drift=drift,
                   mech_probe=mech, mech_train=mech_train, viol=viol, mix=s['mix'], hip=s['hip'],
                   secs=round(time.time() - t0, 1))
        with open(os.path.join(outdir, f"{tag}_s{seed}.json"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log(f"done viol={viol}")
    except Exception as e:                                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        hub.err = e
    finally:
        with hub.cv:
            hub.alive -= 1
            hub.cv.notify_all()


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    P = dict(nA=int(A.get('nA', 300)), nS=int(A.get('nS', 1000)), evaln=int(A.get('evaln', 80)), drift=int(A.get('drift', 300)))
    outdir = A.get('out', os.path.join(HERE, 'out', 'sweep'))
    os.makedirs(outdir, exist_ok=True)
    jobs = []
    for line in open(A['jobs']):
        p = line.split()
        if not p or p[0].startswith('#'):
            continue
        jobs.append((p[0], int(p[1]), {k: float(v) for k, v in (x.split('=') for x in p[2:])}))
    torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
    cfg = Cfg(U=1 << int(A.get('logU', 17)))
    wkw = dict(eye_layout='retino')
    inits, births, worlds = [], [], []
    for tag, seed, prm in jobs:
        wA = World("A", seed=seed, tutor="recast", **wkw)
        init, ear, F0 = birth_init(wA, seed, cfg)
        for _ in range(3):
            inits.append(init)
        births.append((ear, F0)); worlds.append(wA)
    br = Brain(len(inits), inits, cfg, 'cuda')
    for i, (tag, seed, prm) in enumerate(jobs):
        for k, v in prm.items():
            assert k in PER_AGENT, k
            br.prm[k][3 * i:3 * i + 3] = v
    hub = Hub(br)
    hub.alive = len(jobs)
    t0 = time.time()
    th = []
    for i, (tag, seed, prm) in enumerate(jobs):
        rows = dict(main=3 * i, P=3 * i + 1, X=3 * i + 2)
        ag = Agent(hub, rows['main'], births[i][0], births[i][1], seed, rows)
        th.append(threading.Thread(target=protocol, args=(hub, ag, tag, seed, prm, P, outdir, worlds[i]), daemon=True))
    for t in th:
        t.start()
    hub.loop()
    for t in th:
        t.join()
    dt = time.time() - t0
    print('SWEEP', json.dumps(dict(protocols=len(jobs), rows=br.B, batched_ticks=hub.ticks, secs=round(dt, 1),
                                   brain_secs=round(hub.t_step, 1), ms_per_batched_tick=round(1000 * hub.t_step / max(hub.ticks, 1), 2),
                                   gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20), error=str(hub.err))), flush=True)


if __name__ == "__main__":
    main()
