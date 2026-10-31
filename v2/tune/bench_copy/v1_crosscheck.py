# -*- coding: utf-8 -*-
"""Cross-check of the bench against ARCH_v1's integrated toy (archv1_copy.py = arch/synth/archv1.py):
 (a) archv1.Talker with run.py's protocol (listen, teach, PAUSE 5; free replies with the 45/60/12 rule) on corpus pairs
 (b) the bench World (script mode, force) driving the SAME archv1.Brain as an agent        -> world/timing equivalence
 (c) the bench World driving RefV1 (body.py + T + P->Y + Hc->Y/END + echo slot)              -> body.py equivalence
 Tick-by-tick mouth output must be identical; then the bench's mechanics metrics are compared with run.py's audit.
usage: python3 v1_crosscheck.py [n_dialogues] [n_probes] [cfg mem|base] [seed]"""
import collections, copy, json, sys, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import archv1_copy as A
from world import World
from ref_v1 import RefV1, CFGS
from metrics import evaluate
import physics

ND = int(sys.argv[1]) if len(sys.argv) > 1 else 30
NPR = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CFG = sys.argv[3] if len(sys.argv) > 3 else 'mem'
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

pool, chars = A.load_corpus()
pairs = A.pairs_of(pool[:ND])
probes = [p for p in A.pairs_of(pool[-500:])][:NPR]


def v1_cfg():
    c = A.Cfg(seed=SEED)
    k = CFGS[CFG]
    for name in ('cue_w', 'theta', 'end_rule', 'rho', 'Ub', 'rep', 'eta_py_event', 'b_py', 'wref_py', 'eta_py2_event',
                 'b_py2', 'wref_py2', 'g_py2', 'eta_cy', 'b_cy', 'wref_cy', 'eta_ce', 'wref_ce', 'g_dec'):
        setattr(c, name, k[name])
    if k['slot']:
        from echo_slot_orig import EchoSlot          # pristine copy of arch/synth/echo_slot.py for archv1 itself
        c.slot = EchoSlot(c.NP, **k['slot'])
    return c


# ---------------- (a) archv1 Talker, run.py protocol
t0 = time.time()
tk = A.Talker(chars, v1_cfg()); tk.warmup()
tk_b = copy.deepcopy(tk)                       # the same warmed-up brain for (b)
trace_a = []
audit = collections.Counter()
def runlens(evs):
    out, cur = [], None
    for e in evs:
        if e['said'] is None:
            if cur is not None: out.append(cur); cur = None
        elif e['onset']:
            if cur is not None: out.append(cur)
            cur = 1
        else: cur = (cur or 0) + 1
    if cur is not None: out.append(cur)
    return out
for q, r in pairs:
    el = tk.listen(q); et = tk.teach(r); ep = tk.silence(5)
    for e in el + et + ep: trace_a.append(e['said'])
    audit['B1'] += sum(e['said'] is not None for e in el)
    audit['guided_exact'] += A.said_text(et + ep) == r
    rl = runlens(et + ep); audit['syl'] += len(rl); audit['not3'] += sum(x != 3 for x in rl)
    audit['garble'] += sum(e['said'] == '~' for e in et + ep)
free_a = []
for q, r in probes:
    el = tk.listen(q); ef = tk.free(n_after_stop=12, max_ticks=60)
    if len(ef) < 45: ef += tk.silence(45 - len(ef))
    for e in el + ef: trace_a.append(e['said'])
    mv = [k for k, e in enumerate(ef) if e['said'] is not None]
    stop = next((e['stop'] for e in ef if 'stop' in e), None)
    free_a.append(dict(said=A.said_text(ef), first=mv[0] if mv else None, stop=stop, B1=sum(e['said'] is not None for e in el),
                       runs=runlens(ef), garble=sum(e['said'] == '~' for e in ef),
                       restarts=sum(1 for k, e in enumerate(ef) if e['GO'] and any('stop' in x for x in ef[:k]))))
print(f'(a) archv1 Talker: {len(trace_a)} ticks, {time.time()-t0:.0f}s', flush=True)


class ArchV1Agent:
    def __init__(self, talker):
        self.b = talker.b
        self.tick_event = None
    def step(self, obs):
        win = obs['ear']; cue = obs['cue']
        cs, ep = None, False
        if cue is not None:
            nz = np.flatnonzero(cue > 0)
            if len(nz): cs = frozenset(int(m) for m in nz)
            else: ep = True
        ev = self.b.step(self.b.ear.encode(win), float(win.sum()), cue=cs, endpush=ep)
        self.tick_event = ev.get('stop') if ev.get('stop') in ('END', 'floor', 'breath') else None
        return ev['force']


def items():
    return [dict(prompt=q, reply=r, kind='force') for q, r in pairs] + \
           [dict(prompt=q, reply=r, kind='probe', task='never', held_out=True) for q, r in probes]


class Rec:
    """records the world's decoded mouth output per tick by wrapping the agent"""
    def __init__(self, agent, world):
        self.a, self.w, self.trace = agent, world, []
    def step(self, obs):
        m = self.a.step(obs)
        act = tuple(np.flatnonzero(np.asarray(m) > 0).tolist())
        self.trace.append(self.w.voice.decode(act))
        self.tick_event = getattr(self.a, 'tick_event', None)
        return m


def bench_run(agent):
    w = World('B', seed=SEED, tutor='force', lead_in=0, gap_dialogue=0, scene_lead=0)
    rec = Rec(agent, w)
    log = w.run_script(rec, items(), stage='E')
    return w, log, rec.trace


t0 = time.time()
wb, log_b, trace_b = bench_run(ArchV1Agent(tk_b))
print(f'(b) bench world + archv1.Brain: {len(trace_b)} ticks, {time.time()-t0:.0f}s', flush=True)
t0 = time.time()
w0 = World('B', seed=SEED)
ref = RefV1(w0, cfg=CFG, seed=SEED)
wc, log_c, trace_c = bench_run(ref)
print(f'(c) bench world + RefV1: {len(trace_c)} ticks, {time.time()-t0:.1f}s', flush=True)

def first_diff(x, y):
    for i, (a, b) in enumerate(zip(x, y)):
        if a != b: return i
    return None if len(x) == len(y) else min(len(x), len(y))
out = dict(cfg=CFG, seed=SEED, pairs=len(pairs), probes=len(probes), ticks=(len(trace_a), len(trace_b), len(trace_c)),
           a_vs_b_first_diff=first_diff(trace_a, trace_b), b_vs_c_first_diff=first_diff(trace_b, trace_c),
           mouth_ticks=sum(x is not None for x in trace_a))
# mechanics: bench evaluate() vs the run.py-style audit of (a)
for name, log in (('b', log_b), ('c', log_c)):
    ev = evaluate(log)['E']
    out[f'bench_{name}_mech'] = ev['mech']; out[f'bench_{name}_guided'] = ev['guided']
    out[f'bench_{name}_free'] = [(r['reply'], r['latency'], r['agent_stop']) for r in log if r['kind'] == 'probe']
out['v1_audit_train'] = dict(audit)
out['v1_free'] = [(x['said'], x['first'], x['stop']) for x in free_a]
out['v1_free_mech'] = dict(B1=sum(x['B1'] for x in free_a), syl=sum(len(x['runs']) for x in free_a),
                           not3=sum(sum(r != 3 for r in x['runs']) for x in free_a), garble=sum(x['garble'] for x in free_a),
                           restarts=sum(x['restarts'] for x in free_a),
                           stops=dict(collections.Counter(x['stop'] for x in free_a)),
                           first_move=dict(collections.Counter(x['first'] for x in free_a)))
out['identical_free_replies_a_b_c'] = [x['said'] for x in free_a] == [r[0] for r in out['bench_b_free']] == [r[0] for r in out['bench_c_free']]
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
