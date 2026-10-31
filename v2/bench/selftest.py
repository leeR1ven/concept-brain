# -*- coding: utf-8 -*-
"""Reproducible checks of the bench: physics, timing, metrics, baselines, body, ARCH_v1 equivalence, speed.

    python3 selftest.py            # ~3-4 min on one core
    python3 selftest.py --long     # + the long ARCH_v1 cross-check (80 dialogues + 40 free replies, ~6 min)
Exit code 0 iff every check passes.  Writes checks/selftest.out.
"""
import collections
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('OMP_NUM_THREADS', '1')

import physics as P                      # noqa: E402
import voice_orig as VO                  # noqa: E402
import voice_v1copy as V1                # noqa: E402
from world import World, LEXICON_CHARS   # noqa: E402
from metrics import evaluate, echo_free, parrot, first_ok   # noqa: E402
from baselines import BASELINES, Silent, run_agent           # noqa: E402
from body import Body                    # noqa: E402

RESULTS = []
OUT_LINES = []


def out(s):
    print(s, flush=True)
    OUT_LINES.append(s)


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    out(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ----------------------------------------------------------------------------------------- physics
def t_physics():
    chars = P.corpus_chars()
    for seed in (0, 1):
        v = P.Voice(chars, seed=seed); o = VO.Voice(chars, seed=seed)
        same = all((v.sound[c] == o.sound[c]).all() for c in chars) and (v.voice['self'] == o.voice['self']).all() \
            and all(v.need[c] == o.need[c] for c in chars)
        rng = np.random.default_rng(5); bad = 0
        for _ in range(3000):
            s = rng.choice(32, rng.integers(1, 9), replace=False)
            f = {f'm{i}': int(i in s) for i in range(32)}
            bad += v.mouth(f) != o.mouth(f)
        check(f'physics: Voice == world/voice.py (seed {seed}, {len(chars)} chars: sounds, self offset, muscle sets)', same)
        check(f'physics: mouth rule == world/voice.py on 3,000 random muscle sets (seed {seed})', bad == 0, f'mismatches {bad}')
    v = P.Voice(chars, seed=0); o = V1.Voice(chars, seed=0)
    s1 = V1.Stream(o, step=2); s2 = P.Stream(8, 6, 2)
    rng = np.random.default_rng(1); bad = 0
    for k in range(4000):
        oc = chars[rng.integers(50)] if rng.random() < 0.5 else None
        sc = chars[rng.integers(50)] if rng.random() < 0.4 else None
        no, ns = bool(rng.random() < 0.3), bool(rng.random() < 0.3)
        w1 = s1.tick(oc, sc, no, ns)
        w2 = s2.tick([(oc, v.hear(oc, 'other') if oc else None, no), (None, None, False),
                      (sc, v.hear(sc, 'self') if sc else None, ns)])
        bad += not (w1 == w2).all()
    check('physics: Stream == ARCH_v1 voice_copy.Stream (4,000 random ticks, other+self, onsets)', bad == 0)
    check('physics: other2 has its own offset', not (v.voice['other2'] == v.voice['self']).all()
          and v.voice['other2'].any())
    # noise
    st0 = P.Stream(8, 6, 2, noise=0.0)
    e0 = sum(st0.tick([(None, None, False)] * 3).sum() for _ in range(50))
    st1 = P.Stream(8, 6, 2, noise=1.0, noise_seed=3); st1b = P.Stream(8, 6, 2, noise=1.0, noise_seed=3)
    w1 = [st1.tick([(None, None, False)] * 3).copy() for _ in range(50)]
    w1b = [st1b.tick([(None, None, False)] * 3).copy() for _ in range(50)]
    check('physics: silence energy exactly 0 without noise; noise=1 gives energy > 0, same seed same noise',
          e0 == 0 and np.mean([w.sum() for w in w1]) > 5 and all((a == b).all() for a, b in zip(w1, w1b)),
          f'mean window energy with noise 1.0: {np.mean([w.sum() for w in w1]):.1f}')
    # lawful
    from physics_check import run as pcheck
    for ph in ('random', 'lawful'):
        r = pcheck(ph, 0)
        ok = r['lex_pairs_over_0_8'] == 0 and r['own_voice_nn_correct'] == 1.0
        if ph == 'lawful':
            ok = ok and r['corr_sound_distance_vs_shared_muscles'] < -0.5
        check(f'physics: {ph}: lexicon chars distinguishable at layer 0 (no pair Jaccard > 0.8), own voice identified',
              ok, f"max J {r['lex_jaccard_max']:.2f}, NN {r['own_voice_nn_correct']:.2f}, corr(sound dist, shared "
                  f"muscles) {r['corr_sound_distance_vs_shared_muscles']:.2f}, J by shared muscles "
                  f"{ {k: round(x, 2) for k, x in r['layer0_jaccard_by_shared_muscles'].items() if x is not None} }")


# ----------------------------------------------------------------------------------------- timing / determinism
def t_timing():
    def sig(log):
        return [(r['task'], r['kind'], r['prompt'], r['reply'], r['t_prompt_start'], r['scene'], r['attend']) for r in log]
    w1, l1 = run_agent(BASELINES['oracle'](), 'BCD', 0, 80, 'model')
    w2, l2 = run_agent(BASELINES['oracle'](), 'BCD', 0, 80, 'model')
    check('timing: a fixed seed gives identical streams and logs (oracle, BCD, 80 dialogues, twice)', sig(l1) == sig(l2))
    w3, l3 = run_agent(Silent(), 'BCD', 0, 80, 'model')
    same_content = [(r['task'], r['prompt'], r['scene'], r['held_out']) for r in l1] == \
        [(r['task'], r['prompt'], r['scene'], r['held_out']) for r in l3]
    check('timing: the content schedule does not depend on the agent (oracle vs silent agent)', same_content)
    pr = [r for r in l1 if r['kind'] == 'probe']
    check('timing: probes: >= 45 ticks after the prompt and >= 13 silent ticks after the stop',
          all(r['t_end'] - r['t_prompt_end'] >= 45 and r['t_end'] - r['t_stop'] >= 13 for r in pr), f'{len(pr)} probes')
    md = [r for r in l1 if r['kind'] == 'model']
    check('timing: model mode: the model reply starts 3 ticks after the prompt (gap_model)',
          all(r['t_model_end'] - r['t_prompt_end'] == 3 + 3 * len(r['model_reply']) for r in md), f'{len(md)} exchanges')
    # force: next line 6 ticks after the stop (5 silent ticks after the stop tick), re-articulation of repeats
    from ref_v1 import RefV1
    wf = World('B', seed=0, tutor='force', lead_in=0, gap_dialogue=0, scene_lead=0)
    a = RefV1(wf, cfg='base', seed=0)
    items = [dict(prompt='你笑什么', reply=r, kind='force') for r in ('哈哈哈', '好好', '对对对', '嗯嗯', '你好')]
    lf = wf.run_script(a, items)
    check('timing: force mode: guided replies exact incl. repeated syllables (哈哈哈 好好 对对对 嗯嗯), next line at stop+6',
          all(r['reply'] == r['cued'] for r in lf) and all(lf[i + 1]['t_prompt_start'] - lf[i]['t_stop'] == 6
                                                          for i in range(len(lf) - 1)),
          str([r['reply'] for r in lf]))
    m = evaluate(l1)
    ov = sum(m[s]['mech']['mouth_while_partner'] for s in m if s != 'ALL')
    runs = [x for r in l1 for x in r.get('runs', [])]
    check('timing: oracle never overlaps a partner, latency 6, every syllable 3 ticks',
          ov == 0 and set(r['latency'] for r in pr) == {6} and set(runs) == {3}, f'{len(runs)} syllables')


# ----------------------------------------------------------------------------------------- metrics
def t_metrics():
    acc = ['红色', '是红色', '红的']
    ok = (echo_free('红色', '苹果是什么颜色', acc) is True and echo_free('苹果是什么颜色', '苹果是什么颜色', acc) is False
          and echo_free('要苹果', '你要苹果吗', ['要苹果', '我要苹果', '要']) is None
          and parrot('苹果在', '苹果在哪里') == 1.0 and parrot('', 'x') is None
          and first_ok('在右边', ['在左边']) and not first_ok('', ['在左边']))
    check('metrics: echo-free / parrot / first-char on hand-made cases', ok)
    recs = []
    for i in range(10):
        good = i < 6
        recs.append(dict(stage='C', task='colour', kind='probe', prompt='苹果是什么颜色', acceptable=acc,
                         reply='红色' if good else '苹果', held_out=i % 2 == 0, needs_scene=True, familiar=True,
                         item=('colour', '苹果'), one_shot=False, runs=[3, 3], garble=0, stop='self', latency=6,
                         mouth_while_partner=0, extra_turns=0))
    r = evaluate(recs)['C']['colour']
    check('metrics: evaluate() on a synthetic log (acc 0.6, held 3/5, echo-free 0.6, parrot (6 x 0.5 + 4 x 1)/10 = 0.7)',
          abs(r['acc'] - 0.6) < 1e-9 and r['held']['n'] == 5 and abs(r['held']['acc'] - 0.6) < 1e-9
          and abs(r['echo_free'] - 0.6) < 1e-9 and abs(r['parrot'] - 0.7) < 1e-9, json.dumps(r, ensure_ascii=False)[:200])


# ----------------------------------------------------------------------------------------- baselines
def t_baselines():
    for stage in ('B', 'C', 'D', 'E'):
        res = {}
        for nm in ('echo', 'lookup', 'oracle'):
            w, log = run_agent(BASELINES[nm](), stage, 0, 400, 'model')
            res[nm] = evaluate(log)[stage]
        e, lk, o = res['echo'], res['lookup'], res['oracle']
        held_ok = (e['ALL']['held']['acc'] or 0) == 0 and (lk['ALL']['held']['acc'] or 0) == 0
        scene_ok = (not e['SCENE'].get('n')) or ((e['SCENE']['acc'] or 0) == 0 and (lk['SCENE']['acc'] or 0) == 0)
        oracle_ok = o['ALL']['acc'] == 1.0 and o['ALL']['first'] == 1.0
        check(f'baselines {stage}: echo and lookup score 0 on held-out items and scene questions; oracle 1.0',
              held_ok and scene_ok and oracle_ok,
              f"held n={e['ALL']['held']['n']}: echo {e['ALL']['held']['acc']}, lookup {lk['ALL']['held']['acc']}; "
              f"scene n={e['SCENE'].get('n')}: echo {e['SCENE'].get('acc')}, lookup {lk['SCENE'].get('acc')}; "
              f"oracle {o['ALL']['acc']} (n={o['ALL']['n']})")


def t_memory_tasks():
    from baselines import DIAGNOSTIC
    w, log = run_agent(DIAGNOSTIC['lastreply'](), 'D', 0, 400, 'model')
    r = evaluate(log)['D']
    w2, log2 = run_agent(BASELINES['prior'](), 'D', 0, 400, 'model')
    r2 = evaluate(log2)['D']
    ok = (r['before']['acc'] or 0) <= 0.15 and (r['name']['acc'] or 0) <= 0.25 and (r2['before']['acc'] or 0) <= 0.2
    check('memory tasks: repeating the last heard reply / the prior does not solve before, name (stage D)', ok,
          f"lastreply: before {r['before']['acc']:.2f} (n={r['before']['n']}), name {r['name']['acc']:.2f}, oneshot "
          f"{r['oneshot']['acc']:.2f}; prior: before {r2['before']['acc']:.2f}, name {r2['name']['acc']:.2f}, "
          f"check {r2['check']['acc']:.2f}")


# ----------------------------------------------------------------------------------------- body: babbling
class BabbleAgent:
    """body.py alone: babbles when idle; keeps a babble the tutor imitated (reward)."""
    def __init__(self, seed=0):
        self.body = Body(babble_rate=0.05, keep='reward', recruit='babble', seed=seed)

    def step(self, obs):
        return self.body.step(float(obs['ear'].sum()), scorer=None, reward=obs['reward'])


def t_body():
    w = World('A', seed=0)
    a = BabbleAgent()
    log = w.run(a, 150)
    b = a.body
    sets = [frozenset(np.flatnonzero(b.prog[y]).tolist()) for y in range(b.n_prog)]
    dec = [w.voice.decode(tuple(sorted(s))) for s in sets]
    imit = sum(len([x for x in r.get('imitations', []) if x[1]]) for r in log if r['kind'] == 'play')
    lex = sum(d in LEXICON_CHARS for d in dec)
    check('body: programs created without any cue from babbling + tutor imitation (stage A)',
          b.n_prog > 0 and all(d not in (None, '~') for d in dec) and b.n_prog <= imit,
          f'{b.n_babbles} babbles, {imit} imitated, {b.n_prog} programs kept (all decode to chars; {lex} in the '
          f'micro-world lexicon)')
    m = evaluate(log)['A']['mech']
    runs = [x for r in log for x in r.get('runs', [])]
    check('body: every complete babble is one 3-tick syllable', set(runs) <= {3} and len(runs) > 0,
          f"{len(runs)} complete syllables, garble {m['garble']}")


def t_options():
    from world import Curriculum
    c = Curriculum(0, eye_layout='retino')
    blocks = all((c.code[(s, v)] // 240 == s).all() for (s, v) in c.code)
    inv = all((c.code[(0, v)] % 240 == c.code[(2, v)] % 240).all() for (s, v) in c.code if s == 0)
    check('options: eye_layout retino = one 240-unit block per slot, the same feature code in every block', blocks and inv)
    w = World('A', seed=0, mouth_vocab='lexicon')
    rng = np.random.default_rng(3)
    dec = [w.voice.decode(tuple(sorted(rng.choice(32, 4, replace=False).tolist()))) for _ in range(3000)]
    v_full = World('A', seed=0).voice
    same_sound = all((w.voice.sound[ch] == v_full.sound[ch]).all() for ch in list(LEXICON_CHARS)[:50])
    check('options: mouth_vocab=lexicon: babbles decode only to lexicon chars or garble; sounds unchanged',
          all(d == '~' or d in LEXICON_CHARS for d in dec) and same_sound,
          f"{sum(d != '~' for d in dec)} of 3000 random sets are lexicon chars")


# ----------------------------------------------------------------------------------------- ARCH_v1 equivalence
def t_v1(long=False):
    args = ['80', '40', 'mem', '0'] if long else ['12', '8', 'mem', '0']
    p = subprocess.run([sys.executable, os.path.join(HERE, 'v1_crosscheck.py')] + args, capture_output=True, text=True,
                       cwd=HERE, timeout=3600)
    txt = p.stdout
    try:
        d = json.loads(txt[txt.index('{'):])
    except Exception:
        check('ARCH_v1 cross-check ran', False, (p.stderr or txt)[-500:])
        return
    ok = d['a_vs_b_first_diff'] is None and d['b_vs_c_first_diff'] is None and d['identical_free_replies_a_b_c']
    check(f"ARCH_v1: archv1.Talker == bench world + archv1.Brain == bench world + RefV1 (body.py), tick by tick "
          f"({d['pairs']} taught pairs, {d['probes']} free replies)", ok,
          f"{d['ticks'][0]} ticks, {d['mouth_ticks']} mouth ticks")
    mb, au, fm = d['bench_c_mech'], d['v1_audit_train'], d['v1_free_mech']
    agree = (mb['mouth_while_partner'] == au['B1'] + fm['B1'] and mb['syllables'] == au['syl'] + fm['syl']
             and mb['not_3_ticks'] == au['not3'] + fm['not3'] and mb['garble'] == au['garble'] + fm['garble']
             and mb['extra_turns'] == fm['restarts'] and mb['agent_stop'] == fm['stops']
             and mb['latency'] == {str(k) if isinstance(k, int) else k: v for k, v in fm['first_move'].items()}
             and d['bench_c_guided']['exact'] == au['guided_exact'] / d['pairs'])
    check('ARCH_v1: bench mechanics metrics == run.py-style audit of the toy (overlap, syllables, not-3, garble, '
          'extra turns, stop reasons, latency, guided exact)', agree,
          f"overlap {mb['mouth_while_partner']}, syllables {mb['syllables']}, not3 {mb['not_3_ticks']}, latency "
          f"{mb['latency']}, stops {mb['agent_stop']}")


# ----------------------------------------------------------------------------------------- life mode
class _Talker:
    """says one fixed char every `every` ticks, regardless of anything (for nonsense / overlap checks)."""
    def __init__(self, ch_set, every=40):
        self.m = np.zeros(32, np.float32); self.m[list(ch_set)] = 1.0
        self.z = np.zeros(32, np.float32); self.every = every; self.k = 0

    def step(self, obs):
        self.k += 1
        return self.m * (1 + (self.k % self.every) % 3) if self.k % self.every < 3 else self.z


class _NightSpy:
    """privileged: records whether eye and ear are all zero during the nights."""
    privileged = True

    def __init__(self):
        self.night = False; self.bad = 0; self.n = 0; self.z = np.zeros(32)

    def bench_scene(self, info):
        self.night = info['night']

    def step(self, obs):
        if self.night:
            self.n += 1
            self.bad += bool(obs['eye'].any() or obs['ear'].any())
        return self.z


def t_life():
    import hashlib
    from life import Life, evaluate_life, RandomNamer, life_baseline
    same = all(open(os.path.join(HERE, f + '.py'), 'rb').read() == open(os.path.join(HERE, 'v1.2', f + '.py'), 'rb').read()
               for f in ('physics', 'world', 'metrics', 'baselines', 'frontend', 'body', 'ref_v1'))
    import bar
    fp = bar.fingerprint()['md5']
    check('life: v1.2 modules unchanged and the v1.2 fingerprint holds', same and fp == 'ab45fe538236076b6954323f8a83f87d', fp)
    t0 = time.time(); L = Life(seed=0); L.run(Silent(), ticks=1000000); dt = time.time() - t0
    check('life: 10^6 ticks of life with a trivial agent in a few minutes', dt < 180,
          f'{L.t} ticks in {dt:.1f} s ({dt / L.t * 1e6:.1f} us/tick), {L.day_i} days, {L.visit_i} visits')

    def sig(log):
        return hashlib.md5(json.dumps([(r.get('type'), r.get('task'), r.get('prompt'), r.get('reply'), r.get('t'),
                                        r.get('t_prompt_start')) for r in log], ensure_ascii=False, default=str).encode()).hexdigest()
    a = Life(seed=2); la = a.run(life_baseline('oracle'), ticks=150000)
    b = Life(seed=2); lb = b.run(life_baseline('oracle'), ticks=150000)
    c = Life(seed=2); lc = c.run(Silent(), ticks=150000)
    content = lambda log: [(r['task'], r['prompt'], r.get('kind')) for r in log if r.get('type') == 'ex']   # noqa: E731
    check('life: fixed seed -> identical life; the content does not depend on the agent (oracle vs silent)',
          sig(la) == sig(lb) and content(la) == content(lc)[:len(content(la))], f'{len(content(la))} exchanges')
    res = {}
    for nm in ('echo', 'lookup', 'oracle'):
        L = Life(seed=1); res[nm] = evaluate_life(L.run(life_baseline(nm), ticks=200000))['groups']
    ok = all((res[n][g]['acc'] or 0) == 0 for n in ('echo', 'lookup') for g in ('scene', 'held_out', 'memory')) and \
        all(res['oracle'][g]['acc'] == 1.0 for g in ('scene', 'held_out', 'memory', 'social'))
    check('life: echo and lookup score 0 on scene (new situations), held-out and memory items; oracle 1.0', ok,
          ', '.join(f"{g} n={res['oracle'][g]['n']}" for g in ('scene', 'held_out', 'memory', 'social')))
    L = Life(seed=0); r = evaluate_life(L.run(RandomNamer(200), ticks=200000))
    ini = [x for x in L.log if x.get('type') == 'init']
    ok = r['initiative']['reward'] > 0 and all(x['reward'] == 0 for x in ini if not x['partner'] or x['night']) and \
        all(x['reward'] > 0 for x in ini if x['category'] == 'naming')
    check('life: naming a visible object while the partner is present earns a reply + reward; alone / night: nothing',
          ok, f"{r['initiative']['categories']}, reward {r['initiative']['reward']:.0f}")
    v = L.voice
    nons = [c for c in v.chars if c not in __import__('world').LEXICON_CHARS][0]
    L2 = Life(seed=0); r2 = evaluate_life(L2.run(_Talker(v.need[nons], 40), ticks=100000))
    cats = r2['initiative']['categories']
    check('life: nonsense gets no reward; talking over the partner is logged as overlap and interruptions',
          r2['initiative']['reward'] == 0 and cats.get('nonsense', 0) > 0 and r2['overlap']['interruptions'] > 0
          and r2['overlap']['overlap_ticks'] > 0, f"{cats}, overlap {r2['overlap']}")
    spy = _NightSpy(); L3 = Life(seed=0); L3.run(spy, ticks=60000)
    check('life: nights are dark and silent (eye and ear all zero)', spy.n > 0 and spy.bad == 0, f'{spy.n} night ticks')
    L4 = Life(seed=3); log4 = L4.run(Silent(), ticks=600000)
    mem = [x for x in log4 if x.get('type') == 'ex' and x['task'] in ('fact_name', 'fact_place', 'novel')]
    d = [x['ticks_since_told'] for x in mem]
    cross = all(L4 is not None and x['visit'] > 0 for x in mem)
    check('life: memory questions come in later visits, hundreds to tens of thousands of ticks after the fact',
          len(d) > 20 and min(d) < 2000 and max(d) > 10000 and cross,
          f'{len(d)} questions, delay min {min(d) if d else None} median {int(np.median(d)) if d else None} max {max(d) if d else None}')
    enc = collections.Counter(min(x['n_experienced'], 5) for x in mem)
    qa = [x for x in log4 if x.get('type') == 'ex' and x.get('acceptable') and 'n_asked_before' in x]
    check('life: facts recur (memory probes after 1, 2, ... 5+ encounters); every probe logs n_experienced, n_asked_before',
          len(enc) >= 4 and enc.get(1, 0) > 0 and enc.get(5, 0) > 0 and all('n_experienced' in x for x in qa),
          f'encounters of memory probes (5 = 5+): {dict(sorted(enc.items()))}')


# ----------------------------------------------------------------------------------------- speed
def t_speed():
    t0 = time.time()
    w, log = run_agent(Silent(), 'BC', 0, 2000, 'model')
    dt = time.time() - t0
    check('speed: a trivial agent runs 2,000 dialogues in < 180 s on one core', dt < 180,
          f'{dt:.1f} s, {w.t} ticks, {dt / w.t * 1e6:.0f} us/tick')


if __name__ == '__main__':
    t0 = time.time()
    long = '--long' in sys.argv
    for f in (t_physics, t_timing, t_metrics, t_baselines, t_memory_tasks, t_body, t_options, t_life, t_speed):
        f()
    t_v1(long)
    n_ok = sum(ok for _, ok in RESULTS)
    out(f'\n{n_ok}/{len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
    os.makedirs(os.path.join(HERE, 'checks'), exist_ok=True)
    with open(os.path.join(HERE, 'checks', 'selftest.out'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(OUT_LINES) + '\n')
    sys.exit(0 if n_ok == len(RESULTS) else 1)
