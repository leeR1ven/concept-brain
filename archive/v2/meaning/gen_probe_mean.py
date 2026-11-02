# -*- coding: utf-8 -*-
"""Class-dependent generalisation probe (review2/learning gen_probe2.py, v21 RAL harness) for the meaning layer.

Same symbolic harness as v21/regress/gen_probe2.py: the ARCH_v2 cognitive core (continuous thought, recency, awake
recall, 10k hippocampus, floor 0.3) with v2.1's role-aware listening (ral=1 ral_other=1 by default here) hears
prompts, speaks through core.speak(), gets the recast tutor (reward, then the correct reply is heard) and recalls in a
quiet moment after every exchange.  Frames (4) x fillers (6 foods, 6 toys):
  offer  你要X吗 -> 要X (all)      eat  你吃X吗 -> 吃X (foods)      play  你玩X吗 -> 玩X (toys)
  taste  X好不好吃 -> 好吃 (food) / 不能吃 (toy)   held out: taste for 2 foods + 2 toys (never heard: heldstream=0)
Core variants:
  layer=mix     v2.1's dict MIX (identity keys)                               [the old cores]
  layer=rand    randmix's random layer (rm_* keys; default c_flat as randmix's probe) with sound codes only
  layer=rand + mean=<mode>   the random layer reading sound + meaning codes (meanlayer.MeanCodes; mean_* keys)
Measures (frozen deep copy, learning off, at the end):
  held   the 4 held taste pairs x 3 replies (the class test);  seen  3 seen fillers per frame x 3 replies
  fault  every seen (frame, filler) pair once, one prompt position replaced by a random other vocabulary item
         (acceptable reply unchanged, as randmix post.py); class-sub: the taste filler replaced by another SEEN filler
         of the same / the other class (correct = the original's class answer)
  codes  within / between class Jaccard of the fillers' meaning codes, NN class purity, mean pairwise Jaccard over all
         items (collapse index), code drift between 2/3 of the life and the end
  lesion the same frozen tests with the meaning half removed (off) and with codes permuted between items (shuffle)
usage: python3 gen_probe_mean.py layer=rand mean=ri seed=0 [n=900] [mean_km=16 ...] [rm_alloc=flat ...] [tag=x]"""
import os, sys, json, time, collections, copy
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _d in ('synth', 'randmix', 'v21'):   # repo: the unchanged ARCH_v2, randmix and v2.1 code
    if os.path.join(HERE, '..', _d) not in sys.path:
        sys.path.insert(1, os.path.join(HERE, '..', _d))
import numpy as np                                   # noqa: E402
from core_v21 import RoleCore, RCfg                  # noqa: E402
from mcore import MeanCore                           # noqa: E402
from meanlayer import code_stats, code_drift         # noqa: E402

A = dict(a.split('=', 1) for a in sys.argv[1:])
LAYER = A.pop('layer', 'rand')
MEAN = A.pop('mean', 'none')
N_DLG = int(A.pop('n', 900))
SEED = int(A.pop('seed', 0))
HELDSTREAM = int(A.pop('heldstream', 0))
TAG = A.pop('tag', '')
LATE = float(A.pop('late', 0))   # late=f: the held fillers are not heard at all before f x n dialogues (new words)
W2 = int(A.pop('w2', 0))    # w2=1: two-syllable fillers as in the bench (小 shared by 3 toys)
OUT = A.pop('out', os.path.join(HERE, 'out', 'gp_mean.jsonl'))


def num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


RM = dict(train=1, prune=1, age=2000, regrow=1, alloc='flat')           # randmix's probe settings for c (flat)
MK = {}
for k in list(A):
    if k.startswith('rm_'):
        RM[k[3:]] = num(A.pop(k))
    elif k.startswith('mean_'):
        MK[k[5:]] = num(A.pop(k))
RAL = dict(ral=1, ral_other=1)
for k in list(A):
    if k in RCfg.RAL:
        RAL[k] = num(A.pop(k))
CFGX = {k: num(v) for k, v in A.items()}

FRAME_CH = list('你要吗玩好不吃能')
if W2:
    FOODS = ['苹果', '香蕉', '西瓜', '饼干', '牛奶', '葡萄']
    TOYS = ['皮球', '汽车', '小狗', '小猫', '小鸟', '鸭子']
else:
    FOODS = ['苹', '蕉', '瓜', '饼', '奶', '葡']
    TOYS = ['球', '车', '狗', '猫', '鸟', '鸭']
FILL = FOODS + TOYS
VOC = FRAME_CH + sorted(set(''.join(FILL)) - set(FRAME_CH), key=lambda c: ''.join(FILL).index(c))
ID = {c: i for i, c in enumerate(VOC)}
FRAMES = {
    'offer': (lambda x: ['你', '要', *x, '吗'], lambda x: ['要', *x]),
    'eat': (lambda x: ['你', '吃', *x, '吗'], lambda x: ['吃', *x]),
    'play': (lambda x: ['你', '玩', *x, '吗'], lambda x: ['玩', *x]),
    'taste': (lambda x: [*x, '好', '不', '好', '吃'], lambda x: ['好', '吃'] if x in FOODS else ['不', '能', '吃']),
}
ALLOWED = {'offer': FILL, 'eat': FOODS, 'play': TOYS, 'taste': FILL}
FN = list(FRAMES)
rng = np.random.default_rng(SEED)
pf, pt = rng.permutation(FOODS), rng.permutation(TOYS)
HELD = {'offer': set(), 'eat': set(), 'play': set(), 'taste': {str(pf[0]), str(pf[1]), str(pt[0]), str(pt[1])}}
PROMPTS = set(''.join(FRAMES[f][0](x)) for f in FN for x in ALLOWED[f])
CLS = {ID[c]: 0 for w in FOODS for c in w}
CLS.update({ID[c]: 1 for w in TOYS for c in w})
FID = {'food': sorted(set(ID[w[0]] for w in FOODS)), 'toy': sorted(set(ID[w[0]] for w in TOYS))}   # (first) syllables


def make_core():
    kw = dict(V=len(VOC), n_eye=48, seed=SEED, retino=False, ct=True, hip_rho=2.0, ctx=False,
              recall_n=8, eta_recall=0.2, rec_pow=0.0, hip_cap=10000, hip_ndg=6000, hip_kdg=120, floor=0.3,
              think_k=3, ior=0)
    kw.update(RAL)
    kw.update(CFGX)
    cfg = RCfg(**kw)
    if LAYER == 'mix':
        return RoleCore(cfg)
    mean = None
    if MEAN != 'none':
        mean = dict(mode=MEAN, **MK)
        if MEAN in ('oracle', 'oracle2'):
            mean['cls'] = CLS
    return MeanCore(cfg, RM, mean)


def ids(chars):
    return [ID[c] for c in chars]


def classify(y, ok, prompt):
    if ok:
        return 'correct'
    if y == '':
        return 'silent'
    if y == prompt:
        return 'echo_prompt'
    if y in PROMPTS or any(p.startswith(y) for p in PROMPTS if len(y) >= 2):
        return 'partner_line'
    return 'other'


def frozen_tests(core, frng, units=False):
    fz = copy.deepcopy(core)
    fz.storing = False
    fz.ral_pend = None

    def ask(q):
        fz.hear_line(ids(q), learn=False, store=False)
        out = fz.speak()
        fz.actor_tr = []
        return ''.join(VOC[i] for i in out)

    res = {}
    held, seen = [], []
    for f in FN:
        for x in sorted(HELD[f]) + [y for y in ALLOWED[f] if y not in HELD[f]][:3]:
            q, a = FRAMES[f][0](x), FRAMES[f][1](x)
            for _ in range(3):
                y = ask(q)
                ok = y == ''.join(a)
                (held if x in HELD[f] else seen).append(dict(frame=f, x=x, reply=y, ok=ok, cls=classify(y, ok, ''.join(q))))
    res['held'] = round(float(np.mean([r['ok'] for r in held])), 3)
    res['held_food'] = round(float(np.mean([r['ok'] for r in held if r['x'] in FOODS])), 3)
    res['held_toy'] = round(float(np.mean([r['ok'] for r in held if r['x'] in TOYS])), 3)
    res['seen'] = round(float(np.mean([r['ok'] for r in seen])), 3)
    res['held_cls'] = dict(collections.Counter(r['cls'] for r in held))
    res['held_ex'] = [(r['x'], r['reply']) for r in held][::3]
    # fault: one prompt position -> a random other vocabulary item (acceptable unchanged)
    fl, cl, ftype = [], [], {}
    for f in FN:
        for x in ALLOWED[f]:
            if x in HELD[f]:
                continue
            q, a = FRAMES[f][0](x), FRAMES[f][1](x)
            p = int(frng.integers(len(q)))
            new = q[p]
            while new == q[p]:
                new = VOC[int(frng.integers(len(VOC)))]
            q2 = q[:p] + [new] + q[p + 1:]
            fl.append(ask(q2) == ''.join(a))
            ftype.setdefault(('filler' if q[p] in x else 'frame') + ('>filler' if any(new in w for w in FILL) else '>frame'), []).append(fl[-1])
    res['fault_rand'] = round(float(np.mean(fl)), 3)
    res['fault_by_type'] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(ftype.items())}
    # class substitution in the taste frame: the filler replaced by another seen filler of the same / other class
    for same in (True, False):
        ok = []
        for x in ALLOWED['taste']:
            if x in HELD['taste']:
                continue
            pool = [y for y in ALLOWED['taste'] if y != x and y not in HELD['taste'] and ((y in FOODS) == (x in FOODS)) == same]
            y = pool[int(frng.integers(len(pool)))]
            q = FRAMES['taste'][0](y)
            ok.append(ask(q) == ''.join(FRAMES['taste'][1](x)))
        res['sub_same' if same else 'sub_other'] = round(float(np.mean(ok)), 3)
    seen_all = []
    for f in FN:
        for x in ALLOWED[f]:
            if x not in HELD[f]:
                seen_all.append(ask(FRAMES[f][0](x)) == ''.join(FRAMES[f][1](x)))
    res['seen_all'] = round(float(np.mean(seen_all)), 3)
    res['fault_drop'] = round(res['seen_all'] - res['fault_rand'], 3)
    if units:
        act = {}
        for x in FILL:
            fz.hear_line(ids(FRAMES['taste'][0](x)), learn=False, store=False)
            st = fz.own_begin(None, None)
            _, Au = fz._total_ro(st)
            fz.actor_tr = []
            act[x] = set(int(u) for u in Au)
        sf = [x for x in FOODS if x not in HELD['taste']]
        stt = [x for x in TOYS if x not in HELD['taste']]
        fo = set.intersection(*[act[x] for x in sf]) - set.union(*[act[x] for x in stt])
        to = set.intersection(*[act[x] for x in stt]) - set.union(*[act[x] for x in sf])
        hf = [x for x in FOODS if x in HELD['taste']]
        ht = [x for x in TOYS if x in HELD['taste']]
        res['class_units'] = dict(food_only=len(fo), toy_only=len(to), n_active=int(np.mean([len(v) for v in act.values()])),
                                  held_food_in_food=round(float(np.mean([len(act[x] & fo) / max(1, len(fo)) for x in hf])), 3),
                                  held_food_in_toy=round(float(np.mean([len(act[x] & to) / max(1, len(to)) for x in hf])), 3),
                                  held_toy_in_toy=round(float(np.mean([len(act[x] & to) / max(1, len(to)) for x in ht])), 3),
                                  held_toy_in_food=round(float(np.mean([len(act[x] & fo) / max(1, len(fo)) for x in ht])), 3))
    return res


def main():
    t0 = time.time()
    core = make_core()
    log = []
    t = [0]
    snaps = {}

    def adv(n):
        t[0] += n; core.now = t[0]
        if core.c.ral:
            core.ral_tick()

    def hear(items):
        core.heard_onset(contingent=False)
        st = core.begin_line(None, None)
        for it in items:
            core.line_item(st, it, True)
        adv(3 * len(items))
        core.line_end(st, True, True)
    mc = getattr(core, 'mcodes', None)
    for d in range(N_DLG):
        if mc is not None and d == (2 * N_DLG) // 3:
            snaps['t23'] = mc.snapshot()
        adv(30)
        for _ in range(int(rng.integers(1, 4))):
            f = FN[int(rng.integers(len(FN)))]
            x = ALLOWED[f][int(rng.integers(len(ALLOWED[f])))]
            q, a = FRAMES[f][0](x), FRAMES[f][1](x)
            held = x in HELD[f]
            if held and not HELDSTREAM:
                continue
            if LATE and d < LATE * N_DLG and x in HELD['taste']:
                continue
            kind = 'probe' if (held or rng.random() < 0.15) else 'recast'
            hear(ids(q)); adv(6)
            out = core.speak()
            if out and core.c.ral:
                core.ral_mouth_onset()
            adv(3 * len(out))
            core.own_finish(core.own)
            y = ''.join(VOC[i] for i in out)
            ok = y == ''.join(a)
            log.append(dict(d=d, frame=f, filler=x, held=held, kind=kind, reply=y, ok=ok))
            if kind == 'recast':
                adv(6)
                core.reward(1.0 if ok else 0.0)
                hear(ids(a)); adv(13)
            else:
                core.actor_tr = []
                adv(14)
            core.awake_recall()
    if mc is not None:
        mc.recompute()
    frng = np.random.default_rng(SEED + 99)
    res = dict(tag=TAG, w2=W2, late=LATE, layer=LAYER, mean=MEAN, mk=MK, rm=RM if LAYER != 'mix' else None, ral=RAL, cfgx=CFGX,
               seed=SEED, n_dlg=N_DLG, heldstream=HELDSTREAM)
    res['frozen'] = frozen_tests(core, np.random.default_rng(SEED + 99), units=LAYER != 'mix')
    last = [r for r in log if r['d'] >= 2 * N_DLG // 3 and r['kind'] == 'probe' and not r['held']]
    res['live_seen'] = round(float(np.mean([r['ok'] for r in last])), 3) if last else None
    if mc is not None:
        fid = {'food': [i for i in FID['food'] if i not in FID['toy']], 'toy': [i for i in FID['toy'] if i not in FID['food']]}
        res['codes'] = code_stats(mc.code, fid)
        hid = [ID[c[0]] for c in HELD['taste']]
        # held fillers: overlap with their own class vs the other class (seen members)
        def jac(a, b):
            return len(set(a.tolist()) & set(b.tolist())) / max(1, len(set(a.tolist()) | set(b.tolist())))
        hs = []
        for h in hid:
            if h not in mc.code:
                continue
            own = [jac(mc.code[h], mc.code[j]) for j in (fid['food'] if h in fid['food'] else fid['toy']) if j != h and j in mc.code]
            oth = [jac(mc.code[h], mc.code[j]) for j in (fid['toy'] if h in fid['food'] else fid['food']) if j in mc.code]
            hs.append((float(np.mean(own)), float(np.mean(oth))))
        res['held_overlap_own_other'] = [round(float(np.mean([a for a, _ in hs])), 3), round(float(np.mean([b for _, b in hs])), 3)] if hs else None
        if 't23' in snaps:
            res['code_drift_last_third'] = code_drift(snaps['t23'], mc.snapshot())
        res['mstats'] = mc.stats
        res['frame_codes'] = {VOC[i]: [int(v) for v in c[:6]] for i, c in mc.code.items() if VOC[i] in ('你', '要', '吃', '玩', '好')}
    KEEP = ('held', 'held_food', 'held_toy', 'seen', 'seen_all', 'fault_rand', 'sub_same', 'sub_other', 'held_ex')
    c3 = copy.deepcopy(core)
    c3.c.copy_next = False; c3.c.copy_comp = False
    r3 = frozen_tests(c3, np.random.default_rng(SEED + 99))
    res['lesion_norec'] = {k: r3[k] for k in KEEP}
    if mc is not None:
        hf = [ID[s] for c in FOODS if c in HELD['taste'] for s in c]
        ht = [ID[s] for c in TOYS if c in HELD['taste'] for s in c]
        for les in ('lay_off', 'off', 'shuffle', 'swap'):
            c2 = copy.deepcopy(core)
            if les == 'lay_off':
                c2.mix.set_lesion('off')
            else:
                c2.mcodes.set_lesion(les, pairs=list(zip(hf, ht)))
            r2 = frozen_tests(c2, np.random.default_rng(SEED + 99))
            res['lesion_' + les] = {k: r2[k] for k in KEEP}
    res['units'] = int(core.mix.n)
    if hasattr(core.mix, 'snapshot'):
        res['layer_snap'] = core.mix.snapshot('end')
    res['ral_stats'] = {k: v for k, v in core.ral_stats.items() if v}
    res['secs'] = round(time.time() - t0)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'a') as fo:
        fo.write(json.dumps(res, ensure_ascii=False, default=str) + '\n')


if __name__ == '__main__':
    main()
