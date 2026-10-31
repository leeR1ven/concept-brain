# -*- coding: utf-8 -*-
"""Review2 / neuro: checks computed from EXISTING ARCH_v2 result files only (no agent is run).
usage: python3 -I log_checks.py <synth_dir>   (synth_dir = arch/v2/synth)"""
import bisect, collections, json, os, pickle, sys
import numpy as np

S = sys.argv[1]
LIFE = os.path.join(S, 'out', 'life')


def jac(a, b):
    a, b = set(a or ''), set(b or '')
    return len(a & b) / max(1, len(a | b))


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


L = pickle.load(open(os.path.join(LIFE, 'final_noctx_s5_log.pkl'), 'rb'))
ex = [r for r in L if r.get('type') == 'ex']
T = max(r['t'] for r in L if 't' in r)
print('== 1. favourite-thought base rate (life seed 5, ARCH_v2)')
c = collections.Counter()
for r in ex:
    for k in ('prompt', 'recast', 'model_reply'):
        if isinstance(r.get(k), str):
            c.update(list(r[k]))
tot = sum(c.values()); top = c.most_common(20)
print(f'  partner syllable types {len(c)}, tokens {tot}; top-20 TOKEN share {sum(n for _, n in top) / tot:.3f} '
      f'(ARCH_v2 used the TYPE share 20/{len(c)} = {20 / len(c):.3f}); measured held share 0.81')
print('  top-20:', ''.join(ch for ch, _ in top))

print('== 2. self-initiated speech')
ini = [r for r in L if r.get('type') == 'init']
nh = [r for r in ini if r['text'] == '你好']
print(f'  initiatives {len(ini)}; 你好 {len(nh)}:', dict(collections.Counter((r["partner"], r["category"]) for r in nh)))
def tenth(rs):
    out = [0] * 10
    for r in rs:
        out[min(9, int(10 * r['t'] / T))] += 1
    return out
print('  你好 to an empty room per tenth of life:', tenth([r for r in nh if not r['partner']]))
print('  rewarded 你好 per tenth of life:     ', tenth([r for r in nh if r['reward'] > 0]))
ev = [(r['t'], r['what']) for r in L if r.get('type') == 'event']; ts = [t for t, _ in ev]
cc = collections.Counter(); d = []
for r in nh:
    if r['partner']:
        continue
    i = bisect.bisect_right(ts, r['t']) - 1
    if i >= 0:
        cc[ev[i][1]] += 1; d.append(r['t'] - ev[i][0])
print('  world event just before an empty-room 你好:', cc.most_common(), 'median lag', float(np.median(d)))
for tag in ('final_novgo', 'final_nooth'):
    Lx = pickle.load(open(os.path.join(LIFE, f'{tag}_s5_log.pkl'), 'rb'))
    nx = [r for r in Lx if r.get('type') == 'init' and r['text'] == '你好']
    print(f'  {tag}: 你好 {len(nx)}, empty room {sum(not r["partner"] for r in nx)}, rewarded {sum(r["reward"] > 0 for r in nx)}')

print('== 3. can a heard line be told apart as a reaction to my own reply? (perceivable cues only)')
QW = set('什么吗哪几谁')
corr, conf, nq, qc, qn = [], [], [], [], []
for i, r in enumerate(ex[:-1]):
    if r.get('kind') == 'recast' and r.get('reply') and r.get('recast'):
        s = max(jac(r['reply'], r['recast']), jac(r['prompt'], r['recast']))
        (conf if r.get('reward', 0) > 0 else corr).append(s)
        if r.get('reward', 0) == 0:
            qc.append(bool(set(r['recast']) & QW))
    if r.get('reply') and ex[i + 1].get('dlg') == r.get('dlg') and r.get('kind') in ('recast', 'probe'):
        n = ex[i + 1]['prompt']
        nq.append(max(jac(r['reply'], n), jac(r['prompt'], n))); qn.append(bool(set(n) & QW))
own_conf = [jac(r['reply'], r['recast']) for r in ex if r.get('kind') == 'recast' and r.get('reply') and r.get('recast') and r.get('reward', 0) > 0]
print(f'  J(own reply, confirmation) = {np.mean(own_conf):.2f} (n {len(own_conf)})')
print(f'  continuity = max(J(own,line), J(question,line)): confirmations {np.mean(conf):.2f}, corrections {np.mean(corr):.2f} '
      f'(n {len(corr)}), next partner lines {np.mean(nq):.2f} (n {len(nq)})')
print(f'  AUC vs next lines: confirmations {auc(conf, nq):.3f}, corrections {auc(corr, nq):.3f}')
print(f'  share containing a question syllable: corrections {np.mean(qc):.3f}, next lines {np.mean(qn):.3f}')

print('== 4. reply latency (free replies, life)')
fr = [r for r in ex if r.get('kind') in ('recast', 'probe') and r.get('latency') is not None]
lat = np.array([r['latency'] for r in fr]); acc = np.array([bool(r.get('acc')) for r in fr])
print('  latency histogram', sorted(collections.Counter(lat.tolist()).items()),
      f'acc at 6: {acc[lat == 6].mean():.2f}, acc > 6: {acc[lat > 6].mean():.2f}')

print('== 5. held-out by task, ARCH_v2 seeds 0-4 (final frozen probes)')
agg = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
for s in range(5):
    d = [json.loads(l) for l in open(os.path.join(S, 'out', 'runs', f'final_noctx_s{s}.json'))][-1]
    for st in 'BCD':
        for task, v in d['table']['s3'][st].items():
            if task in ('ALL', 'SCENE', 'NOSCENE', 'one_shot', '_spoke_pos'):
                continue
            a = agg[task]
            if v[0] is not None and v[1]:
                a[0] += v[0] * v[1]; a[1] += v[1]
            if v[2] is not None and v[3]:
                a[2] += v[2] * v[3]; a[3] += v[3]
for task, (s_, n, hs, hn) in sorted(agg.items()):
    print(f'  {task:8s} all {s_ / max(n, 1):.2f} (n {n})  held-out ' + (f'{hs / hn:.2f} (n {hn})' if hn else '-'))

print('== 6. life memory facts by encounters')
J = json.load(open(os.path.join(LIFE, 'final_noctx_s5.json')))
for k in ('fact_name', 'fact_person', 'fact_place', 'novel'):
    print(' ', k, [(b['encounters'], b['n'], b['acc']) for b in J['summary']['retention'][k]['by_encounters']])
print('== 7. replay window: episodes with sigma > 0.05 are replayable; sigma = 0.9995^lines ->',
      int(np.log(0.05) / np.log(0.9995)), 'stored lines')
