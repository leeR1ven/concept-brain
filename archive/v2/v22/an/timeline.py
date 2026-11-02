# -*- coding: utf-8 -*-
"""Life-log timeline analysis (diagnosis for step 3): partner lines and own utterances in time order.
For every partner line: its type (question / correction / confirmation / social / statement / reply to my initiative),
the gap since my last utterance ended (if I spoke since the previous partner line), whether I speak within W ticks
after it ends, and whether what I then say repeats it.
usage: python3 an/timeline.py <life_log.pkl> [W=12]"""
import collections, pickle, sys
import numpy as np

QW = set('什么吗哪几谁')


def build(L):
    P, U = [], []          # partner lines: (t0, t1, text, type, reward, who); own utterances: (t0, t1, text, src)
    for r in L:
        ty = r.get('type')
        if ty == 'init':
            U.append((r['t'], r['t_end'], r['text'], 'init'))
            if r.get('appropriate'):
                txt = '你好' if r['category'] == 'greeting' else '对是' + r.get('object', '')
                t0 = r['t_end'] + 2
                P.append((t0, t0 + 3 * len(txt), txt, 'init_reply', r['reward'], 'other'))
            continue
        if ty != 'ex':
            continue
        k = r['kind']
        pt = r['prompt']
        if isinstance(pt, str) and pt:
            if k in ('probe', 'recast'):
                pty = 'question'
            elif k == 'social':
                pty = 'social'
            elif k == 'statement':
                pty = 'statement'
            elif k == 'overheard':
                pty = 'overheard_q'
            else:
                pty = k
            P.append((r['t_prompt_start'], r['t_prompt_end'], pt, pty, 0.0, r.get('prompt_by', 'other')))
        if r.get('t_first_mouth') is not None and r.get('reply'):
            U.append((r['t_first_mouth'], r['t_stop'] or r['t_first_mouth'] + 3, r['reply'], k + '_reply'))
        if k == 'recast':
            txt = r['recast']
            t1 = r['t_recast_end']
            good = r['reward'] > 0
            P.append((t1 - 3 * len(txt), t1, txt, 'confirm' if good else ('correct' if r['reply'] else 'tell_silent'),
                      r['reward'], 'other'))
            if r.get('after_recast_t_first_mouth') is not None and r.get('after_recast_reply'):
                U.append((r['after_recast_t_first_mouth'], r['after_recast_t_stop'] or r['after_recast_t_first_mouth'] + 3,
                          r['after_recast_reply'], 'after_recast'))
        if k == 'overheard' and r.get('model_reply'):
            t0m = r['t_prompt_end'] + 3
            P.append((t0m, t0m + 3 * len(r['model_reply']), r['model_reply'], 'overheard_a', 0.0, 'other'))
        if k == 'social' and r.get('reward', 0) > 0:
            pass                              # the reward is a smile on one silent tick, not a line
    P.sort(); U.sort()
    return P, U


def main():
    path = sys.argv[1]
    W = int(dict(a.split('=') for a in sys.argv[2:] if '=' in a).get('W', 12))
    L = pickle.load(open(path, 'rb'))
    P, U = build(L)
    Ut0 = np.array([u[0] for u in U]); Ut1 = np.array([u[1] for u in U])
    rows = []
    for i, p in enumerate(P):
        t0, t1, txt, ty, rw, who = p
        prev_end = P[i - 1][1] if i else -10 ** 9
        # own utterances between the previous partner line's end and this line's start
        m = (Ut1 <= t0) & (Ut0 >= prev_end)
        gap = int(t0 - Ut1[m].max()) if m.any() else None
        # did I speak within W ticks after it ended (before the next partner line)?
        nxt = P[i + 1][0] if i + 1 < len(P) else 10 ** 12
        j = np.searchsorted(Ut0, t1)
        spoke, rep = False, None
        if j < len(U) and U[j][0] - t1 <= W and U[j][0] < nxt:
            spoke, rep = True, U[j][2]
        echo = bool(rep) and (rep == txt or (len(rep) >= 2 and rep in txt) or (len(txt) >= 2 and txt in rep))
        rows.append(dict(ty=ty, gap=gap, spoke=spoke, rep=rep, echo=echo, txt=txt, rw=rw, t=t0,
                         prev_ty=P[i - 1][3] if i else None, gap_prev=int(t0 - prev_end) if i else None))
    by = collections.defaultdict(list)
    for r in rows:
        by[r['ty']].append(r)
    print(f"{path}: {len(P)} partner lines, {len(U)} own utterances")
    print(f"{'type':12s} {'n':>5s} {'afterOwn':>8s} {'gap med':>7s} {'gap p10-p90':>12s} {'spoke<=W':>8s} {'echo|spk':>8s} "
          f"{'q-syl':>6s}")
    for ty, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        g = [r['gap'] for r in rs if r['gap'] is not None]
        sp = [r for r in rs if r['spoke']]
        print(f"{ty:12s} {len(rs):5d} {len(g) / len(rs):8.2f} {np.median(g) if g else -1:7.0f} "
              f"{(str(int(np.percentile(g, 10))) + '-' + str(int(np.percentile(g, 90)))) if g else '-':>12s} "
              f"{len(sp) / len(rs):8.2f} {np.mean([r['echo'] for r in sp]) if sp else 0:8.2f} "
              f"{np.mean([bool(set(r['txt']) & QW) for r in rs]):6.2f}")
    # what follows what: previous line type -> type, with own speech in between
    print('\nafter-own lines by type: gap histogram (ticks since my utterance ended)')
    edges = [0, 6, 8, 10, 15, 20, 30, 45, 60, 100, 10 ** 9]
    for ty, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        g = [r['gap'] for r in rs if r['gap'] is not None]
        if not g:
            continue
        h = np.histogram(g, edges)[0]
        print(f"  {ty:12s} " + ' '.join(f"[{edges[k]},{edges[k + 1] if edges[k + 1] < 10 ** 8 else 'inf'}):{h[k]}" for k in range(len(h))))
    print('\nwhat I say right after corrections / tell_silent (top 15):')
    for ty in ('correct', 'tell_silent', 'confirm'):
        c = collections.Counter(r['rep'] for r in by.get(ty, []) if r['spoke'])
        print(' ', ty, c.most_common(15))
    # the "second look": lines after which my next utterance repeats them
    print('\nprev type of questions that follow my utterance:', collections.Counter(r['prev_ty'] for r in by.get('question', []) if r['gap'] is not None).most_common())
    T = max(r['t'] for r in rows)
    for ty in ('correct', 'tell_silent', 'confirm', 'question'):
        rs = by.get(ty, [])
        tenth = [np.mean([r['spoke'] for r in rs if k * T / 10 <= r['t'] < (k + 1) * T / 10] or [0]) for k in range(10)]
        print(f"  spoke<=W after {ty} per tenth: " + ' '.join(f"{x:.2f}" for x in tenth))


if __name__ == '__main__':
    main()
