# -*- coding: utf-8 -*-
"""R1: a pretrained RWKV-7 (delta-rule fast-weight state) run as ONE unbroken stream that is never reset.

B streams are run as one batch. Each stream: unrelated Chinese chat text with NF invented-name facts spliced in
('我的小狗叫豆米。\n'); fact j ends at the same token position E_j in every stream (filler is cut to fit), subjects are
shuffled per stream.  At checkpoints the state is forked (cloned) and every fact told so far is queried; the fork is
discarded, so queries never enter the stream.  Recall = exact match of both name tokens (teacher-forced 2nd token =
greedy exact match).  Control: the same streams with the state reset every SESSION tokens (today's LLM usage).
Also: cross-entropy on held-out text read from the forked long-stream state vs from a fresh state.
usage: python r1_stream.py model=<pth> streams=8 length=262144 facts=40 session=4096 tag=..
"""
import os, sys, json, time, math, random
os.environ['RWKV_V7_ON'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F
from rwkv7 import RWKV7

A = dict(a.split('=', 1) for a in sys.argv[1:])
OUT = A.get('out', os.path.join(HERE, 'out', 'r1.jsonl'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
torch.backends.cuda.matmul.allow_tf32 = True          # projections only; the delta-rule state stays fp32
T0 = time.time()
from rwkv.utils import PIPELINE                       # noqa: E402  (tokenizer only)


class _Dummy:
    pass


tokz = PIPELINE(_Dummy(), 'rwkv_vocab_v20230424')
enc, dec = tokz.encode, tokz.decode
M = RWKV7(A['model'])
NS, LEN, NF, SESSION = int(A.get('streams', 8)), int(A.get('length', 262144)), int(A.get('facts', 40)), int(A.get('session', 4096))
TAG = A.get('tag', os.path.basename(A['model']))
SUBJ = ['我的小狗', '我的小猫', '我的金鱼', '我的乌龟', '我的鹦鹉', '我的仓鼠', '我的兔子', '我的小马', '我的小羊', '我的刺猬',
        '我妹妹', '我弟弟', '我哥哥', '我姐姐', '我表哥', '我表妹', '我堂弟', '我侄子', '我外甥', '我女儿',
        '我的同桌', '我的老师', '我的邻居', '我的室友', '我的教练', '我的医生', '我的老板', '我的同事', '我的网友', '我的笔友',
        '我的布娃娃', '我的机器人', '我的自行车', '我的小汽车', '我的玩具熊', '我的风筝', '我的小船', '我的吉他', '我的电脑', '我的手机']
NAME_CH = '豆米朵洛琪卡莎妮嘉蔻菲塔娜柯栗瑶芙茜拉珀蒂霏骁琅珂璃茉蕾翎汐岚曦钰黛璇琥萤筱槿'
QFORMS = {'qa': lambda s: f'\n问：{s}叫什么名字？\n答：你{s[1:]}叫', 'cloze': lambda s: f'\n{s}叫'}


def log(d):
    d['t'] = round(time.time() - T0, 1)
    print(json.dumps(d, ensure_ascii=False), flush=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')


D = []
with open(os.path.expanduser('~/v2lab/data/dialogs.jsonl'), encoding='utf-8') as f:
    for l in f:
        d = json.loads(l)
        if len(d) >= 2:
            D.append('\n'.join(x for x in d if isinstance(x, str)) + '\n\n')
HELD, FILL = D[-1500:], D[:-1500]
held_tok = [t for d in HELD for t in enc(d)]
PPL_P, PPL_L = 24, 256
ppl_x = torch.tensor([held_tok[i * PPL_L:(i + 1) * PPL_L] for i in range(PPL_P)], device='cuda')
HEAD = M.w['head.weight']


def filler_tokens(seed, n):
    r = random.Random(seed)
    out = []
    while len(out) < n:
        out.extend(enc(FILL[r.randrange(len(FILL))]))
    return out[:n]


def make_facts(seed):
    r = random.Random(seed)
    subj = SUBJ[:NF]
    r.shuffle(subj)
    facts, used = [], set()
    for s in subj:
        for _ in range(3000):
            nm = ''.join(r.sample(NAME_CH, 2))
            nt, st, pre = enc(nm), enc(f'{s}叫{nm}。\n'), enc(f'{s}叫')
            if nm in used or len(nt) != 2 or st[:len(pre) + 2] != pre + nt:
                continue
            if not all(enc(q(s) + nm)[:len(enc(q(s))) + 2] == enc(q(s)) + nt for q in QFORMS.values()):
                continue
            used.add(nm)
            facts.append(dict(subj=s, name=nm, nt=nt, st=st, q={k: enc(q(s)) for k, q in QFORMS.items()}))
            break
        else:
            raise RuntimeError('no clean name for ' + s)
    return facts


def make_batch():
    r = random.Random(12345)
    ends = sorted(r.sample(range(512, int(LEN * 0.92)), NF))
    for j in range(1, NF):                          # at least 64 tokens between fact ends
        ends[j] = max(ends[j], ends[j - 1] + 64)
    streams, facts = [], []
    for b in range(NS):
        fs = make_facts(b)
        fill = filler_tokens(b * 7 + 1, LEN + 64 * NF)
        toks, fi = [], 0
        for e, f in zip(ends, fs):
            need = e - len(f['st']) - len(toks)
            toks.extend(fill[fi:fi + need]); fi += need
            toks.extend(f['st'])
            assert len(toks) == e
        toks.extend(fill[fi:fi + LEN - len(toks)])
        streams.append(toks[:LEN])
        facts.append(fs)
    return torch.tensor(streams, device='cuda'), facts, ends


@torch.no_grad()
def query(state, facts, ends, cp, chunk=128):
    """fork every stream's state, ask every fact told so far (both forms). returns list of records."""
    told = [j for j, e in enumerate(ends) if e <= cp]
    items = [(b, j, form) for b in range(NS) for j in told for form in QFORMS]
    rec = []
    for c in range(0, len(items), chunk):
        it = items[c:c + chunk]
        seqs = [facts[b][j]['q'][form] + [facts[b][j]['nt'][0]] for b, j, form in it]
        L = max(len(s) for s in seqs)
        X = torch.zeros(len(it), L, dtype=torch.long, device='cuda')
        for i, s in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s)
        bi = torch.tensor([b for b, _, _ in it], device='cuda')
        st = [[t[bi].clone() for t in lay] for lay in state]
        _, h = M.forward(X, st, return_hidden=True, need_logits=False)
        ln = torch.tensor([len(s) for s in seqs], device='cuda')
        ix = torch.arange(len(it), device='cuda')
        z1 = (h[ix, ln - 2] @ HEAD.T).argmax(-1).tolist()
        z2 = (h[ix, ln - 1] @ HEAD.T).argmax(-1).tolist()
        for i, (b, j, form) in enumerate(it):
            nt = facts[b][j]['nt']
            rec.append((b, j, cp - ends[j], sum(1 for e in ends if ends[j] < e <= cp), form,
                        int(z1[i] == nt[0] and z2[i] == nt[1]), int((cp - 1) // SESSION == (ends[j] - 1) // SESSION)))
    return rec


@torch.no_grad()
def heldout_ce(state, b=0):
    st = M.zero_state(PPL_P) if state is None else [[t[b:b + 1].expand(PPL_P, *t.shape[1:]).clone() for t in lay] for lay in state]
    _, h = M.forward(ppl_x, st, return_hidden=True, need_logits=False)
    ce = torch.stack([F.cross_entropy((h[i, :-1] @ HEAD.T).float(), ppl_x[i, 1:], reduction='none') for i in range(PPL_P)])
    return round(float(ce.mean()), 4), round(float(ce[:, 128:].mean()), 4)


def run(X, facts, ends, reset):
    state = M.zero_state(NS)
    cps = set(range(2048, LEN, 8192))
    for e in ends:
        for dly in (8, 64, 256, 1024, 4096, 16384):
            if e + dly < LEN:
                cps.add(e + dly)
    cps = sorted(cps)
    rec, ce_bins, pos, t_fwd, t_q = [], {}, 0, 0.0, 0.0
    for cp in cps + [LEN]:
        while pos < cp:
            e = min(cp, pos + 4096)
            if reset:
                e = min(e, (pos // SESSION + 1) * SESSION)
                if pos % SESSION == 0 and pos > 0:
                    state = M.zero_state(NS)
            t = time.time()
            _, h = M.forward(X[:, pos:e], state, return_hidden=True, need_logits=False)
            if e - pos > 1:
                ce = sum(F.cross_entropy((h[b, :-1] @ HEAD.T).float(), X[b, pos + 1:e], reduction='sum') for b in range(NS))
                bkey = int(math.log2(max(pos, 1)))
                s0, n0 = ce_bins.get(bkey, (0.0, 0))
                ce_bins[bkey] = (s0 + float(ce), n0 + NS * (e - pos - 1))
            torch.cuda.synchronize()
            t_fwd += time.time() - t
            pos = e
        if cp >= LEN:
            break
        t = time.time()
        rec += query(state, facts, ends, cp)
        t_q += time.time() - t
    return rec, ce_bins, t_fwd, t_q


def summarize(rec, cond):
    bins = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, 10 ** 9)]
    for form in QFORMS:
        row = dict(kind='recall', tag=TAG, cond=cond, form=form)
        for lo, hi in bins:
            v = [r[5] for r in rec if r[4] == form and lo <= r[2] < hi]
            row[f'd{lo}-{hi if hi < 10 ** 9 else "inf"}'] = [round(float(np.mean(v)), 3), len(v)] if v else None
        for lo, hi in [(0, 1), (1, 3), (3, 10), (10, 100)]:
            v = [r[5] for r in rec if r[4] == form and lo <= r[3] < hi and r[2] < 4096]
            row[f'after{lo}-{hi}_d<4096'] = [round(float(np.mean(v)), 3), len(v)] if v else None
        if cond == 'reset':
            for same in (1, 0):
                v = [r[5] for r in rec if r[4] == form and r[6] == same]
                row['same_session' if same else 'other_session'] = [round(float(np.mean(v)), 3), len(v)] if v else None
        log(row)


if __name__ == '__main__':
    log(dict(info='start', args=A, L=M.L, C=M.C, H=M.H))
    X, facts, ends = make_batch()
    log(dict(kind='heldout', tag=TAG, state='fresh', ce=heldout_ce(None)))
    for cond in ('unreset', 'reset'):
        t = time.time()
        rec, bins, tf, tq = run(X, facts, ends, cond == 'reset')
        log(dict(kind='stream', tag=TAG, cond=cond, streams=NS, tokens=LEN, secs=round(time.time() - t, 1),
                 fwd_tokens_per_s=round(NS * LEN / tf), query_secs=round(tq, 1),
                 ce_by_log2pos={b: round(s / c, 4) for b, (s, c) in sorted(bins.items())}))
        summarize(rec, cond)
        with open(OUT.replace('.jsonl', f'_{TAG}_{cond}_rec.json'), 'w') as f:
            json.dump(rec, f)
    # long unreset stream (2 streams): held-out CE read from the forked state after 4^k tokens
    LONG = int(A.get('long', 1 << 20))
    state = M.zero_state(2)
    Y = torch.tensor([filler_tokens(999, LONG), filler_tokens(998, LONG)], device='cuda')
    pos, nxt = 0, 1024
    while pos < LONG:
        e = min(LONG, pos + 4096, nxt)
        M.forward(Y[:, pos:e], state, need_logits=False)
        pos = e
        if pos == nxt:
            log(dict(kind='heldout', tag=TAG, state=f'after{pos}', ce=[heldout_ce(state, 0), heldout_ce(state, 1)],
                     S_norm=round(float(sum(lay[1][0].norm() for lay in state) / len(state)), 2)))
            nxt *= 4
    log(dict(info='end', gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)))
