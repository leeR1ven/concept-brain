# -*- coding: utf-8 -*-
"""llm2hebb E2 shared parts: model, facts, chatter stream, old-knowledge probes, the local-plasticity add-on.

Everything runs in the LLM's own text world (a test of the principle, not a component of our model).
"""
import os, json, math, time, random
import numpy as np
import torch
import torch.nn.functional as F

MODEL = os.path.expanduser(os.environ.get('LLM2HEBB_MODEL', '~/v2lab/models/Qwen2.5-0.5B-Instruct'))
CORPUS = os.path.expanduser('~/v2lab/data/dialogs.jsonl')
DEV = 'cuda'

# 50 subjects ("my X"); the first 40 are test facts, the last 10 dev facts (used only to choose layer / threshold).
SUBJ = ['我的小狗', '我的小猫', '我的金鱼', '我的乌龟', '我的鹦鹉', '我的仓鼠', '我的兔子', '我的小马', '我的小羊', '我的刺猬',
        '我妹妹', '我弟弟', '我哥哥', '我姐姐', '我表哥', '我表妹', '我堂弟', '我侄子', '我外甥', '我女儿',
        '我的同桌', '我的老师', '我的邻居', '我的室友', '我的教练', '我的医生', '我的老板', '我的同事', '我的网友', '我的笔友',
        '我的布娃娃', '我的机器人', '我的自行车', '我的小汽车', '我的玩具熊', '我的风筝', '我的小船', '我的吉他', '我的电脑', '我的手机',
        '我家的猫头鹰', '我家的小鸡', '我家的鸭子', '我家的小猪', '我家的奶牛', '我们班的班长', '我们队的队长', '我家的保姆',
        '我们的乐队', '我们的球队']
NAME_CH = '豆米朵洛琪卡莎妮嘉蔻菲塔娜柯栗瑶芙茜拉珀蒂霏骁琅珂璃茉蕾翎汐岚曦钰黛璇琥萤筱槿'

# world-knowledge cloze (kept only where the frozen LM's greedy answer is right at the start)
KNOW = [("中国的首都是", "北京"), ("日本的首都是", "东京"), ("法国的首都是", "巴黎"), ("英国的首都是", "伦敦"),
        ("一年有", "十二个月"), ("一个星期有", "七天"), ("一天有", "二十四小时"), ("太阳从", "东方升起"),
        ("水在零度会", "结冰"), ("熊猫最喜欢吃", "竹子"), ("天空是", "蓝色的"), ("草是", "绿色的"), ("一加一等于", "二"),
        ("李白是唐朝的", "诗人"), ("《红楼梦》的作者是", "曹雪芹"), ("世界上最高的山是", "珠穆朗玛峰"), ("春天过后是", "夏天"),
        ("牛奶是", "白色的"), ("香蕉是", "黄色的"), ("地球绕着", "太阳转"), ("鱼生活在", "水里"), ("苹果是一种", "水果"),
        ("医生在", "医院工作"), ("中国最长的河是", "长江"), ("孙悟空的师父是", "唐僧"), ("《西游记》的作者是", "吴承恩"),
        ("秦始皇统一了", "六国"), ("美国的首都是", "华盛顿"), ("一百减一等于", "九十九"), ("冬天过后是", "春天"),
        ("人有两只", "眼睛"), ("北京是中国的", "首都"), ("猫喜欢抓", "老鼠"), ("蜜蜂会采", "花蜜"), ("月亮绕着", "地球转"),
        ("长城在", "中国"), ("三角形有", "三条边"), ("冰是固态的", "水"), ("企鹅生活在", "南极"), ("诸葛亮是", "三国时期")]


def you(s):
    return ('你们' + s[2:]) if s.startswith('我们') else ('你' + s[1:])


def load_model(dtype=torch.float32):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, attn_implementation='sdpa').to(DEV).eval()
    return tok, m


def enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def load_dialogs():
    D = []
    with open(CORPUS, encoding='utf-8') as f:
        for l in f:
            d = json.loads(l)
            d = [x[:40] for x in d if isinstance(x, str) and x.strip()]
            if len(d) >= 2:
                D.append(d)
    return D


def dialog_lines(d):
    return [('甲：' if i % 2 == 0 else '乙：') + x + '\n' for i, x in enumerate(d)]


def forms(f, ctx_line):
    """test prompts for fact f (the name follows the prompt)."""
    s = f['subj']
    return {'cloze': ctx_line + f'甲：{s}叫',
            'qa': f'甲：{s}叫什么名字？\n乙：{you(s)}叫',
            'qa_hard': f'甲：你还记得{s}的名字吗？\n乙：当然记得，是'}


def make_facts(tok, seed, subj, ctx_line='乙：哈哈好的\n'):
    rng = random.Random(seed)
    used, facts = set(), []
    for s in subj:
        for _ in range(500):
            nm = ''.join(rng.sample(NAME_CH, 2))
            if nm in used:
                continue
            nt = enc(tok, nm)
            if len(nt) != 2:
                continue
            ok = True
            line = f'甲：{s}叫{nm}。\n'
            pre = enc(tok, f'甲：{s}叫')
            full = enc(tok, line)
            if full[:len(pre) + 2] != pre + nt:
                ok = False
            for k, p in forms(dict(subj=s), ctx_line).items():
                pi = enc(tok, p)
                if enc(tok, p + nm + '。\n')[:len(pi) + 2] != pi + nt:
                    ok = False
            if ok:
                used.add(nm)
                facts.append(dict(subj=s, name=nm, ntok=nt, line=line))
                break
        else:
            raise RuntimeError('no name for ' + s)
    return facts


def build_stream(facts, dialogs, seed, chat_per=5, reps=3, delay_lines=1000):
    """events: ('line', text, fact_idx|-1) ... ('test', tag) ('quiet',).
    P1: every fact once (random order), each followed by chat_per chatter lines -> test 'p1'.
    P2: reps-1 more rounds -> test 'p3'.  P3: delay_lines chatter lines -> test 'p3d'."""
    rng = random.Random(seed + 1)
    di = [0]
    order = list(range(len(dialogs)))
    rng.shuffle(order)
    buf = []

    def chat(n):
        out = []
        while len(out) < n:
            if not buf:
                buf.extend(dialog_lines(dialogs[order[di[0] % len(order)]]))
                di[0] += 1
            out.append(buf.pop(0))
        return out

    ev = [('test', 'p0')]
    for r in range(reps):
        idx = list(range(len(facts)))
        rng.shuffle(idx)
        for i in idx:
            ev.append(('line', facts[i]['line'], i))
            if r == 0:
                ev.append(('imm', i))
            for c in chat(chat_per):
                ev.append(('line', c, -1))
            ev.append(('quiet',))
        if r == 0:
            ev.append(('test', 'p1'))
    ev.append(('test', 'p3'))
    for j in range(delay_lines):
        ev.append(('line', chat(1)[0], -1))
        if j % chat_per == chat_per - 1:
            ev.append(('quiet',))
    ev.append(('test', 'p3d'))
    return ev


class Probes:
    """old knowledge: next-token accuracy / NLL on held-out corpus dialogs (content tokens only), and world-knowledge
    cloze (first answer token; only prompts the unmodified model gets right)."""

    def __init__(self, tok, model, dialogs, n=150, maxlen=96):
        self.tok = tok
        texts = [''.join(dialog_lines(d)) for d in dialogs[:n]]
        ids = [enc(tok, t)[:maxlen] for t in texts]
        tag = set(enc(tok, '甲：') + enc(tok, '乙：') + enc(tok, '\n') + enc(tok, '。\n'))
        L = max(len(x) for x in ids)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.X = torch.full((len(ids), L), pad, dtype=torch.long)
        self.M = torch.zeros((len(ids), L), dtype=torch.bool)       # position t predicts t+1
        self.att = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            self.X[i, :len(x)] = torch.tensor(x)
            self.att[i, :len(x)] = 1
            for t in range(len(x) - 1):
                if x[t + 1] not in tag:
                    self.M[i, t] = True
        self.X, self.M, self.att = self.X.to(DEV), self.M.to(DEV), self.att.to(DEV)
        self.know = []
        for p, a in KNOW:
            pi, full = enc(tok, p), enc(tok, p + a)
            ai = full[len(pi):]
            if full[:len(pi)] != pi or not ai:
                continue
            with torch.no_grad():
                z = model(torch.tensor([pi], device=DEV)).logits[0, -1]
            if int(z.argmax()) == ai[0]:
                self.know.append((pi, ai))

    @torch.no_grad()
    def corpus(self, model, bs=10):
        acc, nll, n = 0.0, 0.0, 0
        for b in range(0, self.X.shape[0], bs):
            X, A, M = self.X[b:b + bs], self.att[b:b + bs], self.M[b:b + bs]
            h = model.model(input_ids=X, attention_mask=A).last_hidden_state[M]
            tgt = X[:, 1:][M[:, :-1]]
            z = model.lm_head(h).float()
            lp = torch.log_softmax(z, -1)
            acc += float((z.argmax(-1) == tgt).sum())
            nll += float(-lp.gather(1, tgt[:, None]).sum())
            n += int(M.sum())
        return dict(acc=round(acc / n, 4), ppl=round(math.exp(nll / n), 3))

    @torch.no_grad()
    def knowledge(self, model):
        ok = []
        for p, a in self.know:
            z = model(torch.tensor([p], device=DEV)).logits[0, -1]
            ok.append(int(int(z.argmax()) == a[0]))
        return dict(acc=round(float(np.mean(ok)), 4), n=len(ok))

    def all(self, model):
        r = self.corpus(model)
        r['know'] = self.knowledge(model)['acc']
        return r


def gpu_mem():
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
