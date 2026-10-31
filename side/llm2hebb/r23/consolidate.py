# -*- coding: utf-8 -*-
"""Test 2: replay consolidation of the episodic store into the slow weights (Qwen2.5-0.5B), CLS style.

Facts arrive in 'days' of DAY new facts.  The episodic store keeps every fact from its first hearing (test 1); in the
quiet phase after each day the store replays facts into the network (back-prop, Adam, all decoder layers; embeddings and
tied head frozen), as minibatches of
   new facts (today's)  +  rehearsal of older stored facts  +  ordinary text (interleaving)
Conditions: cls (all three), notext (no ordinary text), noreh (no rehearsal of older facts), newonly (only today's).
Test: the network ALONE (store removed): today's facts, the first 100, all facts so far; paraphrase and QA forms;
neighbour subjects (do they now get an invented name?); corpus perplexity; the 9 world-knowledge probes.
usage: python consolidate.py cond=cls lr=1e-5 nf=2000 day=100 steps=150
"""
import os, sys, json, time, math, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F
import common as C
import facts as FA

A = dict(a.split('=', 1) for a in sys.argv[1:])
OUT = A.get('out', os.path.join(HERE, 'out', 'consolidate.jsonl'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
torch.manual_seed(0)
T0 = time.time()
COND, LR = A.get('cond', 'cls'), float(A.get('lr', 1e-5))
NF, DAY, STEPS = int(A.get('nf', 2000)), int(A.get('day', 100)), int(A.get('steps', 150))
BF, BT = int(A.get('bf', 16)), int(A.get('bt', 16))
tok, model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)
LAYERS = model.model.layers
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0


def log(d):
    d['t'] = round(time.time() - T0, 1)
    print(json.dumps(d, ensure_ascii=False), flush=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')


def pad(seqs):
    L = max(len(s) for s in seqs)
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return X.to(C.DEV), att.to(C.DEV)


@torch.no_grad()
def eval_facts(FF, form='rewrite', bs=256):
    if not FF:
        return None
    ok = []
    for b in range(0, len(FF), bs):
        S = [f['p'][form] + [f['nt'][0]] for f in FF[b:b + bs]]
        X, att = pad(S)
        h = model.model(input_ids=X, attention_mask=att).last_hidden_state
        ln = torch.tensor([len(s) for s in S], device=C.DEV)
        ix = torch.arange(len(S), device=C.DEV)
        z1 = model.lm_head(h[ix, ln - 2]).argmax(-1)
        z2 = model.lm_head(h[ix, ln - 1]).argmax(-1)
        t = torch.tensor([f['nt'] for f in FF[b:b + bs]], device=C.DEV)
        ok.append(((z1 == t[:, 0]) & (z2 == t[:, 1])).float())
    return round(float(torch.cat(ok).mean()), 4)


@torch.no_grad()
def next_argmax(seqs, bs=256):
    out = []
    for b in range(0, len(seqs), bs):
        S = seqs[b:b + bs]
        X, att = pad(S)
        h = model.model(input_ids=X, attention_mask=att).last_hidden_state
        ln = torch.tensor([len(s) for s in S], device=C.DEV)
        out.append(model.lm_head(h[torch.arange(len(S), device=C.DEV), ln - 1]).argmax(-1))
    return torch.cat(out)


dialogs = C.load_dialogs()
probes = C.Probes(tok, model, dialogs[-400:], n=150)
NAME_TOK = set(t for c in C.NAME_CH for t in C.enc(tok, c))
neigh = FA.make(tok, 300, seed=4242)
neigh_seqs = [f['p']['rewrite'] for f in neigh]
neigh0 = next_argmax(neigh_seqs)
facts = FA.make(tok, NF, seed=1, exclude=set(f['subj'] for f in neigh))       # same facts as R3 / test 1
text = [C.enc(tok, ''.join(C.dialog_lines(d)))[:48] for d in dialogs[:20000]]
text = [t for t in text if len(t) >= 8]
rng = random.Random(1)

params = [p for l in LAYERS for p in l.parameters()]
for p in params:
    p.requires_grad_(True)
opt = torch.optim.Adam(params, lr=LR)


def fact_loss(FF):
    S = [f['p']['rewrite'] + f['nt'] for f in FF]
    X, att = pad(S)
    logits = model(input_ids=X, attention_mask=att).logits
    ln = torch.tensor([len(f['p']['rewrite']) for f in FF], device=C.DEV)
    ix = torch.arange(len(FF), device=C.DEV)
    z = torch.stack([logits[ix, ln - 1], logits[ix, ln]], 1).float()          # (B, 2, V)
    t = torch.tensor([f['nt'] for f in FF], device=C.DEV)
    return F.cross_entropy(z.flatten(0, 1), t.flatten())


def text_loss(seqs):
    X, att = pad(seqs)
    logits = model(input_ids=X, attention_mask=att).logits[:, :-1].float()
    m = att[:, 1:].bool()
    return F.cross_entropy(logits[m], X[:, 1:][m])


def evaluate(n, extra):
    model.eval()
    new = facts[n - DAY:n]
    a = next_argmax(neigh_seqs)
    log(dict(mode='consolidate', cond=COND, lr=LR, steps=STEPS, n=n, new_rewrite=eval_facts(new),
             new_para=eval_facts(new, 'para'), new_qa=eval_facts(new, 'qa'), first100=eval_facts(facts[:100]),
             all=eval_facts(facts[:n]), neigh_same=round(float((a == neigh0).float().mean()), 4),
             neigh_bleed=round(float(np.mean([int(t) in NAME_TOK for t in a.tolist()])), 4), **probes.all(model),
             **extra))


log(dict(info='start', args=A, base=probes.all(model)))
tt, nsteps = 0.0, 0
for d in range(NF // DAY):
    new, old = facts[d * DAY:(d + 1) * DAY], facts[:d * DAY]
    t = time.time()
    model.train()
    for s in range(STEPS):
        if COND in ('cls', 'notext') and old:
            fb = rng.sample(new, BF // 2) + [old[rng.randrange(len(old))] for _ in range(BF // 2)]
        else:
            fb = rng.sample(new, BF)
        loss = fact_loss(fb)
        if COND in ('cls', 'noreh'):
            loss = loss + text_loss([text[rng.randrange(len(text))] for _ in range(BT)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        nsteps += 1
    torch.cuda.synchronize()
    tt += time.time() - t
    n = (d + 1) * DAY
    if n in (100, 200, 500, 1000, 2000, 5000, 10000):
        evaluate(n, dict(sec_per_day=round(tt / (d + 1), 1), ms_per_step=round(1000 * tt / nsteps, 1),
                         replays_per_new_fact=round(STEPS * (BF // 2 if (COND in ('cls', 'notext') and d) else BF) / DAY, 1),
                         gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)))
log(dict(info='end'))
