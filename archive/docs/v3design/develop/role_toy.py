"""Symbolic toy: which learning-by-listening rule learns answers from which caregiver behaviour, and which one drifts
when nobody gives feedback?  No acoustics; lines are (speaker, content).  Pure python, < 50 MB.

Learners (what a heard other-voice line teaches as the answer to question q):
  held  : ARCH_v2 - the first other-voice line after q (own lines skipped) is learned as q's answer; 'moving on'
          (a different line) also weakens the agent's own last answer (predicted but absent).
  ral   : ARCH_v2.1 role-aware listening - as held, but the step is cancelled if the agent speaks within W after that
          line, unless the line carries a reward (praise).  Take-up (repeating a correction) therefore cancels it.
  adj   : developmental / generic - every line is a prediction target of the line immediately before it, whoever said
          it; own lines are predicted by the efference copy and teach nothing.  No role rule, no timing window.
          So q teaches only when the very next line after q is another voice's (formats, replayed pairs).
Caregiver correction forms (after a wrong answer):  plain 'a' (today's bench/life/world2), pair 'q? a' (the
caregiver replays the adjacency pair), neg '不是X 是a' (marked contrast, still not adjacent to q).
format p_model: the caregiver asks and answers herself at once (Ninio & Bruner 1978 formats; bench 'model' mode).
"""
import random, collections, sys

NQ, NA = 40, 12

def run(learner, form, p_model, seed, n_train=1000, n_untut=900, p_take=0.8, eta=0.2):
    r = random.Random(seed)
    ans = [r.randrange(NA) for _ in range(NQ)]
    W = collections.defaultdict(lambda: collections.Counter())   # W[q][a] answer strengths

    def say(q):
        c = W[q]
        if c and max(c.values()) > 0:
            best = max(c.values()); cands = [a for a, v in c.items() if v == best]
            return r.choice(cands)
        return r.randrange(NA)

    def learn(q, a, sign=1.0):
        c = W[q]
        if sign > 0:                       # delta rule toward target a: w_s += eta*(1[s==a] - w_s)
            for k in list(c):
                c[k] -= eta * c[k]
            c[a] += eta
        else:                              # predicted but absent: weaken a
            c[a] -= eta * c[a]

    def acc():
        return sum(say_frozen(q) == ans[q] for q in range(NQ)) / NQ

    def say_frozen(q):
        c = W[q]
        if c and max(c.values()) > 0:
            return max(c.items(), key=lambda kv: (kv[1], str(kv[0])))[0]
        return -1

    def exchange(tutored):
        q = r.randrange(NQ); a = ans[q]
        if tutored and r.random() < p_model:
            # format: caregiver asks and answers herself; the child may echo the answer
            # lines: (cg,q) (cg,a) [(me,a) take-up]
            learn(q, a)            # every learner: first other line after q is a, and it is adjacent to q
            return
        x = say(q)
        nxt_q = r.randrange(NQ)
        if not tutored:
            # untutored: (cg,q) (me,x) (cg,next question)
            if learner == 'held':
                learn(q, ('Q', nxt_q))          # the next question is learned as q's answer
                learn(q, x, -0.5)                # moving on = predicted-but-absent weakens x
            elif learner == 'ral':
                pass                             # the next question gets my answer -> cancelled
            elif learner == 'adj':
                pass                             # (me,x) is adjacent to q: own line, no target
            return
        if x == a:
            # praise line '对' carries a reward; the agent often speaks after it
            if learner in ('held', 'ral'):
                learn(q, x)                      # feedback committed (rewarded)
            elif learner == 'adj':
                pass                             # after q came my own x: nothing; '对' follows my x (value only)
            return
        # wrong answer -> correction line(s)
        take = r.random() < p_take
        if form == 'plain':
            if learner == 'held':
                learn(q, a)
            elif learner == 'ral':
                if not take:
                    learn(q, a)                  # cancelled when I speak right after it (no reward on it)
            elif learner == 'adj':
                pass                             # 'a' follows my x, not q
        elif form == 'pair':
            if learner == 'held':
                learn(q, ('Q', q)); learn(q, x, -0.5)   # the first other line after q is the replayed q itself
                learn(q, a)                              # then a follows the replayed q (held thought still q)
            elif learner == 'ral':
                learn(q, a)                              # the replayed q is answered by the cg herself: no cancel
            elif learner == 'adj':
                learn(q, a)                              # (cg,q)(cg,a) adjacent: learned with no rule
        elif form == 'neg':
            if learner == 'held':
                learn(q, ('NEG',)); learn(q, x, -0.5)
            elif learner == 'ral':
                if not take:
                    learn(q, ('NEG',))
            elif learner == 'adj':
                pass                             # '不是X 是a' follows my x
        return

    out = []
    for n in (250, 750):
        for _ in range(n):
            exchange(True)
        out.append(acc())
    for n in (300, 600, 2100):
        for _ in range(n):
            exchange(False)
        out.append(acc())
    return out

def main():
    rows = []
    for p_model in (0.0, 0.3):
        for form in ('plain', 'pair', 'neg'):
            for learner in ('held', 'ral', 'adj'):
                res = [run(learner, form, p_model, s) for s in range(5)]
                m = [sum(x[i] for x in res) / 5 for i in range(5)]
                print(f'p_model {p_model:.1f}  correction {form:5s}  learner {learner:4s}  tutored 250/1000: {m[0]:.2f} {m[1]:.2f}   untutored +300/+900/+3000: {m[2]:.2f} {m[3]:.2f} {m[4]:.2f}')
main()
