# FEAT0 full-window cosine between voices of the same char and between different chars (diagnostic).
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from run_lib import DiagWorld, new_agent, lexicon_chars
import physics as PH
w = DiagWorld('B', seed=0, tutor='model'); a = new_agent(w)
v = w.voice
def full(ch, who):
    st = PH.Stream(v.F, v.T, v.T // 3, 3, v.levels, 0.0)
    a.ear.reset(); none = (None, None, False)
    for k in range(3):
        src = (ch, v.hear(ch, who), k == 0)
        srcs = [none, none, none]; srcs[{'other': 0, 'other2': 1, 'self': 2}[who]] = src
        win = st.tick(tuple(srcs)); a.ear.step(win)
    f = a.ear.feat0().astype(float); return f / np.linalg.norm(f)
lex = sorted(lexicon_chars())[:40]
F = {(c, who): full(c, who) for c in lex for who in ('other', 'other2', 'self')}
same = {pair: np.mean([F[(c, pair[0])] @ F[(c, pair[1])] for c in lex]) for pair in (('other', 'other2'), ('other', 'self'), ('other2', 'self'))}
diff = [F[(c, 'other')] @ F[(d, 'other')] for i, c in enumerate(lex) for d in lex[i+1:]]
print('same char, different voice (mean cos):', {f'{p[0]}-{p[1]}': round(x, 3) for p, x in same.items()})
print('different chars, same voice: mean %.3f  p95 %.3f  max %.3f' % (np.mean(diff), np.percentile(diff, 95), np.max(diff)))
mins = {p: np.min([F[(c, p[0])] @ F[(c, p[1])] for c in lex]) for p in same}
print('same char, different voice (min cos):', {f'{p[0]}-{p[1]}': round(x, 3) for p, x in mins.items()})
# mouth: how often does a random 4-set produce a lexicon char?
rng = np.random.default_rng(0); hit = 0; n = 20000; lexall = lexicon_chars(); garble = 0
for _ in range(n):
    s = tuple(sorted(rng.choice(32, 4, replace=False).tolist())); d = v.decode(s)
    hit += d in lexall; garble += d == '~'
print(f'random 4-sets: {hit/n:.4f} produce a lexicon char ({len(lexall)} chars), {garble/n:.3f} garble, vocab {len(v.chars)}')
