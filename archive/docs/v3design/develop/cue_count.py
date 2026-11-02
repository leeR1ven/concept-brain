"""Is a question perceivably different from a correction in the bench world's own lines?  Lexical cue only
(question syllables 什么吗哪几谁呢); the worlds have no prosody, gaze or affect channel (PROVED from obs specs)."""
import sys, collections
sys.path.insert(0, 'v2/bench')
from life import Life
from baselines import Silent
QS = set('什么吗哪几谁呢')
L = Life(seed=5, tutor='recast')
lg = L.run(Silent(), ticks=1_000_000)
c = collections.Counter()
for r in lg:
    if r.get('type') != 'ex':
        continue
    k = r.get('kind'); p = r.get('prompt') or ''
    if k in ('probe', 'recast', 'model'):
        c['question'] += 1; c['question_qsyl'] += any(ch in QS for ch in p)
    elif k == 'social':
        c['social_init'] += 1; c['social_qsyl'] += any(ch in QS for ch in p)
    elif k == 'statement':
        c['statement'] += 1; c['statement_qsyl'] += any(ch in QS for ch in p)
    rc = r.get('recast')
    if rc:
        c['recast'] += 1; c['recast_qsyl'] += any(ch in QS for ch in rc)
        c['recast_has_bushi'] += ('不是' in rc and r.get('task') != 'check')
for a, b in [('question', 'question_qsyl'), ('social_init', 'social_qsyl'), ('statement', 'statement_qsyl'), ('recast', 'recast_qsyl')]:
    print(f'{a:12s} n={c[a]:5d}  with a question syllable {c[b]/max(1,c[a]):.3f}')
print('recasts containing an explicit 不是 marker (outside the check task):', c['recast_has_bushi'])
