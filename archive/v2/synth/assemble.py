# -*- coding: utf-8 -*-
"""Assemble ARCH_v2.md from the section drafts (header/verdict + static design + additions + results + tail)."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
def read(n):
    p = os.path.join(HERE, n)
    return open(p).read() if os.path.exists(p) else ''
static = read('draft_static.md')
# split the static draft into its numbered sections
secs = {}
for m in re.finditer(r'^## (\d+)\. .*?(?=^## \d+\. |\Z)', static, re.S | re.M):
    secs[int(m.group(1))] = m.group(0).rstrip() + '\n'
def renum(txt, old, new):
    return re.sub(r'^## %d\. ' % old, '## %d. ' % new, txt, count=1, flags=re.M)
parts = [read('draft_head.md')]
parts += [secs[i] for i in (1, 2, 3, 4, 5, 6) if i in secs]
parts.append(read('draft_additions.md'))
parts.append(read('draft_results.md'))
parts.append(read('draft_results2.md'))
parts.append(renum(secs.get(8, ''), 8, 9))      # principles
parts.append(renum(secs.get(9, ''), 9, 10))     # GPU cost
parts.append(renum(secs.get(10, ''), 10, 11))   # porting plan
parts.append(renum(secs.get(11, ''), 11, 12))   # open problems
parts.append(read('draft_tail.md'))
parts.append(read('draft_zh.md'))
doc = '\n'.join(p.rstrip() + '\n' for p in parts if p.strip())
open(os.path.join(HERE, 'ARCH_v2.md'), 'w').write(doc)
print(len(doc.splitlines()), 'lines')
