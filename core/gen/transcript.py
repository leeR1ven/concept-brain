"""What the child actually says (designer 11-13: "你要看他实际究竟说了什么"). Never-heard dialogues are played as in
life (two adults, gaps between turns), the child's mouth on and its own voice heard back, learning off; the transcript
shows each adult turn and every syllable the child says, placed where it said it (toned pinyin).
usage: python -m core.gen.transcript <ckpt> [n=8] [start=0]"""
import sys

import numpy as np
import torch


def main(ckpt, n=8, start=0):
    import core.corpus_life  # noqa: F401
    from ..net import DEV
    from ..world import phon
    from ..world.corpus import script_dialogue
    life = torch.load(ckpt, weights_only=False)['life']
    b, c = life.brain, life.c
    bases = phon.load_inventory()
    nm = lambda s: phon.toned_name(bases, int(s))
    b.learning = False
    print(f'{ckpt.split("/")[-1]}: heard {life.env.n_ex} dialogues', flush=True)
    for di in range(start, start + n):
        src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9311, di]),
                              tuple(c.get('gap_turn', (4, 10))), 8, 1, 1)
        snap = b.snapshot()
        own, rows, cur, who, voices = None, [], [], None, {}
        for x in src:
            a0 = life.coch.a0(([] if x is None else [x[:5]]) + ([] if own is None else [(own, -1, 'own', 0.0, 0.0)]))
            y = life.mouth.say(b.step(torch.as_tensor(life.with_sight(a0), device=DEV), 0.0).cpu().numpy())
            own = y if (y is not None and y != 'garble') else None
            if x is not None:
                v = x[2]
                if v != who and cur:
                    rows.append((who, cur))
                    cur = []
                who = v
                cur.append(nm(x[0]))
            elif cur:
                rows.append((who, cur))
                cur = []
            if y is not None:
                said = '(含混)' if y == 'garble' else nm(y)
                if cur:                                   # it spoke while someone was talking
                    cur.append(f'[孩子: {said}]')
                else:
                    rows.append(('child', [said]))
        if cur:
            rows.append((who, cur))
        b.restore(snap)
        print(f'\n--- 对话 {di} ---')
        last = None
        for who_, words in rows:
            if who_ == 'child':
                if last == 'child':
                    print(' ' + words[0], end='')
                else:
                    print('\n    孩子: ' + words[0], end='')
            else:
                name = voices.setdefault(who_, '甲乙丙丁'[len(voices) % 4])
                print(f'\n{name}: ' + ' '.join(words), end='')
            last = who_
        print(flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], n=int(kw.get('n', 8)), start=int(kw.get('start', 0)))
