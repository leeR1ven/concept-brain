"""A child of two brains (ARCH 五, evolution: random birth -> test -> keep the best -> breed). Every number of the child
is 40 % the father's, 40 % the mother's, 20 % their mean; link strengths are then changed a little at random; the child is born with the
merged links (learned ones too), still has to learn, but should learn faster. Values (the reward settings, MARIO_*)
are innate and are not merged. The parents must come from the same newborn wiring (the same birth seed), so the same
number means the same link in both; where whole-number tables (wiring) differ anyway, each entry is the father's or the
mother's. The child starts with nothing lit.
usage: python mario_breed.py <father.pt> <mother.pt> <child.pt> [seed] [mutation, relative sd, default 0.02]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_brain                                  # noqa: E402,F401  (pickles)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.net import Pop                           # noqa: E402


def is_links(o):
    return hasattr(o, '__dict__') and torch.is_tensor(getattr(o, 'w', None)) and torch.is_tensor(getattr(o, 'idx', None))


def mix(f, m, g, sd):
    """40 % father, 40 % mother, 20 % the mean; then each changed by a relative sd"""
    m = m.to(f.device)
    r = torch.rand(f.shape, generator=g).to(f.device)
    out = torch.where(r < 0.4, f, torch.where(r < 0.8, m, 0.5 * (f + m)))
    if sd > 0:
        out = out * (1.0 + sd * torch.randn(f.shape, generator=g).to(f.device))
    return out


def merge(child, mother, g, sd, seen, stats, depth=0):
    """walk the child (a copy of the father) and the mother side by side; merge every tensor in place"""
    if depth > 14 or id(child) in seen:
        return
    seen.add(id(child))
    if isinstance(child, Pop):                     # activity is not inherited: the child starts with nothing lit
        child.on.zero_(); child.prev.zero_(); child.age.fill_(99); child.age_prev.fill_(99)
        return
    if is_links(child) and is_links(mother) and child.w.shape == mother.w.shape and child.idx.shape == mother.idx.shape:
        same = (child.idx == mother.idx.to(child.idx.device))
        w = mix(child.w, mother.w, g, sd if getattr(child, 'learned', False) else 0.0)   # learned links only change at
        # random (birth wiring sums whole numbers against whole-number thresholds); where the parents' wiring differs, the link
        child.w.copy_(torch.where(same, w, child.w))    # (target and strength together) is the father's
        seen.add(id(child.w)); seen.add(id(child.idx))
        stats['numbers'] += child.w.numel()
        stats['wiring differs'] += int((~same).sum())
        for k, v in vars(child).items():
            if k not in ('w', 'idx'):
                merge(v, getattr(mother, k, None), g, sd, seen, stats, depth + 1)
        return
    if torch.is_tensor(child):
        if not torch.is_tensor(mother) or mother.shape != child.shape or mother.dtype != child.dtype:
            stats['skipped'] += 1
            return
        child_dev = child.device
        m = mother.to(child_dev)
        r = torch.rand(child.shape, generator=g).to(child_dev)
        if child.is_floating_point():                 # other numbers (thresholds, usual levels, rates): merged, not
            stats['numbers'] += child.numel()          # changed at random (a threshold on a whole number moved by 2 %
            stats['differ'] += int((child != m).sum())  # lets units fire with one input fewer: nearly all lit)
            child.copy_(mix(child, m, g, 0.0))
        else:
            d = child != m
            if d.any():
                child.copy_(torch.where(r < 0.5, child, m))
                stats['wiring differs'] += int(d.sum())
        return
    if isinstance(child, (list, tuple)):
        if isinstance(mother, (list, tuple)) and len(mother) == len(child):
            for a, b in zip(child, mother):
                merge(a, b, g, sd, seen, stats, depth + 1)
        return
    if isinstance(child, dict):
        if isinstance(mother, dict):
            for k, v in child.items():
                if k in mother:
                    merge(v, mother[k], g, sd, seen, stats, depth + 1)
        return
    if hasattr(child, '__dict__') and not isinstance(child, type):
        for k, v in vars(child).items():
            merge(v, getattr(mother, k, None), g, sd, seen, stats, depth + 1)


def main():
    fa, mo, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    sd = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
    child = torch.load(fa, weights_only=False, map_location='cpu')     # on the CPU: two big brains need not fit the GPU
    mother = torch.load(mo, weights_only=False, map_location='cpu')
    g = torch.Generator()
    g.manual_seed(seed)
    stats = {'numbers': 0, 'differ': 0, 'wiring differs': 0, 'skipped': 0}
    merge(child, mother, g, sd, set(), stats)
    child.reset()
    torch.save(child, out)
    print(f'child of {os.path.basename(fa)} x {os.path.basename(mo)} (seed {seed}, mutation sd {sd}): '
          f'{stats["numbers"]} numbers merged ({stats["differ"]} differed between the parents), wiring entries that '
          f'differed {stats["wiring differs"]}, tensors skipped {stats["skipped"]} -> {out}', flush=True)


if __name__ == '__main__':
    main()
