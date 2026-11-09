"""Mario with the PFC learning inside, the architecture as it is: one brain watches the training runs one after another
(eye on Mario), learning on. The last layer -> first layer links (each copy's own check, ARCH 四) learn what comes next,
the screen and the buttons alike, because the buttons are copies too; L4 <-> L4 links learn by the timing rule; the
one decay runs. No readout: what the brain would press is the push its last layer gives the button copies, press
minus stop (pfc.e1), the same push that lights them as thoughts.
  life  python mario_life.py life <net.pt> <cells dir> <out.pt> <frames>     (checkpoint every 50 runs and at the end)
  eval  python mario_life.py eval <life.pt> <cells dir> <rec-free: runs comma list or NEW or SEEN>   learning off;
        per button: does the push rank the frames where it goes down next (press), within 10 frames (ahead), comes up
        next (release); AUC 0.5 = no idea"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_brain                                  # noqa: E402,F401  (pickles)
from game_brain import UNITS_PER_KEY as U          # noqa: E402
from game_brain import DEV                        # noqa: E402
from mario_build import NEW, SEEN_TRY, episodes, load   # noqa: E402
from mario_learn import NAMES, auc                 # noqa: E402

torch.set_num_threads(int(os.environ.get('MARIO_THREADS', 4)))


def key_push(b):
    """press minus stop push of each button from the last layer, this tick"""
    p, nv = b.pfc, b.ch.n_vision
    e = p.e1[nv:]
    nk = e.shape[0] // (2 * U)
    e = e.reshape(2 * nk, U).mean(1)
    return (e[:nk] - e[nk:]).cpu().numpy()


def next_push(b):
    """the push the last layer, as it is now, gives the button copies at the next tick (press minus stop): what the
    brain expects itself to do next. After a frame this aims at the buttons chosen on that frame."""
    p, nv = b.pfc, b.ch.n_vision
    a4 = p.L4.on
    d = p.loop.drive(a4) + p.L41.drive(a4)
    if p.G1P is not None:
        d = d + p.G1P.drive(a4)
    e = (d - p.usual)[nv:]
    nk = e.shape[0] // (2 * U)
    e = e.reshape(2 * nk, U).mean(1)
    return (e[:nk] - e[nk:]).cpu().numpy()


def life(net, src, out, frames):
    b = torch.load(net, weights_only=False, map_location=DEV)
    p = b.pfc
    if os.environ.get('MARIO_WIDE_KEYS') == '1':        # the button copies hear the whole last layer (KeyLinks)
        game_brain.keys_hear_all(b)
    if os.environ.get('MARIO_CHECK_STEP'):              # try another step of the copies' own check
        b.c['check_step'] = float(os.environ['MARIO_CHECK_STEP'])
    print('check step', b.c['check_step'], 'margin', b.c['check_margin'], flush=True)
    eps = [n for n, lv in episodes(src) if lv not in NEW]
    held_out = {}
    for n, lv in episodes(src):
        if lv in SEEN_TRY and lv not in held_out:
            held_out[lv] = n
    eps = [n for n in eps if n not in held_out.values()]
    eps = [eps[i] for i in np.random.default_rng(1).permutation(len(eps))]
    t0, frames_seen, every = time.time(), 0, b.c['decay_every']
    n = {'ticks': 0, 'thoughts': 0}

    def tick():                                         # learning and the one decay go tick by tick
        p.learn()
        n['ticks'] += 1
        n['thoughts'] += int(p.x_int.sum())
        if n['ticks'] % every == 0:
            for tb in p.plastic():
                tb.decay()
    print('ticks per frame', game_brain.TPF, flush=True)
    for i, name in enumerate(eps):
        cells, act = load(src, name)
        b.reset()
        keys = np.zeros(len(b.keys_of(0)), bool)
        n['thoughts'] = 0
        for t in range(len(act)):
            b.step(cells[t], keys, tick)
            keys = b.keys_of(act[t])
        frames_seen += len(act)
        ticks = frames_seen
        thoughts = n['thoughts'] / game_brain.TPF
        w = p.L41.w
        print(f'run {i + 1} {name}: {len(act)} frames, total {ticks}, {ticks / (time.time() - t0):.1f} frames/s; thoughts per'
              f' tick {thoughts / len(act):.0f}; L4->L1 links: mean {float(w.mean()):.4f} max {float(w.max()):.3f}'
              f' nonzero {float((w > 0).float().mean()):.3f}' + ('' if not hasattr(p.L41, 'keys') else
              f'; to buttons mean {float(p.L41.keys.w.mean()):.4f} max {float(p.L41.keys.w.max()):.3f}'), flush=True)
        if (i + 1) % 50 == 0 or ticks >= frames or i == len(eps) - 1:
            b.trained = eps[:i + 1]
            torch.save(b, out)
        if ticks >= frames:
            break


def evaluate(path, src, which):
    b = torch.load(path, weights_only=False, map_location=DEV)
    p = b.pfc
    p.adapt = False
    eps = episodes(src)
    if which == 'NEW':
        names = [n for n, lv in eps if lv in NEW]
    elif which == 'SEEN':
        held = {}
        for n, lv in eps:
            if lv in SEEN_TRY and lv not in held:
                held[lv] = n
        names = list(held.values())
    else:
        names = which.split(',')
    S, H = [], []
    for name in names:
        cells, act = load(src, name)
        b.reset()
        keys = np.zeros(len(b.keys_of(0)), bool)
        s, h = [], []
        for t in range(len(act)):
            b.step(cells[t], keys)
            s.append(next_push(b))                     # aims at the buttons chosen on frame t
            keys = b.keys_of(act[t])
            h.append(keys)
        S.append(np.array(s)[10:])
        H.append(np.array(h)[10:])
    nk = S[0].shape[1]
    if os.environ.get('MARIO_SAVE'):                    # the pushes and buttons, for looking again without the brain
        np.savez(os.environ['MARIO_SAVE'], s=np.concatenate(S), h=np.concatenate(H), n=[len(x) for x in S])
    print(f'{os.path.basename(path)} on {which}: {sum(len(x) for x in S)} frames', flush=True)
    # S[i] aims at H[i]: press = the button was up at i - 1 and is down at i; the push at i only knew the brain's own
    # last layer, which saw up to about i - 4
    for k in range(nk):
        sc, up, nx, ah = [], [], [], []
        for s, h in zip(S, H):
            hk = h[:, k]
            T = len(hk)
            a = np.zeros(T - 1, bool)
            for d in range(1, 11):
                a[:T - d] |= hk[d:] & ~hk[:T - d]
            sc.append(s[1:T, k])
            up.append(~hk[:T - 1])
            nx.append(hk[1:])
            ah.append(a)
        sc, up, nx, ah = (np.concatenate(v) for v in (sc, up, nx, ah))
        print(f'   {NAMES[k]:7s} held {100 * nx.mean():4.1f}% | press AUC {auc(sc[up], nx[up]):.3f}  ahead AUC'
              f' {auc(sc[up], ah[up]):.3f}  release AUC {auc(-sc[~up], ~nx[~up]):.3f}', flush=True)
    # timing: the same press AUC with the target moved by d frames (d < 0: the press already happened that long ago)
    for k in range(nk):
        row = []
        for d in range(-4, 3):
            sc, up, nx = [], [], []
            for s, h in zip(S, H):
                hk = h[:, k]
                T = len(hk)
                i = np.arange(max(1, 1 - d), min(T, T - d))
                sc.append(s[i, k])
                up.append(~hk[i + d - 1])
                nx.append(hk[i + d])
            sc, up, nx = (np.concatenate(v) for v in (sc, up, nx))
            row.append(f'{d:+d}:{auc(sc[up], nx[up]):.2f}')
        print(f'   {NAMES[k]:7s} press AUC by target offset  ' + '  '.join(row), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'life':
        life(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4])
