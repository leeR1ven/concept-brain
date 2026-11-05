"""Label-free recognition (the coordinator's caveat): can an unsupervised competitive stage, learned online from the
same hearings with no syllable labels, carry the identify-then-lookup gain? Winner-take-all prototypes with a Hebbian
instar update (w_k += eta (x - w_k) for the winner only) and a conscience term (a unit that wins too often is
handicapped by its own win rate; local), trained on the training hearings in random order for a few passes. Then the
association is Hebbian counting from the winner (one-hot, so counting = least squares): each training hearing's
winner accumulates the follower distribution of the syllable it came from, as the stream would deliver it. Test: a
fresh hearing -> its winner -> the winner's commonest follower; is it the true cue's commonest follower? Also the
winner's purity (does a winner stand for one syllable). numpy only.
usage: python3 proto.py multi.npz [layer=1] [K=600] [passes=3] [n_ex=18]"""
import json
import sys
import time

import numpy as np


def unpack(a, n):
    return np.unpackbits(a, axis=-1)[..., :n].astype(np.float32)


def train_protos(X, K, passes, eta=0.1, conscience=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    W = Xn[rng.choice(len(Xn), K, replace=False)].copy()        # prototypes start as random hearings
    wins = np.zeros(K)
    t = 0
    for _ in range(passes):
        for i in rng.permutation(len(Xn)):
            t += 1
            Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
            s = Wn @ Xn[i] - conscience * (wins / max(t, 1) - 1.0 / K) * K / 10.0
            k = int(np.argmax(s))
            W[k] += eta * (Xn[i] - W[k])
            wins[k] += 1
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)


def main():
    t0 = time.time()
    z = np.load(sys.argv[1])
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    passes = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    n_ex = int(sys.argv[5]) if len(sys.argv) > 5 else 18
    n1 = int(z['n1'])
    N = z['N']
    S = N.shape[0]
    piA = N.sum(1) / N.sum()
    best = N.argmax(1)
    tr = unpack(z['train'][:, :, :, layer], n1)                 # [S, V, R, n]
    V, R = tr.shape[1], tr.shape[2]
    sel = [(v, k) for v in range(V) for k in range(R)][:n_ex] if n_ex >= V else [(0, k) for k in range(n_ex)]
    sel = [(v, k) for k in range(R) for v in range(V)][:n_ex]
    X = np.stack([tr[:, v, k] for v, k in sel], 1).reshape(-1, n1)
    lab = np.repeat(np.arange(S), len(sel))
    W = train_protos(X, K, passes)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    win = (Xn @ W.T).argmax(1)
    # Hebbian association from the winner: the follower distribution the stream delivers after each hearing
    M = np.zeros((K, S))
    np.add.at(M, win, N[lab] / np.maximum(N.sum(1), 1)[lab][:, None])
    # purity: the share of each winner's hearings that come from its commonest syllable
    cnt = np.zeros((K, S))
    np.add.at(cnt, (win, lab), 1.0)
    used = cnt.sum(1) > 0
    purity = float((cnt.max(1)[used] / cnt.sum(1)[used] * cnt.sum(1)[used]).sum() / cnt.sum())
    maj = cnt.argmax(1)
    out = dict(layer=['copies', 'last layer'][layer], K=K, passes=passes, n_ex=len(sel), winners_used=int(used.sum()),
               purity=round(purity, 3), unigram_top1=round(float((piA * (N.sum(0).argmax() == best)).sum()), 3))
    te = unpack(z['test'][:, :, :, layer], n1).reshape(S, -1, n1)
    vo = unpack(z['voice'][:, :, layer], n1)
    for tname, T in (('fresh', te), ('voice4', vo)):
        Tq = T.reshape(-1, n1)
        Tq = Tq / np.maximum(np.linalg.norm(Tq, axis=1, keepdims=True), 1e-9)
        c = np.repeat(np.arange(S), T.shape[1])
        w = piA[c] / piA[c].sum()
        k = (Tq @ W.T).argmax(1)
        ident = maj[k] == c                                      # the winner's majority syllable is the cue
        look = M[k].argmax(1) == best[c]
        out[tname] = dict(identified_via_winner_weighted=round(float((w * ident).sum()), 3),
                          identified_via_winner_unweighted=round(float(ident.mean()), 3),
                          lookup_top1_weighted=round(float((w * look).sum()), 3),
                          lookup_top1_unweighted=round(float(look.mean()), 3))
    out['secs'] = round(time.time() - t0)
    print('PROTO', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
