"""What the corpus asks of the readout (numpy only, no brain). Within turns: how well does the previous syllable
(bigram), the previous two (trigram) or nothing (unigram) predict the next one, over the S commonest syllables?
Also the "A then B" pairs the designer's test would use: for the commonest cue A, the share of its best follower.
usage: python3 corpus_stats.py <train.npz> [S=600]"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np


def main():
    z = np.load(sys.argv[1])
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    s, to = z['syl'].astype(np.int64), z['turn_off']
    f = np.bincount(s)
    syl = np.argsort(-f)[:S]
    ix = -np.ones(int(s.max()) + 1, np.int64)
    ix[syl] = np.arange(S)
    r = ix[s]
    keep = np.ones(len(s) - 1, bool)
    st = to[(to > 0) & (to < len(s))]
    keep[st - 1] = False
    a, b = r[:-1][keep], r[1:][keep]
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    n = len(a)
    half = n // 2                                         # train on the first half, test on the second half
    N = np.zeros((S, S))
    np.add.at(N, (a[:half], b[:half]), 1.0)
    uni = N.sum(0).argmax()
    big = N.argmax(1)
    acc_uni = float((b[half:] == uni).mean())
    acc_big = float((big[a[half:]] == b[half:]).mean())
    acc_big_train = float((big[a[:half]] == b[:half]).mean())
    # trigram: previous two syllables
    keep2 = keep[:-1] & keep[1:]
    a2, a1, b2 = r[:-2][keep2], r[1:-1][keep2], r[2:][keep2]
    ok2 = (a2 >= 0) & (a1 >= 0) & (b2 >= 0)
    a2, a1, b2 = a2[ok2], a1[ok2], b2[ok2]
    h2 = len(a2) // 2
    tri = defaultdict(Counter)
    for x, y, w in zip(a2[:h2], a1[:h2], b2[:h2]):
        tri[(x, y)][w] += 1
    hit = back = 0
    for x, y, w in zip(a2[h2:], a1[h2:], b2[h2:]):
        c = tri.get((x, y))
        if c:
            hit += c.most_common(1)[0][0] == w
        else:
            back += big[y] == w
    acc_tri = (hit + back) / (len(a2) - h2)
    # entropy of the next syllable given the previous one (bits), bigram, on the first half
    P = N / np.maximum(N.sum(1, keepdims=True), 1)
    pa = N.sum(1) / N.sum()
    H = float(-(pa * (P * np.log2(np.maximum(P, 1e-12))).sum(1)).sum())
    # share of the best follower per cue, weighted by how often the cue occurs
    share = N.max(1) / np.maximum(N.sum(1), 1)
    top = np.argsort(-pa)[:20]
    pairs = [(int(syl[c]), int(syl[big[c]]), round(float(share[c]), 2)) for c in top]
    print(json.dumps(dict(S=S, covered_pairs=int(ok.sum()), of_all=round(float(ok.mean()), 3),
                          top1_unigram=round(acc_uni, 3), top1_bigram=round(acc_big, 3),
                          top1_bigram_train=round(acc_big_train, 3), top1_trigram=round(acc_tri, 3),
                          H_next_given_prev_bits=round(H, 2), best_follower_share_weighted=round(float((pa * share).sum()), 3),
                          best_follower_share_median=round(float(np.median(share)), 3),
                          pairs_20_commonest_cues=pairs)))


if __name__ == '__main__':
    main()
