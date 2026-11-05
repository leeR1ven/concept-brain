"""第 1 期：先亮的指向后亮的。Hook: how do you know the next line of a song? Then the smallest piece: two neurons, one rule."""
import sys

import numpy as np

from stage import GOLD, INK, RED, Camera, Gauge, make

W0, ETA, WREF, THETA = 0.12, 0.40, 0.40, 1.00
APOS, BPOS = (-3.3, 0.0, 0.3), (3.3, 0.5, -0.35)


def grow(w):
    return w + ETA / (1 + w / WREF)


def width(w):
    return 0.018 + 0.085 * w


def crowd(f, n=230):
    """a cloud of dim neurons around the two we will follow, each joined to its nearest neighbours"""
    rng = np.random.default_rng(7)
    P = []
    while len(P) < n:
        p = rng.normal(size=3) * (8.5, 6.5, 5.5)
        if np.linalg.norm(p - APOS) > 2.2 and np.linalg.norm(p - BPOS) > 2.2 and all(np.linalg.norm(p - q) > 1.0 for q in P):
            P.append(p)
    nodes = [f.node(p, r=float(rng.uniform(.2, .32))) for p in P]
    edges = []
    Q = np.array(P)
    for i, p in enumerate(Q):
        for j in np.argsort(np.linalg.norm(Q - p, axis=1))[1:4]:
            if i < j or i not in np.argsort(np.linalg.norm(Q - Q[j], axis=1))[1:4]:
                edges.append(f.edge(nodes[i], nodes[j], w=0.018, alpha=0.5, n=2))
    return nodes, edges, Q


def chain(f, nodes, Q, start, steps, t, dt):
    """a run of neurons lighting one after another along neighbours, the signal visibly passing between them"""
    i, seen = start, {start}
    for k in range(steps):
        nodes[i].fire(t + k * dt, hold=dt * 1.6)
        f.ping(t + k * dt, 330 * 2 ** ((k % 5) * 2 / 12), 0.035)
        near = [j for j in np.argsort(np.linalg.norm(Q - Q[i], axis=1))[1:7] if j not in seen]
        if not near:
            break
        j = int(near[0])
        f.edge(nodes[i], nodes[j], w=0.05, alpha=0.0, n=2).pulse(t + k * dt, dt)
        i = j
        seen.add(i)


def build(f):
    A = f.node(APOS, r=.62, label='A', branches=9, seed=1)
    B = f.node(BPOS, r=.62, label='B', branches=9, seed=2)
    ab = f.edge(A, B, bend=(0.2, -1.1, 0.9), w=width(W0), color=(.5, .62, .78))
    ba = f.edge(B, A, bend=(-0.2, 1.3, -1.6), w=0.05, alpha=0.0, color=RED)
    nodes, edges, Q = crowd(f)
    g = Gauge(W0, THETA, 1.3)
    f.gauges.append(g)
    w = W0

    def dim(t, a, dur=1.8):
        for n in nodes:
            n.alpha.to(t, a, dur)
        for e in edges:
            e.alpha.to(t, a * 0.5, dur)

    def rep(t, w_now):
        f.shoot(A, t, (-13, -3, 2.5))
        A.on(t + .34)
        f.ping(t + .34, 523)
        ab.pulse(t + .36, .62)
        A.off(t + .95)
        f.shoot(B, t + .72, (13, 3, -2.5))
        B.on(t + 1.06)
        f.ping(t + 1.06, 659)
        B.off(t + 1.65)
        nw = grow(w_now)
        ab.w.to(t + 1.12, width(nw), .45)
        g.v.to(t + 1.12, nw, .45)
        return nw

    # hook: the whole cloud, a run of neurons playing out a tune
    f.move(0, 14, az=-34, dist=24)
    l = f.say('你有没有过这种经历？一首歌，前奏刚响，你脑子里已经自己唱起下一句了。')
    chain(f, nodes, Q, 12, 16, l['前奏'], .23)
    chain(f, nodes, Q, 150, 16, l['前奏'] + .5, .23)
    l = f.say('没人提醒你，你也没在心里数拍子。可你就是知道，下面该来什么。')
    chain(f, nodes, Q, 12, 16, l.a + .3, .23)
    chain(f, nodes, Q, 90, 16, l['可你'], .2)
    chain(f, nodes, Q, 200, 16, l.a + 1.2, .22)
    l = f.say('这件事，到底是怎么做到的？', .5)
    t0 = f.t
    f.card(t0, t0 + 3.4, '先亮的，指向后亮的', y=500, size=84, color=GOLD)
    f.card(t0, t0 + 3.4, '第 1 期', y=400, size=40, bold=False)
    f.ping(t0, 440, .08)
    l = f.say('我们这个系列，讲的是一个还在研究中的人工大脑。它到底行不行，现在还不知道，但想法特别有意思。')
    l = f.say('而且我们不从整个大脑讲起。今天，咱们只看最小的一块：两个神经元。', .4)

    # dive into A
    t = l['今天']
    f.move(t, 3.4, az=8, el=7, dist=7.5, target=APOS)
    dim(t + .5, .12)
    A.tag.to(f.t + .4, 1, .4)
    l = f.say('先看这一个。你别被神经元这个名字吓到，在这个模型里，它简单得不能再简单。')
    l = f.say('就两种状态：亮，', .35)
    A.on(l['亮'] + .1)
    f.ping(l['亮'] + .1, 523)
    l = f.say('或者灭。', .4)
    A.off(l.a + .1)
    l = f.say('没有半亮，也没有特别亮，就这两种。')

    # B comes into view
    f.move(f.t, 2.4, az=-8, el=9, dist=12.8, target=(0, .2, 0))
    l = f.say('再来一个，我们叫它 B。')
    B.tag.to(l.a + .9, 1, .4)
    l = f.say('你看它俩中间这根细线，这就是连接。线越粗，连接越强。')
    l = f.say('右上角这个条，显示的就是这根线现在有多强。')
    g.alpha.to(l.a, 1, .5)
    l = f.say('来，我们试一下。让 A 亮起来。', .1)
    f.shoot(A, l['亮起来'] - .34, (-13, -3, 2.5))
    A.on(l['亮起来'])
    f.ping(l['亮起来'], 523)
    l = f.say('信号顺着线跑过去，到了 B 这儿，没反应。')
    ab.pulse(l.a, l['到了'] - l.a + .3)
    A.off(l['到了'])
    f.ring(B, l['没反应'] - .1)
    f.ping(l['没反应'] - .1, 196, .07)
    l = f.say('为什么？因为这根线太细了。B 有个门槛，这点信号推不动它。')
    l = f.say('那问题就来了：怎么才能让 A 点亮 B 呢？')
    l = f.say('答案简单到你可能不信，就一句话。', .4)

    # the rule
    l = f.say('先亮的，指向后亮的。', .4)
    rule_start = l.a
    f.ping(l.a, 440, .08)
    l = f.say('意思是，如果 A 刚亮完，B 紧接着也亮了，那从 A 到 B 这根线，就变粗一点。')
    l = f.say('我们让外面的世界帮个忙。先点亮 A，再点亮 B。', .5)
    w = rep(l['先点亮'] - .3, w)
    l = f.say('看到没？线粗了一点。', .3)
    l = f.say('再来一次。', .1)
    w = rep(l.a + .1, w)
    f.wait(1.5)
    l = f.say('再来。', .1)
    w = rep(l.a + .1, w)
    f.wait(1.6)
    l = f.say('你注意看，每来一次，它就往门槛靠近一点。', .3)
    w = rep(l.a + .5, w)
    l = f.say('最后一次。', .1)
    w = rep(l.a + .1, w)
    f.wait(1.8)
    l = f.say('过线了。好，现在最关键的一步来了。', .4)
    f.card(rule_start, l.a + .5, '先亮的，指向后亮的', y=150, color=GOLD)
    f.move(l.a, 2.5, az=12, el=6, dist=11)

    # the test
    l = f.say('这回，外面只点亮 A，B 我们碰都不碰。', .5)
    t = f.t
    f.shoot(A, t, (-13, -3, 2.5))
    A.on(t + .34)
    f.ping(t + .34, 523)
    f.t = t + .4
    l = f.say('信号跑过去，', .05)
    ab.pulse(l.a, l.b - l.a + .15)
    A.off(l.b)
    l = f.say('B 自己亮了！', .5)
    B.on(l.a)
    f.ping(l.a, 784, .13)
    f.move(l.a, .7, dist=9.8, target=(1.2, .2, 0))
    l = f.say('没有人碰它，它自己亮了。')
    B.off(l.b)
    f.move(f.t, 6, az=-28, el=12, dist=12.5, target=(0, .2, 0))
    l = f.say('这就是记住。A 后面跟着 B 这件事，没有存在什么文件里，它就长在这根线上。')
    f.card(l['这就是记住'], l.b, '记忆，就长在连接上', y=150, color=GOLD)
    l = f.say('这个模型的想法就是，听到前奏想到下一句，靠的就是这样的线，只不过要多得多。', .5)

    # at the same moment
    l = f.say('不过这里有个细节。要是 A 和 B 同时亮呢？是不是也该连起来？', .3)
    f.move(l.a, 2.4, az=6, el=8, dist=12.5)
    l = f.say('这个模型的答案是：不加强。', .3)
    A.on(l.a)
    B.on(l.a)
    f.ping(l.a, 523)
    f.ping(l.a, 659)
    A.off(l.b)
    B.off(l.b)
    f.card(l.a, l.b + .3, '同时亮：不加强', y=150, color=INK)
    l = f.say('为什么这么规定？你想啊，要是同时亮也加强，那从 B 到 A 的线也长出来了。', .2)
    ba.alpha.to(l['也长出来'], 1, .7)
    ba.w.to(l['也长出来'], .1, .9)
    l = f.say('A 点亮 B，B 又点亮 A，A 又点亮 B。', .05)
    l2 = f.say('就这么来回闪，根本停不下来。', .4)
    t, dt = l.a, (l2.b - l.a) / 9
    for k in range(9):
        c, o = (A, B) if k % 2 == 0 else (B, A)
        c.on(t + k * dt, .05)
        o.off(t + k * dt, .05)
        (ab if k % 2 == 0 else ba).pulse(t + k * dt, dt, GOLD if k % 2 == 0 else RED)
        f.ping(t + k * dt, 523 if k % 2 == 0 else 659, .06)
    A.off(l2.b)
    B.off(l2.b)
    l = f.say('所以规则只看先后，不看同时。', .5)
    ba.alpha.to(l.a, 0, .6)
    f.card(l.a, l.b + .3, '只看先后，不看同时', y=150, color=GOLD)

    # recap
    l = f.say('好，今天就记三句话。', .2)
    f.move(l.a, 3, az=14, el=10, dist=16, target=(4.8, 0, 0))
    g.alpha.to(l.a, 0, .5)
    a0 = f.t
    x = 1000
    l = f.say('神经元，只有亮和灭。', .3)
    f.card(l.a, a0 + 13, '1. 神经元只有亮和灭', y=380, size=52, bold=False, x=x, anchor='l')
    l = f.say('A 亮完 B 再亮，A 到 B 的线就变粗。', .3)
    f.card(l.a, a0 + 13, '2. A 亮完 B 再亮，线变粗', y=500, size=52, bold=False, color=GOLD, x=x, anchor='l')
    l = f.say('线够粗了，A 一亮，B 就跟着亮。', .6)
    f.card(l.a, a0 + 13, '3. 线够粗，A 一亮 B 就亮', y=620, size=52, bold=False, x=x, anchor='l')
    l = f.say('不过你可能已经想到了：要是顺序反过来呢？总是 B 先亮，A 后亮，这根线会怎么样？', .3)
    f.t = max(f.t, a0 + 13)
    f.move(f.t - 3, 5, az=-10, el=9, dist=13, target=(0, .2, 0))
    l = f.say('下一期，我们接着看。', .8)
    dim(l.a, .3, 2.5)
    f.black.to(f.t, 1, 1.0)


def cam():
    return Camera(-52, 14, 34, (0, 0, 0))


if __name__ == '__main__':
    make(build, cam, sys.argv[1] if len(sys.argv) > 1 else 'ep01.mp4', preset=sys.argv[2] if len(sys.argv) > 2 else 'veryfast',
         every=int(sys.argv[3]) if len(sys.argv) > 3 else None)
