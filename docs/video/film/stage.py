"""A small 3D film engine for the explainer series: spheres, lines and glow in real perspective with a moving camera,
a voice (edge-tts) that drives the timing, burned-in subtitles, soft sound effects. No modelling, no ray tracing: one CPU
draws a 1080p frame in a fraction of a second, so a whole episode renders in minutes and networks can hold thousands of
neurons. An episode is a function build(film) that places events in time; film.say() advances time by the real speech."""
import asyncio
import glob
import hashlib
import math
import os
import re
import subprocess
import sys
import time
import wave

import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1920, 1080, 30, 24000
FOC = 1.15 * H
GOLD, CYAN, RED, INK, DIM = (1, .74, .32), (.36, .78, .96), (.96, .42, .42), (.94, .96, .98), (.58, .64, .71)
VOICE, RATE = 'zh-CN-YunxiNeural', '+4%'
FONT_R = glob.glob('/usr/share/**/NotoSansCJK-Regular.ttc', recursive=True)[0]
FONT_B = glob.glob('/usr/share/**/NotoSansCJK-Bold.ttc', recursive=True)[0]


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return tuple(x + (y - x) * u for x, y in zip(a, b))


class Track:
    """a value that moves to new targets at given times"""

    def __init__(self, v):
        self.v0, self.keys = v, []

    def to(self, t, v, dur=0.4):
        self.keys.append((t, dur, v))
        self.keys.sort(key=lambda k: k[0])
        return self

    def at(self, t):
        v = self.v0
        for t0, d, x in self.keys:
            if t < t0:
                break
            v = x if d <= 0 or t >= t0 + d else v + (x - v) * ease((t - t0) / d)
        return v


class Camera:
    def __init__(self, az, el, dist, target):
        self.az, self.el, self.dist = Track(az), Track(el), Track(dist)
        self.tx, self.ty, self.tz = Track(target[0]), Track(target[1]), Track(target[2])

    def frame(self, t):
        az = math.radians(self.az.at(t) + 3.0 * math.sin(t * 0.21))
        el = math.radians(self.el.at(t) + 1.6 * math.sin(t * 0.29 + 1.0))
        d = self.dist.at(t)
        T = np.array((self.tx.at(t), self.ty.at(t), self.tz.at(t)))
        C = T + d * np.array((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        f = (T - C) / np.linalg.norm(T - C)
        r = np.cross(f, (0, 0, 1.0))
        r /= np.linalg.norm(r)
        return C, r, np.cross(r, f), f


class View:
    def __init__(self, cam, t):
        self.C, self.r, self.u, self.f = cam.frame(t)
        self.d = cam.dist.at(t)

    def project(self, P):
        V = np.asarray(P, float) - self.C
        z = np.maximum(V @ self.f, 0.05)
        s = FOC / z
        return W / 2 + (V @ self.r) * s, H / 2 - (V @ self.u) * s, z, s

    def fog(self, z):
        return float(np.clip(1.25 - (z - self.d) / 26.0, 0.12, 1.0))


class Node:
    def __init__(self, pos, r, label=None, branches=0, seed=0, alpha=1.0):
        self.pos, self.r, self.label = np.array(pos, float), r, label
        self.lit, self.alpha, self.tag = Track(0.0), Track(alpha), Track(0.0)
        self.color = Track(0.0)
        self.br = []
        rng = np.random.default_rng(seed)
        for _ in range(branches):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            P = [self.pos + d * r * 0.9]
            n = int(rng.integers(5, 8))
            for _ in range(n):
                d = d + rng.normal(size=3) * 0.2
                d /= np.linalg.norm(d)
                P.append(P[-1] + d * r * rng.uniform(0.3, 0.42))
            self.br.append(np.array(P))
            k = n // 2
            e = rng.normal(size=3)
            e = d + 0.9 * e / np.linalg.norm(e)
            e /= np.linalg.norm(e)
            self.br.append(np.array([P[k], P[k] + e * r * .45, P[k] + e * r * .85]))

    def on(self, t, dur=0.08):
        self.lit.to(t, 1.0, dur)

    def off(self, t, dur=0.22):
        self.lit.to(t, 0.0, dur)

    def fire(self, t, hold=0.6):
        self.on(t)
        self.off(t + hold)


class Edge:
    def __init__(self, a, b, bend=(0, 0, 0), w=0.02, alpha=1.0, color=(.42, .53, .68), n=44):
        self.a, self.b, self.color = a, b, color
        self.w, self.alpha = Track(w), Track(alpha)
        self.pulses = []
        p0, p1 = a.pos, b.pos
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        p0, p1 = p0 + d * a.r * 0.95, p1 - d * b.r * 0.95
        if np.allclose(bend, 0):
            self.pts = np.array((p0, p1))
        else:
            m = (p0 + p1) / 2 + np.array(bend, float)
            u = np.linspace(0, 1, n)[:, None]
            self.pts = (1 - u) ** 2 * p0 + 2 * u * (1 - u) * m + u ** 2 * p1

    def pulse(self, t, dur=0.6, color=GOLD):
        self.pulses.append((t, dur, color))
        return t + dur


_txt = {}


def text_surface(s, size, color, bold=False):
    key = (s, size, color, bold)
    if key in _txt:
        return _txt[key]
    f = ImageFont.truetype(FONT_B if bold else FONT_R, size, index=2)
    box = f.getbbox(s)
    w, h = box[2] - box[0] + 8, box[3] - box[1] + 8
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4 - box[0], 4 - box[1]), s, font=f, fill=tuple(int(c * 255) for c in color) + (255,))
    a = np.asarray(im, dtype=np.float32)
    al = a[..., 3:] / 255.0
    b = np.concatenate([a[..., 2::-1] * al, a[..., 3:]], axis=2).astype(np.uint8).copy()
    sf = cairo.ImageSurface.create_for_data(b, cairo.FORMAT_ARGB32, w, h, w * 4)
    _txt[key] = (sf, w, h, b)
    return _txt[key]


def draw_text(ctx, s, x, y, size, color=INK, alpha=1.0, bold=False, anchor='c'):
    sf, w, h, _ = text_surface(s, size, color, bold)
    x0 = x - w / 2 if anchor == 'c' else (x - w if anchor == 'r' else x)
    ctx.set_source_surface(sf, x0, y - h / 2)
    ctx.paint_with_alpha(alpha)
    return w


def rrect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


class Gauge:
    """how strong one connection is, against the threshold of the neuron it feeds"""

    def __init__(self, value, theta, top, label='连接强度'):
        self.v, self.alpha, self.theta, self.top, self.label = Track(value), Track(0.0), theta, top, label

    def draw(self, ctx, t):
        al = self.alpha.at(t)
        if al <= 0.01:
            return
        x, y, w, h = 1390, 48, 470, 130
        ctx.set_source_rgba(.05, .08, .12, .62 * al)
        rrect(ctx, x, y, w, h, 16)
        ctx.fill()
        v = self.v.at(t)
        draw_text(ctx, self.label, x + 26, y + 34, 28, DIM, al, anchor='l')
        draw_text(ctx, f'{v:.2f}', x + w - 26, y + 34, 34, GOLD, al, True, anchor='r')
        bx, by, bw, bh = x + 26, y + 66, w - 52, 18
        ctx.set_source_rgba(.22, .28, .36, al)
        rrect(ctx, bx, by, bw, bh, 9)
        ctx.fill()
        fill = bw * min(1.0, v / self.top)
        if fill > 4:
            ctx.set_source_rgba(*GOLD, al)
            rrect(ctx, bx, by, fill, bh, 9)
            ctx.fill()
        tx = bx + bw * self.theta / self.top
        ctx.set_source_rgba(*INK, al)
        ctx.set_line_width(3)
        ctx.move_to(tx, by - 7)
        ctx.line_to(tx, by + bh + 7)
        ctx.stroke()
        draw_text(ctx, '门槛', tx, by + bh + 24, 22, INK, al)


class Line:
    def __init__(self, text, a, b):
        self.text, self.a, self.b = text, a, b

    def __getitem__(self, phrase):
        return self.a + (self.b - self.a) * self.text.index(phrase) / max(1, len(self.text))


def clauses(text):
    parts = re.findall(r'[^，。？！：；…]+[，。？！：；…]*', text)
    out = []
    for p in parts:
        if out and len(out[-1]) + len(p) <= 18:
            out[-1] += p
        else:
            out.append(p)
    return out


class Film:
    def __init__(self, cam, voice_dir='voice', known=None):
        self.cam, self.nodes, self.edges = cam, [], []
        self.lines, self.subs, self.cards, self.pings, self.rings, self.shots, self.gauges = [], [], [], [], [], [], []
        self.t, self.dir, self.known = 0.6, voice_dir, known or {}
        rng = np.random.default_rng(11)
        self.dust = rng.uniform((-34, -34, -20), (34, 34, 20), (260, 3))
        self.black = Track(1.0).to(0.0, 0.0, 0.8)
        self.bg = self._background()

    def _background(self):
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((x - W / 2) / W, (y - H / 2) / H)
        g = np.clip(1 - r * 1.25, 0, 1)[..., None]
        img = np.concatenate([.05 + .10 * g, .035 + .065 * g, .025 + .04 * g, np.ones_like(g)], axis=2)
        self._bg = (img * 255).astype(np.uint8)
        return cairo.ImageSurface.create_for_data(self._bg, cairo.FORMAT_ARGB32, W, H, W * 4)

    def node(self, *a, **k):
        n = Node(*a, **k)
        self.nodes.append(n)
        return n

    def edge(self, *a, **k):
        e = Edge(*a, **k)
        self.edges.append(e)
        return e

    def key(self, text):
        return hashlib.sha1(f'{VOICE}|{RATE}|{text}'.encode()).hexdigest()[:16]

    def say(self, text, pause=0.28):
        dur = self.known.get(text, 0.19 * len(text) + 0.2)
        ln = Line(text, self.t, self.t + dur)
        self.lines.append(ln)
        parts = clauses(text)
        n = sum(len(p) for p in parts)
        a = self.t
        for p in parts:
            b = a + dur * len(p) / n
            self.subs.append((a, b, p.rstrip('，。：；…')))
            a = b
        self.t += dur + pause
        return ln

    def wait(self, s):
        self.t += s
        return self.t

    def move(self, t, dur, az=None, el=None, dist=None, target=None):
        c = self.cam
        for tr, v in ((c.az, az), (c.el, el), (c.dist, dist)):
            if v is not None:
                tr.to(t, v, dur)
        if target is not None:
            for tr, v in zip((c.tx, c.ty, c.tz), target):
                tr.to(t, v, dur)

    def card(self, t0, t1, text, y=150, size=58, color=INK, bold=True, x=W / 2, anchor='c'):
        self.cards.append((t0, t1, text, x, y, size, color, bold, anchor))

    def ping(self, t, f=660.0, g=0.10):
        self.pings.append((t, f, g))

    def ring(self, node, t, color=RED):
        self.rings.append((node, t, color))

    def shoot(self, node, t, frm, dur=0.34, color=CYAN):
        """a signal from the outside world arriving at a neuron"""
        self.shots.append((node, t, dur, np.array(frm, float), color))
        return t + dur

    def _edge(self, ctx, v, e, t):
        al = e.alpha.at(t)
        if al <= 0.01:
            return
        x, y, z, s = v.project(e.pts)
        if float(np.min(z)) < 2.5:
            return
        al *= ease((float(np.min(z)) - 2.5) / 3.0)
        wd = max(0.8, e.w.at(t) * float(np.mean(s)))
        if len(x) == 2:
            wd = min(wd, 2.2)
        ctx.set_source_rgba(*e.color, min(1.0, al * v.fog(float(np.mean(z)))))
        ctx.set_line_width(wd)
        ctx.move_to(x[0], y[0])
        for i in range(1, len(x)):
            ctx.line_to(x[i], y[i])
        ctx.stroke()

    def _glow(self, ctx, x, y, R, color, a):
        g = cairo.RadialGradient(x, y, 0, x, y, R)
        g.add_color_stop_rgba(0, *color, a)
        g.add_color_stop_rgba(0.35, *color, a * 0.38)
        g.add_color_stop_rgba(1, *color, 0)
        ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_source(g)
        ctx.arc(x, y, R, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _node(self, ctx, v, n, t):
        al = n.alpha.at(t)
        if al <= 0.01:
            return
        x, y, z, s = v.project(n.pos)
        x, y, z, s = float(x), float(y), float(z), float(s)
        if z < 2.5:
            return
        al *= ease((z - 2.5) / 3.0)
        R = n.r * s
        if R < 0.8 or x < -R * 4 or x > W + R * 4 or y < -R * 4 or y > H + R * 4:
            return
        lit = n.lit.at(t)
        al *= v.fog(z)
        col = lerp((.19, .27, .39), (1, .78, .40), lit)
        if lit > 0.02:
            self._glow(ctx, x, y, R * 4.2, GOLD, 0.55 * lit * al)
        for P in n.br:
            bx, by, bz, bs = v.project(P)
            ctx.set_source_rgba(*col, al)
            for i in range(len(bx) - 1):
                ctx.set_line_width(max(0.9, n.r * 0.17 * (1 - i / len(bx)) ** 1.1 * bs[i]))
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.move_to(bx[i], by[i])
                ctx.line_to(bx[i + 1], by[i + 1])
                ctx.stroke()
        g = cairo.RadialGradient(x - .34 * R, y - .38 * R, R * .05, x, y, R)
        c0, c1, c2 = lerp((.42, .54, .70), (1, .96, .80), lit), lerp((.17, .24, .35), (1, .76, .34), lit), lerp((.06, .09, .15), (.86, .44, .12), lit)
        g.add_color_stop_rgba(0, *c0, al)
        g.add_color_stop_rgba(.55, *c1, al)
        g.add_color_stop_rgba(1, *c2, al)
        ctx.set_source(g)
        ctx.arc(x, y, R, 0, 2 * math.pi)
        ctx.fill()
        tag = n.tag.at(t)
        if n.label and tag > 0.01:
            size = int(max(24, min(90, R * 1.05)))
            draw_text(ctx, n.label, x, y, size, lerp(INK, (.12, .08, .03), lit), tag * al, True)

    def _pulses(self, ctx, v, e, t):
        for t0, dur, color in e.pulses:
            u = (t - t0) / dur
            if u < 0 or u > 1.16:
                continue
            n = len(e.pts)
            P = e.pts if n > 2 else np.linspace(e.pts[0], e.pts[1], 30)
            x, y, z, s = v.project(P)
            k = len(P)
            head = min(k - 1, int(u * (k - 1)))
            tail = max(0, int((u - 0.28) * (k - 1)))
            wd = max(1.5, (e.w.at(t) + 0.03) * float(np.mean(s)))
            for i in range(tail, head):
                a = (i - tail) / max(1, head - tail)
                ctx.set_source_rgba(*color, a * 0.9)
                ctx.set_line_width(wd)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.move_to(x[i], y[i])
                ctx.line_to(x[i + 1], y[i + 1])
                ctx.stroke()
            if u <= 1:
                self._glow(ctx, float(x[head]), float(y[head]), 0.34 * float(s[head]), color, 1.0)

    def frame(self, ctx, t):
        ctx.set_source_surface(self.bg, 0, 0)
        ctx.paint()
        v = View(self.cam, t)
        dx, dy, dz, ds = v.project(self.dust)
        for i in range(len(dx)):
            if dz[i] > 1 and -20 < dx[i] < W + 20 and -20 < dy[i] < H + 20:
                r = min(9.0, 0.035 * ds[i])
                ctx.set_source_rgba(.62, .74, .9, min(.22, 2.2 / max(r, 1) * .1))
                ctx.arc(dx[i], dy[i], max(.7, r), 0, 6.3)
                ctx.fill()
        order = []
        for e in self.edges:
            order.append((float((e.pts.mean(0) - v.C) @ v.f), 0, e))
        for n in self.nodes:
            order.append((float((n.pos - v.C) @ v.f), 1, n))
        order.sort(key=lambda o: -o[0])
        for _, kind, o in order:
            self._node(ctx, v, o, t) if kind else self._edge(ctx, v, o, t)
        for e in self.edges:
            if e.pulses:
                self._pulses(ctx, v, e, t)
        for node, t0, dur, frm, color in self.shots:
            u = (t - t0) / dur
            if 0 <= u <= 1.1:
                P = np.linspace(frm, node.pos, 24)
                x, y, z, s = v.project(P)
                head = min(23, int(u * 23))
                tail = max(0, head - 10)
                for i in range(tail, head):
                    ctx.set_source_rgba(*color, (i - tail) / 10 * .9)
                    ctx.set_line_width(max(2.0, .08 * float(s[i])))
                    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                    ctx.move_to(x[i], y[i])
                    ctx.line_to(x[i + 1], y[i + 1])
                    ctx.stroke()
                self._glow(ctx, float(x[head]), float(y[head]), .3 * float(s[head]), color, 1.0)
        for node, t0, color in self.rings:
            u = (t - t0) / 0.5
            if 0 <= u <= 1:
                x, y, z, s = v.project(node.pos)
                R = node.r * float(s) * (1.05 + .55 * u)
                ctx.set_source_rgba(*color, (1 - u) * .9)
                ctx.set_line_width(5)
                ctx.arc(float(x), float(y), R, 0, 6.3)
                ctx.stroke()
        for g in self.gauges:
            g.draw(ctx, t)
        for t0, t1, s, x, y, size, color, bold, anchor in self.cards:
            if t0 <= t <= t1:
                a = ease((t - t0) / .3) * ease((t1 - t) / .3)
                sf, w, h, _ = text_surface(s, size, color, bold)
                x0 = x - w / 2 if anchor == 'c' else x
                ctx.set_source_rgba(.03, .05, .08, .55 * a)
                rrect(ctx, x0 - 28, y - h / 2 - 14, w + 56, h + 28, 16)
                ctx.fill()
                draw_text(ctx, s, x, y, size, color, a, bold, anchor)
        for a, b, s in self.subs:
            if a <= t < b:
                sf, w, h, _ = text_surface(s, 46, INK, False)
                ctx.set_source_rgba(.02, .03, .05, .6)
                rrect(ctx, W / 2 - w / 2 - 26, 968 - h / 2 - 10, w + 52, h + 20, 12)
                ctx.fill()
                draw_text(ctx, s, W / 2, 968, 46, INK)
                break
        k = self.black.at(t)
        if k > 0.004:
            ctx.set_source_rgba(0, 0, 0, k)
            ctx.paint()

    def length(self):
        return self.t + 1.2


def tts(texts, folder):
    import edge_tts
    os.makedirs(folder, exist_ok=True)

    async def go():
        for s in texts:
            k = hashlib.sha1(f'{VOICE}|{RATE}|{s}'.encode()).hexdigest()[:16]
            p = f'{folder}/{k}.wav'
            if os.path.exists(p):
                continue
            await edge_tts.Communicate(s, VOICE, rate=RATE).save(f'{folder}/{k}.mp3')
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f'{folder}/{k}.mp3', '-ar', str(SR), '-ac', '1', p],
                           check=True)
    asyncio.run(go())


def read_wav(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def soundtrack(f, out):
    n = int(SR * (f.length() + 1))
    v = np.zeros(n, np.float32)
    for ln in f.lines:
        a = read_wav(f'{f.dir}/{f.key(ln.text)}.wav')
        rms = float(np.sqrt(np.mean(a ** 2)) + 1e-6)
        a = a * (0.12 / rms)
        i = int(ln.a * SR)
        v[i:i + len(a)] += a[:n - i]
    fx = np.zeros(n, np.float32)
    for t, fq, g in f.pings:
        L = int(SR * 0.5)
        x = np.arange(L) / SR
        s = g * np.exp(-x * 9) * (np.sin(2 * np.pi * fq * x) + .3 * np.sin(4 * np.pi * fq * x)) * np.minimum(1, x * 400)
        i = int(t * SR)
        if i + L < n:
            fx[i:i + L] += s.astype(np.float32)
    x = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * h * x + p) * (1 + .4 * np.sin(2 * np.pi * x / m + p)) for h, m, p in
              ((110.0, 17, 0.0), (164.8, 23, 1.0), (220.0, 19, 2.0), (277.2, 29, 3.0))).astype(np.float32) * 0.0075
    env = np.minimum(1, np.minimum(x / 3, (x[-1] - x) / 3)).astype(np.float32)
    mix = v + 0.55 * fx + pad * env
    peak = float(np.abs(mix).max())
    mix = np.tanh(mix * 1.15) / 1.05
    with wave.open(out, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    print('audio', round(len(mix) / SR, 1), 's, peak', round(peak, 2), flush=True)


def make(build, cam, out, fps=FPS, preset='medium', every=None):
    """two passes: collect the sentences, make the voice, then build the timeline with the real lengths"""
    first = Film(cam())
    build(first)
    tts([ln.text for ln in first.lines], 'voice')
    known = {}
    for ln in first.lines:
        p = f'voice/{first.key(ln.text)}.wav'
        known[ln.text] = len(read_wav(p)) / SR
    f = Film(cam(), known=known)
    build(f)
    T = f.length()
    print('length', round(T, 1), 's', len(f.lines), 'lines', flush=True)
    soundtrack(f, 'mix.wav')
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{W}x{H}', '-r', str(fps),
                           '-i', '-', '-i', 'mix.wav', '-c:v', 'libx264', '-preset', preset, '-crf', '19', '-pix_fmt', 'yuv420p',
                           '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    sf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(sf)
    t0 = time.time()
    for i in range(int(T * fps)):
        if i % 300 == 0:
            print(f'frame {i} of {int(T * fps)}, {time.time() - t0:.0f}s', flush=True)
        f.frame(ctx, i / fps)
        sf.flush()
        ff.stdin.write(bytes(sf.get_data()))
        if every and i % every == 0:
            sf.write_to_png(f'still_{i:05d}.png')
    ff.stdin.close()
    ff.wait()
    print('VIDEO', out, round(os.path.getsize(out) / 1e6, 1), 'MB', flush=True)
