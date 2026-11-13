"""Shared look for the teaching videos: dark page, plain shapes, one sentence of Chinese at the bottom saying what is
happening, and a short line at the top saying where we are. The rule shown is always computed, not drawn by hand."""
from manim import (BLACK, DOWN, LEFT, RIGHT, UP, Arrow, Circle, Dot, FadeIn, FadeOut, ReplacementTransform, Scene, Text,
                   Transform, VGroup, config)

config.background_color = '#10151c'
INK, DIM, GOLD, CYAN, RED, DARK = '#eef2f6', '#8a97a6', '#f2b84b', '#55c2e6', '#ef6f6c', '#2a3440'
FONT = 'Noto Sans CJK SC'


def T(s, size=34, color=INK, weight='NORMAL'):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


class Lesson(Scene):
    """say() puts one sentence on the bottom bar; wait time follows its length, so the viewer can read it."""
    caption = None

    def say(self, s, extra=0.0):
        new = T(s, 36).to_edge(DOWN, buff=0.55)
        if new.width > 12.6:
            new.scale_to_fit_width(12.6)
        if self.caption is None:
            self.play(FadeIn(new, shift=UP * 0.1), run_time=0.35)
        else:
            self.play(FadeOut(self.caption), FadeIn(new, shift=UP * 0.1), run_time=0.35)
        self.caption = new
        self.wait(1.0 + 0.2 * len(s) + extra)

    def note(self, s):
        """a caption that changes while the action goes on, with no pause of its own"""
        new = T(s, 36).to_edge(DOWN, buff=0.55)
        if self.caption is None:
            self.add(new)
        else:
            self.remove(self.caption)
            self.add(new)
        self.caption = new

    def title(self, s, sub=None):
        g = VGroup(T(s, 60, INK, 'BOLD'))
        if sub:
            g.add(T(sub, 32, DIM))
            g.arrange(DOWN, buff=0.4)
        self.play(FadeIn(g), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(g), run_time=0.4)

    def topic(self, s):
        new = T(s, 28, DIM).to_corner(UP + LEFT, buff=0.45)
        old = getattr(self, 'topic_m', None)
        self.play(*( [FadeOut(old)] if old else []), FadeIn(new), run_time=0.3)
        self.topic_m = new

    def clear_stage(self, *keep):
        drop = [m for m in self.mobjects if m not in keep and m is not self.caption and m is not getattr(self, 'topic_m', None)]
        if self.caption is not None:
            drop.append(self.caption)
            self.caption = None
        if drop:
            self.play(*[FadeOut(m) for m in drop], run_time=0.4)


class Cell(VGroup):
    """a neuron: one circle, dark or lit, nothing in between"""

    def __init__(self, name, pos, r=0.62):
        self.body = Circle(radius=r, color=DIM, stroke_width=4).set_fill(DARK, 1).move_to(pos)
        self.tag = T(name, 40, INK, 'BOLD').move_to(pos)
        super().__init__(self.body, self.tag)
        self.pos = pos

    def lit(self):
        return [self.body.animate.set_fill(GOLD, 1).set_stroke(GOLD), self.tag.animate.set_color(BLACK)]

    def dark(self):
        return [self.body.animate.set_fill(DARK, 1).set_stroke(DIM), self.tag.animate.set_color(INK)]


def link(a, b, w):
    """a connection from a to b; thickness shows its strength w"""
    return Arrow(a.get_right(), b.get_left(), buff=0.08, color=DIM, stroke_width=max(1.5, 3 + 26 * w),
                 max_tip_length_to_length_ratio=0.1, max_stroke_width_to_length_ratio=30)


class PairLesson(Lesson):
    """two neurons and the line between them"""

    def pulse(self, a, b, color=GOLD):
        d = Dot(a.get_right(), radius=0.13, color=color)
        self.play(FadeIn(d), run_time=0.12)
        self.play(d.animate.move_to(b.get_left()), run_time=0.7)
        self.play(FadeOut(d), run_time=0.12)

    def light_from_outside(self, cell, side):
        d = Dot(cell.get_center() + side * 2.6, radius=0.13, color=CYAN)
        self.play(FadeIn(d), run_time=0.1)
        self.play(d.animate.move_to(cell.get_center() + side * 0.7), run_time=0.4)
        self.play(FadeOut(d), *cell.lit(), run_time=0.15)

    def show_w(self, w):
        new = T(f'连接强度 {w:.2f}', 34, GOLD).move_to(UP * 2.75)
        old = getattr(self, 'w_m', None)
        if old is None:
            self.play(FadeIn(new), run_time=0.3)
        else:
            self.play(Transform(old, new), run_time=0.3)
            return
        self.w_m = new

    def set_link(self, A, B, w):
        new = link(A, B, w)
        self.play(ReplacementTransform(self.arrow, new), run_time=0.35)
        self.arrow = new
