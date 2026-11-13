"""Episode 1: the first thing to learn is one rule between two neurons. One new idea: the one that lights first gets
connected to the one that lights after it. Numbers on screen come from the small rule below."""
from manim import DOWN, LEFT, RIGHT, UP, CurvedArrow, FadeIn, FadeOut, VGroup

from common import CYAN, DIM, GOLD, RED, Cell, PairLesson, T, link

W0, ETA, WREF, THETA = 0.12, 0.40, 0.40, 1.00


def grow(w):
    """A then B: the connection grows, less each time it is already strong"""
    return w + ETA / (1 + w / WREF)


class Ep1(PairLesson):
    def construct(self):
        self.title('第 1 集：先亮的指向后亮的', '从最小的零件开始，一步步搭出一个会学习的大脑')
        self.topic('神经元')
        A = Cell('A', LEFT * 3.2 + DOWN * 0.1)
        self.play(FadeIn(A))
        self.say('这是一个神经元。它只有两种状态。')
        self.play(*A.lit(), run_time=0.3)
        self.say('亮。')
        self.play(*A.dark(), run_time=0.3)
        self.say('或者灭。没有半亮。')

        self.topic('连接')
        B = Cell('B', RIGHT * 3.2 + DOWN * 0.1)
        self.play(FadeIn(B))
        w = W0
        self.arrow = link(A, B, w)
        self.play(FadeIn(self.arrow))
        self.show_w(w)
        self.say('A 和 B 之间有一根连接。线越粗，连接越强。')
        limit = T(f'B 要被 A 点亮，强度至少要 {THETA:.2f}', 30, DIM).move_to(UP * 2.1)
        self.play(FadeIn(limit))
        self.say(f'现在只有 {W0:.2f}，太弱，A 点不亮 B。')

        self.topic('试一下')
        self.play(*A.lit(), run_time=0.2)
        self.pulse(A, B)
        self.play(*A.dark(), run_time=0.3)
        self.say('A 亮了，信号传过去。B 没有反应。')

        self.topic('学习规则')
        rule = T('A 亮后 B 再亮 → A 到 B 的连接变强', 36, GOLD, 'BOLD').move_to(UP * 1.45)
        self.play(FadeIn(rule))
        self.say('学习的第一条规则：先亮的，指向后亮的。')
        world = VGroup(T('外界', 28, CYAN).move_to(LEFT * 6.1 + DOWN * 0.1), T('外界', 28, CYAN).move_to(RIGHT * 6.1 + DOWN * 0.1))
        self.play(FadeIn(world), run_time=0.3)
        self.say('现在让外界先点亮 A，再点亮 B。')
        for i in range(5):
            self.note(f'第 {i + 1} 次：A 先亮，B 后亮，连接变强')
            self.light_from_outside(A, LEFT)
            self.pulse(A, B)
            self.play(*A.dark(), run_time=0.15)
            self.light_from_outside(B, RIGHT)
            nw = grow(w)
            self.show_w(nw)
            self.set_link(A, B, nw)
            w = nw
            self.play(*B.dark(), run_time=0.15)
        self.say(f'重复 5 次，强度从 {W0:.2f} 涨到 {w:.2f}，超过了门槛。')

        self.topic('学会了吗')
        self.play(FadeOut(rule), FadeOut(world), run_time=0.3)
        self.say('这次外界只点亮 A，不碰 B。')
        self.light_from_outside(A, LEFT)
        self.pulse(A, B)
        self.play(*B.lit(), *A.dark(), run_time=0.25)
        self.say('B 自己亮了。')
        self.play(*B.dark(), run_time=0.3)
        self.say('经历过很多次"A 之后是 B"，连接就记住了这个先后。')

        self.topic('同时亮呢')
        self.say('如果 A 和 B 同时亮呢？')
        self.play(*A.lit(), *B.lit(), run_time=0.3)
        self.say('连接不变。同时亮的，不互相加强。')
        self.play(*A.dark(), *B.dark(), run_time=0.3)
        back = CurvedArrow(B.get_bottom(), A.get_bottom(), angle=-0.7, color=RED, stroke_width=8)
        self.play(FadeIn(back), run_time=0.4)
        self.say('假设同时亮也加强，A 和 B 就会互相连上。')
        self.note('A 点亮 B，B 又点亮 A，停不下来')
        for _ in range(3):
            self.play(*A.lit(), *B.dark(), run_time=0.22)
            self.play(*B.lit(), *A.dark(), run_time=0.22)
        self.play(*B.dark(), run_time=0.2)
        self.say('所以只算先后，不算同时。', 0.5)
        self.play(FadeOut(back), run_time=0.3)

        self.topic('这一集学到')
        self.clear_stage()
        pts = VGroup(T('1. 神经元只有亮和灭', 40), T('2. 连接有强有弱', 40), T('3. A 亮后 B 再亮，A 到 B 变强', 40, GOLD),
                     T('4. 同时亮的不加强', 40, GOLD), T('5. 连接够强后，A 一亮，B 就会亮', 40)
                     ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).shift(UP * 0.5)
        for p in pts:
            self.play(FadeIn(p, shift=RIGHT * 0.15), run_time=0.4)
            self.wait(0.8)
        self.say('下一集：如果顺序反过来，这根线会怎样？')
        self.wait(1)
