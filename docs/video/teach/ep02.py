"""Episode 2: the same two neurons, now the other way round. One new idea: B before A weakens A->B, and a neuron that
stays on nets to nothing. Numbers on screen come from the small ledger here."""
from manim import DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, ReplacementTransform, VGroup

from common import CYAN, DIM, GOLD, RED, Cell, PairLesson, T, link

W_START, STEP, THETA = 1.04, 0.15, 1.00          # where episode 1 ended; one change; B's threshold


class Ep2(PairLesson):
    def construct(self):
        self.title('第 2 集：顺序反了会怎样', '接着上一集的两个神经元')
        self.topic('上一集')
        A, B = Cell('A', LEFT * 3.2 + DOWN * 0.1), Cell('B', RIGHT * 3.2 + DOWN * 0.1)
        w = W_START
        self.arrow = link(A, B, w)
        self.play(FadeIn(A), FadeIn(B), FadeIn(self.arrow))
        self.show_w(w)
        limit = T(f'B 要被 A 点亮，强度至少要 {THETA:.2f}', 30, DIM).move_to(UP * 2.1)
        self.play(FadeIn(limit), run_time=0.3)
        self.say('上一集结束时，A 到 B 的连接已经很强，A 一亮，B 就亮。')
        self.play(*A.lit(), run_time=0.2)
        self.pulse(A, B)
        self.play(*B.lit(), *A.dark(), run_time=0.25)
        self.play(*B.dark(), run_time=0.3)

        self.topic('反过来')
        self.say('如果世界变了，总是 B 先亮，A 后亮呢？')
        rule = T('B 亮后 A 再亮 → A 到 B 的连接变弱', 36, RED, 'BOLD').move_to(UP * 1.45)
        self.play(FadeIn(rule))
        self.say('这是第二条规则：顺序反了，连接就变弱。')
        world = VGroup(T('外界', 28, CYAN).move_to(LEFT * 6.1 + DOWN * 0.1), T('外界', 28, CYAN).move_to(RIGHT * 6.1 + DOWN * 0.1))
        self.play(FadeIn(world), run_time=0.3)
        for i in range(4):
            self.note(f'第 {i + 1} 次：B 先亮，A 后亮，连接变弱')
            self.light_from_outside(B, RIGHT)
            self.play(*B.dark(), run_time=0.15)
            self.light_from_outside(A, LEFT)
            w -= STEP
            self.show_w(w)
            self.set_link(A, B, w)
            self.play(*A.dark(), run_time=0.15)
        self.say(f'连接从 {W_START:.2f} 掉到 {w:.2f}，低于门槛了。')

        self.topic('试一下')
        self.play(FadeOut(rule), FadeOut(world), run_time=0.3)
        self.light_from_outside(A, LEFT)
        self.pulse(A, B)
        self.play(*A.dark(), run_time=0.3)
        self.say('A 亮了，B 不再跟着亮。这个先后被忘掉了。')
        self.say('所以连接记住的是"经常发生的先后"，而不是发生过一次的事。')

        self.topic('一直亮着')
        self.clear_stage(A, B, self.arrow, limit, self.w_m)
        self.say('再看一种情况：A 一直亮着，B 中途亮一下。')
        self.play(*A.lit(), run_time=0.3)
        self.wait(0.6)
        self.play(*B.lit(), run_time=0.25)
        plus = T(f'+{STEP:.2f}', 44, GOLD, 'BOLD').next_to(self.arrow, UP, buff=0.35)
        self.play(FadeIn(plus), run_time=0.3)
        self.say('B 亮的时候，A 在它前面：相当于 A 先 B 后，加强。')
        self.play(*B.dark(), run_time=0.25)
        minus = T(f'-{STEP:.2f}', 44, RED, 'BOLD').next_to(self.arrow, DOWN, buff=0.35)
        self.play(FadeIn(minus), run_time=0.3)
        self.say('B 灭了，A 还亮着：相当于 B 先 A 后，减弱。')
        self.say('一加一减，等于没变。')
        self.play(FadeOut(plus), FadeOut(minus), *A.dark(), run_time=0.3)

        self.topic('为什么重要')
        self.clear_stage()
        C = Cell('A', LEFT * 3.6 + DOWN * 0.3, r=0.7)
        outs = [Cell(s, RIGHT * 3.4 + UP * (2.0 - i * 1.15) + DOWN * 0.3, r=0.42) for i, s in enumerate('BCDE')]
        lines = [link(C, o, 0.02) for o in outs]
        self.play(FadeIn(C), *[FadeIn(o) for o in outs], *[FadeIn(l) for l in lines])
        self.play(*C.lit(), run_time=0.3)
        self.say('有的神经元总是亮着，什么事它都在场。')
        for o in outs + outs:
            self.play(*o.lit(), run_time=0.14)
            self.play(*o.dark(), run_time=0.14)
        self.say('别的神经元亮一下灭一下，它的线一根都不会变粗。')
        self.say('所以总是亮的神经元，不会连出一大堆没用的连接。')

        self.topic('这一集学到')
        self.clear_stage()
        pts = VGroup(T('1. B 亮后 A 再亮，A 到 B 变弱', 40, RED), T('2. 不再发生的先后，慢慢被忘掉', 40),
                     T('3. 一直亮着的 A 碰到 B 亮：一加一减，不变', 40, GOLD),
                     T('4. 所以总是亮的神经元，连不出多余的线', 40)).arrange(DOWN, buff=0.45, aligned_edge=LEFT).shift(UP * 0.5)
        for p in pts:
            self.play(FadeIn(p, shift=RIGHT * 0.15), run_time=0.4)
            self.wait(1.0)
        self.say('下一集：一根线能一直变粗吗？')
        self.wait(1)
