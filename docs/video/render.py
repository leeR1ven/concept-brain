"""Render one explainer episode: narration by a local speech model, one slide per line, burned-in subtitles.
usage: HF_ENDPOINT=<mirror> python3 -u render.py ep01_script.json out.mp4 [voice]"""
import json
import math
import os
import random
import subprocess
import sys
import textwrap

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline
from PIL import Image, ImageDraw, ImageFont

W, H, SR = 1280, 720, 24000
BG, INK, DIM, ACC, ACC2, GRID = '#10151c', '#eef2f6', '#8a97a6', '#f2b84b', '#55c2b5', '#2a3440'
REG, BOLD = '/usr/share/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/opentype/noto/NotoSansCJK-Bold.ttc'
font = lambda s, b=False: ImageFont.truetype(BOLD if b else REG, s, index=2)
TTS_WORDS = [('聊天 AI', '聊天人工智能'), ('AGI', '通用人工智能')]


def mid(d, x, y, s, size, fill=INK, bold=False):
    f = font(size, bold)
    b = d.textbbox((0, 0), s, font=f)
    d.text((x - (b[2] - b[0]) / 2, y - (b[3] - b[1]) / 2 - b[1]), s, font=f, fill=fill)


def box(d, x0, y0, x1, y1, s='', size=30, fill=None, line=GRID, color=INK):
    d.rounded_rectangle((x0, y0, x1, y1), 16, fill=fill, outline=line, width=3)
    if s:
        mid(d, (x0 + x1) / 2, (y0 + y1) / 2, s, size, color)


def arrow(d, x0, y, x1, color=DIM):
    d.line((x0, y, x1, y), fill=color, width=4)
    d.polygon([(x1, y), (x1 - 14, y - 9), (x1 - 14, y + 9)], fill=color)


def grid(d, cx, cy, n, m, gap, r, lit):
    for i in range(n):
        for j in range(m):
            x, y = cx + (i - (n - 1) / 2) * gap, cy + (j - (m - 1) / 2) * gap
            on = (i, j) in lit
            d.ellipse((x - r, y - r, x + r, y + r), fill=ACC if on else GRID)


def child(d, x, base, h):
    r = h * 0.16
    d.ellipse((x - r, base - h, x + r, base - h + 2 * r), fill=ACC2)
    d.rounded_rectangle((x - r * 0.9, base - h + 2.2 * r, x + r * 0.9, base), 8, fill=ACC2)


def chat(d):
    box(d, 120, 120, 700, 520, fill='#161d27')
    d.rounded_rectangle((330, 160, 670, 225), 14, fill='#2c3a4b')
    mid(d, 500, 192, '今天天气怎么样？', 26)
    d.rounded_rectangle((150, 250, 520, 330), 14, fill='#23303d')
    mid(d, 335, 290, '我来帮你看看', 26)


def book(d, x, y):
    d.rounded_rectangle((x - 95, y - 125, x + 95, y + 125), 8, fill='#33506b')
    d.line((x - 75, y - 125, x - 75, y + 125), fill='#233a50', width=6)
    mid(d, x + 8, y, '百科全书', 28)


def draw(view):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    random.seed(sum(map(ord, view)))
    if view == 'title':
        mid(d, W / 2, 250, '第 1 期', 34, DIM)
        mid(d, W / 2, 340, '一个一直活着的大脑', 72, INK, True)
        mid(d, W / 2, 445, '讲的是想法，不是成果', 32, ACC)
    elif view in ('chat', 'chat_close'):
        chat(d)
        grid(d, 960, 300, 6, 6, 42, 11, {(random.randrange(6), random.randrange(6)) for _ in range(9)})
        mid(d, 960, 510, '模型的连接', 28, DIM)
        if view == 'chat_close':
            mid(d, 960, 560, '这次对话前后：没有变化', 30, ACC, True)
    elif view == 'frozen':
        for i, (s, c) in enumerate((('训练', None), ('定型', None), ('使用', None))):
            x = 170 + i * 340
            box(d, x, 240, x + 240, 400, s, 44, fill=c)
            if i < 2:
                arrow(d, x + 250, 320, x + 330)
        mid(d, 1000, 450, '连接不再改变', 30, ACC, True)
        mid(d, 1000, 500, '另记的笔记在模型之外', 26, DIM)
    elif view in ('book', 'book_child'):
        book(d, 300, 320)
        mid(d, 300, 490, '每次翻开，内容都一样', 28, DIM)
        if view == 'book_child':
            for i, h in enumerate((110, 170, 240, 320)):
                child(d, 720 + i * 120, 460, h)
            mid(d, 900, 510, '今天和昨天，已经不是同一个', 28, ACC)
    elif view in ('goal', 'goal_note'):
        mid(d, W / 2, 260, '最终目标', 34, DIM)
        mid(d, W / 2, 345, '能实时学习的人工智能架构', 58, INK, True)
        if view == 'goal_note':
            box(d, 440, 440, 840, 520, '方向，不是成果', 36, line=ACC, color=ACC)
    elif view == 'alive_off':
        mid(d, W / 2, 180, '聊天窗口：每次重新开始', 36, DIM)
        for i in range(5):
            x = 130 + i * 210
            box(d, x, 290, x + 170, 370, f'对话 {i + 1}', 26)
    elif view == 'alive_on':
        mid(d, W / 2, 180, '一直活着：连续的一生', 36, ACC, True)
        d.line((120, 330, 1140, 330), fill=ACC2, width=10)
        d.polygon([(1165, 330), (1135, 310), (1135, 350)], fill=ACC2)
        for x, s in ((230, '听'), (470, '想'), (710, '说'), (950, '学')):
            d.ellipse((x - 13, 317, x + 13, 343), fill=ACC)
            mid(d, x, 385, s, 32)
        mid(d, W / 2, 470, '不停思考和行动，不重来', 30, DIM)
    elif view == 'neurons':
        lit = {(random.randrange(14), random.randrange(9)) for _ in range(34)}
        grid(d, W / 2, 310, 14, 9, 52, 15, lit)
        mid(d, W / 2 - 170, 590, '亮', 30, ACC, True)
        mid(d, W / 2 + 170, 590, '不亮', 30, DIM, True)
    elif view == 'neurons_baby':
        pos = lambda cx, i, j: (cx + (i - 3.5) * 40, 300 + (j - 3.5) * 40)
        grid(d, 340, 300, 8, 8, 40, 11, set())
        mid(d, 340, 510, '新生：还没有学到的连接', 30, DIM)
        arrow(d, 560, 300, 720, ACC)
        mid(d, 640, 262, '经历', 30, ACC)
        for _ in range(26):
            a, c = (random.randrange(8), random.randrange(8)), (random.randrange(8), random.randrange(8))
            if a != c and abs(a[0] - c[0]) + abs(a[1] - c[1]) <= 4:
                d.line((*pos(940, *a), *pos(940, *c)), fill=ACC, width=3)
        grid(d, 940, 300, 8, 8, 40, 11, set())
        mid(d, 940, 510, '连接由经历长出来', 30, ACC)
    elif view == 'forms':
        box(d, 170, 220, 590, 430, '会思考的虚拟模型', 38)
        box(d, 690, 220, 1110, 430, '能操控机器的模型', 38)
        mid(d, W / 2, 500, '设想中的最终形态', 30, DIM)
    elif view == 'honest':
        box(d, 180, 200, 600, 400, '设计者的判断', 42, line=ACC, color=ACC)
        mid(d, 390, 450, '架构大致是对的', 28, DIM)
        box(d, 680, 200, 1100, 400, '结论', 42, color=DIM)
        d.line((700, 300, 1080, 300), fill=ACC, width=5)
        mid(d, 890, 450, '到现在还没有被证明', 28, DIM)
    elif view == 'slow':
        for s, y, c, z in (('即便是婴儿', 240, INK, 52), ('也不是马上就会', 330, INK, 52)):
            mid(d, W / 2, y, s, z, c, True)
        mid(d, W / 2, 460, '没得到结果，不一定是出了问题，也可能只是学得慢', 30, ACC)
    elif view == 'open':
        mid(d, W / 2, 230, '代码公开', 60, INK, True)
        mid(d, W / 2, 330, '看懂这些想法，一起判断它对不对', 34, ACC2)
        for i in range(21):
            x = 240 + i * 40
            r = 11 + (i % 3) * 3
            d.ellipse((x - r, 450 - r, x + r, 450 + r), fill=GRID if i % 4 else ACC)
    elif view == 'next':
        mid(d, W / 2, 280, '下一期', 34, DIM)
        mid(d, W / 2, 370, '一张图看完整个大脑', 60, INK, True)
    return im


def subtitle(im, s):
    d = ImageDraw.Draw(im)
    d.rectangle((0, 610, W, H), fill='#0a0e13')
    rows = textwrap.wrap(s, 40)[:2]
    for i, r in enumerate(rows):
        mid(d, W / 2, 665 - (len(rows) - 1) * 21 + i * 42, r, 30)


def main():
    script, out = json.load(open(sys.argv[1])), sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else 'zf_001'
    repo = 'hexgrad/Kokoro-82M-v1.1-zh'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = KPipeline(lang_code='z', repo_id=repo, model=KModel(repo_id=repo).to(dev).eval())
    os.makedirs('clips', exist_ok=True)
    names, total = [], 0.0
    for i, ln in enumerate(script['lines']):
        text = ln['say']
        for a, b in TTS_WORDS:
            text = text.replace(a, b)
        audio = [r.audio.cpu().numpy() for r in pipe(text, voice=voice, speed=1.0)]
        wav = np.concatenate(audio + [np.zeros(int(SR * 0.4))])
        p = f'clips/{i:02d}'
        sf.write(p + '.wav', wav, SR)
        im = draw(ln['view'])
        subtitle(im, ln['say'])
        im.save(p + '.png')
        dur = len(wav) / SR
        total += dur
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-loop', '1', '-framerate', '15', '-i', p + '.png', '-i',
                        p + '.wav', '-t', f'{dur:.3f}', '-vf', 'fade=in:0:5', '-c:v', 'libx264', '-preset', 'veryfast',
                        '-crf', '28', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '64k', p + '.mp4'], check=True)
        names.append(f"file '{p}.mp4'")
        print(f'line {i}: {dur:.1f}s', flush=True)
    open('clips.txt', 'w').write('\n'.join(names))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'clips.txt', '-c', 'copy',
                    '-movflags', '+faststart', out], check=True)
    print(f'RENDERED {out} {total:.0f}s {os.path.getsize(out) / 1e6:.1f}MB', flush=True)


if __name__ == '__main__':
    main()
