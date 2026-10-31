# -*- coding: utf-8 -*-
"""world2 constants: lexicon (all chars in the corpus voice), house + garden layout, held-out pairs, sensory codes.

Everything here depends on the world seed only (never on agents).  The 12 everyday objects, 6 colours, 3 counts and the
names are the bench's (world.py); world2 adds rooms, caregivers, sizes and more names.
"""
from __future__ import annotations

import collections

import numpy as np

# ------------------------------------------------------------------------------------------ lexicon
OBJECTS = ['苹果', '香蕉', '饼干', '牛奶', '西瓜', '葡萄', '小狗', '小猫', '皮球', '汽车', '小鸟', '鸭子']   # = bench
FOODS = OBJECTS[:6]
ANIMALS = ['小狗', '小猫', '小鸟', '鸭子']            # wander by themselves; cannot be taken; always one
TOYS = ['皮球', '汽车']
COLOURS = ['红色', '黄色', '蓝色', '绿色', '白色', '黑色']                                          # = bench
COUNTS = ['一个', '两个', '三个']                                                                 # = bench
SIZES = ['小', '大']
# bench NOVEL minus the two furniture words (椅子 桌子): 38 novel kinds, introduced once each by the caregivers
NOVEL = ['熊猫', '兔子', '气球', '积木', '蛋糕', '面包', '帽子', '杯子', '书包', '飞机', '火车', '雨伞', '鞋子', '袜子',
         '手机', '电脑', '眼镜', '钥匙', '铅笔', '衣服', '裙子', '星星', '月亮', '太阳', '花朵', '草莓', '橘子', '柠檬',
         '番茄', '土豆', '老虎', '狮子', '猴子', '大象', '青蛙', '乌龟', '蝴蝶', '蜜蜂']
NOVEL_FOODS = ['蛋糕', '面包', '草莓', '橘子', '柠檬', '番茄', '土豆']
KINDS = OBJECTS + NOVEL                                     # kind index = position in this list
KIND_ID = {k: i for i, k in enumerate(KINDS)}
EDIBLE = set(FOODS) | set(NOVEL_FOODS)
ROOMS = ['厨房', '客厅', '卧室', '花园']                    # room ids 0..3
CAREGIVERS = ['妈妈', '爸爸', '奶奶']
_NAME_CHARS = '明红华刚丽军美强芳龙云东兰虎雪宝林文飞玉新月冬花英平海波静杰敏霞春秋光亮晨阳晴雨凤鹏博安乐欢康丹青松梅菊竹江河山石金银珠宁涛洋天'
AGENT_NAMES = ['小' + c for c in _NAME_CHARS]               # 64 names; the first 8 are the bench's NAMES
GREET_MORNING = ['早上好', '早', '你好']
GREET = ['你好', '你好呀']
GOODNIGHT = ['晚安']
GREET_WORDS = {'你好', '你好呀', '早上好', '早'}
HUNGRY_WORDS = {'饿', '我饿了', '饿了', '要吃', '我要吃', '吃'}

# ------------------------------------------------------------------------------------------ layout (24 x 24)
# '#' wall/fence  'D' door  'k' kitchen floor  'l' living-room floor  'b' bedroom floor  'g' grass
# 'T' table  'B' bed  'S' sofa  'O' stove  'H' shelf  'Y' tree       (x = column, y = row; row 0 at the top)
LAYOUT = [
    '########################',
    '#OOkkkkkkkk#HHbbbbbbbBB#',
    '#kkkkkkkkkk#bbbbbbbbbBB#',
    '#kkkkTTkkkk#bbbbbbbbbbb#',
    '#kkkkTTkkkk#bbbbbbbbbbb#',
    '#kkkkkkkkkk#bbbbbbbbbbb#',
    '####DDD#########DDD#####',
    '#llllllllllllllllllllll#',
    '#llllllllllllllllllllll#',
    '#llllllTTllllllllllllll#',
    '#llllllllllllllllllllll#',
    '#llllllllllllllSSSlllll#',
    '##########DDD###########',
    '#gggggggggggggggggggggg#',
    '#gggggggggggggggggggggg#',
    '#gggggggggggggggggYgggg#',
    '#gggYgggggggggggggggggg#',
    '#gggggggggggggggggggggg#',
    '#gggggggggggggggggggggg#',
    '#gggggggggggggggggggggg#',
    '#gggggggggYgggggggggggg#',
    '#gggggggggggggggggggggg#',
    '#gggggggggggggggggggggg#',
    '########################',
]
H = W = 24
TERRAIN_CHARS = '#Dklbg TBSOHY'.replace(' ', '')   # terrain code index = position in this string
T_WALL, T_DOOR, T_K, T_L, T_B, T_G, T_TABLE, T_BED, T_SOFA, T_STOVE, T_SHELF, T_TREE = range(12)
T_OUT = 12                                          # outside the map (padding): seen as nothing
WALKABLE_T = {T_DOOR, T_K, T_L, T_B, T_G}
HOLDS_OBJ_T = {T_K, T_L, T_B, T_G, T_TABLE, T_BED, T_SOFA, T_SHELF}   # cells where an object may lie
OPAQUE_T = {T_WALL, T_OUT}                          # walls block sight (doors are open doorways)
FLOOR_ROOM = {T_K: 0, T_L: 1, T_B: 2, T_G: 3}


def room_of_xy(x, y):
    """room id of a map cell: kitchen 0 (x 1-10, y 1-5), living 1 (y 7-11 and the doors), bedroom 2 (x 12-22,
    y 1-5), garden 3 (y 13-22); walls -1."""
    c = LAYOUT[y][x]
    if c == '#':
        return -1
    if c == 'D':
        return 1
    if y <= 5:
        return 0 if x <= 10 else 2
    if y <= 11:
        return 1
    return 3


# usual rooms (tidying puts misplaced objects back): foods -> kitchen, ball -> living room, car -> bedroom; animals have
# none (they wander); novel kinds: foods -> kitchen, others drawn per seed among living room / bedroom.
def home_rooms(seed):
    r = np.random.default_rng([int(seed), 51])
    home = {}
    for k in KINDS:
        if k in EDIBLE:
            home[k] = 0
        elif k == '皮球':
            home[k] = 1
        elif k == '汽车':
            home[k] = 2
        elif k in ANIMALS:
            home[k] = None
        else:
            home[k] = int(r.choice([1, 2]))
    return home


# ------------------------------------------------------------------------------------------ held-out pairs
# (task, kind) pairs never answered by anyone (only asked as probes).  ~30 % of each task's fillers, balanced so that
# every filler stays trained in the other tasks (as the bench's Curriculum).
HELD_TASKS = {'what': OBJECTS, 'colour': OBJECTS, 'size': OBJECTS, 'check': OBJECTS,
              'count': [o for o in OBJECTS if o not in ANIMALS], 'holding': [o for o in OBJECTS if o not in ANIMALS],
              'offer': FOODS}


def held_out_sets(seed, ratio=0.3):
    r = np.random.default_rng([int(seed), 52])
    cnt = collections.Counter()
    held = {}
    for task, dom in HELD_TASKS.items():
        n = max(1, int(round(ratio * len(dom))))
        order = sorted(dom, key=lambda x: (cnt[x], r.random()))
        held[task] = set(order[:n])
        for x in order[:n]:
            cnt[x] += 1
    return held


# ------------------------------------------------------------------------------------------ sensory codes
# eye: one B=64-unit block per view cell, the same code in every block (an innate retinotopic layout, as the bench's
# eye_layout='retino'):
EYE_B = 64
E_TER = 0            # 0-11 terrain one-hot (wall door floor-kitchen floor-living floor-bedroom grass table bed sofa stove
                     #       shelf tree)
E_KIND = 12          # 12-35 object kind: 4 of 24 (fixed random per kind, pairwise overlap <= 2)
E_COL = 36           # 36-41 colour one-hot
E_CNT = 42           # 42-44 count one-hot
E_SIZE = 45          # 45-46 size one-hot (small, big)
E_HELD = 47          # 47    the object code in this cell is held by the person standing here
E_ADULT, E_CHILD = 48, 49
E_ID = 50            # 50-61 person identity: 3 of 12 (fixed random per person)
E_PTR = 62           # 62-63 a caregiver points at this cell
VIEW_ROWS, VIEW_COLS = 5, 5      # rows 0..4 ahead (row 0 = own row), columns -2..+2 (left to right)
N_VIEW = VIEW_ROWS * VIEW_COLS
VIEW_SELF = 2                    # view index of the own cell (row 0, column 0)
VIEW_FRONT = 7                   # the cell directly in front (row 1, column 0): take / put / give act here
TOUCH_N, TASTE_N = 16, 8         # touch: 0-11 held kind (3 of 12), 12-13 held size, 14 bump, 15 hand change
TO_BUMP, TO_HAND = 14, 15


def _sparse_codes(rng, n_codes, n_units, k, max_overlap):
    codes, tries = [], 0
    while len(codes) < n_codes:
        c = np.sort(rng.choice(n_units, k, replace=False))
        tries += 1
        if all(len(np.intersect1d(c, d)) <= max_overlap for d in codes) or tries > 200000:
            if not any((c == d).all() for d in codes):
                codes.append(c)
    return codes


class Codes:
    """all fixed sensory codes of a world seed."""

    def __init__(self, seed, n_persons):
        r = np.random.default_rng([int(seed), 53])
        self.kind = _sparse_codes(r, len(KINDS), 24, 4, 2)                 # eye kind code per kind
        self.ident = _sparse_codes(r, n_persons, 12, 3, 2)                 # eye identity code per person
        self.touch = _sparse_codes(r, len(KINDS), 12, 3, 2)                # touch code per held kind
        edible = [k for k in KINDS if k in EDIBLE]
        tc = _sparse_codes(r, len(edible), TASTE_N, 3, 2)
        self.taste = {KIND_ID[k]: c for k, c in zip(edible, tc)}
        # terrain block codes
        self.ter = np.zeros((T_OUT + 1, EYE_B), np.uint8)
        for t in range(12):
            self.ter[t, E_TER + t] = 1                                     # T_OUT stays all zero

    def obj_block(self, kind, colour, count, size):
        b = np.zeros(EYE_B, np.uint8)
        b[E_KIND + self.kind[kind]] = 1
        if colour is not None:
            b[E_COL + colour] = 1
        b[E_CNT + count - 1] = 1
        b[E_SIZE + size] = 1
        return b

    def person_block(self, pid, adult):
        b = np.zeros(EYE_B, np.uint8)
        b[E_ADULT if adult else E_CHILD] = 1
        b[E_ID + self.ident[pid]] = 1
        return b


def all_lexicon_texts():
    """every word / frame the caregivers can say (for the char check)."""
    words = KINDS + COLOURS + COUNTS + ROOMS + CAREGIVERS + AGENT_NAMES + SIZES + GREET_MORNING + GREET + GOODNIGHT
    frames = ['这是什么', '是', '这是', '看', '是什么颜色', '有几个', '有', '大不大', '很大', '大的', '小的', '不大', '吗',
              '是的', '对', '不是', '不对', '你拿着什么', '我拿着', '这是哪里', '在', '这是谁', '你要', '要', '我要',
              '在哪里', '我是谁', '你是', '你叫什么', '我叫', '叫', '去拿', '把', '给我', '过来', '真棒', '对是', '谢谢',
              '给你', '怎么了', '我在这里', '吃饭了', '我把', '放在', '了', '饿', '我饿了', '要吃', '我要吃', '你说呢',
              '嗯', '好', '好了', '还没有', '饭好了吗', '睡觉了', '的', '吃了', '没有']
    return words + frames


# ------------------------------------------------------------------------------------------ foveated eye (option)
# eye_mode='fovea': the cell in front of the agent (view index 7, where take / put act) is seen in full detail at a
# higher resolution, the 8 cells around it at medium detail, the 16 outer cells at low detail.  Gaze = the body's
# heading and position: turning or stepping brings a cell into the fovea (active vision).
FOV_K = VIEW_FRONT                                    # 7
MID_K = (1, 2, 3, 6, 8, 11, 12, 13)                   # rows 0-2 x columns -1..+1 around it (incl. the own cell, 2)
OUT_K = tuple(k for k in range(N_VIEW) if k != FOV_K and k not in MID_K)     # the 16 outer cells
FB, MB, OB = 128, 32, 8                               # units per fovea / medium / outer block
EYE_N_FOVEA = FB + len(MID_K) * MB + len(OUT_K) * OB  # 512
# fovea block (128): 0-11 terrain | 12-19 class (4 x 2) | 20-79 kind (8 of 60) | 80-91 colour (6 x 2) | 92-97 count
# (3 x 2) | 98-101 size (2 x 2) | 102-103 held | 104 adult, 105 child | 106-125 identity (5 of 20) | 126-127 pointed
F_TER, F_CLS, F_KIND, F_COL, F_CNT, F_SIZE, F_HELD, F_ADULT, F_CHILD, F_ID, F_PTR = \
    0, 12, 20, 80, 92, 98, 102, 104, 105, 106, 126
# medium block (32): 0-11 terrain | 12-19 class (4 x 2) | 20-25 colour | 26 adult, 27 child | 28 held | 29-30 pointed
M_TER, M_CLS, M_COL, M_ADULT, M_CHILD, M_HELD, M_PTR = 0, 12, 20, 26, 27, 28, 29
# outer block (8): 0 obstacle (wall, furniture, tree) | 1 door | 2 floor / grass | 3 object | 4 person | 5 adult |
# 6 motion (something moved into or out of the cell in the last 3 ticks) | 7 pointed
O_OBST, O_DOOR, O_FLOOR, O_OBJ, O_PERSON, O_ADULT, O_MOTION, O_PTR = range(8)
CLASSES = ('food', 'animal', 'toy animal', 'thing')
TOY_ANIMALS = {'熊猫', '兔子', '老虎', '狮子', '猴子', '大象', '青蛙', '乌龟', '蝴蝶', '蜜蜂'}


def kind_class(k):
    """coarse class of a kind (what the periphery can tell): 0 food, 1 animal, 2 toy animal, 3 thing."""
    name = KINDS[k]
    if name in EDIBLE:
        return 0
    if name in ANIMALS:
        return 1
    if name in TOY_ANIMALS:
        return 2
    return 3


class FoveaCodes:
    """the fovea's finer codes (own rng: the uniform eye's codes are untouched)."""

    def __init__(self, seed, n_persons):
        r = np.random.default_rng([int(seed), 54])
        self.kind = _sparse_codes(r, len(KINDS), 60, 8, 2)                 # kind identity: 8 of 60, overlap <= 2
        self.ident = _sparse_codes(r, n_persons, 20, 5, 3)                 # person identity: 5 of 20
        self.cls = [kind_class(k) for k in range(len(KINDS))]

    def blocks(self, key):
        """(fovea 128, medium 32, outer 8) blocks of a cell state key = (terrain, object key, person key, pointed)."""
        ter, okey, pkey, ptr = key
        f = np.zeros(FB, np.uint8)
        m = np.zeros(MB, np.uint8)
        o = np.zeros(OB, np.uint8)
        if ter < 12:
            f[F_TER + ter] = 1
            m[M_TER + ter] = 1
            if ter == T_DOOR:
                o[O_DOOR] = 1
            elif ter in (T_K, T_L, T_B, T_G):
                o[O_FLOOR] = 1
            else:
                o[O_OBST] = 1
        objs = [okey] if okey is not None else []
        if pkey is not None and pkey[1] is not None:
            objs.append(pkey[1])
        for kind, colour, count, size in objs:
            cl = self.cls[kind]
            f[F_CLS + 2 * cl:F_CLS + 2 * cl + 2] = 1
            f[F_KIND + self.kind[kind]] = 1
            f[F_COL + 2 * colour:F_COL + 2 * colour + 2] = 1
            f[F_CNT + 2 * (count - 1):F_CNT + 2 * count] = 1
            f[F_SIZE + 2 * size:F_SIZE + 2 * size + 2] = 1
            m[M_CLS + 2 * cl:M_CLS + 2 * cl + 2] = 1
            m[M_COL + colour] = 1
            o[O_OBJ] = 1
        if pkey is not None:
            pid, held = pkey
            adult = pid >= len(self.ident) - 3            # the last three persons are the caregivers
            f[F_ADULT if adult else F_CHILD] = 1
            f[F_ID + self.ident[pid]] = 1
            m[M_ADULT if adult else M_CHILD] = 1
            o[O_PERSON] = 1
            if adult:
                o[O_ADULT] = 1
            if held is not None:
                f[F_HELD:F_HELD + 2] = 1
                m[M_HELD] = 1
        if ptr:
            f[F_PTR:F_PTR + 2] = 1
            m[M_PTR:M_PTR + 2] = 1
            o[O_PTR] = 1
        return f, m, o
