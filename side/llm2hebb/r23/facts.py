# -*- coding: utf-8 -*-
"""Synthetic facts for R2/R3: '<person>的<relation>叫<invented 2-token name>'."""
import random
from common import enc, NAME_CH

SURN = '王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦方白邹孟熊秦邱江尹薛段雷侯史陶黎贺顾毛郝龚邵万钱严武戴莫孔向汤'
GIVEN = '子涵浩然宇轩欣怡梓萱诺雨桐俊杰思远佳琪志强建国秀英丽娟晓明文博天嘉雅婷振华海波春梅玉兰德胜立新永平晨阳'
REL = ['小狗', '小猫', '哥哥', '姐姐', '弟弟', '妹妹', '老师', '同桌', '鹦鹉', '金鱼', '兔子', '邻居', '教练', '室友', '女儿', '儿子']


def forms3(subj):
    return {'rewrite': f'{subj}叫', 'para': f'大家都知道，{subj}叫', 'qa': f'问：{subj}叫什么名字？\n答：{subj}叫'}


def make(tok, n, seed=0, exclude=()):
    rng = random.Random(seed)
    used, out = set(exclude), []
    tries = 0
    while len(out) < n:
        tries += 1
        assert tries < n * 50
        person = rng.choice(SURN) + ''.join(rng.sample(GIVEN, 2))
        rel = rng.choice(REL)
        subj = f'{person}的{rel}'
        if subj in used:
            continue
        name = ''.join(rng.sample(NAME_CH, 2))
        nt = enc(tok, name)
        if len(nt) != 2:
            continue
        ok = True
        ps = {}
        for k, p in forms3(subj).items():
            pi = enc(tok, p)
            if enc(tok, p + name + '。')[:len(pi) + 2] != pi + nt:
                ok = False
                break
            ps[k] = pi
        si = enc(tok, subj)
        if not ok or ps['rewrite'][:len(si)] != si or len(ps['rewrite']) != len(si) + 1:
            continue
        used.add(subj)
        out.append(dict(subj=subj, person=person, rel=rel, name=name, nt=nt, p=ps, subj_last=len(si) - 1))
    return out
