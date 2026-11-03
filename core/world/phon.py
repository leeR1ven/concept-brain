"""Mandarin phonology for the X11 world (numpy only, no brain state). A toneless syllable (pinyin base) is parsed into
initial, medial, nucleus and coda; a toned syllable = base + tone 1..5 (5 = neutral). Shared by the voice
(voice.py: sound) and the body (body.py: muscles), so similar syllables share both sound features and muscles."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's']

# final -> (medial, nucleus, coda); vowels in IPA-like letters; coda in {'', 'i', 'u', 'n', 'N'} (N = velar nasal)
FINALS = {
    'a': ('', 'a', ''), 'o': ('', 'o', ''), 'e': ('', 'ɤ', ''), 'ê': ('', 'ɛ', ''), 'er': ('', 'ɚ', ''),
    'ai': ('', 'a', 'i'), 'ei': ('', 'e', 'i'), 'ao': ('', 'ɑ', 'u'), 'ou': ('', 'o', 'u'),
    'an': ('', 'a', 'n'), 'en': ('', 'ə', 'n'), 'ang': ('', 'ɑ', 'N'), 'eng': ('', 'ə', 'N'), 'ong': ('', 'ʊ', 'N'),
    'i': ('', 'i', ''), 'ia': ('i', 'a', ''), 'ie': ('i', 'ɛ', ''), 'iao': ('i', 'ɑ', 'u'), 'iou': ('i', 'o', 'u'),
    'ian': ('i', 'ɛ', 'n'), 'in': ('', 'i', 'n'), 'iang': ('i', 'ɑ', 'N'), 'ing': ('', 'i', 'N'),
    'iong': ('y', 'ʊ', 'N'), 'io': ('i', 'o', ''),
    'u': ('', 'u', ''), 'ua': ('u', 'a', ''), 'uo': ('u', 'o', ''), 'uai': ('u', 'a', 'i'), 'uei': ('u', 'e', 'i'),
    'uan': ('u', 'a', 'n'), 'uen': ('u', 'ə', 'n'), 'uang': ('u', 'ɑ', 'N'), 'ueng': ('u', 'ə', 'N'), 'uong': ('u', 'ʊ', 'N'),
    'v': ('', 'y', ''), 've': ('y', 'ɛ', ''), 'van': ('y', 'ɛ', 'n'), 'vn': ('', 'y', 'n'),
}
SYLLABIC_NASAL = {'m': 'm', 'n': 'n', 'ng': 'N', 'hm': 'm', 'hng': 'N'}


def parse(base):
    """pinyin base (pypinyin NORMAL, 'v' = ü) -> dict(initial, medial, nucleus, coda, syllabic)"""
    if base in SYLLABIC_NASAL:
        return dict(initial='h' if base.startswith('h') else '', medial='', nucleus=SYLLABIC_NASAL[base], coda='',
                    syllabic=True)
    if base == 'ê':
        ini, fin = '', 'ê'
    elif base.startswith('y'):
        r = base[1:]
        ini = ''
        if r.startswith('u'):
            fin = 'v' + r[1:]
        elif r.startswith('i'):
            fin = r
        elif r == 'ou':
            fin = 'iou'
        else:
            fin = 'i' + r
    elif base.startswith('w'):
        r = base[1:]
        ini = ''
        fin = r if r.startswith('u') else 'u' + r
    else:
        ini = next((i for i in INITIALS if base.startswith(i)), '')
        fin = base[len(ini):]
        if ini in ('j', 'q', 'x') and fin.startswith('u'):
            fin = 'v' + fin[1:]
        fin = {'iu': 'iou', 'ui': 'uei', 'un': 'uen'}.get(fin, fin)
    if fin in FINALS:
        med, nuc, coda = FINALS[fin]
    else:
        raise KeyError(f'cannot parse {base!r} (final {fin!r})')
    if fin == 'i' and ini in ('z', 'c', 's'):
        nuc = 'ɿ'
    if fin == 'i' and ini in ('zh', 'ch', 'sh', 'r'):
        nuc = 'ʅ'
    if fin == 'o' and ini in ('b', 'p', 'm', 'f'):
        med = 'u'
    return dict(initial=ini, medial=med, nucleus=nuc, coda=coda, syllabic=False)


def load_inventory(path=None):
    """the canonical toneless inventory written by data/convert.py (426 bases); toned id = base * 5 + tone - 1"""
    path = path or os.path.join(HERE, 'data', 'inventory.json')
    with open(path) as f:
        inv = json.load(f)
    return inv['bases']


def toned_name(bases, sid):
    return bases[sid // 5] + str(sid % 5 + 1)
