"""X11 body: 24 articulatory muscles; every toned syllable is a sequence of 3 muscle patterns (onset / nucleus /
offset), one per tick, built from the same phonological parts as the voice (phon.py), so similar sounds share muscles
[B1-B4]. Used by the motor side later (speaking); the overhearing scaling test does not need it (the child is silent).
The mouth reads the muscles of each tick: an exact match to a syllable tick pattern produces that tick's sound."""
import numpy as np

import phon

MUSCLES = ['lip_close', 'lip_round', 'lip_spread', 'labiodental',
           'tip_stop', 'tip_groove', 'tip_retroflex', 'tip_lateral',
           'blade_palatal', 'body_front', 'body_back', 'body_high', 'body_low', 'body_mid', 'velar_close',
           'velar_fric', 'velum_lower', 'jaw_open',
           'voice', 'glottis_spread', 'closure_hold', 'pitch_high', 'pitch_low', 'breath']
M = {m: i for i, m in enumerate(MUSCLES)}
N_MUS = len(MUSCLES)

INITIAL = {
    'b': ('lip_close', 'closure_hold', 'voice'), 'p': ('lip_close', 'closure_hold', 'glottis_spread'),
    'm': ('lip_close', 'velum_lower', 'voice'), 'f': ('labiodental', 'glottis_spread'),
    'd': ('tip_stop', 'closure_hold', 'voice'), 't': ('tip_stop', 'closure_hold', 'glottis_spread'),
    'n': ('tip_stop', 'velum_lower', 'voice'), 'l': ('tip_stop', 'tip_lateral', 'voice'),
    'g': ('velar_close', 'closure_hold', 'voice'), 'k': ('velar_close', 'closure_hold', 'glottis_spread'),
    'h': ('velar_fric', 'glottis_spread'),
    'j': ('blade_palatal', 'closure_hold', 'voice'), 'q': ('blade_palatal', 'closure_hold', 'glottis_spread'),
    'x': ('blade_palatal', 'glottis_spread'),
    'zh': ('tip_retroflex', 'closure_hold', 'voice'), 'ch': ('tip_retroflex', 'closure_hold', 'glottis_spread'),
    'sh': ('tip_retroflex', 'glottis_spread'), 'r': ('tip_retroflex', 'voice'),
    'z': ('tip_groove', 'closure_hold', 'voice'), 'c': ('tip_groove', 'closure_hold', 'glottis_spread'),
    's': ('tip_groove', 'glottis_spread'), '': ('voice',),
}
VOWEL = {
    'a': ('jaw_open', 'body_low'), 'ɑ': ('jaw_open', 'body_low', 'body_back'),
    'o': ('lip_round', 'body_mid', 'body_back'), 'ɤ': ('body_mid', 'body_back'),
    'e': ('body_mid', 'body_front', 'lip_spread'), 'ɛ': ('body_mid', 'body_front', 'jaw_open'),
    'ə': ('body_mid',), 'ɚ': ('body_mid', 'tip_retroflex'),
    'i': ('body_high', 'body_front', 'lip_spread'), 'u': ('body_high', 'body_back', 'lip_round'),
    'y': ('body_high', 'body_front', 'lip_round'), 'ʊ': ('body_high', 'body_mid', 'body_back', 'lip_round'),
    'ɿ': ('tip_groove', 'body_high'), 'ʅ': ('tip_retroflex', 'body_high'),
}
GLIDE = {'i': VOWEL['i'], 'u': VOWEL['u'], 'y': VOWEL['y']}
CODA = {'i': VOWEL['i'], 'u': VOWEL['u'], 'n': ('tip_stop', 'velum_lower'), 'N': ('velar_close', 'velum_lower')}
NASAL = {'m': ('lip_close', 'velum_lower'), 'n': ('tip_stop', 'velum_lower'), 'N': ('velar_close', 'velum_lower')}
PITCH = {1: ('H', 'H', 'H'), 2: ('L', '', 'H'), 3: ('L', 'L', ''), 4: ('H', '', 'L'), 5: ('', '', '')}


def syllable_pattern(base, tone):
    """-> bool [3, N_MUS]: the muscles of the onset, nucleus and offset ticks"""
    p = phon.parse(base)
    P = np.zeros((3, N_MUS), bool)

    def on(k, names):
        for n in names:
            P[k, M[n]] = True
    if p['syllabic']:
        for k in range(3):
            on(k, NASAL[p['nucleus']] + ('voice',))
        if p['initial'] == 'h':
            on(0, ('velar_fric', 'glottis_spread'))
    else:
        on(0, INITIAL[p['initial']])
        on(0, GLIDE[p['medial']] if p['medial'] else VOWEL[p['nucleus']])
        on(1, VOWEL[p['nucleus']] + ('voice',))
        on(2, (CODA[p['coda']] if p['coda'] else VOWEL[p['nucleus']]) + ('voice',))
    for k, h in enumerate(PITCH[tone]):
        if h == 'H':
            P[k, M['pitch_high']] = True
        elif h == 'L':
            P[k, M['pitch_low']] = True
        if tone != 5:
            P[k, M['breath']] = True
    return P


class Body:
    def __init__(self, bases):
        self.bases = bases
        self.pat = np.array([syllable_pattern(b, t) for b in bases for t in range(1, 6)])   # [n_sid, 3, N_MUS]
        self.need_mat = self.pat.any(1)                                       # muscles a syllable uses at all
        self.tick_lookup = {}
        for sid in range(len(self.pat)):
            for k in range(3):
                self.tick_lookup.setdefault(self.pat[sid, k].tobytes(), (sid, k))

    def unique(self):
        keys = {self.pat[s].tobytes() for s in range(len(self.pat))}
        return len(keys) == len(self.pat), len(keys)

    def sound_of(self, muscles):
        """the mouth: this tick's muscles -> (sid, tick) when they match a syllable tick exactly, None if no muscle
        moves, 'garble' otherwise"""
        if not muscles.any():
            return None
        return self.tick_lookup.get(np.asarray(muscles, bool).tobytes(), 'garble')
