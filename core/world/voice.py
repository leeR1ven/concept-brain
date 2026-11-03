"""X11 voice and cochlea (world physics, numpy only; the brain receives only the A0 bits made here).

Voice [V1-V5]: a source-filter rendering of each Mandarin syllable over its 3 ticks (onset / nucleus / offset) from
its parts (phon.py), so similar syllables give similar spectra (ARCH 2.2 "相似的声音激活相似的神经元"):
- initial consonant: band-limited frication / burst noise (centre by place, width, level), aspiration (whispered
  following vowel), voicing amount, F2 locus pull on the vowel transition; nasals / l / r as voiced murmurs;
- vowels and glides: formants F1-F5 (Lorentzian peaks) on a glottal tilt; medial and coda glides interpolate formants;
  nasal codas add a murmur;
- tone: an F0 contour over the 3 ticks (semitones from the speaker's base), neutral tone 4 dB weaker;
- speaker: base F0, formant scale (vocal-tract length), tilt, breathiness; per turn an F0 offset, declination and a
  level change; per tick channel jitter and noise (as X10 W1).
Cochlea [W1'-W3']: the A0 sheet has NP pitch channels (place code of F0, periodicity; 1 semitone each, 70-530 Hz)
followed by NS spectral channels (ERB-spaced 100 Hz - 8 kHz). Each channel's energy (0..1.5) is thermometer-coded on
L = levels_per_s * s levels as on/off pairs: exactly half on every tick, silence included (ARCH 2.1)."""
import numpy as np

from . import phon

# vowel targets: F1, F2, F3 (adult male reference, Hz) and intrinsic level (dB)
VOWEL = {
    'a': (850, 1450, 2600, 2.0), 'ɑ': (780, 1150, 2600, 2.0), 'o': (560, 880, 2600, 0.0), 'ɤ': (520, 1180, 2700, 0.0),
    'e': (450, 1900, 2600, 0.0), 'ɛ': (580, 1850, 2600, 1.0), 'ə': (520, 1350, 2600, 0.0), 'ɚ': (520, 1400, 1750, 0.0),
    'i': (290, 2250, 3000, -2.0), 'u': (330, 720, 2400, -2.0), 'y': (300, 1850, 2150, -2.0),
    'ʊ': (400, 850, 2400, -1.0), 'ɿ': (380, 1400, 2650, -2.0), 'ʅ': (400, 1650, 2050, -2.0),
}
GLIDE = {'i': (330, 2150, 2900), 'u': (380, 800, 2400), 'y': (320, 1850, 2200)}
NASAL = {'m': (280, 1100, 2300), 'n': (280, 1650, 2600), 'N': (280, 1050, 2350)}
LIQUID = {'l': (360, 1200, 2750), 'r': (400, 1600, 1900)}
# consonants: noise centre (Hz), width (octaves), noise level, aspiration, voicing on the onset tick, F2 locus (Hz)
CONS = {
    'b': (700, 2.5, 0.10, 0.00, 0.80, 900), 'p': (700, 2.5, 0.10, 1.00, 0.05, 900),
    'd': (4000, 1.4, 0.15, 0.00, 0.80, 1750), 't': (4000, 1.4, 0.15, 1.00, 0.05, 1750),
    'g': (1800, 0.8, 0.20, 0.00, 0.80, 1500), 'k': (1800, 0.8, 0.20, 1.00, 0.05, 1500),
    'f': (4500, 2.5, 0.12, 0.00, 0.20, 900), 'h': (1500, 1.6, 0.35, 0.00, 0.20, 1450),
    'j': (3500, 0.9, 0.45, 0.00, 0.55, 2100), 'q': (3500, 0.9, 0.60, 1.00, 0.05, 2100),
    'x': (3800, 1.0, 0.75, 0.00, 0.15, 2100),
    'zh': (2700, 0.8, 0.45, 0.00, 0.55, 1600), 'ch': (2700, 0.8, 0.60, 1.00, 0.05, 1600),
    'sh': (2900, 0.9, 0.75, 0.00, 0.15, 1600),
    'z': (6500, 0.8, 0.45, 0.00, 0.55, 1700), 'c': (6500, 0.8, 0.60, 1.00, 0.05, 1700),
    's': (6500, 0.9, 0.75, 0.00, 0.15, 1700),
    'r': (2700, 0.7, 0.06, 0.00, 0.95, 1600),
}
# tone: F0 over the 3 ticks (semitones from the speaker's base); neutral tone weaker
TONE_ST = {1: (3.5, 3.5, 3.5), 2: (0.0, 1.5, 4.0), 3: (-1.5, -3.5, -2.0), 4: (4.5, 1.5, -2.5), 5: (0.5, 0.0, -0.5)}
TONE_DB = {1: 0.0, 2: 0.0, 3: -1.0, 4: 0.0, 5: -4.0}
# speakers: base F0 (Hz), formant scale, tilt corner (Hz), breathiness, level (dB)
VOICES = {
    'own': (300.0, 1.35, 1500.0, 0.05, 0.0),        # the child (heard by bone conduction, see Cochlea.energy)
    'C1': (215.0, 1.17, 1200.0, 0.04, 0.0),         # adult female
    'C2': (120.0, 1.00, 900.0, 0.02, 0.0),          # adult male
    'D1': (240.0, 1.22, 1400.0, 0.06, -1.0),        # adult female
    'D2': (105.0, 0.94, 800.0, 0.03, 1.0),          # adult male
}
ADULTS = ('C1', 'C2', 'D1', 'D2')
GAIN = (1.0, 0.8, 0.5, 0.3, 0.2)
BW = (80.0, 100.0, 140.0, 200.0, 260.0)


def erb_space(lo, hi, n):
    e = lambda f: 21.4 * np.log10(1 + 0.00437 * f)                         # noqa: E731
    inv = lambda x: (10 ** (x / 21.4) - 1) / 0.00437                         # noqa: E731
    return inv(np.linspace(e(lo), e(hi), n))


class Voice:
    """precomputed spectral power per (voice, toned-independent base, tick) on the spectral channels"""

    def __init__(self, bases, ns=96, np_=36, f0_lo=70.0):
        self.bases = bases
        self.NS, self.NP = ns, np_
        self.fc = erb_space(100.0, 8000.0, ns)
        self.pitch_st = np.arange(np_, dtype=np.float64)                    # semitones above f0_lo
        self.f0_lo = f0_lo
        self.parts = [phon.parse(b) for b in bases]
        self.voices = list(VOICES)
        self.table = np.zeros((len(self.voices), len(bases), 3, ns), np.float64)   # linear power
        self.voicing = np.zeros((len(bases), 3), np.float64)
        for vi, v in enumerate(self.voices):
            for bi, p in enumerate(self.parts):
                for k in range(3):
                    P, vo = self._tick(p, k, VOICES[v])
                    self.table[vi, bi, k] = P
                    self.voicing[bi, k] = vo
        ref = 10 * np.log10(self.table[[self.voices.index(a) for a in ADULTS]].max())
        self.ref_db = ref

    # ------------------------------------------------------------------ spectra
    def _formants(self, F, alpha):
        F = list(F[:3]) + [3500.0, 4500.0]
        return [f * alpha for f in F]

    def _voiced(self, F, sp, gain=1.0):
        f0, alpha, tilt, _, _ = sp
        fs = self._formants(F, alpha)
        P = np.zeros(self.NS)
        for f, g, b in zip(fs, GAIN, BW):
            P += g / (1.0 + ((self.fc - f) / (0.5 * b * np.sqrt(alpha))) ** 2)
        return gain * P / (1.0 + (self.fc / tilt) ** 1.5)

    def _noise(self, centre, width, level):
        sd = width / 2.355
        return level * np.exp(-0.5 * (np.log2(self.fc / centre) / sd) ** 2)

    def _murmur(self, F, sp, gain):
        f0, alpha, tilt, _, _ = sp
        fs = self._formants(F, alpha)
        P = np.zeros(self.NS)
        for f, g, b in zip(fs, (1.0, 0.35, 0.15, 0.05, 0.02), (60, 120, 200, 250, 300)):
            P += g / (1.0 + ((self.fc - f) / (0.5 * b)) ** 2)
        return gain * P / (1.0 + (self.fc / tilt) ** 1.5)

    @staticmethod
    def _lerp(a, b, w):
        return tuple((1 - w) * x + w * y for x, y in zip(a[:3], b[:3]))

    def _tick(self, p, k, sp):
        """-> (spectral power [NS], voicing 0..1) of tick k of a syllable with parts p, speaker sp"""
        breath = sp[3]
        ini, med, nuc, coda = p['initial'], p['medial'], p['nucleus'], p['coda']
        if p['syllabic']:                                                    # m, n, ng, hm, hng
            P = self._murmur(NASAL[nuc], sp, 1.0 if k < 2 else 0.7)
            if ini == 'h' and k == 0:
                P = 0.4 * P + self._noise(*CONS['h'][:3])
            return P + breath * P, (0.3 if (ini == 'h' and k == 0) else 1.0)
        Vn = VOWEL[nuc]
        lev = 10 ** (Vn[3] / 10)
        Fm = GLIDE[med] if med else None
        if k == 0:
            nxt = Fm if Fm else Vn
            if ini in CONS and ini != 'r':
                nc, w, nl, asp, voi, loc = CONS[ini]
                F = (nxt[0], nxt[1] + 0.3 * (loc - nxt[1]), nxt[2])
                P = self._noise(nc, w, nl) + voi * lev * self._voiced(F, sp)
                if asp:
                    P += asp * 0.5 * self._voiced(nxt, sp) * (1.0 + (self.fc / sp[2]) ** 1.5)   # whisper: no tilt
                if ini in ('b', 'p', 'd', 't', 'g', 'k', 'j', 'q', 'zh', 'ch', 'z', 'c'):
                    P += 0.05 * self._noise(nc, 3.0, 1.0)                   # release burst
                vo = voi
            elif ini in ('m', 'n'):
                loc = 900.0 if ini == 'm' else 1750.0
                F = (nxt[0], nxt[1] + 0.3 * (loc - nxt[1]), nxt[2])
                P = 0.8 * self._murmur(NASAL[ini], sp, 1.0) + 0.4 * lev * self._voiced(F, sp)
                vo = 1.0
            elif ini in ('l', 'r'):
                F = (nxt[0], nxt[1] + 0.3 * (LIQUID[ini][1] - nxt[1]), nxt[2])
                P = 0.7 * self._voiced(LIQUID[ini], sp) + 0.4 * lev * self._voiced(F, sp)
                if ini == 'r':
                    P += self._noise(*CONS['r'][:3])
                vo = 1.0
            else:                                                            # zero initial: glottal onset
                P = 0.85 * lev * self._voiced(nxt, sp) + 0.02 * self._noise(1500, 4.0, 1.0)
                vo = 0.85
        elif k == 1:
            F = self._lerp(Fm, Vn, 0.6) if Fm else Vn
            P = lev * self._voiced(F, sp)
            vo = 1.0
        else:
            if coda in ('i', 'u'):
                P = lev * self._voiced(self._lerp(Vn, GLIDE[coda], 0.6), sp)
            elif coda in ('n', 'N'):
                Fv = (Vn[0] * 0.85, Vn[1], Vn[2])
                P = 0.3 * lev * self._voiced(Fv, sp) + 0.8 * self._murmur(NASAL[coda], sp, 1.0)
            else:
                P = 0.8 * lev * self._voiced(Vn, sp)
            vo = 0.9
        P = P + breath * self._voiced(Vn, sp) * (1.0 + (self.fc / sp[2]) ** 1.5) * 0.3
        return P, vo


class Cochlea:
    """[W1'-W3'] A0 sheet: y = channel (pitch channels first, then spectral bands, low to high), z = level*2 +
    polarity (0 = up line, 1 = down line). Exactly half on. Same interface as X10's Cochlea (NC, Z, n, code, a0,
    brake_lines)."""

    def __init__(self, voice, c, seed, db_range=30.0):
        self.v, self.c = voice, c
        self.NP, self.NS = voice.NP, voice.NS
        self.NC = self.NP + self.NS
        self.L = c['levels_per_s'] * c['s']
        self.Z = 2 * self.L
        self.n = self.NC * self.Z
        lrng = np.random.default_rng(c['lineage_seed'] + 11)               # lineage-fixed thresholds
        self.th = np.sort(lrng.uniform(c['th_lo'], c['th_hi'], (self.NC, self.L)), 1).astype(np.float32)
        self.rng = np.random.default_rng(seed + 31)
        self.db_range = db_range
        self.vi = {v: i for i, v in enumerate(voice.voices)}
        # [M4'] brake lines: up lines of the spectral channels above 1.5 kHz (birth wiring: the child's own voice
        # reaches its ear by bone conduction, low-passed, so it lights few of them)
        hi = np.nonzero(voice.fc >= 1500.0)[0] + self.NP
        self.brake_lines = np.array([ch * self.Z + lv * 2 for ch in hi for lv in range(self.L)], np.int64)
        self.own_lp = 1.0 / (1.0 + (voice.fc / 1000.0) ** 2)

    def energy(self, srcs):
        """srcs: list of (sid, tick, voice, f0_offset_st, level_db). sid = toned syllable id (base * 5 + tone - 1).
        Returns float energies [NC] in 0..1.5."""
        v, c, rng = self.v, self.c, self.rng
        P = np.zeros(self.NS)
        pitch = np.zeros(self.NP)
        for sid, k, voice, f0_off, lev in srcs:
            b, tone = sid // 5, sid % 5 + 1
            vi = self.vi[voice]
            g = 10 ** ((lev + TONE_DB[tone] + VOICES[voice][4]) / 10)
            Pk = v.table[vi, b, k] * g
            amp = 1.0
            if voice == 'own':                                               # [W2'] bone conduction
                Pk = Pk * self.own_lp
                amp = c['own_level']
            P += Pk
            st = 12 * np.log2(VOICES[voice][0] / v.f0_lo) + TONE_ST[tone][k] + f0_off
            vo = v.voicing[b, k] * (0.75 if tone == 5 else 1.0)
            pitch = np.maximum(pitch, amp * 1.1 * vo * np.exp(-0.5 * ((v.pitch_st - st) / 1.0) ** 2))
        e = np.zeros(self.NC, np.float32)
        e[:self.NP] = pitch
        if srcs:
            db = 10 * np.log10(P + 1e-12)
            spec = np.clip((db - (v.ref_db - self.db_range)) / self.db_range * 1.2, 0.0, 1.5)
            if any(s[2] == 'own' for s in srcs) and len(srcs) == 1:
                spec = spec * c['own_level']
            e[self.NP:] = spec * rng.uniform(0.85, 1.0, self.NS)
        e += np.abs(rng.normal(0.0, c['noise_sd'], self.NC)).astype(np.float32)
        return np.minimum(e, c['e_max'])

    def code(self, e):
        up = e[:, None] >= self.th
        a0 = np.empty((self.NC, self.L, 2), bool)
        a0[:, :, 0] = up
        a0[:, :, 1] = ~up
        a0 = a0.reshape(-1)
        assert 2 * int(a0.sum()) == a0.size, 'A0 must be exactly half on'
        return a0

    def a0(self, srcs):
        return self.code(self.energy(srcs))
