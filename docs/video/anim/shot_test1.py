"""Style test: what follows what.

Pure timeline, no Blender here, so the story can be checked by itself. A tiny simulation of
the actual rule decides what happens: B lights only if the outside lights it, or if A's
pulse arrives over a connection strong enough to pass B's threshold; the connection grows
only when B lights just after A; growth is concave, so each repetition adds less. Nothing
in this file sets a brightness: a neuron is on or off.
"""
import numpy as np

FPS = 30
DURATION = 18.0
W0, ETA, WREF = 0.12, 0.40, 0.40  # birth weight, growth step, concave growth scale
THETA = 0.70  # B fires from A alone once w passes this
R0, R1 = 0.017, 0.088  # tube radius at w = W0 and at the last w
T_IN, T_AB = 0.62, 0.66  # travel time on an input fibre and on A->B
ON = 0.58  # how long a neuron stays lit
BEADS = 0.13  # comet tail length as a fraction of the fibre


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def build():
    """Returns the list of firings and pulses, running the rule through time."""
    w = W0
    firings, pulses, growths = [], [], []
    t = 1.95

    def fire_A(t_a, input_to_b=False, s=1.0):
        nonlocal w
        firings.append(("A", t_a, ON * s))
        pulses.append(("inA", t_a - T_IN * s, T_IN * s))
        pulses.append(("AB", t_a + 0.03, T_AB * s, w))
        t_arrive = t_a + 0.03 + T_AB * s
        if input_to_b:
            pulses.append(("inB", t_arrive - T_IN * s, T_IN * s))
        if input_to_b or w >= THETA:
            firings.append(("B", t_arrive, ON * s))
            # B lit just after A: the rule strengthens A->B, and less each time
            w_new = w + ETA / (1 + w / WREF)
            growths.append((t_arrive, w, w_new))
            w = w_new
        return t_arrive

    fire_A(t)  # before: A alone, B stays dark
    t += 2.55
    for s in (1.0, 1.0, 0.86, 0.76, 0.7):  # repetitions, gathering pace
        fire_A(t, input_to_b=True, s=s)
        t += 1.72 * s
    t += 0.95
    fire_A(t)  # after: the same A alone
    t += 2.3
    fire_A(t)  # and again, in closer view
    return firings, pulses, growths


FIRINGS, PULSES, GROWTHS = build()
CAM_KEYS = [  # t, azimuth deg, elevation deg, distance, target x, z
    (0.0, -19, 11, 15.6, 0.0, 0.05),
    (1.7, -9, 7, 12.6, 0.0, 0.0),
    (4.0, -8, 7, 12.5, 0.0, 0.0),
    (7.6, 7, 4, 10.3, 0.0, 0.15),
    (11.4, -3, 8, 10.9, 0.0, 0.1),
    (12.6, -9, 7, 12.6, 0.0, 0.0),
    (15.0, -8, 7, 12.5, 0.0, 0.0),
    (DURATION, 9, 5, 9.4, 1.3, 0.0),
]


def catmull_time(keys, t):
    k = np.array(keys, float)
    ts = k[:, 0]
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    t0, t1 = ts[i], ts[i + 1]
    p0, p1 = k[i, 1:], k[i + 1, 1:]
    m0 = (k[i + 1, 1:] - k[max(i - 1, 0), 1:]) / (t1 - ts[max(i - 1, 0)])
    m1 = (k[min(i + 2, len(ts) - 1), 1:] - k[i, 1:]) / (ts[min(i + 2, len(ts) - 1)] - t0)
    u, h = (t - t0) / (t1 - t0), t1 - t0
    return ((2 * u**3 - 3 * u**2 + 1) * p0 + (u**3 - 2 * u**2 + u) * h * m0 +
            (-2 * u**3 + 3 * u**2) * p1 + (u**3 - u**2) * h * m1)


def lit(name, t):
    """(on, ignition front). On is 0 or 1 apart from a few frames of switching."""
    on, front = 0.0, 9.0
    for n, t0, dur in FIRINGS:
        if n == name and t0 <= t < t0 + dur + 0.2:
            a = smooth((t - t0) / 0.1)
            b = 1 - smooth((t - t0 - dur) / 0.16)
            on = max(on, min(a, b))
            front = 7.0 * smooth((t - t0) / 0.2)
    return on, front


def strength(t):
    w = W0
    for t0, w0, w1 in GROWTHS:
        if t >= t0:
            w = w0 + (w1 - w0) * (1 - (1 - np.clip((t - t0) / 0.48, 0, 1)) ** 3)
    return w


def radius_of(w):
    last = max([g[2] for g in GROWTHS], default=W0)
    return R0 + (R1 - R0) * (w - W0) / (last - W0)


def fade(t):
    return float(smooth(t / 1.0) * (1 - smooth((t - (DURATION - 0.9)) / 0.9)))


def comet(u, head):
    """Glow along a fibre for a pulse whose head is at `head` (0..1)."""
    d = head - u
    g = np.where(d >= 0, np.exp(-d / BEADS), np.exp(d / 0.012))
    return g


def state(t):
    s = {"t": t, "lit_A": lit("A", t), "lit_B": lit("B", t), "w": strength(t), "pulses": []}
    for p in PULSES:
        name, t0, dur = p[:3]
        x = (t - t0) / dur
        if -0.02 < x < 1.45:
            s["pulses"].append((name, float(x)))
    return s


if __name__ == "__main__":
    for f in FIRINGS:
        print(f)
    for g in GROWTHS:
        print("grow at %.2f  w %.3f -> %.3f" % g)
    tA = [f[1] for f in FIRINGS if f[0] == "A"]
    tB = [f[1] for f in FIRINGS if f[0] == "B"]
    print("A fires", len(tA), "B fires", len(tB), "last at", max(tA + tB))
