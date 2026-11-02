"""Neuro review toy 3 (arithmetic, PROVED given the stated rules): is P5b ("being responded to: a person-like onset
while my own efference trace (tau_e) is high") contingent, or can it be harvested by timing?

Untutored bench timing (ARCH_v3 6.0, PROVED by v2.2 from world.py): after my answer the next probe starts at
max(Q_end + 45, my_stop + 13). Polite W4 partner: never starts while I speak.
I answered early (stop = Q_end + 12). Option: say ONE extra syllable (3 ticks) starting at Q_end + t_x, or stay quiet.
Return to the actor from the partner's next onset: r_5b * gamma^(onset - decision), gamma 0.95 per tick (G24),
with r_5b = w_resp * trace or w_resp * [trace > 0.5], trace = exp(-(onset - my last stop) / tau_e), tau_e 20 (G14).
A contingent version subtracts what the onset would have been worth had I stayed quiet (Watson 1972; Gergely & Watson
1999: contingency = P(response | act) - P(response | no act)); the onset happens anyway, so its value is 0.
Run: python3 -I p5b_arith.py
"""
import math

TAU_E, GAMMA = 20.0, 0.95
stop = 12  # my answer's stop, ticks after Q_end


def onset_after(last_stop):
    return max(45, last_stop + 13)


def ret(t_x, form):
    last_stop = stop if t_x is None else t_x + 3
    on = onset_after(last_stop)
    tr = math.exp(-(on - last_stop) / TAU_E)
    r = tr if form == 'graded' else float(tr > 0.5)
    t0 = stop + 1  # decision point
    return r * GAMMA ** (on - t0), on, tr


for form in ('graded', 'threshold'):
    print(f'== r_5b {form}')
    q = ret(None, form)
    print(f'   stay quiet          : onset Q_end+{q[1]}, trace {q[2]:.2f}, return {q[0]:.3f}')
    best = None
    for t_x in range(stop + 1, 42):
        v = ret(t_x, form)
        if best is None or v[0] > best[1][0]:
            best = (t_x, v)
    for t_x in (stop + 1, 25, 30, 35, best[0]):
        v = ret(t_x, form)
        print(f'   extra at Q_end+{t_x:2d}  : onset Q_end+{v[1]}, trace {v[2]:.2f}, return {v[0]:.3f}  (advantage {v[0] - q[0]:+.3f})')
    print(f'   best extra-turn time Q_end+{best[0]} (the partner onset is predictable from the age bins)')
print('== contingent r_5b (response minus the response expected without my act): every option returns 0; no incentive')
